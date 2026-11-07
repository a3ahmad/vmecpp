// SPDX-License-Identifier: MIT
// Experimental persistent single-partition GPU iteration runtime.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
#include <cuda/std/array>
#endif
#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
#include <cuda.h>
#include <dlfcn.h>

#include <cstdio>
#include <limits>
#include <memory>

#include "gpu_solver_buffer_pool.h"
#endif
#if defined(VMECPP_COORDINATE_CAPTURES)
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#endif

#include "gpu_diagnostics.h"
#include "gpu_forward.h"
#include "gpu_optimistic_diagnostics.h"
#include "gpu_solver.h"
#if defined(VMECPP_PINNED_CPU_LOG)
#include "gpu_pinned_cpu_log.h"
#endif
#if defined(VMECPP_THOMAS_PREFIX)
#include "gpu_thomas_prefix.h"
#if !defined(VMECPP_CACHE_TRIDIAGONAL_FACTORS)
#error "Thomas prefix requires cached tridiagonal factors"
#endif
#endif
#if defined(VMECPP_LATTICE_RESIDUALS)
#include "gpu_lattice_residual.h"
#endif
#if defined(VMECPP_LATTICE_ENERGIES)
#include "gpu_lattice_energies.h"
#endif
#if defined(VMECPP_COOPERATIVE_ENERGIES)
#include "gpu_cooperative_energies.h"
#endif
#if defined(VMECPP_COOPERATIVE_RESIDUALS)
#include "gpu_cooperative_residual.h"
#endif

namespace {
#if defined(VMECPP_COORDINATE_CAPTURES)
// One loaded backend owns this gate. It intentionally outlives thread-local
// cleanup. Callers must join every completed worker before thread destruction.
std::shared_mutex& CaptureGate() {
  static auto* gate = new std::shared_mutex;
  return *gate;
}
constexpr std::array<const char*, 7> kFlagNames{
    "VMECPP_GPU_SOLVER_PINNED_SCALARS", "VMECPP_GPU_RESIDUAL_ROUNDS",
    "VMECPP_GPU_RESIDUAL_GRAPH",        "VMECPP_GPU_SOLVER_GRAPH",
    "VMECPP_GPU_PROFILE_REFERENCE",     "VMECPP_GPU_DIAGNOSTICS_ROUNDS",
    "VMECPP_GPU_DIAGNOSTICS_GRAPH"};
// Host-only activation evidence. Reading these counters never touches CUDA.
thread_local std::array<std::uint64_t, 3> thread_activity{};
#endif
static_assert(sizeof(double) == 8 && sizeof(int) == 4,
              "The scalar packet requires binary64 and a 32-bit norm flag");
void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
struct ControllerReference {
  cuda::std::array<double, 3> scalars_;
  cuda::std::array<double, 10> history_;
  double fsq_;
  int iter1_;
  int iter2_;
  bool pinned_log_allowed_;
  bool reference_enabled_;
  bool cpu_domain_escape_;
};
__device__ bool SameControllerBits(double left, double right) {
  return __double_as_longlong(left) == __double_as_longlong(right);
}
__global__ void SeedController(double* m_state, unsigned long long* m_stats,
                               double* m_residuals,
                               ControllerReference reference,
                               cuda::std::array<double, 3> previous_scalars) {
  for (int i = 0; i < 10; ++i) m_state[i] = reference.history_[i];
  m_state[10] = reference.fsq_;
  for (int i = 0; i < 3; ++i) m_residuals[3 + i] = reference.scalars_[i];
  for (int i = 0; i < 3; ++i) m_state[11 + i] = previous_scalars[i];
  ++m_stats[6];
}

__global__ void LiveController(
    double* m_state, unsigned long long* m_stats, const double* residuals,
    const double* diagnostics, double delta_s, ControllerReference reference,
    unsigned long long* m_log_stats, int* m_failure
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    ,
    const GpuSolverTailControllerControl* tail_control, const int* tail_status
#endif
) {
#if defined(VMECPP_FUSE_DAMPING_TAIL)
  if (tail_control != nullptr) {
    if (*tail_status != static_cast<int>(GpuSolverTailStatus::STEPPED)) return;
    reference.scalars_[2] = tail_control->time_step_;
    reference.iter1_ = tail_control->iter1_;
    reference.iter2_ = tail_control->iter2_;
    reference.pinned_log_allowed_ = tail_control->pinned_log_allowed_;
    reference.reference_enabled_ = false;
    reference.cpu_domain_escape_ = false;
  }
#endif
  const bool check = reference.reference_enabled_;
#if defined(VMECPP_PINNED_CPU_LOG)
  if (!reference.pinned_log_allowed_) {
    if (!check) {
      if (m_failure != nullptr) *m_failure = 1;
      return;
    }
    ++m_stats[0];
    ++m_log_stats[reference.cpu_domain_escape_ ? 2 : 1];
    for (int i = 0; i < 10; ++i) m_state[i] = reference.history_[i];
    m_state[10] = reference.fsq_;
    for (int i = 0; i < 3; ++i) m_state[11 + i] = reference.scalars_[i];
    return;
  }
#endif
  // Proposal locals keep accepted state untouched if the exact port declines.
  cuda::std::array<double, 10> history;
  for (int i = 0; i < 10; ++i) history[i] = m_state[i];
  const double r = __dmul_rn(__dadd_rn(0.0, residuals[3]), diagnostics[6]);
  const double z = __dmul_rn(__dadd_rn(0.0, residuals[4]), diagnostics[6]);
  const double lambda = __dmul_rn(__dadd_rn(0.0, residuals[5]), delta_s);
  const double fsq = __dadd_rn(__dadd_rn(r, z), lambda);
  const double dt = reference.scalars_[2];
  if (reference.iter2_ == reference.iter1_) {
    const double initial = __ddiv_rn(0.15, dt);
    for (int i = 0; i < 10; ++i) history[i] = initial;
  }
  const double oldest = history[0];
  for (int i = 0; i < 9; ++i) history[i] = history[i + 1];
  history[9] = oldest;
  if (reference.iter2_ > reference.iter1_) {
    double numerator = 0.0;
    if (fsq != 0.0) {
      double logarithm = 0.0;
#if defined(VMECPP_PINNED_CPU_LOG)
      ++m_log_stats[0];
      if (!TryPinnedCpuLog(__ddiv_rn(fsq, m_state[10]), logarithm)) {
        ++m_log_stats[2];
        if (!check) {
          if (m_failure != nullptr) *m_failure = 2;
          return;
        }
        ++m_stats[0];
        for (int i = 0; i < 10; ++i) m_state[i] = reference.history_[i];
        m_state[10] = reference.fsq_;
        for (int i = 0; i < 3; ++i) m_state[11 + i] = reference.scalars_[i];
        return;
      }
      ++m_log_stats[3];
#else
      logarithm = log(__ddiv_rn(fsq, m_state[10]));
#endif
      const double magnitude = __longlong_as_double(
          __double_as_longlong(logarithm) & 0x7fffffffffffffffLL);
      numerator = 0.15 < magnitude ? 0.15 : magnitude;
    }
    history[9] = __ddiv_rn(numerator, dt);
  }
  double total = 0.0;
  bool history_difference = false;
  for (int i = 0; i < 10; ++i) {
    total = __dadd_rn(total, history[i]);
    if (check)
      history_difference |=
          !SameControllerBits(history[i], reference.history_[i]);
  }
  const double otav = __ddiv_rn(total, 10.0);
  const double dtau = __ddiv_rn(__dmul_rn(dt, otav), 2.0);
  const double b1 = __dsub_rn(1.0, dtau);
  const double fac = __ddiv_rn(1.0, __dadd_rn(1.0, dtau));
  const bool coefficient_failure =
      check && (!SameControllerBits(fac, reference.scalars_[0]) ||
                !SameControllerBits(b1, reference.scalars_[1]));
  const bool fsq_failure = check && !SameControllerBits(fsq, reference.fsq_);
  ++m_stats[0];
  m_stats[2] += coefficient_failure;
  m_stats[3] += fsq_failure;
  m_stats[4] += history_difference;
  if (coefficient_failure || fsq_failure) {
    for (int i = 0; i < 10; ++i) m_state[i] = reference.history_[i];
    m_state[10] = reference.fsq_;
    m_state[11] = reference.scalars_[0];
    m_state[12] = reference.scalars_[1];
  } else {
    ++m_stats[1];
    m_stats[5] += history_difference;
    for (int i = 0; i < 10; ++i) m_state[i] = history[i];
    m_state[10] = fsq;
    m_state[11] = fac;
    m_state[12] = b1;
  }
  m_state[13] = dt;
}

#if defined(VMECPP_FUSE_DAMPING_TAIL)
__global__ void GateTailController(
    const GpuSolverTailControllerControl* control, const double* residuals,
    const double* diagnostics, double delta_s, const double* state,
    const int* invalid_norm, int* m_status, unsigned long long* m_stats) {
  GpuSolverTailStatus status = GpuSolverTailStatus::STEPPED;
  // Preserve the original single-partition +=0 and left-to-right scaling.
  const double r =
      __dmul_rn(__dmul_rn(__dadd_rn(0.0, residuals[0]), diagnostics[4]), 0.25);
  const double z =
      __dmul_rn(__dmul_rn(__dadd_rn(0.0, residuals[1]), diagnostics[4]), 0.25);
  const double l = __dmul_rn(__dadd_rn(0.0, residuals[2]), diagnostics[5]);
  if (*invalid_norm != 0)
    status = GpuSolverTailStatus::INVALID_NORM;
  else if (r <= control->force_tolerance_ && z <= control->force_tolerance_ &&
           l <= control->force_tolerance_)
    status = GpuSolverTailStatus::CONVERGED;
  else if (!control->pinned_log_allowed_)
    status = GpuSolverTailStatus::GUARD_DECLINED;
  else {
    const double rp = __dmul_rn(__dadd_rn(0.0, residuals[3]), diagnostics[6]);
    const double zp = __dmul_rn(__dadd_rn(0.0, residuals[4]), diagnostics[6]);
    const double lp = __dmul_rn(__dadd_rn(0.0, residuals[5]), delta_s);
    const double fsq = __dadd_rn(__dadd_rn(rp, zp), lp);
    const auto dt_word = static_cast<unsigned long long>(
        __double_as_longlong(control->time_step_));
    const auto dt_exp = (dt_word >> 52) & 2047ULL;
    if (control->iter2_ <= control->iter1_ || (dt_word >> 63) != 0 ||
        dt_exp == 0 || dt_exp == 2047)
      status = GpuSolverTailStatus::DOMAIN_DECLINED;
    else if (fsq != 0.0) {
      const auto word = static_cast<unsigned long long>(
          __double_as_longlong(__ddiv_rn(fsq, state[10])));
      const auto exponent = (word >> 52) & 2047ULL;
      if ((word >> 63) != 0 || exponent == 0 || exponent == 2047)
        status = GpuSolverTailStatus::DOMAIN_DECLINED;
    }
  }
  const int value = static_cast<int>(status);
  *m_status = value;
  ++m_stats[value];
}
#endif

struct ControllerStep {
  cuda::std::array<double*, 6> geometry_;
  cuda::std::array<double*, 6> velocity_;
  cuda::std::array<const double*, 6> forces_;
  const double* coefficients_;
  const int* failure_;
#if defined(VMECPP_FUSE_DAMPING_TAIL)
  const int* tail_status_;
#endif
  int count_;
};
__global__ void LiveControllerStep(ControllerStep state) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = blockIdx.y;
#if defined(VMECPP_FUSE_DAMPING_TAIL)
  if (state.tail_status_ != nullptr &&
      *state.tail_status_ != static_cast<int>(GpuSolverTailStatus::STEPPED))
    return;
#endif
  if (index >= state.count_ || (state.failure_ != nullptr && *state.failure_))
    return;
  const double fac = state.coefficients_[11];
  const double b1 = state.coefficients_[12];
  const double dt = state.coefficients_[13];
  const double velocity =
      __dmul_rn(fac, __dadd_rn(__dmul_rn(b1, state.velocity_[channel][index]),
                               __dmul_rn(dt, state.forces_[channel][index])));
  state.velocity_[channel][index] = velocity;
  state.geometry_[channel][index] =
      __dadd_rn(state.geometry_[channel][index], __dmul_rn(dt, velocity));
}
#endif

#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
GpuBufferPoolDomain BufferDomain() {
  // The driver handle intentionally outlives TLS cleanup. Default builds never
  // load it or call the driver API. No CUDA device/context is selected here.
  using GetContext = CUresult (*)(CUcontext*);
  static auto get_context = [] {
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (driver == nullptr)
      throw std::runtime_error("Cannot identify GPU buffer pool driver domain");
    auto function =
        reinterpret_cast<GetContext>(dlsym(driver, "cuCtxGetCurrent"));
    if (function == nullptr)
      throw std::runtime_error("GPU buffer pool context query unavailable");
    return function;
  }();
  int device = -1;
  Check(cudaGetDevice(&device));
  CUcontext context = nullptr;
  if (get_context(&context) != CUDA_SUCCESS || context == nullptr)
    throw std::runtime_error("GPU buffer pool context domain unavailable");
  thread_local char identity;
  return {device, reinterpret_cast<std::uintptr_t>(context),
          reinterpret_cast<std::uintptr_t>(&identity)};
}
GpuBufferPoolDomain BufferDomainNoThrow() noexcept {
  try {
    return BufferDomain();
  } catch (...) {
    return {-1, 0, 0};
  }
}
void* PoolAllocate(std::size_t bytes) {
  void* pointer = nullptr;
  Check(cudaMalloc(&pointer, bytes));
  return pointer;
}
bool PoolFree(void* m_pointer) noexcept {
  return cudaFree(m_pointer) == cudaSuccess;
}
struct ThreadBufferPool {
  std::shared_ptr<GpuSolverBufferPool> pool_;
  GpuBufferPoolDomain domain_{};
  ~ThreadBufferPool() noexcept {
    // Owned-domain callers must drain all futures/contexts before worker TLS
    // teardown. Active leases prevent early frees if that contract is broken.
    if (!pool_) return;
    bool retired = false;
    try {
#if defined(VMECPP_COORDINATE_CAPTURES)
      std::shared_lock<std::shared_mutex> allocation_guard(CaptureGate());
#endif
      retired = pool_->Shutdown(BufferDomainNoThrow());
    } catch (...) {
      pool_->Shutdown({-1, 0, 0});
    }
    if (!retired)
      std::fputs(
          "GPU buffer pool TLS cleanup incomplete; buffers quarantined\n",
          stderr);
  }
};
std::shared_ptr<GpuSolverBufferPool> AcquireThreadPool(
    GpuBufferPoolDomain domain) {
  thread_local ThreadBufferPool state;
  if (!state.pool_) {
    state.pool_ =
        std::make_shared<GpuSolverBufferPool>(domain, PoolAllocate, PoolFree);
    state.domain_ = domain;
  } else if (domain != state.domain_) {
    throw std::runtime_error("GPU buffer pool thread/device/context changed");
  }
  return state.pool_;
}
#endif

class Solver {
 public:
  Solver() {
#if defined(VMECPP_COORDINATE_CAPTURES)
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
#else
    Check(cudaStreamCreate(&stream_));
#endif
#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
    try {
      pool_domain_ = BufferDomain();
      pool_ = AcquireThreadPool(pool_domain_);
    } catch (...) {
      cudaStreamDestroy(stream_);
      throw;
    }
#endif
  }
  ~Solver() {
#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
    bool retired = cudaStreamSynchronize(stream_) == cudaSuccess;
    for (auto& graph : graphs_) {
      if (graph.executable_ != nullptr &&
          cudaGraphExecDestroy(graph.executable_) != cudaSuccess)
        retired = false;
      if (graph.graph_ != nullptr &&
          cudaGraphDestroy(graph.graph_) != cudaSuccess)
        retired = false;
    }
    const auto domain = BufferDomainNoThrow();
    if (domain.thread_ == pool_domain_.thread_) {
      for (const auto lease : allocations_)
        if (!pool_->Recycle(lease, domain, retired)) retired = false;
    } else {
      // A foreign thread must not mutate the owner's unsynchronized TLS pool.
      retired = false;
    }
    if (retired) {
      if (host_scalar_packet_ != nullptr) cudaFreeHost(host_scalar_packet_);
      if (host_jacobian_ != nullptr) cudaFreeHost(host_jacobian_);
#if defined(VMECPP_FUSE_DAMPING_TAIL)
      if (host_tail_control_ != nullptr) cudaFreeHost(host_tail_control_);
#endif
    } else {
      std::fputs("GPU solver cleanup incomplete; buffers quarantined\n",
                 stderr);
    }
    cudaStreamDestroy(stream_);
#else
    // The stream belongs to this context and outlives all its allocations.
    cudaStreamSynchronize(stream_);
    for (auto& graph : graphs_) {
      if (graph.executable_ != nullptr) cudaGraphExecDestroy(graph.executable_);
      if (graph.graph_ != nullptr) cudaGraphDestroy(graph.graph_);
    }
#if defined(VMECPP_THOMAS_PREFIX)
    if (thomas_prefix_ != nullptr) DestroyGpuThomasPrefix(thomas_prefix_);
#endif
    for (void* pointer : allocations_) cudaFree(pointer);
    if (host_scalar_packet_ != nullptr) cudaFreeHost(host_scalar_packet_);
    if (host_jacobian_ != nullptr) cudaFreeHost(host_jacobian_);
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    if (host_tail_control_ != nullptr) cudaFreeHost(host_tail_control_);
#endif
    cudaStreamDestroy(stream_);
#endif
  }
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  double* Allocate(std::size_t count, const double* source = nullptr) {
    if (count == 0) throw std::runtime_error("Empty GPU solver allocation");
    double* pointer = nullptr;
#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double))
      throw std::runtime_error("GPU solver allocation byte count overflow");
    const auto lease = pool_->Acquire(count * sizeof(double), pool_domain_);
    pointer = static_cast<double*>(lease.pointer_);
    try {
      allocations_.push_back(lease);
    } catch (...) {
      pool_->Recycle(lease, pool_domain_, true);
      throw;
    }
#else
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    try {
      allocations_.push_back(pointer);
    } catch (...) {
      cudaFree(pointer);
      throw;
    }
#endif
    if (source != nullptr) {
      Upload(pointer, source, count);
    } else {
      Check(cudaMemsetAsync(pointer, 0, count * sizeof(double), stream_));
    }
    return pointer;
  }

  void Upload(double* m_destination, const double* source, std::size_t count) {
    Check(cudaMemcpyAsync(m_destination, source, count * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
  }
  void Download(double* m_destination, const double* source,
                std::size_t count) {
    Check(cudaMemcpyAsync(m_destination, source, count * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
  }
  void Synchronize() {
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    int failure = 0;
    if (controller_failure_ != nullptr)
      Check(cudaMemcpyAsync(&failure, controller_failure_, sizeof(failure),
                            cudaMemcpyDeviceToHost, stream_));
#endif
    Check(cudaStreamSynchronize(stream_));
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    if (failure)
      throw std::runtime_error("Exact GPU damping preflight invariant failed");
#endif
  }

  const char* Flag(const char* name) const {
#if defined(VMECPP_COORDINATE_CAPTURES)
    for (std::size_t i = 0; i < kFlagNames.size(); ++i) {
      if (std::strcmp(name, kFlagNames[i]) == 0)
        return flags_[i] ? flags_[i]->c_str() : nullptr;
    }
    throw std::runtime_error("Unknown frozen GPU solver flag");
#else
    return std::getenv(name);
#endif
  }

  void Initialize(const GpuSolverSetup& setup) {
#if defined(VMECPP_COORDINATE_CAPTURES)
    // An active context never rereads mutable process-global configuration.
    for (std::size_t i = 0; i < kFlagNames.size(); ++i) {
      const char* value = std::getenv(kFlagNames[i]);
      if (value != nullptr) flags_[i] = std::string(value);
    }
#endif
    // The opt-in fixes allocation layout for this context's complete lifetime.
    const char* pinned = Flag("VMECPP_GPU_SOLVER_PINNED_SCALARS");
    pinned_scalars_ = pinned != nullptr && pinned[0] == '1';
    const auto& d = setup.basis_.dimensions_;
    const int ns = d[11];
    if (d[0] != 0 || d[2] != 0 || d[3] != ns || d[4] != ns || d[12] != 0 ||
        d[6] <= 0 || setup.forces_.adiabatic_index_ != 0.0) {
      throw std::runtime_error("GPU solver requires one symmetric partition");
    }
    coefficients_ = std::size_t(ns) * d[5] * (d[6] + 1);
    full_ = std::size_t(ns) * d[7] * d[8];
    const std::size_t half = std::size_t(ns - 1) * d[7] * d[8];
    dimensions_ = setup.state_.dimensions_;
    if (dimensions_ != std::array<int, 5>{ns, d[5], d[6], d[1], d[4]}) {
      throw std::runtime_error("GPU solver state dimensions mismatch");
    }
    step_.coefficient_count_ = static_cast<int>(coefficients_);
    step_.stream_ = stream_;
#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
    controller_state_ = Allocate(14);
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    controller_failure_ = reinterpret_cast<int*>(Allocate(1));
#endif
    controller_stats_ = reinterpret_cast<unsigned long long*>(Allocate(7));
#if defined(VMECPP_PINNED_CPU_LOG)
    pinned_log_stats_ = reinterpret_cast<unsigned long long*>(Allocate(4));
#endif
    controller_delta_s_ = setup.fields_.scalars_[0];
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    static_assert(sizeof(GpuSolverTailControllerControl) % sizeof(double) == 0);
    tail_control_ = reinterpret_cast<GpuSolverTailControllerControl*>(
        Allocate(sizeof(GpuSolverTailControllerControl) / sizeof(double)));
    Check(cudaMallocHost(reinterpret_cast<void**>(&host_tail_control_),
                         sizeof(GpuSolverTailControllerControl)));
    tail_controller_stats_ = reinterpret_cast<unsigned long long*>(Allocate(6));
#endif
#endif
    double* state_base = Allocate(12 * coefficients_);
    double* force_base = Allocate(6 * coefficients_);
    for (int i = 0; i < 18; ++i) {
      if (setup.state_.sizes_[i] != coefficients_) {
        throw std::runtime_error("GPU solver state size mismatch");
      }
      double* value = nullptr;
      if (i < 12) {
        value = state_base + i * coefficients_;
        Upload(value, setup.state_.data_[i], coefficients_);
      } else {
        value = force_base + (i - 12) * coefficients_;
        Upload(value, setup.state_.data_[i], coefficients_);
      }
      if (i < 6) {
        step_.geometry_[i] = value;
      } else if (i < 12) {
        step_.velocity_[i - 6] = value;
      } else {
        iteration_.forces_[i - 12] = value;
        step_.forces_[i - 12] = value;
      }
    }
    double* backup_base = Allocate(6 * coefficients_);
    double* last_update_base = Allocate(6 * coefficients_);
    double* last_valid_base = Allocate(6 * coefficients_);
    for (int i = 0; i < 6; ++i) {
      const double* backup = setup.backup_.data_[i];
      if (backup != nullptr && setup.backup_.sizes_[i] != coefficients_)
        throw std::runtime_error("GPU backup size mismatch");
      backup_[i] = backup_base + i * coefficients_;
      Upload(backup_[i], backup == nullptr ? setup.state_.data_[i] : backup,
             coefficients_);
      last_update_[i] = last_update_base + i * coefficients_;
      last_valid_update_[i] = last_valid_base + i * coefficients_;
      Upload(last_update_[i], setup.state_.data_[i], coefficients_);
      Upload(last_valid_update_[i], setup.state_.data_[i], coefficients_);
    }
    preparation_.scalxc_ = Allocate(setup.scalxc_size_, setup.scalxc_);
    preparation_.dimensions_ = dimensions_;
    preparation_.stream_ = stream_;
    inverse_.dimensions_ = d;
    inverse_.stream_ = stream_;
    inverse_.partials_ = Allocate(std::size_t{12} * ns * d[5] * d[7]);
    double* physical_coefficients = Allocate(6 * coefficients_);
    for (int i = 0; i < 6; ++i) {
      preparation_.coefficients_[i] = physical_coefficients + i * coefficients_;
      inverse_.input_[i] = preparation_.coefficients_[i];
    }
    for (int i = 6; i < 16; ++i) {
      inverse_.input_[i] =
          Allocate(setup.basis_.sizes_[i], setup.basis_.data_[i]);
    }
    auto& geometry = constraint_.force_.geometry_;
    double* full_base = Allocate(18 * full_);
    for (int i = 0; i < 18; ++i) {
      geometry.full_[i] = full_base + i * full_;
      inverse_.output_[i] = geometry.full_[i];
      inverse_.output_sizes_[i] = full_;
    }
    double* half_base = Allocate(14 * half);
    for (int i = 0; i < 14; ++i) geometry.half_[i] = half_base + i * half;
    for (int i = 0; i < 10; ++i) {
      geometry.profiles_[i] =
          Allocate(setup.fields_.sizes_[i + 18], setup.fields_.data_[i + 18]);
    }
    geometry.profiles_[10] = Allocate(ns - 1, setup.cached_dvds_);
    for (int i = 0; i < 5; ++i)
      profile_backup_[i] = Allocate(i == 2 || i == 3 ? ns : ns - 1);
    geometry.jacobian_extrema_ = Allocate(2);
    geometry.jacobian_partials_ = Allocate(3 * ((half + 31) / 32));
    geometry.profile_terms_ = Allocate(3 * half);
    geometry.dimensions_ = d;
    geometry.scalars_ = setup.fields_.scalars_;
    geometry.stream_ = stream_;
    auto& force = constraint_.force_;
    force.adiabatic_index_ = setup.forces_.adiabatic_index_;
    for (int i = 0; i < 2; ++i) {
      force.constants_[i] =
          Allocate(setup.forces_.sizes_[i], setup.forces_.data_[i]);
    }
    double* mhd_base = Allocate(12 * half);
    for (int i = 0; i < 12; ++i) force.mhd_[i] = mhd_base + i * half;
    double* lambda_base = Allocate(4 * full_);
    for (int i = 0; i < 4; ++i) force.lambda_[i] = lambda_base + i * full_;
    force.total_pressure_ = Allocate(half);
    for (int i = 0; i < 15; ++i) {
      constraint_.constants_[i] =
          Allocate(setup.constraints_.sizes_[i], setup.constraints_.data_[i]);
    }
    double* constraint_geometry = Allocate(4 * full_);
    double* constraint_forces = Allocate(4 * full_);
    for (int i = 0; i < 4; ++i) {
      constraint_.geometry_[i] = constraint_geometry + i * full_;
      constraint_.forces_[i] = constraint_forces + i * full_;
    }
    const std::size_t toroidal = std::size_t(ns - 1) * (d[5] - 2) * d[7];
    const std::size_t spectral = std::size_t(ns - 1) * (d[5] - 2) * (d[6] + 1);
    for (int i = 0; i < 2; ++i) {
      constraint_.poloidal_[i] = Allocate(toroidal);
      constraint_.spectral_[i] = Allocate(spectral);
      constraint_.toroidal_[i] = Allocate(toroidal);
    }
    forward_.dimensions_ = d;
    forward_.stream_ = stream_;
    forward_.partials_ = Allocate(std::size_t{12} * ns * d[5] * d[7]);
    for (int i = 0; i < 6; ++i) {
      forward_.output_[i] = iteration_.forces_[i];
      forward_.output_sizes_[i] = coefficients_;
    }
    const auto& m = force.mhd_;
    const auto& l = force.lambda_;
    const auto& c = constraint_.constants_;
    const auto& f = constraint_.forces_;
    forward_.input_ = {m[0], m[1], m[2],  m[3],  l[0], l[1], m[4],  m[5],
                       m[6], m[7], l[2],  l[3],  m[8], m[9], m[10], m[11],
                       f[0], f[1], f[2],  f[3],  c[4], c[5], c[6],  c[7],
                       c[8], c[9], c[10], c[11], c[12]};
    preconditioner_.dimensions_ = setup.preconditioner_.dimensions_;
    preconditioner_.scalars_ = setup.preconditioner_.scalars_;
    preconditioner_.stream_ = stream_;
    const auto& h = geometry.half_;
    const auto& v = geometry.full_;
    const auto& q = geometry.profiles_;
    preconditioner_.input_ = {h[0],
                              force.total_pressure_,
                              h[5],
                              h[11],
                              h[6],
                              h[7],
                              h[8],
                              h[9],
                              h[4],
                              h[2],
                              v[8],
                              v[9],
                              v[7],
                              h[3],
                              h[1],
                              v[2],
                              v[3],
                              v[1],
                              q[1],
                              q[0],
                              nullptr,
                              nullptr,
                              q[2],
                              constraint_.geometry_[1],
                              constraint_.geometry_[2],
                              nullptr};
    for (int i : {20, 21, 25}) {
      preconditioner_.input_[i] = Allocate(setup.preconditioner_.sizes_[i],
                                           setup.preconditioner_.data_[i]);
    }
    cache_sizes_ = setup.cached_preconditioner_.sizes_;
    for (int i = 0; i < 20; ++i) {
      preconditioner_.output_[i] =
          Allocate(cache_sizes_[i], setup.cached_preconditioner_.data_[i]);
    }
    constraint_.constants_[3] = preconditioner_.output_[13];
    preconditioner_.terms_ = Allocate(21 * half);
    preconditioner_.sums_ = Allocate(21 * (ns - 1));
    if (pinned_scalars_) {
      // Existing kernels write directly into nonoverlapping packet subranges:
      // residuals[6], diagnostics[7], invalidNorm[1 int], alignment padding.
      scalar_packet_ = Allocate(14);
      residuals_ = scalar_packet_;
      preconditioner_.invalid_norm_ =
          reinterpret_cast<int*>(scalar_packet_ + 13);
      Check(cudaMallocHost(&host_scalar_packet_, 14 * sizeof(double)));
      Check(cudaMallocHost(&host_jacobian_, 2 * sizeof(double)));
    } else {
      // A double-sized allocation satisfies integer alignment and capacity.
      preconditioner_.invalid_norm_ = reinterpret_cast<int*>(Allocate(1));
    }
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    tail_controller_status_ =
        pinned_scalars_ ? reinterpret_cast<int*>(scalar_packet_ + 13) + 1
                        : reinterpret_cast<int*>(Allocate(1));
#endif
    diagnostics_.data_ = {h[6],
                          h[10],
                          h[11],
                          h[12],
                          h[13],
                          q[2],
                          force.constants_[0],
                          q[10],
                          h[7],
                          h[0],
                          step_.geometry_[0],
                          step_.geometry_[1],
                          step_.geometry_[2],
                          step_.geometry_[3]};
    for (int i = 0; i < 7; ++i) {
      diagnostics_.dimensions_[i] = setup.preconditioner_.dimensions_[i];
    }
    diagnostics_.scalars_ = {geometry.scalars_[0], force.adiabatic_index_,
                             geometry.scalars_[2]};
    diagnostics_.sums_ = Allocate(6);
    if (pinned_scalars_) {
      diagnostics_.output_ = scalar_packet_ + 6;
      Upload(diagnostics_.output_, setup.cached_diagnostics_.data(), 7);
    } else {
      diagnostics_.output_ = Allocate(7, setup.cached_diagnostics_.data());
    }
    diagnostics_.stream_ = stream_;
#if defined(VMECPP_COOPERATIVE_ENERGIES)
    cooperative_energies_supported_ =
        GpuCooperativeEnergiesSupported(diagnostics_);
#endif
    iteration_.dimensions_ = dimensions_;
    iteration_.stream_ = stream_;
    constexpr std::array<int, 11> kCacheMap{5,  7,  9,  11, 14, 15,
                                            16, 17, 18, 19, 0};
    for (int i = 0; i < 11; ++i)
      iteration_.constants_[i] = preconditioner_.output_[kCacheMap[i]];
    for (int i = 0; i < 2; ++i) {
      iteration_.factored_upper_[i] = Allocate(coefficients_);
      iteration_.denominators_[i] = Allocate(coefficients_);
    }
#if defined(VMECPP_CACHE_TRIDIAGONAL_FACTORS)
    // The uploaded setup cache can be used before the first refresh. These
    // context-owned factors stay valid until the next matrix refresh.
    LaunchGpuIterationFactor(iteration_);
#endif
#if defined(VMECPP_THOMAS_PREFIX)
    thomas_prefix_ = CreateGpuThomasPrefix(iteration_);
#endif
    if (!pinned_scalars_) residuals_ = Allocate(6);
#if defined(VMECPP_COOPERATIVE_RESIDUALS)
    // Cache capability before stream capture. Unsupported grids retain the
    // existing certified scan path without changing resolution or tolerances.
    cooperative_supported_ = GpuCooperativeResidualsSupported(iteration_);
#endif
    Synchronize();
  }

  void UploadState(const GpuStepInput& state) {
    if (state.dimensions_ != dimensions_)
      throw std::runtime_error("State dimensions changed");
    // Restores replace geometry/velocity only. Prepared forces remain owned by
    // the runtime, including the original cached-force bad-Jacobian behavior.
    for (int i = 0; i < 12; ++i) {
      if (state.sizes_[i] != coefficients_)
        throw std::runtime_error("State size changed");
      double* value = i < 6 ? step_.geometry_[i] : step_.velocity_[i - 6];
      Upload(value, state.data_[i], coefficients_);
    }
    Synchronize();
  }

  bool UseInnerGraph(const char* requested) {
#if defined(VMECPP_COORDINATE_CAPTURES)
    // Warmup holds a shared gate; only RunStage's exclusive branch captures.
    static_cast<void>(requested);
    return false;
#else
    if (requested == nullptr || requested[0] != '1') return false;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(stream_, &capture_status));
    // cudaGraphLaunch cannot be recorded by stream capture. The optimistic
    // APIs can instead enqueue the identical warmed kernels into our parent.
    return capture_status == cudaStreamCaptureStatusNone;
#endif
  }

  void Residuals(bool include_edge_rz) {
    const char* rounds = Flag("VMECPP_GPU_RESIDUAL_ROUNDS");
    if (rounds != nullptr && std::atoi(rounds) > 0) {
#if defined(VMECPP_LATTICE_RESIDUALS)
      // Accept only a complete original-order bitwise recurrence certificate.
      // Invalid components use the original ordered sum. A zero proposal budget
      // retains the original path for the captured-reference control matrix.
      LaunchGpuIterationResidualsLattice(iteration_, include_edge_rz);
      return;
#endif
#if defined(VMECPP_COOPERATIVE_RESIDUALS)
      if (cooperative_supported_) {
        LaunchGpuCooperativeResiduals(iteration_, include_edge_rz,
                                      std::atoi(rounds));
        return;
      }
#endif
      const char* graph = Flag("VMECPP_GPU_RESIDUAL_GRAPH");
      LaunchGpuIterationResidualsOptimistic(
          iteration_, include_edge_rz, std::atoi(rounds), UseInnerGraph(graph));
    } else {
      LaunchGpuIterationResidualsConditional(iteration_, include_edge_rz,
                                             nullptr);
    }
  }

  void EnqueuePrelude() {
    CopyGeometry(last_update_, step_.geometry_);
    CopyGeometry(preparation_.coefficients_, step_.geometry_);
    preparation_.dimensions_[3] = dimensions_[0];
    preparation_.m1_scale_ = 1.0;
    preparation_.zero_z_m1_ = false;
    LaunchGpuIterationPreparation(preparation_);
    LaunchGpuGeometryAxis(preparation_);
    LaunchGpuFourierDevice(inverse_);
    auto& geometry = constraint_.force_.geometry_;
    for (int i = 0; i < 5; ++i)
      Check(cudaMemcpyAsync(
          profile_backup_[i], geometry.profiles_[i + 6],
          (i == 2 || i == 3 ? dimensions_[0] : dimensions_[0] - 1) *
              sizeof(double),
          cudaMemcpyDeviceToDevice, stream_));
    LaunchGpuResidentFields(geometry);
  }

  // The first real execution warms nested graph/workspace caches. Subsequent
  // capture records only device operations, never host readback or sync.
  template <typename Enqueue>
  void RunStage(const std::array<int, 8>& key, Enqueue enqueue) {
    const char* enabled = Flag("VMECPP_GPU_SOLVER_GRAPH");
    if (enabled == nullptr || enabled[0] != '1') {
#if defined(VMECPP_COORDINATE_CAPTURES)
      std::shared_lock<std::shared_mutex> allocation_guard(CaptureGate());
#endif
      enqueue();
      return;
    }
    GraphEntry* entry = nullptr;
    for (auto& graph : graphs_) {
      if (graph.key_ == key) {
        entry = &graph;
        break;
      }
    }
    if (entry == nullptr) {
      graphs_.push_back({key, nullptr, nullptr});
#if defined(VMECPP_COORDINATE_CAPTURES)
      std::shared_lock<std::shared_mutex> allocation_guard(CaptureGate());
#endif
      enqueue();
      return;
    }
    if (entry->executable_ == nullptr) {
#if defined(VMECPP_COORDINATE_CAPTURES)
      // Branch before locking: shared_mutex has no safe lock upgrade.
      std::unique_lock<std::shared_mutex> capture_guard(CaptureGate());
#endif
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      bool capturing = false;
      try {
        Check(
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        capturing = true;
        enqueue();
        const cudaError_t capture_status =
            cudaStreamEndCapture(stream_, &graph);
        capturing = false;
        Check(capture_status);
        Check(cudaGraphInstantiate(&executable, graph, 0));
      } catch (...) {
        if (capturing) cudaStreamEndCapture(stream_, &graph);
        if (executable != nullptr) cudaGraphExecDestroy(executable);
        if (graph != nullptr) cudaGraphDestroy(graph);
        throw;
      }
      entry->graph_ = graph;
      entry->executable_ = executable;
    }
    Check(cudaGraphLaunch(entry->executable_, stream_));
#if defined(VMECPP_COORDINATE_CAPTURES)
    ++thread_activity[2];
#endif
  }

  void Run(const GpuSolverControl& control, GpuSolverScalars& m_output) {
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    const bool fused = tail_controller_armed_;
    tail_controller_armed_ = false;
    m_output.tail_controller_status_ = 0;
#endif
#if defined(VMECPP_COORDINATE_CAPTURES)
    ++thread_activity[1];
#endif
    const char* profile_reference = Flag("VMECPP_GPU_PROFILE_REFERENCE");
    const std::array<int, 8> prelude_key{
        0,
        profile_reference != nullptr && std::strcmp(profile_reference, "1") == 0,
        0,
        0,
        0,
        0,
        0,
        0};
    RunStage(prelude_key, [this] { EnqueuePrelude(); });
    auto& geometry = constraint_.force_.geometry_;
    Download(
        pinned_scalars_ ? host_jacobian_ : m_output.jacobian_extrema_.data(),
        geometry.jacobian_extrema_, 2);
    Synchronize();
    if (pinned_scalars_)
      std::memcpy(m_output.jacobian_extrema_.data(), host_jacobian_,
                  2 * sizeof(double));
    m_output.bad_jacobian_ =
        m_output.jacobian_extrema_[0] * m_output.jacobian_extrema_[1] < 0.0;
    m_output.invalid_norm_ = 0;
    if (m_output.bad_jacobian_) {
      // The original stops before modifying current/volume profiles on bad
      // geometry. Preserve them even though the fused field adapter ran ahead.
      for (int i = 0; i < 5; ++i)
        Check(cudaMemcpyAsync(
            geometry.profiles_[i + 6], profile_backup_[i],
            (i == 2 || i == 3 ? dimensions_[0] : dimensions_[0] - 1) *
                sizeof(double),
            cudaMemcpyDeviceToDevice, stream_));
      return;
    }
    const char* residual_rounds = Flag("VMECPP_GPU_RESIDUAL_ROUNDS");
    const char* residual_graph = Flag("VMECPP_GPU_RESIDUAL_GRAPH");
    const char* diagnostic_rounds = Flag("VMECPP_GPU_DIAGNOSTICS_ROUNDS");
    const char* diagnostic_graph = Flag("VMECPP_GPU_DIAGNOSTICS_GRAPH");
    const std::array<int, 8> tail_key{
#if defined(VMECPP_FUSE_DAMPING_TAIL)
        fused ? 2 : 1,
#else
        1,
#endif
        control.refresh_preconditioner_,
        control.include_edge_rz_,
        control.zero_z_m1_,
        residual_rounds == nullptr ? 0 : std::atoi(residual_rounds),
        residual_graph != nullptr && residual_graph[0] == '1',
        diagnostic_rounds == nullptr ? 0 : std::atoi(diagnostic_rounds),
        diagnostic_graph != nullptr && diagnostic_graph[0] == '1'};
    RunStage(tail_key, [this, &control
#if defined(VMECPP_FUSE_DAMPING_TAIL)
                        ,
                        fused
#endif
    ] {
      EnqueueTail(control
#if defined(VMECPP_FUSE_DAMPING_TAIL)
                  ,
                  fused
#endif
      );
    });
    if (pinned_scalars_) {
      Download(host_scalar_packet_, scalar_packet_, 14);
    } else {
      Download(m_output.invariant_residuals_.data(), residuals_, 3);
      Download(m_output.preconditioned_residuals_.data(), residuals_ + 3, 3);
      Download(m_output.diagnostics_.data(), diagnostics_.output_, 7);
      if (control.refresh_preconditioner_) {
        Check(cudaMemcpyAsync(&m_output.invalid_norm_,
                              preconditioner_.invalid_norm_, sizeof(int),
                              cudaMemcpyDeviceToHost, stream_));
      }
    }
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    if (fused && !pinned_scalars_)
      Check(cudaMemcpyAsync(&m_output.tail_controller_status_,
                            tail_controller_status_, sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
#endif
    Synchronize();
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    if (fused && pinned_scalars_)
      std::memcpy(&m_output.tail_controller_status_,
                  reinterpret_cast<const char*>(host_scalar_packet_) +
                      13 * sizeof(double) + sizeof(int),
                  sizeof(int));
#endif
    if (pinned_scalars_) {
      std::memcpy(m_output.invariant_residuals_.data(), host_scalar_packet_,
                  3 * sizeof(double));
      std::memcpy(m_output.preconditioned_residuals_.data(),
                  host_scalar_packet_ + 3, 3 * sizeof(double));
      std::memcpy(m_output.diagnostics_.data(), host_scalar_packet_ + 6,
                  7 * sizeof(double));
      if (control.refresh_preconditioner_)
        std::memcpy(&m_output.invalid_norm_, host_scalar_packet_ + 13,
                    sizeof(int));
    }
  }

  void EnqueueTail(const GpuSolverControl& control
#if defined(VMECPP_FUSE_DAMPING_TAIL)
                   ,
                   bool fused
#endif
  ) {
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    // Capture the small H2D node in the existing tail graph. The pinned source
    // is owned until teardown and is changed only after the prior Run sync.
    // Canceling an arm before Run therefore never leaves an in-flight upload.
    if (fused)
      Check(cudaMemcpyAsync(tail_control_, host_tail_control_,
                            sizeof(GpuSolverTailControllerControl),
                            cudaMemcpyHostToDevice, stream_));
#endif
    LaunchGpuResidentForces(constraint_.force_);
    LaunchGpuResidentConstraintGeometry(constraint_);
    if (control.refresh_preconditioner_) {
      LaunchGpuPreconditioner(preconditioner_);
#if defined(VMECPP_CACHE_TRIDIAGONAL_FACTORS)
      // Refresh and factorization are captured in the same ordered tail graph.
      // Geometry restores deliberately retain this preconditioner, as on CPU.
      LaunchGpuIterationFactor(iteration_);
#if defined(VMECPP_THOMAS_PREFIX)
      RefreshGpuThomasPrefix(thomas_prefix_, iteration_);
#endif
#endif
    }
    const char* diagnostic_rounds = Flag("VMECPP_GPU_DIAGNOSTICS_ROUNDS");
#if defined(VMECPP_LATTICE_ENERGIES)
    if (!control.refresh_preconditioner_ && diagnostic_rounds != nullptr &&
        std::atoi(diagnostic_rounds) > 0) {
      // Every signed prefix is certified against the original recurrence;
      // refresh norms keep the existing path and their original output cache.
      LaunchGpuLatticeEnergies(diagnostics_);
    } else
#elif defined(VMECPP_COOPERATIVE_ENERGIES)
    if (!control.refresh_preconditioner_ && cooperative_energies_supported_ &&
        diagnostic_rounds != nullptr && std::atoi(diagnostic_rounds) > 0) {
      // Ordinary energies adapt up to32 rounds; refresh norms retain their
      // configured original path and complete output cache.
      LaunchGpuCooperativeEnergies(diagnostics_, 32);
    } else
#endif
        if (diagnostic_rounds != nullptr && std::atoi(diagnostic_rounds) > 0) {
      const char* graph = Flag("VMECPP_GPU_DIAGNOSTICS_GRAPH");
      LaunchGpuDiagnosticsOptimistic(
          diagnostics_, control.refresh_preconditioner_,
          std::atoi(diagnostic_rounds), UseInnerGraph(graph));
    } else if (control.refresh_preconditioner_) {
      LaunchGpuDiagnostics(diagnostics_);
    } else {
      LaunchGpuEnergies(diagnostics_);
    }
    LaunchGpuResidentConstraintFilter(constraint_);
    Check(cudaMemsetAsync(constraint_.forces_[0], 0, 4 * full_ * sizeof(double),
                          stream_));
    LaunchGpuResidentConstraintAssembly(constraint_);
    LaunchGpuForwardDevice(forward_);
    GpuPreparationDeviceState force_preparation{
        iteration_.forces_,  preparation_.scalxc_, dimensions_,
        0.70710678118654746, control.zero_z_m1_,   stream_};
    LaunchGpuIterationPreparation(force_preparation);
    iteration_.residuals_ = residuals_;
    Residuals(control.include_edge_rz_);
    LaunchGpuIterationM1(iteration_);
#if defined(VMECPP_THOMAS_PREFIX)
    LaunchGpuThomasPrefix(thomas_prefix_, iteration_);
#elif defined(VMECPP_CACHE_TRIDIAGONAL_FACTORS)
    LaunchGpuIterationRZFactored(iteration_);
#else
    LaunchGpuIterationRZ(iteration_);
#endif
    LaunchGpuIterationLambda(iteration_);
    iteration_.residuals_ = residuals_ + 3;
    Residuals(true);
    CopyGeometry(last_valid_update_, last_update_);
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    if (fused) {
      GateTailController<<<1, 1, 0, stream_>>>(
          tail_control_, residuals_, diagnostics_.output_, controller_delta_s_,
          controller_state_, preconditioner_.invalid_norm_,
          tail_controller_status_, tail_controller_stats_);
      Check(cudaGetLastError());
      AdvanceController({}, true);
    }
#endif
  }

  void Advance(const std::array<double, 3>& scalars) {
    step_.scalars_ = scalars;
    LaunchGpuIterationStep(step_);
  }
#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
  static ControllerReference ControllerInput(
      const GpuSolverControllerReference& reference) {
    ControllerReference result{};
    for (int i = 0; i < 3; ++i) result.scalars_[i] = reference.scalars_[i];
    for (int i = 0; i < 10; ++i) result.history_[i] = reference.history_[i];
    result.fsq_ = reference.fsq_;
    result.iter1_ = reference.iter1_;
    result.iter2_ = reference.iter2_;
    result.pinned_log_allowed_ = reference.pinned_log_allowed_;
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    result.reference_enabled_ = reference.reference_enabled_;
    result.cpu_domain_escape_ = reference.cpu_domain_escape_;
#else
    result.reference_enabled_ = true;
    result.cpu_domain_escape_ = false;
#endif
    return result;
  }
  void SeedControllerState(
      const std::array<double, 10>& history, double previous_fsq,
      const std::array<double, 3>& preconditioned_residuals,
      const std::array<double, 3>& previous_scalars = {}) {
    GpuSolverControllerReference reference{};
    reference.history_ = history;
    reference.fsq_ = previous_fsq;
    reference.scalars_ = preconditioned_residuals;
    cuda::std::array<double, 3> selected{};
    for (int i = 0; i < 3; ++i) selected[i] = previous_scalars[i];
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    Check(cudaMemsetAsync(controller_failure_, 0, sizeof(int), stream_));
#endif
    SeedController<<<1, 1, 0, stream_>>>(controller_state_, controller_stats_,
                                         residuals_, ControllerInput(reference),
                                         selected);
    Check(cudaGetLastError());
    controller_seeded_ = true;
  }
  void DownloadControllerState(GpuSolverControllerState& m_output) {
    if (!controller_seeded_)
      throw std::runtime_error("Controller state readback requires a seed");
    std::array<double, 14> packet{};
    Download(packet.data(), controller_state_, packet.size());
    Synchronize();
    for (int i = 0; i < 10; ++i) m_output.history_[i] = packet[i];
    m_output.previous_fsq_ = packet[10];
    for (int i = 0; i < 3; ++i) m_output.scalars_[i] = packet[11 + i];
  }
#if defined(VMECPP_FUSE_DAMPING_TAIL)
  void ArmTailController(const GpuSolverTailControllerControl& control) {
    if (!controller_seeded_)
      throw std::runtime_error("Fused controller requires a CPU setup seed");
    if (tail_controller_armed_)
      throw std::runtime_error("Fused controller already armed");
    *host_tail_control_ = control;
    tail_controller_armed_ = true;
  }
  void CancelTailController() { tail_controller_armed_ = false; }
  void TailControllerStats(std::array<unsigned long long, 6>& m_stats) {
    Check(cudaMemcpyAsync(m_stats.data(), tail_controller_stats_,
                          sizeof(m_stats), cudaMemcpyDeviceToHost, stream_));
    Synchronize();
  }
#endif
  void AdvanceController(const GpuSolverControllerReference& reference
#if defined(VMECPP_FUSE_DAMPING_TAIL)
                         ,
                         bool fused = false
#endif
  ) {
    if (!controller_seeded_)
      throw std::runtime_error("Live controller requires a CPU setup seed");
    LiveController<<<1, 1, 0, stream_>>>(
        controller_state_, controller_stats_, residuals_, diagnostics_.output_,
        controller_delta_s_, ControllerInput(reference),
#if defined(VMECPP_PINNED_CPU_LOG)
        pinned_log_stats_
#else
        nullptr
#endif
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
        ,
        controller_failure_
#else
        ,
        nullptr
#endif
#if defined(VMECPP_FUSE_DAMPING_TAIL)
        ,
        fused ? tail_control_ : nullptr,
        fused ? tail_controller_status_ : nullptr
#endif
    );
    Check(cudaGetLastError());
    ControllerStep state{};
    for (int i = 0; i < 6; ++i) {
      state.geometry_[i] = step_.geometry_[i];
      state.velocity_[i] = step_.velocity_[i];
      state.forces_[i] = step_.forces_[i];
    }
    state.coefficients_ = controller_state_;
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
    state.failure_ = controller_failure_;
#endif
#if defined(VMECPP_FUSE_DAMPING_TAIL)
    state.tail_status_ = fused ? tail_controller_status_ : nullptr;
#endif
    state.count_ = step_.coefficient_count_;
    LiveControllerStep<<<dim3((state.count_ + 127) / 128, 6), 128, 0,
                         stream_>>>(state);
    Check(cudaGetLastError());
  }
  void ControllerStats(std::array<unsigned long long, 7>& m_stats) {
    Check(cudaMemcpyAsync(m_stats.data(), controller_stats_, sizeof(m_stats),
                          cudaMemcpyDeviceToHost, stream_));
    Synchronize();
  }
#if defined(VMECPP_PINNED_CPU_LOG)
  void PinnedLogStats(std::array<unsigned long long, 4>& m_stats) {
    Check(cudaMemcpyAsync(m_stats.data(), pinned_log_stats_, sizeof(m_stats),
                          cudaMemcpyDeviceToHost, stream_));
    Synchronize();
  }
#endif
#endif
  void DownloadState(GpuStepOutput& m_output) {
    for (int i = 0; i < 12; ++i) {
      if (m_output.sizes_[i] != coefficients_)
        throw std::runtime_error("Download state size mismatch");
      Download(m_output.data_[i],
               i < 6 ? step_.geometry_[i] : step_.velocity_[i - 6],
               coefficients_);
    }
    Synchronize();
  }
  void DownloadCache(GpuPreconditionerOutput& m_output) {
    for (int i = 0; i < 20; ++i) {
      if (m_output.sizes_[i] != cache_sizes_[i])
        throw std::runtime_error("Download cache size mismatch");
      Download(m_output.data_[i], preconditioner_.output_[i], cache_sizes_[i]);
    }
    Synchronize();
  }

  void CopyGeometry(const std::array<double*, 6>& m_destination,
                    const std::array<double*, 6>& source) {
    Check(cudaMemcpyAsync(m_destination[0], source[0],
                          6 * coefficients_ * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream_));
  }
  void SaveBackup() { CopyGeometry(backup_, step_.geometry_); }
  void RestoreBackup() {
    CopyGeometry(step_.geometry_, backup_);
    Check(cudaMemsetAsync(step_.velocity_[0], 0,
                          6 * coefficients_ * sizeof(double), stream_));
  }
  void DownloadGeometry(GpuSolverGeometry which,
                        GpuPreparationOutput& m_output) {
    const auto& source = which == GpuSolverGeometry::CURRENT  ? step_.geometry_
                         : which == GpuSolverGeometry::BACKUP ? backup_
                         : which == GpuSolverGeometry::LAST_UPDATE
                             ? last_update_
                             : last_valid_update_;
    for (int i = 0; i < 6; ++i) {
      if (m_output.sizes_[i] != coefficients_)
        throw std::runtime_error("Geometry download size mismatch");
      Download(m_output.data_[i], source[i], coefficients_);
    }
    Synchronize();
  }
  void DownloadForces(GpuPreparationOutput& m_output) {
    for (int i = 0; i < 6; ++i) {
      if (m_output.sizes_[i] != coefficients_)
        throw std::runtime_error("Force download size mismatch");
      Download(m_output.data_[i], iteration_.forces_[i], coefficients_);
    }
    Synchronize();
  }
  void DownloadPrintout(double* m_dvds, std::size_t count,
                        std::array<double, 2>& m_axis) {
    if (count != static_cast<std::size_t>(dimensions_[0] - 1))
      throw std::runtime_error("Printout profile size mismatch");
    const auto& geometry = constraint_.force_.geometry_;
    Download(m_dvds, geometry.profiles_[10], count);
    Download(m_axis.data(), geometry.full_[0], 1);
    Download(m_axis.data() + 1, geometry.full_[6], 1);
    Synchronize();
  }

 private:
#if defined(VMECPP_THOMAS_PREFIX)
  void* thomas_prefix_ = nullptr;
#endif
  struct GraphEntry {
    std::array<int, 8> key_;
    cudaGraph_t graph_;
    cudaGraphExec_t executable_;
  };
  cudaStream_t stream_ = nullptr;
#if defined(VMECPP_COORDINATE_CAPTURES)
  std::array<std::optional<std::string>, kFlagNames.size()> flags_{};
#endif
  std::vector<GraphEntry> graphs_;
#if defined(VMECPP_REUSE_DEVICE_BUFFERS)
  std::shared_ptr<GpuSolverBufferPool> pool_;
  GpuBufferPoolDomain pool_domain_{};
  std::vector<GpuBufferLease> allocations_;
#else
  std::vector<void*> allocations_;
#endif
  std::size_t coefficients_ = 0;
  std::size_t full_ = 0;
  std::array<int, 5> dimensions_{};
  std::array<std::size_t, 20> cache_sizes_{};
  std::array<double*, 6> backup_{};
  std::array<double*, 6> last_update_{};
  std::array<double*, 6> last_valid_update_{};
  std::array<double*, 5> profile_backup_{};
  GpuStepDeviceState step_{};
  GpuPreparationDeviceState preparation_{};
  GpuFourierDeviceState inverse_{};
  GpuResidentConstraintDeviceState constraint_{};
  GpuForwardDeviceState forward_{};
  GpuPreconditionerDeviceState preconditioner_{};
  GpuDiagnosticsDeviceState diagnostics_{};
  GpuIterationDeviceState iteration_{};
  double* residuals_ = nullptr;
  bool pinned_scalars_ = false;
#if defined(VMECPP_COOPERATIVE_ENERGIES)
  bool cooperative_energies_supported_ = false;
#endif
#if defined(VMECPP_COOPERATIVE_RESIDUALS)
  bool cooperative_supported_ = false;
#endif
  double* scalar_packet_ = nullptr;
  double* host_scalar_packet_ = nullptr;
  double* host_jacobian_ = nullptr;
#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
  double* controller_state_ = nullptr;
  unsigned long long* controller_stats_ = nullptr;
  double controller_delta_s_ = 0.0;
  bool controller_seeded_ = false;
#if defined(VMECPP_FUSE_DAMPING_TAIL)
  GpuSolverTailControllerControl* tail_control_ = nullptr;
  GpuSolverTailControllerControl* host_tail_control_ = nullptr;
  int* tail_controller_status_ = nullptr;
  unsigned long long* tail_controller_stats_ = nullptr;
  bool tail_controller_armed_ = false;
#endif
#if defined(VMECPP_EVOLVE_DAMPING_BYPASS)
  int* controller_failure_ = nullptr;
#endif
#if defined(VMECPP_PINNED_CPU_LOG)
  unsigned long long* pinned_log_stats_ = nullptr;
#endif
#endif
};
}  // namespace

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
#if defined(VMECPP_COORDINATE_CAPTURES)
  std::shared_lock<std::shared_mutex> allocation_guard(CaptureGate());
#endif
  auto* context = new Solver;
  try {
    context->Initialize(setup);
  } catch (...) {
    delete context;
    throw;
  }
#if defined(VMECPP_COORDINATE_CAPTURES)
  ++thread_activity[0];
#endif
  return context;
}
extern "C" void DestroyGpuSolver(void* m_context) {
#if defined(VMECPP_COORDINATE_CAPTURES)
  std::shared_lock<std::shared_mutex> allocation_guard(CaptureGate());
#endif
  delete static_cast<Solver*>(m_context);
}
extern "C" bool GpuSolverCaptureCoordinated() {
#if defined(VMECPP_COORDINATE_CAPTURES)
  return true;
#else
  return false;
#endif
}
extern "C" void GpuSolverThreadActivity(std::uint64_t* m_counts) {
  if (m_counts == nullptr) return;
  for (std::size_t i = 0; i < 3; ++i) {
#if defined(VMECPP_COORDINATE_CAPTURES)
    m_counts[i] = thread_activity[i];
#else
    m_counts[i] = 0;
#endif
  }
}
extern "C" void UploadGpuSolverState(void* m_context,
                                     const GpuStepInput& state) {
  static_cast<Solver*>(m_context)->UploadState(state);
}
extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output) {
  static_cast<Solver*>(m_context)->Run(control, m_output);
}
extern "C" void AdvanceGpuSolver(void* m_context,
                                 const std::array<double, 3>& scalars) {
  static_cast<Solver*>(m_context)->Advance(scalars);
}
#if defined(VMECPP_LIVE_DAMPING_CONTROLLER)
#if defined(VMECPP_FUSE_DAMPING_TAIL)
extern "C" void ArmGpuSolverTailController(
    void* m_context, const GpuSolverTailControllerControl& control) {
  static_cast<Solver*>(m_context)->ArmTailController(control);
}
extern "C" void CancelGpuSolverTailController(void* m_context) {
  static_cast<Solver*>(m_context)->CancelTailController();
}
extern "C" void ReadGpuSolverTailControllerStats(
    void* m_context, std::array<unsigned long long, 6>& m_stats) {
  static_cast<Solver*>(m_context)->TailControllerStats(m_stats);
}
#endif
extern "C" void SeedGpuSolverController(
    void* m_context, const std::array<double, 10>& history, double previous_fsq,
    const std::array<double, 3>& preconditioned_residuals) {
  static_cast<Solver*>(m_context)->SeedControllerState(
      history, previous_fsq, preconditioned_residuals);
}
extern "C" void SeedGpuSolverControllerState(
    void* m_context, const GpuSolverControllerState& state,
    const std::array<double, 3>& preconditioned_residuals) {
  static_cast<Solver*>(m_context)->SeedControllerState(
      state.history_, state.previous_fsq_, preconditioned_residuals,
      state.scalars_);
}
extern "C" void DownloadGpuSolverControllerState(
    void* m_context, GpuSolverControllerState& m_output) {
  static_cast<Solver*>(m_context)->DownloadControllerState(m_output);
}
extern "C" void AdvanceGpuSolverController(
    void* m_context, const GpuSolverControllerReference& reference) {
  static_cast<Solver*>(m_context)->AdvanceController(reference);
}
extern "C" void ReadGpuSolverControllerStats(
    void* m_context, std::array<unsigned long long, 7>& m_stats) {
  static_cast<Solver*>(m_context)->ControllerStats(m_stats);
}
#if defined(VMECPP_PINNED_CPU_LOG)
extern "C" void ReadGpuSolverPinnedLogStats(
    void* m_context, std::array<unsigned long long, 4>& m_stats) {
  static_cast<Solver*>(m_context)->PinnedLogStats(m_stats);
}
#endif
#endif
extern "C" void DownloadGpuSolverState(void* m_context,
                                       GpuStepOutput& m_output) {
  static_cast<Solver*>(m_context)->DownloadState(m_output);
}
extern "C" void DownloadGpuSolverCache(void* m_context,
                                       GpuPreconditionerOutput& m_output) {
  static_cast<Solver*>(m_context)->DownloadCache(m_output);
}
extern "C" void SaveGpuSolverBackup(void* m_context) {
  static_cast<Solver*>(m_context)->SaveBackup();
}
extern "C" void RestoreGpuSolverBackup(void* m_context) {
  static_cast<Solver*>(m_context)->RestoreBackup();
}
extern "C" void DownloadGpuSolverGeometry(void* m_context,
                                          GpuSolverGeometry which,
                                          GpuPreparationOutput& m_output) {
  static_cast<Solver*>(m_context)->DownloadGeometry(which, m_output);
}
extern "C" void DownloadGpuSolverForces(void* m_context,
                                        GpuPreparationOutput& m_output) {
  static_cast<Solver*>(m_context)->DownloadForces(m_output);
}
extern "C" void DownloadGpuSolverPrintout(void* m_context, double* m_dvds,
                                          std::size_t count,
                                          std::array<double, 2>& m_axis) {
  static_cast<Solver*>(m_context)->DownloadPrintout(m_dvds, count, m_axis);
}
