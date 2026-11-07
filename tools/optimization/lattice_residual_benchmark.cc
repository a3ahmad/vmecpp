// SPDX-License-Identifier: MIT
// Production-shape component comparison, never a whole-solver speedup claim.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_cooperative_residual.h"
#include "gpu_lattice_residual.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
T Symbol(void* library, const char* name) {
  auto result = reinterpret_cast<T>(dlsym(library, name));
  if (result == nullptr) throw std::runtime_error(name);
  return result;
}
struct Api {
  void* original_;
  void* cooperative_;
  void* candidate_;
  decltype(&LaunchGpuIterationResidualsConditional) ordered_;
  decltype(&LaunchGpuIterationResidualsOptimistic) fixed_;
  decltype(&LaunchGpuCooperativeResiduals) cooperative_launch_;
  decltype(&GpuCooperativeResidualsSupported) supported_;
  decltype(&LaunchGpuIterationResidualsLattice) lattice_;
  decltype(&LaunchGpuIterationResidualsLatticePoisoned) poisoned_;
  decltype(&ReadGpuLatticeResidualStats) lattice_stats_;
  decltype(&ReadGpuLatticeResidualFlags) lattice_flags_;
  decltype(&ReadGpuResidualRelaxationStats) fixed_stats_;
  decltype(&ReadGpuCooperativeResidualStats) cooperative_stats_;
  explicit Api(const char* original, const char* cooperative,
               const char* candidate) {
    original_ = dlopen(original, RTLD_NOW | RTLD_GLOBAL);
    if (original_ == nullptr) throw std::runtime_error(dlerror());
    cooperative_ = dlopen(cooperative, RTLD_NOW | RTLD_GLOBAL);
    if (cooperative_ == nullptr) throw std::runtime_error(dlerror());
    candidate_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (candidate_ == nullptr) throw std::runtime_error(dlerror());
    ordered_ = Symbol<decltype(ordered_)>(
        original_, "LaunchGpuIterationResidualsConditional");
    if (dlsym(RTLD_DEFAULT, "LaunchGpuIterationResidualsConditional") !=
        dlsym(original_, "LaunchGpuIterationResidualsConditional"))
      throw std::runtime_error(
          "Original conditional fallback symbol interposed");
    fixed_ = Symbol<decltype(fixed_)>(original_,
                                      "LaunchGpuIterationResidualsOptimistic");
    cooperative_launch_ = Symbol<decltype(cooperative_launch_)>(
        cooperative_, "LaunchGpuCooperativeResiduals");
    supported_ = Symbol<decltype(supported_)>(
        cooperative_, "GpuCooperativeResidualsSupported");
    lattice_ = Symbol<decltype(lattice_)>(candidate_,
                                          "LaunchGpuIterationResidualsLattice");
    poisoned_ = Symbol<decltype(poisoned_)>(
        candidate_, "LaunchGpuIterationResidualsLatticePoisoned");
    lattice_stats_ = Symbol<decltype(lattice_stats_)>(
        candidate_, "ReadGpuLatticeResidualStats");
    lattice_flags_ = Symbol<decltype(lattice_flags_)>(
        candidate_, "ReadGpuLatticeResidualFlags");
    fixed_stats_ = Symbol<decltype(fixed_stats_)>(
        original_, "ReadGpuResidualRelaxationStats");
    cooperative_stats_ = Symbol<decltype(cooperative_stats_)>(
        cooperative_, "ReadGpuCooperativeResidualStats");
  }
  // Keep DSOs loaded until process exit: their TLS workspaces own destructors.
};

template <class Capture>
const double* Expected(const Capture& capture, int index) {
  return capture.expected_.data() +
         std::accumulate(capture.output_.sizes_.begin(),
                         capture.output_.sizes_.begin() + index,
                         std::size_t{0});
}

enum class Method {
  kOrdered,
  kFixed12,
  kFixed24,
  kCooperative32,
  kLattice,
  kPoisoned
};
const char* Name(Method method) {
  switch (method) {
    case Method::kOrdered:
      return "original_ordered";
    case Method::kFixed12:
      return "production_fixed12";
    case Method::kFixed24:
      return "production_fixed24";
    case Method::kCooperative32:
      return "production_cooperative32";
    case Method::kLattice:
      return "lattice";
    case Method::kPoisoned:
      return "lattice_poisoned";
  }
  return "unknown";
}

struct Resources {
  const Api& api_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
  GpuIterationDeviceState state_{};
  std::array<std::vector<double>, 6> host_forces_;
  std::vector<void*> allocations_;
  double* snapshot_ = nullptr;
  double* history_ = nullptr;
  std::array<cudaGraph_t, 6> graphs_{};
  std::array<cudaGraphExec_t, 6> executables_{};
  bool include_edge_;

  Resources(const Api& api, bool include_edge)
      : api_(api), include_edge_(include_edge) {}
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto executable : executables_)
      if (executable != nullptr) cudaGraphExecDestroy(executable);
    for (auto graph : graphs_)
      if (graph != nullptr) cudaGraphDestroy(graph);
    for (void* pointer : allocations_) cudaFree(pointer);
    if (start_ != nullptr) cudaEventDestroy(start_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t count) {
    double* result = nullptr;
    Check(cudaMalloc(&result, count * sizeof(double)));
    allocations_.push_back(result);
    return result;
  }
  template <class Capture>
  void Initialize(const Capture& capture, bool after, int variant) {
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&end_));
    state_.dimensions_ = capture.input_.dimensions_;
    state_.stream_ = stream_;
    for (int channel = 0; channel < 6; ++channel) {
      const auto count = capture.input_.sizes_[channel];
      const double* values = after ? Expected(capture, channel + 12)
                                   : capture.input_.data_[channel];
      host_forces_[channel].assign(values, values + count);
      if (variant == 1) {
        for (std::size_t index = 0; index < count; ++index)
          host_forces_[channel][index] = index % 2 ? -0.0 : 0.0;
      }
      if (variant == 2)
        std::fill(host_forces_[channel].begin(), host_forces_[channel].end(),
                  1e-160);
      if (variant == 3 && channel % 2 == 0)
        host_forces_[channel][0] = std::numeric_limits<double>::quiet_NaN();
      if (variant == 4)
        std::fill(host_forces_[channel].begin(), host_forces_[channel].end(),
                  1e308);
      if (variant == 5 && (channel == 0 || channel == 2))
        host_forces_[channel].back() = std::numeric_limits<double>::quiet_NaN();
      if (variant == 6 && channel == 0)
        host_forces_[channel][host_forces_[channel].size() / 2] =
            std::numeric_limits<double>::quiet_NaN();
      state_.forces_[channel] = Allocate(count);
      Check(cudaMemcpyAsync(
          state_.forces_[channel], host_forces_[channel].data(),
          count * sizeof(double), cudaMemcpyHostToDevice, stream_));
    }
    state_.residuals_ = Allocate(3);
    snapshot_ = Allocate(3);
    history_ = Allocate(3 * 100);
    Check(cudaStreamSynchronize(stream_));
  }
  void Launch(Method method, bool outer) {
    switch (method) {
      case Method::kOrdered:
        api_.ordered_(state_, include_edge_, nullptr);
        break;
      case Method::kFixed12:
        api_.fixed_(state_, include_edge_, 12, !outer);
        break;
      case Method::kFixed24:
        api_.fixed_(state_, include_edge_, 24, !outer);
        break;
      case Method::kCooperative32:
        api_.cooperative_launch_(state_, include_edge_, 32);
        break;
      case Method::kLattice:
        api_.lattice_(state_, include_edge_);
        break;
      case Method::kPoisoned:
        api_.poisoned_(state_, include_edge_);
        break;
    }
  }
  void Direct(Method method) {
    Check(
        cudaMemsetAsync(state_.residuals_, 0x7f, 3 * sizeof(double), stream_));
    Launch(method, false);
    Check(cudaMemcpyAsync(snapshot_, state_.residuals_, 3 * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream_));
  }
  void Capture(Method method) {
    const int index = static_cast<int>(method);
    Direct(method);
    Check(cudaStreamSynchronize(stream_));
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      Check(cudaMemsetAsync(state_.residuals_, 0x7f, 3 * sizeof(double),
                            stream_));
      Launch(method, true);
      Check(cudaMemcpyAsync(snapshot_, state_.residuals_, 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream_));
      const auto status = cudaStreamEndCapture(stream_, &graphs_[index]);
      capturing = false;
      Check(status);
      Check(cudaGraphInstantiate(&executables_[index], graphs_[index], 0));
    } catch (...) {
      if (capturing) cudaStreamEndCapture(stream_, &graphs_[index]);
      throw;
    }
  }
  std::array<double, 3> Download() {
    std::array<double, 3> result{};
    Check(cudaMemcpyAsync(result.data(), snapshot_, sizeof(result),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    return result;
  }
  void UnchangedInputs() {
    for (int channel = 0; channel < 6; ++channel) {
      std::vector<double> actual(host_forces_[channel].size());
      Check(cudaMemcpyAsync(actual.data(), state_.forces_[channel],
                            actual.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream_));
      Check(cudaStreamSynchronize(stream_));
      if (std::memcmp(actual.data(), host_forces_[channel].data(),
                      actual.size() * sizeof(double)) != 0)
        throw std::runtime_error("Force input changed");
    }
  }
};

void Equal(const std::array<double, 3>& expected,
           const std::array<double, 3>& actual) {
  if (std::memcmp(expected.data(), actual.data(), sizeof(expected)) != 0)
    throw std::runtime_error("Residual result bits differ");
}

void Exercise(const Api& api, Resources& resources,
              const std::vector<Method>& methods, const double* captured_cpu,
              bool nonfinite, bool require_fallback, bool timing,
              const std::string& label, bool& m_first) {
  resources.Direct(Method::kOrdered);
  const auto reference = resources.Download();
  if (captured_cpu != nullptr) {
    std::array<double, 3> expected{};
    std::copy_n(captured_cpu, 3, expected.begin());
    Equal(expected, reference);
  }
  for (Method method : methods) {
    resources.Direct(method);
    Equal(reference, resources.Download());
    resources.Capture(method);
    Check(cudaGraphLaunch(resources.executables_[static_cast<int>(method)],
                          resources.stream_));
    Equal(reference, resources.Download());
  }
  resources.Direct(Method::kPoisoned);
  Equal(reference, resources.Download());
  std::array<int, 3> poisoned_flags{};
  api.lattice_flags_(poisoned_flags);
  if (std::any_of(poisoned_flags.begin(), poisoned_flags.end(),
                  [](int value) { return value == 0; }))
    throw std::runtime_error("Poisoned proposal escaped original fallback");
  resources.Capture(Method::kPoisoned);
  Check(cudaGraphLaunch(
      resources.executables_[static_cast<int>(Method::kPoisoned)],
      resources.stream_));
  Equal(reference, resources.Download());
  api.lattice_flags_(poisoned_flags);
  if (std::any_of(poisoned_flags.begin(), poisoned_flags.end(),
                  [](int value) { return value == 0; }))
    throw std::runtime_error("Captured poisoned proposal escaped fallback");
  resources.Direct(Method::kLattice);
  Equal(reference, resources.Download());
  std::array<int, 3> flags{};
  api.lattice_flags_(flags);
  if (require_fallback && std::any_of(flags.begin(), flags.end(),
                                      [](int value) { return value == 0; }))
    throw std::runtime_error(
        "Unsupported subnormal proposal did not fall back");
  resources.UnchangedInputs();
  if (!timing) {
    std::cout << (m_first ? "" : ",") << "{\"case\":\"" << label
              << "\",\"correctness_only\":true,\"nonfinite_stress_fixture\":"
              << (nonfinite ? "true" : "false")
              << ",\"all_output_bits_exact\":true}";
    m_first = false;
    return;
  }
  std::mt19937_64 generator(982434);
  for (int sample = 0; sample < 7; ++sample) {
    std::vector<std::pair<Method, bool>> order;
    for (Method method : methods) {
      order.emplace_back(method, false);
      order.emplace_back(method, true);
    }
    std::shuffle(order.begin(), order.end(), generator);
    int position = 0;
    for (auto [method, graph] : order) {
      std::array<unsigned long long, 4> before{}, after{};
      std::array<unsigned long long, 4> fixed_before{}, fixed_after{};
      std::array<unsigned long long, 37> cooperative_before{},
          cooperative_after{};
      api.lattice_stats_(before);
      if (method == Method::kFixed12 || method == Method::kFixed24)
        api.fixed_stats_(fixed_before);
      if (method == Method::kCooperative32)
        api.cooperative_stats_(cooperative_before);
      Check(cudaEventRecord(resources.start_, resources.stream_));
      for (int repeat = 0; repeat < 100; ++repeat) {
        if (graph)
          Check(
              cudaGraphLaunch(resources.executables_[static_cast<int>(method)],
                              resources.stream_));
        else
          resources.Direct(method);
        Check(cudaMemcpyAsync(resources.history_ + 3 * repeat,
                              resources.snapshot_, 3 * sizeof(double),
                              cudaMemcpyDeviceToDevice, resources.stream_));
      }
      Check(cudaEventRecord(resources.end_, resources.stream_));
      Check(cudaEventSynchronize(resources.end_));
      float milliseconds;
      Check(cudaEventElapsedTime(&milliseconds, resources.start_,
                                 resources.end_));
      api.lattice_stats_(after);
      if (method == Method::kFixed12 || method == Method::kFixed24)
        api.fixed_stats_(fixed_after);
      if (method == Method::kCooperative32)
        api.cooperative_stats_(cooperative_after);
      std::array<double, 300> actual{};
      Check(cudaMemcpyAsync(actual.data(), resources.history_, sizeof(actual),
                            cudaMemcpyDeviceToHost, resources.stream_));
      Check(cudaStreamSynchronize(resources.stream_));
      for (int repeat = 0; repeat < 100; ++repeat) {
        std::array<double, 3> values{};
        std::copy_n(actual.data() + 3 * repeat, 3, values.begin());
        Equal(reference, values);
      }
      std::cout << (m_first ? "" : ",") << "{\"case\":\"" << label
                << "\",\"sample\":" << sample << ",\"order\":" << position++
                << ",\"method\":\"" << Name(method) << "\",\"mode\":\""
                << (graph ? "outer_graph" : "standalone")
                << "\",\"repeats\":100,\"cuda_event_ms\":" << milliseconds
                << ",\"lattice_calls\":" << after[0] - before[0]
                << ",\"lattice_accepted_components\":" << after[1] - before[1]
                << ",\"lattice_fallback_components\":" << after[2] - before[2]
                << ",\"fixed_or_cooperative_calls\":"
                << fixed_after[0] - fixed_before[0] + cooperative_after[0] -
                       cooperative_before[0]
                << ",\"fixed_or_cooperative_fallback_components\":["
                << fixed_after[1] - fixed_before[1] + cooperative_after[1] -
                       cooperative_before[1]
                << ','
                << fixed_after[2] - fixed_before[2] + cooperative_after[2] -
                       cooperative_before[2]
                << ','
                << fixed_after[3] - fixed_before[3] + cooperative_after[3] -
                       cooperative_before[3]
                << ']' << ",\"all_output_bits_exact\":true}";
      m_first = false;
    }
  }
  resources.UnchangedInputs();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 5 || argc > 7)
      throw std::runtime_error(
          "Usage: benchmark capture.bin original.so cooperative.so "
          "candidate.so [check-only|lattice-only] [before|after]");
    const bool check_only = argc > 5;
    const bool lattice_only =
        argc > 5 && std::string(argv[5]) == "lattice-only";
    const int stage_begin = argc > 6 && std::string(argv[6]) == "after" ? 1 : 0;
    const int stage_end = argc > 6 ? stage_begin + 1 : 2;
    const Api api(argv[2], argv[3], argv[4]);
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    std::cout
        << std::setprecision(17)
        << "{\"scope\":\"production-shape component experiment; one unique "
           "ns99 capture; no solver speedup claim\",\"timing_scope\":\"same "
           "forces/output poison/two24byteD2Dsnapshots; standalone fixed12/24 "
           "use innergraph, outer mode embeds kernels; excludes "
           "downloads\",\"samples\":[";
    bool first = true;
    for (int stage = stage_begin; stage < stage_end; ++stage) {
      for (bool include_edge : {true, false}) {
        for (int variant = 0; variant < (check_only ? 7 : 1); ++variant) {
          Resources resources(api, include_edge);
          resources.Initialize(capture, stage != 0, variant);
          std::vector<Method> methods{Method::kLattice};
          if (!lattice_only) {
            methods.insert(methods.begin(), {Method::kOrdered, Method::kFixed12,
                                             Method::kFixed24});
            if (!api.supported_(resources.state_))
              throw std::runtime_error(
                  "Cooperative production reference unsupported");
            methods.push_back(Method::kCooperative32);
          }
          const auto label = std::string(stage ? "after" : "before") +
                             (include_edge ? "_edge" : "_no_edge") +
                             "_variant" + std::to_string(variant);
          Exercise(api, resources, methods,
                   include_edge && variant == 0
                       ? Expected(capture, stage ? 19 : 18)
                       : nullptr,
                   variant >= 3, variant == 2 || variant == 3 || variant == 4,
                   !check_only, label, first);
        }
      }
    }
    std::cout << "],\"all_output_bits_exact\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
