// SPDX-License-Identifier: MIT
// Ordinary energy-only experiment. Tree guesses never replace the original
// recurrence without a finite, bitwise certificate of every original prefix.
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cub/block/block_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>

#include "gpu_cooperative_energies.h"

namespace {
constexpr int kThreads = 256;
constexpr int kMaximumTerms = kThreads * kThreads;
using BlockScan = cub::BlockScan<double, kThreads>;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Shape {
  std::array<int, 2> counts_{};
  std::array<int, 2> tiles_{};
  int grid_ = 0;
};
bool GetShape(const GpuDiagnosticsDeviceState& state, Shape& m_shape) {
  const auto& d = state.dimensions_;
  if (state.scalars_[1] != 0.0 || d[0] < 2 || d[3] != d[0] - 1 ||
      d[4] != d[0] || d[5] < 1 || d[6] < 1 || d[0] - 1 > kMaximumTerms)
    return false;
  const auto half = static_cast<std::size_t>(d[0] - 1) * d[5];
  if (half > kMaximumTerms ||
      static_cast<std::size_t>(d[6]) > kMaximumTerms / half)
    return false;
  m_shape.counts_ = {d[0] - 1, static_cast<int>(half * d[6])};
  for (int i = 0; i < 2; ++i) {
    m_shape.tiles_[i] = (m_shape.counts_[i] + kThreads - 1) / kThreads;
    m_shape.grid_ += m_shape.tiles_[i];
  }
  return true;
}
struct Parameters {
  cuda::std::array<const double*, 8> data_;
  cuda::std::array<int, 2> counts_;
  cuda::std::array<int, 2> tiles_;
  double* prefix_;
  double* tile_totals_;
  double* tile_carries_;
  double* sums_;
  double* output_;
  int* invalid_;
  int* done_;
  unsigned long long* statistics_;
  int n_theta_;
  int max_rounds_;
  double delta_s_;
};

// Literal expressions and index order from gpu_diagnostics.cu OrderedSums.
// component0 volume, component1 thermal, component2 magnetic.
__device__ double Term(const Parameters& p, int component, int i) {
  const auto& v = p.data_;
  if (component == 0) return v[7][i];
  if (component == 1) return (v[6][i] / 1.0) * v[7][i];
  const double magnetic = 0.5 * (v[1][i] * v[3][i] + v[2][i] * v[4][i]);
  return v[0][i] * magnetic * v[5][i % p.n_theta_];
}

__device__ double ScanGuess(const Parameters& m_p, int component, int tile,
                            double value, BlockScan::TempStorage& m_temporary,
                            cooperative_groups::grid_group grid) {
  const int tile_offset = component == 0 ? 0 : m_p.tiles_[0];
  double local_prefix;
  double aggregate;
  BlockScan(m_temporary).InclusiveSum(value, local_prefix, aggregate);
  if (threadIdx.x == 0) m_p.tile_totals_[tile_offset + tile] = aggregate;
  grid.sync();
  // This branch is block-uniform. All256 lanes participate in its collective.
  if (tile == 0) {
    const double total = threadIdx.x < m_p.tiles_[component]
                             ? m_p.tile_totals_[tile_offset + threadIdx.x]
                             : 0.0;
    double carry;
    BlockScan(m_temporary).ExclusiveSum(total, carry);
    if (threadIdx.x < m_p.tiles_[component])
      m_p.tile_carries_[tile_offset + threadIdx.x] = carry;
  }
  grid.sync();
  return __dadd_rn(m_p.tile_carries_[tile_offset + tile], local_prefix);
}

__global__ void PersistentEnergies(Parameters m_p) {
  __shared__ BlockScan::TempStorage temporary;
  const auto grid = cooperative_groups::this_grid();
  const int component = blockIdx.x < m_p.tiles_[0] ? 0 : 1;
  const int tile = component == 0 ? blockIdx.x : blockIdx.x - m_p.tiles_[0];
  const int local = tile * kThreads + threadIdx.x;
  const int offset = component == 0 ? 0 : m_p.counts_[0];
  const int index = offset + local;
  const bool valid = local < m_p.counts_[component];
  const double term = valid ? Term(m_p, component + 1, local) : 0.0;
  const double initial = ScanGuess(m_p, component, tile, term, temporary, grid);
  if (valid) m_p.prefix_[index] = initial;
  grid.sync();
  int rounds = 0;
  for (; rounds < m_p.max_rounds_;) {
    double error = 0.0;
    double previous_prefix = 0.0;
    if (valid) {
      previous_prefix = m_p.prefix_[index];
      const double previous = local == 0 ? 0.0 : m_p.prefix_[index - 1];
      error = __dsub_rn(__dadd_rn(previous, term), previous_prefix);
    }
    // Both barriers occur in every block, including the short thermal block.
    // All old-prefix reads precede every new-prefix write.
    const double correction =
        ScanGuess(m_p, component, tile, error, temporary, grid);
    if (valid) m_p.prefix_[index] = __dadd_rn(previous_prefix, correction);
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x < 2) m_p.invalid_[threadIdx.x] = 0;
    grid.sync();
    if (valid) {
      const double previous = local == 0 ? 0.0 : m_p.prefix_[index - 1];
      const double target = __dadd_rn(previous, term);
      const double prefix = m_p.prefix_[index];
      if (!isfinite(term) || !isfinite(prefix) ||
          __double_as_longlong(target) != __double_as_longlong(prefix))
        atomicOr(m_p.invalid_ + component, 1);
    }
    grid.sync();
    ++rounds;
    if (blockIdx.x == 0 && threadIdx.x == 0)
      *m_p.done_ = !(m_p.invalid_[0] || m_p.invalid_[1]);
    grid.sync();
    if (*m_p.done_) break;
  }
  if (tile == 0 && threadIdx.x == 0) {
    if (m_p.invalid_[component] == 0)
      m_p.sums_[component + 1] =
          m_p.prefix_[offset + m_p.counts_[component] - 1];
    else
      atomicAdd(m_p.statistics_ + component + 1, 1ULL);
    if (component == 0) {
      atomicAdd(m_p.statistics_, 1ULL);
      atomicAdd(m_p.statistics_ + 3 + rounds, 1ULL);
    }
  }
}

// Also recompute the short original volume recurrence, even though ordinary
// Finish retains cached output volume. This preserves raw sums[0..5] exactly.
__global__ void OrderedFallback(Parameters m_p) {
  __shared__ cuda::std::array<double, 128> terms;
  const int component = blockIdx.x;
  if (component > 0 && m_p.invalid_[component - 1] == 0) return;
  const int count = m_p.counts_[component == 2 ? 1 : 0];
  double sum = 0.0;
  for (int start = 0; start < count; start += 128) {
    const int i = start + threadIdx.x;
    if (i < count) terms[threadIdx.x] = Term(m_p, component, i);
    __syncthreads();
    if (threadIdx.x == 0) {
      const int remaining = count - start;
      const int length = remaining < 128 ? remaining : 128;
      for (int index = 0; index < length; ++index) sum += terms[index];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) m_p.sums_[component] = sum;
}

__global__ void Finish(Parameters m_p) {
  const double thermal = 0.0 + m_p.sums_[1] * m_p.delta_s_;
  const double magnetic = 0.0 + fabs(m_p.sums_[2]) * m_p.delta_s_;
  m_p.output_[1] = thermal;
  m_p.output_[2] = magnetic;
  m_p.output_[3] = magnetic + thermal / (0.0 - 1.0);
}

bool Residency(const Shape& shape, int& m_device, int& m_active,
               int& m_multiprocessors) {
  Check(cudaGetDevice(&m_device));
  int cooperative;
  Check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                               m_device));
  if (!cooperative) return false;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &m_active, PersistentEnergies, kThreads, 0));
  Check(cudaDeviceGetAttribute(&m_multiprocessors,
                               cudaDevAttrMultiProcessorCount, m_device));
  return shape.grid_ <= m_active * m_multiprocessors;
}

struct Workspace {
  Shape shape_;
  int device_ = -1;
  int active_ = 0;
  int multiprocessors_ = 0;
  cudaStream_t stream_ = nullptr;
  double* prefix_ = nullptr;
  double* tile_totals_ = nullptr;
  double* tile_carries_ = nullptr;
  int* invalid_ = nullptr;
  int* done_ = nullptr;
  unsigned long long* statistics_ = nullptr;

  void Clear() {
    for (void* pointer :
         {static_cast<void*>(prefix_), static_cast<void*>(tile_totals_),
          static_cast<void*>(tile_carries_), static_cast<void*>(invalid_),
          static_cast<void*>(done_), static_cast<void*>(statistics_)})
      if (pointer != nullptr) cudaFree(pointer);
    prefix_ = tile_totals_ = tile_carries_ = nullptr;
    invalid_ = done_ = nullptr;
    statistics_ = nullptr;
    shape_ = {};
  }
  ~Workspace() { Clear(); }

  void Resize(const Shape& shape, cudaStream_t stream, bool capturing) {
    int device;
    Check(cudaGetDevice(&device));
    if (prefix_ != nullptr && device_ != device)
      throw std::runtime_error("Cooperative energies cannot change device");
    if (prefix_ != nullptr && shape.counts_ == shape_.counts_) {
      if (stream_ != stream) {
        if (capturing)
          throw std::runtime_error("Warm cooperative energies stream first");
        Check(cudaDeviceSynchronize());
        stream_ = stream;
      }
      return;
    }
    if (capturing)
      throw std::runtime_error("Warm cooperative energies before capture");
    int active;
    int multiprocessors;
    if (!Residency(shape, device, active, multiprocessors))
      throw std::runtime_error("Cooperative energies exceed device residency");
    Clear();
    device_ = device;
    active_ = active;
    multiprocessors_ = multiprocessors;
    stream_ = stream;
    Check(cudaMalloc(&prefix_,
                     (shape.counts_[0] + shape.counts_[1]) * sizeof(double)));
    Check(cudaMalloc(&tile_totals_, shape.grid_ * sizeof(double)));
    Check(cudaMalloc(&tile_carries_, shape.grid_ * sizeof(double)));
    Check(cudaMalloc(&invalid_, 2 * sizeof(int)));
    Check(cudaMalloc(&done_, sizeof(int)));
    Check(cudaMalloc(&statistics_, 36 * sizeof(unsigned long long)));
    Check(cudaMemsetAsync(statistics_, 0, 36 * sizeof(unsigned long long),
                          stream));
    shape_ = shape;
  }
};
thread_local Workspace workspace;
}  // namespace

extern "C" bool GpuCooperativeEnergiesSupported(
    const GpuDiagnosticsDeviceState& state) {
  Shape shape;
  if (!GetShape(state, shape)) return false;
  int device;
  int active;
  int multiprocessors;
  return Residency(shape, device, active, multiprocessors);
}

extern "C" void LaunchGpuCooperativeEnergies(GpuDiagnosticsDeviceState& m_state,
                                             int max_rounds) {
  Shape shape;
  if (max_rounds < 1 || max_rounds > 32 || !GetShape(m_state, shape))
    throw std::runtime_error("Unsupported cooperative energy configuration");
  if (m_state.sums_ == nullptr || m_state.output_ == nullptr)
    throw std::runtime_error("Null cooperative energies output/scratch");
  Parameters p{};
  for (int i = 0; i < 8; ++i) {
    if (m_state.data_[i] == nullptr)
      throw std::runtime_error("Null cooperative energies input");
    p.data_[i] = m_state.data_[i];
  }
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Cooperative energy capture invalidated");
  workspace.Resize(shape, stream, capture == cudaStreamCaptureStatusActive);
  for (int i = 0; i < 2; ++i) {
    p.counts_[i] = shape.counts_[i];
    p.tiles_[i] = shape.tiles_[i];
  }
  p.prefix_ = workspace.prefix_;
  p.tile_totals_ = workspace.tile_totals_;
  p.tile_carries_ = workspace.tile_carries_;
  p.invalid_ = workspace.invalid_;
  p.done_ = workspace.done_;
  p.statistics_ = workspace.statistics_;
  p.sums_ = m_state.sums_;
  p.output_ = m_state.output_;
  p.n_theta_ = m_state.dimensions_[5];
  p.delta_s_ = m_state.scalars_[0];
  p.max_rounds_ = max_rounds;
  std::array<void*, 1> arguments{&p};
  Check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(PersistentEnergies),
                                    dim3(shape.grid_), dim3(kThreads),
                                    arguments.data(), 0, stream));
  OrderedFallback<<<3, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  Finish<<<1, 1, 0, stream>>>(p);
  Check(cudaGetLastError());
}

extern "C" void ReadGpuCooperativeEnergiesStats(
    std::array<unsigned long long, 36>& m_statistics) {
  if (workspace.prefix_ == nullptr)
    throw std::runtime_error("No cooperative energies experiment ran");
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(m_statistics.data(), workspace.statistics_,
                   sizeof(m_statistics), cudaMemcpyDeviceToHost));
}

extern "C" void ReadGpuCooperativeEnergiesLayout(std::array<int, 6>& m_layout) {
  if (workspace.prefix_ == nullptr)
    throw std::runtime_error("No cooperative energies experiment ran");
  m_layout = {workspace.shape_.grid_,     kThreads,
              workspace.active_,          workspace.multiprocessors_,
              workspace.shape_.tiles_[0], workspace.shape_.tiles_[1]};
}
