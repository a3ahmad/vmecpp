// SPDX-License-Identifier: MIT
// Parallel scans propose prefixes. Only a finite, bitwise certificate of every
// original sequential recurrence can accept them; failures use ordered sums.
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cub/block/block_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>

#include "gpu_cooperative_residual.h"

namespace {
constexpr int kThreads = 256;
constexpr int kComponents = 3;
constexpr int kMaximumRounds = 32;
using BlockScan = cub::BlockScan<double, kThreads>;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

struct Parameters {
  cuda::std::array<const double*, 6> forces_;
  double* terms_;
  double* prefix_;
  double* tile_totals_;
  double* tile_carries_;
  double* output_;
  int* invalid_;
  int* done_;
  unsigned long long* statistics_;
  int stride_;
  int rz_count_;
  int tiles_;
  int max_rounds_;
};

// Every block executes this function and both grid barriers. The block-uniform
// tile-zero branch scans at most256 tile totals; padded lanes contribute+0.
__device__ double ScanGuess(const Parameters& m_p, double value,
                            BlockScan::TempStorage& m_temporary,
                            cooperative_groups::grid_group grid) {
  const int component = blockIdx.x / m_p.tiles_;
  const int tile = blockIdx.x % m_p.tiles_;
  double local_prefix;
  double aggregate;
  BlockScan(m_temporary).InclusiveSum(value, local_prefix, aggregate);
  if (threadIdx.x == 0)
    m_p.tile_totals_[component * m_p.tiles_ + tile] = aggregate;
  grid.sync();
  if (tile == 0) {
    const double total =
        threadIdx.x < m_p.tiles_
            ? m_p.tile_totals_[component * m_p.tiles_ + threadIdx.x]
            : 0.0;
    double carry;
    BlockScan(m_temporary).ExclusiveSum(total, carry);
    if (threadIdx.x < m_p.tiles_)
      m_p.tile_carries_[component * m_p.tiles_ + threadIdx.x] = carry;
  }
  grid.sync();
  return __dadd_rn(m_p.tile_carries_[component * m_p.tiles_ + tile],
                   local_prefix);
}

__global__ void PersistentResidual(Parameters m_p) {
  __shared__ BlockScan::TempStorage temporary;
  const auto grid = cooperative_groups::this_grid();
  const int component = blockIdx.x / m_p.tiles_;
  const int tile = blockIdx.x % m_p.tiles_;
  const int local = tile * kThreads + threadIdx.x;
  const int index = component * m_p.stride_ + local;
  const bool valid = local < m_p.stride_;
  double term = 0.0;
  if (valid && (component == 2 || local / 2 < m_p.rz_count_)) {
    const double force = m_p.forces_[2 * component + local % 2][local / 2];
    term = __dmul_rn(force, force);
  }
  if (valid) m_p.terms_[index] = term;
  const double initial = ScanGuess(m_p, term, temporary, grid);
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
    // ScanGuess's first grid barrier follows every old-prefix read. Nothing
    // writes prefix until its second grid barrier has completed.
    const double correction = ScanGuess(m_p, error, temporary, grid);
    if (valid) m_p.prefix_[index] = __dadd_rn(previous_prefix, correction);
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x < kComponents)
      m_p.invalid_[threadIdx.x] = 0;
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
      *m_p.done_ = !(m_p.invalid_[0] || m_p.invalid_[1] || m_p.invalid_[2]);
    grid.sync();
    // done is one global decision made after every certificate write. All
    // threads observe it after the same barrier, including padded lanes.
    if (*m_p.done_) break;
  }
  if (tile == 0 && threadIdx.x == 0) {
    if (m_p.invalid_[component] == 0)
      m_p.output_[component] = m_p.prefix_[(component + 1) * m_p.stride_ - 1];
    else
      atomicAdd(m_p.statistics_ + 1 + component, 1ULL);
    if (component == 0) {
      atomicAdd(m_p.statistics_, 1ULL);
      atomicAdd(m_p.statistics_ + 4 + rounds, 1ULL);
    }
  }
}

struct Workspace {
  int stride_ = 0;
  int tiles_ = 0;
  int device_ = -1;
  int active_blocks_per_sm_ = 0;
  int multiprocessors_ = 0;
  cudaStream_t caller_stream_ = nullptr;
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  double* tile_totals_ = nullptr;
  double* tile_carries_ = nullptr;
  int* invalid_ = nullptr;
  int* done_ = nullptr;
  unsigned long long* statistics_ = nullptr;

  void Clear() {
    for (void* pointer :
         {static_cast<void*>(terms_), static_cast<void*>(prefix_),
          static_cast<void*>(tile_totals_), static_cast<void*>(tile_carries_),
          static_cast<void*>(invalid_), static_cast<void*>(done_),
          static_cast<void*>(statistics_)})
      if (pointer != nullptr) cudaFree(pointer);
    terms_ = prefix_ = tile_totals_ = tile_carries_ = nullptr;
    invalid_ = done_ = nullptr;
    statistics_ = nullptr;
    stride_ = tiles_ = 0;
  }
  ~Workspace() { Clear(); }

  void Resize(int stride, cudaStream_t stream, bool capturing) {
    int device;
    Check(cudaGetDevice(&device));
    if (stride_ != 0 && device != device_)
      throw std::runtime_error("Cooperative workspace cannot change device");
    if (stride == stride_ && device == device_) {
      if (stream != caller_stream_) {
        if (capturing)
          throw std::runtime_error("Warm cooperative caller stream first");
        Check(cudaDeviceSynchronize());
        caller_stream_ = stream;
      }
      return;
    }
    if (capturing)
      throw std::runtime_error("Warm cooperative workspace before capture");
    const int tiles = (stride + kThreads - 1) / kThreads;
    if (tiles > kThreads)
      throw std::runtime_error("Cooperative tile-total scan exceeds256 tiles");
    int cooperative;
    Check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                 device));
    if (!cooperative)
      throw std::runtime_error("Device does not support cooperative launch");
    int active_blocks;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, PersistentResidual, kThreads, 0));
    int multiprocessors;
    Check(cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device));
    if (kComponents * tiles > active_blocks * multiprocessors)
      throw std::runtime_error("Cooperative residual grid exceeds residency");
    Clear();
    device_ = device;
    active_blocks_per_sm_ = active_blocks;
    multiprocessors_ = multiprocessors;
    caller_stream_ = stream;
    Check(cudaMalloc(&terms_, kComponents * stride * sizeof(double)));
    Check(cudaMalloc(&prefix_, kComponents * stride * sizeof(double)));
    Check(cudaMalloc(&tile_totals_, kComponents * tiles * sizeof(double)));
    Check(cudaMalloc(&tile_carries_, kComponents * tiles * sizeof(double)));
    Check(cudaMalloc(&invalid_, kComponents * sizeof(int)));
    Check(cudaMalloc(&done_, sizeof(int)));
    Check(cudaMalloc(&statistics_, 37 * sizeof(unsigned long long)));
    Check(cudaMemsetAsync(statistics_, 0, 37 * sizeof(unsigned long long),
                          stream));
    stride_ = stride;
    tiles_ = tiles;
  }
};
thread_local Workspace workspace;
}  // namespace

extern "C" bool GpuCooperativeResidualsSupported(
    const GpuIterationDeviceState& state) {
  const auto& dimensions = state.dimensions_;
  if (dimensions[0] < 2 || dimensions[1] < 1 || dimensions[2] < 0 ||
      dimensions[4] < 1 || dimensions[4] > dimensions[0] ||
      dimensions[4] < dimensions[0] - 1 ||
      dimensions[1] > kThreads * kThreads / 2 ||
      dimensions[2] >= kThreads * kThreads / 2 ||
      dimensions[4] > kThreads * kThreads / 2)
    return false;
  const std::size_t coefficients =
      static_cast<std::size_t>(dimensions[4]) * dimensions[1] *
      (static_cast<std::size_t>(dimensions[2]) + 1);
  if (coefficients > kThreads * kThreads / 2) return false;
  const int tiles =
      static_cast<int>((2 * coefficients + kThreads - 1) / kThreads);
  int device;
  Check(cudaGetDevice(&device));
  int cooperative;
  Check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                               device));
  if (!cooperative) return false;
  int active_blocks;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, PersistentResidual, kThreads, 0));
  int multiprocessors;
  Check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                               device));
  return kComponents * tiles <= active_blocks * multiprocessors;
}

extern "C" void LaunchGpuCooperativeResiduals(GpuIterationDeviceState& m_state,
                                              bool include_edge_rz,
                                              int max_rounds) {
  if (max_rounds < 1 || max_rounds > kMaximumRounds)
    throw std::runtime_error("Cooperative round limit must be within[1,32]");
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Cooperative capture is invalidated");
  const auto& dimensions = m_state.dimensions_;
  if (dimensions[0] < 2 || dimensions[1] < 1 || dimensions[2] < 0 ||
      dimensions[4] < 1 || dimensions[4] > dimensions[0] ||
      dimensions[1] > kThreads * kThreads / 2 ||
      dimensions[2] >= kThreads * kThreads / 2 ||
      dimensions[4] > kThreads * kThreads / 2)
    throw std::runtime_error("Invalid cooperative residual dimensions");
  const std::size_t coefficients =
      static_cast<std::size_t>(dimensions[4]) * dimensions[1] *
      (static_cast<std::size_t>(dimensions[2]) + 1);
  if (coefficients > kThreads * kThreads / 2)
    throw std::runtime_error(
        "Cooperative residual dimensions exceed tile limit");
  workspace.Resize(static_cast<int>(2 * coefficients), stream,
                   capture == cudaStreamCaptureStatusActive);
  Parameters p{};
  for (int i = 0; i < 6; ++i) {
    if (m_state.forces_[i] == nullptr)
      throw std::runtime_error("Null cooperative force input");
    p.forces_[i] = m_state.forces_[i];
  }
  if (m_state.residuals_ == nullptr)
    throw std::runtime_error("Null cooperative residual output");
  p.terms_ = workspace.terms_;
  p.prefix_ = workspace.prefix_;
  p.tile_totals_ = workspace.tile_totals_;
  p.tile_carries_ = workspace.tile_carries_;
  p.invalid_ = workspace.invalid_;
  p.done_ = workspace.done_;
  p.statistics_ = workspace.statistics_;
  p.output_ = m_state.residuals_;
  p.stride_ = workspace.stride_;
  p.tiles_ = workspace.tiles_;
  const std::size_t rz_count =
      include_edge_rz
          ? coefficients
          : static_cast<std::size_t>(dimensions[0] - 1) * dimensions[1] *
                (static_cast<std::size_t>(dimensions[2]) + 1);
  if (rz_count > coefficients)
    throw std::runtime_error("Cooperative R/Z extent exceeds force allocation");
  p.rz_count_ = static_cast<int>(rz_count);
  p.max_rounds_ = max_rounds;
  std::array<void*, 1> arguments{&p};
  Check(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(PersistentResidual), dim3(kComponents * p.tiles_),
      dim3(kThreads), arguments.data(), 0, stream));
  LaunchGpuIterationResidualsConditional(m_state, include_edge_rz, p.invalid_);
}

extern "C" void ReadGpuCooperativeResidualStats(
    std::array<unsigned long long, 37>& m_statistics) {
  if (workspace.stride_ == 0)
    throw std::runtime_error("No cooperative residual experiment ran");
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(m_statistics.data(), workspace.statistics_,
                   sizeof(m_statistics), cudaMemcpyDeviceToHost));
}

extern "C" void ReadGpuCooperativeResidualLayout(std::array<int, 5>& m_layout) {
  if (workspace.stride_ == 0)
    throw std::runtime_error("No cooperative residual experiment ran");
  m_layout = {kComponents * workspace.tiles_, kThreads,
              workspace.active_blocks_per_sm_, workspace.multiprocessors_,
              workspace.tiles_};
}
