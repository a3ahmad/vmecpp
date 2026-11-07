// SPDX-License-Identifier: MIT
// Experimental optimistic accumulation with a bitwise recurrence certificate.
// Based on Kapre and DeHon, ARITH 2007. Intermediate trees are guesses only;
// no answer is accepted unless every prefix satisfies the sequential
// recurrence. A failed certificate selects the original ordered GPU kernel.
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_iteration.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Parameters {
  cuda::std::array<const double*, 6> forces_;
  double* terms_;
  double* prefix_;
  double* errors_;
  double* corrections_;
  int* keys_;
  int* invalid_;
  unsigned long long* statistics_;
  double* output_;
  int stride_;
  int rz_count_;
};

__global__ void PrepareTerms(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * m_p.stride_) return;
  const int component = i / m_p.stride_;
  const int local = i % m_p.stride_;
  const int coefficient = local / 2;
  double term = 0.0;
  if (component == 2 || coefficient < m_p.rz_count_) {
    const double force = m_p.forces_[2 * component + local % 2][coefficient];
    term = force * force;
  }
  // Padded R/Z zeros follow all real terms. Squared-force sums start at +0,
  // so these trailing +0 additions do not alter the sequential result.
  m_p.terms_[i] = term;
  m_p.keys_[i] = component;
}

__global__ void Errors(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * m_p.stride_) return;
  const double previous = i % m_p.stride_ == 0 ? 0.0 : m_p.prefix_[i - 1];
  const double target = __dadd_rn(previous, m_p.terms_[i]);
  // Sign is explicitly target minus current; guesses can be arbitrary.
  m_p.errors_[i] = __dsub_rn(target, m_p.prefix_[i]);
}

__global__ void Correct(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * m_p.stride_) {
    m_p.prefix_[i] = __dadd_rn(m_p.prefix_[i], m_p.corrections_[i]);
  }
}

__global__ void Verify(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * m_p.stride_) return;
  const double previous = i % m_p.stride_ == 0 ? 0.0 : m_p.prefix_[i - 1];
  const double target = __dadd_rn(previous, m_p.terms_[i]);
  // Bit equality includes zero signs and NaN payloads. Induction from +0
  // certifies the complete sequential prefix, not just a small numeric error.
  if (!isfinite(m_p.terms_[i]) || !isfinite(m_p.prefix_[i]) ||
      __double_as_longlong(target) != __double_as_longlong(m_p.prefix_[i])) {
    atomicOr(m_p.invalid_ + i / m_p.stride_, 1);
  }
}

__global__ void Finish(Parameters m_p) {
  const int component = threadIdx.x;
  if (component == 0) atomicAdd(m_p.statistics_, 1ULL);
  if (component >= 3) return;
  if (m_p.invalid_[component] == 0) {
    m_p.output_[component] = m_p.prefix_[(component + 1) * m_p.stride_ - 1];
  } else {
    atomicAdd(m_p.statistics_ + component + 1, 1ULL);
  }
}

struct GraphEntry {
  std::array<double*, 6> forces_;
  double* output_;
  int stride_;
  int rz_count_;
  int rounds_;
  cudaGraph_t graph_;
  cudaGraphExec_t executable_;
};

struct Workspace {
  int count_ = 0;
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  double* errors_ = nullptr;
  double* corrections_ = nullptr;
  int* keys_ = nullptr;
  int* invalid_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  void* temporary_ = nullptr;
  std::size_t temporary_bytes_ = 0;
  cudaStream_t stream_ = nullptr;
  std::vector<GraphEntry> graphs_;

  void Clear() {
    // The caller owns the stream and may already have destroyed it. cudaFree
    // orders outstanding uses of these allocations without a stale handle.
    for (auto& graph : graphs_) {
      cudaGraphExecDestroy(graph.executable_);
      cudaGraphDestroy(graph.graph_);
    }
    graphs_.clear();
    for (void* pointer :
         {static_cast<void*>(terms_), static_cast<void*>(prefix_),
          static_cast<void*>(errors_), static_cast<void*>(corrections_),
          static_cast<void*>(keys_), static_cast<void*>(invalid_),
          static_cast<void*>(statistics_), temporary_}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    count_ = 0;
    terms_ = nullptr;
    prefix_ = nullptr;
    errors_ = nullptr;
    corrections_ = nullptr;
    keys_ = nullptr;
    invalid_ = nullptr;
    statistics_ = nullptr;
    temporary_ = nullptr;
    stream_ = nullptr;
  }
  ~Workspace() { Clear(); }

  void Resize(int count, cudaStream_t stream) {
    if (count_ == count) {
      if (stream_ != stream) Check(cudaDeviceSynchronize());
      stream_ = stream;
      return;
    }
    Clear();
    stream_ = stream;
    Check(cudaMalloc(&terms_, count * sizeof(double)));
    Check(cudaMalloc(&prefix_, count * sizeof(double)));
    Check(cudaMalloc(&errors_, count * sizeof(double)));
    Check(cudaMalloc(&corrections_, count * sizeof(double)));
    Check(cudaMalloc(&keys_, count * sizeof(int)));
    Check(cudaMalloc(&invalid_, 3 * sizeof(int)));
    Check(cudaMalloc(&statistics_, 4 * sizeof(unsigned long long)));
    Check(cudaMemsetAsync(statistics_, 0, 4 * sizeof(unsigned long long),
                          stream));
    temporary_bytes_ = 0;
    Check(cub::DeviceScan::InclusiveScanByKey(nullptr, temporary_bytes_, keys_,
                                              terms_, prefix_, cub::Sum(),
                                              count, cub::Equality(), stream));
    Check(cudaMalloc(&temporary_, temporary_bytes_));
    count_ = count;
  }
};
thread_local Workspace workspace;

void Enqueue(Workspace& m_work, Parameters m_parameters,
             GpuIterationDeviceState& m_state, bool include_edge_rz,
             int rounds) {
  const int blocks = (3 * m_parameters.stride_ + 127) / 128;
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  PrepareTerms<<<blocks, 128, 0, stream>>>(m_parameters);
  Check(cudaGetLastError());
  Check(cub::DeviceScan::InclusiveScanByKey(
      m_work.temporary_, m_work.temporary_bytes_, m_parameters.keys_,
      m_parameters.terms_, m_parameters.prefix_, cub::Sum(),
      3 * m_parameters.stride_, cub::Equality(), stream));
  for (int iteration = 0; iteration < rounds; ++iteration) {
    Errors<<<blocks, 128, 0, stream>>>(m_parameters);
    Check(cudaGetLastError());
    Check(cub::DeviceScan::InclusiveScanByKey(
        m_work.temporary_, m_work.temporary_bytes_, m_parameters.keys_,
        m_parameters.errors_, m_parameters.corrections_, cub::Sum(),
        3 * m_parameters.stride_, cub::Equality(), stream));
    Correct<<<blocks, 128, 0, stream>>>(m_parameters);
    Check(cudaGetLastError());
  }
  Check(cudaMemsetAsync(m_parameters.invalid_, 0, 3 * sizeof(int), stream));
  Verify<<<blocks, 128, 0, stream>>>(m_parameters);
  Check(cudaGetLastError());
  Finish<<<1, 32, 0, stream>>>(m_parameters);
  Check(cudaGetLastError());
  LaunchGpuIterationResidualsConditional(m_state, include_edge_rz,
                                         m_parameters.invalid_);
}
}  // namespace

extern "C" void LaunchGpuIterationResidualsOptimistic(
    GpuIterationDeviceState& m_state, bool include_edge_rz, int rounds,
    bool use_graph) {
  if (rounds < 1 || rounds > 32)
    throw std::runtime_error("Invalid relaxation limit");
  const int coefficients = m_state.dimensions_[4] * m_state.dimensions_[1] *
                           (m_state.dimensions_[2] + 1);
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  workspace.Resize(6 * coefficients, stream);
  Parameters p{};
  for (int i = 0; i < 6; ++i) p.forces_[i] = m_state.forces_[i];
  p.terms_ = workspace.terms_;
  p.prefix_ = workspace.prefix_;
  p.errors_ = workspace.errors_;
  p.corrections_ = workspace.corrections_;
  p.keys_ = workspace.keys_;
  p.invalid_ = workspace.invalid_;
  p.statistics_ = workspace.statistics_;
  p.output_ = m_state.residuals_;
  p.stride_ = 2 * coefficients;
  p.rz_count_ = include_edge_rz
                    ? coefficients
                    : (m_state.dimensions_[0] - 1) * m_state.dimensions_[1] *
                          (m_state.dimensions_[2] + 1);
  if (!use_graph || stream == nullptr) {
    Enqueue(workspace, p, m_state, include_edge_rz, rounds);
    return;
  }
  for (const auto& graph : workspace.graphs_) {
    if (graph.forces_ == m_state.forces_ &&
        graph.output_ == m_state.residuals_ && graph.stride_ == p.stride_ &&
        graph.rz_count_ == p.rz_count_ && graph.rounds_ == rounds) {
      Check(cudaGraphLaunch(graph.executable_, stream));
      return;
    }
  }
  GraphEntry graph{m_state.forces_, m_state.residuals_,
                   p.stride_,       p.rz_count_,
                   rounds,          nullptr,
                   nullptr};
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  Enqueue(workspace, p, m_state, include_edge_rz, rounds);
  Check(cudaStreamEndCapture(stream, &graph.graph_));
  Check(cudaGraphInstantiate(&graph.executable_, graph.graph_, 0));
  workspace.graphs_.push_back(graph);
  Check(cudaGraphLaunch(graph.executable_, stream));
}

extern "C" void ReadGpuResidualRelaxationStats(
    std::array<unsigned long long, 4>& m_statistics) {
  if (workspace.count_ == 0)
    throw std::runtime_error("No residual experiment ran");
  Check(cudaStreamSynchronize(workspace.stream_));
  Check(cudaMemcpy(m_statistics.data(), workspace.statistics_,
                   sizeof(m_statistics), cudaMemcpyDeviceToHost));
}
