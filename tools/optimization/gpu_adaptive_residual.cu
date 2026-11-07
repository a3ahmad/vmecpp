// SPDX-License-Identifier: MIT
// Off-default adaptive optimistic accumulation. Proposed tree prefixes are
// accepted only after a bitwise certificate of every sequential recurrence.
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_adaptive_residual.h"

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
  int* rounds_;
  unsigned long long* statistics_;
  double* output_;
  int stride_;
  int rz_count_;
  int max_rounds_;
  cudaGraphConditionalHandle handle_;
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
  // Same nonnegative squared-force sequence and trailing R/Z padding as the
  // qualified fixed-round prototype; padding preserves the ordered sum.
  m_p.terms_[i] = term;
  m_p.keys_[i] = component;
}
__global__ void Errors(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * m_p.stride_) return;
  const double previous = i % m_p.stride_ == 0 ? 0.0 : m_p.prefix_[i - 1];
  const double target = __dadd_rn(previous, m_p.terms_[i]);
  m_p.errors_[i] = __dsub_rn(target, m_p.prefix_[i]);
}
__global__ void Correct(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * m_p.stride_)
    m_p.prefix_[i] = __dadd_rn(m_p.prefix_[i], m_p.corrections_[i]);
}
__global__ void Verify(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * m_p.stride_) return;
  const double previous = i % m_p.stride_ == 0 ? 0.0 : m_p.prefix_[i - 1];
  const double target = __dadd_rn(previous, m_p.terms_[i]);
  if (!isfinite(m_p.terms_[i]) || !isfinite(m_p.prefix_[i]) ||
      __double_as_longlong(target) != __double_as_longlong(m_p.prefix_[i]))
    atomicOr(m_p.invalid_ + i / m_p.stride_, 1);
}
__global__ void Continue(Parameters m_p) {
  const int rounds = *m_p.rounds_ + 4;
  *m_p.rounds_ = rounds;
  const bool failed = m_p.invalid_[0] || m_p.invalid_[1] || m_p.invalid_[2];
  cudaGraphSetConditional(m_p.handle_, failed && rounds < m_p.max_rounds_);
}
__global__ void Finish(Parameters m_p) {
  const int component = threadIdx.x;
  if (component == 0) {
    atomicAdd(m_p.statistics_, 1ULL);
    atomicAdd(m_p.statistics_ + 4 + *m_p.rounds_ / 4, 1ULL);
  }
  if (component >= 3) return;
  if (m_p.invalid_[component] == 0)
    m_p.output_[component] = m_p.prefix_[(component + 1) * m_p.stride_ - 1];
  else
    atomicAdd(m_p.statistics_ + component + 1, 1ULL);
}

struct GraphEntry {
  std::array<double*, 6> forces_;
  double* output_;
  std::array<int, 5> dimensions_;
  bool include_edge_rz_;
  int max_rounds_;
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
  int* rounds_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  void* temporary_ = nullptr;
  std::size_t temporary_bytes_ = 0;
  cudaStream_t caller_stream_ = nullptr;
  cudaStream_t graph_capture_stream_ = nullptr;
  cudaStream_t body_capture_stream_ = nullptr;
  std::vector<GraphEntry> graphs_;

  void Clear() {
    // External streams may already have been destroyed. cudaFree orders uses
    // without dereferencing their stale handles, matching the existing APIs.
    for (auto& graph : graphs_) {
      cudaGraphExecDestroy(graph.executable_);
      cudaGraphDestroy(graph.graph_);
    }
    graphs_.clear();
    for (void* pointer :
         {static_cast<void*>(terms_), static_cast<void*>(prefix_),
          static_cast<void*>(errors_), static_cast<void*>(corrections_),
          static_cast<void*>(keys_), static_cast<void*>(invalid_),
          static_cast<void*>(rounds_), static_cast<void*>(statistics_),
          temporary_}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    if (graph_capture_stream_ != nullptr)
      cudaStreamDestroy(graph_capture_stream_);
    if (body_capture_stream_ != nullptr)
      cudaStreamDestroy(body_capture_stream_);
    count_ = 0;
    terms_ = prefix_ = errors_ = corrections_ = nullptr;
    keys_ = invalid_ = rounds_ = nullptr;
    statistics_ = nullptr;
    temporary_ = nullptr;
    caller_stream_ = graph_capture_stream_ = body_capture_stream_ = nullptr;
  }
  ~Workspace() { Clear(); }
  void Resize(int count, cudaStream_t stream, bool capturing) {
    if (count_ == count) {
      if (caller_stream_ != stream) {
        if (capturing)
          throw std::runtime_error(
              "Warm adaptive workspace on caller stream before capture");
        Check(cudaDeviceSynchronize());
      }
      caller_stream_ = stream;
      return;
    }
    if (capturing)
      throw std::runtime_error("Allocate adaptive workspace before capture");
    Clear();
    caller_stream_ = stream;
    Check(cudaStreamCreateWithFlags(&graph_capture_stream_,
                                    cudaStreamNonBlocking));
    Check(cudaStreamCreateWithFlags(&body_capture_stream_,
                                    cudaStreamNonBlocking));
    Check(cudaMalloc(&terms_, count * sizeof(double)));
    Check(cudaMalloc(&prefix_, count * sizeof(double)));
    Check(cudaMalloc(&errors_, count * sizeof(double)));
    Check(cudaMalloc(&corrections_, count * sizeof(double)));
    Check(cudaMalloc(&keys_, count * sizeof(int)));
    Check(cudaMalloc(&invalid_, 3 * sizeof(int)));
    Check(cudaMalloc(&rounds_, sizeof(int)));
    Check(cudaMalloc(&statistics_, 13 * sizeof(unsigned long long)));
    Check(cudaMemsetAsync(statistics_, 0, 13 * sizeof(unsigned long long),
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

void Scan(Workspace& m_work, const Parameters& p, const double* input,
          double* m_output, cudaStream_t stream) {
  Check(cub::DeviceScan::InclusiveScanByKey(
      m_work.temporary_, m_work.temporary_bytes_, p.keys_, input, m_output,
      cub::Sum(), 3 * p.stride_, cub::Equality(), stream));
}
void Body(Workspace& m_work, const Parameters& p) {
  const auto stream = m_work.body_capture_stream_;
  const int blocks = (3 * p.stride_ + 127) / 128;
  for (int round = 0; round < 4; ++round) {
    Errors<<<blocks, 128, 0, stream>>>(p);
    Check(cudaGetLastError());
    Scan(m_work, p, p.errors_, p.corrections_, stream);
    Correct<<<blocks, 128, 0, stream>>>(p);
    Check(cudaGetLastError());
  }
  Check(cudaMemsetAsync(p.invalid_, 0, 3 * sizeof(int), stream));
  Verify<<<blocks, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  Continue<<<1, 1, 0, stream>>>(p);
  Check(cudaGetLastError());
}

void Insert(Workspace& m_work, Parameters m_parameters,
            GpuIterationDeviceState& m_state, bool include_edge_rz) {
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  const int blocks = (3 * m_parameters.stride_ + 127) / 128;
  PrepareTerms<<<blocks, 128, 0, stream>>>(m_parameters);
  Check(cudaGetLastError());
  Scan(m_work, m_parameters, m_parameters.terms_, m_parameters.prefix_, stream);
  Check(cudaMemsetAsync(m_parameters.rounds_, 0, sizeof(int), stream));

  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaGraph_t parent = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  std::size_t dependency_count = 0;
  Check(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &parent,
                                    &dependencies, &dependency_count));
  if (status != cudaStreamCaptureStatusActive || parent == nullptr)
    throw std::runtime_error(
        "Adaptive graph insertion requires active capture");
  Check(cudaGraphConditionalHandleCreate(&m_parameters.handle_, parent, 1,
                                         cudaGraphCondAssignDefault));
  cudaGraphNodeParams conditional{};
  conditional.type = cudaGraphNodeTypeConditional;
  conditional.conditional.handle = m_parameters.handle_;
  conditional.conditional.type = cudaGraphCondTypeWhile;
  conditional.conditional.size = 1;
  cudaGraphNode_t node = nullptr;
  Check(cudaGraphAddNode(&node, parent, dependencies, dependency_count,
                         &conditional));
  const cudaGraph_t body = conditional.conditional.phGraph_out[0];
  bool capturing_body = false;
  try {
    Check(cudaStreamBeginCaptureToGraph(m_work.body_capture_stream_, body,
                                        nullptr, nullptr, 0,
                                        cudaStreamCaptureModeThreadLocal));
    capturing_body = true;
    Body(m_work, m_parameters);
    cudaGraph_t captured = nullptr;
    const auto capture_status =
        cudaStreamEndCapture(m_work.body_capture_stream_, &captured);
    capturing_body = false;
    Check(capture_status);
    if (captured != body)
      throw std::runtime_error(
          "Conditional capture returned a different body graph");
  } catch (...) {
    if (capturing_body) {
      cudaGraph_t ignored = nullptr;
      cudaStreamEndCapture(m_work.body_capture_stream_, &ignored);
    }
    throw;
  }
  Check(cudaStreamUpdateCaptureDependencies(stream, &node, 1,
                                            cudaStreamSetCaptureDependencies));
  Finish<<<1, 32, 0, stream>>>(m_parameters);
  Check(cudaGetLastError());
  LaunchGpuIterationResidualsConditional(m_state, include_edge_rz,
                                         m_parameters.invalid_);
}
}  // namespace

extern "C" void LaunchGpuAdaptiveResiduals(GpuIterationDeviceState& m_state,
                                           bool include_edge_rz,
                                           int max_rounds) {
  if (max_rounds < 4 || max_rounds > 32 || max_rounds % 4 != 0)
    throw std::runtime_error(
        "Adaptive limit must be a multiple of four in [4,32]");
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  Check(cudaStreamIsCapturing(stream, &status));
  if (status == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Cannot insert into invalidated capture");
  const int coefficients = m_state.dimensions_[4] * m_state.dimensions_[1] *
                           (m_state.dimensions_[2] + 1);
  workspace.Resize(6 * coefficients, stream,
                   status == cudaStreamCaptureStatusActive);
  Parameters p{};
  for (int i = 0; i < 6; ++i) p.forces_[i] = m_state.forces_[i];
  p.terms_ = workspace.terms_;
  p.prefix_ = workspace.prefix_;
  p.errors_ = workspace.errors_;
  p.corrections_ = workspace.corrections_;
  p.keys_ = workspace.keys_;
  p.invalid_ = workspace.invalid_;
  p.rounds_ = workspace.rounds_;
  p.statistics_ = workspace.statistics_;
  p.output_ = m_state.residuals_;
  p.stride_ = 2 * coefficients;
  p.rz_count_ = include_edge_rz
                    ? coefficients
                    : (m_state.dimensions_[0] - 1) * m_state.dimensions_[1] *
                          (m_state.dimensions_[2] + 1);
  p.max_rounds_ = max_rounds;
  if (status == cudaStreamCaptureStatusActive) {
    Insert(workspace, p, m_state, include_edge_rz);
    return;
  }
  for (const auto& graph : workspace.graphs_) {
    if (graph.forces_ == m_state.forces_ &&
        graph.output_ == m_state.residuals_ &&
        graph.dimensions_ == m_state.dimensions_ &&
        graph.include_edge_rz_ == include_edge_rz &&
        graph.max_rounds_ == max_rounds) {
      Check(cudaGraphLaunch(graph.executable_, stream));
      return;
    }
  }
  GraphEntry graph{m_state.forces_, m_state.residuals_, m_state.dimensions_,
                   include_edge_rz, max_rounds,         nullptr,
                   nullptr};
  auto capture_state = m_state;
  capture_state.stream_ = workspace.graph_capture_stream_;
  bool capturing = false;
  try {
    Check(cudaStreamBeginCapture(workspace.graph_capture_stream_,
                                 cudaStreamCaptureModeThreadLocal));
    capturing = true;
    Insert(workspace, p, capture_state, include_edge_rz);
    const auto capture_status =
        cudaStreamEndCapture(workspace.graph_capture_stream_, &graph.graph_);
    capturing = false;
    Check(capture_status);
    Check(cudaGraphInstantiate(&graph.executable_, graph.graph_, 0));
    workspace.graphs_.push_back(graph);
  } catch (...) {
    if (capturing)
      cudaStreamEndCapture(workspace.graph_capture_stream_, &graph.graph_);
    if (graph.executable_ != nullptr) cudaGraphExecDestroy(graph.executable_);
    if (graph.graph_ != nullptr) cudaGraphDestroy(graph.graph_);
    throw;
  }
  Check(cudaGraphLaunch(graph.executable_, stream));
}

extern "C" void ReadGpuAdaptiveResidualStats(
    std::array<unsigned long long, 13>& m_statistics) {
  if (workspace.count_ == 0)
    throw std::runtime_error("No adaptive residual experiment ran");
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(m_statistics.data(), workspace.statistics_,
                   sizeof(m_statistics), cudaMemcpyDeviceToHost));
}
