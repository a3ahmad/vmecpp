// SPDX-License-Identifier: MIT
// Trees propose prefixes; bitwise recurrence certificates establish the exact
// original ordered sum. Uncertified components use the original recurrence.
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_optimistic_diagnostics.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Parameters {
  cuda::std::array<const double*, 14> data_;
  cuda::std::array<int, 5> offsets_;
  int count_;
  int half_count_;
  int coefficient_count_;
  int modes_;
  int n_theta_;
  int ns_;
  double delta_s_;
  double lamscale_;
  double* sums_;
  double* output_;
  double* terms_;
  double* prefix_;
  double* errors_;
  double* corrections_;
  int* keys_;
  int* invalid_;
  unsigned long long* statistics_;
  bool refresh_norms_;
};

__device__ double Term(const Parameters& p, int component, int i) {
  const auto& v = p.data_;
  if (component == 0) return v[7][i];
  if (component == 1) return (v[6][i] / 1.0) * v[7][i];
  if (component == 2) {
    const double magnetic = 0.5 * (v[1][i] * v[3][i] + v[2][i] * v[4][i]);
    return v[0][i] * magnetic * v[5][i % p.n_theta_];
  }
  if (component == 3) return v[8][i] * v[9][i] * v[9][i] * v[5][i % p.n_theta_];
  if (component == 4)
    return (v[3][i] * v[3][i] + v[4][i] * v[4][i]) * v[5][i % p.n_theta_];
  // Compact rzNorm sequence removes the skipped rcc(0,0) term on each
  // surface without inserting zero additions into the original recurrence.
  const int row_size = 4 * p.modes_ - 1;
  const int row = i / row_size;
  const int local = i % row_size;
  const int coefficient =
      row * p.modes_ + (local < 3 ? 0 : 1 + (local - 3) / 4);
  const int channel = local < 3 ? local + 1 : (local - 3) % 4;
  const int source =
      channel == 0 ? 10 : (channel == 1 ? 12 : (channel == 2 ? 11 : 13));
  const double value = v[source][coefficient];
  return value * value;
}

__global__ void Prepare(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m_p.count_) return;
  int segment = 0;
  while (segment < 3 && i >= m_p.offsets_[segment + 1]) ++segment;
  m_p.keys_[i] = segment + 2;
  m_p.terms_[i] = Term(m_p, segment + 2, i - m_p.offsets_[segment]);
}
__global__ void Errors(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m_p.count_) return;
  const int segment = m_p.keys_[i] - 2;
  const double previous = i == m_p.offsets_[segment] ? 0.0 : m_p.prefix_[i - 1];
  m_p.errors_[i] =
      __dsub_rn(__dadd_rn(previous, m_p.terms_[i]), m_p.prefix_[i]);
}
__global__ void Correct(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m_p.count_)
    m_p.prefix_[i] = __dadd_rn(m_p.prefix_[i], m_p.corrections_[i]);
}
__global__ void Verify(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m_p.count_) return;
  const int component = m_p.keys_[i];
  const double previous =
      i == m_p.offsets_[component - 2] ? 0.0 : m_p.prefix_[i - 1];
  const double target = __dadd_rn(previous, m_p.terms_[i]);
  if (!isfinite(m_p.terms_[i]) || !isfinite(m_p.prefix_[i]) ||
      __double_as_longlong(target) != __double_as_longlong(m_p.prefix_[i]))
    atomicOr(m_p.invalid_ + component, 1);
}
__global__ void Accept(Parameters m_p) {
  const int component = threadIdx.x;
  if (component == 0) atomicAdd(m_p.statistics_, 1ULL);
  if (component < 2 || component >= (m_p.refresh_norms_ ? 6 : 3)) return;
  if (m_p.invalid_[component] == 0) {
    m_p.sums_[component] = m_p.prefix_[m_p.offsets_[component - 1] - 1];
  } else {
    atomicAdd(m_p.statistics_ + component + 1, 1ULL);
  }
}

__global__ void OrderedFallback(Parameters m_p) {
  __shared__ double terms[128];
  const int component = blockIdx.x;
  if (component >= 2 && m_p.invalid_[component] == 0) return;
  const int count = component < 2
                        ? m_p.ns_ - 1
                        : (component == 5 ? 4 * m_p.coefficient_count_ - m_p.ns_
                                          : m_p.half_count_);
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
  const double volume =
      m_p.refresh_norms_ ? 0.0 + m_p.sums_[0] * m_p.delta_s_ : m_p.output_[0];
  const double thermal = 0.0 + m_p.sums_[1] * m_p.delta_s_;
  const double magnetic = 0.0 + fabs(m_p.sums_[2]) * m_p.delta_s_;
  const double density = (magnetic < thermal ? thermal : magnetic) / volume;
  m_p.output_[1] = thermal;
  m_p.output_[2] = magnetic;
  m_p.output_[3] = magnetic + thermal / (0.0 - 1.0);
  if (m_p.refresh_norms_) {
    m_p.output_[0] = volume;
    m_p.output_[4] = 1.0 / ((0.0 + m_p.sums_[3]) * density * density);
    m_p.output_[5] =
        1.0 / ((0.0 + m_p.sums_[4]) * m_p.lamscale_ * m_p.lamscale_);
    m_p.output_[6] = 1.0 / (0.0 + m_p.sums_[5]);
  }
}

struct GraphEntry {
  GpuDiagnosticsDeviceState state_;
  bool refresh_norms_;
  int rounds_;
  cudaGraph_t graph_;
  cudaGraphExec_t executable_;
};
struct Workspace {
  int capacity_ = 0;
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
    capacity_ = 0;
    terms_ = prefix_ = errors_ = corrections_ = nullptr;
    keys_ = invalid_ = nullptr;
    statistics_ = nullptr;
    temporary_ = nullptr;
    stream_ = nullptr;
  }
  ~Workspace() { Clear(); }
  void Resize(int capacity, cudaStream_t stream) {
    if (capacity_ == capacity) {
      if (stream_ != stream) Check(cudaDeviceSynchronize());
      stream_ = stream;
      return;
    }
    Clear();
    stream_ = stream;
    Check(cudaMalloc(&terms_, capacity * sizeof(double)));
    Check(cudaMalloc(&prefix_, capacity * sizeof(double)));
    Check(cudaMalloc(&errors_, capacity * sizeof(double)));
    Check(cudaMalloc(&corrections_, capacity * sizeof(double)));
    Check(cudaMalloc(&keys_, capacity * sizeof(int)));
    Check(cudaMalloc(&invalid_, 6 * sizeof(int)));
    Check(cudaMalloc(&statistics_, 7 * sizeof(unsigned long long)));
    Check(cudaMemsetAsync(statistics_, 0, 7 * sizeof(unsigned long long),
                          stream));
    temporary_bytes_ = 0;
    Check(cub::DeviceScan::InclusiveScanByKey(
        nullptr, temporary_bytes_, keys_, terms_, prefix_, cub::Sum(), capacity,
        cub::Equality(), stream));
    Check(cudaMalloc(&temporary_, temporary_bytes_));
    capacity_ = capacity;
  }
};
thread_local Workspace workspace;

void Enqueue(Workspace& m_work, Parameters parameters, int rounds,
             cudaStream_t stream) {
  const int blocks = (parameters.count_ + 127) / 128;
  Prepare<<<blocks, 128, 0, stream>>>(parameters);
  Check(cudaGetLastError());
  Check(cub::DeviceScan::InclusiveScanByKey(
      m_work.temporary_, m_work.temporary_bytes_, parameters.keys_,
      parameters.terms_, parameters.prefix_, cub::Sum(), parameters.count_,
      cub::Equality(), stream));
  for (int iteration = 0; iteration < rounds; ++iteration) {
    Errors<<<blocks, 128, 0, stream>>>(parameters);
    Check(cudaGetLastError());
    Check(cub::DeviceScan::InclusiveScanByKey(
        m_work.temporary_, m_work.temporary_bytes_, parameters.keys_,
        parameters.errors_, parameters.corrections_, cub::Sum(),
        parameters.count_, cub::Equality(), stream));
    Correct<<<blocks, 128, 0, stream>>>(parameters);
    Check(cudaGetLastError());
  }
  Check(cudaMemsetAsync(parameters.invalid_, 0, 6 * sizeof(int), stream));
  Verify<<<blocks, 128, 0, stream>>>(parameters);
  Check(cudaGetLastError());
  Accept<<<1, 32, 0, stream>>>(parameters);
  Check(cudaGetLastError());
  OrderedFallback<<<parameters.refresh_norms_ ? 6 : 3, 128, 0, stream>>>(
      parameters);
  Check(cudaGetLastError());
  Finish<<<1, 1, 0, stream>>>(parameters);
  Check(cudaGetLastError());
}
}  // namespace

extern "C" void LaunchGpuDiagnosticsOptimistic(
    GpuDiagnosticsDeviceState& m_state, bool refresh_norms, int rounds,
    bool use_graph) {
  const auto& d = m_state.dimensions_;
  if (rounds < 1 || rounds > 32 || m_state.scalars_[1] != 0.0 ||
      d[3] != d[0] - 1 || d[4] != d[0])
    throw std::runtime_error("Invalid optimistic diagnostics configuration");
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  const int half = (d[0] - 1) * d[5] * d[6];
  const int modes = d[1] * (d[2] + 1);
  const int coefficients = d[4] * modes;
  const int capacity = 3 * half + 4 * coefficients - d[0];
  workspace.Resize(capacity, stream);
  Parameters p{};
  for (int i = 0; i < 14; ++i) p.data_[i] = m_state.data_[i];
  p.offsets_ = {0, half, 2 * half, 3 * half, capacity};
  p.count_ = refresh_norms ? capacity : half;
  p.half_count_ = half;
  p.coefficient_count_ = coefficients;
  p.modes_ = modes;
  p.n_theta_ = d[5];
  p.ns_ = d[0];
  p.delta_s_ = m_state.scalars_[0];
  p.lamscale_ = m_state.scalars_[2];
  p.sums_ = m_state.sums_;
  p.output_ = m_state.output_;
  p.terms_ = workspace.terms_;
  p.prefix_ = workspace.prefix_;
  p.errors_ = workspace.errors_;
  p.corrections_ = workspace.corrections_;
  p.keys_ = workspace.keys_;
  p.invalid_ = workspace.invalid_;
  p.statistics_ = workspace.statistics_;
  p.refresh_norms_ = refresh_norms;
  if (!use_graph || stream == nullptr) {
    Enqueue(workspace, p, rounds, stream);
    return;
  }
  for (const auto& graph : workspace.graphs_) {
    const auto& state = graph.state_;
    if (state.data_ == m_state.data_ && state.dimensions_ == d &&
        state.scalars_ == m_state.scalars_ && state.sums_ == m_state.sums_ &&
        state.output_ == m_state.output_ &&
        graph.refresh_norms_ == refresh_norms && graph.rounds_ == rounds) {
      Check(cudaGraphLaunch(graph.executable_, stream));
      return;
    }
  }
  GraphEntry graph{m_state, refresh_norms, rounds, nullptr, nullptr};
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  Enqueue(workspace, p, rounds, stream);
  Check(cudaStreamEndCapture(stream, &graph.graph_));
  Check(cudaGraphInstantiate(&graph.executable_, graph.graph_, 0));
  workspace.graphs_.push_back(graph);
  Check(cudaGraphLaunch(graph.executable_, stream));
}

extern "C" void ReadGpuDiagnosticsRelaxationStats(
    std::array<unsigned long long, 7>& m_statistics) {
  if (workspace.capacity_ == 0)
    throw std::runtime_error("No optimistic diagnostics ran");
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(m_statistics.data(), workspace.statistics_,
                   sizeof(m_statistics), cudaMemcpyDeviceToHost));
}
