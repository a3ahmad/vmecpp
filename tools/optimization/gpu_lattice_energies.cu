// SPDX-License-Identifier: MIT
// Speculative positive-prefix proposals never replace the original recurrence
// without a complete finite bitwise certificate. Signed inputs may fall back.
#include <cuda_runtime.h>

#include <array>
#include <cuda/std/array>
#include <stdexcept>

#include "gpu_lattice_energies.h"
#include "gpu_lattice_prefix.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

struct Parameters {
  cuda::std::array<const double*, 8> data_;
  cuda::std::array<int*, 2> invalid_;
  cuda::std::array<double*, 2> terms_;
  cuda::std::array<const double*, 2> prefix_;
  cuda::std::array<int, 2> counts_;
  unsigned long long* statistics_;
  double* sums_;
  double* output_;
  int n_theta_;
  double delta_s_;
};

// Literal expressions and index order from gpu_diagnostics.cu OrderedSums.
__device__ double Term(const Parameters& p, int component, int i) {
  const auto& v = p.data_;
  if (component == 0) return v[7][i];
  if (component == 1) return (v[6][i] / 1.0) * v[7][i];
  const double magnetic = 0.5 * (v[1][i] * v[3][i] + v[2][i] * v[4][i]);
  return v[0][i] * magnetic * v[5][i % p.n_theta_];
}

__global__ void PrepareTerms(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m_p.counts_[0]) m_p.terms_[0][index] = Term(m_p, 1, index);
  // Negative gsqrt orientation is common. This is only a proposal: every
  // restored ORIGINAL signed prefix is independently certified below.
  if (index < m_p.counts_[1]) m_p.terms_[1][index] = -Term(m_p, 2, index);
}

__device__ double RestorePrefix(const Parameters& p, int component, int i) {
  const double prefix = p.prefix_[component][i];
  if (component == 0) return prefix;
  // From +0, a recurrence of nonpositive terms retains +0 until nonzero.
  // The signed recurrence certificate still rejects any incorrect zero sign.
  return prefix == 0.0 ? 0.0 : -prefix;
}

__global__ void VerifySignedPrefixes(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int component = 0; component < 2; ++component) {
    if (index >= m_p.counts_[component]) continue;
    const double term = Term(m_p, component + 1, index);
    const double previous =
        index == 0 ? 0.0 : RestorePrefix(m_p, component, index - 1);
    const double prefix = RestorePrefix(m_p, component, index);
    const double expected = __dadd_rn(previous, term);
    if (!isfinite(term) || !isfinite(prefix) ||
        __double_as_longlong(expected) != __double_as_longlong(prefix))
      atomicOr(m_p.invalid_[component], 1);
  }
}

__global__ void FinishCertified(Parameters m_p) {
  const int component = threadIdx.x;
  if (component >= 2) return;
  const bool invalid = *m_p.invalid_[component] != 0;
  if (!invalid)
    m_p.sums_[component + 1] =
        RestorePrefix(m_p, component, m_p.counts_[component] - 1);
  atomicAdd(m_p.statistics_ + 2 * component + (invalid ? 1 : 0), 1ULL);
}

// Original ordered 128-term loading/serial-add schedule, conditionally applied
// only to rejected energy proposals. Volume is recomputed on every call.
__global__ void OrderedFallback(Parameters m_p) {
  __shared__ cuda::std::array<double, 128> terms;
  const int component = blockIdx.x;
  if (component > 0 && *m_p.invalid_[component - 1] == 0) return;
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

struct Workspace {
  std::array<int, 2> counts_{};
  std::array<double*, 2> terms_{};
  std::array<double*, 2> prefix_{};
  std::array<void*, 2> contexts_{};
  int* invalid_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  int device_ = -1;
  cudaStream_t stream_ = nullptr;

  void Clear() {
    for (int i = 0; i < 2; ++i) {
      if (contexts_[i] != nullptr) DestroyGpuLatticePrefix(contexts_[i]);
      if (terms_[i] != nullptr) cudaFree(terms_[i]);
      if (prefix_[i] != nullptr) cudaFree(prefix_[i]);
      contexts_[i] = nullptr;
      terms_[i] = prefix_[i] = nullptr;
    }
    counts_ = {};
    if (invalid_ != nullptr) cudaFree(invalid_);
    if (statistics_ != nullptr) cudaFree(statistics_);
    invalid_ = nullptr;
    statistics_ = nullptr;
  }
  ~Workspace() { Clear(); }

  void Resize(const std::array<int, 2>& counts, cudaStream_t stream,
              bool capturing) {
    int device;
    Check(cudaGetDevice(&device));
    if (counts_[0] != 0 && device != device_)
      throw std::runtime_error("Lattice energies cannot change device");
    if (counts == counts_) {
      if (stream != stream_) {
        if (capturing)
          throw std::runtime_error("Warm lattice energies stream first");
        Check(cudaDeviceSynchronize());
        for (auto* context : contexts_) RebindGpuLatticePrefix(context, stream);
        stream_ = stream;
      }
      return;
    }
    if (capturing)
      throw std::runtime_error("Warm lattice energies shape before capture");
    Clear();
    device_ = device;
    stream_ = stream;
    try {
      Check(cudaMalloc(&invalid_, 2 * sizeof(int)));
      Check(cudaMalloc(&statistics_, 4 * sizeof(unsigned long long)));
      Check(cudaMemsetAsync(statistics_, 0, 4 * sizeof(unsigned long long),
                            stream));
      for (int i = 0; i < 2; ++i) {
        const auto bytes = std::size_t(counts[i]) * sizeof(double);
        Check(cudaMalloc(&terms_[i], bytes));
        Check(cudaMalloc(&prefix_[i], bytes));
        contexts_[i] = CreateGpuLatticePrefix(1, counts[i], stream);
      }
      counts_ = counts;
    } catch (...) {
      Clear();
      throw;
    }
  }
};
thread_local Workspace workspace;

void Launch(GpuDiagnosticsDeviceState& m_state, bool corrupt) {
  const auto& d = m_state.dimensions_;
  if (m_state.scalars_[1] != 0.0 || d[0] < 2 || d[3] != d[0] - 1 ||
      d[4] != d[0] || d[5] < 1 || d[6] < 1 || d[0] - 1 > (1 << 21))
    throw std::runtime_error("Unsupported lattice energy configuration");
  const auto half_count = std::size_t(d[0] - 1) * d[5];
  if (half_count > (1 << 21) || std::size_t(d[6]) > (1 << 21) / half_count)
    throw std::runtime_error("Lattice energy input exceeds bounded size");
  if (m_state.sums_ == nullptr || m_state.output_ == nullptr)
    throw std::runtime_error("Null lattice energy output");
  Parameters p{};
  for (int i = 0; i < 8; ++i) {
    if (m_state.data_[i] == nullptr)
      throw std::runtime_error("Null lattice energy input");
    p.data_[i] = m_state.data_[i];
  }
  const std::array<int, 2> counts{d[0] - 1, int(half_count * d[6])};
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Lattice energy capture invalidated");
  workspace.Resize(counts, stream, capture == cudaStreamCaptureStatusActive);
  for (int i = 0; i < 2; ++i) {
    p.terms_[i] = workspace.terms_[i];
    p.counts_[i] = counts[i];
    p.invalid_[i] = workspace.invalid_ + i;
    p.prefix_[i] = workspace.prefix_[i];
  }
  p.sums_ = m_state.sums_;
  p.output_ = m_state.output_;
  p.n_theta_ = d[5];
  p.delta_s_ = m_state.scalars_[0];
  p.statistics_ = workspace.statistics_;
  PrepareTerms<<<(counts[1] + 255) / 256, 256, 0, stream>>>(p);
  Check(cudaGetLastError());
  for (int i = 0; i < 2; ++i) {
    ProposeGpuLatticePrefix(workspace.contexts_[i], workspace.terms_[i],
                            workspace.prefix_[i], corrupt);
    Check(cudaMemcpyAsync(p.invalid_[i],
                          GpuLatticePrefixInvalidDevice(workspace.contexts_[i]),
                          sizeof(int), cudaMemcpyDeviceToDevice, stream));
  }
  VerifySignedPrefixes<<<(counts[1] + 255) / 256, 256, 0, stream>>>(p);
  Check(cudaGetLastError());
  FinishCertified<<<1, 2, 0, stream>>>(p);
  Check(cudaGetLastError());
  OrderedFallback<<<3, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  Finish<<<1, 1, 0, stream>>>(p);
  Check(cudaGetLastError());
}
}  // namespace

extern "C" void LaunchGpuLatticeEnergies(GpuDiagnosticsDeviceState& m_state) {
  Launch(m_state, false);
}
extern "C" void LaunchGpuLatticeEnergiesPoisoned(
    GpuDiagnosticsDeviceState& m_state) {
  Launch(m_state, true);
}
extern "C" void ReadGpuLatticeEnergiesStats(
    std::array<unsigned long long, 8>& m_statistics) {
  if (workspace.counts_[0] == 0)
    throw std::runtime_error("Lattice energy workspace uninitialized");
  for (int i = 0; i < 2; ++i) {
    std::array<unsigned long long, 4> statistics;
    ReadGpuLatticePrefixStats(workspace.contexts_[i], statistics);
    for (int j = 0; j < 4; ++j) m_statistics[4 * i + j] = statistics[j];
  }
  std::array<unsigned long long, 4> certified;
  Check(cudaMemcpy(certified.data(), workspace.statistics_, sizeof(certified),
                   cudaMemcpyDeviceToHost));
  for (int i = 0; i < 2; ++i) {
    m_statistics[4 * i + 1] = certified[2 * i];
    m_statistics[4 * i + 2] = certified[2 * i + 1];
  }
}
