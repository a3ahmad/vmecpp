// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_diagnostics.h"

namespace {
struct State {
  cuda::std::array<const double*, 14> data_;
  int half_count_;
  int coefficient_count_;
  int modes_;
  int n_theta_;
  int ns_;
  double delta_s_;
  double lamscale_;
  double* sums_;
  double* output_;
};
void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void OrderedSums(State m_state) {
  __shared__ double terms[512];
  const int component = blockIdx.x;
  const int count = component < 2 ? m_state.ns_ - 1
                                  : (component == 5 ? m_state.coefficient_count_
                                                    : m_state.half_count_);
  const int channels = component == 5 ? 4 : 1;
  const auto& p = m_state.data_;
  double sum = 0.0;
  for (int start = 0; start < count; start += 128) {
    const int i = start + threadIdx.x;
    if (i < count) {
      double value = 0.0;
      if (component == 0) value = p[7][i];
      if (component == 1) value = (p[6][i] / 1.0) * p[7][i];
      if (component == 2) {
        const double magnetic = 0.5 * (p[1][i] * p[3][i] + p[2][i] * p[4][i]);
        value = p[0][i] * magnetic * p[5][i % m_state.n_theta_];
      }
      if (component == 3)
        value = p[8][i] * p[9][i] * p[9][i] * p[5][i % m_state.n_theta_];
      if (component == 4)
        value = (p[3][i] * p[3][i] + p[4][i] * p[4][i]) *
                p[5][i % m_state.n_theta_];
      if (component < 5)
        terms[threadIdx.x] = value;
      else {
        // rzNorm's source order is rcc (except the 00 offset), zsc, rss, zcs.
        terms[4 * threadIdx.x] = p[10][i] * p[10][i];
        terms[4 * threadIdx.x + 1] = p[12][i] * p[12][i];
        terms[4 * threadIdx.x + 2] = p[11][i] * p[11][i];
        terms[4 * threadIdx.x + 3] = p[13][i] * p[13][i];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int remaining = count - start;
      const int length = remaining < 128 ? remaining : 128;
      for (int index = 0; index < length; ++index) {
        for (int channel = 0; channel < channels; ++channel) {
          if (component == 5 && channel == 0 &&
              (start + index) % m_state.modes_ == 0)
            continue;
          sum += terms[channels * index + channel];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) m_state.sums_[component] = sum;
}

__global__ void Finish(State m_state, bool refresh_norms) {
  const double volume = refresh_norms ? 0.0 + m_state.sums_[0] * m_state.delta_s_ : m_state.output_[0];
  const double thermal = 0.0 + m_state.sums_[1] * m_state.delta_s_;
  const double magnetic = 0.0 + fabs(m_state.sums_[2]) * m_state.delta_s_;
  const double density = (magnetic < thermal ? thermal : magnetic) / volume;
  m_state.output_[1] = thermal;
  m_state.output_[2] = magnetic;
  m_state.output_[3] = magnetic + thermal / (0.0 - 1.0);
  if (refresh_norms) {
    m_state.output_[0] = volume;
    m_state.output_[4] = 1.0 / ((0.0 + m_state.sums_[3]) * density * density);
    m_state.output_[5] =
        1.0 / ((0.0 + m_state.sums_[4]) * m_state.lamscale_ * m_state.lamscale_);
    m_state.output_[6] = 1.0 / (0.0 + m_state.sums_[5]);
  }
}
}  // namespace

namespace {
void LaunchDiagnostics(GpuDiagnosticsDeviceState& m_state, bool refresh_norms) {
  const auto& d = m_state.dimensions_;
  if (m_state.scalars_[1] != 0.0 || d[3] != d[0] - 1 || d[4] != d[0]) {
    throw std::runtime_error(
        "Diagnostics require gamma=0 and one fixed-boundary partition");
  }
  State state{};
  for (int i = 0; i < 14; ++i) state.data_[i] = m_state.data_[i];
  state.half_count_ = (d[0] - 1) * d[5] * d[6];
  state.modes_ = d[1] * (d[2] + 1);
  state.coefficient_count_ = d[4] * state.modes_;
  state.n_theta_ = d[5];
  state.ns_ = d[0];
  state.delta_s_ = m_state.scalars_[0];
  state.lamscale_ = m_state.scalars_[2];
  state.sums_ = m_state.sums_;
  state.output_ = m_state.output_;
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  OrderedSums<<<refresh_norms ? 6 : 3, 128, 0, stream>>>(state);
  Check(cudaGetLastError());
  Finish<<<1, 1, 0, stream>>>(state, refresh_norms);
  Check(cudaGetLastError());
}
}  // namespace
extern "C" void LaunchGpuDiagnostics(GpuDiagnosticsDeviceState& m_state) {
  LaunchDiagnostics(m_state, true);
}
extern "C" void LaunchGpuEnergies(GpuDiagnosticsDeviceState& m_state) {
  LaunchDiagnostics(m_state, false);
}

extern "C" void RunGpuDiagnostics(const GpuDiagnosticsInput& input,
                                  GpuDiagnosticsOutput& m_output) {
  if (m_output.sizes_[0] != 7)
    throw std::runtime_error("Diagnostics output shape mismatch");
  std::vector<void*> allocations;
  const auto allocate = [&](std::size_t count) {
    double* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  };
  try {
    GpuDiagnosticsDeviceState state{};
    state.dimensions_ = input.dimensions_;
    state.scalars_ = input.scalars_;
    for (int i = 0; i < 14; ++i) {
      double* pointer = allocate(input.sizes_[i]);
      Check(cudaMemcpy(pointer, input.data_[i],
                       input.sizes_[i] * sizeof(double),
                       cudaMemcpyHostToDevice));
      state.data_[i] = pointer;
    }
    state.sums_ = allocate(6);
    state.output_ = allocate(7);
    LaunchGpuDiagnostics(state);
    Check(cudaMemcpy(m_output.data_[0], state.output_, 7 * sizeof(double),
                     cudaMemcpyDeviceToHost));
    for (void* pointer : allocations) Check(cudaFree(pointer));
  } catch (...) {
    for (void* pointer : allocations) cudaFree(pointer);
    throw;
  }
}
