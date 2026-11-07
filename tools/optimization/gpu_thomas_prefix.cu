// SPDX-License-Identifier: MIT
// Full original RN recurrence certificate; never an equation-residual test.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_thomas_prefix.h"

#ifndef VMECPP_THOMAS_FORWARD_PREFETCH
#define VMECPP_THOMAS_FORWARD_PREFETCH 1
#endif
#ifndef VMECPP_THOMAS_BACKWARD_PREFETCH
#define VMECPP_THOMAS_BACKWARD_PREFETCH 1
#endif
static_assert(VMECPP_THOMAS_FORWARD_PREFETCH == 1 ||
              VMECPP_THOMAS_FORWARD_PREFETCH == 2 ||
              VMECPP_THOMAS_FORWARD_PREFETCH == 4 ||
              VMECPP_THOMAS_FORWARD_PREFETCH == 8);
static_assert(VMECPP_THOMAS_BACKWARD_PREFETCH == 1 ||
              VMECPP_THOMAS_BACKWARD_PREFETCH == 2 ||
              VMECPP_THOMAS_BACKWARD_PREFETCH == 4 ||
              VMECPP_THOMAS_BACKWARD_PREFETCH == 8);

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Parameters {
  cuda::std::array<double*, 4> forces_;
  cuda::std::array<const double*, 2> lower_, upper_, denominators_;
  double* reciprocal_;
  double* proposed_;
  int* invalid_;
  int ns_, modes_, n_count_, force_count_, matrix_count_;
};
__global__ void Reciprocals(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * m_p.matrix_count_) return;
  const int component = i / m_p.matrix_count_;
  const int local = i % m_p.matrix_count_;
  const int mode = local % m_p.modes_;
  const int row = local / m_p.modes_;
  const int first = mode / m_p.n_count_ > 0 ? 1 : 0;
  m_p.reciprocal_[i] =
      row < first ? 0.0 : __ddiv_rn(1.0, m_p.denominators_[component][local]);
}
__global__ void Proposal(Parameters m_p) {
  const int mode = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = blockIdx.y;
  if (mode >= m_p.modes_) return;
  const int component = channel / 2;
  const int first = mode / m_p.n_count_ > 0 ? 1 : 0;
  const double* __restrict__ rhs = m_p.forces_[channel];
  const double* __restrict__ lower = m_p.lower_[component];
  const double* __restrict__ denominator = m_p.denominators_[component];
  const double* __restrict__ reciprocal =
      m_p.reciprocal_ + component * m_p.matrix_count_;
  double* __restrict__ proposal = m_p.proposed_ + channel * m_p.force_count_;
#if VMECPP_THOMAS_FORWARD_PREFETCH == 1
  double previous = 0.0;
  for (int j = first; j < m_p.ns_ - 1; ++j) {
    const int i = j * m_p.modes_ + mode;
    const double numerator =
        j == first ? rhs[i] : __dsub_rn(rhs[i], __dmul_rn(previous, lower[i]));
    const double estimate = __dmul_rn(numerator, reciprocal[i]);
    const double residual = __fma_rn(-estimate, denominator[i], numerator);
    previous = __fma_rn(residual, reciprocal[i], estimate);
    proposal[i] = previous;
  }
#else
  constexpr int kRows = VMECPP_THOMAS_FORWARD_PREFETCH;
  double previous = 0.0;
  for (int base = first; base < m_p.ns_ - 1; base += kRows) {
    cuda::std::array<double, kRows> rhs_values;
    cuda::std::array<double, kRows> lower_values;
    cuda::std::array<double, kRows> denominators;
    cuda::std::array<double, kRows> reciprocals;
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const int j = base + row;
      if (j < m_p.ns_ - 1) {
        const int i = j * m_p.modes_ + mode;
        rhs_values[row] = rhs[i];
        lower_values[row] = lower[i];
        denominators[row] = denominator[i];
        reciprocals[row] = reciprocal[i];
      }
    }
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const int j = base + row;
      if (j < m_p.ns_ - 1) {
        const double numerator =
            j == first ? rhs_values[row]
                       : __dsub_rn(rhs_values[row],
                                   __dmul_rn(previous, lower_values[row]));
        const double estimate = __dmul_rn(numerator, reciprocals[row]);
        const double residual =
            __fma_rn(-estimate, denominators[row], numerator);
        previous = __fma_rn(residual, reciprocals[row], estimate);
        proposal[j * m_p.modes_ + mode] = previous;
      }
    }
  }
#endif
}
__global__ void Poison(Parameters m_p) {
  const int row = (m_p.ns_ - 1) / 2;
  const int i = row * m_p.modes_;
  m_p.proposed_[i] =
      __longlong_as_double(__double_as_longlong(m_p.proposed_[i]) ^ 1LL);
}
__global__ void VerifyEveryRow(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * m_p.matrix_count_) return;
  const int channel = i / m_p.matrix_count_;
  const int local = i % m_p.matrix_count_;
  const int mode = local % m_p.modes_;
  const int row = local / m_p.modes_;
  const int first = mode / m_p.n_count_ > 0 ? 1 : 0;
  if (row < first) return;
  const int component = channel / 2;
  const double* proposal = m_p.proposed_ + channel * m_p.force_count_;
  const double numerator =
      row == first ? m_p.forces_[channel][local]
                   : __dsub_rn(m_p.forces_[channel][local],
                               __dmul_rn(proposal[local - m_p.modes_],
                                         m_p.lower_[component][local]));
  const double reference =
      __ddiv_rn(numerator, m_p.denominators_[component][local]);
  if (__double_as_longlong(reference) != __double_as_longlong(proposal[local]))
    atomicAdd(m_p.invalid_ + channel * m_p.modes_ + mode, 1);
}
__global__ void OriginalForwardFallback(Parameters m_p) {
  const int mode = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = blockIdx.y;
  if (mode >= m_p.modes_ || m_p.invalid_[channel * m_p.modes_ + mode] == 0)
    return;
  const int component = channel / 2;
  const int first = mode / m_p.n_count_ > 0 ? 1 : 0;
  double* proposal = m_p.proposed_ + channel * m_p.force_count_;
  double previous =
      __ddiv_rn(m_p.forces_[channel][first * m_p.modes_ + mode],
                m_p.denominators_[component][first * m_p.modes_ + mode]);
  proposal[first * m_p.modes_ + mode] = previous;
  for (int j = first + 1; j < m_p.ns_ - 1; ++j) {
    const int i = j * m_p.modes_ + mode;
    const double numerator = __dsub_rn(
        m_p.forces_[channel][i], __dmul_rn(previous, m_p.lower_[component][i]));
    previous = __ddiv_rn(numerator, m_p.denominators_[component][i]);
    proposal[i] = previous;
  }
}
__global__ void OriginalBackwardScatter(Parameters m_p) {
  const int mode = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = blockIdx.y;
  if (mode >= m_p.modes_) return;
  const int first = mode / m_p.n_count_ > 0 ? 1 : 0;
  const double* proposal = m_p.proposed_ + channel * m_p.force_count_;
  const double* upper = m_p.upper_[channel / 2];
  int i = (m_p.ns_ - 2) * m_p.modes_ + mode;
  double next = proposal[i];
  m_p.forces_[channel][i] = next;
#if VMECPP_THOMAS_BACKWARD_PREFETCH == 1
  for (int j = m_p.ns_ - 3; j >= first; --j) {
    i = j * m_p.modes_ + mode;
    next = __dsub_rn(proposal[i], __dmul_rn(upper[i], next));
    m_p.forces_[channel][i] = next;
  }
#else
  constexpr int kRows = VMECPP_THOMAS_BACKWARD_PREFETCH;
  for (int base = m_p.ns_ - 3; base >= first; base -= kRows) {
    cuda::std::array<double, kRows> proposals;
    cuda::std::array<double, kRows> uppers;
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const int j = base - row;
      if (j >= first) {
        const int index = j * m_p.modes_ + mode;
        proposals[row] = proposal[index];
        uppers[row] = upper[index];
      }
    }
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const int j = base - row;
      if (j >= first) {
        next = __dsub_rn(proposals[row], __dmul_rn(uppers[row], next));
        m_p.forces_[channel][j * m_p.modes_ + mode] = next;
      }
    }
  }
#endif
  for (int j = 0; j < first; ++j)
    m_p.forces_[channel][j * m_p.modes_ + mode] = 0.0;
}
struct Context {
  std::array<int, 5> dimensions_;
  Parameters parameters_{};
  cudaStream_t stream_ = nullptr;
  int device_;
  ~Context() {
    cudaStreamSynchronize(stream_);
    if (parameters_.proposed_) cudaFree(parameters_.proposed_);
    if (parameters_.reciprocal_) cudaFree(parameters_.reciprocal_);
    if (parameters_.invalid_) cudaFree(parameters_.invalid_);
  }
  Parameters View(const GpuIterationDeviceState& state) const {
    if (state.dimensions_ != dimensions_ || state.stream_ != stream_)
      throw std::runtime_error("Thomas prefix dimensions/stream changed");
    int device;
    Check(cudaGetDevice(&device));
    if (device != device_)
      throw std::runtime_error("Thomas prefix device changed");
    auto result = parameters_;
    for (int i = 0; i < 4; ++i) {
      if (!state.forces_[i]) throw std::runtime_error("Null Thomas force");
      result.forces_[i] = state.forces_[i];
    }
    for (int i = 0; i < 2; ++i) {
      if (!state.factored_upper_[i] || !state.denominators_[i] ||
          !state.constants_[8 + i])
        throw std::runtime_error("Null Thomas factors");
      result.lower_[i] = state.constants_[8 + i];
      result.upper_[i] = state.factored_upper_[i];
      result.denominators_[i] = state.denominators_[i];
    }
    return result;
  }
};
void Launch(void* context, GpuIterationDeviceState& m_state, bool poison) {
  auto& ctx = *static_cast<Context*>(context);
  const auto p = ctx.View(m_state);
  Check(
      cudaMemsetAsync(p.invalid_, 0, 4 * p.modes_ * sizeof(int), ctx.stream_));
  const dim3 grid((p.modes_ + 31) / 32, 4);
  Proposal<<<grid, 32, 0, ctx.stream_>>>(p);
  Check(cudaGetLastError());
  if (poison) {
    Poison<<<1, 1, 0, ctx.stream_>>>(p);
    Check(cudaGetLastError());
  }
  VerifyEveryRow<<<(4 * p.matrix_count_ + 127) / 128, 128, 0, ctx.stream_>>>(p);
  Check(cudaGetLastError());
  // Kernel completion separates ALL immutable-proposal reads from fallback
  // writes. Original b remains untouched until the final backward/scatter.
  OriginalForwardFallback<<<grid, 32, 0, ctx.stream_>>>(p);
  Check(cudaGetLastError());
  OriginalBackwardScatter<<<grid, 32, 0, ctx.stream_>>>(p);
  Check(cudaGetLastError());
}
void OutsideCapture(const Context& ctx) {
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(ctx.stream_, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error(
        "Thomas snapshot requires completed noncapture use");
  Check(cudaStreamSynchronize(ctx.stream_));
}
}  // namespace

extern "C" void* CreateGpuThomasPrefix(const GpuIterationDeviceState& state) {
  const auto& d = state.dimensions_;
  if (d[0] < 3 || d[0] > 4096 || d[1] < 1 || d[1] > 256 || d[2] < 0 ||
      d[2] > 256 || d[3] != d[0] - 1 || d[4] != d[0] ||
      std::size_t(d[0]) * d[1] * (d[2] + 1) > (1 << 22))
    throw std::runtime_error("Unsupported Thomas prefix shape");
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(static_cast<cudaStream_t>(state.stream_),
                              &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("Create Thomas prefix before capture");
  auto* ctx = new Context;
  try {
    ctx->dimensions_ = d;
    ctx->stream_ = static_cast<cudaStream_t>(state.stream_);
    Check(cudaGetDevice(&ctx->device_));
    auto& p = ctx->parameters_;
    p.ns_ = d[0];
    p.n_count_ = d[2] + 1;
    p.modes_ = d[1] * p.n_count_;
    p.force_count_ = p.ns_ * p.modes_;
    p.matrix_count_ = (p.ns_ - 1) * p.modes_;
    Check(cudaMalloc(&p.proposed_,
                     4 * std::size_t(p.force_count_) * sizeof(double)));
    Check(cudaMalloc(&p.reciprocal_,
                     2 * std::size_t(p.matrix_count_) * sizeof(double)));
    Check(cudaMalloc(&p.invalid_, 4 * p.modes_ * sizeof(int)));
    Check(cudaMemsetAsync(p.proposed_, 0,
                          4 * std::size_t(p.force_count_) * sizeof(double),
                          ctx->stream_));
    Check(cudaMemsetAsync(p.invalid_, 0xff, 4 * p.modes_ * sizeof(int),
                          ctx->stream_));
    RefreshGpuThomasPrefix(ctx, state);
    Check(cudaStreamSynchronize(ctx->stream_));
    return ctx;
  } catch (...) {
    delete ctx;
    throw;
  }
}
extern "C" void RefreshGpuThomasPrefix(void* m_context,
                                       const GpuIterationDeviceState& state) {
  auto& ctx = *static_cast<Context*>(m_context);
  const auto p = ctx.View(state);
  Reciprocals<<<(2 * p.matrix_count_ + 127) / 128, 128, 0, ctx.stream_>>>(p);
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuThomasPrefix(void* m_context,
                                      GpuIterationDeviceState& m_state) {
  Launch(m_context, m_state, false);
}
extern "C" void LaunchGpuThomasPrefixPoisoned(
    void* m_context, GpuIterationDeviceState& m_state) {
  Launch(m_context, m_state, true);
}
extern "C" void ReadGpuThomasPrefixFlags(
    void* m_context, std::array<unsigned long long, 4>& m_statistics) {
  const auto& ctx = *static_cast<Context*>(m_context);
  OutsideCapture(ctx);
  std::vector<int> flags(4 * ctx.parameters_.modes_);
  Check(cudaMemcpy(flags.data(), ctx.parameters_.invalid_,
                   flags.size() * sizeof(int), cudaMemcpyDeviceToHost));
  m_statistics = {flags.size(), 0, 0, 0};
  for (int value : flags) {
    if (value < 0) throw std::runtime_error("Thomas prefix has not executed");
    ++m_statistics[value == 0 ? 1 : 2];
    m_statistics[3] += value;
  }
}
extern "C" void ReadGpuThomasPrefixForward(
    void* m_context, const std::array<double*, 4>& m_outputs) {
  const auto& ctx = *static_cast<Context*>(m_context);
  OutsideCapture(ctx);
  for (int channel = 0; channel < 4; ++channel)
    Check(cudaMemcpy(
        m_outputs[channel],
        ctx.parameters_.proposed_ + channel * ctx.parameters_.force_count_,
        ctx.parameters_.force_count_ * sizeof(double), cudaMemcpyDeviceToHost));
}
extern "C" void DestroyGpuThomasPrefix(void* m_context) {
  delete static_cast<Context*>(m_context);
}
