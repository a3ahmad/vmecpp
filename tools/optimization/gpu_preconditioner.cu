// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Fixed-boundary radial/lambda preconditioner refresh, preserving each original
// angular sum and scalar expression order. Grid-dependent libm powers are
// inputs.
#include <cuda_runtime.h>

#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_preconditioner.h"

namespace {
struct Parameters {
  cuda::std::array<const double*, 26> input_;
  cuda::std::array<double*, 20> output_;
  int ns_;
  int mpol_;
  int ntor_;
  int nfp_;
  int n_theta_;
  int angular_;
  double delta_s_;
  double lamscale_;
  double damping_;
  double tcon0_;
  double* terms_;
  double* sums_;
  int* invalid_norm_;
};
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void PointTerms(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = (m_p.ns_ - 1) * m_p.angular_;
  if (i >= count) return;
  const int component = blockIdx.y;
  const auto& p = m_p.input_;
  const int j = i / m_p.angular_;
  const int l = i % m_p.angular_ % m_p.n_theta_;
  const int base = 8 + 5 * component;
  const double p_tau = -4.0 * p[0][i] * p[1][i] / p[2][i] * p[22][l];
  const double t1a = p[base + 1][i] / m_p.delta_s_;
  const double t2a = 0.25 *
                     (p[base + 2][i + m_p.angular_] / p[18][j] +
                      p[base + 3][i + m_p.angular_]) /
                     p[18][j];
  const double t3a =
      0.25 * (p[base + 2][i] / p[18][j] + p[base + 3][i]) / p[18][j];
  const double t1b =
      0.5 * (p[base][i] + 0.5 / p[18][j] * p[base + 4][i + m_p.angular_]);
  const double t2b = 0.5 * (p[base][i] + 0.5 / p[18][j] * p[base + 4][i]);
  const cuda::std::array<double, 8> terms{
      p_tau * t1a * t1a,
      p_tau * (t1a + t2a) * (-t1a + t3a),
      p_tau * (t1a + t2a) * (t1a + t2a),
      p_tau * (-t1a + t3a) * (-t1a + t3a),
      p_tau * t1b * t2b,
      p_tau * t1b * t1b,
      p_tau * t2b * t2b,
      0.25 * -4.0 * p[3][i] * p[3][i] * p[4][i] * p[22][l]};
  for (int channel = 0; channel < 8; ++channel) {
    m_p.terms_[(8 * component + channel) * count + i] = terms[channel];
  }
  if (component == 0) {
    m_p.terms_[16 * count + i] = p[5][i] / p[4][i] * p[22][l];
    m_p.terms_[17 * count + i] = p[6][i] / p[4][i] * p[22][l];
    m_p.terms_[18 * count + i] = p[7][i] / p[4][i] * p[22][l];
    m_p.terms_[19 * count + i] = p[23][i] * p[23][i] * p[22][l];
    m_p.terms_[20 * count + i] = p[24][i] * p[24][i] * p[22][l];
  }
}

__global__ void OrderedAngularSums(Parameters m_p) {
  __shared__ cuda::std::array<cuda::std::array<double, 129>, 21> tile;
  const int j = blockIdx.x;
  const int count = (m_p.ns_ - 1) * m_p.angular_;
  double sum = 0.0;
  for (int start = 0; start < m_p.angular_; start += 128) {
    const int l = start + threadIdx.x;
    if (l < m_p.angular_) {
      for (int channel = 0; channel < 21; ++channel) {
        tile[channel][threadIdx.x] =
            m_p.terms_[channel * count + j * m_p.angular_ + l];
      }
    }
    __syncthreads();
    if (threadIdx.x < 21) {
      const int remaining = m_p.angular_ - start;
      const int length = remaining < 128 ? remaining : 128;
      for (int i = 0; i < length; ++i) sum += tile[threadIdx.x][i];
    }
    __syncthreads();
  }
  if (threadIdx.x < 21) m_p.sums_[threadIdx.x * (m_p.ns_ - 1) + j] = sum;
}

__device__ double Sum(const Parameters& p, int channel, int j) {
  return p.sums_[channel * (p.ns_ - 1) + j];
}

__global__ void RadialProfiles(Parameters m_p) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > m_p.ns_) return;
  const auto& p = m_p.input_;
  const auto& o = m_p.output_;
  for (int channel = 0; channel < 3; ++channel) {
    double value = 0.0;
    if (j == 0)
      value = Sum(m_p, 16 + channel, 0);
    else if (j < m_p.ns_) {
      value = 0.5 * ((j < m_p.ns_ - 1 ? Sum(m_p, 16 + channel, j) : 0.0) +
                     Sum(m_p, 16 + channel, j - 1));
    }
    o[1 + channel][j] = value;
  }
  if (j >= m_p.ns_ - 1) return;
  for (int component = 0; component < 2; ++component) {
    const int channel = 8 * component;
    const int output = 4 + 4 * component;
    o[output][2 * j] = -Sum(m_p, channel, j);
    o[output][2 * j + 1] = Sum(m_p, channel + 1, j) * p[20][j] * p[21][j];
    o[output + 2][2 * j] = Sum(m_p, channel + 4, j);
    o[output + 2][2 * j + 1] = Sum(m_p, channel + 4, j) * p[20][j] * p[21][j];
    o[output + 1][2 * j] =
        (j > 0 ? Sum(m_p, channel, j - 1) : 0.0) + Sum(m_p, channel, j);
    o[output + 1][2 * j + 1] =
        (j > 0 ? Sum(m_p, channel + 2, j - 1) * p[20][j - 1] * p[20][j - 1]
               : 0.0) +
        Sum(m_p, channel + 3, j) * p[21][j] * p[21][j];
    o[output + 3][2 * j] =
        (j > 0 ? Sum(m_p, channel + 5, j - 1) : 0.0) + Sum(m_p, channel + 6, j);
    o[output + 3][2 * j + 1] =
        (j > 0 ? Sum(m_p, channel + 5, j - 1) * p[20][j - 1] * p[20][j - 1]
               : 0.0) +
        Sum(m_p, channel + 6, j) * p[21][j] * p[21][j];
  }
  o[12][j] = (j > 0 ? Sum(m_p, 15, j - 1) : 0.0) + Sum(m_p, 15, j);
  if (j == 0) {
    o[13][0] = 0.0;
    return;
  }
  const double ar_norm = Sum(m_p, 19, j);
  const double az_norm = Sum(m_p, 20, j);
  if (ar_norm == 0.0 || az_norm == 0.0) {
    atomicOr(m_p.invalid_norm_,
             (ar_norm == 0.0 ? 1 : 0) | (az_norm == 0.0 ? 2 : 0));
    o[13][j] = 0.0;
    return;
  }
  double multiplier =
      m_p.tcon0_ * (1.0 + m_p.ns_ * (1.0 / 60.0 + m_p.ns_ / (200.0 * 120.0)));
  multiplier /= (4.0 * 4.0);
  const double ar = fabs(o[5][2 * j] / ar_norm);
  const double az = fabs(o[9][2 * j] / az_norm);
  const double base = az < ar ? az : ar;
  o[13][j] = base * multiplier * 32 * m_p.delta_s_ * 32 * m_p.delta_s_;
}

__global__ void FourierAssembly(Parameters m_p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int modes = m_p.mpol_ * (m_p.ntor_ + 1);
  if (i >= m_p.ns_ * modes) return;
  const int j = i / modes;
  const int m = (i % modes) / (m_p.ntor_ + 1);
  const int n = i % (m_p.ntor_ + 1);
  const auto& o = m_p.output_;
  if (i == 0) o[13][m_p.ns_ - 1] = 0.5 * o[13][m_p.ns_ - 2];
  double lambda = 0.0;
  if (j > 0 && (m > 0 || n > 0)) {
    const double p_factor =
        m_p.damping_ / (4.0 * m_p.lamscale_ * m_p.lamscale_);
    const double tnn = n * m_p.nfp_ * n * m_p.nfp_;
    const int tmm = m * m;
    const double tmn = 2.0 * m * n * m_p.nfp_;
    double faclam =
        tnn * o[1][j] + tmn * copysign(o[2][j], o[1][j]) + tmm * o[3][j];
    if (faclam == 0.0) faclam = -1.0e-10;
    lambda = p_factor / faclam * m_p.input_[25][j * m_p.mpol_ + m];
  }
  o[0][i] = lambda;
  if (j >= m_p.ns_ - 1) return;
  for (int component = 0; component < 2; ++component) {
    const int output = 4 + 4 * component;
    const int parity = m % 2;
    double upper = 0.0, diagonal = 0.0, lower = 0.0;
    if (j > 0 || m == 0) {
      upper =
          -(o[output][2 * j + parity] + o[output + 2][2 * j + parity] * m * m);
      diagonal = -(o[output + 1][2 * j + parity] +
                   o[output + 3][2 * j + parity] * m * m +
                   o[12][j] * n * m_p.nfp_ * n * m_p.nfp_);
      if (j > 0)
        lower = -(o[output][2 * (j - 1) + parity] +
                  o[output + 2][2 * (j - 1) + parity] * m * m);
      if (j == 1 && m == 1) diagonal += lower;
    }
    o[14 + component][i] = upper;
    o[16 + component][i] = diagonal;
    o[18 + component][i] = lower;
  }
}
}  // namespace

extern "C" void LaunchGpuPreconditioner(GpuPreconditionerDeviceState& m_state) {
  const auto& d = m_state.dimensions_;
  if (d[0] < 3 || d[3] != d[0] - 1 || d[4] != d[0]) {
    throw std::runtime_error(
        "Preconditioner requires one fixed-boundary partition");
  }
  Parameters p{};
  for (int i = 0; i < 26; ++i) p.input_[i] = m_state.input_[i];
  for (int i = 0; i < 20; ++i) p.output_[i] = m_state.output_[i];
  p.ns_ = d[0];
  p.mpol_ = d[1];
  p.ntor_ = d[2];
  p.n_theta_ = d[5];
  p.angular_ = d[5] * d[6];
  p.nfp_ = d[7];
  p.delta_s_ = m_state.scalars_[0];
  p.lamscale_ = m_state.scalars_[1];
  p.damping_ = m_state.scalars_[2];
  p.tcon0_ = m_state.scalars_[3];
  p.terms_ = m_state.terms_;
  p.sums_ = m_state.sums_;
  p.invalid_norm_ = m_state.invalid_norm_;
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  Check(cudaMemsetAsync(p.invalid_norm_, 0, sizeof(int), stream));
  PointTerms<<<dim3(((d[0] - 1) * p.angular_ + 127) / 128, 2), 128, 0,
               stream>>>(p);
  Check(cudaGetLastError());
  OrderedAngularSums<<<d[0] - 1, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  RadialProfiles<<<(d[0] + 1 + 127) / 128, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  FourierAssembly<<<(d[0] * d[1] * (d[2] + 1) + 127) / 128, 128, 0, stream>>>(
      p);
  Check(cudaGetLastError());
}

extern "C" void RunGpuPreconditioner(const GpuPreconditionerInput& input,
                                     GpuPreconditionerOutput& m_output) {
  std::vector<void*> allocations;
  const auto allocate = [&](std::size_t bytes) {
    void* pointer = nullptr;
    Check(cudaMalloc(&pointer, bytes));
    allocations.push_back(pointer);
    return pointer;
  };
  try {
    GpuPreconditionerDeviceState state{};
    state.dimensions_ = input.dimensions_;
    state.scalars_ = input.scalars_;
    for (int i = 0; i < 26; ++i) {
      auto* pointer =
          static_cast<double*>(allocate(input.sizes_[i] * sizeof(double)));
      Check(cudaMemcpy(pointer, input.data_[i],
                       input.sizes_[i] * sizeof(double),
                       cudaMemcpyHostToDevice));
      state.input_[i] = pointer;
    }
    for (int i = 0; i < 20; ++i)
      state.output_[i] =
          static_cast<double*>(allocate(m_output.sizes_[i] * sizeof(double)));
    const auto& d = input.dimensions_;
    state.terms_ = static_cast<double*>(
        allocate(std::size_t{21} * (d[0] - 1) * d[5] * d[6] * sizeof(double)));
    state.sums_ = static_cast<double*>(
        allocate(std::size_t{21} * (d[0] - 1) * sizeof(double)));
    state.invalid_norm_ = static_cast<int*>(allocate(sizeof(int)));
    LaunchGpuPreconditioner(state);
    int invalid = 0;
    Check(cudaMemcpy(&invalid, state.invalid_norm_, sizeof(int),
                     cudaMemcpyDeviceToHost));
    if (invalid != 0)
      throw std::runtime_error("Constraint multiplier has zero radial norm");
    for (int i = 0; i < 20; ++i)
      Check(cudaMemcpy(m_output.data_[i], state.output_[i],
                       m_output.sizes_[i] * sizeof(double),
                       cudaMemcpyDeviceToHost));
    for (void* pointer : allocations) Check(cudaFree(pointer));
  } catch (...) {
    for (void* pointer : allocations) cudaFree(pointer);
    throw;
  }
}
