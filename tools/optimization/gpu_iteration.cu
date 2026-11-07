// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <cuda/std/array>
#include <stdexcept>
#include <vector>

#include "gpu_iteration.h"

namespace {
#ifndef VMECPP_RESIDUAL_TILE
#define VMECPP_RESIDUAL_TILE 128
#endif
#ifndef VMECPP_RHS_MODE_TILE
#define VMECPP_RHS_MODE_TILE 32
#endif
// Only RHS chain placement changes; factorization and per-chain arithmetic do
// not.
constexpr int kRhsModeTile = VMECPP_RHS_MODE_TILE;
static_assert(kRhsModeTile == 1 || kRhsModeTile == 2 || kRhsModeTile == 4 ||
              kRhsModeTile == 8 || kRhsModeTile == 16 || kRhsModeTile == 32);
#ifndef VMECPP_RHS_MEMORY_POLICY
#define VMECPP_RHS_MEMORY_POLICY 0
#endif
constexpr int kRhsMemoryPolicy = VMECPP_RHS_MEMORY_POLICY;
static_assert(kRhsMemoryPolicy >= 0 && kRhsMemoryPolicy <= 3);
constexpr int kResidualTile = VMECPP_RESIDUAL_TILE;
static_assert(kResidualTile == 32 || kResidualTile == 128);
struct DeviceState {
  cuda::std::array<double*, 6> forces_;
  cuda::std::array<const double*, 11> constants_;
  cuda::std::array<double*, 2> factored_upper_;
  cuda::std::array<double*, 2> denominators_;
  double* residuals_;
  cuda::std::array<int, 5> dimensions_;
};
DeviceState DeviceView(const GpuIterationDeviceState& state) {
  DeviceState result{};
  for (int i = 0; i < 6; ++i) result.forces_[i] = state.forces_[i];
  for (int i = 0; i < 11; ++i) result.constants_[i] = state.constants_[i];
  for (int i = 0; i < 2; ++i)
    result.factored_upper_[i] = state.factored_upper_[i];
  for (int i = 0; i < 2; ++i) result.denominators_[i] = state.denominators_[i];
  for (int i = 0; i < 5; ++i) result.dimensions_[i] = state.dimensions_[i];
  result.residuals_ = state.residuals_;
  return result;
}
struct StepState {
  cuda::std::array<double*, 6> geometry_;
  cuda::std::array<double*, 6> velocity_;
  cuda::std::array<const double*, 6> forces_;
  int count_;
  double velocity_scale_;
  double conjugation_;
  double time_step_;
};
struct PreparationState {
  cuda::std::array<double*, 6> coefficients_;
  const double* scalxc_;
  int count_;
  int mpol_;
  int n_count_;
  int m1_rows_;
  double m1_scale_;
  bool zero_z_m1_;
};
__global__ void Prepare(PreparationState m_state) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m_state.count_) return;
  const int m = (i / m_state.n_count_) % m_state.mpol_;
  const int j = i / (m_state.mpol_ * m_state.n_count_);
  const double scale = m_state.scalxc_[2 * j + m % 2];
  cuda::std::array<double, 6> values;
  for (int channel = 0; channel < 6; ++channel) {
    values[channel] = m_state.coefficients_[channel][i] * scale;
  }
  if (j < m_state.m1_rows_ && m == 1) {
    const double old_rss = values[1];
    values[1] = (old_rss + values[3]) * m_state.m1_scale_;
    values[3] = (old_rss - values[3]) * m_state.m1_scale_;
    if (m_state.zero_z_m1_) values[3] = 0.0;
  }
  for (int channel = 0; channel < 6; ++channel) {
    m_state.coefficients_[channel][i] = values[channel];
  }
}
__global__ void GeometryAxis(PreparationState m_state) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= m_state.n_count_) return;
  const int axis1 = m_state.n_count_ + n;
  const int first_surface1 = (m_state.mpol_ + 1) * m_state.n_count_ + n;
  for (int channel = 0; channel < 6; ++channel) {
    m_state.coefficients_[channel][axis1] =
        m_state.coefficients_[channel][first_surface1];
  }
  m_state.coefficients_[5][n] =
      m_state.coefficients_[5][m_state.mpol_ * m_state.n_count_ + n];
}

__global__ void Step(StepState m_state) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = blockIdx.y;
  if (index >= m_state.count_) return;
  const double velocity =
      m_state.velocity_scale_ *
      (m_state.conjugation_ * m_state.velocity_[channel][index] +
       m_state.time_step_ * m_state.forces_[channel][index]);
  m_state.velocity_[channel][index] = velocity;
  m_state.geometry_[channel][index] += m_state.time_step_ * velocity;
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

__global__ void M1(DeviceState m_state) {
  const int n_count = m_state.dimensions_[2] + 1;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / n_count;
  const int n = index % n_count;
  if (j >= m_state.dimensions_[3]) return;
  const int parity_index = j * 2 + 1;
  const double ard = m_state.constants_[0][parity_index];
  const double brd = m_state.constants_[1][parity_index];
  const double azd = m_state.constants_[2][parity_index];
  const double bzd = m_state.constants_[3][parity_index];
  const double denominator = ard + brd + azd + bzd;
  const double scale_r = (ard + brd) / denominator;
  const double scale_z = (azd + bzd) / denominator;
  const int coefficient = (j * m_state.dimensions_[1] + 1) * n_count + n;
  m_state.forces_[1][coefficient] *= scale_r;
  m_state.forces_[3][coefficient] *= scale_z;
}

// One lane per independent Fourier-mode system. The j-major layout makes
// adjacent lanes access adjacent doubles without a CPU gather/transpose.
// Each factorization and each RHS keeps the original Thomas operation order.
__global__ void ThomasFactor(DeviceState m_state) {
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  const int mode = blockIdx.x * blockDim.x + threadIdx.x;
  const int component = blockIdx.y;
  if (mode >= modes) return;
  const int first = mode / (m_state.dimensions_[2] + 1) > 0 ? 1 : 0;
  const int end = m_state.dimensions_[0] - 1;
  const double* upper = m_state.constants_[4 + component];
  const double* diagonal = m_state.constants_[6 + component];
  const double* lower = m_state.constants_[8 + component];
  double* a = m_state.factored_upper_[component];
  double* denominators = m_state.denominators_[component];
  denominators[first * modes + mode] = diagonal[first * modes + mode];
  a[first * modes + mode] =
      upper[first * modes + mode] / diagonal[first * modes + mode];
  for (int j = first + 1; j < end - 1; ++j) {
    const int index = j * modes + mode;
    const double denominator =
        diagonal[index] - a[index - modes] * lower[index];
    denominators[index] = denominator;
    a[index] = upper[index] / denominator;
  }
  const int last = (end - 1) * modes + mode;
  denominators[last] = diagonal[last] - a[last - modes] * lower[last];
}

// Solve independent RHS concurrently after common factorization. A cached
// rounded denominator equals the original recomputation with unchanged
// operands.
__global__ void ThomasRightHandSide(DeviceState m_state) {
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  if constexpr (kRhsModeTile < 32) {
    if (threadIdx.x >= kRhsModeTile) return;
  }
  const int mode =
      blockIdx.x * (kRhsModeTile == 32 ? blockDim.x : kRhsModeTile) +
      threadIdx.x;
  const int component = blockIdx.y / 2;
  const int basis = blockIdx.y % 2;
  if (mode >= modes) return;
  const int first = mode / (m_state.dimensions_[2] + 1) > 0 ? 1 : 0;
  const int end = m_state.dimensions_[0] - 1;
#if VMECPP_RHS_MEMORY_POLICY > 0
  // Opt-in scope: Solver and RunGpuIteration own disjoint force, matrix and
  // factor allocations. Arbitrarily overlapping external views are excluded.
  const double* __restrict__ lower = m_state.constants_[8 + component];
  const double* __restrict__ a = m_state.factored_upper_[component];
  const double* __restrict__ denominators = m_state.denominators_[component];
#else
  const double* lower = m_state.constants_[8 + component];
  const double* a = m_state.factored_upper_[component];
  const double* denominators = m_state.denominators_[component];
#endif
  {
#if VMECPP_RHS_MEMORY_POLICY > 0
    double* __restrict__ c = m_state.forces_[2 * component + basis];
#else
    double* c = m_state.forces_[2 * component + basis];
#endif
    for (int j = 0; j < first; ++j) c[j * modes + mode] = 0.0;
    if constexpr (kRhsMemoryPolicy < 2) {
      c[first * modes + mode] /= denominators[first * modes + mode];
      for (int j = first + 1; j < end; ++j) {
        const int index = j * modes + mode;
        const double denominator = denominators[index];
        c[index] = (c[index] - c[index - modes] * lower[index]) / denominator;
      }
      for (int j = end - 2; j >= first; --j) {
        const int index = j * modes + mode;
        c[index] -= a[index] * c[index + modes];
      }
    } else {
      double previous =
          c[first * modes + mode] / denominators[first * modes + mode];
      c[first * modes + mode] = previous;
      int j = first + 1;
      if constexpr (kRhsMemoryPolicy == 3) {
        for (; j + 1 < end; j += 2) {
          const int index = j * modes + mode;
          const int second = index + modes;
          const double rhs0 = c[index];
          const double rhs1 = c[second];
          const double lower0 = lower[index];
          const double lower1 = lower[second];
          const double denominator0 = denominators[index];
          const double denominator1 = denominators[second];
          const double value0 = (rhs0 - previous * lower0) / denominator0;
          const double value1 = (rhs1 - value0 * lower1) / denominator1;
          c[index] = value0;
          c[second] = value1;
          previous = value1;
        }
      }
      for (; j < end; ++j) {
        const int index = j * modes + mode;
        previous = (c[index] - previous * lower[index]) / denominators[index];
        c[index] = previous;
      }
      double next = c[(end - 1) * modes + mode];
      j = end - 2;
      if constexpr (kRhsMemoryPolicy == 3) {
        for (; j - 1 >= first; j -= 2) {
          const int index = j * modes + mode;
          const int second = index - modes;
          const double rhs0 = c[index];
          const double rhs1 = c[second];
          const double upper0 = a[index];
          const double upper1 = a[second];
          const double value0 = rhs0 - upper0 * next;
          const double value1 = rhs1 - upper1 * value0;
          c[index] = value0;
          c[second] = value1;
          next = value1;
        }
      }
      for (; j >= first; --j) {
        const int index = j * modes + mode;
        next = c[index] - a[index] * next;
        c[index] = next;
      }
    }
  }
}

__global__ void Lambda(DeviceState m_state) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = m_state.dimensions_[4] * m_state.dimensions_[1] *
                    (m_state.dimensions_[2] + 1);
  if (index >= count) return;
  m_state.forces_[4][index] *= m_state.constants_[10][index];
  m_state.forces_[5][index] *= m_state.constants_[10][index];
}

// Cooperative loads, sequential additions: a tree reduction would change the
// residual and therefore the subsequent solver trajectory/evaluation identity.
__global__ void Residuals(DeviceState m_state, bool include_edge_rz,
                          const int* invalid_components) {
  if (invalid_components != nullptr && invalid_components[blockIdx.x] == 0)
    return;
  __shared__ double terms[2 * kResidualTile];
  const int component = blockIdx.x;
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  const int rows = component < 2 && !include_edge_rz
                       ? m_state.dimensions_[0] - 1
                       : m_state.dimensions_[4];
  const int count = rows * modes;
  double sum = 0.0;
  for (int start = 0; start < count; start += kResidualTile) {
    const int index = start + threadIdx.x;
    if (index < count) {
      const double even = m_state.forces_[2 * component][index];
      const double odd = m_state.forces_[2 * component + 1][index];
      terms[2 * threadIdx.x] = even * even;
      terms[2 * threadIdx.x + 1] = odd * odd;
    }
    if constexpr (kResidualTile == 32)
      __syncwarp();
    else
      __syncthreads();
    if (threadIdx.x == 0) {
      const int remaining = count - start;
      const int length = remaining < kResidualTile ? remaining : kResidualTile;
      for (int i = 0; i < 2 * length; ++i) sum += terms[i];
    }
    if constexpr (kResidualTile == 32)
      __syncwarp();
    else
      __syncthreads();
  }
  if (threadIdx.x == 0) m_state.residuals_[component] = sum;
}
}  // namespace

extern "C" void LaunchGpuIterationM1(GpuIterationDeviceState& m_state) {
  const int count = m_state.dimensions_[3] * (m_state.dimensions_[2] + 1);
  M1<<<(count + 127) / 128, 128, 0,
       static_cast<cudaStream_t>(m_state.stream_)>>>(DeviceView(m_state));
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuIterationRZ(GpuIterationDeviceState& m_state) {
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  ThomasFactor<<<dim3((modes + 31) / 32, 2), 32, 0,
                 static_cast<cudaStream_t>(m_state.stream_)>>>(
      DeviceView(m_state));
  Check(cudaGetLastError());
  ThomasRightHandSide<<<dim3((modes + kRhsModeTile - 1) / kRhsModeTile, 4), 32,
                        0, static_cast<cudaStream_t>(m_state.stream_)>>>(
      DeviceView(m_state));
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuIterationFactor(GpuIterationDeviceState& m_state) {
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  ThomasFactor<<<dim3((modes + 31) / 32, 2), 32, 0,
                 static_cast<cudaStream_t>(m_state.stream_)>>>(
      DeviceView(m_state));
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuIterationRZFactored(GpuIterationDeviceState& m_state) {
  const int modes = m_state.dimensions_[1] * (m_state.dimensions_[2] + 1);
  ThomasRightHandSide<<<dim3((modes + kRhsModeTile - 1) / kRhsModeTile, 4), 32,
                        0, static_cast<cudaStream_t>(m_state.stream_)>>>(
      DeviceView(m_state));
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuIterationLambda(GpuIterationDeviceState& m_state) {
  const int count = m_state.dimensions_[4] * m_state.dimensions_[1] *
                    (m_state.dimensions_[2] + 1);
  Lambda<<<(count + 127) / 128, 128, 0,
           static_cast<cudaStream_t>(m_state.stream_)>>>(DeviceView(m_state));
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuIterationResiduals(GpuIterationDeviceState& m_state,
                                            bool include_edge_rz) {
  const char* setting = std::getenv("VMECPP_GPU_RESIDUAL_ROUNDS");
  if (setting != nullptr) {
    char* end = nullptr;
    const long rounds = std::strtol(setting, &end, 10);
    if (*end != '\0' || rounds < 1 || rounds > 32) {
      throw std::runtime_error("Residual relaxation rounds must be in [1,32]");
    }
    const char* graph = std::getenv("VMECPP_GPU_RESIDUAL_GRAPH");
    LaunchGpuIterationResidualsOptimistic(
        m_state, include_edge_rz, static_cast<int>(rounds),
        graph != nullptr && std::strcmp(graph, "1") == 0);
    return;
  }
  LaunchGpuIterationResidualsConditional(m_state, include_edge_rz, nullptr);
}

extern "C" void LaunchGpuIterationResidualsConditional(
    GpuIterationDeviceState& m_state, bool include_edge_rz,
    const int* invalid_components) {
  Residuals<<<3, kResidualTile, 0,
              static_cast<cudaStream_t>(m_state.stream_)>>>(
      DeviceView(m_state), include_edge_rz, invalid_components);
  Check(cudaGetLastError());
}

extern "C" void RunGpuIteration(const GpuIterationInput& input,
                                GpuIterationOutput& m_output) {
  if (input.dimensions_[0] != input.dimensions_[4] ||
      input.dimensions_[3] != input.dimensions_[0] - 1) {
    throw std::runtime_error("Iteration replay requires fixed-boundary scope");
  }
  std::vector<void*> allocations;
  const auto allocate = [&](std::size_t count) {
    double* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  };
  GpuIterationDeviceState state{};
  state.dimensions_ = input.dimensions_;
  try {
    for (std::size_t i = 0; i < input.data_.size(); ++i) {
      double* pointer = allocate(input.sizes_[i]);
      Check(cudaMemcpy(pointer, input.data_[i],
                       input.sizes_[i] * sizeof(double),
                       cudaMemcpyHostToDevice));
      if (i < 6)
        state.forces_[i] = pointer;
      else
        state.constants_[i - 6] = pointer;
    }
    for (double*& pointer : state.factored_upper_) {
      pointer = allocate(input.sizes_[0]);
    }
    for (double*& pointer : state.denominators_) {
      pointer = allocate(input.sizes_[0]);
    }
    // Keep stage snapshots on device; downloads occur after the entire chain.
    std::array<double*, 20> snapshots{};
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i] = allocate(m_output.sizes_[i]);
    }
    const auto snapshot = [&](int stage) {
      for (int i = 0; i < 6; ++i) {
        const int output = stage * 6 + i;
        Check(cudaMemcpyAsync(snapshots[output], state.forces_[i],
                              m_output.sizes_[output] * sizeof(double),
                              cudaMemcpyDeviceToDevice));
      }
    };
    state.residuals_ = snapshots[18];
    LaunchGpuIterationResiduals(state, true);
    LaunchGpuIterationM1(state);
    snapshot(0);
    LaunchGpuIterationRZ(state);
    snapshot(1);
    LaunchGpuIterationLambda(state);
    snapshot(2);
    state.residuals_ = snapshots[19];
    LaunchGpuIterationResiduals(state, true);
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
      Check(cudaMemcpy(m_output.data_[i], snapshots[i],
                       m_output.sizes_[i] * sizeof(double),
                       cudaMemcpyDeviceToHost));
    }
    for (void* pointer : allocations) Check(cudaFree(pointer));
  } catch (...) {
    for (void* pointer : allocations) cudaFree(pointer);
    throw;
  }
}

extern "C" void LaunchGpuIterationStep(GpuStepDeviceState& m_state) {
  StepState state{};
  for (int i = 0; i < 6; ++i) {
    state.geometry_[i] = m_state.geometry_[i];
    state.velocity_[i] = m_state.velocity_[i];
    state.forces_[i] = m_state.forces_[i];
  }
  state.count_ = m_state.coefficient_count_;
  state.velocity_scale_ = m_state.scalars_[0];
  state.conjugation_ = m_state.scalars_[1];
  state.time_step_ = m_state.scalars_[2];
  Step<<<dim3((state.count_ + 127) / 128, 6), 128, 0,
         static_cast<cudaStream_t>(m_state.stream_)>>>(state);
  Check(cudaGetLastError());
}

extern "C" void RunGpuIterationStep(const GpuStepInput& input,
                                    GpuStepOutput& m_output) {
  std::vector<void*> allocations;
  GpuStepDeviceState state{};
  const int count =
      input.dimensions_[4] * input.dimensions_[1] * (input.dimensions_[2] + 1);
  state.coefficient_count_ = count;
  state.scalars_ = input.scalars_;
  try {
    for (int i = 0; i < 18; ++i) {
      if (input.sizes_[i] != static_cast<std::size_t>(count)) {
        throw std::runtime_error("Step replay requires one full partition");
      }
      double* pointer = nullptr;
      Check(cudaMalloc(&pointer, input.sizes_[i] * sizeof(double)));
      allocations.push_back(pointer);
      Check(cudaMemcpy(pointer, input.data_[i], count * sizeof(double),
                       cudaMemcpyHostToDevice));
      if (i < 6)
        state.geometry_[i] = pointer;
      else if (i < 12)
        state.velocity_[i - 6] = pointer;
      else
        state.forces_[i - 12] = pointer;
    }
    LaunchGpuIterationStep(state);
    for (int i = 0; i < 12; ++i) {
      if (m_output.sizes_[i] != static_cast<std::size_t>(count)) {
        throw std::runtime_error("Step output dimension mismatch");
      }
      Check(cudaMemcpy(m_output.data_[i], allocations[i],
                       count * sizeof(double), cudaMemcpyDeviceToHost));
    }
    for (void* pointer : allocations) Check(cudaFree(pointer));
  } catch (...) {
    for (void* pointer : allocations) cudaFree(pointer);
    throw;
  }
}

extern "C" void LaunchGpuIterationPreparation(
    GpuPreparationDeviceState& m_state) {
  PreparationState state{};
  for (int i = 0; i < 6; ++i) state.coefficients_[i] = m_state.coefficients_[i];
  state.scalxc_ = m_state.scalxc_;
  state.mpol_ = m_state.dimensions_[1];
  state.n_count_ = m_state.dimensions_[2] + 1;
  state.count_ = m_state.dimensions_[4] * state.mpol_ * state.n_count_;
  state.m1_rows_ = m_state.dimensions_[3];
  state.m1_scale_ = m_state.m1_scale_;
  state.zero_z_m1_ = m_state.zero_z_m1_;
  Prepare<<<(state.count_ + 127) / 128, 128, 0,
            static_cast<cudaStream_t>(m_state.stream_)>>>(state);
  Check(cudaGetLastError());
}

extern "C" void LaunchGpuGeometryAxis(GpuPreparationDeviceState& m_state) {
  if (m_state.dimensions_[0] < 2 || m_state.dimensions_[1] < 2) {
    throw std::runtime_error(
        "Axis extrapolation requires two surfaces and m=1");
  }
  PreparationState state{};
  for (int i = 0; i < 6; ++i) state.coefficients_[i] = m_state.coefficients_[i];
  state.mpol_ = m_state.dimensions_[1];
  state.n_count_ = m_state.dimensions_[2] + 1;
  GeometryAxis<<<(state.n_count_ + 31) / 32, 32, 0,
                 static_cast<cudaStream_t>(m_state.stream_)>>>(state);
  Check(cudaGetLastError());
}

namespace {
void RunGpuPreparationInternal(const GpuPreparationInput& input,
                               GpuPreparationOutput& m_output, bool geometry) {
  GpuPreparationDeviceState state{};
  state.dimensions_ = input.dimensions_;
  state.m1_scale_ = input.scalars_[0];
  state.zero_z_m1_ = input.scalars_[1] != 0.0;
  if (geometry) {
    state.dimensions_[3] = input.dimensions_[0];
    state.m1_scale_ = 1.0;
    state.zero_z_m1_ = false;
  }
  std::vector<void*> allocations;
  try {
    for (int i = 0; i < 7; ++i) {
      double* pointer = nullptr;
      Check(cudaMalloc(&pointer, input.sizes_[i] * sizeof(double)));
      allocations.push_back(pointer);
      Check(cudaMemcpy(pointer, input.data_[i],
                       input.sizes_[i] * sizeof(double),
                       cudaMemcpyHostToDevice));
      if (i < 6)
        state.coefficients_[i] = pointer;
      else
        state.scalxc_ = pointer;
    }
    LaunchGpuIterationPreparation(state);
    if (geometry) LaunchGpuGeometryAxis(state);
    for (int i = 0; i < 6; ++i) {
      if (m_output.sizes_[i] != input.sizes_[i])
        throw std::runtime_error("Preparation output shape mismatch");
      Check(cudaMemcpy(m_output.data_[i], state.coefficients_[i],
                       m_output.sizes_[i] * sizeof(double),
                       cudaMemcpyDeviceToHost));
    }
    for (void* pointer : allocations) Check(cudaFree(pointer));
  } catch (...) {
    for (void* pointer : allocations) cudaFree(pointer);
    throw;
  }
}
}  // namespace

extern "C" void RunGpuIterationPreparation(const GpuPreparationInput& input,
                                           GpuPreparationOutput& m_output) {
  RunGpuPreparationInternal(input, m_output, false);
}
extern "C" void RunGpuGeometryPreparation(const GpuPreparationInput& input,
                                          GpuPreparationOutput& m_output) {
  RunGpuPreparationInternal(input, m_output, true);
}
