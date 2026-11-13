// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_FORWARD_H_
#define VMECPP_EXPERIMENT_GPU_FORWARD_H_

#include <array>
#include <cstddef>

struct GpuForwardInput {
  std::array<const double*, 29> data_;
  std::array<std::size_t, 29> sizes_;
  std::array<int, 13> dimensions_;
};

struct GpuForwardOutput {
  std::array<double*, 6> data_;
  std::array<std::size_t, 6> sizes_;
};

struct GpuForwardDeviceState {
  std::array<const double*, 29> input_;
  std::array<double*, 6> output_;
  std::array<std::size_t, 6> output_sizes_;
  std::array<int, 13> dimensions_;
  double* partials_;
  void* stream_;
};

// Device-only projection: no allocation, upload, download, or synchronization.
extern "C" void LaunchGpuForwardDevice(GpuForwardDeviceState& m_state);

extern "C" void RunGpuForward(const GpuForwardInput& input,
                              GpuForwardOutput& m_output);

#endif  // VMECPP_EXPERIMENT_GPU_FORWARD_H_
