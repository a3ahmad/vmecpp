// Experimental strict-FP64 transform interface; no public Python API changes.
#ifndef VMECPP_EXPERIMENT_GPU_FOURIER_H_
#define VMECPP_EXPERIMENT_GPU_FOURIER_H_

#include <array>
#include <cstddef>

struct GpuFourierInput {
  std::array<const double*, 16> data_;
  std::array<std::size_t, 16> sizes_;
  std::array<int, 13> dimensions_;
};

struct GpuFourierOutput {
  std::array<double*, 18> data_;
  std::array<std::size_t, 18> sizes_;
};

struct GpuDeviceFourierOutput {
  std::array<double*, 18> data_;
  std::array<std::size_t, 18> sizes_;
  std::array<int, 13> dimensions_;
  void* stream_;
};

// All pointers address device memory. Output channels must be contiguous in
// the listed order. Separable scratch has 12*(nsMaxF1-nsMinF1)*mpol*nZeta
// doubles.
struct GpuFourierDeviceState {
  std::array<const double*, 16> input_;
  std::array<double*, 18> output_;
  std::array<std::size_t, 18> output_sizes_;
  std::array<int, 13> dimensions_;
  double* partials_;
  void* stream_;
};
extern "C" void LaunchGpuFourierDevice(GpuFourierDeviceState& m_state);

extern "C" void RunGpuFourier(const GpuFourierInput& input,
                              GpuFourierOutput& m_output);

// Enqueues the transform without copying geometry back or synchronizing.
// Device arrays remain owned by the per-thread transform workspace. Complete
// downstream work on this stream before the next call reuses its pinned input.
extern "C" void RunGpuFourierResident(const GpuFourierInput& input,
                                      GpuFourierOutput& m_shape,
                                      GpuDeviceFourierOutput& m_output);

#endif  // VMECPP_EXPERIMENT_GPU_FOURIER_H_
