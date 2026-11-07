// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_PIPELINE_H_
#define VMECPP_EXPERIMENT_GPU_PIPELINE_H_

#include "gpu_forward.h"

// 24 MHD inputs (last is deltaS), seven constraint inputs, four lambda-force
// inputs, xmpq, and eight precomputed Fourier basis arrays. No physics knobs.
struct GpuPipelineInput {
  std::array<const double*, 44> data_;
  std::array<std::size_t, 44> sizes_;
  std::array<int, 13> dimensions_;
};

struct GpuMhdOutput {
  std::array<double*, 12> data_;
  std::array<std::size_t, 12> sizes_;
};

extern "C" void RunGpuPipeline(const GpuPipelineInput& input,
                               GpuForwardOutput& m_output);
extern "C" void RunGpuMhd(const GpuPipelineInput& input,
                          GpuMhdOutput& m_output);

#endif  // VMECPP_EXPERIMENT_GPU_PIPELINE_H_
