// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_COOPERATIVE_RESIDUAL_H_
#define VMECPP_EXPERIMENT_GPU_COOPERATIVE_RESIDUAL_H_

#include <array>

#include "gpu_iteration.h"

// Capability query only: no allocation/kernel launch. Invalid shapes or
// insufficient cooperative residency return false; CUDA API errors propagate.
extern "C" bool GpuCooperativeResidualsSupported(
    const GpuIterationDeviceState& state);

// Off-default cooperative-grid experiment. Warm outside graph capture first.
// Unsupported cooperative residency/dimensions throw before launching work.
extern "C" void LaunchGpuCooperativeResiduals(GpuIterationDeviceState& m_state,
                                              bool include_edge_rz,
                                              int max_rounds);

// Workspace counts: calls, fallback R/Z/lambda, round histogram [0,32].
extern "C" void ReadGpuCooperativeResidualStats(
    std::array<unsigned long long, 37>& m_statistics);

// grid blocks, block threads, resident blocks/SM, SM count, tiles/component.
extern "C" void ReadGpuCooperativeResidualLayout(std::array<int, 5>& m_layout);

#endif  // VMECPP_EXPERIMENT_GPU_COOPERATIVE_RESIDUAL_H_
