// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_ADAPTIVE_RESIDUAL_H_
#define VMECPP_EXPERIMENT_GPU_ADAPTIVE_RESIDUAL_H_

#include <array>

#include "gpu_iteration.h"

// Off-default experiment. Corrections run in certified groups of four.
// The first call must occur outside capture to allocate/warm the workspace.
// Active capture inserts a conditional directly into the caller's graph;
// otherwise a cached standalone graph is launched onto the supplied stream.
extern "C" void LaunchGpuAdaptiveResiduals(GpuIterationDeviceState& m_state,
                                           bool include_edge_rz,
                                           int max_rounds);

// Thread-local workspace counters: calls; fallback R/Z/lambda; termination
// histogram for 0,4,8,...,32 correction rounds. Resize resets these counters.
extern "C" void ReadGpuAdaptiveResidualStats(
    std::array<unsigned long long, 13>& m_statistics);

#endif  // VMECPP_EXPERIMENT_GPU_ADAPTIVE_RESIDUAL_H_
