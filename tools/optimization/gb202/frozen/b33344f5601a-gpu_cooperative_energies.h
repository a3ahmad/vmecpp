// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_COOPERATIVE_ENERGIES_H_
#define VMECPP_EXPERIMENT_GPU_COOPERATIVE_ENERGIES_H_

#include <array>

#include "gpu_diagnostics.h"

// Ordinary energy-only specialization. Refresh-volume/norm updates remain on
// the original path. Query performs no workspace allocation or kernel launch.
extern "C" bool GpuCooperativeEnergiesSupported(
    const GpuDiagnosticsDeviceState& state);
extern "C" void LaunchGpuCooperativeEnergies(GpuDiagnosticsDeviceState& m_state,
                                             int max_rounds);
// calls, fallback_thermal, fallback_magnetic, stopping histogram [0,32].
extern "C" void ReadGpuCooperativeEnergiesStats(
    std::array<unsigned long long, 36>& m_statistics);
// grid blocks, threads/block, resident blocks/SM, SMs, thermal/magnetic tiles.
extern "C" void ReadGpuCooperativeEnergiesLayout(std::array<int, 6>& m_layout);

#endif  // VMECPP_EXPERIMENT_GPU_COOPERATIVE_ENERGIES_H_
