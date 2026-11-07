// SPDX-License-Identifier: MIT
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_ENERGIES_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_ENERGIES_H_

#include <array>

#include "gpu_diagnostics.h"

// Experimental ordinary-energy specialization. Refresh diagnostics are
// separate. Warm dimensions/stream outside capture; graphs must die before a
// resize. Magnetic proposals reverse signs, then verify the ORIGINAL signed
// recurrence before acceptance.
extern "C" void LaunchGpuLatticeEnergies(GpuDiagnosticsDeviceState& m_state);
extern "C" void LaunchGpuLatticeEnergiesPoisoned(
    GpuDiagnosticsDeviceState& m_state);
// Thermal then magnetic cumulative workspace counters, four per context:
// calls, signed-certificate accepted components, original-fallback components,
// normal proposal segments. Workspace snapshots are not solver totals.
extern "C" void ReadGpuLatticeEnergiesStats(
    std::array<unsigned long long, 8>& m_statistics);

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_ENERGIES_H_
