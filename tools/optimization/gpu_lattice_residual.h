// SPDX-License-Identifier: MIT
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_RESIDUAL_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_RESIDUAL_H_

#include <array>

#include "gpu_iteration.h"

// Experimental opt-in solver path. Warm on the
// actual caller stream before capture. All invalid components use ORIGINAL
// LaunchGpuIterationResidualsConditional, never the naive prefix test kernel.
extern "C" void LaunchGpuIterationResidualsLattice(
    GpuIterationDeviceState& m_state, bool include_edge_rz);
extern "C" void LaunchGpuIterationResidualsLatticePoisoned(
    GpuIterationDeviceState& m_state, bool include_edge_rz);
extern "C" void ReadGpuLatticeResidualStats(
    std::array<unsigned long long, 4>& m_stats);
extern "C" void ReadGpuLatticeResidualFlags(std::array<int, 3>& m_flags);

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_RESIDUAL_H_
