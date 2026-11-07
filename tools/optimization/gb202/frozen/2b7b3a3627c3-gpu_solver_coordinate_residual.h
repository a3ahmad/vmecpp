// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <cstdint>
#include "gpu_residual_capped_scan.h"
// created, destroyed, coordinate calls, large-N generic calls, retire errors,
// outstanding contexts. Read only after all Solver contexts have retired.
extern "C" void GpuSolverCoordinateResidualThreadActivity(
    std::array<std::uint64_t, 6>& m_counts);
extern "C" void ReadGpuSolverCoordinateResidualStats(
    ResidualCappedScanStatistics& m_statistics);
