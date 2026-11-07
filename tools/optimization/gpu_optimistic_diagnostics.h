// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_OPTIMISTIC_DIAGNOSTICS_H_
#define VMECPP_EXPERIMENT_GPU_OPTIMISTIC_DIAGNOSTICS_H_

#include <array>

#include "gpu_diagnostics.h"

// Experimental parallel guesses with an exact certificate for every prefix.
// Failed components use the original ordered accumulation before finishing.
extern "C" void LaunchGpuDiagnosticsOptimistic(
    GpuDiagnosticsDeviceState& m_state, bool refresh_norms, int rounds,
    bool use_graph);
// Calls, then fallback counts for original ordered sum components 0..5.
extern "C" void ReadGpuDiagnosticsRelaxationStats(
    std::array<unsigned long long, 7>& m_statistics);

#endif  // VMECPP_EXPERIMENT_GPU_OPTIMISTIC_DIAGNOSTICS_H_
