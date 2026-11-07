// SPDX-License-Identifier: MIT
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_THOMAS_PREFIX_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_THOMAS_PREFIX_H_

#include <array>

#include "gpu_iteration.h"

// Experimental opt-in solver component. Original force buffers
// stay immutable until final backward/scatter. Forces, factors, lower matrices
// and scratch must not alias. Inputs/factors stay fixed throughout one call.
// Create outside capture; context/stream outlive every graph referencing them.
// Optional VMECPP_THOMAS_FORWARD_PREFETCH/BACKWARD_PREFETCH select 1/2/4/8
// immutable rows loaded early per thread. Default 1 preserves the original
// encoded kernels; each selected RN recurrence and full certificate is
// retained.
extern "C" void* CreateGpuThomasPrefix(const GpuIterationDeviceState& state);
extern "C" void RefreshGpuThomasPrefix(void* m_context,
                                       const GpuIterationDeviceState& state);
extern "C" void LaunchGpuThomasPrefix(void* m_context,
                                      GpuIterationDeviceState& m_state);
// Test-only: flip one interior forward-prefix bit before complete verification.
extern "C" void LaunchGpuThomasPrefixPoisoned(void* m_context,
                                              GpuIterationDeviceState& m_state);
// Host-only, outside capture/concurrent use. LAST flag snapshot, not totals:
// columns, accepted columns, fallback columns, failed individual row checks.
extern "C" void ReadGpuThomasPrefixFlags(
    void* m_context, std::array<unsigned long long, 4>& m_statistics);
// Four host arrays of ns*mpol*(ntor+1) doubles. Inactive entries are +0.
extern "C" void ReadGpuThomasPrefixForward(
    void* m_context, const std::array<double*, 4>& m_outputs);
extern "C" void DestroyGpuThomasPrefix(void* m_context);

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_THOMAS_PREFIX_H_
