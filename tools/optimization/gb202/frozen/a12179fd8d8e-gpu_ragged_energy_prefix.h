// SPDX-License-Identifier: MIT
#ifndef VMECPP_PRIVATE_GPU_RAGGED_ENERGY_PREFIX_H_
#define VMECPP_PRIVATE_GPU_RAGGED_ENERGY_PREFIX_H_
#include <array>
extern "C" __attribute__((visibility("protected"))) void*
CreateGpuRaggedEnergyPrefix(int thermal_count, int magnetic_count,
                            void* stream);
extern "C" __attribute__((visibility("protected"))) void
DestroyGpuRaggedEnergyPrefix(void* m_context);
extern "C" __attribute__((visibility("protected"))) void
ProposeGpuRaggedEnergyPrefix(void* m_context, const double* terms,
                             double* m_prefix, bool corrupt);
extern "C" __attribute__((visibility("protected"))) const int*
GpuRaggedEnergyPrefixInvalidDevice(void* context);
extern "C" __attribute__((visibility("protected"))) void
RebindGpuRaggedEnergyPrefix(void* m_context, void* stream);
extern "C" __attribute__((visibility("protected"))) void
ReadGpuRaggedEnergyPrefixStats(void* context,
                               std::array<unsigned long long, 8>& m_stats);
// All writers must already be retired; TLS-owned statistics storage stays live.
// This diagnostic read never uses the caller's possibly destroyed stream.
extern "C" __attribute__((visibility("hidden"))) void
ReadGpuRaggedEnergyPrefixRetiredStats(
    void* context, std::array<unsigned long long, 8>& m_stats);
#endif
