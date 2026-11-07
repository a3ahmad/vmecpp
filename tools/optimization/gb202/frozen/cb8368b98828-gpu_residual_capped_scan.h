// SPDX-License-Identifier: MIT
#ifndef VMECPP_RESIDUAL_CAPPED_SCAN_PER_SLOT_GPU_H_
#define VMECPP_RESIDUAL_CAPPED_SCAN_PER_SLOT_GPU_H_

#include <array>
#include <cstdint>

#include "gpu_iteration.h"

enum class ResidualCappedScanStatistic : int {
  kCalls,
  kAccepted,
  kDeclined,
  kVisitedSegments,
  kScannedMaps,  // Full Map elements scanned, not partial reservations.
  kTermDomainFailures,
  kGuessFailures,
  kSeedFailures,
  kRangeFailures,
  kMetadataFailures,
  kForcedDeclines,
  kOriginalConditionalFallbacks,
  kCount
};
constexpr int kResidualCappedScanStatistics =
    static_cast<int>(ResidualCappedScanStatistic::kCount);
using ResidualCappedScanStatistics =
    std::array<std::array<std::uint64_t, kResidualCappedScanStatistics>, 3>;

// NEW private interface. Context owns THREE components; generic prefix API and
// selected solver are unchanged. Create/warm outside capture on the caller's
// explicit stream. Never migrate thread/device/dimensions/stream. Launch
// returns only after ENQUEUEING the original same-slot live-force conditional
// fallback; the caller must retain the ordinary stream order before M1/force
// overwrite.
extern "C" void* CreateGpuResidualCappedScan(
    const GpuIterationDeviceState& state);
extern "C" void LaunchGpuResidualCappedScan(void* m_context,
                                            GpuIterationDeviceState& m_state,
                                            bool include_edge_rz,
                                            bool force_invalid);
// Caller must retire all graph/stream references before Destroy. Destruction
// rejects a pending/captured stream; no destruction-fault recovery is
// qualified.
extern "C" void DestroyGpuResidualCappedScan(void* m_context);
// Caller proves all writers retired. Synchronous copy does not use a
// potentially retired saved stream, matching the diagnosed energy readback
// lifetime rule.
extern "C" void ReadGpuResidualCappedScanStats(
    void* context, ResidualCappedScanStatistics& m_statistics);

#endif  // VMECPP_RESIDUAL_CAPPED_SCAN_PER_SLOT_GPU_H_
