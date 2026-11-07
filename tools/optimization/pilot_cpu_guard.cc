// SPDX-License-Identifier: MIT
// Read-only x86 floating-point state guard for the isolated cloud pilot.
#include <immintrin.h>
#include <nvtx3/nvToolsExt.h>

#include <cfenv>
#include <cstdint>

extern "C" std::uint32_t VmecPilotMxcsr() { return _mm_getcsr(); }

extern "C" int VmecPilotRoundingMode() { return std::fegetround(); }

extern "C" bool VmecPilotFloatingPointStateSupported() {
  return std::fegetround() == FE_TONEAREST && (_mm_getcsr() & 0xe040U) == 0;
}

extern "C" int VmecPilotRangePush() {
  return nvtxRangePushA("whole vmecpp.run");
}

extern "C" int VmecPilotRangePop() { return nvtxRangePop(); }
