// SPDX-License-Identifier: MIT
#ifndef VMECPP_RESIDUAL_COORDINATE_MAP_H_
#define VMECPP_RESIDUAL_COORDINATE_MAP_H_
#include <cstdint>

#if defined(__CUDACC__)
#define VMECPP_COORDINATE_HD __host__ __device__
#else
#define VMECPP_COORDINATE_HD
#endif

namespace residual_coordinate_map {
constexpr std::uint32_t kMask = 65535;
constexpr int kMaximumCount = 32767;
struct Map {
  std::uint32_t bits_ = 1;  // Base0, tag1/difference0 is identity.
};
static_assert(sizeof(Map) == 4);

VMECPP_COORDINATE_HD inline std::uint32_t Cost(const Map& map, int parity) {
  const auto base = map.bits_ >> 2;
  const auto tag = map.bits_ & 3U;
  return (parity == 0 ? base
                     : tag == 0 ? base - 1 : tag == 2 ? base + 1 : base) & kMask;
}

VMECPP_COORDINATE_HD inline Map Compose(const Map& earlier, const Map& later) {
  const auto base = earlier.bits_ >> 2;
  const auto earlier_tag = earlier.bits_ & 3U;
  const auto later_tag = later.bits_ & 3U;
  const auto result = (base + Cost(later, base & 1)) & kMask;
  // Adjacent earlier costs send both parities to the same later parity.
  const auto tag = earlier_tag != 1 ? earlier_tag
                   : (base & 1) != 0 ? 2U - later_tag : later_tag;
  return {(result << 2) | tag};
}

VMECPP_COORDINATE_HD inline Map Translate(
    std::uint64_t even_cost, std::uint64_t odd_cost,
    std::uint64_t previous_mantissa, std::uint64_t mantissa) {
  const bool swapped = (previous_mantissa & 1) != 0;
  const auto first = swapped ? odd_cost : even_cost;
  const auto second = swapped ? even_cost : odd_cost;
  const auto delta = static_cast<std::int64_t>(mantissa) -
                     static_cast<std::int64_t>(previous_mantissa);
  const auto base = static_cast<std::uint32_t>(
                        static_cast<std::int64_t>(first) - delta) & kMask;
  const auto tag = second > first ? 2U : second < first ? 0U : 1U;
  return {(base << 2) | tag};
}
}  // namespace residual_coordinate_map
#undef VMECPP_COORDINATE_HD
#endif
