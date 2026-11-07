// SPDX-License-Identifier: MIT
// Final-sum-only positive lattice actions. No generic-prefix API changes.
#ifndef VMECPP_ENERGY_FINAL_SUM_PROTOTYPE_MAP_H_
#define VMECPP_ENERGY_FINAL_SUM_PROTOTYPE_MAP_H_

#include <cstdint>

#if defined(__CUDACC__)
#define VMECPP_ENERGY_HD __host__ __device__
#else
#define VMECPP_ENERGY_HD
#endif

namespace energy_final_map {
constexpr std::uint64_t kLower = std::uint64_t{1} << 52;
constexpr std::uint64_t kCap = std::uint64_t{1} << 53;
constexpr std::uint64_t kFractionMask = kLower - 1;

struct Map {
  std::uint64_t even_ = 0;
  std::uint64_t odd_ = 0;
};

struct SegmentedMap {
  Map map_;
  bool head_ = false;
};

VMECPP_ENERGY_HD inline std::uint64_t Cost(const Map& map, int parity) {
  return parity == 0 ? map.even_ : map.odd_;
}

VMECPP_ENERGY_HD inline std::uint64_t ComposeCost(std::uint64_t earlier,
                                                  const Map& later,
                                                  int parity) {
  // Failure is absorbing. Do not consult a meaningless post-failure parity.
  if (earlier >= kCap) return kCap;
  const auto later_cost = Cost(later, (parity + earlier) & 1);
  if (later_cost >= kCap || later_cost >= kCap - earlier) return kCap;
  return earlier + later_cost;
}

VMECPP_ENERGY_HD inline Map Compose(const Map& earlier, const Map& later) {
  return {ComposeCost(earlier.even_, later, 0),
          ComposeCost(earlier.odd_, later, 1)};
}

VMECPP_ENERGY_HD inline SegmentedMap ComposeSegmented(
    const SegmentedMap& earlier, const SegmentedMap& later) {
  return {later.head_ ? later.map_ : Compose(earlier.map_, later.map_),
          earlier.head_ || later.head_};
}

// Raw bits avoid an aliased floating-point lvalue in the host/shared helper.
// All shifts are guarded before evaluation, including arbitrarily large terms.
VMECPP_ENERGY_HD inline Map TermMap(std::uint64_t raw, int binade,
                                    bool& m_valid) {
  const auto magnitude = raw & ~(std::uint64_t{1} << 63);
  const int field = (raw >> 52) & 2047;
  if (binade < 1 || binade > 2046 || field == 2047 ||
      ((raw >> 63) != 0 && magnitude != 0)) {
    m_valid = false;
    return {};
  }
  if (magnitude == 0) return {};
  const auto mantissa = (raw & kFractionMask) + (field == 0 ? 0 : kLower);
  const int shift = (field == 0 ? 1 : field) - binade;
  if (shift >= 0) {
    if (shift >= 53 || mantissa >= (kCap >> shift)) return {kCap, kCap};
    const auto integer = mantissa << shift;
    return {integer, integer};
  }
  const int places = -shift;
  if (places > 53) return {};
  const auto integer = mantissa >> places;
  const auto remainder = mantissa & ((std::uint64_t{1} << places) - 1);
  const auto half = std::uint64_t{1} << (places - 1);
  if (remainder < half) return {integer, integer};
  if (remainder > half) return {integer + 1, integer + 1};
  return {integer + (integer & 1), integer + ((integer + 1) & 1)};
}

VMECPP_ENERGY_HD inline bool Apply(const Map& map, std::uint64_t& m_mantissa) {
  if (m_mantissa < kLower || m_mantissa >= kCap) return false;
  const auto cost = Cost(map, m_mantissa & 1);
  if (cost >= kCap || cost >= kCap - m_mantissa) return false;
  m_mantissa += cost;
  return true;
}
}  // namespace energy_final_map

#undef VMECPP_ENERGY_HD
#endif  // VMECPP_ENERGY_FINAL_SUM_PROTOTYPE_MAP_H_
