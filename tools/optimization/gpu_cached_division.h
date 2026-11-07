// SPDX-License-Identifier: MIT
// Isolated hypothesis: exponent-scaled exact certificate bounds, not a solver
// integration. Outside the proved normal-bound scope use original RN division.
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_CACHED_DIVISION_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_CACHED_DIVISION_H_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace vmecpp_cached_division_experiment {
struct DivisionResult {
  double value_;
  bool certified_;
};

__device__ inline bool IsNormal(double value) {
  return isfinite(value) && fabs(value) >= 0x1p-1022;
}

// reciprocal must be RN(1 / denominator), using the original signed operand.
// A successful certificate proves the ORIGINAL signed quotient's rounding cell.
__device__ inline DivisionResult Divide(double numerator, double denominator,
                                        double reciprocal) {
  if (!IsNormal(numerator) || !IsNormal(denominator) || !IsNormal(reciprocal))
    return {numerator / denominator, false};
  const double positive_denominator = fabs(denominator);
  const double normalized_numerator =
      denominator < 0.0 ? -numerator : numerator;
  const double positive_reciprocal = fabs(reciprocal);
  const double estimate = __dmul_rn(normalized_numerator, positive_reciprocal);
  if (!IsNormal(estimate)) return {numerator / denominator, false};
  const double residual =
      __fma_rn(-estimate, positive_denominator, normalized_numerator);
  const double candidate = __fma_rn(residual, positive_reciprocal, estimate);
  constexpr std::uint64_t kFractionMask = (std::uint64_t{1} << 52) - 1;
  constexpr std::uint64_t kExponentMask = 0x7ff;
  const auto candidate_bits =
      static_cast<std::uint64_t>(__double_as_longlong(candidate));
  const int candidate_exponent = (candidate_bits >> 52) & kExponentMask;
  if (candidate_exponent < 3 || candidate_exponent > 2045)
    return {numerator / denominator, false};
  const auto denominator_bits =
      static_cast<std::uint64_t>(__double_as_longlong(positive_denominator));
  const int denominator_exponent = (denominator_bits >> 52) & kExponentMask;
  const int bound_exponent =
      denominator_exponent + candidate_exponent - 1023 - 53;
  const bool at_power_of_two = (candidate_bits & kFractionMask) == 0;
  const bool negative_candidate = (candidate_bits >> 63) != 0;
  const int lower_exponent =
      bound_exponent - (at_power_of_two && !negative_candidate ? 1 : 0);
  const int upper_exponent =
      bound_exponent - (at_power_of_two && negative_candidate ? 1 : 0);
  if (lower_exponent < 1 || lower_exponent > 2046 || upper_exponent < 1 ||
      upper_exponent > 2046)
    return {numerator / denominator, false};
  // The half-neighbor gaps are exact powers of two for candidate exponents
  // [3,2045]. Scaling a normal denominator by them is exact when the resulting
  // encoded exponent is [1,2046]; its fraction bits remain unchanged. At a
  // signed power-of-two candidate only the gap toward zero is halved.
  const auto fraction = denominator_bits & kFractionMask;
  const double lower_bound = __longlong_as_double(static_cast<long long>(
      (static_cast<std::uint64_t>(lower_exponent) << 52) | fraction));
  const double upper_bound = __longlong_as_double(static_cast<long long>(
      (static_cast<std::uint64_t>(upper_exponent) << 52) | fraction));
  const double residual_lower =
      __fma_rd(-candidate, positive_denominator, normalized_numerator);
  const double residual_upper =
      __fma_ru(-candidate, positive_denominator, normalized_numerator);
  // Strict inequalities reject midpoint ties. Directed FMAs enclose the exact
  // real residual, proving the candidate is the original RN quotient.
  if (residual_lower > -lower_bound && residual_upper < upper_bound)
    return {candidate, true};
  return {numerator / denominator, false};
}
}  // namespace vmecpp_cached_division_experiment

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_CACHED_DIVISION_H_
