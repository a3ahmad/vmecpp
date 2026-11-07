// SPDX-License-Identifier: MIT
// Isolated experiment: exact-rounding certificate, not used by the solver.
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_CERTIFIED_DIVISION_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_CERTIFIED_DIVISION_H_

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>

namespace vmecpp_division_experiment {
struct DivisionResult {
  double value_;
  bool certified_;
};

__device__ inline bool IsNormal(double value) {
  return isfinite(value) && fabs(value) >= 0x1p-1022;
}

// reciprocal must be RN(1 / denominator), with the original signed denominator.
// The fallback deliberately uses the original operation and original operands.
__device__ inline DivisionResult Divide(double numerator, double denominator,
                                        double reciprocal) {
  if (!IsNormal(numerator) || !IsNormal(denominator) || !IsNormal(reciprocal)) {
    return {numerator / denominator, false};
  }
  const double positive_denominator = fabs(denominator);
  const double normalized_numerator =
      denominator < 0.0 ? -numerator : numerator;
  const double positive_reciprocal = fabs(reciprocal);
  const double estimate = __dmul_rn(normalized_numerator, positive_reciprocal);
  if (!IsNormal(estimate)) return {numerator / denominator, false};
  const double residual =
      __fma_rn(-estimate, positive_denominator, normalized_numerator);
  const double candidate = __fma_rn(residual, positive_reciprocal, estimate);
  if (!IsNormal(candidate)) return {numerator / denominator, false};
  const double predecessor = nextafter(candidate, -CUDART_INF);
  const double successor = nextafter(candidate, CUDART_INF);
  if (!IsNormal(predecessor) || !IsNormal(successor))
    return {numerator / denominator, false};
  // Adjacent finite binary64 subtraction is exact here (Sterbenz's lemma).
  const double lower_half_gap = __dmul_rn(candidate - predecessor, 0.5);
  const double upper_half_gap = __dmul_rn(successor - candidate, 0.5);
  const double lower_bound = __dmul_rd(lower_half_gap, positive_denominator);
  const double upper_bound = __dmul_rd(upper_half_gap, positive_denominator);
  if (!(lower_half_gap > 0.0 && upper_half_gap > 0.0 && lower_bound > 0.0 &&
        upper_bound > 0.0 && isfinite(lower_bound) && isfinite(upper_bound))) {
    return {numerator / denominator, false};
  }
  const double residual_lower =
      __fma_rd(-candidate, positive_denominator, normalized_numerator);
  const double residual_upper =
      __fma_ru(-candidate, positive_denominator, normalized_numerator);
  if (residual_lower > -lower_bound && residual_upper < upper_bound)
    return {candidate, true};
  return {numerator / denominator, false};
}
}  // namespace vmecpp_division_experiment

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_CERTIFIED_DIVISION_H_
