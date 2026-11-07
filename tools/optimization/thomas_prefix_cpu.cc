// SPDX-License-Identifier: MIT
// CPU oracle operators; compile with -ffp-contract=off and no fast-math.
#include <cfenv>
#include <cmath>

extern "C" double CpuReferenceDivide(double numerator, double denominator) {
  return numerator / denominator;
}
extern "C" double CpuReferenceFma(double first, double second, double third) {
  return std::fma(first, second, third);
}
extern "C" double CpuReferenceNumerator(double rhs, double previous,
                                        double lower) {
  const double product = previous * lower;
  return rhs - product;
}
extern "C" double CpuReferenceMultiply(double first, double second) {
  return first * second;
}
extern "C" double CpuReferenceAdd(double first, double second) {
  return first + second;
}
extern "C" int CpuReferenceRoundingMode() { return std::fegetround(); }
