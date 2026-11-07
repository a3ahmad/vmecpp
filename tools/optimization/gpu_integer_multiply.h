// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_INTEGER_MULTIPLY_H_
#define VMECPP_EXPERIMENT_GPU_INTEGER_MULTIPLY_H_

#include <bit>
#include <cstdint>

#if defined(__CUDACC__)
#define VMECPP_INTEGER_MULTIPLY_HD __host__ __device__
#else
#define VMECPP_INTEGER_MULTIPLY_HD
#endif

// An exact normal-result fast-path experiment, not an approximation. The
// conservative exponent guard also declines a few rounds-to-minnormal cases.
VMECPP_INTEGER_MULTIPLY_HD inline bool TryIntegerMultiplyBits(
    std::uint64_t left, std::uint64_t right, std::uint64_t& m_result) {
  constexpr std::uint64_t kFractionMask = (1ULL << 52) - 1;
  constexpr std::uint64_t kHiddenBit = 1ULL << 52;
  const int left_exponent = (left >> 52) & 2047;
  const int right_exponent = (right >> 52) & 2047;
  if (left_exponent == 0 || left_exponent == 2047 || right_exponent == 0 ||
      right_exponent == 2047)
    return false;
  const std::uint64_t a = (left & kFractionMask) | kHiddenBit;
  const std::uint64_t b = (right & kFractionMask) | kHiddenBit;
  const std::uint64_t low = a * b;
#if defined(__CUDA_ARCH__)
  const std::uint64_t high = __umul64hi(a, b);
#else
  const std::uint64_t high =
      static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
#endif
  const int upper_binade = (high >> 41) & 1;
  int exponent = left_exponent + right_exponent - 1023 + upper_binade;
  if (exponent <= 0 || exponent >= 2047) return false;
  const int shift = 52 + upper_binade;
  std::uint64_t mantissa = (low >> shift) | (high << (64 - shift));
  const std::uint64_t remainder = low & ((1ULL << shift) - 1);
  const std::uint64_t half = 1ULL << (shift - 1);
  if (remainder > half || (remainder == half && (mantissa & 1))) ++mantissa;
  if (mantissa == (1ULL << 53)) {
    mantissa >>= 1;
    ++exponent;
    if (exponent == 2047) return false;
  }
  const std::uint64_t sign = (left ^ right) & (1ULL << 63);
  m_result = sign | (static_cast<std::uint64_t>(exponent) << 52) |
             (mantissa & kFractionMask);
  return true;
}

VMECPP_INTEGER_MULTIPLY_HD inline std::uint64_t IntegerMultiplyBits(
    double value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
  return std::bit_cast<std::uint64_t>(value);
#endif
}
VMECPP_INTEGER_MULTIPLY_HD inline double IntegerMultiplyValue(
    std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(static_cast<long long>(bits));
#else
  return std::bit_cast<double>(bits);
#endif
}
VMECPP_INTEGER_MULTIPLY_HD inline double IntegerMultiply(double left,
                                                         double right) {
  std::uint64_t result = 0;
  if (TryIntegerMultiplyBits(IntegerMultiplyBits(left),
                             IntegerMultiplyBits(right), result))
    return IntegerMultiplyValue(result);
#if defined(__CUDA_ARCH__)
  return __dmul_rn(left, right);
#else
  return left * right;
#endif
}
#undef VMECPP_INTEGER_MULTIPLY_HD
#endif  // VMECPP_EXPERIMENT_GPU_INTEGER_MULTIPLY_H_
