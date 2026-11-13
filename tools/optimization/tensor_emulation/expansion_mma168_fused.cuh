// SPDX-License-Identifier: MIT
// Short-K A split + cached MMA. B digits/scales must already be resident.
#ifndef VMECPP_EXPERIMENT_EXPANSION_MMA168_FUSED_CUH_
#define VMECPP_EXPERIMENT_EXPANSION_MMA168_FUSED_CUH_

#include "expansion_mma168_cached.cuh"

namespace vmecpp_experiment::expansion_mma168_fused {

template <class T>
__device__ __forceinline__ unsigned short EncodeDigit(long long digit);

template <>
__device__ __forceinline__ unsigned short EncodeDigit<__half>(long long digit) {
  return __half_as_ushort(__float2half_rn(float(digit)));
}

template <>
__device__ __forceinline__ unsigned short EncodeDigit<__nv_bfloat16>(
    long long digit) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(float(digit)));
}

__device__ __forceinline__ double Normalize(double value, int exponent) {
  const int inverse = -exponent;
  if (inverse >= -1022 && inverse <= 1023) {
    const double factor =
        __longlong_as_double(static_cast<long long>(1023 + inverse) << 52);
    return __dmul_rn(value, factor);
  }
  return scalbn(value, inverse);
}

// Launch block32/grid(ceil(m/16),ceil(n/8)), only K<=16. A is ORIGINAL FP64
// column-major, not digit storage. B uses the frozen slice-major digit layout.
// m_range is the same nonzero-failure gate as SplitKernel, initialized outside
// capture. A is repeated across output column tiles, so counts are diagnostic
// visits, not unique input counts. Valid input issues no atomic operation.
template <class T, int Components>
__global__ void FusedKernel(const double* a, const T* b, const int* scale_b,
                            double* m_c, unsigned long long* m_range, int m,
                            int n, int k, int bits, bool triangular,
                            bool group_diagonal, int requested_pair_cap) {
  using namespace expansion_mma168;
  static_assert(Components == 4 || Components == 6 || Components == 8);
  if (blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 || m < 1 ||
      n < 1 || k < 1 || k > 16 || bits < 1 || bits > 12 || m_range == nullptr) {
    asm volatile("trap;");
    return;
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (bits > 9) {
      asm volatile("trap;");
      return;
    }
  }
  const int lane = int(threadIdx.x);
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row_base = int(blockIdx.x) * 16;
  const int column_base = int(blockIdx.y) * 8;
  cuda::std::array<double, 8> residual;
  double maximum0 = 0.0;
  double maximum1 = 0.0;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    const int row = row_base + group + (word & 1) * 8;
    const int inner = thread * 2 + (word >> 1) * 8;
    residual[2 * word] =
        row < m && inner < k ? a[row + std::size_t(inner) * m] : 0.0;
    residual[2 * word + 1] =
        row < m && inner + 1 < k ? a[row + std::size_t(inner + 1) * m] : 0.0;
    const double value =
        fmax(fabs(residual[2 * word]), fabs(residual[2 * word + 1]));
    if ((word & 1) == 0)
      maximum0 = fmax(maximum0, value);
    else
      maximum1 = fmax(maximum1, value);
  }
  // Four lanes jointly own each complete 16-element row; no CTA barrier.
  maximum0 = fmax(maximum0, __shfl_xor_sync(0xffffffffU, maximum0, 1, 4));
  maximum0 = fmax(maximum0, __shfl_xor_sync(0xffffffffU, maximum0, 2, 4));
  maximum1 = fmax(maximum1, __shfl_xor_sync(0xffffffffU, maximum1, 1, 4));
  maximum1 = fmax(maximum1, __shfl_xor_sync(0xffffffffU, maximum1, 2, 4));
  int exponent0 = 0;
  int exponent1 = 0;
  if (maximum0 != 0.0) {
    frexp(maximum0, &exponent0);
    ++exponent0;
  }
  if (maximum1 != 0.0) {
    frexp(maximum1, &exponent1);
    ++exponent1;
  }
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    const int exponent = (word & 1) == 0 ? exponent0 : exponent1;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int index = 2 * word + half;
      const double original = residual[index];
      residual[index] = Normalize(original, exponent);
      if (residual[index] == 0.0 && original != 0.0) atomicAdd(m_range, 1ULL);
    }
  }
  cuda::std::array<cuda::std::array<unsigned int, 4>, Components> cache_a;
  cuda::std::array<cuda::std::array<unsigned int, 2>, Components> cache_b;
  const double radix =
      __longlong_as_double(static_cast<long long>(1023 + bits) << 52);
  const auto digit_bound = 1LL << (bits - 1);
#pragma unroll
  for (int component = 0; component < Components; ++component) {
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const double low = __dmul_rn(residual[2 * word], radix);
      const double high = __dmul_rn(residual[2 * word + 1], radix);
      const auto low_digit = __double2ll_rn(low);
      const auto high_digit = __double2ll_rn(high);
      if (low_digit > digit_bound || low_digit < -digit_bound)
        atomicAdd(m_range, 1ULL);
      if (high_digit > digit_bound || high_digit < -digit_bound)
        atomicAdd(m_range, 1ULL);
      cache_a[component][word] = unsigned(EncodeDigit<T>(low_digit)) |
                                 (unsigned(EncodeDigit<T>(high_digit)) << 16);
      residual[2 * word] = __dsub_rn(low, double(low_digit));
      residual[2 * word + 1] = __dsub_rn(high, double(high_digit));
    }
#pragma unroll
    for (int word = 0; word < 2; ++word) {
      const int inner = thread * 2 + word * 8;
      const int column = column_base + group;
      const auto slice = std::size_t(component) * k * n;
      cache_b[component][word] =
          LoadPair(b, slice + inner + std::size_t(column) * k,
                   slice + inner + 1 + std::size_t(column) * k,
                   column < n && inner < k, column < n && inner + 1 < k);
    }
  }
  cuda::std::array<double, 4> sums{};
  if (!group_diagonal) {
#pragma unroll
    for (int left = 0; left < Components; ++left) {
#pragma unroll
      for (int right = 0; right < Components; ++right) {
        if (triangular && left + right >= Components) continue;
        cuda::std::array<float, 4> accumulator{};
        Mma<T>(cache_a[left], cache_b[right], accumulator);
        Reconstruct(accumulator, -bits * (left + right + 2), sums);
      }
    }
  } else {
    int cap = requested_pair_cap;
    if (cap > 0) {
      const auto maximum = std::uint64_t{1} << (bits - 1);
      const auto bound =
          std::uint64_t(k) * std::uint64_t(cap) * maximum * maximum;
      if (bound > (std::uint64_t{1} << 24)) {
        asm volatile("trap;");
        return;
      }
    } else {
      cap = SafePairCap(k, bits);
    }
    if (cap < 1) {
      asm volatile("trap;");
      return;
    }
    const int diagonals = triangular ? Components : 2 * Components - 1;
    for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
      cuda::std::array<float, 4> accumulator{};
      int count = 0;
#pragma unroll
      for (int left = 0; left < Components; ++left) {
        const int right = diagonal - left;
        if (right < 0 || right >= Components) continue;
        const auto b_registers =
            expansion_mma168_cached::Select<Components, 2>(cache_b, right);
        Mma<T>(cache_a[left], b_registers, accumulator);
        if (++count == cap) {
          Reconstruct(accumulator, -bits * (diagonal + 2), sums);
          accumulator = {};
          count = 0;
        }
      }
      if (count != 0) Reconstruct(accumulator, -bits * (diagonal + 2), sums);
    }
  }
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int row = row_base + group + (index >> 1) * 8;
    const int column = column_base + thread * 2 + (index & 1);
    if (row < m && column < n) {
      const int exponent = index < 2 ? exponent0 : exponent1;
      m_c[row + std::size_t(column) * m] =
          Normalize(sums[index], -exponent - scale_b[column]);
    }
  }
}

}  // namespace vmecpp_experiment::expansion_mma168_fused
#endif  // VMECPP_EXPERIMENT_EXPANSION_MMA168_FUSED_CUH_
