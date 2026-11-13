// SPDX-License-Identifier: MIT
// Short-K register-cache variant; the original m16n8 backend is unchanged.
#ifndef VMECPP_EXPERIMENT_EXPANSION_MMA168_CACHED_CUH_
#define VMECPP_EXPERIMENT_EXPANSION_MMA168_CACHED_CUH_
#include "expansion_mma168.cuh"

namespace vmecpp_experiment::expansion_mma168_cached {
template <int Count, int Words>
__device__ __forceinline__ cuda::std::array<unsigned int, Words> Select(
    const cuda::std::array<cuda::std::array<unsigned int, Words>, Count>& cache,
    int selected) {
  cuda::std::array<unsigned int, Words> result{};
#pragma unroll
  for (int component = 0; component < Count; ++component) {
#pragma unroll
    for (int word = 0; word < Words; ++word)
      if (component == selected) result[word] = cache[component][word];
  }
  return result;
}

// Only full single chunks K<=16 are cached. Host dispatch keeps the original
// kernel for other shapes, explicitly reported as a non-cached execution.
template <class T, int Components>
__global__ void CachedKernel(const T* a, const T* b, const int* scale_a,
                             const int* scale_b, double* m_c, int m, int n,
                             int k, int bits, bool triangular,
                             bool group_diagonal, int requested_pair_cap) {
  using namespace expansion_mma168;
  static_assert(Components == 4 || Components == 6 || Components == 8);
  if (blockDim.x != 32 || k < 1 || k > 16 || bits < 1 || bits > 12) {
    asm volatile("trap;");
    return;
  }
  const int lane = int(threadIdx.x), group = lane >> 2, thread = lane & 3;
  const int row_base = int(blockIdx.x) * 16, column_base = int(blockIdx.y) * 8;
  cuda::std::array<cuda::std::array<unsigned int, 4>, Components> cache_a;
  cuda::std::array<cuda::std::array<unsigned int, 2>, Components> cache_b;
#pragma unroll
  for (int component = 0; component < Components; ++component) {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const int row = row_base + group + (index & 1) * 8;
      const int column = thread * 2 + (index >> 1) * 8;
      const auto slice = std::size_t(component) * m * k;
      cache_a[component][index] =
          LoadPair(a, slice + row + std::size_t(column) * m,
                   slice + row + std::size_t(column + 1) * m,
                   row < m && column < k, row < m && column + 1 < k);
    }
#pragma unroll
    for (int index = 0; index < 2; ++index) {
      const int row = thread * 2 + index * 8, column = column_base + group;
      const auto slice = std::size_t(component) * k * n;
      cache_b[component][index] =
          LoadPair(b, slice + row + std::size_t(column) * k,
                   slice + row + 1 + std::size_t(column) * k,
                   column < n && row < k, column < n && row + 1 < k);
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
      if (std::uint64_t(k) * cap * (std::uint64_t{1} << (2 * (bits - 1))) >
          (std::uint64_t{1} << 24)) {
        asm volatile("trap;");
        return;
      }
    } else
      cap = SafePairCap(k, bits);
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
        const auto b_registers = Select<Components, 2>(cache_b, right);
        Mma<T>(cache_a[left], b_registers, accumulator);
        if (++count == cap) {
          Reconstruct(accumulator, -bits * (diagonal + 2), sums);
          accumulator = {};
          count = 0;
        }
      }
      if (count) Reconstruct(accumulator, -bits * (diagonal + 2), sums);
    }
  }
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int row = row_base + group + (index >> 1) * 8;
    const int column = column_base + thread * 2 + (index & 1);
    if (row < m && column < n)
      m_c[row + std::size_t(column) * m] =
          ScaleFinal(sums[index], scale_a[row] + scale_b[column]);
  }
}
}  // namespace vmecpp_experiment::expansion_mma168_cached
#endif
