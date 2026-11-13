// SPDX-License-Identifier: MIT
// Experimental register-only m16n8k16 expansion product; no solver integration.
// PTX layout: NVIDIA PTX ISA, Matrix Fragments for mma.m16n8k16, .f16/.bf16.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
#ifndef VMECPP_EXPERIMENT_EXPANSION_MMA168_CUH_
#define VMECPP_EXPERIMENT_EXPANSION_MMA168_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda/std/array>
#include <type_traits>

namespace vmecpp_experiment {
namespace expansion_mma168 {

__device__ __forceinline__ double ScaleFinal(double value, int exponent) {
  if (exponent >= -1022 && exponent <= 1023) {
    const double factor =
        __longlong_as_double(static_cast<long long>(1023 + exponent) << 52);
    return __dmul_rn(value, factor);
  }
  return scalbn(value, exponent);
}

template <class T>
__device__ __forceinline__ unsigned short DigitBits(T value);

template <>
__device__ __forceinline__ unsigned short DigitBits(__half value) {
  return __half_as_ushort(value);
}

template <>
__device__ __forceinline__ unsigned short DigitBits(__nv_bfloat16 value) {
  return __bfloat16_as_ushort(value);
}

template <class T>
__device__ __forceinline__ unsigned int LoadPair(const T* data,
                                                 std::size_t first,
                                                 std::size_t second,
                                                 bool first_valid,
                                                 bool second_valid) {
  const auto low = first_valid ? DigitBits(data[first]) : 0U;
  const auto high = second_valid ? DigitBits(data[second]) : 0U;
  return low | (high << 16);
}

template <class T>
__device__ __forceinline__ void Mma(const cuda::std::array<unsigned int, 4>& a,
                                    const cuda::std::array<unsigned int, 2>& b,
                                    cuda::std::array<float, 4>& m_c) {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>);
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(m_c[0]), "+f"(m_c[1]), "+f"(m_c[2]), "+f"(m_c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(m_c[0]), "+f"(m_c[1]), "+f"(m_c[2]), "+f"(m_c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

template <class T>
__device__ __forceinline__ void ComputePair(
    const T* a, const T* b, int m, int n, int k, int left, int right,
    int row_base, int column_base, int start, int stop,
    cuda::std::array<float, 4>& m_accumulator) {
  const int lane = int(threadIdx.x);
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const std::size_t a_slice = std::size_t(left) * m * k;
  const std::size_t b_slice = std::size_t(right) * k * n;
  for (int inner = start; inner < stop; inner += 16) {
    cuda::std::array<unsigned int, 4> a_registers;
    cuda::std::array<unsigned int, 2> b_registers;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const int row = row_base + group + (index & 1) * 8;
      const int column = inner + thread_in_group * 2 + (index >> 1) * 8;
      a_registers[index] =
          LoadPair(a, a_slice + row + std::size_t(column) * m,
                   a_slice + row + std::size_t(column + 1) * m,
                   row < m && column < stop, row < m && column + 1 < stop);
    }
#pragma unroll
    for (int index = 0; index < 2; ++index) {
      const int row = inner + thread_in_group * 2 + index * 8;
      const int column = column_base + group;
      b_registers[index] =
          LoadPair(b, b_slice + row + std::size_t(column) * k,
                   b_slice + row + 1 + std::size_t(column) * k,
                   column < n && row < stop, column < n && row + 1 < stop);
    }
    Mma<T>(a_registers, b_registers, m_accumulator);
  }
}

__host__ __device__ inline int SafePairCap(int inner_count, int bits) {
  if (inner_count < 1 || bits < 1 || bits > 12) return 0;
  const auto maximum = std::uint64_t{1} << (bits - 1);
  const auto cost = std::uint64_t(inner_count) * maximum * maximum;
  return int((std::uint64_t{1} << 24) / cost);
}

__device__ __forceinline__ void Reconstruct(
    const cuda::std::array<float, 4>& accumulator, int weight,
    cuda::std::array<double, 4>& m_sums) {
  // Kernel guards imply -192<=weight<=-2: this is a normal exact power of two.
  const double factor =
      __longlong_as_double(static_cast<long long>(1023 + weight) << 52);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    m_sums[index] =
        __dadd_rn(m_sums[index], __dmul_rn(double(accumulator[index]), factor));
  }
}

// Caller must launch block32, grid(ceil(m/16), ceil(n/8)). The input digit
// layouts/scales exactly match WmmaKernel. Digits must obey the balanced
// bound |d|<=2^(bits-1) and fixed weights exponent-bits*(slice+1).
// group_diagonal=false retains left/right/K-chunk reconstruction order.
// group_diagonal=true uses increasing diagonal/chunk/group order, a separately
// reassociated accuracy configuration. It never mixes unequal weights.
template <class T>
__global__ void Mma168Kernel(const T* a, const T* b, const int* scale_a,
                             const int* scale_b, double* m_c, int m, int n,
                             int k, int components, int bits, int k_block,
                             bool triangular, bool group_diagonal = false,
                             int requested_pair_cap = 0) {
  if (blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 || m < 1 ||
      n < 1 || k < 1 || components < 1 || components > 8 || bits < 1 ||
      bits > 12 || k_block < 1) {
    asm volatile("trap;");
    return;
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (bits > 9) {
      asm volatile("trap;");
      return;
    }
  }
  const int row_base = int(blockIdx.x) * 16;
  const int column_base = int(blockIdx.y) * 8;
  cuda::std::array<double, 4> sums{};
  if (!group_diagonal) {
    for (int left = 0; left < components; ++left) {
      for (int right = 0; right < components; ++right) {
        if (triangular && left + right >= components) continue;
        for (int start = 0; start < k; start += k_block) {
          const int stop = min(k, start + k_block);
          cuda::std::array<float, 4> accumulator{};
          ComputePair(a, b, m, n, k, left, right, row_base, column_base, start,
                      stop, accumulator);
          Reconstruct(accumulator, -bits * (left + right + 2), sums);
        }
      }
    }
  } else {
    const int diagonals = triangular ? components : 2 * components - 1;
    for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
      for (int start = 0; start < k; start += k_block) {
        const int stop = min(k, start + k_block);
        int cap = requested_pair_cap;
        if (cap > 0) {
          if (cap > components ||
              std::uint64_t(stop - start) * cap *
                      (std::uint64_t{1} << (2 * (bits - 1))) >
                  (std::uint64_t{1} << 24)) {
            asm volatile("trap;");
            return;
          }
        } else
          cap = SafePairCap(stop - start, bits);
        if (cap < 1) {
          asm volatile("trap;");
          return;
        }
        cuda::std::array<float, 4> accumulator{};
        int count = 0;
        for (int left = 0; left < components; ++left) {
          const int right = diagonal - left;
          if (right < 0 || right >= components) continue;
          ComputePair(a, b, m, n, k, left, right, row_base, column_base, start,
                      stop, accumulator);
          if (++count == cap) {
            Reconstruct(accumulator, -bits * (diagonal + 2), sums);
            accumulator = {};
            count = 0;
          }
        }
        if (count != 0) Reconstruct(accumulator, -bits * (diagonal + 2), sums);
      }
    }
  }
  const int group = int(threadIdx.x) >> 2;
  const int thread_in_group = int(threadIdx.x) & 3;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int row = row_base + group + (index >> 1) * 8;
    const int column = column_base + thread_in_group * 2 + (index & 1);
    if (row < m && column < n) {
      m_c[row + std::size_t(column) * m] =
          ScaleFinal(sums[index], scale_a[row] + scale_b[column]);
    }
  }
}

}  // namespace expansion_mma168
}  // namespace vmecpp_experiment
#endif  // VMECPP_EXPERIMENT_EXPANSION_MMA168_CUH_
