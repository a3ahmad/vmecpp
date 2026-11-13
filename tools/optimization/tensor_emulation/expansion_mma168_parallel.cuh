// SPDX-License-Identifier: MIT
// Experimental group-parallel product and ordered FP64 reconstruction.
// Reuses the frozen official-PTX-layout MMA168 helpers without changing them.
#ifndef VMECPP_EXPERIMENT_EXPANSION_MMA168_PARALLEL_CUH_
#define VMECPP_EXPERIMENT_EXPANSION_MMA168_PARALLEL_CUH_

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "expansion_mma168.cuh"

namespace vmecpp_experiment {
namespace expansion_mma168_parallel {

struct Group {
  int left_begin_;
  int pair_count_;
  int diagonal_;
  int start_;
  int stop_;
  int weight_;
  int bits_;
};

// Fixed balanced radix and range admission are the caller's responsibility,
// exactly as for grouped Mma168Kernel. Plan construction precedes capture.
// Every group obeys the absolute FP32 accumulator bound using ACTIVE K, not
// padded K. The enumeration order matches grouped Mma168Kernel exactly.
inline std::vector<Group> BuildGroups(int k, int components, int bits,
                                      int k_block, bool triangular,
                                      int requested_pair_cap = 0) {
  if (k < 1 || components < 1 || components > 8 || bits < 1 || bits > 12 ||
      k_block < 1 || requested_pair_cap < 0) {
    throw std::runtime_error("Invalid bounded MMA168 parallel configuration");
  }
  std::vector<Group> groups;
  const int diagonal_count = triangular ? components : 2 * components - 1;
  for (int diagonal = 0; diagonal < diagonal_count; ++diagonal) {
    const int first_left =
        diagonal < components ? 0 : diagonal - components + 1;
    const int last_left = diagonal < components ? diagonal : components - 1;
    for (int start = 0; start < k;) {
      const int length = k - start < k_block ? k - start : k_block;
      const int stop = start + length;
      int cap = expansion_mma168::SafePairCap(length, bits);
      if (requested_pair_cap > 0 && requested_pair_cap < cap) {
        cap = requested_pair_cap;
      }
      if (cap < 1) {
        throw std::runtime_error(
            "FP32 exact-group bound requires a smaller K block or radix");
      }
      for (int first = first_left; first <= last_left;) {
        const int available = last_left - first + 1;
        const int count = available < cap ? available : cap;
        groups.push_back({first, count, diagonal, start, stop,
                          -bits * (diagonal + 2), bits});
        if (groups.size() > 65535) {
          throw std::runtime_error(
              "MMA168 parallel groups exceed CUDA grid.z limit");
        }
        first += count;
      }
      start = stop;
    }
  }
  return groups;
}

inline int GroupCount(int k, int components, int bits, int k_block,
                      bool triangular, int requested_pair_cap = 0) {
  return int(
      BuildGroups(k, components, bits, k_block, triangular, requested_pair_cap)
          .size());
}

inline std::size_t ScratchElements(int m, int n, int groups) {
  if (m < 1 || n < 1 || groups < 1 || groups > 65535) {
    throw std::runtime_error("Invalid MMA168 parallel scratch shape");
  }
  const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
  if (std::size_t(m) > limit / std::size_t(n) ||
      std::size_t(m) * std::size_t(n) > limit / std::size_t(groups)) {
    throw std::runtime_error("MMA168 parallel scratch size overflow");
  }
  return std::size_t(m) * n * groups;
}

inline std::size_t ScratchBytes(int m, int n, int groups) {
  return ScratchElements(m, n, groups) * sizeof(float);
}

// Launch block32, grid(ceil(m/16),ceil(n/8),group_count). Groups are a device
// copy of BuildGroups output. Every valid scratch element has one writer;
// no initial memset, FP64 accumulation, shared memory, or barrier is required.
template <class T>
__global__ void ParallelKernel(const T* a, const T* b, const Group* groups,
                               float* m_partials, int m, int n, int k,
                               int group_count) {
  if (blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 ||
      int(blockIdx.z) >= group_count || m < 1 || n < 1 || k < 1) {
    asm volatile("trap;");
    return;
  }
  const Group group = groups[blockIdx.z];
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (group.bits_ > 9) {
      asm volatile("trap;");
      return;
    }
  }
  const int row_base = int(blockIdx.x) * 16;
  const int column_base = int(blockIdx.y) * 8;
  cuda::std::array<float, 4> accumulator{};
  for (int pair = 0; pair < group.pair_count_; ++pair) {
    const int left = group.left_begin_ + pair;
    const int right = group.diagonal_ - left;
    expansion_mma168::ComputePair(a, b, m, n, k, left, right, row_base,
                                  column_base, group.start_, group.stop_,
                                  accumulator);
  }
  const int lane_group = int(threadIdx.x) >> 2;
  const int thread_in_group = int(threadIdx.x) & 3;
  const auto group_offset = std::size_t(blockIdx.z) * m * n;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int row = row_base + lane_group + (index >> 1) * 8;
    const int column = column_base + thread_in_group * 2 + (index & 1);
    if (row < m && column < n) {
      m_partials[group_offset + row + std::size_t(column) * m] =
          accumulator[index];
    }
  }
}

// Launch on the SAME stream after ParallelKernel. A plain per-element scan
// preserves BuildGroups order; no cross-block reduction or atomic is used.
// Original per-output row/column exponent scaling remains generic scalbn.
__global__ void FinishKernel(const float* partials, const Group* groups,
                             const int* scale_a, const int* scale_b,
                             double* m_c, int m, int n, int group_count) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto elements = std::size_t(m) * n;
  if (index >= elements) return;
  double sum = 0.0;
  for (int group = 0; group < group_count; ++group) {
    // BuildGroups guarantees -192<=weight<=-2 for every supported component.
    const int weight = groups[group].weight_;
    const double factor =
        __longlong_as_double(static_cast<long long>(1023 + weight) << 52);
    const double value =
        double(partials[std::size_t(group) * elements + index]);
    sum = __dadd_rn(sum, __dmul_rn(value, factor));
  }
  m_c[index] = expansion_mma168::ScaleFinal(
      sum, scale_a[index % m] + scale_b[index / m]);
}

}  // namespace expansion_mma168_parallel
}  // namespace vmecpp_experiment
#endif  // VMECPP_EXPERIMENT_EXPANSION_MMA168_PARALLEL_CUH_
