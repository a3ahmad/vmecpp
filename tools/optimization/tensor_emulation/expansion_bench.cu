// SPDX-License-Identifier: MIT
// Independent accuracy experiment, not a VMEC arithmetic replacement.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <memory>

#include "expansion_common.h"
#include "expansion_mma168.cuh"
#include "expansion_mma168_cached.cuh"
#include "expansion_mma168_fused.cuh"
#include "expansion_mma168_parallel.cuh"

namespace {
const std::array<std::string, 8> kModes{"native", "ordered", "fp16-4",
                                        "fp16-6", "fp16-8",  "bf16-4",
                                        "bf16-6", "bf16-8"};
Options Parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string name(argv[index]);
    if (name == "--execute")
      options.execute_ = true;
    else if (name == "--reuse-b")
      options.reuse_b_ = true;
    else if (name == "--direct")
      options.direct_ = true;
    else if (name == "--profile-one")
      options.profile_one_ = true;
    else if (name == "--group-diagonals")
      options.group_diagonals_ = true;
    else {
      if (++index == argc) throw std::runtime_error("Missing option value");
      const std::string value(argv[index]);
      if (name == "--input")
        options.input_ = value;
      else if (name == "--report")
        options.report_ = value;
      else if (name == "--output")
        options.output_ = value;
      else if (name == "--expect-cublas")
        options.expect_cublas_ = value;
      else if (name == "--mode")
        options.mode_ = value;
      else if (name == "--backend")
        options.backend_ = value;
      else if (name == "--split")
        options.split_ = value;
      else if (name == "--terms")
        options.terms_ = value;
      else if (name == "--calls")
        options.calls_ = std::stoi(value);
      else if (name == "--samples")
        options.samples_ = std::stoi(value);
      else if (name == "--warmups")
        options.warmups_ = std::stoi(value);
      else if (name == "--k-block")
        options.k_block_ = std::stoi(value);
      else if (name == "--digits")
        options.digits_ = std::stoi(value);
      else if (name == "--graph-batch")
        options.graph_batch_ = std::stoi(value);
      else if (name == "--m")
        options.m_ = std::stoi(value);
      else if (name == "--n")
        options.n_ = std::stoi(value);
      else if (name == "--k")
        options.k_ = std::stoi(value);
      else
        throw std::runtime_error("Unknown option " + name);
    }
  }
  if (options.mode_ != "all" && options.mode_ != "fp16-all" &&
      options.mode_ != "bf16-all" &&
      std::find(kModes.begin(), kModes.end(), options.mode_) == kModes.end())
    throw std::runtime_error("Unknown mode");
  if ((options.backend_ != "wmma" && options.backend_ != "batched" &&
       options.backend_ != "mma168" && options.backend_ != "mma168-cache" &&
       options.backend_ != "mma168-parallel" &&
       options.backend_ != "mma168-fused") ||
      (options.split_ != "row" && options.split_ != "warp" &&
       options.split_ != "quantized") ||
      (options.terms_ != "full" && options.terms_ != "triangular") ||
      options.calls_ < 1 || options.calls_ > 10000 || options.samples_ < 1 ||
      options.samples_ > 101 || options.samples_ % 2 == 0 ||
      options.warmups_ < 0 || options.warmups_ > 100 || options.k_block_ < 0 ||
      options.digits_ < 0 || options.digits_ > 12 || options.graph_batch_ < 1 ||
      options.graph_batch_ > 128)
    throw std::runtime_error("Invalid bounded configuration");
  if (options.execute_ &&
      (options.report_.empty() || options.expect_cublas_.empty()))
    throw std::runtime_error("Execution requires report and cuBLAS owner");
  return options;
}
__device__ __forceinline__ long long RoundShiftEven(long long value,
                                                    int shift) {
  if (shift == 0) return value;
  const auto raw = static_cast<unsigned long long>(value);
  const auto magnitude = value < 0 ? 0ULL - raw : raw;
  auto quotient = magnitude >> shift;
  const auto remainder = magnitude & ((1ULL << shift) - 1);
  const auto half = 1ULL << (shift - 1);
  quotient += remainder > half || (remainder == half && (quotient & 1));
  return value < 0 ? -static_cast<long long>(quotient)
                   : static_cast<long long>(quotient);
}
template <class T>
__device__ __forceinline__ T IntDigit(int value) {
  if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(float(value));
  else
    return __float2bfloat16_rn(float(value));
}
template <class T>
__global__ void QuantizedSplitKernel(const double* input, T* m_digits,
                                     int* m_scales, int groups, int inner,
                                     bool matrix_a, int components, int bits,
                                     unsigned long long* m_range) {
  const int lane = int(threadIdx.x) & 31;
  const int group = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
  if (group >= groups) return;
  double maximum = 0;
  for (int index = lane; index < inner; index += 32) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    maximum = fmax(maximum, fabs(input[address]));
  }
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  int exponent = 0;
  if (lane == 0) {
    if (maximum != 0) {
      frexp(maximum, &exponent);
      ++exponent;
    }
    m_scales[group] = exponent;
  }
  exponent = __shfl_sync(0xffffffff, exponent, 0);
  const int total_bits = components * bits;
  if (total_bits > 64) {
    asm volatile("trap;");
    return;
  }
  const auto elements = std::size_t(groups) * inner;
  for (int index = lane; index < inner; index += 32) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    const double scaled = scalbn(input[address], total_bits - exponent);
    long long remainder = 0;
    if (!isfinite(scaled) || scaled >= 0x1p63 || scaled <= -0x1p63)
      atomicAdd(m_range, 1ULL);
    else
      remainder = __double2ll_rn(scaled);
    for (int component = 0; component < components; ++component) {
      const int shift = bits * (components - component - 1);
      const auto digit = RoundShiftEven(remainder, shift);
      if (digit > (1LL << (bits - 1)) || digit < -(1LL << (bits - 1)))
        atomicAdd(m_range, 1ULL);
      m_digits[std::size_t(component) * elements + address] =
          IntDigit<T>(int(digit));
      const auto residue = static_cast<unsigned long long>(remainder) -
                           (static_cast<unsigned long long>(digit) << shift);
      remainder = cuda::std::bit_cast<long long>(residue);
    }
  }
}
__device__ double NormalWeight(int exponent) {
  return __longlong_as_double(static_cast<unsigned long long>(exponent + 1023)
                              << 52);
}
int CeilLog2(int value) {
  int result = 0;
  for (int power = 1; power < value; power *= 2) ++result;
  return result;
}
template <class T>
__device__ T IntegerDigit(double value);
template <>
__device__ __half IntegerDigit(double value) {
  return __float2half_rn(float(value));
}
template <>
__device__ __nv_bfloat16 IntegerDigit(double value) {
  return __float2bfloat16_rn(float(value));
}
template <class T>
__global__ void SplitKernel(const double* input, T* m_digits, int* m_scales,
                            int groups, int inner, bool matrix_a,
                            int components, int bits,
                            unsigned long long* m_range) {
  const int group = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (group >= groups) return;
  double maximum = 0;
  for (int index = 0; index < inner; ++index) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    maximum = fmax(maximum, fabs(input[address]));
  }
  int exponent = 0;
  if (maximum != 0) {
    frexp(maximum, &exponent);
    ++exponent;
  }
  m_scales[group] = exponent;
  const auto elements = std::size_t(groups) * inner;
  for (int index = 0; index < inner; ++index) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    double residual = scalbn(input[address], -exponent);
    if (residual == 0 && input[address] != 0) atomicAdd(m_range, 1ULL);
    for (int component = 0; component < components; ++component) {
      const double scaled = __dmul_rn(residual, NormalWeight(bits));
      const auto digit = __double2int_rn(scaled);
      if (digit > (1 << (bits - 1)) || digit < -(1 << (bits - 1)))
        atomicAdd(m_range, 1ULL);
      m_digits[std::size_t(component) * elements + address] =
          IntegerDigit<T>(double(digit));
      residual = __dsub_rn(scaled, double(digit));
    }
  }
}
template <class T>
__global__ void WarpSplitKernel(const double* input, T* m_digits, int* m_scales,
                                int groups, int inner, bool matrix_a,
                                int components, int bits,
                                unsigned long long* m_range) {
  const int lane = int(threadIdx.x) & 31;
  const int group = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
  if (group >= groups) return;
  double maximum = 0;
  for (int index = lane; index < inner; index += 32) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    maximum = fmax(maximum, fabs(input[address]));
  }
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  int exponent = 0;
  if (lane == 0) {
    if (maximum != 0) {
      frexp(maximum, &exponent);
      ++exponent;
    }
    m_scales[group] = exponent;
  }
  exponent = __shfl_sync(0xffffffff, exponent, 0);
  const auto elements = std::size_t(groups) * inner;
  const double factor = NormalWeight(bits);
  for (int index = lane; index < inner; index += 32) {
    const auto address = matrix_a ? group + std::size_t(index) * groups
                                  : index + std::size_t(group) * inner;
    double residual = scalbn(input[address], -exponent);
    if (residual == 0 && input[address] != 0) atomicAdd(m_range, 1ULL);
    for (int component = 0; component < components; ++component) {
      const double scaled = __dmul_rn(residual, factor);
      const int digit = __double2int_rn(scaled);
      if (digit > (1 << (bits - 1)) || digit < -(1 << (bits - 1)))
        atomicAdd(m_range, 1ULL);
      m_digits[std::size_t(component) * elements + address] =
          IntegerDigit<T>(double(digit));
      residual = __dsub_rn(scaled, double(digit));
    }
  }
}
__global__ void OrderedKernel(const double* a, const double* b, double* m_c,
                              int m, int n, int k) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= std::size_t(m) * n) return;
  const int row = int(index % m), column = int(index / m);
  double sum = 0;
  for (int inner = 0; inner < k; ++inner)
    sum = __dadd_rn(sum, __dmul_rn(a[row + std::size_t(inner) * m],
                                   b[inner + std::size_t(column) * k]));
  m_c[index] = sum;
}
template <class T>
__global__ void WmmaKernel(const T* a, const T* b, const int* scale_a,
                           const int* scale_b, double* m_c, int m, int n, int k,
                           int components, int bits, int k_block,
                           bool triangular) {
  namespace wmma = nvcuda::wmma;
  __shared__ __align__(32) cuda::std::array<T, 256> tile_a;
  __shared__ __align__(32) cuda::std::array<T, 256> tile_b;
  __shared__ __align__(32) cuda::std::array<float, 256> tile_c;
  const int lane = int(threadIdx.x);
  const int row_base = int(blockIdx.x) * 16, col_base = int(blockIdx.y) * 16;
  cuda::std::array<double, 8> sums{};
  for (int left = 0; left < components; ++left) {
    for (int right = 0; right < components; ++right) {
      if (triangular && left + right >= components) continue;
      for (int start = 0; start < k; start += k_block) {
        const int stop = min(k, start + k_block);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (int inner = start; inner < stop; inner += 16) {
          for (int index = lane; index < 256; index += 32) {
            const int ar = row_base + index / 16, ak = inner + index % 16;
            const int bk = inner + index % 16, bc = col_base + index / 16;
            tile_a[index] =
                ar < m && ak < stop
                    ? a[std::size_t(left) * m * k + ar + std::size_t(ak) * m]
                    : IntegerDigit<T>(0);
            tile_b[index] =
                bc < n && bk < stop
                    ? b[std::size_t(right) * k * n + bk + std::size_t(bc) * k]
                    : IntegerDigit<T>(0);
          }
          __syncwarp();
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, tile_a.data(), 16);
          wmma::load_matrix_sync(fb, tile_b.data(), 16);
          wmma::mma_sync(accumulator, fa, fb, accumulator);
          __syncwarp();
        }
        wmma::store_matrix_sync(tile_c.data(), accumulator, 16,
                                wmma::mem_row_major);
        __syncwarp();
        for (int part = 0; part < 8; ++part) {
          const int index = lane + part * 32;
          sums[part] = __dadd_rn(
              sums[part], __dmul_rn(double(tile_c[index]),
                                    NormalWeight(-bits * (left + right + 2))));
        }
        __syncwarp();
      }
    }
  }
  for (int part = 0; part < 8; ++part) {
    const int index = lane + part * 32;
    const int row = row_base + index / 16, column = col_base + index % 16;
    if (row < m && column < n)
      m_c[row + std::size_t(column) * m] =
          scalbn(sums[part], scale_a[row] + scale_b[column]);
  }
}
template <class T>
__global__ void GroupedWmmaKernel(const T* a, const T* b, const int* scale_a,
                                  const int* scale_b, double* m_c, int m, int n,
                                  int k, int components, int bits, int k_block,
                                  bool triangular, int capacity) {
  namespace wmma = nvcuda::wmma;
  __shared__ __align__(32) cuda::std::array<T, 256> tile_a;
  __shared__ __align__(32) cuda::std::array<T, 256> tile_b;
  __shared__ __align__(32) cuda::std::array<float, 256> tile_c;
  const int lane = int(threadIdx.x);
  const int row_base = int(blockIdx.x) * 16, col_base = int(blockIdx.y) * 16;
  cuda::std::array<double, 8> sums{};
  const int diagonals = triangular ? components : 2 * components - 1;
  for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
    const int first = max(0, diagonal - components + 1);
    const int last = min(components - 1, diagonal);
    const double weight = NormalWeight(-bits * (diagonal + 2));
    for (int group = first; group <= last; group += capacity) {
      const int group_end = min(last + 1, group + capacity);
      for (int start = 0; start < k; start += k_block) {
        const int stop = min(k, start + k_block);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (int left = group; left < group_end; ++left) {
          const int right = diagonal - left;
          for (int inner = start; inner < stop; inner += 16) {
            for (int index = lane; index < 256; index += 32) {
              const int ar = row_base + index / 16, ak = inner + index % 16;
              const int bk = inner + index % 16, bc = col_base + index / 16;
              tile_a[index] =
                  ar < m && ak < stop
                      ? a[std::size_t(left) * m * k + ar + std::size_t(ak) * m]
                      : IntegerDigit<T>(0);
              tile_b[index] =
                  bc < n && bk < stop
                      ? b[std::size_t(right) * k * n + bk + std::size_t(bc) * k]
                      : IntegerDigit<T>(0);
            }
            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, tile_a.data(), 16);
            wmma::load_matrix_sync(fb, tile_b.data(), 16);
            wmma::mma_sync(accumulator, fa, fb, accumulator);
            __syncwarp();
          }
        }
        wmma::store_matrix_sync(tile_c.data(), accumulator, 16,
                                wmma::mem_row_major);
        __syncwarp();
        for (int part = 0; part < 8; ++part)
          sums[part] = __dadd_rn(
              sums[part], __dmul_rn(double(tile_c[lane + part * 32]), weight));
        __syncwarp();
      }
    }
  }
  for (int part = 0; part < 8; ++part) {
    const int index = lane + part * 32;
    const int row = row_base + index / 16, column = col_base + index % 16;
    if (row < m && column < n)
      m_c[row + std::size_t(column) * m] =
          scalbn(sums[part], scale_a[row] + scale_b[column]);
  }
}
__global__ void ReconstructKernel(const float* partials, const int* weights,
                                  const int* scale_a, const int* scale_b,
                                  double* m_c, int m, int n, int pairs,
                                  int chunks) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= std::size_t(m) * n) return;
  double sum = 0;
  for (int pair = 0; pair < pairs; ++pair)
    for (int chunk = 0; chunk < chunks; ++chunk)
      sum = __dadd_rn(
          sum,
          scalbn(double(partials[(std::size_t(chunk) * pairs + pair) * m * n +
                                 index]),
                 weights[pair]));
  m_c[index] = scalbn(sum, scale_a[index % m] + scale_b[index / m]);
}
struct Device {
  cudaStream_t stream_{};
  cublasHandle_t handle_{};
  double *a_ = nullptr, *b_ = nullptr, *c_ = nullptr;
  void* workspace_ = nullptr;
  Device(const Matrices& matrices) {
    Check(cudaSetDevice(0));
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Check(cublasCreate(&handle_));
    Check(cublasSetStream(handle_, stream_));
    Check(cudaMalloc(&workspace_, 64 * 1024 * 1024));
    Check(cublasSetWorkspace(handle_, workspace_, 64 * 1024 * 1024));
    Check(cudaMalloc(&a_, matrices.a_.size() * sizeof(double)));
    Check(cudaMalloc(&b_, matrices.b_.size() * sizeof(double)));
    Check(cudaMalloc(&c_, matrices.reference_.size() * sizeof(double)));
    Check(cudaMemcpyAsync(a_, matrices.a_.data(),
                          matrices.a_.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    Check(cudaMemcpyAsync(b_, matrices.b_.data(),
                          matrices.b_.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    Check(cudaStreamSynchronize(stream_));
  }
  void Close() {
    Check(cublasDestroy(handle_));
    Check(cudaFree(workspace_));
    Check(cudaFree(c_));
    Check(cudaFree(b_));
    Check(cudaFree(a_));
    Check(cudaStreamDestroy(stream_));
  }
};
template <class T>
struct Expansion {
  T *a_ = nullptr, *b_ = nullptr;
  int *scale_a_ = nullptr, *scale_b_ = nullptr, *weights_ = nullptr;
  unsigned long long* range_ = nullptr;
  float* partials_ = nullptr;
  void **pa_ = nullptr, **pb_ = nullptr, **pc_ = nullptr;
  int m_, n_, k_, components_, bits_, block_, pairs_, chunks_;
  bool triangular_, batched_;
  bool mma168_, warp_split_;
  bool cached_, quantized_;
  bool parallel_;
  bool fused_;
  vmecpp_experiment::expansion_mma168_parallel::Group* groups_ = nullptr;
  int group_count_ = 0;
  int group_capacity_;
  Expansion(const Matrices& matrices, int components, int bits, int block,
            bool triangular, bool batched, bool mma168, bool warp_split,
            bool cached, bool quantized, bool parallel, bool fused,
            int group_capacity, cudaStream_t stream)
      : m_(matrices.m_),
        n_(matrices.n_),
        k_(matrices.k_),
        components_(components),
        bits_(bits),
        block_(block),
        triangular_(triangular),
        batched_(batched),
        mma168_(mma168),
        warp_split_(warp_split),
        cached_(cached),
        quantized_(quantized),
        parallel_(parallel),
        fused_(fused && k_ <= 16 && block_ >= k_),
        group_capacity_(group_capacity) {
    Check(cudaMalloc(&a_, std::size_t(components_) * m_ * k_ * sizeof(T)));
    Check(cudaMalloc(&b_, std::size_t(components_) * k_ * n_ * sizeof(T)));
    Check(cudaMalloc(&scale_a_, m_ * sizeof(int)));
    Check(cudaMalloc(&scale_b_, n_ * sizeof(int)));
    Check(cudaMalloc(&range_, sizeof(*range_)));
    Check(cudaMemsetAsync(range_, 0, sizeof(*range_), stream));
    std::vector<std::array<int, 2>> pairs;
    std::vector<int> weights;
    for (int left = 0; left < components_; ++left)
      for (int right = 0; right < components_; ++right)
        if (!triangular_ || left + right < components_) {
          pairs.push_back({left, right});
          weights.push_back(-bits_ * (left + right + 2));
        }
    pairs_ = int(pairs.size());
    chunks_ = (k_ + block_ - 1) / block_;
    if (parallel_) {
      namespace parallel = vmecpp_experiment::expansion_mma168_parallel;
      const auto groups = parallel::BuildGroups(k_, components_, bits_, block_,
                                                triangular_, group_capacity_);
      group_count_ = int(groups.size());
      const auto bytes = parallel::ScratchBytes(m_, n_, group_count_);
      if (bytes > (std::size_t(4) << 30))
        throw std::runtime_error("Parallel scratch exceeds4GiB");
      Check(cudaMalloc(&groups_, groups.size() * sizeof(groups[0])));
      Check(cudaMalloc(&partials_, bytes));
      Check(cudaMemcpyAsync(groups_, groups.data(),
                            groups.size() * sizeof(groups[0]),
                            cudaMemcpyHostToDevice, stream));
      Check(cudaStreamSynchronize(stream));
    }
    if (!batched_) return;
    const auto elements = std::size_t(chunks_) * pairs_ * m_ * n_;
    if (elements > (std::size_t(1) << 30))
      throw std::runtime_error("Batched partial allocation exceeds4GiB");
    Check(cudaMalloc(&partials_, elements * sizeof(float)));
    Check(cudaMalloc(&weights_, pairs_ * sizeof(int)));
    Check(cudaMemcpyAsync(weights_, weights.data(), pairs_ * sizeof(int),
                          cudaMemcpyHostToDevice, stream));
    std::vector<void*> pa, pb, pc;
    for (int chunk = 0; chunk < chunks_; ++chunk)
      for (int pair = 0; pair < pairs_; ++pair) {
        pa.push_back(a_ + std::size_t(pairs[pair][0]) * m_ * k_ +
                     std::size_t(chunk * block_) * m_);
        pb.push_back(b_ + std::size_t(pairs[pair][1]) * k_ * n_ +
                     chunk * block_);
        pc.push_back(partials_ +
                     (std::size_t(chunk) * pairs_ + pair) * m_ * n_);
      }
    Check(cudaMalloc(&pa_, pa.size() * sizeof(void*)));
    Check(cudaMalloc(&pb_, pb.size() * sizeof(void*)));
    Check(cudaMalloc(&pc_, pc.size() * sizeof(void*)));
    Check(cudaMemcpyAsync(pa_, pa.data(), pa.size() * sizeof(void*),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaMemcpyAsync(pb_, pb.data(), pb.size() * sizeof(void*),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaMemcpyAsync(pc_, pc.data(), pc.size() * sizeof(void*),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaStreamSynchronize(stream));
  }
  void SplitA(const Device& device) {
    if (quantized_ && components_ * bits_ <= 64)
      QuantizedSplitKernel<T><<<(m_ + 3) / 4, 128, 0, device.stream_>>>(
          device.a_, a_, scale_a_, m_, k_, true, components_, bits_, range_);
    else if (warp_split_)
      WarpSplitKernel<T><<<(m_ + 3) / 4, 128, 0, device.stream_>>>(
          device.a_, a_, scale_a_, m_, k_, true, components_, bits_, range_);
    else
      SplitKernel<T><<<(m_ + 127) / 128, 128, 0, device.stream_>>>(
          device.a_, a_, scale_a_, m_, k_, true, components_, bits_, range_);
    Check(cudaGetLastError());
  }
  void SplitB(const Device& device) {
    if (quantized_ && components_ * bits_ <= 64)
      QuantizedSplitKernel<T><<<(n_ + 3) / 4, 128, 0, device.stream_>>>(
          device.b_, b_, scale_b_, n_, k_, false, components_, bits_, range_);
    else if (warp_split_)
      WarpSplitKernel<T><<<(n_ + 3) / 4, 128, 0, device.stream_>>>(
          device.b_, b_, scale_b_, n_, k_, false, components_, bits_, range_);
    else
      SplitKernel<T><<<(n_ + 127) / 128, 128, 0, device.stream_>>>(
          device.b_, b_, scale_b_, n_, k_, false, components_, bits_, range_);
    Check(cudaGetLastError());
  }
  void Multiply(const Device& device, cudaDataType_t type) {
    if (fused_) {
#define VMECPP_LAUNCH_FUSED(Q)                                           \
  vmecpp_experiment::expansion_mma168_fused::FusedKernel<T, Q>           \
      <<<dim3((m_ + 15) / 16, (n_ + 7) / 8), 32, 0, device.stream_>>>(   \
          device.a_, b_, scale_b_, device.c_, range_, m_, n_, k_, bits_, \
          triangular_, group_capacity_ > 0, group_capacity_)
      if (components_ == 4) {
        VMECPP_LAUNCH_FUSED(4);
      } else if (components_ == 6) {
        VMECPP_LAUNCH_FUSED(6);
      } else if (components_ == 8) {
        VMECPP_LAUNCH_FUSED(8);
      } else
        throw std::runtime_error("Unsupported fused component count");
#undef VMECPP_LAUNCH_FUSED
      Check(cudaGetLastError());
      return;
    }
    if (parallel_) {
      namespace parallel = vmecpp_experiment::expansion_mma168_parallel;
      parallel::ParallelKernel<T>
          <<<dim3((m_ + 15) / 16, (n_ + 7) / 8, group_count_), 32, 0,
             device.stream_>>>(a_, b_, groups_, partials_, m_, n_, k_,
                               group_count_);
      Check(cudaGetLastError());
      parallel::FinishKernel<<<(std::size_t(m_) * n_ + 255) / 256, 256, 0,
                               device.stream_>>>(partials_, groups_, scale_a_,
                                                 scale_b_, device.c_, m_, n_,
                                                 group_count_);
      Check(cudaGetLastError());
      return;
    }
    if (!batched_) {
      if (cached_ && k_ <= 16 && block_ >= k_) {
#define VMECPP_LAUNCH_CACHED(Q)                                        \
  vmecpp_experiment::expansion_mma168_cached::CachedKernel<T, Q>       \
      <<<dim3((m_ + 15) / 16, (n_ + 7) / 8), 32, 0, device.stream_>>>( \
          a_, b_, scale_a_, scale_b_, device.c_, m_, n_, k_, bits_,    \
          triangular_, group_capacity_ > 0, group_capacity_)
        if (components_ == 4) {
          VMECPP_LAUNCH_CACHED(4);
        } else if (components_ == 6) {
          VMECPP_LAUNCH_CACHED(6);
        } else if (components_ == 8) {
          VMECPP_LAUNCH_CACHED(8);
        } else
          throw std::runtime_error("Unsupported cache component count");
#undef VMECPP_LAUNCH_CACHED
      } else if (mma168_)
        vmecpp_experiment::expansion_mma168::Mma168Kernel<T>
            <<<dim3((m_ + 15) / 16, (n_ + 7) / 8), 32, 0, device.stream_>>>(
                a_, b_, scale_a_, scale_b_, device.c_, m_, n_, k_, components_,
                bits_, block_, triangular_, group_capacity_ > 0,
                group_capacity_);
      else if (group_capacity_ > 0)
        GroupedWmmaKernel<T>
            <<<dim3((m_ + 15) / 16, (n_ + 15) / 16), 32, 0, device.stream_>>>(
                a_, b_, scale_a_, scale_b_, device.c_, m_, n_, k_, components_,
                bits_, block_, triangular_, group_capacity_);
      else
        WmmaKernel<T>
            <<<dim3((m_ + 15) / 16, (n_ + 15) / 16), 32, 0, device.stream_>>>(
                a_, b_, scale_a_, scale_b_, device.c_, m_, n_, k_, components_,
                bits_, block_, triangular_);
      Check(cudaGetLastError());
      return;
    }
    const float alpha = 1, beta = 0;
    for (int chunk = 0; chunk < chunks_; ++chunk) {
      const int offset = chunk * pairs_;
      Check(cublasGemmBatchedEx(
          device.handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_,
          std::min(block_, k_ - chunk * block_), &alpha,
          const_cast<const void* const*>(pa_ + offset), type, m_,
          const_cast<const void* const*>(pb_ + offset), type, k_, &beta,
          pc_ + offset, CUDA_R_32F, m_, pairs_, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    ReconstructKernel<<<(std::size_t(m_) * n_ + 127) / 128, 128, 0,
                        device.stream_>>>(partials_, weights_, scale_a_,
                                          scale_b_, device.c_, m_, n_, pairs_,
                                          chunks_);
    Check(cudaGetLastError());
  }
  void Close() {
    for (void* pointer :
         {static_cast<void*>(a_), static_cast<void*>(b_),
          static_cast<void*>(scale_a_), static_cast<void*>(scale_b_),
          static_cast<void*>(weights_), static_cast<void*>(range_),
          static_cast<void*>(partials_), static_cast<void*>(pa_),
          static_cast<void*>(pb_), static_cast<void*>(pc_),
          static_cast<void*>(groups_)})
      Check(cudaFree(pointer));
  }
};
struct Error {
  std::uint64_t different_ = 0, ulp_ = 0;
  double absolute_ = 0, relative_l2_ = 0;
};
Error Compare(const std::vector<double>& expected,
              const std::vector<double>& actual) {
  Error error;
  long double squares = 0, denominator = 0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(actual[index]))
      throw std::runtime_error("Nonfinite output");
    const auto left = OrderedBits(expected[index]),
               right = OrderedBits(actual[index]);
    error.ulp_ =
        std::max(error.ulp_, left > right ? left - right : right - left);
    error.different_ += std::bit_cast<std::uint64_t>(expected[index]) !=
                        std::bit_cast<std::uint64_t>(actual[index]);
    const long double difference =
        static_cast<long double>(actual[index]) - expected[index];
    error.absolute_ = std::max(error.absolute_, double(fabsl(difference)));
    squares += difference * difference;
    denominator += static_cast<long double>(expected[index]) * expected[index];
  }
  error.relative_l2_ =
      double(sqrtl(denominator ? squares / denominator : squares));
  return error;
}
template <class T>
void Run(const Options& options, const Matrices& matrices, Device& m_device,
         std::vector<double>& m_native) {
  const bool native = options.mode_ == "native",
             ordered = options.mode_ == "ordered";
  const bool expansion = !native && !ordered;
  const bool bf16 = options.mode_.starts_with("bf16");
  const int components = expansion ? std::stoi(options.mode_.substr(5)) : 0;
  const int block = options.k_block_ == 0
                        ? matrices.k_
                        : std::min(options.k_block_, matrices.k_);
  const int limit = bf16 ? 9 : 12;
  const int bits = options.digits_
                       ? options.digits_
                       : std::min(bf16 ? 8 : 11, (24 - CeilLog2(block)) / 2);
  if (expansion &&
      (bits < 1 || bits > limit || 2 * (bits - 1) + CeilLog2(block) > 24))
    throw std::runtime_error(
        "Digit width violates exact FP32 integer-subdot bound");
  Check(cublasSetMathMode(
      m_device.handle_,
      native ? CUBLAS_PEDANTIC_MATH
             : cublasMath_t(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  const auto allocation_start = Raw();
  if (options.group_diagonals_ && options.backend_ == "batched")
    throw std::runtime_error("Grouping requires custom tensor backend");
  if (options.backend_ == "mma168-parallel" && !options.group_diagonals_)
    throw std::runtime_error(
        "Parallel backend requires explicit diagonal grouping");
  const int group_capacity =
      expansion && options.group_diagonals_
          ? int(std::min<unsigned long long>(
                components, (1ULL << 24) / (std::uint64_t(block) *
                                            (1ULL << (2 * (bits - 1))))))
          : 0;
  if (expansion && options.group_diagonals_ && group_capacity < 1)
    throw std::runtime_error("Unsafe grouped FP32 bound");
  std::unique_ptr<Expansion<T>> state;
  if (expansion)
    state.reset(new Expansion<T>(
        matrices, components, bits, block, options.terms_ == "triangular",
        options.backend_ == "batched",
        options.backend_ == "mma168" || options.backend_ == "mma168-cache" ||
            options.backend_ == "mma168-fused",
        options.split_ != "row", options.backend_ == "mma168-cache",
        options.split_ == "quantized", options.backend_ == "mma168-parallel",
        options.backend_ == "mma168-fused", group_capacity, m_device.stream_));
  Check(cudaStreamSynchronize(m_device.stream_));
  const auto allocation_end = Raw();
  cudaEvent_t begin{}, end{};
  Check(cudaEventCreate(&begin));
  Check(cudaEventCreate(&end));
  double b_setup_ms = 0;
  if (expansion && options.reuse_b_) {
    Check(cudaEventRecord(begin, m_device.stream_));
    state->SplitB(m_device);
    Check(cudaEventRecord(end, m_device.stream_));
    Check(cudaEventSynchronize(end));
    float elapsed = 0;
    Check(cudaEventElapsedTime(&elapsed, begin, end));
    b_setup_ms = elapsed;
  }
  const double alpha = 1, beta = 0;
  auto computation = [&] {
    if (native)
      Check(cublasGemmEx(
          m_device.handle_, CUBLAS_OP_N, CUBLAS_OP_N, matrices.m_, matrices.n_,
          matrices.k_, &alpha, m_device.a_, CUDA_R_64F, matrices.m_,
          m_device.b_, CUDA_R_64F, matrices.k_, &beta, m_device.c_, CUDA_R_64F,
          matrices.m_, CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    else if (ordered) {
      OrderedKernel<<<(matrices.reference_.size() + 127) / 128, 128, 0,
                      m_device.stream_>>>(m_device.a_, m_device.b_, m_device.c_,
                                          matrices.m_, matrices.n_,
                                          matrices.k_);
      Check(cudaGetLastError());
    } else {
      if (!state->fused_) state->SplitA(m_device);
      if (!options.reuse_b_) state->SplitB(m_device);
      state->Multiply(m_device, bf16 ? CUDA_R_16BF : CUDA_R_16F);
    }
  };
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  bool use_graph = false;
  auto launch = [&] {
    if (use_graph)
      Check(cudaGraphLaunch(graph_exec, m_device.stream_));
    else
      computation();
  };
  auto timed = [&](int count) {
    const auto raw = Raw();
    Check(cudaEventRecord(begin, m_device.stream_));
    for (int call = 0; call < count; ++call) launch();
    Check(cudaEventRecord(end, m_device.stream_));
    Check(cudaEventSynchronize(end));
    const auto finish = Raw();
    float elapsed = 0;
    Check(cudaEventElapsedTime(&elapsed, begin, end));
    const int operations = use_graph ? options.graph_batch_ : 1;
    return std::array<double, 2>{
        double(elapsed) / count / operations,
        double(finish - raw) / 1e6 / count / operations};
  };
  const auto first = timed(1);
  for (int warm = 0; warm < options.warmups_; ++warm) computation();
  Check(cudaStreamSynchronize(m_device.stream_));
  const auto capture_start = Raw();
  std::array<double, 2> first_graph{};
  if (!options.direct_) {
    Check(cudaStreamBeginCapture(m_device.stream_,
                                 cudaStreamCaptureModeThreadLocal));
    for (int operation = 0; operation < options.graph_batch_; ++operation)
      computation();
    Check(cudaStreamEndCapture(m_device.stream_, &graph));
    Check(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
    use_graph = true;
    first_graph = timed(1);
  }
  const auto capture_end = Raw();
  std::vector<double> events, raw;
  for (int sample = 0; sample < options.samples_; ++sample) {
    const auto elapsed = timed(options.calls_);
    events.push_back(elapsed[0]);
    raw.push_back(elapsed[1]);
  }
  if (options.profile_one_) {
    Check(cudaProfilerStart());
    launch();
    Check(cudaStreamSynchronize(m_device.stream_));
    Check(cudaProfilerStop());
  }
  std::vector<double> output(matrices.reference_.size());
  Check(cudaMemcpyAsync(output.data(), m_device.c_,
                        output.size() * sizeof(double), cudaMemcpyDeviceToHost,
                        m_device.stream_));
  unsigned long long range = 0;
  if (expansion)
    Check(cudaMemcpyAsync(&range, state->range_, sizeof(range),
                          cudaMemcpyDeviceToHost, m_device.stream_));
  Check(cudaStreamSynchronize(m_device.stream_));
  if (native) m_native = output;
  const auto original_error = Compare(matrices.reference_, output);
  const bool has_native = m_native.size() == output.size();
  const auto native_error = has_native ? Compare(m_native, output) : Error{};
  long double omitted = 0;
  if (expansion && options.terms_ == "triangular") {
    long double normalized = 0;
    for (int left = 0; left < components; ++left)
      for (int right = 0; right < components; ++right)
        if (left + right >= components)
          normalized += ldexpl(1.0L, -bits * (left + right));
    std::vector<int> sa(matrices.m_), sb(matrices.n_);
    if (state->fused_) {
      // The fused kernel keeps A exponents in registers, not this allocation.
      // Derive the identical input-only exponents outside every timed interval.
      for (int row = 0; row < matrices.m_; ++row) {
        double maximum = 0;
        for (int inner = 0; inner < matrices.k_; ++inner)
          maximum = std::max(
              maximum,
              std::abs(matrices.a_[row + std::size_t(inner) * matrices.m_]));
        if (maximum != 0) {
          std::frexp(maximum, &sa[row]);
          ++sa[row];
        }
      }
    } else {
      Check(cudaMemcpy(sa.data(), state->scale_a_, sa.size() * sizeof(int),
                       cudaMemcpyDeviceToHost));
    }
    Check(cudaMemcpy(sb.data(), state->scale_b_, sb.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    omitted = ldexpl(normalized * matrices.k_,
                     *std::max_element(sa.begin(), sa.end()) +
                         *std::max_element(sb.begin(), sb.end()));
  }
  if (!options.output_.empty()) {
    std::ofstream saved(options.output_, std::ios::binary);
    saved.write(reinterpret_cast<const char*>(output.data()),
                output.size() * sizeof(double));
    if (!saved) throw std::runtime_error("Output write failed");
  }
  if (graph_exec) Check(cudaGraphExecDestroy(graph_exec));
  if (graph) Check(cudaGraphDestroy(graph));
  Check(cudaEventDestroy(begin));
  Check(cudaEventDestroy(end));
  const int pair_count = expansion ? state->pairs_ : 1;
  const int parallel_group_count = expansion ? state->group_count_ : 0;
  const bool fused_active = expansion && state->fused_;
  if (state) state->Close();
  int version = 0;
  Check(cublasGetVersion(m_device.handle_, &version));
  std::ofstream report(options.report_);
  report
      << std::setprecision(17)
      << "{\"complete\":true,\"mode\":" << Quote(options.mode_)
      << ",\"type\":" << Quote(expansion ? (bf16 ? "BF16" : "FP16") : "FP64")
      << ",\"backend\":"
      << Quote(expansion
                   ? options.backend_
                   : (ordered ? "scalar_ordered_control" : "cublas_pedantic"))
      << ",\"split\":" << Quote(options.split_)
      << ",\"cached_operands_active\":"
      << (expansion && options.backend_ == "mma168-cache" &&
                  matrices.k_ <= 16 && block >= matrices.k_
              ? "true"
              : "false")
      << ",\"fused_A_split_active\":" << (fused_active ? "true" : "false")
      << ",\"quantized_split_A_active\":"
      << (expansion && !fused_active && options.split_ == "quantized" &&
                  components * bits <= 64
              ? "true"
              : "false")
      << ",\"quantized_split_active\":"
      << (expansion && options.split_ == "quantized" && components * bits <= 64
              ? "true"
              : "false")
      << ",\"quantized_split_recurrence_fallback\":"
      << (expansion && options.split_ == "quantized" && components * bits > 64
              ? "true"
              : "false")
      << ",\"m\":" << matrices.m_ << ",\"n\":" << matrices.n_
      << ",\"k\":" << matrices.k_ << ",\"components\":" << components
      << ",\"digit_bits\":" << (expansion ? bits : 0) << ",\"Kblock\":" << block
      << ",\"pair_count\":" << pair_count
      << ",\"parallel_group_count\":" << parallel_group_count
      << ",\"group_diagonals\":"
      << (options.group_diagonals_ ? "true" : "false")
      << ",\"group_pair_capacity\":" << group_capacity
      << ",\"digit_integer_abs_bound\":"
      << (expansion ? (1ULL << (bits - 1)) : 0)
      << ",\"profile_one\":" << (options.profile_one_ ? "true" : "false")
      << ",\"exact_FP32_subdot_bound\":" << (expansion ? "true" : "false")
      << ",\"graph_captured\":" << (use_graph ? "true" : "false")
      << ",\"graph_batch\":" << (use_graph ? options.graph_batch_ : 1)
      << ",\"first_graph_event_ms_per_operation\":" << first_graph[0]
      << ",\"first_graph_RAW_ms_per_operation\":" << first_graph[1]
      << ",\"reuse_B\":" << (options.reuse_b_ ? "true" : "false")
      << ",\"terms\":" << Quote(options.terms_)
      << ",\"range_failures\":" << range
      << ",\"fallback_count\":0,\"workspace_bytes\":67108864"
      << ",\"mode_allocation_RAW_ms\":"
      << double(allocation_end - allocation_start) / 1e6
      << ",\"B_presplit_event_ms\":" << b_setup_ms
      << ",\"capture_and_first_graph_RAW_ms\":"
      << double(capture_end - capture_start) / 1e6
      << ",\"first_direct_event_ms\":" << first[0]
      << ",\"first_direct_RAW_ms\":" << first[1]
      << ",\"steady_event_median_ms\":" << Median(events)
      << ",\"steady_RAW_diagnostic_median_ms\":" << Median(raw)
      << ",\"calls_per_sample\":" << options.calls_
      << ",\"samples\":" << options.samples_ << ",\"operations_per_sample\":"
      << options.calls_ * (use_graph ? options.graph_batch_ : 1)
      << ",\"original_different_bits\":" << original_error.different_
      << ",\"original_max_ULP\":" << original_error.ulp_
      << ",\"original_max_abs\":" << original_error.absolute_
      << ",\"original_relative_L2\":" << original_error.relative_l2_
      << ",\"native_reference_available\":" << (has_native ? "true" : "false")
      << ",\"native_different_bits\":" << native_error.different_
      << ",\"native_max_abs\":" << native_error.absolute_
      << ",\"native_relative_L2\":" << native_error.relative_l2_
      << ",\"omitted_cross_term_abs_bound\":" << double(omitted)
      << ",\"bound_scope\":\"discarded_digit_products_only_not_total_error\","
         "\"cublas_owner\":"
      << Quote(Owner("cublasGemmEx"))
      << ",\"cudart_owner\":" << Quote(Owner("cudaMalloc"))
      << ",\"cublas_version\":" << version
      << ",\"original_order_guarantee\":false,\"actual_tensor_path_proven\":"
         "false,\"solver_qualified\":false,\"event_samples_ms\":[";
  for (std::size_t i = 0; i < events.size(); ++i)
    report << (i ? "," : "") << events[i];
  report << "]}\n";
  if (!report) throw std::runtime_error("Report write failed");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    if (!options.execute_) {
      std::cout << "{\"phase\":\"PLAN\",\"GPU_calls\":0,\"graph_default\":true,"
                   "\"backend\":"
                << Quote(options.backend_) << "}\n";
      return 0;
    }
    for (char** entry = environ; entry && *entry; ++entry)
      if (std::string(*entry).starts_with("CUBLAS_"))
        throw std::runtime_error("Remove cuBLAS environment overrides");
    if (Owner("cublasGemmEx") !=
        std::filesystem::canonical(options.expect_cublas_).string())
      throw std::runtime_error("Unexpected cuBLAS owner");
    const auto matrices = Inputs(options);
    Device device(matrices);
    std::vector<double> native;
    const bool multi = options.mode_ == "all" || options.mode_ == "fp16-all" ||
                       options.mode_ == "bf16-all";
    std::vector<std::string> modes;
    for (const auto& mode : kModes)
      if (options.mode_ == "all" ||
          ((options.mode_ == "fp16-all" || options.mode_ == "bf16-all") &&
           (mode == "native" || mode == "ordered" ||
            mode.starts_with(options.mode_.substr(0, 4)))) ||
          options.mode_ == mode)
        modes.push_back(mode);
    for (const auto& mode : modes) {
      auto current = options;
      current.mode_ = mode;
      if (multi) {
        current.report_ += "." + mode + ".json";
        if (!current.output_.empty()) current.output_ += "." + mode + ".bin";
      }
      if (mode.starts_with("bf16"))
        Run<__nv_bfloat16>(current, matrices, device, native);
      else
        Run<__half>(current, matrices, device, native);
    }
    device.Close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Expansion benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
