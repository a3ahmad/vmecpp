// SPDX-License-Identifier: MIT
// Positive-sum integer-lattice proposal, always certified against original RN
// recurrence. Exponent guesses and map composition can NEVER accept a result
// without the complete finite bitwise certificate. No novelty claim.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <stdexcept>
#include <vector>

#include "gpu_lattice_prefix.h"

namespace {
constexpr int kThreads = 128;
constexpr int kBins = 2048;
constexpr int kBitmapWords = 64;
constexpr unsigned long long kLowerMantissa = 1ULL << 52;
constexpr unsigned long long kUpperMantissa = 1ULL << 53;
constexpr unsigned long long kFractionMask = kLowerMantissa - 1;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

struct Map {
  unsigned long long even_ = 0;
  unsigned long long odd_ = 0;
};

// Earlier left operand, later right operand. Noncommutative composition on
// Z/(2^64), whose parity is preserved by unsigned modular arithmetic.
struct Compose {
  __host__ __device__ Map operator()(const Map& earlier,
                                     const Map& later) const {
    return {
        earlier.even_ + ((earlier.even_ & 1) ? later.odd_ : later.even_),
        earlier.odd_ + (((1 + earlier.odd_) & 1) ? later.odd_ : later.even_)};
  }
};

struct Parameters {
  const double* terms_ = nullptr;
  double* guess_ = nullptr;
  double* output_ = nullptr;
  double* sums_ = nullptr;
  int* component_keys_ = nullptr;
  int* binade_keys_ = nullptr;
  Map* maps_ = nullptr;
  Map* scanned_ = nullptr;
  int* starts_ = nullptr;
  int* ends_ = nullptr;
  unsigned int* bitmap_ = nullptr;
  unsigned long long* seeds_ = nullptr;
  int* invalid_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  int components_ = 0;
  int stride_ = 0;
};

__device__ int Exponent(double value) {
  return (static_cast<unsigned long long>(__double_as_longlong(value)) >> 52) &
         2047;
}

__device__ Map TermMap(double term, int binade, bool& m_valid) {
  if (term == 0.0) return {};
  if (!isfinite(term) || term < 0.0 || binade < 1 || binade > 2046) {
    m_valid = false;
    return {};
  }
  const auto raw = static_cast<unsigned long long>(__double_as_longlong(term));
  const int field = (raw >> 52) & 2047;
  unsigned long long mantissa = raw & kFractionMask;
  int shift;
  if (field != 0) {
    mantissa += kLowerMantissa;
    shift = field - binade;
  } else {
    shift = 1 - binade;
  }
  if (shift >= 0) {
    if (shift >= 64 || mantissa > (ULLONG_MAX >> shift)) {
      m_valid = false;
      return {};
    }
    const auto integer = mantissa << shift;
    return {integer, integer};
  }
  const int places = -shift;
  if (places > 53) return {};
  const auto integer = mantissa >> places;
  const auto remainder = mantissa & ((1ULL << places) - 1);
  const auto half = 1ULL << (places - 1);
  if (remainder < half) return {integer, integer};
  if (remainder > half) return {integer + 1, integer + 1};
  return {integer + (integer & 1), integer + ((integer + 1) & 1)};
}

__global__ void InitializeKeys(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m_p.components_ * m_p.stride_)
    m_p.component_keys_[index] = index / m_p.stride_;
}

__global__ void Reset(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m_p.components_ * kBins) {
    m_p.starts_[index] = INT_MAX;
    m_p.ends_[index] = -1;
    m_p.seeds_[index] = 0;
  }
  if (index < m_p.components_ * kBitmapWords) m_p.bitmap_[index] = 0;
  if (index < m_p.components_) m_p.invalid_[index] = 0;
}

__global__ void PrepareMaps(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.components_ * m_p.stride_) return;
  const int component = index / m_p.stride_;
  const int local = index % m_p.stride_;
  const double value = m_p.guess_[index];
  const double term = m_p.terms_[index];
  const int field = Exponent(value);
  const int previous = local == 0 ? -1 : Exponent(m_p.guess_[index - 1]);
  bool valid = isfinite(value) && value >= 0.0 && isfinite(term) &&
               term >= 0.0 && !(field == 0 && value != 0.0) &&
               field >= previous;
  const int bin = component * kBins + field;
  m_p.binade_keys_[index] = bin;
  Map mapping{};
  if (field != previous) {
    // Atomics remain race-free even for unsupported decreasing exponent guesses
    // that revisit an old key. Such components are flagged for fallback.
    atomicMin(m_p.starts_ + bin, local);
    atomicOr(m_p.bitmap_ + component * kBitmapWords + field / 32,
             1U << (field % 32));
  } else if (field != 0) {
    mapping = TermMap(term, field, valid);
  } else if (term != 0.0) {
    valid = false;
  }
  const int next =
      local + 1 == m_p.stride_ ? -1 : Exponent(m_p.guess_[index + 1]);
  if (field != next) atomicMax(m_p.ends_ + bin, local);
  m_p.maps_[index] = mapping;
  if (!valid) atomicOr(m_p.invalid_ + component, 1);
}

#if defined(VMECPP_LATTICE_WARP_SEEDS)
static_assert(kBitmapWords == 64, "Warp metadata requires 64 bitmap words");
// Parallelize immutable metadata loads; lane zero retains every RN transition.
__global__ void SeedSegments(Parameters m_p) {
  const int lane = threadIdx.x % 32;
  const int component = blockIdx.x;
  const unsigned int low_word = m_p.bitmap_[component * kBitmapWords + lane];
  const unsigned int high_word =
      m_p.bitmap_[component * kBitmapWords + lane + 32];
  double previous = 0.0;
  const unsigned int low_mask = __ballot_sync(0xffffffffU, low_word != 0);
  const unsigned int high_mask = __ballot_sync(0xffffffffU, high_word != 0);
  for (int half = 0; half < 2; ++half) {
    unsigned int remaining_words = half == 0 ? low_mask : high_mask;
    while (remaining_words != 0) {
      const int owner = __ffs(remaining_words) - 1;
      remaining_words &= remaining_words - 1;
      const int word = half * 32 + owner;
      unsigned int occupied =
          __shfl_sync(0xffffffffU, half == 0 ? low_word : high_word, owner);
      const int own_field = word * 32 + lane;
      const int own_bin = component * kBins + own_field;
      int start = -1;
      int end = -1;
      double term = 0.0;
      Map mapping{};
      const bool present = (occupied & (1U << lane)) != 0;
      if (present && own_field > 0 && own_field <= 2046) {
        start = m_p.starts_[own_bin];
        end = m_p.ends_[own_bin];
        if (start >= 0 && start <= end && end < m_p.stride_) {
          term = m_p.terms_[component * m_p.stride_ + start];
          mapping = m_p.scanned_[component * m_p.stride_ + end];
        }
      }
      while (occupied != 0) {
        const int bit = __ffs(occupied) - 1;
        occupied &= occupied - 1;
        const int field = word * 32 + bit;
        const int bin = component * kBins + field;
        const int selected_start = __shfl_sync(0xffffffffU, start, bit);
        const int selected_end = __shfl_sync(0xffffffffU, end, bit);
        const double selected_term = __shfl_sync(0xffffffffU, term, bit);
        const auto even = __shfl_sync(0xffffffffU, mapping.even_, bit);
        const auto odd = __shfl_sync(0xffffffffU, mapping.odd_, bit);
        if (lane != 0) continue;
        if (field == 0) {
          previous = 0.0;
          continue;
        }
        if (field > 2046 || selected_start < 0 ||
            selected_start > selected_end || selected_end >= m_p.stride_) {
          atomicOr(m_p.invalid_ + component, 1);
          continue;
        }
        const double seed = __dadd_rn(previous, selected_term);
        const auto raw =
            static_cast<unsigned long long>(__double_as_longlong(seed));
        const auto mantissa = kLowerMantissa + (raw & kFractionMask);
        const auto delta = (mantissa & 1) ? odd : even;
        if (!isfinite(seed) || seed <= 0.0 || Exponent(seed) != field ||
            delta >= kUpperMantissa - mantissa) {
          atomicOr(m_p.invalid_ + component, 1);
          previous = 0.0;
          continue;
        }
        m_p.seeds_[bin] = raw;
        const auto end_bits = (static_cast<unsigned long long>(field) << 52) |
                              (mantissa + delta - kLowerMantissa);
        previous = __longlong_as_double(static_cast<long long>(end_bits));
        atomicAdd(m_p.statistics_ + 3, 1ULL);
      }
    }
  }
}

#else
__global__ void SeedSegments(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  double previous = 0.0;
  for (int word = 0; word < kBitmapWords; ++word) {
    unsigned int occupied = m_p.bitmap_[component * kBitmapWords + word];
    while (occupied != 0) {
      const int bit = __ffs(occupied) - 1;
      occupied &= occupied - 1;
      const int field = word * 32 + bit;
      const int bin = component * kBins + field;
      if (field == 0) {
        previous = 0.0;
        continue;
      }
      const int start = m_p.starts_[bin];
      const int end = m_p.ends_[bin];
      if (field > 2046 || start < 0 || start > end || end >= m_p.stride_) {
        atomicOr(m_p.invalid_ + component, 1);
        continue;
      }
      const double seed =
          __dadd_rn(previous, m_p.terms_[component * m_p.stride_ + start]);
      const auto raw =
          static_cast<unsigned long long>(__double_as_longlong(seed));
      const auto mantissa = kLowerMantissa + (raw & kFractionMask);
      const auto mapping = m_p.scanned_[component * m_p.stride_ + end];
      const auto delta = (mantissa & 1) ? mapping.odd_ : mapping.even_;
      if (!isfinite(seed) || seed <= 0.0 || Exponent(seed) != field ||
          delta >= kUpperMantissa - mantissa) {
        atomicOr(m_p.invalid_ + component, 1);
        previous = 0.0;
        continue;
      }
      m_p.seeds_[bin] = raw;
      const auto end_bits = (static_cast<unsigned long long>(field) << 52) |
                            (mantissa + delta - kLowerMantissa);
      previous = __longlong_as_double(static_cast<long long>(end_bits));
      atomicAdd(m_p.statistics_ + 3, 1ULL);
    }
  }
}

#endif

__global__ void Assemble(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.components_ * m_p.stride_) return;
  const int component = index / m_p.stride_;
  const int bin = m_p.binade_keys_[index];
  const int field = bin % kBins;
  const auto raw = m_p.seeds_[bin];
  if (field == 0) {
    m_p.output_[index] = 0.0;
    return;
  }
  const auto mantissa = kLowerMantissa + (raw & kFractionMask);
  const auto mapping = m_p.scanned_[index];
  const auto delta = (mantissa & 1) ? mapping.odd_ : mapping.even_;
  if (raw == 0 || field > 2046 || delta >= kUpperMantissa - mantissa) {
    atomicOr(m_p.invalid_ + component, 1);
    m_p.output_[index] = 0.0;
    return;
  }
  const auto result = (static_cast<unsigned long long>(field) << 52) |
                      (mantissa + delta - kLowerMantissa);
  m_p.output_[index] = __longlong_as_double(static_cast<long long>(result));
}

__global__ void Verify(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.components_ * m_p.stride_) return;
  const int component = index / m_p.stride_;
  const double previous = index % m_p.stride_ ? m_p.output_[index - 1] : 0.0;
  const double term = m_p.terms_[index];
  const double proposed = m_p.output_[index];
  const double target = __dadd_rn(previous, term);
  if (!isfinite(term) || !isfinite(proposed) ||
      __double_as_longlong(target) != __double_as_longlong(proposed))
    atomicOr(m_p.invalid_ + component, 1);
}

__global__ void CorruptProposal(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  const int index = component * m_p.stride_ + m_p.stride_ / 2;
  const auto raw = __double_as_longlong(m_p.output_[index]);
  m_p.output_[index] = __longlong_as_double(raw ^ 1LL);
}

__global__ void Ordered(const double* terms, double* m_prefix, int components,
                        int stride, const int* invalid, bool full_prefix) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= components ||
      (invalid != nullptr && invalid[component] == 0))
    return;
  double sum = 0.0;
  for (int local = 0; local < stride; ++local) {
    const int index = component * stride + local;
    sum = __dadd_rn(sum, terms[index]);
    if (full_prefix) m_prefix[index] = sum;
  }
  if (!full_prefix) m_prefix[component] = sum;
}

__global__ void RecordStatistics(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  if (component == 0) atomicAdd(m_p.statistics_, 1ULL);
  atomicAdd(m_p.statistics_ + (m_p.invalid_[component] ? 2 : 1), 1ULL);
}

__global__ void Finish(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component < m_p.components_ && m_p.invalid_[component] == 0)
    m_p.sums_[component] = m_p.output_[(component + 1) * m_p.stride_ - 1];
}

__global__ void RecordStatisticsAndFinish(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  // Verify is a separate preceding kernel. Its flags and all prefixes are
  // immutable here. Components write distinct sums; statistics are integers.
  const bool invalid = m_p.invalid_[component] != 0;
  if (component == 0) atomicAdd(m_p.statistics_, 1ULL);
  atomicAdd(m_p.statistics_ + (invalid ? 2 : 1), 1ULL);
  if (!invalid)
    m_p.sums_[component] = m_p.output_[(component + 1) * m_p.stride_ - 1];
}

class Workspace {
 public:
  Parameters parameters_;
  cudaStream_t stream_;
  void* temporary_ = nullptr;
  std::size_t temporary_bytes_ = 0;
  std::vector<void*> allocations_;

  Workspace(int components, int stride, void* stream)
      : stream_(static_cast<cudaStream_t>(stream)) {
    if (components < 1 || components > 32 || stride < 1 || stride > (1 << 22) ||
        stream == nullptr)
      throw std::runtime_error("Invalid lattice prefix dimensions/stream");
    parameters_.components_ = components;
    parameters_.stride_ = stride;
  }
  ~Workspace() {
    // The caller may already have retired its stream. cudaFree orders uses of
    // these allocations without touching a stale stream handle.
    for (void* pointer : allocations_) cudaFree(pointer);
  }
  template <typename T>
  T* Allocate(std::size_t count) {
    void* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(T)));
    allocations_.push_back(pointer);
    return static_cast<T*>(pointer);
  }
  void Initialize() {
    auto& p = parameters_;
    const int count = p.components_ * p.stride_;
    p.guess_ = Allocate<double>(count);
    p.component_keys_ = Allocate<int>(count);
    p.binade_keys_ = Allocate<int>(count);
    p.maps_ = Allocate<Map>(count);
    p.scanned_ = Allocate<Map>(count);
    p.starts_ = Allocate<int>(p.components_ * kBins);
    p.ends_ = Allocate<int>(p.components_ * kBins);
    p.seeds_ = Allocate<unsigned long long>(p.components_ * kBins);
    p.bitmap_ = Allocate<unsigned int>(p.components_ * kBitmapWords);
    p.invalid_ = Allocate<int>(p.components_);
    p.statistics_ = Allocate<unsigned long long>(4);
    Check(cudaMemsetAsync(p.statistics_, 0, 4 * sizeof(unsigned long long),
                          stream_));
    InitializeKeys<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream_>>>(
        p);
    Check(cudaGetLastError());
    std::size_t guesses = 0;
    Check(cub::DeviceScan::InclusiveScanByKey(
        nullptr, guesses, p.component_keys_, p.guess_, p.guess_, cub::Sum(),
        count, cub::Equality(), stream_));
    std::size_t maps = 0;
    Check(cub::DeviceScan::InclusiveScanByKey(nullptr, maps, p.binade_keys_,
                                              p.maps_, p.scanned_, Compose{},
                                              count, cub::Equality(), stream_));
    temporary_bytes_ = std::max(guesses, maps);
    temporary_ = Allocate<unsigned char>(temporary_bytes_);
    Check(cudaStreamSynchronize(stream_));
  }
  void Run(const double* terms, double* m_output, bool corrupt = false,
           bool fallback = true, double* m_sums = nullptr) {
    auto p = parameters_;
    p.terms_ = terms;
    p.output_ = m_output;
    p.sums_ = m_sums;
    const int count = p.components_ * p.stride_;
    const int blocks = (count + kThreads - 1) / kThreads;
    Reset<<<(p.components_ * kBins + kThreads - 1) / kThreads, kThreads, 0,
            stream_>>>(p);
    Check(cudaGetLastError());
    std::size_t temporary = temporary_bytes_;
    Check(cub::DeviceScan::InclusiveScanByKey(
        temporary_, temporary, p.component_keys_, terms, p.guess_, cub::Sum(),
        count, cub::Equality(), stream_));
    PrepareMaps<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    temporary = temporary_bytes_;
    Check(cub::DeviceScan::InclusiveScanByKey(
        temporary_, temporary, p.binade_keys_, p.maps_, p.scanned_, Compose{},
        count, cub::Equality(), stream_));
#if defined(VMECPP_LATTICE_WARP_SEEDS)
    SeedSegments<<<p.components_, 32, 0, stream_>>>(p);
#else
    SeedSegments<<<1, 32, 0, stream_>>>(p);
#endif
    Check(cudaGetLastError());
    Assemble<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    if (corrupt) {
      CorruptProposal<<<1, 32, 0, stream_>>>(p);
      Check(cudaGetLastError());
    }
    Verify<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    if (fallback) {
      Ordered<<<1, 32, 0, stream_>>>(terms, m_output, p.components_, p.stride_,
                                     p.invalid_, true);
      Check(cudaGetLastError());
    }
    if (m_sums == nullptr) {
      RecordStatistics<<<1, 32, 0, stream_>>>(p);
    } else {
      RecordStatisticsAndFinish<<<1, 32, 0, stream_>>>(p);
    }
    Check(cudaGetLastError());
  }
};
}  // namespace

extern "C" void* CreateGpuLatticePrefix(int components, int stride,
                                        void* stream) {
  auto* result = new Workspace(components, stride, stream);
  try {
    result->Initialize();
  } catch (...) {
    delete result;
    throw;
  }
  return result;
}
extern "C" void DestroyGpuLatticePrefix(void* m_context) {
  delete static_cast<Workspace*>(m_context);
}
extern "C" void RunGpuLatticePrefix(void* m_context, const double* terms,
                                    double* m_prefix) {
  static_cast<Workspace*>(m_context)->Run(terms, m_prefix);
}
extern "C" void RunGpuLatticePrefixCorrupted(void* m_context,
                                             const double* terms,
                                             double* m_prefix) {
  static_cast<Workspace*>(m_context)->Run(terms, m_prefix, true);
}
extern "C" void ProposeGpuLatticePrefix(void* m_context, const double* terms,
                                        double* m_prefix, bool corrupt) {
  static_cast<Workspace*>(m_context)->Run(terms, m_prefix, corrupt, false);
}
extern "C" void FinishGpuLatticePrefix(void* m_context, const double* prefix,
                                       double* m_sums) {
  auto* context = static_cast<Workspace*>(m_context);
  auto p = context->parameters_;
  p.output_ = const_cast<double*>(prefix);
  p.sums_ = m_sums;
  Finish<<<1, 32, 0, context->stream_>>>(p);
  Check(cudaGetLastError());
}
extern "C" void ProposeAndFinishGpuLatticePrefix(void* m_context,
                                                 const double* terms,
                                                 double* m_prefix,
                                                 double* m_sums, bool corrupt) {
  if (m_sums == nullptr)
    throw std::runtime_error("Null fused lattice sum output");
  static_cast<Workspace*>(m_context)->Run(terms, m_prefix, corrupt, false,
                                          m_sums);
}
extern "C" const int* GpuLatticePrefixInvalidDevice(void* context) {
  return static_cast<Workspace*>(context)->parameters_.invalid_;
}
extern "C" void RebindGpuLatticePrefix(void* m_context, void* stream) {
  if (stream == nullptr) throw std::runtime_error("Null lattice stream");
  static_cast<Workspace*>(m_context)->stream_ =
      static_cast<cudaStream_t>(stream);
}
extern "C" void LaunchGpuLatticeOrdered(const double* terms, double* m_prefix,
                                        int components, int stride,
                                        void* stream) {
  if (components < 1 || components > 32 || stride < 1 || stream == nullptr)
    throw std::runtime_error("Invalid ordered prefix dimensions/stream");
  Ordered<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      terms, m_prefix, components, stride, nullptr, true);
  Check(cudaGetLastError());
}
extern "C" void LaunchGpuLatticeOrderedSums(const double* terms, double* m_sums,
                                            int components, int stride,
                                            void* stream) {
  if (components < 1 || components > 32 || stride < 1 || stream == nullptr)
    throw std::runtime_error("Invalid ordered sums dimensions/stream");
  Ordered<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      terms, m_sums, components, stride, nullptr, false);
  Check(cudaGetLastError());
}
extern "C" void ReadGpuLatticePrefixStats(
    void* m_context, std::array<unsigned long long, 4>& m_stats) {
  const auto* context = static_cast<Workspace*>(m_context);
  Check(cudaMemcpyAsync(m_stats.data(), context->parameters_.statistics_,
                        sizeof(m_stats), cudaMemcpyDeviceToHost,
                        context->stream_));
  Check(cudaStreamSynchronize(context->stream_));
}
extern "C" void ReadGpuLatticePrefixFlags(void* m_context, int* m_flags) {
  const auto* context = static_cast<Workspace*>(m_context);
  Check(cudaMemcpyAsync(m_flags, context->parameters_.invalid_,
                        context->parameters_.components_ * sizeof(int),
                        cudaMemcpyDeviceToHost, context->stream_));
  Check(cudaStreamSynchronize(context->stream_));
}
