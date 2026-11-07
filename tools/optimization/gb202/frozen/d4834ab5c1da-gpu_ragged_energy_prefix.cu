// SPDX-License-Identifier: MIT
#if defined(VMECPP_LATTICE_ENERGY_RAGGED)
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

#include "gpu_ragged_energy_prefix.h"
#include "ragged_shape.h"
#include "coordinate_map.h"
#ifndef VMECPP_ENERGY_COORDINATE_MAP
#define VMECPP_ENERGY_COORDINATE_MAP 0
#endif

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

struct TermActionMap {
  unsigned long long even_ = 0;
  unsigned long long odd_ = 0;
};

#if VMECPP_ENERGY_COORDINATE_MAP
using Map = residual_coordinate_map::Map;
#else
using Map = TermActionMap;
#endif

// Earlier left operand, later right operand. Noncommutative composition on
// Z/(2^64), whose parity is preserved by unsigned modular arithmetic.
struct Compose {
  __host__ __device__ Map operator()(const Map& earlier,
                                     const Map& later) const {
#if VMECPP_ENERGY_COORDINATE_MAP
    return residual_coordinate_map::Compose(earlier, later);
#else
    return {
        earlier.even_ + ((earlier.even_ & 1) ? later.odd_ : later.even_),
        earlier.odd_ + (((1 + earlier.odd_) & 1) ? later.odd_ : later.even_)};
#endif
  }
};

#define VMECPP_ENERGY_PINNED_256X3 1
// Private pinned CUB policy: only block threads/items differ per architecture.
static_assert(VMECPP_ENERGY_PINNED_256X3 == 0 || VMECPP_ENERGY_PINNED_256X3 == 1);
#if VMECPP_ENERGY_PINNED_256X3
static_assert(CUB_VERSION == 200802);
using StockMapPolicyHub =
    cub::detail::scan_by_key::policy_hub<int*, Map, Map, Compose>;


struct ResidualMapPolicyHub {
  template <class Stock>
  using Tuned =
      cub::AgentScanByKeyPolicy<256, 3, Stock::LOAD_ALGORITHM,
                                Stock::LOAD_MODIFIER, Stock::SCAN_ALGORITHM,
                                Stock::STORE_ALGORITHM,
                                typename Stock::detail::delay_constructor_t>;

  struct Policy800
      : cub::ChainedPolicy<800, Policy800, StockMapPolicyHub::Policy520> {
    using ScanByKeyPolicyT =
        Tuned<StockMapPolicyHub::Policy800::ScanByKeyPolicyT>;
  };
  struct Policy860 : cub::ChainedPolicy<860, Policy860, Policy800> {
    using ScanByKeyPolicyT =
        Tuned<StockMapPolicyHub::Policy860::ScanByKeyPolicyT>;
  };
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy860> {
    using ScanByKeyPolicyT =
        Tuned<StockMapPolicyHub::Policy900::ScanByKeyPolicyT>;
  };
  struct Policy1000 : cub::ChainedPolicy<1000, Policy1000, Policy900> {
    using ScanByKeyPolicyT =
        Tuned<StockMapPolicyHub::Policy1000::ScanByKeyPolicyT>;
  };
  using MaxPolicy = Policy1000;
};
#endif

cudaError_t ScanEnergyCoordinateMaps(void* m_temporary, std::size_t& m_bytes,
                                   int* keys, Map* maps, Map* m_scanned,
                                   int count, cudaStream_t stream) {
#if VMECPP_ENERGY_PINNED_256X3
  // Match the public inclusive API's NVTX scope and exact iterator/offset
  // types.
  CUB_DETAIL_NVTX_RANGE_SCOPE_IF(m_temporary,
                                 "cub::DeviceScan::InclusiveScanByKey");
  using Offset = cub::detail::choose_offset_t<int>;
  return cub::DispatchScanByKey<
      int*, Map*, Map*, cub::Equality, Compose, cub::NullType, Offset, Map,
      ResidualMapPolicyHub>::Dispatch(m_temporary, m_bytes, keys, maps,
                                      m_scanned, cub::Equality(), Compose(),
                                      cub::NullType(), count, stream);
#else
  return cub::DeviceScan::InclusiveScanByKey(m_temporary, m_bytes, keys, maps,
                                             m_scanned, Compose(), count,
                                             cub::Equality(), stream);
#endif
}

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
  RaggedShape shape_{};
};

__device__ int Exponent(double value) {
  return (static_cast<unsigned long long>(__double_as_longlong(value)) >> 52) &
         2047;
}

__device__ TermActionMap TermMap(double term, int binade, bool& m_valid) {
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

#if VMECPP_ENERGY_COORDINATE_MAP
__device__ bool CoordinateEnding(Map mapping, unsigned long long seed,
                                 unsigned int start_low,
                                 unsigned long long guessed_mantissa,
                                 unsigned long long& m_mantissa) {
  const auto initial = (static_cast<unsigned int>(seed) - start_low) & 65535U;
  const auto ending =
      (initial + residual_coordinate_map::Cost(mapping, initial & 1)) & 65535U;
  const long long error = ending < 32768U ? ending
                                        : static_cast<long long>(ending) - 65536;
  const auto result = static_cast<long long>(guessed_mantissa) + error;
  if (result < static_cast<long long>(kLowerMantissa) ||
      result >= static_cast<long long>(kUpperMantissa)) return false;
  m_mantissa = static_cast<unsigned long long>(result);
  return true;
}
#endif

__global__ void RaggedInitializeKeys(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m_p.shape_.Count())
    m_p.component_keys_[index] = m_p.shape_.Component(index);
}

__global__ void RaggedReset(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m_p.components_ * kBins) {
    m_p.starts_[index] = INT_MAX;
    m_p.ends_[index] = -1;
    m_p.seeds_[index] = 0;
  }
  if (index < m_p.components_ * kBitmapWords) m_p.bitmap_[index] = 0;
  if (index < m_p.components_) m_p.invalid_[index] = 0;
}

__global__ void RaggedPrepareMaps(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.shape_.Count()) return;
  const int component = m_p.shape_.Component(index);
  const int local = m_p.shape_.Local(index);
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
    const auto action = TermMap(term, field, valid);
#if VMECPP_ENERGY_COORDINATE_MAP
    // Never narrow a huge original action into an apparently valid zero.
    if (action.even_ >= kUpperMantissa || action.odd_ >= kUpperMantissa) {
      valid = false;
    } else {
      const auto previous_raw = static_cast<unsigned long long>(
          __double_as_longlong(m_p.guess_[index - 1]));
      const auto current_raw = static_cast<unsigned long long>(
          __double_as_longlong(value));
      // Even an original zero action needs -DeltaG in the coordinate frame.
      mapping = residual_coordinate_map::Translate(
          action.even_, action.odd_,
          kLowerMantissa + (previous_raw & kFractionMask),
          kLowerMantissa + (current_raw & kFractionMask));
    }
#else
    mapping = action;
#endif
  } else if (term != 0.0) {
    valid = false;
  }
  const int next = local + 1 == m_p.shape_.Length(component)
                       ? -1
                       : Exponent(m_p.guess_[index + 1]);
  if (field != next) atomicMax(m_p.ends_ + bin, local);
  m_p.maps_[index] = mapping;
  if (!valid) atomicOr(m_p.invalid_ + component, 1);
}

#if defined(VMECPP_LATTICE_WARP_SEEDS)
static_assert(kBitmapWords == 64, "Warp metadata requires 64 bitmap words");
// Parallelize immutable metadata loads; lane zero retains every RN transition.
__global__ void RaggedSeedSegments(Parameters m_p) {
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
#if VMECPP_ENERGY_COORDINATE_MAP
      unsigned int start_low = 0;
      unsigned long long guessed_end = 0;
#endif
      const bool present = (occupied & (1U << lane)) != 0;
      if (present && own_field > 0 && own_field <= 2046) {
        start = m_p.starts_[own_bin];
        end = m_p.ends_[own_bin];
        if (start >= 0 && start <= end && end < m_p.shape_.Length(component)) {
          term = m_p.terms_[m_p.shape_.Offset(component) + start];
          mapping = m_p.scanned_[m_p.shape_.Offset(component) + end];
#if VMECPP_ENERGY_COORDINATE_MAP
          const int offset = m_p.shape_.Offset(component);
          start_low = static_cast<unsigned long long>(
              __double_as_longlong(m_p.guess_[offset + start])) & 65535U;
          const auto end_raw = static_cast<unsigned long long>(
              __double_as_longlong(m_p.guess_[offset + end]));
          guessed_end = kLowerMantissa + (end_raw & kFractionMask);
#endif
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
#if VMECPP_ENERGY_COORDINATE_MAP
        Map selected_map{};
        selected_map.bits_ = __shfl_sync(0xffffffffU, mapping.bits_, bit);
        const auto selected_start_low = __shfl_sync(0xffffffffU, start_low, bit);
        const auto selected_guessed_end = __shfl_sync(0xffffffffU, guessed_end, bit);
#else
        const auto even = __shfl_sync(0xffffffffU, mapping.even_, bit);
        const auto odd = __shfl_sync(0xffffffffU, mapping.odd_, bit);
#endif
        if (lane != 0) continue;
        if (field == 0) {
          previous = 0.0;
          continue;
        }
        if (field > 2046 || selected_start < 0 ||
            selected_start > selected_end ||
            selected_end >= m_p.shape_.Length(component)) {
          atomicOr(m_p.invalid_ + component, 1);
          continue;
        }
        const double seed = __dadd_rn(previous, selected_term);
        const auto raw =
            static_cast<unsigned long long>(__double_as_longlong(seed));
        const auto mantissa = kLowerMantissa + (raw & kFractionMask);
#if VMECPP_ENERGY_COORDINATE_MAP
        unsigned long long ending_mantissa = 0;
        const bool range_valid = CoordinateEnding(selected_map, raw,
            selected_start_low, selected_guessed_end, ending_mantissa);
#else
        const auto delta = (mantissa & 1) ? odd : even;
        const bool range_valid = delta < kUpperMantissa - mantissa;
        const auto ending_mantissa = mantissa + delta;
#endif
        if (!isfinite(seed) || seed <= 0.0 || Exponent(seed) != field ||
            !range_valid) {
          atomicOr(m_p.invalid_ + component, 1);
          previous = 0.0;
          continue;
        }
        m_p.seeds_[bin] = raw;
        const auto end_bits = (static_cast<unsigned long long>(field) << 52) |
                              (ending_mantissa - kLowerMantissa);
        previous = __longlong_as_double(static_cast<long long>(end_bits));
        atomicAdd(m_p.statistics_ + 4 * component + 3, 1ULL);
      }
    }
  }
}

#else
__global__ void RaggedSeedSegments(Parameters m_p) {
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
      if (field > 2046 || start < 0 || start > end ||
          end >= m_p.shape_.Length(component)) {
        atomicOr(m_p.invalid_ + component, 1);
        continue;
      }
      const double seed =
          __dadd_rn(previous, m_p.terms_[m_p.shape_.Offset(component) + start]);
      const auto raw =
          static_cast<unsigned long long>(__double_as_longlong(seed));
      const auto mantissa = kLowerMantissa + (raw & kFractionMask);
      const auto mapping = m_p.scanned_[m_p.shape_.Offset(component) + end];
#if VMECPP_ENERGY_COORDINATE_MAP
      const int offset = m_p.shape_.Offset(component);
      const auto start_raw = static_cast<unsigned long long>(
          __double_as_longlong(m_p.guess_[offset + start]));
      const auto end_raw = static_cast<unsigned long long>(
          __double_as_longlong(m_p.guess_[offset + end]));
      unsigned long long ending_mantissa = 0;
      const bool range_valid = CoordinateEnding(mapping, raw, start_raw & 65535U,
          kLowerMantissa + (end_raw & kFractionMask), ending_mantissa);
#else
      const auto delta = (mantissa & 1) ? mapping.odd_ : mapping.even_;
      const bool range_valid = delta < kUpperMantissa - mantissa;
      const auto ending_mantissa = mantissa + delta;
#endif
      if (!isfinite(seed) || seed <= 0.0 || Exponent(seed) != field ||
          !range_valid) {
        atomicOr(m_p.invalid_ + component, 1);
        previous = 0.0;
        continue;
      }
      m_p.seeds_[bin] = raw;
      const auto end_bits = (static_cast<unsigned long long>(field) << 52) |
                            (ending_mantissa - kLowerMantissa);
      previous = __longlong_as_double(static_cast<long long>(end_bits));
      atomicAdd(m_p.statistics_ + 4 * component + 3, 1ULL);
    }
  }
}

#endif

__global__ void RaggedAssemble(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.shape_.Count()) return;
  const int component = m_p.shape_.Component(index);
  const int bin = m_p.binade_keys_[index];
  const int field = bin % kBins;
  const auto raw = m_p.seeds_[bin];
  if (field == 0) {
    m_p.output_[index] = 0.0;
    return;
  }
  const auto mantissa = kLowerMantissa + (raw & kFractionMask);
  const auto mapping = m_p.scanned_[index];
#if VMECPP_ENERGY_COORDINATE_MAP
  const int start = m_p.starts_[bin];
  unsigned long long ending_mantissa = 0;
  bool range_valid = raw != 0 && field <= 2046 && start >= 0 &&
                     start <= m_p.shape_.Local(index);
  if (range_valid) {
    const auto start_raw = static_cast<unsigned long long>(__double_as_longlong(
        m_p.guess_[m_p.shape_.Offset(component) + start]));
    const auto current_raw = static_cast<unsigned long long>(
        __double_as_longlong(m_p.guess_[index]));
    range_valid = CoordinateEnding(mapping, raw, start_raw & 65535U,
        kLowerMantissa + (current_raw & kFractionMask), ending_mantissa);
  }
  if (!range_valid) {
#else
  const auto delta = (mantissa & 1) ? mapping.odd_ : mapping.even_;
  const auto ending_mantissa = mantissa + delta;
  if (raw == 0 || field > 2046 || delta >= kUpperMantissa - mantissa) {
#endif
    atomicOr(m_p.invalid_ + component, 1);
    m_p.output_[index] = 0.0;
    return;
  }
  const auto result = (static_cast<unsigned long long>(field) << 52) |
                      (ending_mantissa - kLowerMantissa);
  m_p.output_[index] = __longlong_as_double(static_cast<long long>(result));
}

__global__ void RaggedVerify(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.shape_.Count()) return;
  const int component = m_p.shape_.Component(index);
  const double previous =
      m_p.shape_.Local(index) ? m_p.output_[index - 1] : 0.0;
  const double term = m_p.terms_[index];
  const double proposed = m_p.output_[index];
  const double target = __dadd_rn(previous, term);
  if (!isfinite(term) || !isfinite(proposed) ||
      __double_as_longlong(target) != __double_as_longlong(proposed))
    atomicOr(m_p.invalid_ + component, 1);
}

__global__ void RaggedCorruptProposal(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  const int index =
      m_p.shape_.Offset(component) + m_p.shape_.Length(component) / 2;
  const auto raw = __double_as_longlong(m_p.output_[index]);
  m_p.output_[index] = __longlong_as_double(raw ^ 1LL);
}

__global__ void RaggedRecordStatistics(Parameters m_p) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= m_p.components_) return;
  atomicAdd(m_p.statistics_ + 4 * component, 1ULL);
  atomicAdd(m_p.statistics_ + 4 * component + (m_p.invalid_[component] ? 2 : 1),
            1ULL);
}

#if VMECPP_ENERGY_COORDINATE_MAP
class Workspace;
thread_local Workspace* last_coordinate_workspace = nullptr;
#endif
class Workspace {
 public:
  Parameters parameters_;
  cudaStream_t stream_;
  void* temporary_ = nullptr;
  std::size_t temporary_bytes_ = 0;
  std::vector<void*> allocations_;

  Workspace(int thermal_count, int magnetic_count, void* stream)
      : stream_(static_cast<cudaStream_t>(stream)) {
    if (thermal_count < 1 || magnetic_count < 1 || thermal_count > (1 << 21) ||
        magnetic_count > (1 << 21) || stream == nullptr)
      throw std::runtime_error("Invalid lattice prefix dimensions/stream");
    parameters_.components_ = 2;
    parameters_.shape_ = {thermal_count, magnetic_count};
  }
  ~Workspace() {
#if VMECPP_ENERGY_COORDINATE_MAP
    if (last_coordinate_workspace == this) last_coordinate_workspace = nullptr;
#endif
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
    const int count = p.shape_.Count();
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
    p.statistics_ = Allocate<unsigned long long>(8);
    Check(cudaMemsetAsync(p.statistics_, 0, 8 * sizeof(unsigned long long),
                          stream_));
    RaggedInitializeKeys<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                           stream_>>>(p);
    Check(cudaGetLastError());
    std::size_t guesses = 0;
    Check(cub::DeviceScan::InclusiveScanByKey(
        nullptr, guesses, p.component_keys_, p.guess_, p.guess_, cub::Sum(),
        count, cub::Equality(), stream_));
    std::size_t maps = 0;
    Check(ScanEnergyCoordinateMaps(nullptr, maps, p.binade_keys_, p.maps_, p.scanned_, count, stream_));
    temporary_bytes_ = std::max(guesses, maps);
    temporary_ = Allocate<unsigned char>(temporary_bytes_);
    Check(cudaStreamSynchronize(stream_));
  }
  void Run(const double* terms, double* m_output, bool corrupt = false,
           bool unused_fallback = false, double* m_sums = nullptr) {
#if VMECPP_ENERGY_COORDINATE_MAP
    last_coordinate_workspace = this;
#endif
    auto p = parameters_;
    p.terms_ = terms;
    p.output_ = m_output;
    p.sums_ = m_sums;
    const int count = p.shape_.Count();
    const int blocks = (count + kThreads - 1) / kThreads;
    RaggedReset<<<(p.components_ * kBins + kThreads - 1) / kThreads, kThreads,
                  0, stream_>>>(p);
    Check(cudaGetLastError());
    std::size_t temporary = temporary_bytes_;
    Check(cub::DeviceScan::InclusiveScanByKey(
        temporary_, temporary, p.component_keys_, terms, p.guess_, cub::Sum(),
        count, cub::Equality(), stream_));
    RaggedPrepareMaps<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    temporary = temporary_bytes_;
    Check(ScanEnergyCoordinateMaps(temporary_, temporary, p.binade_keys_, p.maps_, p.scanned_, count, stream_));
#if defined(VMECPP_LATTICE_WARP_SEEDS)
    RaggedSeedSegments<<<p.components_, 32, 0, stream_>>>(p);
#else
    RaggedSeedSegments<<<1, 32, 0, stream_>>>(p);
#endif
    Check(cudaGetLastError());
    RaggedAssemble<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    if (corrupt) {
      RaggedCorruptProposal<<<1, 32, 0, stream_>>>(p);
      Check(cudaGetLastError());
    }
    RaggedVerify<<<blocks, kThreads, 0, stream_>>>(p);
    Check(cudaGetLastError());
    RaggedRecordStatistics<<<1, 32, 0, stream_>>>(p);
    Check(cudaGetLastError());
  }
};
}  // namespace

extern "C" void* CreateGpuRaggedEnergyPrefix(int thermal_count,
                                             int magnetic_count, void* stream) {
  auto* result = new Workspace(thermal_count, magnetic_count, stream);
  try {
    result->Initialize();
  } catch (...) {
    delete result;
    throw;
  }
  return result;
}
extern "C" void DestroyGpuRaggedEnergyPrefix(void* m_context) {
  delete static_cast<Workspace*>(m_context);
}
extern "C" void ProposeGpuRaggedEnergyPrefix(void* m_context,
                                             const double* terms,
                                             double* m_prefix, bool corrupt) {
  static_cast<Workspace*>(m_context)->Run(terms, m_prefix, corrupt);
}
extern "C" const int* GpuRaggedEnergyPrefixInvalidDevice(void* context) {
  return static_cast<Workspace*>(context)->parameters_.invalid_;
}
extern "C" void RebindGpuRaggedEnergyPrefix(void* m_context, void* stream) {
  if (stream == nullptr) throw std::runtime_error("Null ragged lattice stream");
  static_cast<Workspace*>(m_context)->stream_ =
      static_cast<cudaStream_t>(stream);
}
extern "C" void ReadGpuRaggedEnergyPrefixStats(
    void* context, std::array<unsigned long long, 8>& m_stats) {
  const auto* value = static_cast<Workspace*>(context);
  Check(cudaMemcpyAsync(m_stats.data(), value->parameters_.statistics_,
                        sizeof(m_stats), cudaMemcpyDeviceToHost,
                        value->stream_));
  Check(cudaStreamSynchronize(value->stream_));
}

extern "C" void ReadGpuRaggedEnergyPrefixRetiredStats(
    void* context, std::array<unsigned long long, 8>& m_stats) {
  // Only after graph/stream retirement, while the context's allocations live.
  // The stream belongs to Solver and can be destroyed before this TLS context.
  const auto* value = static_cast<Workspace*>(context);
  Check(cudaMemcpy(m_stats.data(), value->parameters_.statistics_,
                   sizeof(m_stats), cudaMemcpyDeviceToHost));
}

#endif

#if VMECPP_ENERGY_COORDINATE_MAP
// Actual GPU call counters, last TLS shape, caller retires all graph writers.
extern "C" void ReadGpuEnergyCoordinateMapActivity(
    std::array<unsigned long long, 2>& m_counts) {
  m_counts = {};
  if (last_coordinate_workspace == nullptr) return;
  std::array<unsigned long long, 8> statistics{};
  Check(cudaMemcpy(statistics.data(),
      last_coordinate_workspace->parameters_.statistics_, sizeof(statistics),
      cudaMemcpyDeviceToHost));
  m_counts = {statistics[0], statistics[4]};
}
#endif
