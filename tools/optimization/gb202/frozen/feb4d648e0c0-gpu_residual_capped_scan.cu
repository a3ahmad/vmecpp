#ifndef VMECPP_RESIDUAL_SEED_RAW_GUARD
#define VMECPP_RESIDUAL_SEED_RAW_GUARD 0
#endif
#ifndef VMECPP_RESIDUAL_COORDINATE_TWO_PASS
#define VMECPP_RESIDUAL_COORDINATE_TWO_PASS 0
#endif
#ifndef VMECPP_RESIDUAL_PINNED_256X3
#define VMECPP_RESIDUAL_PINNED_256X3 1
#endif
#ifndef VMECPP_RESIDUAL_CAPPED_SCAN
#define VMECPP_RESIDUAL_CAPPED_SCAN 0
#endif
#if VMECPP_RESIDUAL_CAPPED_SCAN
// SPDX-License-Identifier: MIT
// Private per-slot residual final certificate. Source-only, UNBUILT.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#ifndef VMECPP_RESIDUAL_APPROX_FUSION
#define VMECPP_RESIDUAL_APPROX_FUSION 0
#endif
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
#include <cub/block/block_scan.cuh>
#endif
#include <cuda/std/array>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "coordinate_map.h"
#include "energy_final_map.h"
#ifndef VMECPP_RESIDUAL_COORDINATE_MAP
#define VMECPP_RESIDUAL_COORDINATE_MAP 0
#endif
#include "gpu_residual_capped_scan.h"

namespace {
// Private capped-map CUB pipeline. No generic prefix API is changed.
constexpr int kThreads = 128;
constexpr int kBins = 2048;
constexpr int kBitmapWords = kBins / 32;
enum Failure : int {
  kTermDomain = 1,
  kGuess = 2,
  kSeed = 4,
  kRange = 8,
  kMetadata = 16,
  kForced = 32
};
#if VMECPP_RESIDUAL_COORDINATE_MAP
using residual_coordinate_map::Map;
#else
using energy_final_map::Map;
#endif
struct Compose {
  __host__ __device__ Map operator()(const Map& earlier,
                                     const Map& later) const {
#if VMECPP_RESIDUAL_COORDINATE_MAP
    return residual_coordinate_map::Compose(earlier, later);
#else
    return energy_final_map::Compose(earlier, later);
#endif
  }
};
// Private pinned CUB policy: only block threads/items differ per architecture.
static_assert(VMECPP_RESIDUAL_PINNED_256X3 == 0 ||
              VMECPP_RESIDUAL_PINNED_256X3 == 1);
#if VMECPP_RESIDUAL_PINNED_256X3
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

cudaError_t ScanCappedResidualMaps(void* m_temporary, std::size_t& m_bytes,
                                   int* keys, Map* maps, Map* m_scanned,
                                   int count, cudaStream_t stream) {
#if VMECPP_RESIDUAL_PINNED_256X3
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

#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
static_assert(VMECPP_RESIDUAL_COORDINATE_MAP == 1);
constexpr int kCoordinateThreads = 256;
constexpr int kCoordinateItems = 3;
constexpr int kCoordinateTile = kCoordinateThreads * kCoordinateItems;
constexpr std::uint32_t kCoordinateHead = std::uint32_t{1} << 18;
constexpr std::uint32_t kCoordinateMapMask = kCoordinateHead - 1;

// A right aggregate containing any global head discards the left prefix.
// OR retains whether the local prefix already contains a segment head.
struct ComposeCoordinateNode {
  __host__ __device__ Map operator()(const Map& earlier,
                                     const Map& later) const {
    const Map left{earlier.bits_ & kCoordinateMapMask};
    const Map right{later.bits_ & kCoordinateMapMask};
    auto result = (later.bits_ & kCoordinateHead) != 0
                      ? right
                      : residual_coordinate_map::Compose(left, right);
    result.bits_ |= (earlier.bits_ | later.bits_) & kCoordinateHead;
    return result;
  }
};
#endif

struct Parameters {
  const double* terms_ = nullptr;
  const double* guess_ = nullptr;
  double* sum_ = nullptr;
  Map* maps_ = nullptr;
  Map* scanned_ = nullptr;
  int* binade_keys_ = nullptr;
  int* starts_ = nullptr;
  int* ends_ = nullptr;
  unsigned int* bitmap_ = nullptr;
  int* invalid_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  int count_ = 0;
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
  Map* tile_prefix_ = nullptr;
  int global_offset_ = 0;
#endif
};
using Views = cuda::std::array<Parameters, 3>;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__device__ int Exponent(double number) {
  return (static_cast<std::uint64_t>(__double_as_longlong(number)) >> 52) &
         2047;
}

__global__ void InitializeKeys(int* m_keys, int count, int stride) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) m_keys[index] = index / stride;
}

__global__ void Reset(Views m_views, bool forced) {
  const auto m_p = m_views[blockIdx.y];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kBins) {
    m_p.starts_[index] = INT_MAX;
    m_p.ends_[index] = -1;
  }
  if (index < kBitmapWords) m_p.bitmap_[index] = 0;
  if (index == 0) *m_p.invalid_ = forced ? kForced : 0;
}

__global__ void PrepareCappedMaps(Views m_views) {
  const int component = blockIdx.y;
  const auto m_p = m_views[component];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.count_) return;
  const double term = m_p.terms_[index];
  const double guess = m_p.guess_[index];
  const int field = Exponent(guess);
  const int previous = index == 0 ? -1 : Exponent(m_p.guess_[index - 1]);
  const int next =
      index + 1 == m_p.count_ ? -1 : Exponent(m_p.guess_[index + 1]);
  int invalid = 0;
#if VMECPP_RESIDUAL_COORDINATE_MAP
  // Proof requires TOTAL component length, not the current binade length.
  if (m_p.count_ > residual_coordinate_map::kMaximumCount) invalid |= kMetadata;
#endif
  if (!isfinite(term) || term < 0.0) invalid |= kTermDomain;
  if (!isfinite(guess) || guess < 0.0 || field > 2046 || field < previous ||
      (field == 0 && guess != 0.0))
    invalid |= kGuess;
  // Every term is validated, including the identity-map global run head.
  if (field == 0 && term != 0.0) invalid |= kTermDomain;
  m_p.binade_keys_[index] = component * kBins + field;
  Map mapping{};
  if (field <= 2046) {
    if (field != previous) {
      atomicMin(m_p.starts_ + field, index);
      atomicOr(m_p.bitmap_ + field / 32, 1U << (field % 32));
    } else if (field != 0) {
      bool valid = true;
      const auto original_map = energy_final_map::TermMap(
          static_cast<std::uint64_t>(__double_as_longlong(term)), field, valid);
      if (!valid) invalid |= kTermDomain;
#if VMECPP_RESIDUAL_COORDINATE_MAP
      // Never modulo-pack a saturated failure sentinel into a valid identity.
      if (original_map.even_ >= energy_final_map::kCap ||
          original_map.odd_ >= energy_final_map::kCap) {
        invalid |= kRange;
      } else {
        const auto previous_raw = static_cast<std::uint64_t>(
            __double_as_longlong(m_p.guess_[index - 1]));
        const auto current_raw =
            static_cast<std::uint64_t>(__double_as_longlong(guess));
        mapping = residual_coordinate_map::Translate(
            original_map.even_, original_map.odd_,
            energy_final_map::kLower +
                (previous_raw & energy_final_map::kFractionMask),
            energy_final_map::kLower +
                (current_raw & energy_final_map::kFractionMask));
      }
#else
      mapping = original_map;
#endif
    }
    if (field != next) atomicMax(m_p.ends_ + field, index);
  }
  m_p.maps_[index] = mapping;
  if (invalid != 0) atomicOr(m_p.invalid_, invalid);
}

#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
__device__ Map PrepareCoordinateNode(Parameters m_p, int index) {
  const double term = m_p.terms_[index];
  const double guess = m_p.guess_[index];
  const int field = Exponent(guess);
  const int previous = index == 0 ? -1 : Exponent(m_p.guess_[index - 1]);
  const int next =
      index + 1 == m_p.count_ ? -1 : Exponent(m_p.guess_[index + 1]);
  int invalid = 0;
#if VMECPP_RESIDUAL_COORDINATE_MAP
  // Proof requires TOTAL component length, not the current binade length.
  if (m_p.count_ > residual_coordinate_map::kMaximumCount) invalid |= kMetadata;
#endif
  if (!isfinite(term) || term < 0.0) invalid |= kTermDomain;
  if (!isfinite(guess) || guess < 0.0 || field > 2046 || field < previous ||
      (field == 0 && guess != 0.0))
    invalid |= kGuess;
  // Every term is validated, including the identity-map global run head.
  if (field == 0 && term != 0.0) invalid |= kTermDomain;
  Map mapping{};
  if (field <= 2046) {
    if (field != previous) {
      atomicMin(m_p.starts_ + field, index);
      atomicOr(m_p.bitmap_ + field / 32, 1U << (field % 32));
    } else if (field != 0) {
      bool valid = true;
      const auto original_map = energy_final_map::TermMap(
          static_cast<std::uint64_t>(__double_as_longlong(term)), field, valid);
      if (!valid) invalid |= kTermDomain;
#if VMECPP_RESIDUAL_COORDINATE_MAP
      // Never modulo-pack a saturated failure sentinel into a valid identity.
      if (original_map.even_ >= energy_final_map::kCap ||
          original_map.odd_ >= energy_final_map::kCap) {
        invalid |= kRange;
      } else {
        const auto previous_raw = static_cast<std::uint64_t>(
            __double_as_longlong(m_p.guess_[index - 1]));
        const auto current_raw =
            static_cast<std::uint64_t>(__double_as_longlong(guess));
        mapping = residual_coordinate_map::Translate(
            original_map.even_, original_map.odd_,
            energy_final_map::kLower +
                (previous_raw & energy_final_map::kFractionMask),
            energy_final_map::kLower +
                (current_raw & energy_final_map::kFractionMask));
      }
#else
      mapping = original_map;
#endif
    }
    if (field != next) atomicMax(m_p.ends_ + field, index);
  }
  if (invalid != 0) atomicOr(m_p.invalid_, invalid);
  if (field != previous) mapping.bits_ |= kCoordinateHead;
  return mapping;
}

__global__ void LocalCoordinateScan(Views m_views, int stride,
                                    Map* m_tile_aggregates) {
  using Scan = cub::BlockScan<Map, kCoordinateThreads>;
  __shared__ typename Scan::TempStorage storage;
  const ComposeCoordinateNode compose{};
  cuda::std::array<Map, kCoordinateItems> items{};
  Map thread_aggregate{};
  const int base =
      blockIdx.x * kCoordinateTile + threadIdx.x * kCoordinateItems;
  for (int item = 0; item < kCoordinateItems; ++item) {
    const int global = base + item;
    if (global < 3 * stride) {
      const int component = global / stride;
      items[item] = PrepareCoordinateNode(m_views[component], global % stride);
    }
    thread_aggregate = compose(thread_aggregate, items[item]);
  }
  Map prefix{};
  Map block_aggregate{};
  Scan(storage).ExclusiveScan(thread_aggregate, prefix, Map{}, compose,
                              block_aggregate);
  if (threadIdx.x == 0) m_tile_aggregates[blockIdx.x] = block_aggregate;
  for (int item = 0; item < kCoordinateItems; ++item) {
    prefix = compose(prefix, items[item]);
    const int global = base + item;
    if (global < 3 * stride) {
      m_views[global / stride].scanned_[global % stride] = prefix;
    }
  }
}

__global__ void ScanCoordinateSummaries(const Map* aggregates,
                                        Map* m_tile_prefix, int count) {
  using Scan = cub::BlockScan<Map, 128>;
  __shared__ typename Scan::TempStorage storage;
  Map input{};
  if (threadIdx.x < count) input = aggregates[threadIdx.x];
  Map prefix{};
  Scan(storage).InclusiveScan(input, prefix, ComposeCoordinateNode{});
  if (threadIdx.x < count) m_tile_prefix[threadIdx.x] = prefix;
}

#endif

__device__ void CountDecision(Parameters m_p, int invalid,
                              unsigned long long segments) {
  const auto count = [&](ResidualCappedScanStatistic field,
                         unsigned long long number) {
    atomicAdd(m_p.statistics_ + static_cast<int>(field), number);
  };
  count(ResidualCappedScanStatistic::kCalls, 1);
  count(invalid ? ResidualCappedScanStatistic::kDeclined
                : ResidualCappedScanStatistic::kAccepted,
        1);
  count(ResidualCappedScanStatistic::kVisitedSegments, segments);
  count(ResidualCappedScanStatistic::kScannedMaps, m_p.count_);
  if (invalid & kTermDomain)
    count(ResidualCappedScanStatistic::kTermDomainFailures, 1);
  if (invalid & kGuess) count(ResidualCappedScanStatistic::kGuessFailures, 1);
  if (invalid & kSeed) count(ResidualCappedScanStatistic::kSeedFailures, 1);
  if (invalid & kRange) count(ResidualCappedScanStatistic::kRangeFailures, 1);
  if (invalid & kMetadata)
    count(ResidualCappedScanStatistic::kMetadataFailures, 1);
  if (invalid & kForced) count(ResidualCappedScanStatistic::kForcedDeclines, 1);
}

// Warp metadata preloads, but lane zero retains the complete original RN chain.
// All lanes execute every shuffle. A rejected component reads no metadata/tail.
__global__ void FoldCappedTails(Views m_views) {
  const auto m_p = m_views[blockIdx.x];
  const int lane = threadIdx.x;
  int invalid = *m_p.invalid_;
  if (invalid != 0) {
    if (lane == 0) CountDecision(m_p, invalid, 0);
    return;
  }
  const unsigned int low = m_p.bitmap_[lane];
  const unsigned int high = m_p.bitmap_[lane + 32];
  const unsigned int low_mask = __ballot_sync(0xffffffffU, low != 0);
  const unsigned int high_mask = __ballot_sync(0xffffffffU, high != 0);
  double previous = 0.0;
  int expected_start = 0;
  unsigned long long segments = 0;
  for (int half = 0; half < 2; ++half) {
    unsigned int words = half == 0 ? low_mask : high_mask;
    while (words != 0) {
      const int owner = __ffs(words) - 1;
      words &= words - 1;
      const int word = half * 32 + owner;
      unsigned int occupied =
          __shfl_sync(0xffffffffU, half == 0 ? low : high, owner);
      const int own_field = word * 32 + lane;
      const bool present = (occupied & (1U << lane)) != 0;
      int start = -1;
      int end = -1;
      double term = 0.0;
      Map map{};
#if VMECPP_RESIDUAL_COORDINATE_MAP
      std::uint32_t start_low = 0;
      std::uint64_t end_mantissa = 0;
#endif
      if (present && own_field <= 2046) {
        start = m_p.starts_[own_field];
        end = m_p.ends_[own_field];
        if (start >= 0 && start <= end && end < m_p.count_ && own_field != 0) {
          term = m_p.terms_[start];
          map = m_p.scanned_[end];
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
          if (m_p.count_ <= residual_coordinate_map::kMaximumCount) {
            const int tile = (m_p.global_offset_ + end) / kCoordinateTile;
            if ((map.bits_ & kCoordinateHead) == 0 && tile > 0) {
              map = ComposeCoordinateNode{}(m_p.tile_prefix_[tile - 1], map);
            }
            map.bits_ &= kCoordinateMapMask;
          }
#endif
#if VMECPP_RESIDUAL_COORDINATE_MAP
          const auto start_raw = static_cast<std::uint64_t>(
              __double_as_longlong(m_p.guess_[start]));
          const auto end_raw =
              static_cast<std::uint64_t>(__double_as_longlong(m_p.guess_[end]));
          start_low = start_raw & residual_coordinate_map::kMask;
          end_mantissa = energy_final_map::kLower +
                         (end_raw & energy_final_map::kFractionMask);
#endif
        }
      }
      while (occupied != 0) {
        const int bit = __ffs(occupied) - 1;
        occupied &= occupied - 1;
        const int field = word * 32 + bit;
        const int selected_start = __shfl_sync(0xffffffffU, start, bit);
        const int selected_end = __shfl_sync(0xffffffffU, end, bit);
        const double selected_term = __shfl_sync(0xffffffffU, term, bit);
#if VMECPP_RESIDUAL_COORDINATE_MAP
        Map selected_map{};
        selected_map.bits_ = __shfl_sync(0xffffffffU, map.bits_, bit);
        const auto selected_start_low =
            __shfl_sync(0xffffffffU, start_low, bit);
        const auto selected_end_mantissa =
            __shfl_sync(0xffffffffU, end_mantissa, bit);
#else
        const auto even = __shfl_sync(0xffffffffU, map.even_, bit);
        const auto odd = __shfl_sync(0xffffffffU, map.odd_, bit);
        const Map selected_map{even, odd};
#endif
        if (lane != 0 || invalid != 0) continue;
        ++segments;
        if (field > 2046 || selected_start != expected_start ||
            selected_start < 0 || selected_end < selected_start ||
            selected_end >= m_p.count_) {
          invalid |= kMetadata;
          continue;
        }
        expected_start = selected_end + 1;
        if (field == 0) {
          if (previous != 0.0) invalid |= kMetadata;
          previous = 0.0;
          continue;
        }
        const double seed = __dadd_rn(previous, selected_term);
#if VMECPP_RESIDUAL_SEED_RAW_GUARD
        const auto raw = static_cast<std::uint64_t>(__double_as_longlong(seed));
        // Dominating guards reject field>2046; field0 already continued.
        // Full sign+exponent bits must equal the positive normal field.
        if ((raw >> 52) != static_cast<std::uint64_t>(field)) {
#else
        if (!isfinite(seed) || seed <= 0.0 || Exponent(seed) != field) {
#endif
          invalid |= kSeed;
          continue;
        }
#if !VMECPP_RESIDUAL_SEED_RAW_GUARD
        const auto raw = static_cast<std::uint64_t>(__double_as_longlong(seed));
#endif
        auto mantissa =
            energy_final_map::kLower + (raw & energy_final_map::kFractionMask);
#if VMECPP_RESIDUAL_COORDINATE_MAP
        const auto initial =
            (static_cast<std::uint32_t>(mantissa) - selected_start_low) &
            residual_coordinate_map::kMask;
        const auto ending = (initial + residual_coordinate_map::Cost(
                                           selected_map, initial & 1)) &
                            residual_coordinate_map::kMask;
        const std::int64_t error =
            ending < 32768 ? ending : static_cast<std::int64_t>(ending) - 65536;
        const auto reconstructed =
            static_cast<std::int64_t>(selected_end_mantissa) + error;
        if (reconstructed <
                static_cast<std::int64_t>(energy_final_map::kLower) ||
            reconstructed >=
                static_cast<std::int64_t>(energy_final_map::kCap)) {
          invalid |= kRange;
          continue;
        }
        mantissa = static_cast<std::uint64_t>(reconstructed);
#else
        if (!energy_final_map::Apply(selected_map, mantissa)) {
          invalid |= kRange;
          continue;
        }
#endif
        const auto final_bits = (static_cast<std::uint64_t>(field) << 52) |
                                (mantissa - energy_final_map::kLower);
        previous = __longlong_as_double(static_cast<long long>(final_bits));
      }
    }
  }
  if (lane == 0) {
    if (invalid == 0 && expected_start != m_p.count_) invalid |= kMetadata;
    *m_p.invalid_ = invalid;
    if (invalid == 0) *m_p.sum_ = previous;
    CountDecision(m_p, invalid, segments);
  }
}

struct Allocation {
  void* pointer_;
  std::size_t bytes_;
};

// Failed constructor retirement retains ALL still-owned device buffers until
// process exit. No healthy context is published and no fallback/retry is made.
std::vector<std::vector<Allocation>>& Quarantine() {
  static thread_local auto* retained = new std::vector<std::vector<Allocation>>;
  return *retained;
}

struct ResidualTerms {
  cuda::std::array<const double*, 6> forces_{};
  double* terms_ = nullptr;
  int stride_ = 0;
  int rz_count_ = 0;
};

__global__ void PrepareResidualTerms(ResidualTerms m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 3 * m_p.stride_) return;
  const int component = index / m_p.stride_;
  const int local = index % m_p.stride_;
  const int coefficient = local / 2;
  double term = 0.0;
  if (component == 2 || coefficient < m_p.rz_count_) {
    const double force = m_p.forces_[2 * component + local % 2][coefficient];
    term = __dmul_rn(force, force);
  }
  m_p.terms_[index] = term;
}

#include "approximation_kernels.inc"

__global__ void RecordFallbacks(Views m_views) {
  const auto p = m_views[threadIdx.x];
  if (*p.invalid_ != 0) {
    atomicAdd(
        p.statistics_ +
            static_cast<int>(
                ResidualCappedScanStatistic::kOriginalConditionalFallbacks),
        1ULL);
  }
}

class ResidualCappedScanContext {
 public:
  Views views_{};
  cudaStream_t stream_ = nullptr;
  std::thread::id owner_ = std::this_thread::get_id();
  std::array<int, 5> dimensions_{};
  int device_ = -1;
  int stride_ = 0;
  int blocks_ = 0;
#if VMECPP_RESIDUAL_APPROX_FUSION
  int approx_tiles_ = 0;
  double* approx_summaries_ = nullptr;
  double* approx_carries_ = nullptr;
#endif
  int* keys_ = nullptr;
  int* binade_keys_ = nullptr;
  Map* maps_ = nullptr;
  Map* scanned_ = nullptr;
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
  Map* tile_aggregates_ = nullptr;
  Map* tile_prefix_ = nullptr;
#endif
  int* invalid_ = nullptr;
  double* terms_ = nullptr;
  double* guesses_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  void* temporary_ = nullptr;
  std::size_t temporary_bytes_ = 0;
  std::vector<Allocation> allocations_;
  bool poisoned_ = false;

  explicit ResidualCappedScanContext(const GpuIterationDeviceState& state)
      : stream_(static_cast<cudaStream_t>(state.stream_)),
        dimensions_(state.dimensions_) {
    const auto& d = dimensions_;
    if (stream_ == nullptr || d[0] < 2 || d[1] < 1 || d[2] < 0 || d[4] < 1 ||
        d[4] > d[0] || d[1] > 65536 || d[2] >= 65536 || d[4] > 65536) {
      throw std::runtime_error("Invalid residual-final shape/stream");
    }
    const auto coefficients =
        std::size_t(d[4]) * d[1] * (std::size_t(d[2]) + 1);
    if (coefficients > (1 << 21))
      throw std::runtime_error("Residual-final extent exceeds bound");
    stride_ = static_cast<int>(2 * coefficients);
    blocks_ = (stride_ + kThreads - 1) / kThreads;
    Check(cudaGetDevice(&device_));
    cudaStreamCaptureStatus capture;
    Check(cudaStreamIsCapturing(stream_, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::runtime_error("Create residual-final context outside capture");
    try {
      terms_ = Allocate<double>(3 * std::size_t(stride_));
      guesses_ = Allocate<double>(3 * std::size_t(stride_));
#if VMECPP_RESIDUAL_APPROX_FUSION
      if (stride_ <= residual_coordinate_map::kMaximumCount) {
        approx_tiles_ = (stride_ + kApproxTile - 1) / kApproxTile;
        if (approx_tiles_ > kApproxThreads)
          throw std::runtime_error("Approximation summary exceeds fixed CTA");
        approx_summaries_ = Allocate<double>(3 * std::size_t(approx_tiles_));
        approx_carries_ = Allocate<double>(3 * std::size_t(approx_tiles_));
      }
#endif
      keys_ = Allocate<int>(3 * std::size_t(stride_));
      binade_keys_ = Allocate<int>(3 * std::size_t(stride_));
      maps_ = Allocate<Map>(3 * std::size_t(stride_));
      scanned_ = Allocate<Map>(3 * std::size_t(stride_));
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
      tile_aggregates_ = Allocate<Map>(128);
      tile_prefix_ = Allocate<Map>(128);
#endif
      invalid_ = Allocate<int>(3);
      statistics_ =
          Allocate<unsigned long long>(3 * kResidualCappedScanStatistics);
      Check(cudaMemsetAsync(
          statistics_, 0,
          3 * kResidualCappedScanStatistics * sizeof(unsigned long long),
          stream_));
      for (int component = 0; component < 3; ++component) {
        auto& p = views_[component];
        p.count_ = stride_;
        p.terms_ = terms_ + component * stride_;
        p.guess_ = guesses_ + component * stride_;
        p.maps_ = maps_ + component * stride_;
        p.scanned_ = scanned_ + component * stride_;
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
        p.tile_prefix_ = tile_prefix_;
        p.global_offset_ = component * stride_;
#endif
        p.binade_keys_ = binade_keys_ + component * stride_;
        p.starts_ = Allocate<int>(kBins);
        p.ends_ = Allocate<int>(kBins);
        p.bitmap_ = Allocate<unsigned int>(kBitmapWords);
        p.invalid_ = invalid_ + component;
        p.statistics_ = statistics_ + component * kResidualCappedScanStatistics;
      }
      InitializeKeys<<<(3 * stride_ + kThreads - 1) / kThreads, kThreads, 0,
                       stream_>>>(keys_, 3 * stride_, stride_);
      Check(cudaGetLastError());
      Check(cub::DeviceScan::InclusiveScanByKey(
          nullptr, temporary_bytes_, keys_, terms_, guesses_, cub::Sum(),
          3 * stride_, cub::Equality(), stream_));
      std::size_t map_bytes = 0;
      Check(ScanCappedResidualMaps(nullptr, map_bytes, binade_keys_, maps_,
                                   scanned_, 3 * stride_, stream_));
      temporary_bytes_ = std::max(temporary_bytes_, map_bytes);
      temporary_ = Allocate<unsigned char>(temporary_bytes_);
      Check(cudaStreamSynchronize(stream_));
    } catch (...) {
      if (cudaStreamSynchronize(stream_) == cudaSuccess) {
        FreeRetired();
      }
      RetainRemaining();
      throw;
    }
  }

  template <class T>
  T* Allocate(std::size_t count) {
    T* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(T)));
    try {
      allocations_.push_back({pointer, count * sizeof(T)});
    } catch (...) {
      cudaFree(pointer);
      throw;
    }
    return pointer;
  }

  void RetainRemaining() {
    if (!allocations_.empty()) Quarantine().push_back(std::move(allocations_));
  }

  bool FreeRetired() noexcept {
    bool all_freed = true;
    for (auto& allocation : allocations_) {
      if (allocation.pointer_ == nullptr) continue;
      if (cudaFree(allocation.pointer_) == cudaSuccess) {
        allocation.pointer_ = nullptr;
      } else {
        all_freed = false;
      }
    }
    if (all_freed) allocations_.clear();
    return all_freed;
  }

  void CheckOwner() const {
    if (std::this_thread::get_id() != owner_)
      throw std::runtime_error("Foreign residual-final owner thread");
    int device;
    Check(cudaGetDevice(&device));
    if (device != device_)
      throw std::runtime_error("Residual-final CUDA device changed");
    if (poisoned_)
      throw std::runtime_error("Residual-final context is poisoned");
  }

  void CheckState(const GpuIterationDeviceState& state) const {
    CheckOwner();
    if (state.dimensions_ != dimensions_ || state.stream_ != stream_ ||
        state.residuals_ == nullptr)
      throw std::runtime_error("Residual-final shape/stream/output changed");
    const auto destination = reinterpret_cast<std::uintptr_t>(state.residuals_);
    const auto overlaps = [&](const void* pointer, std::size_t bytes) {
      const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
      return destination <= begin ? begin - destination < 3 * sizeof(double)
                                  : destination - begin < bytes;
    };
    for (const auto* force : state.forces_) {
      if (force == nullptr ||
          overlaps(force, std::size_t(stride_ / 2) * sizeof(double)))
        throw std::runtime_error("Residual-final force/output overlap or null");
    }
    for (const auto& allocation : allocations_) {
      if (overlaps(allocation.pointer_, allocation.bytes_))
        throw std::runtime_error("Residual-final output overlaps scratch");
    }
  }

  void Launch(GpuIterationDeviceState& m_state, bool include_edge_rz,
              bool force_invalid) {
    CheckState(m_state);
    const auto count = include_edge_rz
                           ? std::size_t(stride_ / 2)
                           : std::size_t(dimensions_[0] - 1) * dimensions_[1] *
                                 (std::size_t(dimensions_[2]) + 1);
    if (count > std::size_t(stride_ / 2))
      throw std::runtime_error("Residual-final R/Z active extent changed");
    ResidualTerms terms{};
    for (int i = 0; i < 6; ++i) terms.forces_[i] = m_state.forces_[i];
    terms.terms_ = terms_;
    terms.stride_ = stride_;
    terms.rz_count_ = static_cast<int>(count);
    auto views = views_;
    for (int i = 0; i < 3; ++i) views[i].sum_ = m_state.residuals_ + i;
    try {
      std::size_t bytes = temporary_bytes_;
#if VMECPP_RESIDUAL_APPROX_FUSION
      if (stride_ <= residual_coordinate_map::kMaximumCount) {
        Approximation approx{approx_summaries_, approx_carries_, guesses_, approx_tiles_};
        FusedTermsApproximation<<<dim3(approx_tiles_, 3), kApproxThreads, 0,
                                  stream_>>>(views, terms, approx, force_invalid);
        Check(cudaGetLastError());
        ScanApproximationSummaries<<<3, kApproxThreads, 0, stream_>>>(approx);
        Check(cudaGetLastError());
        FixupApproximation<<<dim3(blocks_, 3), kThreads, 0, stream_>>>(views, approx);
        Check(cudaGetLastError());
      } else {
#endif
      Reset<<<dim3(kBins / kThreads, 3), kThreads, 0, stream_>>>(views,
                                                                 force_invalid);
      Check(cudaGetLastError());
      PrepareResidualTerms<<<(3 * stride_ + kThreads - 1) / kThreads, kThreads,
                             0, stream_>>>(terms);
      Check(cudaGetLastError());
      Check(cub::DeviceScan::InclusiveScanByKey(
          temporary_, bytes, keys_, terms_, guesses_, cub::Sum(), 3 * stride_,
          cub::Equality(), stream_));
#if VMECPP_RESIDUAL_APPROX_FUSION
      }
#endif
#if VMECPP_RESIDUAL_COORDINATE_TWO_PASS
      if (stride_ <= residual_coordinate_map::kMaximumCount) {
        const int tiles = (3 * stride_ + kCoordinateTile - 1) / kCoordinateTile;
        LocalCoordinateScan<<<tiles, kCoordinateThreads, 0, stream_>>>(
            views, stride_, tile_aggregates_);
        Check(cudaGetLastError());
        ScanCoordinateSummaries<<<1, 128, 0, stream_>>>(tile_aggregates_,
                                                        tile_prefix_, tiles);
        Check(cudaGetLastError());
      } else {
        PrepareCappedMaps<<<dim3(blocks_, 3), kThreads, 0, stream_>>>(views);
        Check(cudaGetLastError());
        bytes = temporary_bytes_;
        Check(ScanCappedResidualMaps(temporary_, bytes, binade_keys_, maps_,
                                     scanned_, 3 * stride_, stream_));
      }
#else
      PrepareCappedMaps<<<dim3(blocks_, 3), kThreads, 0, stream_>>>(views);
      Check(cudaGetLastError());
      bytes = temporary_bytes_;
      Check(ScanCappedResidualMaps(temporary_, bytes, binade_keys_, maps_,
                                   scanned_, 3 * stride_, stream_));
#endif
      FoldCappedTails<<<3, 32, 0, stream_>>>(views);
      Check(cudaGetLastError());
      // SAME stream and SAME live force state; fallback precedes the caller's
      // M1/Thomas/Lambda overwrite. Never defer invariant fallback to slot2.
      LaunchGpuIterationResidualsConditional(m_state, include_edge_rz,
                                             invalid_);
      RecordFallbacks<<<1, 3, 0, stream_>>>(views);
      Check(cudaGetLastError());
    } catch (...) {
      poisoned_ = true;
      throw;
    }
  }
};

thread_local std::unordered_set<void*> residual_contexts;

ResidualCappedScanContext& GetContext(void* context) {
  if (!residual_contexts.contains(context))
    throw std::runtime_error("Unknown/foreign residual-final context");
  return *static_cast<ResidualCappedScanContext*>(context);
}
}  // namespace

extern "C" void* CreateGpuResidualCappedScan(
    const GpuIterationDeviceState& state) {
  auto* context = new ResidualCappedScanContext(state);
  try {
    residual_contexts.insert(context);
  } catch (...) {
    if (cudaStreamSynchronize(context->stream_) == cudaSuccess)
      context->FreeRetired();
    context->RetainRemaining();
    delete context;
    throw;
  }
  return context;
}

extern "C" void LaunchGpuResidualCappedScan(void* m_context,
                                            GpuIterationDeviceState& m_state,
                                            bool include_edge_rz,
                                            bool force_invalid) {
  GetContext(m_context).Launch(m_state, include_edge_rz, force_invalid);
}

extern "C" void ReadGpuResidualCappedScanStats(
    void* context, ResidualCappedScanStatistics& m_statistics) {
  auto& owner = GetContext(context);
  owner.CheckOwner();
  static_assert(sizeof(m_statistics) ==
                3 * kResidualCappedScanStatistics * sizeof(unsigned long long));
  Check(cudaMemcpy(m_statistics.data(), owner.statistics_, sizeof(m_statistics),
                   cudaMemcpyDeviceToHost));
}

extern "C" void DestroyGpuResidualCappedScan(void* m_context) {
  auto& owner = GetContext(m_context);
  if (std::this_thread::get_id() != owner.owner_)
    throw std::runtime_error("Foreign residual-final destruction thread");
  int device;
  Check(cudaGetDevice(&device));
  if (device != owner.device_)
    throw std::runtime_error("Foreign residual-final destruction device");
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(owner.stream_, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error(
        "Retire residual-final capture before destruction");
  Check(cudaStreamQuery(owner.stream_));
  // Graph handles must already be retired by the caller; a query cannot prove
  // absence of graph references. No further launches may use this context.
  if (!owner.FreeRetired()) {
    owner.poisoned_ = true;
    throw std::runtime_error(
        "Residual-final destruction retained failed frees");
  }
  residual_contexts.erase(m_context);
  delete &owner;
}

#endif
