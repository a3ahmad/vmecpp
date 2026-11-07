// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_SOLVER_BUFFER_POOL_H_
#define VMECPP_EXPERIMENT_GPU_SOLVER_BUFFER_POOL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Host-only bounded ownership core. CUDA calls and scope checks belong to the
// adapter. A caller returns each lease after its stream and graphs are retired.
struct GpuBufferPoolDomain {
  int device_;
  std::uintptr_t context_;
  std::uintptr_t thread_;
  bool operator==(const GpuBufferPoolDomain&) const = default;
};
struct GpuBufferLease {
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class GpuSolverBufferPool {
 public:
  using Allocate = void* (*)(std::size_t);
  using Free = bool (*)(void*) noexcept;
  static constexpr std::size_t kCacheEntries = 256;
  static constexpr std::size_t kActiveEntries = 512;
  static constexpr std::size_t kDefaultBytes = 128 * 1024 * 1024;

  GpuSolverBufferPool(GpuBufferPoolDomain domain, Allocate allocate, Free free,
                      std::size_t byte_limit = kDefaultBytes)
      : domain_(domain),
        allocate_(allocate),
        free_(free),
        byte_limit_(byte_limit) {}
  GpuSolverBufferPool(const GpuSolverBufferPool&) = delete;
  GpuSolverBufferPool& operator=(const GpuSolverBufferPool&) = delete;

  GpuBufferLease Acquire(std::size_t bytes, GpuBufferPoolDomain domain) {
    if (closed_ || domain != domain_ || bytes == 0 ||
        active_count_ == kActiveEntries)
      throw std::runtime_error("GPU buffer pool scope/size/lifetime violation");
    GpuBufferLease lease;
    for (std::size_t i = 0; i < cached_count_; ++i) {
      if (cached_[i].bytes_ != bytes) continue;
      lease = cached_[i];
      cached_[i] = cached_[--cached_count_];
      cached_bytes_ -= bytes;
      ++statistics_[0];
      break;
    }
    if (lease.pointer_ == nullptr) {
      lease = {allocate_(bytes), bytes};
      if (lease.pointer_ == nullptr)
        throw std::runtime_error("GPU buffer pool allocator returned null");
      ++statistics_[1];
    }
    for (std::size_t i = 0; i < active_count_; ++i) {
      if (active_[i].pointer_ == lease.pointer_)
        throw std::runtime_error(
            "GPU buffer pool allocator aliased an active lease");
    }
    active_[active_count_++] = lease;
    return lease;
  }

  // Failed stream/graph cleanup or a changed domain quarantines the pointer:
  // neither caching nor cudaFree is safe to assume in a failed CUDA context.
  bool Recycle(GpuBufferLease lease, GpuBufferPoolDomain domain,
               bool reusable) noexcept {
    std::size_t index = 0;
    while (index < active_count_ && active_[index].pointer_ != lease.pointer_)
      ++index;
    if (index == active_count_) {
      ++statistics_[5];
      return false;
    }
    const auto owned = active_[index];
    active_[index] = active_[--active_count_];
    bool success =
        reusable && domain == domain_ && owned.bytes_ == lease.bytes_;
    if (!success) {
      ++statistics_[4];
      ++statistics_[5];
    } else if (!closed_ && cached_count_ < kCacheEntries &&
               owned.bytes_ <= byte_limit_ - cached_bytes_) {
      cached_[cached_count_++] = owned;
      cached_bytes_ += owned.bytes_;
      ++statistics_[2];
      if (cached_bytes_ > statistics_[6]) statistics_[6] = cached_bytes_;
    } else {
      ++statistics_[3];
      success = free_(owned.pointer_);
      if (!success) {
        ++statistics_[4];
        ++statistics_[5];
      }
    }
    if (closed_ && active_count_ == 0) success = Drain(domain) && success;
    return success;
  }

  bool Shutdown(GpuBufferPoolDomain domain) noexcept {
    closed_ = true;
    if (active_count_ != 0) {
      ++statistics_[7];
      return false;
    }
    return Drain(domain);
  }
  std::size_t ActiveCount() const { return active_count_; }
  std::size_t CachedCount() const { return cached_count_; }
  std::size_t CachedBytes() const { return cached_bytes_; }
  // hits, fresh, cached returns, evictions, quarantines, cleanup/ownership
  // failures, peak cached bytes, shutdown requested while leases active.
  const std::array<std::uint64_t, 8>& Statistics() const { return statistics_; }

 private:
  bool Drain(GpuBufferPoolDomain domain) noexcept {
    bool success = true;
    while (cached_count_ != 0) {
      const auto lease = cached_[--cached_count_];
      if (domain != domain_ || !free_(lease.pointer_)) {
        ++statistics_[4];
        ++statistics_[5];
        success = false;
      }
    }
    cached_bytes_ = 0;
    return success;
  }
  GpuBufferPoolDomain domain_;
  Allocate allocate_;
  Free free_;
  std::size_t byte_limit_;
  std::array<GpuBufferLease, kCacheEntries> cached_{};
  std::array<GpuBufferLease, kActiveEntries> active_{};
  std::size_t cached_count_ = 0;
  std::size_t cached_bytes_ = 0;
  std::size_t active_count_ = 0;
  std::array<std::uint64_t, 8> statistics_{};
  bool closed_ = false;
};
#endif  // VMECPP_EXPERIMENT_GPU_SOLVER_BUFFER_POOL_H_
