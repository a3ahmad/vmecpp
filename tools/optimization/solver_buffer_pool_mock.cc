// SPDX-License-Identifier: MIT
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gpu_solver_buffer_pool.h"

namespace {
std::unordered_map<void*, std::size_t> live;
std::size_t fresh_calls = 0;
std::size_t free_calls = 0;
bool fail_allocate = false;
bool fail_free = false;
void Require(bool value) {
  if (!value) throw std::runtime_error("Buffer pool CPU control failed");
}
void* Allocate(std::size_t bytes) {
  if (fail_allocate) throw std::bad_alloc();
  void* pointer = std::malloc(bytes);
  if (pointer == nullptr) throw std::bad_alloc();
  live.emplace(pointer, bytes);
  ++fresh_calls;
  return pointer;
}
bool Free(void* m_pointer) noexcept {
  if (fail_free) return false;
  if (live.erase(m_pointer) != 1) std::abort();
  std::free(m_pointer);
  ++free_calls;
  return true;
}
constexpr GpuBufferPoolDomain kDomain{0, 123, 456};
constexpr GpuBufferPoolDomain kOtherDevice{1, 123, 456};
constexpr GpuBufferPoolDomain kOtherContext{0, 124, 456};
constexpr GpuBufferPoolDomain kOtherThread{0, 123, 457};
template <typename Function>
void MustThrow(Function function) {
  bool caught = false;
  try {
    function();
  } catch (...) {
    caught = true;
  }
  Require(caught);
}
}  // namespace

int main() {
  {
    GpuSolverBufferPool pool(kDomain, Allocate, Free, 128);
    auto first = pool.Acquire(64, kDomain);
    std::memset(first.pointer_, 0xa5, first.bytes_);
    Require(pool.Recycle(first, kDomain, true));
    auto reused = pool.Acquire(64, kDomain);
    Require(first.pointer_ == reused.pointer_);
    Require(*static_cast<unsigned char*>(reused.pointer_) == 0xa5);
    auto separate = pool.Acquire(64, kDomain);
    Require(separate.pointer_ != reused.pointer_);
    Require(pool.Recycle(reused, kDomain, true));
    Require(pool.Recycle(separate, kDomain, true));
    auto other_size = pool.Acquire(32, kDomain);
    Require(other_size.pointer_ != separate.pointer_);
    Require(pool.Recycle(other_size, kDomain, true));
    Require(pool.CachedBytes() == 128 && pool.CachedCount() == 2);
    Require(pool.Statistics()[0] == 1 && pool.Statistics()[3] == 1);
    Require(pool.Shutdown(kDomain));
    Require(pool.CachedCount() == 0 && live.empty());
  }
  {
    GpuSolverBufferPool pool(kDomain, Allocate, Free, 4096);
    std::vector<GpuBufferLease> active;
    for (std::size_t i = 0; i < GpuSolverBufferPool::kActiveEntries; ++i)
      active.push_back(pool.Acquire(1, kDomain));
    const auto before = fresh_calls;
    MustThrow([&] { pool.Acquire(1, kDomain); });
    Require(fresh_calls == before);
    for (const auto lease : active) Require(pool.Recycle(lease, kDomain, true));
    Require(pool.CachedCount() == GpuSolverBufferPool::kCacheEntries);
    Require(pool.Statistics()[3] == GpuSolverBufferPool::kActiveEntries -
                                        GpuSolverBufferPool::kCacheEntries);
    Require(pool.Shutdown(kDomain) && live.empty());
  }
  {
    GpuSolverBufferPool pool(kDomain, Allocate, Free, 128);
    fail_allocate = true;
    MustThrow([&] { pool.Acquire(64, kDomain); });
    fail_allocate = false;
    Require(pool.ActiveCount() == 0 && pool.CachedCount() == 0);
    auto lease = pool.Acquire(64, kDomain);
    // Model a bookkeeping exception after CUDA allocation but before ownership
    // enters the Solver allocation vector. The catch returns this unused lease.
    try {
      throw std::bad_alloc();
    } catch (...) {
      Require(pool.Recycle(lease, kDomain, true));
    }
    auto second = pool.Acquire(64, kDomain);
    Require(second.pointer_ == lease.pointer_);
    Require(pool.Recycle(second, kDomain, true));
    Require(pool.Shutdown(kDomain) && live.empty());
  }
  for (const auto changed : {kOtherDevice, kOtherContext, kOtherThread}) {
    GpuSolverBufferPool pool(kDomain, Allocate, Free);
    const auto before = fresh_calls;
    MustThrow([&] { pool.Acquire(8, changed); });
    Require(fresh_calls == before);
    auto lease = pool.Acquire(8, kDomain);
    Require(!pool.Recycle(lease, changed, true));
    Require(pool.ActiveCount() == 0 && pool.CachedCount() == 0);
    Require(pool.Statistics()[4] == 1);
    Require(pool.Shutdown(kDomain));
    Require(Free(lease.pointer_));  // Mock explicit context-reset reclamation.
  }
  {
    GpuSolverBufferPool pool(kDomain, Allocate, Free);
    auto failed = pool.Acquire(8, kDomain);
    Require(!pool.Recycle(failed, kDomain, false));
    Require(pool.CachedCount() == 0 && pool.Statistics()[4] == 1);
    Require(!pool.Recycle(failed, kDomain, true));
    Require(pool.Statistics()[5] == 2);
    Require(pool.Shutdown(kDomain));
    Require(Free(failed.pointer_));
  }
  {
    GpuSolverBufferPool pool(kDomain, Allocate, Free, 0);
    auto lease = pool.Acquire(8, kDomain);
    fail_free = true;
    Require(!pool.Recycle(lease, kDomain, true));
    fail_free = false;
    Require(pool.Statistics()[4] == 1 && pool.Statistics()[5] == 1);
    Require(pool.Shutdown(kDomain));
    Require(Free(lease.pointer_));
  }
  {
    auto pool = std::make_shared<GpuSolverBufferPool>(kDomain, Allocate, Free);
    auto context_owner = pool;
    std::weak_ptr<GpuSolverBufferPool> weak = pool;
    auto cached = pool->Acquire(8, kDomain);
    Require(pool->Recycle(cached, kDomain, true));
    auto active = pool->Acquire(16, kDomain);
    const auto before = free_calls;
    Require(!pool->Shutdown(kDomain));
    Require(free_calls ==
            before);  // No cache/active frees before leases drain.
    pool.reset();
    Require(!weak.expired());
    MustThrow([&] { context_owner->Acquire(8, kDomain); });
    Require(context_owner->Recycle(active, kDomain, true));
    Require(live.empty() && free_calls == before + 2);
    context_owner.reset();
    Require(weak.expired());
  }
  Require(live.empty());
  return 0;
}
