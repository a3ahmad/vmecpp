// SPDX-License-Identifier: MIT
// Optional single-worker diagnostic observer. Never use in timing
// qualification.
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#include "gpu_solver.h"

namespace {
constexpr int kFinalGrid = 99;
constexpr long kDefaultThreshold = 1000;
constexpr long kMaximumThreshold = 1000000;

struct Observer {
  std::mutex mutex_;
  bool enabled_ = false;
  bool refused_ = false;
  bool attempted_ = false;
  bool in_run_ = false;
  long threshold_ = kDefaultThreshold;
  long count_ = 0;
  long generation_ = 0;
  long owner_tid_ = 0;
  int ns_ = 0;
  void* context_ = nullptr;
  std::string marker_;

  Observer() {
    const char* enabled = std::getenv("VMECPP_GPU_PROFILE_STEADY");
    enabled_ = enabled != nullptr && std::string(enabled) == "1";
    if (!enabled_) return;
    if (const char* path = std::getenv("VMECPP_GPU_PROFILE_MARKER"))
      marker_ = path;
    if (const char* threshold =
            std::getenv("VMECPP_GPU_PROFILE_STEADY_COUNT")) {
      errno = 0;
      char* end = nullptr;
      const long parsed = std::strtol(threshold, &end, 10);
      refused_ = errno != 0 || end == threshold || *end != '\0' || parsed < 1 ||
                 parsed > kMaximumThreshold;
      if (!refused_) threshold_ = parsed;
    }
  }
};

Observer& State() {
  // Backend destruction can run during TLS/DSO teardown. Deliberately keep the
  // mutex and marker storage alive until process exit, with no destructor
  // order.
  static Observer* const state = new Observer;
  return *state;
}

long ThreadId() { return syscall(SYS_gettid); }

void Marker(const Observer& state, const char* event, const char* reason,
            int status) noexcept {
  try {
    if (state.marker_.empty()) return;
    std::ofstream file(state.marker_, std::ios::app);
    file << "{\"event\":\"" << event << "\",\"reason\":\"" << reason
         << "\",\"pid\":" << getpid() << ",\"tid\":" << ThreadId()
         << ",\"owner_tid\":" << state.owner_tid_
         << ",\"context_generation\":" << state.generation_
         << ",\"actual_ns\":" << state.ns_ << ",\"target_ns\":" << kFinalGrid
         << ",\"consecutive_successful_runs\":" << state.count_
         << ",\"threshold\":" << state.threshold_ << ",\"status\":" << status
         << "}\n";
  } catch (...) {
    // Missing telemetry invalidates diagnostic evidence, never solver output.
  }
}

void Refuse(Observer& m_state, const char* reason) noexcept {
  m_state.refused_ = true;
  m_state.count_ = 0;
  Marker(m_state, "observer_refused", reason, -1);
}

template <typename Callback>
void Observe(Callback callback) noexcept {
  try {
    Observer& state = State();
    if (!state.enabled_) return;
    std::lock_guard lock(state.mutex_);
    callback(state);
  } catch (...) {
    // Observer allocation, locking, and telemetry must not affect the solver.
  }
}

bool Owner(Observer& m_state) {
  if (m_state.owner_tid_ == 0) m_state.owner_tid_ = ThreadId();
  if (m_state.owner_tid_ != ThreadId()) {
    Refuse(m_state, "cross_thread_use");
    return false;
  }
  return !m_state.refused_;
}
}  // namespace

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
  using Create = void* (*)(const GpuSolverSetup&);
  static const auto original = [] {
    void* symbol = dlsym(RTLD_NEXT, "CreateGpuSolver");
    if (symbol == nullptr)
      throw std::runtime_error("Original Create unavailable");
    return reinterpret_cast<Create>(symbol);
  }();
  void* result = nullptr;
  try {
    result = original(setup);
  } catch (...) {
    Observe([](Observer& m_state) {
      m_state.count_ = 0;
      Marker(m_state, "create_exception", "original_exception", -1);
    });
    throw;
  }
  Observe([&](Observer& m_state) {
    if (!Owner(m_state)) return;
    if (m_state.context_ != nullptr || m_state.in_run_) {
      Refuse(m_state, "multiple_live_contexts");
      return;
    }
    ++m_state.generation_;
    m_state.count_ = 0;
    m_state.ns_ = setup.state_.dimensions_[0];
    m_state.context_ = result;
    if (result == nullptr) Refuse(m_state, "null_context");
  });
  return result;
}

extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output) {
  using Run = void (*)(void*, const GpuSolverControl&, GpuSolverScalars&);
  static const auto original = [] {
    void* symbol = dlsym(RTLD_NEXT, "RunGpuSolver");
    if (symbol == nullptr) throw std::runtime_error("Original Run unavailable");
    return reinterpret_cast<Run>(symbol);
  }();
  Observe([&](Observer& m_state) {
    if (!Owner(m_state)) return;
    if (m_state.context_ != m_context || m_state.in_run_) {
      Refuse(m_state, "shared_or_unknown_context");
      return;
    }
    m_state.in_run_ = true;
  });
  try {
    original(m_context, control, m_output);
  } catch (...) {
    Observe([&](Observer& m_state) {
      m_state.in_run_ = false;
      m_state.count_ = 0;
      Marker(m_state, "run_exception", "original_exception", -1);
    });
    throw;
  }
  Observe([&](Observer& m_state) {
    if (!Owner(m_state)) return;
    m_state.in_run_ = false;
    if (m_state.context_ != m_context) {
      Refuse(m_state, "context_changed_during_run");
      return;
    }
    if (m_output.bad_jacobian_ || m_state.ns_ != kFinalGrid) {
      m_state.count_ = 0;
      return;
    }
    ++m_state.count_;
    if (!m_state.attempted_ && m_state.count_ >= m_state.threshold_) {
      m_state.attempted_ = true;
      // The original Run has returned and its capture/launch scope is closed.
      const cudaError_t status = cudaProfilerStart();
      Marker(m_state, "profiler_start", "steady_final_grid",
             static_cast<int>(status));
    }
  });
}

extern "C" void DestroyGpuSolver(void* m_context) {
  using Destroy = void (*)(void*);
  static const auto original = [] {
    void* symbol = dlsym(RTLD_NEXT, "DestroyGpuSolver");
    if (symbol == nullptr)
      throw std::runtime_error("Original Destroy unavailable");
    return reinterpret_cast<Destroy>(symbol);
  }();
  try {
    original(m_context);
  } catch (...) {
    Observe([](Observer& m_state) { Refuse(m_state, "destroy_exception"); });
    throw;
  }
  Observe([&](Observer& m_state) {
    if (!Owner(m_state)) return;
    if (m_state.context_ != m_context || m_state.in_run_) {
      Refuse(m_state, "destroy_shared_or_unknown_context");
      return;
    }
    m_state.context_ = nullptr;
    m_state.count_ = 0;
    m_state.ns_ = 0;
  });
}
