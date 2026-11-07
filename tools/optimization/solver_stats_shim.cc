// SPDX-License-Identifier: MIT
// Optional observation only: workspace counters are not context deltas.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "gpu_solver.h"

namespace {
constexpr char kCounterScope[] =
    "thread-local GPU-module workspace snapshot; counters may be cumulative "
    "or reset by Resize; reset generation unobserved; no context delta "
    "inferred";

std::string Quote(const std::string& value) {
  std::string result = "\"";
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') {
      result += '\\';
      result += static_cast<char>(character);
    } else if (character < 32 || character >= 127) {
      std::array<char, 7> escaped{};
      std::snprintf(escaped.data(), escaped.size(), "\\u%04x",
                    static_cast<unsigned int>(character));
      result += escaped.data();
    } else {
      result += static_cast<char>(character);
    }
  }
  return result + '"';
}

template <typename Function>
Function Original(const char* name) {
  dlerror();
  void* symbol = dlsym(RTLD_NEXT, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr)
    throw std::runtime_error(error == nullptr ? "Missing original solver API"
                                              : error);
  return reinterpret_cast<Function>(symbol);
}

const char* OutputPath() {
  const char* path = std::getenv("VMECPP_GPU_SOLVER_STATS");
  return path == nullptr || path[0] == '\0' ? nullptr : path;
}

void Append(const std::string& record) noexcept {
  const char* path = OutputPath();
  if (path == nullptr) return;
  const int descriptor =
      open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    std::fputs("solver_stats_shim: stats_error opening explicit output path\n",
               stderr);
    return;
  }
  bool success = flock(descriptor, LOCK_EX) == 0;
  std::size_t offset = 0;
  while (success && offset < record.size()) {
    const ssize_t written =
        write(descriptor, record.data() + offset, record.size() - offset);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) {
      success = false;
      break;
    }
    offset += static_cast<std::size_t>(written);
  }
  flock(descriptor, LOCK_UN);
  close(descriptor);
  if (!success)
    std::fputs("solver_stats_shim: stats_error writing complete JSONL record\n",
               stderr);
}

struct ContextInfo {
  unsigned long long sequence_ = 0;
  std::array<int, 5> dimensions_{};
  std::array<int, 13> basis_dimensions_{};
  long created_thread_id_ = 0;
};
struct Registry {
  std::mutex mutex_;
  unsigned long long next_sequence_ = 0;
  std::unordered_map<void*, ContextInfo> contexts_;
};
Registry& GetRegistry() {
  // Solver bindings can be destroyed from TLS destructors during shutdown.
  // Process-lifetime registry storage avoids static destruction-order hazards.
  static auto* registry = new Registry;
  return *registry;
}

std::string Metadata(void* context, const ContextInfo* info,
                     const char* event) {
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  std::array<char, 32> address{};
  std::snprintf(address.data(), address.size(), "%p", context);
  std::string result =
      "{\"event\":" + Quote(event) + ",\"pid\":" + std::to_string(getpid()) +
      ",\"observed_thread_id\":" + std::to_string(syscall(SYS_gettid)) +
      ",\"monotonic_timestamp_ns\":" + std::to_string(timestamp) +
      ",\"context_address\":" + Quote(address.data()) +
      ",\"context_known\":" + (info == nullptr ? "false" : "true") +
      ",\"context_sequence\":" +
      (info == nullptr ? "null" : std::to_string(info->sequence_)) +
      ",\"ns\":" +
      (info == nullptr ? "null" : std::to_string(info->dimensions_[0])) +
      ",\"dimensions_ns_mpol_ntor_nsMaxF_nsIncl\":";
  if (info == nullptr) {
    result += "null,\"created_thread_id\":null";
  } else {
    result += '[';
    for (int i = 0; i < 5; ++i) {
      if (i != 0) result += ',';
      result += std::to_string(info->dimensions_[i]);
    }
    result +=
        "],\"created_thread_id\":" + std::to_string(info->created_thread_id_);
    result += ",\"basis_dimensions\":[";
    for (int i = 0; i < 13; ++i) {
      if (i != 0) result += ',';
      result += std::to_string(info->basis_dimensions_[i]);
    }
    result += ']';
  }
  return result + ",\"counter_scope\":" + Quote(kCounterScope) +
         ",\"rate_limitations\":" +
         Quote(
             "residual calls count residual evaluations; diagnostics calls "
             "count all updates, not refresh-only per-component scan attempts");
}

template <std::size_t Size>
struct Snapshot {
  std::string status_;
  std::string message_;
  std::array<unsigned long long, Size> counters_{};
};

template <std::size_t Size>
Snapshot<Size> ReadSnapshot(const char* symbol, const char* uninitialized) {
  Snapshot<Size> result;
  using Read = void (*)(std::array<unsigned long long, Size>&);
  dlerror();
  auto read = reinterpret_cast<Read>(dlsym(RTLD_NEXT, symbol));
  const char* lookup_error = dlerror();
  if (lookup_error != nullptr || read == nullptr) {
    result.status_ = "stats_error";
    result.message_ = lookup_error == nullptr ? "Missing optional statistic API"
                                              : lookup_error;
    return result;
  }
  try {
    read(result.counters_);
    result.status_ = "available";
  } catch (const std::exception& error) {
    result.status_ = std::strcmp(error.what(), uninitialized) == 0
                         ? "uninitialized"
                         : "stats_error";
    result.message_ = error.what();
  } catch (...) {
    result.status_ = "stats_error";
    result.message_ = "Non-standard exception from statistic API";
  }
  return result;
}

template <std::size_t Size>
std::string Serialize(const Snapshot<Size>& snapshot, const char* layout) {
  std::string result = "{\"status\":" + Quote(snapshot.status_) +
                       ",\"layout\":" + Quote(layout) + ",\"counters\":";
  if (snapshot.status_ != "available") {
    result += "null";
  } else {
    result += '[';
    for (std::size_t i = 0; i < Size; ++i) {
      if (i != 0) result += ',';
      result += std::to_string(snapshot.counters_[i]);
    }
    result += ']';
  }
  return result + ",\"message\":" + Quote(snapshot.message_) + '}';
}

void ObserveCreate(void* context, const GpuSolverSetup& setup) noexcept {
  if (OutputPath() == nullptr) return;
  try {
    auto& registry = GetRegistry();
    ContextInfo info;
    {
      std::lock_guard lock(registry.mutex_);
      info.sequence_ = ++registry.next_sequence_;
      info.dimensions_ = setup.state_.dimensions_;
      info.basis_dimensions_ = setup.basis_.dimensions_;
      info.created_thread_id_ = syscall(SYS_gettid);
      registry.contexts_[context] = info;
    }
    Append(
        Metadata(context, &info, "create") +
        ",\"stats_sampled\":false,\"observation_status\":\"metadata_only\"}\n");
  } catch (...) {
    std::fputs("solver_stats_shim: stats_error recording create metadata\n",
               stderr);
  }
}

void ObserveDestroy(void* context) noexcept {
  if (OutputPath() == nullptr) return;
  try {
    auto& registry = GetRegistry();
    ContextInfo info;
    bool known = false;
    {
      std::lock_guard lock(registry.mutex_);
      const auto entry = registry.contexts_.find(context);
      if (entry != registry.contexts_.end()) {
        info = entry->second;
        known = true;
        registry.contexts_.erase(entry);
      }
    }
    const auto residual = ReadSnapshot<4>("ReadGpuResidualRelaxationStats",
                                          "No residual experiment ran");
    const auto diagnostics = ReadSnapshot<7>(
        "ReadGpuDiagnosticsRelaxationStats", "No optimistic diagnostics ran");
    const bool valid = known && residual.status_ != "stats_error" &&
                       diagnostics.status_ != "stats_error";
    Append(Metadata(context, known ? &info : nullptr, "before_destroy") +
           ",\"observation_status\":" + Quote(valid ? "ok" : "stats_error") +
           ",\"context_metadata_status\":" +
           Quote(known ? "available" : "missing_create_metadata") +
           ",\"residual\":" +
           Serialize(residual, "calls,fallback_R,fallback_Z,fallback_lambda") +
           ",\"diagnostics\":" +
           Serialize(diagnostics,
                     "calls,fallback_volume,fallback_thermal,fallback_magnetic,"
                     "fallback_fNormRZ,fallback_fNormL,fallback_fNorm1") +
           "}\n");
  } catch (...) {
    // Telemetry must not replace the solver's real destruction/control path.
    std::fputs("solver_stats_shim: stats_error collecting destroy snapshot\n",
               stderr);
  }
}
}  // namespace

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
  using Create = void* (*)(const GpuSolverSetup&);
  static const auto create = Original<Create>("CreateGpuSolver");
  void* context = create(setup);
  ObserveCreate(context, setup);
  return context;
}
extern "C" void DestroyGpuSolver(void* m_context) {
  using Destroy = void (*)(void*);
  static const auto destroy = Original<Destroy>("DestroyGpuSolver");
  ObserveDestroy(m_context);
  destroy(m_context);
}
