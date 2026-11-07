// SPDX-License-Identifier: MIT
// Experimental lattice observation only: workspace counters are not deltas.
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

#include "gpu_cooperative_energies.h"
#include "gpu_lattice_residual.h"
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
  const char* path = std::getenv("VMECPP_GPU_LATTICE_OBSERVER");
  return path == nullptr || path[0] == '\0' ? nullptr : path;
}

void Append(const std::string& record) noexcept {
  const char* path = OutputPath();
  if (path == nullptr) return;
  const int descriptor =
      open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    std::fputs(
        "lattice_solver_stats_shim: stats_error opening explicit output path\n",
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
    std::fputs(
        "lattice_solver_stats_shim: stats_error writing complete JSONL "
        "record\n",
        stderr);
}

struct ContextInfo {
  unsigned long long sequence_ = 0;
  std::array<int, 5> dimensions_{};
  std::array<int, 13> basis_dimensions_{};
  long created_thread_id_ = 0;
  bool lattice_bound_ = false;
  bool energy_bound_ = false;
};
struct Registry {
  std::mutex mutex_;
  unsigned long long next_sequence_ = 0;
  std::unordered_map<void*, ContextInfo> contexts_;
  long owner_thread_id_ = 0;
  int active_runs_ = 0;
  bool unsupported_workers_ = false;
};
thread_local void* active_context = nullptr;
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
             "lattice calls count residual evaluations and acceptance/fallback "
             "components; energy calls count ordinary energy-only updates");
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
      if (registry.owner_thread_id_ == 0)
        registry.owner_thread_id_ = info.created_thread_id_;
      if (registry.owner_thread_id_ != info.created_thread_id_)
        registry.unsupported_workers_ = true;
      registry.contexts_[context] = info;
    }
    Append(
        Metadata(context, &info, "create") +
        ",\"stats_sampled\":false,\"observation_status\":\"metadata_only\"}\n");
  } catch (...) {
    std::fputs(
        "lattice_solver_stats_shim: stats_error recording create metadata\n",
        stderr);
  }
}

class ScopedRun {
 public:
  explicit ScopedRun(void* context) noexcept {
    if (OutputPath() == nullptr) return;
    try {
      auto& registry = GetRegistry();
      std::lock_guard lock(registry.mutex_);
      const long thread = syscall(SYS_gettid);
      if (registry.active_runs_ != 0 || active_context != nullptr ||
          (registry.owner_thread_id_ != 0 &&
           registry.owner_thread_id_ != thread))
        registry.unsupported_workers_ = true;
      ++registry.active_runs_;
      previous_ = active_context;
      active_context = context;
      enabled_ = true;
    } catch (...) {
      std::fputs("lattice_solver_stats_shim: stats_error tracking Run\n",
                 stderr);
    }
  }
  ~ScopedRun() {
    if (!enabled_) return;
    auto& registry = GetRegistry();
    std::lock_guard lock(registry.mutex_);
    --registry.active_runs_;
    active_context = previous_;
  }
  ScopedRun(const ScopedRun&) = delete;
  ScopedRun& operator=(const ScopedRun&) = delete;

 private:
  void* previous_ = nullptr;
  bool enabled_ = false;
};

void MarkBound(bool energy) noexcept {
  if (OutputPath() == nullptr) return;
  try {
    auto& registry = GetRegistry();
    std::lock_guard lock(registry.mutex_);
    const auto entry = registry.contexts_.find(active_context);
    if (active_context == nullptr || entry == registry.contexts_.end()) {
      registry.unsupported_workers_ = true;
      return;
    }
    if (energy)
      entry->second.energy_bound_ = true;
    else
      entry->second.lattice_bound_ = true;
  } catch (...) {
    std::fputs(
        "lattice_solver_stats_shim: stats_error tracking module binding\n",
        stderr);
  }
}

template <std::size_t Size>
Snapshot<Size> Unobserved(bool allowed) {
  Snapshot<Size> result;
  result.status_ = allowed ? "uninitialized" : "sampling_forbidden";
  result.message_ =
      allowed ? "No module binding observed in this context"
              : "Observer requires one host thread and no active Run/capture";
  return result;
}

void ObserveDestroy(void* context) noexcept {
  if (OutputPath() == nullptr) return;
  try {
    auto& registry = GetRegistry();
    ContextInfo info;
    bool known = false;
    // Keep the mutex through GPU reads: a new Run wrapper cannot start capture
    // between the no-active-Run check and the completed snapshot read.
    std::lock_guard lock(registry.mutex_);
    {
      const auto entry = registry.contexts_.find(context);
      if (entry != registry.contexts_.end()) {
        info = entry->second;
        known = true;
        registry.contexts_.erase(entry);
      }
    }
    const char* gil = std::getenv("VMECPP_RELEASE_GIL");
    const bool allowed = known && !registry.unsupported_workers_ &&
                         registry.active_runs_ == 0 &&
                         info.created_thread_id_ == syscall(SYS_gettid) &&
                         !(gil != nullptr && std::strcmp(gil, "1") == 0);
    const auto lattice = allowed && info.lattice_bound_
                             ? ReadSnapshot<4>("ReadGpuLatticeResidualStats",
                                               "No lattice residual ran")
                             : Unobserved<4>(allowed);
    const auto energy =
        allowed && info.energy_bound_
            ? ReadSnapshot<36>("ReadGpuCooperativeEnergiesStats",
                               "No cooperative energies experiment ran")
            : Unobserved<36>(allowed);
    const bool valid = allowed && lattice.status_ != "stats_error" &&
                       energy.status_ != "stats_error";
    Append(
        Metadata(context, known ? &info : nullptr, "before_destroy") +
        ",\"observation_status\":" + Quote(valid ? "ok" : "stats_error") +
        ",\"context_metadata_status\":" +
        Quote(known ? "available" : "missing_create_metadata") +
        ",\"lattice_host_binding_observed\":" +
        (info.lattice_bound_ ? "true" : "false") +
        ",\"energy_host_binding_observed\":" +
        (info.energy_bound_ ? "true" : "false") + ",\"lattice\":" +
        Serialize(
            lattice,
            "calls,accepted_components,fallback_components,normal_segments") +
        ",\"cooperative_energy\":" +
        Serialize(energy,
                  "calls,fallback_thermal,fallback_magnetic,stopping_histogram_"
                  "0_to_32") +
        "}\n");
  } catch (...) {
    // Telemetry must not replace the solver's real destruction/control path.
    std::fputs(
        "lattice_solver_stats_shim: stats_error collecting destroy snapshot\n",
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

extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output) {
  using Run = void (*)(void*, const GpuSolverControl&, GpuSolverScalars&);
  static const auto run = Original<Run>("RunGpuSolver");
  ScopedRun scope(m_context);
  run(m_context, control, m_output);
}

extern "C" void LaunchGpuIterationResidualsLattice(
    GpuIterationDeviceState& m_state, bool include_edge_rz) {
  using Launch = void (*)(GpuIterationDeviceState&, bool);
  static const auto launch =
      Original<Launch>("LaunchGpuIterationResidualsLattice");
  launch(m_state, include_edge_rz);
  MarkBound(false);
}

extern "C" void LaunchGpuCooperativeEnergies(GpuDiagnosticsDeviceState& m_state,
                                             int max_rounds) {
  using Launch = void (*)(GpuDiagnosticsDeviceState&, int);
  static const auto launch = Original<Launch>("LaunchGpuCooperativeEnergies");
  launch(m_state, max_rounds);
  MarkBound(true);
}
