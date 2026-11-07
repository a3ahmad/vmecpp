// SPDX-License-Identifier: MIT
// Opt-in outer-throughput experiment. No VMEC arithmetic or inputs are changed.
#include <Python.h>
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "vmecpp/vmec/vmec/vmec.h"

namespace {
constexpr std::string_view kLoopSymbol =
    "_ZN6vmecpp4Vmec20SolveEquilibriumLoopEiiNS_14VmecCheckpointERbS2_";

// LD_PRELOAD initializes this before Python can install a redirect. Loading the
// library after Python initialization is deliberately rejected for GIL release.
struct StartupStreams {
  std::ios_base::Init initialization_;
  std::streambuf* cout_ = std::cout.rdbuf();
  std::streambuf* cerr_ = std::cerr.rdbuf();
  bool trusted_ = !Py_IsInitialized();
};
const StartupStreams kStartupStreams;

bool Enabled() {
  const char* value = std::getenv("VMECPP_RELEASE_GIL");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

void* NextLoopSymbol() {
  // A tail call would make RTLD_NEXT see the caller's library (such as libffi).
  // The static initialization keeps work after dlsym in this interposer.
  static void* const next = dlsym(RTLD_NEXT, kLoopSymbol.data());
  return next;
}

bool OwnsGil() { return Py_IsInitialized() && PyGILState_Check(); }

bool GpuRequested() {
  for (char** variable = environ; variable != nullptr && *variable != nullptr;
       ++variable) {
    const std::string_view entry(*variable);
    const auto separator = entry.find('=');
    if (separator == std::string_view::npos ||
        !entry.substr(0, separator).starts_with("VMECPP_GPU_"))
      continue;
    const auto value = entry.substr(separator + 1);
    if (!value.empty() && value != "0") return true;
  }
  return false;
}

struct BackendAudit {
  int count_ = 0;
  bool coordinated_ = true;
};

int InspectBackend(dl_phdr_info* information, std::size_t, void* m_data) {
  if (information->dlpi_name == nullptr || information->dlpi_name[0] == '\0')
    return 0;
  void* library = dlopen(information->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
  if (library == nullptr) return 0;
  void* create = dlsym(library, "CreateGpuSolver");
  Dl_info owner{};
  const auto base = reinterpret_cast<void*>(information->dlpi_addr);
  if (create != nullptr && dladdr(create, &owner) != 0 &&
      owner.dli_fbase == base) {
    auto& audit = *static_cast<BackendAudit*>(m_data);
    ++audit.count_;
    void* coordinated = dlsym(library, "GpuSolverCaptureCoordinated");
    Dl_info flag_owner{};
    using Advertisement = bool (*)();
    audit.coordinated_ &= coordinated != nullptr &&
                          dladdr(coordinated, &flag_owner) != 0 &&
                          flag_owner.dli_fbase == base &&
                          reinterpret_cast<Advertisement>(coordinated)();
  }
  dlclose(library);
  return 0;
}

std::string GpuSafetyReason() {
  if (!GpuRequested()) return {};
  BackendAudit audit;
  dl_iterate_phdr(InspectBackend, &audit);
  if (audit.count_ != 1)
    return "GPU GIL release requires exactly one solver backend";
  if (!audit.coordinated_)
    return "GPU GIL release requires a capture-coordinated solver backend";
  return {};
}

std::string UnsafeReason(bool verbose) {
  const std::string gpu_reason = GpuSafetyReason();
  if (!gpu_reason.empty()) return gpu_reason;
  if (verbose) return "GIL release requires verbose=False";
  if (!kStartupStreams.trusted_)
    return "GIL shim must be preloaded before Python initializes";
  if (std::cout.rdbuf() != kStartupStreams.cout_ ||
      std::cerr.rdbuf() != kStartupStreams.cerr_)
    return "C++ ostream redirection is forbidden for GIL release";
  for (const char* name : {"stdout", "stderr"}) {
    const std::string original = std::string("__") + name + "__";
    PyObject* current = PySys_GetObject(name);
    PyObject* initial = PySys_GetObject(original.c_str());
    if (current == nullptr || initial == nullptr || current != initial)
      return "Python stdout/stderr redirection is forbidden for GIL release";
  }
  for (const char* name :
       {"CGA_PROFILE", "VMECPP_GPU_SOLVER_TRACE", "VMECPP_PROFILE_STAGES",
        "VMECPP_PROFILE_STAGES_ONLY", "VMECPP_GPU_PROFILE_MARKER",
        "VMECPP_FORWARD_CAPTURE", "VMECPP_RESIDENT_CAPTURE",
        "VMECPP_ITERATION_CAPTURE", "VMECPP_PIPELINE_CAPTURE",
        "VMECPP_STEP_CAPTURE", "VMECPP_DIAGNOSTICS_CAPTURE",
        "VMECPP_PREPARATION_CAPTURE", "VMECPP_MHD_CAPTURE",
        "VMECPP_PRECONDITIONER_CAPTURE"}) {
    const char* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0')
      return "Shared trace/capture paths are forbidden for GIL release";
  }
  return {};
}

class ScopedThreadRelease {
 public:
  explicit ScopedThreadRelease(bool release) {
    if (release) thread_state_ = PyEval_SaveThread();
  }
  ~ScopedThreadRelease() {
    if (thread_state_ != nullptr) PyEval_RestoreThread(thread_state_);
  }
  ScopedThreadRelease(const ScopedThreadRelease&) = delete;
  ScopedThreadRelease& operator=(const ScopedThreadRelease&) = delete;
  bool Released() const { return thread_state_ != nullptr; }

 private:
  PyThreadState* thread_state_ = nullptr;
};
}  // namespace

namespace vmecpp {
absl::StatusOr<Vmec::SolveEqLoopStatus> Vmec::SolveEquilibriumLoop(
    int thread_id, int iterations_before_checkpointing,
    VmecCheckpoint checkpoint, bool& m_lreset_internal, bool& m_liter_flag) {
  using Method = absl::StatusOr<SolveEqLoopStatus> (*)(
      Vmec*, int, int, VmecCheckpoint, bool&, bool&);
  // Resolve before dropping the GIL; never run Python inside a static
  // initializer that another Python thread could wait for while holding the
  // GIL.
  const auto next = reinterpret_cast<Method>(NextLoopSymbol());
  if (next == nullptr)
    return absl::InternalError("Next VMEC loop symbol missing");
  if (!Enabled())
    return next(this, thread_id, iterations_before_checkpointing, checkpoint,
                m_lreset_internal, m_liter_flag);
  if (num_threads_ != 1 || thread_id != 0)
    return absl::FailedPreconditionError("GIL release requires max_threads=1");
  const bool release = OwnsGil();
  const std::string reason =
      release ? UnsafeReason(verbose_) : GpuSafetyReason();
  if (!reason.empty()) return absl::FailedPreconditionError(reason);
  ScopedThreadRelease guard(release);
  return next(this, thread_id, iterations_before_checkpointing, checkpoint,
              m_lreset_internal, m_liter_flag);
}
}  // namespace vmecpp

// Batch-driver preflight; PyDLL callers retain the GIL. This invokes no CUDA
// API.
extern "C" bool VmecppGilReleaseReady() {
  return Enabled() && OwnsGil() && UnsafeReason(false).empty();
}

#ifdef VMECPP_GIL_SHIM_TESTING
namespace {
std::atomic<bool> test_active{false};
struct TestActiveWindow {
  TestActiveWindow() { test_active.store(true); }
  ~TestActiveWindow() { test_active.store(false); }
};
}  // namespace

extern "C" int VmecppGilGuardActive() { return test_active.load() ? 1 : 0; }
extern "C" void* VmecppGilNextLoopAddress() { return NextLoopSymbol(); }

// Test-only C entry point. No VMEC solver or GPU operation is invoked.
extern "C" int VmecppGilGuardExercise(int verbose, int throw_after_release,
                                      int milliseconds) {
  const bool owned_before = OwnsGil();
  if (!(owned_before ? UnsafeReason(verbose != 0) : GpuSafetyReason()).empty())
    return 16;
  int result = owned_before ? 1 : 0;
  try {
    ScopedThreadRelease guard(owned_before);
    TestActiveWindow active;
    if (guard.Released()) result |= 2;
    if (OwnsGil()) result |= 32;
    if (milliseconds > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    if (throw_after_release != 0) throw std::runtime_error("test exception");
  } catch (const std::runtime_error&) {
    result |= 8;
  }
  if (OwnsGil()) result |= 4;
  return result;
}
#endif
