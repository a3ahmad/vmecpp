// SPDX-License-Identifier: MIT
// Diagnostic-only host iteration observations; never preload for timing.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "gpu_solver.h"
#include "gpu_thomas_prefix.h"

namespace {
template <typename Function>
Function Original(const char* name) {
  dlerror();
  void* symbol = dlsym(RTLD_NEXT, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr)
    throw std::runtime_error("Missing original Thomas observation API");
  return reinterpret_cast<Function>(symbol);
}
bool Enabled() {
  const char* path = std::getenv("VMECPP_GPU_THOMAS_OBSERVER");
  return path != nullptr && path[0] != '\0';
}
struct Observation {
  void* solver_ = nullptr;
  void* thomas_ = nullptr;
  unsigned long long sequence_ = 0;
  std::array<int, 5> dimensions_{};
  // Successful host updates, bad-Jacobian returns, and SUMS of the last flags:
  // columns, accepted columns, fallback columns, failed individual row checks.
  std::array<unsigned long long, 6> counts_{};
};
thread_local Observation observation;
thread_local unsigned long long next_sequence = 0;
void Append() {
  const auto& o = observation;
  const std::string record =
      "{\"pid\":" + std::to_string(getpid()) +
      " ,\"tid\":" + std::to_string(syscall(SYS_gettid)) +
      " ,\"sequence\":" + std::to_string(o.sequence_) +
      " ,\"ns\":" + std::to_string(o.dimensions_[0]) +
      " ,\"mpol\":" + std::to_string(o.dimensions_[1]) +
      " ,\"ntor\":" + std::to_string(o.dimensions_[2]) + " ,\"counts\":[" +
      std::to_string(o.counts_[0]) + "," + std::to_string(o.counts_[1]) + "," +
      std::to_string(o.counts_[2]) + "," + std::to_string(o.counts_[3]) + "," +
      std::to_string(o.counts_[4]) + "," + std::to_string(o.counts_[5]) +
      "],\"scope\":\"Per owned context, sum last flags ONCE after each "
      "successful host RunGpuSolver; bad-Jacobian returns excluded. Warmup "
      "and capture construction device evaluations not counted separately. "
      "Diagnostic synchronous reads; never timing evidence.\"}\n";
  const int descriptor = open(std::getenv("VMECPP_GPU_THOMAS_OBSERVER"),
                              O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (descriptor < 0) throw std::runtime_error("Observer output open failed");
  bool success = flock(descriptor, LOCK_EX) == 0;
  std::size_t offset = 0;
  while (success && offset < record.size()) {
    const auto count =
        write(descriptor, record.data() + offset, record.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0)
      success = false;
    else
      offset += count;
  }
  flock(descriptor, LOCK_UN);
  close(descriptor);
  if (!success) throw std::runtime_error("Observer output write failed");
}
}  // namespace

extern "C" void* CreateGpuThomasPrefix(const GpuIterationDeviceState& state) {
  static const auto original =
      Original<void* (*)(const GpuIterationDeviceState&)>(
          "CreateGpuThomasPrefix");
  if (Enabled() && observation.thomas_ != nullptr)
    throw std::runtime_error("Observer rejects overlapping TLS contexts");
  void* context = original(state);
  if (Enabled()) observation.thomas_ = context;
  return context;
}
extern "C" void DestroyGpuThomasPrefix(void* m_context) {
  static const auto original =
      Original<void (*)(void*)>("DestroyGpuThomasPrefix");
  if (Enabled() && observation.thomas_ != m_context)
    throw std::runtime_error("Observer Thomas context owner mismatch");
  original(m_context);
  if (Enabled()) observation.thomas_ = nullptr;
}
extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
  static const auto original =
      Original<void* (*)(const GpuSolverSetup&)>("CreateGpuSolver");
  if (Enabled() && observation.solver_ != nullptr)
    throw std::runtime_error("Observer rejects overlapping TLS solvers");
  void* context = original(setup);
  if (Enabled()) {
    if (observation.thomas_ == nullptr)
      throw std::runtime_error("Observer Thomas binding absent");
    observation.solver_ = context;
    observation.sequence_ = ++next_sequence;
    observation.dimensions_ = setup.state_.dimensions_;
    observation.counts_.fill(0);
  }
  return context;
}
extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output) {
  static const auto original =
      Original<void (*)(void*, const GpuSolverControl&, GpuSolverScalars&)>(
          "RunGpuSolver");
  if (Enabled() && observation.solver_ != m_context)
    throw std::runtime_error("Observer Run context owner mismatch");
  original(m_context, control, m_output);
  if (!Enabled()) return;
  if (m_output.bad_jacobian_) {
    ++observation.counts_[1];
    return;
  }
  static const auto read =
      Original<void (*)(void*, std::array<unsigned long long, 4>&)>(
          "ReadGpuThomasPrefixFlags");
  std::array<unsigned long long, 4> flags{};
  read(observation.thomas_, flags);
  const auto& d = observation.dimensions_;
  if (flags[0] != 4ULL * d[1] * (d[2] + 1) || flags[0] != flags[1] + flags[2])
    throw std::runtime_error("Observer column conservation failed");
  ++observation.counts_[0];
  for (int i = 0; i < 4; ++i) observation.counts_[i + 2] += flags[i];
}
extern "C" void DestroyGpuSolver(void* m_context) {
  static const auto original = Original<void (*)(void*)>("DestroyGpuSolver");
  if (Enabled()) {
    if (observation.solver_ != m_context || observation.thomas_ == nullptr)
      throw std::runtime_error("Observer Destroy context owner mismatch");
    Append();
  }
  original(m_context);
  if (Enabled()) observation.solver_ = nullptr;
}
