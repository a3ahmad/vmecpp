// SPDX-License-Identifier: MIT
// Diagnostic-only host ranges; no CUDA calls or physics changes.
#include <dlfcn.h>
#include <nvtx3/nvToolsExt.h>

#include <array>
#include <stdexcept>
#include <string>

#include "gpu_solver.h"

namespace {
template <typename T>
T Next(const char* name) {
  void* symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) throw std::runtime_error(name);
  return reinterpret_cast<T>(symbol);
}
class Range {
 public:
  explicit Range(const char* name) { nvtxRangePushA(name); }
  ~Range() { nvtxRangePop(); }
};
thread_local void* current_context = nullptr;
}  // namespace

extern "C" void ProfileNvtxPush(const char* name) { nvtxRangePushA(name); }
extern "C" void ProfileNvtxPop() { nvtxRangePop(); }

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
  static const auto original =
      Next<void* (*)(const GpuSolverSetup&)>("CreateGpuSolver");
  if (current_context != nullptr)
    throw std::runtime_error("Single-context diagnostic only");
  const std::string label =
      "GPU context ns=" + std::to_string(setup.state_.dimensions_[0]);
  nvtxRangePushA(label.c_str());
  try {
    current_context = original(setup);
    return current_context;
  } catch (...) {
    nvtxRangePop();
    throw;
  }
}

extern "C" void DestroyGpuSolver(void* m_context) {
  static const auto original = Next<void (*)(void*)>("DestroyGpuSolver");
  original(m_context);
  if (m_context == current_context && current_context != nullptr) {
    current_context = nullptr;
    nvtxRangePop();
  }
}

extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output) {
  static const auto original =
      Next<void (*)(void*, const GpuSolverControl&, GpuSolverScalars&)>(
          "RunGpuSolver");
  Range range("GPU force update");
  original(m_context, control, m_output);
}

extern "C" void AdvanceGpuSolver(void* m_context,
                                 const std::array<double, 3>& scalars) {
  static const auto original =
      Next<void (*)(void*, const std::array<double, 3>&)>("AdvanceGpuSolver");
  Range range("GPU advance");
  original(m_context, scalars);
}
