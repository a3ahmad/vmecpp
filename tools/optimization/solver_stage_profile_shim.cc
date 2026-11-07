// SPDX-License-Identifier: MIT
// Optional profiler observer: start collection at the first final-grid context.
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "gpu_solver.h"

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup) {
  using Create = void* (*)(const GpuSolverSetup&);
  static const auto original = [] {
    void* symbol = dlsym(RTLD_NEXT, "CreateGpuSolver");
    if (symbol == nullptr)
      throw std::runtime_error("Original GPU solver constructor unavailable");
    return reinterpret_cast<Create>(symbol);
  }();
  void* result = original(setup);
  constexpr int kFinalGrid = 99;
  static std::atomic<bool> started{false};
  const char* enabled = std::getenv("VMECPP_GPU_PROFILE_FINAL_GRID");
  if (enabled != nullptr && enabled[0] == '1' &&
      setup.state_.dimensions_[0] == kFinalGrid && !started.exchange(true)) {
    const cudaError_t status = cudaProfilerStart();
    // Telemetry cannot replace the original constructor result or exception.
    // A nonzero status invalidates the profiling gate, not the solver output.
    try {
      if (const char* path = std::getenv("VMECPP_GPU_PROFILE_MARKER")) {
        std::ofstream file(path, std::ios::app);
        file << "{\"event\":\"profiler_start\",\"pid\":" << getpid()
             << ",\"actual_ns\":" << setup.state_.dimensions_[0]
             << ",\"target_ns\":" << kFinalGrid
             << ",\"status\":" << static_cast<int>(status) << "}\n";
      }
    } catch (...) {
      // A missing marker prevents the profile from being accepted as evidence.
    }
  }
  return result;
}
