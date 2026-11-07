// SPDX-License-Identifier: MIT
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "gpu_diagnostics.h"
#include "gpu_iteration.h"
#include "gpu_preconditioner.h"
#include "resident_replay_capture.h"

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "Usage: iteration-replay capture.bin library.so");
#if defined(VMECPP_GEOMETRY_REPLAY)
    auto capture =
        vmecpp_experiment::Read<GpuPreparationInput, GpuPreparationOutput>(
            argv[1], "VMECGPR1");
#elif defined(VMECPP_PRECONDITIONER_REPLAY)
    auto capture =
        vmecpp_experiment::Read<GpuPreconditionerInput,
                                GpuPreconditionerOutput>(argv[1], "VMECGPC2");
#elif defined(VMECPP_PREPARATION_REPLAY)
    auto capture =
        vmecpp_experiment::Read<GpuPreparationInput, GpuPreparationOutput>(
            argv[1], "VMECGPR1");
#elif defined(VMECPP_DIAGNOSTICS_REPLAY)
    auto capture =
        vmecpp_experiment::Read<GpuDiagnosticsInput, GpuDiagnosticsOutput>(
            argv[1], "VMECGDG1");
#elif defined(VMECPP_STEP_REPLAY)
    auto capture = vmecpp_experiment::Read<GpuStepInput, GpuStepOutput>(
        argv[1], "VMECGST1");
#else
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
#endif
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(dlerror());
#if defined(VMECPP_GEOMETRY_REPLAY)
    using Function =
        void (*)(const GpuPreparationInput&, GpuPreparationOutput&);
    auto run =
        reinterpret_cast<Function>(dlsym(library, "RunGpuGeometryPreparation"));
#elif defined(VMECPP_PRECONDITIONER_REPLAY)
    using Function =
        void (*)(const GpuPreconditionerInput&, GpuPreconditionerOutput&);
    auto run =
        reinterpret_cast<Function>(dlsym(library, "RunGpuPreconditioner"));
#elif defined(VMECPP_PREPARATION_REPLAY)
    using Function =
        void (*)(const GpuPreparationInput&, GpuPreparationOutput&);
    auto run = reinterpret_cast<Function>(
        dlsym(library, "RunGpuIterationPreparation"));
#elif defined(VMECPP_DIAGNOSTICS_REPLAY)
    using Function =
        void (*)(const GpuDiagnosticsInput&, GpuDiagnosticsOutput&);
    auto run = reinterpret_cast<Function>(dlsym(library, "RunGpuDiagnostics"));
#elif defined(VMECPP_STEP_REPLAY)
    using Function = void (*)(const GpuStepInput&, GpuStepOutput&);
    auto run =
        reinterpret_cast<Function>(dlsym(library, "RunGpuIterationStep"));
#else
    using Function = void (*)(const GpuIterationInput&, GpuIterationOutput&);
    auto run = reinterpret_cast<Function>(dlsym(library, "RunGpuIteration"));
#endif
    if (run == nullptr) throw std::runtime_error("Iteration adapter missing");
    constexpr int kWarmups = 10;
    constexpr int kRepeats = 100;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kWarmups + kRepeats; ++i) {
      run(capture.input_, capture.output_);
      const auto differences =
          vmecpp_experiment::Compare(capture.expected_, capture.actual_);
      if (differences != 0) {
        std::fprintf(stderr, "Repeat %d different values: %zu\n", i,
                     differences);
        return 2;
      }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
#if !defined(VMECPP_GEOMETRY_REPLAY) && !defined(VMECPP_STEP_REPLAY) && \
    !defined(VMECPP_DIAGNOSTICS_REPLAY) &&                              \
    !defined(VMECPP_PREPARATION_REPLAY) &&                              \
    !defined(VMECPP_PRECONDITIONER_REPLAY)
    if (std::getenv("VMECPP_GPU_RESIDUAL_ROUNDS") != nullptr) {
      using Statistics = void (*)(std::array<unsigned long long, 4>&);
      auto statistics = reinterpret_cast<Statistics>(
          dlsym(library, "ReadGpuResidualRelaxationStats"));
      if (statistics == nullptr)
        throw std::runtime_error("Residual statistics missing");
      std::array<unsigned long long, 4> values{};
      statistics(values);
      std::printf(
          "{\"experiment\":\"optimistic "
          "residual\",\"executed_calls\":%llu,\"fallback_calls\":[%llu,%llu,%"
          "llu]}\n",
          values[0], values[1], values[2], values[3]);
    }
#endif
    std::printf(
        "{\"measurement\":\"executed captured iteration-stage replay; "
        "allocation, downloads and comparison included, not solver "
        "timing\",\"different_values\":0,\"checked_executions\":%d,\"elapsed_"
        "seconds\":%.9g}\n",
        kWarmups + kRepeats, seconds);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
