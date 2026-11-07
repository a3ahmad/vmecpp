// SPDX-License-Identifier: MIT
// Upload a real captured residual input once; measure device scheduling and
// retain every result on device for bitwise comparison after timing.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_iteration.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4 ||
        (std::string(argv[3]) != "before" && std::string(argv[3]) != "after")) {
      throw std::runtime_error(
          "Usage: residual-benchmark iteration.bin library.so before|after");
    }
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    const bool after = std::string(argv[3]) == "after";
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(dlerror());
    using Launch = void (*)(GpuIterationDeviceState&, bool);
    auto launch =
        reinterpret_cast<Launch>(dlsym(library, "LaunchGpuIterationResiduals"));
    if (launch == nullptr)
      throw std::runtime_error("Residual launch API missing");
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    GpuIterationDeviceState state{};
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = stream;
    std::size_t offset =
        after ? std::accumulate(capture.output_.sizes_.begin(),
                                capture.output_.sizes_.begin() + 12,
                                std::size_t{0})
              : 0;
    std::vector<void*> allocations;
    for (int i = 0; i < 6; ++i) {
      const double* input =
          after ? capture.expected_.data() + offset : capture.input_.data_[i];
      const auto count = capture.input_.sizes_[i];
      Check(cudaMalloc(&state.forces_[i], count * sizeof(double)));
      allocations.push_back(state.forces_[i]);
      Check(cudaMemcpyAsync(state.forces_[i], input, count * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      if (after) offset += count;
    }
    Check(cudaMalloc(&state.residuals_, 3 * sizeof(double)));
    allocations.push_back(state.residuals_);
    constexpr int kWarmups = 10;
    constexpr int kRepeats = 100;
    double* history = nullptr;
    Check(cudaMalloc(&history, (kWarmups + kRepeats) * 3 * sizeof(double)));
    allocations.push_back(history);
    cudaEvent_t begin = nullptr, end = nullptr;
    Check(cudaEventCreate(&begin));
    Check(cudaEventCreate(&end));
    for (int i = 0; i < kWarmups + kRepeats; ++i) {
      if (i == kWarmups) Check(cudaEventRecord(begin, stream));
      launch(state, true);
      Check(cudaMemcpyAsync(history + 3 * i, state.residuals_,
                            3 * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
    }
    Check(cudaEventRecord(end, stream));
    Check(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    Check(cudaEventElapsedTime(&milliseconds, begin, end));
    std::vector<double> actual((kWarmups + kRepeats) * 3);
    Check(cudaMemcpy(actual.data(), history, actual.size() * sizeof(double),
                     cudaMemcpyDeviceToHost));
    const auto residual_offset =
        std::accumulate(capture.output_.sizes_.begin(),
                        capture.output_.sizes_.begin() + 18, std::size_t{0}) +
        (after ? 3 : 0);
    const double* expected = capture.expected_.data() + residual_offset;
    for (int i = 0; i < kWarmups + kRepeats; ++i) {
      if (std::memcmp(actual.data() + 3 * i, expected, 3 * sizeof(double)) !=
          0) {
        throw std::runtime_error(
            "Residual benchmark changed a captured output");
      }
    }
    if (std::getenv("VMECPP_GPU_RESIDUAL_ROUNDS") != nullptr) {
      using Statistics = void (*)(std::array<unsigned long long, 4>&);
      auto statistics = reinterpret_cast<Statistics>(
          dlsym(library, "ReadGpuResidualRelaxationStats"));
      if (statistics == nullptr)
        throw std::runtime_error("Residual statistics API missing");
      std::array<unsigned long long, 4> values{};
      statistics(values);
      std::printf(
          "{\"experiment\":\"optimistic "
          "residual\",\"executed_calls\":%llu,\"fallback_calls\":[%llu,%llu,%"
          "llu]}\n",
          values[0], values[1], values[2], values[3]);
    }
    std::printf(
        "{\"measurement\":\"executed CUDA-event component timing; input "
        "uploaded once, includes scheduling gaps and 24-byte D2D result "
        "snapshots, excludes validation downloads; not full-solver "
        "timing\",\"stage\":\"%s\",\"mean_microseconds\":%.9g,\"measured_"
        "repeats\":%d,\"warmups\":%d,\"bitwise_checked_executions\":%d,"
        "\"different_values\":0}\n",
        argv[3], milliseconds * 1000.0 / kRepeats, kRepeats, kWarmups,
        kRepeats + kWarmups);
    for (void* pointer : allocations) Check(cudaFree(pointer));
    Check(cudaEventDestroy(begin));
    Check(cudaEventDestroy(end));
    Check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
