// SPDX-License-Identifier: MIT
// Connected captured M1 -> cached/uncached RZ controls; no physics changes.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <cstdio>
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
using Launch = void (*)(GpuIterationDeviceState&);
Launch Resolve(void* library, const char* name) {
  auto result = reinterpret_cast<Launch>(dlsym(library, name));
  if (result == nullptr) throw std::runtime_error(name);
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: rhs-tile-benchmark iteration.bin rhs.so control|timing");
    const bool timing = std::string(argv[3]) == "timing";
    if (!timing && std::string(argv[3]) != "control")
      throw std::runtime_error("Unknown mode");
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(dlerror());
    const auto factor = Resolve(library, "LaunchGpuIterationFactor");
    const auto cached = Resolve(library, "LaunchGpuIterationRZFactored");
    const auto uncached = Resolve(library, "LaunchGpuIterationRZ");
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    GpuIterationDeviceState state{};
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = stream;
    std::vector<void*> allocations;
    std::array<const double*, 6> initial{}, expected{};
    std::size_t offset = 0;
    for (int channel = 0; channel < 6; ++channel) {
      initial[channel] = capture.expected_.data() + offset;
      offset += capture.output_.sizes_[channel];
      Check(cudaMalloc(&state.forces_[channel],
                       capture.input_.sizes_[channel] * sizeof(double)));
      allocations.push_back(state.forces_[channel]);
    }
    for (int channel = 0; channel < 6; ++channel) {
      expected[channel] = capture.expected_.data() + offset;
      offset += capture.output_.sizes_[6 + channel];
    }
    for (int channel = 0; channel < 11; ++channel) {
      double* buffer = nullptr;
      const auto bytes = capture.input_.sizes_[6 + channel] * sizeof(double);
      Check(cudaMalloc(&buffer, bytes));
      allocations.push_back(buffer);
      Check(cudaMemcpyAsync(buffer, capture.input_.data_[6 + channel], bytes,
                            cudaMemcpyHostToDevice, stream));
      state.constants_[channel] = buffer;
    }
    const auto matrix_bytes = static_cast<std::size_t>(state.dimensions_[0]) *
                              state.dimensions_[1] *
                              (state.dimensions_[2] + 1) * sizeof(double);
    for (int component = 0; component < 2; ++component) {
      Check(cudaMalloc(&state.factored_upper_[component], matrix_bytes));
      allocations.push_back(state.factored_upper_[component]);
      Check(cudaMalloc(&state.denominators_[component], matrix_bytes));
      allocations.push_back(state.denominators_[component]);
    }
    factor(state);
    Check(cudaStreamSynchronize(stream));
    int checks = 0;
    for (const bool use_cached : {false, true}) {
      for (const bool graph : {false, true}) {
        const auto launch = use_cached ? cached : uncached;
        cudaGraph_t definition = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (graph) {
          Check(
              cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          launch(state);
          Check(cudaStreamEndCapture(stream, &definition));
          Check(cudaGraphInstantiate(&executable, definition, nullptr, nullptr,
                                     0));
        }
        cudaEvent_t begin = nullptr, end = nullptr;
        Check(cudaEventCreate(&begin));
        Check(cudaEventCreate(&end));
        double milliseconds = 0.0;
        const int repeats = timing ? 25 : 3;
        constexpr int kWarmups = 2;
        for (int repeat = 0; repeat < repeats + kWarmups; ++repeat) {
          for (int channel = 0; channel < 6; ++channel) {
            Check(
                cudaMemcpyAsync(state.forces_[channel], initial[channel],
                                capture.input_.sizes_[channel] * sizeof(double),
                                cudaMemcpyHostToDevice, stream));
          }
          Check(cudaEventRecord(begin, stream));
          if (graph)
            Check(cudaGraphLaunch(executable, stream));
          else
            launch(state);
          Check(cudaEventRecord(end, stream));
          Check(cudaEventSynchronize(end));
          float elapsed = 0.0f;
          Check(cudaEventElapsedTime(&elapsed, begin, end));
          if (repeat >= kWarmups) milliseconds += elapsed;
          for (int channel = 0; channel < 6; ++channel) {
            const auto count = capture.input_.sizes_[channel];
            std::vector<double> actual(count);
            Check(cudaMemcpy(actual.data(), state.forces_[channel],
                             count * sizeof(double), cudaMemcpyDeviceToHost));
            if (std::memcmp(actual.data(), expected[channel],
                            count * sizeof(double)) != 0)
              throw std::runtime_error("RZ force output bits changed");
          }
          ++checks;
        }
        std::printf(
            "{\"cached\":%s,\"graph\":%s,\"timing_requested\":%s,"
            "\"measured_repeats\":%d,\"excluded_warmups\":%d,"
            "\"kernel_interval_us_mean\":%.9g,\"different_bits\":0}\n",
            use_cached ? "true" : "false", graph ? "true" : "false",
            timing ? "true" : "false", repeats, kWarmups,
            milliseconds * 1000.0 / repeats);
        Check(cudaEventDestroy(begin));
        Check(cudaEventDestroy(end));
        if (executable != nullptr) Check(cudaGraphExecDestroy(executable));
        if (definition != nullptr) Check(cudaGraphDestroy(definition));
      }
    }
    Check(cudaStreamSynchronize(stream));
    for (void* pointer : allocations) Check(cudaFree(pointer));
    Check(cudaStreamDestroy(stream));
    dlclose(library);
    std::printf("{\"checked_frames\":%d,\"all_six_forces_exact\":true}\n",
                checks);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
