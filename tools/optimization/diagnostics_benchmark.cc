// SPDX-License-Identifier: MIT
// Component replay and randomized CUDA-event timing; no full-solver claim.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_diagnostics.h"
#include "gpu_optimistic_diagnostics.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceResources {
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  std::vector<void*> allocations_;
  DeviceResources() {
    Check(cudaStreamCreate(&stream_));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&stop_));
  }
  ~DeviceResources() {
    cudaStreamSynchronize(stream_);
    for (void* pointer : allocations_) cudaFree(pointer);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t size, const double* source = nullptr) {
    double* result = nullptr;
    Check(cudaMalloc(&result, size * sizeof(double)));
    allocations_.push_back(result);
    if (source != nullptr)
      Check(cudaMemcpyAsync(result, source, size * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
    return result;
  }
};
struct Configuration {
  int rounds_;
  bool refresh_;
};
struct Measurement {
  Configuration configuration_;
  int repetition_;
  double microseconds_;
  std::array<unsigned long long, 7> statistics_;
};
void Launch(GpuDiagnosticsDeviceState& m_state, Configuration configuration) {
  if (configuration.rounds_ == 0) {
    if (configuration.refresh_)
      LaunchGpuDiagnostics(m_state);
    else
      LaunchGpuEnergies(m_state);
  } else {
    LaunchGpuDiagnosticsOptimistic(m_state, configuration.refresh_,
                                   configuration.rounds_, true);
  }
}
void RequireEqual(const std::array<double, 7>& expected,
                  const std::array<double, 7>& actual) {
  for (int i = 0; i < 7; ++i) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(double)) != 0) {
      std::fprintf(stderr, "component %d expected %.17g actual %.17g\n", i,
                   expected[i], actual[i]);
      throw std::runtime_error("Diagnostics byte mismatch");
    }
  }
}
void Reset(GpuDiagnosticsDeviceState& m_state,
           const std::array<double, 7>& seed) {
  Check(cudaMemcpyAsync(m_state.output_, seed.data(), sizeof(seed),
                        cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(m_state.stream_)));
}
std::array<double, 7> Read(const GpuDiagnosticsDeviceState& state) {
  std::array<double, 7> result{};
  Check(cudaMemcpyAsync(result.data(), state.output_, sizeof(result),
                        cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(state.stream_)));
  Check(cudaStreamSynchronize(static_cast<cudaStream_t>(state.stream_)));
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error(
          "Usage: diagnostics-benchmark capture output-json [iterations]");
    const int iterations = argc == 4 ? std::stoi(argv[3]) : 100;
    if (iterations < 0 || iterations > 10000)
      throw std::runtime_error("Invalid iteration count");
    auto capture =
        vmecpp_experiment::Read<GpuDiagnosticsInput, GpuDiagnosticsOutput>(
            argv[1], "VMECGDG1");
    if (capture.expected_.size() != 7)
      throw std::runtime_error("Expected seven diagnostic outputs");
    DeviceResources resources;
    GpuDiagnosticsDeviceState state{};
    for (int i = 0; i < 14; ++i)
      state.data_[i] =
          resources.Allocate(capture.input_.sizes_[i], capture.input_.data_[i]);
    state.dimensions_ = capture.input_.dimensions_;
    state.scalars_ = capture.input_.scalars_;
    state.sums_ = resources.Allocate(6);
    state.output_ = resources.Allocate(7);
    state.stream_ = resources.stream_;
    std::array<double, 7> expected{};
    std::copy(capture.expected_.begin(), capture.expected_.end(),
              expected.begin());
    auto retained = expected;
    retained[0] = expected[0] * 1.25;
    retained[4] = 3.25;
    retained[5] = 4.25;
    retained[6] = 5.25;
    std::vector<Configuration> configurations;
    for (bool refresh : {false, true})
      for (int rounds : {0, 4, 8, 12, 16})
        configurations.push_back({rounds, refresh});
    // Validate every configuration before collecting any timing.
    for (const auto& configuration : configurations) {
      Reset(state, retained);
      Launch(state, configuration);
      RequireEqual(configuration.refresh_ ? expected : retained, Read(state));
    }
    std::puts(
        "PASS: all ten configurations byte-exact before timing; energies-only "
        "cached slots retained.");
    std::fflush(stdout);
    if (iterations == 0) {
      std::ofstream output(argv[2]);
      if (!output) throw std::runtime_error("Cannot create correctness output");
      output << "{\"correctness\": \"all ten configurations byte exact; "
                "energies-only cache retained\", \"timed\": false}\n";
      return 0;
    }
    std::mt19937 generator(2077);
    std::vector<Measurement> measurements;
    for (int repetition = 0; repetition < 3; ++repetition) {
      std::shuffle(configurations.begin(), configurations.end(), generator);
      for (const auto& configuration : configurations) {
        Reset(state, retained);
        for (int i = 0; i < 5; ++i) Launch(state, configuration);
        Check(cudaStreamSynchronize(resources.stream_));
        std::array<unsigned long long, 7> before{}, after{};
        if (configuration.rounds_ != 0)
          ReadGpuDiagnosticsRelaxationStats(before);
        Check(cudaEventRecord(resources.start_, resources.stream_));
        for (int i = 0; i < iterations; ++i) Launch(state, configuration);
        Check(cudaEventRecord(resources.stop_, resources.stream_));
        Check(cudaEventSynchronize(resources.stop_));
        float milliseconds = 0;
        Check(cudaEventElapsedTime(&milliseconds, resources.start_,
                                   resources.stop_));
        if (configuration.rounds_ != 0)
          ReadGpuDiagnosticsRelaxationStats(after);
        for (int i = 0; i < 7; ++i) after[i] -= before[i];
        RequireEqual(configuration.refresh_ ? expected : retained, Read(state));
        measurements.push_back(
            {configuration, repetition,
             static_cast<double>(milliseconds) * 1000.0 / iterations, after});
      }
    }
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("Cannot create benchmark output");
    output << std::setprecision(17)
           << "{\n  \"scope\": \"single captured diagnostic component; no "
              "full-solver speed claim\",\n"
           << "  \"correctness\": \"byte exact for all configurations before "
              "and after timing\",\n"
           << "  \"iterations\": " << iterations << ",\n  \"warmups\": 5,\n"
           << "  \"random_seed\": 2077,\n  \"measurements\": [\n";
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto& measurement = measurements[i];
      output << "    {\"refresh\": "
             << (measurement.configuration_.refresh_ ? "true" : "false")
             << ", \"rounds\": " << measurement.configuration_.rounds_
             << ", \"graph\": "
             << (measurement.configuration_.rounds_ ? "true" : "false")
             << ", \"repetition\": " << measurement.repetition_
             << ", \"measured_cuda_event_microseconds\": "
             << measurement.microseconds_ << ", \"calls_and_fallbacks\": [";
      for (int j = 0; j < 7; ++j) {
        if (j) output << ", ";
        output << measurement.statistics_[j];
      }
      output << "]}" << (i + 1 == measurements.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    std::puts(
        "PASS: all timed outputs byte-exact; randomized component timings "
        "recorded.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
