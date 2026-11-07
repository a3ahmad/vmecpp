// SPDX-License-Identifier: MIT
// Real captured residual component checks; never a full-solver speed claim.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_adaptive_residual.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class Capture>
const double* Expected(const Capture& capture, int index) {
  return capture.expected_.data() +
         std::accumulate(capture.output_.sizes_.begin(),
                         capture.output_.sizes_.begin() + index,
                         std::size_t{0});
}
void RequireEqual(const double* expected, const double* actual) {
  if (std::memcmp(expected, actual, 3 * sizeof(double)) != 0) {
    for (int i = 0; i < 3; ++i)
      std::fprintf(stderr, "component%d expected%.17g actual%.17g\n", i,
                   expected[i], actual[i]);
    throw std::runtime_error(
        "Residual output differs from CPU/ordered reference");
  }
}
struct Graph {
  int max_rounds_;
  cudaGraph_t graph_;
  cudaGraphExec_t executable_;
};
struct Resources {
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
  std::vector<void*> allocations_;
  std::vector<Graph> graphs_;
  GpuIterationDeviceState state_{};
  double* captured_output_ = nullptr;
  double* history_ = nullptr;
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto& graph : graphs_) {
      cudaGraphExecDestroy(graph.executable_);
      cudaGraphDestroy(graph.graph_);
    }
    for (void* pointer : allocations_) cudaFree(pointer);
    if (start_ != nullptr) cudaEventDestroy(start_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t count, const double* source = nullptr) {
    double* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    try {
      allocations_.push_back(pointer);
    } catch (...) {
      cudaFree(pointer);
      throw;
    }
    if (source != nullptr)
      Check(cudaMemcpyAsync(pointer, source, count * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
    return pointer;
  }
  void CaptureOuter(int max_rounds) {
    for (const auto& graph : graphs_)
      if (graph.max_rounds_ == max_rounds) return;
    Graph graph{max_rounds, nullptr, nullptr};
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      // Poisoned output plus downstream snapshot tests both upstream and
      // downstream capture dependencies, not just a formerly valid result.
      Check(cudaMemsetAsync(state_.residuals_, 0x7f, 3 * sizeof(double),
                            stream_));
      LaunchGpuAdaptiveResiduals(state_, true, max_rounds);
      Check(cudaMemcpyAsync(captured_output_, state_.residuals_,
                            3 * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream_));
      const auto status = cudaStreamEndCapture(stream_, &graph.graph_);
      capturing = false;
      Check(status);
      Check(cudaGraphInstantiate(&graph.executable_, graph.graph_, 0));
      graphs_.push_back(graph);
    } catch (...) {
      if (capturing) cudaStreamEndCapture(stream_, &graph.graph_);
      if (graph.executable_ != nullptr) cudaGraphExecDestroy(graph.executable_);
      if (graph.graph_ != nullptr) cudaGraphDestroy(graph.graph_);
      throw;
    }
  }
};
struct Configuration {
  int mode_;  // 0 ordered, 1 adaptive standalone, 2 active outer insertion.
  int max_rounds_;
};
void Launch(Resources& m_resources, Configuration configuration) {
  if (configuration.mode_ == 0) {
    LaunchGpuIterationResidualsConditional(m_resources.state_, true, nullptr);
  } else if (configuration.mode_ == 1) {
    LaunchGpuAdaptiveResiduals(m_resources.state_, true,
                               configuration.max_rounds_);
  } else {
    m_resources.CaptureOuter(configuration.max_rounds_);
    for (const auto& graph : m_resources.graphs_)
      if (graph.max_rounds_ == configuration.max_rounds_) {
        Check(cudaGraphLaunch(graph.executable_, m_resources.stream_));
        return;
      }
    throw std::runtime_error("Outer graph cache missing");
  }
}
const double* Output(const Resources& resources, Configuration configuration) {
  return configuration.mode_ == 2 ? resources.captured_output_
                                  : resources.state_.residuals_;
}
struct Measurement {
  Configuration configuration_;
  int repetition_;
  double microseconds_;
  std::array<unsigned long long, 13> statistics_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5 ||
        (std::string(argv[2]) != "before" && std::string(argv[2]) != "after"))
      throw std::runtime_error(
          "Usage: adaptive-residual-benchmark iteration.bin before|after "
          "output.json [repeats]");
    const int repeats = argc == 5 ? std::stoi(argv[4]) : 20;
    if (repeats < 1 || repeats > 100)
      throw std::runtime_error("Repeat count must be within[1,100]");
    const bool after = std::string(argv[2]) == "after";
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&resources.start_));
    Check(cudaEventCreate(&resources.end_));
    auto& state = resources.state_;
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = resources.stream_;
    for (int i = 0; i < 6; ++i)
      state.forces_[i] = resources.Allocate(
          capture.input_.sizes_[i],
          after ? Expected(capture, 12 + i) : capture.input_.data_[i]);
    state.residuals_ = resources.Allocate(3);
    resources.captured_output_ = resources.Allocate(3);
    resources.history_ = resources.Allocate(3 * (repeats + 2));
    Check(cudaStreamSynchronize(resources.stream_));
    const double* expected = Expected(capture, after ? 19 : 18);
    std::array<double, 3> reference{};
    LaunchGpuIterationResidualsConditional(state, true, nullptr);
    Check(cudaMemcpyAsync(reference.data(), state.residuals_, sizeof(reference),
                          cudaMemcpyDeviceToHost, resources.stream_));
    Check(cudaStreamSynchronize(resources.stream_));
    RequireEqual(expected, reference.data());
    std::puts(
        "PASS: original ordered GPU residual equals captured CPU triple.");
    std::fflush(stdout);
    std::vector<Configuration> configurations{{0, 0}};
    for (int rounds : {4, 8, 12, 16, 24, 32}) {
      configurations.push_back({1, rounds});
      configurations.push_back({2, rounds});
    }
    // All outputs pass before collecting any CUDA-event timings.
    for (const auto& configuration : configurations) {
      for (int repeat = 0; repeat < 3; ++repeat) {
        std::printf("CHECK mode%d max%d repeat%d\n", configuration.mode_,
                    configuration.max_rounds_, repeat);
        std::fflush(stdout);
        Launch(resources, configuration);
        std::array<double, 3> actual{};
        Check(cudaMemcpyAsync(actual.data(), Output(resources, configuration),
                              sizeof(actual), cudaMemcpyDeviceToHost,
                              resources.stream_));
        Check(cudaStreamSynchronize(resources.stream_));
        RequireEqual(expected, actual.data());
        RequireEqual(reference.data(), actual.data());
      }
    }
    std::puts(
        "PASS: standalone and active-outer insertion exact for all six limits "
        "before timing.");
    std::fflush(stdout);
    std::mt19937 generator(2077);
    std::vector<Measurement> measurements;
    for (int repetition = 0; repetition < 3; ++repetition) {
      std::shuffle(configurations.begin(), configurations.end(), generator);
      for (const auto& configuration : configurations) {
        std::array<unsigned long long, 13> before{}, after_statistics{};
        for (int i = 0; i < 2; ++i) {
          Launch(resources, configuration);
          Check(cudaMemcpyAsync(resources.history_ + 3 * (repeats + i),
                                Output(resources, configuration),
                                3 * sizeof(double), cudaMemcpyDeviceToDevice,
                                resources.stream_));
        }
        Check(cudaStreamSynchronize(resources.stream_));
        ReadGpuAdaptiveResidualStats(before);
        Check(cudaEventRecord(resources.start_, resources.stream_));
        for (int i = 0; i < repeats; ++i) {
          Launch(resources, configuration);
          Check(cudaMemcpyAsync(
              resources.history_ + 3 * i, Output(resources, configuration),
              3 * sizeof(double), cudaMemcpyDeviceToDevice, resources.stream_));
        }
        Check(cudaEventRecord(resources.end_, resources.stream_));
        Check(cudaEventSynchronize(resources.end_));
        float milliseconds = 0;
        Check(cudaEventElapsedTime(&milliseconds, resources.start_,
                                   resources.end_));
        ReadGpuAdaptiveResidualStats(after_statistics);
        for (int i = 0; i < 13; ++i) after_statistics[i] -= before[i];
        std::vector<double> history(3 * (repeats + 2));
        Check(cudaMemcpy(history.data(), resources.history_,
                         history.size() * sizeof(double),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < repeats + 2; ++i)
          RequireEqual(expected, history.data() + 3 * i);
        measurements.push_back({configuration, repetition,
                                milliseconds * 1000.0 / repeats,
                                after_statistics});
      }
    }
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Cannot open measurement JSON");
    output
        << std::setprecision(17)
        << "{\"scope\":\"real captured residual component; no fullsolver "
           "speedup claim\","
        << "\"timing_scope\":\"CUDAevents include scheduling and24B D2D "
           "history; outer case also includes poisoned-output memset and24B "
           "dependency snapshot\","
        << "\"statistics_scope\":\"same-workspace within-case deltas; no "
           "Resize between snapshots\","
        << "\"statistics_layout\":\"calls,fallbackR,Z,lambda,"
           "terminationhistogram0,4,8,12,16,20,24,28,32\","
        << "\"stage\":\"" << argv[2]
        << "\",\"captured_cpu_exact\":true,\"all_outputs_bitwise_exact\":true,"
        << "\"repeats\":" << repeats
        << ",\"warmups\":2,\"random_seed\":2077,\"measurements\":[\n";
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto& measurement = measurements[i];
      output << "{\"mode\":" << measurement.configuration_.mode_
             << ",\"max_rounds\":" << measurement.configuration_.max_rounds_
             << ",\"repetition\":" << measurement.repetition_
             << ",\"measured_mean_microseconds\":" << measurement.microseconds_
             << ",\"statistics\":[";
      for (int j = 0; j < 13; ++j) {
        if (j) output << ',';
        output << measurement.statistics_[j];
      }
      output << "]}" << (i + 1 == measurements.size() ? "\n" : ",\n");
    }
    output << "]}\n";
    std::puts(
        "PASS: every repeated output exact; component timings and round "
        "histograms recorded.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
