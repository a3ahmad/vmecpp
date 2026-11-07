// SPDX-License-Identifier: MIT
// Actual captured iteration component only; never a whole-solver speed claim.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_thomas_prefix.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Graph {
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};
struct Resources {
  GpuIterationDeviceState state_{};
  cudaStream_t stream_ = nullptr;
  std::vector<void*> allocations_;
  std::array<Graph, 3> graphs_{};
  void* context_ = nullptr;
  double* forces_ = nullptr;
  double* seed_ = nullptr;
  std::size_t count_ = 0;
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
  using Ordinary = void (*)(GpuIterationDeviceState&);
  Ordinary ordinary_ = nullptr;
  ~Resources() {
    if (stream_) cudaStreamSynchronize(stream_);
    for (auto& graph : graphs_) {
      if (graph.executable_) cudaGraphExecDestroy(graph.executable_);
      if (graph.graph_) cudaGraphDestroy(graph.graph_);
    }
    if (context_) DestroyGpuThomasPrefix(context_);
    for (auto* pointer : allocations_) cudaFree(pointer);
    if (begin_) cudaEventDestroy(begin_);
    if (end_) cudaEventDestroy(end_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t count, const double* source = nullptr) {
    double* pointer;
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    allocations_.push_back(pointer);
    if (source)
      Check(cudaMemcpyAsync(pointer, source, count * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
    return pointer;
  }
  void Enqueue(int method) {
    Check(cudaMemcpyAsync(forces_, seed_, 6 * count_ * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream_));
    if (method == 0)
      ordinary_(state_);
    else if (method == 1)
      LaunchGpuThomasPrefix(context_, state_);
    else
      LaunchGpuThomasPrefixPoisoned(context_, state_);
  }
  std::vector<double> ReadForces() {
    std::vector<double> values(6 * count_);
    Check(cudaMemcpyAsync(values.data(), forces_,
                          values.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    return values;
  }
  void Equal(const std::vector<double>& reference) {
    const auto values = ReadForces();
    if (values.size() != reference.size() ||
        std::memcmp(values.data(), reference.data(),
                    values.size() * sizeof(double)))
      throw std::runtime_error("Full six-force bit mismatch");
  }
  void Capture(int method) {
    Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    Enqueue(method);
    Check(cudaStreamEndCapture(stream_, &graphs_[method].graph_));
    Check(cudaGraphInstantiate(&graphs_[method].executable_,
                               graphs_[method].graph_, 0));
  }
  void Execute(int configuration) {
    if (configuration < 2)
      Enqueue(configuration);
    else
      Check(cudaGraphLaunch(graphs_[configuration - 2].executable_, stream_));
  }
};
std::vector<double> Stage(const std::vector<double>& expected,
                          std::size_t count, int stage) {
  return std::vector<double>(expected.begin() + stage * 6 * count,
                             expected.begin() + (stage + 1) * 6 * count);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8)
      throw std::runtime_error(
          "Usage: benchmark capture prefixCPU rhs3.so output repeats samples "
          "[profile_graph_configuration]");
    const int repeats = std::stoi(argv[5]), samples = std::stoi(argv[6]);
    if (repeats < 1 || repeats > 100 || samples < 1 || samples > 10)
      throw std::runtime_error("Invalid repeat bound");
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&resources.begin_));
    Check(cudaEventCreate(&resources.end_));
    resources.count_ = capture.input_.sizes_[0];
    auto& state = resources.state_;
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = resources.stream_;
    resources.forces_ = resources.Allocate(6 * resources.count_);
    resources.seed_ = resources.Allocate(6 * resources.count_);
    for (int channel = 0; channel < 6; ++channel) {
      state.forces_[channel] = resources.forces_ + channel * resources.count_;
      Check(cudaMemcpyAsync(state.forces_[channel],
                            capture.input_.data_[channel],
                            resources.count_ * sizeof(double),
                            cudaMemcpyHostToDevice, resources.stream_));
    }
    for (int i = 0; i < 11; ++i)
      state.constants_[i] = resources.Allocate(capture.input_.sizes_[i + 6],
                                               capture.input_.data_[i + 6]);
    for (auto& value : state.factored_upper_)
      value = resources.Allocate(resources.count_);
    for (auto& value : state.denominators_)
      value = resources.Allocate(resources.count_);
    LaunchGpuIterationM1(state);
    resources.Equal(Stage(capture.expected_, resources.count_, 0));
    Check(cudaMemcpyAsync(resources.seed_, resources.forces_,
                          6 * resources.count_ * sizeof(double),
                          cudaMemcpyDeviceToDevice, resources.stream_));
    LaunchGpuIterationFactor(state);
    void* library = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    resources.ordinary_ = reinterpret_cast<Resources::Ordinary>(
        dlsym(library, "LaunchGpuIterationRZFactored"));
    if (!resources.ordinary_)
      throw std::runtime_error("Missing BEST policy3 entry");
    resources.context_ = CreateGpuThomasPrefix(state);
    std::array<unsigned long long, 4> counts{};
    bool uninitialized = false;
    try {
      ReadGpuThomasPrefixFlags(resources.context_, counts);
    } catch (const std::runtime_error&) {
      uninitialized = true;
    }
    if (!uninitialized)
      throw std::runtime_error("Unexecuted context falsely active");
    std::ifstream prefix_file(argv[2], std::ios::binary);
    std::array<char, 8> magic{};
    std::array<int, 5> dimensions{};
    std::uint64_t count;
    prefix_file.read(magic.data(), 8);
    prefix_file.read(reinterpret_cast<char*>(dimensions.data()),
                     sizeof(dimensions));
    prefix_file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (std::string(magic.data(), 8) != "THPFCPU1" ||
        dimensions != state.dimensions_ || count != resources.count_)
      throw std::runtime_error("CPU prefix shape mismatch");
    std::array<std::vector<double>, 4> forward_reference;
    for (auto& values : forward_reference) {
      values.resize(count);
      prefix_file.read(reinterpret_cast<char*>(values.data()),
                       count * sizeof(double));
    }
    if (!prefix_file) throw std::runtime_error("Truncated CPU prefix");
    const auto check_prefix = [&] {
      std::array<std::vector<double>, 4> actual;
      std::array<double*, 4> pointers;
      for (int i = 0; i < 4; ++i) {
        actual[i].resize(count);
        pointers[i] = actual[i].data();
      }
      ReadGpuThomasPrefixForward(resources.context_, pointers);
      for (int i = 0; i < 4; ++i)
        if (std::memcmp(actual[i].data(), forward_reference[i].data(),
                        count * sizeof(double)))
          throw std::runtime_error("Every CPU prefix not bitwise equal");
    };
    std::array<unsigned long long, 4> normal_counts{}, poison_counts{};
    for (int method = 0; method < 3; ++method) {
      resources.Enqueue(method);
      resources.Equal(Stage(capture.expected_, count, 1));
      if (method) {
        check_prefix();
        ReadGpuThomasPrefixFlags(resources.context_, counts);
        if (method == 1)
          normal_counts = counts;
        else {
          poison_counts = counts;
          if (counts[2] < 1) throw std::runtime_error("Poison not rejected");
        }
      }
      LaunchGpuIterationLambda(state);
      resources.Equal(Stage(capture.expected_, count, 2));
      resources.Capture(method);
      Check(cudaGraphLaunch(resources.graphs_[method].executable_,
                            resources.stream_));
      resources.Equal(Stage(capture.expected_, count, 1));
      if (method) check_prefix();
    }
    for (int test = 0; test < 5; ++test) {
      const double original = capture.expected_[0];
      const double value =
          test == 0   ? -0.0
          : test == 1 ? 0x1p-1074
          : test == 2
              ? 0x1.fffffffffffffp1023
              : std::bit_cast<double>(std::uint64_t{0x7ff8000000000123});
      if (test < 4)
        Check(cudaMemcpyAsync(resources.seed_, &value, sizeof(double),
                              cudaMemcpyHostToDevice, resources.stream_));
      else {
        const double zero = 0.0;
        Check(cudaMemcpyAsync(state.denominators_[0], &zero, sizeof(double),
                              cudaMemcpyHostToDevice, resources.stream_));
      }
      resources.Enqueue(0);
      const auto expected = resources.ReadForces();
      resources.Enqueue(1);
      resources.Equal(expected);
      if (test < 4)
        Check(cudaMemcpyAsync(resources.seed_, &original, sizeof(double),
                              cudaMemcpyHostToDevice, resources.stream_));
      else {
        const double denominator = capture.input_.data_[12][0];
        Check(cudaMemcpyAsync(state.denominators_[0], &denominator,
                              sizeof(double), cudaMemcpyHostToDevice,
                              resources.stream_));
      }
      RefreshGpuThomasPrefix(resources.context_, state);
    }
    // Inactive m>0 axis entries must become literal +0 even for NaN input;
    // the excluded last full-grid row must retain its original bits.
    for (double value :
         {3.0, std::bit_cast<double>(std::uint64_t{0x7ff8000000000123})}) {
      const int axis_mode = state.dimensions_[2] + 1;
      const double original = capture.expected_[axis_mode];
      Check(cudaMemcpyAsync(resources.seed_ + axis_mode, &value, sizeof(double),
                            cudaMemcpyHostToDevice, resources.stream_));
      resources.Enqueue(0);
      const auto reference = resources.ReadForces();
      if (std::bit_cast<std::uint64_t>(reference[axis_mode]) != 0)
        throw std::runtime_error("Original axis is not positive zero");
      resources.Enqueue(1);
      resources.Equal(reference);
      Check(cudaMemcpyAsync(resources.seed_ + axis_mode, &original,
                            sizeof(double), cudaMemcpyHostToDevice,
                            resources.stream_));
    }
    if (argc == 8) {
      const int configuration = std::stoi(argv[7]);
      if (configuration != 2 && configuration != 3)
        throw std::runtime_error("Profile only ordinary/proposal warm graph");
      resources.Execute(configuration);
      Check(cudaStreamSynchronize(resources.stream_));
      Check(cudaProfilerStart());
      resources.Execute(configuration);
      Check(cudaStreamSynchronize(resources.stream_));
      Check(cudaProfilerStop());
      resources.Equal(Stage(capture.expected_, count, 1));
      if (configuration == 3) check_prefix();
    }
    std::ofstream output(argv[4]);
    output << std::setprecision(17)
           << "{\"scope\":\"ONE actual99capture BESTpolicy3 vs4kernel "
              "proposal/cert/fallback/backsub; CUDAevents include common "
              "immutable input D2D reset, no solver "
              "speedup\",\"all18capturedforces_exact\":true,\"every_original_"
              "CPU_forward_prefix_exact\":true,\"poison_fallback_exact\":true,"
              "\"five_adversarial_cases_exact\":true,"
              "\"nonzero_and_NaN_axis_zero_exact\":true,\"repeats\":"
           << repeats << ",\"normal_flags\":[";
    for (int i = 0; i < 4; ++i) output << (i ? "," : "") << normal_counts[i];
    output << "],\"poison_flags\":[";
    for (int i = 0; i < 4; ++i) output << (i ? "," : "") << poison_counts[i];
    output << "],\"samples\":[";
    std::array<int, 4> order{0, 1, 2, 3};
    std::mt19937 random(2631);
    bool first = true;
    for (int sample = 0; sample < samples; ++sample) {
      std::shuffle(order.begin(), order.end(), random);
      for (int configuration : order) {
        Check(cudaEventRecord(resources.begin_, resources.stream_));
        for (int i = 0; i < repeats; ++i) resources.Execute(configuration);
        Check(cudaEventRecord(resources.end_, resources.stream_));
        Check(cudaEventSynchronize(resources.end_));
        float milliseconds;
        Check(cudaEventElapsedTime(&milliseconds, resources.begin_,
                                   resources.end_));
        resources.Equal(Stage(capture.expected_, count, 1));
        if (configuration % 2) {
          check_prefix();
          ReadGpuThomasPrefixFlags(resources.context_, counts);
        }
        if (!first) output << ',';
        first = false;
        output << "{\"configuration\":" << configuration
               << ",\"sample\":" << sample
               << ",\"mean_microseconds\":" << milliseconds * 1000.0 / repeats
               << '}';
      }
    }
    output << "]}\n";
    output.close();
    if (!output) throw std::runtime_error("Report write failed");
    std::puts(
        "PASS "
        "BESTpolicy3/full18/everyprefix/poison/adversarial/direct+graph/"
        "everytimedsample exact");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
