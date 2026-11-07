// SPDX-License-Identifier: MIT
// Full resident-field graph diagnostic; never a pure profile-kernel timing.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_resident.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Resources {
  GpuResidentDeviceState state_{};
  std::vector<void*> allocations_;
  std::array<const double*, 38> outputs_{};
  std::vector<std::size_t> sizes_;
  double* input_ = nullptr;
  double* seed_ = nullptr;
  std::size_t input_bytes_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
  void* library_ = nullptr;
  using Launch = void (*)(GpuResidentDeviceState&);
  Launch launch_ = nullptr;
  std::string name_;
  ~Resources() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
    for (auto* allocation : allocations_) cudaFree(allocation);
    if (begin_) cudaEventDestroy(begin_);
    if (end_) cudaEventDestroy(end_);
    if (stream_) cudaStreamDestroy(stream_);
    // Keep CUDA modules loaded until process teardown, after all graphs die.
  }
  double* Allocate(std::size_t count) {
    double* result;
    Check(cudaMalloc(&result, count * sizeof(double)));
    allocations_.push_back(result);
    return result;
  }
  void Enqueue() {
    Check(cudaMemcpyAsync(input_, seed_, input_bytes_, cudaMemcpyDeviceToDevice,
                          stream_));
    launch_(state_);
  }
  void Verify(const std::vector<double>& expected) {
    std::vector<double> actual(expected.size());
    std::size_t offset = 0;
    for (int i = 0; i < 38; ++i) {
      Check(cudaMemcpyAsync(actual.data() + offset, outputs_[i],
                            sizes_[i] * sizeof(double), cudaMemcpyDeviceToHost,
                            stream_));
      offset += sizes_[i];
    }
    Check(cudaStreamSynchronize(stream_));
    if (std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(double)))
      throw std::runtime_error("ALL38 output arrays not bitwise equal");
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6)
      throw std::runtime_error(
          "Usage: profile-benchmark capture output repeats samples "
          "libraries...");
    const int repeats = std::stoi(argv[3]), samples = std::stoi(argv[4]);
    if (repeats < 1 || repeats > 100 || samples < 1 || samples > 10)
      throw std::runtime_error("Invalid bounded repetition count");
    auto capture = vmecpp_experiment::Read<GpuResidentInput, GpuResidentOutput>(
        argv[1], "VMECGPU5");
    std::vector<std::unique_ptr<Resources>> resources;
    for (int argument = 5; argument < argc; ++argument) {
      auto value = std::make_unique<Resources>();
      auto& x = *value;
      x.name_ = argv[argument];
      x.library_ = dlopen(argv[argument], RTLD_NOW | RTLD_LOCAL);
      if (!x.library_) throw std::runtime_error(dlerror());
      x.launch_ = reinterpret_cast<Resources::Launch>(
          dlsym(x.library_, "LaunchGpuResidentFields"));
      if (!x.launch_) throw std::runtime_error("Missing field API");
      Check(cudaStreamCreateWithFlags(&x.stream_, cudaStreamNonBlocking));
      Check(cudaEventCreate(&x.begin_));
      Check(cudaEventCreate(&x.end_));
      const auto count = capture.values_.size();
      x.input_ = x.Allocate(count);
      x.seed_ = x.Allocate(count);
      x.input_bytes_ = count * sizeof(double);
      Check(cudaMemcpyAsync(x.seed_, capture.values_.data(), x.input_bytes_,
                            cudaMemcpyHostToDevice, x.stream_));
      auto& state = x.state_;
      state.dimensions_ = capture.input_.dimensions_;
      state.scalars_ = capture.input_.scalars_;
      state.stream_ = x.stream_;
      std::size_t offset = 0;
      for (int i = 0; i < 28; ++i) {
        if (i < 18)
          state.full_[i] = x.input_ + offset;
        else
          state.profiles_[i - 18] = x.input_ + offset;
        offset += capture.input_.sizes_[i];
      }
      const int ns = state.dimensions_[11];
      const std::size_t half =
          std::size_t(ns - 1) * state.dimensions_[7] * state.dimensions_[8];
      for (auto& pointer : state.half_) pointer = x.Allocate(half);
      state.profiles_[10] = x.Allocate(ns - 1);
      state.jacobian_extrema_ = x.Allocate(2);
      state.jacobian_partials_ = x.Allocate(3 * ((half + 31) / 32));
      state.profile_terms_ = x.Allocate(3 * half);
      for (int i = 0; i < 18; ++i) x.outputs_[i] = state.full_[i];
      for (int i = 0; i < 14; ++i) x.outputs_[18 + i] = state.half_[i];
      x.outputs_[32] = state.profiles_[10];
      for (int i = 0; i < 4; ++i) x.outputs_[33 + i] = state.profiles_[6 + i];
      x.outputs_[37] = state.jacobian_extrema_;
      x.sizes_.assign(capture.output_.sizes_.begin(),
                      capture.output_.sizes_.end());
      x.Enqueue();
      x.Verify(capture.expected_);
      Check(
          cudaStreamBeginCapture(x.stream_, cudaStreamCaptureModeThreadLocal));
      x.Enqueue();
      Check(cudaStreamEndCapture(x.stream_, &x.graph_));
      Check(cudaGraphInstantiate(&x.executable_, x.graph_, 0));
      Check(cudaGraphLaunch(x.executable_, x.stream_));
      x.Verify(capture.expected_);
      resources.push_back(std::move(value));
    }
    std::ofstream output(argv[2]);
    output
        << std::setprecision(17)
        << "{\"scope\":\"measured CUDAevents: full resident field graph plus "
           "input D2D reset, excludes D2H checks; not isolated profile kernel "
           "or solver latency\",\"all38arrays_exact\":true,\"repeats\":"
        << repeats << ",\"samples\":[";
    std::vector<int> order(resources.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(1741);
    bool first = true;
    for (int sample = 0; sample < samples; ++sample) {
      std::shuffle(order.begin(), order.end(), random);
      for (int index : order) {
        auto& x = *resources[index];
        Check(cudaEventRecord(x.begin_, x.stream_));
        for (int i = 0; i < repeats; ++i)
          Check(cudaGraphLaunch(x.executable_, x.stream_));
        Check(cudaEventRecord(x.end_, x.stream_));
        Check(cudaEventSynchronize(x.end_));
        float milliseconds;
        Check(cudaEventElapsedTime(&milliseconds, x.begin_, x.end_));
        x.Verify(capture.expected_);
        if (!first) output << ',';
        first = false;
        output << "{\"library\":\"" << x.name_ << "\",\"sample\":" << sample
               << ",\"mean_microseconds\":" << milliseconds * 1000.0 / repeats
               << '}';
      }
    }
    output << "]}\n";
    output.close();
    if (!output) throw std::runtime_error("Report write failure");
    std::puts(
        "PASS all38arrays direct/capture/every sample; no allocation in "
        "capture");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
