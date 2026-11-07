// SPDX-License-Identifier: MIT
// One profiled captured residual invocation. No loops in the profiler region.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_cooperative_residual.h"
#include "gpu_lattice_residual.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T>
T Symbol(void* library, const char* name) {
  const auto result = reinterpret_cast<T>(dlsym(library, name));
  if (result == nullptr) throw std::runtime_error(name);
  return result;
}
template <typename Capture>
const double* Expected(const Capture& capture, int channel) {
  return capture.expected_.data() +
         std::accumulate(capture.output_.sizes_.begin(),
                         capture.output_.sizes_.begin() + channel,
                         std::size_t{0});
}
struct Resources {
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  GpuIterationDeviceState state_{};
  double* snapshot_ = nullptr;
  std::vector<void*> allocations_;
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    for (void* pointer : allocations_) cudaFree(pointer);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t count) {
    double* result = nullptr;
    Check(cudaMalloc(&result, count * sizeof(double)));
    allocations_.push_back(result);
    return result;
  }
  void Verify(const double* expected) {
    std::array<double, 3> actual{};
    Check(cudaMemcpyAsync(actual.data(), snapshot_, sizeof(actual),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    if (std::memcmp(actual.data(), expected, sizeof(actual)) != 0)
      throw std::runtime_error("Counter probe changed captured CPU residual");
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8)
      throw std::runtime_error(
          "Usage: probe capture.bin original.so cooperative.so lattice.so "
          "before|after lattice|cooperative output.json");
    const std::string stage = argv[5];
    const std::string method = argv[6];
    if ((stage != "before" && stage != "after") ||
        (method != "lattice" && method != "cooperative"))
      throw std::runtime_error("Unsupported stage or method");
    void* original = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (original == nullptr) throw std::runtime_error(dlerror());
    void* cooperative = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
    if (cooperative == nullptr) throw std::runtime_error(dlerror());
    void* lattice = dlopen(argv[4], RTLD_NOW | RTLD_LOCAL);
    if (lattice == nullptr) throw std::runtime_error(dlerror());
    const auto launch_lattice =
        Symbol<decltype(&LaunchGpuIterationResidualsLattice)>(
            lattice, "LaunchGpuIterationResidualsLattice");
    const auto launch_cooperative =
        Symbol<decltype(&LaunchGpuCooperativeResiduals)>(
            cooperative, "LaunchGpuCooperativeResiduals");
    const auto supported = Symbol<decltype(&GpuCooperativeResidualsSupported)>(
        cooperative, "GpuCooperativeResidualsSupported");
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
    auto& state = resources.state_;
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = resources.stream_;
    for (int channel = 0; channel < 6; ++channel) {
      const auto count = capture.input_.sizes_[channel];
      const double* values = stage == "before"
                                 ? capture.input_.data_[channel]
                                 : Expected(capture, channel + 12);
      state.forces_[channel] = resources.Allocate(count);
      Check(cudaMemcpyAsync(state.forces_[channel], values,
                            count * sizeof(double), cudaMemcpyHostToDevice,
                            resources.stream_));
    }
    state.residuals_ = resources.Allocate(3);
    resources.snapshot_ = resources.Allocate(3);
    if (method == "cooperative" && !supported(state))
      throw std::runtime_error("Cooperative reference unsupported");
    const auto enqueue = [&] {
      Check(cudaMemsetAsync(state.residuals_, 0x7f, 3 * sizeof(double),
                            resources.stream_));
      if (method == "lattice")
        launch_lattice(state, true);
      else
        launch_cooperative(state, true, 32);
      Check(cudaMemcpyAsync(resources.snapshot_, state.residuals_,
                            3 * sizeof(double), cudaMemcpyDeviceToDevice,
                            resources.stream_));
    };
    const double* expected = Expected(capture, stage == "before" ? 18 : 19);
    enqueue();
    resources.Verify(expected);
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(resources.stream_,
                                   cudaStreamCaptureModeThreadLocal));
      capturing = true;
      enqueue();
      const auto status =
          cudaStreamEndCapture(resources.stream_, &resources.graph_);
      capturing = false;
      Check(status);
      Check(cudaGraphInstantiate(&resources.executable_, resources.graph_, 0));
    } catch (...) {
      if (capturing) cudaStreamEndCapture(resources.stream_, &resources.graph_);
      throw;
    }
    Check(cudaGraphLaunch(resources.executable_, resources.stream_));
    resources.Verify(expected);
    std::size_t nodes;
    Check(cudaGraphGetNodes(resources.graph_, nullptr, &nodes));
    std::vector<cudaGraphNode_t> handles(nodes);
    Check(cudaGraphGetNodes(resources.graph_, handles.data(), &nodes));
    std::array<int, 3> kinds{};  // kernel, memset, memcpy.
    for (auto node : handles) {
      cudaGraphNodeType type;
      Check(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel)
        ++kinds[0];
      else if (type == cudaGraphNodeTypeMemset)
        ++kinds[1];
      else if (type == cudaGraphNodeTypeMemcpy)
        ++kinds[2];
      else
        throw std::runtime_error("Unexpected graph node type");
    }
    if (kinds[0] < 1 || kinds[0] > 32)
      throw std::runtime_error("Counter region kernel bound exceeded");
    // Ncu must use --profile-from-start off. Exactly ONE complete graph lies
    // between these API boundaries. Warmups, allocation and D2H are outside.
    Check(cudaProfilerStart());
    Check(cudaGraphLaunch(resources.executable_, resources.stream_));
    Check(cudaStreamSynchronize(resources.stream_));
    Check(cudaProfilerStop());
    resources.Verify(expected);
    std::ofstream output(argv[7]);
    output << "{\"scope\":\"one instrumented residual invocation; shared GPU "
              "diagnostic only, no roofline extrapolation\","
           << "\"stage\":\"" << stage << "\",\"method\":\"" << method
           << "\",\"ns\":" << state.dimensions_[0]
           << ",\"include_edge\":true,\"api_warmup_calls\":1,\"graph_warmup_"
              "calls\":1,"
           << "\"profile_region_graph_calls\":1,\"graph_nodes\":" << nodes
           << ",\"kernel_nodes\":" << kinds[0]
           << ",\"memset_nodes\":" << kinds[1]
           << ",\"memcpy_nodes\":" << kinds[2]
           << ",\"all_output_bits_exact\":true,\"kernel_metric_traffic_"
              "excludes_copy_engine_nodes\":true}\n";
    output.close();
    if (!output) throw std::runtime_error("Counter report write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
