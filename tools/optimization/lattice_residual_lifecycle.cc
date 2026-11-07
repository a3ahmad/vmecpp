// SPDX-License-Identifier: MIT
// Synthetic component extents/stream lifetimes only, never solver fidelity.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gpu_lattice_residual.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T>
T Symbol(void* library, const char* name) {
  auto value = reinterpret_cast<T>(dlsym(library, name));
  if (value == nullptr) throw std::runtime_error(name);
  return value;
}
struct Resources {
  cudaStream_t first_ = nullptr;
  cudaStream_t second_ = nullptr;
  GpuIterationDeviceState state_{};
  std::vector<void*> allocations_;
  ~Resources() {
    if (first_ != nullptr) cudaStreamSynchronize(first_);
    if (second_ != nullptr) cudaStreamSynchronize(second_);
    for (void* pointer : allocations_) cudaFree(pointer);
    if (first_ != nullptr) cudaStreamDestroy(first_);
    if (second_ != nullptr) cudaStreamDestroy(second_);
  }
};
std::array<double, 3> Download(const GpuIterationDeviceState& state) {
  std::array<double, 3> result{};
  const auto stream = static_cast<cudaStream_t>(state.stream_);
  Check(cudaMemcpyAsync(result.data(), state.residuals_, sizeof(result),
                        cudaMemcpyDeviceToHost, stream));
  Check(cudaStreamSynchronize(stream));
  return result;
}
void Equal(const std::array<double, 3>& expected,
           const std::array<double, 3>& actual) {
  if (std::memcmp(expected.data(), actual.data(), sizeof(expected)) != 0)
    throw std::runtime_error("Lifecycle residual bit mismatch");
}

void RejectCold(decltype(&LaunchGpuIterationResidualsLattice) launch,
                GpuIterationDeviceState& m_state, bool edge) {
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  bool rejected = false;
  try {
    launch(m_state, edge);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  cudaGraph_t graph = nullptr;
  const auto status = cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr) Check(cudaGraphDestroy(graph));
  Check(status);
  if (!rejected) throw std::runtime_error("Cold capture was not rejected");
}

void WarmCapture(decltype(&LaunchGpuIterationResidualsLattice) launch,
                 GpuIterationDeviceState& m_state, bool edge,
                 const std::array<double, 3>& reference) {
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  launch(m_state, edge);
  Equal(reference, Download(m_state));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  try {
    Check(
        cudaMemsetAsync(m_state.residuals_, 0x7f, 3 * sizeof(double), stream));
    launch(m_state, edge);
  } catch (...) {
    cudaStreamEndCapture(stream, &graph);
    if (graph != nullptr) cudaGraphDestroy(graph);
    throw;
  }
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphInstantiate(&executable, graph, 0));
  Check(cudaGraphLaunch(executable, stream));
  Equal(reference, Download(m_state));
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: lifecycle capture.bin original.so candidate.so");
    void* original = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (original == nullptr) throw std::runtime_error(dlerror());
    void* candidate = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
    if (candidate == nullptr) throw std::runtime_error(dlerror());
    const auto ordered =
        Symbol<decltype(&LaunchGpuIterationResidualsConditional)>(
            original, "LaunchGpuIterationResidualsConditional");
    const auto launch = Symbol<decltype(&LaunchGpuIterationResidualsLattice)>(
        candidate, "LaunchGpuIterationResidualsLattice");
    const auto poison =
        Symbol<decltype(&LaunchGpuIterationResidualsLatticePoisoned)>(
            candidate, "LaunchGpuIterationResidualsLatticePoisoned");
    const auto flags = Symbol<decltype(&ReadGpuLatticeResidualFlags)>(
        candidate, "ReadGpuLatticeResidualFlags");
    auto capture =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[1], "VMECGIT1");
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.first_, cudaStreamNonBlocking));
    Check(cudaStreamCreateWithFlags(&resources.second_, cudaStreamNonBlocking));
    if (resources.first_ == resources.second_)
      throw std::runtime_error("Distinct live streams unexpectedly alias");
    auto& state = resources.state_;
    state.dimensions_ = capture.input_.dimensions_;
    state.stream_ = resources.first_;
    for (int channel = 0; channel < 6; ++channel) {
      Check(cudaMalloc(&state.forces_[channel],
                       capture.input_.sizes_[channel] * sizeof(double)));
      resources.allocations_.push_back(state.forces_[channel]);
      Check(cudaMemcpyAsync(state.forces_[channel],
                            capture.input_.data_[channel],
                            capture.input_.sizes_[channel] * sizeof(double),
                            cudaMemcpyHostToDevice, resources.first_));
    }
    Check(cudaMalloc(&state.residuals_, 3 * sizeof(double)));
    resources.allocations_.push_back(state.residuals_);
    Check(cudaStreamSynchronize(resources.first_));
    int cold_guards = 0;
    int warm_graphs = 0;
    int poisoned = 0;
    for (int ns : {25, 51, 99}) {
      state.stream_ = resources.first_;
      state.dimensions_[0] = ns;
      state.dimensions_[3] = ns - 1;
      state.dimensions_[4] = ns;
      RejectCold(launch, state, true);
      ++cold_guards;
      for (bool edge : {true, false}) {
        ordered(state, edge, nullptr);
        const auto reference = Download(state);
        WarmCapture(launch, state, edge, reference);
        ++warm_graphs;
        // The first stream remains live while rebinding to the distinct second.
        state.stream_ = resources.second_;
        RejectCold(launch, state, edge);
        ++cold_guards;
        WarmCapture(launch, state, edge, reference);
        ++warm_graphs;
        poison(state, edge);
        Equal(reference, Download(state));
        std::array<int, 3> invalid{};
        flags(invalid);
        if (invalid != std::array<int, 3>{1, 1, 1})
          throw std::runtime_error(
              "Poisoned lifecycle proposal escaped fallback");
        ++poisoned;
        state.stream_ = resources.first_;
        WarmCapture(launch, state, edge, reference);
        ++warm_graphs;
      }
    }
    std::cout << "{\"scope\":\"synthetic component extents, not solver "
                 "fidelity configs\","
              << "\"distinct_live_streams\":true,\"cold_capture_rejections\":"
              << cold_guards << ",\"warm_graph_cases\":" << warm_graphs
              << ",\"poisoned_fallback_cases\":" << poisoned
              << ",\"extents\":[25,51,99],\"all_output_bits_exact\":true}\n";
    // Keep original/candidate libraries mapped until their TLS cleanup
    // finishes.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
