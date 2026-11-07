// SPDX-License-Identifier: MIT
// Experimental ordinary-graph dispatch; not enabled in qualified builds.
#ifndef VMECPP_EXPERIMENT_GPU_DEVICE_GRAPH_DISPATCH_H_
#define VMECPP_EXPERIMENT_GPU_DEVICE_GRAPH_DISPATCH_H_

#include <cuda_runtime.h>

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vmecpp_experiment {
namespace device_dispatch_detail {
inline void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void Dispatch(cudaGraphExec_t good, cudaGraphExec_t bad,
                         const double* jacobian, int* m_status) {
  if (threadIdx.x == 0) {
    const bool invalid = __dmul_rn(jacobian[0], jacobian[1]) < 0.0;
    m_status[1] = invalid;
    m_status[0] =
        cudaGraphLaunch(invalid ? bad : good, cudaStreamGraphTailLaunch);
  }
  // Kept explicitly: isolated factorial controls distinguish this end barrier.
  __syncthreads();
}
}  // namespace device_dispatch_detail

class GpuDeviceGraphDispatch {
 public:
  GpuDeviceGraphDispatch() {
    using device_dispatch_detail::Check;
    Check(cudaMalloc(&device_status_, 2 * sizeof(int)));
    try {
      Check(cudaMallocHost(&host_status_, 2 * sizeof(int)));
    } catch (...) {
      cudaFree(device_status_);
      throw;
    }
  }
  ~GpuDeviceGraphDispatch() {
    // The owning solver synchronizes its stream before destroying these graphs.
    for (const auto& entry : entries_) {
      cudaGraphExecDestroy(entry.executable_);
      cudaGraphDestroy(entry.graph_);
      cudaGraphExecDestroy(entry.good_executable_);
      cudaGraphDestroy(entry.good_graph_);
    }
    if (bad_executable_ != nullptr) cudaGraphExecDestroy(bad_executable_);
    if (bad_graph_ != nullptr) cudaGraphDestroy(bad_graph_);
    cudaFree(device_status_);
    cudaFreeHost(host_status_);
  }
  GpuDeviceGraphDispatch(const GpuDeviceGraphDispatch&) = delete;
  GpuDeviceGraphDispatch& operator=(const GpuDeviceGraphDispatch&) = delete;

  bool Contains(const std::array<int, 8>& key) const {
    for (const auto& entry : entries_)
      if (entry.key_ == key) return true;
    return false;
  }

  template <typename Prelude, typename Good, typename Restore>
  void Prepare(const std::array<int, 8>& key, cudaGraph_t warmed_good_graph,
               cudaStream_t stream, const double* jacobian, Prelude prelude,
               Good good, Restore restore) {
    using device_dispatch_detail::Check;
    if (Contains(key)) return;
    cudaGraph_t graph = nullptr;
    cudaGraph_t good_graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphExec_t good_executable = nullptr;
    bool capturing = false;
    try {
      if (warmed_good_graph == nullptr)
        throw std::runtime_error("Device tail requires a warmed host graph");
      // Capture a distinct graph definition as well as a distinct executable.
      // The warmed host graph only proves that nested workspaces are ready.
      Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      good();
      Check(cudaStreamEndCapture(stream, &good_graph));
      capturing = false;
      Check(cudaGraphInstantiate(&good_executable, good_graph,
                                 cudaGraphInstantiateFlagDeviceLaunch));
      Check(cudaGraphUpload(good_executable, stream));
      Check(cudaStreamSynchronize(stream));
      if (bad_executable_ == nullptr) {
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        capturing = true;
        restore();
        Check(cudaStreamEndCapture(stream, &bad_graph_));
        capturing = false;
        Check(cudaGraphInstantiate(&bad_executable_, bad_graph_,
                                   cudaGraphInstantiateFlagDeviceLaunch));
        Check(cudaGraphUpload(bad_executable_, stream));
        Check(cudaStreamSynchronize(stream));
      }
      Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      prelude();
      device_dispatch_detail::Dispatch<<<1, 1, 0, stream>>>(
          good_executable, bad_executable_, jacobian, device_status_);
      Check(cudaStreamEndCapture(stream, &graph));
      capturing = false;
      Check(cudaGraphInstantiate(&executable, graph, 0));
      entries_.push_back({key, graph, executable, good_executable, good_graph});
    } catch (...) {
      if (capturing) {
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(stream, &abandoned);
        if (abandoned != nullptr) cudaGraphDestroy(abandoned);
      }
      if (executable != nullptr) cudaGraphExecDestroy(executable);
      if (graph != nullptr) cudaGraphDestroy(graph);
      if (good_executable != nullptr) cudaGraphExecDestroy(good_executable);
      if (good_graph != nullptr) cudaGraphDestroy(good_graph);
      throw;
    }
  }

  void Launch(const std::array<int, 8>& key, cudaStream_t stream) {
    using device_dispatch_detail::Check;
    for (const auto& entry : entries_) {
      if (entry.key_ != key) continue;
      Check(cudaGraphLaunch(entry.executable_, stream));
      // This copy is dependent on completion of the entire device tail.
      Check(cudaMemcpyAsync(host_status_, device_status_, 2 * sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
      return;
    }
    throw std::runtime_error("Device dispatch graph was not prepared");
  }

  bool BadJacobianAfterSynchronization() const {
    device_dispatch_detail::Check(static_cast<cudaError_t>(host_status_[0]));
    return host_status_[1] != 0;
  }

 private:
  struct Entry {
    std::array<int, 8> key_;
    cudaGraph_t graph_;
    cudaGraphExec_t executable_;
    cudaGraphExec_t good_executable_;
    cudaGraph_t good_graph_;
  };
  std::vector<Entry> entries_;
  cudaGraph_t bad_graph_ = nullptr;
  cudaGraphExec_t bad_executable_ = nullptr;
  int* device_status_ = nullptr;
  int* host_status_ = nullptr;
};
}  // namespace vmecpp_experiment
#endif  // VMECPP_EXPERIMENT_GPU_DEVICE_GRAPH_DISPATCH_H_
