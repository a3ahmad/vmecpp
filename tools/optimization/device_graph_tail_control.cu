// SPDX-License-Identifier: MIT
// Isolated control: full-block barriers inside device-launched ordinary graphs.
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/array>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VMECPP_GRAPH_CONTINUE_BARRIER
#define VMECPP_GRAPH_CONTINUE_BARRIER 1
#endif
#ifndef VMECPP_GRAPH_DISPATCH_BARRIER
#define VMECPP_GRAPH_DISPATCH_BARRIER 1
#endif
#ifndef VMECPP_GRAPH_SCAN_COUNT
#define VMECPP_GRAPH_SCAN_COUNT 521
#endif
#ifndef VMECPP_GRAPH_KEY_STRIDE
#define VMECPP_GRAPH_KEY_STRIDE 128
#endif

namespace {
constexpr int kThreads = 128;
constexpr int kCount = VMECPP_GRAPH_SCAN_COUNT;
constexpr int kKeyStride = VMECPP_GRAPH_KEY_STRIDE;
struct State {
  int calls_;
  int failures_;
  int launch_error_;
  int target_;
};
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void OwnScan(const double* input, double* m_output,
                        const State* state) {
  __shared__ cuda::std::array<double, kThreads> values;
  const int lane = threadIdx.x;
  const int index = blockIdx.x * kThreads + lane;
  values[lane] = index < kCount ? input[index] * (state->calls_ + 1) : 0.0;
  __syncthreads();
  for (int offset = 1; offset < kThreads; offset *= 2) {
    const double previous = values[lane];
    const double earlier = lane >= offset ? values[lane - offset] : 0.0;
    __syncthreads();
    values[lane] = lane >= offset ? __dadd_rn(earlier, previous) : previous;
    __syncthreads();
  }
  if (index < kCount) m_output[index] = values[lane];
}
template <cub::BlockScanAlgorithm kAlgorithm>
__global__ void CubScan(const double* input, double* m_output,
                        const State* state) {
  using Scan = cub::BlockScan<double, kThreads, kAlgorithm>;
  __shared__ typename Scan::TempStorage storage;
  const int index = blockIdx.x * kThreads + threadIdx.x;
  const double value =
      index < kCount ? input[index] * (state->calls_ + 1) : 0.0;
  double scanned = 0.0;
  Scan(storage).InclusiveSum(value, scanned);
  if (index < kCount) m_output[index] = scanned;
}
__global__ void VerifyIteration(const double* output, State* m_state) {
  const int index = blockIdx.x * kThreads + threadIdx.x;
  if (index < kCount &&
      output[index] != (m_state->calls_ + 1) * (index % kKeyStride + 1.0))
    atomicAdd(&m_state->failures_, 1);
}
__global__ void ScaleInput(double* m_input, const State* state) {
  const int index = blockIdx.x * kThreads + threadIdx.x;
  if (index < kCount) m_input[index] = state->calls_ + 1.0;
}
__global__ void Continue(State* m_state, bool self_launch) {
  if (threadIdx.x == 0) {
    ++m_state->calls_;
    if (self_launch && m_state->calls_ < m_state->target_) {
      const cudaError_t status =
          cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
      if (status != cudaSuccess) m_state->launch_error_ = status;
    }
  }
#if VMECPP_GRAPH_CONTINUE_BARRIER
  __syncthreads();
#endif
}
__global__ void Dispatch(cudaGraphExec_t body, State* m_state) {
  if (threadIdx.x == 0 && m_state->target_ > 0) {
    const cudaError_t status = cudaGraphLaunch(body, cudaStreamGraphTailLaunch);
    if (status != cudaSuccess) m_state->launch_error_ = status;
  }
#if VMECPP_GRAPH_DISPATCH_BARRIER
  __syncthreads();
#endif
}
}  // namespace

int main(int argc, char** argv) {
  cudaStream_t stream = nullptr;
  cudaGraph_t body = nullptr, parent = nullptr;
  cudaGraphExec_t body_exec = nullptr, parent_exec = nullptr;
  double* input = nullptr;
  double* output = nullptr;
  double* host_output = nullptr;
  int* keys = nullptr;
  void* temporary = nullptr;
  std::size_t temporary_bytes = 0;
  State* state = nullptr;
  State* host_state = nullptr;
  auto cleanup = [&] {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (parent_exec != nullptr) cudaGraphExecDestroy(parent_exec);
    if (body_exec != nullptr) cudaGraphExecDestroy(body_exec);
    if (parent != nullptr) cudaGraphDestroy(parent);
    if (body != nullptr) cudaGraphDestroy(body);
    if (input != nullptr) cudaFree(input);
    if (output != nullptr) cudaFree(output);
    if (state != nullptr) cudaFree(state);
    if (keys != nullptr) cudaFree(keys);
    if (temporary != nullptr) cudaFree(temporary);
    if (host_output != nullptr) cudaFreeHost(host_output);
    if (host_state != nullptr) cudaFreeHost(host_state);
    if (stream != nullptr) cudaStreamDestroy(stream);
  };
  try {
    if (argc != 3)
      throw std::runtime_error(
          "Usage: control own|cub-raking|cub-warp|device-scan "
          "host|tail|dispatch|tail-wide|dispatch-wide");
    const std::string kernel = argv[1], mode = argv[2];
    if ((kernel != "own" && kernel != "cub-raking" && kernel != "cub-warp" &&
         kernel != "device-scan") ||
        (mode != "host" && mode != "tail" && mode != "dispatch" &&
         mode != "tail-wide" && mode != "dispatch-wide"))
      throw std::runtime_error("Unsupported control mode");
    if (kernel != "device-scan" && kKeyStride != kThreads)
      throw std::runtime_error("Block scan requires keys aligned to its block");
    const bool device = mode != "host";
    const bool self_launch = mode == "tail" || mode == "tail-wide";
    const int controller_threads =
        mode.find("wide") == std::string::npos ? 1 : kThreads;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Check(cudaMalloc(&input, kCount * sizeof(double)));
    Check(cudaMalloc(&output, kCount * sizeof(double)));
    Check(cudaMalloc(&state, sizeof(State)));
    Check(cudaMallocHost(&host_output, kCount * sizeof(double)));
    Check(cudaMallocHost(&host_state, sizeof(State)));
    const std::vector<double> ones(kCount, 1.0);
    Check(cudaMemcpyAsync(input, ones.data(), kCount * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaStreamSynchronize(stream));
    if (kernel == "device-scan") {
      std::vector<int> host_keys(kCount);
      for (int i = 0; i < kCount; ++i) host_keys[i] = i / kKeyStride;
      Check(cudaMalloc(&keys, kCount * sizeof(int)));
      Check(cudaMemcpyAsync(keys, host_keys.data(), kCount * sizeof(int),
                            cudaMemcpyHostToDevice, stream));
      Check(cub::DeviceScan::InclusiveScanByKey(
          nullptr, temporary_bytes, keys, input, output, cub::Sum{}, kCount,
          cub::Equality{}, stream));
      Check(cudaMalloc(&temporary, temporary_bytes));
      Check(cudaStreamSynchronize(stream));
    }
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (kernel == "own")
      OwnScan<<<(kCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          input, output, state);
    else if (kernel == "cub-raking")
      CubScan<cub::BLOCK_SCAN_RAKING>
          <<<(kCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
              input, output, state);
    else if (kernel == "cub-warp")
      CubScan<cub::BLOCK_SCAN_WARP_SCANS>
          <<<(kCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
              input, output, state);
    else {
      ScaleInput<<<(kCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          input, state);
      Check(cub::DeviceScan::InclusiveScanByKey(
          temporary, temporary_bytes, keys, input, output, cub::Sum{}, kCount,
          cub::Equality{}, stream));
    }
    VerifyIteration<<<(kCount + kThreads - 1) / kThreads, kThreads, 0,
                      stream>>>(output, state);
    Continue<<<1, controller_threads, 0, stream>>>(state, self_launch);
    Check(cudaStreamEndCapture(stream, &body));
    Check(cudaGraphInstantiate(
        &body_exec, body, device ? cudaGraphInstantiateFlagDeviceLaunch : 0));
    if (device) {
      Check(cudaGraphUpload(body_exec, stream));
      Check(cudaStreamSynchronize(stream));
      Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      Dispatch<<<1, controller_threads, 0, stream>>>(body_exec, state);
      Check(cudaStreamEndCapture(stream, &parent));
      Check(cudaGraphInstantiate(&parent_exec, parent, 0));
    }
    int checked_iterations = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
      for (const int target : std::array<int, 4>{0, 1, 3, 17}) {
        *host_state = {0, 0, 0, target};
        Check(cudaMemcpyAsync(state, host_state, sizeof(State),
                              cudaMemcpyHostToDevice, stream));
        Check(cudaMemsetAsync(output, 0, kCount * sizeof(double), stream));
        if (self_launch)
          Check(cudaGraphLaunch(parent_exec, stream));
        else
          for (int i = 0; i < target; ++i)
            Check(cudaGraphLaunch(device ? parent_exec : body_exec, stream));
        // No intermediate synchronization: downstream copies must wait for
        // the entire device tail chain, including its self-launched graphs.
        Check(cudaMemcpyAsync(host_output, output, kCount * sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
        Check(cudaMemcpyAsync(host_state, state, sizeof(State),
                              cudaMemcpyDeviceToHost, stream));
        Check(cudaStreamSynchronize(stream));
        if (host_state->calls_ != target || host_state->failures_ != 0 ||
            host_state->launch_error_ != 0)
          throw std::runtime_error(
              "Tail-chain count/iteration/launch gate failed");
        for (int i = 0; i < kCount; ++i)
          if (host_output[i] != target * (i % kKeyStride + 1.0))
            throw std::runtime_error("Downstream output ordering gate failed");
        checked_iterations += target;
      }
    }
    cleanup();
    std::printf(
        "PASS kernel=%s mode=%s cases=12 checked_iterations=%d "
        "reference_block=128 count=%d key_stride=%d downstream_copy=exact\n",
        kernel.c_str(), mode.c_str(), checked_iterations, kCount, kKeyStride);
    return 0;
  } catch (const std::exception& error) {
    if (stream != nullptr) {
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
          status != cudaStreamCaptureStatusNone) {
        cudaGraph_t ignored = nullptr;
        cudaStreamEndCapture(stream, &ignored);
        if (ignored != nullptr) cudaGraphDestroy(ignored);
      }
    }
    cleanup();
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
