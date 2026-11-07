// SPDX-License-Identifier: MIT
// Experimental integrated offload. The generated kernel derives from the
// repository's existing CUDA implementation, preserving its attribution.
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "generated_fourier_kernel.cuh"
#include "gpu_fourier.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

class Workspace {
 public:
  Workspace() { Check(cudaStreamCreate(&stream_)); }
  ~Workspace() {
    cudaFree(device_input_);
    cudaFree(device_output_);
    cudaFree(device_partials_);
    cudaFreeHost(host_input_);
    cudaFreeHost(host_output_);
    cudaStreamDestroy(stream_);
  }
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  void Resize(std::size_t inputs, std::size_t outputs, std::size_t partials) {
    if (partials > partial_capacity_) {
      Check(cudaFree(device_partials_));
      device_partials_ = nullptr;
      Check(cudaMalloc(&device_partials_, partials * sizeof(double)));
      partial_capacity_ = partials;
    }
    if (inputs > input_capacity_) {
      Check(cudaFree(device_input_));
      Check(cudaFreeHost(host_input_));
      device_input_ = nullptr;
      host_input_ = nullptr;
      Check(cudaMalloc(&device_input_, inputs * sizeof(double)));
      Check(cudaMallocHost(&host_input_, inputs * sizeof(double)));
      input_capacity_ = inputs;
    }
    if (outputs > output_capacity_) {
      Check(cudaFree(device_output_));
      Check(cudaFreeHost(host_output_));
      device_output_ = nullptr;
      host_output_ = nullptr;
      Check(cudaMalloc(&device_output_, outputs * sizeof(double)));
      Check(cudaMallocHost(&host_output_, outputs * sizeof(double)));
      output_capacity_ = outputs;
    }
  }

  cudaStream_t stream_;
  double* device_input_ = nullptr;
  double* device_output_ = nullptr;
  double* device_partials_ = nullptr;
  double* host_input_ = nullptr;
  double* host_output_ = nullptr;

 private:
  std::size_t input_capacity_ = 0;
  std::size_t output_capacity_ = 0;
  std::size_t partial_capacity_ = 0;
};
}  // namespace

extern "C" void LaunchGpuFourierDevice(GpuFourierDeviceState& m_state) {
  const auto& d = m_state.dimensions_;
  const auto& device_inputs = m_state.input_;
  const auto& device_outputs = m_state.output_;
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  const auto outputs = std::accumulate(m_state.output_sizes_.begin(),
      m_state.output_sizes_.end(), std::size_t{0});
  std::size_t offset = 0;
  for (std::size_t i = 0; i < device_outputs.size(); ++i) {
    if (device_outputs[i] != device_outputs[0] + offset) {
      throw std::runtime_error("Device inverse requires contiguous output channels");
    }
    offset += m_state.output_sizes_[i];
  }
  Check(cudaMemsetAsync(device_outputs[0], 0, outputs * sizeof(double),
                        stream));
  KernelParams params{d[0], d[1], d[2], d[3],  d[4],  d[5],      d[6],
                      d[7], d[8], d[9], d[10], d[11], d[12] != 0};
#ifdef VMECPP_GPU_SEPARABLE
  int tasks = (d[3] - d[2]) * d[5] * d[7];
#ifdef VMECPP_GPU_PARTIAL_COMPONENTS
  tasks *= 12;
#endif
  ToroidalPartialsKernel<<<(tasks + 127) / 128, 128, 0, stream>>>(
      device_inputs[0], device_inputs[1], device_inputs[2], device_inputs[3],
      device_inputs[4], device_inputs[5], device_inputs[12], device_inputs[13],
      device_inputs[14], device_inputs[15], m_state.partials_, params);
  Check(cudaGetLastError());
#endif
#ifdef VMECPP_GPU_FLAT_GRID
#ifndef VMECPP_GPU_BLOCK_SIZE
#define VMECPP_GPU_BLOCK_SIZE 256
#endif
  const int block_size = VMECPP_GPU_BLOCK_SIZE;
  const int points = (d[3] - d[2]) * d[7] * d[9];
  const dim3 grid((points + block_size - 1) / block_size,
                  VMECPP_INVERSE_SPLIT_OUTPUT_GROUPS ? 2 : 1);
#else
  const dim3 grid(d[3] - d[2], d[7]);
  // One warp per independent radial/toroidal slice, with serial m/n sums.
  const int block_size = ((d[9] + 31) / 32) * 32;
#endif
  FourierToRealKernel<<<grid, block_size, 0, stream>>>(
#ifdef VMECPP_GPU_SEPARABLE
      m_state.partials_,
#endif
      device_inputs[0], device_inputs[1], device_inputs[2], device_inputs[3],
      device_inputs[4], device_inputs[5], device_inputs[6], device_inputs[7],
      device_inputs[8], device_inputs[9], device_inputs[10], device_inputs[11],
      device_inputs[12], device_inputs[13], device_inputs[14],
      device_inputs[15], device_outputs[0], device_outputs[1],
      device_outputs[2], device_outputs[3], device_outputs[4],
      device_outputs[5], device_outputs[6], device_outputs[7],
      device_outputs[8], device_outputs[9], device_outputs[10],
      device_outputs[11], device_outputs[12], device_outputs[13],
      device_outputs[14], device_outputs[15], device_outputs[16],
      device_outputs[17], params);
  Check(cudaGetLastError());
}

namespace {
void RunGpuFourierInternal(const GpuFourierInput& input,
                           GpuFourierOutput& m_output, bool materialize,
                           GpuDeviceFourierOutput* m_device) {
  thread_local Workspace workspace;
  const std::size_t inputs =
      std::accumulate(input.sizes_.begin(), input.sizes_.end(), std::size_t{0});
  const std::size_t outputs = std::accumulate(
      m_output.sizes_.begin(), m_output.sizes_.end(), std::size_t{0});
  const auto& d = input.dimensions_;
  std::size_t partials = 0;
#ifdef VMECPP_GPU_SEPARABLE
  partials = std::size_t{12} * (d[3] - d[2]) * d[5] * d[7];
#endif
  workspace.Resize(inputs, outputs, partials);
  std::array<const double*, 16> device_inputs;
  std::array<double*, 18> device_outputs;
  std::size_t offset = 0;
  for (std::size_t index = 0; index < input.sizes_.size(); ++index) {
    std::memcpy(workspace.host_input_ + offset, input.data_[index],
                input.sizes_[index] * sizeof(double));
    device_inputs[index] = workspace.device_input_ + offset;
    offset += input.sizes_[index];
  }
  offset = 0;
  for (std::size_t index = 0; index < m_output.sizes_.size(); ++index) {
    device_outputs[index] = workspace.device_output_ + offset;
    offset += m_output.sizes_[index];
  }
  Check(cudaMemcpyAsync(workspace.device_input_, workspace.host_input_,
                        inputs * sizeof(double), cudaMemcpyHostToDevice,
                        workspace.stream_));
  GpuFourierDeviceState state{device_inputs, device_outputs, m_output.sizes_,
      input.dimensions_, workspace.device_partials_, workspace.stream_};
  LaunchGpuFourierDevice(state);
  if (m_device != nullptr) {
    m_device->data_ = device_outputs;
    m_device->sizes_ = m_output.sizes_;
    m_device->dimensions_ = d;
    m_device->stream_ = workspace.stream_;
  }
  if (!materialize) {
    return;
  }
  Check(cudaMemcpyAsync(workspace.host_output_, workspace.device_output_,
                        outputs * sizeof(double), cudaMemcpyDeviceToHost,
                        workspace.stream_));
  Check(cudaStreamSynchronize(workspace.stream_));
  offset = 0;
  for (std::size_t index = 0; index < m_output.sizes_.size(); ++index) {
    std::memcpy(m_output.data_[index], workspace.host_output_ + offset,
                m_output.sizes_[index] * sizeof(double));
    offset += m_output.sizes_[index];
  }
  // Capture one real final-grid call for inexpensive, exact kernel replay.
  const char* capture_path = std::getenv("VMECPP_GPU_CAPTURE");
  static thread_local bool captured = false;
  if (capture_path != nullptr && !captured && d[3] - d[2] >= 99) {
    const std::array<char, 8> kMagic{'V', 'M', 'E', 'C', 'G', 'P', 'U', '1'};
    static_assert(sizeof(std::size_t) == 8);
    static_assert(sizeof(int) == 4);
    std::ofstream capture(capture_path, std::ios::binary);
    capture.write(kMagic.data(), kMagic.size());
    capture.write(reinterpret_cast<const char*>(d.data()), sizeof(d));
    capture.write(reinterpret_cast<const char*>(input.sizes_.data()),
                  sizeof(input.sizes_));
    capture.write(reinterpret_cast<const char*>(m_output.sizes_.data()),
                  sizeof(m_output.sizes_));
    capture.write(reinterpret_cast<const char*>(workspace.host_input_),
                  inputs * sizeof(double));
    capture.write(reinterpret_cast<const char*>(workspace.host_output_),
                  outputs * sizeof(double));
    if (!capture) {
      throw std::runtime_error("Failed to capture GPU replay fixture");
    }
    captured = true;
  }
}
}  // namespace

extern "C" void RunGpuFourier(const GpuFourierInput& input,
                              GpuFourierOutput& m_output) {
  RunGpuFourierInternal(input, m_output, true, nullptr);
}

extern "C" void RunGpuFourierResident(const GpuFourierInput& input,
                                      GpuFourierOutput& m_shape,
                                      GpuDeviceFourierOutput& m_output) {
  if (std::getenv("VMECPP_GPU_CAPTURE") != nullptr) {
    throw std::runtime_error(
        "Resident transform uses the explicit reference capture adapter");
  }
  RunGpuFourierInternal(input, m_shape, false, &m_output);
}
