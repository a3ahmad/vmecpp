// SPDX-License-Identifier: MIT
// Experimental overlap of ordinary energy diagnostics with independent forces.
#ifndef VMECPP_EXPERIMENT_GPU_ENERGY_OVERLAP_H_
#define VMECPP_EXPERIMENT_GPU_ENERGY_OVERLAP_H_

#include <cuda_runtime.h>

#include <stdexcept>

#include "gpu_lattice_energies.h"

namespace vmecpp_experiment {
class GpuEnergyOverlap {
 public:
  GpuEnergyOverlap() {
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    try {
      Check(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
      Check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    } catch (...) {
      if (ready_ != nullptr) cudaEventDestroy(ready_);
      cudaStreamDestroy(stream_);
      throw;
    }
  }
  ~GpuEnergyOverlap() noexcept {
    SynchronizeForCleanup();
    cudaEventDestroy(done_);
    cudaEventDestroy(ready_);
    cudaStreamDestroy(stream_);
  }
  GpuEnergyOverlap(const GpuEnergyOverlap&) = delete;
  GpuEnergyOverlap& operator=(const GpuEnergyOverlap&) = delete;

  void Enqueue(GpuDiagnosticsDeviceState m_state, cudaStream_t m_main_stream) {
    if (m_main_stream == nullptr)
      throw std::runtime_error("Energy overlap requires an owned main stream");
    // This fork follows all field producers. The auxiliary branch only reads
    // immutable field/mass inputs and writes its own scratch and energy slots.
    Check(cudaEventRecord(ready_, m_main_stream));
    Check(cudaStreamWaitEvent(stream_, ready_, 0));
    m_state.stream_ = stream_;
    LaunchGpuLatticeEnergies(m_state);
    Check(cudaEventRecord(done_, stream_));
  }
  void Join(cudaStream_t m_main_stream) {
    Check(cudaStreamWaitEvent(m_main_stream, done_, 0));
  }
  cudaError_t SynchronizeForCleanup() noexcept {
    // Also covers a host exception after the fork but before the normal join.
    return cudaStreamSynchronize(stream_);
  }

 private:
  static void Check(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  cudaEvent_t done_ = nullptr;
};
}  // namespace vmecpp_experiment
#endif  // VMECPP_EXPERIMENT_GPU_ENERGY_OVERLAP_H_
