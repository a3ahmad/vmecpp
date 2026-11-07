// SPDX-License-Identifier: MIT
// The proposal is speculative; the complete certificate and ORIGINAL ordered
// residual fallback determine every reported scalar. Experimental opt-in path.
#include <cuda_runtime.h>

#include <array>
#include <cuda/std/array>
#include <stdexcept>

#include "gpu_lattice_prefix.h"
#include "gpu_lattice_residual.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Parameters {
  cuda::std::array<const double*, 6> forces_;
  double* terms_;
  int stride_;
  int rz_count_;
};
__global__ void PrepareTerms(Parameters m_p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 3 * m_p.stride_) return;
  const int component = index / m_p.stride_;
  const int local = index % m_p.stride_;
  const int coefficient = local / 2;
  double term = 0.0;
  if (component == 2 || coefficient < m_p.rz_count_) {
    const double force = m_p.forces_[2 * component + local % 2][coefficient];
    term = __dmul_rn(force, force);
  }
  // Excluded R/Z LCFS terms are trailing +0, exactly as the fixed-round path.
  m_p.terms_[index] = term;
}

struct Workspace {
  int stride_ = 0;
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  void* context_ = nullptr;
  void Clear() {
    if (context_ != nullptr) DestroyGpuLatticePrefix(context_);
    context_ = nullptr;
    if (terms_ != nullptr) cudaFree(terms_);
    if (prefix_ != nullptr) cudaFree(prefix_);
    terms_ = prefix_ = nullptr;
    stride_ = 0;
  }
  ~Workspace() { Clear(); }
  void Resize(int stride, cudaStream_t stream, bool capturing) {
    int device;
    Check(cudaGetDevice(&device));
    if (stride_ != 0 && device != device_)
      throw std::runtime_error("Lattice workspace cannot change device");
    if (stride == stride_) {
      if (stream != stream_) {
        if (capturing)
          throw std::runtime_error("Warm lattice stream before capture");
        Check(cudaDeviceSynchronize());
        RebindGpuLatticePrefix(context_, stream);
        stream_ = stream;
      }
      return;
    }
    if (capturing)
      throw std::runtime_error("Warm lattice dimensions before capture");
    Clear();
    device_ = device;
    stream_ = stream;
    try {
      Check(cudaMalloc(&terms_, 3 * std::size_t(stride) * sizeof(double)));
      Check(cudaMalloc(&prefix_, 3 * std::size_t(stride) * sizeof(double)));
      context_ = CreateGpuLatticePrefix(3, stride, stream);
      stride_ = stride;
    } catch (...) {
      Clear();
      throw;
    }
  }
};
thread_local Workspace workspace;

void Launch(GpuIterationDeviceState& m_state, bool include_edge_rz,
            bool corrupt) {
  const auto& dimensions = m_state.dimensions_;
  if (dimensions[0] < 2 || dimensions[1] < 1 || dimensions[2] < 0 ||
      dimensions[4] < 1 || dimensions[4] > dimensions[0] ||
      dimensions[1] > 65536 || dimensions[2] >= 65536 || dimensions[4] > 65536)
    throw std::runtime_error("Invalid lattice residual dimensions");
  const std::size_t coefficients = std::size_t(dimensions[4]) * dimensions[1] *
                                   (std::size_t(dimensions[2]) + 1);
  if (coefficients > (1 << 21))
    throw std::runtime_error("Lattice residual exceeds bounded prototype size");
  const std::size_t rz_count =
      include_edge_rz ? coefficients
                      : std::size_t(dimensions[0] - 1) * dimensions[1] *
                            (std::size_t(dimensions[2]) + 1);
  if (rz_count > coefficients || m_state.residuals_ == nullptr ||
      m_state.stream_ == nullptr)
    throw std::runtime_error("Invalid lattice residual extent/output/stream");
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Lattice stream capture invalidated");
  workspace.Resize(static_cast<int>(2 * coefficients), stream,
                   capture == cudaStreamCaptureStatusActive);
  Parameters p{};
  for (int i = 0; i < 6; ++i) {
    if (m_state.forces_[i] == nullptr)
      throw std::runtime_error("Null lattice force input");
    p.forces_[i] = m_state.forces_[i];
  }
  p.terms_ = workspace.terms_;
  p.stride_ = workspace.stride_;
  p.rz_count_ = static_cast<int>(rz_count);
  PrepareTerms<<<(3 * p.stride_ + 127) / 128, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
#if defined(VMECPP_LATTICE_FUSE_FINISH)
  ProposeAndFinishGpuLatticePrefix(workspace.context_, workspace.terms_,
                                   workspace.prefix_, m_state.residuals_,
                                   corrupt);
#else
  ProposeGpuLatticePrefix(workspace.context_, workspace.terms_,
                          workspace.prefix_, corrupt);
  FinishGpuLatticePrefix(workspace.context_, workspace.prefix_,
                         m_state.residuals_);
#endif
  LaunchGpuIterationResidualsConditional(
      m_state, include_edge_rz,
      GpuLatticePrefixInvalidDevice(workspace.context_));
}
}  // namespace

extern "C" void LaunchGpuIterationResidualsLattice(
    GpuIterationDeviceState& m_state, bool include_edge_rz) {
  Launch(m_state, include_edge_rz, false);
}
extern "C" void LaunchGpuIterationResidualsLatticePoisoned(
    GpuIterationDeviceState& m_state, bool include_edge_rz) {
  Launch(m_state, include_edge_rz, true);
}
extern "C" void ReadGpuLatticeResidualStats(
    std::array<unsigned long long, 4>& m_stats) {
  if (workspace.context_ == nullptr)
    throw std::runtime_error("No lattice residual ran");
  ReadGpuLatticePrefixStats(workspace.context_, m_stats);
}
extern "C" void ReadGpuLatticeResidualFlags(std::array<int, 3>& m_flags) {
  if (workspace.context_ == nullptr)
    throw std::runtime_error("No lattice residual ran");
  ReadGpuLatticePrefixFlags(workspace.context_, m_flags.data());
}
