// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Experimental separable forward transform, derived from VMEC++ 0.4.11.
// Each l and k reduction retains the reference order; FMA must be disabled.
#include <cuda_runtime.h>
#ifndef VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR
#define VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR 0
#endif
#if VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR && !defined(VMECPP_FORWARD_SPLIT_GROUPS)
#error "Output clear elision requires the selected three-group kernel"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gpu_forward.h"
#ifdef VMECPP_GPU_PIPELINE
#include "gpu_pipeline.h"
#include "mhd_point.h"
#include "radial_executor.h"
#endif

#ifdef VMECPP_FORWARD_SPLIT_GROUPS
constexpr int kForwardGroups = 3;
#else
constexpr int kForwardGroups = 1;
#endif

namespace {
struct ForwardPointers {
  const double* armn_e_;
  const double* armn_o_;
  const double* azmn_e_;
  const double* azmn_o_;
  const double* blmn_e_;
  const double* blmn_o_;
  const double* brmn_e_;
  const double* brmn_o_;
  const double* bzmn_e_;
  const double* bzmn_o_;
  const double* clmn_e_;
  const double* clmn_o_;
  const double* crmn_e_;
  const double* crmn_o_;
  const double* czmn_e_;
  const double* czmn_o_;
  const double* frcon_e_;
  const double* frcon_o_;
  const double* fzcon_e_;
  const double* fzcon_o_;
  const double* xmpq_;
  const double* cosmui_;
  const double* sinmui_;
  const double* cosmumi_;
  const double* sinmumi_;
  const double* cosnv_;
  const double* sinnv_;
  const double* cosnvn_;
  const double* sinnvn_;
};

struct ForwardParameters {
  int ns_min_f_;
  int ns_max_f_;
  int ns_max_including_lcfs_;
  int ns_;
  int mpol_;
  int ntor_;
  int n_zeta_;
  int n_theta_eff_;
  int n_theta_reduced_;
  int nnyq2_;
};

#ifdef VMECPP_FORWARD_SPLIT_GROUPS
#include "gpu_forward_groups_kernels.h"
__device__ int forward_fp32_enabled = 0;
__device__ unsigned long long forward_precision_counts[2] = {0, 0};
#include "gpu_forward_precision_kernels.cuh"

#else
__global__ void ForwardPartials(ForwardPointers d, ForwardParameters p,
                                double* __restrict__ m_partials) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int count =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * p.n_zeta_;
  if (task >= count) {
    return;
  }
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % p.mpol_;
  const int jF = task / (p.n_zeta_ * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) {
    return;
  }
  const bool even = m % 2 == 0;
  const double* armn = even ? d.armn_e_ : d.armn_o_;
  const double* azmn = even ? d.azmn_e_ : d.azmn_o_;
  const double* blmn = even ? d.blmn_e_ : d.blmn_o_;
  const double* brmn = even ? d.brmn_e_ : d.brmn_o_;
  const double* bzmn = even ? d.bzmn_e_ : d.bzmn_o_;
  const double* clmn = even ? d.clmn_e_ : d.clmn_o_;
  const double* crmn = even ? d.crmn_e_ : d.crmn_o_;
  const double* czmn = even ? d.czmn_e_ : d.czmn_o_;
  const double* frcon = even ? d.frcon_e_ : d.frcon_o_;
  const double* fzcon = even ? d.fzcon_e_ : d.fzcon_o_;
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  double rmkcc = 0.0, rmkcc_n = 0.0, rmkss = 0.0, rmkss_n = 0.0;
  double zmksc = 0.0, zmksc_n = 0.0, zmkcs = 0.0, zmkcs_n = 0.0;
  double lmksc = 0.0, lmksc_n = 0.0, lmkcs = 0.0, lmkcs_n = 0.0;
  const int base = ((jF - p.ns_min_f_) * p.n_zeta_ + k) * p.n_theta_eff_;
  const int mbase = m * p.n_theta_reduced_;
  for (int l = 0; l < p.n_theta_reduced_; ++l) {
    const int idx = base + l;
    const int idx_ml = mbase + l;
    const double cosmui = d.cosmui_[idx_ml];
    const double sinmui = d.sinmui_[idx_ml];
    const double cosmumi = d.cosmumi_[idx_ml];
    const double sinmumi = d.sinmumi_[idx_ml];
    lmksc += blmn[idx] * cosmumi;
    lmkcs += blmn[idx] * sinmumi;
    lmkcs_n -= clmn[idx] * cosmui;
    lmksc_n -= clmn[idx] * sinmui;
    if (rz) {
      rmkcc_n -= crmn[idx] * cosmui;
      zmkcs_n -= czmn[idx] * cosmui;
      rmkss_n -= crmn[idx] * sinmui;
      zmksc_n -= czmn[idx] * sinmui;
      const double tempR = armn[idx] + d.xmpq_[m] * frcon[idx];
      const double tempZ = azmn[idx] + d.xmpq_[m] * fzcon[idx];
      rmkcc += tempR * cosmui + brmn[idx] * sinmumi;
      rmkss += tempR * sinmui + brmn[idx] * cosmumi;
      zmksc += tempZ * sinmui + bzmn[idx] * cosmumi;
      zmkcs += tempZ * cosmui + bzmn[idx] * sinmumi;
    }
  }
  m_partials[0 * count + task] = rmkcc;
  m_partials[1 * count + task] = rmkcc_n;
  m_partials[2 * count + task] = rmkss;
  m_partials[3 * count + task] = rmkss_n;
  m_partials[4 * count + task] = zmksc;
  m_partials[5 * count + task] = zmksc_n;
  m_partials[6 * count + task] = zmkcs;
  m_partials[7 * count + task] = zmkcs_n;
  m_partials[8 * count + task] = lmksc;
  m_partials[9 * count + task] = lmksc_n;
  m_partials[10 * count + task] = lmkcs;
  m_partials[11 * count + task] = lmkcs_n;
}

__global__ void ForwardFinish(
    ForwardPointers d, ForwardParameters p, const double* __restrict__ partials,
    double* __restrict__ m_frcc, double* __restrict__ m_frss,
    double* __restrict__ m_fzsc, double* __restrict__ m_fzcs,
    double* __restrict__ m_flsc, double* __restrict__ m_flcs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * p.n_zeta_;
  const int points =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * (p.ntor_ + 1);
  if (index >= points) {
    return;
  }
  const int n = index % (p.ntor_ + 1);
  const int m = (index / (p.ntor_ + 1)) % p.mpol_;
  const int jF = index / ((p.ntor_ + 1) * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) {
    return;
  }
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  double frcc = 0.0, frss = 0.0, fzsc = 0.0, fzcs = 0.0;
  double flsc = 0.0, flcs = 0.0;
  for (int k = 0; k < p.n_zeta_; ++k) {
    const int task = ((jF - p.ns_min_f_) * p.mpol_ + m) * p.n_zeta_ + k;
    const int idx_kn = k * (p.nnyq2_ + 1) + n;
    const double cosnv = d.cosnv_[idx_kn];
    const double sinnv = d.sinnv_[idx_kn];
    const double cosnvn = d.cosnvn_[idx_kn];
    const double sinnvn = d.sinnvn_[idx_kn];
    if (rz) {
      frcc += partials[0 * count + task] * cosnv +
              partials[1 * count + task] * sinnvn;
      frss += partials[2 * count + task] * sinnv +
              partials[3 * count + task] * cosnvn;
      fzsc += partials[4 * count + task] * cosnv +
              partials[5 * count + task] * sinnvn;
      fzcs += partials[6 * count + task] * sinnv +
              partials[7 * count + task] * cosnvn;
    }
    if (jF >= 1) {
      flsc += partials[8 * count + task] * cosnv +
              partials[9 * count + task] * sinnvn;
      flcs += partials[10 * count + task] * sinnv +
              partials[11 * count + task] * cosnvn;
    }
  }
  m_frcc[index] = frcc;
  m_frss[index] = frss;
  m_fzsc[index] = fzsc;
  m_fzcs[index] = fzcs;
  m_flsc[index] = flsc;
  m_flcs[index] = flcs;
}

#endif

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

class ForwardWorkspace {
 public:
  ForwardWorkspace() { Check(cudaStreamCreate(&stream_)); }
  ~ForwardWorkspace() {
    ResetGraph();
    cudaFree(input_);
    cudaFree(output_);
    cudaFree(partials_);
    cudaFree(forces_);
    cudaFreeHost(host_input_);
    cudaFreeHost(host_output_);
    cudaStreamDestroy(stream_);
  }
  ForwardWorkspace(const ForwardWorkspace&) = delete;
  ForwardWorkspace& operator=(const ForwardWorkspace&) = delete;
  void ResetGraph() {
    if (graph_exec_ != nullptr) {
      cudaGraphExecDestroy(graph_exec_);
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
    }
    graph_exec_ = nullptr;
    graph_ = nullptr;
  }
  void Resize(std::size_t inputs, std::size_t outputs, std::size_t partials,
              std::size_t forces = 0) {
    if (inputs > inputs_ || outputs > outputs_ || partials > partial_count_ ||
        forces > force_count_) {
      ResetGraph();
    }
    if (inputs > inputs_) {
      Check(cudaFree(input_));
      input_ = nullptr;
      Check(cudaFreeHost(host_input_));
      host_input_ = nullptr;
      Check(cudaMalloc(&input_, inputs * sizeof(double)));
      Check(cudaMallocHost(&host_input_, inputs * sizeof(double)));
      inputs_ = inputs;
    }
    if (outputs > outputs_) {
      Check(cudaFree(output_));
      output_ = nullptr;
      Check(cudaFreeHost(host_output_));
      host_output_ = nullptr;
      Check(cudaMalloc(&output_, outputs * sizeof(double)));
      Check(cudaMallocHost(&host_output_, outputs * sizeof(double)));
      outputs_ = outputs;
    }
    if (partials > partial_count_) {
      Check(cudaFree(partials_));
      partials_ = nullptr;
      Check(cudaMalloc(&partials_, partials * sizeof(double)));
      partial_count_ = partials;
    }
    if (forces > force_count_) {
      Check(cudaFree(forces_));
      forces_ = nullptr;
      Check(cudaMalloc(&forces_, forces * sizeof(double)));
      force_count_ = forces;
    }
  }
  cudaStream_t stream_;
  double* input_ = nullptr;
  double* output_ = nullptr;
  double* partials_ = nullptr;
  double* forces_ = nullptr;
  double* host_input_ = nullptr;
  double* host_output_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  std::array<int, 13> graph_dimensions_{};
  std::array<std::size_t, 44> graph_input_sizes_{};
  std::vector<std::size_t> graph_output_sizes_;
  double graph_delta_s_ = 0.0;

 private:
  std::size_t inputs_ = 0;
  std::size_t outputs_ = 0;
  std::size_t partial_count_ = 0;
  std::size_t force_count_ = 0;
};
}  // namespace

// Process/context-global mode, one active equilibrium per process.
// The CPU owner switches only between completed iteration graphs.
extern "C" cudaError_t SetGpuForwardTransformPrecision(int precision,
                                                       void* stream) {
  static int current_precision = 64;
  static const int kEnabled = 1;
  static const int kDisabled = 0;
  if (precision != 32 && precision != 64) return cudaErrorInvalidValue;
  if (precision == current_precision) return cudaSuccess;
  const int* value = precision == 32 ? &kEnabled : &kDisabled;
  const cudaError_t status =
      stream == nullptr
          ? cudaMemcpyToSymbol(forward_fp32_enabled, value, sizeof(int), 0,
                               cudaMemcpyHostToDevice)
          : cudaMemcpyToSymbolAsync(forward_fp32_enabled, value, sizeof(int), 0,
                                    cudaMemcpyHostToDevice,
                                    static_cast<cudaStream_t>(stream));
  if (status == cudaSuccess) current_precision = precision;
  return status;
}

extern "C" cudaError_t GetGpuForwardTransformPrecisionCounts(
    unsigned long long* m_out) {
  if (m_out == nullptr) return cudaErrorInvalidValue;
  return cudaMemcpyFromSymbol(m_out, forward_precision_counts,
                              2 * sizeof(unsigned long long), 0,
                              cudaMemcpyDeviceToHost);
}

extern "C" void LaunchGpuForwardDevice(GpuForwardDeviceState& m_state) {
  const auto& d = m_state.dimensions_;
  const auto& ptr = m_state.input_;
  const auto& out = m_state.output_;
  if (d[12] != 0 || d[0] != 0 || d[2] != 0 || d[4] <= 0 || d[5] <= 0 ||
      d[6] < 0 || d[7] <= 0 || d[9] <= 0) {
    throw std::runtime_error(
        "Resident forward transform requires one fixed-boundary partition");
  }
  const int tasks = (d[4] - d[0]) * d[5] * d[7];
  const int points = (d[4] - d[0]) * d[5] * (d[6] + 1);
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (m_state.output_sizes_[i] != static_cast<std::size_t>(points) ||
        out[i] != out[0] + i * points) {
      throw std::runtime_error(
          "Resident forward outputs must be contiguous equal-sized channels");
    }
  }
  const ForwardPointers pointers{
      ptr[0],  ptr[1],  ptr[2],  ptr[3],  ptr[4],  ptr[5],  ptr[6],  ptr[7],
      ptr[8],  ptr[9],  ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15],
      ptr[16], ptr[17], ptr[18], ptr[19], ptr[20], ptr[21], ptr[22], ptr[23],
      ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]};
  const ForwardParameters parameters{d[0], d[1], d[4], d[11], d[5],
                                     d[6], d[7], d[8], d[9],  d[10]};
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
#if !VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR
  Check(cudaMemsetAsync(out[0], 0, 6 * std::size_t(points) * sizeof(double),
                        stream));
#endif
  ForwardPartialsMixedPrecision<<<dim3((tasks + 127) / 128, kForwardGroups),
                                  128, 0, stream>>>(pointers, parameters,
                                                    m_state.partials_);
  Check(cudaGetLastError());
  ForwardFinishMixedPrecision<<<dim3((points + 127) / 128, kForwardGroups), 128,
                                0, stream>>>(pointers, parameters,
                                             m_state.partials_, out[0], out[1],
                                             out[2], out[3], out[4], out[5]);
  Check(cudaGetLastError());
}

extern "C" void RunGpuForward(const GpuForwardInput& input,
                              GpuForwardOutput& m_output) {
  const auto& d = input.dimensions_;
  if (d[12] != 0 || d[0] != 0 || d[2] != 0 || d[4] <= 0 || d[5] <= 0 ||
      d[6] < 0 || d[7] <= 0 || d[9] <= 0) {
    throw std::runtime_error(
        "Forward prototype requires a single fixed-boundary partition");
  }
  thread_local ForwardWorkspace workspace;
  const std::size_t inputs =
      std::accumulate(input.sizes_.begin(), input.sizes_.end(), std::size_t{0});
  const std::size_t outputs = std::accumulate(
      m_output.sizes_.begin(), m_output.sizes_.end(), std::size_t{0});
  const int tasks = (d[4] - d[0]) * d[5] * d[7];
  const int points = (d[4] - d[0]) * d[5] * (d[6] + 1);
  workspace.Resize(inputs, outputs, std::size_t{12} * tasks);
  std::array<const double*, 29> ptr;
  std::array<double*, 6> out;
  std::size_t offset = 0;
  for (std::size_t index = 0; index < ptr.size(); ++index) {
    std::memcpy(workspace.host_input_ + offset, input.data_[index],
                input.sizes_[index] * sizeof(double));
    ptr[index] = workspace.input_ + offset;
    offset += input.sizes_[index];
  }
  offset = 0;
  for (std::size_t index = 0; index < out.size(); ++index) {
    out[index] = workspace.output_ + offset;
    offset += m_output.sizes_[index];
  }
  const ForwardPointers pointers{
      ptr[0],  ptr[1],  ptr[2],  ptr[3],  ptr[4],  ptr[5],  ptr[6],  ptr[7],
      ptr[8],  ptr[9],  ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15],
      ptr[16], ptr[17], ptr[18], ptr[19], ptr[20], ptr[21], ptr[22], ptr[23],
      ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]};
  const ForwardParameters parameters{d[0], d[1], d[4], d[11], d[5],
                                     d[6], d[7], d[8], d[9],  d[10]};
  Check(cudaMemcpyAsync(workspace.input_, workspace.host_input_,
                        inputs * sizeof(double), cudaMemcpyHostToDevice,
                        workspace.stream_));
  Check(cudaMemsetAsync(workspace.output_, 0, outputs * sizeof(double),
                        workspace.stream_));
  ForwardPartialsMixedPrecision<<<dim3((tasks + 127) / 128, kForwardGroups),
                                  128, 0, workspace.stream_>>>(
      pointers, parameters, workspace.partials_);
  Check(cudaGetLastError());
  ForwardFinishMixedPrecision<<<dim3((points + 127) / 128, kForwardGroups), 128,
                                0, workspace.stream_>>>(
      pointers, parameters, workspace.partials_, out[0], out[1], out[2], out[3],
      out[4], out[5]);
  Check(cudaGetLastError());
  Check(cudaMemcpyAsync(workspace.host_output_, workspace.output_,
                        outputs * sizeof(double), cudaMemcpyDeviceToHost,
                        workspace.stream_));
  Check(cudaStreamSynchronize(workspace.stream_));
  offset = 0;
  for (std::size_t index = 0; index < out.size(); ++index) {
    std::memcpy(m_output.data_[index], workspace.host_output_ + offset,
                m_output.sizes_[index] * sizeof(double));
    offset += m_output.sizes_[index];
  }
}

#ifdef VMECPP_GPU_PIPELINE
namespace {
std::array<const double*, 44> UploadPipelineInputs(
    const GpuPipelineInput& input, ForwardWorkspace& m_workspace,
    bool enqueue) {
  const char* mode = std::getenv("VMECPP_GPU_UPLOAD_MODE");
  const bool batch = mode != nullptr && std::strcmp(mode, "batch") == 0;
  const bool direct = mode != nullptr && std::strcmp(mode, "direct") == 0;
  const bool parallel = mode != nullptr && std::strcmp(mode, "parallel") == 0;
  if (mode != nullptr && !batch && !direct && !parallel &&
      std::strcmp(mode, "packed") != 0) {
    throw std::runtime_error("Unknown GPU upload mode");
  }
  if (!enqueue && (batch || direct)) {
    throw std::runtime_error("Graph upload requires packed or parallel mode");
  }
  std::array<const double*, 44> pointers;
  std::array<void*, 44> destinations;
  std::array<void*, 44> sources;
  std::array<std::size_t, 44> sizes;
  std::array<std::size_t, 44> offsets;
  std::size_t count = 0;
  std::size_t offset = 0;
  for (std::size_t index = 0; index < pointers.size(); ++index) {
    pointers[index] = m_workspace.input_ + offset;
    offsets[index] = offset;
    const std::size_t bytes = input.sizes_[index] * sizeof(double);
    if (bytes != 0) {
      if (batch) {
        destinations[count] = m_workspace.input_ + offset;
        sources[count] = const_cast<double*>(input.data_[index]);
        sizes[count] = bytes;
        ++count;
      } else if (direct) {
        Check(cudaMemcpyAsync(m_workspace.input_ + offset, input.data_[index],
                              bytes, cudaMemcpyHostToDevice,
                              m_workspace.stream_));
      } else if (!parallel) {
        std::memcpy(m_workspace.host_input_ + offset, input.data_[index],
                    bytes);
      }
    }
    offset += input.sizes_[index];
  }
  if (parallel) {
    static const int threads = [] {
      const char* value = std::getenv("VMECPP_GPU_PACK_THREADS");
      if (value == nullptr) {
        return 4;
      }
      char* end = nullptr;
      const long count = std::strtol(value, &end, 10);
      if (*value == '\0' || *end != '\0' || count < 1 || count > 32) {
        throw std::runtime_error("GPU pack threads must be between 1 and 32");
      }
      return static_cast<int>(count);
    }();
    thread_local RadialExecutor executor(threads);
    executor.Run(0, static_cast<int>(pointers.size()), [&](int index) {
      if (input.sizes_[index] != 0) {
        std::memcpy(m_workspace.host_input_ + offsets[index],
                    input.data_[index], input.sizes_[index] * sizeof(double));
      }
    });
  }
  if (batch) {
    cudaMemcpyAttributes attributes{};
    // All sources stay alive and unchanged until the stream synchronization.
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    std::size_t first = 0;
    std::size_t failed = 0;
    Check(cudaMemcpyBatchAsync(destinations.data(), sources.data(),
                               sizes.data(), count, &attributes, &first, 1,
                               &failed, m_workspace.stream_));
  } else if (!direct && enqueue) {
    Check(cudaMemcpyAsync(m_workspace.input_, m_workspace.host_input_,
                          offset * sizeof(double), cudaMemcpyHostToDevice,
                          m_workspace.stream_));
  }
  return pointers;
}

__global__ void MhdPointKernel(vmecpp_experiment::MhdForceView view,
                               vmecpp_experiment::MhdForceParameters parameters,
                               int points, double* __restrict__ m_forces) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= points) {
    return;
  }
  const int jF = index / parameters.angular_size_ + parameters.ns_min_f_;
  const int kl = index % parameters.angular_size_;
  vmecpp_experiment::MhdPointForces value;
  vmecpp_experiment::ComputeMhdForcePoint(view, parameters, jF, kl, value);
  m_forces[0 * points + index] = value.armn_e_;
  m_forces[1 * points + index] = value.armn_o_;
  m_forces[2 * points + index] = value.azmn_e_;
  m_forces[3 * points + index] = value.azmn_o_;
  m_forces[4 * points + index] = value.brmn_e_;
  m_forces[5 * points + index] = value.brmn_o_;
  m_forces[6 * points + index] = value.bzmn_e_;
  m_forces[7 * points + index] = value.bzmn_o_;
  m_forces[8 * points + index] = value.crmn_e_;
  m_forces[9 * points + index] = value.crmn_o_;
  m_forces[10 * points + index] = value.czmn_e_;
  m_forces[11 * points + index] = value.czmn_o_;
}

__global__ void ConstraintKernel(
    vmecpp_experiment::MhdForceParameters p, int points,
    const double* __restrict__ r_con, const double* __restrict__ z_con,
    const double* __restrict__ r_con0, const double* __restrict__ z_con0,
    const double* __restrict__ g_con, const double* __restrict__ ru_full,
    const double* __restrict__ zu_full, const double* __restrict__ sqrt_sf,
    double* __restrict__ m_forces) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= points) {
    return;
  }
  const int jF = index / p.angular_size_ + p.ns_min_f_;
  const double sqrt_s = sqrt_sf[jF - p.ns_min_f1_];
  const double brcon = (r_con[index] - r_con0[index]) * g_con[index];
  const double bzcon = (z_con[index] - z_con0[index]) * g_con[index];
  m_forces[4 * points + index] += brcon;
  m_forces[6 * points + index] += bzcon;
  m_forces[5 * points + index] += brcon * sqrt_s;
  m_forces[7 * points + index] += bzcon * sqrt_s;
  m_forces[12 * points + index] = ru_full[index] * g_con[index];
  m_forces[14 * points + index] = zu_full[index] * g_con[index];
  m_forces[13 * points + index] = m_forces[12 * points + index] * sqrt_s;
  m_forces[15 * points + index] = m_forces[14 * points + index] * sqrt_s;
}

template <class Output>
void RunForcePipeline(const GpuPipelineInput& input, Output& m_output,
                      bool finish) {
  const auto& d = input.dimensions_;
  if (d[0] != 0 || d[2] != 0 || d[12] != 0 || d[11] <= 1 || d[5] <= 0 ||
      d[6] < 0 || d[7] <= 0 || d[8] <= 0 || input.sizes_[23] != 1) {
    throw std::runtime_error(
        "Force pipeline requires one symmetric fixed-boundary partition");
  }
  const int force_rows = std::min(d[1], d[11] - 1) - d[0];
  const int force_points = force_rows * d[7] * d[8];
  const int tasks = (d[4] - d[0]) * d[5] * d[7];
  const int points = (d[4] - d[0]) * d[5] * (d[6] + 1);
  if (force_points <= 0 || tasks <= 0 || points <= 0) {
    throw std::runtime_error("Invalid force-pipeline dimensions");
  }
  if (!finish) {
    for (const auto size : m_output.sizes_) {
      if (size != static_cast<std::size_t>(force_points)) {
        throw std::runtime_error(
            "MHD replay must contain only active force rows");
      }
    }
  }
  thread_local ForwardWorkspace workspace;
  const std::size_t inputs =
      std::accumulate(input.sizes_.begin(), input.sizes_.end(), std::size_t{0});
  const std::size_t outputs = std::accumulate(
      m_output.sizes_.begin(), m_output.sizes_.end(), std::size_t{0});
  workspace.Resize(inputs, outputs, std::size_t{12} * tasks,
                   std::size_t{16} * force_points);
  const char* graph_flag = std::getenv("VMECPP_GPU_GRAPH");
  const bool graph = graph_flag != nullptr && std::strcmp(graph_flag, "1") == 0;
  const auto ptr = UploadPipelineInputs(input, workspace, !graph);
  std::size_t offset = 0;
  const vmecpp_experiment::MhdForceView view{
      ptr[0],  ptr[1],  ptr[2],  ptr[3],  ptr[4],  ptr[5],  ptr[6],  ptr[7],
      ptr[8],  ptr[9],  ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15],
      ptr[16], ptr[17], ptr[18], ptr[19], ptr[20], ptr[21], ptr[22]};
  const vmecpp_experiment::MhdForceParameters parameters{
      d[0], d[2], 0, d[11] - 1, d[7] * d[8], input.data_[23][0]};
  const auto enqueue = [&] {
    if (graph) {
      Check(cudaMemcpyAsync(workspace.input_, workspace.host_input_,
                            inputs * sizeof(double), cudaMemcpyHostToDevice,
                            workspace.stream_));
    }
    MhdPointKernel<<<(force_points + 127) / 128, 128, 0, workspace.stream_>>>(
        view, parameters, force_points, workspace.forces_);
    Check(cudaGetLastError());
    if (finish) {
      ConstraintKernel<<<(force_points + 127) / 128, 128, 0,
                         workspace.stream_>>>(
          parameters, force_points, ptr[24], ptr[25], ptr[26], ptr[27], ptr[28],
          ptr[29], ptr[30], ptr[21], workspace.forces_);
      Check(cudaGetLastError());
      std::array<double*, 16> force;
      for (std::size_t index = 0; index < force.size(); ++index) {
        force[index] = workspace.forces_ + index * force_points;
      }
      const ForwardPointers pointers{
          force[0],  force[1],  force[2],  force[3],  ptr[31],   ptr[32],
          force[4],  force[5],  force[6],  force[7],  ptr[33],   ptr[34],
          force[8],  force[9],  force[10], force[11], force[12], force[13],
          force[14], force[15], ptr[35],   ptr[36],   ptr[37],   ptr[38],
          ptr[39],   ptr[40],   ptr[41],   ptr[42],   ptr[43]};
      const ForwardParameters fp{d[0], d[1], d[4], d[11], d[5],
                                 d[6], d[7], d[8], d[9],  d[10]};
      Check(cudaMemsetAsync(workspace.output_, 0, outputs * sizeof(double),
                            workspace.stream_));
      ForwardPartialsMixedPrecision<<<dim3((tasks + 127) / 128, kForwardGroups),
                                      128, 0, workspace.stream_>>>(
          pointers, fp, workspace.partials_);
      Check(cudaGetLastError());
      std::array<double*, 6> out;
      offset = 0;
      for (std::size_t index = 0; index < out.size(); ++index) {
        out[index] = workspace.output_ + offset;
        offset += m_output.sizes_[index];
      }
      ForwardFinishMixedPrecision<<<dim3((points + 127) / 128, kForwardGroups),
                                    128, 0, workspace.stream_>>>(
          pointers, fp, workspace.partials_, out[0], out[1], out[2], out[3],
          out[4], out[5]);
      Check(cudaGetLastError());
    }
    Check(cudaMemcpyAsync(
        workspace.host_output_, finish ? workspace.output_ : workspace.forces_,
        outputs * sizeof(double), cudaMemcpyDeviceToHost, workspace.stream_));
  };
  if (graph) {
    const bool same_output_sizes =
        workspace.graph_output_sizes_.size() == m_output.sizes_.size() &&
        std::equal(workspace.graph_output_sizes_.begin(),
                   workspace.graph_output_sizes_.end(),
                   m_output.sizes_.begin());
    if (workspace.graph_dimensions_ != d ||
        workspace.graph_input_sizes_ != input.sizes_ || !same_output_sizes ||
        std::memcmp(&workspace.graph_delta_s_, &parameters.delta_s_,
                    sizeof(double)) != 0) {
      workspace.ResetGraph();
    }
    if (workspace.graph_exec_ == nullptr) {
      Check(cudaStreamBeginCapture(workspace.stream_,
                                   cudaStreamCaptureModeThreadLocal));
      enqueue();
      Check(cudaStreamEndCapture(workspace.stream_, &workspace.graph_));
      Check(cudaGraphInstantiate(&workspace.graph_exec_, workspace.graph_, 0));
      workspace.graph_dimensions_ = d;
      workspace.graph_input_sizes_ = input.sizes_;
      workspace.graph_output_sizes_.assign(m_output.sizes_.begin(),
                                           m_output.sizes_.end());
      workspace.graph_delta_s_ = parameters.delta_s_;
    }
    Check(cudaGraphLaunch(workspace.graph_exec_, workspace.stream_));
  } else {
    enqueue();
  }
  Check(cudaStreamSynchronize(workspace.stream_));
  offset = 0;
  for (std::size_t index = 0; index < m_output.sizes_.size(); ++index) {
    std::memcpy(m_output.data_[index], workspace.host_output_ + offset,
                m_output.sizes_[index] * sizeof(double));
    offset += m_output.sizes_[index];
  }
}
}  // namespace

extern "C" void RunGpuPipeline(const GpuPipelineInput& input,
                               GpuForwardOutput& m_output) {
  RunForcePipeline(input, m_output, true);
}
extern "C" void RunGpuMhd(const GpuPipelineInput& input,
                          GpuMhdOutput& m_output) {
  RunForcePipeline(input, m_output, false);
}
#endif  // VMECPP_GPU_PIPELINE
