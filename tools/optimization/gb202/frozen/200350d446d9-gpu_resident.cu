// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Resident geometry/field stages derived from VMEC++ v0.4.11. FMA is disabled.
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda/std/array>
#include <numeric>
#include <stdexcept>

#include "geometry_point.h"
#include "gpu_resident.h"
#include "mhd_point.h"
#ifdef VMECPP_GPU_RESIDENT_CHAIN
#include "gpu_forward.h"
#include "gpu_fourier.h"
#endif

namespace {
struct FullPointers {
  double* r1_e_;
  double* r1_o_;
  double* ru_e_;
  double* ru_o_;
  double* rv_e_;
  double* rv_o_;
  double* z1_e_;
  double* z1_o_;
  double* zu_e_;
  double* zu_o_;
  double* zv_e_;
  double* zv_o_;
  double* lu_e_;
  double* lu_o_;
  double* lv_e_;
  double* lv_o_;
  double* r_con_;
  double* z_con_;
};
struct HalfPointers {
  double* r12_;
  double* ru12_;
  double* zu12_;
  double* rs_;
  double* zs_;
  double* tau_;
  double* gsqrt_;
  double* guu_;
  double* guv_;
  double* gvv_;
  double* bsupu_;
  double* bsupv_;
  double* bsubu_;
  double* bsubv_;
};
struct ProfilePointers {
  double* sqrt_sf_;
  double* sqrt_sh_;
  double* w_int_;
  double* phip_f_;
  double* phip_h_;
  double* curr_h_;
  double* chip_h_;
  double* iota_h_;
  double* chip_f_;
  double* iota_f_;
  double* d_v_ds_h_;
};
struct Parameters {
  int ns_;
  int angular_size_;
  int n_theta_eff_;
  double delta_s_;
  double d_s_half_ds_interp_;
  double lamscale_;
  double sign_of_jacobian_;
};

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

__global__ void GeometryKernel(FullPointers f, HalfPointers m_h,
                               ProfilePointers q, Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (p.ns_ - 1) * p.angular_size_) {
    return;
  }
  const int jH = index / p.angular_size_;
  const int kl = index % p.angular_size_;
  const vmecpp_experiment::GeometryPointView view{
      f.r1_e_, f.r1_o_, f.z1_e_, f.z1_o_, f.ru_e_, f.ru_o_,    f.zu_e_,
      f.zu_o_, f.rv_e_, f.rv_o_, f.zv_e_, f.zv_o_, q.sqrt_sf_, q.sqrt_sh_};
  const vmecpp_experiment::GeometryPointParameters parameters{
      p.angular_size_, 0, 0, p.delta_s_, p.d_s_half_ds_interp_};
  vmecpp_experiment::GeometryPoint value;
  vmecpp_experiment::ComputeGeometryPoint(view, parameters, jH, kl, value);
  m_h.r12_[index] = value.r12_;
  m_h.ru12_[index] = value.ru12_;
  m_h.zu12_[index] = value.zu12_;
  m_h.rs_[index] = value.rs_;
  m_h.zs_[index] = value.zs_;
  m_h.tau_[index] = value.tau_;
  m_h.gsqrt_[index] = value.gsqrt_;
  m_h.guu_[index] = value.guu_;
  m_h.guv_[index] = value.guv_;
  m_h.gvv_[index] = value.gvv_;
}

__global__ void JacobianPartialsKernel(const double* tau, int count,
                                       double* m_partials) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x % 32;
  const int warp = index / 32;
  const int warps = (count + 31) / 32;
  const double value = index < count ? tau[index] : CUDART_INF;
  const bool usable = index < count && isfinite(value) && value != 0.0;
  double min_tau = usable ? value : CUDART_INF;
  double max_tau = usable ? value : -CUDART_INF;
  int special = index < count && !usable;
  for (int offset = 16; offset > 0; offset /= 2) {
    const double other_min = __shfl_down_sync(0xffffffff, min_tau, offset);
    const double other_max = __shfl_down_sync(0xffffffff, max_tau, offset);
    const int other_special = __shfl_down_sync(0xffffffff, special, offset);
    if (other_min < min_tau) {
      min_tau = other_min;
    }
    if (other_max > max_tau) {
      max_tau = other_max;
    }
    special |= other_special;
  }
  if (lane == 0 && warp < warps) {
    m_partials[warp] = min_tau;
    m_partials[warps + warp] = max_tau;
    m_partials[2 * warps + warp] = special ? 1.0 : 0.0;
  }
}

__global__ void JacobianExtremaKernel(const double* tau, int count,
                                      const double* partials,
                                      double* m_extrema) {
  const int lane = threadIdx.x;
  const int warps = (count + 31) / 32;
  double min_tau = CUDART_INF, max_tau = -CUDART_INF;
  int special = 0;
  for (int index = lane; index < warps; index += 32) {
    const double value_min = partials[index],
                 value_max = partials[warps + index];
    if (value_min < min_tau) {
      min_tau = value_min;
    }
    if (value_max > max_tau) {
      max_tau = value_max;
    }
    special |= partials[2 * warps + index] != 0.0;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    const double other_min = __shfl_down_sync(0xffffffff, min_tau, offset);
    const double other_max = __shfl_down_sync(0xffffffff, max_tau, offset);
    const int other_special = __shfl_down_sync(0xffffffff, special, offset);
    if (other_min < min_tau) {
      min_tau = other_min;
    }
    if (other_max > max_tau) {
      max_tau = other_max;
    }
    special |= other_special;
  }
  if (lane != 0) {
    return;
  }
  if (special != 0) {
    // The original zero sentinel and NaN comparison behavior are
    // order-sensitive. Preserve them exactly; normal finite nonzero min/max
    // needs no arithmetic reassociation and selects the same bit pattern in
    // parallel.
    min_tau = 0.0;
    max_tau = 0.0;
    for (int index = 0; index < count; ++index) {
      const double value = tau[index];
      if (value < min_tau || min_tau == 0.0) {
        min_tau = value;
      }
      if (value > max_tau || max_tau == 0.0) {
        max_tau = value;
      }
    }
  }
  m_extrema[0] = min_tau;
  m_extrema[1] = max_tau;
}

__global__ void ScaleLambdaKernel(FullPointers m_f, ProfilePointers q,
                                  Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= p.ns_ * p.angular_size_) {
    return;
  }
  const int jF = index / p.angular_size_;
  m_f.lu_e_[index] *= p.lamscale_;
  m_f.lu_o_[index] *= p.lamscale_;
  m_f.lv_e_[index] *= p.lamscale_;
  m_f.lv_o_[index] *= p.lamscale_;
  m_f.lu_e_[index] += q.phip_f_[jF];
}

__global__ void InitialBContraKernel(FullPointers f, HalfPointers m_h,
                                     ProfilePointers q, Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (p.ns_ - 1) * p.angular_size_) {
    return;
  }
  const int jH = index / p.angular_size_;
  const int outside = index + p.angular_size_;
  const double sqrt_sh = q.sqrt_sh_[jH];
  m_h.bsupu_[index] = 0.5 *
                      ((f.lv_e_[index] + f.lv_e_[outside]) +
                       sqrt_sh * (f.lv_o_[index] + f.lv_o_[outside])) /
                      m_h.gsqrt_[index];
  m_h.bsupv_[index] = 0.5 *
                      ((f.lu_e_[index] + f.lu_e_[outside]) +
                       sqrt_sh * (f.lu_o_[index] + f.lu_o_[outside])) /
                      m_h.gsqrt_[index];
}

__global__ void OrderedProfilesKernel(HalfPointers h, ProfilePointers m_q,
                                      Parameters p) {
  const int jH = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= p.ns_ - 1) {
    return;
  }
  double volume = 0.0, jv_plasma = 0.0, avg_guu_gsqrt = 0.0;
  for (int kl = 0; kl < p.angular_size_; ++kl) {
    const int index = jH * p.angular_size_ + kl;
    const int l = kl % p.n_theta_eff_;
    volume += h.gsqrt_[index] * m_q.w_int_[l];
    jv_plasma +=
        (h.guu_[index] * h.bsupu_[index] + h.guv_[index] * h.bsupv_[index]) *
        m_q.w_int_[l];
    avg_guu_gsqrt += h.guu_[index] / h.gsqrt_[index] * m_q.w_int_[l];
  }
  volume *= p.sign_of_jacobian_;
  m_q.d_v_ds_h_[jH] = volume;
  if (avg_guu_gsqrt != 0.0) {
    m_q.chip_h_[jH] = (m_q.curr_h_[jH] - jv_plasma) / avg_guu_gsqrt;
  }
  if (m_q.phip_h_[jH] != 0.0) {
    m_q.iota_h_[jH] = m_q.chip_h_[jH] / m_q.phip_h_[jH];
  }
}

__global__ void ProfileTermsKernel(HalfPointers h, ProfilePointers q,
                                   Parameters p, double* m_terms) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = (p.ns_ - 1) * p.angular_size_;
  if (index >= count) {
    return;
  }
  const int l = index % p.angular_size_ % p.n_theta_eff_;
  m_terms[index] = h.gsqrt_[index] * q.w_int_[l];
  m_terms[count + index] =
      (h.guu_[index] * h.bsupu_[index] + h.guv_[index] * h.bsupv_[index]) *
      q.w_int_[l];
  m_terms[2 * count + index] = h.guu_[index] / h.gsqrt_[index] * q.w_int_[l];
}

__global__ void TiledOrderedProfilesKernel(const double* terms,
                                           ProfilePointers m_profiles,
                                           Parameters p) {
  constexpr int kRows = 32, kAngles = 32, kPitch = 33, kPlane = kRows * kPitch;
  __shared__ cuda::std::array<double, 3 * kPlane> tile;
  const int row = threadIdx.x;
  const int jH = blockIdx.x * kRows + row;
  const int count = (p.ns_ - 1) * p.angular_size_;
  double volume = 0.0, jv_plasma = 0.0, avg_guu_gsqrt = 0.0;
  for (int first = 0; first < p.angular_size_; first += kAngles) {
    for (int position = threadIdx.x; position < kRows * kAngles;
         position += blockDim.x) {
      const int load_row = position / kAngles;
      const int angle = position % kAngles;
      const int load_j = blockIdx.x * kRows + load_row;
      const int kl = first + angle;
      const bool valid = load_j < p.ns_ - 1 && kl < p.angular_size_;
      const int index = load_j * p.angular_size_ + kl;
      const int destination = load_row * kPitch + angle;
      tile[destination] = valid ? terms[index] : 0.0;
      tile[kPlane + destination] = valid ? terms[count + index] : 0.0;
      tile[2 * kPlane + destination] = valid ? terms[2 * count + index] : 0.0;
    }
    __syncthreads();
    if (row < kRows && jH < p.ns_ - 1) {
      const int angles = min(kAngles, p.angular_size_ - first);
      for (int angle = 0; angle < angles; ++angle) {
        const int index = row * kPitch + angle;
        volume += tile[index];
        jv_plasma += tile[kPlane + index];
        avg_guu_gsqrt += tile[2 * kPlane + index];
      }
    }
    __syncthreads();
  }
  if (row < kRows && jH < p.ns_ - 1) {
    volume *= p.sign_of_jacobian_;
    m_profiles.d_v_ds_h_[jH] = volume;
    if (avg_guu_gsqrt != 0.0) {
      m_profiles.chip_h_[jH] =
          (m_profiles.curr_h_[jH] - jv_plasma) / avg_guu_gsqrt;
    }
    if (m_profiles.phip_h_[jH] != 0.0) {
      m_profiles.iota_h_[jH] = m_profiles.chip_h_[jH] / m_profiles.phip_h_[jH];
    }
  }
}

#ifdef VMECPP_PROFILE_TILE_ROWS
// Opt-in tiling experiment. The original default kernel above is unchanged.
// Each row/component retains the original increasing-angle addition sequence.
template <int kRows, int kThreads, bool kSplitChains>
__global__ void TiledOrderedProfilesRowsKernel(const double* terms,
                                               ProfilePointers m_profiles,
                                               Parameters p) {
  static_assert(kRows == 1 || kRows == 2 || kRows == 4 || kRows == 8 ||
                kRows == 16 || kRows == 32);
  static_assert(kThreads == 32 || kThreads == 128);
  static_assert(!kSplitChains || (kThreads == 32 && 3 * kRows <= 32));
  constexpr int kAngles = 32, kPitch = 33, kPlane = kRows * kPitch;
  __shared__ cuda::std::array<double, 3 * kPlane> tile;
  const int row = kSplitChains ? threadIdx.x % kRows : threadIdx.x;
  const int component = kSplitChains ? threadIdx.x / kRows : 0;
  const int jH = blockIdx.x * kRows + row;
  const int count = (p.ns_ - 1) * p.angular_size_;
  double volume = 0.0, jv_plasma = 0.0, avg_guu_gsqrt = 0.0;
  double component_sum = 0.0;
  for (int first = 0; first < p.angular_size_; first += kAngles) {
    for (int position = threadIdx.x; position < kRows * kAngles;
         position += kThreads) {
      const int load_row = position / kAngles;
      const int angle = position % kAngles;
      const int load_j = blockIdx.x * kRows + load_row;
      const int kl = first + angle;
      const bool valid = load_j < p.ns_ - 1 && kl < p.angular_size_;
      const int index = load_j * p.angular_size_ + kl;
      const int destination = load_row * kPitch + angle;
      tile[destination] = valid ? terms[index] : 0.0;
      tile[kPlane + destination] = valid ? terms[count + index] : 0.0;
      tile[2 * kPlane + destination] = valid ? terms[2 * count + index] : 0.0;
    }
    if constexpr (kThreads == 32)
      __syncwarp();
    else
      __syncthreads();
    if (row < kRows && jH < p.ns_ - 1 && (!kSplitChains || component < 3)) {
      const int angles = min(kAngles, p.angular_size_ - first);
      for (int angle = 0; angle < angles; ++angle) {
        const int index = row * kPitch + angle;
        if constexpr (kSplitChains) {
          component_sum += tile[component * kPlane + index];
        } else {
          volume += tile[index];
          jv_plasma += tile[kPlane + index];
          avg_guu_gsqrt += tile[2 * kPlane + index];
        }
      }
    }
    if constexpr (kThreads == 32)
      __syncwarp();
    else
      __syncthreads();
  }
  if constexpr (kSplitChains) {
    // Every lane participates, including padded rows and unused components.
    volume = __shfl_sync(0xffffffff, component_sum, row);
    jv_plasma = __shfl_sync(0xffffffff, component_sum, kRows + row);
    avg_guu_gsqrt = __shfl_sync(0xffffffff, component_sum, 2 * kRows + row);
  }
  if (row < kRows && jH < p.ns_ - 1 && component == 0) {
    volume *= p.sign_of_jacobian_;
    m_profiles.d_v_ds_h_[jH] = volume;
    if (avg_guu_gsqrt != 0.0) {
      m_profiles.chip_h_[jH] =
          (m_profiles.curr_h_[jH] - jv_plasma) / avg_guu_gsqrt;
    }
    if (m_profiles.phip_h_[jH] != 0.0) {
      m_profiles.iota_h_[jH] = m_profiles.chip_h_[jH] / m_profiles.phip_h_[jH];
    }
  }
}
#endif

__global__ void InterpolateProfilesKernel(ProfilePointers m_q, Parameters p) {
  const int jF = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF >= p.ns_) {
    return;
  }
  if (jF == 0) {
    m_q.iota_f_[0] = 1.5 * m_q.iota_h_[0] - 0.5 * m_q.iota_h_[1];
  } else if (jF == p.ns_ - 1) {
    m_q.chip_f_[jF] = 2.0 * m_q.chip_h_[p.ns_ - 2] - m_q.chip_h_[p.ns_ - 3];
    m_q.iota_f_[jF] =
        1.5 * m_q.iota_h_[p.ns_ - 2] - 0.5 * m_q.iota_h_[p.ns_ - 3];
  } else {
    m_q.chip_f_[jF] = 0.5 * (m_q.chip_h_[jF] + m_q.chip_h_[jF - 1]);
    m_q.iota_f_[jF] = 0.5 * (m_q.iota_h_[jF] + m_q.iota_h_[jF - 1]);
  }
}

__global__ void FinishFieldsKernel(HalfPointers m_h, ProfilePointers q,
                                   Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (p.ns_ - 1) * p.angular_size_) {
    return;
  }
  const int jH = index / p.angular_size_;
  m_h.bsupu_[index] += q.chip_h_[jH] / m_h.gsqrt_[index];
  m_h.bsubu_[index] =
      m_h.guu_[index] * m_h.bsupu_[index] + m_h.guv_[index] * m_h.bsupv_[index];
  m_h.bsubv_[index] =
      m_h.guv_[index] * m_h.bsupu_[index] + m_h.gvv_[index] * m_h.bsupv_[index];
}

__global__ void TotalPressureKernel(HalfPointers h, const double* mass_h,
                                    Parameters p, double* m_pressure) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (p.ns_ - 1) * p.angular_size_) {
    return;
  }
  const int jH = index / p.angular_size_;
  // This specialization is gated on the input gamma being exactly zero.
  // pow(dVdsH,0) is 1; no pressure profile or physics setting is changed.
  const double pres_h = mass_h[jH] / 1.0;
  const double magnetic_pressure = 0.5 * (h.bsupu_[index] * h.bsubu_[index] +
                                          h.bsupv_[index] * h.bsubv_[index]);
  m_pressure[index] = magnetic_pressure + pres_h;
}

__global__ void LambdaForceKernel(FullPointers f, HalfPointers h,
                                  ProfilePointers q, const double* blending,
                                  Parameters p, double* m_lambda) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = p.ns_ * p.angular_size_;
  if (index >= count) {
    return;
  }
  const int jF = index / p.angular_size_;
  const int inside = index - p.angular_size_;
  const int outside = index;
  double sqrt_s_hi = 0.0, sqrt_s_ho = 0.0;
  double bsubu_i = 0.0, bsubv_i = 0.0, gvv_gsqrt_i = 0.0, guv_bsupu_i = 0.0;
  double bsubu_o = 0.0, bsubv_o = 0.0, gvv_gsqrt_o = 0.0, guv_bsupu_o = 0.0;
  if (jF > 0) {
    sqrt_s_hi = q.sqrt_sh_[jF - 1];
    bsubu_i = h.bsubu_[inside];
    bsubv_i = h.bsubv_[inside];
    gvv_gsqrt_i = h.gvv_[inside] / h.gsqrt_[inside];
    guv_bsupu_i = h.guv_[inside] * h.bsupu_[inside];
  }
  if (jF < p.ns_ - 1) {
    sqrt_s_ho = q.sqrt_sh_[jF];
    bsubu_o = h.bsubu_[outside];
    bsubv_o = h.bsubv_[outside];
    gvv_gsqrt_o = h.gvv_[outside] / h.gsqrt_[outside];
    guv_bsupu_o = h.guv_[outside] * h.bsupu_[outside];
  }
  const double gvv_gsqrt_lu_e =
      0.5 * (gvv_gsqrt_i + gvv_gsqrt_o) * f.lu_e_[index];
  const double gvv_gsqrt_lu_o =
      0.5 * (gvv_gsqrt_i * sqrt_s_hi + gvv_gsqrt_o * sqrt_s_ho) *
      f.lu_o_[index];
  const double gvv_gsqrt_lu = gvv_gsqrt_lu_e + gvv_gsqrt_lu_o;
  double bsubv_alternative = gvv_gsqrt_lu;
  const double guv_bsupu = 0.5 * (guv_bsupu_i + guv_bsupu_o);
  bsubv_alternative += guv_bsupu;
  const double bsubv_average = 0.5 * (bsubv_o + bsubv_i);
  double blmn =
      bsubv_average * (1.0 - blending[jF]) + bsubv_alternative * blending[jF];
  if (jF > 0) {
    blmn *= -p.lamscale_;
  }
  m_lambda[index] = blmn;
  m_lambda[count + index] = blmn * q.sqrt_sf_[jF];
  double clmn = 0.5 * (bsubu_o + bsubu_i);
  if (jF > 0) {
    clmn *= -p.lamscale_;
  }
  m_lambda[2 * count + index] = clmn;
  m_lambda[3 * count + index] = clmn * q.sqrt_sf_[jF];
}

__global__ void ResidentMhdKernel(vmecpp_experiment::MhdForceView view,
                                  vmecpp_experiment::MhdForceParameters p,
                                  int count, double* m_forces) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int jF = index / p.angular_size_;
  const int kl = index % p.angular_size_;
  vmecpp_experiment::MhdPointForces value;
  vmecpp_experiment::ComputeMhdForcePoint(view, p, jF, kl, value);
  m_forces[index] = value.armn_e_;
  m_forces[count + index] = value.armn_o_;
  m_forces[2 * count + index] = value.azmn_e_;
  m_forces[3 * count + index] = value.azmn_o_;
  m_forces[4 * count + index] = value.brmn_e_;
  m_forces[5 * count + index] = value.brmn_o_;
  m_forces[6 * count + index] = value.bzmn_e_;
  m_forces[7 * count + index] = value.bzmn_o_;
  m_forces[8 * count + index] = value.crmn_e_;
  m_forces[9 * count + index] = value.crmn_o_;
  m_forces[10 * count + index] = value.czmn_e_;
  m_forces[11 * count + index] = value.czmn_o_;
}

class ResidentForceWorkspace {
 public:
  ~ResidentForceWorkspace() {
    cudaFree(constants_);
    cudaFree(values_);
    cudaFreeHost(host_constants_);
  }
  ResidentForceWorkspace() = default;
  ResidentForceWorkspace(const ResidentForceWorkspace&) = delete;
  ResidentForceWorkspace& operator=(const ResidentForceWorkspace&) = delete;
  void Resize(std::size_t constants, std::size_t values) {
    if (constants > constant_capacity_) {
      Check(cudaFree(constants_));
      constants_ = nullptr;
      Check(cudaFreeHost(host_constants_));
      host_constants_ = nullptr;
      Check(cudaMalloc(&constants_, constants * sizeof(double)));
      Check(cudaMallocHost(&host_constants_, constants * sizeof(double)));
      constant_capacity_ = constants;
    }
    if (values > value_capacity_) {
      Check(cudaFree(values_));
      values_ = nullptr;
      Check(cudaMalloc(&values_, values * sizeof(double)));
      value_capacity_ = values;
    }
  }
  double* constants_ = nullptr;
  double* host_constants_ = nullptr;
  double* values_ = nullptr;

 private:
  std::size_t constant_capacity_ = 0, value_capacity_ = 0;
};

class ResidentWorkspace {
 public:
  ResidentWorkspace() { Check(cudaStreamCreate(&stream_)); }
  ~ResidentWorkspace() {
    cudaFree(input_);
    cudaFree(half_);
    cudaFree(volume_);
    cudaFree(extrema_);
    cudaFree(terms_);
    cudaFreeHost(host_input_);
    cudaFreeHost(host_output_);
    cudaStreamDestroy(stream_);
  }
  ResidentWorkspace(const ResidentWorkspace&) = delete;
  ResidentWorkspace& operator=(const ResidentWorkspace&) = delete;
  void Resize(std::size_t inputs, std::size_t outputs, std::size_t half_points,
              int ns) {
    if (inputs > input_capacity_) {
      Check(cudaFree(input_));
      input_ = nullptr;
      Check(cudaFreeHost(host_input_));
      host_input_ = nullptr;
      Check(cudaMalloc(&input_, inputs * sizeof(double)));
      Check(cudaMallocHost(&host_input_, inputs * sizeof(double)));
      input_capacity_ = inputs;
    }
    if (outputs > output_capacity_) {
      Check(cudaFreeHost(host_output_));
      host_output_ = nullptr;
      Check(cudaMallocHost(&host_output_, outputs * sizeof(double)));
      output_capacity_ = outputs;
    }
    if (half_points > half_capacity_) {
      Check(cudaFree(half_));
      half_ = nullptr;
      Check(cudaMalloc(&half_, 14 * half_points * sizeof(double)));
      half_capacity_ = half_points;
      Check(cudaFree(extrema_));
      extrema_ = nullptr;
      Check(cudaMalloc(&extrema_,
                       (2 + 3 * ((half_points + 31) / 32)) * sizeof(double)));
      Check(cudaFree(terms_));
      terms_ = nullptr;
      Check(cudaMalloc(&terms_, 3 * half_points * sizeof(double)));
    }
    if (ns > radial_capacity_) {
      Check(cudaFree(volume_));
      volume_ = nullptr;
      Check(cudaMalloc(&volume_, ns * sizeof(double)));
      radial_capacity_ = ns;
    }
  }
  cudaStream_t stream_;
  double* input_ = nullptr;
  double* half_ = nullptr;
  double* volume_ = nullptr;
  double* extrema_ = nullptr;
  double* host_input_ = nullptr;
  double* host_output_ = nullptr;
  double* terms_ = nullptr;

 private:
  std::size_t input_capacity_ = 0, output_capacity_ = 0, half_capacity_ = 0;
  int radial_capacity_ = 0;
};
}  // namespace

extern "C" void LaunchGpuResidentFields(GpuResidentDeviceState& m_state) {
  const auto& f = m_state.full_;
  const auto& h = m_state.half_;
  const auto& q = m_state.profiles_;
  const auto& d = m_state.dimensions_;
  const auto& s = m_state.scalars_;
  if (d[0] != 0 || d[2] != 0 || d[11] < 3 || d[12] != 0 || d[6] <= 0) {
    throw std::runtime_error(
        "Resident fields require the symmetric fixed-boundary single "
        "partition");
  }
  const FullPointers full{f[0],  f[1],  f[2],  f[3],  f[4],  f[5],
                          f[6],  f[7],  f[8],  f[9],  f[10], f[11],
                          f[12], f[13], f[14], f[15], f[16], f[17]};
  const HalfPointers half{h[0], h[1], h[2], h[3],  h[4],  h[5],  h[6],
                          h[7], h[8], h[9], h[10], h[11], h[12], h[13]};
  const ProfilePointers profiles{q[0], q[1], q[2], q[3], q[4], q[5],
                                 q[6], q[7], q[8], q[9], q[10]};
  const Parameters p{d[11], d[7] * d[8], d[8], s[0], s[1], s[2], s[3]};
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  const int half_points = (p.ns_ - 1) * p.angular_size_;
  GeometryKernel<<<(half_points + 127) / 128, 128, 0, stream>>>(full, half,
                                                                profiles, p);
  Check(cudaGetLastError());
  JacobianPartialsKernel<<<(half_points + 255) / 256, 256, 0, stream>>>(
      half.tau_, half_points, m_state.jacobian_partials_);
  Check(cudaGetLastError());
  JacobianExtremaKernel<<<1, 32, 0, stream>>>(half.tau_, half_points,
                                              m_state.jacobian_partials_,
                                              m_state.jacobian_extrema_);
  Check(cudaGetLastError());
  ScaleLambdaKernel<<<(p.ns_ * p.angular_size_ + 127) / 128, 128, 0, stream>>>(
      full, profiles, p);
  Check(cudaGetLastError());
  InitialBContraKernel<<<(half_points + 127) / 128, 128, 0, stream>>>(
      full, half, profiles, p);
  Check(cudaGetLastError());
  const char* profile_reference = std::getenv("VMECPP_GPU_PROFILE_REFERENCE");
  if (profile_reference != nullptr &&
      std::strcmp(profile_reference, "1") == 0) {
    OrderedProfilesKernel<<<(p.ns_ - 1 + 127) / 128, 128, 0, stream>>>(
        half, profiles, p);
    Check(cudaGetLastError());
  } else {
    ProfileTermsKernel<<<(half_points + 127) / 128, 128, 0, stream>>>(
        half, profiles, p, m_state.profile_terms_);
    Check(cudaGetLastError());
#ifdef VMECPP_PROFILE_TILE_ROWS
    constexpr int kProfileRows = VMECPP_PROFILE_TILE_ROWS;
    constexpr int kProfileThreads = kProfileRows <= 8 ? 32 : 128;
#ifdef VMECPP_PROFILE_SPLIT_CHAINS
    constexpr bool kProfileSplitChains = true;
#else
    constexpr bool kProfileSplitChains = false;
#endif
    TiledOrderedProfilesRowsKernel<kProfileRows, kProfileThreads,
                                   kProfileSplitChains>
        <<<(p.ns_ - 1 + kProfileRows - 1) / kProfileRows, kProfileThreads, 0,
           stream>>>(m_state.profile_terms_, profiles, p);
#else
    TiledOrderedProfilesKernel<<<(p.ns_ - 1 + 31) / 32, 128, 0, stream>>>(
        m_state.profile_terms_, profiles, p);
#endif
    Check(cudaGetLastError());
  }
  InterpolateProfilesKernel<<<(p.ns_ + 127) / 128, 128, 0, stream>>>(profiles,
                                                                     p);
  Check(cudaGetLastError());
  FinishFieldsKernel<<<(half_points + 127) / 128, 128, 0, stream>>>(
      half, profiles, p);
  Check(cudaGetLastError());
}

extern "C" void LaunchGpuResidentForces(GpuResidentForceDeviceState& m_state) {
  if (m_state.adiabatic_index_ != 0.0) {
    throw std::runtime_error(
        "Resident pressure specialization requires unchanged gamma=0 input");
  }
  const auto& g = m_state.geometry_;
  const auto& f = g.full_;
  const auto& h = g.half_;
  const auto& q = g.profiles_;
  const auto& d = g.dimensions_;
  const auto& s = g.scalars_;
  const Parameters p{d[11], d[7] * d[8], d[8], s[0], s[1], s[2], s[3]};
  const int half_count = (p.ns_ - 1) * p.angular_size_;
  const int full_count = p.ns_ * p.angular_size_;
  for (std::size_t i = 0; i < m_state.mhd_.size(); ++i) {
    if (m_state.mhd_[i] != m_state.mhd_[0] + i * half_count) {
      throw std::runtime_error("Resident MHD channels must be contiguous");
    }
  }
  for (std::size_t i = 0; i < m_state.lambda_.size(); ++i) {
    if (m_state.lambda_[i] != m_state.lambda_[0] + i * full_count) {
      throw std::runtime_error("Resident lambda channels must be contiguous");
    }
  }
  const FullPointers full{f[0],  f[1],  f[2],  f[3],  f[4],  f[5],
                          f[6],  f[7],  f[8],  f[9],  f[10], f[11],
                          f[12], f[13], f[14], f[15], f[16], f[17]};
  const HalfPointers half{h[0], h[1], h[2], h[3],  h[4],  h[5],  h[6],
                          h[7], h[8], h[9], h[10], h[11], h[12], h[13]};
  const ProfilePointers profiles{q[0], q[1], q[2], q[3], q[4], q[5],
                                 q[6], q[7], q[8], q[9], q[10]};
  const auto stream = static_cast<cudaStream_t>(g.stream_);
  TotalPressureKernel<<<(half_count + 127) / 128, 128, 0, stream>>>(
      half, m_state.constants_[0], p, m_state.total_pressure_);
  Check(cudaGetLastError());
  LambdaForceKernel<<<(full_count + 127) / 128, 128, 0, stream>>>(
      full, half, profiles, m_state.constants_[1], p, m_state.lambda_[0]);
  Check(cudaGetLastError());
  const vmecpp_experiment::MhdForceView view{
      h[0],  h[1], h[2],  h[3],  h[4],
      h[5],  h[6], h[10], h[11], m_state.total_pressure_,
      f[0],  f[1], f[2],  f[3],  f[8],
      f[9],  f[7], f[4],  f[5],  f[10],
      f[11], q[0], q[1]};
  const vmecpp_experiment::MhdForceParameters parameters{
      0, 0, 0, p.ns_ - 1, p.angular_size_, p.delta_s_};
  ResidentMhdKernel<<<(half_count + 127) / 128, 128, 0, stream>>>(
      view, parameters, half_count, m_state.mhd_[0]);
  Check(cudaGetLastError());
}

extern "C" void RunGpuResidentFields(const GpuResidentInput& input,
                                     GpuResidentOutput& m_output) {
  thread_local ResidentWorkspace workspace;
  const int ns = input.dimensions_[11];
  if (ns < 3) {
    throw std::runtime_error("Invalid resident radial size");
  }
  const std::size_t half_points =
      std::size_t(ns - 1) * input.dimensions_[7] * input.dimensions_[8];
  const auto inputs =
      std::accumulate(input.sizes_.begin(), input.sizes_.end(), std::size_t{0});
  const auto outputs = std::accumulate(m_output.sizes_.begin(),
                                       m_output.sizes_.end(), std::size_t{0});
  workspace.Resize(inputs, outputs, half_points, ns);
  GpuResidentDeviceState state{};
  std::size_t offset = 0;
  for (std::size_t i = 0; i < input.data_.size(); ++i) {
    if (input.sizes_[i] != 0) {
      std::memcpy(workspace.host_input_ + offset, input.data_[i],
                  input.sizes_[i] * sizeof(double));
    }
    if (i < 18) {
      state.full_[i] = workspace.input_ + offset;
    } else {
      state.profiles_[i - 18] = workspace.input_ + offset;
    }
    offset += input.sizes_[i];
  }
  for (std::size_t i = 0; i < 14; ++i) {
    state.half_[i] = workspace.half_ + i * half_points;
  }
  state.profiles_[10] = workspace.volume_;
  state.jacobian_extrema_ = workspace.extrema_;
  state.jacobian_partials_ = workspace.extrema_ + 2;
  state.profile_terms_ = workspace.terms_;
  state.dimensions_ = input.dimensions_;
  state.scalars_ = input.scalars_;
  state.stream_ = workspace.stream_;
  Check(cudaMemcpyAsync(workspace.input_, workspace.host_input_,
                        inputs * sizeof(double), cudaMemcpyHostToDevice,
                        workspace.stream_));
  LaunchGpuResidentFields(state);
  std::array<const double*, 38> result;
  for (std::size_t i = 0; i < 18; ++i) {
    result[i] = state.full_[i];
  }
  for (std::size_t i = 0; i < 14; ++i) {
    result[18 + i] = state.half_[i];
  }
  result[32] = state.profiles_[10];
  result[33] = state.profiles_[6];
  result[34] = state.profiles_[7];
  result[35] = state.profiles_[8];
  result[36] = state.profiles_[9];
  result[37] = state.jacobian_extrema_;
  offset = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const std::size_t expected =
        i < 18
            ? input.sizes_[i]
            : (i < 32 ? half_points
                      : (i < 35 ? std::size_t(ns - 1)
                                : (i < 37 ? std::size_t(ns) : std::size_t{2})));
    if (m_output.sizes_[i] != expected) {
      throw std::runtime_error("Resident reference output shape mismatch");
    }
    Check(cudaMemcpyAsync(workspace.host_output_ + offset, result[i],
                          expected * sizeof(double), cudaMemcpyDeviceToHost,
                          workspace.stream_));
    offset += expected;
  }
  Check(cudaStreamSynchronize(workspace.stream_));
  offset = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    std::memcpy(m_output.data_[i], workspace.host_output_ + offset,
                m_output.sizes_[i] * sizeof(double));
    offset += m_output.sizes_[i];
  }
}

#ifdef VMECPP_GPU_RESIDENT_CHAIN
namespace {
GpuResidentDeviceState PrepareResidentFourierFields(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    ResidentWorkspace& m_workspace, std::size_t output_count) {
  if (fourier.dimensions_ != reference.dimensions_) {
    throw std::runtime_error(
        "Inverse and field references have different dimensions");
  }
  GpuFourierOutput shape{};
  std::copy_n(reference.sizes_.begin(), 18, shape.sizes_.begin());
  GpuDeviceFourierOutput geometry{};
  RunGpuFourierResident(fourier, shape, geometry);
  const auto stream = static_cast<cudaStream_t>(geometry.stream_);
  auto& workspace = m_workspace;
  const int ns = reference.dimensions_[11];
  const std::size_t half_points =
      std::size_t(ns - 1) * reference.dimensions_[7] * reference.dimensions_[8];
  const auto inputs = std::accumulate(reference.sizes_.begin() + 18,
                                      reference.sizes_.end(), std::size_t{0});
  const auto outputs = output_count;
  workspace.Resize(inputs, outputs, half_points, ns);
  GpuResidentDeviceState state{};
  state.full_ = geometry.data_;
  state.dimensions_ = reference.dimensions_;
  state.scalars_ = reference.scalars_;
  state.stream_ = stream;
  state.jacobian_extrema_ = workspace.extrema_;
  state.jacobian_partials_ = workspace.extrema_ + 2;
  state.profile_terms_ = workspace.terms_;
  std::size_t offset = 0;
  for (std::size_t i = 18; i < reference.data_.size(); ++i) {
    if (reference.sizes_[i] != 0) {
      std::memcpy(workspace.host_input_ + offset, reference.data_[i],
                  reference.sizes_[i] * sizeof(double));
    }
    state.profiles_[i - 18] = workspace.input_ + offset;
    offset += reference.sizes_[i];
  }
  state.profiles_[10] = workspace.volume_;
  for (std::size_t i = 0; i < 14; ++i) {
    state.half_[i] = workspace.half_ + i * half_points;
  }
  Check(cudaMemcpyAsync(workspace.input_, workspace.host_input_,
                        inputs * sizeof(double), cudaMemcpyHostToDevice,
                        stream));
  LaunchGpuResidentFields(state);
  return state;
}
}  // namespace

extern "C" void RunGpuResidentFourierFields(const GpuFourierInput& fourier,
                                            const GpuResidentInput& reference,
                                            GpuResidentOutput& m_output) {
  thread_local ResidentWorkspace workspace;
  const auto outputs = std::accumulate(m_output.sizes_.begin(),
                                       m_output.sizes_.end(), std::size_t{0});
  auto state =
      PrepareResidentFourierFields(fourier, reference, workspace, outputs);
  const auto stream = static_cast<cudaStream_t>(state.stream_);
  const int ns = reference.dimensions_[11];
  const std::size_t half_points =
      std::size_t(ns - 1) * reference.dimensions_[7] * reference.dimensions_[8];
  // Only the validation boundary materializes arrays. No geometry H2D/D2H
  // occurs between the inverse transform and the field kernels above.
  std::array<const double*, 38> result;
  for (std::size_t i = 0; i < 18; ++i) {
    result[i] = state.full_[i];
  }
  for (std::size_t i = 0; i < 14; ++i) {
    result[18 + i] = state.half_[i];
  }
  result[32] = state.profiles_[10];
  result[33] = state.profiles_[6];
  result[34] = state.profiles_[7];
  result[35] = state.profiles_[8];
  result[36] = state.profiles_[9];
  result[37] = state.jacobian_extrema_;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const std::size_t expected =
        i < 18
            ? reference.sizes_[i]
            : (i < 32 ? half_points
                      : (i < 35 ? std::size_t(ns - 1)
                                : (i < 37 ? std::size_t(ns) : std::size_t{2})));
    if (m_output.sizes_[i] != expected) {
      throw std::runtime_error("Resident chain output shape mismatch");
    }
    Check(cudaMemcpyAsync(workspace.host_output_ + offset, result[i],
                          expected * sizeof(double), cudaMemcpyDeviceToHost,
                          stream));
    offset += expected;
  }
  Check(cudaStreamSynchronize(stream));
  offset = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    std::memcpy(m_output.data_[i], workspace.host_output_ + offset,
                m_output.sizes_[i] * sizeof(double));
    offset += m_output.sizes_[i];
  }
}

extern "C" void RunGpuResidentMhdChain(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& constants,
    GpuResidentForceOutput& m_output) {
  const int ns = reference.dimensions_[11];
  if (constants.adiabatic_index_ != 0.0 ||
      constants.sizes_[0] != std::size_t(ns - 1) ||
      constants.sizes_[1] != std::size_t(ns)) {
    throw std::runtime_error(
        "Resident force constants do not match the unchanged gamma=0 deck");
  }
  const std::size_t half_count =
      std::size_t(ns - 1) * reference.dimensions_[7] * reference.dimensions_[8];
  const std::size_t full_count =
      std::size_t(ns) * reference.dimensions_[7] * reference.dimensions_[8];
  const std::size_t outputs = 13 * half_count + 4 * full_count;
  for (std::size_t i = 0; i < m_output.sizes_.size(); ++i) {
    const auto expected = i < 12 || i == 16 ? half_count : full_count;
    if (m_output.sizes_[i] != expected) {
      throw std::runtime_error("Resident MHD output shape mismatch");
    }
  }
  thread_local ResidentWorkspace workspace;
  thread_local ResidentForceWorkspace force_workspace;
  auto geometry =
      PrepareResidentFourierFields(fourier, reference, workspace, outputs);
  const auto stream = static_cast<cudaStream_t>(geometry.stream_);
  const std::size_t constant_count = constants.sizes_[0] + constants.sizes_[1];
  force_workspace.Resize(constant_count, outputs);
  std::memcpy(force_workspace.host_constants_, constants.data_[0],
              constants.sizes_[0] * sizeof(double));
  std::memcpy(force_workspace.host_constants_ + constants.sizes_[0],
              constants.data_[1], constants.sizes_[1] * sizeof(double));
  Check(cudaMemcpyAsync(
      force_workspace.constants_, force_workspace.host_constants_,
      constant_count * sizeof(double), cudaMemcpyHostToDevice, stream));
  GpuResidentForceDeviceState state{};
  state.geometry_ = geometry;
  state.adiabatic_index_ = constants.adiabatic_index_;
  state.constants_ = {force_workspace.constants_,
                      force_workspace.constants_ + constants.sizes_[0]};
  for (std::size_t i = 0; i < 12; ++i) {
    state.mhd_[i] = force_workspace.values_ + i * half_count;
  }
  for (std::size_t i = 0; i < 4; ++i) {
    state.lambda_[i] =
        force_workspace.values_ + 12 * half_count + i * full_count;
  }
  state.total_pressure_ =
      force_workspace.values_ + 12 * half_count + 4 * full_count;
  LaunchGpuResidentForces(state);
  // Materialize only at the reference comparison boundary, never between
  // inverse, geometry, magnetic fields, pressure, lambda, and MHD forces.
  Check(cudaMemcpyAsync(workspace.host_output_, force_workspace.values_,
                        outputs * sizeof(double), cudaMemcpyDeviceToHost,
                        stream));
  Check(cudaStreamSynchronize(stream));
  std::size_t offset = 0;
  for (std::size_t i = 0; i < m_output.sizes_.size(); ++i) {
    std::memcpy(m_output.data_[i], workspace.host_output_ + offset,
                m_output.sizes_[i] * sizeof(double));
    offset += m_output.sizes_[i];
  }
}

extern "C" void RunGpuResidentForceEvalWithStages(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output, GpuResidentPreForceStage pre_stage,
    GpuForceEvaluationTail tail, void* m_context) {
  const auto& d = reference.dimensions_;
  const int ns = d[11];
  const std::size_t full = std::size_t(ns) * d[7] * d[8];
  const std::size_t half = std::size_t(ns - 1) * d[7] * d[8];
  const std::size_t coefficients = std::size_t(ns) * d[5] * (d[6] + 1);
  if (d[4] != ns || d[1] != ns - 1 || forces.adiabatic_index_ != 0.0 ||
      forces.sizes_[0] != std::size_t(ns - 1) ||
      forces.sizes_[1] != std::size_t(ns) || constraints.sizes_[0] != full ||
      constraints.sizes_[1] != full ||
      constraints.sizes_[2] < std::size_t(d[5]) ||
      constraints.sizes_[3] != std::size_t(ns)) {
    throw std::runtime_error(
        "Resident force evaluation constants do not match the fixed-boundary "
        "deck");
  }
  for (std::size_t i = 0; i < 10; ++i) {
    if (m_output.sizes_[i] != (i < 6 ? coefficients : full)) {
      throw std::runtime_error(
          "Resident force evaluation output shape mismatch");
    }
  }
  const std::size_t output_count = 6 * coefficients + 4 * full;
  thread_local ResidentWorkspace workspace;
  thread_local ResidentForceWorkspace force_workspace;
  thread_local ResidentForceWorkspace evaluation_workspace;
  auto geometry =
      PrepareResidentFourierFields(fourier, reference, workspace, output_count);
  const auto stream = static_cast<cudaStream_t>(geometry.stream_);
  const std::size_t force_count = 13 * half + 4 * full;
  const std::size_t force_constants = forces.sizes_[0] + forces.sizes_[1];
  force_workspace.Resize(force_constants, force_count);
  std::memcpy(force_workspace.host_constants_, forces.data_[0],
              forces.sizes_[0] * sizeof(double));
  std::memcpy(force_workspace.host_constants_ + forces.sizes_[0],
              forces.data_[1], forces.sizes_[1] * sizeof(double));
  Check(cudaMemcpyAsync(
      force_workspace.constants_, force_workspace.host_constants_,
      force_constants * sizeof(double), cudaMemcpyHostToDevice, stream));
  GpuResidentConstraintDeviceState state{};
  state.force_.geometry_ = geometry;
  state.force_.adiabatic_index_ = forces.adiabatic_index_;
  state.force_.constants_ = {force_workspace.constants_,
                             force_workspace.constants_ + forces.sizes_[0]};
  for (std::size_t i = 0; i < 12; ++i) {
    state.force_.mhd_[i] = force_workspace.values_ + i * half;
  }
  for (std::size_t i = 0; i < 4; ++i) {
    state.force_.lambda_[i] = force_workspace.values_ + 12 * half + i * full;
  }
  state.force_.total_pressure_ = force_workspace.values_ + 12 * half + 4 * full;
  const std::size_t toroidal = std::size_t(ns - 1) * (d[5] - 2) * d[7];
  const std::size_t spectral = std::size_t(ns - 1) * (d[5] - 2) * (d[6] + 1);
  const std::size_t forward_partials = std::size_t{12} * ns * d[5] * d[7];
  const std::size_t scratch = 8 * full + 4 * toroidal + 2 * spectral +
                              forward_partials + 6 * coefficients;
  const auto constant_count = std::accumulate(
      constraints.sizes_.begin(), constraints.sizes_.end(), std::size_t{0});
  evaluation_workspace.Resize(constant_count, scratch);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < constraints.data_.size(); ++i) {
    std::memcpy(evaluation_workspace.host_constants_ + offset,
                constraints.data_[i], constraints.sizes_[i] * sizeof(double));
    state.constants_[i] = evaluation_workspace.constants_ + offset;
    offset += constraints.sizes_[i];
  }
  Check(cudaMemcpyAsync(
      evaluation_workspace.constants_, evaluation_workspace.host_constants_,
      constant_count * sizeof(double), cudaMemcpyHostToDevice, stream));
  double* cursor = evaluation_workspace.values_;
  for (auto& value : state.geometry_) {
    value = cursor;
    cursor += full;
  }
  for (auto& value : state.poloidal_) {
    value = cursor;
    cursor += toroidal;
  }
  for (auto& value : state.spectral_) {
    value = cursor;
    cursor += spectral;
  }
  for (auto& value : state.toroidal_) {
    value = cursor;
    cursor += toroidal;
  }
  for (auto& value : state.forces_) {
    value = cursor;
    cursor += full;
  }
  Check(
      cudaMemsetAsync(state.forces_[0], 0, 4 * full * sizeof(double), stream));
  GpuForwardDeviceState forward{};
  forward.partials_ = cursor;
  cursor += forward_partials;
  double* coefficient_base = cursor;
  for (std::size_t i = 0; i < 6; ++i) {
    forward.output_[i] = cursor;
    cursor += coefficients;
    forward.output_sizes_[i] = coefficients;
  }
  forward.dimensions_ = d;
  forward.stream_ = stream;
  const auto& m = state.force_.mhd_;
  const auto& l = state.force_.lambda_;
  const auto& c = state.constants_;
  const auto& f = state.forces_;
  forward.input_ = {m[0], m[1], m[2],  m[3],  l[0], l[1], m[4],  m[5],
                    m[6], m[7], l[2],  l[3],  m[8], m[9], m[10], m[11],
                    f[0], f[1], f[2],  f[3],  c[4], c[5], c[6],  c[7],
                    c[8], c[9], c[10], c[11], c[12]};
  if (pre_stage == nullptr) {
    LaunchGpuResidentConstraints(state);
    LaunchGpuResidentForces(state.force_);
  } else {
    LaunchGpuResidentConstraintGeometry(state);
    LaunchGpuResidentForces(state.force_);
    pre_stage(state, m_context);
    LaunchGpuResidentConstraintFilter(state);
  }
  LaunchGpuResidentConstraintAssembly(state);
  LaunchGpuForwardDevice(forward);
  if (tail != nullptr) tail(forward.output_, d, stream, m_context);
  // These downloads are solely for exact comparison of the complete force
  // evaluation. Its internal stages above share device-owned arrays.
  Check(cudaMemcpyAsync(workspace.host_output_, coefficient_base,
                        6 * coefficients * sizeof(double),
                        cudaMemcpyDeviceToHost, stream));
  Check(cudaMemcpyAsync(workspace.host_output_ + 6 * coefficients,
                        state.geometry_[0], 4 * full * sizeof(double),
                        cudaMemcpyDeviceToHost, stream));
  Check(cudaStreamSynchronize(stream));
  offset = 0;
  for (std::size_t i = 0; i < 10; ++i) {
    std::memcpy(m_output.data_[i], workspace.host_output_ + offset,
                m_output.sizes_[i] * sizeof(double));
    offset += m_output.sizes_[i];
  }
}

extern "C" void RunGpuResidentForceEvalWithTail(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output, GpuForceEvaluationTail tail,
    void* m_context) {
  RunGpuResidentForceEvalWithStages(fourier, reference, forces, constraints,
                                    m_output, nullptr, tail, m_context);
}

extern "C" void RunGpuResidentForceEval(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output) {
  RunGpuResidentForceEvalWithTail(fourier, reference, forces, constraints,
                                  m_output, nullptr, nullptr);
}
#endif  // VMECPP_GPU_RESIDENT_CHAIN
