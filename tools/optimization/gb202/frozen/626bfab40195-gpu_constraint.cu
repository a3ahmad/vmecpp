// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Resident constraint filtering, retaining every l/k/n/m reduction order.
#include <cuda_runtime.h>

#include <stdexcept>

#include "gpu_resident.h"

namespace {
struct Parameters {
  int ns_;
  int mpol_;
  int ntor_;
  int n_zeta_;
  int n_theta_;
  int nnyq2_;
};
struct SpatialPointers {
  const double* ru_e_;
  const double* ru_o_;
  const double* zu_e_;
  const double* zu_o_;
  const double* r_con_;
  const double* z_con_;
  const double* r_con0_;
  const double* z_con0_;
  const double* sqrt_sf_;
  double* effective_;
  double* ru_full_;
  double* zu_full_;
  double* filtered_;
};
struct BasisPointers {
  const double* cosmui_;
  const double* sinmui_;
  const double* cosmu_;
  const double* sinmu_;
  const double* cosnv_;
  const double* sinnv_;
  const double* faccon_;
  const double* tcon_;
};
struct ScratchPointers {
  double* poloidal0_;
  double* poloidal1_;
  double* spectral0_;
  double* spectral1_;
  double* toroidal0_;
  double* toroidal1_;
};
struct AssemblyPointers {
  double* br_e_;
  double* br_o_;
  double* bz_e_;
  double* bz_o_;
  double* frcon_e_;
  double* frcon_o_;
  double* fzcon_e_;
  double* fzcon_o_;
};

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

__global__ void ConstraintGeometryKernel(SpatialPointers m_fields,
                                         Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int angular = p.n_zeta_ * p.n_theta_;
  if (index >= p.ns_ * angular) {
    return;
  }
  const int jF = index / angular;
  const double sqrt_s = m_fields.sqrt_sf_[jF];
  m_fields.ru_full_[index] =
      m_fields.ru_e_[index] + sqrt_s * m_fields.ru_o_[index];
  m_fields.zu_full_[index] =
      m_fields.zu_e_[index] + sqrt_s * m_fields.zu_o_[index];
  if (jF == 0) {
    m_fields.effective_[index] = 0.0;
    return;
  }
  m_fields.effective_[index] =
      (m_fields.r_con_[index] - m_fields.r_con0_[index]) *
          m_fields.ru_full_[index] +
      (m_fields.z_con_[index] - m_fields.z_con0_[index]) *
          m_fields.zu_full_[index];
}

__global__ void ConstraintPoloidalKernel(SpatialPointers f, BasisPointers b,
                                         ScratchPointers m_work, Parameters p) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int modes = p.mpol_ - 2;
  const int count = (p.ns_ - 1) * modes * p.n_zeta_;
  if (task >= count) {
    return;
  }
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % modes + 1;
  const int jF = task / (p.n_zeta_ * modes);
  double w0 = 0.0, w1 = 0.0;
  if (jF > 0) {
    for (int l = 0; l < p.n_theta_; ++l) {
      const int spatial = (jF * p.n_zeta_ + k) * p.n_theta_ + l;
      const int basis = m * p.n_theta_ + l;
      w0 += f.effective_[spatial] * b.sinmui_[basis];
      w1 += f.effective_[spatial] * b.cosmui_[basis];
    }
  }
  m_work.poloidal0_[task] = w0;
  m_work.poloidal1_[task] = w1;
}

__global__ void ConstraintSpectralKernel(BasisPointers b,
                                         ScratchPointers m_work, Parameters p) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int modes = p.mpol_ - 2;
  const int count = (p.ns_ - 1) * modes * (p.ntor_ + 1);
  if (task >= count) {
    return;
  }
  const int n = task % (p.ntor_ + 1);
  const int m = (task / (p.ntor_ + 1)) % modes;
  const int jF = task / ((p.ntor_ + 1) * modes);
  double gsc = 0.0, gcs = 0.0;
  if (jF > 0) {
    for (int k = 0; k < p.n_zeta_; ++k) {
      const int spatial = (jF * modes + m) * p.n_zeta_ + k;
      const int basis = k * (p.nnyq2_ + 1) + n;
      gsc += b.cosnv_[basis] * m_work.poloidal0_[spatial] * b.tcon_[jF];
      gcs += b.sinnv_[basis] * m_work.poloidal1_[spatial] * b.tcon_[jF];
    }
  }
  m_work.spectral0_[task] = gsc;
  m_work.spectral1_[task] = gcs;
}

__global__ void ConstraintToroidalKernel(BasisPointers b,
                                         ScratchPointers m_work, Parameters p) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int modes = p.mpol_ - 2;
  const int count = (p.ns_ - 1) * modes * p.n_zeta_;
  if (task >= count) {
    return;
  }
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % modes;
  const int jF = task / (p.n_zeta_ * modes);
  double w0 = 0.0, w1 = 0.0;
  if (jF > 0) {
    for (int n = 0; n < p.ntor_ + 1; ++n) {
      const int spatial = (jF * modes + m) * (p.ntor_ + 1) + n;
      const int basis = k * (p.nnyq2_ + 1) + n;
      w0 += m_work.spectral0_[spatial] * b.cosnv_[basis];
      w1 += m_work.spectral1_[spatial] * b.sinnv_[basis];
    }
  }
  m_work.toroidal0_[task] = w0;
  m_work.toroidal1_[task] = w1;
}

__global__ void ConstraintSynthesisKernel(SpatialPointers m_fields,
                                          BasisPointers b, ScratchPointers work,
                                          Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int angular = p.n_zeta_ * p.n_theta_;
  if (index >= p.ns_ * angular) {
    return;
  }
  const int jF = index / angular;
  const int k = (index % angular) / p.n_theta_;
  const int l = index % p.n_theta_;
  double value = 0.0;
  if (jF > 0 && jF < p.ns_ - 1) {
    for (int m = 1; m < p.mpol_ - 1; ++m) {
      const int task = (jF * (p.mpol_ - 2) + m - 1) * p.n_zeta_ + k;
      const int basis = m * p.n_theta_ + l;
      value += b.faccon_[m] * (work.toroidal0_[task] * b.sinmu_[basis] +
                               work.toroidal1_[task] * b.cosmu_[basis]);
    }
  }
  m_fields.filtered_[index] = value;
}

__global__ void ConstraintAssemblyKernel(SpatialPointers f,
                                         AssemblyPointers m_forces,
                                         Parameters p) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int angular = p.n_zeta_ * p.n_theta_;
  if (index >= (p.ns_ - 1) * angular) {
    return;
  }
  const int jF = index / angular;
  const double brcon =
      (f.r_con_[index] - f.r_con0_[index]) * f.filtered_[index];
  const double bzcon =
      (f.z_con_[index] - f.z_con0_[index]) * f.filtered_[index];
  m_forces.br_e_[index] += brcon;
  m_forces.bz_e_[index] += bzcon;
  m_forces.br_o_[index] += brcon * f.sqrt_sf_[jF];
  m_forces.bz_o_[index] += bzcon * f.sqrt_sf_[jF];
  m_forces.frcon_e_[index] = f.ru_full_[index] * f.filtered_[index];
  m_forces.fzcon_e_[index] = f.zu_full_[index] * f.filtered_[index];
  m_forces.frcon_o_[index] = m_forces.frcon_e_[index] * f.sqrt_sf_[jF];
  m_forces.fzcon_o_[index] = m_forces.fzcon_e_[index] * f.sqrt_sf_[jF];
}

SpatialPointers Spatial(const GpuResidentConstraintDeviceState& state) {
  const auto& f = state.force_.geometry_.full_;
  const auto& q = state.force_.geometry_.profiles_;
  const auto& c = state.constants_;
  const auto& o = state.geometry_;
  return {f[2], f[3], f[8], f[9], f[16], f[17], c[0],
          c[1], q[0], o[0], o[1], o[2],  o[3]};
}
Parameters GetParameters(const GpuResidentConstraintDeviceState& state) {
  const auto& d = state.force_.geometry_.dimensions_;
  if (d[0] != 0 || d[2] != 0 || d[12] != 0 || d[5] < 3 || d[8] != d[9]) {
    throw std::runtime_error(
        "Resident constraint requires one symmetric fixed-boundary partition");
  }
  return {d[11], d[5], d[6], d[7], d[8], d[10]};
}
}  // namespace

extern "C" void LaunchGpuResidentConstraintGeometry(
    GpuResidentConstraintDeviceState& m_state) {
  const auto p = GetParameters(m_state);
  const auto spatial = Spatial(m_state);
  const auto stream =
      static_cast<cudaStream_t>(m_state.force_.geometry_.stream_);
  const int full_points = p.ns_ * p.n_zeta_ * p.n_theta_;
  ConstraintGeometryKernel<<<(full_points + 127) / 128, 128, 0, stream>>>(
      spatial, p);
  Check(cudaGetLastError());
}

extern "C" void LaunchGpuResidentConstraintFilter(
    GpuResidentConstraintDeviceState& m_state) {
  const auto p = GetParameters(m_state);
  const auto spatial = Spatial(m_state);
  const auto& c = m_state.constants_;
  const BasisPointers basis{c[5], c[6], c[13], c[14], c[9], c[10], c[2], c[3]};
  const ScratchPointers work{m_state.poloidal_[0], m_state.poloidal_[1],
                             m_state.spectral_[0], m_state.spectral_[1],
                             m_state.toroidal_[0], m_state.toroidal_[1]};
  const auto stream =
      static_cast<cudaStream_t>(m_state.force_.geometry_.stream_);
  const int full_points = p.ns_ * p.n_zeta_ * p.n_theta_;
  const int toroidal = (p.ns_ - 1) * (p.mpol_ - 2) * p.n_zeta_;
  const int spectral = (p.ns_ - 1) * (p.mpol_ - 2) * (p.ntor_ + 1);
  ConstraintPoloidalKernel<<<(toroidal + 127) / 128, 128, 0, stream>>>(
      spatial, basis, work, p);
  Check(cudaGetLastError());
  ConstraintSpectralKernel<<<(spectral + 127) / 128, 128, 0, stream>>>(basis,
                                                                       work, p);
  Check(cudaGetLastError());
  ConstraintToroidalKernel<<<(toroidal + 127) / 128, 128, 0, stream>>>(basis,
                                                                       work, p);
  Check(cudaGetLastError());
  ConstraintSynthesisKernel<<<(full_points + 127) / 128, 128, 0, stream>>>(
      spatial, basis, work, p);
  Check(cudaGetLastError());
}

extern "C" void LaunchGpuResidentConstraints(
    GpuResidentConstraintDeviceState& m_state) {
  LaunchGpuResidentConstraintGeometry(m_state);
  LaunchGpuResidentConstraintFilter(m_state);
}

extern "C" void LaunchGpuResidentConstraintAssembly(
    GpuResidentConstraintDeviceState& m_state) {
  const auto p = GetParameters(m_state);
  const auto spatial = Spatial(m_state);
  const auto& m = m_state.force_.mhd_;
  const auto& c = m_state.forces_;
  const AssemblyPointers output{m[4], m[5], m[6], m[7], c[0], c[1], c[2], c[3]};
  const auto stream =
      static_cast<cudaStream_t>(m_state.force_.geometry_.stream_);
  const int count = (p.ns_ - 1) * p.n_zeta_ * p.n_theta_;
  ConstraintAssemblyKernel<<<(count + 127) / 128, 128, 0, stream>>>(spatial,
                                                                    output, p);
  Check(cudaGetLastError());
}
