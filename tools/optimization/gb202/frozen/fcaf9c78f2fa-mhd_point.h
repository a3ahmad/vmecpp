// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Generated from v0.4.11 computeMHDForces; arithmetic order is unchanged.
#ifndef VMECPP_EXPERIMENT_MHD_POINT_H_
#define VMECPP_EXPERIMENT_MHD_POINT_H_
#ifdef __CUDACC__
#define VMECPP_MHD_INLINE __host__ __device__ __forceinline__
#else
#define VMECPP_MHD_INLINE inline
#endif
namespace vmecpp_experiment {
struct MhdForceView {
  const double* r12_;
  const double* ru12_;
  const double* zu12_;
  const double* rs_;
  const double* zs_;
  const double* tau_;
  const double* gsqrt_;
  const double* bsupu_;
  const double* bsupv_;
  const double* total_pressure_;
  const double* r1_e_;
  const double* r1_o_;
  const double* ru_e_;
  const double* ru_o_;
  const double* zu_e_;
  const double* zu_o_;
  const double* z1_o_;
  const double* rv_e_;
  const double* rv_o_;
  const double* zv_e_;
  const double* zv_o_;
  const double* sqrt_sf_;
  const double* sqrt_sh_;
};
struct MhdForceParameters {
  int ns_min_f_;
  int ns_min_f1_;
  int ns_min_h_;
  int ns_max_h_;
  int angular_size_;
  double delta_s_;
};
struct MhdPointForces {
  double armn_e_;
  double armn_o_;
  double azmn_e_;
  double azmn_o_;
  double brmn_e_;
  double brmn_o_;
  double bzmn_e_;
  double bzmn_o_;
  double crmn_e_;
  double crmn_o_;
  double czmn_e_;
  double czmn_o_;
};
struct MhdHalfProducts {
  double p_ = 0.0;
  double rup_ = 0.0;
  double zup_ = 0.0;
  double rsp_ = 0.0;
  double zsp_ = 0.0;
  double taup_ = 0.0;
  double gbubu_ = 0.0;
  double gbubv_ = 0.0;
  double gbvbv_ = 0.0;
};
VMECPP_MHD_INLINE MhdHalfProducts ComputeHalfProducts(
    const MhdForceView& v, const MhdForceParameters& p, int jH, int kl) {
  const int index = (jH - p.ns_min_h_) * p.angular_size_ + kl;
  MhdHalfProducts result;
  result.p_ = v.r12_[index] * v.total_pressure_[index];
  result.rup_ = v.ru12_[index] * result.p_;
  result.zup_ = v.zu12_[index] * result.p_;
  result.rsp_ = v.rs_[index] * result.p_;
  result.zsp_ = v.zs_[index] * result.p_;
  result.taup_ = v.tau_[index] * v.total_pressure_[index];
  result.gbubu_ = v.gsqrt_[index] * v.bsupu_[index] * v.bsupu_[index];
  result.gbubv_ = v.gsqrt_[index] * v.bsupu_[index] * v.bsupv_[index];
  result.gbvbv_ = v.gsqrt_[index] * v.bsupv_[index] * v.bsupv_[index];
  return result;
}
VMECPP_MHD_INLINE void ComputeMhdForcePoint(
    const MhdForceView& v, const MhdForceParameters& p, int jF, int kl,
    MhdPointForces& m_output) {
  MhdHalfProducts inside;
  MhdHalfProducts outside;
  double sqrtSHi = 1.0;
  double sqrtSHo = 1.0;
  if (jF > p.ns_min_f_ || p.ns_min_f_ > 0) {
    const int jH = jF > p.ns_min_f_ ? jF - 1 : p.ns_min_h_;
    inside = ComputeHalfProducts(v, p, jH, kl);
    sqrtSHi = v.sqrt_sh_[jH - p.ns_min_h_];
  }
  if (jF < p.ns_max_h_) {
    outside = ComputeHalfProducts(v, p, jF, kl);
    sqrtSHo = v.sqrt_sh_[jF - p.ns_min_h_];
  }
  const double sFull = v.sqrt_sf_[jF - p.ns_min_f1_] * v.sqrt_sf_[jF - p.ns_min_f1_];
  const int idx_g = (jF - p.ns_min_f1_) * p.angular_size_ + kl;
  m_output.armn_e_ = (outside.zup_ - inside.zup_) / p.delta_s_ +
          0.5 * (outside.taup_ + inside.taup_) -
          0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.r1_e_[idx_g] -
          0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
              v.r1_o_[idx_g];
  m_output.armn_o_ = (outside.zup_ * sqrtSHo - inside.zup_ * sqrtSHi) / p.delta_s_ -
          0.25 * (outside.p_ / sqrtSHo + inside.p_ / sqrtSHi) * v.zu_e_[idx_g] -
          0.25 * (outside.p_ + inside.p_) * v.zu_o_[idx_g] +
          0.5 * (outside.taup_ * sqrtSHo + inside.taup_ * sqrtSHi) -
          0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
              v.r1_e_[idx_g] -
          0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.r1_o_[idx_g] * sFull;
  m_output.azmn_e_ = -(outside.rup_ - inside.rup_) / p.delta_s_;
  m_output.azmn_o_ = -(outside.rup_ * sqrtSHo - inside.rup_ * sqrtSHi) / p.delta_s_ +
          0.25 * (outside.p_ / sqrtSHo + inside.p_ / sqrtSHi) * v.ru_e_[idx_g] +
          0.25 * (outside.p_ + inside.p_) * v.ru_o_[idx_g];
  m_output.brmn_e_ = 0.5 * (outside.zsp_ + inside.zsp_) +
          0.25 * (outside.p_ / sqrtSHo + inside.p_ / sqrtSHi) * v.z1_o_[idx_g] -
          0.5 * (outside.gbubu_ + inside.gbubu_) * v.ru_e_[idx_g] -
          0.5 * (outside.gbubu_ * sqrtSHo + inside.gbubu_ * sqrtSHi) *
              v.ru_o_[idx_g];
  m_output.brmn_o_ = 0.5 * (outside.zsp_ * sqrtSHo + inside.zsp_ * sqrtSHi) +
          0.25 * (outside.p_ + inside.p_) * v.z1_o_[idx_g] -
          0.5 * (outside.gbubu_ * sqrtSHo + inside.gbubu_ * sqrtSHi) *
              v.ru_e_[idx_g] -
          0.5 * (outside.gbubu_ + inside.gbubu_) * v.ru_o_[idx_g] * sFull;
  m_output.bzmn_e_ = -0.5 * (outside.rsp_ + inside.rsp_) -
          0.25 * (outside.p_ / sqrtSHo + inside.p_ / sqrtSHi) * v.r1_o_[idx_g] -
          0.5 * (outside.gbubu_ + inside.gbubu_) * v.zu_e_[idx_g] -
          0.5 * (outside.gbubu_ * sqrtSHo + inside.gbubu_ * sqrtSHi) *
              v.zu_o_[idx_g];
  m_output.bzmn_o_ = -0.5 * (outside.rsp_ * sqrtSHo + inside.rsp_ * sqrtSHi) -
          0.25 * (outside.p_ + inside.p_) * v.r1_o_[idx_g] -
          0.5 * (outside.gbubu_ * sqrtSHo + inside.gbubu_ * sqrtSHi) *
              v.zu_e_[idx_g] -
          0.5 * (outside.gbubu_ + inside.gbubu_) * v.zu_o_[idx_g] * sFull;
  m_output.brmn_e_ += -0.5 * (outside.gbubv_ + inside.gbubv_) * v.rv_e_[idx_g] -
            0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.rv_o_[idx_g];
  m_output.brmn_o_ += -0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.rv_e_[idx_g] -
            0.5 * (outside.gbubv_ + inside.gbubv_) * v.rv_o_[idx_g] * sFull;
  m_output.bzmn_e_ += -0.5 * (outside.gbubv_ + inside.gbubv_) * v.zv_e_[idx_g] -
            0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.zv_o_[idx_g];
  m_output.bzmn_o_ += -0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.zv_e_[idx_g] -
            0.5 * (outside.gbubv_ + inside.gbubv_) * v.zv_o_[idx_g] * sFull;
  m_output.crmn_e_ = 0.5 * (outside.gbubv_ + inside.gbubv_) * v.ru_e_[idx_g] +
            0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.ru_o_[idx_g] +
            0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.rv_e_[idx_g] +
            0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
                v.rv_o_[idx_g];
  m_output.crmn_o_ = 0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.ru_e_[idx_g] +
            0.5 * (outside.gbubv_ + inside.gbubv_) * v.ru_o_[idx_g] * sFull +
            0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
                v.rv_e_[idx_g] +
            0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.rv_o_[idx_g] * sFull;
  m_output.czmn_e_ = 0.5 * (outside.gbubv_ + inside.gbubv_) * v.zu_e_[idx_g] +
            0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.zu_o_[idx_g] +
            0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.zv_e_[idx_g] +
            0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
                v.zv_o_[idx_g];
  m_output.czmn_o_ = 0.5 * (outside.gbubv_ * sqrtSHo + inside.gbubv_ * sqrtSHi) *
                v.zu_e_[idx_g] +
            0.5 * (outside.gbubv_ + inside.gbubv_) * v.zu_o_[idx_g] * sFull +
            0.5 * (outside.gbvbv_ * sqrtSHo + inside.gbvbv_ * sqrtSHi) *
                v.zv_e_[idx_g] +
            0.5 * (outside.gbvbv_ + inside.gbvbv_) * v.zv_o_[idx_g] * sFull;
}
}  // namespace vmecpp_experiment
#undef VMECPP_MHD_INLINE
#endif  // VMECPP_EXPERIMENT_MHD_POINT_H_
