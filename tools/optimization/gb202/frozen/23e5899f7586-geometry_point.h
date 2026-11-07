// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Generated from v0.4.11. Preserve every parenthesis and arithmetic order.
#ifndef VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
#define VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
#ifdef __CUDACC__
#define VMECPP_GEOMETRY_POINT_INLINE __host__ __device__ __forceinline__
#else
#define VMECPP_GEOMETRY_POINT_INLINE inline
#endif
namespace vmecpp_experiment {
struct GeometryPointView {
  const double* r1_e_;
  const double* r1_o_;
  const double* z1_e_;
  const double* z1_o_;
  const double* ru_e_;
  const double* ru_o_;
  const double* zu_e_;
  const double* zu_o_;
  const double* rv_e_;
  const double* rv_o_;
  const double* zv_e_;
  const double* zv_o_;
  const double* sqrt_sf_;
  const double* sqrt_sh_;
};
struct GeometryPointParameters {
  int angular_size_;
  int ns_min_f1_;
  int ns_min_h_;
  double delta_s_;
  double d_s_half_ds_interp_;
};
struct GeometryPoint {
  double r12_;
  double ru12_;
  double zu12_;
  double rs_;
  double zs_;
  double tau_;
  double gsqrt_;
  double guu_;
  double guv_;
  double gvv_;
};
VMECPP_GEOMETRY_POINT_INLINE void ComputeGeometryPoint(
    const GeometryPointView& v, const GeometryPointParameters& p,
    int jH, int kl, GeometryPoint& m_output) {
  const int inner_index = (jH - p.ns_min_f1_) * p.angular_size_ + kl;
  const int outer_index = (jH + 1 - p.ns_min_f1_) * p.angular_size_ + kl;
  struct Surface {
    double r1e_;
    double r1o_;
    double z1e_;
    double z1o_;
    double rue_;
    double ruo_;
    double zue_;
    double zuo_;
    double rve_;
    double rvo_;
    double zve_;
    double zvo_;
  };
  const Surface inside{
    v.r1_e_[inner_index],
    v.r1_o_[inner_index],
    v.z1_e_[inner_index],
    v.z1_o_[inner_index],
    v.ru_e_[inner_index],
    v.ru_o_[inner_index],
    v.zu_e_[inner_index],
    v.zu_o_[inner_index],
    v.rv_e_[inner_index],
    v.rv_o_[inner_index],
    v.zv_e_[inner_index],
    v.zv_o_[inner_index]
  };
  const double r1e_o = v.r1_e_[outer_index];
  const double r1o_o = v.r1_o_[outer_index];
  const double z1e_o = v.z1_e_[outer_index];
  const double z1o_o = v.z1_o_[outer_index];
  const double rue_o = v.ru_e_[outer_index];
  const double ruo_o = v.ru_o_[outer_index];
  const double zue_o = v.zu_e_[outer_index];
  const double zuo_o = v.zu_o_[outer_index];
  const double rve_o = v.rv_e_[outer_index];
  const double rvo_o = v.rv_o_[outer_index];
  const double zve_o = v.zv_e_[outer_index];
  const double zvo_o = v.zv_o_[outer_index];
  const double sqrtSH = v.sqrt_sh_[jH - p.ns_min_h_];
  const double sF_i = v.sqrt_sf_[jH - p.ns_min_f1_] *
                      v.sqrt_sf_[jH - p.ns_min_f1_];
  const double sF_o = v.sqrt_sf_[jH + 1 - p.ns_min_f1_] *
                      v.sqrt_sf_[jH + 1 - p.ns_min_f1_];
// R on half-grid
      m_output.r12_ = 0.5 * ((inside.r1e_ + r1e_o) +
                          sqrtSH * (inside.r1o_ + r1o_o));

      // dRdTheta on half-grid
      m_output.ru12_ = 0.5 * ((inside.rue_ + rue_o) +
                           sqrtSH * (inside.ruo_ + ruo_o));

      // dZdTheta on half-grid
      m_output.zu12_ = 0.5 * ((inside.zue_ + zue_o) +
                           sqrtSH * (inside.zuo_ + zuo_o));

      // \tilde{dRds} on half-grid
      m_output.rs_ =
          ((r1e_o - inside.r1e_) + sqrtSH * (r1o_o - inside.r1o_)) /
          p.delta_s_;

      // \tilde{dZds} on half-grid
      m_output.zs_ =
          ((z1e_o - inside.z1e_) + sqrtSH * (z1o_o - inside.z1o_)) /
          p.delta_s_;

      // sqrt(g)/R on half-grid: assemble as governed by product rule
      double tau1 = m_output.ru12_ * m_output.zs_ - m_output.rs_ * m_output.zu12_;
      double tau2 = ruo_o * z1o_o + inside.ruo_ * inside.z1o_ -
                    zuo_o * r1o_o - inside.zuo_ * inside.r1o_ +
                    (rue_o * z1o_o + inside.rue_ * inside.z1o_ -
                     zue_o * r1o_o - inside.zue_ * inside.r1o_) /
                        sqrtSH;
      double tau_val = tau1 + p.d_s_half_ds_interp_ * tau2;

      m_output.tau_ = tau_val;

      
m_output.gsqrt_ = m_output.tau_ * m_output.r12_;
m_output.guu_ = 0.5 * ((inside.rue_ * inside.rue_ +
                           inside.zue_ * inside.zue_) +
                          (rue_o * rue_o + zue_o * zue_o) +
                          sF_i * (inside.ruo_ * inside.ruo_ +
                                  inside.zuo_ * inside.zuo_) +
                          sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                   sqrtSH * ((inside.rue_ * inside.ruo_ +
                              inside.zue_ * inside.zuo_) +
                             (rue_o * ruo_o + zue_o * zuo_o));
m_output.gvv_ = 0.5 * (inside.r1e_ * inside.r1e_ + r1e_o * r1e_o +
                          sF_i * inside.r1o_ * inside.r1o_ +
                          sF_o * r1o_o * r1o_o) +
                   sqrtSH * (inside.r1e_ * inside.r1o_ + r1e_o * r1o_o);
m_output.guv_ = 0.5 * ((inside.rue_ * inside.rve_ +
                             inside.zue_ * inside.zve_) +
                            (rue_o * rve_o + zue_o * zve_o) +
                            sF_i * (inside.ruo_ * inside.rvo_ +
                                    inside.zuo_ * inside.zvo_) +
                            sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                            sqrtSH * ((inside.rue_ * inside.rvo_ +
                                       inside.zue_ * inside.zvo_) +
                                      (rue_o * rvo_o + zue_o * zvo_o) +
                                      (inside.rve_ * inside.ruo_ +
                                       inside.zve_ * inside.zuo_) +
                                      (rve_o * ruo_o + zve_o * zuo_o)));
m_output.gvv_ += 0.5 * ((inside.rve_ * inside.rve_ +
                              inside.zve_ * inside.zve_) +
                             (rve_o * rve_o + zve_o * zve_o) +
                             sF_i * (inside.rvo_ * inside.rvo_ +
                                     inside.zvo_ * inside.zvo_) +
                             sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
                      sqrtSH * ((inside.rve_ * inside.rvo_ +
                                 inside.zve_ * inside.zvo_) +
                                (rve_o * rvo_o + zve_o * zvo_o));
}
}  // namespace vmecpp_experiment
#undef VMECPP_GEOMETRY_POINT_INLINE
#endif  // VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
