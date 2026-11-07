// SPDX-License-Identifier: MIT
// Isolated N6/N12 launch study. Independent chains retain original RN order.
// N6 and N12 duplicate tempR/tempZ work across cc/ss or sc/cs chains. These
// extra instructions are overhead, not additional useful transform work.
#ifndef VMECPP_EXPERIMENT_GPU_FORWARD_SCALAR_GROUPS_KERNELS_H_
#define VMECPP_EXPERIMENT_GPU_FORWARD_SCALAR_GROUPS_KERNELS_H_

#if VMECPP_FORWARD_PARTIAL_GROUPS != 6 && VMECPP_FORWARD_PARTIAL_GROUPS != 12
#error "The scalar-group study requires six or twelve partial groups"
#endif

__global__ void ForwardPartials(ForwardPointers d, ForwardParameters p,
                                double* __restrict__ m_partials) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int count =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * p.n_zeta_;
  if (task >= count) return;
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % p.mpol_;
  const int jF = task / (p.n_zeta_ * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) return;
  const int group = blockIdx.y;
  constexpr int kChannels = 12 / VMECPP_FORWARD_PARTIAL_GROUPS;
  const int pair = group * kChannels / 2;
  const bool main_enabled = kChannels == 2 || group % 2 == 0;
  const bool derivative_enabled = kChannels == 2 || group % 2 == 1;
  const bool even = m % 2 == 0;
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  const bool enabled = pair >= 4 || rz;
  const int base = ((jF - p.ns_min_f_) * p.n_zeta_ + k) * p.n_theta_eff_;
  const int mbase = m * p.n_theta_reduced_;
  double main_sum = 0.0, derivative_sum = 0.0;
  if (enabled) {
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      if (pair < 2) {
        const double* armn = even ? d.armn_e_ : d.armn_o_;
        const double* brmn = even ? d.brmn_e_ : d.brmn_o_;
        const double* crmn = even ? d.crmn_e_ : d.crmn_o_;
        const double* frcon = even ? d.frcon_e_ : d.frcon_o_;
        if (derivative_enabled)
          derivative_sum -= crmn[idx] * (pair == 0 ? cosmui : sinmui);
        if (main_enabled) {
          const double tempR = armn[idx] + d.xmpq_[m] * frcon[idx];
          if (pair == 0)
            main_sum += tempR * cosmui + brmn[idx] * sinmumi;
          else
            main_sum += tempR * sinmui + brmn[idx] * cosmumi;
        }
      } else if (pair < 4) {
        const double* azmn = even ? d.azmn_e_ : d.azmn_o_;
        const double* bzmn = even ? d.bzmn_e_ : d.bzmn_o_;
        const double* czmn = even ? d.czmn_e_ : d.czmn_o_;
        const double* fzcon = even ? d.fzcon_e_ : d.fzcon_o_;
        if (derivative_enabled)
          derivative_sum -= czmn[idx] * (pair == 2 ? sinmui : cosmui);
        if (main_enabled) {
          const double tempZ = azmn[idx] + d.xmpq_[m] * fzcon[idx];
          if (pair == 2)
            main_sum += tempZ * sinmui + bzmn[idx] * cosmumi;
          else
            main_sum += tempZ * cosmui + bzmn[idx] * sinmumi;
        }
      } else {
        const double* blmn = even ? d.blmn_e_ : d.blmn_o_;
        const double* clmn = even ? d.clmn_e_ : d.clmn_o_;
        if (derivative_enabled)
          derivative_sum -= clmn[idx] * (pair == 4 ? sinmui : cosmui);
        if (main_enabled)
          main_sum += blmn[idx] * (pair == 4 ? cosmumi : sinmumi);
      }
    }
  }
  // The LCFS R/Z zeros must overwrite scratch despite shorter R/Z inputs.
  if constexpr (kChannels == 2) {
    m_partials[(2 * group) * count + task] = main_sum;
    m_partials[(2 * group + 1) * count + task] = derivative_sum;
  } else {
    m_partials[group * count + task] = main_enabled ? main_sum : derivative_sum;
  }
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
  if (index >= points) return;
  const int n = index % (p.ntor_ + 1);
  const int m = (index / (p.ntor_ + 1)) % p.mpol_;
  const int jF = index / ((p.ntor_ + 1) * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) return;
  const int group = blockIdx.y;
  const bool enabled =
      group >= 4 ? jF >= 1 : jF < p.ns_max_f_ && jF < p.ns_ - 1;
  double sum = 0.0;
  if (enabled) {
    for (int k = 0; k < p.n_zeta_; ++k) {
      const int task = ((jF - p.ns_min_f_) * p.mpol_ + m) * p.n_zeta_ + k;
      const int idx_kn = k * (p.nnyq2_ + 1) + n;
      if (group % 2 == 0)
        sum += partials[(2 * group) * count + task] * d.cosnv_[idx_kn] +
               partials[(2 * group + 1) * count + task] * d.sinnvn_[idx_kn];
      else
        sum += partials[(2 * group) * count + task] * d.sinnv_[idx_kn] +
               partials[(2 * group + 1) * count + task] * d.cosnvn_[idx_kn];
    }
  }
  if (group == 0)
    m_frcc[index] = sum;
  else if (group == 1)
    m_frss[index] = sum;
  else if (group == 2)
    m_fzsc[index] = sum;
  else if (group == 3)
    m_fzcs[index] = sum;
  else if (group == 4)
    m_flsc[index] = sum;
  else
    m_flcs[index] = sum;
}

#endif  // VMECPP_EXPERIMENT_GPU_FORWARD_SCALAR_GROUPS_KERNELS_H_
