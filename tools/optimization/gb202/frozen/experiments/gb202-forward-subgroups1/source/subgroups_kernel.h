// SPDX-License-Identifier: MIT
// Independent value/derivative recurrences; original l order and FP expressions.
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
  const int group = blockIdx.y / 2;
  const bool derivative = blockIdx.y % 2 != 0;
  const bool even = m % 2 == 0;
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  const int base = ((jF - p.ns_min_f_) * p.n_zeta_ + k) * p.n_theta_eff_;
  const int mbase = m * p.n_theta_reduced_;
  double first = 0.0, second = 0.0;
  if (group == 0 && rz && !derivative) {
    const double* armn = even ? d.armn_e_ : d.armn_o_;
    const double* brmn = even ? d.brmn_e_ : d.brmn_o_;
    const double* frcon = even ? d.frcon_e_ : d.frcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      const double tempR = armn[idx] + d.xmpq_[m] * frcon[idx];
      first += tempR * cosmui + brmn[idx] * sinmumi;
      second += tempR * sinmui + brmn[idx] * cosmumi;
    }
  } else if (group == 0 && rz && derivative) {
    const double* crmn = even ? d.crmn_e_ : d.crmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      first -= crmn[idx] * d.cosmui_[idx_ml];
      second -= crmn[idx] * d.sinmui_[idx_ml];
    }
  } else if (group == 1 && rz && !derivative) {
    const double* azmn = even ? d.azmn_e_ : d.azmn_o_;
    const double* bzmn = even ? d.bzmn_e_ : d.bzmn_o_;
    const double* fzcon = even ? d.fzcon_e_ : d.fzcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      const double tempZ = azmn[idx] + d.xmpq_[m] * fzcon[idx];
      first += tempZ * sinmui + bzmn[idx] * cosmumi;
      second += tempZ * cosmui + bzmn[idx] * sinmumi;
    }
  } else if (group == 1 && rz && derivative) {
    const double* czmn = even ? d.czmn_e_ : d.czmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      first -= czmn[idx] * d.sinmui_[idx_ml];
      second -= czmn[idx] * d.cosmui_[idx_ml];
    }
  } else if (group == 2 && !derivative) {
    const double* blmn = even ? d.blmn_e_ : d.blmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      first += blmn[idx] * d.cosmumi_[idx_ml];
      second += blmn[idx] * d.sinmumi_[idx_ml];
    }
  } else if (group == 2 && derivative) {
    const double* clmn = even ? d.clmn_e_ : d.clmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      first -= clmn[idx] * d.sinmui_[idx_ml];
      second -= clmn[idx] * d.cosmui_[idx_ml];
    }
  }
  // Both subgroups preserve the original four-channel partial layout.
  const int channel = 4 * group + (derivative ? 1 : 0);
  m_partials[channel * count + task] = first;
  m_partials[(channel + 2) * count + task] = second;
}
