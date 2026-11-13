// SPDX-License-Identifier: MIT
// SPDX-License-Identifier: MIT
// Experimental opt-in R/Z/lambda launch groups. Included after the private
// ForwardPointers/ForwardParameters declarations in an isolated source copy.
// No basis conversion, reassociation, allocation, or synchronization is added.
#ifndef VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_DOUBLE_HELPERS_H_
#define VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_DOUBLE_HELPERS_H_

__device__ __forceinline__ void ForwardPartialsFp64Body(
    ForwardPointers d, ForwardParameters p, double* __restrict__ m_partials) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int count =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * p.n_zeta_;
  if (task >= count) return;
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % p.mpol_;
  const int jF = task / (p.n_zeta_ * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) return;
  const int group = blockIdx.y;
  const bool even = m % 2 == 0;
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  const int base = ((jF - p.ns_min_f_) * p.n_zeta_ + k) * p.n_theta_eff_;
  const int mbase = m * p.n_theta_reduced_;
  double first = 0.0, first_n = 0.0, second = 0.0, second_n = 0.0;
  if (group == 0 && rz) {
    const double* armn = even ? d.armn_e_ : d.armn_o_;
    const double* brmn = even ? d.brmn_e_ : d.brmn_o_;
    const double* crmn = even ? d.crmn_e_ : d.crmn_o_;
    const double* frcon = even ? d.frcon_e_ : d.frcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      first_n -= crmn[idx] * cosmui;
      second_n -= crmn[idx] * sinmui;
      const double tempR = armn[idx] + d.xmpq_[m] * frcon[idx];
      first += tempR * cosmui + brmn[idx] * sinmumi;
      second += tempR * sinmui + brmn[idx] * cosmumi;
    }
  } else if (group == 1 && rz) {
    const double* azmn = even ? d.azmn_e_ : d.azmn_o_;
    const double* bzmn = even ? d.bzmn_e_ : d.bzmn_o_;
    const double* czmn = even ? d.czmn_e_ : d.czmn_o_;
    const double* fzcon = even ? d.fzcon_e_ : d.fzcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      second_n -= czmn[idx] * cosmui;
      first_n -= czmn[idx] * sinmui;
      const double tempZ = azmn[idx] + d.xmpq_[m] * fzcon[idx];
      first += tempZ * sinmui + bzmn[idx] * cosmumi;
      second += tempZ * cosmui + bzmn[idx] * sinmumi;
    }
  } else if (group == 2) {
    const double* blmn = even ? d.blmn_e_ : d.blmn_o_;
    const double* clmn = even ? d.clmn_e_ : d.clmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const double cosmui = d.cosmui_[idx_ml];
      const double sinmui = d.sinmui_[idx_ml];
      const double cosmumi = d.cosmumi_[idx_ml];
      const double sinmumi = d.sinmumi_[idx_ml];
      first += blmn[idx] * cosmumi;
      second += blmn[idx] * sinmumi;
      second_n -= clmn[idx] * cosmui;
      first_n -= clmn[idx] * sinmui;
    }
  }
  // Inactive R/Z rows still overwrite all four channels with literal +0.
  // They must never read the shorter R/Z input allocations at the LCFS.
  m_partials[(4 * group) * count + task] = first;
  m_partials[(4 * group + 1) * count + task] = first_n;
  m_partials[(4 * group + 2) * count + task] = second;
  m_partials[(4 * group + 3) * count + task] = second_n;
}

__device__ __forceinline__ void ForwardFinishFp64Body(
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
#if VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR
  if (jF == 0 && m > 0) {
    if (blockIdx.y == 0) {
      m_frcc[index] = 0.0;
      m_frss[index] = 0.0;
    } else if (blockIdx.y == 1) {
      m_fzsc[index] = 0.0;
      m_fzcs[index] = 0.0;
    } else {
      m_flsc[index] = 0.0;
      m_flcs[index] = 0.0;
    }
    return;
  }
#else
  if (jF == 0 && m > 0) return;
#endif
  const int group = blockIdx.y;
  const bool enabled =
      group == 2 ? jF >= 1 : jF < p.ns_max_f_ && jF < p.ns_ - 1;
  double first = 0.0, second = 0.0;
  if (enabled) {
    for (int k = 0; k < p.n_zeta_; ++k) {
      const int task = ((jF - p.ns_min_f_) * p.mpol_ + m) * p.n_zeta_ + k;
      const int idx_kn = k * (p.nnyq2_ + 1) + n;
      const double cosnv = d.cosnv_[idx_kn];
      const double sinnv = d.sinnv_[idx_kn];
      const double cosnvn = d.cosnvn_[idx_kn];
      const double sinnvn = d.sinnvn_[idx_kn];
      first += partials[(4 * group) * count + task] * cosnv +
               partials[(4 * group + 1) * count + task] * sinnvn;
      second += partials[(4 * group + 2) * count + task] * sinnv +
                partials[(4 * group + 3) * count + task] * cosnvn;
    }
  }
  if (group == 0) {
    m_frcc[index] = first;
    m_frss[index] = second;
  } else if (group == 1) {
    m_fzsc[index] = first;
    m_fzcs[index] = second;
  } else {
    m_flsc[index] = first;
    m_flcs[index] = second;
  }
}

#endif  // VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_DOUBLE_HELPERS_H_

// SPDX-License-Identifier: MIT
// Experimental opt-in R/Z/lambda launch groups. Included after the private
// ForwardPointers/ForwardParameters declarations in an isolated source copy.
// No basis conversion, reassociation, allocation, or synchronization is added.
#ifndef VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_FLOAT_HELPERS_H_
#define VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_FLOAT_HELPERS_H_

__device__ __forceinline__ void ForwardPartialsFp32Body(
    ForwardPointers d, ForwardParameters p, double* __restrict__ m_partials) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int count =
      (p.ns_max_including_lcfs_ - p.ns_min_f_) * p.mpol_ * p.n_zeta_;
  if (task >= count) return;
  const int k = task % p.n_zeta_;
  const int m = (task / p.n_zeta_) % p.mpol_;
  const int jF = task / (p.n_zeta_ * p.mpol_) + p.ns_min_f_;
  if (jF == 0 && m > 0) return;
  const int group = blockIdx.y;
  const bool even = m % 2 == 0;
  const bool rz = jF < p.ns_max_f_ && jF < p.ns_ - 1;
  const int base = ((jF - p.ns_min_f_) * p.n_zeta_ + k) * p.n_theta_eff_;
  const int mbase = m * p.n_theta_reduced_;
  float first = 0.0, first_n = 0.0, second = 0.0, second_n = 0.0;
  if (group == 0 && rz) {
    const double* armn = even ? d.armn_e_ : d.armn_o_;
    const double* brmn = even ? d.brmn_e_ : d.brmn_o_;
    const double* crmn = even ? d.crmn_e_ : d.crmn_o_;
    const double* frcon = even ? d.frcon_e_ : d.frcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const float cosmui = static_cast<float>(d.cosmui_[idx_ml]);
      const float sinmui = static_cast<float>(d.sinmui_[idx_ml]);
      const float cosmumi = static_cast<float>(d.cosmumi_[idx_ml]);
      const float sinmumi = static_cast<float>(d.sinmumi_[idx_ml]);
      first_n -= static_cast<float>(crmn[idx]) * cosmui;
      second_n -= static_cast<float>(crmn[idx]) * sinmui;
      const float tempR =
          static_cast<float>(armn[idx]) +
          static_cast<float>(d.xmpq_[m]) * static_cast<float>(frcon[idx]);
      first += tempR * cosmui + static_cast<float>(brmn[idx]) * sinmumi;
      second += tempR * sinmui + static_cast<float>(brmn[idx]) * cosmumi;
    }
  } else if (group == 1 && rz) {
    const double* azmn = even ? d.azmn_e_ : d.azmn_o_;
    const double* bzmn = even ? d.bzmn_e_ : d.bzmn_o_;
    const double* czmn = even ? d.czmn_e_ : d.czmn_o_;
    const double* fzcon = even ? d.fzcon_e_ : d.fzcon_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const float cosmui = static_cast<float>(d.cosmui_[idx_ml]);
      const float sinmui = static_cast<float>(d.sinmui_[idx_ml]);
      const float cosmumi = static_cast<float>(d.cosmumi_[idx_ml]);
      const float sinmumi = static_cast<float>(d.sinmumi_[idx_ml]);
      second_n -= static_cast<float>(czmn[idx]) * cosmui;
      first_n -= static_cast<float>(czmn[idx]) * sinmui;
      const float tempZ =
          static_cast<float>(azmn[idx]) +
          static_cast<float>(d.xmpq_[m]) * static_cast<float>(fzcon[idx]);
      first += tempZ * sinmui + static_cast<float>(bzmn[idx]) * cosmumi;
      second += tempZ * cosmui + static_cast<float>(bzmn[idx]) * sinmumi;
    }
  } else if (group == 2) {
    const double* blmn = even ? d.blmn_e_ : d.blmn_o_;
    const double* clmn = even ? d.clmn_e_ : d.clmn_o_;
    for (int l = 0; l < p.n_theta_reduced_; ++l) {
      const int idx = base + l;
      const int idx_ml = mbase + l;
      const float cosmui = static_cast<float>(d.cosmui_[idx_ml]);
      const float sinmui = static_cast<float>(d.sinmui_[idx_ml]);
      const float cosmumi = static_cast<float>(d.cosmumi_[idx_ml]);
      const float sinmumi = static_cast<float>(d.sinmumi_[idx_ml]);
      first += static_cast<float>(blmn[idx]) * cosmumi;
      second += static_cast<float>(blmn[idx]) * sinmumi;
      second_n -= static_cast<float>(clmn[idx]) * cosmui;
      first_n -= static_cast<float>(clmn[idx]) * sinmui;
    }
  }
  // Inactive R/Z rows still overwrite all four channels with literal +0.
  // They must never read the shorter R/Z input allocations at the LCFS.
  m_partials[(4 * group) * count + task] = first;
  m_partials[(4 * group + 1) * count + task] = first_n;
  m_partials[(4 * group + 2) * count + task] = second;
  m_partials[(4 * group + 3) * count + task] = second_n;
}

__device__ __forceinline__ void ForwardFinishFp32Body(
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
#if VMECPP_FORWARD_ELIDE_OUTPUT_CLEAR
  if (jF == 0 && m > 0) {
    if (blockIdx.y == 0) {
      m_frcc[index] = 0.0;
      m_frss[index] = 0.0;
    } else if (blockIdx.y == 1) {
      m_fzsc[index] = 0.0;
      m_fzcs[index] = 0.0;
    } else {
      m_flsc[index] = 0.0;
      m_flcs[index] = 0.0;
    }
    return;
  }
#else
  if (jF == 0 && m > 0) return;
#endif
  const int group = blockIdx.y;
  const bool enabled =
      group == 2 ? jF >= 1 : jF < p.ns_max_f_ && jF < p.ns_ - 1;
  float first = 0.0, second = 0.0;
  if (enabled) {
    for (int k = 0; k < p.n_zeta_; ++k) {
      const int task = ((jF - p.ns_min_f_) * p.mpol_ + m) * p.n_zeta_ + k;
      const int idx_kn = k * (p.nnyq2_ + 1) + n;
      const float cosnv = static_cast<float>(d.cosnv_[idx_kn]);
      const float sinnv = static_cast<float>(d.sinnv_[idx_kn]);
      const float cosnvn = static_cast<float>(d.cosnvn_[idx_kn]);
      const float sinnvn = static_cast<float>(d.sinnvn_[idx_kn]);
      first +=
          static_cast<float>(partials[(4 * group) * count + task]) * cosnv +
          static_cast<float>(partials[(4 * group + 1) * count + task]) * sinnvn;
      second +=
          static_cast<float>(partials[(4 * group + 2) * count + task]) * sinnv +
          static_cast<float>(partials[(4 * group + 3) * count + task]) * cosnvn;
    }
  }
  if (group == 0) {
    m_frcc[index] = first;
    m_frss[index] = second;
  } else if (group == 1) {
    m_fzsc[index] = first;
    m_fzcs[index] = second;
  } else {
    m_flsc[index] = first;
    m_flcs[index] = second;
  }
}

#endif  // VMECPP_EXPERIMENT_GPU_FORWARD_PRECISION_FLOAT_HELPERS_H_

// Uniform device-time flag read makes existing CUDA graph nodes switch mode.
__global__ void ForwardPartialsMixedPrecision(ForwardPointers d,
                                              ForwardParameters p,
                                              double* __restrict__ m_partials) {
  const bool use_fp32 =
      *reinterpret_cast<volatile int*>(&forward_fp32_enabled) != 0;
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    atomicAdd(&forward_precision_counts[use_fp32 ? 1 : 0], 1ULL);
  if (use_fp32)
    ForwardPartialsFp32Body(d, p, m_partials);
  else
    ForwardPartialsFp64Body(d, p, m_partials);
}

__global__ void ForwardFinishMixedPrecision(
    ForwardPointers d, ForwardParameters p, const double* __restrict__ partials,
    double* __restrict__ m_frcc, double* __restrict__ m_frss,
    double* __restrict__ m_fzsc, double* __restrict__ m_fzcs,
    double* __restrict__ m_flsc, double* __restrict__ m_flcs) {
  const bool use_fp32 =
      *reinterpret_cast<volatile int*>(&forward_fp32_enabled) != 0;
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    atomicAdd(&forward_precision_counts[use_fp32 ? 1 : 0], 1ULL);
  if (use_fp32)
    ForwardFinishFp32Body(d, p, partials, m_frcc, m_frss, m_fzsc, m_fzcs,
                          m_flsc, m_flcs);
  else
    ForwardFinishFp64Body(d, p, partials, m_frcc, m_frss, m_fzsc, m_fzcs,
                          m_flsc, m_flcs);
}
