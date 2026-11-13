// SPDX-License-Identifier: MIT
// Research-only device-time dispatch; one equilibrium per process.
namespace vmecpp_mixed_inverse {
__device__ int precision_fp32 = 0;
__device__ unsigned long long precision_counts[2] = {};
struct FloatReadView {
  const double* data_;
  __device__ float operator[](int index) const {
    return static_cast<float>(data_[index]);
  }
};

__device__ __forceinline__ void ToroidalPartialsKernelFp64(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ cosnvn, const double* __restrict__ sinnvn,
    double* __restrict__ m_partials, KernelParams params) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int task_count =
      (params.ns_max_f1 - params.ns_min_f1) * params.mpol * params.n_zeta;
  if (task >= task_count) {
    return;
  }
  const int k = task % params.n_zeta;
  const int m = (task / params.n_zeta) % params.mpol;
  const int jF = task / (params.n_zeta * params.mpol) + params.ns_min_f1;
  if (jF == 0 && m > 1) {
    return;
  }
  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;
  // Accumulate from toroidal modes n.
  double rmkcc = 0.0, rmkcc_n = 0.0;
  double rmkss = 0.0, rmkss_n = 0.0;
  double zmksc = 0.0, zmksc_n = 0.0;
  double zmkcs = 0.0, zmkcs_n = 0.0;
  double lmksc = 0.0, lmksc_n = 0.0;
  double lmkcs = 0.0, lmkcs_n = 0.0;

  for (int n = 0; n < ntorp1; ++n) {
    const int idx_kn = k * nnyq2p1 + n;
    const double cosnv_val = cosnv[idx_kn];
    const double sinnv_val = sinnv[idx_kn];
    const double cosnvn_val = cosnvn[idx_kn];
    const double sinnvn_val = sinnvn[idx_kn];

    const int idx_mn = ((jF - params.ns_min_f1) * params.mpol + m) * ntorp1 + n;

    rmkcc += rmncc[idx_mn] * cosnv_val;
    rmkcc_n += rmncc[idx_mn] * sinnvn_val;
    rmkss += rmnss[idx_mn] * sinnv_val;
    rmkss_n += rmnss[idx_mn] * cosnvn_val;
    zmksc += zmnsc[idx_mn] * cosnv_val;
    zmksc_n += zmnsc[idx_mn] * sinnvn_val;
    zmkcs += zmncs[idx_mn] * sinnv_val;
    zmkcs_n += zmncs[idx_mn] * cosnvn_val;
    lmksc += lmnsc[idx_mn] * cosnv_val;
    lmksc_n += lmnsc[idx_mn] * sinnvn_val;
    lmkcs += lmncs[idx_mn] * sinnv_val;
    lmkcs_n += lmncs[idx_mn] * cosnvn_val;
  }

  m_partials[0 * task_count + task] = rmkcc;
  m_partials[1 * task_count + task] = rmkcc_n;
  m_partials[2 * task_count + task] = rmkss;
  m_partials[3 * task_count + task] = rmkss_n;
  m_partials[4 * task_count + task] = zmksc;
  m_partials[5 * task_count + task] = zmksc_n;
  m_partials[6 * task_count + task] = zmkcs;
  m_partials[7 * task_count + task] = zmkcs_n;
  m_partials[8 * task_count + task] = lmksc;
  m_partials[9 * task_count + task] = lmksc_n;
  m_partials[10 * task_count + task] = lmkcs;
  m_partials[11 * task_count + task] = lmkcs_n;
}

__device__ __forceinline__ void ToroidalPartialsKernelFp32(
    const double* __restrict__ rmncc_fp64,
    const double* __restrict__ rmnss_fp64,
    const double* __restrict__ zmnsc_fp64,
    const double* __restrict__ zmncs_fp64,
    const double* __restrict__ lmnsc_fp64,
    const double* __restrict__ lmncs_fp64,
    const double* __restrict__ cosnv_fp64,
    const double* __restrict__ sinnv_fp64,
    const double* __restrict__ cosnvn_fp64,
    const double* __restrict__ sinnvn_fp64, double* __restrict__ m_partials,
    KernelParams params) {
  const FloatReadView rmncc{rmncc_fp64};
  const FloatReadView rmnss{rmnss_fp64};
  const FloatReadView zmnsc{zmnsc_fp64};
  const FloatReadView zmncs{zmncs_fp64};
  const FloatReadView lmnsc{lmnsc_fp64};
  const FloatReadView lmncs{lmncs_fp64};
  const FloatReadView cosnv{cosnv_fp64};
  const FloatReadView sinnv{sinnv_fp64};
  const FloatReadView cosnvn{cosnvn_fp64};
  const FloatReadView sinnvn{sinnvn_fp64};
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
  const int task_count =
      (params.ns_max_f1 - params.ns_min_f1) * params.mpol * params.n_zeta;
  if (task >= task_count) {
    return;
  }
  const int k = task % params.n_zeta;
  const int m = (task / params.n_zeta) % params.mpol;
  const int jF = task / (params.n_zeta * params.mpol) + params.ns_min_f1;
  if (jF == 0 && m > 1) {
    return;
  }
  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;
  // Accumulate from toroidal modes n.
  float rmkcc = 0.0f, rmkcc_n = 0.0f;
  float rmkss = 0.0f, rmkss_n = 0.0f;
  float zmksc = 0.0f, zmksc_n = 0.0f;
  float zmkcs = 0.0f, zmkcs_n = 0.0f;
  float lmksc = 0.0f, lmksc_n = 0.0f;
  float lmkcs = 0.0f, lmkcs_n = 0.0f;

  for (int n = 0; n < ntorp1; ++n) {
    const int idx_kn = k * nnyq2p1 + n;
    const float cosnv_val = cosnv[idx_kn];
    const float sinnv_val = sinnv[idx_kn];
    const float cosnvn_val = cosnvn[idx_kn];
    const float sinnvn_val = sinnvn[idx_kn];

    const int idx_mn = ((jF - params.ns_min_f1) * params.mpol + m) * ntorp1 + n;

    rmkcc += rmncc[idx_mn] * cosnv_val;
    rmkcc_n += rmncc[idx_mn] * sinnvn_val;
    rmkss += rmnss[idx_mn] * sinnv_val;
    rmkss_n += rmnss[idx_mn] * cosnvn_val;
    zmksc += zmnsc[idx_mn] * cosnv_val;
    zmksc_n += zmnsc[idx_mn] * sinnvn_val;
    zmkcs += zmncs[idx_mn] * sinnv_val;
    zmkcs_n += zmncs[idx_mn] * cosnvn_val;
    lmksc += lmnsc[idx_mn] * cosnv_val;
    lmksc_n += lmnsc[idx_mn] * sinnvn_val;
    lmkcs += lmncs[idx_mn] * sinnv_val;
    lmkcs_n += lmncs[idx_mn] * cosnvn_val;
  }

  m_partials[0 * task_count + task] = rmkcc;
  m_partials[1 * task_count + task] = rmkcc_n;
  m_partials[2 * task_count + task] = rmkss;
  m_partials[3 * task_count + task] = rmkss_n;
  m_partials[4 * task_count + task] = zmksc;
  m_partials[5 * task_count + task] = zmksc_n;
  m_partials[6 * task_count + task] = zmkcs;
  m_partials[7 * task_count + task] = zmkcs_n;
  m_partials[8 * task_count + task] = lmksc;
  m_partials[9 * task_count + task] = lmksc_n;
  m_partials[10 * task_count + task] = lmkcs;
  m_partials[11 * task_count + task] = lmkcs_n;
}

__global__ void ToroidalPartialsKernelDispatch(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ cosnvn, const double* __restrict__ sinnvn,
    double* __restrict__ m_partials, KernelParams params) {
  const int mode = precision_fp32 != 0;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    atomicAdd(&precision_counts[mode], 1ULL);
  if (mode)
    ToroidalPartialsKernelFp32(rmncc, rmnss, zmnsc, zmncs, lmnsc, lmncs, cosnv,
                               sinnv, cosnvn, sinnvn, m_partials, params);
  else
    ToroidalPartialsKernelFp64(rmncc, rmnss, zmnsc, zmncs, lmnsc, lmncs, cosnv,
                               sinnv, cosnvn, sinnvn, m_partials, params);
}

__device__ __forceinline__ void FourierToRealKernelSeparableFp64(
    const double* __restrict__ partials, const double* __restrict__ rmncc,
    const double* __restrict__ rmnss, const double* __restrict__ zmnsc,
    const double* __restrict__ zmncs, const double* __restrict__ lmnsc,
    const double* __restrict__ lmncs, const double* __restrict__ xmpq,
    const double* __restrict__ sqrt_sf, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ cosmum,
    const double* __restrict__ sinmum, const double* __restrict__ cosnv,
    const double* __restrict__ sinnv, const double* __restrict__ cosnvn,
    const double* __restrict__ sinnvn, double* __restrict__ r1_e,
    double* __restrict__ r1_o, double* __restrict__ ru_e,
    double* __restrict__ ru_o, double* __restrict__ rv_e,
    double* __restrict__ rv_o, double* __restrict__ z1_e,
    double* __restrict__ z1_o, double* __restrict__ zu_e,
    double* __restrict__ zu_o, double* __restrict__ zv_e,
    double* __restrict__ zv_o, double* __restrict__ lu_e,
    double* __restrict__ lu_o, double* __restrict__ lv_e,
    double* __restrict__ lv_o, double* __restrict__ r_con,
    double* __restrict__ z_con, KernelParams params) {
  const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int jF =
      point_index / (params.n_zeta * params.n_theta_reduced) + params.ns_min_f1;
  const int k = (point_index / params.n_theta_reduced) % params.n_zeta;
  const int l = point_index % params.n_theta_reduced;

  if (jF >= params.ns_max_f1 || k >= params.n_zeta ||
      l >= params.n_theta_reduced) {
    return;
  }

  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
  const int output_index =
      ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;
  double r1_e_accumulator = r1_e[output_index];
  double r1_o_accumulator = r1_o[output_index];
  double ru_e_accumulator = ru_e[output_index];
  double ru_o_accumulator = ru_o[output_index];
  double rv_e_accumulator = rv_e[output_index];
  double rv_o_accumulator = rv_o[output_index];
  double z1_e_accumulator = z1_e[output_index];
  double z1_o_accumulator = z1_o[output_index];
  double zu_e_accumulator = zu_e[output_index];
  double zu_o_accumulator = zu_o[output_index];
  double zv_e_accumulator = zv_e[output_index];
  double zv_o_accumulator = zv_o[output_index];
  double lu_e_accumulator = lu_e[output_index];
  double lu_o_accumulator = lu_o[output_index];
  double lv_e_accumulator = lv_e[output_index];
  double lv_o_accumulator = lv_o[output_index];
  const bool constraint_active =
      params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs;
  int constraint_index = 0;
  double r_con_accumulator = 0.0;
  double z_con_accumulator = 0.0;
  if (constraint_active) {
    constraint_index =
        ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;
    r_con_accumulator = r_con[constraint_index];
    z_con_accumulator = z_con[constraint_index];
  }
#endif
#if VMECPP_INVERSE_LOAD_HOIST
  double sqrt_sf_point = 0.0;
  if (constraint_active && params.mpol > 1)
    sqrt_sf_point = sqrt_sf[jF - params.ns_min_f1];
#endif
  // Loop over poloidal modes m.
  for (int m = 0; m < params.mpol; ++m) {
    const bool m_even = (m % 2 == 0);

    // Axis only gets contributions from m=0,1.
    int jMin = 1;
    if (m == 0 || m == 1) {
      jMin = 0;
    }
    if (jF < jMin) {
      continue;
    }

#if VMECPP_INVERSE_LOAD_HOIST
    double xmpq_mode = 0.0;
    if (constraint_active) xmpq_mode = xmpq[m];
#endif
    const int idx_ml = m * params.n_theta_reduced + l;
    const double cosmu_val = cosmu[idx_ml];
    const double sinmu_val = sinmu[idx_ml];
    const double cosmum_val = cosmum[idx_ml];
    const double sinmum_val = sinmum[idx_ml];

    const int task_count =
        (params.ns_max_f1 - params.ns_min_f1) * params.mpol * params.n_zeta;
    const int task =
        ((jF - params.ns_min_f1) * params.mpol + m) * params.n_zeta + k;
    const double rmkcc = partials[0 * task_count + task];
    const double rmkcc_n = partials[1 * task_count + task];
    const double rmkss = partials[2 * task_count + task];
    const double rmkss_n = partials[3 * task_count + task];
    const double zmksc = partials[4 * task_count + task];
    const double zmksc_n = partials[5 * task_count + task];
    const double zmkcs = partials[6 * task_count + task];
    const double zmkcs_n = partials[7 * task_count + task];
    const double lmksc = partials[8 * task_count + task];
    const double lmksc_n = partials[9 * task_count + task];
    const double lmkcs = partials[10 * task_count + task];
    const double lmkcs_n = partials[11 * task_count + task];
    // Compute output index.
    const int idx_kl =
        ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
    // Preserve each original update; uniform parity chooses scalar registers.
    if (m_even) {
      r1_e_accumulator += rmkcc * cosmu_val + rmkss * sinmu_val;
      ru_e_accumulator += rmkcc * sinmum_val + rmkss * cosmum_val;
      rv_e_accumulator += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
      z1_e_accumulator += zmksc * sinmu_val + zmkcs * cosmu_val;
      zu_e_accumulator += zmksc * cosmum_val + zmkcs * sinmum_val;
      zv_e_accumulator += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
      lu_e_accumulator += lmksc * cosmum_val + lmkcs * sinmum_val;
      lv_e_accumulator -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;
    } else {
      r1_o_accumulator += rmkcc * cosmu_val + rmkss * sinmu_val;
      ru_o_accumulator += rmkcc * sinmum_val + rmkss * cosmum_val;
      rv_o_accumulator += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
      z1_o_accumulator += zmksc * sinmu_val + zmkcs * cosmu_val;
      zu_o_accumulator += zmksc * cosmum_val + zmkcs * sinmum_val;
      zv_o_accumulator += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
      lu_o_accumulator += lmksc * cosmum_val + lmkcs * sinmum_val;
      lv_o_accumulator -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;
    }
#else
    // Select even/odd arrays based on m parity.
    double* r1 = m_even ? r1_e : r1_o;
    double* ru = m_even ? ru_e : ru_o;
    double* rv = m_even ? rv_e : rv_o;
    double* z1 = m_even ? z1_e : z1_o;
    double* zu = m_even ? zu_e : zu_o;
    double* zv = m_even ? zv_e : zv_o;
    double* lu = m_even ? lu_e : lu_o;
    double* lv = m_even ? lv_e : lv_o;

    // Use atomic adds since multiple m values write to the same output
    // location.
    r1[idx_kl] += rmkcc * cosmu_val + rmkss * sinmu_val;
    ru[idx_kl] += rmkcc * sinmum_val + rmkss * cosmum_val;
    rv[idx_kl] += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
    z1[idx_kl] += zmksc * sinmu_val + zmkcs * cosmu_val;
    zu[idx_kl] += zmksc * cosmum_val + zmkcs * sinmum_val;
    zv[idx_kl] += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
    lu[idx_kl] += lmksc * cosmum_val + lmkcs * sinmum_val;
    // lv gets a negative sign.
    lv[idx_kl] -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;

#endif
    // Constraint force computation.
    if (params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs) {
#if VMECPP_INVERSE_LOAD_HOIST
      const double sqrt_s = m_even ? 1.0 : sqrt_sf_point;
      const double con_factor = xmpq_mode * sqrt_s;
#else
      const double sqrt_s = m_even ? 1.0 : sqrt_sf[jF - params.ns_min_f1];
      const double con_factor = xmpq[m] * sqrt_s;
#endif

      const int idx_con =
          ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
      r_con_accumulator += (rmkcc * cosmu_val + rmkss * sinmu_val) * con_factor;
      z_con_accumulator += (zmksc * sinmu_val + zmkcs * cosmu_val) * con_factor;
#else
      r_con[idx_con] += (rmkcc * cosmu_val + rmkss * sinmu_val) * con_factor;
      z_con[idx_con] += (zmksc * sinmu_val + zmkcs * cosmu_val) * con_factor;
#endif
    }
  }
#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
  r1_e[output_index] = r1_e_accumulator;
  r1_o[output_index] = r1_o_accumulator;
  ru_e[output_index] = ru_e_accumulator;
  ru_o[output_index] = ru_o_accumulator;
  rv_e[output_index] = rv_e_accumulator;
  rv_o[output_index] = rv_o_accumulator;
  z1_e[output_index] = z1_e_accumulator;
  z1_o[output_index] = z1_o_accumulator;
  zu_e[output_index] = zu_e_accumulator;
  zu_o[output_index] = zu_o_accumulator;
  zv_e[output_index] = zv_e_accumulator;
  zv_o[output_index] = zv_o_accumulator;
  lu_e[output_index] = lu_e_accumulator;
  lu_o[output_index] = lu_o_accumulator;
  lv_e[output_index] = lv_e_accumulator;
  lv_o[output_index] = lv_o_accumulator;
  if (constraint_active) {
    r_con[constraint_index] = r_con_accumulator;
    z_con[constraint_index] = z_con_accumulator;
  }
#endif
}

__device__ __forceinline__ void FourierToRealKernelSeparableFp32(
    const double* __restrict__ partials_fp64,
    const double* __restrict__ rmncc_fp64,
    const double* __restrict__ rmnss_fp64,
    const double* __restrict__ zmnsc_fp64,
    const double* __restrict__ zmncs_fp64,
    const double* __restrict__ lmnsc_fp64,
    const double* __restrict__ lmncs_fp64, const double* __restrict__ xmpq_fp64,
    const double* __restrict__ sqrt_sf_fp64,
    const double* __restrict__ cosmu_fp64,
    const double* __restrict__ sinmu_fp64,
    const double* __restrict__ cosmum_fp64,
    const double* __restrict__ sinmum_fp64,
    const double* __restrict__ cosnv_fp64,
    const double* __restrict__ sinnv_fp64,
    const double* __restrict__ cosnvn_fp64,
    const double* __restrict__ sinnvn_fp64, double* __restrict__ r1_e,
    double* __restrict__ r1_o, double* __restrict__ ru_e,
    double* __restrict__ ru_o, double* __restrict__ rv_e,
    double* __restrict__ rv_o, double* __restrict__ z1_e,
    double* __restrict__ z1_o, double* __restrict__ zu_e,
    double* __restrict__ zu_o, double* __restrict__ zv_e,
    double* __restrict__ zv_o, double* __restrict__ lu_e,
    double* __restrict__ lu_o, double* __restrict__ lv_e,
    double* __restrict__ lv_o, double* __restrict__ r_con,
    double* __restrict__ z_con, KernelParams params) {
  const FloatReadView partials{partials_fp64};
  const FloatReadView rmncc{rmncc_fp64};
  const FloatReadView rmnss{rmnss_fp64};
  const FloatReadView zmnsc{zmnsc_fp64};
  const FloatReadView zmncs{zmncs_fp64};
  const FloatReadView lmnsc{lmnsc_fp64};
  const FloatReadView lmncs{lmncs_fp64};
  const FloatReadView xmpq{xmpq_fp64};
  const FloatReadView sqrt_sf{sqrt_sf_fp64};
  const FloatReadView cosmu{cosmu_fp64};
  const FloatReadView sinmu{sinmu_fp64};
  const FloatReadView cosmum{cosmum_fp64};
  const FloatReadView sinmum{sinmum_fp64};
  const FloatReadView cosnv{cosnv_fp64};
  const FloatReadView sinnv{sinnv_fp64};
  const FloatReadView cosnvn{cosnvn_fp64};
  const FloatReadView sinnvn{sinnvn_fp64};
  const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int jF =
      point_index / (params.n_zeta * params.n_theta_reduced) + params.ns_min_f1;
  const int k = (point_index / params.n_theta_reduced) % params.n_zeta;
  const int l = point_index % params.n_theta_reduced;

  if (jF >= params.ns_max_f1 || k >= params.n_zeta ||
      l >= params.n_theta_reduced) {
    return;
  }

  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
  const int output_index =
      ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;
  float r1_e_accumulator = r1_e[output_index];
  float r1_o_accumulator = r1_o[output_index];
  float ru_e_accumulator = ru_e[output_index];
  float ru_o_accumulator = ru_o[output_index];
  float rv_e_accumulator = rv_e[output_index];
  float rv_o_accumulator = rv_o[output_index];
  float z1_e_accumulator = z1_e[output_index];
  float z1_o_accumulator = z1_o[output_index];
  float zu_e_accumulator = zu_e[output_index];
  float zu_o_accumulator = zu_o[output_index];
  float zv_e_accumulator = zv_e[output_index];
  float zv_o_accumulator = zv_o[output_index];
  float lu_e_accumulator = lu_e[output_index];
  float lu_o_accumulator = lu_o[output_index];
  float lv_e_accumulator = lv_e[output_index];
  float lv_o_accumulator = lv_o[output_index];
  const bool constraint_active =
      params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs;
  int constraint_index = 0;
  float r_con_accumulator = 0.0f;
  float z_con_accumulator = 0.0f;
  if (constraint_active) {
    constraint_index =
        ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;
    r_con_accumulator = r_con[constraint_index];
    z_con_accumulator = z_con[constraint_index];
  }
#endif
#if VMECPP_INVERSE_LOAD_HOIST
  float sqrt_sf_point = 0.0f;
  if (constraint_active && params.mpol > 1)
    sqrt_sf_point = sqrt_sf[jF - params.ns_min_f1];
#endif
  // Loop over poloidal modes m.
  for (int m = 0; m < params.mpol; ++m) {
    const bool m_even = (m % 2 == 0);

    // Axis only gets contributions from m=0,1.
    int jMin = 1;
    if (m == 0 || m == 1) {
      jMin = 0;
    }
    if (jF < jMin) {
      continue;
    }

#if VMECPP_INVERSE_LOAD_HOIST
    float xmpq_mode = 0.0f;
    if (constraint_active) xmpq_mode = xmpq[m];
#endif
    const int idx_ml = m * params.n_theta_reduced + l;
    const float cosmu_val = cosmu[idx_ml];
    const float sinmu_val = sinmu[idx_ml];
    const float cosmum_val = cosmum[idx_ml];
    const float sinmum_val = sinmum[idx_ml];

    const int task_count =
        (params.ns_max_f1 - params.ns_min_f1) * params.mpol * params.n_zeta;
    const int task =
        ((jF - params.ns_min_f1) * params.mpol + m) * params.n_zeta + k;
    const float rmkcc = partials[0 * task_count + task];
    const float rmkcc_n = partials[1 * task_count + task];
    const float rmkss = partials[2 * task_count + task];
    const float rmkss_n = partials[3 * task_count + task];
    const float zmksc = partials[4 * task_count + task];
    const float zmksc_n = partials[5 * task_count + task];
    const float zmkcs = partials[6 * task_count + task];
    const float zmkcs_n = partials[7 * task_count + task];
    const float lmksc = partials[8 * task_count + task];
    const float lmksc_n = partials[9 * task_count + task];
    const float lmkcs = partials[10 * task_count + task];
    const float lmkcs_n = partials[11 * task_count + task];
    // Compute output index.
    const int idx_kl =
        ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
    // Preserve each original update; uniform parity chooses scalar registers.
    if (m_even) {
      r1_e_accumulator += rmkcc * cosmu_val + rmkss * sinmu_val;
      ru_e_accumulator += rmkcc * sinmum_val + rmkss * cosmum_val;
      rv_e_accumulator += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
      z1_e_accumulator += zmksc * sinmu_val + zmkcs * cosmu_val;
      zu_e_accumulator += zmksc * cosmum_val + zmkcs * sinmum_val;
      zv_e_accumulator += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
      lu_e_accumulator += lmksc * cosmum_val + lmkcs * sinmum_val;
      lv_e_accumulator -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;
    } else {
      r1_o_accumulator += rmkcc * cosmu_val + rmkss * sinmu_val;
      ru_o_accumulator += rmkcc * sinmum_val + rmkss * cosmum_val;
      rv_o_accumulator += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
      z1_o_accumulator += zmksc * sinmu_val + zmkcs * cosmu_val;
      zu_o_accumulator += zmksc * cosmum_val + zmkcs * sinmum_val;
      zv_o_accumulator += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
      lu_o_accumulator += lmksc * cosmum_val + lmkcs * sinmum_val;
      lv_o_accumulator -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;
    }
#else
    // Select even/odd arrays based on m parity.
    float* r1 = m_even ? r1_e : r1_o;
    float* ru = m_even ? ru_e : ru_o;
    float* rv = m_even ? rv_e : rv_o;
    float* z1 = m_even ? z1_e : z1_o;
    float* zu = m_even ? zu_e : zu_o;
    float* zv = m_even ? zv_e : zv_o;
    float* lu = m_even ? lu_e : lu_o;
    float* lv = m_even ? lv_e : lv_o;

    // Use atomic adds since multiple m values write to the same output
    // location.
    r1[idx_kl] += rmkcc * cosmu_val + rmkss * sinmu_val;
    ru[idx_kl] += rmkcc * sinmum_val + rmkss * cosmum_val;
    rv[idx_kl] += rmkcc_n * cosmu_val + rmkss_n * sinmu_val;
    z1[idx_kl] += zmksc * sinmu_val + zmkcs * cosmu_val;
    zu[idx_kl] += zmksc * cosmum_val + zmkcs * sinmum_val;
    zv[idx_kl] += zmksc_n * sinmu_val + zmkcs_n * cosmu_val;
    lu[idx_kl] += lmksc * cosmum_val + lmkcs * sinmum_val;
    // lv gets a negative sign.
    lv[idx_kl] -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;

#endif
    // Constraint force computation.
    if (params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs) {
#if VMECPP_INVERSE_LOAD_HOIST
      const float sqrt_s = m_even ? 1.0f : sqrt_sf_point;
      const float con_factor = xmpq_mode * sqrt_s;
#else
      const float sqrt_s = m_even ? 1.0f : sqrt_sf[jF - params.ns_min_f1];
      const float con_factor = xmpq[m] * sqrt_s;
#endif

      const int idx_con =
          ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;

#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
      r_con_accumulator += (rmkcc * cosmu_val + rmkss * sinmu_val) * con_factor;
      z_con_accumulator += (zmksc * sinmu_val + zmkcs * cosmu_val) * con_factor;
#else
      r_con[idx_con] += (rmkcc * cosmu_val + rmkss * sinmu_val) * con_factor;
      z_con[idx_con] += (zmksc * sinmu_val + zmkcs * cosmu_val) * con_factor;
#endif
    }
  }
#if VMECPP_INVERSE_REGISTER_ACCUMULATORS
  r1_e[output_index] = r1_e_accumulator;
  r1_o[output_index] = r1_o_accumulator;
  ru_e[output_index] = ru_e_accumulator;
  ru_o[output_index] = ru_o_accumulator;
  rv_e[output_index] = rv_e_accumulator;
  rv_o[output_index] = rv_o_accumulator;
  z1_e[output_index] = z1_e_accumulator;
  z1_o[output_index] = z1_o_accumulator;
  zu_e[output_index] = zu_e_accumulator;
  zu_o[output_index] = zu_o_accumulator;
  zv_e[output_index] = zv_e_accumulator;
  zv_o[output_index] = zv_o_accumulator;
  lu_e[output_index] = lu_e_accumulator;
  lu_o[output_index] = lu_o_accumulator;
  lv_e[output_index] = lv_e_accumulator;
  lv_o[output_index] = lv_o_accumulator;
  if (constraint_active) {
    r_con[constraint_index] = r_con_accumulator;
    z_con[constraint_index] = z_con_accumulator;
  }
#endif
}

__global__ void FourierToRealKernelSeparableDispatch(
    const double* __restrict__ partials, const double* __restrict__ rmncc,
    const double* __restrict__ rmnss, const double* __restrict__ zmnsc,
    const double* __restrict__ zmncs, const double* __restrict__ lmnsc,
    const double* __restrict__ lmncs, const double* __restrict__ xmpq,
    const double* __restrict__ sqrt_sf, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ cosmum,
    const double* __restrict__ sinmum, const double* __restrict__ cosnv,
    const double* __restrict__ sinnv, const double* __restrict__ cosnvn,
    const double* __restrict__ sinnvn, double* __restrict__ r1_e,
    double* __restrict__ r1_o, double* __restrict__ ru_e,
    double* __restrict__ ru_o, double* __restrict__ rv_e,
    double* __restrict__ rv_o, double* __restrict__ z1_e,
    double* __restrict__ z1_o, double* __restrict__ zu_e,
    double* __restrict__ zu_o, double* __restrict__ zv_e,
    double* __restrict__ zv_o, double* __restrict__ lu_e,
    double* __restrict__ lu_o, double* __restrict__ lv_e,
    double* __restrict__ lv_o, double* __restrict__ r_con,
    double* __restrict__ z_con, KernelParams params) {
  const int mode = precision_fp32 != 0;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    atomicAdd(&precision_counts[mode], 1ULL);
  if (mode)
    FourierToRealKernelSeparableFp32(
        partials, rmncc, rmnss, zmnsc, zmncs, lmnsc, lmncs, xmpq, sqrt_sf,
        cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn, r1_e, r1_o,
        ru_e, ru_o, rv_e, rv_o, z1_e, z1_o, zu_e, zu_o, zv_e, zv_o, lu_e, lu_o,
        lv_e, lv_o, r_con, z_con, params);
  else
    FourierToRealKernelSeparableFp64(
        partials, rmncc, rmnss, zmnsc, zmncs, lmnsc, lmncs, xmpq, sqrt_sf,
        cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn, r1_e, r1_o,
        ru_e, ru_o, rv_e, rv_o, z1_e, z1_o, zu_e, zu_o, zv_e, zv_o, lu_e, lu_o,
        lv_e, lv_o, r_con, z_con, params);
}

}  // namespace vmecpp_mixed_inverse
