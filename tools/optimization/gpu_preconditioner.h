// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_PRECONDITIONER_H_
#define VMECPP_EXPERIMENT_GPU_PRECONDITIONER_H_
#include <array>
#include <cstddef>

struct GpuPreconditionerInput {
  // r12, totalPressure, tau, bsupv, gsqrt, guu, guv, gvv;
  // zs, zu12, zu_e, zu_o, z1_o; rs, ru12, ru_e, ru_o, r1_o;
  // sqrtSH, sqrtSF, sm, sp, wInt, ruFull, zuFull, fixed lambda powers.
  std::array<const double*, 26> data_;
  std::array<std::size_t, 26> sizes_;
  // ns, mpol, ntor, nsMaxF, nsMaxFIncludingLcfs, nThetaEff, nZeta, nfp.
  std::array<int, 8> dimensions_;
  // deltaS, lamscale, dampingFactor, tcon0.
  std::array<double, 4> scalars_;
};
struct GpuPreconditionerOutput {
  // lambdaPreconditioner; bLambda,dLambda,cLambda; arm,ard,brm,brd;
  // azm,azd,bzm,bzd; cxd,tcon; ar,az,dr,dz,br,bz.
  std::array<double*, 20> data_;
  std::array<std::size_t, 20> sizes_;
};

struct GpuPreconditionerDeviceState {
  std::array<const double*, 26> input_;
  std::array<double*, 20> output_;
  std::array<int, 8> dimensions_;
  std::array<double, 4> scalars_;
  double* terms_;  // 21 * (ns-1) * nZeta * nThetaEff.
  double* sums_;   // 21 * (ns-1).
  int* invalid_norm_;
  void* stream_;
};
extern "C" void LaunchGpuPreconditioner(GpuPreconditionerDeviceState& m_state);
extern "C" void RunGpuPreconditioner(const GpuPreconditionerInput& input,
                                     GpuPreconditionerOutput& m_output);

#endif  // VMECPP_EXPERIMENT_GPU_PRECONDITIONER_H_
