// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_RESIDENT_H_
#define VMECPP_EXPERIMENT_GPU_RESIDENT_H_

#include <array>
#include <cstddef>

// Full fields use the inverse transform's existing 18-array order. Profiles:
// sqrtSF, sqrtSH, wInt, phipF, phipH, currH, chipH, iotaH, chipF, iotaF.
// Scalars: deltaS, dSHalfDsInterp, lamscale, signOfJacobian.
struct GpuResidentInput {
  std::array<const double*, 28> data_;
  std::array<std::size_t, 28> sizes_;
  std::array<int, 13> dimensions_;
  std::array<double, 4> scalars_;
};

// Full fields (18); r12, ru12, zu12, rs, zs, tau, gsqrt, guu, guv, gvv,
// bsupu, bsupv, bsubu, bsubv (14); dVdsH, chipH, iotaH, chipF, iotaF (5);
// minTau/maxTau (one two-element array).
struct GpuResidentOutput {
  std::array<double*, 38> data_;
  std::array<std::size_t, 38> sizes_;
};

// Every pointer below addresses device memory. Adjacent field stages share
// this state and stream; no host materialization occurs between their kernels.
struct GpuResidentDeviceState {
  std::array<double*, 18> full_;
  std::array<double*, 14> half_;
  std::array<double*, 11> profiles_;
  double* jacobian_extrema_;
  double* jacobian_partials_;
  double* profile_terms_;
  std::array<int, 13> dimensions_;
  std::array<double, 4> scalars_;
  void* stream_;
};

struct GpuResidentForceConstants {
  std::array<const double*, 2> data_;  // massH, radialBlending
  std::array<std::size_t, 2> sizes_;
  double adiabatic_index_;
};

struct GpuResidentForceOutput {
  std::array<double*, 17>
      data_;  // 12 MHD forces, four lambda forces, totalPressure
  std::array<std::size_t, 17> sizes_;
};

struct GpuResidentForceDeviceState {
  GpuResidentDeviceState geometry_;
  std::array<const double*, 2> constants_;
  double adiabatic_index_;
  double* total_pressure_;
  std::array<double*, 4> lambda_;  // Contiguous full-grid channels.
  std::array<double*, 12> mhd_;    // Contiguous active force-grid channels.
};

struct GpuResidentConstraintConstants {
  // rCon0, zCon0, faccon, tcon, xmpq, eight forward basis arrays, cosmu, sinmu.
  std::array<const double*, 15> data_;
  std::array<std::size_t, 15> sizes_;
};

struct GpuResidentConstraintDeviceState {
  GpuResidentForceDeviceState force_;
  std::array<const double*, 15> constants_;
  // gConEff, ruFull, zuFull, gCon, including the allocated LCFS row.
  std::array<double*, 4> geometry_;
  std::array<double*, 2> poloidal_;
  std::array<double*, 2> spectral_;
  std::array<double*, 2> toroidal_;
  // Contiguous frcon_e/o, fzcon_e/o channels on the active force grid.
  std::array<double*, 4> forces_;
};

struct GpuResidentForceEvalOutput {
  // Six Fourier forces, then gConEff, ruFull, zuFull, gCon for validation.
  std::array<double*, 10> data_;
  std::array<std::size_t, 10> sizes_;
};

extern "C" void LaunchGpuResidentFields(GpuResidentDeviceState& m_state);
extern "C" void LaunchGpuResidentForces(GpuResidentForceDeviceState& m_state);
extern "C" void LaunchGpuResidentConstraints(
    GpuResidentConstraintDeviceState& m_state);
extern "C" void LaunchGpuResidentConstraintGeometry(
    GpuResidentConstraintDeviceState& m_state);
extern "C" void LaunchGpuResidentConstraintFilter(
    GpuResidentConstraintDeviceState& m_state);
extern "C" void LaunchGpuResidentConstraintAssembly(
    GpuResidentConstraintDeviceState& m_state);

// Reference/replay adapter only: uploads captured CPU inputs and materializes
// all outputs for byte comparison. This is not the production residency API.
extern "C" void RunGpuResidentFields(const GpuResidentInput& input,
                                     GpuResidentOutput& m_output);

struct GpuFourierInput;
// Chain validation adapter: ignores all 18 host geometry pointers in reference,
// obtaining geometry directly from the resident inverse transform instead.
extern "C" void RunGpuResidentFourierFields(const GpuFourierInput& fourier,
                                            const GpuResidentInput& reference,
                                            GpuResidentOutput& m_output);

extern "C" void RunGpuResidentMhdChain(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& constants,
    GpuResidentForceOutput& m_output);

extern "C" void RunGpuResidentForceEval(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output);

// Experimental integration seam: the callback receives GPU-owned Fourier
// forces and enqueues the next stages on the same stream before any download.
using GpuForceEvaluationTail = void (*)(const std::array<double*, 6>&,
                                        const std::array<int, 13>&, void*,
                                        void*);
extern "C" void RunGpuResidentForceEvalWithTail(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output, GpuForceEvaluationTail tail,
    void* m_context);

using GpuResidentPreForceStage = void (*)(GpuResidentConstraintDeviceState&,
                                          void*);
extern "C" void RunGpuResidentForceEvalWithStages(
    const GpuFourierInput& fourier, const GpuResidentInput& reference,
    const GpuResidentForceConstants& forces,
    const GpuResidentConstraintConstants& constraints,
    GpuResidentForceEvalOutput& m_output, GpuResidentPreForceStage pre_stage,
    GpuForceEvaluationTail tail, void* m_context);

#endif  // VMECPP_EXPERIMENT_GPU_RESIDENT_H_
