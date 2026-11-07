// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_SOLVER_H_
#define VMECPP_EXPERIMENT_GPU_SOLVER_H_

#include <array>

#include "gpu_fourier.h"
#include "gpu_iteration.h"
#include "gpu_preconditioner.h"
#include "gpu_resident.h"

// Initialization uses the original CPU setup iteration's profiles and caches.
// All recurring bulk state is GPU-owned. Host control/backup mirroring is an
// explicit intermediate integration step, not the final full-GPU controller.
struct GpuSolverSetup {
  GpuFourierInput basis_;
  GpuResidentInput fields_;
  GpuResidentForceConstants forces_;
  GpuResidentConstraintConstants constraints_;
  GpuPreconditionerInput preconditioner_;
  GpuPreconditionerOutput cached_preconditioner_;
  GpuStepInput state_;
  // Optional original CPU restart backup; null channels use current geometry.
  GpuPreparationInput backup_{};
  const double* cached_dvds_ = nullptr;
  const double* scalxc_;
  std::size_t scalxc_size_;
  std::array<double, 7> cached_diagnostics_;
};

struct GpuSolverControl {
  bool refresh_preconditioner_;
  bool include_edge_rz_;
  bool zero_z_m1_;
};
struct GpuSolverScalars {
  std::array<double, 3> invariant_residuals_;
  std::array<double, 3> preconditioned_residuals_;
  // volume, thermal, magnetic, MHD, fNormRZ, fNormL, fNorm1.
  std::array<double, 7> diagnostics_;
  std::array<double, 2> jacobian_extrema_;
  int invalid_norm_;
  bool bad_jacobian_;
  int tail_controller_status_ = 0;
};

extern "C" void* CreateGpuSolver(const GpuSolverSetup& setup);
extern "C" void DestroyGpuSolver(void* m_context);
extern "C" void UploadGpuSolverState(void* m_context,
                                     const GpuStepInput& state);
extern "C" void RunGpuSolver(void* m_context, const GpuSolverControl& control,
                             GpuSolverScalars& m_output);
extern "C" void AdvanceGpuSolver(void* m_context,
                                 const std::array<double, 3>& scalars);
// Optional stage1 controller: CPU references protect every live GPU step.
struct GpuSolverControllerReference {
  std::array<double, 3> scalars_;
  std::array<double, 10> history_;
  double fsq_;
  int iter1_;
  int iter2_;
  bool pinned_log_allowed_ = false;
  bool reference_enabled_ = true;
  bool cpu_domain_escape_ = false;
};
// Optional fused-tail protocol. Non-step outcomes leave controller/x/v
// untouched.
enum class GpuSolverTailStatus {
  UNARMED = 0,
  STEPPED = 1,
  CONVERGED = 2,
  GUARD_DECLINED = 3,
  DOMAIN_DECLINED = 4,
  INVALID_NORM = 5
};
struct GpuSolverTailControllerControl {
  double time_step_ = 0.0;
  double force_tolerance_ = 0.0;
  int iter1_ = 0;
  int iter2_ = 0;
  bool pinned_log_allowed_ = false;
};
extern "C" void ArmGpuSolverTailController(
    void* m_context, const GpuSolverTailControllerControl& control);
extern "C" void CancelGpuSolverTailController(void* m_context);
// One counter for each GpuSolverTailStatus, per context.
extern "C" void ReadGpuSolverTailControllerStats(
    void* m_context, std::array<unsigned long long, 6>& m_stats);

struct GpuSolverControllerState {
  std::array<double, 10> history_;
  double previous_fsq_;
  // Last selected velocity_scale, conjugation_parameter, time_step.
  std::array<double, 3> scalars_;
};
extern "C" void SeedGpuSolverControllerState(
    void* m_context, const GpuSolverControllerState& state,
    const std::array<double, 3>& preconditioned_residuals);
// Own-stream readback only; does not advance history, geometry or counters.
extern "C" void DownloadGpuSolverControllerState(
    void* m_context, GpuSolverControllerState& m_output);
extern "C" void SeedGpuSolverController(
    void* m_context, const std::array<double, 10>& history, double previous_fsq,
    const std::array<double, 3>& preconditioned_residuals);
extern "C" void AdvanceGpuSolverController(
    void* m_context, const GpuSolverControllerReference& reference);
// steps, accepted, coefficient failures, fsq failures, history differences,
// history-only differences, explicit seeds. Counters are per-context.
extern "C" void ReadGpuSolverControllerStats(
    void* m_context, std::array<unsigned long long, 7>& m_stats);
// Exact-log attempts, guard declines, domain declines, accepted exact-log
// calls.
extern "C" void ReadGpuSolverPinnedLogStats(
    void* m_context, std::array<unsigned long long, 4>& m_stats);
extern "C" void DownloadGpuSolverState(void* m_context,
                                       GpuStepOutput& m_output);
extern "C" void DownloadGpuSolverCache(void* m_context,
                                       GpuPreconditionerOutput& m_output);

enum class GpuSolverGeometry {
  CURRENT,
  BACKUP,
  LAST_UPDATE,
  LAST_VALID_UPDATE
};
extern "C" void SaveGpuSolverBackup(void* m_context);
extern "C" void RestoreGpuSolverBackup(void* m_context);
extern "C" void DownloadGpuSolverGeometry(void* m_context,
                                          GpuSolverGeometry which,
                                          GpuPreparationOutput& m_output);
extern "C" void DownloadGpuSolverForces(void* m_context,
                                        GpuPreparationOutput& m_output);
extern "C" void DownloadGpuSolverPrintout(void* m_context, double* m_dvds,
                                          std::size_t count,
                                          std::array<double, 2>& m_axis);

#endif  // VMECPP_EXPERIMENT_GPU_SOLVER_H_
