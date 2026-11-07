// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_ITERATION_H_
#define VMECPP_EXPERIMENT_GPU_ITERATION_H_

#include <array>
#include <cstddef>

// Symmetric, three-dimensional, fixed-boundary, single-partition scope.
// Force channels: frcc, frss, fzsc, fzcs, flsc, flcs.
struct GpuIterationDeviceState {
  std::array<double*, 6> forces_;
  // ard, brd, azd, bzd; ar, az, dr, dz, br, bz; lambdaPreconditioner.
  std::array<const double*, 11> constants_;
  std::array<double*, 2> factored_upper_;
  std::array<double*, 2> denominators_;
  double* residuals_;
  // ns, mpol, ntor, nsMaxF, nsMaxFIncludingLcfs.
  std::array<int, 5> dimensions_;
  void* stream_;
};

extern "C" void LaunchGpuIterationM1(GpuIterationDeviceState& m_state);
// Cache only the R/Z matrix factorization. Run once after initializing the
// matrix/factor buffers, and again after every write to constants_[4..9].
// Dimensions and those six matrices must remain unchanged until the next
// factorization. All calls must use the same ordered stream and context-owned
// buffers. Geometry/velocity backup restores do not invalidate these factors.
extern "C" void LaunchGpuIterationFactor(GpuIterationDeviceState& m_state);
// Apply current forces using an already valid factorization. M1 scaling still
// runs after the invariant residual and before this call on every iteration.
extern "C" void LaunchGpuIterationRZFactored(GpuIterationDeviceState& m_state);
// Compatibility entry point: always factor and solve, as before.
extern "C" void LaunchGpuIterationRZ(GpuIterationDeviceState& m_state);
extern "C" void LaunchGpuIterationLambda(GpuIterationDeviceState& m_state);
extern "C" void LaunchGpuIterationResiduals(GpuIterationDeviceState& m_state,
                                            bool include_edge_rz);
extern "C" void LaunchGpuIterationResidualsConditional(
    GpuIterationDeviceState& m_state, bool include_edge_rz,
    const int* invalid_components);
extern "C" void LaunchGpuIterationResidualsOptimistic(
    GpuIterationDeviceState& m_state, bool include_edge_rz, int rounds,
    bool use_graph);
extern "C" void ReadGpuResidualRelaxationStats(
    std::array<unsigned long long, 4>& m_statistics);

struct GpuIterationInput {
  std::array<const double*, 17> data_;
  std::array<std::size_t, 17> sizes_;
  std::array<int, 5> dimensions_;
};
struct GpuIterationOutput {
  // Six forces after each of M1, RZ and lambda; residual triples before/after.
  std::array<double*, 20> data_;
  std::array<std::size_t, 20> sizes_;
};
// Capture/replay adapter; production kernels above use device-owned pointers.
extern "C" void RunGpuIteration(const GpuIterationInput& input,
                                GpuIterationOutput& m_output);

struct GpuStepDeviceState {
  std::array<double*, 6> geometry_;
  std::array<double*, 6> velocity_;
  std::array<const double*, 6> forces_;
  int coefficient_count_;
  // velocity_scale, conjugation_parameter, time_step.
  std::array<double, 3> scalars_;
  void* stream_;
};
extern "C" void LaunchGpuIterationStep(GpuStepDeviceState& m_state);
struct GpuStepInput {
  std::array<const double*, 18> data_;
  std::array<std::size_t, 18> sizes_;
  std::array<int, 5> dimensions_;
  std::array<double, 3> scalars_;
};
struct GpuStepOutput {
  std::array<double*, 12> data_;
  std::array<std::size_t, 12> sizes_;
};
extern "C" void RunGpuIterationStep(const GpuStepInput& input,
                                    GpuStepOutput& m_output);

struct GpuPreparationDeviceState {
  std::array<double*, 6> coefficients_;
  const double* scalxc_;
  std::array<int, 5> dimensions_;
  double m1_scale_;
  bool zero_z_m1_;
  void* stream_;
};
struct GpuPreparationInput {
  // Six Fourier coefficient channels, then scalxc.
  std::array<const double*, 7> data_;
  std::array<std::size_t, 7> sizes_;
  std::array<int, 5> dimensions_;
  // m=1 constraint scaling, whether zeroZForceForM1 ran.
  std::array<double, 2> scalars_;
};
struct GpuPreparationOutput {
  std::array<double*, 6> data_;
  std::array<std::size_t, 6> sizes_;
};
extern "C" void LaunchGpuIterationPreparation(
    GpuPreparationDeviceState& m_state);
extern "C" void RunGpuIterationPreparation(const GpuPreparationInput& input,
                                           GpuPreparationOutput& m_output);
extern "C" void LaunchGpuGeometryAxis(GpuPreparationDeviceState& m_state);
extern "C" void RunGpuGeometryPreparation(const GpuPreparationInput& input,
                                          GpuPreparationOutput& m_output);
#endif  // VMECPP_EXPERIMENT_GPU_ITERATION_H_
