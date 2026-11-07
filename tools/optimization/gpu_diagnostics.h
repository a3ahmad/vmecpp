// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_GPU_DIAGNOSTICS_H_
#define VMECPP_EXPERIMENT_GPU_DIAGNOSTICS_H_
#include <array>
#include <cstddef>

struct GpuDiagnosticsInput {
  // gsqrt, bsupu, bsupv, bsubu, bsubv, wInt, massH, dVdsH, guu, r12,
  // decomposed rcc, rss, zsc, zcs.
  std::array<const double*, 14> data_;
  std::array<std::size_t, 14> sizes_;
  // ns, mpol, ntor, nsMaxF, nsMaxFIncludingLcfs, nThetaEff, nZeta.
  std::array<int, 7> dimensions_;
  // deltaS, gamma (must be zero for this specialization), lamscale.
  std::array<double, 3> scalars_;
};
struct GpuDiagnosticsOutput {
  // plasmaVolume, thermalEnergy, magneticEnergy, mhdEnergy, fNormRZ/L/1.
  std::array<double*, 1> data_;
  std::array<std::size_t, 1> sizes_;
};
struct GpuDiagnosticsDeviceState {
  std::array<const double*, 14> data_;
  std::array<int, 7> dimensions_;
  std::array<double, 3> scalars_;
  double* sums_;    // Six doubles of device scratch.
  double* output_;  // Seven doubles, retained on device.
  void* stream_;
};
extern "C" void LaunchGpuDiagnostics(GpuDiagnosticsDeviceState& m_state);
// Update energies every iteration while retaining cached volume/norms between
// the original preconditioner refresh iterations.
extern "C" void LaunchGpuEnergies(GpuDiagnosticsDeviceState& m_state);
extern "C" void RunGpuDiagnostics(const GpuDiagnosticsInput& input,
                                  GpuDiagnosticsOutput& m_output);
#endif  // VMECPP_EXPERIMENT_GPU_DIAGNOSTICS_H_
