// SPDX-License-Identifier: MIT
// Persistent solver replay against one matched original CPU iteration.
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gpu_diagnostics.h"
#include "gpu_pipeline.h"
#include "gpu_solver.h"
#include "resident_replay_capture.h"

namespace {
struct ConstraintInput {
  std::array<const double*, 2> data_;
  std::array<std::size_t, 2> sizes_;
  std::array<int, 13> dimensions_;
};
struct ConstraintOutput {
  std::array<double*, 4> data_;
  std::array<std::size_t, 4> sizes_;
};
template <class Capture>
const double* Expected(const Capture& capture, int index) {
  return capture.expected_.data() +
         std::accumulate(capture.output_.sizes_.begin(),
                         capture.output_.sizes_.begin() + index,
                         std::size_t{0});
}
void RequireEqual(const double* expected, const double* actual,
                  std::size_t count, const char* label) {
  std::size_t differences = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (std::memcmp(expected + i, actual + i, sizeof(double)) != 0) {
      if (differences < 8)
        std::fprintf(stderr, "%s[%zu] expected %.17g actual %.17g\n", label, i,
                     expected[i], actual[i]);
      ++differences;
    }
  }
  if (differences != 0) throw std::runtime_error(label);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Usage: solver-replay optimization-root");
    const std::filesystem::path root(argv[1]);
    auto inverse = vmecpp_experiment::Read<GpuFourierInput, GpuFourierOutput>(
        (root / "real-transform.bin").c_str(), "VMECGPU1");
    auto fields = vmecpp_experiment::Read<GpuResidentInput, GpuResidentOutput>(
        (root / "resident-fields/real-fields.bin").c_str(), "VMECGPU5");
    auto constraints =
        vmecpp_experiment::Read<ConstraintInput, ConstraintOutput>(
            (root /
             "resident-constraints-capture/real-fields.bin.constraints.bin")
                .c_str(),
            "VMECGCC1");
    auto pipeline = vmecpp_experiment::Read<GpuPipelineInput, GpuForwardOutput>(
        (root / "a100-fused-results/fused/real-pipeline.bin").c_str(),
        "VMECGPU4");
    auto preparation =
        vmecpp_experiment::Read<GpuPreparationInput, GpuPreparationOutput>(
            (root / "resident-iteration10/real-preparation.bin").c_str(),
            "VMECGPR1");
    auto iteration =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            (root / "resident-iteration10/real-iteration.bin").c_str(),
            "VMECGIT1");
    auto step = vmecpp_experiment::Read<GpuStepInput, GpuStepOutput>(
        (root / "resident-iteration10/real-step.bin").c_str(), "VMECGST1");
    auto diagnostics =
        vmecpp_experiment::Read<GpuDiagnosticsInput, GpuDiagnosticsOutput>(
            (root / "resident-iteration10/real-diagnostics.bin").c_str(),
            "VMECGDG1");
    auto refresh = vmecpp_experiment::Read<GpuPreconditionerInput,
                                           GpuPreconditionerOutput>(
        (root / "resident-preconditioner-capture-retry/real-preconditioner.bin")
            .c_str(),
        "VMECGPC2");
    for (int i = 0; i < 6; ++i) {
      RequireEqual(Expected(pipeline, i), preparation.input_.data_[i],
                   pipeline.output_.sizes_[i], "matched raw forces");
      RequireEqual(Expected(preparation, i), iteration.input_.data_[i],
                   preparation.output_.sizes_[i], "matched prepared forces");
      RequireEqual(Expected(iteration, 12 + i), step.input_.data_[12 + i],
                   step.input_.sizes_[12 + i], "matched step forces");
    }
    GpuSolverSetup setup{};
    setup.basis_ = inverse.input_;
    for (int i = 0; i < 6; ++i) setup.basis_.data_[i] = nullptr;
    setup.fields_ = fields.input_;
    for (int i = 0; i < 18; ++i) setup.fields_.data_[i] = nullptr;
    std::ifstream file(
        root / "resident-constants/real-fields.bin.constants.bin",
        std::ios::binary);
    std::array<char, 8> magic{};
    file.read(magic.data(), 8);
    if (std::memcmp(magic.data(), "VMECGPC1", 8) != 0)
      throw std::runtime_error("Invalid force constants capture");
    file.read(reinterpret_cast<char*>(&setup.forces_.adiabatic_index_),
              sizeof(double));
    file.read(reinterpret_cast<char*>(setup.forces_.sizes_.data()),
              sizeof(setup.forces_.sizes_));
    const auto count = setup.forces_.sizes_[0] + setup.forces_.sizes_[1];
    if (count > 1000000) throw std::runtime_error("Invalid constants size");
    std::vector<double> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(double));
    if (!file) throw std::runtime_error("Incomplete constants capture");
    setup.forces_.data_ = {values.data(),
                           values.data() + setup.forces_.sizes_[0]};
    constexpr std::array<int, 11> kPipelineMap{26, 27, 35, 36, 37, 38,
                                               39, 40, 41, 42, 43};
    for (int i = 0; i < 15; ++i) {
      if (i == 2 || i == 3) {
        setup.constraints_.data_[i] = constraints.input_.data_[i - 2];
        setup.constraints_.sizes_[i] = constraints.input_.sizes_[i - 2];
      } else if (i >= 13) {
        setup.constraints_.data_[i] = inverse.input_.data_[i - 5];
        setup.constraints_.sizes_[i] = inverse.input_.sizes_[i - 5];
      } else {
        const int source = kPipelineMap[i < 2 ? i : i - 2];
        setup.constraints_.data_[i] = pipeline.input_.data_[source];
        setup.constraints_.sizes_[i] = pipeline.input_.sizes_[source];
      }
    }
    setup.preconditioner_ = refresh.input_;
    for (int i = 0; i < 26; ++i)
      if (i != 20 && i != 21 && i != 25)
        setup.preconditioner_.data_[i] = nullptr;
    setup.cached_preconditioner_ = refresh.output_;
    // Zero input cache proves refresh recomputes rather than echoing reference.
    setup.cached_diagnostics_.fill(0.0);
    setup.state_ = step.input_;
    setup.scalxc_ = preparation.input_.data_[6];
    setup.scalxc_size_ = preparation.input_.sizes_[6];
    std::unique_ptr<void, decltype(&DestroyGpuSolver)> context(
        CreateGpuSolver(setup), DestroyGpuSolver);
    GpuSolverScalars scalars{};
    GpuSolverControl control{true, true, preparation.input_.scalars_[1] != 0.0};
    RunGpuSolver(context.get(), control, scalars);
    if (scalars.bad_jacobian_ || scalars.invalid_norm_ != 0)
      throw std::runtime_error("Unexpected invalid geometry/norm");
    RequireEqual(Expected(iteration, 18), scalars.invariant_residuals_.data(),
                 3, "invariant residuals");
    RequireEqual(Expected(iteration, 19),
                 scalars.preconditioned_residuals_.data(), 3,
                 "preconditioned residuals");
    RequireEqual(diagnostics.expected_.data(), scalars.diagnostics_.data(), 7,
                 "diagnostics");
    RequireEqual(Expected(fields, 37), scalars.jacobian_extrema_.data(), 2,
                 "Jacobian extrema");
    DownloadGpuSolverCache(context.get(), refresh.output_);
    RequireEqual(refresh.expected_.data(), refresh.actual_.data(),
                 refresh.actual_.size(), "refreshed cache");
    control.refresh_preconditioner_ = false;
    RunGpuSolver(context.get(), control, scalars);
    RequireEqual(Expected(iteration, 18), scalars.invariant_residuals_.data(),
                 3, "repeat invariant residuals");
    RequireEqual(Expected(iteration, 19),
                 scalars.preconditioned_residuals_.data(), 3,
                 "repeat preconditioned residuals");
    RequireEqual(diagnostics.expected_.data(), scalars.diagnostics_.data(), 7,
                 "repeat diagnostics");
    GpuPreparationOutput geometry_output{};
    std::vector<double> geometry_values(6 * step.input_.sizes_[0]);
    for (int i = 0; i < 6; ++i) {
      geometry_output.data_[i] =
          geometry_values.data() + i * step.input_.sizes_[0];
      geometry_output.sizes_[i] = step.input_.sizes_[i];
    }
    DownloadGpuSolverGeometry(context.get(), GpuSolverGeometry::BACKUP,
                              geometry_output);
    for (int i = 0; i < 6; ++i)
      RequireEqual(step.input_.data_[i], geometry_output.data_[i],
                   geometry_output.sizes_[i], "initial restart backup");
    DownloadGpuSolverForces(context.get(), geometry_output);
    for (int i = 0; i < 6; ++i)
      RequireEqual(step.input_.data_[12 + i], geometry_output.data_[i],
                   geometry_output.sizes_[i], "prepared force download");
    std::vector<double> dvds(fields.output_.sizes_[32]);
    std::array<double, 2> axis{};
    DownloadGpuSolverPrintout(context.get(), dvds.data(), dvds.size(), axis);
    RequireEqual(Expected(fields, 32), dvds.data(), dvds.size(),
                 "printout dVdsH");
    RequireEqual(Expected(fields, 0), axis.data(), 1, "printout axis R");
    RequireEqual(Expected(fields, 6), axis.data() + 1, 1, "printout axis Z");
    AdvanceGpuSolver(context.get(), step.input_.scalars_);
    DownloadGpuSolverState(context.get(), step.output_);
    RequireEqual(step.expected_.data(), step.actual_.data(),
                 step.actual_.size(), "advanced geometry/velocity");
    RunGpuSolver(context.get(), control, scalars);
    for (int i : {0, 4, 5, 6})
      RequireEqual(diagnostics.expected_.data() + i,
                   scalars.diagnostics_.data() + i, 1,
                   "retained diagnostic cache");
    DownloadGpuSolverCache(context.get(), refresh.output_);
    RequireEqual(refresh.expected_.data(), refresh.actual_.data(),
                 refresh.actual_.size(), "retained matrix cache");
    for (auto which :
         {GpuSolverGeometry::CURRENT, GpuSolverGeometry::LAST_UPDATE,
          GpuSolverGeometry::LAST_VALID_UPDATE}) {
      DownloadGpuSolverGeometry(context.get(), which, geometry_output);
      RequireEqual(step.expected_.data(), geometry_values.data(),
                   geometry_values.size(), "evolved device snapshots");
    }
    SaveGpuSolverBackup(context.get());
    GpuStepInput altered = step.input_;
    std::vector<double> altered_values = step.actual_;
    for (double& value : altered_values) value = 2.0;
    for (int i = 0; i < 12; ++i)
      altered.data_[i] = altered_values.data() + i * step.input_.sizes_[0];
    UploadGpuSolverState(context.get(), altered);
    RestoreGpuSolverBackup(context.get());
    DownloadGpuSolverState(context.get(), step.output_);
    RequireEqual(step.expected_.data(), step.actual_.data(),
                 geometry_values.size(), "device backup restore");
    std::vector<double> zeros(geometry_values.size(), 0.0);
    RequireEqual(zeros.data(), step.actual_.data() + geometry_values.size(),
                 zeros.size(), "restore zeroes velocity");
    std::puts(
        "PASS: refresh, same-state repeat, step byte parity; evolved-state "
        "cache invariants. No full-trajectory claim.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
