// SPDX-License-Identifier: MIT
// Sequential control-matrix replay; no timing or full-trajectory claim.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
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

struct CpuReference {
  std::array<const double*, 6> forces_;
  const double* invariant_;
  const double* preconditioned_;
  const double* diagnostics_;
  const double* jacobian_;
  const double* cache_;
  const double* stepped_;
  std::size_t cache_size_;
};
struct Frame {
  std::string label_;
  GpuSolverScalars scalars_{};
  std::vector<double> cache_;
  std::vector<double> forces_;
  std::vector<double> state_;
  std::vector<double> snapshots_;
  std::vector<double> printout_;
};
Frame CaptureFrame(void* m_context, const GpuSolverSetup& setup,
                   const GpuSolverScalars& scalars, std::string label) {
  Frame result;
  result.label_ = std::move(label);
  result.scalars_ = scalars;
  const std::size_t coefficients = setup.state_.sizes_[0];
  const auto& sizes = setup.cached_preconditioner_.sizes_;
  result.cache_.resize(
      std::accumulate(sizes.begin(), sizes.end(), std::size_t{0}));
  GpuPreconditionerOutput cache{};
  std::size_t offset = 0;
  for (int i = 0; i < 20; ++i) {
    cache.data_[i] = result.cache_.data() + offset;
    cache.sizes_[i] = sizes[i];
    offset += sizes[i];
  }
  DownloadGpuSolverCache(m_context, cache);
  result.forces_.resize(6 * coefficients);
  GpuPreparationOutput geometry{};
  for (int i = 0; i < 6; ++i) {
    geometry.data_[i] = result.forces_.data() + i * coefficients;
    geometry.sizes_[i] = coefficients;
  }
  DownloadGpuSolverForces(m_context, geometry);
  result.state_.resize(12 * coefficients);
  GpuStepOutput state{};
  for (int i = 0; i < 12; ++i) {
    state.data_[i] = result.state_.data() + i * coefficients;
    state.sizes_[i] = coefficients;
  }
  DownloadGpuSolverState(m_context, state);
  result.snapshots_.resize(24 * coefficients);
  constexpr std::array<GpuSolverGeometry, 4> kSnapshots{
      GpuSolverGeometry::CURRENT, GpuSolverGeometry::BACKUP,
      GpuSolverGeometry::LAST_UPDATE, GpuSolverGeometry::LAST_VALID_UPDATE};
  for (int snapshot = 0; snapshot < 4; ++snapshot) {
    for (int i = 0; i < 6; ++i)
      geometry.data_[i] =
          result.snapshots_.data() + (6 * snapshot + i) * coefficients;
    DownloadGpuSolverGeometry(m_context, kSnapshots[snapshot], geometry);
  }
  const std::size_t half_rows = setup.state_.dimensions_[0] - 1;
  result.printout_.resize(half_rows + 2);
  std::array<double, 2> axis{};
  DownloadGpuSolverPrintout(m_context, result.printout_.data(), half_rows,
                            axis);
  std::copy(axis.begin(), axis.end(), result.printout_.begin() + half_rows);
  return result;
}
void RequireCpu(const Frame& frame, const CpuReference& reference,
                std::size_t coefficients) {
  RequireEqual(reference.invariant_, frame.scalars_.invariant_residuals_.data(),
               3, "CPU invariant");
  RequireEqual(reference.preconditioned_,
               frame.scalars_.preconditioned_residuals_.data(), 3,
               "CPU preconditioned");
  RequireEqual(reference.diagnostics_, frame.scalars_.diagnostics_.data(), 7,
               "CPU diagnostics");
  RequireEqual(reference.jacobian_, frame.scalars_.jacobian_extrema_.data(), 2,
               "CPU Jacobian");
  RequireEqual(reference.cache_, frame.cache_.data(), reference.cache_size_,
               "CPU cache20");
  for (int i = 0; i < 6; ++i)
    RequireEqual(reference.forces_[i], frame.forces_.data() + i * coefficients,
                 coefficients, "CPU force6");
}
void RequireFrame(const Frame& expected, const Frame& actual) {
  if (expected.label_ != actual.label_)
    throw std::runtime_error("Frame order mismatch");
  RequireEqual(expected.scalars_.invariant_residuals_.data(),
               actual.scalars_.invariant_residuals_.data(), 3,
               "variant invariant");
  RequireEqual(expected.scalars_.preconditioned_residuals_.data(),
               actual.scalars_.preconditioned_residuals_.data(), 3,
               "variant preconditioned");
  RequireEqual(expected.scalars_.diagnostics_.data(),
               actual.scalars_.diagnostics_.data(), 7, "variant diagnostics");
  RequireEqual(expected.scalars_.jacobian_extrema_.data(),
               actual.scalars_.jacobian_extrema_.data(), 2, "variant Jacobian");
  if (expected.scalars_.invalid_norm_ != actual.scalars_.invalid_norm_ ||
      expected.scalars_.bad_jacobian_ != actual.scalars_.bad_jacobian_)
    throw std::runtime_error("Variant scalar flags mismatch");
  for (const auto* vectors :
       {&expected.cache_, &expected.forces_, &expected.state_,
        &expected.snapshots_, &expected.printout_}) {
    const std::vector<double>* other =
        vectors == &expected.cache_       ? &actual.cache_
        : vectors == &expected.forces_    ? &actual.forces_
        : vectors == &expected.state_     ? &actual.state_
        : vectors == &expected.snapshots_ ? &actual.snapshots_
                                          : &actual.printout_;
    if (vectors->size() != other->size())
      throw std::runtime_error("Variant shape mismatch");
    RequireEqual(vectors->data(), other->data(), vectors->size(),
                 expected.label_.c_str());
  }
}
void SetFlag(const char* name, bool enabled) {
  if (setenv(name, enabled ? "1" : "0", 1) != 0)
    throw std::runtime_error("Cannot set replay environment");
}
std::vector<Frame> RunScenario(const GpuSolverSetup& setup,
                               const CpuReference& reference, bool zero_m1,
                               bool outer_graph, bool pinned, bool optimistic) {
  SetFlag("VMECPP_GPU_SOLVER_GRAPH", outer_graph);
  SetFlag("VMECPP_GPU_SOLVER_PINNED_SCALARS", pinned);
  SetFlag("VMECPP_GPU_RESIDUAL_GRAPH", optimistic);
  SetFlag("VMECPP_GPU_DIAGNOSTICS_GRAPH", optimistic);
  if (setenv("VMECPP_GPU_RESIDUAL_ROUNDS", optimistic ? "12" : "0", 1) != 0 ||
      setenv("VMECPP_GPU_DIAGNOSTICS_ROUNDS", optimistic ? "16" : "0", 1) != 0)
    throw std::runtime_error("Cannot set replay rounds");
  // The context is destroyed before this function returns. Nested workspaces
  // are thread-local and must never be exercised by two live contexts here.
  std::unique_ptr<void, decltype(&DestroyGpuSolver)> context(
      CreateGpuSolver(setup), DestroyGpuSolver);
  const std::size_t coefficients = setup.state_.sizes_[0];
  std::vector<Frame> frames;
  GpuSolverScalars scalars{};
  const GpuSolverControl captured{true, true, zero_m1};
  for (int repetition = 0; repetition < 3; ++repetition) {
    RunGpuSolver(context.get(), captured, scalars);
    if (scalars.bad_jacobian_ || scalars.invalid_norm_ != 0)
      throw std::runtime_error("Unexpected captured geometry/norm failure");
    frames.push_back(
        CaptureFrame(context.get(), setup, scalars,
                     "CPU captured control " + std::to_string(repetition)));
    RequireCpu(frames.back(), reference, coefficients);
  }
  AdvanceGpuSolver(context.get(), setup.state_.scalars_);
  frames.push_back(
      CaptureFrame(context.get(), setup, scalars, "CPU captured step"));
  RequireEqual(reference.stepped_, frames.back().state_.data(),
               12 * coefficients, "CPU captured step12");
  UploadGpuSolverState(context.get(), setup.state_);
  frames.push_back(
      CaptureFrame(context.get(), setup, scalars, "initial state upload"));
  for (int i = 0; i < 12; ++i)
    RequireEqual(setup.state_.data_[i],
                 frames.back().state_.data() + i * coefficients, coefficients,
                 "initial upload x/v");
  for (bool refresh : {false, true})
    for (bool edge : {false, true})
      for (bool zero : {false, true})
        for (int repetition = 0; repetition < 3; ++repetition) {
          // Arbitrary control triples are component probes, not a valid VMEC
          // controller schedule. Keep each force evaluation on the captured
          // geometry while still checking its resulting time step.
          UploadGpuSolverState(context.get(), setup.state_);
          const GpuSolverControl control{refresh, edge, zero};
          RunGpuSolver(context.get(), control, scalars);
          if (scalars.bad_jacobian_ || scalars.invalid_norm_ != 0) {
            std::fprintf(stderr,
                         "control %d%d%d repeat %d: badJac=%d invalidNorm=%d "
                         "minTau=%.17g maxTau=%.17g\n",
                         refresh, edge, zero, repetition, scalars.bad_jacobian_,
                         scalars.invalid_norm_, scalars.jacobian_extrema_[0],
                         scalars.jacobian_extrema_[1]);
            throw std::runtime_error(
                "Unexpected control geometry/norm failure");
          }
          AdvanceGpuSolver(context.get(), setup.state_.scalars_);
          const std::string label =
              "control " + std::to_string(refresh) + std::to_string(edge) +
              std::to_string(zero) + " repeat " + std::to_string(repetition);
          frames.push_back(CaptureFrame(context.get(), setup, scalars, label));
        }
  SaveGpuSolverBackup(context.get());
  frames.push_back(CaptureFrame(context.get(), setup, scalars, "saved backup"));
  const auto saved_state = frames.back().state_;
  const auto saved_forces = frames.back().forces_;
  GpuStepInput altered = setup.state_;
  std::vector<double> altered_values(12 * coefficients, 2.0);
  std::fill(altered_values.begin() + 6 * coefficients, altered_values.end(),
            -3.0);
  for (int i = 0; i < 12; ++i)
    altered.data_[i] = altered_values.data() + i * coefficients;
  UploadGpuSolverState(context.get(), altered);
  frames.push_back(
      CaptureFrame(context.get(), setup, scalars, "altered state upload"));
  RequireEqual(altered_values.data(), frames.back().state_.data(),
               altered_values.size(), "altered upload x/v");
  RequireEqual(saved_forces.data(), frames.back().forces_.data(),
               saved_forces.size(), "upload retains force6");
  RestoreGpuSolverBackup(context.get());
  frames.push_back(
      CaptureFrame(context.get(), setup, scalars, "restored backup"));
  RequireEqual(saved_state.data(), frames.back().state_.data(),
               6 * coefficients, "backup restores x6");
  const std::vector<double> zeros(6 * coefficients, 0.0);
  RequireEqual(zeros.data(), frames.back().state_.data() + 6 * coefficients,
               zeros.size(), "backup clears v6");
  UploadGpuSolverState(context.get(), setup.state_);
  frames.push_back(CaptureFrame(context.get(), setup, scalars,
                                "final original state upload"));
  for (int i = 0; i < 12; ++i)
    RequireEqual(setup.state_.data_[i],
                 frames.back().state_.data() + i * coefficients, coefficients,
                 "final upload x/v");
  RunGpuSolver(context.get(), captured, scalars);
  frames.push_back(CaptureFrame(context.get(), setup, scalars,
                                "CPU captured control after transitions"));
  RequireCpu(frames.back(), reference, coefficients);
  AdvanceGpuSolver(context.get(), setup.state_.scalars_);
  frames.push_back(CaptureFrame(context.get(), setup, scalars,
                                "CPU captured step after transitions"));
  RequireEqual(reference.stepped_, frames.back().state_.data(),
               12 * coefficients, "CPU step12 after transitions");
  return frames;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::runtime_error(
          "Usage: solver-control-replay optimization-root");
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

    setup.cached_dvds_ = Expected(fields, 32);
    CpuReference reference{};
    for (int i = 0; i < 6; ++i)
      reference.forces_[i] = step.input_.data_[12 + i];
    reference.invariant_ = Expected(iteration, 18);
    reference.preconditioned_ = Expected(iteration, 19);
    reference.diagnostics_ = diagnostics.expected_.data();
    reference.jacobian_ = Expected(fields, 37);
    reference.cache_ = refresh.expected_.data();
    reference.stepped_ = step.expected_.data();
    reference.cache_size_ = refresh.expected_.size();
    const bool zero_m1 = preparation.input_.scalars_[1] != 0.0;
    const auto baseline =
        RunScenario(setup, reference, zero_m1, false, false, false);
    std::printf(
        "PASS: ordered reference, %zu recorded frames; context destroyed.\n",
        baseline.size());
    std::fflush(stdout);
    for (const auto& flags :
         {std::array<bool, 2>{false, false}, std::array<bool, 2>{true, false},
          std::array<bool, 2>{false, true}, std::array<bool, 2>{true, true}}) {
      const auto actual =
          RunScenario(setup, reference, zero_m1, flags[0], flags[1], true);
      if (actual.size() != baseline.size())
        throw std::runtime_error("Frame count mismatch");
      for (std::size_t i = 0; i < actual.size(); ++i)
        RequireFrame(baseline[i], actual[i]);
      std::printf(
          "PASS: optimized outer_graph=%d pinned=%d, %zu frames exact; context "
          "destroyed.\n",
          flags[0], flags[1], actual.size());
      std::fflush(stdout);
    }
    std::puts(
        "PASS: 8 control triples x3 repeats per variant, CPU captured parity, "
        "backup/restore/uploads; no timing/full-trajectory claim.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
