// SPDX-License-Identifier: MIT
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gpu_fourier.h"
#include "gpu_pipeline.h"
#include "gpu_resident.h"
#include "resident_replay_capture.h"
#ifdef VMECPP_ITERATION_TAIL_REPLAY
#include <cuda_runtime.h>

#include "gpu_diagnostics.h"
#include "gpu_iteration.h"
#include "gpu_preconditioner.h"

namespace {
void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct TailContext {
  GpuPreconditionerDeviceState refresh_{};
  GpuDiagnosticsDeviceState diagnostics_{};
  using Refresh = void (*)(GpuPreconditionerDeviceState&);
  using Diagnostics = void (*)(GpuDiagnosticsDeviceState&);
  Refresh refresh_function_ = nullptr;
  Diagnostics diagnostics_function_ = nullptr;
  bool refresh_enabled_ = false;
  std::array<std::size_t, 20> refresh_sizes_{};
  std::vector<double> refresh_expected_;
  std::vector<double> diagnostics_expected_;
  GpuIterationDeviceState state_{};
  GpuPreparationDeviceState preparation_{};
  using Iteration = void (*)(GpuIterationDeviceState&);
  using Residual = void (*)(GpuIterationDeviceState&, bool);
  using Preparation = void (*)(GpuPreparationDeviceState&);
  Iteration m1_ = nullptr;
  Iteration rz_ = nullptr;
  Iteration lambda_ = nullptr;
  Residual residual_ = nullptr;
  Preparation prepare_ = nullptr;
  using Step = void (*)(GpuStepDeviceState&);
  Step step_function_ = nullptr;
  GpuStepDeviceState step_{};
  std::array<double*, 12> initial_step_{};
  bool run_step_ = false;
  double* residual_storage_ = nullptr;
  std::vector<void*> allocations_;
  double* Allocate(std::size_t size) {
    double* pointer = nullptr;
    Check(cudaMalloc(&pointer, size * sizeof(double)));
    allocations_.push_back(pointer);
    return pointer;
  }
  ~TailContext() {
    for (void* pointer : allocations_) cudaFree(pointer);
  }
};
template <class Capture>
const double* ExpectedArray(const Capture& capture, int index) {
  return capture.expected_.data() +
         std::accumulate(capture.output_.sizes_.begin(),
                         capture.output_.sizes_.begin() + index,
                         std::size_t{0});
}
void RequireSame(const double* a, std::size_t a_size, const double* b,
                 std::size_t b_size) {
  if (a_size != b_size || std::memcmp(a, b, a_size * sizeof(double)) != 0) {
    throw std::runtime_error(
        "Refresh/diagnostic capture does not share the resident state");
  }
}
void BeforeForces(GpuResidentConstraintDeviceState& m_fields, void* m_context) {
  auto& c = *static_cast<TailContext*>(m_context);
  const auto& g = m_fields.force_.geometry_;
  const auto& h = g.half_;
  const auto& f = g.full_;
  const auto& q = g.profiles_;
  const double* sm = c.refresh_.input_[20];
  const double* sp = c.refresh_.input_[21];
  const double* powers = c.refresh_.input_[25];
  c.refresh_.input_ = {h[0],
                       m_fields.force_.total_pressure_,
                       h[5],
                       h[11],
                       h[6],
                       h[7],
                       h[8],
                       h[9],
                       h[4],
                       h[2],
                       f[8],
                       f[9],
                       f[7],
                       h[3],
                       h[1],
                       f[2],
                       f[3],
                       f[1],
                       q[1],
                       q[0],
                       sm,
                       sp,
                       q[2],
                       m_fields.geometry_[1],
                       m_fields.geometry_[2],
                       powers};
  c.refresh_.stream_ = g.stream_;
  c.refresh_function_(c.refresh_);
  const auto& o = c.refresh_.output_;
  c.state_.constants_ = {o[5],  o[7],  o[9],  o[11], o[14], o[15],
                         o[16], o[17], o[18], o[19], o[0]};
  m_fields.constants_[3] = o[13];
  c.diagnostics_.data_ = {h[6],
                          h[10],
                          h[11],
                          h[12],
                          h[13],
                          q[2],
                          m_fields.force_.constants_[0],
                          q[10],
                          h[7],
                          h[0],
                          c.initial_step_[0],
                          c.initial_step_[1],
                          c.initial_step_[2],
                          c.initial_step_[3]};
  c.diagnostics_.stream_ = g.stream_;
  c.diagnostics_function_(c.diagnostics_);
}
void Tail(const std::array<double*, 6>& m_forces,
          const std::array<int, 13>& dimensions, void* stream,
          void* m_context) {
  auto& context = *static_cast<TailContext*>(m_context);
  if (dimensions[11] != context.state_.dimensions_[0] ||
      dimensions[5] != context.state_.dimensions_[1] ||
      dimensions[6] != context.state_.dimensions_[2]) {
    throw std::runtime_error("Resident iteration dimensions differ");
  }
  context.preparation_.coefficients_ = m_forces;
  context.preparation_.stream_ = stream;
  context.state_.forces_ = m_forces;
  context.state_.stream_ = stream;
  context.prepare_(context.preparation_);
  context.state_.residuals_ = context.residual_storage_;
  context.residual_(context.state_, true);
  context.m1_(context.state_);
  context.rz_(context.state_);
  context.lambda_(context.state_);
  context.state_.residuals_ = context.residual_storage_ + 3;
  context.residual_(context.state_, true);
  if (context.run_step_) {
    context.step_.stream_ = stream;
    for (int i = 0; i < 6; ++i) {
      context.step_.forces_[i] = m_forces[i];
      Check(cudaMemcpyAsync(
          context.step_.geometry_[i], context.initial_step_[i],
          context.step_.coefficient_count_ * sizeof(double),
          cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
      Check(cudaMemcpyAsync(
          context.step_.velocity_[i], context.initial_step_[i + 6],
          context.step_.coefficient_count_ * sizeof(double),
          cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
    }
    context.step_function_(context.step_);
  }
}
}  // namespace
#endif

struct ConstraintCaptureInput {
  std::array<const double*, 2> data_;
  std::array<std::size_t, 2> sizes_;
  std::array<int, 13> dimensions_;
};
struct ConstraintCaptureOutput {
  std::array<double*, 4> data_;
  std::array<std::size_t, 4> sizes_;
};

namespace {
int ReplayCount(const char* name, int default_value, int minimum) {
  const char* value = std::getenv(name);
  if (value == nullptr) return default_value;
  char* end = nullptr;
  const long count = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || count < minimum || count > 1000) {
    throw std::runtime_error("Invalid replay execution count");
  }
  return static_cast<int>(count);
}
}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    constexpr int kArguments = 10;
#else
    constexpr int kArguments = 7;
#endif
    if (argc != kArguments
#ifdef VMECPP_ITERATION_TAIL_REPLAY
        && argc != kArguments + 1 && argc != kArguments + 3
#endif
    ) {
      throw std::runtime_error(
          "Usage: resident_force_eval_replay inverse.bin fields.bin "
          "constants.bin constraints.bin pipeline.bin library.so");
    }
    auto inverse = vmecpp_experiment::Read<GpuFourierInput, GpuFourierOutput>(
        argv[1], "VMECGPU1");
    auto fields = vmecpp_experiment::Read<GpuResidentInput, GpuResidentOutput>(
        argv[2], "VMECGPU5");
    auto auxiliary =
        vmecpp_experiment::Read<ConstraintCaptureInput,
                                ConstraintCaptureOutput>(argv[4], "VMECGCC1");
    auto pipeline = vmecpp_experiment::Read<GpuPipelineInput, GpuForwardOutput>(
        argv[5], "VMECGPU4");
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    auto preparation =
        vmecpp_experiment::Read<GpuPreparationInput, GpuPreparationOutput>(
            argv[7], "VMECGPR1");
    auto iteration =
        vmecpp_experiment::Read<GpuIterationInput, GpuIterationOutput>(
            argv[8], "VMECGIT1");
    const auto force_count = pipeline.expected_.size();
    if (preparation.expected_.size() != force_count ||
        std::memcmp(pipeline.expected_.data(), preparation.values_.data(),
                    force_count * sizeof(double)) != 0 ||
        std::memcmp(preparation.expected_.data(), iteration.values_.data(),
                    force_count * sizeof(double)) != 0) {
      throw std::runtime_error(
          "Force evaluation, preparation and preconditioner captures do not "
          "share state");
    }
    void* iteration_library = dlopen(argv[9], RTLD_NOW | RTLD_LOCAL);
    if (iteration_library == nullptr) throw std::runtime_error(dlerror());
    TailContext tail;
    tail.m1_ = reinterpret_cast<TailContext::Iteration>(
        dlsym(iteration_library, "LaunchGpuIterationM1"));
    tail.rz_ = reinterpret_cast<TailContext::Iteration>(
        dlsym(iteration_library, "LaunchGpuIterationRZ"));
    tail.lambda_ = reinterpret_cast<TailContext::Iteration>(
        dlsym(iteration_library, "LaunchGpuIterationLambda"));
    tail.residual_ = reinterpret_cast<TailContext::Residual>(
        dlsym(iteration_library, "LaunchGpuIterationResiduals"));
    tail.prepare_ = reinterpret_cast<TailContext::Preparation>(
        dlsym(iteration_library, "LaunchGpuIterationPreparation"));
    if (!tail.m1_ || !tail.rz_ || !tail.lambda_ || !tail.residual_ ||
        !tail.prepare_)
      throw std::runtime_error("Iteration device API missing");
    tail.state_.dimensions_ = iteration.input_.dimensions_;
    tail.preparation_.dimensions_ = preparation.input_.dimensions_;
    tail.preparation_.m1_scale_ = preparation.input_.scalars_[0];
    tail.preparation_.zero_z_m1_ = preparation.input_.scalars_[1] != 0.0;
    for (int i = 0; i < 11; ++i) {
      const auto size = iteration.input_.sizes_[i + 6];
      double* pointer = tail.Allocate(size);
      Check(cudaMemcpy(pointer, iteration.input_.data_[i + 6],
                       size * sizeof(double), cudaMemcpyHostToDevice));
      tail.state_.constants_[i] = pointer;
    }
    double* scalxc = tail.Allocate(preparation.input_.sizes_[6]);
    Check(cudaMemcpy(scalxc, preparation.input_.data_[6],
                     preparation.input_.sizes_[6] * sizeof(double),
                     cudaMemcpyHostToDevice));
    tail.preparation_.scalxc_ = scalxc;
    for (double*& pointer : tail.state_.factored_upper_)
      pointer = tail.Allocate(iteration.input_.sizes_[0]);
    for (double*& pointer : tail.state_.denominators_)
      pointer = tail.Allocate(iteration.input_.sizes_[0]);
    tail.residual_storage_ = tail.Allocate(6);
    // No CPU force array is used by the chained device execution.
    std::fill_n(preparation.input_.data_.begin(), 6, nullptr);
    std::fill_n(iteration.input_.data_.begin(), 6, nullptr);
    std::vector<double> expected_step;
    if (argc >= 11) {
      auto step = vmecpp_experiment::Read<GpuStepInput, GpuStepOutput>(
          argv[10], "VMECGST1");
      if (step.input_.dimensions_ != iteration.input_.dimensions_ ||
          std::memcmp(step.input_.data_[12],
                      iteration.expected_.data() + 2 * force_count,
                      force_count * sizeof(double)) != 0) {
        throw std::runtime_error(
            "Preconditioner and step captures do not share force state");
      }
      expected_step = step.expected_;
      tail.step_function_ = reinterpret_cast<TailContext::Step>(
          dlsym(iteration_library, "LaunchGpuIterationStep"));
      if (!tail.step_function_)
        throw std::runtime_error("Device coefficient update missing");
      tail.step_.scalars_ = step.input_.scalars_;
      tail.step_.coefficient_count_ = step.input_.sizes_[0];
      for (int i = 0; i < 12; ++i) {
        tail.initial_step_[i] = tail.Allocate(step.input_.sizes_[i]);
        Check(cudaMemcpy(tail.initial_step_[i], step.input_.data_[i],
                         step.input_.sizes_[i] * sizeof(double),
                         cudaMemcpyHostToDevice));
        double* pointer = tail.Allocate(step.input_.sizes_[i]);
        if (i < 6)
          tail.step_.geometry_[i] = pointer;
        else
          tail.step_.velocity_[i - 6] = pointer;
      }
      tail.run_step_ = true;
    }
#endif
    const auto geometry_size =
        std::accumulate(fields.input_.sizes_.begin(),
                        fields.input_.sizes_.begin() + 18, std::size_t{0});
    if (geometry_size != inverse.expected_.size() ||
        std::memcmp(inverse.expected_.data(), fields.values_.data(),
                    geometry_size * sizeof(double)) != 0) {
      throw std::runtime_error(
          "Inverse and field references do not share geometry");
    }
    std::size_t offset = auxiliary.output_.sizes_[0];
    for (int source : {29, 30, 28}) {
      const int target = source == 29 ? 1 : (source == 30 ? 2 : 3);
      if (auxiliary.output_.sizes_[target] != pipeline.input_.sizes_[source] ||
          std::memcmp(auxiliary.expected_.data() + offset,
                      pipeline.input_.data_[source],
                      pipeline.input_.sizes_[source] * sizeof(double)) != 0) {
        throw std::runtime_error(
            "Constraint and projection references do not share state");
      }
      offset += auxiliary.output_.sizes_[target];
    }
    GpuResidentForceConstants forces{};
    std::ifstream file(argv[3], std::ios::binary);
    std::array<char, 8> magic{};
    file.read(magic.data(), 8);
    if (std::memcmp(magic.data(), "VMECGPC1", 8) != 0) {
      throw std::runtime_error("Invalid force constants capture");
    }
    file.read(reinterpret_cast<char*>(&forces.adiabatic_index_),
              sizeof(double));
    file.read(reinterpret_cast<char*>(forces.sizes_.data()),
              sizeof(forces.sizes_));
    const auto count = forces.sizes_[0] + forces.sizes_[1];
    if (!file || count > 1000000) {
      throw std::runtime_error("Invalid constant dimensions");
    }
    std::vector<double> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(double));
    if (!file) {
      throw std::runtime_error("Incomplete force constants capture");
    }
    forces.data_ = {values.data(), values.data() + forces.sizes_[0]};
    GpuResidentConstraintConstants constraints{};
    constraints.data_ = {pipeline.input_.data_[26], pipeline.input_.data_[27],
                         auxiliary.input_.data_[0], auxiliary.input_.data_[1],
                         pipeline.input_.data_[35], pipeline.input_.data_[36],
                         pipeline.input_.data_[37], pipeline.input_.data_[38],
                         pipeline.input_.data_[39], pipeline.input_.data_[40],
                         pipeline.input_.data_[41], pipeline.input_.data_[42],
                         pipeline.input_.data_[43], inverse.input_.data_[8],
                         inverse.input_.data_[9]};
    constraints.sizes_ = {
        pipeline.input_.sizes_[26], pipeline.input_.sizes_[27],
        auxiliary.input_.sizes_[0], auxiliary.input_.sizes_[1],
        pipeline.input_.sizes_[35], pipeline.input_.sizes_[36],
        pipeline.input_.sizes_[37], pipeline.input_.sizes_[38],
        pipeline.input_.sizes_[39], pipeline.input_.sizes_[40],
        pipeline.input_.sizes_[41], pipeline.input_.sizes_[42],
        pipeline.input_.sizes_[43], inverse.input_.sizes_[8],
        inverse.input_.sizes_[9]};
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    if (argc == 13) {
      auto refresh = vmecpp_experiment::Read<GpuPreconditionerInput,
                                             GpuPreconditionerOutput>(
          argv[11], "VMECGPC2");
      auto diagnostic =
          vmecpp_experiment::Read<GpuDiagnosticsInput, GpuDiagnosticsOutput>(
              argv[12], "VMECGDG1");
      const auto& r = refresh.input_;
      for (const auto [a, b] :
           std::array{std::pair{0, 0}, std::pair{1, 9}, std::pair{2, 5},
                      std::pair{3, 8}, std::pair{4, 6}, std::pair{8, 4},
                      std::pair{9, 2}, std::pair{10, 14}, std::pair{11, 15},
                      std::pair{12, 16}, std::pair{13, 3}, std::pair{14, 1},
                      std::pair{15, 12}, std::pair{16, 13}, std::pair{17, 11},
                      std::pair{18, 22}, std::pair{19, 21}}) {
        RequireSame(r.data_[a], r.sizes_[a], pipeline.input_.data_[b],
                    pipeline.input_.sizes_[b]);
      }
      for (int i = 0; i < 3; ++i)
        RequireSame(r.data_[5 + i], r.sizes_[5 + i],
                    ExpectedArray(fields, 25 + i),
                    fields.output_.sizes_[25 + i]);
      RequireSame(r.data_[22], r.sizes_[22], fields.input_.data_[20],
                  fields.input_.sizes_[20]);
      RequireSame(r.data_[23], r.sizes_[23], ExpectedArray(auxiliary, 1),
                  auxiliary.output_.sizes_[1]);
      RequireSame(r.data_[24], r.sizes_[24], ExpectedArray(auxiliary, 2),
                  auxiliary.output_.sizes_[2]);
      if (!std::equal(diagnostic.input_.dimensions_.begin(),
                      diagnostic.input_.dimensions_.end(),
                      r.dimensions_.begin()) ||
          r.scalars_[0] != fields.input_.scalars_[0] ||
          r.scalars_[1] != fields.input_.scalars_[2]) {
        throw std::runtime_error(
            "Refresh/diagnostic dimensions or constants differ");
      }
      const auto& d = diagnostic.input_;
      constexpr std::array<int, 5> field_arrays{24, 28, 29, 30, 31};
      for (int i = 0; i < 5; ++i)
        RequireSame(d.data_[i], d.sizes_[i],
                    ExpectedArray(fields, field_arrays[i]),
                    fields.output_.sizes_[field_arrays[i]]);
      RequireSame(d.data_[5], d.sizes_[5], fields.input_.data_[20],
                  fields.input_.sizes_[20]);
      RequireSame(d.data_[6], d.sizes_[6], forces.data_[0], forces.sizes_[0]);
      RequireSame(d.data_[7], d.sizes_[7], ExpectedArray(fields, 32),
                  fields.output_.sizes_[32]);
      RequireSame(d.data_[8], d.sizes_[8], ExpectedArray(fields, 25),
                  fields.output_.sizes_[25]);
      RequireSame(d.data_[9], d.sizes_[9], ExpectedArray(fields, 18),
                  fields.output_.sizes_[18]);
      auto step_reference =
          vmecpp_experiment::Read<GpuStepInput, GpuStepOutput>(argv[10],
                                                               "VMECGST1");
      for (int i = 0; i < 4; ++i)
        RequireSame(d.data_[10 + i], d.sizes_[10 + i],
                    step_reference.input_.data_[i],
                    step_reference.input_.sizes_[i]);
      tail.refresh_function_ = reinterpret_cast<TailContext::Refresh>(
          dlsym(iteration_library, "LaunchGpuPreconditioner"));
      tail.diagnostics_function_ = reinterpret_cast<TailContext::Diagnostics>(
          dlsym(iteration_library, "LaunchGpuDiagnostics"));
      if (!tail.refresh_function_ || !tail.diagnostics_function_)
        throw std::runtime_error("Refresh/diagnostic device API missing");
      tail.refresh_.dimensions_ = r.dimensions_;
      tail.refresh_.scalars_ = r.scalars_;
      tail.refresh_sizes_ = refresh.output_.sizes_;
      tail.refresh_expected_ = refresh.expected_;
      for (int i = 0; i < 20; ++i)
        tail.refresh_.output_[i] = tail.Allocate(refresh.output_.sizes_[i]);
      for (int i : {20, 21, 25}) {
        double* pointer = tail.Allocate(r.sizes_[i]);
        Check(cudaMemcpy(pointer, r.data_[i], r.sizes_[i] * sizeof(double),
                         cudaMemcpyHostToDevice));
        tail.refresh_.input_[i] = pointer;
      }
      const auto rows = r.dimensions_[0] - 1;
      tail.refresh_.terms_ = tail.Allocate(std::size_t{21} * rows *
                                           r.dimensions_[5] * r.dimensions_[6]);
      tail.refresh_.sums_ = tail.Allocate(std::size_t{21} * rows);
      tail.refresh_.invalid_norm_ = reinterpret_cast<int*>(tail.Allocate(1));
      tail.diagnostics_.dimensions_ = d.dimensions_;
      tail.diagnostics_.scalars_ = d.scalars_;
      tail.diagnostics_.sums_ = tail.Allocate(6);
      tail.diagnostics_.output_ = tail.Allocate(7);
      tail.diagnostics_expected_ = diagnostic.expected_;
      tail.refresh_enabled_ = true;
    }
#endif
    GpuResidentForceEvalOutput output{};
    std::copy_n(pipeline.output_.sizes_.begin(), 6, output.sizes_.begin());
    std::copy_n(auxiliary.output_.sizes_.begin(), 4, output.sizes_.begin() + 6);
    std::vector<double> expected = pipeline.expected_;
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    const auto force_offset =
        std::accumulate(iteration.output_.sizes_.begin(),
                        iteration.output_.sizes_.begin() + 12, std::size_t{0});
    std::copy_n(iteration.expected_.begin() + force_offset, force_count,
                expected.begin());
    const std::vector<double> expected_residuals(iteration.expected_.end() - 6,
                                                 iteration.expected_.end());
#endif
    expected.insert(expected.end(), auxiliary.expected_.begin(),
                    auxiliary.expected_.end());
    std::vector<double> actual(expected.size());
    offset = 0;
    for (std::size_t i = 0; i < 10; ++i) {
      output.data_[i] = actual.data() + offset;
      offset += output.sizes_[i];
    }
    std::fill_n(fields.input_.data_.begin(), 18, nullptr);
    void* library = dlopen(argv[6], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      throw std::runtime_error(dlerror());
    }
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    using Function = void (*)(
        const GpuFourierInput&, const GpuResidentInput&,
        const GpuResidentForceConstants&, const GpuResidentConstraintConstants&,
        GpuResidentForceEvalOutput&, GpuForceEvaluationTail, void*);
    auto run = reinterpret_cast<Function>(
        dlsym(library, "RunGpuResidentForceEvalWithTail"));
    using StagedFunction = void (*)(
        const GpuFourierInput&, const GpuResidentInput&,
        const GpuResidentForceConstants&, const GpuResidentConstraintConstants&,
        GpuResidentForceEvalOutput&, GpuResidentPreForceStage,
        GpuForceEvaluationTail, void*);
    auto run_stages = reinterpret_cast<StagedFunction>(
        dlsym(library, "RunGpuResidentForceEvalWithStages"));
    const auto execute = [&] {
      if (tail.refresh_enabled_) {
        if (!run_stages)
          throw std::runtime_error("Resident refresh integration API missing");
        run_stages(inverse.input_, fields.input_, forces, constraints, output,
                   BeforeForces, Tail, &tail);
      } else {
        run(inverse.input_, fields.input_, forces, constraints, output, Tail,
            &tail);
      }
      std::vector<double> residuals(6);
      Check(cudaMemcpy(residuals.data(), tail.residual_storage_,
                       6 * sizeof(double), cudaMemcpyDeviceToHost));
      if (vmecpp_experiment::Compare(expected_residuals, residuals) != 0)
        throw std::runtime_error("Resident residual mismatch");
      if (tail.run_step_) {
        std::vector<double> actual_step(expected_step.size());
        const auto count = tail.step_.coefficient_count_;
        for (int i = 0; i < 12; ++i) {
          const double* pointer =
              i < 6 ? tail.step_.geometry_[i] : tail.step_.velocity_[i - 6];
          Check(cudaMemcpy(actual_step.data() + i * count, pointer,
                           count * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (vmecpp_experiment::Compare(expected_step, actual_step) != 0)
          throw std::runtime_error("Resident coefficient update mismatch");
      }
      if (tail.refresh_enabled_) {
        int invalid = 0;
        Check(cudaMemcpy(&invalid, tail.refresh_.invalid_norm_, sizeof(int),
                         cudaMemcpyDeviceToHost));
        if (invalid != 0)
          throw std::runtime_error("GPU refresh reported invalid norm");
        std::vector<double> refreshed(tail.refresh_expected_.size());
        std::size_t cursor = 0;
        for (int i = 0; i < 20; ++i) {
          Check(cudaMemcpy(refreshed.data() + cursor, tail.refresh_.output_[i],
                           tail.refresh_sizes_[i] * sizeof(double),
                           cudaMemcpyDeviceToHost));
          cursor += tail.refresh_sizes_[i];
        }
        if (vmecpp_experiment::Compare(tail.refresh_expected_, refreshed) != 0)
          throw std::runtime_error("Connected GPU refresh differs");
        std::vector<double> diagnostic(7);
        Check(cudaMemcpy(diagnostic.data(), tail.diagnostics_.output_,
                         7 * sizeof(double), cudaMemcpyDeviceToHost));
        if (vmecpp_experiment::Compare(tail.diagnostics_expected_,
                                       diagnostic) != 0)
          throw std::runtime_error("Connected GPU energy/norm differs");
      }
    };
#else
    using Function = void (*)(const GpuFourierInput&, const GpuResidentInput&,
                              const GpuResidentForceConstants&,
                              const GpuResidentConstraintConstants&,
                              GpuResidentForceEvalOutput&);
    auto run =
        reinterpret_cast<Function>(dlsym(library, "RunGpuResidentForceEval"));
    if (run == nullptr) {
      throw std::runtime_error("Resident force evaluation entry missing");
    }
    const auto execute = [&] {
      run(inverse.input_, fields.input_, forces, constraints, output);
    };
#endif
    if (run == nullptr)
      throw std::runtime_error("Resident evaluation entry missing");
    execute();
    auto differences = vmecpp_experiment::Compare(expected, actual);
    if (differences != 0) {
      std::fprintf(stderr, "Different values: %zu\n", differences);
      return 2;
    }
    const int kWarmups = ReplayCount("VMECPP_REPLAY_WARMUPS", 10, 0);
    const int kRepeats = ReplayCount("VMECPP_REPLAY_REPEATS", 100, 1);
    for (int i = 0; i < kWarmups; ++i) {
      execute();
      if (vmecpp_experiment::Compare(expected, actual) != 0) return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
      execute();
      if (vmecpp_experiment::Compare(expected, actual) != 0) return 2;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    differences = vmecpp_experiment::Compare(expected, actual);
#ifdef VMECPP_ITERATION_TAIL_REPLAY
    if (std::getenv("VMECPP_GPU_RESIDUAL_ROUNDS") != nullptr) {
      using Statistics = void (*)(std::array<unsigned long long, 4>&);
      auto statistics = reinterpret_cast<Statistics>(
          dlsym(iteration_library, "ReadGpuResidualRelaxationStats"));
      if (statistics == nullptr)
        throw std::runtime_error("Residual statistics missing");
      std::array<unsigned long long, 4> values{};
      statistics(values);
      std::printf(
          "{\"experiment\":\"optimistic "
          "residual\",\"executed_calls\":%llu,\"fallback_calls\":[%llu,%llu,%"
          "llu]}\n",
          values[0], values[1], values[2], values[3]);
    }
#endif
    std::printf(
        "{\"measurement\":\"executed resident force/iteration chain replay; "
        "includes validation downloads and comparisons, not solver "
        "timing\",\"different_values\":%zu,\"host_geometry_pointers\":\"null\","
        "\"repeats\":%d,\"warmups\":%d,\"mean_microseconds\":%.9g}\n",
        differences, kRepeats, kWarmups, seconds * 1e6 / kRepeats);
    return differences == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
