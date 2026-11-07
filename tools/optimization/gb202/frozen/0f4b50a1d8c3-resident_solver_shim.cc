// SPDX-License-Identifier: MIT
// Experimental persistent GPU body with the original CPU solver controller.
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <vector>

#include "gpu_solver.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace {
void* Original(const char* name) {
  static void* library = [] {
    const char* path = std::getenv("VMECPP_FORWARD_REFERENCE_SO");
    if (path == nullptr)
      throw std::runtime_error("Reference extension required");
    void* result = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (result == nullptr) throw std::runtime_error("Reference not loaded");
    return result;
  }();
  void* result = dlsym(library, name);
  if (result == nullptr) throw std::runtime_error("Reference method missing");
  return result;
}

struct Binding {
  vmecpp::Vmec* owner_ = nullptr;
  const vmecpp::IdealMhdModel* model_ = nullptr;
  vmecpp::FourierGeometry* geometry_ = nullptr;
  vmecpp::FourierGeometry* backup_ = nullptr;
  vmecpp::FourierVelocity* velocity_ = nullptr;
  vmecpp::FourierForces* forces_ = nullptr;
  void* context_ = nullptr;
  int ns_ = 0;
  bool gpu_update_ = false;
  bool resident_control_ = false;
  std::array<std::vector<double>, 12> mirrors_;
  std::function<absl::StatusOr<bool>()> materialize_;
  ~Binding() { Reset(); }
  void Reset() {
    if (context_ != nullptr) DestroyGpuSolver(context_);
    context_ = nullptr;
    gpu_update_ = false;
    resident_control_ = false;
    materialize_ = {};
  }
};
thread_local Binding binding;

std::array<std::span<double>, 6> Geometry(vmecpp::FourierGeometry& m_x) {
  return {m_x.rmncc, m_x.rmnss, m_x.zmnsc, m_x.zmncs, m_x.lmnsc, m_x.lmncs};
}
std::array<std::span<double>, 6> Velocity(vmecpp::FourierVelocity& m_v) {
  return {m_v.vrcc, m_v.vrss, m_v.vzsc, m_v.vzcs, m_v.vlsc, m_v.vlcs};
}
std::array<std::span<double>, 6> Forces(vmecpp::FourierForces& m_f) {
  return {m_f.frcc, m_f.frss, m_f.fzsc, m_f.fzcs, m_f.flsc, m_f.flcs};
}
GpuStepInput State(vmecpp::FourierGeometry& m_x, vmecpp::FourierVelocity& m_v,
                   vmecpp::FourierForces& m_f,
                   const std::array<int, 5>& dimensions) {
  GpuStepInput result{};
  result.dimensions_ = dimensions;
  const std::array<std::array<std::span<double>, 6>, 3> groups{
      Geometry(m_x), Velocity(m_v), Forces(m_f)};
  for (int group = 0; group < 3; ++group) {
    for (int i = 0; i < 6; ++i) {
      result.data_[group * 6 + i] = groups[group][i].data();
      result.sizes_[group * 6 + i] = groups[group][i].size();
    }
  }
  return result;
}
void SaveMirrors(const GpuStepInput& state) {
  for (int i = 0; i < 12; ++i) {
    binding.mirrors_[i].assign(state.data_[i],
                               state.data_[i] + state.sizes_[i]);
  }
}
bool Changed(const GpuStepInput& state) {
  for (int i = 0; i < 12; ++i) {
    if (binding.mirrors_[i].size() != state.sizes_[i] ||
        std::memcmp(binding.mirrors_[i].data(), state.data_[i],
                    state.sizes_[i] * sizeof(double)) != 0)
      return true;
  }
  return false;
}
GpuPreparationOutput GeometryOutput(
    const std::array<std::span<double>, 6>& values) {
  GpuPreparationOutput output{};
  for (int i = 0; i < 6; ++i) {
    output.data_[i] = values[i].data();
    output.sizes_[i] = values[i].size();
  }
  return output;
}
absl::Status Flush() {
  if (binding.context_ == nullptr) return absl::OkStatus();
  if (binding.materialize_) {
    const auto result = binding.materialize_();
    if (!result.ok()) return result.status();
  }
  if (binding.resident_control_) {
    const auto state =
        State(*binding.geometry_, *binding.velocity_, *binding.forces_, {});
    GpuStepOutput output{};
    for (int i = 0; i < 12; ++i) {
      output.data_[i] = const_cast<double*>(state.data_[i]);
      output.sizes_[i] = state.sizes_[i];
    }
    DownloadGpuSolverState(binding.context_, output);
    auto forces = GeometryOutput(Forces(*binding.forces_));
    DownloadGpuSolverForces(binding.context_, forces);
    auto backup = GeometryOutput(Geometry(*binding.backup_));
    DownloadGpuSolverGeometry(binding.context_, GpuSolverGeometry::BACKUP,
                              backup);
  }
  binding.Reset();
  return absl::OkStatus();
}
void Trace(const vmecpp::FlowControl& fc, int iter1, int iter2, bool gpu,
           bool refresh, bool bad, const std::array<double, 7>& diagnostics) {
  const char* path = std::getenv("VMECPP_GPU_SOLVER_TRACE");
  if (path == nullptr) return;
  std::ofstream file(path, std::ios::app);
  file << std::setprecision(17) << "{\"ns\":" << fc.ns << ",\"iter1\":" << iter1
       << ",\"iter2\":" << iter2 << ",\"gpu\":" << gpu
       << ",\"refresh\":" << refresh << ",\"bad_jacobian\":" << bad
       << ",\"residuals\":[" << fc.fsqr << ',' << fc.fsqz << ',' << fc.fsql
       << ',' << fc.fsqr1 << ',' << fc.fsqz1 << ',' << fc.fsql1
       << "],\"invariant\":[" << fc.fResInvar[0] << ',' << fc.fResInvar[1]
       << ',' << fc.fResInvar[2] << "],\"preconditioned\":[" << fc.fResPrecd[0]
       << ',' << fc.fResPrecd[1] << ',' << fc.fResPrecd[2]
       << "],\"diagnostics\":[";
  for (int i = 0; i < 7; ++i) {
    if (i != 0) file << ',';
    file << diagnostics[i];
  }
  file << "]}\n";
}
}  // namespace

namespace vmecpp {
absl::StatusOr<bool> IdealMhdModel::update(
    FourierGeometry& m_decomposed_x, FourierGeometry& m_physical_x,
    FourierForces& m_decomposed_f, FourierForces& m_physical_f,
    bool& m_need_restart, int& m_last_preconditioner_update,
    int& m_last_full_update_nestor, FlowControl& m_fc, const int iter1,
    const int iter2, const VmecCheckpoint& checkpoint,
    const int iterations_before_checkpointing, bool verbose) {
  using Method = absl::StatusOr<bool> (*)(
      IdealMhdModel*, FourierGeometry&, FourierGeometry&, FourierForces&,
      FourierForces&, bool&, int&, int&, FlowControl&, int, int,
      const VmecCheckpoint&, int, bool);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel6updateERNS_15FourierGeometryES2_RNS_"
               "13FourierForcesES4_RbRiS6_RNS_11FlowControlEiiRKNS_"
               "14VmecCheckpointEib"));
  auto cpu_update = [&] {
    return original(this, m_decomposed_x, m_physical_x, m_decomposed_f,
                    m_physical_f, m_need_restart, m_last_preconditioner_update,
                    m_last_full_update_nestor, m_fc, iter1, iter2, checkpoint,
                    iterations_before_checkpointing, verbose);
  };
  auto diagnostics = [&] {
    return std::array<double, 7>{m_h_.plasmaVolume,   m_h_.thermalEnergy,
                                 m_h_.magneticEnergy, m_h_.mhdEnergy,
                                 m_h_.fNormRZ,        m_h_.fNormL,
                                 m_h_.fNorm1};
  };
  const bool enabled = std::getenv("VMECPP_GPU_SOLVER") != nullptr;
  const bool supported =
      !m_fc.lfreeb && !s_.lasym && s_.lthreed && ncurr == 1 &&
      adiabaticIndex == 0.0 && m_fc.max_threads() == 1 && r_.nsMinF == 0 &&
      r_.nsMinF1 == 0 && r_.nsMinH == 0 && r_.nsMaxF == m_fc.ns - 1 &&
      checkpoint == VmecCheckpoint::NONE;
  if (!enabled || !supported || iter2 == iter1 || binding.model_ != this ||
      binding.velocity_ == nullptr || binding.ns_ != m_fc.ns) {
    if (binding.context_ != nullptr && binding.resident_control_) {
      const auto status = Flush();
      if (!status.ok()) return status;
    }
    binding.Reset();
    const auto result = cpu_update();
    Trace(m_fc, iter1, iter2, false,
          (iter2 - iter1) % FlowControl::kPreconditionerUpdateInterval == 0,
          m_fc.restart_reason == RestartReason::BAD_JACOBIAN, diagnostics());
    return result;
  }
  const std::array<int, 5> dimensions{m_fc.ns, s_.mpol, s_.ntor, r_.nsMaxF,
                                      r_.nsMaxFIncludingLcfs};
  GpuStepInput state =
      State(m_decomposed_x, *binding.velocity_, m_decomposed_f, dimensions);
  const std::array<std::span<double>, 20> cache{lambdaPreconditioner,
                                                bLambda,
                                                dLambda,
                                                cLambda,
                                                arm,
                                                ard,
                                                brm,
                                                brd,
                                                azm,
                                                azd,
                                                bzm,
                                                bzd,
                                                cxd,
                                                tcon,
                                                ar,
                                                az,
                                                dr,
                                                dz,
                                                br,
                                                bz};
  if (binding.context_ == nullptr) {
    GpuSolverSetup setup{};
    setup.state_ = state;
    setup.scalxc_ = m_p_.scalxc.data();
    setup.scalxc_size_ = m_p_.scalxc.size();
    setup.cached_diagnostics_ = diagnostics();
    setup.cached_dvds_ = m_p_.dVdsH.data();
    const char* resident_control =
        std::getenv("VMECPP_GPU_SOLVER_RESIDENT_CONTROL");
    binding.resident_control_ =
        resident_control != nullptr && std::strcmp(resident_control, "1") == 0;
    if (binding.resident_control_) {
      const auto backup = Geometry(*binding.backup_);
      for (int i = 0; i < 6; ++i) {
        setup.backup_.data_[i] = backup[i].data();
        setup.backup_.sizes_[i] = backup[i].size();
      }
    }
    const std::array<int, 13> dims{
        r_.nsMinF, r_.nsMaxF, r_.nsMinF1, r_.nsMaxF1,   r_.nsMaxFIncludingLcfs,
        s_.mpol,   s_.ntor,   s_.nZeta,   s_.nThetaEff, s_.nThetaReduced,
        s_.nnyq2,  m_fc.ns,   0};
    setup.basis_.dimensions_ = dims;
    const std::array<std::span<const double>, 10> basis{
        xmpq,      m_p_.sqrtSF, t_.cosmu, t_.sinmu,  t_.cosmum,
        t_.sinmum, t_.cosnv,    t_.sinnv, t_.cosnvn, t_.sinnvn};
    for (int i = 0; i < 10; ++i) {
      setup.basis_.data_[6 + i] = basis[i].data();
      setup.basis_.sizes_[6 + i] = basis[i].size();
    }
    setup.fields_.dimensions_ = dims;
    setup.fields_.scalars_ = {m_fc.deltaS, dSHalfDsInterp, constants_.lamscale,
                              static_cast<double>(signOfJacobian)};
    const std::array<std::span<const double>, 28> fields{
        r1_e,       r1_o,       ru_e,       ru_o,        rv_e,
        rv_o,       z1_e,       z1_o,       zu_e,        zu_o,
        zv_e,       zv_o,       lu_e,       lu_o,        lv_e,
        lv_o,       rCon,       zCon,       m_p_.sqrtSF, m_p_.sqrtSH,
        s_.wInt,    m_p_.phipF, m_p_.phipH, m_p_.currH,  m_p_.chipH,
        m_p_.iotaH, m_p_.chipF, m_p_.iotaF};
    for (int i = 0; i < 28; ++i) {
      setup.fields_.data_[i] = fields[i].data();
      setup.fields_.sizes_[i] = fields[i].size();
    }
    setup.forces_ = {{m_p_.massH.data(), m_p_.radialBlending.data()},
                     {m_p_.massH.size(), m_p_.radialBlending.size()},
                     adiabaticIndex};
    const std::array<std::span<const double>, 15> constraints{
        rCon0,     zCon0,     faccon,     tcon,       xmpq,
        t_.cosmui, t_.sinmui, t_.cosmumi, t_.sinmumi, t_.cosnv,
        t_.sinnv,  t_.cosnvn, t_.sinnvn,  t_.cosmu,   t_.sinmu};
    for (int i = 0; i < 15; ++i) {
      setup.constraints_.data_[i] = constraints[i].data();
      setup.constraints_.sizes_[i] = constraints[i].size();
    }
    std::vector<double> powers(m_fc.ns * s_.mpol);
    for (int j = 0; j < m_fc.ns; ++j) {
      for (int m = 0; m < s_.mpol; ++m) {
        const double pwr = std::min(m * m / (16.0 * 16.0), 8.0);
        powers[j * s_.mpol + m] = std::pow(m_p_.sqrtSF[j], pwr);
      }
    }
    const std::array<std::span<const double>, 26> preconditioner{
        r12,     totalPressure, tau,     bsupv,       gsqrt,
        guu,     guv,           gvv,     zs,          zu12,
        zu_e,    zu_o,          z1_o,    rs,          ru12,
        ru_e,    ru_o,          r1_o,    m_p_.sqrtSH, m_p_.sqrtSF,
        m_p_.sm, m_p_.sp,       s_.wInt, ruFull,      zuFull,
        powers};
    setup.preconditioner_.dimensions_ = {
        m_fc.ns,      s_.mpol,  s_.ntor, r_.nsMaxF, r_.nsMaxFIncludingLcfs,
        s_.nThetaEff, s_.nZeta, s_.nfp};
    setup.preconditioner_.scalars_ = {m_fc.deltaS, constants_.lamscale,
                                      dampingFactor, tcon0};
    for (int i = 0; i < 26; ++i) {
      setup.preconditioner_.data_[i] = preconditioner[i].data();
      setup.preconditioner_.sizes_[i] = preconditioner[i].size();
    }
    for (int i = 0; i < 20; ++i) {
      setup.cached_preconditioner_.data_[i] = cache[i].data();
      setup.cached_preconditioner_.sizes_[i] = cache[i].size();
    }
    binding.context_ = CreateGpuSolver(setup);
    if (!binding.resident_control_) SaveMirrors(state);
  } else if (!binding.resident_control_ && Changed(state)) {
    UploadGpuSolverState(binding.context_, state);
    SaveMirrors(state);
  }
  const double old_fsqr = m_fc.fsqr;
  const double old_fsqz = m_fc.fsqz;
  std::array<std::vector<double>, 6> update_geometry;
  const auto geometry = Geometry(m_decomposed_x);
  if (!binding.resident_control_) {
    for (int i = 0; i < 6; ++i) {
      update_geometry[i].assign(geometry[i].begin(), geometry[i].end());
    }
  }
  // At an iteration limit the original physical arrays describe the last
  // pre-step geometry, while decomposed_x already describes the post-step
  // geometry. Keep that distinction when materializing a nonconverged stage.
  auto materialize = [this, geometry, cache,
                      update_geometry = std::move(update_geometry), old_fsqr,
                      old_fsqz, &m_decomposed_x, &m_physical_x, &m_decomposed_f,
                      &m_physical_f, &m_last_preconditioner_update,
                      &m_last_full_update_nestor, &m_fc, iter1, iter2,
                      checkpoint, iterations_before_checkpointing,
                      verbose](GpuSolverGeometry source) {
    std::array<std::vector<double>, 6> current_geometry;
    for (int i = 0; i < 6; ++i) {
      current_geometry[i].assign(geometry[i].begin(), geometry[i].end());
      if (!binding.resident_control_) {
        std::copy(update_geometry[i].begin(), update_geometry[i].end(),
                  geometry[i].begin());
      }
    }
    if (binding.resident_control_) {
      auto output = GeometryOutput(geometry);
      DownloadGpuSolverGeometry(binding.context_, source, output);
    }
    const std::array<double, 6> current_residuals{
        m_fc.fsqr, m_fc.fsqz, m_fc.fsql, m_fc.fsqr1, m_fc.fsqz1, m_fc.fsql1};
    const auto current_invariant = m_fc.fResInvar;
    const auto current_preconditioned = m_fc.fResPrecd;
    const auto current_restart_reason = m_fc.restart_reason;
    const int current_last_preconditioner_update = m_last_preconditioner_update;
    const int current_last_full_update_nestor = m_last_full_update_nestor;
    GpuPreconditionerOutput output{};
    for (int i = 0; i < 20; ++i) {
      output.data_[i] = cache[i].data();
      output.sizes_[i] = cache[i].size();
    }
    DownloadGpuSolverCache(binding.context_, output);
    m_fc.fsqr = old_fsqr;
    m_fc.fsqz = old_fsqz;
    bool materialized_need_restart = false;
    const auto result = original(
        this, m_decomposed_x, m_physical_x, m_decomposed_f, m_physical_f,
        materialized_need_restart, m_last_preconditioner_update,
        m_last_full_update_nestor, m_fc, iter1, iter2, checkpoint,
        iterations_before_checkpointing, verbose);
    for (int i = 0; i < 6; ++i) {
      std::copy(current_geometry[i].begin(), current_geometry[i].end(),
                geometry[i].begin());
    }
    m_fc.fsqr = current_residuals[0];
    m_fc.fsqz = current_residuals[1];
    m_fc.fsql = current_residuals[2];
    m_fc.fsqr1 = current_residuals[3];
    m_fc.fsqz1 = current_residuals[4];
    m_fc.fsql1 = current_residuals[5];
    m_fc.fResInvar = current_invariant;
    m_fc.fResPrecd = current_preconditioned;
    m_fc.restart_reason = current_restart_reason;
    m_last_preconditioner_update = current_last_preconditioner_update;
    m_last_full_update_nestor = current_last_full_update_nestor;
    return result;
  };
  const bool refresh =
      (iter2 - iter1) % FlowControl::kPreconditionerUpdateInterval == 0;
  const GpuSolverControl control{
      refresh, (iter2 - iter1) < 50 && ((old_fsqr + old_fsqz) < 1.0e-6),
      old_fsqz < 1.0e-6 || iter2 < 2};
  GpuSolverScalars scalars{};
  RunGpuSolver(binding.context_, control, scalars);
  binding.gpu_update_ = true;
  if (scalars.bad_jacobian_) {
    // Bad Jacobians overwrite geometry and Jacobian ingredients only. First
    // materialize the previous valid update, retaining its other fields.
    const auto previous_materialize = binding.materialize_;
    binding.materialize_ = [previous_materialize, materialize]() {
      if (previous_materialize) {
        const auto previous = previous_materialize();
        if (!previous.ok()) return previous;
      }
      return materialize(GpuSolverGeometry::LAST_UPDATE);
    };
    m_fc.restart_reason = RestartReason::BAD_JACOBIAN;
    Trace(m_fc, iter1, iter2, true, refresh, true, diagnostics());
    return false;
  }
  binding.materialize_ = [materialize = std::move(materialize)] {
    return materialize(GpuSolverGeometry::LAST_VALID_UPDATE);
  };
  if (scalars.invalid_norm_ != 0) {
    return absl::InternalError("GPU preconditioner normalization failed");
  }
  m_h_.plasmaVolume = scalars.diagnostics_[0];
  m_h_.thermalEnergy = scalars.diagnostics_[1];
  m_h_.magneticEnergy = scalars.diagnostics_[2];
  m_h_.mhdEnergy = scalars.diagnostics_[3];
  m_h_.fNormRZ = scalars.diagnostics_[4];
  m_h_.fNormL = scalars.diagnostics_[5];
  m_h_.fNorm1 = scalars.diagnostics_[6];
  evalFResInvar(std::vector<double>(scalars.invariant_residuals_.begin(),
                                    scalars.invariant_residuals_.end()));
  evalFResPrecd(std::vector<double>(scalars.preconditioned_residuals_.begin(),
                                    scalars.preconditioned_residuals_.end()));
  if (refresh) m_last_preconditioner_update = iter2;
  Trace(m_fc, iter1, iter2, true, refresh, false, diagnostics());
  if (m_fc.fsqr <= m_fc.ftolv && m_fc.fsqz <= m_fc.ftolv &&
      m_fc.fsql <= m_fc.ftolv) {
    if (binding.resident_control_) {
      const auto status = Flush();
      if (!status.ok()) return status;
      m_need_restart = false;
      return false;
    }
    GpuPreconditionerOutput output{};
    for (int i = 0; i < 20; ++i) {
      output.data_[i] = cache[i].data();
      output.sizes_[i] = cache[i].size();
    }
    DownloadGpuSolverCache(binding.context_, output);
    // Materialize all CPU output fields at the same state without adding a
    // controller iteration or history entry. Preserve the pre-update force
    // decision inputs so the original branch order remains unchanged.
    m_fc.fsqr = old_fsqr;
    m_fc.fsqz = old_fsqz;
    const auto result = cpu_update();
    binding.Reset();
    return result;
  }
  m_need_restart = false;
  return false;
}

absl::StatusOr<bool> Vmec::SolveEquilibrium(
    VmecCheckpoint checkpoint, int iterations_before_checkpointing) {
  using Method = absl::StatusOr<bool> (*)(Vmec*, VmecCheckpoint, int);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp4Vmec16SolveEquilibriumENS_14VmecCheckpointEi"));
  const auto result =
      original(this, checkpoint, iterations_before_checkpointing);
  if (result.ok() && !*result && binding.context_ != nullptr &&
      binding.model_ == m_[0].get() && binding.materialize_) {
    const auto status = Flush();
    if (!status.ok()) return status;
  }
  return result;
}

absl::StatusOr<Vmec::SolveEqLoopStatus> Vmec::SolveEquilibriumLoop(
    int thread_id, int iterations_before_checkpointing,
    VmecCheckpoint checkpoint, bool& m_lreset_internal, bool& m_liter_flag) {
  using Method = absl::StatusOr<SolveEqLoopStatus> (*)(
      Vmec*, int, int, VmecCheckpoint, bool&, bool&);
  static Method original = reinterpret_cast<Method>(Original(
      "_ZN6vmecpp4Vmec20SolveEquilibriumLoopEiiNS_14VmecCheckpointERbS2_"));
  const auto result = original(this, thread_id, iterations_before_checkpointing,
                               checkpoint, m_lreset_internal, m_liter_flag);
  // Every return can precede host multigrid reads or an axis-guess retry that
  // writes the CPU coefficients directly. Return full ownership before either.
  if (binding.context_ != nullptr && binding.resident_control_ &&
      binding.owner_ == this) {
    const auto status = Flush();
    if (result.ok() && !status.ok()) return status;
  }
  return result;
}

void Vmec::RestartIteration(double& m_delt0r, int thread_id) {
  using Method = void (*)(Vmec*, double&, int);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp4Vmec16RestartIterationERdi"));
  if (binding.context_ == nullptr || !binding.resident_control_ ||
      binding.owner_ != this) {
    original(this, m_delt0r, thread_id);
    return;
  }
  if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    RestoreGpuSolverBackup(binding.context_);
    m_delt0r = m_delt0r * 0.9;
    fc_.ijacob = fc_.ijacob + 1;
    iter1_ = iter2_;
    fc_.restart_reason = RestartReason::NO_RESTART;
  } else if (fc_.restart_reason == RestartReason::BAD_PROGRESS) {
    RestoreGpuSolverBackup(binding.context_);
    m_delt0r = m_delt0r / 1.03;
    fc_.restart_reason = RestartReason::NO_RESTART;
  } else {
    SaveGpuSolverBackup(binding.context_);
  }
}

void FourierGeometry::ComputeSpectralWidth(
    const FourierBasisFastPoloidal& fourier_basis,
    RadialProfiles& m_radial_profiles, int p, int q) const {
  using Method =
      void (*)(const FourierGeometry*, const FourierBasisFastPoloidal&,
               RadialProfiles&, int, int);
  static Method original = reinterpret_cast<Method>(
      Original("_ZNK6vmecpp15FourierGeometry20ComputeSpectralWidthERKNS_"
               "24FourierBasisFastPoloidalERNS_14RadialProfilesEii"));
  if (binding.context_ != nullptr && binding.resident_control_ &&
      binding.backup_ == this) {
    auto output = GeometryOutput(Geometry(*binding.backup_));
    DownloadGpuSolverGeometry(binding.context_, GpuSolverGeometry::BACKUP,
                              output);
  }
  original(this, fourier_basis, m_radial_profiles, p, q);
}

void Vmec::Printout(double delt0r, int thread_id, int iter2) {
  using Method = void (*)(Vmec*, double, int, int);
  static Method original =
      reinterpret_cast<Method>(Original("_ZN6vmecpp4Vmec8PrintoutEdii"));
  if (binding.context_ != nullptr && binding.resident_control_ &&
      binding.owner_ == this) {
    std::array<double, 2> axis{};
    DownloadGpuSolverPrintout(binding.context_, p_[thread_id]->dVdsH.data(),
                              p_[thread_id]->dVdsH.size(), axis);
    h_.SetGeometricOffset({axis[0], axis[1]});
  }
  original(this, delt0r, thread_id, iter2);
}

void Vmec::PerformTimeStep(double fac, double b1, double time_step,
                           int thread_id) {
  using Method = void (*)(Vmec*, double, double, double, int);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp4Vmec15PerformTimeStepEdddi"));
  if (binding.context_ != nullptr && binding.gpu_update_ &&
      binding.model_ == m_[thread_id].get()) {
    AdvanceGpuSolver(binding.context_, {fac, b1, time_step});
    if (!binding.resident_control_) {
      auto state = State(*decomposed_x_[thread_id], *decomposed_v_[thread_id],
                         *decomposed_f_[thread_id], {});
      GpuStepOutput output{};
      for (int i = 0; i < 12; ++i) {
        output.data_[i] = const_cast<double*>(state.data_[i]);
        output.sizes_[i] = state.sizes_[i];
      }
      DownloadGpuSolverState(binding.context_, output);
      SaveMirrors(state);
    }
  } else {
    original(this, fac, b1, time_step, thread_id);
  }
  binding.owner_ = this;
  binding.model_ = m_[thread_id].get();
  binding.geometry_ = decomposed_x_[thread_id].get();
  binding.backup_ = physical_x_backup_[thread_id].get();
  binding.velocity_ = decomposed_v_[thread_id].get();
  binding.forces_ = decomposed_f_[thread_id].get();
  binding.ns_ = fc_.ns;
  binding.gpu_update_ = false;
}
}  // namespace vmecpp
