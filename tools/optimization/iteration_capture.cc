// SPDX-License-Identifier: MIT
// Reference-only interposition: original CPU methods always execute.
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "gpu_diagnostics.h"
#include "gpu_iteration.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace {
struct Capture {
  GpuIterationInput input_{};
  std::array<std::vector<double>, 17> inputs_;
  std::array<std::vector<double>, 20> outputs_;
  bool ready_ = false;
  bool done_ = false;
};
thread_local Capture capture;
thread_local std::array<std::vector<double>, 7> preparation_inputs;
thread_local bool zero_z_m1 = false;
thread_local const void* preparation_target = nullptr;
void* Original(const char* name) {
  static void* library = [] {
    const char* path = std::getenv("VMECPP_FORWARD_REFERENCE_SO");
    if (path == nullptr) throw std::runtime_error("Reference library required");
    void* result = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (result == nullptr) throw std::runtime_error("Reference not loaded");
    return result;
  }();
  void* result = dlsym(library, name);
  if (result == nullptr) throw std::runtime_error("Reference method missing");
  return result;
}
void Snapshot(const vmecpp::FourierForces& forces, int stage) {
  const std::array<std::span<const double>, 6> values{forces.frcc, forces.frss,
                                                      forces.fzsc, forces.fzcs,
                                                      forces.flsc, forces.flcs};
  for (int i = 0; i < 6; ++i) {
    capture.outputs_[stage * 6 + i].assign(values[i].begin(), values[i].end());
  }
}
std::vector<double> Residuals(const vmecpp::FourierForces& forces) {
  using Method =
      void (*)(const vmecpp::FourierForces*, std::vector<double>&, bool);
  static Method original = reinterpret_cast<Method>(
      Original("_ZNK6vmecpp13FourierForces9residualsERSt6vectorIdSaIdEEb"));
  std::vector<double> result(3);
  original(&forces, result, true);
  return result;
}
}  // namespace

namespace vmecpp {
void FourierCoeffs::decomposeInto(FourierCoeffs& m_x,
                                  const std::vector<double>& scalxc) const {
  if (std::getenv("VMECPP_PREPARATION_CAPTURE") != nullptr && !capture.done_ &&
      ns >= 99 && m_x.nsMax_ == ns - 1) {
    const std::array<std::span<const double>, 7> values{rcc, rss, zsc,   zcs,
                                                        lsc, lcs, scalxc};
    for (int i = 0; i < 7; ++i) {
      preparation_inputs[i].assign(values[i].begin(), values[i].end());
    }
    preparation_target = &m_x;
  }
  using Method = void (*)(const FourierCoeffs*, FourierCoeffs&,
                          const std::vector<double>&);
  static Method original = reinterpret_cast<Method>(Original(
      "_ZNK6vmecpp13FourierCoeffs13decomposeIntoERS0_RKSt6vectorIdSaIdEE"));
  original(this, m_x, scalxc);
}

void FourierForces::zeroZForceForM1() {
  zero_z_m1 = true;
  using Method = void (*)(FourierForces*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13FourierForces15zeroZForceForM1Ev"));
  original(this);
}

void IdealMhdModel::computeForceNorms(const FourierGeometry& decomposed_x) {
  using Method = void (*)(IdealMhdModel*, const FourierGeometry&);
  static Method original = reinterpret_cast<Method>(Original(
      "_ZN6vmecpp13IdealMhdModel17computeForceNormsERKNS_15FourierGeometryE"));
  original(this, decomposed_x);
  static thread_local bool done = false;
  const char* path = std::getenv("VMECPP_DIAGNOSTICS_CAPTURE");
  if (path == nullptr || done || m_fc_.ns < 99) return;
  if (m_fc_.lfreeb || s_.lasym || !s_.lthreed || r_.nsMinF != 0 ||
      r_.nsMaxF != m_fc_.ns - 1) {
    throw std::runtime_error("Diagnostics capture scope mismatch");
  }
  const std::array<std::span<const double>, 14> values{gsqrt,
                                                       bsupu,
                                                       bsupv,
                                                       bsubu,
                                                       bsubv,
                                                       s_.wInt,
                                                       m_p_.massH,
                                                       m_p_.dVdsH,
                                                       guu,
                                                       r12,
                                                       decomposed_x.rmncc,
                                                       decomposed_x.rmnss,
                                                       decomposed_x.zmnsc,
                                                       decomposed_x.zmncs};
  GpuDiagnosticsInput input{};
  input.dimensions_ = {
      m_fc_.ns,     s_.mpol, s_.ntor, r_.nsMaxF, r_.nsMaxFIncludingLcfs,
      s_.nThetaEff, s_.nZeta};
  input.scalars_ = {m_fc_.deltaS, adiabaticIndex, constants_.lamscale};
  for (int i = 0; i < 14; ++i) input.sizes_[i] = values[i].size();
  const std::array<double, 7> expected{m_h_.plasmaVolume,   m_h_.thermalEnergy,
                                       m_h_.magneticEnergy, m_h_.mhdEnergy,
                                       m_h_.fNormRZ,        m_h_.fNormL,
                                       m_h_.fNorm1};
  const std::array<std::size_t, 1> sizes{expected.size()};
  std::ofstream file(path, std::ios::binary);
  file.write("VMECGDG1", 8);
  file.write(reinterpret_cast<const char*>(input.dimensions_.data()),
             sizeof(input.dimensions_));
  file.write(reinterpret_cast<const char*>(input.scalars_.data()),
             sizeof(input.scalars_));
  file.write(reinterpret_cast<const char*>(input.sizes_.data()),
             sizeof(input.sizes_));
  file.write(reinterpret_cast<const char*>(sizes.data()), sizeof(sizes));
  for (const auto& array : values) {
    file.write(reinterpret_cast<const char*>(array.data()),
               array.size() * sizeof(double));
  }
  file.write(reinterpret_cast<const char*>(expected.data()), sizeof(expected));
  file.flush();
  if (!file) throw std::runtime_error("Diagnostics reference capture failed");
  done = true;
}

void IdealMhdModel::applyM1Preconditioner(FourierForces& m_decomposed_f) {
  const bool record = std::getenv("VMECPP_ITERATION_CAPTURE") != nullptr &&
                      !capture.done_ && m_fc_.ns >= 99;
  if (record) {
    if (m_fc_.lfreeb || s_.lasym || !s_.lthreed || r_.nsMinF != 0 ||
        r_.nsMaxF != m_fc_.ns - 1) {
      throw std::runtime_error("Iteration capture scope mismatch");
    }
    capture.input_.dimensions_ = {m_fc_.ns, s_.mpol, s_.ntor, r_.nsMaxF,
                                  r_.nsMaxFIncludingLcfs};
    const std::array<std::span<const double>, 17> values{m_decomposed_f.frcc,
                                                         m_decomposed_f.frss,
                                                         m_decomposed_f.fzsc,
                                                         m_decomposed_f.fzcs,
                                                         m_decomposed_f.flsc,
                                                         m_decomposed_f.flcs,
                                                         ard,
                                                         brd,
                                                         azd,
                                                         bzd,
                                                         ar,
                                                         az,
                                                         dr,
                                                         dz,
                                                         br,
                                                         bz,
                                                         lambdaPreconditioner};
    for (std::size_t i = 0; i < values.size(); ++i) {
      capture.inputs_[i].assign(values[i].begin(), values[i].end());
    }
    capture.outputs_[18] = Residuals(m_decomposed_f);
    capture.ready_ = true;
    const char* preparation_path = std::getenv("VMECPP_PREPARATION_CAPTURE");
    if (preparation_path != nullptr) {
      if (preparation_target != static_cast<FourierCoeffs*>(&m_decomposed_f)) {
        throw std::runtime_error(
            "Preparation capture does not match the preconditioner target");
      }
      GpuPreparationInput input{};
      input.dimensions_ = capture.input_.dimensions_;
      input.scalars_ = {1.0 / std::numbers::sqrt2, zero_z_m1 ? 1.0 : 0.0};
      for (int i = 0; i < 7; ++i)
        input.sizes_[i] = preparation_inputs[i].size();
      const std::array<std::span<const double>, 6> expected{
          m_decomposed_f.frcc, m_decomposed_f.frss, m_decomposed_f.fzsc,
          m_decomposed_f.fzcs, m_decomposed_f.flsc, m_decomposed_f.flcs};
      std::array<std::size_t, 6> sizes{};
      for (int i = 0; i < 6; ++i) sizes[i] = expected[i].size();
      std::ofstream file(preparation_path, std::ios::binary);
      file.write("VMECGPR1", 8);
      file.write(reinterpret_cast<const char*>(input.dimensions_.data()),
                 sizeof(input.dimensions_));
      file.write(reinterpret_cast<const char*>(input.scalars_.data()),
                 sizeof(input.scalars_));
      file.write(reinterpret_cast<const char*>(input.sizes_.data()),
                 sizeof(input.sizes_));
      file.write(reinterpret_cast<const char*>(sizes.data()), sizeof(sizes));
      for (const auto& values : preparation_inputs) {
        file.write(reinterpret_cast<const char*>(values.data()),
                   values.size() * sizeof(double));
      }
      for (const auto& values : expected) {
        file.write(reinterpret_cast<const char*>(values.data()),
                   values.size() * sizeof(double));
      }
      file.flush();
      if (!file)
        throw std::runtime_error("Preparation reference capture failed");
    }
  }
  using Method = void (*)(IdealMhdModel*, FourierForces&);
  static Method original = reinterpret_cast<Method>(Original(
      "_ZN6vmecpp13IdealMhdModel21applyM1PreconditionerERNS_13FourierForcesE"));
  original(this, m_decomposed_f);
  if (record) Snapshot(m_decomposed_f, 0);
  zero_z_m1 = false;
}

absl::Status IdealMhdModel::applyRZPreconditioner(
    FourierForces& m_decomposed_f) {
  if (capture.ready_ && !capture.done_) {
    const std::array<const std::vector<double>*, 6> matrices{&ar, &az, &dr,
                                                             &dz, &br, &bz};
    for (int i = 0; i < 6; ++i) capture.inputs_[10 + i] = *matrices[i];
  }
  using Method = absl::Status (*)(IdealMhdModel*, FourierForces&);
  static Method original = reinterpret_cast<Method>(Original(
      "_ZN6vmecpp13IdealMhdModel21applyRZPreconditionerERNS_13FourierForcesE"));
  absl::Status result = original(this, m_decomposed_f);
  if (capture.ready_ && !capture.done_) Snapshot(m_decomposed_f, 1);
  return result;
}

void IdealMhdModel::applyLambdaPreconditioner(FourierForces& m_decomposed_f) {
  using Method = void (*)(IdealMhdModel*, FourierForces&);
  static Method original =
      reinterpret_cast<Method>(Original("_ZN6vmecpp13IdealMhdModel25applyLambda"
                                        "PreconditionerERNS_13FourierForcesE"));
  original(this, m_decomposed_f);
  if (!capture.ready_ || capture.done_) return;
  Snapshot(m_decomposed_f, 2);
  capture.outputs_[19] = Residuals(m_decomposed_f);
  GpuIterationOutput output{};
  for (std::size_t i = 0; i < capture.inputs_.size(); ++i) {
    capture.input_.sizes_[i] = capture.inputs_[i].size();
  }
  for (std::size_t i = 0; i < capture.outputs_.size(); ++i) {
    output.sizes_[i] = capture.outputs_[i].size();
  }
  std::ofstream file(std::getenv("VMECPP_ITERATION_CAPTURE"), std::ios::binary);
  file.write("VMECGIT1", 8);
  file.write(reinterpret_cast<const char*>(capture.input_.dimensions_.data()),
             sizeof(capture.input_.dimensions_));
  file.write(reinterpret_cast<const char*>(capture.input_.sizes_.data()),
             sizeof(capture.input_.sizes_));
  file.write(reinterpret_cast<const char*>(output.sizes_.data()),
             sizeof(output.sizes_));
  for (const auto& values : capture.inputs_) {
    file.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(double));
  }
  for (const auto& values : capture.outputs_) {
    file.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(double));
  }
  file.flush();
  if (!file) throw std::runtime_error("Iteration reference capture failed");
  capture.done_ = true;
}
}  // namespace vmecpp

namespace vmecpp {
void Vmec::PerformTimeStep(double fac, double b1, double time_step,
                           int thread_id) {
  static thread_local bool done = false;
  const char* path = std::getenv("VMECPP_STEP_CAPTURE");
  const bool record = path != nullptr && !done && fc_.ns >= 99;
  GpuStepInput input{};
  std::array<std::vector<double>, 18> values;
  const auto copy_values = [&] {
    const auto& x = *decomposed_x_[thread_id];
    const auto& v = *decomposed_v_[thread_id];
    const auto& f = *decomposed_f_[thread_id];
    const std::array<std::span<const double>, 18> arrays{
        x.rmncc, x.rmnss, x.zmnsc, x.zmncs, x.lmnsc, x.lmncs,
        v.vrcc,  v.vrss,  v.vzsc,  v.vzcs,  v.vlsc,  v.vlcs,
        f.frcc,  f.frss,  f.fzsc,  f.fzcs,  f.flsc,  f.flcs};
    for (std::size_t i = 0; i < arrays.size(); ++i) {
      values[i].assign(arrays[i].begin(), arrays[i].end());
    }
  };
  if (record) {
    if (thread_id != 0 || r_.size() != 1 || s_.lasym || !s_.lthreed) {
      throw std::runtime_error("Step capture requires one symmetric partition");
    }
    input.dimensions_ = {fc_.ns, s_.mpol, s_.ntor, r_[0]->nsMaxF,
                         r_[0]->nsMaxFIncludingLcfs};
    input.scalars_ = {fac, b1, time_step};
    copy_values();
    for (std::size_t i = 0; i < values.size(); ++i)
      input.sizes_[i] = values[i].size();
  }
  using Method = void (*)(Vmec*, double, double, double, int);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp4Vmec15PerformTimeStepEdddi"));
  original(this, fac, b1, time_step, thread_id);
  if (record) {
    std::ofstream file(path, std::ios::binary);
    file.write("VMECGST1", 8);
    file.write(reinterpret_cast<const char*>(input.dimensions_.data()),
               sizeof(input.dimensions_));
    file.write(reinterpret_cast<const char*>(input.scalars_.data()),
               sizeof(input.scalars_));
    file.write(reinterpret_cast<const char*>(input.sizes_.data()),
               sizeof(input.sizes_));
    std::array<std::size_t, 12> sizes{};
    std::copy_n(input.sizes_.begin(), 12, sizes.begin());
    file.write(reinterpret_cast<const char*>(sizes.data()), sizeof(sizes));
    for (const auto& array : values) {
      file.write(reinterpret_cast<const char*>(array.data()),
                 array.size() * sizeof(double));
    }
    copy_values();
    for (int i = 0; i < 12; ++i) {
      file.write(reinterpret_cast<const char*>(values[i].data()),
                 values[i].size() * sizeof(double));
    }
    file.flush();
    if (!file) throw std::runtime_error("Step reference capture failed");
    done = true;
  }
}
}  // namespace vmecpp
