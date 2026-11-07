// SPDX-License-Identifier: MIT
// Reference-only capture of actual preconditioner refresh and assembly.
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <vector>

#include "gpu_preconditioner.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

namespace {
struct PreconditionerCapture {
  GpuPreconditionerInput input_{};
  std::array<std::vector<double>, 26> values_;
  std::vector<double> tcon_;
  bool ready_ = false;
  bool done_ = false;
};
thread_local PreconditionerCapture preconditioner_capture;
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
  if (result == nullptr)
    throw std::runtime_error("Preconditioner reference method missing");
  return result;
}
}  // namespace

namespace vmecpp {
void IdealMhdModel::updateRadialPreconditioner() {
  auto& capture = preconditioner_capture;
  const bool record = std::getenv("VMECPP_PRECONDITIONER_CAPTURE") != nullptr &&
                      !capture.done_ && m_fc_.ns >= 99;
  if (record) {
    if (m_fc_.lfreeb || s_.lasym || !s_.lthreed || r_.nsMinF != 0 ||
        r_.nsMinF1 != 0 || r_.nsMinH != 0 || r_.nsMaxF != m_fc_.ns - 1) {
      throw std::runtime_error("Preconditioner capture scope mismatch");
    }
    std::vector<double> powers(m_fc_.ns * s_.mpol);
    for (int j = 0; j < m_fc_.ns; ++j) {
      for (int m = 0; m < s_.mpol; ++m) {
        const int tmm = m * m;
        const double pwr = std::min(tmm / (16.0 * 16.0), 8.0);
        powers[j * s_.mpol + m] = std::pow(m_p_.sqrtSF[j], pwr);
      }
    }
    // These powers depend only on the fixed radial grid and poloidal mode.
    // Cache the original host-libm values once at grid setup; no changed pow
    // implementation or approximate device transcendental is substituted.
    const std::array<std::span<const double>, 26> values{
        r12,     totalPressure, tau,     bsupv,       gsqrt,
        guu,     guv,           gvv,     zs,          zu12,
        zu_e,    zu_o,          z1_o,    rs,          ru12,
        ru_e,    ru_o,          r1_o,    m_p_.sqrtSH, m_p_.sqrtSF,
        m_p_.sm, m_p_.sp,       s_.wInt, ruFull,      zuFull,
        powers};
    capture.input_.dimensions_ = {
        m_fc_.ns,     s_.mpol,  s_.ntor, r_.nsMaxF, r_.nsMaxFIncludingLcfs,
        s_.nThetaEff, s_.nZeta, s_.nfp};
    capture.input_.scalars_ = {m_fc_.deltaS, constants_.lamscale, dampingFactor,
                               tcon0};
    for (int i = 0; i < 26; ++i) {
      capture.values_[i].assign(values[i].begin(), values[i].end());
      capture.input_.sizes_[i] = values[i].size();
    }
    capture.ready_ = true;
  }
  using Method = void (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel26updateRadialPreconditionerEv"));
  original(this);
}

absl::Status IdealMhdModel::constraintForceMultiplier() {
  using Method = absl::Status (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel25constraintForceMultiplierEv"));
  absl::Status status = original(this);
  auto& capture = preconditioner_capture;
  if (capture.ready_ && !capture.done_ && status.ok()) capture.tcon_ = tcon;
  return status;
}

void IdealMhdModel::assembleRZPreconditioner() {
  using Method = void (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel24assembleRZPreconditionerEv"));
  original(this);
  auto& capture = preconditioner_capture;
  if (!capture.ready_ || capture.done_) return;
  if (capture.tcon_.size() != tcon.size()) {
    throw std::runtime_error("Constraint multiplier capture not ready");
  }
  const std::array<std::span<const double>, 20> values{lambdaPreconditioner,
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
                                                       capture.tcon_,
                                                       ar,
                                                       az,
                                                       dr,
                                                       dz,
                                                       br,
                                                       bz};
  GpuPreconditionerOutput output{};
  for (int i = 0; i < 20; ++i) output.sizes_[i] = values[i].size();
  std::ofstream file(std::getenv("VMECPP_PRECONDITIONER_CAPTURE"),
                     std::ios::binary);
  file.write("VMECGPC2", 8);
  file.write(reinterpret_cast<const char*>(capture.input_.dimensions_.data()),
             sizeof(capture.input_.dimensions_));
  file.write(reinterpret_cast<const char*>(capture.input_.scalars_.data()),
             sizeof(capture.input_.scalars_));
  file.write(reinterpret_cast<const char*>(capture.input_.sizes_.data()),
             sizeof(capture.input_.sizes_));
  file.write(reinterpret_cast<const char*>(output.sizes_.data()),
             sizeof(output.sizes_));
  for (const auto& array : capture.values_) {
    file.write(reinterpret_cast<const char*>(array.data()),
               array.size() * sizeof(double));
  }
  for (const auto& array : values) {
    file.write(reinterpret_cast<const char*>(array.data()),
               array.size() * sizeof(double));
  }
  file.flush();
  if (!file)
    throw std::runtime_error("Preconditioner reference capture failed");
  capture.done_ = true;
}
}  // namespace vmecpp
