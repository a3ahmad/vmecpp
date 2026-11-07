// SPDX-License-Identifier: MIT
// Reference capture only. It calls the original CPU methods unconditionally.
#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_resident.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

namespace {
struct CaptureState {
  std::array<std::vector<double>, 28> values_;
  GpuResidentInput input_{};
  std::array<std::vector<double>, 2> constants_;
  double adiabatic_index_ = 0.0;
  bool ready_ = false;
  bool done_ = false;
  bool constraint_done_ = false;
};
thread_local CaptureState capture_state;

void* Original(const char* name) {
  static void* library = [] {
    const char* path = std::getenv("VMECPP_FORWARD_REFERENCE_SO");
    if (path == nullptr) {
      throw std::runtime_error("Reference extension path required");
    }
    void* result = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (result == nullptr) {
      throw std::runtime_error("Reference extension not loaded");
    }
    return result;
  }();
  void* result = dlsym(library, name);
  if (result == nullptr) {
    throw std::runtime_error("Resident reference method missing");
  }
  return result;
}
}  // namespace

namespace vmecpp {
void IdealMhdModel::computeJacobian() {
  const char* path = std::getenv("VMECPP_RESIDENT_CAPTURE");
  if (path != nullptr && !capture_state.done_ && m_fc_.ns >= 99) {
    if (m_fc_.lfreeb || s_.lasym || !s_.lthreed || ncurr != 1 ||
        r_.nsMinF != 0 || r_.nsMinF1 != 0 || r_.nsMinH != 0 ||
        r_.nsMaxH != m_fc_.ns - 1) {
      throw std::runtime_error(
          "Resident capture requires the official symmetric fixed-boundary "
          "scope");
    }
    auto& input = capture_state.input_;
    capture_state.constants_[0] = m_p_.massH;
    capture_state.constants_[1] = m_p_.radialBlending;
    capture_state.adiabatic_index_ = adiabaticIndex;
    input = {{r1_e.data(),        r1_o.data(),        ru_e.data(),
              ru_o.data(),        rv_e.data(),        rv_o.data(),
              z1_e.data(),        z1_o.data(),        zu_e.data(),
              zu_o.data(),        zv_e.data(),        zv_o.data(),
              lu_e.data(),        lu_o.data(),        lv_e.data(),
              lv_o.data(),        rCon.data(),        zCon.data(),
              m_p_.sqrtSF.data(), m_p_.sqrtSH.data(), s_.wInt.data(),
              m_p_.phipF.data(),  m_p_.phipH.data(),  m_p_.currH.data(),
              m_p_.chipH.data(),  m_p_.iotaH.data(),  m_p_.chipF.data(),
              m_p_.iotaF.data()},
             {r1_e.size(),        r1_o.size(),        ru_e.size(),
              ru_o.size(),        rv_e.size(),        rv_o.size(),
              z1_e.size(),        z1_o.size(),        zu_e.size(),
              zu_o.size(),        zv_e.size(),        zv_o.size(),
              lu_e.size(),        lu_o.size(),        lv_e.size(),
              lv_o.size(),        rCon.size(),        zCon.size(),
              m_p_.sqrtSF.size(), m_p_.sqrtSH.size(), s_.wInt.size(),
              m_p_.phipF.size(),  m_p_.phipH.size(),  m_p_.currH.size(),
              m_p_.chipH.size(),  m_p_.iotaH.size(),  m_p_.chipF.size(),
              m_p_.iotaF.size()},
             {r_.nsMinF, r_.nsMaxF, r_.nsMinF1, r_.nsMaxF1,
              r_.nsMaxFIncludingLcfs, s_.mpol, s_.ntor, s_.nZeta, s_.nThetaEff,
              s_.nThetaReduced, s_.nnyq2, m_fc_.ns, 0},
             {m_fc_.deltaS, dSHalfDsInterp, constants_.lamscale,
              static_cast<double>(signOfJacobian)}};
    for (std::size_t i = 0; i < input.data_.size(); ++i) {
      auto& values = capture_state.values_[i];
      values.assign(input.data_[i], input.data_[i] + input.sizes_[i]);
      input.data_[i] = values.data();
    }
    capture_state.ready_ = true;
  }
  using Method = void (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel15computeJacobianEv"));
  original(this);
  if (m_fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    capture_state.ready_ = false;
  }
}

void IdealMhdModel::computeBCo() {
  using Method = void (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel10computeBCoEv"));
  original(this);
  const char* path = std::getenv("VMECPP_RESIDENT_CAPTURE");
  if (path == nullptr || capture_state.done_ || !capture_state.ready_) {
    return;
  }
  std::array<double, 2> extrema{0.0, 0.0};
  for (double value : tau) {
    if (value < extrema[0] || extrema[0] == 0.0) {
      extrema[0] = value;
    }
    if (value > extrema[1] || extrema[1] == 0.0) {
      extrema[1] = value;
    }
  }
  const GpuResidentOutput output{
      {r1_e.data(),       r1_o.data(),       ru_e.data(),
       ru_o.data(),       rv_e.data(),       rv_o.data(),
       z1_e.data(),       z1_o.data(),       zu_e.data(),
       zu_o.data(),       zv_e.data(),       zv_o.data(),
       lu_e.data(),       lu_o.data(),       lv_e.data(),
       lv_o.data(),       rCon.data(),       zCon.data(),
       r12.data(),        ru12.data(),       zu12.data(),
       rs.data(),         zs.data(),         tau.data(),
       gsqrt.data(),      guu.data(),        guv.data(),
       gvv.data(),        bsupu.data(),      bsupv.data(),
       bsubu.data(),      bsubv.data(),      m_p_.dVdsH.data(),
       m_p_.chipH.data(), m_p_.iotaH.data(), m_p_.chipF.data(),
       m_p_.iotaF.data(), extrema.data()},
      {r1_e.size(),       r1_o.size(),       ru_e.size(),
       ru_o.size(),       rv_e.size(),       rv_o.size(),
       z1_e.size(),       z1_o.size(),       zu_e.size(),
       zu_o.size(),       zv_e.size(),       zv_o.size(),
       lu_e.size(),       lu_o.size(),       lv_e.size(),
       lv_o.size(),       rCon.size(),       zCon.size(),
       r12.size(),        ru12.size(),       zu12.size(),
       rs.size(),         zs.size(),         tau.size(),
       gsqrt.size(),      guu.size(),        guv.size(),
       gvv.size(),        bsupu.size(),      bsupv.size(),
       bsubu.size(),      bsubv.size(),      m_p_.dVdsH.size(),
       m_p_.chipH.size(), m_p_.iotaH.size(), m_p_.chipF.size(),
       m_p_.iotaF.size(), extrema.size()}};
  const auto& input = capture_state.input_;
  std::ofstream capture(path, std::ios::binary);
  capture.write("VMECGPU5", 8);
  capture.write(reinterpret_cast<const char*>(input.dimensions_.data()),
                sizeof(input.dimensions_));
  capture.write(reinterpret_cast<const char*>(input.scalars_.data()),
                sizeof(input.scalars_));
  capture.write(reinterpret_cast<const char*>(input.sizes_.data()),
                sizeof(input.sizes_));
  capture.write(reinterpret_cast<const char*>(output.sizes_.data()),
                sizeof(output.sizes_));
  for (std::size_t i = 0; i < input.data_.size(); ++i) {
    capture.write(reinterpret_cast<const char*>(input.data_[i]),
                  input.sizes_[i] * sizeof(double));
  }
  for (std::size_t i = 0; i < output.data_.size(); ++i) {
    capture.write(reinterpret_cast<const char*>(output.data_[i]),
                  output.sizes_[i] * sizeof(double));
  }
  capture.flush();
  if (!capture) {
    throw std::runtime_error("Resident CPU reference capture failed");
  }
  std::ofstream constants(std::string(path) + ".constants.bin",
                          std::ios::binary);
  constants.write("VMECGPC1", 8);
  constants.write(
      reinterpret_cast<const char*>(&capture_state.adiabatic_index_),
      sizeof(double));
  const std::array<std::size_t, 2> sizes{capture_state.constants_[0].size(),
                                         capture_state.constants_[1].size()};
  constants.write(reinterpret_cast<const char*>(sizes.data()), sizeof(sizes));
  for (const auto& values : capture_state.constants_) {
    constants.write(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(double));
  }
  constants.flush();
  if (!constants) {
    throw std::runtime_error("Resident profile constant capture failed");
  }
  capture_state.done_ = true;
  capture_state.ready_ = false;
}

void IdealMhdModel::computeMHDForces() {
  const char* path = std::getenv("VMECPP_RESIDENT_CAPTURE");
  if (path != nullptr && capture_state.done_ &&
      !capture_state.constraint_done_ &&
      m_fc_.ns == capture_state.input_.dimensions_[11]) {
    std::ofstream file(std::string(path) + ".constraints.bin",
                       std::ios::binary);
    file.write("VMECGCC1", 8);
    const auto& dimensions = capture_state.input_.dimensions_;
    file.write(reinterpret_cast<const char*>(dimensions.data()),
               sizeof(dimensions));
    const std::array<std::size_t, 2> inputs{faccon.size(), tcon.size()};
    const std::array<std::size_t, 4> outputs{gConEff.size(), ruFull.size(),
                                             zuFull.size(), gCon.size()};
    file.write(reinterpret_cast<const char*>(inputs.data()), sizeof(inputs));
    file.write(reinterpret_cast<const char*>(outputs.data()), sizeof(outputs));
    file.write(reinterpret_cast<const char*>(faccon.data()),
               faccon.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(tcon.data()),
               tcon.size() * sizeof(double));
    const std::array<const double*, 4> values{gConEff.data(), ruFull.data(),
                                              zuFull.data(), gCon.data()};
    for (std::size_t i = 0; i < values.size(); ++i) {
      file.write(reinterpret_cast<const char*>(values[i]),
                 outputs[i] * sizeof(double));
    }
    file.flush();
    if (!file) {
      throw std::runtime_error("Resident constraint reference capture failed");
    }
    capture_state.constraint_done_ = true;
  }
  using Method = void (*)(IdealMhdModel*);
  static Method original = reinterpret_cast<Method>(
      Original("_ZN6vmecpp13IdealMhdModel16computeMHDForcesEv"));
  original(this);
}
}  // namespace vmecpp
