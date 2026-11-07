// SPDX-License-Identifier: MIT
// Isolated interposition experiment. The reference extension remains untouched.
#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "gpu_forward.h"
#ifdef VMECPP_GPU_PIPELINE
#include "gpu_pipeline.h"
#endif
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

namespace {
constexpr const char* kForwardSymbol =
    "_ZN6vmecpp33ForcesToFourier3DSymmFastPoloidalERKNS_"
    "15RealSpaceForcesERKSt6vectorIdSaIdEERKNS_18RadialPartitioningERKNS_"
    "11FlowControlERKNS_5SizesERKNS_24FourierBasisFastPoloidalENS_"
    "19VacuumPressureStateERNS_13FourierForcesE";

auto OriginalForward() {
  using Function = decltype(&vmecpp::ForcesToFourier3DSymmFastPoloidal);
  static Function original = [] {
    const char* path = std::getenv("VMECPP_FORWARD_REFERENCE_SO");
    if (path == nullptr) {
      throw std::runtime_error("Reference extension path required");
    }
    void* handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
      throw std::runtime_error("Reference extension is not loaded");
    }
    auto result = reinterpret_cast<Function>(dlsym(handle, kForwardSymbol));
    if (result == nullptr ||
        result == &vmecpp::ForcesToFourier3DSymmFastPoloidal) {
      throw std::runtime_error("Cannot resolve the original forward transform");
    }
    return result;
  }();
  return original;
}

template <class Input, class Output>
void Capture(const char* path, const Input& input, const Output& output,
             const char* magic = "VMECGPU2") {
  static_assert(sizeof(std::size_t) == 8 && sizeof(int) == 4);
  std::ofstream capture(path, std::ios::binary);
  capture.write(magic, 8);
  capture.write(reinterpret_cast<const char*>(input.dimensions_.data()),
                sizeof(input.dimensions_));
  capture.write(reinterpret_cast<const char*>(input.sizes_.data()),
                sizeof(input.sizes_));
  capture.write(reinterpret_cast<const char*>(output.sizes_.data()),
                sizeof(output.sizes_));
  for (std::size_t index = 0; index < input.data_.size(); ++index) {
    capture.write(reinterpret_cast<const char*>(input.data_[index]),
                  input.sizes_[index] * sizeof(double));
  }
  for (std::size_t index = 0; index < output.data_.size(); ++index) {
    capture.write(reinterpret_cast<const char*>(output.data_[index]),
                  output.sizes_[index] * sizeof(double));
  }
  capture.flush();
  if (!capture) {
    throw std::runtime_error("Forward capture failed");
  }
}
#ifdef VMECPP_GPU_PIPELINE
struct PipelineState {
  GpuPipelineInput input_{};
  const void* owner_ = nullptr;
  const double* force_data_ = nullptr;
  bool ready_ = false;
};
thread_local PipelineState pipeline_state;

bool Enabled(const char* key) {
  const char* value = std::getenv(key);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

void* OriginalMemberSymbol(const char* symbol) {
  static void* handle = [] {
    const char* path = std::getenv("VMECPP_FORWARD_REFERENCE_SO");
    if (path == nullptr) {
      throw std::runtime_error("Reference extension path required");
    }
    void* result = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (result == nullptr) {
      throw std::runtime_error("Reference extension is not loaded");
    }
    return result;
  }();
  void* result = dlsym(handle, symbol);
  if (result == nullptr) {
    throw std::runtime_error("Missing original MHD method");
  }
  return result;
}
#endif
}  // namespace

namespace vmecpp {
void ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
#ifdef VMECPP_GPU_PIPELINE
  if (Enabled("VMECPP_GPU_FORCE_PIPELINE")) {
    if (!pipeline_state.ready_ ||
        pipeline_state.force_data_ != d.armn_e.data()) {
      throw std::runtime_error("MHD/forward pipeline state is incomplete");
    }
    GpuForwardOutput output{
        {m_physical_forces.frcc.data(), m_physical_forces.frss.data(),
         m_physical_forces.fzsc.data(), m_physical_forces.fzcs.data(),
         m_physical_forces.flsc.data(), m_physical_forces.flcs.data()},
        {m_physical_forces.frcc.size(), m_physical_forces.frss.size(),
         m_physical_forces.fzsc.size(), m_physical_forces.fzcs.size(),
         m_physical_forces.flsc.size(), m_physical_forces.flcs.size()}};
    RunGpuPipeline(pipeline_state.input_, output);
    pipeline_state.ready_ = false;
    return;
  }
#endif
  const GpuForwardInput input{
      {d.armn_e.data(),   d.armn_o.data(),  d.azmn_e.data(),  d.azmn_o.data(),
       d.blmn_e.data(),   d.blmn_o.data(),  d.brmn_e.data(),  d.brmn_o.data(),
       d.bzmn_e.data(),   d.bzmn_o.data(),  d.clmn_e.data(),  d.clmn_o.data(),
       d.crmn_e.data(),   d.crmn_o.data(),  d.czmn_e.data(),  d.czmn_o.data(),
       d.frcon_e.data(),  d.frcon_o.data(), d.fzcon_e.data(), d.fzcon_o.data(),
       xmpq.data(),       fb.cosmui.data(), fb.sinmui.data(), fb.cosmumi.data(),
       fb.sinmumi.data(), fb.cosnv.data(),  fb.sinnv.data(),  fb.cosnvn.data(),
       fb.sinnvn.data()},
      {d.armn_e.size(),   d.armn_o.size(),  d.azmn_e.size(),  d.azmn_o.size(),
       d.blmn_e.size(),   d.blmn_o.size(),  d.brmn_e.size(),  d.brmn_o.size(),
       d.bzmn_e.size(),   d.bzmn_o.size(),  d.clmn_e.size(),  d.clmn_o.size(),
       d.crmn_e.size(),   d.crmn_o.size(),  d.czmn_e.size(),  d.czmn_o.size(),
       d.frcon_e.size(),  d.frcon_o.size(), d.fzcon_e.size(), d.fzcon_o.size(),
       xmpq.size(),       fb.cosmui.size(), fb.sinmui.size(), fb.cosmumi.size(),
       fb.sinmumi.size(), fb.cosnv.size(),  fb.sinnv.size(),  fb.cosnvn.size(),
       fb.sinnvn.size()},
      {rp.nsMinF, rp.nsMaxF, rp.nsMinF1, rp.nsMaxF1, rp.nsMaxFIncludingLcfs,
       s.mpol, s.ntor, s.nZeta, s.nThetaEff, s.nThetaReduced, s.nnyq2, fc.ns,
       static_cast<int>(fc.lfreeb)}};
  GpuForwardOutput output{
      {m_physical_forces.frcc.data(), m_physical_forces.frss.data(),
       m_physical_forces.fzsc.data(), m_physical_forces.fzcs.data(),
       m_physical_forces.flsc.data(), m_physical_forces.flcs.data()},
      {m_physical_forces.frcc.size(), m_physical_forces.frss.size(),
       m_physical_forces.fzsc.size(), m_physical_forces.fzcs.size(),
       m_physical_forces.flsc.size(), m_physical_forces.flcs.size()}};
  const char* enabled = std::getenv("VMECPP_GPU_FORWARD");
  const char* capture_path = std::getenv("VMECPP_FORWARD_CAPTURE");
  if (enabled != nullptr && std::strcmp(enabled, "1") == 0) {
    if (capture_path != nullptr) {
      throw std::runtime_error("Capture requires the CPU reference path");
    }
    RunGpuForward(input, output);
    return;
  }
  OriginalForward()(d, xmpq, rp, fc, s, fb, vacuum_pressure_state,
                    m_physical_forces);
  static thread_local bool captured = false;
  if (capture_path != nullptr && !captured && fc.ns >= 99) {
    Capture(capture_path, input, output);
    captured = true;
  }
#ifdef VMECPP_GPU_PIPELINE
  const char* pipeline_capture = std::getenv("VMECPP_PIPELINE_CAPTURE");
  static thread_local bool pipeline_captured = false;
  if (pipeline_capture != nullptr && !pipeline_captured && fc.ns >= 99) {
    if (!pipeline_state.ready_ ||
        pipeline_state.force_data_ != d.armn_e.data()) {
      throw std::runtime_error("Cannot capture incomplete CPU force pipeline");
    }
    Capture(pipeline_capture, pipeline_state.input_, output, "VMECGPU4");
    pipeline_captured = true;
  }
  pipeline_state.ready_ = false;
#endif
}

#ifdef VMECPP_GPU_PIPELINE
void IdealMhdModel::computeMHDForces() {
  const bool enabled = Enabled("VMECPP_GPU_FORCE_PIPELINE");
  const char* capture = std::getenv("VMECPP_PIPELINE_CAPTURE");
  const char* mhd_capture = std::getenv("VMECPP_MHD_CAPTURE");
  if (enabled || capture != nullptr || mhd_capture != nullptr) {
    if (m_fc_.lfreeb || s_.lasym || !s_.lthreed || r_.nsMinF != 0 ||
        r_.nsMinF1 != 0 || r_.nsMinH != 0 || r_.nsMaxH != m_fc_.ns - 1) {
      throw std::runtime_error(
          "Force pipeline supports only the official single-partition "
          "symmetric scope");
    }
    pipeline_state.input_ = {
        {r12.data(),         ru12.data(),
         zu12.data(),        rs.data(),
         zs.data(),          tau.data(),
         gsqrt.data(),       bsupu.data(),
         bsupv.data(),       totalPressure.data(),
         r1_e.data(),        r1_o.data(),
         ru_e.data(),        ru_o.data(),
         zu_e.data(),        zu_o.data(),
         z1_o.data(),        rv_e.data(),
         rv_o.data(),        zv_e.data(),
         zv_o.data(),        m_p_.sqrtSF.data(),
         m_p_.sqrtSH.data(), &m_fc_.deltaS,
         rCon.data(),        zCon.data(),
         rCon0.data(),       zCon0.data(),
         gCon.data(),        ruFull.data(),
         zuFull.data(),      blmn_e.data(),
         blmn_o.data(),      clmn_e.data(),
         clmn_o.data(),      xmpq.data(),
         t_.cosmui.data(),   t_.sinmui.data(),
         t_.cosmumi.data(),  t_.sinmumi.data(),
         t_.cosnv.data(),    t_.sinnv.data(),
         t_.cosnvn.data(),   t_.sinnvn.data()},
        {r12.size(),         ru12.size(),
         zu12.size(),        rs.size(),
         zs.size(),          tau.size(),
         gsqrt.size(),       bsupu.size(),
         bsupv.size(),       totalPressure.size(),
         r1_e.size(),        r1_o.size(),
         ru_e.size(),        ru_o.size(),
         zu_e.size(),        zu_o.size(),
         z1_o.size(),        rv_e.size(),
         rv_o.size(),        zv_e.size(),
         zv_o.size(),        m_p_.sqrtSF.size(),
         m_p_.sqrtSH.size(), 1,
         rCon.size(),        zCon.size(),
         rCon0.size(),       zCon0.size(),
         gCon.size(),        ruFull.size(),
         zuFull.size(),      blmn_e.size(),
         blmn_o.size(),      clmn_e.size(),
         clmn_o.size(),      xmpq.size(),
         t_.cosmui.size(),   t_.sinmui.size(),
         t_.cosmumi.size(),  t_.sinmumi.size(),
         t_.cosnv.size(),    t_.sinnv.size(),
         t_.cosnvn.size(),   t_.sinnvn.size()},
        {r_.nsMinF, r_.nsMaxF, r_.nsMinF1, r_.nsMaxF1, r_.nsMaxFIncludingLcfs,
         s_.mpol, s_.ntor, s_.nZeta, s_.nThetaEff, s_.nThetaReduced, s_.nnyq2,
         m_fc_.ns, static_cast<int>(m_fc_.lfreeb)}};
    pipeline_state.owner_ = this;
    pipeline_state.force_data_ = armn_e.data();
    pipeline_state.ready_ = true;
  }
  if (enabled) {
    if (capture != nullptr || mhd_capture != nullptr) {
      throw std::runtime_error(
          "Pipeline capture requires the CPU reference path");
    }
    return;
  }
  using Member = void (*)(IdealMhdModel*);
  static Member original = reinterpret_cast<Member>(
      OriginalMemberSymbol("_ZN6vmecpp13IdealMhdModel16computeMHDForcesEv"));
  original(this);
  static thread_local bool captured = false;
  if (mhd_capture != nullptr && !captured && m_fc_.ns >= 99) {
    const std::size_t count =
        static_cast<std::size_t>(std::min(r_.nsMaxF, m_fc_.ns - 1) -
                                 r_.nsMinF) *
        s_.nZnT;
    GpuMhdOutput output{
        {armn_e.data(), armn_o.data(), azmn_e.data(), azmn_o.data(),
         brmn_e.data(), brmn_o.data(), bzmn_e.data(), bzmn_o.data(),
         crmn_e.data(), crmn_o.data(), czmn_e.data(), czmn_o.data()},
        {count, count, count, count, count, count, count, count, count, count,
         count, count}};
    Capture(mhd_capture, pipeline_state.input_, output, "VMECGPU3");
    captured = true;
  }
}

void IdealMhdModel::assembleTotalForces() {
  if (Enabled("VMECPP_GPU_FORCE_PIPELINE")) {
    if (!pipeline_state.ready_ || pipeline_state.owner_ != this) {
      throw std::runtime_error("MHD/assembly pipeline state is incomplete");
    }
    return;
  }
  using Member = void (*)(IdealMhdModel*);
  static Member original = reinterpret_cast<Member>(
      OriginalMemberSymbol("_ZN6vmecpp13IdealMhdModel19assembleTotalForcesEv"));
  original(this);
}
#endif
}  // namespace vmecpp
