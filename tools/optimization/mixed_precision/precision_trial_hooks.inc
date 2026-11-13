// SPDX-License-Identifier: MIT
// Experimental one-equilibrium-per-process precision policy and diagnostics.
// Included inside the shim's anonymous namespace. No controller changes.
struct PrecisionTrial {
  using Setter = int (*)(int, void*);
  using Getter = int (*)(unsigned long long*);
  void* owner_ = nullptr;
  int ns_ = -1;
  int calls_ = 0;
  int float_calls_ = 0;
  int double_calls_ = 0;
  int active_ = -1;
  std::array<unsigned long long, 4> initial_{};
  Setter inverse_set_ = nullptr;
  Setter forward_set_ = nullptr;
  Getter inverse_get_ = nullptr;
  Getter forward_get_ = nullptr;
};
thread_local PrecisionTrial precision_trial;

const char* PrecisionMode() {
  return std::getenv("VMECPP_EXPERIMENT_TRANSFORM_MODE");
}

void PrecisionNumber(std::ostream& m_file, double value) {
  if (std::isfinite(value))
    m_file << value;
  else
    m_file << "null";
}

std::array<unsigned long long, 4> PrecisionCounts() {
  std::array<unsigned long long, 4> result{};
  if (precision_trial.inverse_get_(result.data()) != 0 ||
      precision_trial.forward_get_(result.data() + 2) != 0) {
    throw std::runtime_error("Experimental precision counter read failed");
  }
  return result;
}

void PrecisionSnapshot(const char* event) {
  const char* path = std::getenv("VMECPP_EXPERIMENT_PRECISION_TRACE");
  if (path == nullptr || precision_trial.ns_ < 0) return;
  const auto counts = PrecisionCounts();
  std::ofstream file(path, std::ios::app);
  file << "{\"event\":\"" << event << "\",\"ns\":" << precision_trial.ns_
       << ",\"gpu_force_calls\":" << precision_trial.calls_
       << ",\"fp32_force_calls\":" << precision_trial.float_calls_
       << ",\"fp64_force_calls\":" << precision_trial.double_calls_
       << ",\"precision\":" << precision_trial.active_
       << ",\"actual_kernel_counts_order\":[\"inverse64\",\"inverse32\","
          "\"forward64\",\"forward32\"]"
       << ",\"actual_kernel_counts\":[";
  for (int index = 0; index < 4; ++index) {
    if (index) file << ',';
    file << counts[index] - precision_trial.initial_[index];
  }
  file << "]}\n";
  if (!file) throw std::runtime_error("Precision trace write failed");
}

void PrecisionResolve() {
  if (precision_trial.inverse_set_ != nullptr) return;
  const char* path = std::getenv("VMECPP_EXPERIMENT_RESIDENT_SO");
  if (path == nullptr)
    throw std::runtime_error("Precision resident owner required");
  void* library = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
  if (library == nullptr)
    throw std::runtime_error("Precision resident not loaded");
  precision_trial.inverse_set_ = reinterpret_cast<PrecisionTrial::Setter>(
      dlsym(library, "SetGpuInverseTransformPrecision"));
  precision_trial.forward_set_ = reinterpret_cast<PrecisionTrial::Setter>(
      dlsym(library, "SetGpuForwardTransformPrecision"));
  precision_trial.inverse_get_ = reinterpret_cast<PrecisionTrial::Getter>(
      dlsym(library, "GetGpuInverseTransformPrecisionCounts"));
  precision_trial.forward_get_ = reinterpret_cast<PrecisionTrial::Getter>(
      dlsym(library, "GetGpuForwardTransformPrecisionCounts"));
  if (precision_trial.inverse_set_ == nullptr ||
      precision_trial.forward_set_ == nullptr ||
      precision_trial.inverse_get_ == nullptr ||
      precision_trial.forward_get_ == nullptr)
    throw std::runtime_error("Precision entry point missing");
}

void PrecisionBeforeForce(void* owner, int ns, int iter1, int iter2) {
  const char* mode = PrecisionMode();
  if (mode == nullptr) return;
  const std::string selected(mode);
  if (selected != "fp64" && selected != "fp32" && selected != "warmup")
    throw std::runtime_error("Invalid experimental precision mode");
  int budget = 0;
  if (selected == "warmup") {
    const char* value = std::getenv("VMECPP_EXPERIMENT_FP32_CALLS");
    char* end = nullptr;
    const long parsed = value == nullptr ? -1 : std::strtol(value, &end, 10);
    if (parsed < 1 || parsed > 10000 || end == nullptr || *end != '\0')
      throw std::runtime_error("Invalid experimental FP32 call budget");
    budget = int(parsed);
  }
  PrecisionResolve();
  if (precision_trial.owner_ != owner || precision_trial.ns_ != ns) {
    if (precision_trial.ns_ >= 0) PrecisionSnapshot("stage_end");
    precision_trial.owner_ = owner;
    precision_trial.ns_ = ns;
    precision_trial.calls_ = 0;
    precision_trial.float_calls_ = 0;
    precision_trial.double_calls_ = 0;
    precision_trial.active_ = -1;
    precision_trial.initial_ = PrecisionCounts();
  }
  const int requested = selected == "fp32" || (selected == "warmup" &&
                                               precision_trial.calls_ < budget)
                            ? 32
                            : 64;
  if (requested != precision_trial.active_) {
    if (precision_trial.inverse_set_(requested, nullptr) != 0 ||
        precision_trial.forward_set_(requested, nullptr) != 0)
      throw std::runtime_error("Precision switch failed");
    precision_trial.active_ = requested;
    PrecisionSnapshot("precision_switch");
    const char* path = std::getenv("VMECPP_EXPERIMENT_PRECISION_TRACE");
    if (path != nullptr) {
      std::ofstream file(path, std::ios::app);
      file << "{\"event\":\"switch_iteration\",\"ns\":" << ns
           << ",\"iter1\":" << iter1 << ",\"iter2\":" << iter2
           << ",\"precision\":" << requested << "}\n";
    }
  }
  ++precision_trial.calls_;
  if (requested == 32)
    ++precision_trial.float_calls_;
  else
    ++precision_trial.double_calls_;
}

void PrecisionMaterialized(const vmecpp::FlowControl& fc, bool status_ok,
                           bool reference_force_evaluated) {
  const char* path = std::getenv("VMECPP_EXPERIMENT_PRECISION_TRACE");
  if (PrecisionMode() == nullptr || path == nullptr) return;
  std::ofstream file(path, std::ios::app);
  file << std::setprecision(17)
       << "{\"event\":\"reference_fp64_materialization\",\"ns\":" << fc.ns
       << ",\"reference_update_ok\":" << (status_ok ? "true" : "false")
       << ",\"reference_force_evaluated\":"
       << (reference_force_evaluated ? "true" : "false")
       << ",\"reference_restart_reason\":"
       << static_cast<int>(fc.restart_reason)
       << ",\"precision\":" << precision_trial.active_ << ",\"ftol\":";
  PrecisionNumber(file, fc.ftolv);
  file << ",\"fp64_residuals\":[";
  PrecisionNumber(file, fc.fsqr);
  file << ',';
  PrecisionNumber(file, fc.fsqz);
  file << ',';
  PrecisionNumber(file, fc.fsql);
  const bool passed = status_ok && reference_force_evaluated &&
                      std::isfinite(fc.fsqr) && std::isfinite(fc.fsqz) &&
                      std::isfinite(fc.fsql) && fc.fsqr <= fc.ftolv &&
                      fc.fsqz <= fc.ftolv && fc.fsql <= fc.ftolv;
  file << "],\"fp64_force_tolerance_passed\":" << (passed ? "true" : "false")
       << "}\n";
  if (!file) throw std::runtime_error("FP64 certificate trace write failed");
}
