// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Benchmark tool for comparing compute backend performance (CPU vs CUDA).
//
// Usage:
//   backend_benchmark [options]
//
// Options:
//   --ns=<int>        Number of radial surfaces (default: 100)
//   --nzeta=<int>     Number of toroidal grid points (default: 36)
//   --ntheta=<int>    Number of poloidal grid points (default: 36)
//   --iterations=<int> Number of benchmark iterations (default: 100)
//   --warmup=<int>    Number of warmup iterations (default: 10)
//
// The benchmark measures the time for backend physics operations:
//   1. ComputeJacobian
//   2. ComputeMetricElements
//   3. ComputeMHDForces

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "vmecpp/common/compute_backend/compute_backend.h"
#include "vmecpp/common/compute_backend/compute_backend_cpu.h"
#include "vmecpp/common/compute_backend/compute_backend_factory.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace {

struct BenchmarkConfig {
  int ns = 100;
  int nzeta = 36;
  int ntheta = 36;
  int iterations = 100;
  int warmup = 10;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
};

struct TimingResult {
  double mean_us;
  double std_us;
  double min_us;
  double max_us;
};

struct BenchmarkResult {
  std::string backend_name;
  TimingResult jacobian;
  TimingResult metric;
  TimingResult mhd_forces;
  double total_mean_us;
  bool available;
};

// Parse command line arguments.
BenchmarkConfig ParseArgs(int argc, char* argv[]) {
  BenchmarkConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--ns=", 0) == 0) {
      config.ns = std::stoi(arg.substr(5));
    } else if (arg.rfind("--nzeta=", 0) == 0) {
      config.nzeta = std::stoi(arg.substr(8));
    } else if (arg.rfind("--ntheta=", 0) == 0) {
      config.ntheta = std::stoi(arg.substr(9));
    } else if (arg.rfind("--iterations=", 0) == 0) {
      config.iterations = std::stoi(arg.substr(13));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      config.warmup = std::stoi(arg.substr(9));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: backend_benchmark [options]\n"
                << "\n"
                << "Options:\n"
                << "  --ns=<int>         Number of radial surfaces (default: "
                   "100)\n"
                << "  --nzeta=<int>      Toroidal grid points (default: 36)\n"
                << "  --ntheta=<int>     Poloidal grid points (default: 36)\n"
                << "  --iterations=<int> Benchmark iterations (default: 100)\n"
                << "  --warmup=<int>     Warmup iterations (default: 10)\n";
      std::exit(0);
    }
  }

  return config;
}

// Calculate timing statistics.
TimingResult CalcStats(const std::vector<double>& times) {
  TimingResult result{};
  if (times.empty()) return result;

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  result.mean_us = sum / static_cast<double>(times.size());

  double sq_sum = 0.0;
  for (double t : times) {
    sq_sum += (t - result.mean_us) * (t - result.mean_us);
  }
  result.std_us = std::sqrt(sq_sum / static_cast<double>(times.size()));

  result.min_us = *std::min_element(times.begin(), times.end());
  result.max_us = *std::max_element(times.begin(), times.end());

  return result;
}

// Data container for benchmark inputs/outputs.
struct BenchmarkData {
  // Grid dimensions
  int ns;
  int grid_size;  // (ns-1) * nZeta * nThetaEff for half-grid operations

  // Geometry arrays (full-grid, even/odd)
  std::vector<double> r1_e, r1_o;
  std::vector<double> z1_e, z1_o;
  std::vector<double> ru_e, ru_o;
  std::vector<double> zu_e, zu_o;
  std::vector<double> rv_e, rv_o;
  std::vector<double> zv_e, zv_o;

  // Jacobian outputs (half-grid)
  std::vector<double> tau;
  std::vector<double> r12, ru12, zu12;
  std::vector<double> rs, zs;

  // Metric outputs (half-grid)
  std::vector<double> gsqrt;
  std::vector<double> guu, guv, gvv;

  // Radial profiles
  std::vector<double> sqrtSF, sqrtSH;

  // Magnetic field (for MHD forces)
  std::vector<double> bsupu, bsupv;
  std::vector<double> totalPressure;

  // MHD force outputs
  std::vector<double> armn_e, armn_o;
  std::vector<double> azmn_e, azmn_o;
  std::vector<double> brmn_e, brmn_o;
  std::vector<double> bzmn_e, bzmn_o;
  std::vector<double> crmn_e, crmn_o;
  std::vector<double> czmn_e, czmn_o;

  void Initialize(const BenchmarkConfig& config,
                  const vmecpp::RadialPartitioning& rp,
                  const vmecpp::Sizes& s) {
    ns = config.ns;
    int full_grid_size = (rp.nsMaxF1 - rp.nsMinF1) * s.nZeta * s.nThetaEff;
    int half_grid_size =
        (rp.nsMaxFIncludingLcfs - rp.nsMinF) * s.nZeta * s.nThetaEff;
    grid_size = half_grid_size;

    // Allocate geometry arrays
    r1_e.resize(full_grid_size);
    r1_o.resize(full_grid_size);
    z1_e.resize(full_grid_size);
    z1_o.resize(full_grid_size);
    ru_e.resize(full_grid_size);
    ru_o.resize(full_grid_size);
    zu_e.resize(full_grid_size);
    zu_o.resize(full_grid_size);
    rv_e.resize(full_grid_size);
    rv_o.resize(full_grid_size);
    zv_e.resize(full_grid_size);
    zv_o.resize(full_grid_size);

    // Allocate Jacobian outputs
    tau.resize(half_grid_size);
    r12.resize(half_grid_size);
    ru12.resize(half_grid_size);
    zu12.resize(half_grid_size);
    rs.resize(half_grid_size);
    zs.resize(half_grid_size);

    // Allocate metric outputs
    gsqrt.resize(half_grid_size);
    guu.resize(half_grid_size);
    guv.resize(half_grid_size);
    gvv.resize(half_grid_size);

    // Allocate radial profiles
    sqrtSF.resize(config.ns);
    sqrtSH.resize(config.ns);

    // Allocate magnetic field arrays
    bsupu.resize(half_grid_size);
    bsupv.resize(half_grid_size);
    totalPressure.resize(half_grid_size);

    // Allocate MHD force outputs
    armn_e.resize(half_grid_size);
    armn_o.resize(half_grid_size);
    azmn_e.resize(half_grid_size);
    azmn_o.resize(half_grid_size);
    brmn_e.resize(half_grid_size);
    brmn_o.resize(half_grid_size);
    bzmn_e.resize(half_grid_size);
    bzmn_o.resize(half_grid_size);
    crmn_e.resize(half_grid_size);
    crmn_o.resize(half_grid_size);
    czmn_e.resize(half_grid_size);
    czmn_o.resize(half_grid_size);

    // Initialize with realistic values
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    double R0 = 5.5;  // Major radius
    double a = 0.5;   // Minor radius

    for (int j = 0; j < config.ns; ++j) {
      double s_val =
          static_cast<double>(j) / static_cast<double>(config.ns - 1);
      sqrtSF[j] = std::sqrt(s_val);
      sqrtSH[j] = std::sqrt(s_val + 0.5 / static_cast<double>(config.ns - 1));
    }

    // Fill geometry with tokamak-like shape
    for (size_t i = 0; i < r1_e.size(); ++i) {
      r1_e[i] = R0 + a * dist(rng);
      r1_o[i] = dist(rng) * 0.01;
      z1_e[i] = a * dist(rng);
      z1_o[i] = dist(rng) * 0.01;
      ru_e[i] = dist(rng) * 0.1;
      ru_o[i] = dist(rng) * 0.01;
      zu_e[i] = dist(rng) * 0.1;
      zu_o[i] = dist(rng) * 0.01;
      rv_e[i] = dist(rng) * 0.01;
      rv_o[i] = dist(rng) * 0.001;
      zv_e[i] = dist(rng) * 0.01;
      zv_o[i] = dist(rng) * 0.001;
    }

    // Initialize magnetic field
    for (size_t i = 0; i < bsupu.size(); ++i) {
      bsupu[i] = 0.3 + dist(rng) * 0.01;
      bsupv[i] = 1.0 + dist(rng) * 0.01;
      totalPressure[i] = 1e4 * (1.0 - static_cast<double>(i) /
                                          static_cast<double>(bsupu.size()));
    }
  }
};

// Run benchmark for a single backend.
BenchmarkResult RunBenchmark(vmecpp::ComputeBackend* backend,
                             const BenchmarkConfig& config,
                             const vmecpp::Sizes& s,
                             vmecpp::RadialPartitioning& rp,
                             BenchmarkData& data) {
  BenchmarkResult result;
  result.backend_name = backend->GetName();
  result.available = backend->IsAvailable();

  if (!result.available) {
    result.jacobian = {0, 0, 0, 0};
    result.metric = {0, 0, 0, 0};
    result.mhd_forces = {0, 0, 0, 0};
    result.total_mean_us = 0;
    return result;
  }

  double deltaS = 1.0 / static_cast<double>(config.ns - 1);

  // Create input/output structs for Jacobian
  vmecpp::JacobianInput jac_input{data.r1_e, data.r1_o, data.z1_e, data.z1_o,
                                  data.ru_e, data.ru_o, data.zu_e, data.zu_o,
                                  data.sqrtSH, deltaS};

  vmecpp::JacobianOutput jac_output{data.tau,  data.r12,  data.ru12,
                                    data.zu12, data.rs, data.zs};

  // Create input/output structs for Metric
  vmecpp::MetricInput metric_input{
      data.r1_e, data.r1_o, data.z1_e, data.z1_o,  data.ru_e,
      data.ru_o, data.zu_e, data.zu_o, data.rv_e,  data.rv_o,
      data.zv_e, data.zv_o, data.tau,  data.r12, data.sqrtSF,
      data.sqrtSH, /*lthreed=*/true};

  vmecpp::MetricOutput metric_output{data.gsqrt, data.guu, data.guv, data.gvv};

  // Create input/output structs for MHD forces
  vmecpp::MHDForcesInput mhd_input{
      data.r1_e,     data.r1_o,         data.z1_e,  data.z1_o,
      data.ru_e,     data.ru_o,         data.zu_e,  data.zu_o,
      data.rv_e,     data.rv_o,         data.zv_e,  data.zv_o,
      data.r12,      data.ru12,         data.zu12,  data.rs,
      data.zs,       data.tau,          data.gsqrt, data.bsupu,
      data.bsupv,    data.totalPressure, data.sqrtSF, data.sqrtSH,
      deltaS, /*lfreeb=*/false, /*lthreed=*/true, config.ns};

  vmecpp::MHDForcesOutput mhd_output{data.armn_e, data.armn_o, data.azmn_e,
                                     data.azmn_o, data.brmn_e, data.brmn_o,
                                     data.bzmn_e, data.bzmn_o, data.crmn_e,
                                     data.crmn_o, data.czmn_e, data.czmn_o};

  // Warmup for Jacobian
  for (int i = 0; i < config.warmup; ++i) {
    backend->ComputeJacobian(jac_input, rp, s, jac_output);
  }
  backend->Synchronize();

  // Benchmark Jacobian
  std::vector<double> jac_times;
  jac_times.reserve(config.iterations);
  for (int i = 0; i < config.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    backend->ComputeJacobian(jac_input, rp, s, jac_output);
    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    jac_times.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  result.jacobian = CalcStats(jac_times);

  // Warmup for Metric
  for (int i = 0; i < config.warmup; ++i) {
    backend->ComputeMetricElements(metric_input, rp, s, metric_output);
  }
  backend->Synchronize();

  // Benchmark Metric
  std::vector<double> metric_times;
  metric_times.reserve(config.iterations);
  for (int i = 0; i < config.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    backend->ComputeMetricElements(metric_input, rp, s, metric_output);
    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    metric_times.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  result.metric = CalcStats(metric_times);

  // Warmup for MHD Forces
  for (int i = 0; i < config.warmup; ++i) {
    backend->ComputeMHDForces(mhd_input, rp, s, mhd_output);
  }
  backend->Synchronize();

  // Benchmark MHD Forces
  std::vector<double> mhd_times;
  mhd_times.reserve(config.iterations);
  for (int i = 0; i < config.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    backend->ComputeMHDForces(mhd_input, rp, s, mhd_output);
    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    mhd_times.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  result.mhd_forces = CalcStats(mhd_times);

  result.total_mean_us =
      result.jacobian.mean_us + result.metric.mean_us + result.mhd_forces.mean_us;

  return result;
}

void PrintResults(const std::vector<BenchmarkResult>& results,
                  const BenchmarkConfig& config) {
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "  VMEC++ Compute Backend Benchmark\n";
  std::cout << "========================================\n";
  std::cout << "\n";
  std::cout << "Configuration:\n";
  std::cout << "  Radial surfaces (ns):     " << config.ns << "\n";
  std::cout << "  Toroidal grid (nzeta):    " << config.nzeta << "\n";
  std::cout << "  Poloidal grid (ntheta):   " << config.ntheta << "\n";
  std::cout << "  Field periods (nfp):      " << config.nfp << "\n";
  std::cout << "  Iterations:               " << config.iterations << "\n";
  std::cout << "  Warmup iterations:        " << config.warmup << "\n";
  std::cout << "\n";

  // Calculate grid size
  int64_t grid_points =
      static_cast<int64_t>(config.ns) * config.nzeta * config.ntheta;
  std::cout << "Problem size:\n";
  std::cout << "  Total grid points:        " << grid_points << "\n";
  std::cout << "\n";

  std::cout << "Results (times in microseconds):\n";
  std::cout << "\n";
  std::cout << std::left << std::setw(12) << "Backend" << std::right
            << std::setw(14) << "Jacobian" << std::setw(14) << "Metric"
            << std::setw(14) << "MHDForces" << std::setw(14) << "Total"
            << std::setw(10) << "Status" << "\n";
  std::cout << std::string(78, '-') << "\n";

  const BenchmarkResult* cpu_result = nullptr;
  for (const auto& r : results) {
    if (r.backend_name.find("CPU") != std::string::npos && r.available) {
      cpu_result = &r;
      break;
    }
  }

  for (const auto& r : results) {
    std::cout << std::left << std::setw(12) << r.backend_name;

    if (!r.available) {
      std::cout << std::right << std::setw(14) << "-" << std::setw(14) << "-"
                << std::setw(14) << "-" << std::setw(14) << "-" << std::setw(10)
                << "N/A" << "\n";
      continue;
    }

    std::cout << std::right << std::fixed << std::setprecision(1)
              << std::setw(14) << r.jacobian.mean_us << std::setw(14)
              << r.metric.mean_us << std::setw(14) << r.mhd_forces.mean_us
              << std::setw(14) << r.total_mean_us << std::setw(10) << "OK"
              << "\n";
  }

  // Print speedup comparison
  if (cpu_result != nullptr) {
    std::cout << "\n";
    std::cout << "Speedup vs CPU:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Backend" << std::right
              << std::setw(12) << "Jacobian" << std::setw(12) << "Metric"
              << std::setw(12) << "MHDForces" << std::setw(12) << "Total"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& r : results) {
      if (!r.available || r.backend_name.find("CPU") != std::string::npos) {
        continue;
      }

      double jac_speedup = cpu_result->jacobian.mean_us / r.jacobian.mean_us;
      double met_speedup = cpu_result->metric.mean_us / r.metric.mean_us;
      double mhd_speedup = cpu_result->mhd_forces.mean_us / r.mhd_forces.mean_us;
      double total_speedup = cpu_result->total_mean_us / r.total_mean_us;

      std::cout << std::left << std::setw(12) << r.backend_name << std::right
                << std::fixed << std::setprecision(2) << std::setw(11)
                << jac_speedup << "x" << std::setw(11) << met_speedup << "x"
                << std::setw(11) << mhd_speedup << "x" << std::setw(11)
                << total_speedup << "x" << "\n";
    }
  }

  std::cout << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  BenchmarkConfig config = ParseArgs(argc, argv);

  // Create Sizes object using the proper constructor.
  vmecpp::Sizes s(/*lasym=*/false, config.nfp, config.mpol, config.ntor,
                  config.ntheta, config.nzeta);

  // Create radial partitioning and configure it for the benchmark.
  vmecpp::RadialPartitioning rp;
  rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, config.ns,
                              /*lfreeb=*/false, /*printout=*/false);

  // Initialize benchmark data
  BenchmarkData data;
  data.Initialize(config, rp, s);

  std::cout << "Initializing backends...\n";

  // Collect results.
  std::vector<BenchmarkResult> results;

  // CPU backend (always available).
  {
    vmecpp::ComputeBackendCpu cpu_backend;
    std::cout << "  CPU backend: " << cpu_backend.GetName() << "\n";
    results.push_back(RunBenchmark(&cpu_backend, config, s, rp, data));
  }

  // CUDA backend (if available).
  {
    vmecpp::BackendConfig cuda_config;
    cuda_config.type = vmecpp::BackendType::kCuda;
    auto cuda_result = vmecpp::ComputeBackendFactory::Create(cuda_config);
    if (cuda_result.ok()) {
      auto& cuda_backend = cuda_result.value();
      std::cout << "  CUDA backend: " << cuda_backend->GetName() << "\n";
      results.push_back(
          RunBenchmark(cuda_backend.get(), config, s, rp, data));
    } else {
      std::cout << "  CUDA backend: not available\n";
      BenchmarkResult cuda_unavailable;
      cuda_unavailable.backend_name = "CUDA";
      cuda_unavailable.available = false;
      results.push_back(cuda_unavailable);
    }
  }

  PrintResults(results, config);

  return 0;
}
