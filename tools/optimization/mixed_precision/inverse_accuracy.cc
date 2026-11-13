// SPDX-License-Identifier: MIT
// Adapted from profile_tiles_benchmark.cc: resident inverse event pairs.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_fourier.h"
#include "resident_replay_capture.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
struct Arm {
  using Launch = void (*)(GpuFourierDeviceState&);
  GpuFourierDeviceState state_{};
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
  std::vector<double*> allocations_;
  double* input_ = nullptr;
  double* output_ = nullptr;
  void* library_ = nullptr;
  Launch launch_ = nullptr;
  std::string owner_;
  std::string report_prefix_;
  std::array<std::uint64_t, 18> sizes_{};
  int verification_ = 0;
  bool exact_required_ = true;

  double* Allocate(std::size_t count) {
    double* result = nullptr;
    Check(cudaMalloc(&result, count * sizeof(double)));
    allocations_.push_back(result);
    return result;
  }
  void Enqueue() { launch_(state_); }
  void Verify(const std::vector<double>& expected,
              const std::vector<double>& original_input) {
    std::vector<double> actual(expected.size());
    Check(cudaMemcpyAsync(actual.data(), output_,
                          actual.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
    std::vector<double> input(original_input.size());
    Check(cudaMemcpyAsync(input.data(), input_, input.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    Require(
        !exact_required_ || std::memcmp(actual.data(), expected.data(),
                                        actual.size() * sizeof(double)) == 0,
        "FP64 control differs from capture");
    const std::string prefix =
        report_prefix_ + ".check" + std::to_string(verification_++);
    std::ofstream raw(prefix + ".f64", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(actual.data()),
              actual.size() * sizeof(double));
    Require(bool(raw), "Output array save failed");
    std::ofstream stats(prefix + ".json");
    stats << std::setprecision(17) << "{\"arrays\":[";
    std::size_t offset = 0;
    for (int array = 0; array < 18; ++array) {
      long double squared_error = 0, squared_reference = 0;
      double maximum_error = 0, maximum_reference = 0;
      std::uint64_t different = 0, nonfinite = 0;
      for (std::uint64_t i = 0; i < sizes_[array]; ++i) {
        const double value = actual[offset + i],
                     reference = expected[offset + i];
        if (!std::isfinite(value)) ++nonfinite;
        if (std::bit_cast<std::uint64_t>(value) !=
            std::bit_cast<std::uint64_t>(reference))
          ++different;
        const double error = std::abs(value - reference);
        maximum_error = std::max(maximum_error, error);
        maximum_reference = std::max(maximum_reference, std::abs(reference));
        squared_error += static_cast<long double>(error) * error;
        squared_reference += static_cast<long double>(reference) * reference;
      }
      if (array) stats << ',';
      stats << "{\"index\":" << array << ",\"count\":" << sizes_[array]
            << ",\"different_bits\":" << different
            << ",\"nonfinite\":" << nonfinite
            << ",\"max_abs\":" << maximum_error
            << ",\"max_reference\":" << maximum_reference << ",\"relative_l2\":"
            << (squared_reference > 0
                    ? std::sqrt(squared_error / squared_reference)
                    : std::sqrt(squared_error))
            << '}';
      offset += sizes_[array];
      Require(nonfinite == 0, "Nonfinite inverse output");
    }
    stats << "],\"input_unchanged_checked\":true}\n";
    Require(bool(stats), "Statistics save failed");
    Require(std::memcmp(input.data(), original_input.data(),
                        input.size() * sizeof(double)) == 0,
            "Readonly inverse input changed");
  }
  void Close() {
    Check(cudaStreamSynchronize(stream_));
    Check(cudaGraphExecDestroy(executable_));
    Check(cudaGraphDestroy(graph_));
    for (auto* pointer : allocations_) Check(cudaFree(pointer));
    Check(cudaEventDestroy(begin_));
    Check(cudaEventDestroy(end_));
    Check(cudaStreamDestroy(stream_));
    allocations_.clear();
    stream_ = nullptr;
    // Retain libraries through process exit, after graph/kernel retirement.
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::puts(
        "PLAN ONLY: inverse-accuracy INPUT REPORT REPEATS PAIRS BASELINE "
        "CANDIDATE PRECISION");
    return 0;
  }
  try {
    Require(argc == 8,
            "Expected INPUT REPORT REPEATS PAIRS BASELINE CANDIDATE");
    const int precision = std::stoi(argv[7]);
    Require(precision == 32 || precision == 64, "Precision must be32 or64");
    const int repeats = std::stoi(argv[3]);
    const int pairs = std::stoi(argv[4]);
    Require(repeats >= 1 && repeats <= 100 && pairs >= 2 && pairs <= 10,
            "Bounded repeats1..100 and pairs2..10 required");
    Require(!std::filesystem::exists(argv[2]), "Fresh report required");
    auto capture = vmecpp_experiment::Read<GpuFourierInput, GpuFourierOutput>(
        argv[1], "VMECGPU1");
    std::array<Arm, 2> arms;
    for (int variant = 0; variant < 2; ++variant) {
      auto& arm = arms[variant];
      arm.owner_ = std::filesystem::canonical(argv[5 + variant]).string();
      arm.library_ = dlopen(arm.owner_.c_str(), RTLD_NOW | RTLD_LOCAL);
      Require(arm.library_ != nullptr, "Inverse module load failed");
      auto* address = dlsym(arm.library_, "LaunchGpuFourierDevice");
      Dl_info info{};
      Require(
          address != nullptr && dladdr(address, &info) != 0 &&
              info.dli_fname != nullptr &&
              std::filesystem::canonical(info.dli_fname).string() == arm.owner_,
          "Exact inverse launch owner mismatch");
      arm.launch_ = reinterpret_cast<Arm::Launch>(address);
      arm.exact_required_ = variant == 0 || precision == 64;
      arm.report_prefix_ =
          std::string(argv[2]) + ".arm" + std::to_string(variant);
      for (int i = 0; i < 18; ++i) arm.sizes_[i] = capture.output_.sizes_[i];
      if (variant == 1) {
        using Setter = cudaError_t (*)(int, void*);
        auto setter = reinterpret_cast<Setter>(
            dlsym(arm.library_, "SetGpuInverseTransformPrecision"));
        Dl_info setter_info{};
        Require(
            setter && dladdr(reinterpret_cast<void*>(setter), &setter_info) &&
                setter_info.dli_fname &&
                std::filesystem::canonical(setter_info.dli_fname).string() ==
                    arm.owner_,
            "Precision setter owner mismatch");
        Check(setter(precision, nullptr));
      }

      Check(cudaStreamCreateWithFlags(&arm.stream_, cudaStreamNonBlocking));
      Check(cudaEventCreate(&arm.begin_));
      Check(cudaEventCreate(&arm.end_));
      arm.input_ = arm.Allocate(capture.values_.size());
      arm.output_ = arm.Allocate(capture.expected_.size());
      Check(cudaMemcpyAsync(arm.input_, capture.values_.data(),
                            capture.values_.size() * sizeof(double),
                            cudaMemcpyHostToDevice, arm.stream_));
      auto& state = arm.state_;
      state.dimensions_ = capture.input_.dimensions_;
      state.output_sizes_ = capture.output_.sizes_;
      state.stream_ = arm.stream_;
      std::size_t offset = 0;
      for (int i = 0; i < 16; ++i) {
        state.input_[i] = arm.input_ + offset;
        offset += capture.input_.sizes_[i];
      }
      offset = 0;
      for (int i = 0; i < 18; ++i) {
        state.output_[i] = arm.output_ + offset;
        offset += capture.output_.sizes_[i];
      }
      const auto& d = state.dimensions_;
      state.partials_ =
          arm.Allocate(std::size_t{12} * (d[3] - d[2]) * d[5] * d[7]);
      for (int warm = 0; warm < 3; ++warm) {
        arm.Enqueue();
        arm.Verify(capture.expected_, capture.values_);
      }
      Check(cudaStreamBeginCapture(arm.stream_,
                                   cudaStreamCaptureModeThreadLocal));
      arm.Enqueue();
      Check(cudaStreamEndCapture(arm.stream_, &arm.graph_));
      Check(cudaGraphInstantiate(&arm.executable_, arm.graph_, 0));
      Check(cudaGraphLaunch(arm.executable_, arm.stream_));
      arm.Verify(capture.expected_, capture.values_);
    }
    std::ofstream output(argv[2]);
    Require(bool(output), "Report open failed");
    output << std::setprecision(17)
           << "{\"scope\":\"Measured CUDAevents: device "
              "outputclear+ToroidalPartials+inverse; "
              "host transfers and verification outside event bounds; not "
              "isolated main kernel or solver timing\","
              "\"repeats\":"
           << repeats << ",\"pairs\":" << pairs
           << ",\"excluded_pair\":0,\"baseline_owner\":\"" << arms[0].owner_
           << "\",\"candidate_owner\":\"" << arms[1].owner_
           << "\",\"samples\":[";
    bool first = true;
    for (int mode = 0; mode < 2; ++mode) {
      for (int pair = 0; pair < pairs; ++pair) {
        const int first_variant = (mode + pair) & 1;
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
          const int variant = first_variant ^ ordinal;
          auto& arm = arms[variant];
          Check(cudaEventRecord(arm.begin_, arm.stream_));
          for (int repeat = 0; repeat < repeats; ++repeat) {
            if (mode == 0)
              arm.Enqueue();
            else
              Check(cudaGraphLaunch(arm.executable_, arm.stream_));
          }
          Check(cudaEventRecord(arm.end_, arm.stream_));
          Check(cudaEventSynchronize(arm.end_));
          float milliseconds = 0;
          Check(cudaEventElapsedTime(&milliseconds, arm.begin_, arm.end_));
          Require(std::isfinite(milliseconds) && milliseconds > 0,
                  "Invalid CUDA event interval");
          arm.Verify(capture.expected_, capture.values_);
          if (!first) output << ',';
          first = false;
          output << "{\"mode\":\"" << (mode == 0 ? "direct" : "graph")
                 << "\",\"pair\":" << pair << ",\"ordinal\":" << ordinal
                 << ",\"variant\":" << variant << ",\"calls\":" << repeats
                 << ",\"mean_us\":"
                 << static_cast<double>(milliseconds) * 1000.0 / repeats
                 << ",\"numeric_exactness_required\":false,\"input_unchanged\":"
                    "true}";
        }
      }
    }
    for (auto& arm : arms) arm.Close();
    output << "],\"complete\":true,\"retired\":true}\n";
    output.close();
    Require(bool(output), "Report write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    // On failure retain possibly-live resources until owned process exit.
    return 1;
  }
}
