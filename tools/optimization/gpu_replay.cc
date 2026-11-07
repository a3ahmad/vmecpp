// Replay one captured real transform; no synthetic equilibrium inputs.
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(VMECPP_REPLAY_RESIDENT)
#include "gpu_resident.h"
using ReplayInput = GpuResidentInput;
using ReplayOutput = GpuResidentOutput;
constexpr const char* kMagic = "VMECGPU5";
constexpr const char* kEntry = "RunGpuResidentFields";
#elif defined(VMECPP_REPLAY_PIPELINE)
#include "gpu_pipeline.h"
using ReplayInput = GpuPipelineInput;
using ReplayOutput = GpuForwardOutput;
constexpr const char* kMagic = "VMECGPU4";
constexpr const char* kEntry = "RunGpuPipeline";
#elif defined(VMECPP_REPLAY_MHD)
#include "gpu_pipeline.h"
using ReplayInput = GpuPipelineInput;
using ReplayOutput = GpuMhdOutput;
constexpr const char* kMagic = "VMECGPU3";
constexpr const char* kEntry = "RunGpuMhd";
#elif defined(VMECPP_REPLAY_FORWARD)
#include "gpu_forward.h"
using ReplayInput = GpuForwardInput;
using ReplayOutput = GpuForwardOutput;
constexpr const char* kMagic = "VMECGPU2";
constexpr const char* kEntry = "RunGpuForward";
#else
#include "gpu_fourier.h"
using ReplayInput = GpuFourierInput;
using ReplayOutput = GpuFourierOutput;
constexpr const char* kMagic = "VMECGPU1";
constexpr const char* kEntry = "RunGpuFourier";
#endif

int Run(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: gpu_replay fixture.bin library.so\n");
    return 1;
  }
  std::ifstream capture(argv[1], std::ios::binary);
  std::array<char, 8> magic;
  capture.read(magic.data(), magic.size());
  if (std::memcmp(magic.data(), kMagic, 8) != 0) {
    throw std::runtime_error("Invalid capture header");
  }
  ReplayInput input{};
  ReplayOutput output{};
  capture.read(reinterpret_cast<char*>(input.dimensions_.data()),
               sizeof(input.dimensions_));
#ifdef VMECPP_REPLAY_RESIDENT
  capture.read(reinterpret_cast<char*>(input.scalars_.data()),
               sizeof(input.scalars_));
#endif
  capture.read(reinterpret_cast<char*>(input.sizes_.data()),
               sizeof(input.sizes_));
  capture.read(reinterpret_cast<char*>(output.sizes_.data()),
               sizeof(output.sizes_));
  if (!capture) throw std::runtime_error("Incomplete capture header");
  for (const auto size : input.sizes_) {
    if (size > 10000000) throw std::runtime_error("Input capture too large");
  }
  for (const auto size : output.sizes_) {
    if (size > 10000000) throw std::runtime_error("Output capture too large");
  }
  const std::size_t inputs =
      std::accumulate(input.sizes_.begin(), input.sizes_.end(), std::size_t{0});
  const std::size_t outputs = std::accumulate(
      output.sizes_.begin(), output.sizes_.end(), std::size_t{0});
  if (inputs > 10000000 || outputs > 10000000) {
    throw std::runtime_error("Invalid capture dimensions");
  }
  std::vector<double> values(inputs);
  std::vector<double> expected(outputs);
  std::vector<double> actual(outputs);
  capture.read(reinterpret_cast<char*>(values.data()), inputs * sizeof(double));
  capture.read(reinterpret_cast<char*>(expected.data()),
               outputs * sizeof(double));
  if (!capture) throw std::runtime_error("Incomplete capture");
  std::size_t offset = 0;
  for (std::size_t index = 0; index < input.sizes_.size(); ++index) {
    input.data_[index] = values.data() + offset;
    offset += input.sizes_[index];
  }
  offset = 0;
  for (std::size_t index = 0; index < output.sizes_.size(); ++index) {
    output.data_[index] = actual.data() + offset;
    offset += output.sizes_[index];
  }
  // Keep the library loaded through process exit so TLS CUDA buffers can clean
  // up.
  void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) throw std::runtime_error(dlerror());
  using Transform = void (*)(const ReplayInput&, ReplayOutput&);
  auto transform = reinterpret_cast<Transform>(dlsym(library, kEntry));
  if (transform == nullptr) throw std::runtime_error("Transform entry missing");
  transform(input, output);
  std::size_t differences = 0;
  double max_absolute = 0;
  for (std::size_t index = 0; index < outputs; ++index) {
    if (std::memcmp(&actual[index], &expected[index], sizeof(double)) != 0) {
      ++differences;
      max_absolute =
          std::max(max_absolute, std::abs(actual[index] - expected[index]));
    }
  }
  constexpr int kRepeats = 100;
  constexpr int kWarmups = 10;
  for (int repeat = 0; repeat < kWarmups; ++repeat) transform(input, output);
  const auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < kRepeats; ++repeat) transform(input, output);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  for (std::size_t index = 0; index < outputs; ++index) {
    if (std::memcmp(&actual[index], &expected[index], sizeof(double)) != 0) {
      ++differences;
      max_absolute =
          std::max(max_absolute, std::abs(actual[index] - expected[index]));
    }
  }
  std::printf(
      "{\"measurement\":\"measured replay; includes transfers and host "
      "copies\","
      "\"different_values\":%zu,\"max_absolute_difference\":%.17g,"
      "\"repeats\":%d,\"warmups\":%d,\"mean_microseconds\":%.9g,"
      "\"input_bytes\":%zu,\"output_bytes\":%zu}\n",
      differences, max_absolute, kRepeats, kWarmups, seconds * 1e6 / kRepeats,
      inputs * sizeof(double), outputs * sizeof(double));
  return differences == 0 ? 0 : 2;
}

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
