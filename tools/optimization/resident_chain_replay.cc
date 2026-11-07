// SPDX-License-Identifier: MIT
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gpu_fourier.h"
#include "gpu_resident.h"
#include "resident_replay_capture.h"

using vmecpp_experiment::Compare;
using vmecpp_experiment::Read;

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "Usage: resident_chain_replay inverse.bin fields.bin library.so");
    }
    auto inverse = Read<GpuFourierInput, GpuFourierOutput>(argv[1], "VMECGPU1");
    auto fields =
        Read<GpuResidentInput, GpuResidentOutput>(argv[2], "VMECGPU5");
    const auto geometry_values =
        std::accumulate(fields.input_.sizes_.begin(),
                        fields.input_.sizes_.begin() + 18, std::size_t{0});
    if (geometry_values != inverse.expected_.size() ||
        std::memcmp(inverse.expected_.data(), fields.values_.data(),
                    geometry_values * sizeof(double)) != 0) {
      throw std::runtime_error(
          "Captures do not share the same intermediate geometry");
    }
    // A correct chain must not read these host geometry arrays at all.
    std::fill_n(fields.input_.data_.begin(), 18, nullptr);
    void* library = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      throw std::runtime_error(dlerror());
    }
    using Function = void (*)(const GpuFourierInput&, const GpuResidentInput&,
                              GpuResidentOutput&);
    auto run = reinterpret_cast<Function>(
        dlsym(library, "RunGpuResidentFourierFields"));
    if (run == nullptr) {
      throw std::runtime_error("Resident chain entry missing");
    }
    run(inverse.input_, fields.input_, fields.output_);
    std::size_t differences = Compare(fields.expected_, fields.actual_);
    if (differences != 0) {
      std::fprintf(stderr, "Different output values: %zu\n", differences);
      return 2;
    }
    constexpr int kWarmups = 10, kRepeats = 100;
    for (int i = 0; i < kWarmups; ++i) {
      run(inverse.input_, fields.input_, fields.output_);
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
      run(inverse.input_, fields.input_, fields.output_);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    differences = Compare(fields.expected_, fields.actual_);
    std::printf(
        "{\"measurement\":\"executed inverse-to-field chain; includes "
        "validation downloads, not solver "
        "timing\",\"different_values\":%zu,\"host_geometry_pointers\":\"null\","
        "\"repeats\":%d,\"warmups\":%d,\"mean_microseconds\":%.9g}\n",
        differences, kRepeats, kWarmups, seconds * 1e6 / kRepeats);
    return differences == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
