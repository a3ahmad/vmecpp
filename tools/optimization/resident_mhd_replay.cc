// SPDX-License-Identifier: MIT
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu_fourier.h"
#include "gpu_pipeline.h"
#include "gpu_resident.h"
#include "resident_replay_capture.h"

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      throw std::runtime_error(
          "Usage: resident_mhd_replay inverse.bin fields.bin constants.bin "
          "mhd.bin library.so");
    }
    auto inverse = vmecpp_experiment::Read<GpuFourierInput, GpuFourierOutput>(
        argv[1], "VMECGPU1");
    auto fields = vmecpp_experiment::Read<GpuResidentInput, GpuResidentOutput>(
        argv[2], "VMECGPU5");
    auto mhd = vmecpp_experiment::Read<GpuPipelineInput, GpuMhdOutput>(
        argv[4], "VMECGPU3");
    const auto geometry_size =
        std::accumulate(fields.input_.sizes_.begin(),
                        fields.input_.sizes_.begin() + 18, std::size_t{0});
    if (geometry_size != inverse.expected_.size() ||
        std::memcmp(inverse.expected_.data(), fields.values_.data(),
                    geometry_size * sizeof(double)) != 0) {
      throw std::runtime_error(
          "Inverse and field captures do not share geometry");
    }
    std::array<std::size_t, 38> field_offsets{};
    for (std::size_t i = 1; i < field_offsets.size(); ++i) {
      field_offsets[i] = field_offsets[i - 1] + fields.output_.sizes_[i - 1];
    }
    const std::array<std::pair<int, int>, 20> shared{
        {{0, 18}, {1, 19}, {2, 20}, {3, 21}, {4, 22},  {5, 23}, {6, 24},
         {7, 28}, {8, 29}, {10, 0}, {11, 1}, {12, 2},  {13, 3}, {14, 8},
         {15, 9}, {16, 7}, {17, 4}, {18, 5}, {19, 10}, {20, 11}}};
    for (const auto& [source, target] : shared) {
      if (mhd.input_.sizes_[source] != fields.output_.sizes_[target] ||
          std::memcmp(mhd.input_.data_[source],
                      fields.expected_.data() + field_offsets[target],
                      mhd.input_.sizes_[source] * sizeof(double)) != 0) {
        throw std::runtime_error(
            "Field and MHD references do not share physical state");
      }
    }
    GpuResidentForceConstants constants{};
    std::ifstream file(argv[3], std::ios::binary);
    std::array<char, 8> magic{};
    file.read(magic.data(), 8);
    if (std::memcmp(magic.data(), "VMECGPC1", 8) != 0) {
      throw std::runtime_error("Invalid constants capture");
    }
    file.read(reinterpret_cast<char*>(&constants.adiabatic_index_),
              sizeof(double));
    file.read(reinterpret_cast<char*>(constants.sizes_.data()),
              sizeof(constants.sizes_));
    const auto constant_count = constants.sizes_[0] + constants.sizes_[1];
    if (!file || constant_count > 1000000) {
      throw std::runtime_error("Invalid constant dimensions");
    }
    std::vector<double> values(constant_count);
    file.read(reinterpret_cast<char*>(values.data()),
              constant_count * sizeof(double));
    if (!file) {
      throw std::runtime_error("Incomplete constants capture");
    }
    constants.data_ = {values.data(), values.data() + constants.sizes_[0]};
    GpuResidentForceOutput output{};
    std::vector<double> expected = mhd.expected_;
    std::copy_n(mhd.output_.sizes_.begin(), 12, output.sizes_.begin());
    for (std::size_t i = 0; i < 4; ++i) {
      output.sizes_[12 + i] = mhd.input_.sizes_[31 + i];
      expected.insert(expected.end(), mhd.input_.data_[31 + i],
                      mhd.input_.data_[31 + i] + mhd.input_.sizes_[31 + i]);
    }
    output.sizes_[16] = mhd.input_.sizes_[9];
    expected.insert(expected.end(), mhd.input_.data_[9],
                    mhd.input_.data_[9] + mhd.input_.sizes_[9]);
    std::vector<double> actual(expected.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < output.data_.size(); ++i) {
      output.data_[i] = actual.data() + offset;
      offset += output.sizes_[i];
    }
    // Force the implementation to obtain geometry from its GPU predecessor.
    std::fill_n(fields.input_.data_.begin(), 18, nullptr);
    void* library = dlopen(argv[5], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      throw std::runtime_error(dlerror());
    }
    using Function =
        void (*)(const GpuFourierInput&, const GpuResidentInput&,
                 const GpuResidentForceConstants&, GpuResidentForceOutput&);
    auto run =
        reinterpret_cast<Function>(dlsym(library, "RunGpuResidentMhdChain"));
    if (run == nullptr) {
      throw std::runtime_error("Resident MHD chain entry missing");
    }
    const auto execute = [&] {
      run(inverse.input_, fields.input_, constants, output);
    };
    execute();
    auto differences = vmecpp_experiment::Compare(expected, actual);
    if (differences != 0) {
      std::fprintf(stderr, "Different values: %zu\n", differences);
      return 2;
    }
    constexpr int kWarmups = 10, kRepeats = 100;
    for (int i = 0; i < kWarmups; ++i) {
      execute();
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
      execute();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    differences = vmecpp_experiment::Compare(expected, actual);
    std::printf(
        "{\"measurement\":\"executed resident inverse-to-MHD chain; includes "
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
