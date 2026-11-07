// SPDX-License-Identifier: MIT
// Adversarial checks for the original VMEC zero-sentinel/NaN extrema behavior.
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "gpu_resident.cu"

int main() {
  try {
    const double positive_zero = 0.0;
    const double negative_zero = -0.0;
    const double nan = std::bit_cast<double>(std::uint64_t{0x7ff8000000001234});
    const double signaling_nan =
        std::bit_cast<double>(std::uint64_t{0x7ff0000000001234});
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> cases{
        {1.0},
        {-1.0},
        {positive_zero},
        {negative_zero},
        {1.0, positive_zero, 2.0},
        {2.0, negative_zero, 1.0},
        {-1.0, positive_zero, -2.0},
        {positive_zero, nan, 1.0},
        {nan, 1.0, -1.0},
        {1.0, nan, -1.0},
        {negative_zero, signaling_nan, 3.0},
        {1.0, signaling_nan, -1.0},
        {infinity, -infinity},
        {infinity, positive_zero, 1.0},
        {std::numeric_limits<double>::denorm_min(),
         -std::numeric_limits<double>::denorm_min()}};
    std::mt19937_64 generator(0);
    for (int count : {31, 32, 33, 255, 256, 257, 32928}) {
      std::vector<double> values(count);
      for (auto& value : values) {
        value = static_cast<double>(generator() % 100000 + 1) / 1024.0 - 50.0;
      }
      cases.push_back(values);
      values[count / 2] = negative_zero;
      cases.push_back(values);
      values[count / 2] = nan;
      cases.push_back(values);
    }
    int checked = 0;
    for (const auto& values : cases) {
      std::array<double, 2> expected{0.0, 0.0};
      for (double value : values) {
        if (value < expected[0] || expected[0] == 0.0) {
          expected[0] = value;
        }
        if (value > expected[1] || expected[1] == 0.0) {
          expected[1] = value;
        }
      }
      ResidentWorkspace workspace;
      workspace.Resize(values.size(), 0, values.size(), 3);
      Check(cudaMemcpyAsync(workspace.input_, values.data(),
                            values.size() * sizeof(double),
                            cudaMemcpyHostToDevice, workspace.stream_));
      const int count = static_cast<int>(values.size());
      JacobianPartialsKernel<<<(count + 255) / 256, 256, 0,
                               workspace.stream_>>>(workspace.input_, count,
                                                    workspace.extrema_ + 2);
      Check(cudaGetLastError());
      JacobianExtremaKernel<<<1, 32, 0, workspace.stream_>>>(
          workspace.input_, count, workspace.extrema_ + 2, workspace.extrema_);
      Check(cudaGetLastError());
      std::array<double, 2> actual;
      Check(cudaMemcpyAsync(actual.data(), workspace.extrema_, sizeof(actual),
                            cudaMemcpyDeviceToHost, workspace.stream_));
      Check(cudaStreamSynchronize(workspace.stream_));
      if (std::memcmp(actual.data(), expected.data(), sizeof(actual)) != 0) {
        std::fprintf(stderr, "Extrema bit mismatch in case %d\n", checked);
        return 2;
      }
      ++checked;
    }
    std::printf(
        "Passed %d extrema cases, including signed zero, NaNs, infinities, "
        "subnormals, and warp boundaries\n",
        checked);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
