// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_RESIDENT_REPLAY_CAPTURE_H_
#define VMECPP_EXPERIMENT_RESIDENT_REPLAY_CAPTURE_H_
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace vmecpp_experiment {
template <class Input, class Output>
struct Capture {
  Input input_{};
  Output output_{};
  std::vector<double> values_;
  std::vector<double> expected_;
  std::vector<double> actual_;
};

template <class Input, class Output>
Capture<Input, Output> Read(const char* path, const char* expected_magic) {
  Capture<Input, Output> result;
  std::ifstream file(path, std::ios::binary);
  std::array<char, 8> magic{};
  file.read(magic.data(), 8);
  if (std::memcmp(magic.data(), expected_magic, 8) != 0) {
    throw std::runtime_error("Capture magic mismatch");
  }
  auto& input = result.input_;
  auto& output = result.output_;
  file.read(reinterpret_cast<char*>(input.dimensions_.data()),
            sizeof(input.dimensions_));
  if constexpr (requires { input.scalars_; }) {
    file.read(reinterpret_cast<char*>(input.scalars_.data()),
              sizeof(input.scalars_));
  }
  file.read(reinterpret_cast<char*>(input.sizes_.data()), sizeof(input.sizes_));
  file.read(reinterpret_cast<char*>(output.sizes_.data()),
            sizeof(output.sizes_));
  if (!file) {
    throw std::runtime_error("Incomplete capture header");
  }
  std::size_t inputs = 0, outputs = 0;
  for (auto count : input.sizes_) {
    if (count > 10000000) {
      throw std::runtime_error("Invalid input size");
    }
    inputs += count;
  }
  for (auto count : output.sizes_) {
    if (count > 10000000) {
      throw std::runtime_error("Invalid output size");
    }
    outputs += count;
  }
  if (inputs > 10000000 || outputs > 10000000) {
    throw std::runtime_error("Capture too large");
  }
  result.values_.resize(inputs);
  result.expected_.resize(outputs);
  result.actual_.resize(outputs);
  file.read(reinterpret_cast<char*>(result.values_.data()),
            inputs * sizeof(double));
  file.read(reinterpret_cast<char*>(result.expected_.data()),
            outputs * sizeof(double));
  if (!file) {
    throw std::runtime_error("Incomplete capture data");
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < input.data_.size(); ++i) {
    input.data_[i] = result.values_.data() + offset;
    offset += input.sizes_[i];
  }
  offset = 0;
  for (std::size_t i = 0; i < output.data_.size(); ++i) {
    output.data_[i] = result.actual_.data() + offset;
    offset += output.sizes_[i];
  }
  return result;
}

std::size_t Compare(const std::vector<double>& expected,
                    const std::vector<double>& actual) {
  std::size_t differences = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(double)) != 0) {
      if (differences < 16) {
        std::fprintf(stderr, "Output %zu: expected %.17g actual %.17g\n", i,
                     expected[i], actual[i]);
      }
      ++differences;
    }
  }
  return differences;
}
}  // namespace vmecpp_experiment
#endif  // VMECPP_EXPERIMENT_RESIDENT_REPLAY_CAPTURE_H_
