#pragma once
// SPDX-License-Identifier: MIT
// Isolated FP64 GEMM accuracy/performance experiment; never a solver
// replacement.
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern char** environ;
namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void Check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
std::uint64_t Raw() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
    throw std::runtime_error("RAW clock failed");
  return std::uint64_t(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
std::string Quote(const std::string& value) {
  std::string result = "\"";
  for (unsigned char byte : value) {
    if (byte == '"' || byte == '\\') result += '\\';
    if (byte < 32 || byte > 126)
      throw std::runtime_error("Non-ASCII JSON string");
    result += char(byte);
  }
  return result + '"';
}
std::string Owner(const char* symbol) {
  void* pointer = dlsym(RTLD_DEFAULT, symbol);
  Dl_info information{};
  if (pointer == nullptr || !dladdr(pointer, &information))
    throw std::runtime_error("Library symbol owner missing");
  return std::filesystem::canonical(information.dli_fname).string();
}
double Median(std::vector<double> m_values) {
  std::sort(m_values.begin(), m_values.end());
  return m_values[m_values.size() / 2];
}
struct Options {
  bool execute_ = false, reuse_b_ = false, direct_ = false;
  bool profile_one_ = false, group_diagonals_ = false;
  std::string mode_ = "all", backend_ = "wmma", terms_ = "full";
  std::string split_ = "row";
  std::string input_, report_, output_, expect_cublas_;
  int m_ = 256, n_ = 256, k_ = 256;
  int calls_ = 20, samples_ = 7, warmups_ = 3, k_block_ = 0, digits_ = 0;
  int graph_batch_ = 1;
};
struct Matrices {
  int m_, n_, k_;
  std::vector<double> a_, b_, reference_;
};
std::size_t Product(std::uint64_t left, std::uint64_t right) {
  if (left == 0 || right == 0 || left > 2147483647 || right > 2147483647 ||
      left > (std::size_t(1) << 29) / right)
    throw std::runtime_error("Matrix size exceeds bounded FP64 allocation");
  return std::size_t(left * right);
}
Matrices Inputs(const Options& options) {
  std::array<std::uint64_t, 3> dimensions{std::uint64_t(options.m_),
                                          std::uint64_t(options.n_),
                                          std::uint64_t(options.k_)};
  std::ifstream input;
  if (!options.input_.empty()) {
    input.open(options.input_, std::ios::binary);
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    if (magic != std::array<char, 8>{'T', 'E', 'G', 'E', 'M', 'M', '1', '\0'})
      throw std::runtime_error("Invalid TEGEMM1 magic");
    static_assert(std::endian::native == std::endian::little);
    input.read(reinterpret_cast<char*>(dimensions.data()), sizeof(dimensions));
    if (!input) throw std::runtime_error("Incomplete dimensions");
  }
  const auto a_count = Product(dimensions[0], dimensions[2]);
  const auto b_count = Product(dimensions[2], dimensions[1]);
  const auto c_count = Product(dimensions[0], dimensions[1]);
  Matrices matrices{int(dimensions[0]),           int(dimensions[1]),
                    int(dimensions[2]),           std::vector<double>(a_count),
                    std::vector<double>(b_count), std::vector<double>(c_count)};
  if (input.is_open()) {
    for (auto* values : {&matrices.a_, &matrices.b_, &matrices.reference_}) {
      input.read(reinterpret_cast<char*>(values->data()),
                 values->size() * sizeof(double));
      if (!input) throw std::runtime_error("Incomplete matrix data");
    }
    if (input.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Unexpected trailing input bytes");
  } else {
    std::uint64_t random = 0x509013500131ULL;
    for (auto* values : {&matrices.a_, &matrices.b_})
      for (auto& value : *values) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        value = double(std::int64_t(random % 2049) - 1024) / 1024.0;
      }
    for (int column = 0; column < matrices.n_; ++column)
      for (int row = 0; row < matrices.m_; ++row) {
        long double sum = 0;
        for (int inner = 0; inner < matrices.k_; ++inner)
          sum += static_cast<long double>(
                     matrices.a_[row + std::size_t(inner) * matrices.m_]) *
                 matrices.b_[inner + std::size_t(column) * matrices.k_];
        matrices.reference_[row + std::size_t(column) * matrices.m_] =
            double(sum);
      }
  }
  for (const auto* values : {&matrices.a_, &matrices.b_, &matrices.reference_})
    for (double value : *values)
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite input/reference");
  return matrices;
}
std::uint64_t OrderedBits(double value) {
  auto bits = std::bit_cast<std::uint64_t>(value);
  return (bits >> 63) ? ~bits : bits | (std::uint64_t(1) << 63);
}

}  // namespace
