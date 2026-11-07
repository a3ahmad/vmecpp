// SPDX-License-Identifier: MIT
// Independent correctness harness; the included logarithm is LGPL-2.1-or-later.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "gpu_pinned_cpu_log.h"

namespace {
struct Case {
  std::uint64_t input_;
  std::uint64_t expected_;
  std::uint64_t eligible_;
  std::uint64_t observed_;
};
struct Result {
  std::uint64_t value_;
  std::uint64_t eligible_;
};
static_assert(sizeof(Case) == 32 && sizeof(Result) == 16);
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void Evaluate(const Case* cases, Result* m_results,
                         std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  double result = 1234.5;
  const bool eligible = TryPinnedCpuLog(
      __longlong_as_double(static_cast<long long>(cases[index].input_)),
      result);
  m_results[index + 1] = {
      static_cast<std::uint64_t>(__double_as_longlong(result)), eligible};
}
}  // namespace

int main(int argc, char** argv) {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Case* device_cases = nullptr;
  Result* device_results = nullptr;
  auto cleanup = [&] {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (executable != nullptr) cudaGraphExecDestroy(executable);
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (device_results != nullptr) cudaFree(device_results);
    if (device_cases != nullptr) cudaFree(device_cases);
    if (stream != nullptr) cudaStreamDestroy(stream);
  };
  try {
    if (argc != 2) throw std::runtime_error("Usage: pinned-log corpus.bin");
    std::ifstream file(argv[1], std::ios::binary);
    std::array<char, 8> magic{};
    std::uint64_t count = 0;
    file.read(magic.data(), 8);
    file.read(reinterpret_cast<char*>(&count), 8);
    if (std::memcmp(magic.data(), "VMECGPL1", 8) || count > 5000000 || !count)
      throw std::runtime_error("Invalid GPU log corpus");
    std::vector<Case> cases(count);
    file.read(reinterpret_cast<char*>(cases.data()), count * sizeof(Case));
    if (!file || file.peek() != std::ifstream::traits_type::eof())
      throw std::runtime_error("Incomplete GPU log corpus");
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Check(cudaMalloc(&device_cases, count * sizeof(Case)));
    Check(cudaMalloc(&device_results, (count + 2) * sizeof(Result)));
    Check(cudaMemcpyAsync(device_cases, cases.data(), count * sizeof(Case),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaStreamSynchronize(stream));
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    Evaluate<<<(count + 127) / 128, 128, 0, stream>>>(device_cases,
                                                      device_results, count);
    Check(cudaGetLastError());
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&executable, graph, 0));
    std::vector<Result> direct;
    for (const bool captured : {false, true}) {
      Check(cudaMemsetAsync(device_results, 0xa5, (count + 2) * sizeof(Result),
                            stream));
      if (captured)
        Check(cudaGraphLaunch(executable, stream));
      else
        Evaluate<<<(count + 127) / 128, 128, 0, stream>>>(
            device_cases, device_results, count);
      Check(cudaGetLastError());
      std::vector<Result> actual(count + 2);
      Check(cudaMemcpyAsync(actual.data(), device_results,
                            actual.size() * sizeof(Result),
                            cudaMemcpyDeviceToHost, stream));
      std::vector<Case> unchanged(count);
      Check(cudaMemcpyAsync(unchanged.data(), device_cases,
                            unchanged.size() * sizeof(Case),
                            cudaMemcpyDeviceToHost, stream));
      Check(cudaStreamSynchronize(stream));
      if (std::memcmp(unchanged.data(), cases.data(), count * sizeof(Case)))
        throw std::runtime_error("Input corpus mutated");
      constexpr std::uint64_t kGuard = 0xa5a5a5a5a5a5a5a5ULL;
      for (const auto index : {std::size_t{0}, std::size_t{count + 1}})
        if (actual[index].value_ != kGuard || actual[index].eligible_ != kGuard)
          throw std::runtime_error("Output guard overwritten");
      std::size_t eligible = 0, observed = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (actual[i + 1].value_ != cases[i].expected_ ||
            actual[i + 1].eligible_ != cases[i].eligible_) {
          std::fprintf(
              stderr,
              "case %zu input=%016llx expected=%016llx "
              "actual=%016llx eligible=%llu/%llu\n",
              i, static_cast<unsigned long long>(cases[i].input_),
              static_cast<unsigned long long>(cases[i].expected_),
              static_cast<unsigned long long>(actual[i + 1].value_),
              static_cast<unsigned long long>(cases[i].eligible_),
              static_cast<unsigned long long>(actual[i + 1].eligible_));
          throw std::runtime_error("Pinned CPU log GPU mismatch");
        }
        eligible += actual[i + 1].eligible_;
        observed += cases[i].observed_;
      }
      if (captured && std::memcmp(actual.data(), direct.data(),
                                  actual.size() * sizeof(Result)))
        throw std::runtime_error("Direct/graph outputs differ");
      if (!captured) direct = actual;
      std::printf(
          "{\"graph\":%d,\"cases\":%llu,\"eligible\":%zu,"
          "\"declined_unchanged\":%llu,\"observed_CPU_pairs\":%zu,"
          "\"all_output_bits_exact\":true,\"guards_input_unchanged\":true}\n",
          captured, static_cast<unsigned long long>(count), eligible,
          static_cast<unsigned long long>(count - eligible), observed);
    }
    cleanup();
    return 0;
  } catch (const std::exception& error) {
    cleanup();
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
