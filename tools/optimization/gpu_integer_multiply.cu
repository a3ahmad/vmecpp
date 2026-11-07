// SPDX-License-Identifier: MIT
#if !defined(VMECPP_INTEGER_MULTIPLY_CPU)
#include <cuda_runtime.h>

#include <cuda/std/array>
#endif

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_integer_multiply.h"

namespace {
struct Case {
  std::uint64_t left_;
  std::uint64_t right_;
  std::uint64_t expected_;
  std::uint64_t exact_expected_;
};
static_assert(sizeof(Case) == 32);
std::vector<Case> Load(const char* path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic(8, '\0');
  file.read(magic.data(), 8);
  std::uint64_t count = 0;
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (magic != "VMECIMU1" || count == 0 || count > 2000000)
    throw std::runtime_error("Invalid multiply corpus header");
  std::vector<Case> cases(count);
  file.read(reinterpret_cast<char*>(cases.data()), count * sizeof(Case));
  if (!file || file.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("Incomplete multiply corpus");
  return cases;
}
#if !defined(VMECPP_INTEGER_MULTIPLY_CPU)
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Result {
  std::uint64_t candidate_;
  std::uint64_t native_;
  std::uint64_t integer_;
};
__global__ void CheckProducts(const Case* cases, Result* m_results, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const auto item = cases[index];
  const double left = IntegerMultiplyValue(item.left_);
  const double right = IntegerMultiplyValue(item.right_);
  std::uint64_t result = 0;
  const bool integer = TryIntegerMultiplyBits(item.left_, item.right_, result);
  const double candidate =
      integer ? IntegerMultiplyValue(result) : __dmul_rn(left, right);
  m_results[index] = {IntegerMultiplyBits(candidate),
                      IntegerMultiplyBits(__dmul_rn(left, right)), integer};
}
template <bool kInteger, int kChains, bool kVerify>
__global__ void Chains(const double* seeds, const double* factors,
                       double* m_output, std::uint64_t* m_stats, int threads,
                       int iterations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= threads) return;
  cuda::std::array<double, kChains> values;
  for (int chain = 0; chain < kChains; ++chain)
    values[chain] = seeds[index * kChains + chain];
  std::uint64_t accepted = 0, differences = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const double factor = factors[iteration & 63];
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      if constexpr (kVerify) {
        std::uint64_t result = 0;
        const bool integer =
            TryIntegerMultiplyBits(IntegerMultiplyBits(values[chain]),
                                   IntegerMultiplyBits(factor), result);
        const double candidate = integer ? IntegerMultiplyValue(result)
                                         : __dmul_rn(values[chain], factor);
        const double native = __dmul_rn(values[chain], factor);
        accepted += integer;
        differences +=
            IntegerMultiplyBits(candidate) != IntegerMultiplyBits(native);
        values[chain] = candidate;
      } else if constexpr (kInteger) {
        values[chain] = IntegerMultiply(values[chain], factor);
      } else {
        values[chain] = __dmul_rn(values[chain], factor);
      }
    }
  }
  for (int chain = 0; chain < kChains; ++chain)
    m_output[index * kChains + chain] = values[chain];
  if constexpr (kVerify) {
    m_stats[2 * index] = accepted;
    m_stats[2 * index + 1] = differences;
  }
}
void Launch(bool integer, bool verify, bool dependent, const double* seeds,
            const double* factors, double* m_output, std::uint64_t* m_stats,
            cudaStream_t stream) {
  const int threads = dependent ? 440 : 32768;
  const int iterations = dependent ? 512 : 256;
  const int blocks = (threads + 127) / 128;
  if (dependent) {
    if (verify)
      Chains<true, 1, true><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
    else if (integer)
      Chains<true, 1, false><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
    else
      Chains<false, 1, false><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
  } else {
    if (verify)
      Chains<true, 8, true><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
    else if (integer)
      Chains<true, 8, false><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
    else
      Chains<false, 8, false><<<blocks, 128, 0, stream>>>(
          seeds, factors, m_output, m_stats, threads, iterations);
  }
  Check(cudaGetLastError());
}
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2)
      throw std::runtime_error(
          "Usage: integer-multiply corpus.bin [proof|benchmark]");
    const auto cases = Load(argv[1]);
#if defined(VMECPP_INTEGER_MULTIPLY_CPU)
    std::size_t accepted = 0;
    for (const auto item : cases) {
      std::uint64_t result = 0;
      const bool integer =
          TryIntegerMultiplyBits(item.left_, item.right_, result);
      if (integer && (item.exact_expected_ == 0 || result != item.expected_))
        throw std::runtime_error(
            "CPU integer product differs from exact Fraction result");
      if (item.exact_expected_ != 0) {
        const double candidate =
            IntegerMultiply(IntegerMultiplyValue(item.left_),
                            IntegerMultiplyValue(item.right_));
        if (IntegerMultiplyBits(candidate) != item.expected_)
          throw std::runtime_error(
              "CPU candidate/fallback differs from exact Fraction product");
      }
      accepted += integer;
    }
    std::printf(
        "{\"CPU_cases\":%zu,\"integer_accepted\":%zu,\"fallback\":%zu,\"bit_"
        "mismatches\":0}\n",
        cases.size(), accepted, cases.size() - accepted);
#else
    if (argc != 3 || (std::string(argv[2]) != "proof" &&
                      std::string(argv[2]) != "benchmark"))
      throw std::runtime_error("Expected proof or benchmark mode");
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Case* device_cases = nullptr;
    Result* device_results = nullptr;
    Check(cudaMalloc(&device_cases, cases.size() * sizeof(Case)));
    Check(cudaMalloc(&device_results, cases.size() * sizeof(Result)));
    Check(cudaMemcpyAsync(device_cases, cases.data(),
                          cases.size() * sizeof(Case), cudaMemcpyHostToDevice,
                          stream));
    CheckProducts<<<(cases.size() + 127) / 128, 128, 0, stream>>>(
        device_cases, device_results, static_cast<int>(cases.size()));
    Check(cudaGetLastError());
    std::vector<Result> results(cases.size());
    Check(cudaMemcpyAsync(results.data(), device_results,
                          results.size() * sizeof(Result),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
      if (results[i].candidate_ != results[i].native_ ||
          (cases[i].exact_expected_ &&
           results[i].candidate_ != cases[i].expected_))
        throw std::runtime_error(
            "GPU candidate differs from literal native/Fraction product");
      accepted += results[i].integer_;
    }
    std::printf(
        "{\"GPU_proof_cases\":%zu,\"integer_accepted\":%zu,\"fallback\":%zu,"
        "\"all_bits_exact\":true}\n",
        cases.size(), accepted, cases.size() - accepted);
    cudaGraph_t proof_graph = nullptr;
    cudaGraphExec_t proof_executable = nullptr;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CheckProducts<<<(cases.size() + 127) / 128, 128, 0, stream>>>(
        device_cases, device_results, static_cast<int>(cases.size()));
    Check(cudaStreamEndCapture(stream, &proof_graph));
    Check(cudaGraphInstantiate(&proof_executable, proof_graph, 0));
    Check(cudaGraphLaunch(proof_executable, stream));
    std::vector<Result> graph_results(cases.size());
    Check(cudaMemcpyAsync(graph_results.data(), device_results,
                          graph_results.size() * sizeof(Result),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (results[i].candidate_ != graph_results[i].candidate_ ||
          results[i].native_ != graph_results[i].native_ ||
          results[i].integer_ != graph_results[i].integer_)
        throw std::runtime_error("Direct/graph product proof differs");
    }
    std::printf("{\"graph_product_proof_exact\":true}\n");
    Check(cudaGraphExecDestroy(proof_executable));
    Check(cudaGraphDestroy(proof_graph));
    Check(cudaFree(device_cases));
    Check(cudaFree(device_results));
    if (std::string(argv[2]) == "benchmark") {
      constexpr int kValues = 32768 * 8;
      std::vector<double> seeds(kValues), factors(64);
      std::mt19937_64 random(19631);
      for (double& seed : seeds) seed = 0.75 + (random() % 65536) / 65536.0;
      for (int i = 0; i < 64; ++i)
        factors[i] = 1.0 + (i % 2 ? -1.0 : 1.0) * (i + 1) * 0x1p-25;
      double *device_seeds = nullptr, *device_factors = nullptr,
             *output = nullptr;
      std::uint64_t* stats = nullptr;
      Check(cudaMalloc(&device_seeds, kValues * sizeof(double)));
      Check(cudaMalloc(&device_factors, 64 * sizeof(double)));
      Check(cudaMalloc(&output, kValues * sizeof(double)));
      Check(cudaMalloc(&stats, 2 * 32768 * sizeof(std::uint64_t)));
      Check(cudaMemcpyAsync(device_seeds, seeds.data(),
                            kValues * sizeof(double), cudaMemcpyHostToDevice,
                            stream));
      Check(cudaMemcpyAsync(device_factors, factors.data(), 64 * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      std::vector<double> expected_independent(kValues),
          expected_dependent(440);
      for (const bool dependent : {false, true}) {
        auto& expected = dependent ? expected_dependent : expected_independent;
        Launch(false, false, dependent, device_seeds, device_factors, output,
               stats, stream);
        Check(cudaMemcpyAsync(expected.data(), output,
                              expected.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
        Check(cudaStreamSynchronize(stream));
        Launch(true, true, dependent, device_seeds, device_factors, output,
               stats, stream);
        const int threads = dependent ? 440 : 32768;
        std::vector<std::uint64_t> observed(2 * threads);
        Check(cudaMemcpyAsync(observed.data(), stats,
                              observed.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost, stream));
        Check(cudaStreamSynchronize(stream));
        std::uint64_t total = 0;
        for (int i = 0; i < threads; ++i) {
          total += observed[2 * i];
          if (observed[2 * i + 1])
            throw std::runtime_error("Untimed chain product mismatch");
        }
        std::printf(
            "{\"chain_validation\":\"%s\",\"actual_integer_accepted\":%llu,"
            "\"actual_total_multiplies\":%llu,\"mismatches\":0}\n",
            dependent ? "dependent440" : "independent8",
            static_cast<unsigned long long>(total),
            static_cast<unsigned long long>(dependent ? 440ULL * 512
                                                      : 32768ULL * 8 * 256));
      }
      struct Job {
        bool integer_;
        bool dependent_;
        bool graph_;
        cudaGraph_t definition_ = nullptr;
        cudaGraphExec_t executable_ = nullptr;
      };
      std::vector<Job> jobs;
      for (const bool integer : {false, true})
        for (const bool dependent : {false, true})
          for (const bool graph : {false, true}) {
            Job job{integer, dependent, graph};
            if (graph) {
              Check(cudaStreamBeginCapture(stream,
                                           cudaStreamCaptureModeThreadLocal));
              Launch(integer, false, dependent, device_seeds, device_factors,
                     output, stats, stream);
              Check(cudaStreamEndCapture(stream, &job.definition_));
              Check(cudaGraphInstantiate(&job.executable_, job.definition_, 0));
            }
            jobs.push_back(job);
          }
      for (int repeat = -1; repeat < 7; ++repeat) {
        std::shuffle(jobs.begin(), jobs.end(), random);
        for (const auto& job : jobs) {
          cudaEvent_t begin = nullptr, end = nullptr;
          Check(cudaEventCreate(&begin));
          Check(cudaEventCreate(&end));
          Check(cudaEventRecord(begin, stream));
          if (job.graph_)
            Check(cudaGraphLaunch(job.executable_, stream));
          else
            Launch(job.integer_, false, job.dependent_, device_seeds,
                   device_factors, output, stats, stream);
          Check(cudaEventRecord(end, stream));
          Check(cudaEventSynchronize(end));
          float ms = 0.0f;
          Check(cudaEventElapsedTime(&ms, begin, end));
          const auto& expected =
              job.dependent_ ? expected_dependent : expected_independent;
          std::vector<double> observed(expected.size());
          Check(cudaMemcpyAsync(observed.data(), output,
                                observed.size() * sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
          Check(cudaStreamSynchronize(stream));
          for (std::size_t i = 0; i < expected.size(); ++i)
            if (IntegerMultiplyBits(observed[i]) !=
                IntegerMultiplyBits(expected[i]))
              throw std::runtime_error(
                  "Timed chain output differs from native reference");
          if (repeat >= 0)
            std::printf(
                "{\"diagnostic_trial\":%d,\"integer\":%d,\"dependent\":%d,"
                "\"graph\":%d,\"CUDAevent_ms\":%.9g}\n",
                repeat, job.integer_, job.dependent_, job.graph_, ms);
          Check(cudaEventDestroy(begin));
          Check(cudaEventDestroy(end));
        }
      }
      for (const auto& job : jobs) {
        if (job.executable_) Check(cudaGraphExecDestroy(job.executable_));
        if (job.definition_) Check(cudaGraphDestroy(job.definition_));
      }
      Check(cudaFree(stats));
      Check(cudaFree(output));
      Check(cudaFree(device_factors));
      Check(cudaFree(device_seeds));
    }
    Check(cudaStreamDestroy(stream));
#endif
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
