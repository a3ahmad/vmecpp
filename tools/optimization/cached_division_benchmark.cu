// SPDX-License-Identifier: MIT
// Standalone proof/performance experiment; no solver integration.
#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_cached_division.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Input {
  double numerator_;
  double denominator_;
};
struct Record {
  double numerator_;
  double denominator_;
  double reciprocal_;
  double result_;
  double reference_;
  std::uint64_t certified_;
};
static_assert(sizeof(Record) == 48);

__global__ void Evaluate(const Input* inputs, Record* m_records, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const auto input = inputs[index];
  const double reciprocal = __ddiv_rn(1.0, input.denominator_);
  const auto result = vmecpp_cached_division_experiment::Divide(
      input.numerator_, input.denominator_, reciprocal);
  m_records[index] = {input.numerator_,
                      input.denominator_,
                      reciprocal,
                      result.value_,
                      input.numerator_ / input.denominator_,
                      result.certified_};
}

// Independent chains, each with a serial recurrence resembling Thomas RHS.
// Reciprocals are precomputed outside the timed kernel, as at factor refresh.
template <bool kCertified>
__global__ void Chain(const Input* inputs, const double* reciprocals,
                      double* m_outputs, int chains, int rows, int repeats) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= chains) return;
  double result = 0.125;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (int row = 0; row < rows; ++row) {
      const int position = row * chains + index;
      const auto input = inputs[position];
      const double numerator = input.numerator_ - result * 0.125;
      if constexpr (kCertified) {
        result = vmecpp_cached_division_experiment::Divide(
                     numerator, input.denominator_, reciprocals[position])
                     .value_;
      } else {
        result = numerator / input.denominator_;
      }
    }
  }
  m_outputs[index] = result;
}

template <bool kCertified>
float TimeChain(const Input* inputs, const double* reciprocals,
                double* m_outputs, int chains, int rows, int repeats) {
  cudaEvent_t begin;
  cudaEvent_t end;
  Check(cudaEventCreate(&begin));
  Check(cudaEventCreate(&end));
  Check(cudaEventRecord(begin));
  Chain<kCertified><<<(chains + 31) / 32, 32>>>(inputs, reciprocals, m_outputs,
                                                chains, rows, repeats);
  Check(cudaGetLastError());
  Check(cudaEventRecord(end));
  Check(cudaEventSynchronize(end));
  float elapsed = 0.0f;
  Check(cudaEventElapsedTime(&elapsed, begin, end));
  Check(cudaEventDestroy(begin));
  Check(cudaEventDestroy(end));
  return elapsed;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: benchmark RECORDS.bin");
    std::mt19937_64 generator(982434);
    std::vector<Input> inputs;
    for (int i = 0; i < 200000; ++i) {
      inputs.push_back({std::bit_cast<double>(generator()),
                        std::bit_cast<double>(generator())});
    }
    const std::array<double, 16> adversarial{
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.0,
        7.0,
        0x1p-1022,
        -0x1p-1022,
        0x1p-1074,
        -0x1p-1074,
        0x1.fffffffffffffp+1023,
        -0x1.fffffffffffffp+1023,
        INFINITY,
        -INFINITY,
        std::bit_cast<double>(std::uint64_t{0x7ff8000000001234}),
        std::bit_cast<double>(std::uint64_t{0x7ff0000000001234})};
    for (double numerator : adversarial) {
      for (double denominator : adversarial)
        inputs.push_back({numerator, denominator});
    }
    for (int exponent = -1022; exponent <= 1023; ++exponent) {
      const double value = std::ldexp(1.0, exponent);
      for (double denominator : {value, nextafter(value, 0.0),
                                 nextafter(value, INFINITY), 1.0, 3.0, 7.0}) {
        for (double numerator :
             {value, nextafter(value, 0.0), nextafter(value, INFINITY)}) {
          for (double numerator_sign : {-1.0, 1.0})
            for (double denominator_sign : {-1.0, 1.0})
              inputs.push_back(
                  {numerator * numerator_sign, denominator * denominator_sign});
        }
      }
    }
    Input* device_inputs = nullptr;
    Record* device_records = nullptr;
    const int count = static_cast<int>(inputs.size());
    Check(cudaMalloc(&device_inputs, inputs.size() * sizeof(Input)));
    Check(cudaMalloc(&device_records, inputs.size() * sizeof(Record)));
    Check(cudaMemcpy(device_inputs, inputs.data(),
                     inputs.size() * sizeof(Input), cudaMemcpyHostToDevice));
    Evaluate<<<(count + 127) / 128, 128>>>(device_inputs, device_records,
                                           count);
    Check(cudaGetLastError());
    std::vector<Record> records(inputs.size());
    Check(cudaMemcpy(records.data(), device_records,
                     records.size() * sizeof(Record), cudaMemcpyDeviceToHost));
    std::uint64_t certified = 0;
    std::uint64_t mismatches = 0;
    for (const auto& record : records) {
      certified += record.certified_;
      if (std::bit_cast<std::uint64_t>(record.result_) !=
          std::bit_cast<std::uint64_t>(record.reference_))
        ++mismatches;
    }
    std::ofstream output(argv[1], std::ios::binary);
    output.write(reinterpret_cast<const char*>(records.data()),
                 records.size() * sizeof(Record));
    output.close();
    if (!output) throw std::runtime_error("Record write failed");
    Check(cudaFree(device_records));
    Check(cudaFree(device_inputs));

    constexpr int kChains = 264;
    constexpr int kRows = 99;
    constexpr int kRepeats = 100;
    inputs.clear();
    std::vector<double> reciprocals;
    std::uniform_real_distribution<double> numerator_distribution(-1.0, 1.0);
    std::uniform_real_distribution<double> denominator_distribution(1.0, 2.0);
    for (int index = 0; index < kChains * kRows; ++index) {
      const double denominator = denominator_distribution(generator);
      inputs.push_back({numerator_distribution(generator), denominator});
      reciprocals.push_back(1.0 / denominator);
    }
    double* device_reciprocals = nullptr;
    double* device_outputs = nullptr;
    Check(cudaMalloc(&device_inputs, inputs.size() * sizeof(Input)));
    Check(cudaMalloc(&device_reciprocals, reciprocals.size() * sizeof(double)));
    Check(cudaMalloc(&device_outputs, 2 * kChains * sizeof(double)));
    Check(cudaMemcpy(device_inputs, inputs.data(),
                     inputs.size() * sizeof(Input), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(device_reciprocals, reciprocals.data(),
                     reciprocals.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    TimeChain<false>(device_inputs, device_reciprocals, device_outputs, kChains,
                     kRows, 1);
    TimeChain<true>(device_inputs, device_reciprocals, device_outputs + kChains,
                    kChains, kRows, 1);
    std::array<float, 5> baseline_times{};
    std::array<float, 5> candidate_times{};
    for (int repeat = 0; repeat < 5; ++repeat) {
      if (repeat % 2 == 0) {
        baseline_times[repeat] =
            TimeChain<false>(device_inputs, device_reciprocals, device_outputs,
                             kChains, kRows, kRepeats);
        candidate_times[repeat] =
            TimeChain<true>(device_inputs, device_reciprocals,
                            device_outputs + kChains, kChains, kRows, kRepeats);
      } else {
        candidate_times[repeat] =
            TimeChain<true>(device_inputs, device_reciprocals,
                            device_outputs + kChains, kChains, kRows, kRepeats);
        baseline_times[repeat] =
            TimeChain<false>(device_inputs, device_reciprocals, device_outputs,
                             kChains, kRows, kRepeats);
      }
    }
    std::array<double, 2 * kChains> outputs{};
    Check(cudaMemcpy(outputs.data(), device_outputs, sizeof(outputs),
                     cudaMemcpyDeviceToHost));
    int chain_mismatches = 0;
    for (int index = 0; index < kChains; ++index)
      if (std::bit_cast<std::uint64_t>(outputs[index]) !=
          std::bit_cast<std::uint64_t>(outputs[index + kChains]))
        ++chain_mismatches;
    Check(cudaFree(device_inputs));
    Check(cudaFree(device_reciprocals));
    Check(cudaFree(device_outputs));
    std::cout << "{\"cases\":" << count << ",\"certified\":" << certified
              << ",\"bit_mismatches\":" << mismatches
              << ",\"chain_bit_mismatches\":" << chain_mismatches
              << ",\"chains\":" << kChains << ",\"rows\":" << kRows
              << ",\"chain_repeats\":" << kRepeats << ",\"baseline_ms\":[";
    for (int index = 0; index < 5; ++index)
      std::cout << (index == 0 ? "" : ",") << baseline_times[index];
    std::cout << "],\"candidate_ms\":[";
    for (int index = 0; index < 5; ++index)
      std::cout << (index == 0 ? "" : ",") << candidate_times[index];
    std::cout << "]}\n";
    return mismatches == 0 && chain_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
