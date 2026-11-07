// Diagnostic only: compare CUDA log with observed original CPU libm results.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

struct LogInput {
  std::uint64_t argument_;
  std::uint64_t reference_;
};

struct LogOutput {
  std::uint64_t logarithm_;
  std::uint64_t damping_numerator_;
};

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void LogProbeKernel(const LogInput* input, LogOutput* m_output,
                               std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const double value = log(__longlong_as_double(input[index].argument_));
  const double absolute = fabs(value);
  // Match std::min operand order, including its NaN behavior.
  const double numerator = 0.15 < absolute ? 0.15 : absolute;
  m_output[index].logarithm_ = __double_as_longlong(value);
  m_output[index].damping_numerator_ = __double_as_longlong(numerator);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Expected input and output paths");
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) % sizeof(LogInput) != 0)
      throw std::runtime_error("Invalid input size");
    std::vector<LogInput> records(static_cast<std::size_t>(size) /
                                  sizeof(LogInput));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(records.data()), size);
    if (!input) throw std::runtime_error("Incomplete input");
    std::vector<LogOutput> output(records.size());
    LogInput* device_input = nullptr;
    LogOutput* device_output = nullptr;
    Check(cudaMalloc(&device_input, records.size() * sizeof(LogInput)));
    Check(cudaMalloc(&device_output, output.size() * sizeof(LogOutput)));
    Check(cudaMemcpy(device_input, records.data(),
                     records.size() * sizeof(LogInput),
                     cudaMemcpyHostToDevice));
    LogProbeKernel<<<(records.size() + 127) / 128, 128>>>(
        device_input, device_output, records.size());
    Check(cudaGetLastError());
    Check(cudaMemcpy(output.data(), device_output,
                     output.size() * sizeof(LogOutput),
                     cudaMemcpyDeviceToHost));
    Check(cudaFree(device_output));
    Check(cudaFree(device_input));
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char*>(output.data()),
                 output.size() * sizeof(LogOutput));
    result.close();
    if (!result) throw std::runtime_error("Incomplete output");
    std::size_t log_differences = 0;
    std::size_t numerator_differences = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
      log_differences += records[i].reference_ != output[i].logarithm_;
      double reference;
      std::memcpy(&reference, &records[i].reference_, sizeof(reference));
      const double absolute = std::abs(reference);
      const double numerator = 0.15 < absolute ? 0.15 : absolute;
      std::uint64_t numerator_bits;
      std::memcpy(&numerator_bits, &numerator, sizeof(numerator));
      numerator_differences += numerator_bits != output[i].damping_numerator_;
    }
    std::printf(
        "{\"records\":%zu,\"log_bit_differences\":%zu,"
        "\"numerator_bit_differences\":%zu,"
        "\"scope\":\"diagnostic; not solver parity qualification\"}\n",
        records.size(), log_differences, numerator_differences);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
