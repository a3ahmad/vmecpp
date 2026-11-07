// SPDX-License-Identifier: MIT
// Counter-matched calibration only; never used to calculate VMEC results.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/std/array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kThreads = 256;
constexpr int kChains = 8;
constexpr std::size_t kComputeCount = 1 << 18;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void Initialize(double* m_input, std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) m_input[index] = 0.25 + (index % 17) * 0x1p-10;
}

__global__ void StrictMixed(const double* input, double* m_output,
                            int iterations) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kComputeCount) return;
  cuda::std::array<double, kChains> values;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain)
    values[chain] = __dadd_rn(input[index], chain * 0x1p-7);
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain)
      values[chain] = __dadd_rn(__dmul_rn(values[chain], 0.999), 0.001);
  }
  double sum = 0.0;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain)
    sum = __dadd_rn(sum, values[chain]);
  m_output[index] = sum;
}

__global__ void InitializeVectors(uint4* m_input, std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const unsigned int value = static_cast<unsigned int>(index);
    m_input[index] = make_uint4(value, value ^ 0x731984U, ~value, value + 17U);
  }
}

__global__ void CopyVectors(const uint4* input, uint4* m_output,
                            std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  unsigned int first, second, third, fourth;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(first), "=r"(second), "=r"(third), "=r"(fourth)
               : "l"(input + index)
               : "memory");
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(m_output + index), "r"(first), "r"(second), "r"(third),
                 "r"(fourth)
               : "memory");
}

struct Resources {
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    if (input_ != nullptr) cudaFree(input_);
    if (output_ != nullptr) cudaFree(output_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
};

int Argument(const char* text, int minimum, int maximum) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value < minimum || value > maximum)
    throw std::runtime_error("Calibration argument outside fixed bounds");
  return static_cast<int>(value);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "Usage: roofline-matched compute iterations[1..1024] | memory "
          "workingMiB[1..512]");
    const bool compute = std::string(argv[1]) == "compute";
    if (!compute && std::string(argv[1]) != "memory")
      throw std::runtime_error("Unknown calibration mode");
    const int parameter = Argument(argv[2], 1, compute ? 1024 : 512);
    const std::size_t bytes = compute ? kComputeCount * sizeof(double)
                                      : (std::size_t(parameter) << 20) / 2;
    const std::size_t count = compute ? kComputeCount : bytes / sizeof(uint4);
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0));
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
    Check(cudaMalloc(&resources.input_, bytes));
    Check(cudaMalloc(&resources.output_, bytes));
    if (compute)
      Initialize<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                   resources.stream_>>>(static_cast<double*>(resources.input_),
                                        count);
    else
      InitializeVectors<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                          resources.stream_>>>(
          static_cast<uint4*>(resources.input_), count);
    Check(cudaGetLastError());
    Check(cudaStreamSynchronize(resources.stream_));
    Check(cudaStreamBeginCapture(resources.stream_,
                                 cudaStreamCaptureModeThreadLocal));
    if (compute)
      StrictMixed<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                    resources.stream_>>>(
          static_cast<double*>(resources.input_),
          static_cast<double*>(resources.output_), parameter);
    else
      CopyVectors<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                    resources.stream_>>>(static_cast<uint4*>(resources.input_),
                                         static_cast<uint4*>(resources.output_),
                                         count);
    Check(cudaGetLastError());
    Check(cudaStreamEndCapture(resources.stream_, &resources.graph_));
    Check(cudaGraphInstantiate(&resources.executable_, resources.graph_, 0));
    for (int warmup = 0; warmup < 3; ++warmup)
      Check(cudaGraphLaunch(resources.executable_, resources.stream_));
    Check(cudaStreamSynchronize(resources.stream_));
    Check(cudaProfilerStart());
    Check(cudaGraphLaunch(resources.executable_, resources.stream_));
    Check(cudaStreamSynchronize(resources.stream_));
    Check(cudaProfilerStop());
    std::vector<unsigned char> output(bytes);
    Check(cudaMemcpy(output.data(), resources.output_, bytes,
                     cudaMemcpyDeviceToHost));
    if (compute) {
      std::array<double, 17> expected{};
      for (int seed = 0; seed < 17; ++seed) {
        for (int chain = 0; chain < kChains; ++chain) {
          double value = 0.25 + seed * 0x1p-10 + chain * 0x1p-7;
          for (int i = 0; i < parameter; ++i) value = value * 0.999 + 0.001;
          expected[seed] += value;
        }
      }
      for (std::size_t index = 0; index < count; ++index)
        if (std::memcmp(output.data() + index * sizeof(double),
                        &expected[index % 17], sizeof(double)) != 0)
          throw std::runtime_error("Strict calibration CPU-bit mismatch");
    } else {
      for (std::size_t index = 0; index < count; ++index) {
        const unsigned int value = static_cast<unsigned int>(index);
        const auto expected =
            make_uint4(value, value ^ 0x731984U, ~value, value + 17U);
        if (std::memcmp(output.data() + index * sizeof(uint4), &expected,
                        sizeof(uint4)) != 0)
          throw std::runtime_error("Memory calibration copy mismatch");
      }
    }
    std::printf(
        "{\"scope\":\"counter-matched calibration only\",\"mode\":\"%s\","
        "\"parameter\":%d,\"working_set_bytes\":%zu,\"l2_bytes\":%d,"
        "\"SMs\":%d,\"all_output_words_verified\":true,"
        "\"profiled_graph_launches\":1,\"warmup_graph_launches\":3}\n",
        compute ? "compute" : "memory", parameter, 2 * bytes,
        properties.l2CacheSize, properties.multiProcessorCount);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
