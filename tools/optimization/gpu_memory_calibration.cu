// SPDX-License-Identifier: MIT
// Standalone logical SM load/store calibration; no equilibrium calculations.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cuda/std/array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kBlockSize = 256;
constexpr int kDefaultPasses = 128;
constexpr int kDefaultSamples = 9;
constexpr int kDefaultWarmups = 3;
constexpr std::uint64_t kSeed = 731984;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

// Each kernel touches the WHOLE working set once. Repetition is across kernels,
// not within each CTA (which would accidentally make a DRAM case cache-hot).
// The opaque volatile asm forces one vector load and store, with no FP work.
__global__ void CopyVectors(const uint4* input, uint4* m_output,
                            std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  cuda::std::array<unsigned int, 4> value{};
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
               : "l"(input + index)
               : "memory");
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(m_output + index), "r"(value[0]), "r"(value[1]),
                 "r"(value[2]), "r"(value[3])
               : "memory");
}

std::uint64_t Pattern(std::size_t index) {
  std::uint64_t value = index + kSeed + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::uint64_t Checksum(const std::vector<std::uint64_t>& input) {
  std::uint64_t result = 14695981039346656037ULL;
  for (std::uint64_t value : input) {
    result ^= value;
    result *= 1099511628211ULL;
  }
  return result;
}

double EpochSeconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string Quoted(const std::string& value) {
  std::ostringstream result;
  result << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') result << '\\';
    if (character < 32 || character > 126) {
      result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << int(character) << std::dec;
    } else {
      result << character;
    }
  }
  result << '"';
  return result.str();
}

struct Regime {
  const char* name_;
  std::size_t target_working_bytes_;
  std::size_t vectors_;
  uint4* input_ = nullptr;
  uint4* output_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  std::size_t graph_nodes_ = 0;
  std::uint64_t expected_checksum_ = 0;
};

class Resources {
 public:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
  std::vector<Regime> regimes_;

  void Initialize() {
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&end_));
    regimes_.reserve(3);
  }
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto& regime : regimes_) {
      if (regime.executable_ != nullptr)
        cudaGraphExecDestroy(regime.executable_);
      if (regime.graph_ != nullptr) cudaGraphDestroy(regime.graph_);
      if (regime.input_ != nullptr) cudaFree(regime.input_);
      if (regime.output_ != nullptr) cudaFree(regime.output_);
    }
    if (start_ != nullptr) cudaEventDestroy(start_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  void Enqueue(const Regime& regime) {
    const auto blocks = static_cast<unsigned int>(
        (regime.vectors_ + kBlockSize - 1) / kBlockSize);
    CopyVectors<<<blocks, kBlockSize, 0, stream_>>>(
        regime.input_, regime.output_, regime.vectors_);
    Check(cudaGetLastError());
  }

  void Add(const char* name, std::size_t working_bytes, int passes) {
    const std::size_t vectors =
        std::max(std::size_t{1}, working_bytes / (2 * sizeof(uint4)));
    regimes_.push_back({name, working_bytes, vectors});
    auto& regime = regimes_.back();
    const std::size_t array_bytes = vectors * sizeof(uint4);
    Check(cudaMalloc(&regime.input_, array_bytes));
    Check(cudaMalloc(&regime.output_, array_bytes));
    std::vector<std::uint64_t> input(array_bytes / sizeof(std::uint64_t));
    for (std::size_t index = 0; index < input.size(); ++index)
      input[index] = Pattern(index);
    regime.expected_checksum_ = Checksum(input);
    Check(cudaMemcpyAsync(regime.input_, input.data(), array_bytes,
                          cudaMemcpyHostToDevice, stream_));
    Check(cudaMemsetAsync(regime.output_, 0, array_bytes, stream_));
    Enqueue(regime);
    Check(cudaStreamSynchronize(stream_));
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      for (int pass = 0; pass < passes; ++pass) Enqueue(regime);
      const auto status = cudaStreamEndCapture(stream_, &regime.graph_);
      capturing = false;
      Check(status);
      Check(cudaGraphGetNodes(regime.graph_, nullptr, &regime.graph_nodes_));
      if (regime.graph_nodes_ != static_cast<std::size_t>(passes))
        throw std::runtime_error("Unexpected captured graph node count");
      Check(cudaGraphInstantiate(&regime.executable_, regime.graph_, 0));
    } catch (...) {
      if (capturing) cudaStreamEndCapture(stream_, &regime.graph_);
      throw;
    }
  }

  std::uint64_t Verify(const Regime& regime) {
    std::vector<std::uint64_t> output(regime.vectors_ * 2);
    Check(cudaMemcpyAsync(output.data(), regime.output_,
                          output.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    for (std::size_t index = 0; index < output.size(); ++index)
      if (output[index] != Pattern(index))
        throw std::runtime_error("Output mismatch at word " +
                                 std::to_string(index));
    const auto checksum = Checksum(output);
    if (checksum != regime.expected_checksum_)
      throw std::runtime_error("Output checksum mismatch");
    return checksum;
  }
};

struct Sample {
  int round_;
  int order_;
  int regime_;
  bool graph_;
  float milliseconds_;
  double started_at_;
  double finished_at_;
  std::uint64_t checksum_;
};

int ReadInteger(const char* argument, int minimum, int maximum) {
  std::size_t consumed;
  const int value = std::stoi(argument, &consumed);
  if (consumed != std::string(argument).size() || value < minimum ||
      value > maximum)
    throw std::runtime_error("Integer argument outside allowed range");
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int device = 0;
    int passes = kDefaultPasses;
    int rounds = kDefaultSamples;
    int warmups = kDefaultWarmups;
    for (int index = 1; index < argc; index += 2) {
      if (index + 1 == argc) throw std::runtime_error("Option requires value");
      const std::string option = argv[index];
      if (option == "--device")
        device = ReadInteger(argv[index + 1], 0, 63);
      else if (option == "--passes")
        passes = ReadInteger(argv[index + 1], 1, 1024);
      else if (option == "--samples")
        rounds = ReadInteger(argv[index + 1], 1, 100);
      else if (option == "--warmups")
        warmups = ReadInteger(argv[index + 1], 1, 100);
      else
        throw std::runtime_error("Unknown option: " + option);
    }
    Check(cudaSetDevice(device));
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, device));
    if (properties.l2CacheSize <= 0)
      throw std::runtime_error("Device did not report an L2 cache size");
    int runtime_version;
    int driver_version;
    Check(cudaRuntimeGetVersion(&runtime_version));
    Check(cudaDriverGetVersion(&driver_version));
    std::size_t free_before;
    std::size_t total_memory;
    Check(cudaMemGetInfo(&free_before, &total_memory));
    const std::size_t l2_bytes = properties.l2CacheSize;
    const std::array<std::size_t, 3> target_bytes{
        std::size_t{1} << 20, l2_bytes / 2,
        std::max(4 * l2_bytes, std::size_t{256} << 20)};
    const std::size_t required =
        target_bytes[0] + target_bytes[1] + target_bytes[2];
    if (required > free_before / 2)
      throw std::runtime_error(
          "Calibration exceeds half available device memory");
    Resources resources;
    resources.Initialize();
    resources.Add("tiny", target_bytes[0], passes);
    resources.Add("l2_candidate", target_bytes[1], passes);
    resources.Add("dram_candidate", target_bytes[2], passes);
    std::size_t free_after;
    std::size_t total_after;
    Check(cudaMemGetInfo(&free_after, &total_after));
    std::vector<Sample> samples;
    std::mt19937_64 generator(kSeed);
    for (int round = 0; round < rounds; ++round) {
      std::array<int, 6> order{0, 1, 2, 3, 4, 5};
      std::shuffle(order.begin(), order.end(), generator);
      for (int position = 0; position < 6; ++position) {
        const int regime_index = order[position] / 2;
        const bool graph = order[position] % 2 != 0;
        const auto& regime = resources.regimes_[regime_index];
        const double started_at = EpochSeconds();
        for (int warmup = 0; warmup < warmups; ++warmup)
          resources.Enqueue(regime);
        Check(cudaEventRecord(resources.start_, resources.stream_));
        if (graph) {
          Check(cudaGraphLaunch(regime.executable_, resources.stream_));
        } else {
          for (int pass = 0; pass < passes; ++pass) resources.Enqueue(regime);
        }
        Check(cudaEventRecord(resources.end_, resources.stream_));
        Check(cudaEventSynchronize(resources.end_));
        float elapsed;
        Check(cudaEventElapsedTime(&elapsed, resources.start_, resources.end_));
        if (!(elapsed > 0.0f))
          throw std::runtime_error("Nonpositive event time");
        const double finished_at = EpochSeconds();
        const auto checksum = resources.Verify(regime);
        samples.push_back({round, position, regime_index, graph, elapsed,
                           started_at, finished_at, checksum});
      }
    }
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (char byte : properties.uuid.bytes)
      uuid << std::setw(2) << int(static_cast<unsigned char>(byte));
    std::array<char, 32> pci_bus{};
    Check(cudaDeviceGetPCIBusId(pci_bus.data(), pci_bus.size(), device));
    std::cout << std::setprecision(17)
              << "{\"scope\":\"logical SM vector load/store; not physical "
                 "bandwidth or solver roofline\","
              << "\"kernel\":\"ld.global.cg.v4.u32 + st.global.cg.v4.u32; no "
                 "FP arithmetic\","
              << "\"device\":{\"ordinal\":" << device
              << ",\"name\":" << Quoted(properties.name)
              << ",\"uuid_hex\":" << Quoted(uuid.str())
              << ",\"pci_bus_id\":" << Quoted(pci_bus.data())
              << ",\"cc_major\":" << properties.major
              << ",\"cc_minor\":" << properties.minor
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"l2_bytes\":" << l2_bytes
              << ",\"total_memory_bytes\":" << total_memory
              << ",\"free_before_bytes\":" << free_before
              << ",\"free_after_allocation_bytes\":" << free_after
              << ",\"runtime_version\":" << runtime_version
              << ",\"driver_version\":" << driver_version << "},"
              << "\"seed\":" << kSeed << ",\"passes_per_sample\":" << passes
              << ",\"warmup_full_passes_per_sample\":" << warmups
              << ",\"sample_rounds\":" << rounds
              << ",\"block_threads\":" << kBlockSize
              << ",\"vector_bytes\":16,\"verification\":\"every word after "
                 "every timed sample, untimed D2H\","
              << "\"timing_scope\":\"CUDA events around full pass chain; "
                 "direct includes host submission bubbles; graph excludes "
                 "capture/instantiate\","
              << "\"telemetry\":\"external runner records actual nvidia-smi "
                 "clocks/power when available\","
              << "\"regimes\":[";
    for (int index = 0; index < 3; ++index) {
      const auto& regime = resources.regimes_[index];
      const std::size_t working = 2 * regime.vectors_ * sizeof(uint4);
      std::cout << (index == 0 ? "" : ",")
                << "{\"name\":" << Quoted(regime.name_)
                << ",\"target_working_bytes\":" << regime.target_working_bytes_
                << ",\"allocated_input_bytes\":" << working / 2
                << ",\"allocated_output_bytes\":" << working / 2
                << ",\"working_set_bytes\":" << working
                << ",\"logical_bytes_per_full_pass\":" << working
                << ",\"logical_bytes_per_sample\":" << working * passes
                << ",\"blocks_per_pass\":"
                << (regime.vectors_ + kBlockSize - 1) / kBlockSize
                << ",\"graph_kernel_nodes\":" << regime.graph_nodes_
                << ",\"expected_checksum_u64\":"
                << Quoted(std::to_string(regime.expected_checksum_)) << '}';
    }
    std::cout << "],\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      const auto& sample = samples[index];
      const auto& regime = resources.regimes_[sample.regime_];
      const double logical_bytes =
          double(2 * regime.vectors_ * sizeof(uint4)) * passes;
      std::cout << (index == 0 ? "" : ",") << "{\"round\":" << sample.round_
                << ",\"order\":" << sample.order_
                << ",\"regime\":" << Quoted(regime.name_)
                << ",\"mode\":" << Quoted(sample.graph_ ? "graph" : "direct")
                << ",\"cuda_event_ms\":" << sample.milliseconds_
                << ",\"logical_gigabytes_per_second\":"
                << logical_bytes / (sample.milliseconds_ * 1e6)
                << ",\"host_started_at\":" << sample.started_at_
                << ",\"host_timed_completion_at\":" << sample.finished_at_
                << ",\"checksum_u64\":"
                << Quoted(std::to_string(sample.checksum_))
                << ",\"all_words_verified\":true}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
