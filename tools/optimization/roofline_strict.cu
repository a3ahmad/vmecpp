// SPDX-License-Identifier: MIT
// Empirical calibration only. These kernels never calculate VMEC results.
// Compile with --fmad=false; explicit FMA is a separate reference category.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/std/array>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t kStreamCount = std::size_t{1} << 23;
constexpr std::size_t kComputeCount = std::size_t{1} << 18;
constexpr int kChains = 8;
constexpr int kBlockSize = 256;
constexpr int kSampleRounds = 3;
enum class Operation { kAdd, kMultiply, kStrictMixed, kExplicitFma };

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void Initialize(double* m_a, double* m_b, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  m_a[i] = 0.25 + (i % 17) * 0.001;
  m_b[i] = 0.5;
}
__global__ void Stream(const double* a, const double* b, double* m_c,
                       std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) m_c[i] = __dadd_rn(a[i], __dmul_rn(0.125, b[i]));
}

template <Operation Mode>
__global__ void Independent(const double* input, double* m_output,
                            std::size_t count, int iterations) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  cuda::std::array<double, kChains> values;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain)
    values[chain] = __dadd_rn(input[i], 0.01 * chain);
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      if constexpr (Mode == Operation::kAdd)
        values[chain] = __dadd_rn(values[chain], 0.000001);
      else if constexpr (Mode == Operation::kMultiply)
        values[chain] = __dmul_rn(values[chain], 0.999999);
      else if constexpr (Mode == Operation::kStrictMixed)
        values[chain] = __dadd_rn(__dmul_rn(values[chain], 0.999), 0.001);
      else
        values[chain] = __fma_rn(values[chain], 0.999, 0.001);
    }
  }
  double checksum = 0.0;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain)
    checksum = __dadd_rn(checksum, values[chain]);
  m_output[i] = checksum;
}

struct ProbeResult {
  unsigned long long start_;
  unsigned long long end_;
  double checksum_;
};

__device__ unsigned long long ClockAfter(double value) {
  unsigned long long result;
  // The predicate genuinely consumes the last FP64 result, preventing the
  // trailing clock read from being scheduled before the dependent chain.
  asm volatile(
      "{ .reg .pred p; setp.eq.f64 p, %1, %1; mov.u64 %0, 0; "
      "@p mov.u64 %0, %%clock64; }"
      : "=l"(result)
      : "d"(value)
      : "memory");
  return result;
}
__global__ void DependentAdd(const double* seeds, int iterations,
                             ProbeResult* m_result) {
  const auto start = clock64();
  // Equal seed values selected by the clock bit force start -> load -> loop
  // ordering without changing the arithmetic input across samples.
  double value = seeds[start & 1ULL];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) value = __dadd_rn(value, 0.000001);
  const auto end = ClockAfter(value);
  m_result->start_ = start;
  m_result->end_ = end;
  m_result->checksum_ = value;
}

struct Resources {
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
  std::array<double*, 3> arrays_{};
  double* seeds_ = nullptr;
  ProbeResult* probes_ = nullptr;
  void Initialize(int repeats) {
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&end_));
    for (auto& pointer : arrays_)
      Check(cudaMalloc(&pointer, kStreamCount * sizeof(double)));
    Check(cudaMalloc(&seeds_, 2 * sizeof(double)));
    Check(cudaMalloc(&probes_, 2 * repeats * sizeof(ProbeResult)));
    const std::array<double, 2> seeds{0.25, 0.25};
    Check(cudaMemcpyAsync(seeds_, seeds.data(), sizeof(seeds),
                          cudaMemcpyHostToDevice, stream_));
    ::Initialize<<<(kStreamCount + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
                   stream_>>>(arrays_[0], arrays_[1], kStreamCount);
    Check(cudaGetLastError());
    Check(cudaStreamSynchronize(stream_));
  }
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto* pointer : arrays_)
      if (pointer != nullptr) cudaFree(pointer);
    if (seeds_ != nullptr) cudaFree(seeds_);
    if (probes_ != nullptr) cudaFree(probes_);
    if (start_ != nullptr) cudaEventDestroy(start_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
};

template <typename Launch>
double Time(Resources& m_resources, int repeats, Launch launch) {
  for (int i = 0; i < 2; ++i) launch(i % repeats);
  Check(cudaStreamSynchronize(m_resources.stream_));
  Check(cudaEventRecord(m_resources.start_, m_resources.stream_));
  for (int i = 0; i < repeats; ++i) launch(i);
  Check(cudaEventRecord(m_resources.end_, m_resources.stream_));
  Check(cudaEventSynchronize(m_resources.end_));
  float milliseconds = 0.0F;
  Check(cudaEventElapsedTime(&milliseconds, m_resources.start_,
                             m_resources.end_));
  if (milliseconds <= 0.0F)
    throw std::runtime_error("Zero calibration interval");
  return milliseconds;
}

void LaunchCase(Resources& m_resources, int which, int iterations) {
  const auto stream = m_resources.stream_;
  const auto& p = m_resources.arrays_;
  const int blocks = (kComputeCount + kBlockSize - 1) / kBlockSize;
  if (which == 0)
    Stream<<<(kStreamCount + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
             stream>>>(p[0], p[1], p[2], kStreamCount);
  else if (which == 1)
    Independent<Operation::kAdd><<<blocks, kBlockSize, 0, stream>>>(
        p[0], p[2], kComputeCount, iterations);
  else if (which == 2)
    Independent<Operation::kMultiply><<<blocks, kBlockSize, 0, stream>>>(
        p[0], p[2], kComputeCount, iterations);
  else if (which == 3)
    Independent<Operation::kStrictMixed><<<blocks, kBlockSize, 0, stream>>>(
        p[0], p[2], kComputeCount, iterations);
  else
    Independent<Operation::kExplicitFma><<<blocks, kBlockSize, 0, stream>>>(
        p[0], p[2], kComputeCount, iterations);
  Check(cudaGetLastError());
}

int Argument(const char* text, int maximum) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 1 || value > maximum)
    throw std::runtime_error("Invalid bounded calibration argument");
  return static_cast<int>(value);
}
double Median(std::vector<unsigned long long> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return values.size() % 2 ? static_cast<double>(values[middle])
                           : 0.5 * static_cast<double>(values[middle - 1]) +
                                 0.5 * static_cast<double>(values[middle]);
}
void JsonString(const char* value) {
  std::putchar('"');
  for (const auto* p = value; *p != '\0'; ++p) {
    if (*p == '"' || *p == '\\') std::putchar('\\');
    std::putchar(*p);
  }
  std::putchar('"');
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 4)
      throw std::runtime_error(
          "Usage: roofline-strict [repeats<=20] [computeIterations<=1024] "
          "[dependentAdds<=262144]");
    const int repeats = argc > 1 ? Argument(argv[1], 20) : 10;
    const int iterations = argc > 2 ? Argument(argv[2], 1024) : 512;
    const int dependent_adds = argc > 3 ? Argument(argv[3], 262144) : 65536;
    const std::size_t requested_bytes = 3 * kStreamCount * sizeof(double) +
                                        2 * sizeof(double) +
                                        2 * repeats * sizeof(ProbeResult);
    if (requested_bytes >= (std::size_t{512} << 20))
      throw std::runtime_error("Calibration allocation exceeds bound");
    int device = 0;
    cudaDeviceProp properties{};
    Check(cudaGetDevice(&device));
    Check(cudaGetDeviceProperties(&properties, device));
    Resources resources;
    resources.Initialize(repeats);
    std::printf(
        "{\"scope\":\"empirical calibration only; no whole-solver roofline "
        "claim\","
        "\"compile_requirement\":\"--fmad=false; explicit RN intrinsics\","
        "\"timing_scope\":\"CUDA events bracket repeated kernels, including "
        "dispatch gaps; warmups excluded\","
        "\"stream_scope\":\"logical global-memory bytes, not measured DRAM "
        "transactions\","
        "\"compute_scope\":\"loop FLOPs only; "
        "seed/checksum/load/store/loop-control overhead remains in elapsed "
        "time\","
        "\"device\":");
    JsonString(properties.name);
    std::printf(
        ",\"compute_capability\":[%d,%d],\"sm_count\":%d,"
        "\"requested_device_allocation_bytes\":%zu,\"stream_elements\":%zu,"
        "\"compute_threads\":%zu,\"block_threads\":%d,\"chains_per_thread\":%d,"
        "\"loop_iterations\":%d,\"repeats\":%d,\"warmup_launches\":2,"
        "\"random_seed\":2077,\"sample_rounds\":%d,\"calibration_samples\":[\n",
        properties.major, properties.minor, properties.multiProcessorCount,
        requested_bytes, kStreamCount, kComputeCount, kBlockSize, kChains,
        iterations, repeats, kSampleRounds);
    const std::array<const char*, 5> names{
        "stream_strict_mul_add", "FP64_add_independent",
        "FP64_multiply_independent", "FP64_strict_mul_then_add_independent",
        "FP64_explicit_FMA_reference"};
    std::array<int, 5> order{0, 1, 2, 3, 4};
    std::mt19937 generator(2077);
    bool first = true;
    for (int sample = 0; sample < kSampleRounds; ++sample) {
      std::shuffle(order.begin(), order.end(), generator);
      for (int which : order) {
        const double milliseconds = Time(resources, repeats, [&](int) {
          LaunchCase(resources, which, iterations);
        });
        double checksum = 0.0;
        Check(cudaMemcpy(&checksum, resources.arrays_[2], sizeof(double),
                         cudaMemcpyDeviceToHost));
        if (!std::isfinite(checksum))
          throw std::runtime_error("Nonfinite compute checksum");
        const unsigned long long bytes =
            which == 0 ? 3ULL * kStreamCount * sizeof(double) * repeats
                       : 2ULL * kComputeCount * sizeof(double) * repeats;
        const unsigned long long flops =
            which == 0
                ? 2ULL * kStreamCount * repeats
                : static_cast<unsigned long long>(kComputeCount) * kChains *
                      iterations * repeats * (which >= 3 ? 2 : 1);
        std::printf(
            "%s{\"category\":\"%s\",\"sample\":%d,\"cuda_event_milliseconds\":%"
            ".9f,"
            "\"logical_bytes\":%llu,\"loop_FLOPs\":%llu,\"checksum_thread0\":%."
            "17g,"
            "\"threads_per_launch\":%zu,\"chains_per_thread\":%d,"
            "\"iterations_per_chain\":%d,\"FP64_FLOPs_per_chain_iteration\":%d,"
            "\"derived_logical_GB_per_second\":%.9f,\"derived_loop_TFLOP_per_"
            "second\":%.9f}",
            first ? "" : ",\n", names[which], sample, milliseconds, bytes,
            flops, checksum, which == 0 ? kStreamCount : kComputeCount,
            which == 0 ? 1 : kChains, which == 0 ? 1 : iterations,
            which == 1 || which == 2 ? 1 : 2, bytes / (milliseconds * 1.0e6),
            flops / (milliseconds * 1.0e9));
        first = false;
      }
    }
    std::array<double, 2> probe_milliseconds{};
    for (int mode = 0; mode < 2; ++mode) {
      probe_milliseconds[mode] = Time(resources, repeats, [&](int repeat) {
        DependentAdd<<<1, 1, 0, resources.stream_>>>(
            resources.seeds_, mode == 0 ? 0 : dependent_adds,
            resources.probes_ + mode * repeats + repeat);
        Check(cudaGetLastError());
      });
    }
    std::vector<ProbeResult> probes(2 * repeats);
    Check(cudaMemcpy(probes.data(), resources.probes_,
                     probes.size() * sizeof(ProbeResult),
                     cudaMemcpyDeviceToHost));
    std::array<std::vector<unsigned long long>, 2> cycles;
    for (int mode = 0; mode < 2; ++mode)
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto& probe = probes[mode * repeats + repeat];
        if (!std::isfinite(probe.checksum_) || probe.end_ < probe.start_)
          throw std::runtime_error("Invalid dependency probe");
        cycles[mode].push_back(probe.end_ - probe.start_);
      }
    std::printf(
        "\n],\"dependent_add_probe\":{"
        "\"scope\":\"one active lane; clock64 elapsed SM cycles include "
        "scheduling, loop control, seed load and trailing predicate; not pure "
        "instruction latency\","
        "\"clock_frequency\":\"not inferred from a nominal specification\","
        "\"dependent_adds_per_launch\":%d,\"launches_per_mode\":%d,"
        "\"baseline_cuda_event_milliseconds\":%.9f,\"chain_cuda_event_"
        "milliseconds\":%.9f,"
        "\"baseline_median_clock_cycles\":%.3f,\"chain_median_clock_cycles\":%."
        "3f,"
        "\"derived_baseline_adjusted_cycles_per_add\":%.9f,\"raw_probes\":[",
        dependent_adds, repeats, probe_milliseconds[0], probe_milliseconds[1],
        Median(cycles[0]), Median(cycles[1]),
        (Median(cycles[1]) - Median(cycles[0])) / dependent_adds);
    for (int i = 0; i < 2 * repeats; ++i)
      std::printf(
          "%s{\"mode\":\"%s\",\"start_clock\":%llu,\"end_clock\":%llu,\"clock_"
          "cycles\":%llu,\"checksum\":%.17g}",
          i == 0 ? "" : ",",
          i < repeats ? "zero_add_baseline" : "dependent_add_chain",
          probes[i].start_, probes[i].end_, probes[i].end_ - probes[i].start_,
          probes[i].checksum_);
    std::puts("]}}");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
