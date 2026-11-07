// SPDX-License-Identifier: MIT
// Isolated correctness/proposal experiment. Never reports whole-solver speedup.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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

#include "gpu_lattice_prefix.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Api {
  void* library_;
  decltype(&CreateGpuLatticePrefix) create_;
  decltype(&DestroyGpuLatticePrefix) destroy_;
  decltype(&RunGpuLatticePrefix) run_;
  decltype(&RunGpuLatticePrefixCorrupted) corrupt_;
  decltype(&LaunchGpuLatticeOrdered) ordered_;
  decltype(&LaunchGpuLatticeOrderedSums) sums_;
  decltype(&ReadGpuLatticePrefixStats) statistics_;
  decltype(&ReadGpuLatticePrefixFlags) flags_;

  explicit Api(const char* path) {
    library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) throw std::runtime_error(dlerror());
    create_ = Symbol<decltype(create_)>("CreateGpuLatticePrefix");
    destroy_ = Symbol<decltype(destroy_)>("DestroyGpuLatticePrefix");
    run_ = Symbol<decltype(run_)>("RunGpuLatticePrefix");
    corrupt_ = Symbol<decltype(corrupt_)>("RunGpuLatticePrefixCorrupted");
    ordered_ = Symbol<decltype(ordered_)>("LaunchGpuLatticeOrdered");
    sums_ = Symbol<decltype(sums_)>("LaunchGpuLatticeOrderedSums");
    statistics_ = Symbol<decltype(statistics_)>("ReadGpuLatticePrefixStats");
    flags_ = Symbol<decltype(flags_)>("ReadGpuLatticePrefixFlags");
  }
  ~Api() { dlclose(library_); }
  template <typename T>
  T Symbol(const char* name) {
    auto result = reinterpret_cast<T>(dlsym(library_, name));
    if (result == nullptr) throw std::runtime_error(name);
    return result;
  }
};

struct Case {
  std::string label_;
  std::vector<double> terms_;
  std::vector<double> expected_;
  int components_ = 1;
  int stride_ = 0;
  bool must_accept_ = false;
};

std::vector<Case> ReadCases(const char* path) {
  static_assert(std::endian::native == std::endian::little);
  std::ifstream input(path, std::ios::binary);
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (std::string(magic.data(), magic.size()) != "VMECLAT1")
    throw std::runtime_error("Case magic mismatch");
  std::uint64_t count;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!input || count > 10000) throw std::runtime_error("Invalid case count");
  std::vector<Case> result;
  for (std::uint64_t index = 0; index < count; ++index) {
    std::uint64_t size;
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!input || size < 1 || size > (1 << 22))
      throw std::runtime_error("Invalid case size");
    Case item;
    item.label_ = "cpu-case-" + std::to_string(index);
    item.stride_ = static_cast<int>(size);
    item.terms_.resize(size);
    item.expected_.resize(size);
    input.read(reinterpret_cast<char*>(item.terms_.data()),
               size * sizeof(double));
    input.read(reinterpret_cast<char*>(item.expected_.data()),
               size * sizeof(double));
    if (!input) throw std::runtime_error("Truncated cases");
    result.push_back(std::move(item));
  }
  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Trailing case bytes");
  return result;
}

struct Resources {
  const Api& api_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  double* reference_ = nullptr;
  double* sums_ = nullptr;
  void* context_ = nullptr;
  std::array<cudaGraph_t, 3> graphs_{};
  std::array<cudaGraphExec_t, 3> executables_{};

  explicit Resources(const Api& api) : api_(api) {}
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto executable : executables_)
      if (executable != nullptr) cudaGraphExecDestroy(executable);
    for (auto graph : graphs_)
      if (graph != nullptr) cudaGraphDestroy(graph);
    if (context_ != nullptr) api_.destroy_(context_);
    for (void* pointer : {terms_, prefix_, reference_, sums_})
      if (pointer != nullptr) cudaFree(pointer);
    if (start_ != nullptr) cudaEventDestroy(start_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  void Initialize(const Case& item) {
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&end_));
    const auto bytes = item.terms_.size() * sizeof(double);
    Check(cudaMalloc(&terms_, bytes));
    Check(cudaMalloc(&prefix_, bytes));
    Check(cudaMalloc(&reference_, bytes));
    Check(cudaMalloc(&sums_, item.components_ * sizeof(double)));
    Check(cudaMemcpyAsync(terms_, item.terms_.data(), bytes,
                          cudaMemcpyHostToDevice, stream_));
    context_ = api_.create_(item.components_, item.stride_, stream_);
  }
  void Enqueue(int method, const Case& item) {
    if (method == 0)
      api_.run_(context_, terms_, prefix_);
    else if (method == 1)
      api_.ordered_(terms_, reference_, item.components_, item.stride_,
                    stream_);
    else
      api_.sums_(terms_, sums_, item.components_, item.stride_, stream_);
  }
  void Capture(int method, const Case& item) {
    Enqueue(method, item);
    Check(cudaStreamSynchronize(stream_));
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      Enqueue(method, item);
      const auto status = cudaStreamEndCapture(stream_, &graphs_[method]);
      capturing = false;
      Check(status);
      Check(cudaGraphInstantiate(&executables_[method], graphs_[method], 0));
    } catch (...) {
      if (capturing) cudaStreamEndCapture(stream_, &graphs_[method]);
      throw;
    }
  }
};

std::vector<double> Download(const Resources& resources, const double* pointer,
                             std::size_t count) {
  std::vector<double> result(count);
  Check(cudaMemcpyAsync(result.data(), pointer, count * sizeof(double),
                        cudaMemcpyDeviceToHost, resources.stream_));
  Check(cudaStreamSynchronize(resources.stream_));
  return result;
}

void Equal(const std::vector<double>& expected,
           const std::vector<double>& actual) {
  if (expected.size() != actual.size())
    throw std::runtime_error("Length mismatch");
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (std::bit_cast<std::uint64_t>(expected[index]) !=
        std::bit_cast<std::uint64_t>(actual[index]))
      throw std::runtime_error("Prefix bit mismatch at " +
                               std::to_string(index));
}

std::array<unsigned long long, 4> CheckCase(const Api& api, const Case& item,
                                            std::size_t& m_nan_exclusions) {
  Resources resources(api);
  resources.Initialize(item);
  resources.Enqueue(1, item);
  const auto reference =
      Download(resources, resources.reference_, item.terms_.size());
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (std::isnan(item.expected_[index])) {
      if (!std::isnan(reference[index]))
        throw std::runtime_error("NaN mismatch");
      ++m_nan_exclusions;
    } else if (std::bit_cast<std::uint64_t>(reference[index]) !=
               std::bit_cast<std::uint64_t>(item.expected_[index])) {
      throw std::runtime_error("CPU reference mismatch");
    }
  }
  std::vector<int> flags(item.components_);
  resources.Enqueue(0, item);
  Equal(reference, Download(resources, resources.prefix_, reference.size()));
  api.flags_(resources.context_, flags.data());
  if (item.must_accept_ && std::any_of(flags.begin(), flags.end(),
                                       [](int value) { return value != 0; }))
    throw std::runtime_error(
        "Noncommutative scan witness unexpectedly fell back");
  resources.Capture(0, item);
  Check(cudaGraphLaunch(resources.executables_[0], resources.stream_));
  Equal(reference, Download(resources, resources.prefix_, reference.size()));
  api.flags_(resources.context_, flags.data());
  if (item.must_accept_ && std::any_of(flags.begin(), flags.end(),
                                       [](int value) { return value != 0; }))
    throw std::runtime_error("Captured scan witness unexpectedly fell back");
  std::array<unsigned long long, 4> statistics{};
  api.statistics_(resources.context_, statistics);
  api.corrupt_(resources.context_, resources.terms_, resources.prefix_);
  Equal(reference, Download(resources, resources.prefix_, reference.size()));
  api.flags_(resources.context_, flags.data());
  if (item.must_accept_ && std::any_of(flags.begin(), flags.end(),
                                       [](int value) { return value == 0; }))
    throw std::runtime_error("Corrupted witness was accepted");
  return statistics;
}

void TimeCase(const Api& api, const Case& item, bool& m_first) {
  Resources resources(api);
  resources.Initialize(item);
  for (int method = 0; method < 3; ++method) resources.Capture(method, item);
  std::mt19937_64 generator(982434);
  constexpr int kRepeats = 100;
  constexpr int kSamples = 7;
  for (int sample = 0; sample < kSamples; ++sample) {
    std::array<int, 6> order{0, 1, 2, 3, 4, 5};
    std::shuffle(order.begin(), order.end(), generator);
    for (int position = 0; position < 6; ++position) {
      const int method = order[position] / 2;
      const bool graph = order[position] % 2 != 0;
      Check(cudaEventRecord(resources.start_, resources.stream_));
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        if (graph)
          Check(cudaGraphLaunch(resources.executables_[method],
                                resources.stream_));
        else
          resources.Enqueue(method, item);
      }
      Check(cudaEventRecord(resources.end_, resources.stream_));
      Check(cudaEventSynchronize(resources.end_));
      float elapsed;
      Check(cudaEventElapsedTime(&elapsed, resources.start_, resources.end_));
      if (method == 0)
        Equal(item.expected_,
              Download(resources, resources.prefix_, item.terms_.size()));
      else if (method == 1)
        Equal(item.expected_,
              Download(resources, resources.reference_, item.terms_.size()));
      else {
        std::vector<double> expected;
        for (int component = 0; component < item.components_; ++component)
          expected.push_back(
              item.expected_[(component + 1) * item.stride_ - 1]);
        Equal(expected, Download(resources, resources.sums_, expected.size()));
      }
      const std::array<const char*, 3> methods{"lattice", "ordered_full_prefix",
                                               "ordered_final_sum"};
      std::cout << (m_first ? "" : ",") << "{\"case\":\"" << item.label_
                << "\",\"components\":" << item.components_
                << ",\"stride\":" << item.stride_ << ",\"sample\":" << sample
                << ",\"order\":" << position << ",\"method\":\""
                << methods[method] << "\",\"mode\":\""
                << (graph ? "graph" : "direct") << "\",\"repeats\":" << kRepeats
                << ",\"cuda_event_ms\":" << elapsed
                << ",\"all_output_bits_exact\":true}";
      m_first = false;
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: benchmark cases.bin library.so captured_tail_count");
    const Api api(argv[2]);
    auto cases = ReadCases(argv[1]);
    const int tail = std::stoi(argv[3]);
    if (tail != 0 && (tail != 6 || cases.size() < 6))
      throw std::runtime_error("Captured tail must be zero or six streams");
    std::vector<Case> timing;
    if (tail == 6) {
      const std::size_t original_cases = cases.size();
      for (int stage = 0; stage < 2; ++stage) {
        Case item;
        item.label_ =
            stage == 0 ? "captured_invariant" : "captured_preconditioned";
        item.components_ = 3;
        item.stride_ = cases[original_cases - 6 + 3 * stage].stride_;
        for (int component = 0; component < 3; ++component) {
          const auto& source =
              cases[original_cases - 6 + 3 * stage + component];
          if (source.stride_ != item.stride_)
            throw std::runtime_error("Captured lengths differ");
          item.terms_.insert(item.terms_.end(), source.terms_.begin(),
                             source.terms_.end());
          item.expected_.insert(item.expected_.end(), source.expected_.begin(),
                                source.expected_.end());
        }
        timing.push_back(item);
        cases.push_back(std::move(item));
      }
    }
    // Half-ulp and one-ulp maps are noncommutative; fallback must not hide an
    // operand-order error. Test lengths on both sides of CUB tile boundaries.
    for (int count :
         {127,  128,   129,   255,   256,   257,   511,   512,  513,  1023,
          1024, 1025,  2047,  2048,  2049,  4095,  4096,  4097, 8191, 8192,
          8193, 16383, 16384, 16385, 21780, 32767, 32768, 32769}) {
      Case item;
      item.label_ = "half_ulp_witness_" + std::to_string(count);
      item.stride_ = count;
      item.must_accept_ = true;
      double sum = 0.0;
      for (int index = 0; index < count; ++index) {
        const double term = index == 0 ? 1.0 : (index % 2 ? 0x1p-53 : 0x1p-52);
        item.terms_.push_back(term);
        sum += term;
        item.expected_.push_back(sum);
      }
      cases.push_back(std::move(item));
    }
    std::array<unsigned long long, 4> totals{};
    std::size_t nan_exclusions = 0;
    for (const auto& item : cases) {
      const auto statistics = CheckCase(api, item, nan_exclusions);
      for (int index = 0; index < 4; ++index)
        totals[index] += statistics[index];
    }
    std::cout
        << "{\"scope\":\"isolated prefix proposal/certificate/fallback, no "
           "solver speedup claim\","
        << "\"correctness_cases\":" << cases.size()
        << ",\"all_gpu_prefix_bits_exact\":true,"
        << "\"cpu_nan_payload_exclusions\":" << nan_exclusions
        << ",\"injection_controls_executed\":" << cases.size()
        << ",\"unmodified_proposal_calls\":" << totals[0]
        << ",\"accepted_components\":" << totals[1]
        << ",\"fallback_components\":" << totals[2]
        << ",\"normal_segments\":" << totals[3]
        << ",\"timing_scope\":\"CUDA events, terms already rounded; graph "
           "excludes capture; direct includes host submission; "
           "ordered_final_sum writes only one value/component\",\"samples\":[";
    bool first = true;
    for (const auto& item : timing) TimeCase(api, item, first);
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
