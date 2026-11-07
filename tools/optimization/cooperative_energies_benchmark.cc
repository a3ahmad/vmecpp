// SPDX-License-Identifier: MIT
// Captured ordinary-energy component gates; never a full-solver speed claim.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_cooperative_energies.h"
#include "gpu_optimistic_diagnostics.h"
#include "resident_replay_capture.h"

namespace {
constexpr int kPacket = 13;  // Seven outputs followed by six raw sums.
using Packet = std::array<double, kPacket>;
using Counters = std::array<unsigned long long, 36>;

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void RequireEqual(const Packet& expected, const double* actual) {
  for (int i = 0; i < kPacket; ++i) {
    if (std::memcmp(expected.data() + i, actual + i, sizeof(double)) != 0) {
      std::fprintf(stderr, "slot%d expected%.17g actual%.17g\n", i, expected[i],
                   actual[i]);
      throw std::runtime_error("Output/raw-sum packet byte mismatch");
    }
  }
}
struct Graph {
  int rounds_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};
struct Resources {
  GpuDiagnosticsDeviceState state_{};
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
  std::vector<void*> allocations_;
  std::vector<Graph> graphs_;
  double* captured_ = nullptr;
  double* history_ = nullptr;

  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto& graph : graphs_) {
      cudaGraphExecDestroy(graph.executable_);
      cudaGraphDestroy(graph.graph_);
    }
    for (void* pointer : allocations_) cudaFree(pointer);
    if (begin_ != nullptr) cudaEventDestroy(begin_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  double* Allocate(std::size_t count, const double* source = nullptr) {
    double* pointer = nullptr;
    Check(cudaMalloc(&pointer, count * sizeof(double)));
    try {
      allocations_.push_back(pointer);
    } catch (...) {
      cudaFree(pointer);
      throw;
    }
    if (source != nullptr)
      Check(cudaMemcpyAsync(pointer, source, count * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
    return pointer;
  }
  void Reset(const Packet& seed) {
    Check(cudaMemcpyAsync(state_.output_, seed.data(), sizeof(seed),
                          cudaMemcpyHostToDevice, stream_));
  }
  Packet Read(const double* pointer) {
    Packet actual{};
    Check(cudaMemcpyAsync(actual.data(), pointer, sizeof(actual),
                          cudaMemcpyDeviceToHost, stream_));
    Check(cudaStreamSynchronize(stream_));
    return actual;
  }
  void Capture(int rounds) {
    for (const auto& graph : graphs_)
      if (graph.rounds_ == rounds) return;
    Graph graph{rounds};
    bool capturing = false;
    try {
      Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      Check(cudaMemsetAsync(state_.output_ + 1, 0x7f, 3 * sizeof(double),
                            stream_));
      Check(cudaMemsetAsync(state_.sums_, 0x7f, 3 * sizeof(double), stream_));
      LaunchGpuCooperativeEnergies(state_, rounds);
      Check(cudaMemcpyAsync(captured_, state_.output_, kPacket * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream_));
      const auto status = cudaStreamEndCapture(stream_, &graph.graph_);
      capturing = false;
      Check(status);
      Check(cudaGraphInstantiate(&graph.executable_, graph.graph_, 0));
      graphs_.push_back(graph);
    } catch (...) {
      if (capturing) cudaStreamEndCapture(stream_, &graph.graph_);
      if (graph.executable_ != nullptr) cudaGraphExecDestroy(graph.executable_);
      if (graph.graph_ != nullptr) cudaGraphDestroy(graph.graph_);
      throw;
    }
  }
};
struct Configuration {
  int mode_;  // 0 ordered;1 fixed16 graph;2 cooperative;3 cooperative outer.
  int rounds_;
};
void Launch(Resources& m_resources, Configuration configuration) {
  if (configuration.mode_ == 0) {
    LaunchGpuEnergies(m_resources.state_);
  } else if (configuration.mode_ == 1) {
    LaunchGpuDiagnosticsOptimistic(m_resources.state_, false, 16, true);
  } else if (configuration.mode_ == 2) {
    LaunchGpuCooperativeEnergies(m_resources.state_, configuration.rounds_);
  } else {
    m_resources.Capture(configuration.rounds_);
    for (const auto& graph : m_resources.graphs_)
      if (graph.rounds_ == configuration.rounds_) {
        Check(cudaGraphLaunch(graph.executable_, m_resources.stream_));
        return;
      }
    throw std::runtime_error("Missing cooperative energies outer graph");
  }
}
const double* Output(const Resources& resources, Configuration configuration) {
  return configuration.mode_ == 3 ? resources.captured_
                                  : resources.state_.output_;
}
void Statistics(Configuration configuration, Counters& m_counters) {
  if (configuration.mode_ == 1) {
    std::array<unsigned long long, 7> fixed{};
    ReadGpuDiagnosticsRelaxationStats(fixed);
    m_counters[0] = fixed[0];
    m_counters[1] = fixed[2];
    m_counters[2] = fixed[3];
  } else {
    ReadGpuCooperativeEnergiesStats(m_counters);
  }
}
struct Measurement {
  Configuration configuration_;
  int repetition_;
  double microseconds_;
  Counters counters_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5)
      throw std::runtime_error(
          "Usage: cooperative-energies-benchmark capture output.json "
          "[repeats] [focused_rounds]");
    const int repeats = argc >= 4 ? std::stoi(argv[3]) : 20;
    const int focused = argc == 5 ? std::stoi(argv[4]) : 0;
    if (repeats < 1 || repeats > 100 || focused < 0 || focused > 32)
      throw std::runtime_error("Unsupported repeats/round count");
    auto capture =
        vmecpp_experiment::Read<GpuDiagnosticsInput, GpuDiagnosticsOutput>(
            argv[1], "VMECGDG1");
    if (capture.expected_.size() != 7)
      throw std::runtime_error("Expected seven captured CPU diagnostics");
    Resources resources;
    Check(cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
    Check(cudaEventCreate(&resources.begin_));
    Check(cudaEventCreate(&resources.end_));
    auto& state = resources.state_;
    state.dimensions_ = capture.input_.dimensions_;
    state.scalars_ = capture.input_.scalars_;
    state.stream_ = resources.stream_;
    if (!GpuCooperativeEnergiesSupported(state))
      throw std::runtime_error("Captured energies grid unsupported");
    auto invalid = state;
    invalid.scalars_[1] = 1.0;
    if (GpuCooperativeEnergiesSupported(invalid))
      throw std::runtime_error("Nonzero gamma unexpectedly supported");
    for (int i = 0; i < 14; ++i)
      state.data_[i] =
          resources.Allocate(capture.input_.sizes_[i], capture.input_.data_[i]);
    state.output_ = resources.Allocate(kPacket);
    state.sums_ = state.output_ + 7;
    resources.captured_ = resources.Allocate(kPacket);
    resources.history_ = resources.Allocate(kPacket * (repeats + 2));
    Packet seed{};
    std::copy(capture.expected_.begin(), capture.expected_.end(), seed.begin());
    seed[0] *= 1.25;
    seed[1] = -17.0;
    seed[2] = -19.0;
    seed[3] = -23.0;
    seed[4] = 3.25;
    seed[5] = 4.25;
    seed[6] = 5.25;
    for (int i = 7; i < kPacket; ++i) seed[i] = 20.25 + i;
    resources.Reset(seed);
    LaunchGpuEnergies(state);
    const Packet expected = resources.Read(state.output_);
    for (int i = 0; i < 7; ++i) {
      const double value = i >= 1 && i <= 3 ? capture.expected_[i] : seed[i];
      if (std::memcmp(&expected[i], &value, sizeof(double)))
        throw std::runtime_error("Ordered energies differ from CPU/cache");
    }
    for (int i = 10; i < kPacket; ++i)
      if (std::memcmp(&expected[i], &seed[i], sizeof(double)))
        throw std::runtime_error("Ordered energies changed raw norm scratch");
    std::vector<Configuration> configurations{{0, 0}, {1, 16}};
    const std::vector<int> limits =
        focused ? std::vector<int>{focused}
                : std::vector<int>{4, 8, 12, 16, 24, 32};
    for (int rounds : limits) {
      configurations.push_back({2, rounds});
      configurations.push_back({3, rounds});
    }
    for (const auto& configuration : configurations) {
      resources.Reset(seed);
      for (int i = 0; i < 3; ++i) {
        Launch(resources, configuration);
        const auto actual = resources.Read(Output(resources, configuration));
        RequireEqual(expected, actual.data());
      }
    }
    std::puts(
        "PASS: captured CPU energies, retained cache and six raw sums "
        "exact for every requested path before timing");
    std::fflush(stdout);
    std::mt19937 generator(2077);
    std::vector<Measurement> measurements;
    for (int repetition = 0; repetition < 3; ++repetition) {
      std::shuffle(configurations.begin(), configurations.end(), generator);
      for (const auto& configuration : configurations) {
        resources.Reset(seed);
        for (int i = 0; i < 2; ++i) {
          Launch(resources, configuration);
          Check(cudaMemcpyAsync(resources.history_ + kPacket * (repeats + i),
                                Output(resources, configuration),
                                kPacket * sizeof(double),
                                cudaMemcpyDeviceToDevice, resources.stream_));
        }
        Check(cudaStreamSynchronize(resources.stream_));
        Counters before{}, after{};
        Statistics(configuration, before);
        Check(cudaEventRecord(resources.begin_, resources.stream_));
        for (int i = 0; i < repeats; ++i) {
          Launch(resources, configuration);
          Check(cudaMemcpyAsync(resources.history_ + kPacket * i,
                                Output(resources, configuration),
                                kPacket * sizeof(double),
                                cudaMemcpyDeviceToDevice, resources.stream_));
        }
        Check(cudaEventRecord(resources.end_, resources.stream_));
        Check(cudaEventSynchronize(resources.end_));
        float milliseconds;
        Check(cudaEventElapsedTime(&milliseconds, resources.begin_,
                                   resources.end_));
        Statistics(configuration, after);
        for (int i = 0; i < 36; ++i) after[i] -= before[i];
        std::vector<double> history(kPacket * (repeats + 2));
        Check(cudaMemcpy(history.data(), resources.history_,
                         history.size() * sizeof(double),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < repeats + 2; ++i)
          RequireEqual(expected, history.data() + kPacket * i);
        measurements.push_back({configuration, repetition,
                                milliseconds * 1000.0 / repeats, after});
      }
    }
    std::array<int, 6> layout{};
    ReadGpuCooperativeEnergiesLayout(layout);
    std::printf("LAYOUT grid%d block%d activeBlocksPerSM%d SM%d tiles%d+%d\n",
                layout[0], layout[1], layout[2], layout[3], layout[4],
                layout[5]);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("Cannot write component report");
    output
        << std::setprecision(17)
        << "{\"scope\":\"captured ordinary energies only; no full-solver "
           "speed claim\",\"exact_cpu_energies_and_cache_and_raw_sums\":true,"
        << "\"timing_scope\":\"CUDAevents include13-double history D2D; "
           "outer additionally poisons energy outputs/raw sums and "
           "snapshots13doubles\",\"repeats\":"
        << repeats
        << ",\"statistics_layout\":\"calls,fallbackThermal,Magnetic,"
           "histogram0..32; mode1 has first3 counters only\","
        << "\"measurements\":[\n";
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto& measurement = measurements[i];
      output << "{\"mode\":" << measurement.configuration_.mode_
             << ",\"rounds\":" << measurement.configuration_.rounds_
             << ",\"repetition\":" << measurement.repetition_
             << ",\"mean_microseconds\":" << measurement.microseconds_
             << ",\"statistics\":[";
      for (int j = 0; j < 36; ++j) {
        if (j) output << ',';
        output << measurement.counters_[j];
      }
      output << "]}" << (i + 1 == measurements.size() ? "\n" : ",\n");
    }
    output << "]}\n";
    std::puts("PASS: every timed13-double history exact");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
