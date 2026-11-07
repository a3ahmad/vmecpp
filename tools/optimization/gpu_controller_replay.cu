// SPDX-License-Identifier: MIT
// Fixed observed damping-controller replay; not an autonomous equilibrium loop.
#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cuda/std/array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Event {
  int ns_;
  int iter1_;
  int iter2_;
  int initialize_;
  std::uint64_t fsq_;
  std::uint64_t time_step_;
  std::uint64_t fac_;
  std::uint64_t b1_;
  cuda::std::array<std::uint64_t, 10> history_;
};
struct LogPair {
  std::uint64_t argument_;
  std::uint64_t value_;
};
struct Output {
  std::uint64_t fac_;
  std::uint64_t b1_;
  cuda::std::array<std::uint64_t, 10> history_;
  std::uint64_t log_argument_;
  std::uint64_t log_value_;
  std::uint64_t numerator_;
  std::uint32_t log_used_;
  std::uint32_t errors_;
};
static_assert(sizeof(Event) == 128 && sizeof(Output) == 128 &&
              sizeof(LogPair) == 16);
static_assert(std::endian::native == std::endian::little);
__device__ double Value(std::uint64_t bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}
__device__ std::uint64_t Bits(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}
template <bool kNativeLog>
__global__ void Replay(const Event* events, std::size_t count,
                       const LogPair* logs, std::size_t log_count,
                       Output* m_output) {
  cuda::std::array<double, 10> history{};
  double previous_fsq = 0.0;
  std::size_t log_index = 0;
  bool initialized = false;
  for (std::size_t i = 0; i < count; ++i) {
    const Event event = events[i];
    Output output{};
    const double dt = Value(event.time_step_);
    const double fsq = Value(event.fsq_);
    const bool reset = event.iter2_ == event.iter1_;
    if (reset != (event.initialize_ != 0)) output.errors_ |= 32;
    if (reset) {
      const double initial = __ddiv_rn(0.15, dt);
      for (int j = 0; j < 10; ++j) history[j] = initial;
      initialized = true;
    }
    if (!initialized) output.errors_ |= 1;
    if (event.iter2_ < event.iter1_) output.errors_ |= 2;
    const double first = history[0];
    for (int j = 0; j < 9; ++j) history[j] = history[j + 1];
    history[9] = first;
    if (event.iter2_ > event.iter1_) {
      double numerator = 0.0;
      if (fsq != 0.0) {
        const double argument = __ddiv_rn(fsq, previous_fsq);
        double logarithm = 0.0;
        if (log_index >= log_count) {
          output.errors_ |= 8;
        } else {
          if (Bits(argument) != logs[log_index].argument_) output.errors_ |= 4;
          if constexpr (kNativeLog)
            logarithm = log(argument);
          else
            logarithm = Value(logs[log_index].value_);
        }
        ++log_index;
        // Literal fabs and std::min operand semantics, not fmin's NaN rule.
        const double magnitude = Value(Bits(logarithm) & 0x7fffffffffffffffULL);
        numerator = 0.15 < magnitude ? 0.15 : magnitude;
        output.log_used_ = 1;
        output.log_argument_ = Bits(argument);
        output.log_value_ = Bits(logarithm);
      }
      output.numerator_ = Bits(numerator);
      history[9] = __ddiv_rn(numerator, dt);
    }
    previous_fsq = fsq;
    double total = 0.0;
    for (int j = 0; j < 10; ++j) {
      total = __dadd_rn(total, history[j]);
      output.history_[j] = Bits(history[j]);
    }
    const double otav = __ddiv_rn(total, 10.0);
    const double dtau = __ddiv_rn(__dmul_rn(dt, otav), 2.0);
    output.b1_ = Bits(__dsub_rn(1.0, dtau));
    output.fac_ = Bits(__ddiv_rn(1.0, __dadd_rn(1.0, dtau)));
    if (i + 1 == count && log_index != log_count) output.errors_ |= 16;
    m_output[i] = output;
  }
}
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T>
void Read(std::ifstream& m_file, T* m_output, std::size_t count) {
  m_file.read(reinterpret_cast<char*>(m_output), count * sizeof(T));
  if (!m_file) throw std::runtime_error("Truncated controller input");
}
}  // namespace

int main(int argc, char** argv) {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Event* device_events = nullptr;
  LogPair* device_logs = nullptr;
  Output* device_outputs = nullptr;
  auto cleanup = [&] {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (executable != nullptr) cudaGraphExecDestroy(executable);
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (device_outputs != nullptr) cudaFree(device_outputs);
    if (device_logs != nullptr) cudaFree(device_logs);
    if (device_events != nullptr) cudaFree(device_events);
    if (stream != nullptr) cudaStreamDestroy(stream);
  };
  try {
    if (argc != 4 ||
        (std::string(argv[3]) != "direct" && std::string(argv[3]) != "graph"))
      throw std::runtime_error(
          "Usage: controller-replay input.bin output.bin direct|graph");
    std::ifstream input(argv[1], std::ios::binary);
    cuda::std::array<char, 8> magic{};
    Read(input, magic.data(), 8);
    if (std::string(magic.data(), 8) != "VMECCTR1")
      throw std::runtime_error("Invalid controller input magic");
    std::uint64_t count = 0, log_count = 0;
    Read(input, &count, 1);
    Read(input, &log_count, 1);
    if (count == 0 || count > 100000 || log_count > count)
      throw std::runtime_error("Invalid controller counts");
    std::vector<Event> events(count);
    std::vector<LogPair> logs(log_count);
    Read(input, events.data(), count);
    Read(input, logs.data(), log_count);
    if (input.peek() != std::ifstream::traits_type::eof())
      throw std::runtime_error("Trailing controller input bytes");
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Check(cudaMalloc(&device_events, count * sizeof(Event)));
    Check(cudaMalloc(&device_logs,
                     (log_count == 0 ? 1 : log_count) * sizeof(LogPair)));
    Check(cudaMalloc(&device_outputs, 2 * count * sizeof(Output)));
    Check(cudaMemcpyAsync(device_events, events.data(), count * sizeof(Event),
                          cudaMemcpyHostToDevice, stream));
    if (log_count != 0)
      Check(cudaMemcpyAsync(device_logs, logs.data(),
                            log_count * sizeof(LogPair), cudaMemcpyHostToDevice,
                            stream));
    Check(cudaStreamSynchronize(stream));
    const bool use_graph = std::string(argv[3]) == "graph";
    if (use_graph)
      Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    Replay<false><<<1, 1, 0, stream>>>(device_events, count, device_logs,
                                       log_count, device_outputs);
    Check(cudaGetLastError());
    Replay<true><<<1, 1, 0, stream>>>(device_events, count, device_logs,
                                      log_count, device_outputs + count);
    Check(cudaGetLastError());
    if (use_graph) {
      Check(cudaStreamEndCapture(stream, &graph));
      Check(cudaGraphInstantiate(&executable, graph, 0));
      Check(cudaGraphLaunch(executable, stream));
    }
    std::vector<Output> outputs(2 * count);
    Check(cudaMemcpyAsync(outputs.data(), device_outputs,
                          outputs.size() * sizeof(Output),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    for (const auto& output : outputs)
      if (output.errors_ != 0)
        throw std::runtime_error(
            "Controller replay input/state/log-index mismatch");
    // CPU-log mode separates literal scalar arithmetic from native CUDA log.
    for (std::size_t i = 0; i < count; ++i) {
      if (outputs[i].fac_ != events[i].fac_ ||
          outputs[i].b1_ != events[i].b1_ ||
          outputs[i].history_ != events[i].history_)
        throw std::runtime_error(
            "GPU literal arithmetic missed original CPU bits");
    }
    std::ofstream file(argv[2], std::ios::binary | std::ios::trunc);
    file.write("VMECGCR1", 8);
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    const std::uint64_t modes = 2;
    file.write(reinterpret_cast<const char*>(&modes), sizeof(modes));
    file.write(reinterpret_cast<const char*>(outputs.data()),
               outputs.size() * sizeof(Output));
    if (!file) throw std::runtime_error("Controller output write failed");
    cleanup();
    std::printf(
        "PASS: %llu actual observed steps; CPU-log arithmetic exact; "
        "native-log output retained; graph=%d. Not an autonomous solver.\n",
        static_cast<unsigned long long>(count), use_graph);
    return 0;
  } catch (const std::exception& error) {
    // Also end a failed active capture before retiring its stream.
    if (stream != nullptr && graph == nullptr) {
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
          status != cudaStreamCaptureStatusNone)
        cudaStreamEndCapture(stream, &graph);
    }
    cleanup();
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
