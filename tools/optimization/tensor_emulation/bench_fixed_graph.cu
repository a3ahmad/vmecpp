// SPDX-License-Identifier: MIT
// Graph-captured fixed-mantissa snapshot; measured direct sources unchanged.
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern char** environ;
namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void Check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
std::uint64_t Raw() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
    throw std::runtime_error("RAW clock failed");
  return std::uint64_t(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
std::string Quote(const std::string& value) {
  std::string result = "\"";
  for (unsigned char byte : value) {
    if (byte == '"' || byte == '\\') result += '\\';
    if (byte < 32 || byte > 126)
      throw std::runtime_error("Non-ASCII JSON string");
    result += char(byte);
  }
  return result + '"';
}
std::string Owner(const char* symbol) {
  void* pointer = dlsym(RTLD_DEFAULT, symbol);
  Dl_info information{};
  if (pointer == nullptr || !dladdr(pointer, &information))
    throw std::runtime_error("Library symbol owner missing");
  return std::filesystem::canonical(information.dli_fname).string();
}
double Median(std::vector<double> m_values) {
  std::sort(m_values.begin(), m_values.end());
  return m_values[m_values.size() / 2];
}
const std::array<std::string, 6> kModes{"native",   "fixed-40", "fixed-48",
                                        "fixed-55", "fixed-64", "fixed-80"};
struct DeviceData {
  cudaStream_t stream_{};
  cublasHandle_t handle_{};
  cudaEvent_t begin_{}, end_{};
  double *a_ = nullptr, *b_ = nullptr, *c_ = nullptr;
  void* workspace_ = nullptr;
};
DeviceData device;
void CloseDevice() {
  if (!device.handle_) return;
  Check(cudaEventDestroy(device.begin_));
  Check(cudaEventDestroy(device.end_));
  Check(cublasDestroy(device.handle_));
  Check(cudaFree(device.workspace_));
  Check(cudaFree(device.c_));
  Check(cudaFree(device.b_));
  Check(cudaFree(device.a_));
  Check(cudaStreamDestroy(device.stream_));
  device = {};
}
struct Options {
  bool execute_ = false;
  bool profile_one_ = false;
  std::string mode_ = "native";
  std::string input_;
  std::string report_;
  std::string output_;
  std::string expect_cublas_;
  int m_ = 256, n_ = 256, k_ = 256;
  int calls_ = 20, samples_ = 7, warmups_ = 3;
  std::size_t workspace_mib_ = 8192;
};
Options Parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string name(argv[index]);
    if (name == "--execute")
      options.execute_ = true;
    else if (name == "--profile-one")
      options.profile_one_ = true;
    else {
      if (++index == argc) throw std::runtime_error("Missing option value");
      const std::string value(argv[index]);
      if (name == "--mode")
        options.mode_ = value;
      else if (name == "--input")
        options.input_ = value;
      else if (name == "--report")
        options.report_ = value;
      else if (name == "--output")
        options.output_ = value;
      else if (name == "--expect-cublas")
        options.expect_cublas_ = value;
      else if (name == "--m")
        options.m_ = std::stoi(value);
      else if (name == "--n")
        options.n_ = std::stoi(value);
      else if (name == "--k")
        options.k_ = std::stoi(value);
      else if (name == "--calls")
        options.calls_ = std::stoi(value);
      else if (name == "--samples")
        options.samples_ = std::stoi(value);
      else if (name == "--warmups")
        options.warmups_ = std::stoi(value);
      else if (name == "--workspace-mib")
        options.workspace_mib_ = std::stoull(value);
      else
        throw std::runtime_error("Unknown option " + name);
    }
  }
  if (options.mode_ != "all" &&
      std::find(kModes.begin(), kModes.end(), options.mode_) == kModes.end())
    throw std::runtime_error("Unsupported mode");
  if (options.calls_ < 1 || options.calls_ > 10000 || options.samples_ < 1 ||
      options.samples_ > 101 || options.samples_ % 2 != 1 ||
      options.warmups_ < 0 || options.warmups_ > 100 ||
      options.workspace_mib_ > 8192)
    throw std::runtime_error("Out-of-bound workload");
  if (options.execute_ &&
      (options.report_.empty() || options.expect_cublas_.empty()))
    throw std::runtime_error(
        "Execution requires report and exact cuBLAS owner");
  return options;
}
struct Matrices {
  int m_, n_, k_;
  std::vector<double> a_, b_, reference_;
};
std::size_t Product(std::uint64_t left, std::uint64_t right) {
  if (left == 0 || right == 0 || left > 2147483647 || right > 2147483647 ||
      left > (std::size_t(1) << 29) / right)
    throw std::runtime_error("Matrix size exceeds bounded FP64 allocation");
  return std::size_t(left * right);
}
Matrices Inputs(const Options& options) {
  std::array<std::uint64_t, 3> dimensions{std::uint64_t(options.m_),
                                          std::uint64_t(options.n_),
                                          std::uint64_t(options.k_)};
  std::ifstream input;
  if (!options.input_.empty()) {
    input.open(options.input_, std::ios::binary);
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    if (magic != std::array<char, 8>{'T', 'E', 'G', 'E', 'M', 'M', '1', '\0'})
      throw std::runtime_error("Invalid TEGEMM1 magic");
    static_assert(std::endian::native == std::endian::little);
    input.read(reinterpret_cast<char*>(dimensions.data()), sizeof(dimensions));
    if (!input) throw std::runtime_error("Incomplete dimensions");
  }
  const auto a_count = Product(dimensions[0], dimensions[2]);
  const auto b_count = Product(dimensions[2], dimensions[1]);
  const auto c_count = Product(dimensions[0], dimensions[1]);
  Matrices matrices{int(dimensions[0]),           int(dimensions[1]),
                    int(dimensions[2]),           std::vector<double>(a_count),
                    std::vector<double>(b_count), std::vector<double>(c_count)};
  if (input.is_open()) {
    for (auto* values : {&matrices.a_, &matrices.b_, &matrices.reference_}) {
      input.read(reinterpret_cast<char*>(values->data()),
                 values->size() * sizeof(double));
      if (!input) throw std::runtime_error("Incomplete matrix data");
    }
    if (input.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Unexpected trailing input bytes");
  } else {
    std::uint64_t random = 0x509013500131ULL;
    for (auto* values : {&matrices.a_, &matrices.b_})
      for (auto& value : *values) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        value = double(std::int64_t(random % 2049) - 1024) / 1024.0;
      }
    for (int column = 0; column < matrices.n_; ++column)
      for (int row = 0; row < matrices.m_; ++row) {
        long double sum = 0;
        for (int inner = 0; inner < matrices.k_; ++inner)
          sum += static_cast<long double>(
                     matrices.a_[row + std::size_t(inner) * matrices.m_]) *
                 matrices.b_[inner + std::size_t(column) * matrices.k_];
        matrices.reference_[row + std::size_t(column) * matrices.m_] =
            double(sum);
      }
  }
  for (const auto* values : {&matrices.a_, &matrices.b_, &matrices.reference_})
    for (double value : *values)
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite input/reference");
  return matrices;
}
std::uint64_t OrderedBits(double value) {
  auto bits = std::bit_cast<std::uint64_t>(value);
  return (bits >> 63) ? ~bits : bits | (std::uint64_t(1) << 63);
}
void Execute(const Options& options, const Matrices& matrices) {
  // Explicit APIs determine every arm. Never silently inherit an environment
  // override that changes precision or the claimed requested strategy.
  for (char** entry = environ; entry && *entry; ++entry) {
    const std::string value(*entry);
    if (value.starts_with("CUBLAS_") ||
        value.starts_with("NVIDIA_TF32_OVERRIDE="))
      throw std::runtime_error("Remove cuBLAS precision/environment overrides");
  }
  const std::string cublas_owner = Owner("cublasGemmEx");
  if (cublas_owner !=
      std::filesystem::canonical(options.expect_cublas_).string())
    throw std::runtime_error("Unexpected cuBLAS owner");
  const std::string runtime_owner = Owner("cudaMalloc");
  const bool native = options.mode_ == "native";
  const int offset = 0;
  const int fixed_bits = native ? 0 : std::stoi(options.mode_.substr(6));
  const auto requested_control = native
                                     ? CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
                                     : CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
  const auto setup_begin = Raw();
  Check(cudaSetDevice(0));
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, 0));
  int runtime_version = 0, driver_version = 0;
  Check(cudaRuntimeGetVersion(&runtime_version));
  Check(cudaDriverGetVersion(&driver_version));
  const bool first_mode = device.handle_ == nullptr;
  if (first_mode) {
    Check(cudaStreamCreateWithFlags(&device.stream_, cudaStreamNonBlocking));
    Check(cublasCreate(&device.handle_));
    Check(cublasSetStream(device.handle_, device.stream_));
    Check(cudaEventCreate(&device.begin_));
    Check(cudaEventCreate(&device.end_));
  }
  const auto stream = device.stream_;
  const auto handle = device.handle_;
  Check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  Check(cublasSetMathMode(handle,
                          native ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));
  const auto strategy = options.mode_ == "dynamic-performant"
                            ? CUBLAS_EMULATION_STRATEGY_PERFORMANT
                            : CUBLAS_EMULATION_STRATEGY_EAGER;
  Check(cublasSetEmulationStrategy(handle, strategy));
  Check(cublasSetFixedPointEmulationMantissaControl(handle, requested_control));
  if (!native)
    Check(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, fixed_bits));
  Check(cublasSetFixedPointEmulationMantissaBitOffset(handle, offset));
  int library_version = 0, actual_offset = 0, max_bits = 0;
  cudaEmulationMantissaControl actual_control{};
  cublasEmulationStrategy_t actual_strategy{};
  Check(cublasGetVersion(handle, &library_version));
  Check(cublasGetEmulationStrategy(handle, &actual_strategy));
  Check(cublasGetFixedPointEmulationMantissaControl(handle, &actual_control));
  Check(cublasGetFixedPointEmulationMantissaBitOffset(handle, &actual_offset));
  Check(cublasGetFixedPointEmulationMaxMantissaBitCount(handle, &max_bits));
  if (actual_offset != offset || actual_strategy != strategy ||
      actual_control != requested_control ||
      (!native && max_bits != fixed_bits))
    throw std::runtime_error("cuBLAS configuration readback mismatch");
  const auto workspace_bytes = options.workspace_mib_ * 1024 * 1024;
  if (first_mode) {
    Check(cudaMalloc(&device.a_, matrices.a_.size() * sizeof(double)));
    Check(cudaMalloc(&device.b_, matrices.b_.size() * sizeof(double)));
    Check(cudaMalloc(&device.c_, matrices.reference_.size() * sizeof(double)));
    if (workspace_bytes != 0)
      Check(cudaMalloc(&device.workspace_, workspace_bytes));
    Check(cudaMemcpyAsync(device.a_, matrices.a_.data(),
                          matrices.a_.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaMemcpyAsync(device.b_, matrices.b_.data(),
                          matrices.b_.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaStreamSynchronize(stream));
  }
  const auto a = device.a_, b = device.b_, c = device.c_;
  Check(cublasSetWorkspace(handle, device.workspace_, workspace_bytes));
  const auto setup_end = Raw();
  const double alpha = 1.0, beta = 0.0;
  const auto compute = native ? CUBLAS_COMPUTE_64F_PEDANTIC
                              : CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
  auto gemm = [&] {
    Check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrices.m_,
                       matrices.n_, matrices.k_, &alpha, a, CUDA_R_64F,
                       matrices.m_, b, CUDA_R_64F, matrices.k_, &beta, c,
                       CUDA_R_64F, matrices.m_, compute, CUBLAS_GEMM_DEFAULT));
  };
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  bool use_graph = false;
  auto launch = [&] {
    if (use_graph)
      Check(cudaGraphLaunch(graph_exec, stream));
    else
      gemm();
  };
  const auto begin = device.begin_, end = device.end_;
  auto timed = [&](int calls) {
    const auto raw_begin = Raw();
    Check(cudaEventRecord(begin, stream));
    for (int call = 0; call < calls; ++call) launch();
    Check(cudaEventRecord(end, stream));
    Check(cudaEventSynchronize(end));
    const auto raw_end = Raw();
    float elapsed = 0;
    Check(cudaEventElapsedTime(&elapsed, begin, end));
    return std::array<double, 2>{double(elapsed) / calls,
                                 double(raw_end - raw_begin) / calls / 1e6};
  };
  const auto first = timed(1);
  for (int call = 0; call < options.warmups_; ++call) gemm();
  Check(cudaStreamSynchronize(stream));
  const auto capture_begin = Raw();
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  gemm();
  Check(cudaStreamEndCapture(stream, &graph));
  const auto capture_end = Raw();
  Check(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
  const auto instantiate_end = Raw();
  use_graph = true;
  const auto first_graph = timed(1);
  std::vector<double> event_ms, raw_ms;
  for (int sample = 0; sample < options.samples_; ++sample) {
    const auto elapsed = timed(options.calls_);
    event_ms.push_back(elapsed[0]);
    raw_ms.push_back(elapsed[1]);
  }
  if (options.profile_one_) {
    Check(cudaProfilerStart());
    launch();
    Check(cudaStreamSynchronize(stream));
    Check(cudaProfilerStop());
  }
  std::vector<double> output(matrices.reference_.size());
  Check(cudaMemcpyAsync(output.data(), c, output.size() * sizeof(double),
                        cudaMemcpyDeviceToHost, stream));
  Check(cudaStreamSynchronize(stream));
  long double squared_error = 0, squared_reference = 0;
  double max_abs = 0, max_rel_nonzero = 0;
  std::uint64_t different_bits = 0, signed_zero = 0, max_ulp = 0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double expected = matrices.reference_[index], actual = output[index];
    if (!std::isfinite(actual))
      throw std::runtime_error("Nonfinite GEMM output");
    const double error = std::abs(actual - expected);
    max_abs = std::max(max_abs, error);
    if (expected != 0)
      max_rel_nonzero = std::max(max_rel_nonzero, error / std::abs(expected));
    const auto left = OrderedBits(actual), right = OrderedBits(expected);
    max_ulp =
        std::max(max_ulp, (actual == 0 && expected == 0)
                              ? std::uint64_t(0)
                              : (left > right ? left - right : right - left));
    const bool differs = std::bit_cast<std::uint64_t>(actual) !=
                         std::bit_cast<std::uint64_t>(expected);
    different_bits += differs;
    signed_zero += differs && actual == 0 && expected == 0;
    const long double difference = static_cast<long double>(actual) - expected;
    squared_error += difference * difference;
    squared_reference += static_cast<long double>(expected) * expected;
  }
  const double relative_l2 =
      squared_reference == 0
          ? double(std::sqrt(squared_error))
          : double(std::sqrt(squared_error / squared_reference));
  if (!options.output_.empty()) {
    std::ofstream saved(options.output_, std::ios::binary | std::ios::trunc);
    saved.write(reinterpret_cast<const char*>(output.data()),
                output.size() * sizeof(double));
    if (!saved) throw std::runtime_error("Output write failed");
  }
  Check(cudaGraphExecDestroy(graph_exec));
  Check(cudaGraphDestroy(graph));
  std::ofstream report(options.report_);
  report << std::setprecision(17)
         << "{\n\"complete\":true,\"mode\":" << Quote(options.mode_)
         << ",\"accuracy_label\":"
         << Quote(native ? "NATIVE_FP64_NOT_ORIGINAL_ORDER_GUARANTEE"
                         : "FIXED_MANTISSA_ACCURACY_EXPERIMENTAL")
         << ",\"m\":" << matrices.m_ << ",\"n\":" << matrices.n_
         << ",\"k\":" << matrices.k_ << ",\"reference\":"
         << Quote(options.input_.empty() ? "synthetic_long_double_dot_then_FP64"
                                         : "supplied_original_ordered_capture")
         << ",\"input\":" << Quote(options.input_)
         << ",\"cublas_owner\":" << Quote(cublas_owner)
         << ",\"cudart_owner\":" << Quote(runtime_owner)
         << ",\"cublas_version\":" << library_version
         << ",\"compiled_CUDART_VERSION\":" << CUDART_VERSION
         << ",\"runtime_version\":" << runtime_version
         << ",\"driver_version\":" << driver_version
         << ",\"device\":" << Quote(properties.name)
         << ",\"SMs\":" << properties.multiProcessorCount
         << ",\"CC_major\":" << properties.major
         << ",\"CC_minor\":" << properties.minor
         << ",\"compute_type\":" << int(compute)
         << ",\"strategy\":" << int(actual_strategy)
         << ",\"mantissa_control\":" << int(actual_control)
         << ",\"mantissa_offset\":" << actual_offset
         << ",\"max_mantissa_bits_readback\":" << max_bits
         << ",\"workspace_bytes\":" << workspace_bytes
         << ",\"workspace_full_documented_ceiling\":"
         << (options.workspace_mib_ == 8192 ? "true" : "false")
         << ",\"setup_H2D_RAW_ms\":" << double(setup_end - setup_begin) / 1e6
         << ",\"shared_allocation_H2D_first_mode\":"
         << (first_mode ? "true" : "false")
         << ",\"first_GEMM_event_ms\":" << first[0]
         << ",\"first_GEMM_RAW_ms\":" << first[1]
         << ",\"graph_captured\":true,\"capture_RAW_ms\":"
         << double(capture_end - capture_begin) / 1e6
         << ",\"instantiate_RAW_ms\":"
         << double(instantiate_end - capture_end) / 1e6
         << ",\"first_graph_event_ms\":" << first_graph[0]
         << ",\"first_graph_RAW_ms\":" << first_graph[1]
         << ",\"steady_calls_per_sample\":" << options.calls_
         << ",\"warmups\":" << options.warmups_
         << ",\"steady_event_median_ms\":" << Median(event_ms)
         << ",\"steady_RAW_diagnostic_median_ms\":" << Median(raw_ms)
         << ",\"steady_event_TFLOPs\":"
         << (2.0 * matrices.m_ * matrices.n_ * matrices.k_ / Median(event_ms) /
             1e9)
         << ",\"different_reference_bits\":" << different_bits
         << ",\"signed_zero_differences\":" << signed_zero
         << ",\"max_ULP\":" << max_ulp << ",\"max_absolute_error\":" << max_abs
         << ",\"max_relative_error_nonzero_reference\":" << max_rel_nonzero
         << ",\"relative_L2_error\":" << relative_l2
         << ",\"profile_one\":" << (options.profile_one_ ? "true" : "false")
         << ",\"actual_tensor_path_proven\":false,\"solver_qualified\":false,"
            "\"event_samples_ms\":[";
  for (std::size_t index = 0; index < event_ms.size(); ++index)
    report << (index ? "," : "") << event_ms[index];
  report << "],\"RAW_diagnostic_samples_ms\":[";
  for (std::size_t index = 0; index < raw_ms.size(); ++index)
    report << (index ? "," : "") << raw_ms[index];
  report << "]}\n";
  if (!report) throw std::runtime_error("Report write failed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    if (!options.execute_) {
      std::cout << "{\"phase\":\"PLAN\",\"GPU_calls\":0,\"mode\":"
                << Quote(options.mode_)
                << ",\"compiled_CUDART_VERSION\":" << CUDART_VERSION << "}\n";
      return 0;
    }
    const auto matrices = Inputs(options);
    if (options.mode_ == "all") {
      for (const auto& mode : kModes) {
        auto current = options;
        current.mode_ = mode;
        current.report_ += "." + mode + ".json";
        if (!current.output_.empty()) current.output_ += "." + mode + ".bin";
        Execute(current, matrices);
      }
    } else {
      Execute(options, matrices);
    }
    CloseDevice();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GEMM benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
