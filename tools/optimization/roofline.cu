// Empirical FP64 streaming and independent FMA calibration ceilings.
// Calibration only: these kernels do not calculate equilibrium results.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

__global__ void Initialize(double* m_a, double* m_b, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    m_a[index] = 0.25 + (index % 17) * 0.001;
    m_b[index] = 0.5;
  }
}

__global__ void Stream(const double* a, const double* b, double* m_c,
                       size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    m_c[index] = a[index] + 0.125 * b[index];
  }
}

__global__ void Compute(const double* a, double* m_c, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  double x0 = a[index];
  double x1 = x0 + 0.01;
  double x2 = x0 + 0.02;
  double x3 = x0 + 0.03;
  double x4 = x0 + 0.04;
  double x5 = x0 + 0.05;
  double x6 = x0 + 0.06;
  double x7 = x0 + 0.07;
#pragma unroll 1
  for (int iteration = 0; iteration < 256; ++iteration) {
    x0 = fma(x0, 0.999, 0.001);
    x1 = fma(x1, 0.999, 0.001);
    x2 = fma(x2, 0.999, 0.001);
    x3 = fma(x3, 0.999, 0.001);
    x4 = fma(x4, 0.999, 0.001);
    x5 = fma(x5, 0.999, 0.001);
    x6 = fma(x6, 0.999, 0.001);
    x7 = fma(x7, 0.999, 0.001);
  }
  m_c[index] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

int main() {
  constexpr size_t kCount = 1 << 25;
  constexpr size_t kComputeCount = 1 << 20;
  constexpr int kRepeats = 20;
  constexpr int kBlockSize = 256;
  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  Check(cudaMalloc(&a, kCount * sizeof(double)));
  Check(cudaMalloc(&b, kCount * sizeof(double)));
  Check(cudaMalloc(&c, kCount * sizeof(double)));
  Initialize<<<kCount / kBlockSize, kBlockSize>>>(a, b, kCount);
  Stream<<<kCount / kBlockSize, kBlockSize>>>(a, b, c, kCount);
  Compute<<<kComputeCount / kBlockSize, kBlockSize>>>(a, c, kComputeCount);
  Check(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t end;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&end));
  Check(cudaEventRecord(start));
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    Stream<<<kCount / kBlockSize, kBlockSize>>>(a, b, c, kCount);
  }
  Check(cudaEventRecord(end));
  Check(cudaEventSynchronize(end));
  float milliseconds = 0;
  Check(cudaEventElapsedTime(&milliseconds, start, end));
  const double bandwidth =
      kCount * sizeof(double) * 3.0 * kRepeats / (milliseconds * 1.0e6);
  std::printf(
      "{\"measurement\":\"derived from CUDA event timing\","
      "\"stream_GB_per_second\":%.6f,",
      bandwidth);
  Check(cudaEventRecord(start));
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    Compute<<<kComputeCount / kBlockSize, kBlockSize>>>(a, c, kComputeCount);
  }
  Check(cudaEventRecord(end));
  Check(cudaEventSynchronize(end));
  Check(cudaEventElapsedTime(&milliseconds, start, end));
  const double compute =
      kComputeCount * 256.0 * 8 * 2 * kRepeats / (milliseconds * 1.0e9);
  std::printf("\"FP64_FMA_TFLOP_per_second\":%.6f,", compute);
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  Check(cudaStreamCreate(&stream));
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    Stream<<<kCount / kBlockSize, kBlockSize, 0, stream>>>(a, b, c, kCount);
  }
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  Check(cudaGraphLaunch(executable, stream));
  Check(cudaStreamSynchronize(stream));
  Check(cudaEventRecord(start, stream));
  Check(cudaGraphLaunch(executable, stream));
  Check(cudaEventRecord(end, stream));
  Check(cudaEventSynchronize(end));
  Check(cudaEventElapsedTime(&milliseconds, start, end));
  const double graph_bandwidth =
      kCount * sizeof(double) * 3.0 * kRepeats / (milliseconds * 1.0e6);
  std::printf("\"graph_stream_GB_per_second\":%.6f}\n", graph_bandwidth);
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
  Check(cudaStreamDestroy(stream));
  Check(cudaGetLastError());
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(end));
  Check(cudaFree(a));
  Check(cudaFree(b));
  Check(cudaFree(c));
}
