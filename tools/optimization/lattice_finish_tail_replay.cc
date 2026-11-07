// SPDX-License-Identifier: MIT
// Control the production tail before ORIGINAL residual fallback can hide a
// bad scalar write. This harness does not replace the production fallback.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_lattice_prefix.h"

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
T Symbol(void* library, const char* name) {
  auto value = reinterpret_cast<T>(dlsym(library, name));
  if (value == nullptr) throw std::runtime_error(name);
  return value;
}
struct Api {
  decltype(&CreateGpuLatticePrefix) create_;
  decltype(&DestroyGpuLatticePrefix) destroy_;
  decltype(&ProposeGpuLatticePrefix) propose_;
  decltype(&FinishGpuLatticePrefix) finish_;
  decltype(&ProposeAndFinishGpuLatticePrefix) fused_;
  decltype(&ReadGpuLatticePrefixStats) statistics_;
  decltype(&ReadGpuLatticePrefixFlags) flags_;
  explicit Api(void* library)
      : create_(Symbol<decltype(create_)>(library, "CreateGpuLatticePrefix")),
        destroy_(
            Symbol<decltype(destroy_)>(library, "DestroyGpuLatticePrefix")),
        propose_(
            Symbol<decltype(propose_)>(library, "ProposeGpuLatticePrefix")),
        finish_(Symbol<decltype(finish_)>(library, "FinishGpuLatticePrefix")),
        fused_(Symbol<decltype(fused_)>(library,
                                        "ProposeAndFinishGpuLatticePrefix")),
        statistics_(Symbol<decltype(statistics_)>(library,
                                                  "ReadGpuLatticePrefixStats")),
        flags_(Symbol<decltype(flags_)>(library, "ReadGpuLatticePrefixFlags")) {
  }
};
struct Resources {
  const Api& api_;
  cudaStream_t stream_ = nullptr;
  void* context_ = nullptr;
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  double* sums_ = nullptr;
  std::vector<cudaGraphExec_t> executions_;
  std::vector<cudaGraph_t> graphs_;
  explicit Resources(const Api& api) : api_(api) {}
  ~Resources() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    for (auto execution : executions_) cudaGraphExecDestroy(execution);
    for (auto graph : graphs_) cudaGraphDestroy(graph);
    if (context_ != nullptr) api_.destroy_(context_);
    if (terms_ != nullptr) cudaFree(terms_);
    if (prefix_ != nullptr) cudaFree(prefix_);
    if (sums_ != nullptr) cudaFree(sums_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
};
std::size_t Run(Resources& m_resources, bool fused, bool corrupt, bool graph) {
  auto enqueue = [&]() {
    Check(cudaMemsetAsync(m_resources.sums_, 0xa5, 3 * sizeof(double),
                          m_resources.stream_));
    if (fused) {
      m_resources.api_.fused_(m_resources.context_, m_resources.terms_,
                              m_resources.prefix_, m_resources.sums_, corrupt);
    } else {
      m_resources.api_.propose_(m_resources.context_, m_resources.terms_,
                                m_resources.prefix_, corrupt);
      m_resources.api_.finish_(m_resources.context_, m_resources.prefix_,
                               m_resources.sums_);
    }
  };
  if (!graph) {
    enqueue();
    return 0;
  }
  cudaGraph_t captured = nullptr;
  Check(cudaStreamBeginCapture(m_resources.stream_,
                               cudaStreamCaptureModeThreadLocal));
  enqueue();
  Check(cudaStreamEndCapture(m_resources.stream_, &captured));
  std::size_t nodes = 0;
  Check(cudaGraphGetNodes(captured, nullptr, &nodes));
  m_resources.graphs_.push_back(captured);
  cudaGraphExec_t execution = nullptr;
  Check(cudaGraphInstantiate(&execution, captured, 0));
  m_resources.executions_.push_back(execution);
  Check(cudaGraphLaunch(execution, m_resources.stream_));
  return nodes;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Usage: tail-replay original.so candidate.so");
    if (dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL) == nullptr)
      throw std::runtime_error(dlerror());
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(dlerror());
    const Api api(library);
    int cases = 0;
    const double sentinel = std::bit_cast<double>(0xa5a5a5a5a5a5a5a5ULL);
    for (int stride : {257, 32769}) {
      Resources resources(api);
      Check(
          cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking));
      resources.context_ = api.create_(3, stride, resources.stream_);
      std::vector<double> terms(3 * stride);
      std::vector<double> expected(3 * stride);
      for (int component = 0; component < 3; ++component) {
        double sum = 0.0;
        for (int i = 0; i < stride; ++i) {
          const double value =
              component == 0
                  ? (i == 0 ? 1.0 : (i % 2 ? 0x1p-53 : 0x1p-52))
                  : (component == 1 ? std::numeric_limits<double>::denorm_min()
                                    : 0.0);
          terms[component * stride + i] = value;
          sum += value;
          expected[component * stride + i] = sum;
        }
      }
      Check(cudaMalloc(&resources.terms_, terms.size() * sizeof(double)));
      Check(cudaMalloc(&resources.prefix_, terms.size() * sizeof(double)));
      Check(cudaMalloc(&resources.sums_, 3 * sizeof(double)));
      Check(cudaMemcpyAsync(resources.terms_, terms.data(),
                            terms.size() * sizeof(double),
                            cudaMemcpyHostToDevice, resources.stream_));
      std::array<std::size_t, 2> separate_nodes{};
      for (bool fused : {false, true}) {
        for (bool corrupt : {false, true}) {
          for (bool graph : {false, true}) {
            std::array<unsigned long long, 4> before{}, after{};
            api.statistics_(resources.context_, before);
            const auto nodes = Run(resources, fused, corrupt, graph);
            if (graph) {
              if (!fused)
                separate_nodes[corrupt] = nodes;
              else if (nodes + 1 != separate_nodes[corrupt])
                throw std::runtime_error(
                    "Fusion did not remove exactly one graph node");
            }
            std::array<double, 3> sums{};
            Check(cudaMemcpyAsync(sums.data(), resources.sums_, sizeof(sums),
                                  cudaMemcpyDeviceToHost, resources.stream_));
            std::vector<double> prefix(terms.size());
            Check(cudaMemcpyAsync(prefix.data(), resources.prefix_,
                                  prefix.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost, resources.stream_));
            Check(cudaStreamSynchronize(resources.stream_));
            std::array<int, 3> flags{};
            api.flags_(resources.context_, flags.data());
            api.statistics_(resources.context_, after);
            if (after[0] - before[0] != 1)
              throw std::runtime_error("Tail statistics call count changed");
            const auto failed =
                std::count_if(flags.begin(), flags.end(),
                              [](int value) { return value != 0; });
            if (after[2] - before[2] !=
                    static_cast<unsigned long long>(failed) ||
                after[1] - before[1] != 3ULL - failed)
              throw std::runtime_error(
                  "Tail component statistics differ from flags");
            if ((corrupt && failed != 3) || (!corrupt && flags[0] != 0))
              throw std::runtime_error("Mandatory certificate control failed");
            for (int component = 0; component < 3; ++component) {
              if (flags[component]) {
                if (std::memcmp(&sums[component], &sentinel, sizeof(double)))
                  throw std::runtime_error(
                      "Invalid sum changed before ORIGINAL fallback");
              } else {
                const auto offset = component * stride;
                if (std::memcmp(prefix.data() + offset,
                                expected.data() + offset,
                                stride * sizeof(double)) ||
                    std::memcmp(&sums[component],
                                expected.data() + offset + stride - 1,
                                sizeof(double)))
                  throw std::runtime_error(
                      "Certified prefix/final sum bits changed");
              }
            }
            ++cases;
          }
        }
      }
      // A prior fused call must not leak its sums pointer into generic Propose.
      Check(cudaMemsetAsync(resources.sums_, 0xa5, 3 * sizeof(double),
                            resources.stream_));
      api.propose_(resources.context_, resources.terms_, resources.prefix_,
                   false);
      std::array<double, 3> sums{};
      Check(cudaMemcpyAsync(sums.data(), resources.sums_, sizeof(sums),
                            cudaMemcpyDeviceToHost, resources.stream_));
      Check(cudaStreamSynchronize(resources.stream_));
      for (double value : sums)
        if (std::memcmp(&value, &sentinel, sizeof(double)))
          throw std::runtime_error(
              "Generic proposal now writes a prior sums buffer");
    }
    std::cout << "{\"tail_cases\":" << cases
              << ",\"invalid_sums_untouched\":true,\"full_certificate\":true,"
                 "\"generic_proposal_side_effect_unchanged\":true,"
                 "\"one_graph_node_removed\":true,"
                 "\"timed\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
