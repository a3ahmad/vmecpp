// SPDX-License-Identifier: MIT
// Included only for the opt-in prototype, inside the energy module namespace.
struct RaggedWorkspace {
  std::array<int, 2> counts_{};
  double* terms_ = nullptr;
  double* prefix_ = nullptr;
  void* context_ = nullptr;
  int* invalid_ = nullptr;
  unsigned long long* statistics_ = nullptr;
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
  void Clear() {
    if (context_ != nullptr) DestroyGpuRaggedEnergyPrefix(context_);
    context_ = nullptr;
    if (terms_ != nullptr) cudaFree(terms_);
    if (prefix_ != nullptr) cudaFree(prefix_);
    if (invalid_ != nullptr) cudaFree(invalid_);
    if (statistics_ != nullptr) cudaFree(statistics_);
    terms_ = prefix_ = nullptr;
    invalid_ = nullptr;
    statistics_ = nullptr;
    counts_ = {};
  }
  ~RaggedWorkspace() { Clear(); }
  void Resize(const std::array<int, 2>& counts, cudaStream_t stream,
              bool capturing) {
    int device;
    Check(cudaGetDevice(&device));
    if (counts_[0] != 0 && device != device_)
      throw std::runtime_error("Ragged energy workspace cannot change device");
    if (counts == counts_) {
      if (stream != stream_) {
        if (capturing)
          throw std::runtime_error("Warm ragged energy stream before capture");
        Check(cudaDeviceSynchronize());
        RebindGpuRaggedEnergyPrefix(context_, stream);
        stream_ = stream;
      }
      return;
    }
    if (capturing)
      throw std::runtime_error("Warm ragged energy shape before capture");
    Clear();
    device_ = device;
    stream_ = stream;
    try {
      const auto count = std::size_t(counts[0]) + counts[1];
      Check(cudaMalloc(&terms_, count * sizeof(double)));
      Check(cudaMalloc(&prefix_, count * sizeof(double)));
      Check(cudaMalloc(&invalid_, 2 * sizeof(int)));
      Check(cudaMalloc(&statistics_, 4 * sizeof(unsigned long long)));
      Check(cudaMemsetAsync(statistics_, 0, 4 * sizeof(unsigned long long),
                            stream));
      context_ = CreateGpuRaggedEnergyPrefix(counts[0], counts[1], stream);
      counts_ = counts;
    } catch (...) {
      Clear();
      throw;
    }
  }
};
thread_local RaggedWorkspace ragged_workspace;
thread_local bool last_ragged_mode = false;

bool UseRaggedEnergy() {
  const char* value = std::getenv("VMECPP_GPU_LATTICE_ENERGY_RAGGED");
  if (value == nullptr || std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  throw std::runtime_error("Invalid ragged energy runtime opt-in");
}

void LaunchRaggedEnergy(GpuDiagnosticsDeviceState& m_state, bool corrupt) {
  const auto& d = m_state.dimensions_;
  if (m_state.scalars_[1] != 0.0 || d[0] < 2 || d[3] != d[0] - 1 ||
      d[4] != d[0] || d[5] < 1 || d[6] < 1 || d[0] - 1 > (1 << 21))
    throw std::runtime_error("Unsupported ragged energy configuration");
  const auto half_count = std::size_t(d[0] - 1) * d[5];
  if (half_count > (1 << 21) || std::size_t(d[6]) > (1 << 21) / half_count)
    throw std::runtime_error("Ragged energy input exceeds bounded size");
  if (m_state.sums_ == nullptr || m_state.output_ == nullptr)
    throw std::runtime_error("Null ragged energy output");
  Parameters p{};
  for (int i = 0; i < 8; ++i) {
    if (m_state.data_[i] == nullptr)
      throw std::runtime_error("Null ragged energy input");
    p.data_[i] = m_state.data_[i];
  }
  const std::array<int, 2> counts{d[0] - 1, int(half_count * d[6])};
  const auto stream = static_cast<cudaStream_t>(m_state.stream_);
  cudaStreamCaptureStatus capture;
  Check(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusInvalidated)
    throw std::runtime_error("Ragged energy capture invalidated");
  ragged_workspace.Resize(counts, stream,
                          capture == cudaStreamCaptureStatusActive);
  for (int i = 0; i < 2; ++i) {
    const int offset = i == 0 ? 0 : counts[0];
    p.terms_[i] = ragged_workspace.terms_ + offset;
    p.prefix_[i] = ragged_workspace.prefix_ + offset;
    p.counts_[i] = counts[i];
    p.invalid_[i] = ragged_workspace.invalid_ + i;
  }
  p.sums_ = m_state.sums_;
  p.output_ = m_state.output_;
  p.n_theta_ = d[5];
  p.delta_s_ = m_state.scalars_[0];
  p.statistics_ = ragged_workspace.statistics_;
  PrepareTerms<<<(counts[1] + 255) / 256, 256, 0, stream>>>(p);
  Check(cudaGetLastError());
  ProposeGpuRaggedEnergyPrefix(ragged_workspace.context_,
                               ragged_workspace.terms_,
                               ragged_workspace.prefix_, corrupt);
  for (int i = 0; i < 2; ++i) {
    Check(cudaMemcpyAsync(
        p.invalid_[i],
        GpuRaggedEnergyPrefixInvalidDevice(ragged_workspace.context_) + i,
        sizeof(int), cudaMemcpyDeviceToDevice, stream));
  }
  VerifySignedPrefixes<<<(counts[1] + 255) / 256, 256, 0, stream>>>(p);
  Check(cudaGetLastError());
  FinishCertified<<<1, 2, 0, stream>>>(p);
  Check(cudaGetLastError());
  OrderedFallback<<<3, 128, 0, stream>>>(p);
  Check(cudaGetLastError());
  Finish<<<1, 1, 0, stream>>>(p);
  Check(cudaGetLastError());
  last_ragged_mode = true;
}

void ReadRaggedEnergyStats(std::array<unsigned long long, 8>& m_statistics) {
  if (ragged_workspace.counts_[0] == 0)
    throw std::runtime_error("No ragged energy proposal ran");
  ReadGpuRaggedEnergyPrefixStats(ragged_workspace.context_, m_statistics);
  std::array<unsigned long long, 4> certified{};
  Check(cudaMemcpy(certified.data(), ragged_workspace.statistics_,
                   sizeof(certified), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 2; ++i) {
    m_statistics[4 * i + 1] = certified[2 * i];
    m_statistics[4 * i + 2] = certified[2 * i + 1];
  }
}
