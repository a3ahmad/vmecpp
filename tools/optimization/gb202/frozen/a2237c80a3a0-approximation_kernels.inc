// SPDX-License-Identifier: MIT
// Private positive-RN-tree guesses. No exact residual output is published here.
#if VMECPP_RESIDUAL_APPROX_FUSION
constexpr int kApproxThreads = 128;
constexpr int kApproxItems = 4;
constexpr int kApproxTile = kApproxThreads * kApproxItems;
struct Approximation {
  double* summaries_;
  double* carries_;
  double* guesses_;
  int tiles_;
};

__global__ void FusedTermsApproximation(Views m_views, ResidualTerms terms,
                                       Approximation m_approx, bool forced) {
  const int component = blockIdx.y;
  const auto m_p = m_views[component];
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int reset_stride = gridDim.x * blockDim.x;
  // These disjoint metadata writes need no dependency on term evaluation.
  // The next kernel provides the same global boundary before Map consumers.
  for (int i = linear; i < kBins; i += reset_stride) {
    m_p.starts_[i] = INT_MAX;
    m_p.ends_[i] = -1;
  }
  for (int i = linear; i < kBitmapWords; i += reset_stride) m_p.bitmap_[i] = 0;
  if (linear == 0) *m_p.invalid_ = forced ? kForced : 0;

  cuda::std::array<double, kApproxItems> values{};
  for (int item = 0; item < kApproxItems; ++item) {
    const int local = blockIdx.x * kApproxTile + threadIdx.x * kApproxItems + item;
    const int coefficient = local / 2;
    double term = 0.0;
    if (local < m_p.count_) {
      // Literal original force-square body and excluded R/Z +0 semantics.
      if (component == 2 || coefficient < terms.rz_count_) {
        const double force = terms.forces_[2 * component + local % 2][coefficient];
        term = __dmul_rn(force, force);
      }
      terms.terms_[component * terms.stride_ + local] = term;
    }
    values[item] = term;
  }
  using Scan = cub::BlockScan<double, kApproxThreads>;
  __shared__ typename Scan::TempStorage storage;
  double aggregate = 0.0;
  double thread_total = values[0];
  for (int item = 1; item < kApproxItems; ++item)
    thread_total = __dadd_rn(thread_total, values[item]);
  double thread_prefix = 0.0;
  Scan(storage).ExclusiveSum(thread_total, thread_prefix, aggregate);
  for (int item = 0; item < kApproxItems; ++item) {
    const int local = blockIdx.x * kApproxTile + threadIdx.x * kApproxItems + item;
    thread_prefix = __dadd_rn(thread_prefix, values[item]);
    if (local < m_p.count_)
      m_approx.guesses_[component * m_p.count_ + local] = thread_prefix;
  }
  if (threadIdx.x == 0)
    m_approx.summaries_[component * m_approx.tiles_ + blockIdx.x] = aggregate;
}

__global__ void ScanApproximationSummaries(Approximation m_approx) {
  const int component = blockIdx.x;
  const int index = threadIdx.x;
  const double value = index < m_approx.tiles_
                           ? m_approx.summaries_[component * m_approx.tiles_ + index]
                           : 0.0;
  using Scan = cub::BlockScan<double, kApproxThreads>;
  __shared__ typename Scan::TempStorage storage;
  double prefix = 0.0;
  Scan(storage).ExclusiveSum(value, prefix);
  if (index < m_approx.tiles_)
    m_approx.carries_[component * m_approx.tiles_ + index] = prefix;
}

__global__ void FixupApproximation(Views m_views, Approximation approx) {
  const int component = blockIdx.y;
  const auto m_p = m_views[component];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m_p.count_) return;
  // Carry and local tree cover DISJOINT sets of original terms, once each.
  const int tile = index / kApproxTile;
  const double carry = approx.carries_[component * approx.tiles_ + tile];
  approx.guesses_[component * m_p.count_ + index] =
      __dadd_rn(carry, m_p.guess_[index]);
}
#endif
