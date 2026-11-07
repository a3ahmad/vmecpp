// SPDX-License-Identifier: MIT
// Experimental opt-in exact-prefix path and standalone test APIs.
#ifndef VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_PREFIX_H_
#define VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_PREFIX_H_

#include <array>

// All arrays are device resident. One context may have only one active caller.
// Calls require a live caller-owned explicit stream. The context must outlive
// every graph referring to its allocations; cleanup tolerates a retired stream.
extern "C" void* CreateGpuLatticePrefix(int components, int stride,
                                        void* stream);
extern "C" void DestroyGpuLatticePrefix(void* m_context);
// Proposal + complete finite bitwise recurrence certificate + ordered fallback.
// Every output prefix is materialized, including on fallback.
extern "C" void RunGpuLatticePrefix(void* m_context, const double* terms,
                                    double* m_prefix);
// Negative-control entry point: flip one proposed prefix bit before verifying.
extern "C" void RunGpuLatticePrefixCorrupted(void* m_context,
                                             const double* terms,
                                             double* m_prefix);
// Production-shape adapter APIs: no naive ordered fallback is launched here.
extern "C" void ProposeGpuLatticePrefix(void* m_context, const double* terms,
                                        double* m_prefix, bool corrupt);
extern "C" void FinishGpuLatticePrefix(void* m_context, const double* prefix,
                                       double* m_sums);
// Opt-in production tail: same proposal/certificate, then one statistics+sum
// kernel. Invalid sums remain untouched for ORIGINAL force-based fallback.
// Existing generic full-prefix and separate proposal/finish APIs are unchanged.
extern "C" void ProposeAndFinishGpuLatticePrefix(void* m_context,
                                                 const double* terms,
                                                 double* m_prefix,
                                                 double* m_sums, bool corrupt);
extern "C" const int* GpuLatticePrefixInvalidDevice(void* context);
// Caller must synchronize all old uses outside capture before rebinding.
extern "C" void RebindGpuLatticePrefix(void* m_context, void* stream);
extern "C" void LaunchGpuLatticeOrdered(const double* terms, double* m_prefix,
                                        int components, int stride,
                                        void* stream);
// Terms are already rounded squares; this baseline writes only final sums.
extern "C" void LaunchGpuLatticeOrderedSums(const double* terms, double* m_sums,
                                            int components, int stride,
                                            void* stream);
// Cumulative calls, accepted components, fallback components, normal segments.
extern "C" void ReadGpuLatticePrefixStats(
    void* m_context, std::array<unsigned long long, 4>& m_stats);
extern "C" void ReadGpuLatticePrefixFlags(void* m_context, int* m_flags);

#endif  // VMECPP_TOOLS_OPTIMIZATION_GPU_LATTICE_PREFIX_H_
