# Measured native Fourier counters

Three actual protected-counter captures completed on RTX3090, driver595.99.02,
82SMs. Independent seal `c981c06c` verifies all205 retrieved files,63 counter
values, original packet checks, target identities, source hashes, retirement
and host resource checks. These are kernel diagnostics, not solver speedups.

| ns99 kernel | Measured GPU us | Derived GFLOP/s | Measured FP64 pipe % | Derived DRAM GB/s |
| --- | ---: | ---: | ---: | ---: |
| Inverse Fourier | 53.728 | 227.223 | 70.38 | 506.044 |
| Forward partials | 55.968 | 210.336 | 68.95 | 122.584 |
| Forward finish | 37.984 | 162.470 | 59.77 | 62.598 |

Every number above comes from executed captures. Rates divide actual instruction
counts or bytes by recorded kernel duration. FP64 FLOPs count DADD+DMUL+2*DFMA;
all three DFMA counts were0. These counters used10 replay passes, unmodified
clocks and uncontrolled caches; the profiler's warnings remain part of evidence.
No confidence interval or sustained calibration is claimed.

## Distance from the arithmetic ceiling

At the three recorded SM clocks (1.97042/1.86238/1.66347GHz), the architectural
strict add/multiply ceilings are323.149/305.431/272.809GFLOP/s. The achieved
fractions are70.315/68.865/59.555%. These are derived theoretical comparisons,
not measured sustained rooflines. At fixed clock and unchanged instruction work,
the ideal arithmetic headroom is about1.42x/1.45x/1.68x. Eliminating redundant
work can change that bound. FP64 and DRAM percentages must not be multiplied.

The low generic issue rates (5.94/3.95/5.61%) are not recoverable-idle fractions:
consumer FP64 instruction throughput is much lower than generic issue capacity.
Measured active-warps percentages are25.61/43.10/24.90. Short-scoreboard stalls
are60.28/28.67/74.71%, and long-scoreboard stalls31.83/5.38/4.19%; these identify
questions about dependencies and latency hiding, not additive wall-time losses.

## Fidelity and scope

The original unchanged inverse/forward consumers use the same compiled launch
functions as the selected resident backend. Shapes were1040x32 for inverse,
186x3 blocks of128 for partials and86x3 blocks of128 for finish. Each capture
selected matching launch11, after one original check and10 warmups.

Two unprofiled controls and three profiled applications each completed111 calls
and returned zero different values and zero maximum absolute difference at the
original first/final packet checks. Intermediate packets were not all checked.
No equilibrium solver or reported benchmark metrics ran in this diagnostic.
The benchmark timing table is unchanged.

The desktop daemon remained present; resource label is
`DESKTOP_BACKGROUND_DIAGNOSTIC`. Host checks before/after matched its PID/birth
and excluded other compute clients. Container visibility alone was not treated
as global quiet. The isolated container had profiling capabilities; no host
driver policy, clocks or other processes were changed. All children retired,
data were retrieved and the exact labeled container was removed.

The first launch failed before GPU work because of a shallow helper path. The
second collected one valid inverse capture but failed its old clock-unit parser.
Both failed attempts remain preserved. The successful run changed only the
caller layout and a parser accepting `hz` as cycles/second; GPU math was unchanged.

These results support good arithmetic utilization for these three consumer-GPU
kernels. They do not establish an A100 or whole-solver roofline. The next profiling
targets are CURRENT residual scans/seeding and submission/synchronization costs.

Evidence: [independent finding](../../build/optimization/native3090-counter-actual3-independent1/FINDING.md),
[raw-derived rates](../../build/optimization/native3090-counter-actual3-independent1/rates.csv),
[audit](../../build/optimization/native3090-counter-actual3-independent1/audit.json)
and [seal](../../build/optimization/native3090-counter-actual3-independent1/seal.json).
