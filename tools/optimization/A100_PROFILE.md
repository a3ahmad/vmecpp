# Current A100 profile: measured findings

One actual public-P3/P3 CLEAN solve completed on an A100 PCIe host with an
EPYC 7232P CPU. Independent audit `191ad881` verifies the original request/deck,
all 12 metric values, complete original NetCDF bytes/100 numeric fields and
50 diagnostic arrays. Maximum relative metric drift is **0** for every metric.
All modes, tolerances, multigrid steps, iteration caps, nstep and max_threads=1
were preserved. This is a diagnostic profile, not a new speedup comparison.

The measured NVTX call interval is **8.912758545 s**. Nsight recorded 1,038,197
target kernels, with **4.683223652 s summed duration** and **4.683223364 s
interval union**. Collection warnings remain: these are observed records,
not a complete activity attribution, GPU-idle measurement or roofline fraction.
Do not compare this interval with the earlier 6.192948 s unprofiled SXM result;
the host, GPU and instrumentation differ.

## Where device time goes

All numbers below are executed, instrumented measurements aggregated across
the actual solve's shapes. Anonymous Map names were separated using kernel-bound
mangled names and the pinned source.

| Observed family | Calls | Summed device time s |
| --- | ---: | ---: |
| Residual lattice-prefix Map scan | 27,520 | 0.286230311 |
| Ragged energy-prefix Map scan | 13,217 | 0.118712984 |
| Double scan, combined family | 40,737 | 0.228679954 |
| SeedSegments | 27,520 | 0.215943229 |
| Inverse Fourier transform | 13,761 | 0.335049614 |
| ForwardPartials | 13,760 | 0.158093887 |
| ForwardFinish | 13,760 | 0.203872734 |
| Thomas Proposal | 13,760 | 0.188554597 |
| Geometry | 13,761 | 0.067398289 |
| MHD | 13,760 | 0.081695487 |

Geometry/MHD together are **3.18% of the recorded kernel-duration sum** on this
profiled host. That demotes their block-size tuning as the primary A100 target.
The prepared 32/64-thread variants remain default-off and GPU-untested. This
fraction does not establish their unprofiled share on another GPU.

The captured device reports **108 SMs**. At ns=99, 19,542 residual Map scans
used grid86/block256 and totalled 0.215192775 s of recorded kernel duration.
One such launch can place a block on at most 86 of 108 SMs (79.63%). This is a
launch-coverage ceiling, not achieved occupancy or a roofline percentage.
It supports testing the prepared smaller scan tile; extra tiles also increase
lookback work, so a speedup remains unproven.
[Recorded launch-coverage check](../../build/optimization/a100-scan-launch-coverage1/actual1.json).

## Transfers and synchronization

| Observed operation | Calls | Bytes | Summed duration s |
| --- | ---: | ---: | ---: |
| StreamSynchronize host API | 27,779 | N/A | 5.255892057 |
| GraphLaunch host API | 27,477 | N/A | 1.882180468 |
| MemcpyAsync host API | 40,503 | N/A | 0.264808491 |
| Host to device | 1,584 | 49,945,520 | 0.005367450 |
| Device to host | 28,824 | 61,911,256 | 0.050551200 |
| Device to device | 146,263 | 22,065,656,168 | 0.264202715 |
| Device memset | 56,776 | 72,459,111,172 | 0.113458969 |

Host/device copies contribute **0.055918650 s of recorded device duration**.
The evidence directs attention to fine-grained control and submission
dependencies rather than bulk PCIe traffic. Synchronization API elapsed time
overlaps GPU execution: its union intersects recorded kernels for 4.198093104 s
and copies/memsets for 0.376759599 s. These quantities are not additive causes
or measured removable CPU cost. Profiling perturbation still needs a matched,
same-host comparison.

## Multigrid attribution

Source-compatible Geometry/Lambda launch anchors associate the 41 observed
graph groups with these conditional sums:

| ns-compatible shape | Groups | Kernel records | Kernel-duration sum s |
| --- | ---: | ---: | ---: |
| 25 | 9 | 147,779 | 0.572312608 |
| 51 | 13 | 147,632 | 0.583161744 |
| 99 | 19 | 726,570 | 3.457790958 |
| No graph membership | N/A | 16,216 | 0.069958342 |

These reconcile the totals. They are **conditional kernel sums, not stage wall
times**: graph-executable generations and some standalone work remain
unresolved. Collection warnings and family-specific coverage differences are
retained.

## What follows

1. Optimize the scan/certification chains using exact ordered arithmetic,
   guarded fusion or redundant-work removal; retain original fallback and
   original-output checks.
2. Trace the host decisions between graph submissions and synchronization.
   Measure instrumentation overhead on the same host before claiming a
   scheduling gap or GPU-control speedup.
3. Investigate Fourier kernels and required device copies/zeroing with matched
   source/shape counters. A100 FP64 and bandwidth roofline distance is still
   unmeasured. The local counter probe remains unqualified and the local card
   became occupied; no counter result or new performance gain is reported.

The successful profile/export/analysis jobs all exited 0 and retired without
cleanup signals. Retrieval and lease destruction are verified. The profiling
fix balances NVTX calls without branching on tool-dependent return values, as
recommended by [NVIDIA](https://docs.nvidia.com/cuda/profiler-users-guide/index.html).
It changes no physics. Profiling child environments now filter credential key
names; raw profiler evidence remains private because metadata can be sensitive.

Evidence: [independent execution audit](../../build/optimization/a100-whole-nsys-live7-independent1/actual-seal.json),
[redacted source-aware finding](../../build/optimization/a100-whole-stage-analysis1/FINDING-current1.md)
and [analysis seal](../../build/optimization/a100-whole-stage-analysis1/actual-current-seal1.json).
