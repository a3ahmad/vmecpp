# Exact-parity VMEC acceleration: current findings

For the completed RTX 5090 pass, read [GB202.md](GB202.md). It measured
1.62-1.64x independent-solve throughput with zero metric drift. Historical
comparisons below are separate and are not multiplied into that result.

The primary native six-case comparison remains 13.74-14.28x faster VMEC than
stock CPU, with bit-exact reported metrics and equilibrium fields. Its complete
table/build recipe is in [SUMMARY.md](SUMMARY.md) and [README.md](README.md).
The following newer measured results are separate comparisons; none are
multiplied into that CPU ratio.

| Executed comparison | Measured result | Scope |
| --- | --- | --- |
| CLEAN ragged vs selected, native | 1.0296-1.0338x VMEC | 72 calls, three boundaries/two tracks |
| CLEAN ragged vs selected, A100 | 6.525313 s to 6.192948 s; paired 1.052811x | 14 calls, one P3 fixture; five measured pairs |
| Private MPS, width 2 P3 | 32.20-32.31 s to 20.07-20.18 s; 1.601-1.604x throughput | 12 calls, two boundaries |
| Private MPS, width 3 P1 | 39.34-39.53 s to 20.02-20.06 s; 1.961-1.974x throughput | 18 calls, three boundaries |
| Private MPS, width 6 mixed | 86.24-86.67 s to 33.32-33.54 s; 2.571-2.601x throughput | 36 calls, three boundaries/two tracks |

All entries are executed measurements or arithmetic derived from them.
The 80 calls in the newer A100/MPS campaigns passed independent original-output
checks, including 924 numeric metrics and 36 unchanged nulls. Maximum relative
metric drift is 0 for every numeric metric. Physical repeats match. MPS pilots
have only two repetitions and unqualified resource conditions; each width is
compared only to its matching serial composition. A100 full fresh-call
uncertainty includes no change. These limits keep the results out of any
generalized scaling or whole-solver roofline claim.

Changes: the qualified GPU path uses strict FP64, resident arrays, ordered
Fourier grouping and cached Thomas work. CLEAN ragged energy avoids unnecessary
prefix work while retaining exact certification/fallback. The new private-MPS
runner overlaps independent single-threaded solves in separate OS processes;
owned services and workers have finite lifetimes and explicit retirement proof.
The capped-CUB residual rewrite passed normal and all three sanitizers, but
its small component-only timing signal is not selected into the solver.

Throughput comes with a cost tradeoff. Summed measured cold-call time increased
about 22% at width 2, 42-43% at width 3 and 106-110% at width 6. These are pilot
interval sums, not production charges. The actual CGA ledger charges summed
per-request elapsed time; no accounting was renormalized. Cache/request-ID,
timeout ownership and GPU-profile integration must be addressed before deployment.

Profile evidence also changes priorities: in this accelerated A100 P3 fixture,
VMEC occupies 39-41% of the complete adapter envelope on the same clock.
The remaining work is material, but is not all imports or removable CPU cost.
The [current A100 PCIe profile](A100_PROFILE.md) passed original-output checks.
Its 1,038,197 recorded kernels prioritize scan/certification and Fourier work;
Geometry/MHD are 3.18% of recorded kernel-duration sum. Recorded host/device
copies total 0.055919 s. These instrumented observations are not unprofiled
shares, removable synchronization cost or measured stage wall times.
Near-roofline performance remains unproven. Default-off scan and conditional
graph prototypes are next; no additional speedup is claimed.

Physics, modes, tolerances, multigrid schedule, iteration caps, nstep and
single-thread VMEC stay unchanged. No approximate precision, altered accounting,
benchmark original or submission edit was introduced. All trial-owned MPS
services retired, and the latest confirmed A100 lease was destroyed after
retrieval. A subsequent HTTP400 creation attempt returned no confirmed instance
ID; its ambiguity guard remains active.

Evidence and limitations: [native energy](ENERGY_BATCHING.md),
[A100 timings/per-metric drift](A100_RAGGED.md),
[MPS timings/per-metric drift/cost](THROUGHPUT.md),
[profile evidence](ROOFLINE.md) and [active work](NEXT_STEPS.md).

## Latest measured counter diagnostic

Three [native Fourier captures](COUNTER_PROFILE.md) passed independent audit:
inverse/partials/finish measured 53.728/55.968/37.984 us and 70.38/68.95/59.77%
FP64-pipe activity on RTX 3090. Original packet endpoints were bitwise exact.
This adds no solver speedup and establishes no A100 or whole-solver roofline.
