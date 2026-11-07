# VMEC++ optimization: measured results

The latest RTX 5090 result and source handoff are in [GB202.md](GB202.md).
Its measured 1.62-1.64x batching gain is separate from the earlier results below.

The [current A100 profile](A100_PROFILE.md) completed one original-exact P3
solve and recorded 1,038,197 kernels. It changes optimization priorities without
changing the timing table: no new speedup or near-roofline claim is made.

The newer [independent-solve throughput pilot](THROUGHPUT.md) executed36 calls
on native RTX3090. Private MPS completed six-call suites in33.32-33.54s versus
86.24-86.67s serial:2.57-2.60x throughput with exact outputs. Individual VMEC
latency increased2.68-3.47x. This is an unqualified two-repeat scheduling pilot,
not a per-solve gain or a production accounting change; it stays separate from
the primary table and does not compound any ratio below.

The [current A100 P3 pilot](A100_RAGGED.md) executed14 evaluations: two traced
qualifications and six untraced pairs, with pair0 excluded from timing. CLEAN
ragged improved the VMEC paired median1.052811x versus selected on that one
fixture, with zero drift; consistent full fresh-call gains remain unproven.
This is a separate GPU-versus-GPU result, not a full A100 matrix or roofline.

The newer [native energy-batching study](ENERGY_BATCHING.md) measured
1.0296-1.0338x VMEC paired medians versus the selected GPU build across six
cases, with zero metric drift and 30/30 measured pairs improved. This roughly 3%
incremental native result is preferred within its fixed scope; the earlier local
study remains separate. Neither result compounds the original-CPU table below
or establishes whole-solver near-roofline performance.

The selected build gives **13.7369-14.2841x faster VMEC calls** with zero metric
and equilibrium-field drift. The table uses a separate measured timer: the
post-import adapter body, including settings, metrics and artifact writes.
All ratios below come from the same 72 executions; none are compounded.

| Boundary | Track | Original CPU body s | Selected GPU body s | Speedup | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Third frozen fixture | P1 | 84.375638 | 8.035474 | 10.500393x | 0 |
| Third frozen fixture | P3 | 87.303398 | 10.670843 | 8.181490x | 0 |
| public_p2_seed | P1 | 102.987779 | 9.178441 | 11.220618x | 0 |
| public_p2_seed | P3 | 106.066896 | 11.792965 | 8.994082x | 0 |
| public_p3_seed | P1 | 113.175319 | 10.191955 | 11.104377x | 0 |
| public_p3_seed | P3 | 115.873470 | 12.831189 | 9.030611x | 0 |

Measured on Threadripper 2950X / RTX 3090, with VMEC/OpenMP/BLAS1: one excluded
warmup pair and five alternating measured pairs per case. Every numeric metric
has drift 0; three null QI entries stay null. Repeated outputs are bitwise exact.
[Per-metric CSV](../../build/optimization/native-originalcpu-seed72-final-report/per-metric-drift.csv)
and [timer scopes](../../build/optimization/native-originalcpu-seed72-timing-scopes/tables.md).

Hardware limitation: both this RTX3090 cohort and the newer RTX5090 cohort use
consumer GPUs whose native FP64 throughput is1/64 of FP32, versus1/2 on A100
([NVIDIA architecture table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)).
Their roofline fractions do not establish datacenter roofline performance.
The A100 CLEAN pilot above covers one P3 case; its full six-case matrix remains
pending.

What changed: opt-in strict-FP64 GPU kernels, resident data, SparseSeed,
N3 Fourier grouping and Thomas prefetch8/8. Original CPU convergence and damping
control remain selected. [Frozen build recipe](README.md).

What did not change: physics, modes, precision, tolerances, grid schedule,
iteration caps, nstep and single-thread settings. Oracle installation,
benchmark data and submission artifacts remain untouched.

Historical profile evidence: an earlier A100 trace recorded 1,170,370 kernels, 880,627 below
5 us. Scans/seeding, Fourier transforms and launch/control dependencies justify
the current work. Instrumented gaps do not measure unprofiled idle or a roofline
percentage. Whole-solver near-roofline performance remains unproven.
[Profile evidence and priorities](ROOFLINE.md).

Experimental, not selected: the resident-controller rewrite passed all six cases
twice, with original-exact metrics,100NC fields,50diagnostics and repeat traces.
Its focused component passed all three sanitizers (seal65315782). Historical
legacy190 timeout and uncontrolled local timings remain separate and excluded.

An earlier native smoke remains excluded because its CPU authority differed
from canonical metrics by one ULP. Four saved-WOut replays isolated dependency
effects. A private NumPy2.2.6/SciPy1.17.1 environment then passed six actual CPU
solves against canonical whole NC bytes, diagnostics and metrics (audit8826e8d3).
The installed oracle is untouched; no values are rounded or replaced.

The default-OFF pinned-packet variant passed all six cases twice:12 entire
original NC files byte-exact,600 diagnostic fields,138 zero-drift numeric metric
entries,6 unchanged nulls and exact repeats (sealbbaf0e39). The bounded WHILE2
variant passed two publicP3/P3 solves with exact outputs (sealb2cddb45). Both
preserve all164 original device instruction occurrences and pass focused route
and rejection tests. These are correctness results; new-variant sanitizer and
performance qualification remain pending.

The versioned25-iteration loop passed all six cases twice with exact outputs
(seal478b1964). However, its native untraced P3 screen LOST: three measured
alternating pairs, after one excluded warm pair, gave median VMEC8.015067s for
selected versus9.191487s for versioned25, a14.6776% paired slowdown. All eight
calls passed exact-output gates. Adapter medians were14.605921s versus15.759606s.
It stays unselected and outside the main table. Its10.8x fewer returned chunks
did not translate into speed. Next: matched trace attribution before another
rewrite. [Actual timing metadata](../../build/optimization/native-private-v2-screen1-actual/screen-comparison.json).

Adjacent metadata fusion also LOST its native untraced P3 screen: all8 calls
passed saved exact-output gates and clean ownership checks, but three measured
pairs gave selected/fusion VMEC ratios0.873607/0.878615/0.881882 and a13.8155%
median paired slowdown (audit5ec74742). It stays unselected; this is screen-only
evidence with no full SAN qualification or incremental claim against old V2.
[Actual fusion screen](../../build/optimization/native-private-fusion-screen1-actual/screen-comparison.json).

A 48-evaluation scheduling study preserves exact outputs, but throughput ratios
1.021493/1.639494/0.922356 are uncontrolled and excluded. Native-thread/GIL
concurrency is established; actual GPU overlap and MPS state are unmeasured.
[Detailed results](RESULTS.md) and [active evidence/handoff](NEXT_STEPS.md).
