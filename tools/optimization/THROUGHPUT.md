# Native independent-solve throughput pilot

This measured 36-call pilot schedules independent, unchanged equilibria on the
Threadripper2950X/RTX3090. It compares serial execution, six concurrent OS
processes, and six concurrent processes attached to a private CUDA MPS service.
Every solve retains the qualified CLEAN ragged build, original deck, fidelity,
tolerances and single-thread settings. Three boundaries and both P1/P3 tracks
form each six-call suite. Two complete repetitions per arm were executed.

| Arm | Repeat | Six-call makespan s | Calls/s | Matching serial / arm |
| --- | ---: | ---: | ---: | ---: |
| Serial | 0 | 86.242191 | 0.069572 | 1.000000 |
| Concurrent, no MPS | 0 | 56.224738 | 0.106715 | 1.533883 |
| Private MPS | 0 | 33.538498 | 0.178899 | 2.571439 |
| Private MPS | 1 | 33.322414 | 0.180059 | 2.600845 |
| Concurrent, no MPS | 1 | 57.439007 | 0.104459 | 1.508843 |
| Serial | 1 | 86.666451 | 0.069231 | 1.000000 |

All numbers in the table are measured or arithmetic derived from executed
cohort RAW clocks. Makespan runs from suite admission through the last observed
owned retirement, including MPS startup. Full startup/teardown-inclusive times
are retained separately in the raw records. The first serial case includes
memory-sampler overhead; MPS cohorts include membership-query overhead.
Retirement observation has approximately100ms plus query-delay uncertainty.
There are only two repetitions; no confidence interval or controlled-load
performance qualification is claimed. Resource status is UNQUALIFIED_DIAGNOSTIC:
the explicitly bound desktop service remains active and is not claimed quiet.

MPS increases throughput while increasing individual cold-call latency:
measured latency ratios versus matching serial calls range1.94-2.31x. Without
MPS they range3.37-4.05x. These cold calls include scientific imports and metric
work. Actual VMEC-call RAW latency increases2.68-3.47x. All36 distinct VMEC,
adapter and cold-call bookends are in the [latency CSV](throughput-native-latencies.csv).
The ratios must not be multiplied into the earlier CPU-versus-GPU or
ragged-energy comparisons.

Independent post-retrieval audit passed all36 canonical whole NC files,
1800 actual NPY fields,414 numeric metric observations,18 unchanged nulls,
request/deck identities, provider and FP-control checks. Each boundary/track
repeats across all arms. Every numeric metric has maximum relative drift0;
see the [per-metric CSV](throughput-native-metric-drift.csv) and
[independent seal](../../build/optimization/native-throughput-CONTINUE-independent1/actual-seal.json)
(22977484). The pilot remains outside the primary per-solve benchmark table.

Production adoption is pending. The current CGA launcher charges the sum of
individual subprocess elapsed times. Longer individual calls can consume more
metered budget despite higher throughput; this pilot does not measure actual
production charges or change accounting. Its supervisory bookends also differ
from the real shell/WSL launcher. Request-ID/cache coordination, owned timeout
cleanup and the qualified GPU launch profile need explicit integration. See the
[source-bound adoption finding](../../build/optimization/stella-mps-adoption-review1/FINDING.md).

Both MPS cohorts witnessed exactly their six owned client PIDs simultaneously.
Both private daemon/server pairs retired, and root checked their recorded
PID/birth identities again after the experiment. Complete control and foreground
logs were retrieved. Membership establishes attachment, not causal kernel
overlap or proximity to roofline. A matched timeline/counter experiment is
needed to distinguish overlap, context switching and other scheduling effects.

The source capsule and bounded continuation protocol are under
`build/optimization/native-throughput-pilot2` and
`build/optimization/native-throughput-continue-root-stage1`. Default invocation
performs source checks only. Execution requires fresh, externally generated
grants binding actual GPU/desktop identities and measured capacity evidence;
see the capsules' ROOT_RUN_PLAN.txt files. No GPU mode, clock, global MPS
service, physics setting, benchmark original or submission artifact changed.

## Smaller existing search widths

A separate30-call screen executed two P3 cases at width2 and three P1 cases at
width3, each in S,P,M,M,P,S order. Every width uses its own matching serial
composition and repetition. The sets differ, so this is not a controlled
width-only scaling curve. Original case indices, scientific settings and
per-case CPU assignments stayed fixed.

| Case set / width | Rep | Serial s | Private MPS s | Throughput ratio | Sum cold-call time / serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two P3 boundaries /2 | 0 | 32.305628 | 20.180506 | 1.600833 | 1.215236 |
| Two P3 boundaries /2 | 1 | 32.199685 | 20.071861 | 1.604220 | 1.215957 |
| Three P1 boundaries /3 | 0 | 39.338072 | 20.064019 | 1.960628 | 1.422904 |
| Three P1 boundaries /3 | 1 | 39.529851 | 20.022147 | 1.974306 | 1.429289 |

All entries derive from these executed pilot intervals. Summed cold-call time
is a measured pilot cost proxy, not a production ledger charge. No elapsed cost
is divided by concurrency or renormalized. All-in throughput ratios including
service teardown are1.59763/1.60094 at width2 and1.95657/1.97023 at width3.
Per-call latency increases, while each batch finishes earlier. The producer
passed all30 exact scientific gates and four normal private-MPS lifecycles.
Independent audit95937424 verifies30 whole NC files,1500 actual NPY fields,
342 numeric metrics,18 unchanged nulls, all identities/providers/owners and
four exact requested-width lifecycles. All eight service identities were checked
retired again after the trial. See the
[width audit](../../build/optimization/native-throughput-widths-independent1/actual-seal.json)
and [per-width/per-metric CSV](throughput-width-metric-drift.csv).
Resource status remains unqualified.

The source-bound adoption review identifies these as existing search widths;
using six candidates in those rounds would change the search. Pair acquisition
has different independent-work availability. No production scheduler, budget,
cache, search order or submission artifact changed in either pilot.
[Thirty-call records](../../build/optimization/native-throughput-widths-results1/terminal.json)
and [width timing/cost table](../../build/optimization/native-throughput-widths-results1/throughput-comparison.json).

Actual six-wide evidence: [36-call comparison](../../build/optimization/native-throughput-pilot2-results2/throughput-comparison.json),
[scientific records](../../build/optimization/native-throughput-pilot2-results2/terminal.json),
[outer retirement](../../build/optimization/native-throughput-continue-root-results1/terminal.json)
and [private-service log manifest](../../build/optimization/native-throughput-continue-root-results1/private-MPS-evidence1-manifest.json).
