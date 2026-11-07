# VMEC++ exact-parity acceleration: results and ongoing work

## Additional hardware and scheduling pilots, 2026-11-04

These executed pilots are separate from the qualified direct72 table below.
No pilot ratio is multiplied into that table, and no new controller is selected.

The later native two-iteration controller pilot completed8 exact evaluations
(two excluded warmups, three alternating measured pairs). VMEC RAW medians
were7.993949660s selected versus8.734436741s candidate: the candidate took
9.263% longer, with every paired ratio below1. Root independently recomputed
all239 transferred hashes,1168 raw numeric fields and canonical identities.
All965 frozen native inputs stayed unchanged and all processes were reaped.
The candidate is rejected. See `native-chunk2-pilot3/FINDING.md` and
`native-chunk2-root-audit1/audit.json`; this is one-case pilot evidence only.

| Executed pilot | Measured evidence | Derived result and limit |
| --- | --- | --- |
| A100 PCIe, six cases |18 equilibria: one matchedCPU144e and two GPU calls per case; all metrics, requests, decks, raw NC and diagnostic arrays exact | GPU VMEC medians5.12-6.75s;18.74-19.65x versus matchedCPU144e. No stockwheel baseline, alternating pairs, excluded warmups or confidence intervals on this host |
| RTX3090 fresh-process concurrency |16 evaluations, two public boundaries on P3; three measured paired workloads; zero metric/raw drift | Median paired request-throughput ratio1.290957x. Individual VMEC solves became slower; includes startup/setup/import work |
| RTX3090 Fourier launch layout |31 measured warmed pairs on one captured ns99 input; exact component outputs and guards | Graph component ratio1.037127x; no full-solver gain established |

Receipts: `build/optimization/a100-hardware2-actual-audit1/table.md`,
`native-process-batch-pilot1/FINDING.txt` and
`native-forward-layout-independent1/audit.json`. All timings are measured;
ratios and medians are derived from executed records.

A subsequent profile-only A100 lease completed one additional GPU anchor:
measured VMEC6.731627s,12 reported metrics and150 raw numeric fields exactly
match the prior matched CPU baseline. Its profiled attempt failed at an NVTX
marker guard before entering VMEC; it is not a completed equilibrium or timing
sample. All122 retrieved files were independently rehashed, children reaped,
and providerpostflight7 confirmed zero active optimization leases. Evidence:
`build/optimization/a100-profileonly-actual-audit1/`. Lease time charge is only
an estimate, approximatelyUSD0.0387 excluding network/minimum charges.

Timing scope correction: the existing adapter-wall tables retain their original
numeric values. Their outer timer excludes interpreter startup and imports
completed before its bookend, but includes lazy ConStellaration imports inside
`run_official`. Earlier descriptions saying this timer excluded imports were
incorrect. The latest campaign's separately saved inner timer is reported below;
historical ratios must not be relabeled as inner-body measurements.

## Direct original stock CPU versus selected GPU

This completed same-host campaign directly compares original stock CPU33587
with the selected SparseSeed/N3/Thomas8 GPU144e configuration. No historical
ratios are multiplied. It executed 72 evaluations on Threadripper 2950X / RTX
3090, CPU 0, VMEC/OpenMP/BLAS threads 1: 12 excluded audited warmups and five
alternating measured pairs per boundary/track. All intervals below are derived
from measured pairs; no number is an estimate.

The primary timer is the post-import adapter body, first solve in each fresh
process, including settings, first-use costs, metrics and artifact writes within
its saved bookends. It is not isolated `forward_model` or a proven warmed solve.

| Boundary | Track | Original CPU body s | Selected GPU body s | Ratio | Paired 95% interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Third frozen fixture | P1 | 84.375638 | 8.035474 | 10.500393x | 10.464710-10.601876 | 0 |
| Third frozen fixture | P3 | 87.303398 | 10.670843 | 8.181490x | 8.072304-8.823141 | 0 |
| public_p2_seed | P1 | 102.987779 | 9.178441 | 11.220618x | 11.209123-11.356025 | 0 |
| public_p2_seed | P3 | 106.066896 | 11.792965 | 8.994082x | 8.903193-9.131571 | 0 |
| public_p3_seed | P1 | 113.175319 | 10.191955 | 11.104377x | 11.054257-11.152753 | 0 |
| public_p3_seed | P3 | 115.873470 | 12.831189 | 9.030611x | 8.972210-9.115278 | 0 |

The separately measured VMEC-call ratios are **13.736949-14.284143x**; outer
fresh-call ratios, including lazy imports, are **5.561327-7.312858x**. See
[all three measured scope tables](../../build/optimization/native-originalcpu-seed72-timing-scopes/tables.md)
for each case, raw pairs and intervals. The root independently rehashed all72
reports and recomputed all18 medians/ratios: [root audit](../../build/optimization/native-originalcpu-seed72-root-audit1/audit.json).

All69 numeric case/metric entries have absolute and relative drift0; three null
QI entries remain null. All72 original metric/request identities, decks, actual
raw NetCDF numeric fields, captured numerical diagnostic hashes and repeats are
exact. All798 dependency/launch paths were rehashed; processes are reaped and
cleanup actions empty. Raw diagnostic arrays were not separately serialized.
[Per-metric CSV](../../build/optimization/native-originalcpu-seed72-final-report/per-metric-drift.csv),
[full qualification](../../build/optimization/native-originalcpu-seed72-final-report/audit.json),
[strict provenance audit](../../build/optimization/native-originalcpu-seed72-final-report/strict-original-audit.json),
and [sealed primary deliverable](../../build/optimization/native-originalcpu-seed72-qualified/summary.json).

The legacy supervisor purpose text names an older incremental comparison; it
was preserved unchanged. Actual immutable plan, commands, loaded core hashes,
per-run flags and all provenance establish this direct comparison.

## Unselected follow-ups and cross-host identity limit

The timing status below supersedes categorical conclusions from older WSL
screens. An external Windows-QPC experiment found CLOCK_MONOTONIC/perf_counter
under-reporting RAW-bounded intervals by0.53-7.77%, with84-87ns clock brackets.
These new trials cannot reconstruct old pilot RAW values or provide a correction
factor. Preserve their recorded numbers as clock-reported exploratory screens.
Main direct72 tables were measured on the native host, whose clocks independently
passed coherence checks. [External clock evidence](../../build/optimization/clock-bracket1/FINDING.txt).

Packed GPU-controller fusion and its bulk-read guard follow-up each passed16
official exact evaluations and12 capped lifecycle probes. Their separate
eight-evaluation one-case pilots (one excluded pair, three measured pairs) were
recorded larger WSL MONOTONIC times: VMEC6.676305 to7.722356s (+15.67%) for packed,
and6.374105 to6.926504s (+8.67%) for bulk-read. Cross-campaign differences do
not isolate the bulk-read solver benefit. Neither candidate is selected or a
six-case speed qualification. [Packed pilot](../../build/optimization/packed-controller-transfers1/pilot-summary.json),
[bulk-read pilot](../../build/optimization/bulk-read-guard1/pilot-summary.json).

A subsequent corrective NATIVE GateLive-versus-selected screen executed8 fresh
processes, one excluded warmup pair and3 measured pairs. Actual VMEC RAW pairs
were8.022696629/8.432426291,8.006332295/8.428948909,8.115250756/8.412627225s
(selected/candidate). Median paired ratio0.9514102291 means candidate duration
was5.107% longer; MONOTONIC ratio0.9514100492 agrees. All original metric/request
identities, decks, raw NC fields, diagnostics and repeats exact;809 dependencies
unchanged and all processes reaped. This remains one-case internal selection
evidence, not a six-case timing qualification, trajectory proof or isolated
effect of the Gate/Live fusion. [Native finding](../../build/optimization/native-gatelive-clock-screen1/FINDING.txt).

The first A100 pilot executed only its matched-current CPU144e baseline. It stopped
at one-ULP `max_elongation` drift (relative1.6108953933853254e-16), despite
identical actual deck, equilibrium numerical hashes and diagnostic hashes.
No GPU candidate, repeat, calibration or profile ran. All37 retrieved files
verified and lease teardown is verified. This run is excluded from comparable
speed tables. Saved-equilibrium postprocessing replay is required before a
new lease. [Actual pilot finding](../../build/optimization/a100-current-actual-audit1/FINDING.txt).

A separate, subsequently executed A100 PCIe pilot used a mandatory saved-output
CPU compatibility gate followed by counter access checking before any fixture
solve. Four fresh CPU processes (default/Haswell, two repeats each) matched the
original metric and all2082 reference arrays exactly, independently reloaded
and compared. This covers one saved-output metric path, not every official
metric or the cause of the first host's drift. Ncu's query exited0 but reported
ERR_NVGPUCTRPERM; the runner correctly stopped, with no counter kernel, fixture
equilibrium or profile executed. All45 retrieved files verified; runtime and
payload remained unchanged and lease teardown is verified. No performance or
roofline result is inferred. [Second pilot audit](../../build/optimization/a100-compat-actual-audit1/terminal-audit.json).

The subsequent timing-and-trace pilot actually executed one publicP3/P3 matched
CPU/GPU pair on A100 PCIe / EPYC7543P. Measured VMEC calls were92.341533s and
6.402298s; adapter bodies102.100360s and16.050959s. These are single observations,
not qualified speedups or cross-GPU rankings. Both original global identities,
decks, diagnostics and wout gates passed; after retrieval the unchanged raw
comparison also found all150 numeric NC/diagnostic fields exact. No candidate
repeat, remaining case, calibration or trace executed: plain orchestration
Python lacked netCDF4. The campaign is incomplete and excluded from main tables.
All82 retrieved files independently rehashed, teardown verified and independent
provider inventory found0leases. Quote0.592222USD/h times observed297.451431s
gives an estimated time charge USD0.048933, excluding transfers/billing minima.
[Terminal audit](../../build/optimization/a100-hardware-actual-audit1/terminal-audit.json),
[retrieval and output audit](../../build/optimization/a100-hardware-actual-audit1/retrieval-audit.json).

## Latest qualified incremental result: SparseSeed + N3 Forward

This clean native campaign compares qualified Prefetch8 with the combined
SparseSeed and N3 Forward changes. It executed 72 evaluations: 12 excluded,
audited warmups and 60 measured evaluations, five alternating measured pairs
for each of three boundaries on P1/P3. Hardware was Threadripper 2950X /
RTX 3090, CPU 0, VMEC max_threads=1, OpenMP/BLAS 1. Adapter times below are
measured outer fresh-call medians, including lazy ConStellaration imports.
Ratios and paired
95% bootstrap intervals are derived from executed measurements; none is an
estimate. All six intervals exceed no change.

| Boundary | Track | Prefetch8 s | Combined s | Ratio | Paired 95% interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Third frozen fixture | P1 | 14.621159 | 13.987494 | 1.045302x | 1.039207-1.049535 | 0 |
| Third frozen fixture | P3 | 17.303394 | 16.600489 | 1.042342x | 1.042342-1.045606 | 0 |
| public_p2_seed | P1 | 15.949480 | 15.163486 | 1.051835x | 1.042803-1.060813 | 0 |
| public_p2_seed | P3 | 18.462027 | 17.717576 | 1.042018x | 1.039149-1.049943 | 0 |
| public_p3_seed | P1 | 16.955526 | 16.139856 | 1.050538x | 1.048094-1.053057 | 0 |
| public_p3_seed | P3 | 19.618431 | 18.818449 | 1.042511x | 1.038595-1.050817 | 0 |

The same 72 executed reports also contain a separate **post-import adapter body
(first solve in a fresh process)** timer. This begins after lazy imports and
directory creation, and ends before equilibrium metadata/digests are assembled.
It includes boundary/settings work, first-use costs, metrics and artifact writes
within those bookends; it is neither isolated `forward_model` nor a proven warmed
steady-state solve. The following measured medians and derived intervals exclude
the same 12 audited warmups and use the same five paired samples per case.

| Boundary | Track | Prefetch8 body s | Combined body s | Ratio | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Third frozen fixture | P1 | 8.650452 | 8.018631 | 1.078794x | 1.075983-1.083046 |
| Third frozen fixture | P3 | 11.295273 | 10.589076 | 1.066691x | 1.059407-1.074595 |
| public_p2_seed | P1 | 9.903744 | 9.127236 | 1.085076x | 1.077276-1.091980 |
| public_p2_seed | P3 | 12.450148 | 11.740408 | 1.060453x | 1.059094-1.065376 |
| public_p3_seed | P1 | 10.997146 | 10.145609 | 1.083932x | 1.079590-1.085421 |
| public_p3_seed | P3 | 13.569690 | 12.771447 | 1.062502x | 1.049249-1.069030 |

Instrumented VMEC-call median ratios from these executions are
1.102255-1.110264x. Per-run outer-minus-inner residual medians are
5.968639-6.069112 s across variants/cases; this includes pre/post-inner work and
call/profiler-dispatch overhead and is not a pure import measurement.
[All three scope tables and raw per-run fields](../../build/optimization/native-seed-forward72-timing-scopes/tables.md),
[72-report frozen-source audit](../../build/optimization/native-seed-forward72-timing-scopes/audit.json),
[per-run CSV](../../build/optimization/native-seed-forward72-timing-scopes/per-run-scopes.csv).
The audit reproduces every original outer value and interval exactly. No solver
was rerun, and metric/identity qualification remains unchanged.

This is a **combined incremental GPU-versus-GPU result**. It does not isolate
the two changes individually and must not be multiplied into historical
CPU-to-GPU or intermediate ratios. The third fixture is
`0078c85f...-111246be.json`. Full names, full-precision values, VMEC-only
medians, raw pairs and intervals are in the [qualified report](../../build/optimization/native-seed-forward72-final-report/qualified-pairs.json).

### Unselected fused-update pilot

A separate WSL RTX5090/P3-seed/P3-track pilot compared the qualified SparseSeed/N3
build with the combined fused damping/state update. It executed four profiling
evaluations and eight unprofiled evaluations: one excluded warmup pair and three
alternating measured pairs, CPU20/threads1. All12 original identities, decks,
raw numerical outputs and diagnostic hashes matched exactly. Frozen files stayed
unchanged and owned processes were reaped. This is one case, not the six-case
performance qualification.

| Actual timer scope | Baseline median s | Fused median s |
| --- | ---: | ---: |
| Whole fresh process | 16.473748 | 17.289648 |
| Outer adapter call, including lazy imports | 10.387690 | 11.405476 |
| Post-import adapter body | 9.358434 | 10.371499 |
| VMEC call | 6.311810 | 7.341555 |

The fused candidate is slower in this pilot and is not selected. Actual Nsight
traces also show waits moving from StreamSynchronize into MemcpyAsync; summed
API time is not an additive explanation of the regression. Source inspection
finds a separate pageable4B failure readback at each synchronization. A private
packed/pinned readback change is being implemented while preserving immediate
failure checks; no improvement is claimed before it passes its own gates.
[All raw pilot pairs and scopes](../../build/optimization/fused-profile-screen1/pilot-summary.json),
[clipped trace intervals and API groups](../../build/optimization/fused-profile-screen1/profile-comparison.json).
Profiler-wrapper VMEC wall time includes ProfilerStart/Stop, whereas the inner
NVTX span excludes stop/flushing; these are different timing scopes.

### Main six-case qualification evidence

Comparison tolerance is exactly zero. All 69 numeric case/metric entries have
absolute and relative drift 0; the three null QI entries remain null.
[Per-metric CSV](../../build/optimization/native-seed-forward72-final-report/per-metric-drift.csv). All 72 actual official metric/request
identities were independently recomputed, and every serialized raw NetCDF
numeric field was re-read and checked. Original actual decks, captured
diagnostic hashes and repeats match exactly; raw diagnostic arrays were not
separately serialized. Physics settings and the original CPU convergence/
damping controller are unchanged; live GPU control, pool, GIL shim and queue
are OFF. The nine runtime flags are unchanged from Prefetch8.

All 707 configuration dependencies plus launch inputs (711 files total)
rehashed exactly. Supervisor/coordinator exited 0, known owned PIDs were gone,
groups/adopted children were empty, and no cleanup signals were required.
[Actual-artifact audit](../../build/optimization/native-seed-forward72-final-report/audit.json), [strict original-reference audit](../../build/optimization/native-seed-forward72-final-report/strict-original-audit.json),
[postflight](../../build/optimization/native-seed-forward72-final-report/postflight-final.json), [evidence hashes](../../build/optimization/native-seed-forward72-final-report/evidence-hashes.json).
The audit explicitly notes a legacy supervisor description; actual commands,
wrapper configurations, build labels and library hashes establish this
comparison. Failed prelaunch diagnostics and the separate corrected native
cross-host correctness attempt remain preserved outside the timing table.

SparseSeed changes seed scheduling while retaining the full rounding
certificate and original ordered fallback. N3 splits independent R/Z/lambda
Fourier chains into three launch groups without changing their original
theta/k arithmetic order, twelve partial arrays or six outputs. On one
captured input, seven randomized component rounds measured graph medians
51.488 to 35.103 us; matched counters explain lower register use and better
latency hiding. Those component measurements are not the full-solver ratio.
[Component proof/counters](../../build/optimization/forward-groups1/finding.json).
N6/N12 variants passed exactness/sanitizers but were slower and are not used:
[negative finding](../../build/optimization/forward-scalar-groups1/finding.json).
[Combined local qualification](../../build/optimization/seed-forward-qualification1/final-summary.json),
[native cross-host audit](../../build/optimization/native-seed-forward-crosshost-results2/independent-audit.json).

The whole-solver timeline predates this combined build. Current matched
N3 calibration covers one cache-warm ns99 kernel input and empirical strict
FP64/memory references; it does not establish whole-solver or absolute-peak
roofline efficiency. See [ROOFLINE.md](ROOFLINE.md). The earlier
[stage diagnostic](../../build/optimization/native-prefetch8-qualification-payload/stage-pair-completed.json)
contains single observations and is excluded from paired speed claims.
Reproduce the frozen build with [README.md](README.md#qualified-sparseseed--n3-forward-build).

## Earlier qualified incremental result: Prefetch8

This earlier clean native campaign compares qualified Thomas row1 with Prefetch8.
It executed 72 evaluations: 12 excluded, audited warmups and 60 measured evaluations,
five alternating measured pairs for each of three boundaries on P1/P3. Hardware
was Threadripper 2950X / RTX 3090, CPU 0, VMEC max_threads=1, OpenMP/BLAS 1. The table
contains measured outer fresh-call adapter medians, including lazy imports.
Ratios and
paired-bootstrap 95% intervals are derived from executed measurements. All six
intervals exceed no change; none of these numbers is estimated.

| Boundary | Track | Thomas row1 s | Prefetch8 s | Ratio | Paired 95% interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Third frozen fixture | P1 |15.003101 |14.670667 |1.022660x |1.021244-1.029778 |0 |
| Third frozen fixture | P3 |17.565538 |17.211817 |1.020551x |1.018016-1.025135 |0 |
| public_p2_seed | P1 |16.337608 |15.848799 |1.030842x |1.027036-1.036215 |0 |
| public_p2_seed | P3 |18.920033 |18.410205 |1.027693x |1.023792-1.029643 |0 |
| public_p3_seed | P1 |17.442808 |16.974123 |1.027612x |1.022506-1.031907 |0 |
| public_p3_seed | P3 |20.016418 |19.522376 |1.025306x |1.019791-1.029336 |0 |

The third fixture is `0078c85f...-111246be.json`. Full names, full-precision values,
VMEC-only medians, all raw pairs and confidence intervals are in the
[qualified report](../../build/optimization/native-prefetch8-qualification72-final-report/qualified-pairs.json).
This is incremental GPU-versus-GPU latency. It must not be multiplied into
the earlier 14-to-22, 22-to-Thomas or original-CPU-to-14 comparisons.

Comparison tolerance is exactly zero: all 69 numeric case/metric entries have
absolute and relative drift 0; the three null QI entries remain null. The
[per-metric CSV](../../build/optimization/native-prefetch8-qualification72-final-report/per-metric-drift.csv)
reports every entry. Every official request/metric identity was independently
recomputed. Actual decks, all serialized raw NetCDF numeric fields, captured
diagnostic hashes and repeats match immutable original references. Raw diagnostic
arrays were not separately serialized. All 504 configuration dependencies plus
launch inputs (508 files total) rehashed exactly. Supervisor/coordinator exited 0;
the supervisor was reaped, groups/adopted children were empty, and no cleanup
signals were needed. [Actual-artifact audit](../../build/optimization/native-prefetch8-qualification72-final-report/audit.json),
[postflight proof](../../build/optimization/native-prefetch8-qualification72-final-report/postflight-final.json),
[evidence hashes](../../build/optimization/native-prefetch8-qualification72-final-report/evidence-hashes.json).

Prefetch8 changes only early immutable loads in the certified Thomas component;
CPU control, physics, resolution, convergence tolerances and cadence are unchanged.
The frozen build uses the qualified Thomas backend/header/shim with only helper,
helper-header and builder overrides. Default row1 encoded kernels remain exact.
On one captured input, seven randomized component rounds measured graph medians
59.965 to 41.366 us. Matched Nsight counters measured summed kernel time 65.376 to
42.176 us; Proposal long-scoreboard stalls fell 50.33% to 10.15%, backscatter 64.28%
to 17.96%. Registers increased without spills. These are component observations,
not full-solver speedups or roofline efficiency. Counter sums exclude graph input
copies/memset; event timings include the common reset. The compiler evaluates 440
extra discarded first-row DADD/DMUL operations; selected recurrence bits remain
exact. [Component evidence](../../build/optimization/thomas-prefetch1/finding.json),
[counter scope/data](../../build/optimization/thomas-prefetch1/counter-finding.json),
[local qualification](../../build/optimization/prefetch-qualification1/matrix-independent-audit.json),
[native12 cross-host audit](../../build/optimization/native-prefetch8-crosshost-results/independent-audit.json).

## Earlier qualified incremental result:22 to Thomas row1

The preceding completed native campaign compares frozen22 with Thomas-prefix1:
72 executed evaluations, including12 excluded warmups and60 measured evaluations.
Measured adapter improvements are1.0040-1.0103x across the six cases. All original
metrics, identities, decks, numerical outputs and repeats match exactly.
Two paired95% intervals include1, so those cases do not establish an improvement.
See the [six-case table](../../build/optimization/native-thomas-vs22-audit/table.md),
[per-metric CSV](../../build/optimization/native-thomas-vs22-audit/per-metric-drift.csv)
and [independent audit](../../build/optimization/native-thomas-vs22-audit/audit.json).
These are incremental GPU-versus-GPU measurements. Do not multiply them into
the historical CPU comparison below. Component speedups are not adapter speedups.

The current Thomas whole-solver Nsight diagnostic executed two P3/P3 evaluations
with exact original outputs. Its instrumented inner VMEC interval is12.197872s:
kernel union8.017064s (65.725%), all GPU activity union8.364451s, and3.833421s
without recorded target GPU activity. These intervals measure trace coverage,
not occupancy or roofline efficiency. The trace contains1,170,370 kernel records
and27,782 stream synchronizations. API durations overlap GPU execution.

Actual host-transfer payload durations sum to5.032ms HtoD and43.468ms DtoH;
bulk host-transfer bandwidth is not the dominant recorded cost. Device copies
sum to189.374ms and22.066GB of logical payload. Synchronization latency remains
separate from payload-transfer duration. SeedSegments, ForwardPartials and
FourierToReal are the largest summed kernel times at796.319,731.446 and611.642ms.
See the [diagnostic finding](../../build/optimization/native-thomas-full-profile1/FINDINGS.txt)
and [independent interval review](../../build/optimization/native-thomas-full-profile1/root-independent-review.json).
Profile timings are excluded from all latency tables. GPU-context scopes by
radial resolution exclude CPU initialization and are not full stage timings.

A separate P3/P3 stage diagnostic used the existing source timers around actual
radial initialization and equilibrium solves. Both variants matched original
outputs exactly. These are single measured observations, excluded from the
72-evaluation paired campaign; they do not establish a speedup.

| ns | Thomas row1 initialize s | Thomas row1 solve s | Prefetch8 initialize s | Prefetch8 solve s |
| --- | ---: | ---: | ---: | ---: |
|25 |0.000728 |0.999519 |0.000738 |0.978726 |
|51 |0.004342 |1.051108 |0.004724 |1.015704 |
|99 |0.007570 |7.047129 |0.007457 |6.665708 |

The [actual stage records](../../build/optimization/native-prefetch8-qualification-payload/stage-pair-completed.json)
include iteration counts, retries and residuals; the retained logs contain all
six original timer lines. Physics settings and convergence cadence are unchanged.

Frozen22 adds a separately measured incremental improvement over frozen14:

| Boundary | Track | GPU14 adapter s | GPU22 adapter s | Speedup | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Third frozen fixture | P1 |18.340142 |15.188549 |1.207498x |0 |
| Third frozen fixture | P3 |20.844171 |17.796694 |1.171238x |0 |
| public_p2_seed | P1 |18.558466 |16.512264 |1.123920x |0 |
| public_p2_seed | P3 |21.131503 |19.126763 |1.104813x |0 |
| public_p3_seed | P1 |20.844796 |17.646882 |1.181217x |0 |
| public_p3_seed | P3 |23.582442 |20.287119 |1.162434x |0 |

All times are measured native-host adapter medians, five alternating pairs per
case after one excluded warmup pair:72 actual evaluations,12 warmups/60 measured.
Speedups are derived ratios of measured medians. VMEC-only medians improve from
10.19-12.72s to7.13-9.50s; exact per-case values and paired intervals are in the
[incremental data](../../build/optimization/native22-vs14-posthoc-report/qualified-incremental-pairs.json).
This compares14 with22, not originalCPU with22; no compounded speedup claim.

The original supervisor exited1 after all benchmark children exited0 because its
reporter import lacked the package root. Its failed status/logs remain intact.
Independent local recovery qualified the completed measurement data: actual
rawNetCDF, recomputed official metric/request identities, original global
reports/decks/diagnostics, repeats, runtime flags and331 frozen dependencies all
exact. Coordinator exit0 is inferred from the executed success branch; no direct
saved exit receipt is invented. All owned processes are gone, no cleanup signals.
[Post-hoc data audit](../../build/optimization/native22-vs14-posthoc-report/posthoc-data-qualification.json),
[per-metric drift](../../build/optimization/native22-vs14-posthoc-report/per-metric-drift.csv).
The original CPU-vs14 main table remains unchanged.

The dedicated native solve queue is now independently qualified for throughput:
**1.161676x median paired gain** over four inline workers, bootstrap95% interval
1.003070-1.300070x. Five measured six-case pairs, after one excluded warmup pair:

| Pair | Inline service s | Solve queue s | Throughput ratio |
| --- | ---: | ---: | ---: |
| 1 |43.741056 |37.253421 |1.174149x |
| 2 |43.368215 |37.332444 |1.161676x |
| 3 |37.778419 |37.662812 |1.003070x |
| 4 |39.985384 |36.374262 |1.099277x |
| 5 |46.566863 |35.818746 |1.300070x |

Every time is measured on the native RTX3090 host. All five pairs favor the
queue, with broad variation. Both service processes have mask0-9; inline has
four dedicated adapter/solve workers0-3, queue has four native workers0-3 plus
six adapters4-9. Auxiliary threads may inherit the common ten-core mask. This
compares queueing plus dedicated worker allocation, not an algorithm-only gain.
Existing serial setup/postprocessing locks remain. Each VMEC solve stays single
threaded. All84 actual evaluations (12 cross-host gate,72 paired) pass original
metrics/request identities, decks, raw numerical output, diagnostics, repeats
and TLS ownership checks with zero drift. All processes reaped normally without
cleanup signals. Eight archives/841 covered files verified independently.
[Raw pair table](../../build/optimization/native-queue18-results/batch-throughput-table.csv),
[independent84 audit](../../build/optimization/native-queue18-results/independent84-artifact-audit.json).
This is throughput, not a single-evaluation speedup; do not multiply gains from
separate campaigns.

Separate native resident18 batch qualification is complete:72 actual evaluations,
one excluded warmup plus five alternating measured six-case workload pairs.
Four workers versus one yield1.547080x median paired throughput, bootstrap95%
interval1.543088-1.664903. Median makespans71.939753/46.570021s. All original
metrics, decks, raw numerical output and repeat gates match exactly. This is
throughput, not single-evaluation latency or a multiplier for the main table.
Original reporting failed after all measurements and cleanup; independent raw
artifact audit and corrected statistics are preserved in
[recovered-qualified-summary.json](../../build/optimization/native-batch18-results/recovered-qualified-summary.json).

The separate warmed four-worker full-round diagnostic preserves12 original
evaluations exactly. Its measured six-job NVTX window is45.683480s; recorded
GPU graph-inflight union28.736991s, no-graph16.946489s. Post-native phases
overlap15.549429s of no-graph time, including a10.080914s largest gap and
5.691676s final tail. Phases overlap across workers and are not additive.
Graph spans are not kernel busy time, occupancy or roofline utilization.
[Full-round analysis](../../build/optimization/validation-agent/native-round-profile3/round-analysis.json)
motivates a separate solve queue before further controller micro-optimizations.
Actual worker exit0 and all owned-PID absence were independently verified;
the supervisor's teardown PermissionError is preserved, not reported as success.

The latest qualified frozen resident14 achieves **4.4302-5.8312x measured adapter
speedup**, with zero metric drift across all six fixed boundary/track cases.
The72-evaluation native original-vs14 campaign is terminal, retrieved and audited:
12 excluded warmups,60 measured evaluations, five alternating pairs per case.
Hardware is Threadripper2950X/RTX3090, CPU0, VMEC max_threads=1 and OMP/BLAS1.
No profiler, trace, GIL-release shim or changed physics knob ran in this campaign.

| Frozen boundary | Track | Original CPU median s | Resident14 median s | Speedup | Max relative drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Public record1087ab3b174e | P1 |90.667903 |18.379146 |4.9332x |0 |
| Public record1087ab3b174e | P3 |93.072719 |21.008627 |4.4302x |0 |
| public_p2_seed | P1 |109.224275 |18.730863 |5.8312x |0 |
| public_p2_seed | P3 |111.968552 |21.239253 |5.2718x |0 |
| public_p3_seed | P1 |120.467203 |21.036700 |5.7265x |0 |
| public_p3_seed | P3 |121.868646 |23.608339 |5.1621x |0 |

Every time is measured; ratios/paired-bootstrap95% intervals are derived from
executed samples. All69 numeric metric cells have absolute/relative drift0;
three null metrics stay null. Original reference hash33587eac..., baseline no
GPU flags/preload, candidate only frozen14. Actual decks, numerical wout fields,
captured diagnostic hashes, warmup/measured repeats and all runtime provenance
pass. Postflight225 dependency hashes and18 original-reference files match.
Cleanup verifies the supervisor reaped, coordinator group empty and guard complete
with no cleanup actions. Archive SHA256:
`6562ea780a219e251ea8d7eb3e97bd375fc9be08d2f283aecfc9c9a545308a53`.

Authoritative files are under `solver14-qualification72-final-report/`:
`qualified-pairs.json`, `per-metric-drift.csv`, `audit.json`, `terminal-summary.json`
and `evidence-hashes.json`. Raw verified checkpoint:
`solver14-qualification72-checkpoints/c072-complete-1793774013492006350/`.
Resident14 combines factor caching, cooperative residual max32 and ordinary-energy
max32 with refresh diagnostics16. Later lattice/capture-coordinated/prefetch/pool
experiments are excluded. The goal remains active; near-roofline performance is
unproven. [SUMMARY.md](SUMMARY.md) is the current one-page result.

## Earlier qualified resident8 campaign

The persistent GPU solver achieves 3.401-4.149x measured adapter speedup across
three boundaries and two tracks, with zero metric drift, byte-identical actual
decks and exact numerical equilibrium and additional diagnostic hashes. The
original-baseline campaign completed72 evaluations: one excluded warmup pair
plus five measured alternating pairs per case. All six cases pass the hardened
reporter's original-reference qualification gates. This is a qualified
improvement; near-roofline full-solver performance remains unproven and the goal
remains active. The previous mixed CPU/GPU table at1.539-1.584x is historical.

The native host is anis-ubuntu-desktop, AMD Threadripper2950X and RTX3090.
Both variants use CPU0 and VMEC max_threads=1, with OpenMP/BLAS thread counts1.
Timings below are measured adapter-wall medians in seconds; speedups are derived
ratios of those medians. Each row contains five measured alternating pairs.
Adapter timing excludes interpreter startup but includes lazy imports performed
inside the timed call. No tracing or profiler ran in this campaign.

| Frozen boundary | Track | Baseline s | GPU s | Speedup | Maximum relative metric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Public record1087ab3b174e | P1 |90.5469 |24.9043 |3.6358x |0 |
| Public record1087ab3b174e | P3 |93.8753 |27.6052 |3.4006x |0 |
| public_p2_seed | P1 |109.7666 |26.4563 |4.1490x |0 |
| public_p2_seed | P3 |111.5407 |29.0882 |3.8346x |0 |
| public_p3_seed | P1 |120.0891 |29.4683 |4.0752x |0 |
| public_p3_seed | P3 |122.1861 |32.1344 |3.8023x |0 |

Per-metric absolute/relative differences, raw pairs, paired-bootstrap intervals
and all qualification gates are recorded in
`build/optimization/resident8-native72-summary.json`. Each numeric metric has
maximum absolute and relative drift0; null P1 QI values stay null. Every repeat,
including warmups, preserves its request/metrics and numerical identities.
The terminal733-file archive was SHA256 verified:
`68fdcf8ddc222b28316742af5053d950778cfd99325230a137d216ff21876060`.
Full evidence is under
`secondary-campaign-checkpoints/c072-complete-1793759209591650595/`.
The independent actual-artifact audit passes all72 records and all six cases:
`validation-agent/native72-final-independent-audit.json`. It verifies original
references, actual decks, output hashes, repeat determinism, frozen library
recipes and CPU/thread placement independently of the timing reporter.

Canonical `resident-solver8-capture` keeps geometry, velocity and restart state
on the GPU, packs device copies, uses pinned scalar packets, and captures larger
iteration graphs. Original CPU scalar convergence, damping and multigrid control
remain. The six-case audit is `validation-agent/solver8-correctness-summary.json`;
extra public P1/P1 default mpol=ntor12 and native secondary-host repeat checks
also pass. Near-roofline performance remains unproven.

A separate local one-case residual12-versus24 selection executed eight runs,
including one excluded warmup pair and three measured alternating pairs.
Adapter medians are 23.5725 and 21.6798 seconds, a derived 1.0873x ratio.
Every run matches the global reference exactly. This is internal selection
evidence, not a qualified six-case speedup; the qualified candidate stays at12.
See `internal-round-budget-selection/`. Round24 now passes all twelve six-case
correctness executions and repeat trajectories; that is not timing qualification.

Newer cooperative residual kernels pass full local/native trajectories. A native
one-case selection executed eight unprofiled evaluations, with three measured
pairs: adapter medians31.9938/25.7450s (derived1.24272x), VMEC medians21.2369/
15.1148s (1.40504x). This is preliminary, not a new main-table result. Combined
factor caching, cooperative residuals and ordinary cooperative energies in
solver14 pass two native smokes plus twelve matrix executions with zero drift.
The original-versus14 campaign is now complete and qualified above; these
one-case selection timings remain separate from the main table.
Evidence: `secondary-cooperative11-selection-results/summary.json` and
`secondary-cooperative14-results/results/summary.json`.

The positive-prefix integer-lattice residual adapter passes280 measured component
samples and six focused sanitizer jobs. On one unique captured ns99 state, graph
medians are42.46/39.80us before/after preconditioning, versus96.33/61.38us for
the cooperative kernel. These are component timings, not whole-equilibrium
speedups. Every proposal is checked against the complete original rounded
recurrence; rejected components use the original ordered residual fallback.
Opt-in solver17 now passes the35-frame captured CPU control matrix and two
complete P3/P3 solves. All metrics, decks, numerical outputs and repeat hashes
are exact; both original CPU trajectories match13,780 events with13,761 GPU
updates. Initial undefined bad-Jacobian residual fields are excluded explicitly,
as in earlier trajectory checks. These traced shared-GPU runs do not qualify
timing or broader fixture coverage. Evidence:
`resident-solver17-lattice-controls/smoke/validation.json`.
See `lattice-residual1/timing-analysis.json` and its `local-gates/` records.

Near-roofline performance is still distant: scoped cooperative-kernel replays
report70.29-73.79% barrier stalls,6.74-9.06% peak L2 throughput and5.75-9.38%
peak DRAM throughput. These captured-state counters are not whole-solver
percentages. The lattice path targets repeated corrections and global barriers;
thread batching separately targets unused hardware capacity.
See `cooperative-residual-roofline/counter-values.json`.

Solver17 now also passes all twelve six-case evaluations, including independent
actual-deck/NetCDF/diagnostic audits and six exact repeat trajectories. Its two
observed fine-grid workspace snapshots each record19542 residual calls,58626
accepted components and zero fallbacks. Coordinated18 passes one single-worker
and four repeated two-worker requests exactly. These are correctness results,
not throughput qualification. See `validation-agent/solver17-matrix-independent-audit.json`,
`lattice-observer1/observed17-workspace-audit.json` and
`validation-agent/batch-driver/solver18-batch-independent-audit.json`.

A new steady17 Nsight window starts after1000 consecutive successful fine-grid
updates. All profiled outputs/decks/numerical hashes remain exact. Its164788
kernels sum1.319079s: Thomas RHS23.31%, ordinary energies14.49%, ordered profiles
9.17%. Recorded target activity covers45.20% of its2.990s activity span; this
is not unprofiled GPU utilization. D2H6.105MB costs3.074ms and D2D5.270GB costs
18.717ms in this window. Dependency-heavy kernels and scheduling now outrank
transfer bandwidth as measured priorities. Evidence:
`resident-solver17-lattice-controls/profile-steady/timeline-analysis.json` and
`validation-agent/solver17-steady-profile-independent-audit.json`.
The Nsight CLI returned14.306s before the application finished; acceptance uses
its independently verified terminal/gone worker, not the CLI-only status.

Separate batching diagnostics executed16 actual evaluations across1/2/4workers.
All original metrics, request/deck identities, raw NetCDF numerical fields,
diagnostics and repeated outputs match exactly. Node timestamps prove concurrent
kernels on distinct streams, beyond host-call or graph-span overlap:

| Workers | Actual capture s | Recorded kernel coverage | Time with cross-stream kernels |
| --- | ---: | ---: | ---: |
| 1 |15.000154 |38.121% |0% |
| 2 |15.000388 |52.223% |31.458% |
| 4 |14.088606 |63.743% |48.079% |

These are instrumented target-process statistics under shared Windows GPU load,
not whole-GPU utilization or clean evaluation-throughput measurements. Four
boundary kernels in the last capture are clipped to its actual14.088606s window.
The longest recorded post-startup all-activity gap decreases17.789/4.779/4.660ms.
Several large gaps coincide with graph teardown and memory frees; association
alone does not prove a cause. Original requests continue to completion after
capture; actual worker cleanup is verified independently of the early CLI exit.
Evidence: `validation-agent/batch-diagnostic/root-scheduling-comparison.json`,
`batch-diagnostic-aggregate-audit.json` and per-stage `root-overlap-analysis.json`.
A separate unprofiled native1vs4 campaign is being staged for clean throughput.

The A100 pilot completed all six original/candidate correctness pairs exactly,
and all264 retrieved files pass checksum verification. The instance was destroyed
after collection, with teardown verified. Measured lease lifetime is1178.52s;
quote-derived compute estimate isUSD0.2201, excluding unverified invoice fees.
No external volume was created. These traced pilot runs do not qualify a new
timing table. Evidence: `a100-pilot8-1793759093/controller/` and
`validation-agent/a100-pilot-final-profile-and-tools-audit.json`.

Measured A100 calibration medians are2.61686TF/s strict multiply/add,
5.06314TF/s explicit FMA reference, and1345.71GB/s logical streaming. The
dependent-add probe measures27.00084SMcycles/add including scheduling, loop,
load and predicate overhead. These scoped empirical calibration results do not
establish achieved solver throughput, DRAM transactions or pure instruction
latency. Native GPU counters were denied with ERR_NVGPUCTRPERM; no A100
counter-derived roofline percentage is reported.

Its real ns99 CUDA node timeline was recovered offline from the raw qdstrm
using the exact matching importer. In the instrumented2.966s activity span,
scans/init/error/correction kernels account for46.45% of summed kernel duration,
Thomas factor/RHS18.46%, and residual9.71%. This motivates fewer scan launches,
factor reuse, and more parallel work per launch. These are summed kernel shares,
not unprofiled process percentages. Full analysis and raw/imported hashes are
under `validation-agent/a100-offline-import/`.

Factor caching passes native replay controls and two complete R12 P3/P3 solves
with exact outputs, decks and original CPU trajectories. A separate eight-run
local selection has three measured pairs: adapter medians37.3084/35.3493s,
derived ratio1.05542; VMEC medians26.5119/24.2208s. All global identities and
repeat gates pass. This is preliminary one-case evidence with appreciable timing
variation, not an addition to the qualified main table. Evidence:
`secondary-factor10-results/` and `factor-cache-internal-selection/`.

Scan-input fusion passes48 component processes and focused memory/synchronization
checks, but is9-11% slower with graphs on the measured RTX5090 captured state.
Local counters show80 registers/thread in both versions; scan duration rises
7.392 to9.376us and L2 traffic2.050176 to2.744160MB in the selected replayed
kernel. This remains an isolated experiment. Likewise, certified reciprocal
division passes347568 bit comparisons and296195 exact-rational certificate
checks, but is3.524x slower on the tested serial chains. Neither is promoted.
See `residual-correction-input-fusion/` and `certified-division-study/`.

At 2026-11-03 16:00 PST, the new persistent GPU solver completed its first full
public P3/P3 evaluation with zero metric drift and exact request/metrics, all
numerical wout fields, and additional diagnostic hashes. Its trace records
13,761 GPU updates. Both independent executions now pass these checks. The
complete original CPU trajectory also matches both GPU trajectories across
13,780 events; only undefined residual fields on the initial bad-Jacobian exit
are excluded explicitly. This is a
correctness milestone, not a qualified speedup result: tracing and overlapping
validation workloads invalidate a performance comparison. The original CPU
controller and geometry/velocity mirrors remain in this intermediate rewrite.
Evidence: `build/optimization/resident-solver1/smoke/validation.json`.
The connected replay also passes byte comparisons for fresh preconditioners,
diagnostics, residuals, and stepping; memcheck, racecheck, and synccheck each
report zero errors. Bounded replay evidence is under `resident-solver1/sanitizers`.

At 2026-11-03 16:14 PST, `resident-solver4` also passes two complete independent
public P3/P3 evaluations and the entire defined CPU trajectory in each run.
This version keeps ordinary geometry/velocity/restart backups on the GPU and
uses certified graph-based diagnostic16/residual12 accumulations. Original CPU
scalar control, rare setup/reporting/terminal downloads, and physics remain.
Each run executes 13,761 GPU updates, including genuine bad-Jacobian recovery.
Measured diagnostic adapter times are 82.8634 and 80.1327 seconds; VMEC regions
are 71.2844 and 68.6269 seconds. These traced local correctness runs are not a
paired speedup campaign and are excluded from the qualified table. Its six-case
matrix is now complete: all twelve executions across the three frozen boundaries
and P1/P3 pass exact output, byte-identical INDATA, and repeat gates. Maximum
absolute and relative drift is zero for every numeric metric. The independent
audit is `validation-agent/solver4-correctness-summary.json`. These remain
diagnostic correctness timings; no new paired speedup is claimed. Evidence:
`build/optimization/resident-solver4/smoke/validation.json`.

A real solver4 Nsight Systems node-level capture covers a delayed ten-second
window of the public P3/P3 solve; the complete profiled evaluation still matches
all outputs and its deck exactly. Recorded target GPU activity covers 32.97%
of the 9.964 s first-to-last activity span. This is instrumented trace coverage,
not SM utilization or unprofiled hardware idleness. Residual kernels account
for 42.54% of summed kernel duration, CUB scans 18.05%, Thomas RHS 7.17%, and
Thomas factorization 3.73%. D2H traffic is 5.53 MB across 10,093 records, dominated
by periodic reporting arrays; D2D traffic is 4.83 GB across 67,729 records.
CUDA API sums identify copy and launch overhead, but node tracing adds overhead
and overlapping API durations cannot be treated as whole-process percentages.
Evidence: `resident-solver4/profile/{command.json,parity.json,analysis.json}`.
These results motivate contiguous device storage/copies and larger iteration
graphs. They do not establish near-roofline performance.

The nonconvergent terminal integration also passes twelve executed lifecycle
probes with copied decks capped at one, two or three iterations per grid. Cap1
produces the same expected bad-Jacobian exception in both builds; cap2/cap3
check exact outputs, repeats and traces, including GPU bad-Jacobian recovery.
These changed-deck tests are noncomparable correctness probes, not speedups.
Evidence: `validation-agent/solver2-lifecycle-retry/lifecycle-validation.json`.

A separate captured-frame diagnostic benchmark measures ordinary energy sums
at 3799.97 us ordered versus 171.32 us with sixteen certified correction rounds
in a CUDA Graph; refresh sums measure 3812.74 versus 190.45 us. Three randomized
process-order repetitions each time 100 calls after warmup. All configurations
pass byte parity; sixteen-round graphs have zero fallbacks on this captured
state. Existing user GPU load remained, so these are scoped local component
measurements, not whole-solver speedups or roofline evidence. Memcheck,
synccheck and bounded racecheck pass with zero errors. A preceding full-timing
racecheck timeout is recorded and excluded. Evidence:
`build/optimization/optimistic-diagnostics-compile/`.

## Historical mixed CPU/GPU summary

The authoritative baseline is the installed VMEC++ 0.4.11 oracle extension,
Python 3.12, constellaration 0.2.6, and booz-xform 0.0.9. Three frozen public
boundaries were evaluated on P1 and P3 through the unchanged official CGA adapter.
The third fixture has normalized boundary hash beginning `1087ab3b174e` and
comes from an existing successful public-boundary record. Public P1 was rejected
as a fixture because its P3 QI calculation fails; discovery retains the failure.

The candidate in the latest qualified table combines the separable inverse transform, fused GPU
MHD/constraint/forward region, and four persistent CPU workers for independent
outputs. It uses strict FP64 arithmetic, original per-output accumulation order,
and disabled FMA contraction. VMEC itself retains max_threads=1. Baseline and
candidate run serially on the same host, alternating order. Baseline affinity
is CPU 0; the candidate uses CPUs 0-3. Each of six cases has one excluded warmup
pair and five measured pairs, for 72 executed evaluations. The original inverse
GPU and CPU-locality campaigns remain below as separate results.

Acceptance tolerance is **zero**. Every numeric metric, request-plus-metric
hash, and captured numerical wout field matches exactly. Each build reproduces
those hashes across repeats. Null P1 QI remains null. This establishes parity
for the executed fixtures and hosts, not arbitrary inputs. No physics setting,
multigrid schedule, tolerance, iteration limit, nstep, or boundary was changed.
The original GPU table keeps the CPU solver single threaded. The subsequent CPU
table uses four workers for independent output calculations while retaining each
output's original reduction order and VMEC's max_threads=1. Original data and submission
artifacts are untouched; outputs are isolated in `build/optimization`.

All six original-baseline cases also match between the local WSL workstation
and A100 host, including request/metric identities and every captured numerical
field. The comparison is saved in `build/optimization/crosshost-baseline-parity.json`.

Profiling identifies transforms and MHD forces as major costs, with 87.5% of
stage solve time in the final radial grid in one instrumented solve. The original
inverse-only GPU prototype returns about 54 GB to the CPU per traced solve.
The latest completed mixed configuration qualifies separable transforms,
the fused force region, and deterministic CPU parallelism together. Broader
iteration residency remains separate from that benchmark.

All task leases have been destroyed with teardown verified. The latest A100
PCIe lease cost approximately USD 2.18 by rate times recorded duration,
**estimated**, excluding storage/transfer and fees. Earlier leases add about
USD 2.11 estimated compute, for approximately USD 4.29 total. Invoice charges
remain unverified against the USD 50 cap. The final retrieval verified SHA256
for 850 files before destruction. Credentials were not exported.

## Latest completed mixed CPU/GPU matrix

These are measured adapter medians on the same EPYC 7662/A100 PCIe host.
The CPU baseline and candidate each have five measured repeats per case;
warmups are excluded. Intervals are paired-bootstrap 95% intervals and reflect
only this small fixed campaign. Additional jxbout, mercier, and
threed1_volumetrics numerical hashes are captured and exact, as are every
reported metric, request/metric identity, numerical wout field, and repeat.

| Boundary | Track | Baseline s | Candidate s | Speedup | 95% interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Public P3 | P1 | 138.468 | 87.439 | 1.5836 | 1.5672-1.5960 | 0 |
| Public P3 | P3 | 141.525 | 90.430 | 1.5650 | 1.5569-1.5677 | 0 |
| Public P2 | P1 | 126.632 | 80.175 | 1.5794 | 1.5429-1.5967 | 0 |
| Public P2 | P3 | 129.241 | 83.072 | 1.5558 | 1.5489-1.5673 | 0 |
| Third frozen fixture | P1 | 104.473 | 67.191 | 1.5549 | 1.5503-1.5675 | 0 |
| Third frozen fixture | P3 | 107.440 | 69.797 | 1.5393 | 1.5321-1.5493 | 0 |

Unrounded values, per-metric maximum absolute/relative drift, repeat gates,
and raw pairs are in `build/optimization/paired-combined-summary.json` and
`a100-fused-results/paired-combined/samples.json`. Every numeric metric listed
in the per-metric table below also has maximum absolute and relative drift zero
in this campaign. P1 QI is null on both builds. All 36 A100 baseline executions
also match the workstation baseline request/metric and numerical wout hashes;
see `crosshost-combined-baseline-parity.json`. Extra diagnostic cross-host
coverage is limited by the older workstation campaign's captured sections.

The frozen configuration enables VMECPP_GPU_FOURIER=1,
VMECPP_GPU_FORCE_PIPELINE=1, and VMECPP_DETERMINISTIC_FOURIER_THREADS=4.
OMP, OpenBLAS, and MKL thread limits stay at one. Upload mode is the default
packed path. The later parallel packing, CPU preconditioner, CUDA Graph, and
resident-iteration experiments are absent. The frozen preload SHA256 values are
`d3f2dd37fdf18d0756e949266b182814dd698fdd85d8830e0f6697631f90cbfb`
(forward shim) and
`c25bbc45b2c596191d6df4b23e79351c7855a00501d4a32500d3fc963b893d5a`
(separable inverse). Per-run provenance also records extension and harness
hashes. This table does not multiply independent campaign speedup ratios.

## Completed deterministic CPU matrix

These are measured adapter medians from five alternating baseline/candidate
pairs per case on the local WSL workstation, with an excluded warmup pair.
The candidate combines persistent workers for independent Fourier outputs with
independent MHD, geometry, and force assembly work. It does not use GPU kernels.
Acceptance is zero drift for metrics, request/metric identity, and captured
numerical wout fields; all six cases and repeats pass. All reported metric
drifts in the per-metric table below are also zero for this matrix.

| Boundary | Track | Baseline s | CPU s | Speedup | 95% paired interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Public P3 | P1 | 72.988 | 49.481 | 1.4751 | 1.224-1.496 | 0 |
| Public P3 | P3 | 54.373 | 50.064 | 1.0861 | 1.054-1.546 | 0 |
| Public P2 | P1 | 46.578 | 38.147 | 1.2210 | 1.221-1.495 | 0 |
| Public P2 | P3 | 48.552 | 40.182 | 1.2083 | 1.208-1.481 | 0 |
| Third frozen fixture | P1 | 38.685 | 31.630 | 1.2230 | 1.223-1.472 | 0 |
| Third frozen fixture | P3 | 39.818 | 33.326 | 1.1948 | 1.193-1.490 | 0 |

Raw pairs, hashes, and per-metric drift are in
`build/optimization/paired-pointwise-summary.json` and `paired-pointwise/`.
The broad intervals and markedly different timing regimes require further
attribution; do not use the fastest case as a general speedup claim. These are
end-to-end observations, not proof of scaling efficiency or bandwidth saturation.

## Completed GPU matrix

Times are **measured medians**, in adapter wall seconds, including metric
calculation. Extra numerical-output capture occurs after the timed adapter.
Both builds ran on the same AMD EPYC 7532/A100 SXM4 host; these are not speedups
against the faster local workstation CPU. Intervals are paired-bootstrap 95%
intervals from five measured pairs, with precision limited by sample size.

| Boundary | Track | Baseline s | GPU s | Speedup | 95% interval | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Public P3 | P1 | 131.030 | 109.208 | 1.1998 | 1.1983-1.2011 | 0 |
| Public P3 | P3 | 134.060 | 112.144 | 1.1954 | 1.1938-1.2019 | 0 |
| Public P2 | P1 | 119.361 | 99.817 | 1.1958 | 1.1853-1.1973 | 0 |
| Public P2 | P3 | 122.422 | 102.729 | 1.1917 | 1.1790-1.2017 | 0 |
| Third frozen fixture | P1 | 98.972 | 83.021 | 1.1921 | 1.1826-1.1953 | 0 |
| Third frozen fixture | P3 | 101.829 | 85.883 | 1.1857 | 1.1813-1.1876 | 0 |

Unrounded timings and numerical hashes are authoritative in
`build/optimization/a100-results/paired-gpu/samples.json` and
`build/optimization/paired-a100-summary.json`.

| Metric | Max absolute drift | Max relative drift |
| --- | ---: | ---: |
| aspect_ratio | 0 | 0 |
| aspect_ratio_over_edge_rotational_transform | 0 | 0 |
| average_triangularity | 0 | 0 |
| axis_magnetic_mirror_ratio | 0 | 0 |
| axis_rotational_transform_over_n_field_periods | 0 | 0 |
| edge_magnetic_mirror_ratio | 0 | 0 |
| edge_rotational_transform_over_n_field_periods | 0 | 0 |
| flux_compression_in_regions_of_bad_curvature | 0 | 0 |
| max_elongation | 0 | 0 |
| minimum_normalized_magnetic_gradient_scale_length | 0 | 0 |
| qi (P3; null on P1) | 0 | 0 |
| vacuum_well | 0 | 0 |

Zero baseline metrics require exact equality for zero relative drift. A nonzero
change is not hidden by a denominator floor. Null values are checked for equality
and are not assigned a numeric relative error.

## Profile evidence

A 20-second native py-spy sample collected 1,999 samples without errors.
Exclusive leaves: inverse transform 335 (16.76%), MHD forces 312 (15.61%),
forward transform 293 (14.66%), and force assembly 102 (5.10%). Sampling shares
are not exact stage wall times. One instrumented unmodified-source solve
measured 2.3993 s at ns=25, 5.3432 s at ns=51, and 54.4186 s at ns=99, reaching
iter2 9773 with four retries. These are measured observations of one solve.

The local full-solver Nsight Systems trace contains 13,780 inverse kernels:
3.36475 s device kernel time, 54.197 GB D2H in 4.36998 s, and 6.35569 GB H2D in
0.56012 s. Synchronization overlaps device work and must not be added to those
durations. Profiler wall times are excluded from the benchmark table.

For a captured real final-grid transform, RTX 5090 Nsight Compute measured
412.42 us for the initial kernel and 205.25 us for the dense 32-thread launch,
with exact output bytes. This is a device-kernel improvement, not a twofold
solver speedup. The dense kernel uses 70.69% of FP64 pipe capacity but about 35%
of advertised fused-operation peak. FMA is disabled for parity. Register-output
and tested unroll variants gave no meaningful further benefit. Captured-data
memcheck reported zero errors for the dense variant.

Unprofiled A100 calibration measured approximately 1388 GB/s stream bandwidth
and 7.316 TFLOP/s FP64 FMA; these are not solver throughput. Hardware-counter
collection on that host failed with ERR_NVGPUCTRPERM. Local counters work.
No full-pipeline near-roofline result is established.

## CPU findings, validation, and remaining work

The CPU locality patch retains each output's accumulation order. All six cases
passed ten measured pairs, with exact metrics, numerical fields, and repeat
hashes. Every timing interval includes 1.0. Detailed timings are in
`build/optimization/paired-locality-summary.json`.

The locality/stage-timer source passed the baseline Bazel build and all 14 C++
test targets. Three drift-helper unit tests passed. GPU validation covers the
executed integration runs and captured-data replay, not every upstream test.

An initial AVX2 build exposed Eigen allocator incompatibility across translation
units. A consistent wide-vector build changed wdot history despite matching
metrics. Neither is promoted. A later ABI-compatible AVX2 configuration matched
all numerical fields in one smoke run; performance remains unqualified.

Current experiments: separable GPU sums, ordinary two/four-thread repeat probes,
and parallel independent Fourier transforms with serial solver reductions.
These must pass their own full-run gates. Larger device-resident geometry,
force, and forward-transform regions are needed to attack the transfer boundary.
FFT, tensor-core, fast-math, relaxed-fidelity, and warm-restart configurations
have not been qualified as comparable alternatives.

## Subsequent component and parity evidence

The separable GPU transform completed two P3 evaluations with exact metrics and
all numerical wout fields. Local timings were variable and remain unqualified.
For a captured real transform, Nsight Compute measured its two kernels at
22.91 and 40.86 us, compared with 205.25 us for the dense single-stage kernel.
Five randomized, transfer-inclusive replay trials per variant measured medians
of 809.964 us (dense) and 748.813 us (separable). This much smaller improvement
shows that reducing arithmetic alone cannot remove the host/device boundary.
Compute Sanitizer reported zero errors for the separable replay.

Splitting independent toroidal components across threads reduced that stage
to 15.42 us and raised achieved occupancy to 74.05%, compared with 9.24% for
the original separable stage. The second kernel measured 41.66 us. Captured
output bytes match exactly. This variant has not passed a full fixture matrix.

Ordinary VMEC threading is rejected as comparable on the tested P3 fixture:
two threads changed metrics by up to 5.76068e-8 relative to baseline, and four
threads by up to 9.16359e-8. Four-thread repeats differed by up to 6.10032e-8
(vacuum_well). The two-thread repeats matched each other, but not baseline.
These configurations remain outside the completed table regardless of timing.

Parallel independent Fourier transforms using serial solver reductions passed
two four-core and two eight-core P3 runs with exact metrics and numerical fields.
Nested OpenMP worker creation adds overhead; a persistent team was tested
instead. Its dispatch/recovery test passed normally and with address/
undefined-behavior sanitizers. The persistent and broader pointwise builds
passed all 14 C++ test targets. The pointwise full matrix is reported above.
ThreadSanitizer could not run: it exited with an unexpected-memory-mapping error;
that check is not counted as passed. The pointwise preview compiles successfully
and includes geometry metrics, field components, differential-volume rows, MHD
forces, and force assembly. Every reduction within an angular row retains its
serial order.

A strict-FP64 standalone GPU forward transform passed captured-data replay with
zero differing bytes and two complete P3 runs with identical metrics and wout
fields. Measured local adapter times were 59.190 and 58.466 s, versus 53.581 s
for the CPU-reference capture run. These are unpaired qualification runs, not
a benchmark gain. Transfer-inclusive replay averaged 753.319 us over 100 calls
after ten warmups (5,310,608 input bytes and 522,720 output bytes per call).
This negative result motivates testing fused MHD forces, constraint assembly,
and the forward transform with intermediate force arrays retained on the GPU.
The fused prototype passed captured MHD and complete-pipeline replay with zero
differing bytes, followed by two complete P3 evaluations with exact metrics and
numerical wout fields. Measured A100 PCIe adapter times were 115.985 and 116.047 s,
versus 142.490 s for its CPU-reference capture run. This is an unpaired
qualification result, not a six-case speedup claim. Transfer-inclusive A100
pipeline replay averaged 1253.382 us over 100 calls after ten warmups, with
8,521,664 input bytes and 522,720 output bytes per call.

The same captured pipeline passed local RTX replay with zero differing bytes
and Compute Sanitizer memcheck with zero errors. Local Nsight Compute measured
MHD, constraint, forward partial, and forward finish kernels at 21.34, 4.99,
42.37, and 35.42 us respectively. MHD achieved 12.96% occupancy; forward partials
9.21%, and forward finish 8.18%. No near-roofline claim is justified. A separate
local Nsight Systems trace averaged 315.314 us H2D, 18.171 us D2H, and 67.821 us
for the four device kernels combined per call (111 observed calls, including
warmups and correctness checks). These profiler observations are not added to
unprofiled replay timings; clocks and collection overhead differ.

A100 hardware counters again returned ERR_NVGPUCTRPERM. Its raw Systems trace
was retrieved and imported locally after the remote CLI lacked its importer.
The combined inverse-transform/fused-force/CPU-worker configuration is under
independent qualification. New runs additionally hash numerical jxbout, mercier,
and threed1_volumetrics fields and record process and main-thread CPU time;
the older completed tables only establish the wout coverage stated above.

In the first expanded-output combined P3 probe, baseline measured 142.038 s and
fused forces plus the separable inverse measured 92.748 s. All captured metrics,
wout, jxbout, mercier, and volumetric numerical fields match. The ratio is about
1.53x, derived from one unpaired qualification comparison, not a qualified
six-case benchmark. Repeats and a four-worker combination remain under test.

Five randomized local replay trials per upload mode measured medians of
1267.722 us for packed input, 1347.238 us for direct copies, and 1904.955 us for
batched copies. All output bytes match; direct and batched uploads are rejected
for performance on this device. A separate randomized campaign measured
1216.411 us packed, 1301.555 us with a one-worker pool, and 1084.294 us with
four persistent packing workers, using the same CPU affinity {0,2,4,6}.
This is only replay evidence; full-solver qualification is in progress.

An import-state diagnostic measured the same arithmetic control mode in the
original and CPU candidate environments: MXCSR 0x1fa2 after forward_model import,
with FTZ and DAZ both disabled. The differing exception-status bits during
intermediate imports do not establish a timing cause. Timing regimes remain
unresolved rather than being attributed to an unmeasured explanation.

For 99 complete post-warmup A100 replay periods, the measured trace averages
1447.192 us per period: 70.992 us kernels (4.91%), 480.067 us H2D/D2H (33.17%),
3.619 us memset (0.25%), and 892.514 us without recorded work on that stream
(61.67%). The last category combines host packing, scheduling, launch, and
synchronization gaps; it is not separately attributed. These are component
replay observations, not full-solver utilization or a percentage of roofline.
`analyze_pipeline_trace.py` derives them from device events without adding
overlapping API durations. Eliminating all four kernels would cap this replay's
improvement at about 1.052x with everything else fixed, a derived upper bound,
not an executed speedup.

The combined A100 qualification subsequently measured 92.007 and 92.748 s with
one CPU worker, and 90.422 and 91.898 s with four, versus 142.038 s baseline.
Every captured numerical diagnostic matches. The independently timed six-case
matrix uses the four-worker frozen binary and excludes these selection probes.
Local parallel-packing P3 qualifications measured 48.892 and 49.136 s, with exact
metrics and all captured diagnostics; the frozen combined local qualification
measured 60.603 s and its baseline 72.307 s. These remain unpaired observations.
The separate 30-second native profile captured 2662 samples with two errors;
it is not a completed or timed benchmark solve.

## Options and the remaining roofline gap

| Approach | Evidence and current decision | Next useful test |
| --- | --- | --- |
| CPU locality alone | Exact six-case matrix, no demonstrated gain | Do not promote |
| CPU independent outputs | Exact six-case matrix, 1.09-1.48x, variable timing | Secondary-host repeats and utilization attribution |
| Ordinary VMEC threads | Metric drift and four-thread repeat variation | Alternative configuration only |
| GPU inverse only | Exact six-case A100 matrix, 1.19-1.20x | Combine with broader resident work |
| GPU forward only | Exact P3 repeats, slower locally | Reject as standalone change |
| Fused GPU force region | Exact captured data and P3 repeats, modest gain | Combined independent fixture matrix |
| Faster GPU kernel launches | Device kernels improved, transfers dominate | Retain residency across geometry, fields, and forces |
| Direct or batched uploads | Exact replay, slower on tested RTX | Do not promote |
| Parallel host packing | Exact replay, smaller transfer-inclusive cost | Whole-solver parity and paired timing |
| Independent equilibrium batching | Not executed | Measure throughput separately from per-solve latency |
| FFT, tensor cores, fast math, relaxed fidelity | Not qualified for identity | Treat drifted results as alternatives |

The measured counters do not justify calling these kernels bandwidth-bound
simply because VMEC uses small radial matrices. The profiled MHD and forward
kernels have small grids, low achieved occupancy, and many cycles without an
eligible warp. The pipeline also spends much more time transferring data than
executing kernels. Buying a GPU with higher arithmetic throughput alone cannot
remove those measured costs. Broader residency requires porting geometry and
field calculations plus their ordered reductions, with exact final-state
materialization and the same full-output gates. No speedup for that unbuilt
approach is estimated here. Datacenter FP64 hardware and stronger CPU SIMD
remain hardware options, not unexecuted benchmark claims.

Priority order is broader device residency, host packing and graph submission,
more independent output work per kernel, CPU SIMD/cache and ordered solver
dependencies, then independent-equilibrium throughput and hardware selection.
Keep the original rounding order within each dependency chain throughout.
The CPU arithmetic-intensity/sustained-bandwidth model and A100 application
counters are incomplete, so no single full-solver roofline percentage is claimed.

The literature supports this direction but does not establish additional speedup
on our fixtures. NVIDIA's [transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
prioritizes keeping intermediate data on the device. [CUDA Graph launch work](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)
addresses repeated submission overhead, not the remaining data traffic.
[FPRev](https://arxiv.org/abs/2411.00442) can help audit accumulation order.
[ReproBLAS](https://eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf)
provides order-independent reproducibility, which does not promise the existing
sequential result; it is not an identity-preserving replacement by default.
[VMEX implementation notes](https://vmec-jax.readthedocs.io/en/latest/explanation/solver-landscape.html)
discuss batched Thomas kernels and Newton-Krylov tails. Mapping independent
Thomas chains is a potential exact-order optimization; replacing the iteration
or factorization needs separate identity qualification. [DESC's different basis
and solver](https://arxiv.org/abs/2203.17173) belong to the alternative-configuration
track, not the comparable benchmark. None of these external performance claims
is reported as an execution of our workload.

## Resident iteration development (not yet integrated into a complete solve)

The resident inverse-to-field path executes inverse Fourier synthesis, Jacobian,
metric, lambda normalization, differential volume, the ordered toroidal-current
constraint, profile interpolation, and covariant/contravariant fields on one
CUDA stream. The GPU arrays feed subsequent stages directly. The chain replay
sets all 18 host geometry pointers to null and still matches every one of 38
captured output arrays byte for byte, before and after 100 repeated calls.
The unmodified CPU capture solve matches the reference metrics, wout, jxbout,
mercier, and volumetric fields. Memcheck reports zero errors. This is intermediate
array qualification on one captured real state, not a completed GPU solver or
whole-fixture qualification. Downloads at the validation boundary intentionally
materialize all arrays for comparison.

The first conservative port exposed a 5.225 ms mean serial Jacobian-extrema
kernel in the local Systems trace. Parallel exact min/max selection reduced
its two kernels to approximately 1.29 and 4.97 us in the later trace. Zeros,
infinities, and NaNs take the original ordered fallback; 36 adversarial cases
passed, including signed zero, NaN payloads, subnormals, and warp boundaries.
The full resident chain retains exact output bytes after this change.

The next profile identified the ordered volume/current integrals at 188.72 us
mean. Coalesced point-term generation plus shared-memory tiled accumulation
reduced their combined measured kernel time to about 42.85 us, while retaining
the exact original addition order within each surface. All 38 arrays still
match, and memcheck passes. Summed mean kernel durations in that trace are about
103.75 us for the inverse-to-field chain; this is device work in a component
trace, not an end-to-end wall time. The validation replay includes all-array
downloads and is not used as a solver speedup.

The resident chain was subsequently extended through pressure, lambda forces,
and MHD forces. Its 12 MHD outputs, four lambda outputs, and total pressure
match the CPU reference byte for byte before and after 100 repeated calls with
null host geometry pointers. Memcheck passes. The actual mass profile is nonzero
and was captured; the pressure specialization is gated on unchanged gamma=0.
This is not a zero-pressure substitution. Timelines and snapshots are in
`build/optimization/resident-mhd/`.

The inverse-to-field chain also passed on the secondary RTX 3090, with zero
differing bytes across the same 100 repeats. Its validation timing is not
promoted as a solver benchmark. In 111 local trace windows from inverse-kernel
entry through field completion there are zero intermediate D2H bytes. The only
intermediate H2D is the debug adapter's 7200-byte profile initialization; full
array downloads occur after the validation boundary. The production-facing
launch API consumes device-owned state directly.

The resident force evaluation now also includes constraint filtering,
constraint-force assembly, and forward projection. All six Fourier-force
channels and four constraint-geometry arrays match the captured CPU reference
in a 100-repeat replay with null host geometry pointers. Memcheck, racecheck,
and synccheck pass. Its replay adapter downloads final validation arrays;
the reported replay time is not a solver speedup. Evidence is in
`build/optimization/resident-force-eval/`.

The standalone iteration-stage gate captures a real solve's M1, radial
Thomas, and lambda preconditioner inputs and outputs, plus force residuals.
The capture preserves the complete reported metric, numerical wout, and
additional diagnostic hashes. All captured GPU outputs match byte for byte
over 110 executions. Memcheck, racecheck, and synccheck pass, including the
coefficient/velocity update gate. A subsequent capture also qualifies all
seven energy/volume/force-normalization values and physical-force preparation
through those same checks. Full CPU reference evaluations retain exact metrics,
wout, and additional diagnostic hashes. Evidence is in
`build/optimization/resident-iteration6/`, `resident-iteration7/`, and
`resident-iteration10/`.

The connected chain now runs inverse, geometry, fields, pressure, lambda/MHD
forces, constraint filtering/assembly, forward projection, coefficient
preparation, M1/radial/lambda preconditioning, both raw residual evaluations,
and the coefficient/velocity update on GPU. Captures are checked to share the
same real intermediate state before joining them. Host geometry and intermediate
force input pointers are null. Each warmup and measured replay is compared,
with zero differing output bytes. Memcheck, racecheck, and synccheck pass.
The Nsight timeline checks 111 windows from inverse entry through the completed
coefficient step: all have zero intermediate D2H bytes. The reference adapter
still uploads 579296 bytes of captured profile/constraint initialization within
each window; these are not CPU-computed intermediates of the iteration. Final
debug comparisons download arrays after the GPU step. See
`build/optimization/resident-connected-step/`. Replay wall time includes these
initializations, downloads, and comparisons and is not a solver benchmark.

The initial radial solve kernel measured 228.4 us median locally. Splitting
factorization from independent RHS solves retains all divisions and Thomas
ordering, passes captured parity, and measures about 47.0 + 92.0 us median in
the connected trace. This is a component improvement, not an end-to-end ratio.
A shared-memory prefetch trial for the residuals showed no convincing benefit
and was removed; its immutable experiment remains in `resident-iteration8/`.

Nsight Compute counters for the ordered residual kernel show 3 blocks across
170 SMs, 0.37% SM throughput and 0.04% DRAM throughput. These are measured
counter-derived kernel statistics, not a percent of whole-solver roofline.
The kernel is far from saturating the device; exact ordered accumulation leaves
very few independent dependency chains. The profiled 848.90 us duration includes
counter-replay effects and must not replace unprofiled timing. The report and
CSV are under `resident-iteration9/residual-ncu*`. Priorities are to finish the
resident solve/control path, improve the ordered reduction's instruction and
dependency behavior, test on FP64-capable datacenter hardware, and quantify
independent-boundary batching separately from single-solve latency. No near-peak
claim or unexecuted batch speedup is made.

The preconditioner refresh now separately qualifies all 20 outputs: lambda
preconditioner and profiles, R/Z radial coefficients, constraint multiplier,
and assembled tridiagonal matrices. Every output matches the real CPU capture
over 110 executions, with memcheck/racecheck/synccheck passes. Original host-libm
powers depending only on radial grid and mode are captured once as constants;
the refresh itself executes on GPU. One reference capture hit its observation
timeout; a retried capture completed and retained exact full-evaluation hashes.
The timed-out capture is not counted as a completed run.

The connected physics body now includes fresh GPU preconditioners and all seven
energy/volume/normalization values. All 111 ordinary executions and memcheck
pass. Full-length race instrumentation exceeded five minutes and is not counted
as passed. Bounded racecheck and synccheck each pass three actual executions,
covering graph creation and reuse. The expanded timeline checks 111 inverse-to-
coefficient-step windows and finds zero intermediate D2H bytes in every window.
See
`build/optimization/resident-connected-refresh/`.

In the refresh-heavy replay, node-level CUDA Graph tracing attributes 86.1% of
recorded kernel duration to ordered energy/normalization sums (3559.0 us median
for that kernel). The earlier graph-level trace omitted individual graph nodes
and is not used for this breakdown. This replay
refreshes preconditioners/norms on every call, whereas the actual solver refresh
schedule must remain unchanged. It is not a whole-solve time breakdown. The
generic certified-sum approach is a candidate for these sums, but has not been
implemented or measured there. Full-loop integration takes priority.

Geometry preparation, persistent iteration state, multigrid transitions,
convergence/restart control, and full-trajectory/output parity remain unfinished.
The complete resident solver has not run. The completed mixed CPU/GPU table uses none
of this new path. Raw snapshots and traces are under `resident-fields/`,
`resident-chain/`, `resident-reduction3/`, and `resident-tiled/` in
`build/optimization`. The old materializing inverse API was separately replayed
after adding the resident entry and still has zero differing output bytes.

CUDA Graph trials passed captured-data parity and memcheck but did not
demonstrate a timing benefit: randomized replay medians were 1177.32 us packed
without graphs versus 1257.01 us with graphs, and 1066.71 versus 1173.18 us with
parallel packing. Timing variability is substantial. They remain outside the
benchmark candidate rather than being assumed beneficial from the literature.

The profile-guided per-mode CPU staging/Thomas and independent constraint
changes passed the full Bazel build, all 14 C++ tests with their flags enabled,
and four P3 full-output parity probes. These are unpaired qualification runs;
no new CPU speedup is promoted from their timings.

## Certified optimistic residual experiment

The primary literature describes parallel prefix proposals followed by
correction and recurrence checks to reproduce sequential accumulation
([Kapre and DeHon, 2007](https://ic.ese.upenn.edu/pdf/fpaccum_arith2007.pdf)).
Our experiment accepts a proposal only when every prefix satisfies the original
rounded recurrence bit for bit, starting at +0. Failed certificates and
nonfinite proposals select the original ordered GPU kernel. This is not an
approximate stopping criterion or a changed residual. A CPU algorithm probe
on the six actual residual vectors required 1-10 correction rounds.

These are measured medians of three randomized component measurements on the
local GPU. Each measurement uploads the captured input once, excludes 10 warmups,
times 100 calls using CUDA events, and checks every result against the original
CPU capture. Timing includes scheduling gaps and a 24-byte device-to-device
result snapshot; validation downloads occur after timing. All configurations
below have zero differing output bytes, including cases that fall back.

| GPU residual schedule | Before preconditioning us | After preconditioning us |
| --- | ---: | ---: |
| Original ordered kernel | 572.033 | 567.079 |
| 12 corrections, ordinary launches | 717.552 | 750.705 |
| 12 corrections, CUDA Graph | 180.107 | 144.577 |
| 8 corrections, CUDA Graph | 653.676 | 110.326 |
| 4 corrections, CUDA Graph | 616.850 | 617.629 |

Twelve corrections with graphs produced no fallbacks on either captured stage
in these measurements. The shorter schedules fall back on some components;
they do not change outputs, but can cost more than the original kernel. The
12-correction graph variant also passes connected iteration-body byte parity,
memcheck, racecheck, and synccheck. These results do not establish full-solver
speedup or coverage of an entire trajectory. The extra parallel work must not
be counted as useful solver FLOPs to inflate a roofline claim.

The initial component harness crashed at cleanup; GDB identified synchronization
of an already-destroyed caller stream. The workspace cleanup was corrected and
the complete randomized campaign rerun successfully. Failed-run timings are
excluded. A separate single-warp ordered trial was slower (about 611 versus
573 us median) and is not enabled by default. Raw data and code snapshots are in
`build/optimization/resident-iteration12/` through `resident-iteration15/`,
with the accepted component samples in
`resident-iteration15/residual-benchmark-samples.json` and its summary JSON.

## Secondary-host corroboration

The Threadripper 2950X matrix completed 36 executions: one warmup pair and two
measured pairs per boundary/track. All six cases have zero metric drift, exact
numerical wout fields, and identical repeat hashes. Every original-baseline
case also matches the local workstation baseline across hosts. The short
campaign does not meet the five-pair qualification rule for the main table.

| Boundary | Track | Measured baseline median s | Measured CPU median s | Observed ratio |
| --- | --- | ---: | ---: | ---: |
| Public P3 | P1 | 118.775 | 94.002 | 1.264 |
| Public P3 | P3 | 123.556 | 95.369 | 1.296 |
| Public P2 | P1 | 110.480 | 86.089 | 1.283 |
| Public P2 | P3 | 111.297 | 88.968 | 1.251 |
| Third frozen fixture | P1 | 90.426 | 72.175 | 1.253 |
| Third frozen fixture | P3 | 94.250 | 74.574 | 1.264 |

Per-metric drift, repeats, and raw pairs are in
`build/optimization/secondary-pointwise-summary.json`; cross-host checks are in
`crosshost-secondary-parity.json`. Additional diagnostic sections were not
captured by that campaign's older harness.

See [README.md](README.md) for build/reproduction instructions and [patches](patches)
for the measured CPU patches. Raw profiles, fixture provenance, dependency
snapshots, and experiment binaries are under `build/optimization`.
