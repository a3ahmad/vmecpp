# Measured component limits

Latest actual evidence: [three native Fourier counter captures](COUNTER_PROFILE.md)
passed all63 requested values, original packet endpoints and independent audit.
RTX3090 FP64-pipe activity is70.38/68.95/59.77% for inverse/partials/finish.
Theoretical fixed-work arithmetic headroom is about1.42x/1.45x/1.68x, not a
measured sustained roofline or an A100/whole-solver speedup. Scans/seeding and
submission costs remain the next profiling targets.

## Current full-call evidence

The [current A100 P3 pilot](A100_RAGGED.md) completed14 original-exact
evaluations. Its five measured pairs give1.052811x VMEC speedup for CLEAN
ragged versus selected; consistent fresh-call gains remain unproven. This
single-fixture result is separate from the native six-case comparison.

[Same-clock analysis](../../build/optimization/accelerated-call-fractions1/FINDING.md)
puts accelerated VMEC at39-41% of the complete A100 adapter envelope and35-37%
of the fresh worker-process envelope in this fixture. These are per-call
fractions with explicit bookends, not unrelated median ratios. Outside time
includes imports, request preparation, metrics, persistence and other work;
it is not all imports, pure CPU work, GPU idle or a proven removable cost.
The user's historical94-97% CPU-only scope must not be carried forward silently.

The [native MPS pilots](THROUGHPUT.md) preserve exact outputs at widths2/3/6,
but improve throughput while increasing individual latency and summed interval
cost. They neither establish kernel overlap nor qualify a production budget
change. Source review of Geometry/MHD finds independent points, but existing
counter rows lack FP64 pipe utilization. Low generic issue rates on consumer
FP64 alone do not establish bubbles. Block32/64 variants are CPU-built with
matching device instructions, but remain GPU-untested and unselected.

An independent read of the saved historical selected-build RTX5090 captures
now quantifies their executed FP64 rates. Geometry99 executed6,455,554 FP64
FLOPs in15,456ns (417.673GFLOP/s); MHD99 executed8,533,700 in20,352ns
(419.305GFLOP/s). FLOPs count DADD+DMUL+2*DFMA from actual instruction counters.
Measured active-warps percentages were17.81/19.13 and issue percentages7.10/7.50.
Measured DRAM rates were6.062/20.038GB/s; L2 rates604.826/577.200GB/s. These
are instrumented historical kernel samples, not current CLEAN or A100 rates.
They do not establish a sustained roofline or a whole-solver fraction. See the
[counter derivation](../../build/optimization/historical-selected-fp64-rates1/counter-rates.json).
The separate clock-normalized ceilings in that finding are theoretical bounds,
not measured sustained performance.

The [current A100 PCIe software-node trace](A100_PROFILE.md) completed one
original-exact P3 solve: 1,038,197 observed kernels, 4.683224 s kernel union
inside an 8.912759 s instrumented whole-call interval. Collection warnings
remain. Scans/certification and Fourier work lead device investigation;
Geometry/MHD are 3.18% of recorded kernel-duration sum. Host/device copies total
0.055919 s of recorded device duration. The 27,779 synchronization API calls
overlap GPU work; their 5.255892 s elapsed sum is not removable CPU overhead.

Source audit finds 27,521 hot synchronization points: two reads per valid
iteration, except the bad-Jacobian branch. A conditional prelude/tail merge
could retain the exact branch and original tail while removing one round trip;
its default-off prototype is not GPU-qualified. A same-host unprofiled/graph/
node/unprofiled screen is needed before quantifying tracing perturbation.
Conditional ns-compatible kernel sums and the
[stage ledger](../../build/optimization/a100-current-stage-ledger1/FINDING.md)
are not measured multigrid stage wall times. No new A100 hardware counters or
achieved roofline percentage is reported.

Controlled profiling is now the priority before further candidate tuning. The
expanded local script retains exact full-call output guards and targets the
actual scan/seed/Fourier names and ns99 launch shapes from the current trace.
Its first hardware sample has not run. Local GPU availability is being checked;
the native3090 counter policy is admin-only and noninteractive sudo failed.
Fresh bounded A100 VM searches, including multi-GPU offers and relaxed CPU/
network filters up to3USD/hour, returned zero matching offers at those query
times. A failed counter request on one ordinary A100 host does not establish
that every host blocks counters: the next host's actual policy must be read
before upload or GPU work. No new lease or counter measurement is claimed.

## Latest residual evidence and next experiment

The final-only residual prototype a97 is not selected. A native RTX3090 matched
screen actually consumed12800 packets across four ns99 cases in direct and
host-graph modes. All four graph CUDA-event CURRENT/CANDIDATE paired medians
are below1 (0.89865,0.88592,0.97800,0.96353); its fewer graph nodes did not give
a speedup. Outputs and repeated typed statistics are exact. Resource status is
UNQUALIFIED_DIAGNOSTIC. See the
[native screen](../../build/optimization/residual-final-native-timing-independent1/FINDING.md).

Two separately instrumented RTX5090 calls recorded13 current nodes and9
candidate nodes. The largest observed candidate node is FoldFinal11.024us,
followed by double CUB scan4.880us, ComposeFieldsWarp3.675us and
BuildPartials3.464us. Both retain incomplete-collection warnings, so these are
observed-node self times only. They establish neither complete activity,
reclaimable idle, a whole-solver fraction nor a hardware-roofline percentage.
See the [trace audit](../../build/optimization/residual-final-profile2-independent1/FINDING.md).

A distinct capped-CUB successor replaces manual partial construction and
composition with a materialized ordered scan of capped integer maps and three
warp tail consumers. It keeps literal force squares, full domain/coverage/seed
guards, original same-slot live-force fallback and the generic prefix API.
Its first actual GPU normal gate passed160 packets and18 original CPU scalar
anchors; independent sealc88a17eb confirms every real row accepts four natural
calls and declines four forced calls per component. All three sanitizer runs
also passed independent audit5b2a28f7:480packets/2400words and zero tool errors
or race warnings. A matched12800-packet WSL timing screen completed, with mixed
graph event medians1.05340/0.91799/1.03982/1.11239 (CURRENT/CANDIDATE). Its clocks
and resource state remain unqualified, and native replication is pending.
Whole-solver integration is unmeasured. This tests the measured fold/dependency
problem; it is not an accepted speedup.

The next whole-P3 A100 attempt59457905 reached direct SSH, exact upload and
owned supervisor startup, then stopped before bootstrap because observed
SYS_ADMIN effective/bounding bits were false. Its six diagnostic files were
retrieved and teardown was verified. No solver result or hardware-counter
measurement came from that attempt. The whole-timing workload contains no
profiler, so an isolated successor will retain the capability witness without
requiring counter privileges; the actual counter-profiling gate stays separate.

## Actual cuFFT toroidal-only alternative

Two local actual runs use the fresh canonical actual7 ns99 input and cuFFT11400
on RTX5090/driver610.88. The literal N3 ordered DFT matches the CPU output bits;
the toroidal24 FFT changes57701/65340 component values, max absolute error
5.30825383648903e-16. Both persisted output arrays repeat bitwise across runs.
An independent array audit recomputes all six mismatch counts and maxima.
This alternative is ineligible for the identity-preserving main comparison;
full-solver metric drift is unmeasured, with no full-solver FFT integration.

The untraced32 randomized paired samples measure event medians0.0679680035ms
ordered versus0.0725120008ms FFT, paired baseline/FFT ratio median0.9462889215.
Scope includes output zeroing, original weighted theta partials, FFT packing,
reused FFT plan, finish/scaling/unpacking and ordinary-stream enqueue gaps.
Cold plan creation2.170431ms is separate. No competing compute app was listed;
desktop GPU utilization was7%, so this is a limited local component comparison,
not a solver speedup or a pure kernel-throughput roofline measurement.
Actual evidence: fourier-library-toroidal1/local-untraced-run1/{terminal.json,
independent-array-audit.json,component/report.json}; the preceding closure run
records actual CUDA/cuFFT/driver libraries and excludes traced timing claims.

## Latest selected-controller graph-level diagnostic

The versioned production25 loop now has two full publicP3/P3 exact-output solves,
independently audited under sealc0703386. Both return669 flat chunks for13192
Update attempts, versus7210 chunks in the earlier two-slot controller. This is
10.8x fewer flat-chunk returns, not an all-readback count or measured wall-time
gain. Scope/host-lease handoffs still exist. Default OFF retains164 original
instruction occurrences; ON changes32 packet-metadata entries with other math
words/resources and operation order unchanged. Whole-solver roofline, six-case
qualification, new SAN and performance remain unproven. Focused conditional-body
node profiling is being prepared to locate the remaining device pipeline cost.

The local selected solve was captured successfully; its orchestration later
failed because the coordinator used a Python without NumPy. This failed terminal
is preserved. Separate saved-output and SQLite audits passed: original metrics,
NC100/NPZ50, positive selected activity, target PID79488, range/provenance and
owned retirement. Root exported the saved report without rerunning the solve.
The missing flat-controller capture subsequently completed terminal0 using the
pinned scientific Python coordinator. Its separate scientific and SQLite audits
passed; the earlier failed selected orchestration remains failed.

| Instrumented selected-call observation | Actual count | Summed time s |
| --- | ---: | ---: |
| Graph-launch API | 27,477 | 0.594027392 |
| Stream-synchronize API | 27,782 | 5.208113482 |
| Graph-instantiation API | 41 | 0.008811799 |
| Device-to-host copies | 28,824 | 0.012399384 |

The frequent 16-byte (13,761) and 112-byte (13,760) readbacks are already pinned.
Device-to-host volume was61,911,256 bytes; all40,328 copies uniquely correlate
with actual runtime APIs. Graph-envelope union was4.159648132s, visible direct
kernel union0.026400817s and all observable GPU-envelope union4.208815880s.
Caller API union was6.463701871s; API time outside those observable envelopes
was2.263371740s. These overlapping quantities are not additive. Graph envelopes
are not SM-busy time, and synchronization includes queued device work. None of
these values establishes CPU compute, reclaimable idle, unprofiled latency or
a whole-solver roofline percentage.

This evidence prioritizes reducing launch/wait boundaries over optimizing the
already small graph-instantiation cost or merely reducing readback bytes.
CUDA conditional WHILE graphs now have bounded two-slot correctness evidence,
with unchanged arithmetic, journal limits, control-key handoffs and failure
guards. Two full publicP3/P3 calls passed original whole NC bytes,100numeric
fields,50actual diagnostics each, metrics, CPU trajectories and repeated arrays.
Actual WHILE3[19,7210,7210] confirms created/submitted/valid-packet activity;
independent sealb2cddb45 binds that result. Comparative performance is unmeasured.
The
[CUDA12.9 guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)
documents device conditional evaluation and the body-node restrictions.

Evidence: `flat-outer-graph-selected-export1/receipt.json`,
`flat-outer-graph-analysis1/selected1/analysis.json` and
`flat-outer-graph-profile1-independent1/selected-SQLite-audit.json`.

The flat trace has14,284 uniquely correlated copies. Its frequent 1,808-byte
Device-to-Pageable group has7,987 calls: correlated API union4.888054661s versus
only0.003142614s summed device-copy duration. This difference includes queued
work/wait placement and possible staging; it is not measured pure-copy cost or
recoverable time. Observable GPU-envelope union was5.260984468s, not SM-busy time.
Independent saved SQLite comparison:
`flat-outer-graph-profile2-independent1/actual-flat-SQLite-comparison-audit.json`
(80ef3537).

The separate default-OFF pinned-packet prototype also passed two full original-
exact publicP3/P3 calls and repeats, with activity[16,7987,16], owner-bound reads
and retired allocations; independent sealf3654dff. Its later six-case/two-repeat
matrix also passed:12 original whole NC files,600 diagnostic fields,138 numeric
metric entries with zero drift,6 nulls and six repeat pairs, independently
audited under sealbbaf0e39. These are correctness runs under observed background
GPU activity. Both prototypes preserve all164
old device instruction occurrences. The base natural-bootstrap component's
memcheck/synccheck/racecheck passed independent audit65315782; that scoped pass
does not repair the older legacy190 timeout or qualify these new variants under
SAN. No pinned-transfer gain, reduced host wait or roofline improvement follows
from their current correctness results.

The native six-call CPU/selected/flat smoke is complete, but its original CPU
baseline differs from canonical local metrics by one ULP in axis mirror ratio and QI
despite exact whole NC/50diagnostics. It is excluded from main comparisons.
Local saved-WOut replay reproduces all12 canonical metrics with no solver calls
(seal6e045051). Root's current-native replay also reproduced all12 onsite metrics
with zero solver calls (native-metric-replay-current1-root-receipt.json).
A fresh private three-package intervention (NumPy2.2.6/SciPy1.17.1/Simsopt1.8.1)
then matched all12 canonical metrics on the same WOut, with no expected-metric
input and zero solver calls. Original installed packages were untouched.
Receipt: `native-metric-replay-pins3-1-root-receipt.json`, result033a0e7b.
This restores one-fixture replay identity, not full-matrix qualification or
causality for one package. A later NumPy-only replay restores QI but leaves the
axis mismatch; NumPy2.2.6/SciPy1.17.1 together restore all12 while retaining native
Simsopt1.11.1 (configd91d320e, result29692d9d). This is the smallest tested bundle
for that one fixture. The original smoke remains excluded. This identity
investigation supplies no new hardware ceiling or speedup measurement.

The current split Fourier kernels reach67-68% and60-62% of the strongest
strict-FP64 reference measured so far on one captured ns99 input. This is not
a whole-solver roofline percentage, absolute hardware ceiling or speedup
qualification. The newer reference search below supersedes the older
reference fractions; historical measurements are retained unchanged.

Qualification update: the later unroll study measured92-93% GPU utilization
and about360W before/after its component jobs despite an empty WSL compute
process list. Independent Windows inventory confirmed shared activity that
WSL did not expose. That study's timings are uncontrolled and cannot select
a speedup. Earlier wave and six-finish snapshots recorded only the WSL
compute list, so shared Windows load was not excluded there either. Their
actual counts, correctness and recorded ratios remain preserved, but local
timing/reference fractions are provisional pending controlled Linux
reproduction. Main native direct72 qualification is unaffected. Evidence:
`forward-finish-unroll1/qualification.json` and `shared-load-context1/note.json`.

## New final-only energy regression: measured serial fold cost

Root executed one unchanged consumed-output component process under the
compatible private Nsight 2026.5.2. The profiler/target and all CPU postprocess
jobs exited0, all owned children were reaped, frozen hashes were unchanged,
and all 1,400 consumed plus 280 original-oracle packets passed. The actual
trace contains 32,864 kernel records, including 32,200 graph-body records.
This is an instrumented RTX5090 component experiment, not a solver timing or
an exclusive-load performance qualification. Independent trace and packet
review passed in `energy-final-profile-independent1/audit.json`; raw evidence
is `energy-final-profile1/gpu-profile1/`.

| Graph containing candidate FoldFinal | Fold calls | Fold self-time sum ms | All graph kernel self-time sum ms | Fold fraction |
| --- | ---: | ---: | ---: | ---: |
| 5 | 350 | 30.707734 | 39.537235 | 77.6679% |
| 11 | 350 | 27.547341 | 34.886666 | 78.9624% |

All numbers above are measured from the saved node trace. FoldFinal launches
one block with one thread. Across the entire process it has 702 records and
58.421550 ms summed self-time; BuildPartials has 702 records and only
1.776653 ms. The source serially walks tile partials through FindPartial and
Apply. This is measured motivation to compose ordered tile maps in parallel
within each exponent range, retaining the original cross-range seed arithmetic
and all metadata/domain checks. No such replacement is qualified yet.

The full-process OrderedSums total, 940.531777 ms, belongs to the original
oracle controls outside these graphs. Calling it the candidate hotspot would
misattribute control work. Kernel sums omit graph gaps and transfers and are
not additive with overlapping API intervals. These fractions do not measure
memory bandwidth, occupancy, an absolute roofline or whole-solver efficiency.
The earlier negative unprofiled timing pilot remains separate and unselected.

## A100 diagnostic3: actual graph nodes now visible

Two new diagnostic solves completed on A100/EPYC7662. Root independently
verified139 retrieved file hashes and recomputed every successful graph-launch
node gate. All27,477 host graph launches have recorded node activity;
1,153,961 graph kernel records plus16,409 direct kernels are present, and no
whole-graph envelopes remain. The lease was destroyed after retrieval;
providerpostflight10 independently confirmed zero optimization leases.

Measured marked call:9.819548346s. Actual original-run RAW bracket:
9.819492170-9.819494440s. Unprofiled anchor RAW bracket:
6.760317499-6.760319069s. These are separate fresh-process diagnostics;
node instrumentation materially perturbs the call. The unchanged profiled
report timer24.122322703s includes wrapper/profiler lifecycle costs.

Largest recorded kernel families (summed durations, instrumented call only):

| Kernel | Calls | Summed time ms |
|---|---:|---:|
| Lattice Map scan-by-key | 53,954 | 464.819 |
| SeedSegments | 53,954 | 359.586 |
| FourierToRealKernelSeparable | 13,761 | 335.270 |
| Double scan-by-key | 53,954 | 280.964 |
| ForwardFinish | 13,760 | 208.514 |
| Proposal | 13,760 | 190.810 |
| ForwardPartials | 13,760 | 159.079 |

Median recorded kernel duration is2.688us;880,627 of1,170,370 records are
below5us. Per-launch graph kernel counts range13-121. This motivates examining
small-kernel fusion and lattice scan/seeding dependencies while preserving
exact arithmetic, rather than assuming Fourier compute is the sole limit.

Kernel interval union is4.996916602s; kernel/copy/memset union5.446501140s.
The4.373047206s complement includes3.655964650s intersecting CUDA API time.
GraphLaunch APIs alone sum2.660398163s. These instrumented gaps are not proven
unprofiled idle, recoverable time, occupancy, bandwidth or roofline measurements.
No additional solver speedup is qualified by this trace.

Evidence: `a100-profileonly3-root-actual-audit1/{audit.json,timeline.json,
kernel-shape.json}`. Quoted-rate lease time chargeUSD0.055333 is estimated,
excluding transfers, billing minima and invoice adjustments.
## Actual magnetic scan counters: underfilled captured invocations

Three finite RTX5090 Ncu captures completed with all 27 component cases exact
after each profile. Independent audit verified one intended kernel/PID per
report, all 34 requested numeric counters and units, all 2,598 frozen inputs,
actual tool injection, watchdog reaping and no remaining owned processes.

| Captured kernel | GPU interval us | Grid blocks | Threads/block | Eligible warps/cycle | Barrier stall % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Magnetic SeedSegments | 5.120 | 1 | 32 | 0.128636 | 0 |
| Magnetic integer Map scan | 6.368 | 43 | 256 | 0.143713 | 49.967292 |
| Magnetic double guess scan | 7.328 | 11 | 224 | 0.090065 | 28.516250 |

The device reports 170 SMs. This is concrete grid underfill and dependency/barrier
evidence for these captured invocations, not a whole-solver occupancy bound.
Map scan executed zero DP arithmetic instructions; double guess scan recorded
99,594 DADD instructions. Faster FP64 arithmetic alone cannot address the
integer Map scan. Smaller CUB tiles/items are the next test, preserving exact
Map composition and the complete original recurrence certificate/fallback.

All three recorded DRAM byte counters were zero, not missing. Profile-observed
aggregate L2 byte counts were 154,944 / 2,095,840 / 1,442,752 respectively.
These are not useful problem-byte counts or isolated bandwidth measurements.
Caches/clocks were uncontrolled and shared Windows load was not excluded;
seven replay passes may perturb these short kernels. A matched tiny-kernel
control is needed to assess the traffic/measurement floor. No hardware roofline
fraction or whole-solver speedup follows from these profiles.

Evidence: `thermal-counter-independent1/actual-counter-audit.json` and
`thermal-ordered-tools1/gpu-ncu1/`. The separate single-component guess-scan
prototype passed its actual 27-case consumed-output gate, including corruption
fallback and direct/capture repeats. Its existing 72 kernel/architecture entries
remain instruction-identical; six CUB sum/init entries are added. Subsequent
sanitizer/timing results follow below; it is not selected.

## Ordinary guess scan: actual execution, exact output, no qualified gain

The opt-in single-component InclusiveSum path executed on hardware. One Ncu
capture identified its new const-double DeviceScanKernel at grid8 x416threads,
with all 34 counters present and all 27 consumed-output controls exact. It
recorded 133,416 DADD instructions and 147,511 integer instructions. Those counts
describe this captured invocation; they are not a paired timing comparison
against the earlier keyed scan. The independent all-27-case memcheck also passed
with zero errors, all 2,623 frozen inputs exact and no remaining owned processes.

A separate executed event study persisted and independently checked all 3,200
104-byte packets. Seven randomized measured pairs per mode followed the
preexcluded pair0. Each interval contains two24B memsets plus energy work.

| Provisional component scope | OFF median us | ON median us | Median paired OFF/ON | ON faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Direct submission | 279.40352 | 295.58848 | 1.012120 | 5/7 |
| Captured single-component graph | 74.20928 | 79.27232 | 0.889162 | 1/7 |

Direct paired ratios ranged 0.53-1.44; captured ratios ranged 0.61-1.26.
The differing summaries and scatter do not establish an improvement. Windows
clocks were 300MHz before and 2797MHz after; foreign load was not excluded.
These are measured provisional component values, not controlled regressions,
whole-solver speedups or a reason to change the selected build. Full-solver
qualification remains unexecuted for this prototype.

Evidence: `lattice-single-counter-independent1/actual-counter-audit.json`,
`lattice-single-tools1/actual-memcheck-audit.json`, and
`lattice-single-independent1/actual-timing-audit.json`.

Three separate default-OFF keyed-scan tile variants (128x2,128x4,256x2) have
strict SM80/SM120 full builds and have now passed 27 consumed-output cases each
(81 actual cases). Pristine/OFF host text and all 78 kernel/architecture
occurrences match exactly. Positive actual grids, multi-component whole-prefix
coverage, sanitizer and timing gates remain pending. Estimated grids must not
be reported as observed grid sizes. No policy variant is selected.

## Assemble/certificate fusion: compiled, not GPU qualified

The A100 diagnostic recorded 53,954 lattice Assemble launches totaling
125.352 ms. This is instrumented kernel duration, not recoverable unprofiled
time. A new opt-in candidate replaces Assemble followed by Verify with one
kernel. It reconstructs the predecessor proposal from immutable maps/seeds,
then retains the complete original finite, round-to-nearest bitwise check.
One policy reconstructs both proposals; the other shuffles interior-warp
predecessors and reconstructs warp boundaries. Overlapping terms/output
buffers delegate the original launch group. Seed and both scans are unchanged.

Four actual strict SM80/SM120 builds passed. An independent CPU audit verified
that source outside the opt-in include/launch group matches the original apart
from whitespace, and pristine/OFF host text plus the complete device fatbin
are byte-identical. A separate CPU indexing proof checks 56 cases and 318,008
proposal/predecessor rows, including component/warp/tail boundaries, poison and
named special-value bit patterns. This does not prove GPU arithmetic or safety.
No GPU run or speedup is claimed. Component/prefix, sanitizer, controlled timing
and full-solver exactness gates remain required before selection.

Evidence: `lattice-assemble-verify-design1/{CPU-final-proof.json,PLAN.md}` and
`lattice-assemble-verify-root-review1/audit.json`. A procedural qualification
harness is being prepared while local GPU work remains on hold.

## First complete A100 whole-call diagnostic

Diagnostic2 completed two actual fresh-process solves with exact original
identities and150 raw numerical fields. All137 retrieved hashes were checked
independently by root and validation. Lease59180370 was destroyed after final
retrieval; providerpostflight8 independently confirmed zero optimization leases.
The quoted-rate time charge is an estimate ofUSD0.03822 excluding transfers
and billing minima, not a measured invoice.

The profiled original VMEC call has a RAW bracket6.854005125-6.854006595s and
a measured Nsight NVTX interval6.854263058s. The unchanged outer report timer
11.428690553s includes combined FP-guard/profiler lifecycle/NVTX wrapper costs;
it is not pure VMEC time or a performance qualification.

The SQLite export contains27477 graph-inflight intervals,16409 standalone
kernel records,40328 copies and770 memsets for the target process. All are
contained in the whole-call marker. Graph/kernel/copy/memset envelope union is
5.259118496s (76.7277% of the marked interval); the complement is1.595144562s.
Of that complement,1.061361529s intersects target runtime API intervals.
These are measured recorded intervals, not SM-busy fractions or proof of
device idle. API/GPU durations overlap and must not be added as causes.

All standalone kernel records have graphNodeId=0 andgraphId=0: repeated graph
body kernels are not individually enumerated despite the requested node mode.
The69.530413ms standalone kernel sum is therefore not total GPU work, and no
per-stage/per-kernel hotspot or whole-solver roofline can be inferred from it.
Next native graph-marker smoke explicitly tests node visibility before a
matched selected/chunk2 trace. Evidence:
`a100-profileonly2-root-audit1/audit.json` and
`a100-profileonly2-independent1/{FINDING.md,graph-coverage-audit.json,clock-scope-audit.json}`.

## Reference tail and block-count follow-up

`roofline-wave-search1/analysis.json` records42 additional actual Ncu profiles:
12 strict-RN count/chain/block reference configurations and two unchanged
Fourier kernels, each in three randomized rounds. Twelve normal all-output
CPU bit checks and two focused memchecks passed. All56 GPU/profiler jobs and
42 CPU exports completed and were reaped. This is a reference experiment,
not a solver speedup or a change to any VMEC grid.

At262144 reference threads,128-thread blocks outperformed256-thread blocks
by derived1.07845x and1.07734x for8 and32 chains. At261120 threads, an equal
number of blocks per SM, those ratios collapsed to1.000312x and1.000138x.
Actual occupancy APIs confirm170 SMs and1536 maximum threads per SM, but the
32-chain/128-thread case is register-limited to7 resident blocks per SM.
Thus max-thread alignment is not necessarily a whole resident wave.

The strongest observed reference was1044480 threads,32 chains,128-thread
blocks:286.003929 executed FLOPs per mean SM cycle, approximately833.135
GFLOP/s. Close aligned/longer references differ by about0.2%; no statistical
winner or absolute hardware ceiling is claimed. Within-round comparisons
put unchanged Partials at67.52-67.79% and Finish at59.72-61.64% of this
stronger reference. All executed DFMA counts remain zero. The contrasts
support tail imbalance as the source of much of the earlier block-size gap;
direct scheduler-tail timing was not measured.

An isolated six-output finish candidate reduced SM120 registers54 to40,
passed all real/bounds/special/output guards and memcheck, but increased the
graph component median36.1093 to40.2969us in31 warmed measured pairs.
This candidate is rejected; a resource-count improvement is not a measured
latency improvement. Evidence: `forward-finish-six1/analysis.json`.

## Stronger strict-FP64 reference search

`roofline-chain-search1/analysis.json` records24 actual Ncu profiles: six
strict-RN chain/block configurations and two Fourier kernels, each measured
in three randomized rounds on RTX5090. Six normal all-output CPU bit checks
and two focused memcheck runs passed. All56 compiler-independent GPU/profiler
and export jobs completed and were reaped. Independent recomputation is in
`roofline-chain-independent1/audit.json`.

The best observed reference used32 independent chains and128 threads per
block. Its median executed FLOPs per mean SM cycle was265.371725 versus
245.398147 for the older eight-chain,256-thread reference: a derived1.081393x
stronger reference. Its measured median rate was774.650 GFLOP/s and FP64
pipe activity78.05%. All reference DFMA counts were zero; multiply and add
remain separate strict-RN operations.

Against that reference in the same randomized rounds, Partials reached
72.21-73.08% and Finish65.00-65.81%, normalized by actual SM cycles. Counts
are GPU-wide FLOPs divided by mean SM cycles, not individual-SM throughput.
The fixed262144-thread workload was not aligned to a whole resident wave;
tail underfill remains an untested explanation for block-size differences.
These observations strengthen the reference without proving absolute peak.

## Earlier matched reference measurements

Clock caveat: a later [external Windows-QPC check](../../build/optimization/clock-bracket1/FINDING.txt)
found WSL MONOTONIC/perf_counter interval disagreement. No historical records or
Nsight counter values below were changed or retrospectively rescaled. These
tables retain their exported GPU-metric provenance; reference fractions use
hardware clock-domain cycles. WSL host elapsed/API/NVTX whole-call times must
not be treated as independently validated wall time or combined across clock
domains. Native paired benchmark clocks passed independent coherence checks.

Durations are measured medians of three randomized Nsight Compute trials on
the same RTX5090; rates and fractions are derived from measured counters and
durations, not estimated. Physical DRAM and L2 counters are separate.
Clocks were observed, not locked; within-round fractions use measured cycles
in the corresponding SM, L2 or DRAM clock domain.

| Kernel/reference | Kernel us | Executed FP64 GFLOP/s | Physical L2 GB/s | Physical DRAM GB/s | FP64 pipe-active % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Split ForwardPartials | 23.936 | 491.816 | 798.689 | 0.118 | 57.04 |
| Split ForwardFinish | 14.752 | 418.334 | 342.753 | 0.017 | 51.26 |
| Independent strict multiply then add reference | 757.344 | 714.425 | 5.796 | 0.000 | 72.39 |
| 48MiB working-set copy reference | 10.688 | 0.000 | 4724.811 | 0.287 | 0.00 |
| 384MiB working-set copy reference | 226.464 | 0.000 | 1778.739 | 1536.151 | 0.00 |

The three within-round, clock-normalized FP64 reference fractions are
78.61-79.22% for Partials and70.56-70.89% for Finish. Corresponding L2 reference
fractions are16.34-16.85% and7.10-7.57%. These are ratios to observed empirical
references, not proven hardware ceilings. Executed FP64 work includes all
predicated DADD+DMUL+2*DFMA instructions, including overhead. No missing count
was imputed. The reference uses eight independent strict RN chains; its FP64
pipe-active metric is72.39%, so it is not claimed to attain absolute peak.

The exported `executed_FLOPs_per_SM_cycle` field divides aggregate GPU FLOPs
by mean SM cycles. It is not an individual SM's instruction rate; that rate
would additionally divide by170 SMs. Same-GPU reference fractions cancel that
constant, so their values are unchanged. Independent recomputation confirmed
all15 trials and their correctness, counter, hash and process receipts.

The inference is that FP64 arithmetic/dependency throughput is a more relevant
limit for these two kernels than bulk DRAM bandwidth. The measurements do not
establish this for every VMEC stage, boundary or GPU. The existing whole-solver
timeline has separate host/launch gaps; a component rate cannot explain them.

## Instruction and scheduling follow-up

Six additional actual Ncu profiles used the same frozen ns99 capture, with
three trials per kernel. All output comparisons were exact, inputs remained
unchanged and all profiler children were reaped without cleanup. These are
component counter measurements, not full-solver timings.

| Median measured counter | Partials | Finish |
| --- | ---: | ---: |
| Eligible warps per active cycle | 0.10 | 0.09 |
| Issue activity, percent | 8.23 | 8.54 |
| Active warps, percent | 24.68 | 13.49 |
| Short scoreboard stalls, percent | 41.17 | 42.69 |
| Fixed-latency wait stalls, percent | 24.81 | 41.47 |
| Long scoreboard stalls, percent | 9.40 | 9.36 |
| Math-pipe throttle stalls, percent | 0.11 | 0.09 |
| Integer fraction of thread instructions, derived | 9.61% | 21.92% |

Stall metrics retain their individual Ncu denominators; do not sum them or
interpret them as fractions of whole-solver time. NVIDIA defines short
scoreboard as a dependency on non-L1TEX MIO work and wait as a fixed-latency
execution dependency. High short scoreboard alone does not prove shared-memory
bank conflicts. See the [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
and [compute triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

Low eligibility and issue activity suggest testing latency hiding and
independent work before another bandwidth rewrite. This is an inference;
source/SASS-level producer attribution is still needed to distinguish the
specific dependencies. Preserve each accumulation's operation order while
testing block geometry, independent outputs and load scheduling. Do not
reassociate a reduction to increase instruction-level parallelism.
Raw reports, six exactness receipts, terminal process records and derived
medians are in `build/optimization/forward-issue-profile1/analysis.json` and
its adjacent evidence manifest.

A separate native py-spy0.4.3 run completed two exact selected-build evaluations
with all children reaped. Of2961 observed stacks containing the actual timed
VMEC wrapper,1839 contained explicit CUDA synchronization paths,88 launch paths
and73 copy/memset paths. Named CPU VMEC leaves also appeared, including Fourier
and magnetic-axis work. Four sampling-lag warnings at100Hz prevent treating
these counts as unbiased percentages or elapsed-time fractions. They do provide
direct stack evidence that calling-thread CPU time includes CUDA waiting; it
cannot all be assigned to CPU arithmetic. The independent audit and complete
counts are `build/optimization/native-cpu-stack1/independent-stack-audit.json`.
A separate20Hz follow-up completed two more exact evaluations, with722 total
samples, zero sampler errors and zero lag warnings. Of349 observed VMEC-wrapper
samples,311 contained explicit CUDA synchronization calls. All34 retrieved
evidence hashes and terminal/reaped state verified. This supports synchronization
reduction as an optimization target; collapsed stacks still lack timestamps
and cannot establish GPU idle time or precise elapsed-time fractions. See
`build/optimization/native-cpu-stack2/root-stack-audit.json`. Both externally
profiled studies remain excluded from clean timing tables, regardless of the
older benchmark report's internal profiled flag.

## Measured launch-layout candidate, not selected

The next isolated experiment transposes the three R/Z/lambda block-group axes
to interleave independent output work. Arithmetic and reduction order are
unchanged. SM80/SM120 compilation preserves all7680 device encoding words when
the option is OFF, and each kernel's FP64 instruction stream remains identical
when ON; SM120 integer/NOP scheduling also changes. The initial assumption that
total candidate code size would match failed and is preserved in the evidence.

On native RTX3090, the warmed follow-up used the unchanged component harness,
400 launches per sample and34 randomized rounds, with the first3 rounds excluded
before execution. All18 output arrays,12 bounds/special fixtures, direct/graph
repeats and rejection guards passed in all three actual harness invocations.

| Measured GPU-event component interval | Grouped median us | Interleaved median us | Median paired ratio,31 pairs |
| --- | ---: | ---: | ---: |
| Direct submission | 87.045202 | 83.731680 | 1.039688x |
| Graph replay | 85.436964 | 82.377195 | 1.037127x |

These are one captured ns99 input and within-process component measurements,
not full-solver speedups or a fixture qualification. The noisier first7-round
screen is retained separately. No gain is multiplied into the selected result.
The candidate remains unselected pending broader checks. Exact source mirrors
passed repository hooks; Bazel was absent, so no full Bazel build/test is claimed.
See `build/optimization/native-forward-layout1/root-audit.json` and
`build/optimization/forward-group-layout1/sass-inspection.json`.

## Current next steps in order

1. Retain the actual six-case/two-repeat flat-solver correctness result and the
   separate focused three-SAN pass. Historical legacy190 failures remain failed.
   The pinned variant also has a six-case/two-repeat exact full matrix; WHILE2
   has two publicP3/P3 exact full calls. The versioned production25 loop also
   passed two publicP3/P3 calls with extended groups and zero drift. Broader
   WHILE/SAN qualification and
   native metric-environment identity
   remain distinct gates before any selection or latency claim.
2. Compare the qualified rewrite directly with the selected original-controller
   GPU build on the same six cases. Use fresh measured pairs and independent
   clock evidence. Do not multiply its ratio by historical GPU or CPU results.
3. Use the completed selected/flat Nsys diagnostics to target actual transfers
   and synchronization. Compare source-defined controller handoffs
   with their trace events. A smaller launch count alone is not a speedup, and
   profiled gaps are not measurements of unprofiled idle time.
4. Collect matched Ncu counters for the kernels that dominate the qualified
   rewrite, with calibration on the same hardware and working-set size. Report
   the relevant memory level, instruction mix, occupancy and dependency stalls;
   a DRAM-only or dense-FP64 ceiling cannot explain every tiny integer scan.
   Whole-solver near-roofline performance remains unproven.
5. Measure independent-solve scheduling separately. The current selected build
   passed 17 actual staged concurrency correctness evaluations, including all
   six cases twice in the final stage, with original CPU arrays and metrics
   exact. These instrumented stages establish no throughput gain. A clean
   sequential-versus-four-native-worker observed-background pilot completed48
   evaluations. Whole-invocation QPC ratios1.021493/1.639494/0.922356 are variable
   and uncontrolled, not a repeatable gain. Existing clock disagreement and
   unresolved foreign activity remain explicit limitations.
6. Use the resulting profiles to choose further fusion or work batching. Retain
   the slower earlier two-slot and pinned-log pilots as negative evidence. The
   new graph carries state and moves Begin/Gather inside capture, but its
   additional scalar kernels can still outweigh removed host work. Measure
   before selecting it or increasing its chunk size.

## Earlier priorities before current local qualification

1. Keep the qualified [thermal comparison](THERMAL_ORDERED.md) separate: its
   72-call A100 experiment preserves all original outputs, with VMEC median
   ratios1.039824-1.059868x. Two per-case intervals include no gain and consistent
   fresh-call gains remain unproven. The selected build is retained. Mapped
   readback still needs sanitizer/fault/lifecycle and clean-timing gates after
   its exact twelve-call matrix. Do not compound historical ratios.
2. Use the actual magnetic scan/seed counters above to test smaller scan tiles
   and matched tiny-kernel controls. Separate profile-observed aggregate traffic
   from useful bytes. Trace gaps alone cannot identify a bandwidth ceiling.
3. Qualify the separately compiled Assemble/Verify fusion and three scan tiles
   with the frozen procedural54 whole-prefix/11safe-alias harness, sanitizers
   and controlled component timing on a cleared or leased GPU. CPU build/index
   proofs are not GPU qualification. No new captured-data transfer is implied.
   Preserve the unkeyed-scan negative/provisional timing finding above.
4. Investigate the original CPU-controller dependency and recorded failures
   before another device-control rewrite. Prior scan-input/controller fusions
   were slower; fewer launches alone prove no gain. CTA-private repeated seeding
   is a separate source design, not implemented or measured. Keep integer Map
   composition and the complete finite recurrence certificate intact.
5. Measure independent-solve scheduling as throughput separately from latency.
   The completed native batch pilot improved throughput 1.291x but slowed each
   request. Current-selected batching has CPU preparation only; it needs its
   own concurrency/array/lifetime gates. Do not multiply historical throughput
   into the qualified latency table.

Literature check: CUB already implements single-pass decoupled look-back; a
generic replacement with the same algorithm is not a demonstrated innovation.
Its floating-point scan result can vary, so only a speculative guess may use
this path; accepted output must still pass the unchanged exact certificate.
See [NVIDIA DeviceScan documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html).
This check does not upgrade the pinned local library or establish a speedup.

## Earlier proposed order (superseded)

1. Integrate the ordinary two-iteration graph into the original CPU control
   path, then qualify whole trajectories, all fixtures and paired native wall
   time. Its captured real component passed direct/capture/replay and all three
   focused sanitizers with 128 arrays bitwise exact. This does not qualify a
   full equilibrium. The earlier GateLive fusion was slower in a corrective
   native screen and remains unselected.
2. Prepare a current frozen pilot on a GPU with stronger native FP64 throughput.
   Run correctness, matching calibration and actual whole solves before making
   a hardware speedup claim. Rent only after the payload and bounded retrieval/
   teardown procedure are ready. No new lease was used for this measurement.
3. Continue the measured SeedSegments and radial-solve dependency work. Their
   integer/certificate/division behavior is not described by this FP64 reference.
4. Use the instruction/stall evidence above to test independent work and
   scheduling without changing arithmetic order. Conditional device-launch
   experiments retain their failing sanitizer evidence; the ordinary graph
   path avoids that mechanism but still needs full-solver validation.
5. Re-profile the winning full solver and evaluate independent-solve batching
   separately as throughput. Do not multiply historical incremental ratios.

FP64 emulation or reduced precision remains an alternative configuration until
its own output-identity and accuracy evidence exists. It is excluded from the
exact-parity timing table.

## FP64 emulation literature follow-up, 2026-11-05

This is a literature finding, not an executed VMEC experiment. The current
strict-FP64 build and pinned oracle are unchanged.

NVIDIA now documents fixed-point FP64 GEMM emulation using Ozaki-I/II on compute
capabilities 8.x through 12.x, starting with CUDA13.0 Update2. Dynamic precision
selects a mantissa size and can fall back to native FP64. The documented accuracy
target does not promise identity with VMEC's original ordered operations.
Explicit per-handle workspace is required for the library's reproducibility
guarantee; workspace allocation can otherwise trigger fallback. Any trial must
use a separate toolchain because the current experiment uses CUDA12.9.
Source: [cuBLAS fixed-point emulation and reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#fixed-point).

The Ozaki-II paper reports measured GEMM acceleration on RTX4090 and GH200
using integer Tensor Cores and modular reconstruction. These are the authors'
matrix-product results, not our hardware measurements or VMEC speedups.
Source: [Ozaki Scheme II, revised September2026](https://arxiv.org/abs/2504.08009).

A newer deconstruction-aware study identifies operand conversion, integer-pipe
work and residue-plane traffic as additional costs. Its abstract explicitly
states that all results are model projections pending measurement; its proposed
hardware assist is not an available software optimization for this workload.
Source: [Ozaki2.5, September2026](https://arxiv.org/abs/2609.09095).

Inference for this solver: first test only a frozen Fourier matrix-product
component with its actual small shapes, conversion and reconstruction included.
Constant basis data might amortize conversion; changing geometry data cannot be
assumed reusable. Compare against the selected N3 kernel, not a dense-GEMM peak.
This does not replace the dependent tridiagonal scan or certify the original
rounding order. Keep any changed result in an alternative configuration, with
per-metric drift, repeatability and full-equilibrium behavior measured before a
stellarator-design accuracy claim. No emulation speedup is estimated here.

## Native control-loop regression, 2026-11-05

An untraced eight-call P3 screen on the native RTX3090 completed with exact
outputs. After excluding one warm pair, three alternating measured pairs gave
selected VMEC times8.038381/8.015067/8.013296s versus versioned25 times
9.183994/9.191487/9.195661s. The median paired ratio is0.872010x: versioned25
is14.6776% slower. It remains unselected. Counts of669 returned chunks versus
the earlier7210 are structural evidence, not performance gains. Saved exact
timing metadata and independent arithmetic review are under
`build/optimization/native-private-v2-screen1-actual/`.

The earlier local component node capture identified348 inner-body records over
three bodies plus6 entry/gather records. Its JacobianExtrema samples were
5.302us,5.145us and2436.204us; the cause of that outlier is unproven. These three
instrumented bodies cannot establish native full-solver costs or a roofline
percentage. Source inspection identifies added scalar and speculative restart
sites, but matched native traces are required to attribute the regression.

Two actual matched whole-call traces then completed on the native RTX3090
(root30260, configdf4e30fd), both with exact outputs and retired targets. The
recorded graph counts/durations were27,477/6.806831s selected and1,763/8.263201s
versioned25. Selected's graph count also matches its actual activity getter.
Recorded GPU memcpy durations were81.196ms and34.704ms respectively; the1,446
11376B packet copies themselves took2.763302ms. Their count matches the actual
route formula669 + 2*108 +561. These are instrumented observations, not projected
speedups or additive wall-time components.

Both reports warn that not all CUDA events might have been collected and say
hardware tracing is unsupported, with legacy software tracing used instead.
Thus complete coverage and hidden graph-body idle time remain unproven. The
recorded regression points inside graph execution; this mode cannot separate
guarded math/control kernels from conditional scheduling gaps. CPU memcpy API
time includes queued GPU waiting and is not recoverable transfer cost. Reports,
SQLite exports, diagnostics and root analysis are preserved under
`build/optimization/native-private-v2-profile2-actual/`.

The subsequent native node:host-only probe also completed both full solves
exactly (root20845, confige283d0be), but FAILED whole-body visibility: selected
has1,170,370 kernel records, while V2 records only3,013 BeginSlot and3,014 Finish
entries against13,192 actual Update attempts. Final-grid ForwardPartials records
are9,771 selected versus400 V2. These partial V2 records cannot measure the hot
final stage or resolve its internal gaps. Reports/SQLite/witnesses are retained
under `build/optimization/native-private-v2-profile3-actual/`.

The selected trace supports a narrower per-shape observation. Independent
read-only SQLite audit47ee25c3 binds PID526094 and database6a45cd8d. The actual
ForwardPartials symbol has exactly three ordered grid runs,47x3 then96x3 then
186x3, mapped to the known ns25/51/99 shapes. Times below are profile-relative
seconds. The window starts at the first ForwardPartials start and ends at the
last same-grid ForwardPartials end; it is not a complete stage boundary.

| ns | ForwardPartials count | First start s | Last end s | Anchor span s | Observed kernel sum s | Observed kernel union s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1994 | 0.086285355 | 1.737124031 | 1.650838676 | 0.576659583 | 0.576659391 |
| 51 | 1995 | 1.780803538 | 3.763385927 | 1.982582389 | 0.727755242 | 0.727728138 |
| 99 | 9771 | 3.835629352 | 15.386056423 | 11.550427071 | 5.348410504 | 5.345126915 |

ForwardPartials itself sums34.885270/58.055113/512.420905ms in these runs.
Within the ns99 anchor, the largest observed named groups include
FourierToRealKernelSeparable514.857825ms, ForwardPartials512.420905ms,
ResidentMhdKernel353.196406ms, ToroidalPartialsKernel317.962055ms and
ForwardFinish310.048385ms. These are instrumented kernel sums, not routine
speedups or a full-solver cost model. The anchor sums/unions clip intersecting
kernel intervals; one ns99 boundary-crossing record is clipped. Work before,
after and between anchors remains unassigned. Gaps are included in spans;
subtracting kernel sums does not establish unprofiled idle or CPU stage wall.
Both trace warnings and incomplete V2 body coverage remain in force.
[Actual ordered runs, timestamps and kernel rankings](../../build/optimization/native-private-v2-profile3-actual/analysis/selected-stage-anchors-independent.json).

The isolated A-D adjacent metadata fusion backend (38aa7dd5) ACTUALLY completed
an untraced native publicP3/P3 screen, with all8 saved canonical exact-output and
route gates passing and all owned targets cleanly reaped. After one excluded
warm pair, selected VMEC RAW durations were8.046097089/8.029229492/8.074986836s
versus fusion9.210202983/9.138509101/9.156541193s. The three paired slowdowns
were14.46796/13.81552/13.39388%, median13.8155%; it is rejected and unselected.
Independent metadata audit5ec74742 verifies all five clock comparisons. This is
one-fixture screen evidence, not full SAN/edge qualification or a roofline gain.
Separate old-V2 runs cannot establish an incremental fusion gain. Actual saved
metadata is in `build/optimization/native-private-fusion-screen1-actual/`.

## Exact-division verifier: measured component follow-up

On RTX5090, a verifier-only residual-interval and signed-zero certificate
passed all original prefix/force/poison/adversarial controls on ns25/51/99 and
all three sanitizers. Twelve fresh untraced component calls (two excluded
warmups, five alternating measured pairs) produced CURRENT/ON ratios with
median1.034688x, range1.026057-1.046858. This is a component result, not a
whole-solver gain; production integration and full-solver qualification remain
pending. Physics, proposal arithmetic and the original division fallback stayed
fixed in this verifier-only study.

The new actual Verify capture has337 blocks x128 threads and38 registers.
DFMA count37754 matches the nonzero guard rows;4970 exact-zero rows take the
earlier integer branch. The historical same-geometry CURRENT capture has
338828 DFMA,89978 DMUL and42284 DADD, versus37754/42284/42284 in the new one.
Integer instructions rise from3196412 to4481006. These are measured counters,
not proportional speedup estimates. Profile duration5.152us versus5.312us is
not paired unprofiled timing; the observed DRAM traffic also differs,0 versus
3012608 bytes. Saved CURRENT reuse is explicit. Independent audit rehashed
3085 source/input bindings and checked all12 consumed reports and retirement
witnesses; raw GPU force payloads were not separately archived for a second
byte comparison. [Audit and scope](../../build/optimization/exact-division-gpu2-followup1-independent1/actual-seal.json).

Consumer FP64 issue ceilings are hardware-specific. RTX3090/5090 results do
not establish A100 utilization or predict whether extra integer certification
is profitable there. A matched current-candidate A100 study remains required.

## Reproduction and evidence

`roofline_matched.cu` is a standalone calibration and never calculates an
equilibrium. Its strict-compute outputs are checked bitwise against CPU RN
results; every memory-copy word is checked. Normal compute and three memory
sizes passed, plus focused compute/memory memcheck. Each profiled Fourier run
also retained its CPU/original-GPU exactness checks. All15 actual profiled jobs
and their exports exited0; frozen sources/binaries stayed unchanged and owned
processes were reaped. No physical clocks, driver settings or physics knobs
were changed.

Actual commands, tool/source/binary hashes, pre/post telemetry, profiler reports,
raw exports, every sample and recomputed fractions are in
`build/optimization/roofline-matched1/`. The main files are `analysis.json`,
`kernel-samples.csv`, `execution1/status.json` and `execution1/postflight.json`.
The input is the frozen `forward-prototype/real-forward.bin`; this is one
capture, not the three-boundary benchmark distribution. Profiler trials remain
outside the main unprofiled wall-time table.
