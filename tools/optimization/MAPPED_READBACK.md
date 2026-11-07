# Experimental mapped scalar readback

Status: the actual 12-evaluation correctness matrix passed: three frozen
boundaries x P1/P3 x two repeats. Timing, sanitizer and fault/lifecycle
qualification remain pending. The selected qualified build is unchanged.

The transport publishes16 Jacobian bytes and112 scalar-packet bytes into mapped
host storage, with independent phase counters and system release/acquire. It
retains same-stream CUDA completion/error checks, original CPU control and all
physics settings. Private cold+graph scratch initialization settles runtime
flags before exact domain checks and realReady0. Failed retirement quarantines
storage and blocks subsequent candidate GPU work in that process.

Measured outcomes from two executed evaluations:12metrics with absolute and
relative drift0, original request/metric identity and decks exact, full CPU
trajectory exact,100rawNC variables and50saved actualdiagnostic fields exact.
Mapped callbacks naturally differ:13761Jacobian versus13760scalar handoffs,
with every context accounted. All282frozen inputs stayed unchanged; processes
were reaped without cleanup actions. Legacy NPZ10fields match the original;
all23newselected/mapped fields also match. Metadata schema expansion is not
reported metric drift. No per-step private coefficient-reference proof is
claimed. All elapsed values in these traced runs are diagnostic observations.

Evidence in build/optimization:
- resident-solver-mapped2/runs3/gpu-first-gate1/validation.json
- mapped2-firstgate-independent1/actual-two-physics-audit.json
- root-first-solver-report1/per-metric-drift.csv
- resident-solver-mapped2/{source,build-commands.json,offline-proof2.json}
- mapped-phase-component3/normal-independent-audit.json

The subsequent matrix preserved all 138 numeric metric cells with absolute and
relative drift zero, six nulls, original identities and decks, 1,200 raw NC and
600 diagnostic fields. All six repeat trajectory/array pairs were exact. The
original CPU trajectory anchor was checked for public-P3/P3 in both repeats;
no original CPU trajectory anchor is claimed for other cases. All 88 context
records passed routing/counter checks. All 340 frozen inputs were unchanged,
and all 14 recorded owner/worker generations retired without cleanup actions.

Matrix evidence in build/optimization:
- resident-solver-mapped-qualification1/runs1/matrix/validation.json
- resident-solver-mapped-qualification1/runs1/launch/terminal.json
- mapped-matrix-independent1/actual-matrix-audit.json
- mapped-matrix-independent1/final-receipt.json

Fault/poison controls, lifecycle coverage, sanitizers and clean controlled timing
are required before promotion. No physics knob, precision or thread setting
changed. Traced correctness runs do not establish a new speedup.
