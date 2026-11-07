# Exact-parity energy batching: native incremental result

The clean ragged-energy build is preferred for this frozen native six-case
cohort: **1.0296-1.0338x VMEC paired median speedup** over the current selected
GPU build, with **zero-tolerance metric and equilibrium drift**. All 30 measured
VMEC pairs improved. This GPU-versus-GPU result is separate from the original
CPU comparison in [SUMMARY.md](SUMMARY.md); do not compound the ratios.

Measured on Threadripper 2950X / RTX3090, native Linux, using the pinned private
scientific environment and one VMEC/BLAS thread. Each of three boundaries was
tested on P1 and P3: one excluded warm pair plus five balanced measured pairs,
72 fresh calls total. Times below are median RAW-clock VMEC-function brackets.
Speedup is the median of five paired ratios, not the ratio of duration medians.

| Boundary | Track | Selected GPU s | Batched GPU s | Paired speedup | Metric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| public_p3_seed | P1 | 7.995592 | 7.748066 | 1.031947x | 0 |
| public_p3_seed | P3 | 8.005598 | 7.752247 | 1.033510x | 0 |
| public_p2_seed | P1 | 7.098066 | 6.876844 | 1.032169x | 0 |
| public_p2_seed | P3 | 7.111118 | 6.894328 | 1.031794x | 0 |
| Third frozen boundary | P1 | 6.005430 | 5.809105 | 1.033796x | 0 |
| Third frozen boundary | P3 | 5.983309 | 5.811357 | 1.029589x | 0 |

The [native per-case, per-metric CSV](energy-batching-native-metric-drift.csv)
was derived from all 72 actual native reports against six frozen original 3358
reports: **828 numeric observations have absolute/relative drift 0; 36 nulls stay
null**. Null-only rows are NA. The independent audit read 72 whole original-exact
NetCDF files (7200 numeric fields), 3600 actual diagnostic NPZ payload/dtype/shape
fields, requests and decks. All 60 within-build repeats and 36 cross-build pairs
were exact. Positive selected3 and Ragged4 counters, actual backend/provider
owners, canonical FP controls, clock PID/TID and saved retirement receipts passed.
Sticky FP status bits were retained, not cleared. No CPU trajectory was captured
in this native timing study; available trajectories passed separate local gates.

The separate thermal and magnetic prefix pipelines share one ragged batch.
Original ordered recurrences, complete bitwise certification and fallback remain.
Physics, modes, precision, tolerances, multigrid schedule, iteration limits,
nstep, single-thread settings and CPU convergence/damping are unchanged. Thermal
work was retained. No profiler ran during timing. Benchmark/submission artifacts
and the installed oracle were not modified; no withheld native data was used.

Adapter-body ratios are **1.0135-1.0234x**, with 30/30 measured pairs positive.
Ragged4 is read outside VMEC timing but inside the candidate adapter bracket.
RAW, MONO, epoch and CPU clocks remain separate; paired RAW/MONO ratios agree
closely here, without establishing calibration or continuous quietness. Each
case's descriptive 95% median interval, enumerating all 5^5 resamples, exceeds 1;
five pairs remain a sparse sample. No universal gain, whole-solver roofline
fraction, near-roofline result or combined-optimization gain is established.

Build: the [clean fresh-output seven-TU recipe](../../build/optimization/resident-solver-energy-ragged-clean1/README.md)
produced backend be306 without old objects or failure logs. Its
[106-stream proof](../../build/optimization/resident-solver-energy-ragged-clean1/proof1/proof.json)
matches encoded GPU instructions, resources and constants to measured d184.
Native backend 8508 is a DT_NEEDED/SONAME/RUNPATH-only rehome of be306, with unchanged
program/data sections, complete fatbins and symbol semantics; see the
[native staging proof](../../build/optimization/native-energy-ragged-stage1/carrier1/relocation.json).
The clean build also passed the separate
[six-case, two-repeat qualification](../../build/optimization/resident-solver-energy-ragged-clean-matrix-independent1/actual-audit.json).
An earlier statistics getter crash was diagnosed and fixed as a retired-stream
read issue; its failed runs remain preserved rather than counted as passes.

Native evidence: [table, intervals and limitations](../../build/optimization/native-energy-ragged-paired-analysis1/RESULT.md),
[72-call physical/clock audit](../../build/optimization/native-energy-ragged-paired-analysis1/physical-audit1.json)
and [sealed receipt](../../build/optimization/native-energy-ragged-paired-analysis1/seal.json).
The earlier Core Ultra 9 285K / RTX5090 WSL2 study remains separate: measured d184
VMEC ratios 1.0184-1.0392x, 29/30 positive, adapter 1.0037-1.0296x, with a material
RAW/MONO discrepancy and one interval including no gain. Its
[local table and clock caveats](../../build/optimization/resident-solver-energy-ragged-paired-analysis1/RESULT.md)
and [local metric CSV](energy-batching-metric-drift.csv) are not replaced or pooled
with the native study. Neither study changes the original CPU comparison table.
