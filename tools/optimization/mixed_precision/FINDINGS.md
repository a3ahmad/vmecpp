# Mixed-precision whole-solver pilot

**No candidate demonstrated both valid baseline identity and repeatable speedup.**
The retained FP64 solver remains the result. This was one public boundary on
track P3, with two candidate repetitions per policy; no broader three-boundary
P1/P3 qualification was run because the pilots did not justify it.

The experiment changed inverse/forward transform precision and its switching
policy. Original decks, fidelity, multigrid stages, tolerances, iteration limits
and thread settings were preserved. All 16 saved actual input decks were
byte-identical to the original reference. Each B/C/C/B cohort used fresh owned
processes; every child was reaped without cleanup actions. No user benchmark
inputs or original reference artifacts were modified.

| Candidate policy | Successful candidates | ns99 force calls | Max relative metric drift | QPC baseline/candidate pairs |
|---|---:|---:|---:|---:|
| New shim/resident, FP64 control | 2/2 | 9,771 | 0, exact | control validation only |
| Continuous FP32 | 0/2 | 39,965 | No final metrics | Not a successful-solve comparison |
| First 1,000 calls/stage FP32, then FP64 | 2/2 | 25,155 | QI 1.34749% | 0.58166 / 0.69016 |
| First 25 calls/stage FP32, then FP64 | 2/2 | 9,771 | Gradient scale 25.344 ppm | 0.93863 / 1.07877 |

A ratio above one favors the candidate. Windows QPC marker intervals measure a
cold owned process including imports, VMEC, full CGA metrics, persistence and
retirement, plus unsubtracted delivery jitter. They are not VMEC-only times.
Warm25 intervals were B22.304/C23.763/C24.264/B26.175 seconds: mixed results,
not repeatable speedup. Linux RAW/MONO/epoch clocks disagree and are retained
only as diagnostics. Two repetitions provide no robust confidence interval.
[All process intervals and ratios](results/qpc-pairs.csv).

## Accuracy and convergence

The new FP64 control matched all 12 metrics, all 100 numeric NetCDF fields and
all 146 saved numeric model/diagnostic fields in each call. Both mixed policies
passed the independent terminal FP64 force-tolerance certificate, yet changed
physical outputs and metrics. Certification is not output identity.

Warm25 used only 75 FP32 force calls out of 13,761. It still changed 10/12
metrics: gradient-scale relative drift 2.53444e-5, vacuum well 1.28373e-5 and
QI 1.09168e-6. Spectral-field relative-L2 errors were 6.12e-8 for B, 4.35e-8
for R, 1.31e-7 for Z and 9.52e-9 for full-grid iota. Warm1000 required 29,145
force calls versus 13,761, so added convergence work outweighed any cheaper
transform work. Its physical and diagnostic errors were materially larger.

All three successful candidate policies repeated their metrics and all 146
saved numeric arrays bit-for-bit. Deterministic drift remains drift. Continuous
FP32 failed with SOLVER_NONCONVERGENCE and produced no final metrics/arrays;
its last materialization checkpoint is not evidence of a terminal force floor.
[All 12 metric drifts](results/metric-maxima.csv),
[physical/model errors](results/model-array-errors.csv),
[stage/kernel counts](results/stage-counters.csv),
[certificates](results/terminal-certificates.csv),
[repeat checks](results/repeat-identity.csv).

Separate transform component error measurements around 1e-7 to 1e-4 are scoped
to those component tests; they neither bound solver drift nor establish a
whole-solver speedup. The pilot was stopped without promoting a mixed policy.
It does not rule out every possible mixed-precision algorithm or other fixture.
Runtime sources and reproduction instructions are in [README.md](README.md);
[sanitized provenance](results/sources.json) records executed artifact hashes.
Figures are saved only under ignored build output, not in the repository.

An earlier control attempt with per-iteration tracing was stopped and excluded
from these completed cohorts. Its incomplete outputs and cleanup record remain
preserved. No timing or scientific result is claimed for that attempt.
