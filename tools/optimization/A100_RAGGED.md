# A100 CLEAN ragged-energy pilot

This executed single-boundary P3 pilot compares selected3937 with CLEAN
ragged-energy be306 on an A100-SXM4-40GB, AMD EPYC7713, driver590.48.01.
It is a GPU-versus-GPU comparison. It does not compare against a timed CPU
baseline, compound earlier gains, or extend the native six-case result.

Fourteen actual evaluations ran: two traced candidate qualifications, then
12 untraced calls comprising one excluded warm pair and five alternating
measured pairs. All original modes, grids, tolerances, iteration caps, precision
and single-thread settings stayed unchanged.
The actual deck has mpol10/ntor10, ntheta26/nzeta24, ns25/51/99,
niter2000/2000/40000, ftol1e-17/1e-17/1e-15, nstep250, nfp3,
lasym=false and ncurr1. This matches its original-build authority exactly.

| Measured scope / clock | Selected median s | CLEAN median s | Paired speed ratio | Descriptive 95% interval |
| --- | ---: | ---: | ---: | --- |
| VMEC call, RAW lower bracket | 6.525313 | 6.192948 | 1.052811 | [1.046450,1.058471] |
| Post-import adapter body, MONO | 10.480882 | 10.173857 | 1.027151 | [1.016331,1.057531] |
| Outer oracle fresh call, MONO | 16.143314 | 15.840144 | 1.022274 | [0.993638,1.038353] |
| Fresh benchmark call, RAW lower bracket | 16.368995 | 16.090962 | 1.022014 | [0.993139,1.038433] |
| Fresh worker process, RAW | 17.766426 | 17.370553 | 1.023725 | [0.989279,1.041393] |

All table values are measured or derived from executed pairs. Ratios are medians
of paired ratios, not ratios of the separately displayed marginal medians.
Intervals are the retained small-sample bootstrap summaries; no generalized
confidence or controlled-load qualification is claimed. RAW and MONO scopes
remain separate. The fifth fresh-process pair was slower for CLEAN and is
retained. Consistent full fresh-call improvement remains unproven.

Independent post-retrieval checks passed all14 canonical whole NC files,
1400 numeric NC names,700 actual NPY fields and168 numeric metric observations,
plus request/deck/boundary identities and repeats. Maximum relative drift is0
for every metric. Both qualification traces match the original3358 CPU anchor
under the existing initial undefined-residual policy. Typed routes, provider,
source/archive bindings and saved ownership/retirement checks passed. FP-control
guard success is source-enforced; per-call full FP status words were not saved.
See the [independent audit](../../build/optimization/a100-whole-p3-live3-independent1/actual-seal.json)
and [per-pair/per-metric CSV](a100-ragged-per-metric-drift.csv), which includes
the validated warm pair for numerical checks only.

This resolves actual guest qualification of the current CLEAN build on one
fixture. It establishes neither a full A100 fixture matrix nor near-roofline
performance. No hardware counters were collected. The earlier two failed leases
remain failed: one stopped at an unnecessary profiler-capability prerequisite;
the next stopped because saved captured provenance was sent through a live-run
gate. The isolated fixes preserved capability observations, all ordinary live
identity checks and the original saved data. No metric was rounded or replaced.

The lease was destroyed after final retrieval. All 433 retrieved files and 434
regular archive members were verified; all 14 workers and the supervisor retired
cleanly.
The oracle installation, benchmark originals and submission artifacts were
untouched. Source, release and bounded execution recipes are frozen under
`build/optimization/a100-clean-ragged-equilibrium3`,
`a100-clean-ragged-release3-root1` and `a100-clean-ragged-pipeline3`.

Actual evidence: [five-pair timing summary](../../build/optimization/a100-whole-p3-live3/upload1/controller/snapshot-0004/results/analysis/summary.json),
[final retrieval and teardown](../../build/optimization/a100-whole-p3-live3/upload1/controller/controller-state.json)
and [lease closure](../../build/optimization/a100-whole-p3-live3/lease.json).
