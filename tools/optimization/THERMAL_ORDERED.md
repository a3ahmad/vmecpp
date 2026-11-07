# Qualified ordered-thermal experiment

This is a separate incremental comparison against the qualified GPU build.
It does not replace the original CPU baseline or establish a consistent
fresh-call wall-time gain. No physics, resolution, tolerance, multigrid,
iteration limit, nstep, current profile or single-thread setting changed.

The change skips the generic thermal prefix and retains the original ordered
thermal reduction. Thermal terms are not elided. The magnetic reduction is
unchanged. The source change is opt-in at compile time:
`VMECPP_LATTICE_ORDERED_THERMAL`.

| Boundary | Track | Selected VMEC s | Thermal VMEC s | Ratio | Max relative metric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Third frozen fixture | P1 | 5.103800 | 4.891347 | 1.043435x | 0 |
| Third frozen fixture | P3 | 5.080559 | 4.825670 | 1.052819x | 0 |
| public_p2_seed | P1 | 5.918169 | 5.621506 | 1.052773x | 0 |
| public_p2_seed | P3 | 5.916769 | 5.582556 | 1.059868x | 0 |
| public_p3_seed | P1 | 6.684929 | 6.363203 | 1.050560x | 0 |
| public_p3_seed | P3 | 6.648074 | 6.393461 | 1.039824x | 0 |

Per-call durations were measured on EPYC7662/A100 PCIE; medians and ratios are
derived. Each case has one excluded warmup pair and five alternating measured
pairs. Four paired VMEC confidence intervals favor thermal; third/P3 and
publicP2/P3 include no gain. Keep all five pairs and the full uncertainty.
Adapter-body medians improve less; two fresh-call medians regress. No uniform
end-to-end improvement, batch throughput or roofline efficiency is claimed.

The 72 main evaluations and two separate traced qualifications passed exact
original identities, repeats, 7,400 physical NC variables and 3,700 diagnostic
fields. All 852 numeric metric cells have zero drift and 36 nulls are preserved.
Both qualification CPU trajectories match all 13,780 events; main timing calls
have no trace, observer or profiler. The lease was destroyed after retrieval.

The existing build recipe is [README.md](README.md). Exact executed thermal
backend argv are in `build/optimization/thermal-ordered1/commands.json` (key
`on`); strict options include `--fmad=false`, SM80/SM120 and unchanged Thomas
prefetch8/8, SparseSeed and N3 dependencies. The actual source is
`thermal-ordered1/source/gpu_lattice_energies.cu`, SHA256
`923ea18af76a1f6eaddf6f5c7c1638db0245f94f8f0a134179440f3f32526c24`.
The qualified backend SHA256 is
`a0c106880ff57f26d99c2962cb1264de73cc0e625e8f892dd4e48bcb97b474fc`.

Exact shim link argv are in `resident-solver-ordered-thermal1/commands.json`;
the original CPU controller and whole144e Python overlay remain mandatory.
The nine runtime flags and original dependencies are frozen in
`clean-linux-variants1/root-release1/payload/variants-plan.json`. The approved
build/configuration archive is `root-release1/approved-thermal.tar.gz` under
that experiment, SHA256
`84a61aaf3d915860db48b45897a362ee029a518abe9a65bf5a0b030a6b89e3b5`.
Relocate copies only, with unchanged text/fatbin proofs; retain the originals.
Do not replay recorded output paths over frozen artifacts or install into the
original oracle. A rebuilt or relocated binary needs its own exactness checks.

[All timer scopes, confidence intervals and each measured pair](../../build/optimization/a100-thermal-compare1-independent-audit1/FINDING.txt),
[per-metric original comparisons for every call](../../build/optimization/a100-thermal-compare1-independent-audit1/per-metric-original-drift-all74.csv),
[independent physical-array and trajectory audit](../../build/optimization/a100-thermal-compare1-independent-audit1/audit.json).

Profiling evidence motivating this isolated reduction change and subsequent
fusion/scan-policy work is in [ROOFLINE.md](ROOFLINE.md). No historical speed
ratios are multiplied into this table. The prior selected build is retained.
