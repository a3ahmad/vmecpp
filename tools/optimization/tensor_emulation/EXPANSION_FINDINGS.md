# FP16/BF16 expansion findings

On the measured actual VMEC contraction exports, no expansion variant beats its
same-run native DGEMM or flat ordered FP64 control on RTX5090 / SM120. Register
MMA, cached operands and fused A splitting substantially reduce expansion cost,
but do not reverse that result. Tiny synthetic timing ratios are not evidence
of an actual-contraction speedup.

The final snapshot has **18 suites, 1,214 mode reports, 607/607 byte-identical
repeat comparisons, and 1,214 finite outputs**. Exact Fraction checks cover
2,464 selected case-occurrence dots and 38,848 arm checks. Actual exports have
894 reports, repeated-row proxies 192, and public synthetic stress inputs 128.
These are not 1,214 independent solves. No expansion output matches supplied C
bit-for-bit on actual exports or proxies; 42 synthetic expansion reports do.
[Inventory and source receipts](results/expansion-completed-inventory.csv) link
the per-suite accuracy CSVs: maxabs, relative-L2, worst relative error excluding
zero references, zero counts/errors, cancellation-scaled errors, repeat hashes,
range/fallback counts, precision and execution flags. The redundant combined
accuracy CSV is retained only in the ignored build directory.

## Six-component FP16 near-roundoff candidate

This explicit accuracy-oriented comparison uses final `fused8`, not the fastest
lower-accuracy BF16 point. Times are microseconds per contraction: the median of
two run-level steady medians. B is reused; graph batch 32 amortizes submission.

| Contraction | Digit bits | FP16 q6 us | Native / ordered us | Maxabs vs C | Relative-L2 vs C |
|---|---:|---:|---:|---:|---:|
| K11 | 10 | 7.597 | 4.708 / 2.190 | 6.66E-016 | 1.71E-016 |
| K10 | 10 | 7.537 | 2.987 / 2.158 | 4.44E-016 | 1.24E-016 |
| K48 | 9 | 25.209 | 5.353 / 4.093 | 7.16E-016 | 6.83E-015 |

K11 is inverse toroidal, K10 inverse main, K48 forward finish. Fused A splitting
is active only for K<=16; K48 uses the nonfused fallback. The
[fastest q4/q6/q8 table](results/expansion-final8-fastest-by-depth.csv) is a
separate tradeoff, not an equal-accuracy recommendation. Full products versus
triangular products and graph batch 1 versus 32 change the scope; separate-run
comparisons are descriptive rather than randomized causal estimates.

Fusion matches recurrence-based cached outputs in **72/72 complete-array
comparisons** at matched arithmetic settings. The separate quantized splitter
matches only 14/48 (34 differ, up to 1.83e-9); do not claim its numerical identity.
[Comparison hashes](results/expansion-fusion-byte-comparison.csv) preserve both.
Stress tests show why finite depth is not general FP64 emulation: balanced
exponent ranges can lose the entire result (maxabs 16, relative-L2=1) even with
zero range/fallback counters. Cancellation differences versus sequential C may
instead reflect reassociation; selected exact-real dots separate those cases.
Subnormal differences reach 2e-323. Exact FP32 subdots do not bound finite-q loss,
omitted products, scaling or reconstruction.

## Actual dispatch and remaining bottleneck

The [saved profiler summary](results/expansion-profile-summary.json) proves
25,632,768 FP16 tensor operations for fused q6 inverse-main, and BF16=0; the
BF16 b9 control reverses these counts. FP16 achieved 3.66% occupancy (1.76 active
warps/SM), 72 registers/thread, 1.34% DRAM throughput and 51.84% short-scoreboard
stall cycles. Composite SM compute 42.80% is not FP64/tensor roofline efficiency.
One warmed Nsys observation captures a single 7.388us fused kernel from FP64 A
load through split, tensor products and FP64 output; B presplitting is excluded.
These profiled times are not benchmark ratios. No instruction-level cause is
claimed without PC attribution. No complete expansion transform or solver
acceptance follows from these standalone components.

Method: [implementation and timing scope](EXPANSION.md),
[CPU reader](expansion_accuracy.py), [q6 detailed metrics](results/expansion-final8-fp16-q6.csv).
Primary background: [FP16 Ozaki construction](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/)
and [BF16 decomposition](https://arxiv.org/abs/1904.06376); their guarantees are
not transferred to this finite-depth triangular implementation.
