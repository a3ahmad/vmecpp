# CUDA FP64 emulation on actual VMEC contractions

The completed experiment finds a real size/accuracy crossover for GEMM, but
**no demonstrated original-Fourier or whole-solver speedup**. All configurations
are alternatives: even native cuBLAS GEMM changes the original ordered arithmetic.
None of the360 complete output arrays matched its original ordered reference
bitwise. No CGA metrics, iteration trajectories or equilibrium outputs were
qualified here.

CUDA13.5.0 and cuBLAS13.10.0.20 were installed alongside the preserved toolkits.
The comparison used separate pinned CUDA13.1/cuBLAS13.2.0.9 and13.5 processes
on the RTX5090 (GB202). Existing12.9/13.1 SDKs and manual default13.1 remain.
Library/binary ownership checks passed in the execution receipts.

## Measured timing and accuracy

Four completed suites cover thin original-shape contractions, full2D dense
diagnostics, and dynamic/fixed mantissa requests. There are360 mode reports
and180/180 repeated full-output byte comparisons passed. This is repeatability
within each case/version/mode, not original evaluation identity.

The tables below use medians of two executed process repetitions, in
microseconds. Timing is steady CUDA-event **library-call** time including
internal conversion/reconstruction, excluding external matrix packing, basis
construction and solver integration. Modes ran in fixed sequential order;
these are descriptive measurements, not randomized paired confidence bounds.
Errors compare against original ordered FP64 C, not solver metrics.

Thin inverse toroidal contraction, M982,N24,K11:

| Requested mode | CUDA13.1 us | CUDA13.5 us |13.5 max absolute error |
|---|---:|---:|---:|
| Native pedantic FP64 |17.070 |18.876 |2.22e-16 |
| Dynamic eager |120.479 |208.188 |2.22e-16 |
| Dynamic performant |17.043 |14.005 |2.22e-16 |
| Reduced offset -12 |120.951 |293.046 |6.66e-16 |

Dynamic eager was slower than native in all five thin screened cases. The
requested performant output matched native bytes across the tested cases;
its timing advantage is not evidence of tensor dispatch. Every tested fixed
mantissa40..80 request was slower than native in the four thin fixed cases.

Largest full2D split-output diagnostic, M12544,N336,K220:

| Requested mode | CUDA13.1 us | CUDA13.5 us |13.5 native/mode ratio |13.5 max absolute error |
|---|---:|---:|---:|---:|
| Native pedantic FP64 |1282.591 |1280.850 |1.000 |2.44e-15 |
| Fixed80 bits |1158.312 |579.454 |2.210 |8.88e-16 |
| Fixed64 bits |890.391 |384.590 |3.331 |8.88e-16 |
| Fixed55 bits |657.636 |272.804 |4.695 |8.88e-16 |
| Fixed48 bits |657.446 |259.680 |4.933 |7.90e-14 |
| Fixed40 bits |555.962 |238.908 |5.361 |2.26e-11 |

Ratios are medians of per-repetition native/requested ratios, not ratios of
the displayed marginal medians. At the unreplicated M196 shape, all these
fixed modes were slower than native. Dynamic eager never beat native across
the dense sweep; reduced offset -12 crossed native at repeat16 (1.070x) and
reached1.806x at repeat64. Full case/version/mode timing, full-output maximum
absolute/relative/ULP errors and ranges appear in [aggregate.csv](results/aggregate.csv).

These dense matrices repeat the SAME captured radial rows1/4/16/64 times;
they are not distinct boundaries, independent equilibria or solver throughput.
Half of K is zero-masked to preserve separate even/odd output meanings. Dense
nominal arithmetic is12.32x the original separable work for this r1 value alone,
and excludes the other outputs sharing the original Fourier kernels. Matrix
packing and output extraction remain additional work. Thus5.361x versus a dense
DGEMM control cannot be reported as5.361x original Fourier acceleration.

## Accuracy and actual dispatched arithmetic

The CPU exact-reference analysis used binary64 operands as exact rational
numbers, summed products exactly, then rounded once. That mathematical GEMM
target differs from original separable/grouped C and from native GEMM rounding.
There were480 selected case-occurrence dots and11,520 saved-arm comparisons;
selection emphasized worst absolute/ULP, near-zero and cancellation cases.
Repeated rows/cases do not make those counts independent new physics inputs.
Fixed55 selected exact-dot relative error reached9.39e-14 despite small absolute
error. Neither requested bit count nor observed error is a stellarator-design
tolerance or a guarantee of monotonic accuracy across modes.

Profiling establishes dispatch only for its observed target:

- The fixed55 NVJET kernel had30,720 IMMA instructions and251,658,240 INT8
  operations, with zero FP64 operations in that kernel. This is actual INT8
  tensor arithmetic evidence, beyond an API request or kernel-name inference.
- The thin dynamic route's profiled DP tail had8192 DMMA instructions and
  4,194,304 FP64 operations, while its NVJET kernel had zero IMMA instructions.
  Dynamic/default behavior on that target is mixed/fallback; it cannot be
  labeled a universally pure INT8 emulation route.

Nsight Systems severity2 incomplete-event diagnostics are retained. No complete
timeline percentage, total-route arithmetic percentage or all-mode dispatch
claim is inferred from those captures. Performance counters are separate from
the unprofiled event measurements.

## Decision and graph fairness check

Keep the original ordered solver path unchanged. The larger-K fixed-mantissa
results justify further algorithm research as explicitly non-comparable
configurations; the measured thin shapes do not justify solver adoption.
Next decisions need the whole transform, packing/epilogue costs and full
equilibrium metric checks. They must not combine component ratios into a
predicted solver speedup.

The final CUDA-graph screen completed another 24 reports. All 12 graph repeat
comparisons and all 24 graph/direct output comparisons were byte-identical.
For the thin toroidal case, native direct/graph times were 17.53/9.97 us;
fixed55 times were 109.22/45.32 us. Graphs reduced submission overhead, but
fixed55 remained slower than graph-native. For dense repeat64, native
direct/graph times were 1280.85/1286.03 us and fixed55 272.80/263.33 us.
These are descriptive medians from separate two-repetition suites, not paired
causal estimates. See [graph comparisons](results/graph-direct-comparison.csv).
There were 384 completed reports across all six suites. Graphs did not change
the decision above; whole-solver speedup and metric compatibility remain untested.

The report is derived from completed `screen1`, `dense1`, `fixed1`, and
`fixed-dense1` validation CSVs. [Aggregate source hashes](results/aggregate-sources.json)
bind those CSVs without including raw operands, outputs, profiler artifacts or
private host paths. All numeric table entries above are measured or derived
from executed measurements; no unexecuted run is reported.
