# Actual VMEC Fourier contractions for GEMM research

`mapping_export.py` reads the approved public `VMECGPU1` inverse and `VMECGPU2`
forward capture formats. It uses only CPU NumPy multiply/add operations, serial
reduction loops and binary I/O. No matmul, BLAS call, CUDA call or solver DSO is
used. Original capture files are rehashed after reading and never modified.

```sh
python3 tools/optimization/tensor_emulation/mapping_export.py \
  --captures build/optimization/public-stage-capture-run1/actual7/generated-public \
  --out /tmp/vmec-gemm-export-001 --ns 25 51 99
```

Choose a fresh output directory. NumPy is required. `--families` can select
`inverse-toroidal`, `inverse-main`, `forward-finish`, `forward-poloidal`, or
`forward-poloidal-packed`. The opt-in `inverse-full2d` family below tests a
larger reduction dimension. The selected numerical
source manifest is checked and its four actual transform source hashes are
included in the output provenance. Capture dimensions, source SHA256 values,
output comparison and every exported file hash are recorded in `manifest.json`.

## Binary contract

Each file is exactly eight bytes `TEGEMM1\0`, then little-endian uint64 M,N,K,
then FP64 little-endian column-major A[M,K], B[K,N], ordered-reference C[M,N].
Use GEMM transA=N, transB=N, lda=M, ldb=K, ldc=M, alpha=1, beta=0. No padding is
inserted. The CPU exporter performs all necessary row-major capture-to-column-
major packing explicitly. Matrix packing belongs outside device timing, but
must be included in any future end-to-end solver experiment.

## Mapping

Let R be the retained radial count, Mpol the original poloidal modes, Z the
original toroidal points, T the reduced poloidal points, and Q=ntor+1.

| Family | GEMM M,N,K | Operand mapping | Ordered reference |
|---|---|---|---|
| Inverse toroidal, 12 channels | active(R*Mpol),Z,Q | A[jF,m;n] is one of six actual coefficient arrays; B[n;k] is actual cosnv/sinnv/cosnvn/sinnvn | n ascending, one multiply then add per term |
| Inverse main, 8 outputs x 2 parities | R*Z,T,2*count(parity modes) | A[jF,k;2i:2i+2] is the relevant pair of toroidal partials; B[2i:2i+2;l] is the corresponding cosmu/sinmu/cosmum/sinmum pair | m ascending within parity; add the two products, then update accumulator |
| Forward finish, 6 outputs | R*Mpol,Q,2*Z | A[jF,m;2k:2k+2] is the relevant pair of ordered forward poloidal partials; B[2k:2k+2;n] is cosnv/sinnvn or sinnv/cosnvn | k ascending; add the two products, then update accumulator |

For ns99 the actual dimensions are inverse toroidal982x24x11, inverse
main2376x14x10, and forward finish990x11x48. The eight axis rows jF=0,m>1
are absent from inverse toroidal exports because the original kernel skips
their writes. Their coefficient rows are not benchmarked as invented work.
Inverse main uses original zero-masked skipped modes; output row metadata
retains the original radial/toroidal interpretation. The lv sign is represented
by negated B for GEMM, while its reference uses the original subtract update.

Captured basis arrays already contain VMEC mode scaling, derivative factors,
and forward integration weights. Do not synthesize trigonometry, renormalize,
change nfp, or multiply a second quadrature/derivative factor. The exporter
slices the retained mpol rows from the larger captured basis table exactly as
the original indexed kernel does.

## Non-GEMM operations and fidelity boundary

The full ordered reference also reconstructs the inverse constraint outputs.
Their factor xmpq[m]*sqrt_s is multiplied AFTER the two-product sum. Moving it
into either GEMM operand changes rounding, so these outputs are not exported
as if they were equivalent single GEMMs.

Forward poloidal partials are reconstructed with the exact original formulas,
including tempR=armn+xmpq[m]*frcon and tempZ=azmn+xmpq[m]*fzcon before each
weighted term, R/Z LCFS exclusions, axis exclusions and lambda jF>=1 finish
rule. A broader algebraic GEMM decomposition would separate armn/brmn/frcon
products and combine them later; that changes this per-l rounding order.
It is now exported only under the explicitly alternative forward-poloidal
families described below; the original reference remains unchanged.

## Dominant forward-poloidal kernel: complete decomposition costs

There are ten physical field types (ARMN, BRMN, FRCON, CRMN, AZMN, BZMN,
FZCON, CZMN, BLMN, CLMN) with separate even/odd inputs. Each field needs two
of four bases. The small alternative therefore requires **40 GEMMs**, not one:
20 per parity. Each has K=14 and N=5 for the actual retained parity modes.
M is radial*nZeta:2352 for R/Z and2376 for lambda at ns99. Using N=10 per
parity would compute half unused mode columns and must count that extra work.

The packed alternative performs **two GEMMs**, one per parity, by stacking all
ten field matrices in A and concatenating all four basis matrices in B. At
ns99 each is M=23760,N=20,K=14. It computes two unneeded basis combinations
per field, and zero-pads the shorter R/Z input allocations to the lambda radial
extent without reading past their input buffers. The packed two-GEMM total is
26,611,200 nominal2MNK operations versus13,198,080 for40 small GEMMs:2.016293x
the product work. Both alternatives also compute rows/modes later discarded by
the original axis mask; these cannot be reported as useful extra physics.

Neither GEMM family alone produces the original12 partial channels. They need
an extraction/recombination epilogue: four R/Z value channels each combine two
products and an xmpq-weighted third product, totaling four multiplies and eight
adds per retained R/Z point/mode; six derivative products are negated; two
lambda value products are copied. Original axis/LCFS masks must be applied.
Packing, extraction, epilogue and these extra products must be included in an
end-to-end comparison rather than timing one GEMM and claiming a kernel win.

`ns99-forward-poloidal-reference.npz` retains original ordered12 partials,
the explicitly decomposed ordered-product result, xmpq and radial indices.
The manifest records the exact product-to-partial formulas. Packed selected
product results were bitwise checked against their small-product counterparts.
This decomposition itself already changes82,846 partial values on the actual
ns99 capture (maximum absolute drift2.51534904e-17), before any Ozaki emulation.
It is thus an alternative configuration even if Ozaki perfectly reproduces
the decomposed GEMMs. No equilibrium metric equivalence follows from that error.

Every emitted C is an ordered contraction reference. For inverse main and
forward finish, standard GEMM can change grouping even in native pedantic
FP64. Therefore report three independent comparisons: native GEMM vs ordered
C, Ozaki GEMM vs ordered C, and Ozaki GEMM vs native GEMM. Report max absolute,
relative and ULP drift, signed-zero differences, and nonfinite counts. A small
GEMM error does not prove equilibrium-metric parity. These experiments are
explicitly reassociated alternatives and cannot enter the identity table.

The exporter additionally compares ALL18 inverse and ALL6 forward reconstructed
arrays bitwise to the capture's original expected arrays, recording success or
failure. It does not silently require or assert metric equivalence, and no
GPU timing or solver performance result is produced by this CPU export.

## Full2D product-basis inverse: larger-K alternative

To test emulation with K larger than11/14, not assume thin-GEMM conclusions
apply universally:

```sh
python3 tools/optimization/tensor_emulation/mapping_export.py \
  --captures build/optimization/public-stage-capture-run1/actual7/generated-public \
  --out /tmp/vmec-full2d-001 --ns 99 --families inverse-full2d \
  --repeat-rows 1 4 16 64
```

For each retained nonaxis radial row, A stacks actual RMNCC[m,n] then
RMNSS[m,n]. B stacks precomputed `cosmu[m,l]*cosnv[k,n]` and
`sinmu[m,l]*sinnv[k,n]`; columns are `k*nThetaReduced+l`. All captured basis
normalization is retained; there is no new sine/cosine evaluation or scaling.
The axis row is excluded because higher m do not contribute there; it would
need the original axis path in a later integrated experiment.

Three configurations are exported, all explicitly reassociated alternatives:

| Configuration, repeat1 | M,N,K | C reference | Nominal2MNK |
|---|---|---|---|
| r1-sum diagnostic | 98,336,220 | Original captured r1_even + r1_odd, added once | 14,488,320 |
| r1-even direct comparison | 98,336,110 | Original captured r1_even | 7,244,160 |
| r1-split-output | 196,336,220 | Captured r1_even rows stacked over r1_odd rows | 28,976,640 |

The split-output version masks opposite-parity A coefficients to zero and
uses one common B. It retains the two required output-buffer meanings, with
twice the rows and half the K entries zero per row. The sum version is only
a diagnostic: it does not replace the parity-split solver buffers and is not
the physical geometry combination involving radial sqrt_s.

Value-only original separable arithmetic is2,352,000 strict operations for
both parities at this shape; the sum reference adds32,928 more. The dense sum
uses6.16x that separable product work, and split-output uses12.32x including
its zero-mask work. These are NOT full inverse-kernel operation counts: the
actual original kernels also compute many other outputs/derivatives. Preparing
the full B requires73,920 FP64 products for the K220 cases; packing, masking,
output extraction and original axis handling must be counted for integration.

The CPU dense reference evaluates the new K order with separate multiply/add
operations and compares it against the original captured C. At ns99 repeat1,
split-output already changes55,795 values, max absolute2.442490654e-15 and
relative L2 error4.233877421e-16 before Ozaki. The sum diagnostic changes29,128
values, max absolute3.552713679e-15. These are measured CPU arithmetic drifts,
not equilibrium metric errors or GPU accuracy/performance results.

Repeat factors1/4/16/64 duplicate the SAME actual radial rows to measure GEMM
size crossover. They do not represent independent equilibria, extra boundaries,
or demonstrated solver throughput. The output records repetition scope and
extra nominal work explicitly. Previous actual3/actual4 exports remain unchanged.
