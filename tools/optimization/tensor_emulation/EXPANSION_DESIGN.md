# Short-contraction FP16/BF16 expansion recommendation

This is CPU/source research, not a new measured speedup. Prefer **six scaled
FP16 slices for K11/14 as the first accuracy-oriented candidate**, retaining
four-slice and BF16 variants as explicitly approximate controls. Do not label
fixed4/6 residual casts as guaranteed FP64 emulation. Keep the original ordered
GPU transform unchanged; all GEMM/expansion alternatives remain outside the
evaluation-identity configuration.

## What transfers from the primary literature

Mukunoki et al.'s FP16 Ozaki construction extracts scaled slices chosen so each
FP32 component dot is exact, then reconstructs in FP64; its correctly-rounded
variant additionally uses accurate summation. It is more than repeatedly
casting a residual. Required depth depends on K and exponent spread. Its
estimated-accuracy fast mode truncates products and has different guarantees
from the correctly-rounded mode. Power-of-two scaling and reusable workspace
are useful here; its published large-GEMM performance is not a VMEC prediction.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/),
[authors' poster](https://sighpc.ipsj.or.jp/HPCAsia2020/hpcasia2020_posters/poster_09.pdf).

The newer AMX paper fixes six BF16 slices and tests triangular6/10/15/21
component products, widening each FP32 component result into an FP64 sum.
It explicitly restricts operand/residual exponent range, implements no general
scaling/fallback, and does not promise a universal FP64 threshold. Transfer
its fused packing, component reuse and wider reconstruction ideas. AMX tile
scheduling, CPU crossover and FP32 reduction behavior do not establish GPU
accuracy/performance. Six slices and six products are different parameters.
[Cui and Liu, 2026](https://arxiv.org/html/2609.04663v1).

Henry/Tang/Heinecke's familiar six-product schedule uses THREE BF16 components
to target FP32-oriented accuracy; it is not a proof of FP64 accuracy from six
BF16 components. [Primary paper](https://arxiv.org/html/1904.06376v1).

## Accumulation and exponent safeguards

The following is a conservative engineering derivation, not a theorem that
fixed slice count guarantees FP64. For common-exponent integer limbs with at
most pA/pB significant bits, require `pA+pB+ceil(log2(K)) <= 24`, plus normal
finite products, to keep the absolute FP32 partial-sum bound representable.
The condition must hold at every accumulation step; arbitrary residual casts
with differing scales do not automatically satisfy it.

| Actual K | Conservative equal-limb bits | Six-slice nominal aligned-bit budget |
|---:|---:|---:|
|11/14 |10 |60 |
|48 |9 |54 |
|110/220 |8 |48 |

Exponent gaps, cancellation and omitted cross products consume accuracy beyond
this nominal budget. Four ordinary FP16 slices have only about44 significand
bits of relative coverage; six BF16 slices about48. These are useful probes,
not generic53-bit guarantees. For longer K, allow adaptive depth beyond six or
an explicit failure/fallback rather than silently accepting an accuracy label.

An improved balanced-radix variant tightens that conservative estimate. Set
`e=ilogb(panel_max)+2`, `r0=scalbn(x,-e)`, then repeatedly compute
`d_i=RN(scalbn(r_i,b))` and `r_next=scalbn(r_i,b)-d_i`. When scaling remains
representable, `|r_i|<=1/2` and `|d_i|<=2^(b-1)` for every slice, including the
endpoint carry. Reconstruction weights are exactly `e-b*(i+1)`. For b10 FP16,
all integer digits through magnitude512 are representable. Thus the stronger
absolute bound is `K*group_pairs*2^18<=2^24`: up to5 equal-weight pairs at K11
or4 at K14 can share one FP32 accumulator. Six slices with triangular21 products
then need7 or8 FP64 reconstructions respectively. This preserves60 radix bits
(roughly59 relative to the panel maximum), not guaranteed entrywise accuracy.
It is a source/math design, not a measured speedup.

Balanced integer digits also change the BF16 precision tradeoff: b9 digits
range through magnitude256, and EVERY integer in [-256,256] is exactly BF16
representable. Six such slices preserve54 radix bits; the normalized residual
bound is2^-55, approximately2^-53 relative to the panel maximum. Thus the
earlier48-bit estimate applies to ordinary six-step BF16 residual casts, not
this balanced/scaled radix9 encoding. Its K11/14 group caps23/18 accommodate
the largest six-pair diagonal, giving six reconstructions for triangular21
products. This still does not prove entrywise or cancellation-safe FP64 output.
FP16 b11 digits through magnitude1024 are also exact, providing more retained
bits, but their K11/14 group cap is only1: triangular21 then needs21 FP64
reconstructions. These are useful different accuracy/cost settings to measure.

Equal `i+j` only implies equal weight if exponents follow that fixed radix
schedule. Independent adaptive per-slice exponents break the grouping argument.
Padding must be literal zero, and partial K blocks must use their actual active
length in the bound. A group starts from FP32 zero and cannot mix weights.
Check scaled-nonzero loss and all finite/normal conditions; use `scalbn`
directly instead of constructing potentially overflowing `2^e` values.

Use row scales for A and column scales for B along the contraction direction.
Store separate power-of-two exponents per slice and reconstruct with exact
power-of-two rescaling when representable. Compute extraction residuals in FP64.
Do not downcast the original FP64 through FP32 first. Check residual exhaustion,
nonfinite values, overflow, subnormal loss and scaled-zero loss. FP16's narrow
exponent range makes unscaled repeated residual casts particularly unsafe;
BF16 has wider range but does not cover arbitrary FP64 inputs/residuals.

Never accumulate distinct high/low component products into ONE FP32 output:
that can discard all low-order recovery. Start each component product from
FP32 zero, then widen it and reconstruct in a fixed FP64 order. A correctly
rounded target needs an accurate-sum algorithm and a residual/product-tail
bound, not merely FP64 additions. For a six-slice reference, keep all36 products
initially. Triangular21/15/10 products are separately labeled approximation
settings; do not equate a longer expansion with monotonically better results.

NVIDIA specifies FP32-or-better accumulation for these MMA forms, but leaves
reduction order, rounding and subnormal handling unspecified. Verify actual
normal-limb behavior and dispatch; an API compute-type request alone is not
proof. Do not reuse one GPU's numerical behavior as a cross-host bitwise claim.
[NVIDIA PTX numerical semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

## Fair original-transform comparator, ready now

The existing `inverse-device-timing1` consumer calls the retained
`LaunchGpuFourierDevice(GpuFourierDeviceState&)` interface:16 input channels,
18 contiguous output channels,13 dimension entries and12-channel separable
scratch. It measures output clear + ToroidalPartials + inverse main, with
H2D/D2H, allocations and verification outside events. It verifies ALL18 arrays
bitwise and input immutability after each warm/timed batch, then retires graphs,
events, allocations and streams. This is the comparator to rerun, rather than
the old transfer-heavy `gpu_replay` or historical approximately40us values.

`expansion_transform_comparator.py` is a small PLAN/exec wrapper around that
unchanged consumer. It hash-checks its retained binary, public ns99 capture,
current native resident60d and private loader. Both alternating arms use the
SAME current native module,100 calls and8 pairs; pair0 is declared excluded.
The wrapper is site-bound to the retained ignored assets, not a new wheel or
new benchmark framework. Root supplies the verified current driver directory,
GPU lock and externally owned60-second timeout:

```sh
python3 tools/optimization/tensor_emulation/expansion_transform_comparator.py \
  --driver-dir "$ACTIVE_WSL_DRIVER_DIR" --report /tmp/FRESH-native-inverse.json
# Add --execute only under the existing root-owned GPU run/timeout.
```

The PLAN path was CPU-executed: all four asset hashes passed, with no GPU call.
This wrapper intentionally creates a clean target environment and is for
unprofiled timing; it must not be inserted below Nsight and strip injection.
Graph timing still can contain submission gaps; retain all reported intervals.

The isolated `expansion_mma168.cuh` header is an optional prototype for the
same integer digit buffers and scale arrays. Native PTX m16n8k16 keeps four
outputs per lane, without WMMA shared staging. For M2376,N14, output CTA count
increases149->298 compared with16x16 WMMA; for M982,N24 it increases124->186
while reducing padded N32->24. Grouping is opt-in with a checked balanced-limb
cap and deterministic FP64 reconstruction. Its half/BF16 SM120 instantiations
were CPU-compiled successfully with strict flags:40 registers, zero shared,
local or stack bytes. No GPU parity or speedup is claimed for this header.

TEGEMM jobs are individual sub-contractions. Inverse toroidal exports12 jobs,
inverse main16, while the original also computes constraints with post-pair
scaling. Forward's40 small products or2 packed products still require masks,
packing, extraction and epilogue. Summing independently timed sub-GEMMs does
not produce reliable complete-transform time. Full2D parity-split GEMMs also
perform12.32x the original value-only separable work and omit other outputs.
An expansion implementation is not a full-transform candidate until it writes
all18 required outputs and includes these costs. Do not weaken the original
consumer's bitwise gate to admit it: use a separate accuracy-reporting arm.

## Bounded decision sequence

1. Compare new4/6-slice outputs to exact rational stored-operand GEMM, native
   GEMM, and original grouped C separately, including cancellation/near-zero,
   signed zero, exponent boundaries, repeated runs and actual FP32 component
   roundoff. Record extraction residuals and retained-product schedule.
2. Benchmark one consumed graph containing conversion, all component products,
   rescaling/reconstruction and epilogue. Time cached basis splitting separately
   only if its cache lifetime is valid; A changes every solver iteration.
3. Rerun the full native18-output comparator on the SAME resource window. A
   sub-GEMM win is not a transform win; a sub-GEMM already slower than the full
   native call is a strong negative result for that current implementation.
4. Only a complete-transform advantage warrants full-equilibrium metric and
   convergence testing. No scientific tolerance is selected by this design.

No new GPU profile, workload, lease or speedup measurement was performed for
this document; the isolated header received only the CPU compilation stated
above. Existing expansion implementation files owned by the GPU agent were
not edited.
