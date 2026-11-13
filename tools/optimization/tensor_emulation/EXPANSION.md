# Standalone FP16/BF16 expansion experiment

This changes no solver arithmetic or installed runtime. Root alone executes
GPU work. PLAN is the default. Inputs use the same TEGEMM1 format as bench.cu.
Every report compares the output with supplied original-order C; native
pedantic GEMM and an ascending separate-DMUL/DADD scalar control are distinct
comparators. Neither control is claimed to be the current VMEC implementation.

Build on CPU only, in a fresh directory:

```sh
/usr/bin/python3 -I -S -B tools/optimization/tensor_emulation/build_expansion.py \
  --cuda /usr/local/cuda-13.5 --out build/optimization/tensor-expansion-13.5-FRESH
```

The build receipt records binary/source/library hashes. Successful builds
retain exact copies of all source/header inputs. Earlier snapshots remain
preserved. Build3 was not sealed because a header changed during compilation;
build7 compiled but its fused triangular-bound metadata needed correction.
Build8 is the final formatted, source-frozen handoff. Its executable SHA256 is
`42ecde2fd0cc82cf3a87534972ee2623e20e811cd1b2560c4a15f45aeed24cf7`.
The public CUDA/header bytes match the build8 receipt and retained snapshots.

Root-owned example, using the active driver directory verified for this host:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="$ACTIVE_WSL_DRIVER_DIR:/usr/lib/wsl/lib:/usr/local/cuda-13.5/lib64" \
  /usr/bin/timeout --signal=TERM --kill-after=8s 90s \
  build/optimization/tensor-expansion-13.5-build8/expansion_bench --execute \
  --input build/optimization/tensor-emulation-20261112/mapping/actual3/ns99-inv-toroidal-00.bin \
  --mode all --backend mma168-fused --split warp --group-diagonals \
  --terms triangular --reuse-b --graph-batch 32 \
  --calls 20 --samples 7 --warmups 3 \
  --expect-cublas /usr/local/cuda-13.5/lib64/libcublas.so \
  --report /tmp/FRESH-expansion --output /tmp/FRESH-expansion-C
```

All writes eight mode-suffixed JSON/binary pairs: native, ordered, fp16-4,
fp16-6, fp16-8, bf16-4, bf16-6, bf16-8. Single named modes are supported.
`--backend` selects batched cuBLAS, WMMA, or register-loaded m16n8 PTX via
`batched`, `wmma`, or `mma168`. Additional explicit backends are:

- `mma168-cache`: keep all A/B component operands in registers for K<=16 and
  one complete K chunk. Other shapes execute the original `mma168` route;
  `cached_operands_active` reports whether caching was actually active.
- `mma168-parallel`: independent diagonal-group CTAs write FP32 scratch, then
  a same-stream kernel reconstructs in the declared group order. The group
  plan is uploaded before capture. This backend requires diagonal grouping.
- `mma168-fused`: for K<=16 and a complete K chunk, the tensor kernel reads
  original FP64 A and combines its row maximum, balanced split, cached MMA
  and reconstruction. There is no A-digit materialization or separate A-split
  launch. Other shapes execute the original `mma168` route with a separate
  split; `fused_A_split_active` reports actual activation. B is not fused.

Grouping requires a custom tensor backend. `fp16-all` and `bf16-all` select
native, ordered, and only the chosen format's 4/6/8-component modes. This
allows format-specific explicit digit widths without admitting another type.
`--split row` is the initial one-thread-per-panel implementation; `warp` uses
one warp per row/column. `--digits` overrides the conservative automatic digit
width, and `--k-block` bounds FP32 subdot length (zero means the full K).
Explicit type limits are 12 radix bits for FP16 and 9 for BF16; the subdot
bound still rejects unsafe combinations. Automatic widths remain conservative.

`--split quantized` uses one power-of-two FP64 scaling and one signed64 RNE
quantization, followed by integer balanced-digit extraction. It is active only
when q*b<=64; wider cases explicitly use the warp recurrence. This can change
the individual balanced digits and triangular truncation results, even when
the complete quantized value agrees. The portable CPU check validates signed
round-shifts, carry endpoints, modulo subtraction and integer reconstruction;
it does not prove GPU correctness or triangular equivalence.
In the fused backend A ALWAYS uses its balanced recurrence, even when
`--split quantized` is requested. B may use the quantized split, including its
one-time cached presplit. Separate activation fields distinguish these paths.

For each A row/B column, e=frexp(maxabs).exponent+1 gives normalized magnitude
strictly below0.5. The balanced fixed-radix recurrence is
`d=RNE(r*2^b); r_next=r*2^b-d`, with fixed component weight `2^(e-b*(s+1))`.
It bounds integer digits by `2^(b-1)`, including a possible carry endpoint.
Power-of-two normalization is range-checked for lost nonzero inputs; device
range-failure counts are retained. These checks do not prove negligible
finite-component error for arbitrary exponent ranges or cancellation.

FP32 subdots are exact integer arithmetic when
`Kblock * groupPairs * 2^(2*(b-1)) <= 2^24`. All zero-padded K entries are zero.
Grouped diagonals combine only equal-weight component products, obeying this
same bound before FP64 widening. Full products retain q*q pairs; triangular
keeps i+j<q and explicitly drops the others. The reported omitted-cross-term
bound covers only those dropped digit products, not finite-q loss, FP64
reconstruction rounding, input-scaling underflow or original-order differences.
Exact FP32 subdots do not imply an exact FP64 dot or signed-zero preservation.

Steady graph timing includes variable A splitting, all component GEMMs and
FP64 reconstruction. Without --reuse-b it also includes B splitting. With
reuse enabled, B splitting is measured once separately and excluded from
steady calls: this requires a genuinely reusable basis in any integration.
The cached B contains its digit arrays and column scales, not merely raw B.
Fused A normalization/splitting remains inside every measured computation.
No allocation occurs inside the captured computation. --graph-batch N records
N complete computations in one graph, divides steady/event/RAW times by N,
and reports the actual operations per sample. Default N1 preserves the first
protocol. --direct disables capture. --profile-one additionally brackets one
warmed graph launch (N operations when batched), or one direct computation,
with cudaProfilerStart/Stop; it launches no profiler itself.

CUDA events measure the complete queued pipeline, not just tensor instructions.
RAW is a host diagnostic under WSL and is not silently calibrated. GPU paths,
speedups, full-transform benefit and solver eligibility require actual
measurements. First executed small-shape results and subsequent optimizations
must remain separate; replication of real rows is not independent boundary
coverage. Accuracy scripts and stress inputs are maintained separately.
No mode has a generic FP64 accuracy or original-VMEC bitwise guarantee.
