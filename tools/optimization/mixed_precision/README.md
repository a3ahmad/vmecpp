# Mixed-precision transform research

See [the measured findings and tradeoff table](FINDINGS.md). The pilot tests
FP32 transforms within an otherwise FP64 solver and FP64 finishing; it does
not test a fully FP32 solver or three-component 16-bit tensor arithmetic.

This isolated prototype preserves the double pointer ABI, device storage, solver
state, physics deck, and transform traversal order. It is not a qualified solver
replacement. The FP32 branches cast inputs to float and perform local additions
and multiplications in float; outputs and toroidal partials are stored as double.
The FP64 branches copy the original retained transform bodies. CUDA 12.9,
`--fmad=false`, and SM120 compilation match the retained toolchain.

The resident DSO exports unmangled controls:

- `SetGpuInverseTransformPrecision(int bits, void* stream)`
- `SetGpuForwardTransformPrecision(int bits, void* stream)`
- `GetGpuInverseTransformPrecisionCounts(unsigned long long* counts)`
- `GetGpuForwardTransformPrecisionCounts(unsigned long long* counts)`

Precision defaults to 64. Both inverse and forward wrappers read a device flag
when the kernel executes, including during replay of an already captured graph.
The agreed caller uses `stream=nullptr`: setters synchronously copy four bytes
only on a precision transition, after the preceding solver readback has drained.
Inverse rejects a nonnull stream. Each transform increments two physical device
counters, once for each of its two kernels. Counter order is `[FP64, FP32]`.
Getters are synchronous and must run after the caller's normal drain.

Flags and counters belong to the CUDA module, not an individual solver. Use one
equilibrium at a time per process/context; differently scheduled concurrent
threads are unsupported. Independent process batching has separate modules.

The root-owned experimental shim controls the policy with
`VMECPP_EXPERIMENT_TRANSFORM_MODE=fp64|fp32|warmup`,
`VMECPP_EXPERIMENT_FP32_CALLS`, `VMECPP_EXPERIMENT_PRECISION_TRACE`, and
`VMECPP_EXPERIMENT_RESIDENT_SO`. The call budget is per radial stage and must
remain monotonic across solver-context resets. See `precision_trial_hooks.inc`
for the actual policy and CPU materialization certificate admission.

Build steps use fresh absolute output directories:

```sh
python3 tools/optimization/mixed_precision/build_inverse.py --out /fresh/inverse
python3 tools/optimization/mixed_precision/build_resident.py --out /fresh/resident
python3 tools/optimization/mixed_precision/build_shim.py \
  --out /fresh/shim --backend /fresh/resident/solver-backend.so
```

The current narrow recipe pins the actual research inverse and forward objects
under `build/optimization/`; it is not a portable source-only rebuild recipe.
Resident and constraint objects are reused byte-for-byte, while only inverse
and forward objects change. A private copy of the retained backend receives a
DT_NEEDED replacement for the experimental resident. Frozen originals are never
modified. The first runtime gate must establish candidate FP64 mode is bit-exact
to the retained solver before FP32 or warmup results are interpreted. No such
runtime qualification is implied by a successful CPU compile.

## Saved trial runner and analysis

`run_trial.py` is a published copy of the measured private `run2.py`; `worker.py`
retains the existing scientific worker wrapper. `analyze_trial.py` reads saved
results and reports metric and physical-array drift, including failed solves.
These scripts explicitly require this checkout's `build/optimization` authority,
canonical fixture reports, dependency snapshot, and finite owner. They are not
portable standalone tools and do not install packages or change the baseline.
The runner plans by default; actual execution requires `--execute` and an
externally coordinated GPU resource window. Approximate modes require a matching
completed candidate FP64 exact-control receipt.

```sh
python3 tools/optimization/mixed_precision/run_trial.py \
  --backend /absolute/candidate/solver-backend.so \
  --resident /absolute/candidate/resident.so \
  --shim /absolute/candidate/solver-shim.so --cases 1 --require-exact \
  --out /fresh/control
python3 tools/optimization/mixed_precision/analyze_trial.py \
  --run /saved/trial --out /fresh/analysis
```

Before postmeasurement Google formatting, all measured mixed-precision source
files, private runner originals, and build receipts were copied to
`build/optimization/mixed-precision-solver1/measured-source/` with a SHA256
manifest. The measured resident (`49b5d265...`), backend (`40e74922...`), and
corrected shim (`1e42d2e4...`) remain unchanged. Google formatting changed
whitespace, adjacent string-literal wrapping, and include ordering in the shim
and accuracy consumers. The inverse and forward CUDA implementation files and
headers were already formatted and remain byte-exact. These postmeasurement
formatting changes do not qualify a newly rebuilt binary.
