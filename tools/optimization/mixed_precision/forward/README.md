# FP32-local forward transform trial

This isolated experimental TU keeps the selected double input, partial and
output ABI. The default device flag selects the original FP64 expressions.
The alternative casts each floating input/basis load to float before its
multiply/add and uses float local accumulators in BOTH poloidal partials and
toroidal finish; results are widened into the original double buffers.
Indexing, Fourier basis/scaling, axis and LCFS masks, literal zero stores,
launch shapes and allocation layout are unchanged. No fast math or FMA is
enabled. Controller arithmetic, force norms, preconditioner and deck settings
are outside this TU and remain FP64 under the trial protocol.

The mode is read uniformly on the device every kernel execution, including
CUDA graph replays. A host-only capture-time branch would not support a late
precision switch. Original FP64 helper bodies retain their arithmetic; the
wrapper branch and diagnostic atomic are new, so physical parity still needs
an actual default64 comparison before any qualification claim.

Exports:

- `SetGpuForwardTransformPrecision(int precision, void* stream)` returns a
  `cudaError_t`. Only32/64 are accepted. Default64 and unchanged-mode requests
  return success without a copy. A NULL stream uses synchronous
  `cudaMemcpyToSymbol`; non-NULL uses the supplied stream and immutable static
  host0/1 sources, avoiding asynchronous stack-lifetime hazards.
- `GetGpuForwardTransformPrecisionCounts(unsigned long long* out2)` returns
  a `cudaError_t` and copies `[FP64, FP32]` execution counts synchronously.
  Read only after the consuming solver streams have drained.

Each partial/finish wrapper increments its selected count once on its first
physical thread/block. A complete transform therefore adds TWO counts.
Graph capture alone does not increment these device counters. Both precision
modes incur the same diagnostic atomic. There is no counter reset API; use
before/after differences.

The flag/cached host mode are process/context global. This trial requires ONE
active equilibrium/context per process; independent threads with different
precision budgets are unsupported. The CPU owner must finish an iteration
before changing both inverse and forward flags, check setter status, and use
the device counts to verify the actual schedule. Budget/iteration policy is
implemented by the separate experimental solver shim, not this TU.

## Accuracy and convergence limitation

FP32 is an explicitly non-comparable accuracy alternative. For orientation,
the standard float summation factors gamma14 and gamma24 are approximately
8.34e-7 and1.43e-6 before input conversion and paired-expression errors; these
scale absolute dot magnitudes, not necessarily the cancellation-small output.
They are analytical warning scales, not a measured residual/convergence floor.
The high-fidelity force tolerances cannot be assumed achievable in FP32.

A bounded FP32 warm-up followed by FP64 finishing may recover convergence, but
can also change restart decisions, trajectory, final metrics or convergence.
No tolerance, bitwise identity, robustness or speedup is promised. Report an
actual error/trace/metric comparison against the unchanged FP64 call. Retain
the original zero-FP32 mode as the control, and keep failed trials as evidence.

## CPU build

Use the original selected forward flags with CUDA12.9 SM120, C++20, `-O3`,
`--fmad=false`, `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`,
`-DVMECPP_FORWARD_SPLIT_GROUPS`, `-DVMECPP_FORWARD_ELIDE_OUTPUT_CLEAR=1`,
and PIC. `source-provenance.json` binds the source cohort and build receipt.
The combined resident linker must use this new forward object plus the separate
mixed inverse object and the unchanged selected resident/constraint objects.
This directory does not install a wheel or modify the original solver build.
No GPU execution was performed by the source author.
