# Experimental device graph dispatch

This work targets the Jacobian host round trip in `Solver::Run`. It is outside
the qualified timing table. No speedup is established for this prototype.

The qualified Thomas whole-VMEC trace contains27,782 stream synchronizations
and27,477 host graph launches. Recorded GPU kernel coverage is65.725% of the
instrumented12.197872s inner call. This motivates testing device scheduling;
it does not establish removable latency or a roofline percentage.

The prototype preserves the original CPU convergence and multigrid controller.
For an eligible ordinary iteration, one host graph runs the original prelude,
then a GPU controller chooses either the original good-tail graph or a graph
that restores the same five cached profiles on a bad Jacobian. The host reads
Jacobian extrema and the original scalar packet after that entire device tail
completes. It checks the device branch against the original host multiplication.
Refresh iterations, graph warmup, unpinned scalars and unsupported host floating
point modes retain the original path. No iteration, tolerance or check is skipped.

The implementation is opt-in and built from a private copy of the qualified
Thomas sources. Working controller/prefetch sources and all qualified binaries
are untouched. The helper is `gpu_device_graph_dispatch.h`; the original
`gpu_solver.h` and solver shim remain byte-identical in the experiment.

## Executed controls

- Initial local24 controls: all normal, memory and race checks passed; three
  device-tail synchronization checks failed. Failures are preserved.
- Local follow-up10 controls: own full-block scan, single dispatch, self-tail,
  and wider controller variants passed normal and synchronization checks with
  explicit end-of-controller barriers.
- Native factorial72 controls: all36 normal checks passed. All18 synchronization
  checks with the end-of-dispatch barrier passed; all18 without it failed.
  An end-of-continuation barrier did not change this distinction.
- Native focused12 memory/race checks and local24 normal/memory/synchronization/
  race checks passed for the two passing barrier variants. Each control checks
  every scan iteration, invocation count and downstream host-copy ordering for
  bounded chains, including the zero-iteration case. These are not solver tests.
- Solver prototypes2 and3: five configurations x38 frames matched bitwise,
  including a folded interior Z surface, cached-force step and backup restore.
  Memory checks passed. Synchronization checks failed, including host launches
  of graphs instantiated with device-launch capability. Race/full-solve gates
  were not executed for these failed variants.

Prototype4 preserves original host graph executables and instantiates separate
device-only tail executables. Its38-frame normal and memory checks passed;
synchronization failed with224 reported errors at the global CUB guess scan.
Race and full-oracle gates were not executed. Independent standalone global-scan
controls passed all48 normal/memory/synchronization/race jobs, including the
actual three-component residual shape. A generic CUB failure is not established.
Read-only ELF inspection found no duplicated exported scan symbols across the
backend, resident and RHS libraries; an interposition fix is unsupported.
The native factorial archive and its102 verified files are bound
by `build/optimization/device-tail-controls3/native-focused-independent-review.json`.

## Additional failure isolation

Prototype5 captures a fresh graph definition for its device-only tail. Its
38-frame normal and memory controls passed; synchronization still failed224
locally. Native prototype4 passed normal38-frame controls but synchronization
failed256. The issue is not confined to WSL. Neither variant is qualified;
race/full-oracle gates were not executed for these failures.

The independent native prefix study audited85 completed controls and338 file
hashes. Ordinary host graphs passed; host execution of DeviceLaunch-flagged
prefix graphs failed synchronization; device dispatch of those same prefix
bodies passed. A shared-memory negative control was detected in all four
tested launch modes. This establishes detection for that control, not universal
instrumentation coverage. An earlier optimized negative control had no
WARPSYNC instruction and is excluded from coverage.

Mixed ordinary/device use and fresh ordinary graphs after device dispatch
passed. These controls do not support a general graph-definition-order or
persistent function-poisoning explanation. Full-solver failure remains
unresolved. The next diagnostic records actual graph flags, node composition,
workspace identities and launch origins around the first failing solver path,
without adding synchronization or changing arithmetic. Evidence and the plan:
`build/optimization/native-prefix-isolation-summary1/`.

The actual first-failure diagnostic is now executed. Its first version called
CUDA kernel-node metadata getters;13 returned error98 and the normal replay
subsequently failed. That attempt is preserved and provides no solver
qualification. A fresh version removes all added CUDA API queries and keeps
only a bounded CPU event ring around original definitions, launches and syncs.
Normal replay passed all38 frames; synccheck reproduced exit91 at the first
parent host launch, with actual parent flags0 and good/bad child flags4. The
original stream synchronization returned999. No events were dropped, frozen
inputs stayed unchanged and all owned processes were reaped. Source inspection
places this doubleguess CUB scan in child-tail residual/energy work; the parent
prelude has no matching scan. Host backtrace and launch metadata do not identify
which child callsite failed. Solver pointer reuse across earlier scenarios
also prevents treating raw addresses as context generations. Raw evidence:
`build/optimization/device-tail-first-failure2/`
and `build/optimization/resident-solver-device-tail-trace2/first-failure1/`.

## Actual child-callsite differentials

Four isolated variants replace exactly one prefix proposal, only while defining
the good device child, with the literal RN ordered prefix plus verification.
All ordinary warmup/capture paths remain unchanged. Each full38-frame normal
replay passed, but all four synchronization checks failed91. Reported error
totals were2464 for thermal removal and224 for each other removal. These counts
do not include the tool's separately suppressed-error messages. No single
necessary callsite was isolated. Path statistics differ truthfully and are
not claimed identical to the original proposal statistics.

Replacing all four child prefix proposals also passed normal38 frames but
failed synchronization with128 errors. The first reported kernel changed from
CUB to lattice-energy `OrderedFallback` at0x780. Offline SASS identifies that
instruction as an unpredicated full-CTA `BAR.SYNC.DEFER_BLOCKING`, not WARPSYNC;
the same-named optimistic-diagnostics kernel has a MOV at that offset. The
preceding instruction is `BSYNC.RECONVERGENT` after a partial shared-store branch.
This static observation does not prove dynamic participation. The result
broadens the observed failure beyond CUB and does not establish a CUB defect.
Evidence: `build/optimization/device-tail-drop-one1/` and
`build/optimization/device-tail-drop-all1/`.

A separate actual full-solver differential changes only the parent Dispatch
launch from one thread to32. All88 compiled GPU functions retain identical
encoded instruction words in both architectures; the launch dimension changes
in host code. Normal38 frames passed, while synchronization again failed91
with the original CUB Policy1000+0x53b0 report and224 errors. The full-warp
parent therefore does not resolve this failure. Frozen inputs stayed unchanged
and all owned processes were reaped. Evidence:
`build/optimization/device-tail-parent-warp1/` and
`build/optimization/resident-solver-device-tail-warp1/first-failure1/audit.json`.

None of these diagnostics qualifies device dispatch or measures a speedup.

## Build and remaining gates

The exact absolute-path command for the current prototype is saved in
`build/optimization/device-tail-solver1/split-executable-build-command.json`.
Its frozen input copy is `build/optimization/device-tail-solver1/input`, and the
build output is `build/optimization/resident-solver-device-tail4`. Default
configuration is OFF; enabling it requires coordinated captures and excludes
buffer pooling and the live-controller experiment. The private builder snapshots
all sources and linked component hashes before compiling.

`device_tail_dispatch_frozen_thomas.patch` is the reviewable prototype4 patch
against the exact qualified Thomas source snapshot. A read-only dry run passed.
This patch remains experimental because its synchronization gate failed.

Required next gates are explicit good/bad activation, all physical/cache/frame
comparisons, memory/synchronization/race checks, full original-CPU trajectory
smokes, the six-case repeat matrix, native cross-host checks and unprofiled paired
timings. Refresh-device control and an autonomous GPU iteration loop remain
separate work. A passing simple graph control does not qualify a full solver.
