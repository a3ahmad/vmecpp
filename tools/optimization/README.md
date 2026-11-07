# Isolated optimization experiments

The latest completed hardware pass is documented in [GB202.md](GB202.md), with
versioned source and aggregate six-case results. The historical campaigns below
retain their original hardware and measurement scopes.

The maintained fork is <https://github.com/a3ahmad/vmecpp>. Its original source
is <https://github.com/proximafusion/vmecpp>. For a fresh clone, retain the latter
as a fetch-only upstream while pushing changes to this fork:

```sh
git remote add upstream https://github.com/proximafusion/vmecpp.git
git remote set-url --push upstream disabled://read-only
git config remote.pushDefault origin
```

These tools run against an existing pinned ConStellaration oracle installation.
They do not install into that environment or write into the stella repository.
Results belong under the ignored `build/optimization` directory. Set STELLA_ROOT to your separate ConStellaration checkout before using the examples.

See the [one-page current findings](ONE_PAGE.md) for measured results, exactness,
resource/accounting limits and remaining profiling work.

The [current A100 kernel profile](A100_PROFILE.md) passed original-output checks
with zero drift. It identifies scan/certification and Fourier work, frequent
host synchronization, and small host/device transfer volume. It supplies no
new speedup or roofline percentage.

The [36-call native MPS pilot](THROUGHPUT.md) measured2.57-2.60x six-solve
throughput with exact outputs. Individual VMEC calls slowed2.68-3.47x. Resource
status remains unqualified, and deployment needs launcher/accounting integration;
this is separate from per-solve acceleration and is not compounded below.

The [current A100 P3 pilot](A100_RAGGED.md) passed14 actual evaluations with
zero drift. Its five measured pairs gave1.052811x VMEC speedup for CLEAN ragged
versus selected; consistent full fresh-call gains remain unproven. This
single-fixture GPU comparison stays separate from the primary six-case table.

The [energy-batching report](ENERGY_BATCHING.md) now leads with a 72-call native
GPU-versus-GPU study: 1.0296-1.0338x VMEC paired medians across six cases, with
zero drift and 30/30 measured pairs improved. This roughly 3% incremental result
uses the clean source rebuild; its scope, earlier local study and roofline
limitations remain explicit. The original-CPU comparison below is unchanged;
these ratios are not compounded.

## Qualified SparseSeed + N3 Forward build

The latest direct stock CPU versus selected GPU campaign executed 72 evaluations
on Threadripper 2950X / RTX 3090: **13.736949-14.284143x for VMEC calls**,
**8.181490-11.220618x for the post-import adapter body**, and
**5.561327-7.312858x for the outer fresh call**. All original metrics, canonical
identities, decks, raw numerical fields, diagnostic hashes and repeats match at
zero tolerance. These are direct same-host measured ratios, not compounded
historical gains. See the [primary six-case table and intervals](../../build/optimization/native-originalcpu-seed72-qualified/primary-table.md),
[per-metric drift](../../build/optimization/native-originalcpu-seed72-final-report/per-metric-drift.csv)
and [separate timer scopes](../../build/optimization/native-originalcpu-seed72-timing-scopes/tables.md).

The following incremental campaign remains separate from that direct comparison.
The SparseSeed + N3 native comparison measured **1.042018-1.051835x outer fresh-call speedup**
over qualified Prefetch8 for the combined SparseSeed + N3 Forward build. All six
paired 95% intervals exceed 1, and all 69 numeric case/metric entries have zero
absolute/relative drift; three null QI entries remain null. This is a combined
incremental GPU comparison, with no individual-feature or multiplied CPU ratio.
See the [six-case table](../../build/optimization/native-seed-forward72-final-report/table.md),
[per-metric CSV](../../build/optimization/native-seed-forward72-final-report/per-metric-drift.csv)
and [72-record independent audit](../../build/optimization/native-seed-forward72-final-report/audit.json).

The newer [ordered-thermal experiment](THERMAL_ORDERED.md) is a separate
A100 GPU-versus-GPU comparison with zero drift in 74 actual evaluations.
Its VMEC medians improve, but consistent fresh-call gains remain unproven.
It does not change the selected build or compound the direct CPU ratios above.

That outer timer includes lazy ConStellaration imports inside `run_official`.
Earlier descriptions claiming it excluded imports were incorrect. The same
executed reports separately measure the **post-import adapter body (first solve
in a fresh process)**: 1.060453-1.085076x derived median ratios. Instrumented
VMEC-call ratios are 1.102255-1.110264x. These are distinct scopes of the same
executions, with no new runs or changed original numeric tables. The body timer
still includes settings, first-use costs, metrics and artifact work within its
bookends. Outer-minus-inner also includes post-body metadata/digests and call
overhead, so it does not measure imports alone. See [separate scope tables and
bookends](../../build/optimization/native-seed-forward72-timing-scopes/tables.md)
and [all-72 frozen-source audit](../../build/optimization/native-seed-forward72-timing-scopes/audit.json).

Use the immutable source snapshot below. Its backend/header/shim match frozen
`resident-solver-warp-seeds2`; the linked complete resident aggregate contains
the qualified N3 Forward kernels and unchanged companion modules. Original CPU
convergence/damping control remains. Live GPU control, allocation pool, GIL shim
and solve queue are OFF. Physics, modes, tolerances and iteration schedules are
unchanged. The core remains the matching exported v0.4.11 reference extension.

From WSL, with the existing frozen prerequisites and a new output directory:

```bash
task_root=$(git rev-parse --show-toplevel)
task_opt="$task_root/build/optimization"
task_out="$task_opt/reproduce-seed-forward"
"$task_opt/tools-env/bin/python" \
  "$task_opt/resident-solver-seed-forward1/source/build_solver.py" \
  --source "$task_opt/source-v0.4.11" --out "$task_out" \
  --cuda /usr/local/cuda-12.9 \
  --reference-extension "$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  --resident "$task_opt/forward-groups1/full-aggregate/resident.so" \
  --iteration "$task_opt/rhs-memory-policy/policy3/rhs.so" \
  --cache-factors --thomas-prefix \
  --thomas-prefetch-forward 8 --thomas-prefetch-backward 8 \
  --lattice-residuals --lattice-energies --coordinate-captures \
  --warp-lattice-seeds
```

The strict FP64 SM80/SM120 build disables FMA contraction. Use the complete
aggregate above, which preserves all 17 public GPU entry points; the component
`split/forward.so` alone is insufficient. Authoritative [build commands](../../build/optimization/resident-solver-seed-forward1/commands.json),
[source hashes](../../build/optimization/resident-solver-seed-forward1/source-hashes.json),
[source attribution](../../build/optimization/seed-forward-qualification1/build-attribution.json)
and [aggregate companion/export proof](../../build/optimization/forward-groups1/full-aggregate/proof.json)
bind the inputs. Rebuilds or relocated copies require fresh parity gates before
timing. No qualified binary needs to be rerun to review the existing evidence.

For one evaluation, clear inherited experiments and set exactly these nine
`VMECPP_` runtime flags. Keep VMEC/OMP/BLAS at 1. No tracing or observer is enabled:

```bash
task_clean_env=()
while IFS= read -r task_env_name; do
  case "$task_env_name" in
    VMECPP_*|NSYS_*|NVPROF_*|NV_COMPUTE_SANITIZER_*)
      task_clean_env+=(-u "$task_env_name") ;;
  esac
done < <(compgen -e)
taskset -c 0 env "${task_clean_env[@]}" \
  -u LD_PRELOAD -u CGA_PROFILE -u CUDA_INJECTION64_PATH -u NVTX_INJECTION64_PATH \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$task_root" \
  LD_PRELOAD="$task_out/solver-shim.so" \
  VMECPP_FORWARD_REFERENCE_SO="$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  VMECPP_GPU_SOLVER=1 VMECPP_GPU_SOLVER_RESIDENT_CONTROL=1 \
  VMECPP_GPU_RESIDUAL_ROUNDS=32 VMECPP_GPU_RESIDUAL_GRAPH=1 \
  VMECPP_GPU_DIAGNOSTICS_ROUNDS=16 VMECPP_GPU_DIAGNOSTICS_GRAPH=1 \
  VMECPP_GPU_SOLVER_GRAPH=1 VMECPP_GPU_SOLVER_PINNED_SCALARS=1 \
  "$task_opt/gpu-env/bin/python" -m tools.optimization.benchmark \
  --stella "$STELLA_ROOT" \
  --seed "$task_opt/fixtures-frozen/public_p3_seed.json" \
  --track P3 --out "$task_opt/reproduce-seed-forward-evaluation"
```

The 32/16 proposal controls are implementation settings, not VMEC tolerances.
The measured native configuration used CPU 0 / Threadripper 2950X / RTX 3090,
VMEC/OMP/BLAS 1 and exactly the nine flags above. Its [candidate configuration](../../build/optimization/native-seed-forward-qualification-checkpoints/c072-complete-1793804746313461077/candidate-config.json)
pins 707 dependencies; [postflight](../../build/optimization/native-seed-forward72-final-report/postflight-final.json)
checks all 711 dependency/launch inputs and every actual flag/affinity/thread
record. Only the single shim is preloaded. [Local complete qualification](../../build/optimization/seed-forward-qualification1/final-summary.json)
and [native cross-host qualification](../../build/optimization/native-seed-forward-crosshost-results2/independent-audit.json)
include original identities, actual decks, raw outputs and repeated trajectories.

The earlier [stage diagnostic](../../build/optimization/native-prefetch8-qualification-payload/stage-pair-completed.json)
is separate and cannot establish this combined build's paired stage speedup.
The whole-solver timeline predates this build. Matched N3 calibration is one
cache-warm ns99 component scope against empirical references, not a whole-solver
or absolute-peak roofline result; see [ROOFLINE.md](ROOFLINE.md). Historical
comparisons below remain separate. [Full evidence and limits](RESULTS.md).

## Qualified Prefetch8 build

This historical clean native campaign measured **1.020551-1.030842x additional adapter
improvement over qualified Thomas row1**, with all six paired 95% intervals above
no change and zero numeric metric drift. This is an incremental GPU comparison;
do not multiply it into earlier campaigns. See the
[six-case table](../../build/optimization/native-prefetch8-qualification72-final-report/table.md),
[per-metric CSV](../../build/optimization/native-prefetch8-qualification72-final-report/per-metric-drift.csv)
and [actual-artifact audit](../../build/optimization/native-prefetch8-qualification72-final-report/audit.json).

Use this immutable source snapshot, not the working builder. It copies the
qualified Thomas backend/header/shim and changes only helper/header/builder
inputs for prefetch. Original CPU convergence/damping control remains active;
the experimental live GPU controller, allocation pool, GIL shim and solve queue
are OFF. Physics settings, modes, tolerances and iteration schedules are unchanged.

With the existing pinned v0.4.11 core, matching exported extension, complete
profile aggregate and RHS policy3 prerequisites, run from WSL:

```bash
task_root=$(git rev-parse --show-toplevel)
task_opt="$task_root/build/optimization"
task_out="$task_opt/reproduce-prefetch8"
"$task_opt/tools-env/bin/python" \
  "$task_opt/resident-solver-thomas-prefetch1/source/build_solver.py" \
  --source "$task_opt/source-v0.4.11" --out "$task_out" \
  --cuda /usr/local/cuda-12.9 \
  --reference-extension "$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  --resident "$task_opt/profile-tiles-full1/resident.so" \
  --iteration "$task_opt/rhs-memory-policy/policy3/rhs.so" \
  --cache-factors --thomas-prefix \
  --thomas-prefetch-forward 8 --thomas-prefetch-backward 8 \
  --lattice-residuals --lattice-energies --coordinate-captures
```

The output directory must be new. The strict FP64 SM80/SM120 build disables FMA
contraction. Prefetch defaults remain row1; the explicit 8 values select early
immutable loads without changing the selected rounded recurrence or certificate.
Authoritative [commands](../../build/optimization/resident-solver-thomas-prefetch1/commands.json),
[source hashes](../../build/optimization/resident-solver-thomas-prefetch1/source-hashes.json)
and [frozen-input proof](../../build/optimization/thomas-prefetch1/solver-inputs-proof.json)
identify every input. Rebuilds or relocated copies require fresh exact-parity
checks before timing.

For one evaluation, clear inherited experiments and set exactly these nine
`VMECPP_` runtime flags. Keep VMEC/OMP/BLAS at 1. No tracing or observer is enabled:

```bash
task_clean_env=()
while IFS= read -r task_env_name; do
  case "$task_env_name" in
    VMECPP_*|NSYS_*|NVPROF_*|NV_COMPUTE_SANITIZER_*)
      task_clean_env+=(-u "$task_env_name") ;;
  esac
done < <(compgen -e)
taskset -c 0 env "${task_clean_env[@]}" \
  -u LD_PRELOAD -u CGA_PROFILE -u CUDA_INJECTION64_PATH -u NVTX_INJECTION64_PATH \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$task_root" \
  LD_PRELOAD="$task_out/solver-shim.so" \
  VMECPP_FORWARD_REFERENCE_SO="$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  VMECPP_GPU_SOLVER=1 VMECPP_GPU_SOLVER_RESIDENT_CONTROL=1 \
  VMECPP_GPU_RESIDUAL_ROUNDS=32 VMECPP_GPU_RESIDUAL_GRAPH=1 \
  VMECPP_GPU_DIAGNOSTICS_ROUNDS=16 VMECPP_GPU_DIAGNOSTICS_GRAPH=1 \
  VMECPP_GPU_SOLVER_GRAPH=1 VMECPP_GPU_SOLVER_PINNED_SCALARS=1 \
  "$task_opt/gpu-env/bin/python" -m tools.optimization.benchmark \
  --stella "$STELLA_ROOT" \
  --seed "$task_opt/fixtures-frozen/public_p3_seed.json" \
  --track P3 --out "$task_opt/reproduce-prefetch8-evaluation"
```

The 32/16 proposal controls are implementation settings, not VMEC tolerances.
The measured native campaign used CPU 0 on Threadripper 2950X / RTX 3090; local WSL
timings are not substitutes for those measurements. Its exact relocated
[candidate configuration](../../build/optimization/native-prefetch8-qualification-checkpoints/c072-complete-1793796533563879616/candidate-config.json)
pins 504 dependencies and the single-shim preload order. The
[stage diagnostic](../../build/optimization/native-prefetch8-qualification-payload/stage-pair-completed.json)
is separate from paired timing. The current whole-solver profile predates
Prefetch8; near-roofline performance remains unproven. [Limits and evidence](RESULTS.md).

## Qualified resident14 build

The original CPU-paired result is resident14:4.43-5.83x measured adapter speedup,
zero drift. See [SUMMARY.md](SUMMARY.md) and the
[72-run audit](../../build/optimization/solver14-qualification72-final-report/audit.json).
Use the immutable resident14 source snapshot for exact reproduction; the working
sources also contain newer default-off experiments. Required prerequisite
components are the pinned v0.4.11 core/exported reference extension, frozen
resident-device-inverse library, and factor1 iteration library. Their build
commands/source/dependency hashes are retained in their experiment directories;
the sections below describe building these isolated components.

From WSL, with these existing prerequisites:

```bash
task_root=$(git rev-parse --show-toplevel)
task_opt="$task_root/build/optimization"
task_out="$task_opt/reproduce-resident14"
"$task_opt/tools-env/bin/python" \
  "$task_opt/resident-solver14-cooperative-energies/source/build_solver.py" \
  --source "$task_opt/source-v0.4.11" --out "$task_out" \
  --cuda /usr/local/cuda-12.9 \
  --reference-extension "$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  --resident "$task_opt/resident-device-inverse/resident.so" \
  --iteration "$task_opt/resident-iteration-factor1/iteration.so" \
  --cache-factors --cooperative-residuals --cooperative-energies
```

The output directory must be new. This builds the strict FP64 SM80/SM120 backend,
shim and replay tool; it changes no oracle installation. The authoritative
[frozen commands](../../build/optimization/resident-solver14-cooperative-energies/commands.json)
and [native runtime configuration](../../build/optimization/solver14-qualification72-plan/candidate-config.json)
include all source/dependency hashes and the relocated native library paths.
Run a fresh correctness check before using a rebuilt/relocated binary.

For a single local evaluation, keep VMEC/OMP/BLAS at1 and use these exact flags:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$task_root" \
  LD_PRELOAD="$task_out/solver-shim.so" \
  VMECPP_FORWARD_REFERENCE_SO="$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  VMECPP_GPU_SOLVER=1 VMECPP_GPU_SOLVER_RESIDENT_CONTROL=1 \
  VMECPP_GPU_RESIDUAL_ROUNDS=32 VMECPP_GPU_RESIDUAL_GRAPH=1 \
  VMECPP_GPU_DIAGNOSTICS_ROUNDS=16 VMECPP_GPU_DIAGNOSTICS_GRAPH=1 \
  VMECPP_GPU_SOLVER_GRAPH=1 VMECPP_GPU_SOLVER_PINNED_SCALARS=1 \
  "$task_opt/gpu-env/bin/python" -m tools.optimization.benchmark \
  --stella "$STELLA_ROOT" \
  --seed "$task_opt/fixtures-frozen/public_p3_seed.json" \
  --track P3 --out "$task_opt/reproduce-resident14-evaluation"
```

The32/16 values govern certified implementation proposals, not VMEC force
convergence tolerances. Ordinary cooperative energies use their frozen max32;
refresh diagnostics retain16. Every failed certificate selects the original
ordered result. Actual INDATA must match the original reference exactly.
This recipe does not enable outer batching, GIL release, lattice energies,
RHS-prefetch or allocation reuse; those have separate experimental evidence.

## Measured frozen22 incremental build

Frozen22 adds1.1048-1.2075x measured adapter speedup over14 on all six cases,
with zero metric drift. All72 actual measurements passed independent raw-output,
identity, configuration and repeat audits. The supervisor's reporter import
failed after measurements; the separate
[post-hoc data audit](../../build/optimization/native22-vs14-posthoc-report/posthoc-data-qualification.json)
preserves that failed orchestration status. This is an incremental comparison;
do not multiply its ratios into the earlier CPU table.

Use the immutable22 source snapshot and complete profile aggregate, RHS policy3
library and matching exported extension already built in this workspace:

```bash
task_root=$(git rev-parse --show-toplevel)
task_opt="$task_root/build/optimization"
task_out="$task_opt/reproduce-resident22"
"$task_opt/tools-env/bin/python" \
  "$task_opt/resident-solver22-full-profile/source/build_solver.py" \
  --source "$task_opt/source-v0.4.11" --out "$task_out" \
  --cuda /usr/local/cuda-12.9 \
  --reference-extension "$task_opt/gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so" \
  --resident "$task_opt/profile-tiles-full1/resident.so" \
  --iteration "$task_opt/rhs-memory-policy/policy3/rhs.so" \
  --cache-factors --lattice-residuals --lattice-energies --coordinate-captures
```

Use the same single-evaluation environment flags shown above with this new
`task_out` and a new evaluation output directory. Frozen22 keeps the CPU scalar
controller, original refresh diagnostics and all physics settings. Signed energy
and residual proposals retain complete original-rounding certificates/fallbacks;
RHS prefetch and profile tiling preserve original arithmetic order. Allocation
pooling and the new live GPU damping prototype are separate, unmeasured changes.

The actual executed
[build commands](../../build/optimization/resident-solver22-full-profile/commands.json),
[source hashes](../../build/optimization/resident-solver22-full-profile/source-hashes.json)
and relocated native candidate configuration in the verified terminal checkpoint
are authoritative. Rebuild/relocation requires fresh exact-parity checks.
The [solve-queue recipe](../../build/optimization/native-queue18-results/RUN_BUILD_CONFIG_RECIPE.txt)
is separate and pins solver18; frozen22 plus queueing has not been measured.

## Official evaluations

Run from WSL using the oracle interpreter. Set `PYTHONDONTWRITEBYTECODE=1` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` (export each variable).

```bash
~/.cga/envs/oracle_v026/bin/python tools/optimization/benchmark.py \
  --stella "$STELLA_ROOT" \
  --seed "$STELLA_ROOT/artifacts/seeds/public_p3_seed.json" \
  --track P1 --out build/optimization/example
```

Output directories must not exist. The harness records actual solver input,
extension checksums, canonical request/metrics hashes, metrics, selected output
array digests, solver timing, and adapter timing. The outer adapter timer excludes
Python startup and imports completed before its bookend, but includes lazy
imports inside `run_official`. The saved `result.wall_seconds` is a separate
post-import adapter-body timer with different bookends. `suite.py` also records
child-process wall time.
Input capture is diagnostic overhead and occurs before the solver timer.

`suite.py --stella PATH --out NEW_DIRECTORY --samples 6` runs the three public
seeds on P1/P3 serially. Sample zero is a warmup. Discovery runs under contention
are suitable for convergence/parity checks, not the main performance table.
The initial suite does not replace failed fixtures automatically.

`summarize.py DIRECTORY --reference BASELINE_DIRECTORY` compares successful
reports by seed and track. Its diagnostic median explicitly includes warmups;
do not use that median as the final benchmark. A matching request/metrics hash
is a local reproducibility check, not a replacement for the production result
identity calculation. Changed build provenance must be retained separately.

## GPU calibration

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 tools/optimization/roofline.cu \
  -o build/optimization/roofline
build/optimization/roofline
```

Select the actual target architecture on another GPU. The streaming working set
is 768 MiB; the compute calibration uses eight independent FP64 FMA chains.
Reported FLOPs count the FMA loop only. Logical streaming bytes count two reads
and one write; obtain physical traffic from profiler counters separately.

CUDA event timings cover the launch sequence, including device idle gaps.
The additional CUDA Graph replay measurement tests launch-gap sensitivity.
These measurements are calibration experiments, not solver speedups. Run them
without other GPU work before treating them as empirical hardware ceilings.

```bash
nsys profile --trace=cuda,nvtx --sample=none \
  --output=build/optimization/timeline build/optimization/roofline
/usr/local/cuda/bin/ncu --kernel-name regex:Stream --launch-count 1 \
  --set roofline --export build/optimization/kernel build/optimization/roofline
```

Never use application wall time from an Nsight Compute replay run as a speedup.
Keep raw profiler reports and unprofiled timings separate.

## Validation and leasing

```bash
python -m unittest tools.optimization.test_summarize
ruff check tools/optimization
pre-commit run --files tools/optimization/benchmark.py \
  tools/optimization/suite.py tools/optimization/summarize.py \
  tools/optimization/test_summarize.py tools/optimization/README.md
```

`benchmark.py` never creates a lease. `lease.py create` explicitly creates one
pilot and starts an independent expiry watchdog. Credentials are read from
environment variables only. Before creating a lease, prepare and validate the experiment
bundle, establish an independent expiry watchdog, and reserve retrieval time
within the lease limit. Record the instance identifier and billing rate, copy
and checksum results, destroy the instance, and verify removal of billable
storage. The authorized total cap is USD 50, with a USD 25 target and the remainder
held as a buffer. The lifecycle tool selects offers below USD 1/hour and defaults
to two-hour expiry, with one active instance. The active A100 matrix received
one recorded extension to three hours based on measured solve durations.

## Reproduce the CPU candidate

Start with the upstream `v0.4.11` tag in an isolated clone. Preserve the installed
reference extension and its checksum. Apply `patches/transform-locality.patch`
and `patches/stage-timers.patch` to that clone. Build/install into a separate
Python 3.12 environment using the upstream release defaults, without replacing
the oracle installation or upgrading its dependencies. The measured candidate
uses both patches, with `VMECPP_PROFILE_STAGES` absent during timing runs.

The locality patch moves independent toroidal output slices outside the poloidal
loop; each output retains its original m/n accumulation order. No fast-math,
resolution, tolerance, convergence, or threading change is included.

`paired.py` runs baseline/candidate serially with alternating order, fixed CPU
placement, one excluded warmup pair, and five measured pairs per fixture/track.
It checks all metrics and numerical wout fields after each pair. `report_pairs.py`
also checks repeat determinism and provides paired-bootstrap intervals. The
coordinator was paused once for separate diagnostics; the affected coordinator
process-wall sample is excluded as recorded in `build/optimization/pause-audit.json`.
Its child finished before profiling began, so adapter/solver timings remain valid.

## Experimental GPU candidate

`prepare_gpu.py` wires the symmetric inverse Fourier transform into the isolated
0.4.11 clone using the existing repository CUDA kernel as a starting point. It
removes unnecessary atomics and retains serial m/n accumulation. Compile the
generated `gpu_fourier.cu` with CUDA 12.9, C++20, `-O3 --fmad=false`, PIC, and
native targets `sm_80` and `sm_120`. Configure the clone with
`VMECPP_GPU_FOURIER_EXPERIMENT=ON` and a stable `PYTHON_EXECUTABLE` path, then
build `_vmecpp`. Keep the resulting extension in a separate environment.

Only `VMECPP_GPU_FOURIER=1` enables offload. The experiment covers the supplied
3D stellarator-symmetric fixtures; remaining solver stages execute on CPU.
CUDA errors propagate as exceptions, rather than silently falling back inside
the experimental transform. Full-solve parity and measured timeline evidence
are required before interpreting any microbenchmark benefit as a solver gain.

## Historical A100 results and followup experiments

[RESULTS.md](RESULTS.md) contains the completed A100 and CPU conclusions,
per-metric drift, profile evidence, and limits. `FINDINGS.md` is historical.
The completed CPU locality campaign has ten measured pairs per case; the
broader four-worker pointwise campaign has five. Both completed A100 campaigns
also have five. All task leases were destroyed with teardown verified; final
fused-campaign retrieval checksum-verified 850 files.

The original inverse-only A100 table predates the following separable and fused
stages. That earlier mixed table combines them with the four-worker CPU candidate.
For a separable GPU transform, generate the dense header with
`prepare_flat_kernel.py`, then pass it to `prepare_separable_kernel.py --out DIR`.
Compile `gpu_fourier.cu` with that directory on the include path and the flags
`-DVMECPP_GPU_FLAT_GRID -DVMECPP_GPU_BLOCK_SIZE=32 -DVMECPP_GPU_SEPARABLE
-DFourierToRealKernel=FourierToRealKernelSeparable`. Retain strict FP64 flags
and the original kernel attribution. A shared wrapper library can interpose
`RunGpuFourier` via `LD_PRELOAD` in a separate process. Record its hash as well
as the extension hash; never substitute it silently in the completed campaign.
`gpu_replay.cc` compares every output byte against the real captured call before
and after repeated execution. Whole-solver and repeated fixture checks remain
necessary because a single captured call is insufficient qualification.

`parallel_fourier.py SOURCE` adds opt-in parallelism over independent radial
transforms in the isolated source. Build with OpenMP, set
`VMECPP_DETERMINISTIC_FOURIER_THREADS=4` and `OMP_MAX_ACTIVE_LEVELS=2`, and keep
the actual solver `max_threads=1`. Do not pin this experiment to only one CPU.
The per-output summation order stays serial; parity must still be demonstrated.

`persistent_fourier.py SOURCE` replaces those nested OpenMP launches with the
shared `radial_executor.h` worker pool. Set the thread count once per process;
the solver still uses `max_threads=1`. `radial_executor_test.cc` exercises
dispatch, exception propagation, recovery, and shutdown. `parallel_pointwise.py`
then distributes disjoint angular MHD-force slices and radial force-assembly
outputs across that same pool. Radial force history remains serial within each
slice, and scratch-array updates touch only the slice owned by that worker.
It also covers independent geometry-metric slices, magnetic-field components,
and differential-volume rows, retaining each row's angular summation order.
The latest completed mixed A100 table uses both persistent and pointwise changes.
The original inverse-only A100 table uses neither.

`build_forward.py --source SOURCE --out NEW_DIR --cuda /usr/local/cuda-12.9
--pipeline` builds an isolated MHD/constraint/forward prototype. It snapshots
its source, commands, and binary hashes. The source must be the matching
v0.4.11 tree with its CMake dependency headers; a portable header manifest and
`--force-point-header` can instead supply those already generated inputs.
Set `LD_PRELOAD=NEW_DIR/forward-shim.so` and
`VMECPP_FORWARD_REFERENCE_SO` to the absolute experimental extension path.
For CPU-reference capture, set `VMECPP_MHD_CAPTURE` and
`VMECPP_PIPELINE_CAPTURE` to distinct output files. Run `mhd-replay` and
`pipeline-replay` on those captures with the shim path as their second argument.
For the fused solver path set `VMECPP_GPU_FORCE_PIPELINE=1`, with capture flags
unset. This prototype only supports the fixed-boundary, stellarator-symmetric,
single-partition scope checked at entry, not arbitrary checkpoint APIs.

`VMECPP_GPU_UPLOAD_MODE=packed` is the original frozen prototype behavior.
`direct` and `batch` test uploads without the explicit host packing copy;
batch uses the CUDA 12.9 batched-copy API. These modes require independent
replay and whole-solver validation. Runtime configuration and preload hashes
are captured in provenance. Do not modify a frozen library during a campaign.

For formal comparisons, `paired.py --candidate-cpus 0,2,4,6 --cpu 0` allows the
candidate to use four selected CPUs while keeping the baseline on CPU 0. Record
the topology, actual thread count, extension hash, and any preload-library hash.
`benchmark.py` captures the selected experimental environment and preload hashes.

`thread_probe.py --threads 2` accepts the other `benchmark.py` arguments. It
records an explicit runtime override, since the official adapter hardcodes
single-threaded VMEC. Its request hash alone must not be treated as proof that
the effective runtime configuration equals the official request. Compare
numeric metrics and every captured numerical field; label any drift as an
alternative configuration, outside the main table.

The AVX2 build is a separate experiment. Applying incompatible Eigen alignment
settings across the core and binding caused an isolated trial to abort; that
configuration is rejected. Consistent SIMD flags preserve metrics on one smoke
fixture but change the `wdot` diagnostic history, so it has not been promoted.

## Resident iteration prototype

This is a separate, incomplete GPU solver port. It is not the binary used for
either completed main table. `build_resident.py` snapshots its inputs and builds
strict FP64 geometry, field, pressure, lambda-force, and MHD-force stages.

```bash
python tools/optimization/build_resident.py \
  --source build/optimization/source-v0.4.11 \
  --out build/optimization/new-resident-build \
  --cuda /usr/local/cuda-12.9 \
  --inverse-kernel build/optimization/gpu-separable/generated_fourier_kernel.cuh
```

Use an isolated, matching v0.4.11 extension with exported CPU methods for
reference capture. Set `LD_PRELOAD=NEW_BUILD/resident-capture.so`,
`VMECPP_FORWARD_REFERENCE_SO` to that extension, and
`VMECPP_RESIDENT_CAPTURE=NEW_OUTPUT/fields.bin` for one ordinary CPU evaluation.
The shim always calls the original CPU methods. It records a real final-grid
input/output state plus `.constants.bin` and `.constraints.bin` sidecars.
Verify that the complete evaluation still matches the original oracle first.
The force-constant sidecar captures the actual mass and radial-blending profiles;
the resident pressure specialization rejects gamma values other than the
fixture's unchanged zero value.

`resident-chain-replay inverse.bin fields.bin NEW_BUILD/resident.so` compares
38 arrays after the inverse-to-field chain. It rejects captures with different
intermediate geometry and sets host geometry pointers to null before execution.
`resident-mhd-replay inverse.bin fields.bin fields.bin.constants.bin mhd.bin
NEW_BUILD/resident.so` additionally compares 12 MHD forces, four lambda forces,
and total pressure. Obtain `inverse.bin` with the existing inverse capture and
`mhd.bin` with the CPU `VMECPP_MHD_CAPTURE` adapter described above. Matching
shared states are checked before the captures can be joined.

Run Compute Sanitizer memcheck and retain the raw Nsight Systems timeline.
Reference adapters deliberately download outputs at the final comparison
boundary; their wall times are not production solver speedups.
`LaunchGpuResidentFields` and `LaunchGpuResidentForces` operate directly on
device pointers and their shared stream. `resident-force-eval-replay inverse.bin
fields.bin fields.bin.constants.bin fields.bin.constraints.bin pipeline.bin
NEW_BUILD/resident.so` covers the connected constraint/force/forward region.

Build the separate iteration-stage library against the same exported reference
extension (the reference shim links its Abseil dependency from that library):

```bash
python tools/optimization/build_iteration.py \
  --source build/optimization/source-v0.4.11 \
  --out build/optimization/new-iteration-build \
  --cuda /usr/local/cuda-12.9 \
  --reference-extension /absolute/path/to/matching/_vmecpp.cpython-312-x86_64-linux-gnu.so
```

For reference capture preload `iteration-capture.so`, set
`VMECPP_FORWARD_REFERENCE_SO` to the same extension, and set distinct output
paths in `VMECPP_ITERATION_CAPTURE`, `VMECPP_STEP_CAPTURE`,
`VMECPP_DIAGNOSTICS_CAPTURE`, and `VMECPP_PREPARATION_CAPTURE`.
The original CPU methods always execute. Verify complete evaluation hashes
before accepting the reference captures.

`iteration-replay`, `step-replay`, `diagnostics-replay`, and
`preparation-replay` each take `CAPTURE.bin NEW_ITERATION_BUILD/iteration.so`.
They compare every output byte on every one of 110 executions. Their allocation,
download, and comparison timings are not production performance numbers.

`NEW_BUILD/resident-iteration-replay` takes the six force-evaluation arguments
above, followed by `preparation.bin iteration.bin iteration.so` and optionally
`step.bin`. It first verifies that all captured intermediate states match.
The connected callback consumes only GPU-owned force pointers, retaining the
same stream through preparation, preconditioning, residuals, and coefficient
evolution. Each replay restores captured initial step arrays on device for
comparison; this is not a full solver trajectory.
`VMECPP_PRECONDITIONER_CAPTURE=FILE` additionally captures a real radial/lambda
refresh, constraint multiplier, and assembled matrices. Check it with
`preconditioner-replay FILE NEW_ITERATION_BUILD/iteration.so`. Append
`preconditioner.bin diagnostics.bin` after the step argument to connect both
stages to the physics-body replay. It checks shared real inputs before joining
and compares all 20 refresh arrays and seven diagnostic values after the GPU
coefficient step. Persistent geometry preparation/iteration state, multigrid,
and solver control remain before full-solve qualification.

`VMECPP_GPU_RESIDUAL_ROUNDS=12 VMECPP_GPU_RESIDUAL_GRAPH=1` enables a separate
certified optimistic-accumulation experiment in the iteration library. Every
prefix must satisfy the original rounded recurrence bit for bit; otherwise the
original ordered kernel supplies that component's answer. Nonfinite proposals
also fall back. This is not enabled in the qualified full-evaluation tables.
`residual-benchmark iteration.bin iteration.so before` (or `after`) uploads one
real input once, excludes 10 warmups, times 100 calls with CUDA events, and
checks all 110 results after timing. It reports fallback counts. Run randomized
independent processes and preserve the selected environment and binary hashes.

For expensive sanitizer instrumentation only, the connected replay accepts
`VMECPP_REPLAY_WARMUPS=0 VMECPP_REPLAY_REPEATS=2`, giving three checked executions
including its initial call. Reports print the actual counts. Retain the default
full-repeat ordinary parity run separately; a timed-out sanitizer is not a pass.

## Reproduce the historical qualified mixed configuration

Use the isolated v0.4.11 source and pinned Python dependency set. Apply the CPU
locality/timer patches and GPU wiring described above, then apply, in order,
`parallel_fourier.py SOURCE`, `persistent_fourier.py SOURCE`, and
`parallel_pointwise.py SOURCE`. Build into a new environment. Build the separable
inverse wrapper and fused `build_forward.py --pipeline` library as above.
The immutable measured fused sources/commands are preserved in
`build/optimization/a100-fused-results/fused/`; its source snapshots take
precedence over later experimental edits when reproducing that campaign.

Launch the candidate with both wrapper libraries in LD_PRELOAD,
VMECPP_GPU_FOURIER=1, VMECPP_GPU_FORCE_PIPELINE=1,
VMECPP_DETERMINISTIC_FOURIER_THREADS=4, and VMECPP_FORWARD_REFERENCE_SO pointing
at its own extension. Keep OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and
MKL_NUM_THREADS at one. Leave graph, parallel-packing, later CPU preconditioner,
and capture flags unset; upload mode is default packed. Retain all binary
hashes and use a separate executable wrapper for the candidate environment.

Run `paired.py --baseline-python BASELINE_PYTHON --candidate-python
CANDIDATE_WRAPPER --stella STELLA --manifest FROZEN_MANIFEST --out NEW_OUTPUT
--pairs 6 --cpu 0 --candidate-cpus 0,1,2,3`. Report using
`report_pairs.py NEW_OUTPUT/samples.json --out SUMMARY.json`. Six pairs means
one excluded warmup plus five measured pairs for each of six boundary/track
cases. The executed campaign's initial and resume commands are in
`a100-fused-results/combined-matrix-command.json` and
`extended-matrix-command.json`; per-run provenance records all effective
settings and library hashes. The resident iteration prototype is absent.

## Experimental persistent solver with original CPU control

The first complete public P3/P3 solve passed zero-tolerance output checks in
`resident-solver1/smoke/validation.json`. This intermediate solver retains CPU
scalar control and geometry/velocity mirrors; it is not fully GPU controlled
and has no qualified speedup table yet.

Build using `build_solver.py --source SOURCE_V0411 --out NEW_SOLVER_BUILD
--cuda /usr/local/cuda-12.9 --reference-extension MATCHING_EXTENSION
--resident DEVICE_INVERSE_RESIDENT/resident.so --iteration ITERATION17/iteration.so`.
The resident dependency must export LaunchGpuFourierDevice; the iteration
dependency must export LaunchGpuEnergies. The build snapshots sources, commands,
and dependency hashes. `NEW_SOLVER_BUILD/solver-replay OPTIMIZATION_ROOT` checks
fresh caches, diagnostics, residuals, repeated state, and the captured step.

For a complete solve, preload `NEW_SOLVER_BUILD/solver-shim.so`, set
`VMECPP_GPU_SOLVER=1` and `VMECPP_FORWARD_REFERENCE_SO=MATCHING_EXTENSION`, and
use the isolated matching Python environment. Keep OMP/BLAS/MKL threads at one.
Use `validate_resident_solver.py` for repeated strict correctness checks;
its automatic trace proves that GPU updates actually execute. Trace timing
is diagnostic only. The six-case mode checks frozen decks and all captured
numerical outputs against the qualified baseline. The separate
`validate_resident_lifecycle.py` changes iteration caps only in isolated copies
to test nonconvergent output materialization; those configurations are explicitly
noncomparable and cannot enter a speedup table.

## CPU controller diagnostics

`cpu_log_observer.py --out NEW_DIRECTORY` builds a diagnostic LD_PRELOAD library
and checks64 CPU cases over four rounding modes. It forwards actual libm `log`,
records input/output bits with caller identity, and preserves errno and floating
point state. Set `VMECPP_CPU_LOG_OBSERVER_PREFIX` only in isolated diagnostic
children. Binary records are Linux x86-64 specific and decoded by `read_records`;
incomplete traces are rejected. These observations are not timing samples.

`cuda_log_probe.cu` compares frozen observed CPU input/output pairs with CUDA
`log` and the original ordered clamp. Its input is little-endian uint64 pairs
(argument bits, original result bits); output pairs are CUDA result and clamp
bits. Compile with `--fmad=false --prec-div=true --prec-sqrt=true`. Two executed
probes on the actual P3/P3 corpus differ on9/13,760 logs and7 clamp outputs,
each by at most1ULP for the raw log. Downstream step identity remains unproven.
See `build/optimization/cuda-log-probe1/comparison.json` for actual evidence.

The optional `VMECPP_CPU_CONTROLLER_TRACE` in newly built diagnostic shims writes
per-process JSONL containing bit patterns for fac, b1, time step, current fsq and
the ten-entry invTau history. Frozen qualified shims remain unchanged. Replay
the complete history before concluding whether a log difference changes a step;
never promote approximate arithmetic from a component-only comparison.

`replay_controller_log.py` reconstructs the original ten-entry history and every
fac/b1 bit before comparing candidate CUDA log results. Both executed P3/P3
traces reconstruct13,778 original steps exactly;70 candidate history states
differ, but no final step coefficient differs. This observation does not qualify
an autonomous controller or other fixtures. Evidence is in
`build/optimization/controller-trace22/replay-18512.json` and the actual-artifact
audit beside it.

The optional `--assumed-log-radius` explores interval screening with observed
CPU-history recovery on ambiguous steps. It is not a proven generic certificate
or implemented runtime fallback. On that trace, assumed radius2 accepts97.844%
of coefficients; this is a feasibility count, not a timing or speedup estimate.
[CUDA12.9.1 Table19](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#standard-functions)
lists1ULP for double log; that does not establish CPU libm identity.
[GNU's math accuracy documentation](https://sourceware.org/glibc/manual/latest/html_node/Errors-in-Math-Functions.html)
does not promise correctly rounded log. Pinned implementation/IFUNC error bounds
and real CPU history recovery remain required for a generic certified path.
## Latest counter evidence

The [native Fourier counter profile](COUNTER_PROFILE.md) completed three protected
captures with original packet checks and independently verified cleanup. FP64
pipe activity was59.77-70.38%; this is not an A100 or whole-solver roofline claim.
