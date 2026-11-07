# Retained GB202 numerical source snapshot

This source-only snapshot preserves the numerical cohort used for the measured
SM120 resident `60d24820102c953dedc85a8487ec3023fdf7de921ff7e82ff25294718309e078`
and backend `2eb80b39e9051961721f8cf9b28c520f970a599d19a8197b8707c86cb52dcf2d`.
It includes the policy3 RHS and original iteration dependency sources. No
binary, fixture, benchmark dataset, raw profiler capture, credential or runtime
environment is included. It is experimental offload code, not an installed
VMEC++ package or a portable replacement wheel.

`sources.json` records every retained source SHA256, original private-build
path relative to the repository, the exact per-translation-unit source closure,
and historical compile commands with `${ORIGINAL_REPO}` placeholders. The
`frozen/` files are **byte-exact** copies; do not autoformat them. Content shared
by several cohorts is stored once. Hash prefixes distinguish different contents
that historically shared a filename. Numerical source edits in this snapshot: 0.

The retained algorithms are packed five-profile backup/restore, original
tile768 residual coordinate scan with approximation fusion and raw-seed guard,
coordinate energy without its later negative seed-guard variant, inverse
register accumulation/load hoist, and forward output-clear elision. Rejected
inverse output-group/angular variants and forward subgroup/cache experiments
are not silently included.

## CPU build recipe

Requirements: Linux, Python3, CUDA12.9 `nvcc` with SM120 support, matching CUDA
headers/CUB and cudart, and g++. No GPU is required to plan or compile. From any
clone location, use an absolute fresh output path:

```sh
python3 tools/optimization/gb202/build.py --out /tmp/vmecpp-gb202-build-001
python3 tools/optimization/gb202/build.py --out /tmp/vmecpp-gb202-build-001 --nvcc /usr/local/cuda-12.9/bin/nvcc --execute
```

The first command validates source hashes and prints a plan without creating
the output directory or invoking compilers/CUDA. The second compiles all 17
numerical TUs for SM120 and links `iteration.so`, `rhs.so`, `resident.so` and
`solver-backend.so`. Only the qualified per-TU strict flags/macros are used;
FMA remains disabled. It reuses the existing bounded CPU compiler owner and
does not force nice levels or CPU affinity. `--cuda-lib` can specify the cudart
library directory if it is not next to nvcc. CUDA library ABI/toolchain selection
remains the caller's responsibility. The recipe does not execute any output.

This reconstructed fresh-source recipe was **CPU-compiled and linked**: all
17 numerical TUs compiled successfully, and all four corrected links passed.
Compile jobs took 65.17 seconds of wall time in total; corrected link jobs took
1.71 seconds. CPU compute time was not recorded. The first backend link failed
because g++ as-needed omitted the original iteration DSO dependency; the
metadata-only correction explicitly retains dependencies and names that same
iteration DSO in the backend link. The original failure remains recorded.
`BUILD_RECEIPT.json` records actual status, output hashes and ELF dependencies.
No output was loaded and no CUDA workload or physical qualification was run.

The measured original build compiled two TUs and pruned already-qualified
SM120 objects for the rest. This recipe preserves their source/flags but
freshly compiles them instead. Changing source/output
locations and linking dependency paths changes metadata and DT_NEEDED paths;
the resulting binary SHA256 values need not match the old measured DSOs.
Inspect new output hashes/dependencies and repeat the existing exact physical
checks before operational adoption. No performance result is implied by a build.

The source snapshot also retains the accepted host shim and its local headers,
but does not compile it: that requires the pinned VMEC++0.4.11 C++ headers,
extension ABI and Eigen/Abseil dependencies. The active numerical TUs are
self-contained except CUDA/system headers. The manifest lists upstream shim
includes and an inactive optional pipeline `mhd_point.h` include; that latter
header is already present in the resident cohort and the pipeline macro is not
enabled by the selected forward compile. The Python oracle, activation shim
owner checks, deck/expected outputs and qualified runtime remain separate
prerequisites. See the operational handoff for their site-bound setup.

## Unselected experiment sources

`frozen/experiments/` preserves byte-exact ASCII source snapshots for inverse
output groups, inverse guarded angular specialization, forward recurrence
subgroups, and forward explicit cache-global loads. `experiments.json` records
their original repository-relative paths, SHA256 values and actual CPU compile/
link commands. Small historical Python preparation/build recipes are retained
as site-bound provenance: they refer to the original private source/object
cohorts and are not standalone portable build commands. These four experiments
are not included in the selected `sources.json` or operational build recipe.
No binary, raw timing/profile data or fixture accompanies these snapshots.
