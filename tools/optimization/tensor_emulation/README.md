# FP64 GEMM emulation experiment

The measured findings are split into two experiments:

- [CUDA/cuBLAS Ozaki emulation](FINDINGS.md).
- [Custom 4/6/8-component FP16/BF16 expansion](EXPANSION_FINDINGS.md), with
  [build and execution instructions](EXPANSION.md).

Neither experiment has established a new identity-preserving Stella solver
speedup. Component accuracy is separate from final equilibrium metric parity.

This standalone benchmark changes no solver code. PLAN is the default. Root
owns the GPU lock and every execution. Separate builds select the exact toolkit
library; each report verifies its requested cuBLAS owner and records versions.

CPU-only build, using fresh output directories:

```sh
/usr/bin/python3 -I -S -B tools/optimization/tensor_emulation/build.py \
  --cuda /usr/local/cuda-13.1 --out build/optimization/tensor-emulation-13.1-build1
/usr/bin/python3 -I -S -B tools/optimization/tensor_emulation/build.py \
  --cuda /usr/local/cuda-13.5 --out build/optimization/tensor-emulation-13.5-build1
```

The build receipt binds source, headers, exact library files and binary SHA256.
Run each toolkit in a separate clean process; do not mix their library paths.
Example root-owned invocation, with a finite external timeout and fresh reports:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="$ACTIVE_WSL_DRIVER_DIR:/usr/lib/wsl/lib:/usr/local/cuda-13.5/lib64" \
  /usr/bin/timeout --signal=TERM --kill-after=8s 60s \
  build/optimization/tensor-emulation-13.5-build1/bench --execute \
  --input build/optimization/tensor-emulation-20261112/mapping/actual3/ns99-inv-toroidal-00.bin \
  --mode all --workspace-mib 8192 --calls 20 --samples 7 --warmups 3 \
  --expect-cublas /usr/local/cuda-13.5/lib64/libcublas.so \
  --report /tmp/FRESH-gemm --output /tmp/FRESH-gemm-output
```

`all` writes six `REPORT.MODE.json` and optional `OUTPUT.MODE.bin` files in one
process. The CUDA context, handle, events, workspace, input allocations and H2D
upload are shared once. Modes run native, dynamic-eager, dynamic-performant,
reduced-4, reduced-8, reduced-12. Single `--mode native` or another named mode
writes the exact report/output paths. `--profile-one` additionally places ONE
consumed, warmed GEMM inside cudaProfilerStart/Stop, after the timing samples.
It does not itself launch a profiler or prove a tensor kernel was dispatched.

Input format, agreed with the capture exporter: eight magic bytes `TEGEMM1\0`,
three little-endian uint64 dimensions M,N,K, then FP64 column-major A(M,K),
B(K,N), and original reference C(M,N). All calls use NN, alpha1, beta0,
ldaM/ldbK/ldcM. Exact file length and finite values are checked. Without input,
deterministic synthetic matrices use an extended-precision CPU dot reference.
Real captures preserve original grouped arithmetic, so even native pedantic
GEMM may differ through reassociation. No solver parity is inferred.

The official [cuBLAS 13.5 emulation documentation](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation)
defines dynamic mantissa control, eager/performant selection and signed bit
offsets. Zero offset targets FP64 accuracy, not the original ordered bit pattern.
Negative offsets are explicitly reduced-accuracy experiments. Eager is a
request, and ADP/unsupported paths may still select native code; actual kernel
profiling is required to identify Ozaki-I/Ozaki-II or fallback.

Classic cuBLAS documents no fixed-point workspace-size query. Its documented
upper bound is 8GB. This benchmark conservatively allocates8192MiB once and
uses cublasSetWorkspace after setting the stream. A smaller explicit workspace
may cause library allocations or native fallback; it is reported, not hidden.
First-call and steady event intervals include every internal ADP conversion,
GEMM and reconstruction for each invocation. Setup/H2D is separate. Host RAW
intervals include submission and synchronization and remain WSL diagnostics.
Events include any host gaps between queued work; they are not kernel busy time.
No timing or accuracy result is claimed before root executes the benchmark.

## WSL driver-library selection

On this host, root's initial process selected a stale Linux580 JIT component
alongside the active Windows621 driver. Successful processes put the actual
Windows-projected driver directory FIRST in their process-local library path.
Set `ACTIVE_WSL_DRIVER_DIR` to the directory verified for the current host;
the observed directory was
`/usr/lib/wsl/drivers/nv_dispi.inf_amd64_cee48bd6d33f95ab`.
Keep `/usr/lib/wsl/lib` and the chosen toolkit library directory after it.
This is a per-process loader choice, not a driver installation, global linker
configuration change, or claim that arbitrary JIT/driver versions can be mixed.
The measured benchmark and build1 binaries were preserved unchanged.

## Fixed-mantissa snapshot v1

`bench_fixed.cu` is a separate source snapshot. Build it with the same recipe
and `--source tools/optimization/tensor_emulation/bench_fixed.cu`, using fresh
`tensor-emulation-13.1-build2` or `tensor-emulation-13.5-build2` output paths.
Its `--mode all` means native, fixed-40, fixed-48, fixed-55, fixed-64, fixed-80.
Each fixed arm sets `CUDA_EMULATION_MANTISSA_CONTROL_FIXED` and its exact
maximum bit count, then verifies both through API readback. Offset is zero,
strategy is eager. Single named modes are also supported. The same context,
workspace, inputs, timers and original-C error comparisons are reused.
Fixed bit count is an experimental accuracy/performance control; error versus
original ordered arithmetic or native GEMM need not decrease monotonically.
Actual kernel profiling remains necessary to identify dispatched arithmetic.
