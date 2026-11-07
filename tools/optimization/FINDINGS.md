# Historical exploratory findings (superseded)

For completed benchmark results and current status, read [RESULTS.md](RESULTS.md).
The text below records earlier discovery work and is not the current lease or
benchmark status.

No comparable speedup is claimed yet. The CPU locality change and integrated
GPU transform both preserve all reported metrics and captured numerical wout
fields on the public P3/P1 smoke fixture. The full paired matrix is running.
Two bounded A100 pilots were created and both were destroyed with teardown
verified. The first stalled during image setup. The replacement was reachable,
but automatic approval review blocked workload uploads pending explicit user
approval. Only hardware details were retrieved; no remote solve was run.

## Executed measurements

The installed VMEC++ 0.4.11 extension was exercised through the official adapter
using the public P3 seed. All outputs were isolated under `build/optimization`.
The host was running other oracle workers, so these are smoke-test timings,
not controlled performance benchmarks.

| Boundary / track | Adapter wall seconds | VMEC wall seconds | Result |
| --- | ---: | ---: | --- |
| Public P3 / P1, first smoke | 141.599 | 136.393 | Success |
| Public P3 / P1, repeat | See raw report | 158.916 | Success; exact metric and request/metric hashes |
| Public P3 / P3, discovery | 206.796 | 192.165 | Success |

These timings are measured. Differences between them are not speedups.

A 20-second native py-spy sample of the first smoke captured 1,999 samples,
zero sampling errors. Leading exclusive leaf counts were:

| Native function | Samples | Share of all samples |
| --- | ---: | ---: |
| FourierToReal3DSymmFastPoloidal | 335 | 16.76% |
| IdealMhdModel::computeMHDForces | 312 | 15.61% |
| ForcesToFourier3DSymmFastPoloidal | 293 | 14.66% |
| Unresolved address 0x7a7de2b8984a | 130 | 6.50% |
| IdealMhdModel::assembleTotalForces | 102 | 5.10% |

Shares are derived from measured sample counts, not full-run wall percentages.
Raw evidence: `build/optimization/native-smoke.json`.

## GPU evidence and next choices

The integrated prototype runs the actual inverse Fourier transform on CUDA,
with explicit opt-in selection, packed pinned transfers, reusable buffers,
serial m/n accumulation, and contraction disabled. It does not silently fall
back to CPU. The first full official P3/P1 run completed with exact metric and
numerical-wout parity. Its 63.386 s solver time is a smoke measurement, not a
comparable speedup. The local device does not yet show a reliable benefit.

The stage-timer smoke measured 2.399 s at ns=25, 5.343 s at ns=51, and 54.419 s
at ns=99. The final stage accounts for approximately 87.5% of those solve times
(derived). It terminated at iter2=9773 with the prescribed final tolerance;
the schedule and tolerances were not altered.

The local RTX 5090 streaming calibration reached 81.03% of Nsight Compute's
reported DRAM throughput, with a 536.51 us device kernel duration. This is a
calibration result, not an equilibrium calculation. Nsight reports a 64:1 ratio
of peak FP32 to FP64 throughput for this device. A datacenter GPU with stronger
FP64 capability is therefore worth comparing after solver kernels are ready;
this is a hypothesis, not a measured solver recommendation.

The host-driven calibration loop produced much lower effective throughput.
One later unprofiled trial measured 129.44 GB/s with direct launches and
170.45 GB/s with CUDA Graph replay. Host contention and device scheduling remain
uncontrolled. These are pipeline diagnostics, not established hardware ceilings.
Nsight Systems raw evidence and summaries are stored with the calibration.
Compute Sanitizer memcheck reported zero errors; its timings are excluded.

Preferred GPU prototype: preserve FP64 arithmetic and accumulation order across
independent transform outputs, then retain adjacent geometry/force operations
on device if transfer-inclusive timings justify it. Existing checkout CUDA
support is not proof of solver integration; tiled paths include CPU delegates.

## Constraints and validation

Physics settings and benchmark artifacts were not modified. The installed oracle
environment was not modified. Matching upstream source v0.4.11 was cloned into
the ignored build directory. An isolated matching-source build and required
Bazel baseline build passed. After hydrating the checksum-verified Git LFS
NetCDF test fixture in the isolated clone, all 14 C++ test targets passed for
the locality/diagnostic source changes.

Three comparison regression tests passed (one ULP, zero baseline, null/missing).
Ruff lint and format checks passed for Python tooling. Repository pre-commit
hooks passed for the initial experiment tooling; new GPU/cloud additions still
require their final hook run.

Vast authentication was resolved using the user-provided environment credential,
without printing or serializing it. Both pilots used no external volume and
had independent two-hour destroy watchdogs. Teardown was verified for instances
58929461 and 58929938. Rate-times-elapsed estimates are USD 0.0485 and USD 0.0320,
respectively; these are estimates, not invoiced charges, and exclude any separate
storage/transfer fees. State and hardware records are under `build/optimization`.
