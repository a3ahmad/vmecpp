# Optimization status and remaining work

The current architecture-specific results and qualification status are in
[GB202.md](GB202.md). Earlier measured campaigns remain documented in
[RESULTS.md](RESULTS.md), [A100_RAGGED.md](A100_RAGGED.md),
[ENERGY_BATCHING.md](ENERGY_BATCHING.md), and [THROUGHPUT.md](THROUGHPUT.md).
These reports describe different builds, hardware, clocks, and workload scopes;
their speedup ratios must not be multiplied.

## Completed work

- Experimental GPU source, bounded launchers, output checks, and test fixtures
  are isolated from the original oracle and submission artifacts. Generated
  binaries, captured arrays, private profiler reports, and runtime receipts
  belong under ignored `build/optimization`, not in the source repository.
- The qualified CLEAN campaigns retain the original requests, input decks,
  tolerances, metrics, numerical WOut fields, actual diagnostics, and repeat
  checks. Per-metric drift tables accompany their measured results.
- Independent-solve scheduling was measured on the RTX 3090 using serial,
  concurrent-process, and private-MPS arms. Higher throughput came with longer
  individual solve latency and higher summed-time budget proxies. Width-2,
  width-3, and width-6 results use different case compositions, each matched
  only to its own serial baseline. They are not a width-only scaling curve.
- A100 component and whole-solver experiments establish that consumer-GPU
  performance cannot be extrapolated to datacenter hardware. Event timing,
  profiled kernel duration, whole-call wall time, and computed CGA metrics are
  reported separately.
- Local GB202 research uses Windows QPC receipt intervals for host wall-time
  comparisons. Linux RAW, MONO, and epoch clocks are retained as diagnostics;
  observed disagreement is not silently corrected. Component events retain
  their full declared envelope and all measured outliers.
- The GB202 forward recurrence-subgroup experiment reduced registers and
  improved representative component timing, but its small whole-solver screen
  was mixed. It remains experimental and default OFF. Explicit CG cache loads,
  inverse output grouping, and an angular launch-shape variant did not improve
  their matched graph comparisons. See GB202.md for exact artifacts and scope.

## Remaining evidence-based work

1. Preserve the completed three-boundary, two-track qualification record for
   the exact final build. Before promoting any different binary, repeat the
   relevant original request/deck, twelve-metric, full numerical output,
   diagnostic, and repeat checks. A single-fixture component or VMEC-only
   research screen does not establish full CGA qualification.
2. Integrate batching only at an existing independent-work boundary. Keep
   request/cache ownership, search order, per-process cleanup, and original
   fidelity unchanged. Do not widen adaptive search rounds merely to fill a
   GPU. Verify MPS availability and actual membership on the chosen system;
   successful source compilation is not evidence of service support.
3. Resolve accounting before production adoption: faster batches can increase
   the sum of individual process elapsed times charged by the existing meter.
   Report throughput, individual latency, and unrenormalized summed-time cost
   separately. Pilot time proxies are not billing records.
4. Base further tuning on the actual target workload and measured bottleneck.
   More occupancy alone did not establish a faster inverse transform. A warp
   stall category alone does not establish a shared-memory bank conflict, and
   a trace without recorded work does not by itself establish GPU idle time.
5. Keep experimental proposals opt-in until their measured end-to-end benefit
   is convincing. Retain negative results, failures, warnings, and limitations;
   do not promote small or mixed screens into the main per-solve table.

## Reproduction and publication

Use [README.md](README.md) for the source/build workflow. Runtime paths, SSH
hosts, interpreter locations, and SSH configuration are explicit watcher
arguments; there are no personal-machine defaults. Watchers perform bounded
status reads or verified evidence retrieval and never launch GPU work.

Do not publish credentials, private SSH configuration, provider state files,
raw captured inputs, unrelated process metadata, or submission artifacts.
Only explicitly approved fixtures and dependencies may be transferred for an
experiment. A source checkout is not an authorization to allocate a machine,
change GPU settings, or interfere with another workload.

Changes must follow VMECPP_NAMING_GUIDE.md, preserve the documented Fourier
basis and arithmetic order, use ASCII, and pass focused pre-commit checks.
No general framework, broad benchmark rerun, or new GPU trial is required to
maintain these documentation and watcher utilities.
