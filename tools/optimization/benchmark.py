"""Run isolated official evaluations without touching benchmark artifacts.

Run with the pinned oracle Python and supply the stella repository explicitly. Each
invocation is one fresh-process sample. Output directories must be new.
"""

from __future__ import annotations

# Imports depend on the explicitly selected external oracle environment.
# ruff: noqa: PLC0415, T201
import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--track", choices=("P1", "P3"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    stella = args.stella.resolve()
    out = args.out.resolve()
    if out == stella or stella in out.parents:
        parser.error("outputs must be outside the stella repository")
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(stella / "src"))

    hashing = importlib.import_module("cga.contracts.hashing")
    canonical_bytes = hashing.canonical_bytes
    sha256_canonical = hashing.sha256_canonical
    normalize_boundary_payload = importlib.import_module(
        "cga.data.boundary"
    ).normalize_boundary_payload
    run_official = importlib.import_module("cga.oracle.official_v026").run_official

    np = importlib.import_module("numpy")

    import vmecpp

    seed = json.loads(args.seed.read_text())
    boundary = normalize_boundary_payload(json.loads(seed["boundary_json"]))
    request = {
        "request_id": "isolated-optimization",
        "design_id": sha256_canonical(boundary),
        "boundary_json": json.dumps(boundary, sort_keys=True),
        "mhd_parameters": {
            "builder": "constellaration.mhd.ideal_mhd_parameters."
            "boundary_to_ideal_mhd_parameters"
        },
        "fidelity": "official_high",
        "skip_qi": args.track == "P1",
        "threads": 1,
    }
    (out / "request.json").write_bytes(canonical_bytes(request))
    package = Path(vmecpp.__file__).parent
    extensions = list(package.rglob("_vmecpp*.so"))
    provenance = {
        "measurement": "measured",
        "seed_sha256": digest(args.seed),
        "harness_sha256": digest(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "load_average_at_start": os.getloadavg(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "campaign_wrapper_sha256": os.environ.get("CGA_OPTIMIZATION_WRAPPER_SHA256"),
        "campaign_configuration_sha256": os.environ.get(
            "CGA_OPTIMIZATION_CONFIG_SHA256"
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "vmecpp",
                "constellaration",
                "booz-xform",
                "numpy",
                "jax",
                "jaxlib",
                "simsopt",
                "pydantic",
                "scipy",
                "matplotlib",
            )
        },
        "extensions": {str(path): digest(path) for path in extensions},
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "experimental_runtime": {
            key: os.environ.get(key)
            for key in (
                "VMECPP_GPU_FOURIER",
                "VMECPP_GPU_FORWARD",
                "VMECPP_GPU_FORCE_PIPELINE",
                "VMECPP_GPU_UPLOAD_MODE",
                "VMECPP_GPU_PACK_THREADS",
                "VMECPP_GPU_GRAPH",
                "VMECPP_PARALLEL_PRECONDITIONER",
                "VMECPP_PARALLEL_CONSTRAINT",
                "VMECPP_RESIDENT_CAPTURE",
                "VMECPP_ITERATION_CAPTURE",
                "VMECPP_STEP_CAPTURE",
                "VMECPP_DIAGNOSTICS_CAPTURE",
                "VMECPP_PREPARATION_CAPTURE",
                "VMECPP_PRECONDITIONER_CAPTURE",
                "VMECPP_GPU_RESIDUAL_ROUNDS",
                "VMECPP_GPU_RESIDUAL_GRAPH",
                "VMECPP_GPU_SOLVER",
                "VMECPP_GPU_SOLVER_RESIDENT_CONTROL",
                "VMECPP_GPU_SOLVER_GRAPH",
                "VMECPP_GPU_SOLVER_PINNED_SCALARS",
                "VMECPP_GPU_SOLVER_TRACE",
                "VMECPP_GPU_DIAGNOSTICS_ROUNDS",
                "VMECPP_GPU_DIAGNOSTICS_GRAPH",
                "VMECPP_PROFILE_STAGES",
                "VMECPP_PROFILE_STAGES_ONLY",
                "VMECPP_GPU_PROFILE_REFERENCE",
                "VMECPP_PIPELINE_CAPTURE",
                "VMECPP_MHD_CAPTURE",
                "VMECPP_FORWARD_REFERENCE_SO",
                "VMECPP_FORWARD_CAPTURE",
                "VMECPP_DETERMINISTIC_FOURIER_THREADS",
                "OMP_MAX_ACTIVE_LEVELS",
            )
        },
        "preloaded_libraries": {
            str(path): digest(path)
            for value in os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
            if (path := Path(value)).is_file()
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    original_run = vmecpp.run
    timings = []
    process_timings = []
    main_thread_timings = []
    numerical_wout_hashes = []
    numerical_diagnostic_hashes = []
    outputs = []

    def timed_run(*positional, **keywords):
        if positional:
            indata = positional[0]
        else:
            indata = keywords.get("indata", keywords.get("input"))
        if indata is not None:
            indata.save(out / f"actual_indata_{len(timings)}.json")
        started = time.perf_counter()
        process_started = time.process_time()
        thread_started = time.thread_time()
        try:
            output = original_run(*positional, **keywords)
        finally:
            timings.append(time.perf_counter() - started)
            process_timings.append(time.process_time() - process_started)
            main_thread_timings.append(time.thread_time() - thread_started)
        outputs.append(output)
        return output

    vmecpp.run = timed_run
    started = time.perf_counter()
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    try:
        result = run_official(request, out)
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(out / "python.pstats"))
    elapsed = time.perf_counter() - started
    for index, output in enumerate(outputs):
        output.wout.save(out / f"raw_wout_{index}.nc")
        hashes = {}
        for name, value in output.wout.model_dump().items():
            if isinstance(value, int | float | bool | np.ndarray | np.number):
                array = np.ascontiguousarray(np.asarray(value))
                hashes[name] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
        numerical_wout_hashes.append(hashes)
        diagnostics = {}
        for section in ("jxbout", "mercier", "threed1_volumetrics"):
            fields = {}
            for name, value in getattr(output, section).model_dump().items():
                if isinstance(value, int | float | bool | np.ndarray | np.number):
                    array = np.ascontiguousarray(np.asarray(value))
                    fields[name] = {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
            diagnostics[section] = fields
        numerical_diagnostic_hashes.append(diagnostics)
    metrics = result.get("metrics")
    report = {
        "status": result["status"],
        "track": args.track,
        "seed": args.seed.name,
        "measurement": "measured",
        "profiled": args.profile,
        "adapter_wall_seconds": elapsed,
        "load_average_at_end": os.getloadavg(),
        "vmec_wall_seconds": timings,
        "vmec_process_cpu_seconds": process_timings,
        "vmec_main_thread_cpu_seconds": main_thread_timings,
        "numerical_wout_hashes": numerical_wout_hashes,
        "numerical_diagnostic_hashes": numerical_diagnostic_hashes,
        "metrics": metrics,
        "metrics_sha256": sha256_canonical(metrics),
        "request_metrics_sha256": sha256_canonical(
            {"request": request, "metrics": metrics}
        ),
        "result": result,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in (
                    "result",
                    "numerical_wout_hashes",
                    "numerical_diagnostic_hashes",
                )
            }
        )
    )
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
