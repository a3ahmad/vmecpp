"""Plan or explicitly run isolated batch-throughput experiments, never latency pairs."""

# CLI reporting and full-future exception draining are intentional.
# ruff: noqa: T201, BLE001

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.optimization.campaign_gates import complete_capture, pair_gate, read_capture
from tools.optimization.summarize import metric_drift

GPU_FLAGS = frozenset(
    {
        "VMECPP_GPU_SOLVER",
        "VMECPP_GPU_SOLVER_RESIDENT_CONTROL",
        "VMECPP_GPU_RESIDUAL_ROUNDS",
        "VMECPP_GPU_RESIDUAL_GRAPH",
        "VMECPP_GPU_DIAGNOSTICS_ROUNDS",
        "VMECPP_GPU_DIAGNOSTICS_GRAPH",
        "VMECPP_GPU_SOLVER_GRAPH",
        "VMECPP_GPU_SOLVER_PINNED_SCALARS",
    }
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_file(module) -> Path:
    if module.__file__ is None:
        message = "Batch provenance requires a file-backed module"
        raise RuntimeError(message)
    return Path(module.__file__).resolve()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class FrozenArtifacts:
    def __init__(self, paths):
        self.hashes = {str(path.resolve()): digest(path) for path in sorted(set(paths))}

    def check(self) -> None:
        for name, expected in self.hashes.items():
            path = Path(name)
            if not path.is_file() or digest(path) != expected:
                message = "An immutable batch input/source/library changed"
                raise RuntimeError(message)


def environment_fingerprint() -> str:
    # Raw values, including credentials, are never serialized or reported.
    result = hashlib.sha256()
    for name in sorted(os.environ):
        result.update(name.encode() + b"\0" + os.environ[name].encode() + b"\0")
    return result.hexdigest()


@dataclass(frozen=True)
class Job:
    key: str
    seed: str
    track: str
    round: int
    warmup: bool
    request_json: str
    out: Path


@dataclass
class NativeCall:
    input_snapshot: Any
    started: float
    ended: float = 0
    thread_cpu_seconds: float = 0
    output: Any = None
    activity_before: tuple | None = None
    activity_after: tuple | None = None


@dataclass
class RequestState:
    job: Job
    submitted: float
    started: float = 0
    finished: float = 0
    evaluation_started: float = 0
    evaluation_ended: float = 0
    owner: int = 0
    lock_owned: bool = False
    in_vmec: bool = False
    lock_wait_seconds: float = 0
    calls: list[NativeCall] = field(default_factory=list)
    result: dict | None = None
    error: dict | None = None


class CaptureDispatcher:
    """One immutable wrapper with request-local state and explicit lock ownership."""

    def __init__(
        self,
        original,
        fingerprint=environment_fingerprint,
        activity=None,
        require_gpu=False,
        require_graph=False,
    ):
        self.original = original
        self.gate = threading.RLock()
        self.local = threading.local()
        self.fingerprint = fingerprint
        self.expected_environment = fingerprint()
        self.activity = activity
        self.require_gpu = require_gpu
        self.require_graph = require_graph
        if require_gpu and activity is None:
            message = "GPU batch capture requires actual backend activity counters"
            raise ValueError(message)

    def check_environment(self) -> None:
        if self.fingerprint() != self.expected_environment:
            message = "Process environment changed after batch initialization"
            raise RuntimeError(message)

    @contextmanager
    def request_scope(self, state: RequestState):
        if getattr(self.local, "request", None) is not None:
            message = "Reentrant batch requests are forbidden"
            raise RuntimeError(message)
        self.check_environment()
        state.owner = threading.get_ident()
        self.local.request = state
        try:
            waiting = time.perf_counter()
            self.gate.acquire()
            state.lock_wait_seconds += time.perf_counter() - waiting
            state.lock_owned = True
            yield
            self.check_environment()
        finally:
            if state.lock_owned:
                state.lock_owned = False
                self.gate.release()
            self.local.request = None

    def __call__(self, *positional, **keywords):
        state = getattr(self.local, "request", None)
        if (
            state is None
            or state.owner != threading.get_ident()
            or not state.lock_owned
            or state.in_vmec
        ):
            message = "VMEC call lacks exclusive ownership of a batch request"
            raise RuntimeError(message)
        if (
            keywords.get("max_threads") != 1
            or keywords.get("verbose", True) is not False
        ):
            message = "Batch VMEC calls require max_threads=1 and verbose=False"
            raise RuntimeError(message)
        self.check_environment()
        input_value = positional[0] if positional else keywords["input"]
        frozen_input = copy.deepcopy(input_value)
        activity_before = self.activity() if self.activity else None
        state.in_vmec = True
        state.lock_owned = False
        self.gate.release()
        started = time.perf_counter()
        cpu_started = time.thread_time()
        call = NativeCall(frozen_input, started, activity_before=activity_before)
        try:
            call.output = self.original(*positional, **keywords)
            return call.output
        finally:
            call.ended = time.perf_counter()
            call.thread_cpu_seconds = time.thread_time() - cpu_started
            try:
                call.activity_after = self.activity() if self.activity else None
            finally:
                waiting = time.perf_counter()
                self.gate.acquire()
                state.lock_wait_seconds += time.perf_counter() - waiting
                state.lock_owned = True
                state.in_vmec = False
                state.calls.append(call)
            self.check_environment()
            if self.require_gpu and call.output is not None:
                if call.activity_before is None or call.activity_after is None:
                    message = "GPU backend activation counters are missing"
                    raise RuntimeError(message)
                delta = [
                    after - before
                    for before, after in zip(
                        call.activity_before, call.activity_after, strict=True
                    )
                ]
                if (
                    len(delta) != 3
                    or min(delta) < 0
                    or delta[0] <= 0
                    or delta[1] <= 0
                    or self.require_graph
                    and delta[2] <= 0
                ):
                    message = "Actual per-request GPU backend activation proof failed"
                    raise RuntimeError(message)


def install_capture(module, dispatcher: CaptureDispatcher) -> None:
    if (
        hasattr(module, "_isolated_batch_capture")
        or module.run is not dispatcher.original
    ):
        message = "VMEC capture may be installed exactly once before workers start"
        raise RuntimeError(message)
    module._isolated_batch_capture = dispatcher
    module.run = dispatcher


def run_rounds(
    groups,
    dispatcher,
    evaluate,
    finalize,
    workers,
    executor_factory: Callable[..., Any] = ThreadPoolExecutor,
    verify_artifacts=lambda: None,
):
    """Drain every submitted future before any worker/TLS destructor can run."""
    executor = executor_factory(max_workers=workers)
    submitted = {}
    collected = set()
    rounds = []
    records = []

    def worker(job, submitted_at):
        state = RequestState(job, submitted_at, started=time.perf_counter())
        try:
            job.out.mkdir(parents=True, exist_ok=False)
            with dispatcher.request_scope(state):
                state.evaluation_started = time.perf_counter()
                try:
                    state.result = evaluate(json.loads(job.request_json), job.out)
                finally:
                    state.evaluation_ended = time.perf_counter()
        except BaseException as error:
            state.error = {"type": type(error).__name__, "message": str(error)}
        finally:
            state.finished = time.perf_counter()
        return state

    try:
        for jobs in groups:
            verify_artifacts()
            started = time.perf_counter()
            futures = []
            for job in jobs:
                submitted_at = time.perf_counter()
                future = executor.submit(worker, job, submitted_at)
                submitted[future] = job
                futures.append(future)
            states = []
            terminal_errors = []
            for future in as_completed(futures):
                try:
                    states.append(future.result())
                except BaseException as error:
                    if not future.done():
                        raise
                    terminal_errors.append(
                        {
                            "job": submitted[future].key,
                            "gate": {"exact": False},
                            "timing": None,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )
                collected.add(future)
            ended = time.perf_counter()
            round_records = terminal_errors.copy()
            records.extend(terminal_errors)
            for state in sorted(states, key=lambda value: value.job.key):
                try:
                    record = finalize(state)
                except Exception as error:
                    record = {
                        "job": state.job.key,
                        "gate": {"exact": False},
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                records.append(record)
                round_records.append(record)
            verify_artifacts()
            exact = all(record.get("gate", {}).get("exact") for record in round_records)
            rounds.append(
                {
                    "round": jobs[0].round,
                    "warmup": jobs[0].warmup,
                    "submitted_jobs": len(jobs),
                    "all_futures_collected": True,
                    "makespan_seconds": ended - started,
                    "all_gates_exact": exact,
                    "throughput_evaluations_per_second": len(jobs) / (ended - started)
                    if exact
                    else None,
                    "scope": "first submission through all request completions; audit hashes/IO excluded",
                }
            )
            if not exact:
                break
    finally:
        # Do not cancel on first failure: exiting a worker can destroy TLS CUDA
        # state while a different worker is still capturing a graph.
        active_error = sys.exc_info()[1]
        interrupted = None
        for future in submitted:
            while future not in collected:
                try:
                    future.result()
                except BaseException as error:
                    if not future.done():
                        if interrupted is None:
                            interrupted = error
                        continue
                if future.done():
                    collected.add(future)
        executor.shutdown(wait=True, cancel_futures=False)
        if active_error is None and interrupted is not None:
            raise interrupted
    return rounds, records


def timing_record(state: RequestState) -> dict:
    return {
        "queue_seconds": state.started - state.submitted,
        "submit_to_finish_seconds": state.finished - state.submitted,
        "service_seconds": state.finished - state.started,
        "adapter_wall_seconds": state.evaluation_ended - state.evaluation_started
        if state.evaluation_started
        else None,
        "post_vmec_tail_seconds": state.finished - state.calls[-1].ended
        if state.calls
        else None,
        "serialized_gate_wait_seconds": state.lock_wait_seconds,
        "vmec_wall_seconds": [call.ended - call.started for call in state.calls],
        "vmec_main_thread_cpu_seconds": [
            call.thread_cpu_seconds for call in state.calls
        ],
        "backend_activity_deltas": [
            [
                after - before
                for before, after in zip(
                    call.activity_before, call.activity_after, strict=True
                )
            ]
            if call.activity_before is not None and call.activity_after is not None
            else None
            for call in state.calls
        ],
    }


def numeric_fields(model, np) -> dict:
    """Match the canonical benchmark's shape/dtype/contiguous-byte hashes."""
    fields = {}
    for name, value in model.model_dump().items():
        if isinstance(value, int | float | bool | np.ndarray | np.number):
            array = np.ascontiguousarray(np.asarray(value))
            fields[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    return fields


def repeat_coverage(records) -> dict:
    cases = {}
    for record in records:
        key = f"{record.get('seed')}:{record.get('track')}"
        case = cases.setdefault(key, {"executed_samples": 0, "identities": set()})
        case["executed_samples"] += 1
        case["identities"].add(
            (
                record.get("request_metrics_sha256"),
                record.get("numeric_output_identity"),
            )
        )
    consistent = all(len(case["identities"]) == 1 for case in cases.values())
    repeated = bool(cases) and all(
        case["executed_samples"] >= 2 for case in cases.values()
    )
    return {
        "all_cases_repeated": repeated,
        "observed_identities_consistent": consistent,
        "repeated_outputs_exact": consistent if repeated else None,
        "cases": {
            key: {
                "executed_samples": case["executed_samples"],
                "distinct_output_identities": len(case["identities"]),
                "repeat_check_performed": case["executed_samples"] >= 2,
            }
            for key, case in cases.items()
        },
        "scope": "Per-case repeated samples; singleton cases cannot establish determinism",
    }


def planned_cases(args):
    fixtures = json.loads(args.manifest.read_text())
    seeds = [Path(item["path"]) for item in fixtures]
    seeds = [
        seed if seed.is_absolute() else args.manifest.parent / seed for seed in seeds
    ]
    cases = [(seed, track) for seed in seeds for track in ("P1", "P3")]
    if args.case:
        wanted = set(args.case)
        cases = [
            (seed, track) for seed, track in cases if f"{seed.name}:{track}" in wanted
        ]
        if len(cases) != len(wanted):
            message = (
                "A requested case is missing or duplicated in the fixture manifest"
            )
            raise ValueError(message)
    identifiers = [f"{seed.stem}-{track}" for seed, track in cases]
    if not cases or len(identifiers) != len(set(identifiers)):
        message = "Batch requires unique case identifiers and at least one case"
        raise ValueError(message)
    return cases


class LibraryInfo(ctypes.Structure):
    _fields_: ClassVar[Any] = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def library_owner(function) -> Path:
    lookup = ctypes.CDLL(None).dladdr
    lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(LibraryInfo)]
    lookup.restype = ctypes.c_int
    information = LibraryInfo()
    if not lookup(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(information)):
        message = "Cannot identify the defining backend library"
        raise RuntimeError(message)
    return Path(information.filename.decode()).resolve()


def reference_extension(environment, loaded_extensions=None) -> Path:
    name = environment.get("VMECPP_FORWARD_REFERENCE_SO")
    if not name or not Path(name).is_absolute() or not Path(name).is_file():
        message = "GPU batches require an existing absolute reference extension"
        raise RuntimeError(message)
    reference = Path(name).resolve()
    if loaded_extensions is not None and (
        str(reference) not in loaded_extensions
        or digest(reference) != loaded_extensions[str(reference)]
    ):
        message = "Reference extension must match the frozen pinned Python extension"
        raise RuntimeError(message)
    return reference


def preflight_runtime(args):
    if sys.version_info[:2] != (3, 12):
        message = "Pinned batch execution requires Python3.12"
        raise RuntimeError(message)
    if os.environ.get("VMECPP_RELEASE_GIL") != "1":
        message = "A preloaded opt-in GIL guard is required"
        raise RuntimeError(message)
    for name, value in os.environ.items():
        if (
            name.startswith("VMECPP_GPU_")
            and value
            and value != "0"
            and name not in GPU_FLAGS
        ):
            message = "Batch execution forbids direct component GPU configuration"
            raise RuntimeError(message)
        if value and (
            name == "CGA_PROFILE"
            or name.endswith(("_TRACE", "_CAPTURE", "_MARKER"))
            or "OBSERVER" in name
            or name in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")
        ):
            message = (
                "Batch execution forbids profiler/observer/shared-capture configuration"
            )
            raise RuntimeError(message)
    gpu = os.environ.get("VMECPP_GPU_SOLVER") == "1"
    if gpu != (args.backend == "gpu"):
        message = "Requested backend disagrees with the fixed process configuration"
        raise RuntimeError(message)
    if gpu:
        reference_extension(os.environ)
    if args.backend == "cpu" and any(
        value and value != "0"
        for name, value in os.environ.items()
        if name.startswith("VMECPP_GPU_")
    ):
        message = "CPU batches must not carry GPU runtime flags"
        raise RuntimeError(message)
    guard = ctypes.PyDLL(None)
    ready = guard.VmecppGilReleaseReady
    ready.restype = ctypes.c_bool
    if not ready():
        message = "GIL/stream/backend coordination preflight rejected this process"
        raise RuntimeError(message)
    activity_snapshot: Callable[[], tuple[int, ...]] | None = None
    if gpu:
        coordinated = guard.GpuSolverCaptureCoordinated
        coordinated.restype = ctypes.c_bool
        if not coordinated():
            message = "A capture-coordinated GPU backend is required"
            raise RuntimeError(message)
        activity = guard.GpuSolverThreadActivity
        activity.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        activity.restype = None
        owner = library_owner(guard.CreateGpuSolver)
        if library_owner(activity) != owner or library_owner(coordinated) != owner:
            message = (
                "Backend activation counters and coordination must have the same owner"
            )
            raise RuntimeError(message)

        def read_activity():
            counts = (ctypes.c_uint64 * 3)()
            activity(counts)
            return tuple(counts)

        activity_snapshot = read_activity
    else:
        owner = None
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            message = "All native numerical thread limits must be fixed at1"
            raise RuntimeError(message)
    return {"activity": activity_snapshot, "backend_library": owner}


def execute(args, cases) -> dict:
    preflight_runtime(args)
    if args.out == args.stella or args.stella in args.out.parents:
        message = "Batch outputs must be outside the benchmark repository"
        raise ValueError(message)
    args.out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.stella / "src"))
    np = importlib.import_module("numpy")
    vmecpp = importlib.import_module("vmecpp")
    if (
        getattr(vmecpp.run, "__name__", None) != "run"
        or not hasattr(vmecpp.run, "__code__")
        or Path(vmecpp.run.__code__.co_filename).resolve() != module_file(vmecpp)
    ):
        message = "Stock vmecpp.run is required before installing the batch collector"
        raise RuntimeError(message)
    hashing = importlib.import_module("cga.contracts.hashing")
    normalize = importlib.import_module("cga.data.boundary").normalize_boundary_payload
    evaluate = importlib.import_module("cga.oracle.official_v026").run_official
    if (
        Path(evaluate.__code__.co_filename).resolve()
        != (args.stella / "src/cga/oracle/official_v026.py").resolve()
    ):
        message = "The official adapter did not load from the requested frozen source"
        raise RuntimeError(message)
    importlib.import_module("constellaration.forward_model")
    jax = importlib.import_module("jax")
    if jax.default_backend() != "cpu":
        message = "Batch postprocessing requires the existing CPU JAX backend"
        raise RuntimeError(message)
    package = module_file(vmecpp).parent
    provenance = {
        "measurement": "measured batch correctness/throughput experiment",
        "harness_sha256": digest(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "vmecpp",
                "constellaration",
                "booz-xform",
                "numpy",
                "pydantic",
                "jax",
                "jaxlib",
            )
        },
        "extensions": {
            str(path): digest(path) for path in package.rglob("_vmecpp*.so")
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "experimental_runtime": {
            name: value
            for name, value in os.environ.items()
            if name.startswith("VMECPP_") or name == "CGA_PROFILE"
        },
        "preloaded_libraries": {
            name: digest(Path(name))
            for name in os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
        },
    }
    if [
        provenance["packages"][name]
        for name in ("vmecpp", "constellaration", "booz-xform")
    ] != ["0.4.11", "0.2.6", "0.0.9"]:
        message = "Pinned solver and benchmark package versions are required"
        raise RuntimeError(message)
    if args.backend == "gpu":
        reference = reference_extension(os.environ, provenance["extensions"])
        provenance["reference_extension"] = {
            "path": str(reference),
            "sha256": digest(reference),
        }
    requests = {}
    references = {}
    frozen_paths = [
        args.manifest,
        Path(__file__),
        Path(evaluate.__code__.co_filename),
        args.stella / "configs/oracle.yaml",
        module_file(vmecpp),
    ]
    frozen_paths.extend(
        Path(__file__).with_name(name) for name in ("campaign_gates.py", "summarize.py")
    )
    frozen_paths.extend(Path(name) for name in provenance["preloaded_libraries"])
    frozen_paths.extend(Path(name) for name in provenance["extensions"])
    for name in (
        "constellaration.forward_model",
        "constellaration.mhd.vmec_utils",
        "constellaration.mhd.vmec_settings",
        "cga.data.boundary",
        "cga.contracts.hashing",
    ):
        frozen_paths.append(module_file(importlib.import_module(name)))
    for seed, track in cases:
        frozen_paths.append(seed)
        boundary = normalize(json.loads(json.loads(seed.read_text())["boundary_json"]))
        request = {
            "request_id": "isolated-optimization",
            "design_id": hashing.sha256_canonical(boundary),
            "boundary_json": json.dumps(boundary, sort_keys=True),
            "mhd_parameters": {
                "builder": "constellaration.mhd.ideal_mhd_parameters.boundary_to_ideal_mhd_parameters"
            },
            "fidelity": "official_high",
            "skip_qi": track == "P1",
            "threads": 1,
        }
        requests[(seed.name, track)] = json.dumps(
            request, sort_keys=True, allow_nan=False
        )
        reference = args.references / f"{seed.stem}-{track}"
        reference_report = json.loads((reference / "report.json").read_text())
        reference_capture = read_capture(reference)
        references[(seed.name, track)] = (reference_report, reference_capture)
        frozen_paths.extend(reference.glob("*.json"))
        if (
            not complete_capture(reference_report, reference_capture)
            or hashing.sha256_canonical(
                {"request": request, "metrics": reference_report["metrics"]}
            )
            != reference_report["request_metrics_sha256"]
        ):
            message = "Immutable original reference/request preflight failed"
            raise RuntimeError(message)
    runtime = preflight_runtime(args)
    if runtime["backend_library"] is not None:
        frozen_paths.append(runtime["backend_library"])
    # Freeze experimental dependent DSOs too; no new backend/module loads are
    # permitted after the workers start.
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split(maxsplit=5)[-1]
        if (
            name.startswith("/")
            and ".so" in name
            and any(
                token in Path(name).name
                for token in (
                    "solver",
                    "resident",
                    "iteration",
                    "residual",
                    "diagnostic",
                    "pipeline",
                    "fourier",
                )
            )
        ):
            frozen_paths.append(Path(name))
    frozen = FrozenArtifacts(frozen_paths)
    dispatcher = CaptureDispatcher(
        vmecpp.run,
        activity=runtime["activity"],
        require_gpu=args.backend == "gpu",
        require_graph=args.backend == "gpu"
        and os.environ.get("VMECPP_GPU_SOLVER_GRAPH") == "1",
    )
    install_capture(vmecpp, dispatcher)
    groups = []
    for number in range(args.warmup_rounds + args.rounds):
        jobs = []
        for seed, track in cases:
            for replica in range(args.replicates):
                key = f"{seed.stem}-{track}-round{number:02d}-copy{replica:02d}"
                jobs.append(
                    Job(
                        key,
                        seed.name,
                        track,
                        number,
                        number < args.warmup_rounds,
                        requests[(seed.name, track)],
                        args.out / key,
                    )
                )
        groups.append(jobs)

    def finalize(state):
        job = state.job
        request = json.loads(job.request_json)
        (job.out / "request.json").write_bytes(hashing.canonical_bytes(request))
        save(job.out / "provenance.json", provenance)
        outputs = []
        diagnostics = []
        for index, call in enumerate(state.calls):
            call.input_snapshot.save(job.out / f"actual_indata_{index}.json")
            if call.output is not None:
                call.output.wout.save(job.out / f"raw_wout_{index}.nc")
                outputs.append(numeric_fields(call.output.wout, np))
                diagnostics.append(
                    {
                        name: numeric_fields(getattr(call.output, name), np)
                        for name in ("jxbout", "mercier", "threed1_volumetrics")
                    }
                )
        result = state.result or {"status": "WORKER_CRASH", "metrics": None}
        metrics = result.get("metrics")
        timing = timing_record(state)
        report = {
            "status": result["status"],
            "track": job.track,
            "seed": job.seed,
            "profiled": False,
            "measurement": "measured batch experiment",
            "adapter_wall_seconds": timing["adapter_wall_seconds"],
            "vmec_wall_seconds": timing["vmec_wall_seconds"],
            "numerical_wout_hashes": outputs,
            "numerical_diagnostic_hashes": diagnostics,
            "metrics": metrics,
            "metrics_sha256": hashing.sha256_canonical(metrics),
            "request_metrics_sha256": hashing.sha256_canonical(
                {"request": request, "metrics": metrics}
            ),
            "result": result,
        }
        save(job.out / "report.json", report)
        reference_report, reference_capture = references[(job.seed, job.track)]
        gate = pair_gate(
            reference_report,
            report,
            reference_capture,
            read_capture(job.out),
            require_harness=False,
        )
        if state.error is not None:
            gate["exact"] = False
        return {
            "job": job.key,
            "seed": job.seed,
            "track": job.track,
            "warmup": job.warmup,
            "round": job.round,
            "timing": timing,
            "gate": gate,
            "metric_drift": metric_drift(reference_report["metrics"], metrics or {}),
            "error": state.error,
            "request_metrics_sha256": report["request_metrics_sha256"],
            "numeric_output_identity": hashing.sha256_canonical([outputs, diagnostics]),
            "main_table_eligible": False,
        }

    rounds, records = run_rounds(
        groups,
        dispatcher,
        evaluate,
        finalize,
        args.workers,
        verify_artifacts=frozen.check,
    )
    dispatcher.check_environment()
    repeats = repeat_coverage(records)
    measured = [row for row in rounds if not row["warmup"]]
    if args.correctness_only:
        for row in rounds:
            row["throughput_evaluations_per_second"] = None
    exact = (
        repeats["observed_identities_consistent"]
        and len(rounds) == len(groups)
        and all(row["all_gates_exact"] for row in rounds)
    )
    result = {
        "scope": "batch throughput, never a main-table single-evaluation speedup",
        "all_original_global_gates_exact": exact,
        "repeated_outputs_exact": repeats["repeated_outputs_exact"],
        "repeat_coverage": repeats,
        "workers_requested": args.workers,
        "rounds": rounds,
        "records": records,
        "all_submitted_futures_terminal_before_shutdown": True,
        "main_table_eligible": False,
        "correctness_only": args.correctness_only,
        "eligible_for_batch_throughput_comparison": exact
        and len(measured) >= 2
        and repeats["all_cases_repeated"]
        and not args.correctness_only,
        "measured_evaluations_per_second": sum(
            row["submitted_jobs"] for row in measured
        )
        / sum(row["makespan_seconds"] for row in measured)
        if exact and measured and not args.correctness_only
        else None,
        "timing_scope": "Original vmecpp.run calls may overlap, including their conversions. Setup/postprocessing outside vmecpp.run is serialized. Per-round validation IO/hashing is excluded.",
        "configuration": provenance,
        "environment_fingerprint": dispatcher.expected_environment,
        "frozen_artifacts": frozen.hashes,
    }
    save(args.out / "batch-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "references", "stella", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    for name in ("manifest", "references", "stella", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if not (
        1 <= args.workers <= 32
        and 1 <= args.rounds <= 10
        and 0 <= args.warmup_rounds <= 2
        and 1 <= args.replicates <= 8
    ):
        parser.error("Batch sizes exceed bounded prototype limits")
    cases = planned_cases(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "scope": "PLAN ONLY: no oracle imports, solver/GPU execution or experiment output writes",
                    "backend": args.backend,
                    "correctness_only": args.correctness_only,
                    "workers": args.workers,
                    "cases": [f"{seed.name}:{track}" for seed, track in cases],
                    "warmup_rounds": args.warmup_rounds,
                    "measured_rounds": args.rounds,
                    "replicates_per_case": args.replicates,
                    "planned_evaluations": len(cases)
                    * args.replicates
                    * (args.rounds + args.warmup_rounds),
                    "requirements": [
                        "fresh preloaded GIL guard",
                        "one coordinated GPU backend if GPU enabled",
                        "fixed environment/max_threads1/verboseFalse",
                        "independent inputs/outdirs",
                        "no profilers/observers/shared trace paths",
                        "all futures drained before TLS cleanup",
                    ],
                    "main_table_eligible": False,
                },
                indent=2,
            )
        )
        return 0
    result = execute(args, cases)
    return 0 if result["all_original_global_gates_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
