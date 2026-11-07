"""Prepare/coordinate persistent unprofiled batch services; default plan only."""

# CLI status is intentional; interrupted services must retain cleanup evidence.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import math
import os
import random
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import sysconfig
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from tools.optimization import batch_throughput as batch

REMOTE_ROOT = Path("/bulk/vmecpp-optimization-20261103/batch18-qualification")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_files(files):
    for name, expected in files.items():
        if digest(name) != expected:
            message = "Frozen campaign file changed: " + name
            raise RuntimeError(message)


def paired_order(pair):
    return (1, 4) if pair % 2 == 0 else (4, 1)


def metric_maximum(current, value):
    if value is None:
        return current
    if not isinstance(value, int | float) or not math.isfinite(value):
        message = "Metric drift must be finite numeric data or an explicit null"
        raise ValueError(message)
    return max(current, value)


def workload_coverage(summary):
    records = summary["records"]
    cases = {(row["seed"], row["track"]) for row in records}
    return (
        len(records) == 36
        and len(cases) == 6
        and len({seed for seed, _ in cases}) == 3
        and {track for _, track in cases} == {"P1", "P3"}
        and all(
            len([row for row in records if row["round"] == number]) == 6
            and {
                (row["seed"], row["track"]) for row in records if row["round"] == number
            }
            == cases
            and all(
                row["warmup"] == (number == 0)
                for row in records
                if row["round"] == number
            )
            for number in range(6)
        )
    )


def dependency_snapshot(root):
    paths = [
        root.parent / "baseline/lib/python3.12/site-packages",
        Path(sysconfig.get_path("stdlib")),
    ]
    return {
        str(path): digest(path)
        for directory in paths
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
    }


def physical_affinity(rows):
    anchor = next(row for row in rows if row["cpu"] == 0)
    selected, seen = [], set()
    for row in sorted(rows, key=lambda item: item["cpu"]):
        key = (row["package"], row["core"])
        if row["node"] == anchor["node"] and key not in seen:
            selected.append(row)
            seen.add(key)
    if len(selected) < 5:
        message = "Need four physical solver cores and a separate coordinator core"
        raise RuntimeError(message)
    return {
        "service_cpus": [row["cpu"] for row in selected[:4]],
        "coordinator_cpu": selected[-1]["cpu"],
        "selected_physical_cores": selected[:4],
        "all_topology": rows,
    }


def pinned_executor(cpus):
    assigned = []
    lock = threading.Lock()

    def initialize():
        with lock:
            if sorted(os.sched_getaffinity(0)) != cpus:
                message = "Worker inherited an unexpected process affinity mask"
                raise RuntimeError(message)
            cpu = cpus[len(assigned)]
            assigned.append(cpu)
            os.sched_setaffinity(0, {cpu})

    def factory(max_workers):
        return ThreadPoolExecutor(max_workers=max_workers, initializer=initialize)

    return factory


def service_launch(environment, cpus, python, arguments):
    # Apply Python-specific preloads only after taskset/env have executed.
    # A static interpreter supplies Py* symbols; taskset cannot supply them.
    outer = {name: value for name, value in environment.items() if name != "LD_PRELOAD"}
    command = [
        "/usr/bin/taskset",
        "-c",
        ",".join(str(cpu) for cpu in cpus),
        "/usr/bin/env",
        "LD_PRELOAD=" + environment["LD_PRELOAD"],
        python,
        *arguments,
    ]
    return command, outer


def raw_gate(directory, report):
    np = importlib.import_module("numpy")
    vmecpp = importlib.import_module("vmecpp")

    expected = report["numerical_wout_hashes"]
    if not expected:
        return False
    return all(
        batch.numeric_fields(
            vmecpp.VmecWOut.from_wout_file(directory / f"raw_wout_{index}.nc"), np
        )
        == fields
        for index, fields in enumerate(expected)
    )


def controlled_rounds(
    root,
    configuration,
    workers,
    original,
    groups,
    dispatcher,
    evaluate,
    finalize,
    *,
    verify_artifacts,
    raw_verifier=raw_gate,
):
    """Barrier only after all futures and validation have finished; pool persists."""
    phase = 0
    completed = []
    affinities = {}

    def observed_evaluate(request, output):
        before = sorted(os.sched_getaffinity(0))
        try:
            return evaluate(request, output)
        finally:
            after = sorted(os.sched_getaffinity(0))
            if (
                len(before) != 1
                or after != before
                or before[0] not in configuration["worker_cpus"]
            ):
                message = "Actual per-request single-core affinity changed"
                raise RuntimeError(message)
            affinities[str(output)] = {
                "tid": threading.get_native_id(),
                "start_affinity": before,
                "end_affinity": after,
            }

    def observed_finalize(state):
        record = finalize(state)
        report = json.loads((state.job.out / "report.json").read_text())
        record["raw_netcdf_hashes_exact"] = raw_verifier(state.job.out, report)
        if "worker_cpus" in configuration:
            record["worker_affinity"] = affinities[str(state.job.out)]
        completed.append(record)
        return record

    def barrier():
        nonlocal phase
        verify_artifacts()
        check_files(configuration["files"])
        pair = phase // 2
        if phase % 2 == 0:
            save(
                root / "run" / f"ready-{workers}-{pair:02d}.json",
                {
                    "nonce": configuration["nonce"],
                    "workers": workers,
                    "pair": pair,
                    "pid": os.getpid(),
                },
            )
            grant = root / "run" / f"grant-{workers}-{pair:02d}.json"
            while not grant.exists():
                if time.monotonic() >= configuration["expires_monotonic"]:
                    message = "Persistent service grant deadline exceeded"
                    raise TimeoutError(message)
                time.sleep(0.1)
            value = json.loads(grant.read_text())
            if value != {
                "nonce": configuration["nonce"],
                "workers": workers,
                "pair": pair,
            }:
                message = "Unexpected paired-round grant"
                raise RuntimeError(message)
        else:
            records = [record for record in completed if record["round"] == pair]
            exact = len(records) == 6 and all(
                record["gate"]["exact"] and record["raw_netcdf_hashes_exact"]
                for record in records
            )
            save(
                root / "run" / f"completed-{workers}-{pair:02d}.json",
                {
                    "nonce": configuration["nonce"],
                    "workers": workers,
                    "pair": pair,
                    "warmup": pair == 0,
                    "exact": exact,
                    "records": records,
                    "completed_at": time.time(),
                },
            )
            if not exact:
                message = "Original global/raw-output round gate failed"
                raise RuntimeError(message)
        phase += 1

    options: dict[str, Any] = {"verify_artifacts": barrier}
    if "worker_cpus" in configuration:
        options["executor_factory"] = pinned_executor(configuration["worker_cpus"])
    return original(
        groups,
        dispatcher,
        observed_evaluate if "worker_cpus" in configuration else evaluate,
        observed_finalize,
        workers,
        **options,
    )


def service(root, workers):
    configuration = json.loads((root / "run/launch.json").read_text())
    status = {
        "phase": "running",
        "pid": os.getpid(),
        "birth": birth(os.getpid()),
        "workers": workers,
        "nonce": configuration["nonce"],
        "actual_process_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    save(root / "run" / f"service-{workers}.json", status)
    if (
        status["actual_process_affinity"] != configuration["worker_cpus"]
        or status["nice"] != 0
    ):
        message = "Service did not receive the fixed four-core mask/normal priority"
        raise RuntimeError(message)
    original = batch.run_rounds

    def coordinated(
        groups, dispatcher, evaluate, finalize, actual_workers, *, verify_artifacts
    ):
        if actual_workers != workers:
            message = "Persistent service worker count changed"
            raise RuntimeError(message)
        return controlled_rounds(
            root,
            configuration,
            workers,
            original,
            groups,
            dispatcher,
            evaluate,
            finalize,
            verify_artifacts=verify_artifacts,
        )

    batch.run_rounds = coordinated
    sys.argv = [
        "native-batch-service",
        "--execute",
        "--backend",
        "gpu",
        "--workers",
        str(workers),
        "--warmup-rounds",
        "1",
        "--rounds",
        "5",
        "--replicates",
        "1",
        "--manifest",
        str(root / "fixtures/remote-manifest.json"),
        "--references",
        str(root / "references"),
        "--stella",
        str(root / "stella"),
        "--out",
        str(root / "evals" / f"workers{workers}"),
    ]
    code = 1
    try:
        code = batch.main()
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        status.update(phase="terminal", exit_status=code, finished_at=time.time())
        save(root / "run" / f"service-{workers}.json", status)
    return code


def birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def children(parent):
    result, pending, seen = [], [parent], {parent}
    while pending:
        current = pending.pop()
        for path in Path(f"/proc/{current}/task").glob("*/children"):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                for value in path.read_text().split():
                    pid = int(value)
                    if pid not in seen:
                        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                            result.append((pid, birth(pid)))
                            pending.append(pid)
                            seen.add(pid)
    return result


def cleanup(processes):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, expected in reversed(children(os.getpid())):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                if birth(pid) == expected:
                    os.kill(pid, sig)
        limit = time.monotonic() + 3
        while time.monotonic() < limit and children(os.getpid()):
            for process in processes.values():
                process.poll()
            for pid, _ in children(os.getpid()):
                if pid not in {process.pid for process in processes.values()}:
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(pid, os.WNOHANG)
            time.sleep(0.05)
    return not children(os.getpid())


def wait_file(path, process, expires_monotonic):
    while not path.exists():
        if process.poll() is not None:
            message = "Persistent service exited before its round receipt"
            raise RuntimeError(message)
        if time.monotonic() >= expires_monotonic:
            message = "Independent native batch deadline exceeded"
            raise TimeoutError(message)
        time.sleep(0.2)
    return json.loads(path.read_text())


def checkpoint(root, pair, completed):
    """Archive only a completed pair while both services are at idle barriers."""
    paths = []
    for workers in (1, 4):
        paths.append(root / "run" / f"completed-{workers}-{pair:02d}.json")
        paths.extend(
            path
            for directory in (root / "evals" / f"workers{workers}").glob(
                f"*-round{pair:02d}-copy00"
            )
            for path in directory.rglob("*")
            if path.is_file()
        )
    manifest = {str(path.relative_to(root)): digest(path) for path in paths}
    receipt = root / "checkpoints" / f"pair-{pair:02d}-manifest.json"
    save(receipt, {"completed_evaluations": completed, "pair": pair, "files": manifest})
    archive = root / "checkpoints" / f"pair-{pair:02d}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path in [*paths, receipt]:
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {
        "pair": pair,
        "completed_evaluations": completed,
        "path": str(archive),
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
    }


def final_checkpoint(root):
    paths = [path for path in root.glob("*.json") if path.is_file()]
    paths.extend(root.glob("service-*.log"))
    paths.extend((root / "run").glob("*.json"))
    paths.extend((root / "logs").glob("*.log"))
    paths.extend(root.glob("evals/*/batch-summary.json"))
    manifest = root / "final-manifest.json"
    save(
        manifest,
        {"files": {str(path.relative_to(root)): digest(path) for path in paths}},
    )
    archive = root / "checkpoints/final-metadata.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path in [*paths, manifest]:
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {
        "path": str(archive),
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
    }


def verify_checkpoint(archive, expected_sha, out):
    if digest(archive) != expected_sha:
        message = "Checkpoint archive hash mismatch"
        raise ValueError(message)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
                or member.name in names
            ):
                message = "Unsafe or duplicate checkpoint member"
                raise ValueError(message)
            names.add(member.name)
        out.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                message = "Checkpoint member has no content"
                raise ValueError(message)
            with source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    manifests = [
        path
        for path in out.rglob("*manifest.json")
        if path.is_file()
        and (path.name.startswith("pair-") or path.name == "final-manifest.json")
    ]
    if len(manifests) != 1:
        message = "Checkpoint must contain exactly one covering manifest"
        raise ValueError(message)
    manifest = manifests[0]
    expected = json.loads(manifest.read_text())["files"]
    if names != set(expected) | {str(manifest.relative_to(out))}:
        message = "Checkpoint manifest does not cover every archive file"
        raise ValueError(message)
    for name, sha in expected.items():
        if digest(out / name) != sha:
            message = "Checkpoint member content hash mismatch"
            raise ValueError(message)
    save(
        out / "verification.json",
        {
            "archive_sha256": expected_sha,
            "files_verified": len(expected),
            "all_files_exact": True,
        },
    )


def preflight_existing_campaign():
    parent = REMOTE_ROOT.parent / "solver14-qualification72"
    status = json.loads((parent / "full-campaign-status.json").read_text())
    guard = json.loads((parent / "full-guard-status.json").read_text())
    if (
        status.get("phase")
        not in ("complete", "failed", "qualification_failed", "timed_out")
        or not guard.get("supervisor_reaped")
        or not guard.get("coordinator_group_empty")
    ):
        message = "Native14 campaign is not terminal with confirmed cleanup"
        raise RuntimeError(message)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    if (
        int(query.split(",")[1]) != 0
        or int(query.split(",")[2]) < 8192
        or any(
            "/usr/libexec/gnome-remote-desktop-daemon" not in row
            for row in applications.splitlines()
            if row
        )
    ):
        message = "Native GPU is busy or has insufficient headroom; no batch started"
        raise RuntimeError(message)
    return {
        "native14_terminal": status["phase"],
        "guard_reaped": True,
        "gpu": query,
        "compute_apps": applications,
    }


def campaign(root, deadline):
    if root != REMOTE_ROOT:
        message = "Unexpected isolated native batch root"
        raise ValueError(message)
    stage = json.loads((root / "stage-ready.json").read_text())
    check_files(stage["files"])
    idle = preflight_existing_campaign()
    if (root / "run").exists():
        message = "Batch campaign already exists; never restart implicitly"
        raise RuntimeError(message)
    for name in ("run", "evals", "checkpoints"):
        (root / name).mkdir()
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        message = "Cannot establish native child subreaper"
        raise RuntimeError(message)
    dependencies = dependency_snapshot(root)
    save(root / "dependency-hashes.json", dependencies)
    configuration = {
        "nonce": secrets.token_hex(16),
        "expires_at": time.time() + deadline,
        "expires_monotonic": time.monotonic() + deadline,
        "files": stage["files"],
        "warmup_pairs": 1,
        "measured_pairs": 5,
        "workload_cases": 6,
        "full_evaluations": 72,
        "idle_preflight": idle,
        "worker_cpus": stage["affinity"]["service_cpus"],
        "affinity": stage["affinity"],
    }
    save(root / "run/launch.json", configuration)
    state = {
        "phase": "running",
        "pid": os.getpid(),
        "birth": birth(os.getpid()),
        "started_at": time.time(),
        "completed_evaluations": 0,
        "checkpoints": [],
        "rounds": [],
        "batch_table_qualified": False,
        "main_single_eval_table_eligible": False,
    }
    save(root / "campaign-status.json", state)
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        message = "Launch coordinator at normal priority; it lowers itself only after spawning services"
        raise RuntimeError(message)
    os.sched_setaffinity(0, {stage["affinity"]["coordinator_cpu"]})
    processes, logs = {}, []
    exact = False
    try:
        for workers in (1, 4):
            environment = dict(stage["service_environment"])
            environment.update(CGA_NATIVE_BATCH_NONCE=configuration["nonce"])
            log = (root / f"service-{workers}.log").open("x")
            logs.append(log)
            command, outer_environment = service_launch(
                environment,
                configuration["worker_cpus"],
                stage["python"],
                [
                    "-B",
                    "-m",
                    "tools.optimization.native_batch_campaign",
                    "--root",
                    str(root),
                    "--service",
                    str(workers),
                ],
            )
            processes[workers] = subprocess.Popen(
                command,
                env=outer_environment,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        os.nice(19)
        for pair in range(6):
            for workers in (1, 4):
                wait_file(
                    root / "run" / f"ready-{workers}-{pair:02d}.json",
                    processes[workers],
                    configuration["expires_monotonic"],
                )
            for workers in paired_order(pair):
                wait_file(
                    root / "run" / f"ready-{workers}-{pair:02d}.json",
                    processes[workers],
                    configuration["expires_monotonic"],
                )
                save(
                    root / "run" / f"grant-{workers}-{pair:02d}.json",
                    {"nonce": configuration["nonce"], "workers": workers, "pair": pair},
                )
                receipt = wait_file(
                    root / "run" / f"completed-{workers}-{pair:02d}.json",
                    processes[workers],
                    configuration["expires_monotonic"],
                )
                if not receipt["exact"] or receipt["nonce"] != configuration["nonce"]:
                    message = "Completed paired workload failed strict original gates"
                    raise RuntimeError(message)
                if pair == 5:
                    code = processes[workers].wait(
                        timeout=max(
                            1, configuration["expires_monotonic"] - time.monotonic()
                        )
                    )
                    if code != 0:
                        message = "Final persistent service failed or did not drain"
                        raise RuntimeError(message)
                else:
                    # The preceding service must finish its post-round checks
                    # and reach the next idle barrier before the other runs.
                    wait_file(
                        root / "run" / f"ready-{workers}-{pair + 1:02d}.json",
                        processes[workers],
                        configuration["expires_monotonic"],
                    )
                state["completed_evaluations"] += 6
                state["rounds"].append(
                    {
                        "pair": pair,
                        "workers": workers,
                        "warmup": pair == 0,
                        "exact": True,
                    }
                )
                save(root / "campaign-status.json", state)
            state["checkpoints"].append(
                checkpoint(root, pair, state["completed_evaluations"])
            )
            save(root / "campaign-status.json", state)
        check_files(stage["files"])
        check_files(dependencies)
        exact = True
    except BaseException as error:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        clean = not children(os.getpid()) or cleanup(processes)
        for log in logs:
            log.close()
        state.update(
            phase="complete" if exact and clean else "failed",
            owned_children_reaped=clean,
            finished_at=time.time(),
            frozen_unchanged=all(
                digest(name) == sha for name, sha in stage["files"].items()
            ),
            dependencies_unchanged=exact,
        )
        save(root / "campaign-status.json", state)
    result = report(root)
    state["batch_table_qualified"] = result["batch_table_qualified"]
    save(root / "campaign-status.json", state)
    state["final_metadata"] = final_checkpoint(root)
    save(root / "campaign-status.json", state)
    return 0


def report(root: Path, output: Path | None = None):
    status = json.loads((root / "campaign-status.json").read_text())
    summaries = {
        workers: json.loads(
            (root / "evals" / f"workers{workers}/batch-summary.json").read_text()
        )
        for workers in (1, 4)
    }
    pairs, latencies = [], []
    for pair in range(1, 6):
        durations = {
            workers: summaries[workers]["rounds"][pair]["makespan_seconds"]
            for workers in (1, 4)
        }
        pairs.append(
            {
                "pair": pair,
                "workers1_seconds": durations[1],
                "workers4_seconds": durations[4],
                "workers1_evaluations_per_second": 6 / durations[1],
                "workers4_evaluations_per_second": 6 / durations[4],
                "batch_throughput_ratio_4_over_1": durations[1] / durations[4],
            }
        )
    for workers, summary in summaries.items():
        for seed, track in sorted(
            {(row["seed"], row["track"]) for row in summary["records"]}
        ):
            selected = [
                row
                for row in summary["records"]
                if row["seed"] == seed and row["track"] == track and not row["warmup"]
            ]
            latencies.append(
                {
                    "workers": workers,
                    "seed": seed,
                    "track": track,
                    "measured_samples": len(selected),
                    "submit_to_finish_seconds": [
                        row["timing"]["submit_to_finish_seconds"] for row in selected
                    ],
                    "service_seconds": [
                        row["timing"]["service_seconds"] for row in selected
                    ],
                    "median_submit_to_finish_seconds": statistics.median(
                        row["timing"]["submit_to_finish_seconds"] for row in selected
                    ),
                }
            )
    repeats = {
        str(workers): batch.repeat_coverage(summary["records"])
        for workers, summary in summaries.items()
    }
    qualified = (
        status["phase"] == "complete"
        and status["completed_evaluations"] == 72
        and status["owned_children_reaped"]
        and status["frozen_unchanged"]
        and status["dependencies_unchanged"]
        and all(
            summary["all_original_global_gates_exact"] and workload_coverage(summary)
            for summary in summaries.values()
        )
        and all(value["repeated_outputs_exact"] for value in repeats.values())
        and all(
            record["raw_netcdf_hashes_exact"]
            for summary in summaries.values()
            for record in summary["records"]
        )
    )
    metric_drift = {}
    for summary in summaries.values():
        for record in summary["records"]:
            for name, drift in record["metric_drift"].items():
                entry = metric_drift.setdefault(
                    name,
                    {
                        "exact": True,
                        "max_absolute": 0.0,
                        "max_relative": 0.0,
                        "null_relative_observations": 0,
                    },
                )
                entry["exact"] &= drift["exact"]
                entry["max_absolute"] = metric_maximum(
                    entry["max_absolute"], drift.get("absolute", 0.0)
                )
                relative = drift.get("relative")
                if relative is None:
                    entry["null_relative_observations"] += 1
                else:
                    entry["max_relative"] = metric_maximum(
                        entry["max_relative"], relative
                    )
    generator = random.Random(0)
    ratios = [row["batch_throughput_ratio_4_over_1"] for row in pairs]
    bootstrap = sorted(
        statistics.median(generator.choices(ratios, k=5)) for _ in range(10000)
    )
    result = {
        "scope": "Measured native batch throughput for identical six-case workloads, not single-evaluation solver speedup",
        "batch_table_qualified": qualified,
        "main_single_eval_table_eligible": False,
        "warmup_pair_excluded": True,
        "measured_pairs": pairs,
        "median_paired_batch_throughput_ratio": statistics.median(
            row["batch_throughput_ratio_4_over_1"] for row in pairs
        ),
        "geometric_mean_paired_batch_throughput_ratio": math.exp(
            statistics.mean(
                math.log(row["batch_throughput_ratio_4_over_1"]) for row in pairs
            )
        ),
        "per_request_latencies_separate": latencies,
        "repeat_coverage": repeats,
        "max_absolute_metric_drift": 0 if qualified else None,
        "max_relative_metric_drift": 0 if qualified else None,
        "timing_scope": "Canonical batch makespan: first submission through all request completions; imports and validation IO/hash work excluded. Persistent pools are warmed once, then alternating paired grants serialize GPU work. Other service stays at a drained idle barrier.",
    }
    result["max_drift_per_metric_against_original"] = metric_drift
    result["paired_bootstrap_median_ratio_95_interval"] = [
        bootstrap[250],
        bootstrap[9750],
    ]
    save(
        output if output is not None else root / "batch-qualification-summary.json",
        result,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--service", type=int, choices=(1, 4))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline", type=int, default=2100)
    args = parser.parse_args()
    if args.verify:
        if not args.sha256 or args.out is None:
            parser.error("--verify requires --sha256 and fresh --out")
        verify_checkpoint(args.verify, args.sha256, args.out)
        return 0
    if args.service:
        return service(args.root.resolve(), args.service)
    if args.report:
        report(args.root.resolve())
        return 0
    if not args.execute:
        print(
            json.dumps(
                {
                    "scope": "PLAN ONLY; no solver/GPU execution",
                    "workers": [1, 4],
                    "workload": "three frozen boundaries x P1/P3",
                    "persistent_service_pools": True,
                    "warmup_pairs": 1,
                    "measured_pairs": 5,
                    "full_evaluations": 72,
                    "alternating_orders": [paired_order(pair) for pair in range(6)],
                    "independent_deadline_seconds": args.deadline,
                    "native14_must_be_terminal_and_reaped": True,
                    "main_single_eval_table_eligible": False,
                },
                indent=2,
            )
        )
        return 0
    if not 1200 <= args.deadline <= 2400:
        parser.error("Independent campaign deadline must be1200..2400seconds")
    return campaign(args.root.resolve(), args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
