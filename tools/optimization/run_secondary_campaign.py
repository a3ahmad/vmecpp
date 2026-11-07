"""Supervise only the authorized frozen 72-evaluation secondary campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def save(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path(
        "/bulk/vmecpp-optimization-20261103/resident-solver8-capture-replica"
    ):
        message = "Unexpected authorized campaign directory"
        raise ValueError(message)
    specification = json.loads(args.spec.read_text())
    for name, expected in specification["expected_files"].items():
        if digest(root / name) != expected:
            message = "Frozen launch input changed: " + name
            raise ValueError(message)
    manifest = json.loads((root / "variant-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if not name.startswith(("lib/", "bin/")) and digest(root / name) != expected:
            message = "Frozen campaign source changed: " + name
            raise ValueError(message)
    relocation = json.loads((root / "relocation.json").read_text())
    for item in relocation:
        if digest(root / item["path"]) != item["after"]:
            message = "Relocated campaign binary changed: " + item["path"]
            raise ValueError(message)
    for label in ("baseline", "candidate"):
        configuration = json.loads((root / (label + "-config.json")).read_text())
        for name, expected in configuration["files"].items():
            if digest(Path(name)) != expected:
                message = "Frozen source, fixture or library changed: " + name
                raise ValueError(message)
    plan = json.loads((root / "campaign-plan.json").read_text())
    if (
        plan["full_evaluations"] != 72
        or plan["warmup_pairs"] != 1
        or plan["measured_pairs"] != 5
    ):
        message = "Unexpected frozen campaign schedule"
        raise ValueError(message)
    if (root / "paired").exists() or (root / "full-campaign-status.json").exists():
        message = "Full campaign already exists; refusing a duplicate"
        raise ValueError(message)
    utilization = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if any(int(value) for value in utilization.split()) or any(
        name != "/usr/libexec/gnome-remote-desktop-daemon"
        for name in applications.splitlines()
        if name
    ):
        message = "Secondary GPU is busy; no campaign started"
        raise RuntimeError(message)
    allowed = os.sched_getaffinity(0)
    if len(allowed) > 1:
        os.sched_setaffinity(0, allowed - {0})
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("VMECPP_") or key in (
            "LD_PRELOAD",
            "CGA_PROFILE",
            "CUDA_INJECTION64_PATH",
            "NVTX_INJECTION64_PATH",
            "CGA_OPTIMIZATION_WRAPPER_SHA256",
            "CGA_OPTIMIZATION_CONFIG_SHA256",
        ):
            environment.pop(key)
    environment.update(
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    status_path = root / "full-campaign-status.json"
    started = time.time()
    state = {
        "phase": "starting",
        "purpose": "original baseline versus frozen solver 8",
        "command": plan["full_campaign"],
        "supervisor_pid": os.getpid(),
        "supervisor_sha256": digest(Path(__file__)),
        "spec_sha256": digest(args.spec),
        "started_at": started,
        "timeout_seconds": args.timeout_seconds,
        "expected_evaluations": 72,
        "warmup_evaluations": 12,
        "expected_measured_evaluations": 60,
        "expected_case_pairs": 36,
        "prelaunch_gpu_utilization": utilization,
        "existing_desktop_context": applications,
    }
    save(status_path, state)
    with (root / "logs/full-campaign-coordinator.log").open("x") as output:
        process = subprocess.Popen(
            plan["full_campaign"],
            cwd=root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        state.update(phase="running", coordinator_pid=process.pid)
        timed_out = False
        while process.poll() is None:
            path = root / "paired/samples.json"
            samples = json.loads(path.read_text()) if path.is_file() else []
            completed = {}
            for sample in samples:
                completed.setdefault(
                    (sample["pair"], sample["seed"], sample["track"]), []
                ).append(sample)
            valid_pairs = sum(
                len(rows) == 2
                and {row["label"] for row in rows} == {"baseline", "candidate"}
                and all(
                    row["exit_status"] == 0
                    and (row.get("original_reference") or {})
                    .get("gate", {})
                    .get("exact")
                    for row in rows
                )
                for rows in completed.values()
            )
            state.update(
                updated_at=time.time(),
                completed_evaluations=len(samples),
                completed_case_pairs=valid_pairs,
                completed_measured_evaluations=sum(
                    not sample["warmup"] for sample in samples
                ),
                failed_evaluations=sum(
                    sample["exit_status"] != 0 for sample in samples
                ),
            )
            save(status_path, state)
            if time.time() - started > args.timeout_seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(10)
        state.update(
            phase="timed_out"
            if timed_out
            else "complete"
            if process.returncode == 0
            else "failed",
            exit_status=process.wait(),
            finished_at=time.time(),
        )
    path = root / "paired/samples.json"
    samples = json.loads(path.read_text()) if path.is_file() else []
    completed = {}
    for sample in samples:
        completed.setdefault(
            (sample["pair"], sample["seed"], sample["track"]), []
        ).append(sample)
    final_pairs = sum(
        len(rows) == 2
        and {row["label"] for row in rows} == {"baseline", "candidate"}
        and all(
            row["exit_status"] == 0
            and (row.get("original_reference") or {}).get("gate", {}).get("exact")
            for row in rows
        )
        for rows in completed.values()
    )
    state.update(
        completed_evaluations=len(samples),
        completed_case_pairs=final_pairs,
        completed_measured_evaluations=sum(not sample["warmup"] for sample in samples),
        failed_evaluations=sum(sample["exit_status"] != 0 for sample in samples),
        failed_global_reference_checks=sum(
            not (sample.get("original_reference") or {}).get("gate", {}).get("exact")
            for sample in samples
        ),
    )
    if process.returncode == 0 and not timed_out:
        from tools.optimization.report_pairs import summarize_pairs

        rows = summarize_pairs(
            samples, root / "paired", require_original_reference=True
        )
        save(root / "full-campaign-summary.json", rows)
        qualified = (
            len(samples) == 72
            and len(rows) == 6
            and all(row["main_table_eligible"] for row in rows)
        )
        state.update(main_table_qualified=qualified)
        if not qualified:
            state["phase"] = "qualification_failed"
    save(status_path, state)


if __name__ == "__main__":
    main()
