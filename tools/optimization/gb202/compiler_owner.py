"""Bounded descendant-aware replay qualification; no equilibrium timing claim."""

# The supervisor must persist interruption failures before owned-tree cleanup.
# ruff: noqa: T201, BLE001
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / "build/optimization/resident-solver-energy-overlap1"
TOOL = (
    ROOT
    / "build/optimization/sanitizer-latest-probe/extracted/usr/local/cuda-13.3/compute-sanitizer/compute-sanitizer"
)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def descendants():
    pending = [os.getpid()]
    result = {}
    while pending:
        pid = pending.pop()
        for task in Path(f"/proc/{pid}/task").glob("*/children"):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                for item in task.read_text().split():
                    child = int(item)
                    if child not in result:
                        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                            result[child] = birth(child)
                            pending.append(child)
    return result


def reap(process, job):
    process.poll()
    if process.returncode is not None and "direct_exit_status" not in job:
        job.update(direct_exit_status=process.returncode, direct_reaped_at=time.time())
    for pid in descendants():
        if pid == process.pid:
            continue
        with contextlib.suppress(ChildProcessError, ProcessLookupError):
            reaped, status = os.waitpid(pid, os.WNOHANG)
            if reaped:
                job.setdefault("adopted_reaped", []).append(
                    {"pid": pid, "wait_status": status}
                )


def run_job(command, name, budget, environment, directory):
    job = {
        "name": name,
        "command": command,
        "budget_seconds": budget,
        "started": time.time(),
        "phase": "starting",
        "cleanup_actions": [],
    }
    path = directory / (name + "-status.json")
    process = None
    try:
        with (directory / (name + ".log")).open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            job.update(
                pid=process.pid,
                birth=birth(process.pid),
                uid=os.getuid(),
                phase="running",
            )
            save(path, job)
            deadline = time.monotonic() + budget
            while True:
                reap(process, job)
                owned = descendants()
                job["owned_descendants"] = owned
                save(path, job)
                if process.returncode is not None and not owned:
                    break
                if time.monotonic() >= deadline:
                    message = "Profiler/target descendant deadline"
                    raise TimeoutError(message)
                time.sleep(0.1)
    except BaseException as error:
        job.update(error=repr(error), phase="failed_before_cleanup")
        save(path, job)
    finally:
        job["before_cleanup"] = descendants()
        save(path, job)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            owned = descendants()
            for pid, expected in reversed(list(owned.items())):
                with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                    if (
                        birth(pid) == expected
                        and Path(f"/proc/{pid}").stat().st_uid == os.getuid()
                    ):
                        os.kill(pid, sig)
                        job["cleanup_actions"].append(
                            {"pid": pid, "birth": expected, "signal": int(sig)}
                        )
            save(path, job)
            limit = time.monotonic() + 4
            while descendants() and time.monotonic() < limit:
                if process is not None:
                    reap(process, job)
                time.sleep(0.05)
            if not descendants():
                break
        if process is not None:
            reap(process, job)
        job.update(
            phase="terminal", ended=time.time(), owned_children_reaped=not descendants()
        )
        save(path, job)
    return job


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not (args.execute or args.self_test):
        print(
            "Plan: ordered-reference + graph1/pinned1, 38 frames each; normal/mem/sync/race"
        )
        return
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    os.sched_setaffinity(0, {20})
    environment = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.9/lib64"
    if args.self_test:
        directory = OUT / "lifecycle-tests"
        directory.mkdir(exist_ok=False)
        success = run_job(
            [
                sys.executable,
                "-c",
                "import os,time; p=os.fork(); time.sleep(.4) if p==0 else None",
            ],
            "adopted-outlives-cli",
            5,
            environment,
            directory,
        )
        assert success["direct_exit_status"] == 0
        assert success["owned_children_reaped"]
        assert not success["cleanup_actions"]
        assert success.get("adopted_reaped")
        failure = run_job(
            [sys.executable, "-c", "import os,time; os.fork(); time.sleep(20)"],
            "timeout-owned-tree",
            0.3,
            environment,
            directory,
        )
        assert failure.get("error")
        assert failure["cleanup_actions"]
        assert failure["owned_children_reaped"]
        print("PASS: delayed adopted descendant and timeout tree cleanup")
        return
    directory = OUT / "gpu-gates1"
    directory.mkdir(exist_ok=False)
    frozen_paths = [
        *(OLD / "source").glob("*"),
        OLD / "solver-backend.so",
        OUT / "source.cc",
        OUT / "focused-replay",
        Path(__file__),
        TOOL,
    ]
    old_hashes = json.loads((OLD / "gpu-gates1/status.json").read_text())["hashes"]
    for path, expected in old_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    frozen = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_paths
        if path.is_file()
    }
    status = {
        "supervisor_pid": os.getpid(),
        "supervisor_birth": birth(os.getpid()),
        "phase": "running",
        "jobs": [],
        "files": frozen,
        "scope": "38-frame component control only; no full adapter or speed qualification",
    }
    save(directory / "status.json", status)
    for command, name in (
        ([str(TOOL), "--version"], "tool-version"),
        (
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.used,utilization.gpu",
                "--format=csv",
            ],
            "hardware",
        ),
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=15,
            check=False,
        )
        save(
            directory / (name + ".json"),
            {
                "command": command,
                "exit_status": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        assert result.returncode == 0
    try:
        for check, budget in (
            ("normal", 40),
            ("memcheck", 90),
            ("synccheck", 90),
            ("racecheck", 420),
        ):
            command = (
                []
                if check == "normal"
                else [str(TOOL), "--tool", check, "--error-exitcode", "91"]
            )
            command += [str(OUT / "focused-replay"), str(ROOT / "build/optimization")]
            job = run_job(command, check, budget, environment, directory)
            text = (directory / (check + ".log")).read_text()
            job["exact_control_pass"] = all(
                marker in text
                for marker in (
                    "ordered reference, 38 recorded frames",
                    "outer_graph=1 pinned=1, 38 frames exact",
                    "PASS: 8 control triples",
                )
            )
            pattern = (
                r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
                if check == "racecheck"
                else r"ERROR SUMMARY: 0 errors"
            )
            job["explicit_zero_summary"] = (
                bool(re.search(pattern, text)) if check != "normal" else None
            )
            job["passed"] = (
                job.get("direct_exit_status") == 0
                and not job.get("error")
                and job["owned_children_reaped"]
                and not job["cleanup_actions"]
                and job["exact_control_pass"]
                and (check == "normal" or job["explicit_zero_summary"])
            )
            status["jobs"].append(job)
            save(directory / "status.json", status)
            print(check, job["passed"], flush=True)
            if not job["passed"]:
                break
    finally:
        status.update(
            phase="terminal",
            owned_children_reaped=not descendants(),
            frozen_unchanged=all(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                for path, expected in frozen.items()
            ),
        )
        status["passed"] = (
            len(status["jobs"]) == 4
            and all(job["passed"] for job in status["jobs"])
            and status["owned_children_reaped"]
            and status["frozen_unchanged"]
        )
        save(directory / "status.json", status)
    raise SystemExit(0 if status["passed"] else 1)


if __name__ == "__main__":
    main()
