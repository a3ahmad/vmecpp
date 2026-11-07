"""Bounded, isolated A100 pilot operations; no provider or credential API calls.

Bootstrap/run default to a printed plan. Root owns the external one-hour lease watchdog,
upload, incremental collection, and prompt verified destruction.
"""

from __future__ import annotations

# CLI status is intentional; optional oracle imports belong only in its worker.
# ruff: noqa: T201, PLC0415
import argparse
import contextlib
import copy
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __package__:
    from .campaign_gates import (
        clean_environment,
        complete_capture,
        pair_gate,
        read_capture,
    )
else:
    from campaign_gates import (
        clean_environment,
        complete_capture,
        pair_gate,
        read_capture,
    )

RESERVE_SECONDS = 600
FLAGS = {
    "VMECPP_GPU_SOLVER": "1",
    "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
    "VMECPP_GPU_RESIDUAL_ROUNDS": "12",
    "VMECPP_GPU_RESIDUAL_GRAPH": "1",
    "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
    "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
    "VMECPP_GPU_SOLVER_GRAPH": "1",
    "VMECPP_GPU_SOLVER_PINNED_SCALARS": "1",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inside(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    target = (root / name).resolve()
    if (
        parts.is_absolute()
        or ".." in parts.parts
        or root.resolve() not in target.parents
    ):
        message = "Payload path escapes the isolated root"
        raise ValueError(message)
    return target


def verify_payload(root: Path) -> dict:
    manifest = json.loads((root / "payload-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(inside(root, name)) != expected:
            message = "Frozen payload hash mismatch: " + name
            raise ValueError(message)
    return manifest


def preflight(root: Path) -> dict:
    manifest = verify_payload(root)
    required = (
        "archives/oracle-packages.tar.gz",
        "archives/nsys-cli.tar.gz",
        "archives/ncu-cli.tar.gz",
        "bin/patchelf",
        "bin/solver-replay",
        "bin/roofline-strict",
        "runtime/libcudart.so.12",
        "profiling/solver-stage-profile-shim.so",
        "fixtures/manifest.json",
        "stella-snapshot/configs/oracle.yaml",
    )
    for name in required:
        if name not in manifest["files"]:
            message = "Required pilot asset absent: " + name
            raise ValueError(message)
    fixtures = json.loads((root / "fixtures/manifest.json").read_text())
    if len(fixtures) != 3:
        message = "Pilot requires the unchanged three-fixture main suite"
        raise ValueError(message)
    references = []
    for fixture in fixtures:
        seed = root / "fixtures" / Path(fixture["path"]).name
        for track in ("P1", "P3"):
            directory = root / "references" / f"{seed.stem}-{track}"
            report = json.loads((directory / "report.json").read_text())
            if not complete_capture(report, read_capture(directory)):
                message = "Incomplete immutable global reference"
                raise ValueError(message)
            references.append(
                {
                    "seed": seed.name,
                    "track": track,
                    "report_sha256": digest(directory / "report.json"),
                }
            )
    return {
        "scope": "CPU-only input/hash/dependency preflight; no remote or GPU test claimed",
        "verified_files": len(manifest["files"]),
        "complete_original_references": references,
        "plan": planned_commands(root),
    }


def safe_extract(archive: Path, destination: Path) -> None:
    if sys.version_info < (3, 12):
        message = "Safe pilot extraction requires Python3.12"
        raise RuntimeError(message)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = []
        for member in bundle:
            normalized = PurePosixPath(member.name)
            if not normalized.parts:
                if not member.isdir() or member.size != 0:
                    message = (
                        "Archive root entry must be a directory, never a file/link"
                    )
                    raise ValueError(message)
                continue
            inside(destination, member.name)
            members.append(member)
        bundle.extractall(destination, members=members, filter="data")


def remaining_timeout(
    expires_at: float, maximum: int, reserve: int = RESERVE_SECONDS
) -> int:
    remaining = int(expires_at - time.time() - reserve)
    if remaining <= 0:
        message = "Retrieval reserve reached; no new workload started"
        raise TimeoutError(message)
    return min(maximum, remaining)


def check_lease_window(started_at: float, expires_at: float) -> None:
    if not 0 < expires_at - started_at <= 3600:
        message = "Pilot requires an external watchdog of at most one hour"
        raise ValueError(message)
    remaining_timeout(expires_at, 1)


def environment(root: Path, variant: str | None = None) -> dict[str, str]:
    # Do not forward local provider credentials or arbitrary host configuration.
    result = clean_environment(
        {
            key: value
            for key, value in os.environ.items()
            if key
            in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES")
        }
    )
    result["PYTHONPATH"] = str(root)
    result["TMPDIR"] = str(root / "work/tmp")
    result["XDG_CACHE_HOME"] = str(root / "work/cache")
    result["MPLCONFIGDIR"] = str(root / "work/cache/matplotlib")
    result["CUDA_CACHE_PATH"] = str(root / "work/cache/cuda")
    result["LD_LIBRARY_PATH"] = ":".join(
        (
            str(root / "runtime"),
            str(root / "work/lib"),
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        )
    )
    if variant == "candidate":
        result.update(FLAGS)
        result["VMECPP_FORWARD_REFERENCE_SO"] = str(extension(root, variant))
        result["LD_PRELOAD"] = str(root / "work/lib/solver-shim.so")
    return result


def python_path(root: Path, variant: str) -> Path:
    return root / "work" / variant / "bin/python"


def extension(root: Path, variant: str) -> Path:
    return (
        root
        / "work"
        / variant
        / "lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )


def launch(
    root: Path,
    command: list[str],
    name: str,
    expires_at: float,
    *,
    maximum: int = 180,
    env: dict | None = None,
    check: bool = True,
) -> int | str:
    limit = remaining_timeout(expires_at, maximum)
    status_path = root / "results/commands" / (name + ".json")
    started = time.time()
    status = {
        "command": command,
        "phase": "running",
        "started_at": started,
        "timeout_seconds": limit,
    }
    log = root / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as output:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env or environment(root),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status["pid"] = process.pid
        save(status_path, status)
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = "timeout"
    status.update(
        phase="terminal",
        exit_status=code,
        finished_at=time.time(),
        process_wall_seconds=time.time() - started,
    )
    save(status_path, status)
    if check and code != 0:
        message = "Pilot stage failed: " + name
        raise RuntimeError(message)
    return code


def planned_commands(root: Path) -> dict:
    return {
        "scope": "future one-hour pilot only; no lease is created by this runner",
        "external_watchdog_required": True,
        "maximum_case_pairs": 6,
        "maximum_original_and_candidate_case_evaluations": 12,
        "retrieval_reserve_seconds": RESERVE_SECONDS,
        "bootstrap_os_packages": [
            "binutils",
            "python3-venv",
            "libhdf5-dev",
            "libnetcdf-dev",
            "liblapack-dev",
            "libgomp1",
            "libgl1",
            "libsm6",
            "libxext6",
        ],
        "calibration": [str(root / "bin/roofline-strict"), "10", "512", "65536"],
        "profile_scope": "Nsight first final-grid(ns99) GPU-context node window; Ncu captured-frame hot kernels only, never whole-solver roofline percentages",
        "no_qualification": "One pair per case cannot qualify a speedup table. Native72 campaign is separate.",
    }


def bootstrap(root: Path, expires_at: float) -> None:
    if not str(root).startswith("/opt/vmecpp-pilot-"):
        message = "Bootstrap is restricted to a fresh /opt/vmecpp-pilot-* leased-container root"
        raise ValueError(message)
    manifest = verify_payload(root)
    if sys.version_info[:2] != (3, 12):
        message = "Pinned extension requires Python3.12"
        raise RuntimeError(message)
    (root / "work").mkdir(exist_ok=False)
    (root / "work/tmp").mkdir()
    (root / "work/cache").mkdir()
    # OS package changes occur only inside the newly leased container.
    for source in (
        Path("/etc/apt/sources.list"),
        Path("/etc/apt/sources.list.d/ubuntu.sources"),
    ):
        if source.is_file():
            text = source.read_text()
            source.write_text(
                text.replace(
                    "http://archive.ubuntu.com", "https://archive.ubuntu.com"
                ).replace("http://security.ubuntu.com", "https://security.ubuntu.com")
            )
    packages = planned_commands(root)["bootstrap_os_packages"]
    launch(
        root,
        [
            "apt-get",
            "-o",
            "Acquire::Retries=1",
            "-o",
            "Acquire::http::Timeout=15",
            "update",
            "-qq",
        ],
        "apt-update",
        expires_at,
        maximum=120,
    )
    apt_env = environment(root)
    apt_env["DEBIAN_FRONTEND"] = "noninteractive"
    launch(
        root,
        ["apt-get", "install", "-y", "-qq", *packages],
        "apt-libraries",
        expires_at,
        maximum=180,
        env=apt_env,
    )
    for variant in ("baseline", "candidate"):
        launch(
            root,
            [sys.executable, "-m", "venv", str(root / "work" / variant)],
            "venv-" + variant,
            expires_at,
            maximum=60,
        )
    baseline_site = root / "work/baseline/lib/python3.12/site-packages"
    candidate_site = root / "work/candidate/lib/python3.12/site-packages"
    safe_extract(root / "archives/oracle-packages.tar.gz", baseline_site)
    (baseline_site / "pilot.pth").write_text(str(root) + "\n")
    shutil.copytree(root / "python/vmecpp", candidate_site / "vmecpp")
    (candidate_site / "oracle.pth").write_text(
        str(baseline_site) + "\n" + str(root) + "\n"
    )
    if (
        digest(extension(root, "baseline")) != manifest["original_extension_sha256"]
        or digest(extension(root, "candidate")) != manifest["matching_extension_sha256"]
    ):
        message = "Extracted original/matching extension hash mismatch"
        raise ValueError(message)
    shutil.copytree(root / "lib", root / "work/lib")
    (root / "work/bin").mkdir()
    shutil.copy2(root / "bin/solver-replay", root / "work/bin/solver-replay")
    relocation = []
    targets = [
        *sorted((root / "work/lib").glob("*.so")),
        root / "work/bin/solver-replay",
    ]
    for path in targets:
        before = digest(path)
        dynamic = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        for needed in re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic):
            if needed.startswith("/mnt/f/"):
                replacement = (
                    str(extension(root, "candidate"))
                    if "_vmecpp" in needed
                    else Path(needed).name
                )
                launch(
                    root,
                    [
                        str(root / "bin/patchelf"),
                        "--replace-needed",
                        needed,
                        replacement,
                        str(path),
                    ],
                    "replace-" + path.name + "-" + Path(needed).name,
                    expires_at,
                    maximum=15,
                )
        rpath = ":".join(
            (
                str(root / "work/lib"),
                str(root / "runtime"),
                str(extension(root, "candidate").parent),
            )
        )
        launch(
            root,
            [str(root / "bin/patchelf"), "--set-rpath", rpath, str(path)],
            "rpath-" + path.name,
            expires_at,
            maximum=15,
        )
        relocation.append(
            {
                "path": str(path.relative_to(root)),
                "before": before,
                "after": digest(path),
            }
        )
    save(root / "relocation.json", relocation)
    shutil.copytree(root / "profiling", root / "work/profiling")
    observer = root / "work/profiling/solver-stage-profile-shim.so"
    before = digest(observer)
    launch(
        root,
        [
            str(root / "bin/patchelf"),
            "--set-rpath",
            str(root / "runtime"),
            str(observer),
        ],
        "observer-rpath",
        expires_at,
        maximum=15,
    )
    save(
        root / "profile-observer-relocation.json",
        {
            "path": str(observer.relative_to(root)),
            "before": before,
            "after": digest(observer),
        },
    )
    targets.append(observer)
    if (root / "optional/lib").is_dir():
        shutil.copytree(root / "optional/lib", root / "work/optional/lib")
        shutil.copytree(root / "optional/bin", root / "work/optional/bin")
        optional_relocation = []
        for path in [
            *sorted((root / "work/optional/lib").glob("*.so")),
            *sorted((root / "work/optional/bin").iterdir()),
        ]:
            before = digest(path)
            dynamic = subprocess.run(
                ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
            ).stdout
            for needed in re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic):
                if needed.startswith("/mnt/f/"):
                    launch(
                        root,
                        [
                            str(root / "bin/patchelf"),
                            "--replace-needed",
                            needed,
                            Path(needed).name,
                            str(path),
                        ],
                        "optional-replace-" + path.name + "-" + Path(needed).name,
                        expires_at,
                        maximum=15,
                    )
            launch(
                root,
                [
                    str(root / "bin/patchelf"),
                    "--set-rpath",
                    ":".join((str(root / "work/optional/lib"), str(root / "runtime"))),
                    str(path),
                ],
                "optional-rpath-" + path.name,
                expires_at,
                maximum=15,
            )
            optional_relocation.append(
                {
                    "path": str(path.relative_to(root)),
                    "before": before,
                    "after": digest(path),
                }
            )
        save(root / "optional-relocation.json", optional_relocation)
        targets.extend(root / record["path"] for record in optional_relocation)
    for name, archive in (("nsys", "nsys-cli.tar.gz"), ("ncu", "ncu-cli.tar.gz")):
        safe_extract(root / "archives" / archive, root / "work/profilers" / name)
    for index, path in enumerate(
        [
            *targets,
            extension(root, "baseline"),
            extension(root, "candidate"),
            root / "bin/roofline-strict",
        ]
    ):
        result = subprocess.run(
            ["ldd", str(path)],
            env=environment(root),
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        (root / "logs" / f"ldd-{index}.log").write_text(output)
        if "not found" in output or result.returncode != 0:
            message = "Unresolved copied ELF dependency: " + path.name
            raise RuntimeError(message)
    for variant in ("baseline", "candidate"):
        launch(
            root,
            [
                str(python_path(root, variant)),
                "-c",
                "import importlib.metadata as m; import vmecpp, constellaration, booz_xform; assert [m.version(x) for x in ('vmecpp','constellaration','booz-xform')]==['0.4.11','0.2.6','0.0.9']; print('Pinned imports verified')",
            ],
            "imports-" + variant,
            expires_at,
            maximum=60,
        )
    save(
        root / "bootstrap-status.json",
        {
            "phase": "complete",
            "finished_at": time.time(),
            "immutable_payload_verified": True,
            "relocated_copies": relocation,
        },
    )


def make_wrappers(root: Path) -> dict[str, Path]:
    wrappers = {}
    files = {
        name: sha
        for name, sha in verify_payload(root)["files"].items()
        if name.startswith(("tools/", "stella-snapshot/", "fixtures/", "references/"))
    }
    for item in json.loads((root / "relocation.json").read_text()):
        files[item["path"]] = item["after"]
    for variant in ("baseline", "candidate"):
        files_variant = {
            **files,
            str(extension(root, variant).relative_to(root)): digest(
                extension(root, variant)
            ),
        }
        for path in extension(root, variant).parents[1].glob("*.py"):
            files_variant[str(path.relative_to(root))] = digest(path)
        path = root / "work/bin" / (variant + "-python")
        config = {
            "variant": variant,
            "files": files_variant,
            "flags": FLAGS if variant == "candidate" else {},
            "python": str(python_path(root, variant)),
            "wrapper": str(path),
        }
        save(root / "work" / (variant + "-config.json"), config)
        argv = [
            str(python_path(root, "baseline")),
            "-B",
            str(Path(__file__).resolve()),
            "exec",
            "--root",
            str(root),
            "--variant",
            variant,
            "--",
        ]
        path.write_text("#!/bin/sh\nexec " + shlex.join(argv) + ' "$@"\n')
        path.chmod(0o755)
        wrappers[variant] = path
    return wrappers


def execute_wrapper(root: Path, variant: str, arguments: list[str]) -> None:
    config_path = root / "work" / (variant + "-config.json")
    config = json.loads(config_path.read_text())
    for name, expected in config["files"].items():
        if digest(inside(root, name)) != expected:
            message = "Frozen execution input changed: " + name
            raise ValueError(message)
    env = environment(root, variant)
    env["CGA_OPTIMIZATION_WRAPPER_SHA256"] = hashlib.sha256(
        Path(config["wrapper"]).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    env["CGA_OPTIMIZATION_CONFIG_SHA256"] = digest(config_path)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if variant == "candidate" and "--out" in arguments:
        output = Path(arguments[arguments.index("--out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        env["VMECPP_GPU_SOLVER_TRACE"] = str(output.with_suffix(".gpu-trace.jsonl"))
    os.execve(config["python"], [config["python"], "-B", *arguments], env)


def pilot_pair_gate(baseline, candidate, baseline_capture, candidate_capture, **kwargs):
    """Keep diagnostic trace overhead outside campaign timing qualification."""
    captures = copy.deepcopy((baseline_capture, candidate_capture))
    traced = False
    for capture in captures:
        runtime = capture.get("provenance", {}).get("experimental_runtime", {})
        traced = traced or runtime.get("VMECPP_GPU_SOLVER_TRACE") is not None
        runtime.pop("VMECPP_GPU_SOLVER_TRACE", None)
    gate = pair_gate(baseline, candidate, *captures, **kwargs)
    gate["diagnostic_trace_present"] = traced
    gate["main_table_eligible"] = False
    gate["scope"] = (
        "pilot correctness only; any trace overhead excludes timing qualification"
    )
    return gate


def report_gate(root: Path, output: Path, seed: Path, track: str) -> dict:
    reference = root / "references" / f"{seed.stem}-{track}"
    return pilot_pair_gate(
        json.loads((reference / "report.json").read_text()),
        json.loads((output / "report.json").read_text()),
        read_capture(reference),
        read_capture(output),
        require_harness=False,
    )


def run_pilot(
    root: Path, expires_at: float, maximum_pairs: int, *, include_optional: bool = False
) -> None:
    if maximum_pairs not in range(1, 7):
        message = "Pilot permits at most six case pairs"
        raise ValueError(message)
    if json.loads((root / "bootstrap-status.json").read_text())["phase"] != "complete":
        message = "Verified bootstrap must finish before the pilot"
        raise ValueError(message)
    verify_payload(root)
    if (root / "pilot-status.json").exists():
        message = "Refusing a duplicate pilot"
        raise ValueError(message)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    state = {
        "purpose": "one-hour A100 pilot only; not performance qualification",
        "phase": "running",
        "started_at": time.time(),
        "expires_at": expires_at,
        "retrieval_reserve_seconds": RESERVE_SECONDS,
        "cpu_affinity": [cpu],
        "case_pairs": [],
        "profiles": [],
    }
    save(root / "pilot-status.json", state)
    try:
        wrappers = make_wrappers(root)
        launch(
            root,
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            "gpu-inventory",
            expires_at,
            maximum=15,
        )
        devices = (root / "logs/gpu-inventory.log").read_text().splitlines()
        if len(devices) != 1 or "A100" not in devices[0]:
            message = "This frozen pilot targets A100 hardware"
            raise RuntimeError(message)
        launch(
            root,
            [str(root / "bin/roofline-strict"), "10", "512", "65536"],
            "strict-calibration",
            expires_at,
            maximum=60,
        )
        calibration = json.loads((root / "logs/strict-calibration.log").read_text())
        save(root / "results/strict-calibration.json", calibration)
        fixtures = json.loads((root / "fixtures/manifest.json").read_text())
        cases = [
            (root / "fixtures" / Path(fixture["path"]).name, track)
            for fixture in fixtures
            for track in ("P1", "P3")
        ]
        cases.sort(
            key=lambda case: (
                case[0].name != "public_p3_seed.json" or case[1] != "P3",
                case[0].name,
                case[1],
            )
        )
        for index, (seed, track) in enumerate(cases[:maximum_pairs]):
            # Retain extra profiling/retrieval headroom; estimate is not a result.
            prior = json.loads((root / "pilot-plan.json").read_text())[
                "prior_baseline_case_medians_seconds"
            ]
            baseline_estimate = next(
                item["seconds"]
                for item in prior
                if item["seed"] == seed.name and item["track"] == track
            )
            pair_estimate = max(120, int(2 * baseline_estimate * 1.25))
            if expires_at - time.time() < RESERVE_SECONDS + 300 + pair_estimate:
                state["stopped_before_next_pair"] = "insufficient conservative reserve"
                break
            observations = {}
            for variant in ("baseline", "candidate"):
                output = root / "results/pilots" / f"{seed.stem}-{track}" / variant
                command = [
                    str(wrappers[variant]),
                    str(root / "tools/optimization/benchmark.py"),
                    "--stella",
                    str(root / "stella-snapshot"),
                    "--seed",
                    str(seed),
                    "--track",
                    track,
                    "--out",
                    str(output),
                ]
                launch(
                    root, command, f"pilot-{index}-{variant}", expires_at, maximum=300
                )
                gate = report_gate(root, output, seed, track)
                observations[variant] = {
                    "output": str(output.relative_to(root)),
                    "global_gate": gate,
                }
                if variant == "candidate":
                    trace = output.with_suffix(".gpu-trace.jsonl")
                    gpu_updates = sum(
                        bool(json.loads(line).get("gpu"))
                        for line in trace.read_text().splitlines()
                        if line
                    )
                    observations[variant]["gpu_updates"] = gpu_updates
                    if gpu_updates <= 0:
                        message = (
                            "Candidate did not demonstrate actual GPU solver updates"
                        )
                        raise RuntimeError(message)
                if not gate["exact"]:
                    message = "Original-global identity/deck/output parity failed"
                    raise RuntimeError(message)
            a = root / observations["baseline"]["output"]
            b = root / observations["candidate"]["output"]
            gate = pilot_pair_gate(
                json.loads((a / "report.json").read_text()),
                json.loads((b / "report.json").read_text()),
                read_capture(a),
                read_capture(b),
            )
            state["case_pairs"].append(
                {
                    "seed": seed.name,
                    "track": track,
                    "gate": gate,
                    "observations": observations,
                    "timing_scope": "correctness-only pilot pair; candidate scalar trace excludes timing qualification",
                }
            )
            save(root / "pilot-status.json", state)
            if not gate["exact"]:
                message = "Paired pilot parity failed"
                raise RuntimeError(message)
        if not state["case_pairs"]:
            message = (
                "No complete baseline/candidate global-parity pair; profiles forbidden"
            )
            raise RuntimeError(message)
        state["canonical_parity_complete"] = True
        if expires_at - time.time() >= RESERVE_SECONDS + 300:
            profiles(root, expires_at, state)
        else:
            state["profiles_skipped"] = "retrieval reserve"
        if include_optional and expires_at - time.time() >= RESERVE_SECONDS + 420:
            optional_tools(root, expires_at, state)
        elif include_optional:
            state["optional_tools_skipped"] = (
                "canonical work/retrieval reserve takes priority"
            )
        state["phase"] = "complete"
    except Exception as error:
        state.update(
            phase="failed", failure_type=type(error).__name__, failure=str(error)
        )
        raise
    finally:
        state["finished_at"] = time.time()
        save(root / "pilot-status.json", state)


def profiles(root: Path, expires_at: float, state: dict) -> None:
    observer_record = json.loads(
        (root / "profile-observer-relocation.json").read_text()
    )
    if digest(inside(root, observer_record["path"])) != observer_record["after"]:
        message = "Copied profiling observer changed after bootstrap"
        raise RuntimeError(message)
    output = root / "results/profiles/nsys-evaluation"
    nsys = root / "work/profilers/nsys/target-linux-x64/nsys"
    prefix = root / "results/profiles/timeline"
    marker = root / "results/profiles/final-grid-marker.jsonl"
    profile_environment = environment(root, "candidate")
    profile_environment["LD_PRELOAD"] = ":".join(
        (
            str(root / "work/profiling/solver-stage-profile-shim.so"),
            str(root / "work/lib/solver-shim.so"),
        )
    )
    profile_environment["VMECPP_GPU_PROFILE_FINAL_GRID"] = "1"
    profile_environment["VMECPP_GPU_PROFILE_MARKER"] = str(marker)
    env_args = [
        "env",
        *[key + "=" + value for key, value in profile_environment.items()],
    ]
    command = [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--duration=3",
        "--kill=none",
        "--wait=all",
        "--force-overwrite=true",
        "--output=" + str(prefix),
        *env_args,
        str(python_path(root, "candidate")),
        str(Path(__file__).resolve()),
        "profile-worker",
        "--root",
        str(root),
        "--evaluation-out",
        str(output),
    ]
    code = launch(
        root, command, "nsys-node-window", expires_at, maximum=180, check=False
    )
    worker = root / "results/profiles/nsys-worker-status.json"
    deadline = min(expires_at - RESERVE_SECONDS, time.time() + 120)
    while time.time() < deadline:
        if (
            worker.is_file()
            and json.loads(worker.read_text()).get("phase") == "terminal"
        ):
            break
        time.sleep(1)
    worker_status = json.loads(worker.read_text()) if worker.is_file() else {}
    if worker_status.get("phase") != "terminal" and worker_status.get("pid"):
        pid = worker_status["pid"]
        command_line = Path(f"/proc/{pid}/cmdline")
        if command_line.is_file():
            command_bytes = command_line.read_bytes()
            if (
                str(root).encode() in command_bytes
                and b"profile-worker" in command_bytes
            ):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(worker_status["pgid"], signal.SIGKILL)
    marker_result = final_grid_marker(marker, worker_status.get("pid"))
    gate = None
    if (
        code == 0
        and worker_status.get("exit_status") == 0
        and marker_result["exact"]
        and (output / "report.json").is_file()
    ):
        gate = report_gate(root, output, root / "fixtures/public_p3_seed.json", "P3")
    state["profiles"].append(
        {
            "kind": "Nsight node window",
            "exit_status": code,
            "worker_status": worker_status,
            "final_grid_marker": marker_result,
            "parity": gate,
            "report_created": prefix.with_suffix(".nsys-rep").is_file(),
            "scope": "up to three seconds beginning after first real ns99 GPU-context creation; not whole evaluation timing or SM utilization",
        }
    )
    save(root / "pilot-status.json", state)
    if worker_status.get("phase") != "terminal":
        state["ncu_skipped"] = (
            "Nsight worker terminal proof absent; no overlapping counter job started"
        )
        save(root / "pilot-status.json", state)
        message = (
            "Nsight worker terminal proof absent; stop pilot workloads for cleanup"
        )
        raise RuntimeError(message)
    ncu = root / "work/profilers/ncu/target/linux-desktop-glibc_2_11_3-x64/ncu"
    query = launch(
        root,
        [str(ncu), "--query-metrics", "--query-metrics-mode", "all"],
        "ncu-query",
        expires_at,
        maximum=30,
        check=False,
    )
    available = (root / "logs/ncu-query.log").read_text()
    failure = ncu_query_failure(query, available)
    if failure is not None:
        state["ncu_skipped"] = failure
        state["ncu_counter_permission_denied"] = "ERR_NVGPUCTRPERM" in available
        save(root / "pilot-status.json", state)
        return
    wanted = (
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    )
    metrics = [name for name in wanted if name in available]
    l2_traffic = next(
        (
            name
            for name in ("lts__t_bytes.sum", "lts__t_sectors.sum")
            if name in available
        ),
        None,
    )
    if l2_traffic:
        metrics.append(l2_traffic)
    if "gpu__time_duration.sum" not in metrics:
        state["ncu_skipped"] = "required elapsed-time metric unavailable"
        return
    for name, ordered in (
        ("Residuals", True),
        ("ThomasRightHandSide", True),
        ("DeviceScanByKeyKernel", False),
    ):
        if expires_at - time.time() < RESERVE_SECONDS + 90:
            break
        env = environment(root)
        env.update(FLAGS)
        env.update(
            VMECPP_GPU_SOLVER_GRAPH="0",
            VMECPP_GPU_RESIDUAL_GRAPH="0",
            VMECPP_GPU_DIAGNOSTICS_GRAPH="0",
        )
        if ordered:
            env.update(
                VMECPP_GPU_RESIDUAL_ROUNDS="0", VMECPP_GPU_DIAGNOSTICS_ROUNDS="0"
            )
        report = root / "results/profiles" / ("ncu-" + name)
        command = [
            str(ncu),
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:.*" + name + ".*",
            "--launch-count",
            "1",
            "--cache-control",
            "none",
            "--metrics",
            ",".join(metrics),
            "--force-overwrite",
            "--export",
            str(report),
            str(root / "work/bin/solver-replay"),
            str(root / "captures"),
        ]
        code = launch(
            root, command, "ncu-" + name, expires_at, maximum=60, env=env, check=False
        )
        log_text = (root / "logs" / ("ncu-" + name + ".log")).read_text()
        created = report.with_suffix(".ncu-rep").is_file()
        counter_validation = {"exact": False, "missing_metrics": metrics}
        if code == 0 and created:
            exported = launch(
                root,
                [
                    str(ncu),
                    "--import",
                    str(report.with_suffix(".ncu-rep")),
                    "--csv",
                    "--page",
                    "raw",
                ],
                "ncu-values-" + name,
                expires_at,
                maximum=15,
                check=False,
            )
            if exported == 0:
                counter_validation = validate_ncu_values(
                    (root / "logs" / ("ncu-values-" + name + ".log")).read_text(),
                    metrics,
                )
        required_roofline = {
            "gpu__time_duration.sum",
            "dram__bytes.sum",
            l2_traffic,
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        }
        state["profiles"].append(
            {
                "kind": "Ncu captured-frame kernel",
                "kernel_filter": name,
                "exit_status": code,
                "report_created": created,
                "counter_capture_available": code == 0
                and created
                and "no kernels were profiled" not in log_text.lower()
                and "ERR_NVGPUCTRPERM" not in log_text
                and counter_validation["exact"],
                "metrics_requested": metrics,
                "metrics_unavailable_on_device": [
                    metric for metric in wanted if metric not in metrics
                ],
                "counter_values_validation": counter_validation,
                "roofline_inputs_present": ncu_roofline_inputs_present(
                    counter_validation, required_roofline
                )
                and "ERR_NVGPUCTRPERM" not in log_text,
                "counter_permission_error": "ERR_NVGPUCTRPERM" in log_text,
                "l2_traffic_counter": l2_traffic,
                "cache_control": "none; upstream captured-body kernels warm arrays, but profiler replay can still perturb cache state",
                "FP64_counter_scope": "executed thread instructions, separate from useful algorithm FLOPs; no whole-process inference",
                "scope": "ordered fallback/reference path"
                if ordered
                else "first optimistic diagnostic-refresh prefix scan; not residual-specific or whole-solver evidence",
            }
        )
        save(root / "pilot-status.json", state)


def validate_ncu_values(text: str, requested: list[str]) -> dict:
    """Read long metric rows or wide metric columns, retaining reported units."""
    values = {}
    units = {}
    invalid = set()
    header = None
    layout = None
    wide_units = {}
    for row in csv.reader(io.StringIO(text)):
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            layout = "long"
            continue
        if "ID" in row and any(name in row for name in requested):
            header = row
            layout = "wide"
            wide_units = {}
            continue
        if header is None:
            continue
        if len(row) != len(header):
            if (
                layout == "wide"
                and header.index("ID") < len(row)
                and row[header.index("ID")].strip().isdigit()
            ):
                invalid.update(requested)
            elif layout == "long" and header.index("Metric Name") < len(row):
                name = row[header.index("Metric Name")]
                if name in requested:
                    invalid.add(name)
            continue
        if layout == "wide" and not row[header.index("ID")].strip():
            wide_units = {
                name: row[header.index(name)].strip()
                for name in requested
                if name in header
            }
            continue
        if layout == "long":
            name = row[header.index("Metric Name")]
            entries = [
                (
                    name,
                    row[header.index("Metric Value")],
                    row[header.index("Metric Unit")].strip()
                    if "Metric Unit" in header
                    else "",
                )
            ]
        else:
            entries = [
                (name, row[header.index(name)], wide_units.get(name, ""))
                for name in requested
                if name in header
            ]
        for name, raw_value, unit in entries:
            if name not in requested:
                continue
            try:
                value = float(raw_value.replace(",", ""))
            except ValueError:
                invalid.add(name)
                continue
            if not math.isfinite(value):
                invalid.add(name)
                continue
            values.setdefault(name, []).append(value)
            units.setdefault(name, []).append(unit)
    missing = [name for name in requested if name not in values]
    missing_units = [
        name for name in requested if not units.get(name) or not all(units[name])
    ]
    permission_error = "ERR_NVGPUCTRPERM" in text
    return {
        "exact": bool(requested)
        and not missing
        and not invalid
        and not permission_error,
        "values": values,
        "units": units,
        "missing_metrics": missing,
        "invalid_metrics": sorted(invalid),
        "missing_unit_metrics": missing_units,
        "units_complete": not missing_units and bool(requested),
        "error": "ERR_NVGPUCTRPERM" if permission_error else None,
    }


def ncu_roofline_inputs_present(validation: dict, required: set) -> bool:
    """Require actual finite values and one explicit unit for each value."""
    return (
        bool(validation.get("exact"))
        and bool(required)
        and all(
            validation.get("values", {}).get(name)
            and len(validation.get("units", {}).get(name, []))
            == len(validation["values"][name])
            and all(validation["units"][name])
            for name in required
        )
    )


def ncu_query_failure(exit_status: int | str, text: str) -> str | None:
    if "ERR_NVGPUCTRPERM" in text:
        return "GPU performance counter permission denied (ERR_NVGPUCTRPERM)"
    if exit_status != 0:
        return "metric query failed; retain exact diagnostic log"
    return None


def profile_worker(root: Path, output: Path) -> int:
    import vmecpp
    from tools.optimization import benchmark

    runtime = ctypes.CDLL(str(root / "runtime/libcudart.so.12"))
    original = vmecpp.run
    calls = []
    save(
        root / "results/profiles/nsys-worker-status.json",
        {
            "phase": "running",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "started_at": time.time(),
        },
    )

    def ranged(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            calls.append({"stop_status": runtime.cudaProfilerStop()})

    vmecpp.run = ranged
    code = 1
    try:
        sys.argv = [
            "benchmark.py",
            "--stella",
            str(root / "stella-snapshot"),
            "--seed",
            str(root / "fixtures/public_p3_seed.json"),
            "--track",
            "P3",
            "--out",
            str(output),
        ]
        code = benchmark.main()
        return code
    finally:
        save(
            root / "results/profiles/nsys-worker-status.json",
            {
                "phase": "terminal",
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "exit_status": code,
                "profiler_api_calls": calls,
                "finished_at": time.time(),
            },
        )


def final_grid_marker(path: Path, pid: int | None) -> dict:
    if not path.is_file():
        return {"exact": False, "reason": "missing final-grid profiler marker"}
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
    except json.JSONDecodeError:
        return {"exact": False, "reason": "malformed final-grid profiler marker"}
    exact = (
        len(records) == 1
        and isinstance(records[0], dict)
        and records[0].get("event") == "profiler_start"
        and records[0].get("actual_ns") == records[0].get("target_ns") == 99
        and records[0].get("status") == 0
        and records[0].get("pid") == pid
    )
    return {
        "exact": exact,
        "records": records,
        "scope": "observer marker required; no entry-time profiler-start hook",
    }


def optional_tools(root: Path, expires_at: float, state: dict) -> None:
    """Separate native tooling diagnosis; adaptive code never enters a solve."""
    optional = root / "work/optional"
    if not optional.is_dir():
        state["optional_tools_skipped"] = "optional payload absent"
        return
    for record in json.loads((root / "optional-relocation.json").read_text()):
        if digest(inside(root, record["path"])) != record["after"]:
            message = "Copied optional tooling changed after bootstrap"
            raise RuntimeError(message)
    results = []
    env = environment(root)
    env["LD_LIBRARY_PATH"] = str(optional / "lib") + ":" + env["LD_LIBRARY_PATH"]
    sanitizer = root / "optional/compute-sanitizer/compute-sanitizer"
    launch(
        root,
        [str(sanitizer), "--version"],
        "optional-tool-version",
        expires_at,
        maximum=15,
        env=env,
        check=False,
    )
    commands = []
    for name in ("ordinary", "if-no-setter", "if-with-setter", "minimal"):
        program = str(optional / "bin" / name)
        commands.extend(
            (
                ("normal-" + name, [program]),
                (
                    "memcheck-" + name,
                    [
                        str(sanitizer),
                        "--tool",
                        "memcheck",
                        "--error-exitcode",
                        "90",
                        program,
                    ],
                ),
            )
        )
    capture = str(root / "optional/captures/real-iteration.bin")
    for frame in ("before", "after"):
        output = root / "results/optional" / ("adaptive-" + frame + ".json")
        commands.append(
            (
                "adaptive-normal-" + frame,
                [
                    str(optional / "bin/adaptive-residual-benchmark"),
                    capture,
                    frame,
                    str(output),
                    "1",
                ],
            )
        )
    for name, command in commands:
        if expires_at - time.time() < RESERVE_SECONDS + 45:
            break
        (root / "results/optional").mkdir(exist_ok=True)
        code = launch(
            root,
            command,
            "optional-" + name,
            expires_at,
            maximum=30,
            env=env,
            check=False,
        )
        results.append(
            {
                "name": name,
                "exit_status": code,
                "scope": "optional native-tool/component diagnosis only; no full-solver or adaptive qualification",
            }
        )
        state["optional_native_tools"] = results
        save(root / "pilot-status.json", state)


def collect(root: Path) -> dict:
    target = root / "collections" / ("pilot-results-" + str(time.time_ns()) + ".tar.gz")
    target.parent.mkdir(exist_ok=True)
    names = []
    for directory in (root / "logs", root / "results"):
        if directory.exists():
            names.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and not path.name.endswith(".tmp")
            )
    names.extend(
        path
        for path in (
            root / "pilot-status.json",
            root / "bootstrap-status.json",
            root / "relocation.json",
            root / "optional-relocation.json",
            root / "profile-observer-relocation.json",
            root / "supervisor-status.json",
            root / "controller-launch.json",
            root / "controller-supervisor.py",
        )
        if path.is_file()
    )
    hashes = {}
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(set(names)):
            data = path.read_bytes()
            name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(hashes, indent=2).encode()
        info = tarfile.TarInfo("retrieval-manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return {
        "archive": str(target),
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "files": len(hashes),
        "scope": "point-in-time incremental collection; incomplete/running stages are not passes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "bootstrap",
            "run",
            "exec",
            "profile-worker",
            "optional-tools",
            "collect",
            "verify-collection",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--lease-started-at", type=float)
    parser.add_argument("--lease-expires-at", type=float)
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--include-optional-tools", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--evaluation-out", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--destination", type=Path)
    args, remaining = parser.parse_known_args()
    if remaining and args.action != "exec":
        parser.error(
            "Unknown arguments are not accepted outside the wrapper exec action"
        )
    root = args.root.resolve()
    if args.action == "preflight":
        print(json.dumps(preflight(root), indent=2))
    elif args.action in ("bootstrap", "run", "optional-tools"):
        if not args.execute:
            print(json.dumps(planned_commands(root), indent=2))
            return 0
        if args.lease_started_at is None or args.lease_expires_at is None:
            parser.error("execution requires actual external one-hour lease timestamps")
        check_lease_window(args.lease_started_at, args.lease_expires_at)
        if args.action == "bootstrap":
            bootstrap(root, args.lease_expires_at)
        elif args.action == "run":
            run_pilot(
                root,
                args.lease_expires_at,
                args.max_pairs,
                include_optional=args.include_optional_tools,
            )
        else:
            state = json.loads((root / "pilot-status.json").read_text())
            if state.get("phase") != "complete" or not state.get(
                "canonical_parity_complete"
            ):
                parser.error("optional tools require completed canonical parity work")
            if args.lease_expires_at - time.time() < RESERVE_SECONDS + 420:
                parser.error(
                    "insufficient optional-tool budget after retrieval reserve"
                )
            optional_tools(root, args.lease_expires_at, state)
    elif args.action == "exec":
        if not args.variant:
            parser.error("exec requires variant")
        execute_wrapper(root, args.variant, remaining)
    elif args.action == "profile-worker":
        if not args.evaluation_out:
            parser.error("profile-worker requires evaluation-out")
        return profile_worker(root, args.evaluation_out)
    elif args.action == "collect":
        print(json.dumps(collect(root), indent=2))
    else:
        if not args.archive or not args.destination or not args.expected_sha256:
            parser.error(
                "verify-collection requires archive, destination and expected-sha256"
            )
        if digest(args.archive) != args.expected_sha256:
            parser.error("retrieved archive checksum mismatch")
        args.destination.mkdir(parents=True, exist_ok=False)
        safe_extract(args.archive, args.destination)
        manifest = json.loads(
            (args.destination / "retrieval-manifest.json").read_text()
        )
        for name, expected in manifest.items():
            if digest(inside(args.destination, name)) != expected:
                parser.error("retrieved file checksum mismatch")
        print(json.dumps({"verified_files": len(manifest), "exact": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
