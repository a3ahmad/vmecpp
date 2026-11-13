"""Small full-CGA research comparison using existing finite owner and fixtures."""

# Explicit research CLI prints, runtime-only imports and grouped guard predicates.
# ruff: noqa: T201, PLC0415, PT018

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = REPO / "build/optimization"
AUTHORITY = BASE / "gb202-packed-matrix1/matrix1.json"
AUTHORITY_SHA = "9b9aec1bbc3b6d2052fc970cd11def6fe14746cc0be9a181d66abbd6d4d8d552"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interrupted(*_):
    message = "Research owner interrupted"
    raise InterruptedError(message)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--shim", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default="1")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--driver-libs", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert digest(AUTHORITY) == AUTHORITY_SHA
    cfg = json.loads(AUTHORITY.read_text())
    indices = [int(x) for x in args.cases.split(",")]
    assert indices and len(indices) == len(set(indices))
    assert all(0 <= x < 6 for x in indices)
    backend, resident = (
        args.backend.resolve(strict=True),
        args.resident.resolve(strict=True),
    )
    extra = dict(value.split("=", 1) for value in args.env)
    assert all(key.startswith("VMECPP_") and key not in cfg["flags"] for key in extra)
    assert not any("CAPTURE" in key or "TRACE" in key for key in extra)
    if args.shim:
        extra.setdefault("VMECPP_EXPERIMENT_TRANSFORM_MODE", "fp64")
    if args.require_exact:
        assert extra.get("VMECPP_EXPERIMENT_TRANSFORM_MODE", "fp64") == "fp64"
    source_shim = args.shim.resolve(strict=True) if args.shim else Path(cfg["shim"])
    identity = {
        "backend": digest(backend),
        "resident": digest(resident),
        "shim_source": digest(source_shim),
    }
    if not args.require_exact:
        assert args.control, "Run the candidate default64 exact control first"
        control = json.loads(args.control.read_text())
        assert control["complete"] and control["exact_control"]
        assert control["candidate_identity"] == identity
        assert (
            control["candidate_flags"].get("VMECPP_EXPERIMENT_TRANSFORM_MODE", "fp64")
            == "fp64"
        )
        assert set(indices) <= set(control["case_indices"])
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": "PLAN",
                    "case_indices": indices,
                    "calls": 4 * len(indices),
                    "candidate_identity": identity,
                    "extra_candidate_flags": extra,
                    "GPU_calls": 0,
                }
            )
        )
        return
    assert args.cpu in os.sched_getaffinity(0)
    args.out.mkdir(parents=True, exist_ok=False)
    owner_path = BASE / "resident-solver-energy-ragged-clean1/compiler_owner.py"
    spec = importlib.util.spec_from_file_location("finite_owner", owner_path)
    assert spec is not None
    assert spec.loader is not None
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    for event in (signal.SIGTERM, signal.SIGINT):
        signal.signal(event, interrupted)
    sys.path.insert(0, str(REPO))
    from tools.optimization.summarize import metric_drift
    from tools.optimization.validate_resident_solver import compare_reports

    shim = args.out / "candidate-shim.so"
    shim.write_bytes(source_shim.read_bytes())
    patch = BASE / "a100-compat-payload-final1/bin/patchelf"
    needed = subprocess.check_output(
        [str(patch), "--print-needed", str(shim)], text=True
    ).splitlines()
    old = next(name for name in needed if name.endswith("solver-backend.so"))
    subprocess.run(
        [str(patch), "--replace-needed", old, str(backend), str(shim)], check=True
    )
    core = cfg["flags"]["VMECPP_FORWARD_REFERENCE_SO"]
    old_core = next(
        name
        for name in needed
        if name.endswith("_vmecpp.cpython-312-x86_64-linux-gnu.so")
    )
    if old_core != core:
        subprocess.run(
            [str(patch), "--replace-needed", old_core, core, str(shim)], check=True
        )
    final_needed = subprocess.check_output(
        [str(patch), "--print-needed", str(shim)], text=True
    ).splitlines()
    assert core in final_needed
    assert str(backend) in final_needed
    old_resident = str(BASE / "gb202-packed-sm1201/build2/resident.so")
    role_mapping = {
        cfg["backend"]: str(backend),
        old_resident: str(resident),
        cfg["shim"]: str(shim),
    }
    capture_source = (
        BASE / "resident-solver-energy-ragged-clean-equilibrium1/capture_worker.py"
    )
    worker_source = ROOT / "worker.py"
    rows = []
    status = {
        "complete": False,
        "scope": "ALTERNATIVE_RESEARCH_NOT_QUALIFIED",
        "case_indices": indices,
        "candidate_identity": identity,
        "candidate_flags": extra,
        "exact_control": False,
        "rows": rows,
    }
    status["private_shim_sha256"] = digest(shim)
    status["private_shim_needed"] = final_needed
    status["source_hashes"] = {
        str(path): digest(path)
        for path in (Path(__file__), worker_source, owner_path, AUTHORITY)
    }
    save(args.out / "status.json", status)
    deadline = time.monotonic() + 1800
    try:
        for index in indices:
            case = cfg["cases"][index]
            reference = json.loads(Path(case["original_report"]).read_text())
            for call, role in enumerate(
                ("baseline", "candidate", "candidate", "baseline")
            ):
                assert time.monotonic() + 310 < deadline
                name = f"case{index}-{call}-{role}"
                directory = args.out / name
                directory.mkdir()
                child = copy.deepcopy(cfg["worker_template"])
                child["files"] = dict(cfg["files"])
                child["case"] = case
                child["repo"] = str(REPO)
                child["capture_source"] = str(capture_source)
                child["run_root"] = str(directory / "worker")
                child["stats"] = str(directory / "selected.stats.json")
                child["base"]["cpu"] = args.cpu
                child["base"]["outputs"]["selected"] = str(directory / "sample")
                child["base"]["environments"]["selected"] = {
                    key: value
                    for key, value in cfg["flags"].items()
                    if key != "VMECPP_GPU_SOLVER_TRACE"
                }

                if role == "candidate":
                    assert digest(backend) == identity["backend"]
                    assert digest(resident) == identity["resident"]
                    assert digest(source_shim) == identity["shim_source"]
                    child["backend"] = str(backend)
                    child["base"]["preloads"]["selected"] = str(shim)
                    child["base"]["provider_witness_modules"]["selected"] = [
                        role_mapping.get(path, path)
                        for path in child["base"]["provider_witness_modules"][
                            "selected"
                        ]
                    ]
                    child["base"]["environments"]["selected"].update(extra)
                    if args.shim:
                        child["base"]["environments"]["selected"][
                            "VMECPP_EXPERIMENT_PRECISION_TRACE"
                        ] = str(directory / "precision.trace.jsonl")
                        child["base"]["environments"]["selected"][
                            "VMECPP_EXPERIMENT_RESIDENT_SO"
                        ] = str(resident)
                paths = child["base"]["provider_witness_modules"]["selected"]
                pins = [
                    *paths,
                    str(worker_source),
                    str(capture_source),
                    case["seed"],
                    case["original_report"],
                    case["canonical_report"],
                ]
                for path in pins:
                    child["files"][path] = digest(path)
                child["preflight_files"] = pins
                config_path = directory / "configuration.json"
                save(config_path, child)
                env = {
                    **cfg["base"]["base_environment"],
                    **child["base"]["environments"]["selected"],
                    "LD_PRELOAD": child["base"]["preloads"]["selected"],
                }
                if args.driver_libs:
                    env["LD_LIBRARY_PATH"] = (
                        str(args.driver_libs)
                        + ":/usr/lib/wsl/lib:"
                        + env["LD_LIBRARY_PATH"]
                    )
                env["MAPPED_PARENT_RECORD"] = json.dumps(
                    {
                        "PID": os.getpid(),
                        "UID": os.getuid(),
                        "birth": int(owner.birth(os.getpid())),
                    }
                )
                env["CGA_CHUNK_PILOT_NONCE"] = "mixed-research-" + name
                command = [
                    cfg["base"]["gpu_python"],
                    "-B",
                    str(worker_source),
                    "--config",
                    str(config_path),
                    "--sha",
                    digest(config_path),
                ]
                print(
                    "VMECPP_QPC_MARKER "
                    + json.dumps(
                        {
                            "phase": "batch_begin",
                            "group": name,
                            "arm": role,
                            "repetition": call,
                            "width": 1,
                        }
                    ),
                    flush=True,
                )
                process = owner.run_job(command, name, 300, env, directory)
                print(
                    "VMECPP_QPC_MARKER "
                    + json.dumps(
                        {
                            "phase": "batch_end",
                            "group": name,
                            "arm": role,
                            "repetition": call,
                            "width": 1,
                        }
                    ),
                    flush=True,
                )
                assert (
                    process["owned_children_reaped"] and not process["cleanup_actions"]
                )
                report_path = directory / "sample/report.json"
                report = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else None
                )
                comparison = compare_reports(reference, report) if report else None
                precision_path = directory / "precision.trace.jsonl"
                precision_rows = (
                    [
                        json.loads(line)
                        for line in precision_path.read_text().splitlines()
                    ]
                    if precision_path.exists()
                    else []
                )
                certificates = [
                    r
                    for r in precision_rows
                    if r.get("event") == "reference_fp64_materialization"
                ]
                final_certificate = (
                    max(
                        enumerate(certificates),
                        key=lambda item: (item[1]["ns"], item[0]),
                    )[1]
                    if certificates
                    else None
                )
                row = {
                    "case_index": index,
                    "role": role,
                    "call": call,
                    "process": process,
                    "report": str(report_path),
                    "comparison_original": comparison,
                    "metric_drift": metric_drift(
                        reference["metrics"], report.get("metrics") or {}
                    )
                    if report
                    else None,
                    "precision_trace": str(directory / "precision.trace.jsonl")
                    if role == "candidate" and args.shim
                    else None,
                    "FP64_terminal_certificate_checked": False,
                    "scope": "alternative",
                }
                rows.append(row)
                row["FP64_terminal_certificate_checked"] = final_certificate is not None
                row["FP64_terminal_certificate"] = final_certificate
                row["FP64_terminal_certificate_passed"] = bool(
                    final_certificate
                    and final_certificate.get("fp64_force_tolerance_passed") is True
                )
                row["precision_stage_summaries"] = [
                    r
                    for r in precision_rows
                    if r.get("event") in ("stage_end", "loop_end")
                ]
                save(args.out / "status.json", status)
                if role == "baseline" or args.require_exact:
                    assert (
                        report
                        and comparison is not None
                        and comparison["exact"]
                        and process["direct_exit_status"] == 0
                    ), "FP64 baseline/control drift; outputs preserved"
                    if role == "candidate" and args.shim:
                        assert row[
                            "FP64_terminal_certificate_passed"
                        ], "Missing/failed independent FP64 terminal residual certificate"
        status["complete"] = True
        status["exact_control"] = args.require_exact and all(
            row["comparison_original"]["exact"] for row in rows
        )
    finally:
        status["owned_descendants_remaining"] = owner.descendants()
        save(args.out / "status.json", status)


if __name__ == "__main__":
    main()
