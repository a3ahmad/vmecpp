"""Compile the root-owned experimental shim with original pinned include flags."""

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from tools.optimization.gb202 import compiler_owner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT.parent / "gb202/sources.json").read_text())
    for name, source in manifest["runtime_shim"]["files"].items():
        if name != "resident_solver_shim.cc":
            (out / name).write_bytes((ROOT.parent / "gb202" / source).read_bytes())
    prior = json.loads(
        (
            REPO / "build/optimization/resident-solver-seed-forward1/commands.json"
        ).read_text()
    )[1]
    command = []
    for item in prior:
        translated = item
        if item.endswith("/source/resident_solver_shim.cc"):
            translated = str(ROOT / "resident_solver_shim.cc")
        elif item.endswith("resident-solver-seed-forward1/solver-backend.so"):
            translated = str(args.backend.resolve())
        elif item.endswith("resident-solver-seed-forward1/solver-shim.so"):
            translated = str(out / "solver-shim.so")
        command.append(translated)
    command[1:1] = ["-I" + str(out), "-I" + str(ROOT)]
    command.extend(
        [
            "-Wl,--no-as-needed",
            str(args.backend.resolve().parent / "resident.so"),
            "-Wl,-rpath," + str(args.backend.resolve().parent),
        ]
    )
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
    }
    job = compiler_owner.run_job(command, "shim-compile", 180, environment, out)
    paths = [ROOT / "resident_solver_shim.cc", ROOT / "precision_trial_hooks.inc"]
    output = out / "solver-shim.so"
    receipt = {
        "job": job,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "CUDA_executions": 0,
        "flag_origin": "resident-solver-seed-forward1/commands.json[1]",
    }
    if output.exists():
        receipt["shim_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if job.get("direct_exit_status") == 0 else 1)


if __name__ == "__main__":
    main()
