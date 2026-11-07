"""Plan or CPU-build the retained GB202 numerical source cohort.

This does not build/install VMEC++, load a DSO, or execute CUDA workloads.
"""

# ruff: noqa: T201 - This command intentionally prints its plan and build receipt.

import argparse
import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path

import compiler_owner

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cuda-lib", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.out.is_absolute(), "Use an absolute fresh output directory"
    out = args.out.resolve()
    assert not out.exists(), "Output must be fresh"
    manifest = json.loads((ROOT / "sources.json").read_text())
    for name, value in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        assert data.isascii(), name
        assert sha(ROOT / name) == value["sha256"], name
    nvcc = shutil.which(args.nvcc) or args.nvcc
    cxx = shutil.which(args.cxx) or args.cxx
    cuda_lib = args.cuda_lib or Path(nvcc).resolve().parent.parent / "lib64"
    commands = []
    objects = {}
    for unit in manifest["translation_units"]:
        stage = out / "units" / unit["name"]
        source = next(
            name for name, value in unit["files"].items() if value == unit["source"]
        )
        target = out / "objects" / (unit["name"] + ".o")
        objects[unit["name"]] = str(target)
        commands.append(
            (
                unit["name"],
                [
                    nvcc,
                    *unit["flags"],
                    "-I" + str(stage),
                    "-c",
                    str(stage / source),
                    "-o",
                    str(target),
                ],
            )
        )
    base = ["-shared", "-Wl,--no-undefined"]
    runtime = [
        "-L" + str(cuda_lib),
        "-lcudart",
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath," + str(cuda_lib),
    ]

    def link(name, names, deps=(), extra=()):
        commands.append(
            (
                name + "-link",
                [
                    cxx,
                    *base,
                    *(objects[x] for x in names),
                    "-Wl,--no-as-needed",
                    *(str(out / x) for x in deps),
                    "-Wl,--as-needed",
                    *extra,
                    *runtime,
                    "-o",
                    str(out / (name + ".so")),
                ],
            )
        )

    link(
        "iteration",
        [
            "iteration_" + x
            for x in (
                "gpu_iteration",
                "gpu_diagnostics",
                "gpu_optimistic_residual",
                "gpu_preconditioner",
            )
        ],
    )
    link("rhs", ["rhs"], ["iteration.so"], ["-Wl,-Bsymbolic"])
    link("resident", ["resident", "fourier", "forward", "constraint"])
    link(
        "solver-backend",
        [
            "gpu_solver",
            "gpu_optimistic_diagnostics",
            "gpu_thomas_prefix",
            "gpu_lattice_prefix",
            "gpu_lattice_residual",
            "gpu_lattice_energies",
            "ragged_coordinate",
            "coordinate",
        ],
        ["resident.so", "rhs.so", "iteration.so"],
    )
    plan = {
        "scope": "CPU source rebuild; new-location binaries not yet qualified",
        "source_manifest_sha256": sha(ROOT / "sources.json"),
        "CUDA_executions": 0,
        "commands": commands,
        "original_ancestry": manifest["ancestry"],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    out.mkdir(parents=True)
    (out / "objects").mkdir()
    (out / "logs").mkdir()
    for unit in manifest["translation_units"]:
        stage = out / "units" / unit["name"]
        stage.mkdir(parents=True)
        for name, original in unit["files"].items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / original, target)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = []
    for label, command in commands:
        result = compiler_owner.run_job(
            command, label, 180, dict(os.environ), out / "logs"
        )
        jobs.append(result)
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        assert result["direct_exit_status"] == 0
        assert result["owned_children_reaped"]
        assert not result["cleanup_actions"]
        assert not result.get("error")
    outputs = {p.name: sha(p) for p in out.glob("*.so")}
    (out / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
