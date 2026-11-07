"""Build an immutable GPU iteration-stage replay and original CPU capture shim."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--reference-extension", type=Path, required=True)
    parser.add_argument("--residual-tile", type=int, choices=(32, 128), default=128)
    parser.add_argument(
        "--rhs-mode-tile", type=int, choices=(1, 2, 4, 8, 16, 32), default=32
    )
    parser.add_argument(
        "--rhs-memory-policy", type=int, choices=(0, 1, 2, 3), default=0
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "source"
    snapshot.mkdir()
    for name in (
        "gpu_iteration.h",
        "gpu_iteration.cu",
        "gpu_optimistic_residual.cu",
        "gpu_diagnostics.h",
        "gpu_diagnostics.cu",
        "gpu_preconditioner.h",
        "gpu_preconditioner.cu",
        "preconditioner_capture.cc",
        "iteration_capture.cc",
        "iteration_replay.cc",
        "resident_replay_capture.h",
        "resident_force_eval_replay.cc",
        "gpu_fourier.h",
        "gpu_pipeline.h",
        "gpu_forward.h",
        "gpu_resident.h",
        "residual_benchmark.cc",
        "build_iteration.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    target = "CMakeFiles/vmecpp_core.dir/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc.o"
    result = subprocess.run(
        ["ninja", "-C", str(args.source / "build"), "-t", "commands", target],
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(
        next(x for x in result.stdout.splitlines() if "ideal_mhd_model.cc" in x)
    )
    includes = []
    iterator = iter(command)
    for argument in iterator:
        if argument in ("-I", "-isystem"):
            includes += [argument, next(iterator)]
        elif argument.startswith(("-I", "-D")):
            includes.append(argument)
    commands = [
        [
            str(args.cuda / "bin/nvcc"),
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-DVMECPP_RESIDUAL_TILE=" + str(args.residual_tile),
            "-DVMECPP_RHS_MODE_TILE=" + str(args.rhs_mode_tile),
            "-DVMECPP_RHS_MEMORY_POLICY=" + str(args.rhs_memory_policy),
            "-Xcompiler=-fPIC",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_120,code=sm_120",
            "-shared",
            str(snapshot / "gpu_iteration.cu"),
            str(snapshot / "gpu_diagnostics.cu"),
            str(snapshot / "gpu_optimistic_residual.cu"),
            str(snapshot / "gpu_preconditioner.cu"),
            "-o",
            str(args.out / "iteration.so"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-fopenmp",
            *includes,
            str(snapshot / "iteration_capture.cc"),
            str(snapshot / "preconditioner_capture.cc"),
            "-Wl,--no-as-needed",
            str(args.reference_extension),
            "-Wl,-rpath," + str(args.reference_extension.parent),
            "-ldl",
            "-o",
            str(args.out / "iteration-capture.so"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "iteration-replay"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_STEP_REPLAY",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "step-replay"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_DIAGNOSTICS_REPLAY",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "diagnostics-replay"),
        ],
    ]
    commands.append(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_PREPARATION_REPLAY",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "preparation-replay"),
        ]
    )
    commands.append(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_ITERATION_TAIL_REPLAY",
            "-I" + str(snapshot),
            "-I" + str(args.cuda / "include"),
            str(snapshot / "resident_force_eval_replay.cc"),
            "-L" + str(args.cuda / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(args.cuda / "lib64"),
            "-ldl",
            "-o",
            str(args.out / "resident-iteration-replay"),
        ]
    )
    commands.append(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-I" + str(snapshot),
            "-I" + str(args.cuda / "include"),
            str(snapshot / "residual_benchmark.cc"),
            "-L" + str(args.cuda / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(args.cuda / "lib64"),
            "-ldl",
            "-o",
            str(args.out / "residual-benchmark"),
        ]
    )
    commands.append(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_PRECONDITIONER_REPLAY",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "preconditioner-replay"),
        ]
    )
    commands.append(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_GEOMETRY_REPLAY",
            str(snapshot / "iteration_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "geometry-preparation-replay"),
        ]
    )
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot.iterdir()
    }
    (args.out / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for index, command in enumerate(commands):
        with (args.out / f"build-{index}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
