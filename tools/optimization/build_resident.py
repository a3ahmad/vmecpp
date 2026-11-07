"""Build an immutable resident geometry/field prototype and CPU capture shim."""

from __future__ import annotations

# CLI progress is intentional.
# ruff: noqa: T201
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--inverse-kernel", type=Path)
    parser.add_argument("--profile-rows", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--profile-split-chains", action="store_true")
    args = parser.parse_args()
    if args.profile_split_chains and args.profile_rows not in (1, 2, 4, 8):
        parser.error("--profile-split-chains requires --profile-rows 1,2,4,8")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "source"
    snapshot.mkdir()
    for name in (
        "gpu_resident.cu",
        "gpu_resident.h",
        "resident_capture.cc",
        "gpu_replay.cc",
        "generate_geometry_point.py",
        "generate_force_point.py",
        "build_resident.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.inverse_kernel:
        for name in (
            "gpu_fourier.h",
            "gpu_fourier.cu",
            "resident_chain_replay.cc",
            "resident_mhd_replay.cc",
            "resident_replay_capture.h",
            "gpu_pipeline.h",
            "gpu_forward.h",
            "gpu_forward.cu",
            "gpu_constraint.cu",
            "resident_force_eval_replay.cc",
            "gpu_iteration.h",
            "gpu_preconditioner.h",
            "gpu_diagnostics.h",
        ):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
        shutil.copy2(args.inverse_kernel, snapshot / "generated_fourier_kernel.cuh")
    subprocess.run(
        [
            sys.executable,
            str(snapshot / "generate_geometry_point.py"),
            "--source",
            str(args.source),
            "--out",
            str(snapshot / "geometry_point.h"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(snapshot / "generate_force_point.py"),
            "--source",
            str(args.source),
            "--out",
            str(args.out / "force-generated"),
        ],
        check=True,
    )
    shutil.copy2(args.out / "force-generated/mhd_point.h", snapshot / "mhd_point.h")
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
            "-Xcompiler=-fPIC",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_120,code=sm_120",
            "-I" + str(snapshot),
            "-c",
            str(snapshot / "gpu_resident.cu"),
            "-o",
            str(args.out / "resident.o"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-fopenmp",
            "-I" + str(snapshot),
            *includes,
            str(snapshot / "resident_capture.cc"),
            "-ldl",
            "-o",
            str(args.out / "resident-capture.so"),
        ],
        [
            "g++",
            "-shared",
            str(args.out / "resident.o"),
            "-L" + str(args.cuda / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(args.cuda / "lib64"),
            "-o",
            str(args.out / "resident.so"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_REPLAY_RESIDENT",
            "-I" + str(snapshot),
            str(snapshot / "gpu_replay.cc"),
            "-ldl",
            "-o",
            str(args.out / "resident-replay"),
        ],
    ]
    if args.profile_rows is not None:
        commands[0].insert(1, f"-DVMECPP_PROFILE_TILE_ROWS={args.profile_rows}")
    if args.profile_split_chains:
        commands[0].insert(1, "-DVMECPP_PROFILE_SPLIT_CHAINS")
    if args.inverse_kernel:
        commands[0].insert(1, "-DVMECPP_GPU_RESIDENT_CHAIN")
        commands[2].insert(3, str(args.out / "fourier.o"))
        commands[2].insert(4, str(args.out / "forward.o"))
        commands[2].insert(5, str(args.out / "constraint.o"))
        inverse_command = [
            str(args.cuda / "bin/nvcc"),
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-Xcompiler=-fPIC",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_120,code=sm_120",
            "-DVMECPP_GPU_FLAT_GRID",
            "-DVMECPP_GPU_BLOCK_SIZE=32",
            "-DVMECPP_GPU_SEPARABLE",
            "-DFourierToRealKernel=FourierToRealKernelSeparable",
            "-I" + str(snapshot),
            "-c",
            str(snapshot / "gpu_fourier.cu"),
            "-o",
            str(args.out / "fourier.o"),
        ]
        extra_commands = []
        for source_name, object_name in (
            ("gpu_forward.cu", "forward.o"),
            ("gpu_constraint.cu", "constraint.o"),
        ):
            extra_commands.append(
                [
                    str(args.cuda / "bin/nvcc"),
                    "-std=c++20",
                    "-O3",
                    "--fmad=false",
                    "-Xcompiler=-fPIC",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_120,code=sm_120",
                    "-I" + str(snapshot),
                    "-c",
                    str(snapshot / source_name),
                    "-o",
                    str(args.out / object_name),
                ]
            )
        commands[1:1] = [inverse_command, *extra_commands]
        commands.append(
            [
                "g++",
                "-std=c++20",
                "-O3",
                "-I" + str(snapshot),
                str(snapshot / "resident_chain_replay.cc"),
                "-ldl",
                "-o",
                str(args.out / "resident-chain-replay"),
            ]
        )
        commands.append(
            [
                "g++",
                "-std=c++20",
                "-O3",
                "-I" + str(snapshot),
                str(snapshot / "resident_mhd_replay.cc"),
                "-ldl",
                "-o",
                str(args.out / "resident-mhd-replay"),
            ]
        )
        commands.append(
            [
                "g++",
                "-std=c++20",
                "-O3",
                "-I" + str(snapshot),
                str(snapshot / "resident_force_eval_replay.cc"),
                "-ldl",
                "-o",
                str(args.out / "resident-force-eval-replay"),
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
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2))
    for i, command in enumerate(commands):
        with (args.out / f"build-{i}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Completed resident build step", i, flush=True)
    hashes = {
        str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.rglob("*")
        if p.is_file()
    }
    (args.out / "hashes.json").write_text(json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
