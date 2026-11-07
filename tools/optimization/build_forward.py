"""Build an isolated GPU forward-transform interposition experiment."""

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
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--include-manifest", type=Path)
    parser.add_argument("--force-point-header", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original_tools = Path(__file__).resolve().parent
    tools = args.out / "source"
    tools.mkdir()
    source_names = [
        "gpu_forward.cu",
        "gpu_forward.h",
        "forward_shim.cc",
        "gpu_replay.cc",
        "build_forward.py",
    ]
    if args.pipeline:
        source_names += [
            "gpu_pipeline.h",
            "generate_force_point.py",
            "radial_executor.h",
        ]
    for name in source_names:
        shutil.copy2(original_tools / name, tools / name)
    if args.pipeline:
        if args.force_point_header:
            shutil.copy2(args.force_point_header, tools / "mhd_point.h")
        else:
            generation = [
                sys.executable,
                str(tools / "generate_force_point.py"),
                "--source",
                str(args.source),
                "--out",
                str(args.out / "generated"),
            ]
            (args.out / "generator-command.json").write_text(
                json.dumps(generation, indent=2)
            )
            subprocess.run(generation, check=True)
            shutil.copy2(args.out / "generated/mhd_point.h", tools / "mhd_point.h")
    build = args.source / "build"
    target = (
        "CMakeFiles/vmecpp_core.dir/src/vmecpp/cpp/vmecpp/vmec/"
        "ideal_mhd_model/ideal_mhd_model.cc.o"
    )
    if args.include_manifest:
        manifest = json.loads(args.include_manifest.read_text())
        includes = manifest["definitions"] + [
            "-I" + str((args.include_manifest.parent / path).resolve())
            for path in manifest["include_roots"]
        ]
    else:
        result = subprocess.run(
            ["ninja", "-C", str(build), "-t", "commands", target],
            capture_output=True,
            text=True,
            check=True,
        )
        command = shlex.split(
            next(
                line
                for line in result.stdout.splitlines()
                if "ideal_mhd_model.cc" in line
            )
        )
        includes = []
        iterator = iter(command)
        for argument in iterator:
            if argument in ("-I", "-isystem"):
                includes += [argument, next(iterator)]
            elif argument.startswith(("-I", "-D")):
                includes.append(argument)
    host_object = args.out / "forward-shim.o"
    gpu_object = args.out / "gpu-forward.o"
    library = args.out / "forward-shim.so"
    replay = args.out / "forward-replay"
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
            "-I" + str(tools),
            "-c",
            str(tools / "gpu_forward.cu"),
            "-o",
            str(gpu_object),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-ffp-contract=off",
            "-fopenmp",
            "-I" + str(tools),
            *includes,
            "-c",
            str(tools / "forward_shim.cc"),
            "-o",
            str(host_object),
        ],
        [
            "g++",
            "-shared",
            "-fopenmp",
            str(host_object),
            str(gpu_object),
            "-L" + str(args.cuda / "lib64"),
            "-lcudart",
            "-ldl",
            "-Wl,-rpath," + str(args.cuda / "lib64"),
            "-o",
            str(library),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DVMECPP_REPLAY_FORWARD",
            "-I" + str(tools),
            str(tools / "gpu_replay.cc"),
            "-ldl",
            "-o",
            str(replay),
        ],
    ]
    if args.pipeline:
        commands[0].insert(1, "-DVMECPP_GPU_PIPELINE")
        commands[1].insert(1, "-DVMECPP_GPU_PIPELINE")
        for mode in ("PIPELINE", "MHD"):
            commands.append(
                [
                    value.replace(
                        "-DVMECPP_REPLAY_FORWARD", "-DVMECPP_REPLAY_" + mode
                    ).replace(str(replay), str(args.out / (mode.lower() + "-replay")))
                    for value in commands[3]
                ]
            )
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        with (args.out / f"build-{index}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Completed build step", index, flush=True)
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [library, replay, *tools.iterdir()]
        if path.is_file()
    }
    (args.out / "hashes.json").write_text(json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
