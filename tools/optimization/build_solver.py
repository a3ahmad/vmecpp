"""Build an immutable persistent GPU solver experiment and CPU control shim."""

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
    for name in (
        "source",
        "out",
        "cuda",
        "reference-extension",
        "resident",
        "iteration",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shim-source", type=Path)
    parser.add_argument("--cache-factors", action="store_true")
    parser.add_argument("--thomas-prefix", action="store_true")
    parser.add_argument(
        "--thomas-prefetch-forward", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument(
        "--thomas-prefetch-backward", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--cooperative-residuals", action="store_true")
    parser.add_argument("--lattice-residuals", action="store_true")
    parser.add_argument("--warp-lattice-seeds", action="store_true")
    parser.add_argument("--fuse-lattice-finish", action="store_true")
    parser.add_argument("--coordinate-captures", action="store_true")
    parser.add_argument("--reuse-device-buffers", action="store_true")
    parser.add_argument(
        "--live-damping-controller", "--live-controller", action="store_true"
    )
    parser.add_argument("--pinned-cpu-log", action="store_true")
    parser.add_argument("--evolve-damping-bypass", action="store_true")
    parser.add_argument("--fuse-damping-tail", action="store_true")
    parser.add_argument("--cooperative-energies", action="store_true")
    parser.add_argument("--lattice-energies", action="store_true")
    args = parser.parse_args()
    if (
        args.thomas_prefetch_forward != 1 or args.thomas_prefetch_backward != 1
    ) and not args.thomas_prefix:
        parser.error("Thomas prefetch options require --thomas-prefix")
    if args.pinned_cpu_log and not args.live_damping_controller:
        parser.error("Pinned CPU log requires --live-controller")
    if args.evolve_damping_bypass and not args.pinned_cpu_log:
        parser.error("Evolve damping bypass requires --pinned-cpu-log")
    if args.fuse_damping_tail and not args.evolve_damping_bypass:
        parser.error("Fused damping tail requires --evolve-damping-bypass")
    if args.warp_lattice_seeds and not args.lattice_residuals:
        parser.error("Warp lattice seeds require --lattice-residuals")
    if args.thomas_prefix and not args.cache_factors:
        parser.error("Thomas prefix requires --cache-factors")
    if args.thomas_prefix and args.reuse_device_buffers:
        parser.error("Thomas prefix buffer pooling is not supported")
    if args.lattice_residuals and args.cooperative_residuals:
        parser.error("Choose one residual implementation")
    if args.fuse_lattice_finish and not args.lattice_residuals:
        parser.error("Lattice finish fusion requires --lattice-residuals")
    if args.lattice_energies and not args.lattice_residuals:
        parser.error("Lattice energies require --lattice-residuals")
    if args.lattice_energies and args.cooperative_energies:
        parser.error("Choose one ordinary energy implementation")
    if args.cache_factors:
        symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(args.iteration)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        exported = {line.split()[-1] for line in symbols.splitlines() if line.split()}
        required = {"LaunchGpuIterationFactor", "LaunchGpuIterationRZFactored"}
        if not required.issubset(exported):
            message = "Factor caching requires the factor/RHS iteration APIs"
            raise ValueError(message)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "source"
    snapshot.mkdir()
    names = (
        "gpu_solver.h",
        "gpu_solver.cu",
        "resident_solver_shim.cc",
        "gpu_solver_replay.cc",
        "resident_replay_capture.h",
        "build_solver.py",
        "gpu_fourier.h",
        "gpu_resident.h",
        "gpu_iteration.h",
        "gpu_preconditioner.h",
        "gpu_diagnostics.h",
        "gpu_forward.h",
        "gpu_pipeline.h",
        "gpu_optimistic_diagnostics.h",
        "gpu_optimistic_diagnostics.cu",
    )
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.reuse_device_buffers:
        shutil.copy2(
            Path(__file__).with_name("gpu_solver_buffer_pool.h"),
            snapshot / "gpu_solver_buffer_pool.h",
        )
    if args.pinned_cpu_log:
        for name in (
            "gpu_pinned_cpu_log.h",
            "pinned_cpu_log_guard.h",
            "gpu_pinned_cpu_log.COPYING.LIB",
            "gpu_pinned_cpu_log_provenance.json",
        ):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.cooperative_residuals:
        for name in ("gpu_cooperative_residual.h", "gpu_cooperative_residual.cu"):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.thomas_prefix:
        for name in ("gpu_thomas_prefix.h", "gpu_thomas_prefix.cu"):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.lattice_residuals:
        for name in (
            "gpu_lattice_residual.h",
            "gpu_lattice_residual.cu",
            "gpu_lattice_prefix.h",
            "gpu_lattice_prefix.cu",
        ):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.cooperative_energies:
        for name in ("gpu_cooperative_energies.h", "gpu_cooperative_energies.cu"):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.lattice_energies:
        for name in ("gpu_lattice_energies.h", "gpu_lattice_energies.cu"):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if args.shim_source:
        shutil.copy2(args.shim_source, snapshot / "resident_solver_shim.cc")
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
    backend = args.out / "solver-backend.so"
    libraries = [str(args.resident), str(args.iteration)]
    runtime_paths = [
        args.resident.parent,
        args.iteration.parent,
        args.cuda / "lib64",
        args.out,
    ]
    rpaths = ["-Wl,-rpath," + str(path) for path in runtime_paths]
    commands = [
        [
            str(args.cuda / "bin/nvcc"),
            "-std=c++20",
            "-O3",
            "--fmad=false",
            *(["-DVMECPP_CACHE_TRIDIAGONAL_FACTORS"] if args.cache_factors else []),
            *(["-DVMECPP_THOMAS_PREFIX"] if args.thomas_prefix else []),
            *(
                [
                    "-DVMECPP_THOMAS_FORWARD_PREFETCH="
                    + str(args.thomas_prefetch_forward),
                    "-DVMECPP_THOMAS_BACKWARD_PREFETCH="
                    + str(args.thomas_prefetch_backward),
                ]
                if args.thomas_prefix
                else []
            ),
            *(["-DVMECPP_COOPERATIVE_RESIDUALS"] if args.cooperative_residuals else []),
            *(["-DVMECPP_LATTICE_RESIDUALS"] if args.lattice_residuals else []),
            *(["-DVMECPP_LATTICE_WARP_SEEDS"] if args.warp_lattice_seeds else []),
            *(["-DVMECPP_LATTICE_FUSE_FINISH"] if args.fuse_lattice_finish else []),
            *(["-DVMECPP_COORDINATE_CAPTURES"] if args.coordinate_captures else []),
            *(["-DVMECPP_REUSE_DEVICE_BUFFERS"] if args.reuse_device_buffers else []),
            *(["-DVMECPP_PINNED_CPU_LOG"] if args.pinned_cpu_log else []),
            *(["-DVMECPP_EVOLVE_DAMPING_BYPASS"] if args.evolve_damping_bypass else []),
            *(["-DVMECPP_FUSE_DAMPING_TAIL"] if args.fuse_damping_tail else []),
            *(
                ["-DVMECPP_LIVE_DAMPING_CONTROLLER"]
                if args.live_damping_controller
                else []
            ),
            *(["-DVMECPP_COOPERATIVE_ENERGIES"] if args.cooperative_energies else []),
            *(["-DVMECPP_LATTICE_ENERGIES"] if args.lattice_energies else []),
            "-Xcompiler=-fPIC",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_120,code=sm_120",
            "-shared",
            str(snapshot / "gpu_solver.cu"),
            str(snapshot / "gpu_optimistic_diagnostics.cu"),
            *([str(snapshot / "gpu_thomas_prefix.cu")] if args.thomas_prefix else []),
            *(
                [str(snapshot / "gpu_cooperative_residual.cu")]
                if args.cooperative_residuals
                else []
            ),
            *(
                [
                    str(snapshot / "gpu_lattice_prefix.cu"),
                    str(snapshot / "gpu_lattice_residual.cu"),
                ]
                if args.lattice_residuals
                else []
            ),
            *(
                [str(snapshot / "gpu_cooperative_energies.cu")]
                if args.cooperative_energies
                else []
            ),
            *(
                [str(snapshot / "gpu_lattice_energies.cu")]
                if args.lattice_energies
                else []
            ),
            *libraries,
            *(["-ldl"] if args.reuse_device_buffers else []),
            *("-Xlinker=-rpath," + str(path) for path in runtime_paths),
            "-o",
            str(backend),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-fopenmp",
            *(["-DVMECPP_PINNED_CPU_LOG"] if args.pinned_cpu_log else []),
            *(["-DVMECPP_EVOLVE_DAMPING_BYPASS"] if args.evolve_damping_bypass else []),
            *(["-DVMECPP_FUSE_DAMPING_TAIL"] if args.fuse_damping_tail else []),
            *(
                ["-DVMECPP_LIVE_DAMPING_CONTROLLER"]
                if args.live_damping_controller
                else []
            ),
            *includes,
            str(snapshot / "resident_solver_shim.cc"),
            "-Wl,--no-as-needed",
            str(args.reference_extension),
            str(backend),
            "-Wl,-rpath," + str(args.reference_extension.parent),
            *rpaths,
            "-ldl",
            "-o",
            str(args.out / "solver-shim.so"),
        ],
        [
            "g++",
            "-std=c++20",
            "-O3",
            str(snapshot / "gpu_solver_replay.cc"),
            str(backend),
            *rpaths,
            "-o",
            str(args.out / "solver-replay"),
        ],
    ]
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot.iterdir()
    }
    hashes.update(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.resident, args.iteration, args.reference_extension)
        }
    )
    (args.out / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for index, command in enumerate(commands):
        with (args.out / f"build-{index}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
