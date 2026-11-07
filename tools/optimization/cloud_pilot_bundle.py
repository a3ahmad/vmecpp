"""Assemble a frozen local pilot payload.

This never rents or contacts a host.
"""

from __future__ import annotations

# Assembly progress is intentional.
# ruff: noqa: T201
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assemble(workspace: Path, stella: Path, out: Path, *, optional: bool) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    optimization = workspace / "build/optimization"
    snapshot = optimization / "resident-solver8-capture"
    source_hashes = json.loads((snapshot / "source-hashes.json").read_text())
    for name, expected in source_hashes.items():
        source = Path(name) if Path(name).is_absolute() else snapshot / "source" / name
        if digest(source) != expected:
            message = "Frozen solver8 input changed: " + name
            raise ValueError(message)
    smoke = json.loads((snapshot / "smoke/validation.json").read_text())
    if (
        not smoke["exact"]
        or digest(snapshot / "solver-shim.so")
        != smoke["preload_sha256"][str(snapshot / "solver-shim.so")]
    ):
        message = "Solver8 shim no longer matches its executed exact smoke"
        raise ValueError(message)

    def copy(source: Path, relative: str) -> None:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def tree(source: Path, relative: str) -> None:
        for path in sorted(source.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                copy(path, str(Path(relative) / path.relative_to(source)))

    for name in ("oracle-packages.tar.gz", "nsys-cli.tar.gz", "ncu-cli.tar.gz"):
        copy(optimization / name, "archives/" + name)
    libs = {
        "solver-shim.so": snapshot / "solver-shim.so",
        "solver-backend.so": snapshot / "solver-backend.so",
        "resident.so": optimization / "resident-device-inverse/resident.so",
        "iteration.so": optimization / "resident-iteration17/iteration.so",
    }
    for name, source in libs.items():
        copy(source, "lib/" + name)
    copy(snapshot / "solver-replay", "bin/solver-replay")
    tree(snapshot / "source", "source/solver8")
    tree(optimization / "resident-device-inverse/source", "source/resident")
    tree(
        optimization / "resident-device-inverse/force-generated",
        "source/resident-force-generated",
    )
    tree(optimization / "resident-iteration17/source", "source/iteration")
    for name in ("commands.json", "source-hashes.json"):
        copy(snapshot / name, "provenance/solver8/" + name)
    for directory in ("resident-device-inverse", "resident-iteration17"):
        for name in ("commands.json", "hashes.json", "source-hashes.json"):
            source = optimization / directory / name
            if source.is_file():
                copy(source, f"provenance/{directory}/{name}")
    package = optimization / "gpu-env/lib/python3.12/site-packages/vmecpp"
    tree(package, "python/vmecpp")
    for path in sorted((stella / "src/cga").rglob("*")):
        if (
            path.is_file()
            and path.suffix in (".py", ".typed")
            and "__pycache__" not in path.parts
        ):
            copy(path, "stella-snapshot/" + str(path.relative_to(stella)))
    copy(stella / "configs/oracle.yaml", "stella-snapshot/configs/oracle.yaml")
    tools = (
        "cloud_pilot.py",
        "cloud_pilot_bundle.py",
        "benchmark.py",
        "campaign_gates.py",
        "paired.py",
        "report_pairs.py",
        "summarize.py",
        "validate_resident_solver.py",
        "validate_resident_lifecycle.py",
        "analyze_solver_timeline.py",
    )
    for name in tools:
        copy(workspace / "tools/optimization" / name, "tools/optimization/" + name)
    observer = optimization / "final-grid-profile-observer"
    observer_hashes = json.loads((observer / "hashes.json").read_text())
    for name, expected in observer_hashes.items():
        if digest(Path(name)) != expected:
            message = "Final-grid profiling observer changed"
            raise ValueError(message)
    copy(
        observer / "solver-stage-profile-shim.so",
        "profiling/solver-stage-profile-shim.so",
    )
    copy(
        workspace / "tools/optimization/solver_stage_profile_shim.cc",
        "profiling/source/solver_stage_profile_shim.cc",
    )
    copy(workspace / "tools/optimization/gpu_solver.h", "profiling/source/gpu_solver.h")
    for name in ("command.json", "hashes.json"):
        copy(observer / name, "profiling/provenance/" + name)
    fixtures = json.loads((optimization / "fixtures-frozen/manifest.json").read_text())
    copy(optimization / "fixtures-frozen/manifest.json", "fixtures/manifest.json")
    for fixture in fixtures:
        seed = Path(fixture["path"])
        copy(seed, "fixtures/" + seed.name)
        for track in ("P1", "P3"):
            reference = (
                optimization
                / "a100-fused-results/paired-combined/baseline"
                / f"{seed.stem}-{track}-00"
            )
            for path in sorted(reference.glob("*.json")):
                if path.name in (
                    "report.json",
                    "provenance.json",
                ) or path.name.startswith("actual_indata_"):
                    copy(path, f"references/{seed.stem}-{track}/{path.name}")
    extra = optimization / "validation-agent/extra-mpol12-public-p1"
    if extra.is_dir():
        copy(extra / "public_p1_seed.json", "extra/fixtures/public_p1_seed.json")
        copy(extra / "fixture.json", "extra/source-provenance.json")
        for name in ("report.json", "provenance.json", "actual_indata_0.json"):
            copy(
                extra / "baseline-0" / name,
                "extra/references/public_p1_seed-P1/" + name,
            )
    captures = (
        "real-transform.bin",
        "resident-fields/real-fields.bin",
        "resident-constants/real-fields.bin.constants.bin",
        "resident-constraints-capture/real-fields.bin.constraints.bin",
        "a100-fused-results/fused/real-pipeline.bin",
        "resident-iteration10/real-preparation.bin",
        "resident-iteration10/real-iteration.bin",
        "resident-iteration10/real-step.bin",
        "resident-iteration10/real-diagnostics.bin",
        "resident-preconditioner-capture-retry/real-preconditioner.bin",
    )
    for name in captures:
        copy(optimization / name, "captures/" + name)
    calibration = optimization / "roofline-strict-calibration"
    expected = json.loads((calibration / "hashes.json").read_text())
    for name in (
        "tools/optimization/roofline_strict.cu",
        "build/optimization/roofline-strict-calibration/roofline-strict",
    ):
        if digest(workspace / name) != expected[name]:
            message = "Strict calibration source or binary changed"
            raise ValueError(message)
    copy(calibration / "roofline-strict", "bin/roofline-strict")
    copy(
        workspace / "tools/optimization/roofline_strict.cu",
        "source/calibration/roofline_strict.cu",
    )
    for name in ("build-command.json", "hashes.json", "sass-opcode-summary.json"):
        copy(calibration / name, "provenance/calibration/" + name)
    packaging = optimization / "validation-agent/pilot-packaging-tools"
    copy(packaging / "patchelf", "bin/patchelf")
    copy(packaging / "provenance.json", "provenance/patchelf.json")
    copy(
        Path("/usr/local/cuda-12.9/lib64/libcudart.so.12").resolve(),
        "runtime/libcudart.so.12",
    )
    if optional:
        prepare_optional(workspace, out, copy, tree)
    plan = {
        "purpose": "one-hour frugal A100 pilot, not a full qualification campaign",
        "no_lease_or_key_handling": True,
        "required_external_watchdog_seconds": 3600,
        "maximum_case_pairs": 6,
        "retrieval_reserve_seconds": 600,
        "prior_baseline_case_medians_seconds": [
            {
                "seed": row["seed"],
                "track": row["track"],
                "seconds": row["baseline_seconds"],
            }
            for row in json.loads(
                (optimization / "paired-combined-summary.json").read_text()
            )
        ],
        "canonical_flags": {
            "resident_control": 1,
            "residual_rounds": 12,
            "residual_graph": 1,
            "diagnostics_rounds": 16,
            "diagnostics_graph": 1,
            "outer_graph": 1,
            "pinned_scalars": 1,
        },
        "scope": "Original baseline and candidate must match immutable global identity, all numerical outputs and actual decks before profiles. Optional adaptive code is never used by a full solve.",
        "candidate_scalar_trace": "Each candidate records real GPU update count; diagnostic trace overhead excludes all pilot timings from qualification.",
        "execution": [
            "Root creates the lease only after local preflight succeeds and starts the independent one-hour watchdog before create.",
            "Upload/extract the tar into a fresh /opt/vmecpp-pilot-* leased container directory; verify archive SHA before extraction.",
            "Run cloud_pilot.py bootstrap/run with --execute and actual lease-started-at/expires-at timestamps from the local lease ledger.",
            "Call collect between completed stages, scp the returned archive, and verify-collection locally with its expected SHA. Retrieval is incremental and never requires provider credentials on the host.",
            "Begin final retrieval with at least ten minutes left. Destroy promptly after verified retrieval and confirm teardown_verified. Never wait for expiry on an idle lease.",
        ],
        "estimated_minutes_not_measured": {
            "provision": 3,
            "upload_at_50Mbit_per_second": 2,
            "bootstrap_and_relocation": 5,
            "strict_calibration": 1,
            "six_case_pairs_if_candidate_equals_prior_baseline": 24.93,
            "nsight_and_selected_ncu": 7,
            "optional_tools_if_reserve_allows": 4,
            "retrieval_and_destroy": 8,
        },
        "contingencies": [
            "All allowances are estimates; new A100 CPU/candidate rates are not measurements yet.",
            "Slow upload/apt or slow first pair reduces the number of pilot cases. No automatic extension or second lease.",
            "Counter permission/tool failures remain failures with logs; do not spend the reserve debugging or claim unexecuted captures.",
            "Nsight is a bounded node window; Ncu measures captured-frame kernels. Calibration is not a whole-solver roofline claim.",
            "The extra publicP1 mpol12 fixture/reference is included but never silently added to the six main case pairs.",
        ],
    }
    save(out / "pilot-plan.json", plan)
    manifest = {
        "purpose": plan["purpose"],
        "snapshot": snapshot.name,
        "original_extension_sha256": digest(
            Path(
                "/home/a3ahmad/.cga/envs/oracle_v026/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
            )
        ),
        "matching_extension_sha256": digest(
            package / "cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
        ),
        "files": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    }
    save(out / "payload-manifest.json", manifest)
    archive = out.with_suffix(".tar")
    with tarfile.open(archive, "w") as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=str(path.relative_to(out)), recursive=False)
    report = {
        "archive": str(archive),
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
        "input_files": len(manifest["files"]),
        "compressed_input_archives_are_not_recompressed": True,
        "lease_created": False,
    }
    save(out.with_suffix(".archive.json"), report)
    return report


def prepare_optional(workspace: Path, out: Path, copy, tree) -> None:
    optimization = workspace / "build/optimization"
    prototype = optimization / "adaptive-residual-prototype"
    hashes = json.loads((prototype / "gates-2/hashes.json").read_text())
    for name, expected in hashes.items():
        if digest(workspace / name) != expected:
            message = "Frozen optional adaptive artifact changed: " + name
            raise ValueError(message)
    copy(prototype / "adaptive-residual.so", "optional/lib/adaptive-residual.so")
    copy(
        optimization / "resident-iteration17/iteration.so", "optional/lib/iteration.so"
    )
    copy(
        prototype / "gates-2/adaptive-residual-benchmark",
        "optional/bin/adaptive-residual-benchmark",
    )
    copy(
        prototype / "gates-2/source.cc",
        "optional/source/adaptive-residual-benchmark.cc",
    )
    copy(
        optimization / "resident-iteration10/real-iteration.bin",
        "optional/captures/real-iteration.bin",
    )
    for name in ("gpu_adaptive_residual.cu", "gpu_adaptive_residual.h"):
        copy(workspace / "tools/optimization" / name, "optional/source/" + name)
    for name in ("build-command.json", "harness-build-command.json"):
        copy(prototype / name, "optional/provenance/" + name)
    copy(prototype / "gates-2/hashes.json", "optional/provenance/frozen-hashes.json")
    isolation = prototype / "conditional-tool-isolation"
    commands = []
    for name in ("ordinary", "minimal", "if-no-setter", "if-with-setter"):
        copy(isolation / (name + ".cu"), "optional/source/" + name + ".cu")
        binary = out / "optional/bin" / name
        command = [
            "/usr/local/cuda-12.9/bin/nvcc",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-arch=sm_80",
            str(out / "optional/source" / (name + ".cu")),
            "-o",
            str(binary),
        ]
        with (out / "optional/provenance" / (name + "-build.log")).open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120
            )
        commands.append(command)
    save(out / "optional/provenance/sm80-build-commands.json", commands)
    for name in (
        "status.json",
        "further-isolation-status.json",
        "last-isolation-status.json",
        "tool-versions-and-finding.json",
    ):
        copy(isolation / name, "optional/reference-wsl/" + name)
    tree(Path("/usr/local/cuda-12.9/compute-sanitizer"), "optional/compute-sanitizer")
    save(
        out / "optional/README.json",
        {
            "scope": "optional native tooling diagnosis only; adaptive sanitizer qualification was blocked on WSL",
            "canonical_solver8_uses_no_optional_library": True,
            "minimal_binaries_recompiled_for": "sm80 only; no GPU execution occurred during packaging",
            "priority": "skip unless canonical parity/calibration/profiles are complete and retrieval reserve remains",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stella", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--without-optional-tools", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.workspace.resolve(),
                args.stella.resolve(),
                args.out.resolve(),
                optional=not args.without_optional_tools,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
