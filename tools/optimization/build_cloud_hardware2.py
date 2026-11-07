"""Prepare a new unsealed timing payload; never rent or execute GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from tools.optimization import cloud_pilot_compat as compat
from tools.optimization import cloud_pilot_hardware2 as pilot

ROOT = Path(__file__).resolve().parents[2]


def build(out, base):
    out, base = out.resolve(), base.resolve()
    if out.is_relative_to(base) or base.is_relative_to(out):
        msg = "New preparation must be separate from frozen input"
        raise ValueError(msg)
    compat.preflight(base)
    original = json.loads((base / "payload-manifest.json").read_text())
    shutil.copytree(base, out)
    names = {
        "cloud_pilot_hardware2.py": "cloud_pilot.py",
        "cloud_pilot_hardware2_controller.py": "cloud_pilot_hardware2_controller.py",
        "build_cloud_hardware2.py": "build_cloud_hardware2.py",
    }
    hashes = {}
    for source, target in names.items():
        path = ROOT / "tools/optimization" / source
        shutil.copy2(path, out / "tools/optimization" / target)
        hashes[source] = pilot.digest(path)
    value = json.loads((out / "pilot-config.json").read_text())
    value["counter_policy"] = "timing-and-trace"
    value["raw_compare_dependency_policy"] = "pinned-baseline-subprocess"
    saved = (
        ROOT
        / "build/optimization/a100-hardware-pilot-1793829869/controller/snapshot-0002/results/evaluations"
    )
    closure = out / "profiling/raw-closure"
    for label in ("baseline", "candidate"):
        source = saved / ("public_p3_seed-P3-" + label)
        destination = closure / label
        destination.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    proof = {
        str(path.relative_to(closure)): pilot.digest(path)
        for path in closure.rglob("*")
        if path.is_file()
    }
    pilot.save(closure / "manifest.json", proof)
    package_archive = out / "archives/oracle-packages.tar.gz"
    modules = {}
    wanted = {name + "/__init__.py": name for name in ("numpy", "netCDF4", "cftime")}
    with tarfile.open(package_archive) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in wanted and member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                modules[wanted[name]] = {
                    "archive_member": name,
                    "sha256": hashlib.sha256(stream.read()).hexdigest(),
                }
    if set(modules) != set(wanted.values()):
        message = "Pinned comparator module source missing from dependency archive"
        raise ValueError(message)
    pilot.save(
        closure / "dependencies.json",
        {
            "archive": "archives/oracle-packages.tar.gz",
            "archive_sha256": pilot.digest(package_archive),
            "modules": modules,
        },
    )
    value["required_assets"] = sorted(
        set(value["required_assets"])
        | {
            "profiling/raw-closure/" + name
            for name in (*proof, "manifest.json", "dependencies.json")
        }
    )
    value["permission_denied_policy"] = (
        "retain false status; continue mandatory six-case identity and timing/trace diagnostics; no counter-free roofline claim"
    )
    value["required_assets"] = sorted(
        set(value["required_assets"])
        | {"tools/optimization/" + name for name in names.values()}
    )
    pilot.save(out / "pilot-config.json", value)
    pilot.save(
        out / "provenance/hardware2-preparation.json",
        {
            "ready_for_parent_controlled_pilot": False,
            "sealed": False,
            "source_hashes": hashes,
            "base_manifest_sha256": pilot.digest(base / "payload-manifest.json"),
            "baseline_label": pilot.BASELINE_LABEL,
            "scope": "CPU-only preparation; no new VMEC/GPU/cloud execution",
        },
    )
    manifest = dict(original)
    manifest["files"] = {
        str(path.relative_to(out)): pilot.digest(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
        and path.name != "payload-manifest.json"
        and "__pycache__" not in path.parts
    }
    for name, expected in original["files"].items():
        if (
            name not in ("tools/optimization/cloud_pilot.py", "pilot-config.json")
            and manifest["files"].get(name) != expected
        ):
            raise ValueError("Unexpected frozen input change: " + name)
    pilot.save(out / "payload-manifest.json", manifest)
    pilot.preflight(out)
    compat.preflight(base)
    return {
        "ready_for_parent_controlled_pilot": False,
        "files": len(manifest["files"]),
        "out": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "build/optimization/a100-hardware-payload4",
    )
    args = parser.parse_args()
    if args.out is None:
        return
    build(args.out, args.base)


if __name__ == "__main__":
    main()
