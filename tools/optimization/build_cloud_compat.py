"""Prepare an UNSEALED CPU-compatibility payload from immutable current inputs."""

from __future__ import annotations

# The output is an explicit local preparation recipe, never a rental operation.
# ruff: noqa: T201
import argparse
import json
import shutil
from pathlib import Path

from tools.optimization import cloud_pilot_compat as pilot
from tools.optimization import cloud_pilot_current as current

ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "build/optimization"


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(out, base, probe):
    out, base, probe = (path.resolve() for path in (out, base, probe))
    if out.is_relative_to(base) or base.is_relative_to(out):
        message = "Preparation output must be separate from immutable base"
        raise ValueError(message)
    current.preflight(base)
    original = json.loads((base / "payload-manifest.json").read_text())
    probe_manifest = json.loads((probe / "candidate-manifest.json").read_text())
    ready = json.loads((probe / "probe-ready.json").read_text())
    if not ready.get("CPU_only_private_probe_ready") or ready.get(
        "prospective_cloud_executed"
    ):
        message = "Actual local CPU probe readiness required; no cloud claim"
        raise ValueError(message)
    if pilot.digest(probe / "candidate-manifest.json") != ready["manifest_sha256"]:
        message = "Probe readiness does not bind immutable input manifest"
        raise ValueError(message)
    for name, expected in probe_manifest.items():
        source = (probe / name).resolve()
        if not source.is_relative_to(probe) or pilot.digest(source) != expected:
            message = "Probe path/hash mismatch: " + name
            raise ValueError(message)
    shutil.copytree(base, out)
    replacements = {
        "cloud_pilot_compat.py": "cloud_pilot.py",
        "cloud_cpu_compatibility.py": "cloud_cpu_compatibility.py",
        "cloud_pilot_compat_controller.py": "cloud_pilot_compat_controller.py",
        "build_cloud_compat.py": "build_cloud_compat.py",
    }
    source_hashes = {}
    for name, destination in replacements.items():
        source = ROOT / "tools/optimization" / name
        copy_file(source, out / "tools/optimization" / destination)
        source_hashes[name] = pilot.digest(source)
    probe_root = out / "profiling/cpu-compatibility"
    probe_names = [*probe_manifest, "candidate-manifest.json", "probe-ready.json"]
    for name in probe_names:
        copy_file(probe / name, probe_root / name)
    value = json.loads((out / "pilot-config.json").read_text())
    value["saved_output_cpu_compatibility"] = "required"
    value["required_assets"] = sorted(
        set(value["required_assets"])
        | {"profiling/cpu-compatibility/" + name for name in probe_names}
        | {
            "tools/optimization/cloud_cpu_compatibility.py",
            "tools/optimization/cloud_pilot_compat_controller.py",
        }
    )
    pilot.save(out / "pilot-config.json", value)
    pilot.save(
        out / "provenance/compatibility-preparation.json",
        {
            "sealed": False,
            "ready_for_rental": False,
            "base_manifest_sha256": pilot.digest(base / "payload-manifest.json"),
            "probe_manifest_sha256": ready["manifest_sha256"],
            "source_hashes_at_preparation": source_hashes,
            "scope": "Local CPU-only preparation; no lease/GPU/VMEC work; await source freeze",
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
            message = "Unexpected change to original payload input: " + name
            raise ValueError(message)
    pilot.save(out / "payload-manifest.json", manifest)
    pilot.preflight(out)
    current.preflight(base)
    return {
        "phase": "UNSEALED_PREPARATION_COMPLETE",
        "out": str(out),
        "files": len(manifest["files"]),
        "source_hashes": source_hashes,
        "ready_for_parent_controlled_pilot": False,
        "lease_actions": 0,
        "GPU_runs": 0,
        "VMEC_solves": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--base", type=Path, default=OPT / "a100-current-payload2")
    parser.add_argument("--probe", type=Path, default=OPT / "cpu-compatibility-probe1")
    args = parser.parse_args()
    if args.out is None:
        print("PLAN ONLY: prepare separate unsealed payload; no lease/GPU/VMEC action")
        return
    print(json.dumps(build(args.out, args.base, args.probe), indent=2))


if __name__ == "__main__":
    main()
