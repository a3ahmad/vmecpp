"""Prepare an unsealed two-evaluation profiling diagnostic; no rental actions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.optimization import cloud_pilot_hardware2 as previous
from tools.optimization import cloud_pilot_profileonly2 as pilot
from tools.optimization import nsys_injection_guard as guard

ROOT = Path(__file__).resolve().parents[2]


def build(out, base, prior):
    out, base, prior = (path.resolve() for path in (out, base, prior))
    if out.is_relative_to(base) or base.is_relative_to(out):
        message = "Profile-only output must be separate from prior payload"
        raise ValueError(message)
    previous.preflight(base)
    if pilot.digest(prior) != pilot.PRIOR_CANONICAL_SHA256:
        message = "Exact root-audited eighteen-evaluation receipt required"
        raise ValueError(message)
    original = json.loads((base / "payload-manifest.json").read_text())
    shutil.copytree(base, out)
    names = {
        "cloud_pilot_profileonly2.py": "cloud_pilot.py",
        "cloud_pilot_profileonly2_controller.py": "cloud_pilot_profileonly2_controller.py",
        "nsys_injection_guard.py": "nsys_injection_guard.py",
        "validate_profile_marker.py": "validate_profile_marker.py",
        "build_cloud_profileonly2.py": "build_cloud_profileonly2.py",
    }
    source_hashes = {}
    for source, target in names.items():
        path = ROOT / "tools/optimization" / source
        shutil.copy2(path, out / "tools/optimization" / target)
        source_hashes[source] = pilot.digest(path)
    destination = out / "provenance/prior-canonical-evidence.json"
    shutil.copy2(prior, destination)
    shutil.copy2(
        ROOT / "tools/optimization/profile_only_cpu_target.py",
        out / "profiling/profile-only-cpu-target.py",
    )
    pilot.save(
        out / "profiling/nsys-profile-policy.json",
        {
            "routing": guard.ROUTING_LIBRARY,
            "ordered_preloads": guard.ORDERED_PRELOAD_LIBRARIES,
        },
    )
    value = json.loads((out / "pilot-config.json").read_text())
    value.update(
        diagnostic_mode="anchor-plus-profile",
        prior_canonical_evidence_sha256=pilot.digest(prior),
    )
    value["required_assets"] = sorted(
        set(value["required_assets"])
        | {"tools/optimization/" + name for name in names.values()}
        | {
            "provenance/prior-canonical-evidence.json",
            "profiling/profile-only-cpu-target.py",
            "profiling/nsys-profile-policy.json",
        }
    )
    pilot.save(out / "pilot-config.json", value)
    pilot.save(
        out / "provenance/profile-only-preparation.json",
        {
            "sealed": False,
            "ready_for_parent_controlled_pilot": False,
            "source_hashes": source_hashes,
            "base_manifest_sha256": pilot.digest(base / "payload-manifest.json"),
            "prior_canonical_evidence_sha256": pilot.digest(prior),
            "scope": "Two futureGPUdiagnostics only; no newCPUequilibrium or eighteen-case rerun; CPUpreparation only",
        },
    )
    manifest = dict(original)
    manifest["files"] = {
        str(path.relative_to(out)): pilot.digest(path)
        for path in out.rglob("*")
        if path.is_file()
        and path.name != "payload-manifest.json"
        and "__pycache__" not in path.parts
    }
    for name, expected in original["files"].items():
        if (
            name not in ("tools/optimization/cloud_pilot.py", "pilot-config.json")
            and manifest["files"].get(name) != expected
        ):
            message = "Unexpected change to frozen prior payload: " + name
            raise ValueError(message)
    pilot.save(out / "payload-manifest.json", manifest)
    pilot.preflight(out)
    previous.preflight(base)
    return {
        "phase": "UNSEALED_PROFILE_ONLY_PREPARATION",
        "files": len(manifest["files"]),
        "ready_for_parent_controlled_pilot": False,
        "planned_GPU_evaluations": 2,
        "actual_GPU_or_provider_actions": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "build/optimization/a100-hardware2-payload2"
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=ROOT
        / "build/optimization/a100-hardware2-actual-audit1/terminal-audit.json",
    )
    args = parser.parse_args()
    if args.out is not None:
        build(args.out, args.base, args.prior)


if __name__ == "__main__":
    main()
