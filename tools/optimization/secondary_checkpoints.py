"""Snapshot completed campaign evidence without touching a running evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

TERMINAL = {"complete", "failed", "qualification_failed", "timed_out"}


def snapshot(root: Path, after: int) -> dict:
    if root != Path(
        "/bulk/vmecpp-optimization-20261103/resident-solver8-capture-replica"
    ):
        message = "Unexpected campaign checkpoint directory"
        raise ValueError(message)
    available = os.sched_getaffinity(0)
    if len(available) > 1:
        os.sched_setaffinity(0, available - {0})
    os.nice(19)
    status = json.loads((root / "full-campaign-status.json").read_text())
    samples_path = root / "paired/samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.is_file() else []
    phase = status["phase"]
    metadata = {"phase": phase, "completed_evaluations": len(samples), "archive": None}
    if len(samples) <= after and phase not in TERMINAL:
        return metadata
    files = {}
    if phase in TERMINAL:
        paths = [path for path in (root / "paired").rglob("*") if path.is_file()]
    else:
        paths = []
        for sample in samples[after:]:
            directory = (
                root
                / "paired"
                / sample["label"]
                / f"{Path(sample['seed']).stem}-{sample['track']}-{sample['pair']:02d}"
            )
            paths.extend(directory.glob("*.json"))
            log = directory.parent / (directory.name + ".log")
            if log.is_file():
                paths.append(log)
    paths.extend(
        root / name
        for name in (
            "full-campaign-status.json",
            "full-campaign-summary.json",
            "full-launch-spec.json",
            "full-supervisor-launch.json",
            "campaign-plan.json",
            "variant-manifest.json",
            "relocation.json",
            "baseline-config.json",
            "candidate-config.json",
            "logs/full-campaign-coordinator.log",
            "logs/full-campaign-supervisor.log",
            "tools/optimization/run_secondary_campaign.py",
        )
        if (root / name).is_file()
    )
    for path in paths:
        if not path.resolve().is_relative_to(root):
            message = "Checkpoint path escapes the authorized campaign"
            raise ValueError(message)
        files[str(path.relative_to(root))] = path.read_bytes()
    files["paired/samples.json"] = (json.dumps(samples, indent=2) + "\n").encode()
    files["full-campaign-status.json"] = (json.dumps(status, indent=2) + "\n").encode()
    manifest = {
        "phase": phase,
        "completed_evaluations": len(samples),
        "after": after,
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    files["checkpoint-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination = root / "checkpoints"
    destination.mkdir(exist_ok=True)
    archive = destination / f"c{len(samples):03d}-{phase}-{time.time_ns()}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mtime = int(time.time())
            bundle.addfile(item, io.BytesIO(data))
    metadata.update(
        archive=str(archive),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        bytes=archive.stat().st_size,
        files=len(manifest["files"]),
    )
    return metadata


def verify(archive: Path, out: Path, expected: str) -> dict:
    allowed = Path(
        "/mnt/f/Users/might/Sync/vmecpp/build/optimization/secondary-campaign-checkpoints"
    )
    if not out.resolve().is_relative_to(allowed) or out.exists():
        message = "Unexpected or existing local checkpoint directory"
        raise ValueError(message)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        message = "Checkpoint archive checksum mismatch"
        raise ValueError(message)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if not all(
            member.isfile() and (out / member.name).resolve().is_relative_to(out)
            for member in members
        ):
            message = "Unsafe checkpoint member"
            raise ValueError(message)
        out.mkdir(parents=True)
        bundle.extractall(out, filter="data")
    manifest = json.loads((out / "checkpoint-manifest.json").read_text())
    for name, expected_file in manifest["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected_file:
            message = "Checkpoint file checksum mismatch: " + name
            raise ValueError(message)
    result = {
        "exact": True,
        "phase": manifest["phase"],
        "completed_evaluations": manifest["completed_evaluations"],
        "verified_files": len(manifest["files"]),
        "archive_sha256": expected,
        "directory": str(out),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "verify"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.mode == "snapshot":
        if args.root is None:
            parser.error("snapshot requires root")
        result = snapshot(args.root.resolve(), args.after)
    else:
        if args.archive is None or args.out is None or args.sha256 is None:
            parser.error("verify requires archive, out and sha256")
        result = verify(args.archive, args.out, args.sha256)
    print(json.dumps(result))  # noqa: T201


if __name__ == "__main__":
    main()
