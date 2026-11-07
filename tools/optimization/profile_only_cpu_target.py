"""Pure stdlib CPU target proving exact Nsight profile-exec forwarding."""

import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    root, destination = (Path(value).resolve() for value in sys.argv[1:])
    policy = json.loads((root / "profiling/nsys-profile-policy.json").read_text())
    runtime = json.loads((root / "bootstrap-status.json").read_text())["runtime_files"]
    directory = root / "work/nsys/target-linux-x64"
    expected = [directory / name for name, _ in policy["ordered_preloads"]]
    shim = root / "work/lib/solver-shim.so"
    expected.append(shim)
    entries = os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
    if [Path(value).resolve() for value in entries] != expected:
        message = (
            "Actual forwarded preload order differs from two proxies plus app shim"
        )
        raise RuntimeError(message)
    records = []
    for path, sha in [
        *((directory / name, sha) for name, sha in policy["ordered_preloads"]),
        (shim, runtime["work/lib/solver-shim.so"]),
    ]:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha:
            message = "Forwarded preload hash mismatch"
            raise RuntimeError(message)
        records.append({"path": str(path), "sha256": actual})
    routing = directory / policy["routing"][0]
    if (
        any(
            Path(os.environ.get(key, "")).resolve() != routing
            for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")
        )
        or hashlib.sha256(routing.read_bytes()).hexdigest() != policy["routing"][1]
    ):
        message = "Actual forwarded injection routing differs"
        raise RuntimeError(message)
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)")).resolve()
            if path in expected:
                mapped.add(str(path))
    if {str(path) for path in expected} != mapped:
        message = "Forwarded expected preloads are not actually mapped"
        raise RuntimeError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "exact": True,
                "ordered_preloads": records,
                "actual_expected_mappings": sorted(mapped),
                "routing_file_sha256": policy["routing"][1],
                "target_imports_only_stdlib": True,
                "VMEC_or_application_CUDA_calls": 0,
                "scope": "Actual Nsight -> profile-exec -> pure stdlib target; preload/library static initialization is not a solver/GPU workload",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
