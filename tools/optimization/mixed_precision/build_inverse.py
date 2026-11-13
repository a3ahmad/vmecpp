"""CPU-build the isolated inverse object using the retained CUDA 12.9 flags."""

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from tools.optimization.gb202 import compiler_owner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT.parent / "gb202/sources.json").read_text())
    unit = next(x for x in manifest["translation_units"] if x["name"] == "fourier")
    command = [
        "/usr/local/cuda-12.9/bin/nvcc",
        *unit["flags"],
        "-I" + str(ROOT),
        "-I" + str(ROOT.parent / "gb202/frozen"),
        "-c",
        str(ROOT / "gpu_fourier_mixed.cu"),
        "-o",
        str(out / "fourier.o"),
    ]
    # Include original headers under their ABI names, without modifying snapshots.
    for name, source in unit["files"].items():
        if name != "gpu_fourier.cu":
            data = (ROOT.parent / "gb202" / source).read_bytes()
            (out / name).write_bytes(data)
    command.insert(1, "-I" + str(out))
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    environment = {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
    }
    job = compiler_owner.run_job(command, "inverse-compile", 180, environment, out)
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "gpu_fourier_mixed.cu", ROOT / "inverse_mixed_kernels.cuh")
    }
    receipt = {"job": job, "sources": sources, "CUDA_executions": 0}
    if (out / "fourier.o").is_file():
        receipt["object_sha256"] = hashlib.sha256(
            (out / "fourier.o").read_bytes()
        ).hexdigest()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if job.get("direct_exit_status") == 0 else 1)


if __name__ == "__main__":
    main()
