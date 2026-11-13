"""Link the research resident with two unchanged retained SM120 objects."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from tools.optimization.gb202 import compiler_owner  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    retained = REPO / "build/optimization/gb202-packed-sm1201/build2"
    assert (
        sha(retained / "resident.so")
        == "60d24820102c953dedc85a8487ec3023fdf7de921ff7e82ff25294718309e078"
    )
    assert (
        sha(retained / "solver-backend.so")
        == "2eb80b39e9051961721f8cf9b28c520f970a599d19a8197b8707c86cb52dcf2d"
    )
    inverse = REPO / "build/optimization/mixed-precision1/inverse-build1/fourier.o"
    forward = (
        REPO / "build/optimization/mixed-precision-forward1/build1/forward-counted.o"
    )
    assert (
        sha(inverse)
        == "6d1dbbd1f1035dd6b2dde4aa617ff90a19c6ad3865710b4929b00d4341b5d0ac"
    )
    assert (
        sha(forward)
        == "3e7bc086e94eb7040378664c58f16f2da90dc9ca0e792fb3ef369263e7fcc1c6"
    )
    objects = [
        retained / "objects/resident.o",
        inverse,
        forward,
        retained / "objects/constraint.o",
    ]
    command = [
        "g++",
        "-Wl,--no-undefined",
        "-shared",
        *map(str, objects),
        "-L/usr/local/cuda-12.9/lib64",
        "-lcudart",
        "-Wl,-rpath,/usr/local/cuda-12.9/lib64",
        "-o",
        str(out / "resident.so"),
    ]
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
    }
    jobs = [compiler_owner.run_job(command, "resident-link", 60, environment, out)]
    assert jobs[-1].get("direct_exit_status") == 0
    # Only relocate the dependency of an isolated backend copy.
    backend = out / "solver-backend.so"
    shutil.copyfile(retained / "solver-backend.so", backend)
    prior = str(retained / "resident.so")
    patch = shutil.which("patchelf") or str(
        REPO / "build/optimization/a100-compat-payload-final1/bin/patchelf"
    )
    assert patch
    jobs.append(
        compiler_owner.run_job(
            [patch, "--replace-needed", prior, str(out / "resident.so"), str(backend)],
            "backend-resident-binding",
            30,
            environment,
            out,
        )
    )
    assert jobs[-1].get("direct_exit_status") == 0
    receipt = {
        "scope": "CPU link only; mode64 scientific parity pending",
        "jobs": jobs,
        "objects": {str(p): sha(p) for p in objects},
        "outputs": {p.name: sha(p) for p in out.glob("*.so")},
        "CUDA_executions": 0,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
