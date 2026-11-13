"""CPU-only SM120 compilation for the standalone expansion experiment."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-13.5"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("expansion_bench.cu").resolve()
    common = source.with_name("expansion_common.h")
    toolkit, out = args.cuda.resolve(), args.out.resolve()
    headers = (
        common,
        source.with_name("expansion_mma168.cuh"),
        source.with_name("expansion_mma168_cached.cuh"),
        source.with_name("expansion_mma168_parallel.cuh"),
        source.with_name("expansion_mma168_fused.cuh"),
    )
    inputs = {str(path): sha(path) for path in (source, *headers)}
    out.mkdir(parents=True, exist_ok=False)
    command = [
        str(toolkit / "bin/nvcc"),
        "-std=c++20",
        "-O3",
        "--fmad=false",
        "-arch=sm_120",
        "--cudart",
        "shared",
        "-Xptxas=-v",
        "-Xcompiler=-fno-fast-math,-ffp-contract=off",
        "-I" + str(toolkit / "include/cccl"),
        str(source),
        "-L" + str(toolkit / "lib64"),
        "-lcublas",
        "-ldl",
        "-Xlinker=-rpath",
        "-Xlinker=" + str(toolkit / "lib64"),
        "-o",
        str(out / "expansion_bench"),
    ]
    started = time.monotonic()
    with (out / "build.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
        )
    receipt = {
        "command": command,
        "returncode": result.returncode,
        "CPU_build_elapsed_seconds": time.monotonic() - started,
        "sources": inputs,
        "GPU_calls": 0,
    }
    if result.returncode == 0:
        receipt["binary_sha256"] = sha(out / "expansion_bench")
        for path in (source, *headers):
            assert sha(path) == inputs[str(path)]
            (out / path.name).write_bytes(path.read_bytes())
        receipt["libraries"] = {
            name: {
                "path": str((toolkit / "lib64" / name).resolve()),
                "sha256": sha((toolkit / "lib64" / name).resolve()),
            }
            for name in ("libcublas.so", "libcublasLt.so", "libcudart.so")
        }
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))  # noqa: T201 - Explicit CPU build receipt.
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
