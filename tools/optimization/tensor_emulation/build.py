"""Build the isolated host-side cuBLAS experiment; never execute CUDA APIs."""

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
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).with_name("bench.cu")
    )
    args = parser.parse_args()
    toolkit, out = args.cuda.resolve(), args.out.resolve()
    source = args.source.resolve()
    inputs = [
        source,
        toolkit / "include/cublas_api.h",
        toolkit / "include/library_types.h",
        toolkit / "include/cuda_runtime_api.h",
    ]
    assert source.read_bytes().isascii()
    before = {str(path): sha(path) for path in inputs}
    out.mkdir(exist_ok=False, parents=True)
    command = [
        str(toolkit / "bin/nvcc"),
        "-std=c++20",
        "-O3",
        "--fmad=false",
        "--cudart",
        "shared",
        "-Xcompiler=-fno-fast-math,-ffp-contract=off",
        str(source),
        "-I" + str(toolkit / "include"),
        "-L" + str(toolkit / "lib64"),
        "-lcublas",
        "-ldl",
        "-Xlinker=-rpath",
        "-Xlinker=" + str(toolkit / "lib64"),
        "-o",
        str(out / "bench"),
    ]
    started = time.monotonic()
    with (out / "build.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
        )
    receipt = {
        "command": command,
        "returncode": result.returncode,
        "elapsed_CPU_build_seconds": time.monotonic() - started,
        "source_and_headers": before,
        "CUDA_runtime_calls": 0,
        "GPU_workloads": 0,
    }
    if result.returncode == 0:
        receipt["binary_sha256"] = sha(out / "bench")
        libraries = {}
        for name in ("libcublas.so", "libcublasLt.so", "libcudart.so"):
            path = (toolkit / "lib64" / name).resolve()
            libraries[name] = {"canonical_path": str(path), "sha256": sha(path)}
        receipt["runtime_libraries"] = libraries
        receipt["compiler_version"] = subprocess.check_output(
            [str(toolkit / "bin/nvcc"), "--version"], text=True
        )
        receipt["dynamic_metadata"] = subprocess.check_output(
            ["/usr/bin/readelf", "-d", str(out / "bench")], text=True
        )
    assert before == {str(path): sha(path) for path in inputs}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps(
            {
                "out": str(out),
                "returncode": result.returncode,
                "binary_sha256": receipt.get("binary_sha256"),
                "GPU_calls": 0,
            }
        )
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
