"""Freeze/build the isolated instruction-matched CPU-log GPU experiment."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / "source"
    source.mkdir()
    for name in ("gpu_pinned_cpu_log.h", "gpu_pinned_cpu_log.cu", Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    for name in (
        "source-acquisition.json",
        "table-constant-proof.json",
        "instruction-graph-proof.json",
        "design-and-proof-scope.json",
    ):
        shutil.copy2(args.proof / name, source / name)
    shutil.copy2(args.proof / "upstream/glibc-2.39/COPYING.LIB", source / "COPYING.LIB")
    library = Path("/usr/lib/x86_64-linux-gnu/libm.so.6")
    if (
        digest(library)
        != "fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0"
    ):
        message = "Independent expected-value generator requires the pinned local libm"
        raise ValueError(message)
    dynamic = ctypes.CDLL(None)
    dynamic.dlvsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    dynamic.dlvsym.restype = ctypes.c_void_p
    math_library = ctypes.CDLL(str(library))
    pointer = dynamic.dlvsym(math_library._handle, b"log", b"GLIBC_2.29")
    original = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(pointer)
    data = (args.proof / "cpu-corpus.bin").read_bytes()
    count = struct.unpack_from("<Q", data, 8)[0]
    if data[:8] != b"VMECLOG2" or len(data) != 16 + 24 * count:
        message = "Invalid executed CPU corpus"
        raise ValueError(message)
    rows = []
    accepted = observed = 0
    sentinel = struct.unpack("<Q", struct.pack("<d", 1234.5))[0]
    for index in range(count):
        word, expected, flags = struct.unpack_from("<3Q", data, 16 + 24 * index)
        exponent = (word >> 52) & 2047
        eligible = word >> 63 == 0 and 0 < exponent < 2047
        result = sentinel
        if eligible:
            value = struct.unpack("<d", struct.pack("<Q", word))[0]
            result = struct.unpack("<Q", struct.pack("<d", original(value)))[0]
            if flags & 1 and result != expected:
                message = (
                    "Current independent CPU reference differs from actual capture"
                )
                raise ValueError(message)
        rows.append(struct.pack("<4Q", word, result, int(eligible), flags & 1))
        accepted += eligible
        observed += bool(flags & 1)
    corpus = args.out / "corpus.bin"
    corpus.write_bytes(b"VMECGPL1" + struct.pack("<Q", count) + b"".join(rows))
    replay = Path(__file__).with_name("gpu_controller_replay.cu").read_text()
    if replay.count("logarithm = log(argument);") != 1:
        message = "Controller replay source changed; inspect exact call replacement"
        raise ValueError(message)
    replay = replay.replace(
        "#include <cuda_runtime.h>",
        '#include <cuda_runtime.h>\n#include "gpu_pinned_cpu_log.h"',
    )
    replay = replay.replace(
        "logarithm = log(argument);",
        "{ if (!TryPinnedCpuLog(argument, logarithm)) output.errors_ |= 64; }",
    )
    (source / "pinned_controller_replay.cu").write_text(replay)
    commands = []
    for name, target in [
        ("gpu_pinned_cpu_log.cu", "pinned-log"),
        ("pinned_controller_replay.cu", "controller-replay"),
    ]:
        commands.append(
            [
                "/usr/local/cuda-12.9/bin/nvcc",
                "-std=c++20",
                "-O3",
                "--fmad=false",
                "-lineinfo",
                "-Xptxas=-v",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_120,code=sm_120",
                str(source / name),
                "-o",
                str(args.out / target),
            ]
        )
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=180, check=False
        )
        (args.out / f"build-{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            message = f"Isolated build {index} failed"
            raise RuntimeError(message)
    receipt = {
        "CPU_expected_cases": count,
        "CPU_eligible": accepted,
        "actual_CPU_pairs": observed,
        "GPU_executed": False,
        "reference_SHA256": digest(library),
        "input_CPU_corpus_SHA256": hashlib.sha256(data).hexdigest(),
        "files": {
            str(path.relative_to(args.out)): digest(path)
            for path in [
                *source.iterdir(),
                corpus,
                args.out / "pinned-log",
                args.out / "controller-replay",
            ]
        },
    }
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
