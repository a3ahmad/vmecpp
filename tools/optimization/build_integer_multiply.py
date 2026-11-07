"""Freeze an exact Fraction corpus and build an isolated multiply experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path


def value(word: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", word))[0]


def bits(number: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", number))[0]


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in (
        "gpu_integer_multiply.h",
        "gpu_integer_multiply.cu",
        Path(__file__).name,
    ):
        shutil.copy2(Path(__file__).parent / name, source / name)
    rng = random.Random(928145)
    pairs: set[tuple[int, int]] = set()
    base = 1 << 52
    fractions = [0, 1, 2, 3, (1 << 51) - 1, 1 << 51, base - 3, base - 2, base - 1]
    for exponent in range(1, 2047):
        for partner in (1, 1023, 2046, max(1, min(2046, 2047 - exponent))):
            for fraction in fractions:
                for sign in range(4):
                    pairs.add(
                        (
                            (exponent << 52) | fraction | ((sign & 1) << 63),
                            (partner << 52)
                            | (base - 1 - fraction)
                            | ((sign >> 1) << 63),
                        )
                    )
    ties = [
        (base + 1, base + (1 << 51)),
        (base + 3, base + (1 << 51)),
        (7 * (1 << 50) + 2, 3 * (1 << 51)),
        (7 * (1 << 50) + 6, 3 * (1 << 51)),
        (base + 1, 2 * base - 2),
    ]
    for a, b in ties:
        for exponent in (1, 2, 1022, 1023, 1024, 2045, 2046):
            for sign in range(4):
                pairs.add(
                    (
                        (exponent << 52) | (a - base) | ((sign & 1) << 63),
                        (1023 << 52) | (b - base) | ((sign >> 1) << 63),
                    )
                )
    specials = [
        0,
        1,
        base - 1,
        base,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0x7FF8000000000123,
        0x7FF0000000000001,
    ]
    for a in specials:
        for b in [*specials, bits(1.0), bits(2.0), bits(0.5)]:
            for sign in range(4):
                pairs.add((a | ((sign & 1) << 63), b | ((sign >> 1) << 63)))
    for _ in range(100000):
        pairs.add((rng.getrandbits(64), rng.getrandbits(64)))
    rows = []
    exact_count = 0
    for a, b in sorted(pairs):
        left, right = value(a), value(b)
        exact = math.isfinite(left) and math.isfinite(right)
        expected = 0
        if exact:
            product = Fraction.from_float(left) * Fraction.from_float(right)
            try:
                rounded = float(product)
            except OverflowError:
                rounded = math.copysign(math.inf, -1 if (a ^ b) >> 63 else 1)
            expected = bits(rounded)
            if rounded == 0.0:
                expected = ((a ^ b) >> 63) << 63
            exact_count += 1
        rows.append(struct.pack("<4Q", a, b, expected, int(exact)))
    corpus = output / "corpus.bin"
    corpus.write_bytes(b"VMECIMU1" + struct.pack("<Q", len(rows)) + b"".join(rows))
    commands = [
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-DVMECPP_INTEGER_MULTIPLY_CPU",
            "-x",
            "c++",
            str(source / "gpu_integer_multiply.cu"),
            "-o",
            str(output / "cpu-proof"),
        ],
        [str(output / "cpu-proof"), str(corpus)],
        [
            "/usr/local/cuda-12.9/bin/nvcc",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_120,code=sm_120",
            str(source / "gpu_integer_multiply.cu"),
            "-o",
            str(output / "integer-multiply"),
        ],
    ]
    receipts = []
    for index, command in enumerate(commands):
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        (output / f"command-{index}.log").write_text(run.stdout + run.stderr)
        receipts.append({"argv": command, "returncode": run.returncode})
        (output / "commands.json").write_text(json.dumps(receipts, indent=2))
        if run.returncode:
            message = f"Build/proof command {index} failed"
            raise RuntimeError(message)
    run = subprocess.run(
        [
            "/usr/local/cuda-12.9/bin/cuobjdump",
            "--dump-sass",
            str(output / "integer-multiply"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    (output / "sass.txt").write_text(run.stdout)
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "cases": len(rows),
                "Fraction_expected": exact_count,
                "proof_scope": "Finite inputs exact rational product rounded by Python Fraction-to-float; special inputs GPU native reference only",
                "GPU_executed": False,
                "hashes": {
                    str(path.relative_to(output)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in [
                        *source.iterdir(),
                        corpus,
                        output / "cpu-proof",
                        output / "integer-multiply",
                    ]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
