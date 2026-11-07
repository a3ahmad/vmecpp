"""Probe certified sequential-sum relaxation on captured real force residuals.

This is a CPU algorithm experiment, not a GPU speedup measurement. The final certificate
checks every prefix recurrence bit for bit, with initial sum +0. See Kapre and DeHon,
ARITH 2007, Optimistic Parallelization of Floating-Point Accumulation. Correction signs
here are defined as target minus current.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def parallel_prefix(values: np.ndarray) -> np.ndarray:
    """A fixed Hillis-Steele tree, used only to propose prefix values."""
    result = values.copy()
    offset = 1
    while offset < result.size:
        previous = result.copy()
        result[offset:] = previous[offset:] + previous[:-offset]
        offset *= 2
    return result


def probe(terms: np.ndarray, expected: float, max_rounds: int) -> dict:
    sequential = np.empty_like(terms)
    total = 0.0
    for i, value in enumerate(terms):
        total += float(value)
        sequential[i] = total
    if struct.pack("<d", total) != struct.pack("<d", expected):
        message = "Sequential reconstruction does not match CPU capture"
        raise ValueError(message)
    prefix = parallel_prefix(terms)
    rounds = []
    for iteration in range(max_rounds + 1):
        previous = np.empty_like(prefix)
        previous[0] = 0.0
        previous[1:] = prefix[:-1]
        target = previous + terms
        invalid = int(
            np.count_nonzero(target.view(np.uint64) != prefix.view(np.uint64))
        )
        actual = int(
            np.count_nonzero(sequential.view(np.uint64) != prefix.view(np.uint64))
        )
        rounds.append(
            {
                "round": iteration,
                "invalid_recurrences": invalid,
                "different_sequential_prefixes": actual,
            }
        )
        if invalid == 0:
            if actual != 0:
                message = "A purported recurrence certificate is incorrect"
                raise ValueError(message)
            break
        if iteration == max_rounds:
            break
        correction = target - prefix
        prefix = prefix + parallel_prefix(correction)
    return {
        "terms": int(terms.size),
        "certified": rounds[-1]["invalid_recurrences"] == 0,
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=32)
    args = parser.parse_args()
    data = args.capture.read_bytes()
    if data[:8] != b"VMECGIT1":
        parser.error("requires a real iteration-stage capture")
    offset = 8 + 5 * 4
    sizes = struct.unpack_from("<37Q", data, offset)
    offset += 37 * 8
    arrays = []
    for count in sizes:
        arrays.append(np.frombuffer(data, dtype="<f8", count=count, offset=offset))
        offset += 8 * count
    if offset != len(data):
        parser.error("capture length mismatch")
    inputs, outputs = arrays[:17], arrays[17:]
    rows = []
    for name, values, residuals in (
        ("before_preconditioning", inputs[:6], outputs[18]),
        ("after_preconditioning", outputs[12:18], outputs[19]),
    ):
        for component in range(3):
            terms = np.empty(2 * values[2 * component].size, dtype=np.float64)
            terms[0::2] = values[2 * component] * values[2 * component]
            terms[1::2] = values[2 * component + 1] * values[2 * component + 1]
            rows.append(
                {
                    "stage": name,
                    "component": component,
                    **probe(terms, float(residuals[component]), args.max_rounds),
                }
            )
    args.out.write_text(
        json.dumps(
            {
                "measurement": "executed CPU algorithm probe on actual captured residuals; not GPU timing or full-solve parity",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
