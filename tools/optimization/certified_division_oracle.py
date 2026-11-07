"""Audit the isolated division certificate with exact binary64 rational values."""

from __future__ import annotations

import argparse
import fractions
import json
import math
import random
import struct
import sys
from pathlib import Path


def _normal(value: float) -> bool:
    return math.isfinite(value) and abs(value) >= float.fromhex("0x1p-1022")


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def _rounded(value: fractions.Fraction, direction: int = 0) -> float:
    try:
        result = float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf
    if direction < 0 and fractions.Fraction(result) > value:
        result = math.nextafter(result, -math.inf)
    if direction > 0 and fractions.Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _certificate(numerator: float, denominator: float) -> tuple[float, bool]:
    """CPU exact-rational simulation of device explicit-rounding operations."""
    if not _normal(numerator) or not _normal(denominator):
        return 0.0, False
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    reciprocal = 1.0 / denominator
    estimate = numerator * reciprocal
    if not _normal(reciprocal) or not _normal(estimate):
        return 0.0, False
    exact = fractions.Fraction
    residual = _rounded(exact(numerator) - exact(estimate) * exact(denominator))
    candidate = _rounded(exact(residual) * exact(reciprocal) + exact(estimate))
    if not _normal(candidate):
        return 0.0, False
    predecessor = math.nextafter(candidate, -math.inf)
    successor = math.nextafter(candidate, math.inf)
    if not _normal(predecessor) or not _normal(successor):
        return 0.0, False
    lower_gap = (candidate - predecessor) * 0.5
    upper_gap = (successor - candidate) * 0.5
    lower_bound = _rounded(exact(lower_gap) * exact(denominator), -1)
    upper_bound = _rounded(exact(upper_gap) * exact(denominator), -1)
    if not (
        lower_gap > 0
        and upper_gap > 0
        and 0 < lower_bound < math.inf
        and 0 < upper_bound < math.inf
    ):
        return 0.0, False
    error = exact(numerator) - exact(candidate) * exact(denominator)
    certified = _rounded(error, -1) > -lower_bound and _rounded(error, 1) < upper_bound
    return candidate, certified


def _audit_certified(numerator: float, denominator: float, result: float) -> None:
    exact = fractions.Fraction
    quotient = exact(numerator) / exact(denominator)
    predecessor = exact(math.nextafter(result, -math.inf))
    successor = exact(math.nextafter(result, math.inf))
    assert (predecessor + exact(result)) / 2 < quotient
    assert quotient < (successor + exact(result)) / 2
    assert _bits(float(quotient)) == _bits(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--simulate", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certified = 0
    count = 0
    if args.records:
        payload = args.records.read_bytes()
        if len(payload) % 48:
            message = "Truncated GPU record file"
            raise ValueError(message)
        for (
            numerator,
            denominator,
            reciprocal,
            result,
            reference,
            flag,
        ) in struct.iter_unpack("<dddddQ", payload):
            count += 1
            if _bits(result) != _bits(reference):
                message = f"GPU division bit mismatch at record {count}"
                raise ValueError(message)
            if flag:
                assert _bits(reciprocal) == _bits(1.0 / denominator)
                _audit_certified(numerator, denominator, result)
                certified += 1
        mode = "GPU records audited using exact Fraction arithmetic"
    else:
        generator = random.Random(982434)
        for _ in range(args.simulate):
            numerator, denominator = (
                struct.unpack("<d", generator.getrandbits(64).to_bytes(8, "little"))[0]
                for _ in range(2)
            )
            result, flag = _certificate(numerator, denominator)
            count += 1
            if flag:
                _audit_certified(numerator, denominator, result)
                certified += 1
        mode = "CPU exact-rational simulation only; no GPU execution"
    report = {
        "mode": mode,
        "cases": count,
        "certified": certified,
        "fallback": count - certified,
        "certified_bit_mismatches": 0,
        "seed": 982434,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
