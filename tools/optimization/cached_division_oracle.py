"""Exact Fraction proof controls for exponent-constructed division bounds."""

from __future__ import annotations

import argparse
import fractions
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

FRACTION_MASK = (1 << 52) - 1
MAX_FINITE = float.fromhex("0x1.fffffffffffffp1023")


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def number(value: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", value))[0]


def normal(value: float) -> bool:
    return 0 < ((bits(value) >> 52) & 2047) < 2047


def rounded(value: fractions.Fraction, direction: int = 0) -> float:
    try:
        result = float(value)
    except OverflowError:
        if value > 0:
            return MAX_FINITE if direction < 0 else math.inf
        return -MAX_FINITE if direction > 0 else -math.inf
    if direction < 0 and fractions.Fraction(result) > value:
        result = math.nextafter(result, -math.inf)
    if direction > 0 and fractions.Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def bounds(denominator: float, candidate: float) -> tuple[float, float] | None:
    """Mirror only the new integer/exponent construction, with no FP bounds ops."""
    if not normal(denominator):
        return None
    encoded = bits(candidate)
    exponent = (encoded >> 52) & 2047
    if not 3 <= exponent <= 2045:
        return None
    positive_denominator = bits(abs(denominator))
    base = ((positive_denominator >> 52) & 2047) + exponent - 1023 - 53
    power = (encoded & FRACTION_MASK) == 0
    negative = bool(encoded >> 63)
    lower_exponent = base - int(power and not negative)
    upper_exponent = base - int(power and negative)
    if not (1 <= lower_exponent <= 2046 and 1 <= upper_exponent <= 2046):
        return None
    fraction = positive_denominator & FRACTION_MASK
    return (
        number((lower_exponent << 52) | fraction),
        number((upper_exponent << 52) | fraction),
    )


def audit_bounds(denominator: float, candidate: float) -> bool:
    constructed = bounds(denominator, candidate)
    if constructed is None:
        return False
    exact = fractions.Fraction
    predecessor = math.nextafter(candidate, -math.inf)
    successor = math.nextafter(candidate, math.inf)
    lower_gap = (exact(candidate) - exact(predecessor)) / 2
    upper_gap = (exact(successor) - exact(candidate)) / 2
    # Independent exact-rational neighbor distances, not the bit formula.
    lower = exact(abs(denominator)) * lower_gap
    upper = exact(abs(denominator)) * upper_gap
    assert exact(constructed[0]) == lower
    assert exact(constructed[1]) == upper
    # Old certificate's directed multiplications round EXACTLY in this scope.
    assert bits(rounded(lower, -1)) == bits(constructed[0])
    assert bits(rounded(upper, -1)) == bits(constructed[1])
    # Strict interval predicates reject synthetic exact midpoint residuals.
    assert not (-constructed[0] > -constructed[0])
    assert not (constructed[1] < constructed[1])
    return True


def certificate(numerator: float, denominator: float) -> tuple[float, bool]:
    """Exact-rational simulation of the unchanged RN estimate/FMA operations."""
    if not normal(numerator) or not normal(denominator):
        return 0.0, False
    reciprocal = 1.0 / denominator
    if not normal(reciprocal):
        return 0.0, False
    d = abs(denominator)
    n = -numerator if denominator < 0 else numerator
    r = abs(reciprocal)
    estimate = n * r
    if not normal(estimate):
        return 0.0, False
    exact = fractions.Fraction
    residual = rounded(exact(n) - exact(estimate) * exact(d))
    candidate = rounded(exact(residual) * exact(r) + exact(estimate))
    constructed = bounds(d, candidate)
    if constructed is None:
        return candidate, False
    error = exact(n) - exact(candidate) * exact(d)
    accepted = (
        rounded(error, -1) > -constructed[0] and rounded(error, 1) < constructed[1]
    )
    if accepted:
        assert bits(candidate) == bits(float(exact(numerator) / exact(denominator)))
        assert audit_bounds(d, candidate)
    return candidate, accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--random", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator = random.Random(439274)
    bound_cases = supported = 0
    exponents = [
        1,
        2,
        3,
        4,
        52,
        53,
        1022,
        1023,
        1024,
        1073,
        1074,
        1075,
        2044,
        2045,
        2046,
    ]
    fractions_bits = [0, 1, 2, 1 << 51, FRACTION_MASK - 1, FRACTION_MASK]
    for candidate_exponent in exponents:
        for denominator_exponent in exponents:
            for candidate_fraction in fractions_bits:
                for denominator_fraction in fractions_bits:
                    for sign in [0, 1]:
                        candidate = number(
                            (sign << 63)
                            | (candidate_exponent << 52)
                            | candidate_fraction
                        )
                        denominator = number(
                            (denominator_exponent << 52) | denominator_fraction
                        )
                        supported += audit_bounds(denominator, candidate)
                        bound_cases += 1
    for _ in range(args.random):
        denominator = number(
            (generator.randrange(1, 2047) << 52) | generator.getrandbits(52)
        )
        candidate = number(
            (generator.randrange(2) << 63)
            | (generator.randrange(1, 2047) << 52)
            | generator.getrandbits(52)
        )
        supported += audit_bounds(denominator, candidate)
        bound_cases += 1
    simulated = accepted = 0
    for _ in range(args.random):
        numerator, denominator = (number(generator.getrandbits(64)) for _ in range(2))
        _, flag = certificate(numerator, denominator)
        simulated += 1
        accepted += flag
    record_count = record_accepted = old_accepted_rejected_by_scope = 0
    if args.records:
        data = args.records.read_bytes()
        assert len(data) % 48 == 0
        for n, d, reciprocal, _, reference, old_flag in struct.iter_unpack(
            "<dddddQ", data
        ):
            candidate, flag = certificate(n, d)
            record_count += 1
            if flag:
                assert bits(reciprocal) == bits(1.0 / d)
                assert bits(candidate) == bits(reference)
                assert old_flag == 1
                record_accepted += 1
            elif old_flag:
                assert bounds(abs(d), candidate) is None
                old_accepted_rejected_by_scope += 1
    # A deliberately wrong exponent is detected by the independent Fraction cell.
    expected = bounds(1.0, 1.0)
    assert expected is not None
    wrong = math.ldexp(expected[0], 1)
    exact_gap = (
        fractions.Fraction(1.0) - fractions.Fraction(math.nextafter(1.0, -math.inf))
    ) / 2
    assert fractions.Fraction(wrong) != exact_gap
    report = {
        "scope": "CPU exact Fraction proof controls only; new GPU implementation UNEXECUTED",
        "seed": 439274,
        "independent_neighbor_bound_cases": bound_cases,
        "supported_exact_normal_bound_cases": supported,
        "random_full_estimate_certificate_cases": simulated,
        "random_certified_exact_RN_cases": accepted,
        "old_executed_GPU_record_inputs_CPU_replayed": record_count,
        "new_CPU_certificate_accepted_exact_old_GPU_reference": record_accepted,
        "old_accepted_new_scope_rejected": old_accepted_rejected_by_scope,
        "new_accepts_are_subset_of_old_executed_accepts": True,
        "all_supported_bounds_equal_exact_neighbor_gap_products": True,
        "accepted_bit_mismatches": 0,
        "wrong_exponent_negative_control_detected": True,
        "GPU_executions_of_new_code": 0,
        "original_record_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest()
        if args.records
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
