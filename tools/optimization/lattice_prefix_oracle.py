"""Exact-rational CPU proof tests for an isolated positive-prefix proposal.

The proposal never replaces a result without checking every original binary64 recurrence
from +0. This script makes no GPU performance or novelty claim.
"""

from __future__ import annotations

import argparse
import fractions
import hashlib
import itertools
import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Any

MASK = (1 << 64) - 1
FRACTION_MASK = (1 << 52) - 1
LOWER_MANTISSA = 1 << 52
UPPER_MANTISSA = 1 << 53


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def number(value: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", value))[0]


def exponent(value: float) -> int:
    return (bits(value) >> 52) & 2047


def term_map(term: float, binade: int) -> tuple[int, int] | None:
    """Derive two parity-dependent deltas using only input bits and integers."""
    if term == 0.0:
        return (0, 0)
    if not math.isfinite(term) or term < 0.0 or not 1 <= binade <= 2046:
        return None
    field = exponent(term)
    mantissa = bits(term) & FRACTION_MASK
    if field:
        mantissa += LOWER_MANTISSA
        shift = field - binade
    else:
        shift = 1 - binade
    if shift >= 0:
        if mantissa.bit_length() + shift > 64:
            return None
        integer = mantissa << shift
        return (integer, integer)
    places = -shift
    if places > 53:
        return (0, 0)
    integer = mantissa >> places
    remainder = mantissa & ((1 << places) - 1)
    half = 1 << (places - 1)
    if remainder < half:
        return (integer, integer)
    if remainder > half:
        return ((integer + 1) & MASK, (integer + 1) & MASK)
    return (
        (integer + (integer & 1)) & MASK,
        (integer + ((integer + 1) & 1)) & MASK,
    )


def compose(earlier: tuple[int, int], later: tuple[int, int]) -> tuple[int, int]:
    """Return later after earlier, matching left-to-right inclusive scan order."""
    return (
        (earlier[0] + later[earlier[0] & 1]) & MASK,
        (earlier[1] + later[(1 + earlier[1]) & 1]) & MASK,
    )


def apply(mapping: tuple[int, int], mantissa: int) -> int:
    return (mantissa + mapping[mantissa & 1]) & MASK


def ordered(terms: list[float]) -> list[float]:
    result = []
    accumulator = 0.0
    for term in terms:
        accumulator += term
        result.append(accumulator)
    return result


def certificate(terms: list[float], prefix: list[float]) -> bool:
    if len(terms) != len(prefix):
        return False
    for index, (term, value) in enumerate(zip(terms, prefix, strict=True)):
        previous = prefix[index - 1] if index else 0.0
        if (
            not math.isfinite(term)
            or not math.isfinite(value)
            or bits(previous + term) != bits(value)
        ):
            return False
    return True


def parallel_guess(terms: list[float]) -> list[float]:
    values = terms.copy()
    offset = 1
    while offset < len(values):
        old = values
        values = [
            value + old[index - offset] if index >= offset else value
            for index, value in enumerate(old)
        ]
        offset *= 2
    return values


def propose(terms: list[float], guess: list[float] | None = None) -> tuple[list, bool]:
    guesses = parallel_guess(terms) if guess is None else guess
    keys = []
    invalid = False
    for index, (term, value) in enumerate(zip(terms, guesses, strict=True)):
        field = exponent(value)
        if (
            not math.isfinite(term)
            or term < 0.0
            or not math.isfinite(value)
            or value < 0.0
            or (field == 0 and value != 0.0)
            or (index and field < keys[-1])
        ):
            invalid = True
        keys.append(field)
    result = [0.0] * len(terms)
    previous = 0.0
    for field, indices in itertools.groupby(range(len(terms)), key=keys.__getitem__):
        positions = list(indices)
        if field == 0:
            if any(terms[index] != 0.0 for index in positions):
                invalid = True
            continue
        if not 1 <= field <= 2046:
            invalid = True
            continue
        start = positions[0]
        seed = previous + terms[start]
        if not math.isfinite(seed) or exponent(seed) != field:
            invalid = True
            continue
        seed_mantissa = LOWER_MANTISSA + (bits(seed) & FRACTION_MASK)
        mapping = (0, 0)
        for index in positions:
            if index != start:
                step = term_map(terms[index], field)
                if step is None:
                    invalid = True
                    continue
                mapping = compose(mapping, step)
            delta = mapping[seed_mantissa & 1]
            if delta >= UPPER_MANTISSA - seed_mantissa:
                invalid = True
                continue
            value = seed_mantissa + delta
            result[index] = number((field << 52) | (value - LOWER_MANTISSA))
        previous = result[positions[-1]]
    return result, not invalid and certificate(terms, result)


def rational_round(value: fractions.Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def captured_cases(path: Path) -> tuple[list[list[float]], dict]:
    payload = path.read_bytes()
    assert payload[:8] == b"VMECGIT1"
    dimensions = struct.unpack_from("<5i", payload, 8)
    input_sizes = struct.unpack_from("<17Q", payload, 28)
    output_sizes = struct.unpack_from("<20Q", payload, 164)
    cursor = 324
    inputs = []
    outputs = []
    for sizes, destination in ((input_sizes, inputs), (output_sizes, outputs)):
        for count in sizes:
            assert count <= 10000000
            destination.append(struct.unpack_from("<" + "d" * count, payload, cursor))
            cursor += count * 8
    assert cursor == len(payload)
    result = []
    for stage in (0, 1):
        forces = inputs[:6] if stage == 0 else outputs[12:18]
        reference = outputs[18 + stage]
        for component in range(3):
            terms = [
                value * value
                for pair in zip(
                    forces[2 * component], forces[2 * component + 1], strict=True
                )
                for value in pair
            ]
            prefix = ordered(terms)
            assert bits(prefix[-1]) == bits(reference[component])
            result.append(terms)
    return result, {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dimensions": dimensions,
        "captured_raw_final_sums_exact": 6,
        "cases": len(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    generator = random.Random(982434)
    stats: dict[str, Any] = {
        "map_domain_checks": 0,
        "pair_checks": 0,
        "rounded_pair_checks": 0,
        "associativity_checks": 0,
    }
    exact = fractions.Fraction
    for _ in range(20000):
        field = generator.randint(1, 2046)
        term_field = max(0, min(2046, field + generator.randint(-80, 2)))
        term = number((term_field << 52) | generator.getrandbits(52))
        mapping = term_map(term, field)
        assert mapping is not None
        mantissa = LOWER_MANTISSA
        seed = number(field << 52)
        for parity in (0, 1):
            mantissa = LOWER_MANTISSA | (generator.getrandbits(52) & ~1) | parity
            seed = number((field << 52) | (mantissa - LOWER_MANTISSA))
            reference = rational_round(exact(seed) + exact(term))
            if math.isfinite(reference) and exponent(reference) == field:
                proposed = apply(mapping, mantissa)
                assert LOWER_MANTISSA <= proposed < UPPER_MANTISSA
                assert bits(reference) == (field << 52) | (proposed - LOWER_MANTISSA)
                stats["map_domain_checks"] += 1
        other_term = number((max(0, field - 10) << 52) | generator.getrandbits(52))
        other = term_map(other_term, field)
        assert other is not None
        first = rational_round(exact(seed) + exact(term))
        if math.isfinite(first) and exponent(first) == field:
            second = rational_round(exact(first) + exact(other_term))
            if math.isfinite(second) and exponent(second) == field:
                proposed = apply(compose(mapping, other), mantissa)
                assert bits(second) == (field << 52) | (proposed - LOWER_MANTISSA)
                stats["rounded_pair_checks"] += 1
        # Independently test map composition on an integer seed, including wrap.
        assert apply(compose(mapping, other), mantissa) == apply(
            other, apply(mapping, mantissa)
        )
        stats["pair_checks"] += 1
    for _ in range(20000):
        first, second, third = (
            (generator.getrandbits(64), generator.getrandbits(64)) for _ in range(3)
        )
        assert compose(compose(first, second), third) == compose(
            first, compose(second, third)
        )
        stats["associativity_checks"] += 1
    half = (0, 1)
    one = (1, 1)
    assert apply(compose(half, one), LOWER_MANTISSA) == LOWER_MANTISSA + 1
    assert apply(compose(one, half), LOWER_MANTISSA) == LOWER_MANTISSA + 2
    cases = [
        [0.0] * 64,
        [0.0] * 8 + [1.0] + [2.0**-53, 2.0**-52] * 64,
        [1.0] + [2.0**-53] * 127,
        [math.nextafter(2.0, 0.0), 2.0**-53, 2.0**-52] * 16,
        [float.fromhex("0x1p-1074")] * 64,
        [float.fromhex("0x1p-1022"), float.fromhex("0x1p-1074")] * 64,
        [float.fromhex("0x1.fffffffffffffp+1023")] * 2,
        [math.inf, 0.0],
        [math.nan, 1.0],
        [1.0, -0.5, 2.0],
    ]
    for _ in range(100):
        cases.append(
            [
                math.ldexp(generator.random(), generator.randint(-500, 100)) ** 2
                for _ in range(generator.randint(32, 1024))
            ]
        )
    captures = []
    unique_captures = {}
    for path in args.capture:
        additions, metadata = captured_cases(path)
        if metadata["sha256"] in unique_captures:
            metadata["duplicate_of"] = unique_captures[metadata["sha256"]]
            metadata["included_cases"] = 0
        else:
            unique_captures[metadata["sha256"]] = str(path)
            metadata["included_cases"] = len(additions)
            metadata["cpu_proposals_accepted"] = sum(
                propose(terms)[1] for terms in additions
            )
            cases.extend(additions)
        captures.append(metadata)
    accepted = 0
    corrupted_rejected = 0
    for terms in cases:
        reference = ordered(terms)
        proposal, valid = propose(terms)
        if valid:
            assert [bits(value) for value in proposal] == [
                bits(value) for value in reference
            ]
            accepted += 1
        final = proposal if valid else reference
        assert [bits(value) for value in final] == [bits(value) for value in reference]
        if reference and all(math.isfinite(value) for value in reference):
            corrupt = reference.copy()
            corrupt[len(corrupt) // 2] = math.nextafter(
                corrupt[len(corrupt) // 2], math.inf
            )
            assert not certificate(terms, corrupt)
            corrupted_rejected += 1
    with (args.out / "cases.bin").open("wb") as output:
        output.write(b"VMECLAT1")
        output.write(struct.pack("<Q", len(cases)))
        for terms in cases:
            output.write(struct.pack("<Q", len(terms)))
            output.write(struct.pack("<" + "d" * len(terms), *terms))
            output.write(struct.pack("<" + "d" * len(terms), *ordered(terms)))
    stats.update(
        prefix_cases=len(cases),
        accepted_proposals=accepted,
        ordered_fallback_cases=len(cases) - accepted,
        corrupted_prefixes_rejected=corrupted_rejected,
        cpu_only=True,
        gpu_executed=False,
        captures=captures,
        unique_capture_inputs=len(unique_captures),
        scope="Exact-rational constant-binade proof and CPU proposal tests; no speed or novelty claim",
    )
    (args.out / "cpu-proof.json").write_text(json.dumps(stats, indent=2) + "\n")
    sys.stdout.write(json.dumps(stats) + "\n")


if __name__ == "__main__":
    main()
