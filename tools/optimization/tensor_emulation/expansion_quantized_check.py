"""Small exact CPU checks for signed RNE extraction; never imports CUDA."""

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path


def shift_even(value, shift):
    if not shift:
        return value
    quotient, remainder = divmod(abs(value), 1 << shift)
    half = 1 << (shift - 1)
    quotient += remainder > half or (remainder == half and quotient & 1)
    return -quotient if value < 0 else quotient


def extract(value, components, bits):
    original = value
    digits = []
    for component in range(components):
        shift = bits * (components - component - 1)
        digit = shift_even(value, shift)
        assert digit == round(Fraction(value, 1 << shift))
        assert abs(digit) <= 1 << (bits - 1)
        unsigned = (value - (digit << shift)) & ((1 << 64) - 1)
        value = unsigned if unsigned < 1 << 63 else unsigned - (1 << 64)
        digits.append(digit)
    assert value == 0
    assert (
        sum(
            digit << (bits * (components - index - 1))
            for index, digit in enumerate(digits)
        )
        == original
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rng = random.Random(5090135)
    count = 0
    for components, bits in ((4, 10), (6, 10), (6, 9), (8, 8), (4, 11), (6, 8)):
        width = components * bits
        limit = (1 << (width - 1)) - (1024 if width == 64 else 0)
        values = {0, 1, -1, limit, -limit, limit - 1, -limit + 1}
        for component in range(components - 1):
            shift = bits * (components - component - 1)
            for integer in (0, 1, 2, 3):
                for delta in (-1, 0, 1):
                    value = integer * (1 << shift) + (1 << (shift - 1)) + delta
                    if value <= limit:
                        values.update((value, -value))
        values.update(rng.randint(-limit, limit) for _ in range(64))
        for value in values:
            extract(value, components, bits)
            count += 1
        for value in (
            0.0,
            -0.0,
            0.25,
            -0.25,
            math.nextafter(0.5, 0.0),
            -math.nextafter(0.5, 0.0),
            2.0**-50,
        ):
            scaled = math.ldexp(value, width)
            assert abs(scaled) < 2**63
            extract(round(Fraction(scaled)), components, bits)
            count += 1
    result = {
        "CPU_checks_pass": count,
        "scope": "RNE signed shifts, modulo subtraction, balanced carry, exact integer reconstruction; not GPU or triangular equivalence",
        "wide_recurrence_fallback": [66, 72],
        "GPU_calls": 0,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))  # noqa: T201 - Explicit CPU proof receipt.


if __name__ == "__main__":
    main()
