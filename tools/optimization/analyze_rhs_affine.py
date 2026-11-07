"""CPU-only affine proposals, literal-RN certificates and RHS suffix repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from collections import Counter
from pathlib import Path


def bits(value: float) -> bytes:
    return struct.pack("<d", value)


def read_capture(path: Path) -> tuple[list[int], list[list[float]], list[list[float]]]:
    data = path.read_bytes()
    if data[:8] != b"VMECGIT1":
        msg = "Capture magic mismatch"
        raise ValueError(msg)
    dimensions = list(struct.unpack_from("<5i", data, 8))
    sizes = struct.unpack_from("<37Q", data, 28)
    arrays = []
    offset = 28 + 37 * 8
    for size in sizes:
        arrays.append(list(struct.unpack_from(f"<{size}d", data, offset)))
        offset += size * 8
    if offset != len(data):
        msg = "Capture length mismatch"
        raise ValueError(msg)
    return dimensions, arrays[:17], arrays[17:]


def affine_guess(a: list[float], b: list[float]) -> list[float]:
    """Hillis-Steele inclusive affine scan; reassociation is proposal-only."""
    a, b = a.copy(), b.copy()
    stride = 1
    while stride < len(b):
        old_a, old_b = a, b
        a, b = old_a.copy(), old_b.copy()
        for index in range(stride, len(b)):
            a[index] = old_a[index] * old_a[index - stride]
            product = old_a[index] * old_b[index - stride]
            b[index] = product + old_b[index]
        stride *= 2
    return b


def recurrence(
    b: list[float],
    multiplier: list[float],
    denominator: list[float] | None,
    index: int,
    previous: float,
) -> float:
    if index == 0:
        return b[0] / denominator[0] if denominator is not None else b[0]
    product = previous * multiplier[index]
    difference = b[index] - product
    return difference / denominator[index] if denominator is not None else difference


def study_phase(
    b: list[float],
    multiplier: list[float],
    denominator: list[float] | None,
) -> tuple[dict, list[float]]:
    reference = []
    for index in range(len(b)):
        reference.append(
            recurrence(
                b, multiplier, denominator, index, reference[-1] if index else 0.0
            )
        )
    if denominator is None:
        a = [-value for value in multiplier]
        intercept = b.copy()
    else:
        a = [
            -value / divisor
            for value, divisor in zip(multiplier, denominator, strict=True)
        ]
        intercept = [
            value / divisor for value, divisor in zip(b, denominator, strict=True)
        ]
    a[0] = 0.0
    guess = affine_guess(a, intercept)
    rounds = []
    first_complete = None
    for round_index in range(33):
        literal = [
            recurrence(
                b, multiplier, denominator, index, guess[index - 1] if index else 0.0
            )
            for index in range(len(b))
        ]
        valid = [
            math.isfinite(actual)
            and math.isfinite(wanted)
            and bits(actual) == bits(wanted)
            for actual, wanted in zip(guess, literal, strict=True)
        ]
        first_bad = next(
            (index for index, value in enumerate(valid) if not value), len(b)
        )
        if first_bad == len(b) and first_complete is None:
            first_complete = round_index
        # Every certified prefix must already equal the literal ordered answer.
        if any(
            bits(guess[index]) != bits(reference[index]) for index in range(first_bad)
        ):
            msg = "Certified prefix differs from ordered recurrence"
            raise AssertionError(msg)
        repaired = guess[:first_bad]
        for index in range(first_bad, len(b)):
            repaired.append(
                recurrence(
                    b, multiplier, denominator, index, repaired[-1] if index else 0.0
                )
            )
        if any(bits(x) != bits(y) for x, y in zip(repaired, reference, strict=True)):
            msg = "Ordered suffix repair differs"
            raise AssertionError(msg)
        rounds.append(
            {
                "round": round_index,
                "first_mismatch_traversal_index": first_bad,
                "ordered_repair_rows": len(b) - first_bad,
                "all_prefixes_certified": first_bad == len(b),
            }
        )
        guess = literal  # Synchronous correction uses only the previous-round guess.
    return {
        "rows": len(b),
        "first_complete_round_within32": first_complete,
        "rounds": rounds,
        "all_ordered_repairs_bitwise_exact": True,
    }, reference


def summarize(records: list[dict]) -> dict:
    complete = [row["first_complete_round_within32"] for row in records]
    per_round = []
    for index in range(33):
        tails = [row["rounds"][index]["ordered_repair_rows"] for row in records]
        per_round.append(
            {
                "round": index,
                "uncertified_chains": sum(value > 0 for value in tails),
                "total_repair_rows": sum(tails),
                "mean_repair_rows": statistics.mean(tails),
                "median_repair_rows": statistics.median(tails),
                "worst_repair_rows": max(tails),
            }
        )
    return {
        "chains": len(records),
        "complete_round_histogram": dict(Counter(map(str, complete))),
        "not_complete_by32": sum(value is None for value in complete),
        "per_round": per_round,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    unique = {}
    for path in args.capture:
        unique.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(
            str(path)
        )
    reports = []
    for digest, paths in unique.items():
        dimensions, inputs, outputs = read_capture(Path(paths[0]))
        ns, mpol, ntor, _, _ = dimensions
        modes, end = mpol * (ntor + 1), ns - 1
        actual = [channel.copy() for channel in outputs[:6]]
        records = {"forward": [], "backward": []}
        data_values = []
        for component in range(2):
            upper, diagonal, lower = (
                inputs[10 + component],
                inputs[12 + component],
                inputs[14 + component],
            )
            for mode in range(modes):
                first = int(mode // (ntor + 1) > 0)
                rows = list(range(first, end))
                a, denominators = [], []
                for local, row in enumerate(rows):
                    index = row * modes + mode
                    divisor = (
                        diagonal[index]
                        if local == 0
                        else diagonal[index] - a[-1] * lower[index]
                    )
                    denominators.append(divisor)
                    # Original final-row factorization does not calculate upper/denominator.
                    a.append(upper[index] / divisor if row < end - 1 else 0.0)
                for basis in range(2):
                    channel = 2 * component + basis
                    original_b = [outputs[channel][row * modes + mode] for row in rows]
                    multipliers = [lower[row * modes + mode] for row in rows]
                    forward, exact_forward = study_phase(
                        original_b, multipliers, denominators
                    )
                    backward, reverse_answer = study_phase(
                        exact_forward[::-1], a[::-1], None
                    )
                    identity = {
                        "component": component,
                        "basis": basis,
                        "mode": mode,
                        "first_row": first,
                    }
                    records["forward"].append(identity | forward)
                    records["backward"].append(identity | backward)
                    for row in range(first):
                        actual[channel][row * modes + mode] = 0.0
                    for row, value in zip(rows, reverse_answer[::-1], strict=True):
                        actual[channel][row * modes + mode] = value
                    data_values.extend(original_b + multipliers + denominators + a)
        mismatches = sum(
            bits(value) != bits(wanted)
            for channel, expected in zip(actual, outputs[6:12], strict=True)
            for value, wanted in zip(channel, expected, strict=True)
        )
        if mismatches:
            msg = f"Original CPU RZ capture mismatches: {mismatches}"
            raise AssertionError(msg)
        report = {
            "capture_sha256": digest,
            "paths": paths,
            "dimensions": dimensions,
            "original_all_six_force_capture_bit_mismatches": mismatches,
            "finite_data_values": sum(map(math.isfinite, data_values)),
            "nonfinite_data_values": sum(not math.isfinite(x) for x in data_values),
            "positive_zero_values": sum(
                x == 0.0 and bits(x) == bits(0.0) for x in data_values
            ),
            "negative_zero_values": sum(bits(x) == bits(-0.0) for x in data_values),
            "negative_nonzero_values": sum(x < 0.0 for x in data_values),
            "summary": {phase: summarize(rows) for phase, rows in records.items()},
        }
        (args.out / f"{digest}-chains.json").write_text(json.dumps(records) + "\n")
        reports.append(report)
    result = {
        "scope": "Executed CPU-only feasibility; no GPU timing or performance claim",
        "input_paths": len(args.capture),
        "unique_captures": len(unique),
        "proposal": "Hillis-Steele affine scan; reassociation affects untrusted proposals only",
        "certificate": "All finite original per-row recurrence prefixes compared by FP64 bits; signed zero distinguished",
        "correction": "Synchronous literal original recurrence from previous-round guess, rounds0..32",
        "repair": "Keep fully certified prefix, ordered original recurrence from first mismatch; backward traverses reversed rows",
        "reports": reports,
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
