"""CPU proof of full-prefix certification and original Thomas fallback."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path


def bits(value: float) -> bytes:
    return struct.pack("<d", value)


class Operators:
    def __init__(self, path: Path):
        library = ctypes.CDLL(str(path))
        self.divide = library.CpuReferenceDivide
        self.fma = library.CpuReferenceFma
        self.numerator = library.CpuReferenceNumerator
        self.multiply = library.CpuReferenceMultiply
        self.add = library.CpuReferenceAdd
        for function, count in [
            (self.divide, 2),
            (self.fma, 3),
            (self.numerator, 3),
            (self.multiply, 2),
            (self.add, 2),
        ]:
            function.argtypes = [ctypes.c_double] * count
            function.restype = ctypes.c_double
        assert library.CpuReferenceRoundingMode() == 0


def forward(rhs, lower, denominators, ops, propose=False):
    result = []
    for index, value in enumerate(rhs):
        numerator = (
            value if index == 0 else ops.numerator(value, result[-1], lower[index])
        )
        if propose:
            inverse = ops.divide(1.0, denominators[index])
            estimate = ops.multiply(numerator, inverse)
            residual = ops.fma(-estimate, denominators[index], numerator)
            result.append(ops.fma(residual, inverse, estimate))
        else:
            result.append(ops.divide(numerator, denominators[index]))
    return result


def failures(rhs, lower, denominators, proposal, ops):
    result = []
    for index, value in enumerate(rhs):
        numerator = (
            value
            if index == 0
            else ops.numerator(value, proposal[index - 1], lower[index])
        )
        reference = ops.divide(numerator, denominators[index])
        if bits(reference) != bits(proposal[index]):
            result.append(index)
    return result


def backward(prefix, upper, ops):
    result = list(prefix)
    for index in range(len(result) - 2, -1, -1):
        result[index] = ops.numerator(result[index], upper[index], result[index + 1])
    return result


def captured(path: Path):
    data = path.read_bytes()
    assert data[:8] == b"VMECGIT1"
    dimensions = struct.unpack_from("<5i", data, 8)
    sizes = struct.unpack_from("<17Q", data, 28)
    output_sizes = struct.unpack_from("<20Q", data, 164)
    offset = 324
    inputs, outputs = [], []
    for lengths, target in [(sizes, inputs), (output_sizes, outputs)]:
        for count in lengths:
            assert count <= 10000000
            target.append(
                list(struct.unpack_from("<" + str(count) + "d", data, offset))
            )
            offset += 8 * count
    assert offset == len(data)
    return dimensions, inputs, outputs, hashlib.sha256(data).hexdigest()


def require_arrays(actual, expected):
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected, strict=False):
        assert len(first) == len(second)
        assert all(bits(x) == bits(y) for x, y in zip(first, second, strict=False))


def residuals(forces, ops):
    result = []
    for component in range(3):
        value = 0.0
        for first, second in zip(
            forces[2 * component], forces[2 * component + 1], strict=False
        ):
            value = ops.add(value, ops.multiply(first, first))
            value = ops.add(value, ops.multiply(second, second))
        result.append(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--operators", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    ops = Operators(args.operators)
    dimensions, inputs, expected, source_hash = captured(args.capture)
    ns, mpol, ntor, ns_max, ns_incl = dimensions
    assert ns_max == ns - 1
    assert ns_incl == ns
    modes, n_count, end = mpol * (ntor + 1), ntor + 1, ns - 1
    immutable_input = hashlib.sha256(
        b"".join(bits(v) for a in inputs for v in a)
    ).hexdigest()
    forces = [list(values) for values in inputs[:6]]
    require_arrays([residuals(forces, ops)], [expected[18]])
    for j in range(ns_max):
        ard, brd, azd, bzd = (inputs[i][2 * j + 1] for i in range(6, 10))
        denominator = ops.add(ops.add(ops.add(ard, brd), azd), bzd)
        scale_r = ops.divide(ops.add(ard, brd), denominator)
        scale_z = ops.divide(ops.add(azd, bzd), denominator)
        for n in range(n_count):
            index = (j * mpol + 1) * n_count + n
            forces[1][index] = ops.multiply(forces[1][index], scale_r)
            forces[3][index] = ops.multiply(forces[3][index], scale_z)
    require_arrays(forces, expected[:6])
    factored, denominators = [], []
    for component in range(2):
        upper, diagonal, lower = (inputs[i + component] for i in (10, 12, 14))
        factor, denominator = [0.0] * len(upper), [0.0] * len(diagonal)
        for mode in range(modes):
            first = int(mode // n_count > 0)
            index = first * modes + mode
            denominator[index] = diagonal[index]
            factor[index] = ops.divide(upper[index], diagonal[index])
            for j in range(first + 1, end - 1):
                index = j * modes + mode
                denominator[index] = ops.numerator(
                    diagonal[index], factor[index - modes], lower[index]
                )
                factor[index] = ops.divide(upper[index], denominator[index])
            index = (end - 1) * modes + mode
            denominator[index] = ops.numerator(
                diagonal[index], factor[index - modes], lower[index]
            )
        factored.append(factor)
        denominators.append(denominator)
    prefix_outputs = [[0.0] * len(forces[0]) for _ in range(4)]
    columns = []
    poison_tests = []
    for channel in range(4):
        component = channel // 2
        for mode in range(modes):
            first = int(mode // n_count > 0)
            indices = [j * modes + mode for j in range(first, end)]
            rhs = tuple(forces[channel][i] for i in indices)
            lower = tuple(inputs[14 + component][i] for i in indices)
            diagonal = tuple(denominators[component][i] for i in indices)
            upper = tuple(factored[component][i] for i in indices)
            reference = forward(rhs, lower, diagonal, ops)
            proposal = forward(rhs, lower, diagonal, ops, propose=True)
            failed = failures(rhs, lower, diagonal, proposal, ops)
            certified = reference if failed else proposal
            assert all(
                bits(a) == bits(b) for a, b in zip(certified, reference, strict=False)
            )
            answer = backward(certified, upper, ops)
            for index, original, value in zip(indices, reference, answer, strict=False):
                prefix_outputs[channel][index] = original
                forces[channel][index] = value
            columns.append(
                {
                    "channel": channel,
                    "mode": mode,
                    "rows": len(indices),
                    "accepted": not failed,
                    "failed_rows": failed,
                }
            )
            if channel == mode == 0:
                for label, replace in [
                    (
                        "one_ULP",
                        lambda x: struct.unpack(
                            "<d",
                            (int.from_bytes(bits(x), "little") ^ 1).to_bytes(
                                8, "little"
                            ),
                        )[0],
                    ),
                    ("NaN", lambda _: math.nan),
                ]:
                    poisoned = list(proposal)
                    poisoned[len(poisoned) // 2] = replace(poisoned[len(poisoned) // 2])
                    assert bits(poisoned[-1]) == bits(proposal[-1])
                    bad = failures(rhs, lower, diagonal, poisoned, ops)
                    assert bad
                    restored = backward(forward(rhs, lower, diagonal, ops), upper, ops)
                    assert all(
                        bits(a) == bits(b)
                        for a, b in zip(restored, answer, strict=False)
                    )
                    poison_tests.append(
                        {
                            "case": label,
                            "terminal_prefix_unchanged": True,
                            "failed_rows": bad,
                            "whole_column_fallback_exact": True,
                        }
                    )
    require_arrays(forces, expected[6:12])
    for channel in (4, 5):
        for i, value in enumerate(forces[channel]):
            forces[channel][i] = ops.multiply(value, inputs[16][i])
    require_arrays(forces, expected[12:18])
    require_arrays([residuals(forces, ops)], [expected[19]])
    assert (
        immutable_input
        == hashlib.sha256(b"".join(bits(v) for a in inputs for v in a)).hexdigest()
    )
    generator = random.Random(49232)
    adversarial = []
    fixtures = [
        ("signed_zero", [-0.0, 0.0, -0.0, 0.0], [1.0] * 4),
        (
            "subnormal",
            [float.fromhex("0x1p-1074"), -float.fromhex("0x1p-1074"), 0.0, 1.0],
            [1.0] * 4,
        ),
        ("division_zero", [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ("nonfinite", [math.inf, 1.0, math.nan, -math.inf], [1.0] * 4),
        ("overflow", [1e308] * 4, [1e-308] * 4),
    ]
    for index in range(100):
        fixtures.append(
            (
                f"random{index}",
                [generator.uniform(-1, 1) for _ in range(99)],
                [generator.uniform(0.5, 2) for _ in range(99)],
            )
        )
    for label, rhs, diagonal in fixtures:
        lower = [0.125] * len(rhs)
        reference = forward(tuple(rhs), lower, diagonal, ops)
        proposal = forward(tuple(rhs), lower, diagonal, ops, propose=True)
        bad = failures(rhs, lower, diagonal, proposal, ops)
        selected = reference if bad else proposal
        assert all(
            bits(a) == bits(b) for a, b in zip(selected, reference, strict=False)
        )
        adversarial.append(
            {
                "case": label,
                "rows": len(rhs),
                "accepted": not bad,
                "fallback_exact": True,
            }
        )
    report = {
        "scope": "CPU actual-capture prefix-induction/fallback proof only; no GPU execution/performance claim",
        "capture_sha256": source_hash,
        "dimensions": dimensions,
        "columns": len(columns),
        "active_row_checks": sum(x["rows"] for x in columns),
        "accepted_columns": sum(x["accepted"] for x in columns),
        "fallback_columns": sum(not x["accepted"] for x in columns),
        "all18captured_force_arrays_and_both_residual_triples_bitwise_exact": True,
        "original_input_immutable": True,
        "poison_witnesses": poison_tests,
        "adversarial_column_cases": adversarial,
        "all_columns": columns,
        "induction": "First row checks original b/d bits. If all predecessors match, every row numerator uses original RN(b-RN(y_prev*lower)); exact RNdivision bit-check proves next prefix. ANYfailedrow restores entirecolumn from immutableb before original ordered backsub. Equation residual/last-prefix agreement is insufficient.",
        "GPU_special_NaN_payload_scope_unproved": True,
    }
    (args.out / "proof.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii"
    )
    with (args.out / "prefix-reference.bin").open("wb") as stream:
        stream.write(b"THPFCPU1")
        stream.write(struct.pack("<5iQ", *dimensions, len(prefix_outputs[0])))
        for values in prefix_outputs:
            stream.write(struct.pack("<" + str(len(values)) + "d", *values))
    sys.stdout.write(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["all_columns", "adversarial_column_cases"]
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
