"""Small CPU-only independent-reference edge checks; no GPU or dataset audit."""

import hashlib
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np


def main():
    source = Path(__file__).with_name("analyze_suite.py")
    spec = importlib.util.spec_from_file_location("analysis_cpu_checks", source)
    assert spec is not None
    assert spec.loader is not None
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    warnings.simplefilter("error", RuntimeWarning)
    cases = (
        ("cancellation", [2.0**53, 1.0, -(2.0**53)], [1.0] * 3, 0.0, 1.0),
        ("tie_even_lower", [1.0, 2.0**-53], [1.0, 1.0], 1.0, 1.0),
        ("tie_even_upper", [1.0, 3.0 * 2.0**-53], [1.0, 1.0], 1.0, 1.0 + 2.0**-51),
        (
            "normal_subnormal_boundary",
            [2.0**-1022, 2.0**-1074],
            [1.0, 1.0],
            2.0**-1022,
            math.nextafter(2.0**-1022, math.inf),
        ),
        ("half_min_subnormal", [2.0**-1074], [0.5], 0.0, 0.0),
    )
    for name, left, right, ordered, rounded in cases:
        dots, comparisons = analysis.rational_checks(
            name,
            np.array([left]),
            np.array(right).reshape(-1, 1),
            np.array([ordered]),
            [
                {
                    "array": np.array([rounded]),
                    "version": "CPU_FIXTURE",
                    "repetition": 0,
                    "mode": "CPU_ONLY",
                }
            ],
            32,
        )
        assert dots[0]["rounded_exact_GEMM_hex"] == rounded.hex()
        assert comparisons[0]["correctly_rounded_GEMM_bits"]
        if name == "cancellation":
            assert dots[0]["exact_sum_numerator"] == "1"
        if name == "half_min_subnormal":
            assert int(comparisons[0]["error_exact_denominator"]) == 2**1075
    error, _, _ = analysis.errors(np.array([-0.0]), np.array([0.0]))
    assert error["different_ordered_bits"] == 1
    assert error["signed_zero_differences"] == 1
    assert error["max_ULP_zero_sign_separate"] == 0
    assert analysis.mode_order(
        {"native", "fixed-40", "fixed-80", "fixed-55", "fixed-48", "fixed-64"}
    ) == (
        "native",
        "fixed-80",
        "fixed-64",
        "fixed-55",
        "fixed-48",
        "fixed-40",
    )
    sys.stdout.write(
        json.dumps(
            {
                "CPU_only": True,
                "GPU_calls": 0,
                "cases_passed": 7,
                "analyzer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
