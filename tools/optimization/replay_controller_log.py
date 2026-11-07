"""Replay observed damping history; reject a replay that misses original bits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def value(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def bits(number: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", number))[0]


def coefficients(history: list[float], time_step: float) -> tuple[int, int]:
    """Preserve the original ordered sum, division and two-step dtau product."""
    total = 0.0
    for entry in history:
        total += entry
    otav = total / 10
    dtau = time_step * otav / 2.0
    return bits(1.0 / (1.0 + dtau)), bits(1.0 - dtau)


def replay(events: list[dict], originals: list[tuple], proposals: list[tuple]) -> dict:
    if len(originals) != len(proposals) or not events:
        message = "Missing or inconsistent replay inputs"
        raise ValueError(message)
    original_history: list[float] = []
    candidate_history: list[float] = []
    log_index = 0
    changed_steps = []
    changed_history_events = 0
    previous_fsq = None
    for index, event in enumerate(events):
        fac, b1, time_step_bits, fsq_bits = event["step_bits"]
        time_step, fsq = value(time_step_bits), value(fsq_bits)
        assert len(event["inv_tau_bits"]) == 10
        if event["iter2"] == event["iter1"]:
            original_history = [0.15 / time_step] * 10
            candidate_history = original_history.copy()
        else:
            assert event["iter2"] > event["iter1"]
            assert original_history
            original_history = original_history[1:] + original_history[:1]
            candidate_history = candidate_history[1:] + candidate_history[:1]
            original_numerator = candidate_numerator = 0.0
            if fsq != 0.0:
                argument, logarithm = originals[log_index]
                _candidate_logarithm, candidate_numerator_bits = proposals[log_index]
                assert previous_fsq is not None
                assert bits(fsq / previous_fsq) == argument
                original_numerator = min(abs(value(logarithm)), 0.15)
                candidate_numerator = value(candidate_numerator_bits)
                log_index += 1
            original_history[-1] = original_numerator / time_step
            candidate_history[-1] = candidate_numerator / time_step
        # The CPU oracle must reconstruct every original history and step bit
        # before its candidate differences can be interpreted.
        assert [bits(x) for x in original_history] == event["inv_tau_bits"], index
        assert coefficients(original_history, time_step) == (fac, b1), index
        candidate = coefficients(candidate_history, time_step)
        changed_history_events += [bits(x) for x in original_history] != [
            bits(x) for x in candidate_history
        ]
        if candidate != (fac, b1):
            changed_steps.append(
                {
                    "event": index,
                    "ns": event["ns"],
                    "iter1": event["iter1"],
                    "iter2": event["iter2"],
                    "original_fac_b1_bits": [fac, b1],
                    "candidate_fac_b1_bits": list(candidate),
                }
            )
        previous_fsq = fsq
    assert log_index == len(originals)
    return {
        "scope": "Fixed observed trajectory replay; not an autonomous GPU solver",
        "original_history_and_coefficients_exact": True,
        "step_events": len(events),
        "log_calls_consumed": log_index,
        "candidate_changed_history_events": changed_history_events,
        "candidate_changed_step_events": len(changed_steps),
        "candidate_step_differences": changed_steps,
        "limit": "After a changed step the observed trajectory is counterfactual; no downstream physics or metric conclusion",
    }


def screen_assumed_envelope(
    events: list[dict], proposals: list[tuple], radius: int
) -> dict:
    """Explore conditional acceptance; neither log bounds nor recovery are proven."""
    assert radius > 0
    lower: list[float] = []
    upper: list[float] = []
    cursor = accepted = ambiguous = streak = maximum_streak = 0
    for event in events:
        fac, b1, time_step_bits, fsq_bits = event["step_bits"]
        time_step = value(time_step_bits)
        assert time_step > 0
        actual = [value(x) for x in event["inv_tau_bits"]]
        if event["iter1"] == event["iter2"]:
            lower = [0.15 / time_step] * 10
            upper = lower.copy()
        else:
            assert lower
            lower = lower[1:] + lower[:1]
            upper = upper[1:] + upper[:1]
            lo = hi = 0.0
            if value(fsq_bits) != 0:
                left = right = value(proposals[cursor][0])
                cursor += 1
                for _ in range(radius):
                    left = math.nextafter(left, -math.inf)
                    right = math.nextafter(right, math.inf)
                lo = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
                hi = max(abs(left), abs(right))
                lo, hi = min(lo, 0.15), min(hi, 0.15)
            lower[-1], upper[-1] = lo / time_step, hi / time_step
        assert all(
            lo <= x <= hi for lo, x, hi in zip(lower, actual, upper, strict=True)
        )
        low_coefficients = coefficients(lower, time_step)
        high_coefficients = coefficients(upper, time_step)
        if low_coefficients == high_coefficients:
            assert low_coefficients == (fac, b1)
            accepted += 1
            streak += 1
            maximum_streak = max(maximum_streak, streak)
        else:
            ambiguous += 1
            # This uses observed CPU values as an oracle. It is not an
            # implementation of runtime CPU history reconstruction.
            lower = actual.copy()
            upper = actual.copy()
            streak = 0
    assert cursor == len(proposals)
    return {
        "scope": "Assumed envelope feasibility only; not a certified runtime path",
        "radius_representable_neighbors": radius,
        "accepted_step_coefficients": accepted,
        "ambiguous_steps_oracle_history_collapse": ambiguous,
        "accepted_percent": 100 * accepted / len(events),
        "longest_accepted_streak": maximum_streak,
        "limits": [
            "Observed enclosure is not proof of pinned CPU/CUDA log bounds",
            "Ambiguous steps use observed CPU histories; runtime recovery is not implemented",
            "Acceptance rates are not measured timing or speedup estimates",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gpu-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--assumed-log-radius",
        type=int,
        choices=(1, 2, 4, 8),
        action="append",
        default=[],
    )
    args = parser.parse_args()
    events = [json.loads(line) for line in args.controller.read_text().splitlines()]
    original = list(struct.iter_unpack("<QQ", args.corpus.read_bytes()))
    proposals = list(struct.iter_unpack("<QQ", args.gpu_output.read_bytes()))
    result = replay(events, original, proposals)
    result["assumed_interval_screens"] = [
        screen_assumed_envelope(events, proposals, radius)
        for radius in args.assumed_log_radius
    ]
    result["evidence_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.controller, args.corpus, args.gpu_output)
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
