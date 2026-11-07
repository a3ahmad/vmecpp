"""Render native samples and measured multigrid stages as a standalone figure."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    profile = json.loads(args.profile.read_text())
    frames = profile["shared"]["frames"]
    counts = Counter(
        frames[sample[-1]]["name"]
        for recording in profile["profiles"]
        for sample in recording["samples"]
    )
    total = sum(counts.values())
    top = counts.most_common(8)[::-1]
    stages = re.findall(
        r"VMECPP_STAGE ns=(\d+) initialize=([\d.e+-]+) solve=([\d.e+-]+)",
        args.stages.read_text(),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = [
        name.replace("vmecpp::IdealMhdModel::", "").replace("vmecpp::", "")
        for name, _ in top
    ]
    shares = [100 * count / total for _, count in top]
    axes[0].barh(names, shares, color="#345e8c")
    axes[0].set_xlabel("Exclusive leaf samples (%)")
    axes[0].set_title(f"Native sample window: {total:,} samples")
    axes[1].bar(
        [row[0] for row in stages], [float(row[2]) for row in stages], color="#4e9077"
    )
    axes[1].set_xlabel("Radial grid size ns")
    axes[1].set_ylabel("Equilibrium solve time (seconds)")
    axes[1].set_title("One instrumented official P3-seed/P1 solve")
    figure.suptitle("Measured diagnostics from separate runs; no speedup comparison")
    figure.savefig(args.out, dpi=180)
    figure.savefig(args.out.with_suffix(".pdf"))
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
