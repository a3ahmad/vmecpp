"""Distribute independent angular force slices across the persistent radial team."""

from __future__ import annotations

import argparse
from pathlib import Path


def function_bounds(text: str, signature: str) -> tuple[int, int]:
    start = text.index("{", text.index(signature)) + 1
    position = start
    depth = 1
    while depth:
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
        position += 1
    return start, position - 1


def parallel_rows(body: str, loop: str, first: str, last: str, variable: str) -> str:
    """Wrap only a named, reviewed loop whose output rows are independent."""
    while loop in body:
        start = body.index(loop)
        begin = start + len(loop)
        position = begin
        depth = 1
        while depth:
            if body[position] == "{":
                depth += 1
            elif body[position] == "}":
                depth -= 1
            position += 1
        body = (
            body[:start]
            + f"FourierExecutor().Run({first}, {last}, [&, this](int {variable}) {{"
            + body[begin : position - 1]
            + "});"
            + body[position:]
        )
    return body


def angular_slices(body: str) -> str:
    loop = "for (int kl = 0; kl < s_.nZnT; ++kl)"
    if loop not in body:
        message = "Expected independent angular work"
        raise ValueError(message)
    body = body.replace(loop, "for (int kl = slice_begin; kl < slice_end; ++kl)")
    return (
        """
  const int slices = DeterministicFourierThreads();
  FourierExecutor().Run(0, slices, [&, this](int slice) {
    const int slice_begin = s_.nZnT * slice / slices;
    const int slice_end = s_.nZnT * (slice + 1) / slices;
"""
        + body
        + "\n  });\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source / (
        "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    )
    text = path.read_text()
    if "RadialExecutor& FourierExecutor()" not in text or "slice_begin" in text:
        parser.error("requires the persistent Fourier experiment, applied once")
    start, end = function_bounds(text, "void IdealMhdModel::computeMHDForces()")
    body = text[start:end]
    loop = "for (int kl = 0; kl < s_.nZnT; ++kl)"
    if body.count(loop) < 10:
        parser.error("unexpected force loop layout")
    body = body.replace(loop, "for (int kl = slice_begin; kl < slice_end; ++kl)")
    names = ("P", "rup", "zup", "rsp", "zsp", "taup", "gbubu", "gbubv", "gbvbv")
    for name in names:
        fill = f"absl::c_fill_n(m_ls_.{name}_i, s_.nZnT, 0);"
        shift = f"m_ls_.{name}_i = {name}_o;"
        if body.count(fill) != 1 or body.count(shift) != 1:
            parser.error("unexpected shared scratch update")
        body = body.replace(
            fill,
            f"std::fill(m_ls_.{name}_i.begin() + slice_begin, m_ls_.{name}_i.begin() + slice_end, 0);",
        )
        body = body.replace(
            shift,
            f"std::copy({name}_o.begin() + slice_begin, {name}_o.begin() + slice_end, m_ls_.{name}_i.begin() + slice_begin);",
        )
    wrapper = """
  const int slices = DeterministicFourierThreads();
  FourierExecutor().Run(0, slices, [&, this](int slice) {
    const int slice_begin = s_.nZnT * slice / slices;
    const int slice_end = s_.nZnT * (slice + 1) / slices;
"""
    text = text[:start] + wrapper + body + "\n  });\n" + text[end:]
    signature = "void IdealMhdModel::assembleTotalForces()"
    start, end = function_bounds(text, signature)
    body = text[start:end]
    loop = "  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {"
    if body.count(loop) != 1:
        parser.error("unexpected force assembly loop")
    body = body.replace(
        loop,
        "  FourierExecutor().Run(r_.nsMinF, r_.nsMaxF, [&](int jF) {",
        1,
    )
    position = body.rfind("}")
    body = body[:position] + "});" + body[position + 1 :]
    text = text[:start] + body + text[end:]
    # Geometry metrics carry radial history independently for each angular point.
    start, end = function_bounds(text, "void IdealMhdModel::computeMetricElements()")
    text = text[:start] + angular_slices(text[start:end]) + text[end:]
    # The initial B transform carries radial history. Current-profile angular
    # reductions stay serial within each independently scheduled radial row.
    start, end = function_bounds(text, "void IdealMhdModel::computeBContra()")
    body = text[start:end]
    split = body.index("  if (ncurr == 1)")
    initial, remaining = body[:split], body[split:]
    remaining = parallel_rows(
        remaining,
        "for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {",
        "r_.nsMinH",
        "r_.nsMaxH",
        "jH",
    )
    text = text[:start] + angular_slices(initial) + remaining + text[end:]
    for signature in (
        "void IdealMhdModel::computeBCo()",
        "void IdealMhdModel::updateDifferentialVolume()",
    ):
        start, end = function_bounds(text, signature)
        body = parallel_rows(
            text[start:end],
            "for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {",
            "r_.nsMinH",
            "r_.nsMaxH",
            "jH",
        )
        text = text[:start] + body + text[end:]
    signature = "void IdealMhdModel::geometryFromFourier("
    start, end = function_bounds(text, signature)
    body = parallel_rows(
        text[start:end],
        "for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {",
        "r_.nsMinF",
        "r_.nsMaxFIncludingLcfs",
        "jF",
    )
    text = text[:start] + body + text[end:]
    text.encode("ascii")
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
