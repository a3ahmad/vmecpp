"""Replace experimental nested OpenMP launches with a persistent radial team."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source / (
        "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    )
    text = path.read_text()
    if "DeterministicFourierThreads" not in text or "RadialExecutor" in text:
        parser.error("requires the OpenMP transform experiment, applied once")
    pragma_start = "#ifdef _OPENMP\n#pragma omp parallel for schedule(static)"
    for _ in range(3):
        start = text.index(pragma_start)
        end = text.index("#endif\n", start) + len("#endif\n")
        loop_end = text.index(" {", end)
        loop = text[end:loop_end]
        first = loop.split(" = ", 1)[1].split(";", 1)[0]
        last = loop.split("jF < ", 1)[1].split(";", 1)[0]
        body_start = loop_end + 2
        depth = 1
        body_end = body_start
        while depth:
            if text[body_end] == "{":
                depth += 1
            elif text[body_end] == "}":
                depth -= 1
            body_end += 1
        text = (
            text[:start]
            + f"  FourierExecutor().Run({first}, {last}, [&](int jF) {{"
            + text[body_start : body_end - 1]
            + "});"
            + text[body_end:]
        )
    helper = """
RadialExecutor& FourierExecutor() {
  static thread_local RadialExecutor executor(DeterministicFourierThreads());
  return executor;
}
"""
    marker = "// Set m_h.rCC_LCFS"
    text = text.replace(marker, helper + "\n" + marker, 1)
    text = text.replace(
        "#include <stdexcept>",
        '#include <stdexcept>\n#include "radial_executor.h"',
        1,
    )
    text.encode("ascii")
    path.write_text(text)
    shutil.copy2(Path(__file__).with_name("radial_executor.h"), path.parent)
    build = path.with_name("BUILD.bazel")
    build.write_text(
        build.read_text().replace(
            'hdrs = ["ideal_mhd_model.h", "dft_data.h"],',
            'hdrs = ["ideal_mhd_model.h", "dft_data.h", "radial_executor.h"],',
            1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
