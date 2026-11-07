"""Add opt-in diagnostic multigrid timers to the isolated 0.4.11 source copy."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--baseline-build-log", type=Path, required=True)
    args = parser.parse_args()
    if "Build completed successfully" not in args.baseline_build_log.read_text():
        parser.error(
            "a successful baseline Bazel build is required before instrumentation"
        )
    path = args.source / "src/vmecpp/cpp/vmecpp/vmec/vmec/vmec.cc"
    text = path.read_text()
    before = "      // initialize ns-dependent arrays"
    after = "      if (!reached_checkpoint.ok() || *reached_checkpoint == true) {"
    if text.count(before) != 1 or text.count(after) != 1:
        parser.error("source does not match the expected uninstrumented layout")
    text = text.replace(
        "#include <algorithm>",
        "#include <algorithm>\n#include <chrono>\n#include <cstdlib>",
        1,
    )
    text = text.replace(
        before,
        "      const auto stage_start = std::chrono::steady_clock::now();\n" + before,
        1,
    )
    text = text.replace(
        "      // *HERE* is the *ACTUAL* call to the equilibrium solver !",
        "      const auto solve_start = std::chrono::steady_clock::now();\n"
        "      // *HERE* is the *ACTUAL* call to the equilibrium solver !",
        1,
    )
    diagnostic = r"""
      if (std::getenv("VMECPP_PROFILE_STAGES") != nullptr) {
        const auto stage_end = std::chrono::steady_clock::now();
        const double initialize_seconds =
            std::chrono::duration<double>(solve_start - stage_start).count();
        const double solve_seconds =
            std::chrono::duration<double>(stage_end - solve_start).count();
        std::fprintf(stderr,
                     "VMECPP_STAGE ns=%d initialize=%.9g solve=%.9g "
                     "iter2=%d retries=%d fsqr=%.17g fsqz=%.17g fsql=%.17g "
                     "status=%s\n",
                     fc_.nsval, initialize_seconds, solve_seconds, iter2_,
                     num_eqsolve_retries_, fc_.fsqr, fc_.fsqz, fc_.fsql,
                     VmecStatusAsString(status_).c_str());
      }
"""
    text = text.replace(after, diagnostic + after, 1)
    text.encode("ascii")
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
