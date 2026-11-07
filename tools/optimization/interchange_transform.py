"""Create an isolated transform locality experiment after baseline validation."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--baseline-build-log", type=Path, required=True)
    args = parser.parse_args()
    if "Build completed successfully" not in args.baseline_build_log.read_text():
        parser.error("a successful baseline Bazel build is required")
    path = args.source / "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    text = path.read_text()
    start = text.index("void vmecpp::FourierToReal3DSymmFastPoloidal(")
    end = text.index("// Implemented as a free function", start + 1)
    original = text[start:end]
    inner = "      for (int k = 0; k < s.nZeta; ++k) {\n"
    outer = "  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {\n"
    if original.count(inner) != 1 or original.count(outer) != 1:
        parser.error("unexpected transform loop layout")
    changed = original.replace(inner, "", 1).replace(
        outer, outer + "    for (int k = 0; k < s.nZeta; ++k) {\n", 1
    )
    changed = changed.replace(
        "      }  // k\n    }  // m", "      }  // m\n    }  // k", 1
    )
    # Each output retains its original ascending m/n accumulation order.
    # Only independent toroidal output slices move outside the m loop.
    updated = text[:start] + changed + text[end:]
    updated.encode("ascii")
    path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
