"""Build/test a diagnostic CPU log observer; never replaces CPU libm arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

RECORD = struct.Struct("<8Q256s128s")
MAGIC = 0x564D45434C4F4731
DRIVER = r"""
#include <dlfcn.h>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cinttypes>
#include <cstdio>
#include <cstring>

int main() {
  using Function = double (*)(double);
  const auto function = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, "log"));
  if (function == nullptr) return 2;
  constexpr std::array<std::uint64_t, 16> kInputs{
      0x3ff0000000000000, 0x4000000000000000, 0x3fe0000000000000,
      0x3ff0000000000001, 0x3fefffffffffffff, 0x0000000000000000,
      0x8000000000000000, 0xbff0000000000000, 0x7ff0000000000000,
      0x7ff8000000001234, 0x7ff0000000001234, 0x0010000000000000,
      0x7fefffffffffffff, 0x0000000000000001, 0x3ff029fe55500000,
      0x3fef0123abc40000};
  constexpr std::array kRounding{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD,
                                FE_TOWARDZERO};
  for (const int rounding : kRounding) {
    if (std::fesetround(rounding) != 0) return 3;
    for (const auto bits : kInputs) {
      double input;
      std::memcpy(&input, &bits, sizeof(input));
      std::feclearexcept(FE_ALL_EXCEPT);
      errno = 13;
      const double output = function(input);
      const int error = errno;
      const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
      std::uint64_t result;
      std::memcpy(&result, &output, sizeof(output));
      std::printf("%016" PRIx64 " %016" PRIx64 " %d %d %d\n", bits,
                  result, error, exceptions, std::fegetround());
    }
  }
  return 0;
}
"""


def read_records(path: Path) -> list[dict]:
    """Reject incomplete traces instead of treating partial data as observations."""
    data = path.read_bytes()
    if not data or len(data) % RECORD.size:
        message = "Empty or incomplete CPU log trace"
        raise ValueError(message)
    result = []
    for values in RECORD.iter_unpack(data):
        if values[0] != MAGIC:
            message = "Invalid CPU log record magic"
            raise ValueError(message)
        result.append(
            dict(
                zip(
                    (
                        "magic",
                        "sequence",
                        "thread_id",
                        "caller_offset",
                        "input_bits",
                        "output_bits",
                        "errno",
                        "exceptions",
                    ),
                    values[:8],
                    strict=True,
                ),
                module=values[8].split(b"\0", 1)[0].decode(),
                symbol=values[9].split(b"\0", 1)[0].decode(),
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix(".cc").resolve()
    copied = out / source.name
    copied.write_bytes(source.read_bytes())
    driver_source = out / "driver.cc"
    driver_source.write_text(DRIVER, encoding="ascii")
    library = out / "observer.so"
    driver = out / "driver"
    commands = [
        [
            args.cxx,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-frounding-math",
            str(copied),
            "-ldl",
            "-lm",
            "-o",
            str(library),
        ],
        [
            args.cxx,
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-frounding-math",
            "-rdynamic",
            str(driver_source),
            "-Wl,--no-as-needed",
            "-ldl",
            "-lm",
            "-o",
            str(driver),
        ],
    ]
    for index, command in enumerate(commands):
        completed = subprocess.run(
            command, capture_output=True, timeout=120, check=False
        )
        (out / f"build-{index}.stdout").write_bytes(completed.stdout)
        (out / f"build-{index}.stderr").write_bytes(completed.stderr)
        completed.check_returncode()
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment.pop("VMECPP_CPU_LOG_OBSERVER_PREFIX", None)
    baseline = subprocess.run(
        [str(driver)], env=environment, capture_output=True, check=True, timeout=30
    )
    environment["LD_PRELOAD"] = str(library)
    disabled = subprocess.run(
        [str(driver)], env=environment, capture_output=True, check=True, timeout=30
    )
    environment["VMECPP_CPU_LOG_OBSERVER_PREFIX"] = str(out / "trace")
    observed = subprocess.run(
        [str(driver)], env=environment, capture_output=True, check=True, timeout=30
    )
    assert baseline.stdout == disabled.stdout == observed.stdout
    traces = list(out.glob("trace.*.bin"))
    assert len(traces) == 1
    records = read_records(traces[0])
    lines = observed.stdout.decode("ascii").splitlines()
    assert len(lines) == len(records) == 64
    for index, (line, record) in enumerate(zip(lines, records, strict=True)):
        argument, result, error, exceptions, _rounding = line.split()
        assert record["sequence"] == index
        assert record["input_bits"] == int(argument, 16)
        assert record["output_bits"] == int(result, 16)
        assert record["errno"] == int(error)
        assert record["exceptions"] == int(exceptions)
        assert record["symbol"] == "main"
    bad_trace = out / "truncated.bin"
    bad_trace.write_bytes(traces[0].read_bytes()[:-1])
    try:
        read_records(bad_trace)
    except ValueError:
        pass
    else:
        message = "Truncated trace accepted"
        raise AssertionError(message)
    environment["VMECPP_CPU_LOG_OBSERVER_PREFIX"] = str(out / "missing" / "trace")
    failure = subprocess.run(
        [str(driver)], env=environment, capture_output=True, timeout=30, check=False
    )
    assert failure.returncode == 122
    (out / "baseline.stdout").write_bytes(baseline.stdout)
    (out / "observed.stdout").write_bytes(observed.stdout)
    summary = {
        "purpose": "CPU observer transparency, not solver or GPU qualification",
        "status": "PASS",
        "cases": 64,
        "rounding_modes": 4,
        "checks": ["output bits", "errno", "FP exceptions", "rounding mode"],
        "disabled_transparent": True,
        "truncated_trace_rejected": True,
        "unwritable_output_exit_status": failure.returncode,
        "commands": commands,
        "source_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "solver_executed": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
