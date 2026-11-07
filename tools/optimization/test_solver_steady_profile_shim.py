"""CPU-only LD_PRELOAD tests with a fake CUDA profiler and fake solver backend."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA = Path("/usr/local/cuda-12.9")
MOCK_SOURCE = r"""
#include <cuda_profiler_api.h>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include "gpu_solver.h"
namespace {
std::array<int, 8> tokens{};
int created = 0;
int run_count = 0;
int start_count = 0;
int start_run = -1;
bool active = false;
bool bad = false;
bool throwing = false;
}
extern "C" void MockBad(bool value) { bad = value; }
extern "C" void MockThrow(bool value) { throwing = value; }
extern "C" int MockStarts() { return start_count; }
extern "C" int MockStartRun() { return start_run; }
extern "C" int MockRuns() { return run_count; }
extern "C" cudaError_t cudaProfilerStart() {
  if (active) throw std::runtime_error("Profiler started inside original Run");
  ++start_count;
  start_run = run_count;
  const char* status = std::getenv("MOCK_PROFILER_STATUS");
  return status ? static_cast<cudaError_t>(std::atoi(status)) : cudaSuccess;
}
extern "C" void* CreateGpuSolver(const GpuSolverSetup&) {
  if (throwing) throw std::runtime_error("Original create exception");
  return &tokens.at(created++);
}
extern "C" void DestroyGpuSolver(void*) {
  if (throwing) throw std::runtime_error("Original destroy exception");
}
extern "C" void RunGpuSolver(void*, const GpuSolverControl&,
                             GpuSolverScalars& m_output) {
  active = true;
  ++run_count;
  if (throwing) {
    active = false;
    throw std::runtime_error("Original sentinel exception");
  }
  m_output.bad_jacobian_ = bad;
  active = false;
}
"""
DRIVER_SOURCE = r"""
#include <stdexcept>
#include <string>
#include <thread>
#include "gpu_solver.h"
extern "C" void MockBad(bool);
extern "C" void MockThrow(bool);
extern "C" int MockStarts();
extern "C" int MockStartRun();
extern "C" int MockRuns();
void Require(bool value) {
  if (!value) throw std::runtime_error("Mock assertion failed");
}
void* Create(int ns) {
  GpuSolverSetup setup{};
  setup.state_.dimensions_[0] = ns;
  return CreateGpuSolver(setup);
}
void Run(void* context, int count) {
  GpuSolverScalars output{};
  for (int index = 0; index < count; ++index)
    RunGpuSolver(context, {}, output);
}
int main(int argc, char** argv) {
  Require(argc == 2);
  const std::string scenario(argv[1]);
  if (scenario == "coarse" || scenario == "profiler_error" ||
      scenario == "marker_error") {
    void* context = Create(25);
    Run(context, 10);
    DestroyGpuSolver(context);
    Require(MockStarts() == 0);
    context = Create(99);
    Run(context, 2);
    Require(MockStarts() == 0);
    Run(context, 1);
    Require(MockStarts() == 1 && MockStartRun() == 13);
    Run(context, 5);
    Require(MockStarts() == 1);
    DestroyGpuSolver(context);
    context = Create(99);
    Run(context, 5);
    Require(MockStarts() == 1);
    DestroyGpuSolver(context);
  } else if (scenario == "reset") {
    void* context = Create(99);
    Run(context, 2);
    DestroyGpuSolver(context);
    context = Create(99);
    Run(context, 2);
    Require(MockStarts() == 0);
    MockBad(true);
    Run(context, 1);
    MockBad(false);
    Run(context, 2);
    Require(MockStarts() == 0);
    Run(context, 1);
    Require(MockStarts() == 1 && MockStartRun() == 8);
    DestroyGpuSolver(context);
  } else if (scenario == "exception") {
    void* context = Create(99);
    Run(context, 2);
    MockThrow(true);
    bool caught = false;
    try { Run(context, 1); }
    catch (const std::runtime_error& error) {
      caught = std::string(error.what()) == "Original sentinel exception";
    }
    Require(caught && MockStarts() == 0);
    MockThrow(false);
    Run(context, 2);
    Require(MockStarts() == 0);
    Run(context, 1);
    Require(MockStarts() == 1 && MockStartRun() == 6);
    DestroyGpuSolver(context);
  } else if (scenario == "create_exception") {
    void* context = Create(99);
    Run(context, 2);
    MockThrow(true);
    bool caught = false;
    try { Create(99); }
    catch (const std::runtime_error& error) {
      caught = std::string(error.what()) == "Original create exception";
    }
    Require(caught);
    MockThrow(false);
    Run(context, 2);
    Require(MockStarts() == 0);
    Run(context, 1);
    Require(MockStarts() == 1);
    DestroyGpuSolver(context);
  } else if (scenario == "destroy_exception") {
    void* context = Create(99);
    Run(context, 2);
    MockThrow(true);
    bool caught = false;
    try { DestroyGpuSolver(context); }
    catch (const std::runtime_error& error) {
      caught = std::string(error.what()) == "Original destroy exception";
    }
    Require(caught);
    MockThrow(false);
    Run(context, 4);
    Require(MockStarts() == 0);
    DestroyGpuSolver(context);
  } else if (scenario == "cross_thread") {
    void* context = Create(99);
    std::thread other([&] { Run(context, 1); });
    other.join();
    Run(context, 5);
    Require(MockStarts() == 0 && MockRuns() == 6);
    DestroyGpuSolver(context);
  } else if (scenario == "shared") {
    void* first = Create(99);
    void* second = Create(99);
    Run(first, 5);
    Run(second, 5);
    Require(MockStarts() == 0 && MockRuns() == 10);
    DestroyGpuSolver(first);
    DestroyGpuSolver(second);
  } else if (scenario == "default") {
    void* context = Create(99);
    Run(context, 999);
    Require(MockStarts() == 0);
    Run(context, 1);
    Require(MockStarts() == 1 && MockStartRun() == 1000);
    DestroyGpuSolver(context);
  } else if (scenario == "disabled" || scenario == "invalid_threshold") {
    void* context = Create(99);
    Run(context, 1001);
    Require(MockStarts() == 0 && MockRuns() == 1001);
    DestroyGpuSolver(context);
  } else {
    throw std::runtime_error("Unknown scenario");
  }
}
"""


@unittest.skipUnless(CUDA.is_dir(), "Requires Linux CUDA headers, never a GPU")
class SteadyObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vmec-steady-observer-")
        cls.directory = Path(cls.temporary.name)
        (cls.directory / "mock.cc").write_text(MOCK_SOURCE)
        (cls.directory / "driver.cc").write_text(DRIVER_SOURCE)
        cls.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("VMECPP_") and key != "LD_PRELOAD"
        }
        common = [
            "g++",
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(CUDA / "include"),
            "-I" + str(ROOT / "tools/optimization"),
        ]
        commands = [
            [
                *common,
                "-fPIC",
                "-shared",
                str(cls.directory / "mock.cc"),
                "-o",
                str(cls.directory / "mock.so"),
            ],
            [
                *common,
                "-fPIC",
                "-shared",
                str(ROOT / "tools/optimization/solver_steady_profile_shim.cc"),
                "-ldl",
                "-pthread",
                "-o",
                str(cls.directory / "observer.so"),
            ],
            [
                *common,
                str(cls.directory / "driver.cc"),
                str(cls.directory / "mock.so"),
                "-pthread",
                "-o",
                str(cls.directory / "driver"),
            ],
        ]
        for command in commands:
            subprocess.run(
                command,
                env=cls.environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_scenario(
        self,
        scenario,
        *,
        threshold: str | None = "3",
        enabled=True,
        profiler_status=None,
    ):
        marker = self.directory / f"{scenario}.jsonl"
        if scenario == "marker_error":
            marker = self.directory / "missing-directory" / "marker.jsonl"
        environment = {
            **self.environment,
            "LD_PRELOAD": str(self.directory / "observer.so"),
            "VMECPP_GPU_PROFILE_STEADY": "1" if enabled else "0",
            "VMECPP_GPU_PROFILE_MARKER": str(marker),
        }
        if threshold is not None:
            environment["VMECPP_GPU_PROFILE_STEADY_COUNT"] = threshold
        if profiler_status is not None:
            environment["MOCK_PROFILER_STATUS"] = str(profiler_status)
        result = subprocess.run(
            [str(self.directory / "driver"), scenario],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return (
            [json.loads(line) for line in marker.read_text().splitlines()]
            if marker.exists()
            else []
        )

    def test_coarse_grid_ignored_start_after_run_and_once_per_process(self):
        events = self.run_scenario("coarse")
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "profiler_start"
        assert event["actual_ns"] == 99
        assert event["context_generation"] == 2
        assert event["consecutive_successful_runs"] == 3
        assert event["tid"] == event["owner_tid"]
        assert event["status"] == 0

    def test_context_and_bad_jacobian_reset(self):
        events = self.run_scenario("reset")
        assert len(events) == 1
        assert events[0]["context_generation"] == 2
        assert events[0]["consecutive_successful_runs"] == 3

    def test_original_exception_propagates_and_resets_count(self):
        events = self.run_scenario("exception")
        assert [event["event"] for event in events] == [
            "run_exception",
            "profiler_start",
        ]

    def test_cross_thread_refuses_profiling_but_delegates_solver(self):
        events = self.run_scenario("cross_thread")
        assert events
        assert all(event["event"] == "observer_refused" for event in events)

    def test_create_exception_propagates_and_resets_count(self):
        events = self.run_scenario("create_exception")
        assert [event["event"] for event in events] == [
            "create_exception",
            "profiler_start",
        ]

    def test_destroy_exception_propagates_and_refuses_profiling(self):
        events = self.run_scenario("destroy_exception")
        assert events[0]["reason"] == "destroy_exception"
        assert all(event["event"] == "observer_refused" for event in events)

    def test_multiple_live_contexts_refuse_profiling(self):
        events = self.run_scenario("shared")
        assert events[0]["reason"] == "multiple_live_contexts"
        assert all(event["event"] == "observer_refused" for event in events)

    def test_default_threshold_is_1000(self):
        events = self.run_scenario("default", threshold=None)
        assert events[0]["threshold"] == 1000
        assert events[0]["consecutive_successful_runs"] == 1000

    def test_profiler_error_is_recorded_once_without_changing_solver(self):
        events = self.run_scenario("profiler_error", profiler_status=7)
        assert len(events) == 1
        assert events[0]["status"] == 7

    def test_marker_failure_does_not_change_solver(self):
        assert self.run_scenario("marker_error") == []

    def test_disabled_is_noop(self):
        assert self.run_scenario("disabled", enabled=False) == []

    def test_invalid_threshold_is_not_used(self):
        for threshold in ("0", "1000001", "bad", "3extra"):
            assert self.run_scenario("invalid_threshold", threshold=threshold) == []


if __name__ == "__main__":
    unittest.main()
