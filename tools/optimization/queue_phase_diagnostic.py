"""Opt-in copied queue phase clocks; no runtime launch or per-step instrumentation."""

from __future__ import annotations

import threading
import time


class PhaseRecorder:
    """Record only request/method boundaries in memory until futures drain."""

    def __init__(self, wall=time.perf_counter_ns, cpu=time.thread_time_ns):
        self.wall = wall
        self.cpu = cpu
        self.lock = threading.Lock()
        self.events = {}

    def mark(self, state, phase, role):
        event = {
            "phase": phase,
            "role": role,
            "tid": threading.get_native_id(),
            "wall_ns": self.wall(),
            "thread_cpu_ns": self.cpu(),
        }
        with self.lock:
            self.events.setdefault(state.job.key, []).append(event)

    def snapshot(self, job):
        with self.lock:
            return [dict(event) for event in self.events.get(job, [])]


def replace_once(source, before, after):
    if source.count(before) != 1:
        message = "Diagnostic source anchor changed; review before execution"
        raise ValueError(message)
    return source.replace(before, after)


def instrument_batch_source(source):
    patches = [
        (
            "        try:\n            job.out.mkdir(parents=True, exist_ok=False)",
            (
                "        PHASE_RECORDER.mark(state, 'request_start', 'adapter')\n"
                "        try:\n            job.out.mkdir(parents=True, exist_ok=False)"
            ),
        ),
        (
            "            with dispatcher.request_scope(state):\n                state.evaluation_started",
            (
                "            with dispatcher.request_scope(state):\n"
                "                PHASE_RECORDER.mark(state, 'initial_gate_acquired', 'adapter')\n"
                "                state.evaluation_started"
            ),
        ),
        (
            "                    state.evaluation_ended = time.perf_counter()",
            (
                "                    PHASE_RECORDER.mark(state, 'evaluation_end', 'adapter')\n"
                "                    state.evaluation_ended = time.perf_counter()"
            ),
        ),
        (
            "            state.finished = time.perf_counter()",
            (
                "            PHASE_RECORDER.mark(state, 'request_end', 'adapter')\n"
                "            state.finished = time.perf_counter()"
            ),
        ),
        (
            '        save(job.out / "report.json", report)',
            (
                '        report["diagnostic_request_phase_events"] = PHASE_RECORDER.snapshot(job.key)\n'
                '        report["main_table_eligible"] = False\n'
                '        save(job.out / "report.json", report)'
            ),
        ),
    ]
    for before, after in patches:
        source = replace_once(source, before, after)
    return source


def instrument_queue_source(source):
    patches = [
        (
            "                call.output = self.original(*positional, **keywords)",
            (
                "                PHASE_RECORDER.mark(state, 'native_method_start', 'native')\n"
                "                try:\n"
                "                    call.output = self.original(*positional, **keywords)\n"
                "                finally:\n"
                "                    PHASE_RECORDER.mark(state, 'native_method_end', 'native')"
            ),
        ),
        (
            "        state.in_vmec = True",
            (
                "        PHASE_RECORDER.mark(state, 'setup_end_before_submit', 'adapter')\n"
                "        state.in_vmec = True"
            ),
        ),
        (
            "        finally:\n            waiting = time.perf_counter()\n            self.gate.acquire()",
            (
                "        finally:\n"
                "            PHASE_RECORDER.mark(state, 'native_wait_end_before_gate', 'adapter')\n"
                "            waiting = time.perf_counter()\n            self.gate.acquire()\n"
                "            PHASE_RECORDER.mark(state, 'post_gate_acquired', 'adapter')"
            ),
        ),
    ]
    for before, after in patches:
        source = replace_once(source, before, after)
    return source
