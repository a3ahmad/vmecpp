# Profile-driven optimization experiments

These are executed RTX 3090 experiments, separate from the qualified six-case
benchmark table. Ratios above 1 favor the prototype. Broad final qualification
is deferred; original output checks remain embedded in performance runs.

| Experiment | Measured graph-route result | Decision |
| --- | --- | --- |
| Stock capped scan vs CURRENT | 1.0025-1.0185x CUDA-event ratio, four cases | Small component gain only |
| Capped scan, 128x2 policy vs CURRENT | 0.9894-0.9947x, four cases | Set aside on RTX 3090 |
| Exact FP64 map encoding vs CURRENT | 0.7404-0.7488x, four cases | Set aside on RTX 3090 |
| Aggregate Seed statistics atomics vs CURRENT | 0.9974x event ratio; RAW ratio 1.0002x | No graph benefit; do not promote |

Each scan variant was paired against CURRENT within its own run. Their absolute
times are not directly paired across variants. All 38,400 scan packets matched;
the Seed experiment matched another 3,200 packets and per-pair statistics.
The Seed direct route improved 1.0110x, but that did not carry into graph mode.
These component measurements do not establish a solver speedup or metric drift.

The measured CURRENT Map kernel took 11.072 us, with 6,670,355 integer thread
instructions, no FP64 instructions and 14.93% DRAM throughput. Its 86 blocks
over 82 SMs provide little grid-level latency hiding. Seed took 7.680 us with
only three blocks, 47 DADD instructions and 47 contended statistics increments.
This evidence motivated the launch-policy and statistics experiments above.

The conditional whole-solver prototype also executed: two OFF controls took
7.694 and 7.736 s; two ON trials took 7.814 and 7.819 s. These are measured
single-fixture timings with reversed pair order, not a generalized confidence
claim. Original metrics/request identity, whole NC and 50 diagnostic arrays
passed the embedded original-authority checks in all four calls.
Physical graph launches fell from 27,477 to 13,770, but performance regressed.
The ON path recorded 13,708 combined predicates, 13,707 valid tails and one bad
tail, with no errors or pending buffers. Fewer launches alone did not help.

The first two one-second conditional Nsight windows collected no CUDA activity:
the launcher cleared injection variables after Nsight started. Those failed
captures are preserved and provide no predicate-cost or idle-time evidence.
The launch path is being corrected before selecting the next conditional change.

The A100 SXM4 access experiment found protected counters blocked. Instance
59499184 was destroyed with teardown verification after the provider's 429
backoff. No A100 counter result or new A100 optimization gain is claimed.

Saved results are under build/optimization/native-scan-screen1-analysis1,
native-seed-stats-actual1 and native-conditional-actual1. The main qualified
speedups remain unchanged. Physics settings and benchmark artifacts were not
modified.
