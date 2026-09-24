# Task/event runtimes

Mirage MPK makes tasks, events, queues, and scheduler/worker roles explicit in the runtime contract. This supports device-side readiness and distributed events, but moves scheduling cost onto SMs, atomics, fences, queue memory, and cache traffic.

AutoMegaKernel exposes a related agent-editable task/counter program. Its monotonic counters and static thresholds make dependencies easy to reason about, while per-SM queues and schedule search control placement.

Evidence: `ev-mpk-worker-scheduler-queues`, `ev-mpk-runtime-contract`, `ev-amk-counter-dag`.
