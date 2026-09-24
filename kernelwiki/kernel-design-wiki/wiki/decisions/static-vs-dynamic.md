# Static replay versus dynamic readiness

Choose static replay for fixed shapes, regular dependency graphs, and repeated schedules. Choose a device-ready runtime when routing, arrivals, or task duration vary enough that a fixed order creates measurable bubbles.

A useful hybrid fixes the graph, memory lifetimes, and most placement at compile time, then leaves only ready order or load balancing dynamic. Measure scheduler/queue cost before adding generality.

Evidence: `ev-machete-readiness-scheduler`, `ev-mpk-worker-scheduler-queues`, `ev-amk-counter-dag`.
