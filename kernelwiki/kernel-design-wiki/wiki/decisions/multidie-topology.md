# Multi-die topology decisions

Fleet shows why a flat worker pool is insufficient on MI350: XCDs have private L2 domains, so related tiles should cooperate locally and completion should be aggregated in local counters before a global publish.

The transferable rule is to make cache and signaling scope first-class in task placement. The concrete XCD count and AMD memory primitives are not portable to B300 or a single-die GPU.

Evidence: `ev-fleet-xcd-placement`, `ev-fleet-hierarchical-signals`.
