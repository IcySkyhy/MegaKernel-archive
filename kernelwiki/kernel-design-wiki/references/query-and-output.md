# Query and output patterns

## Four query modes

### Design recommendation

Normalize hardware/topology, workload graph, shapes/batch, latency-vs-throughput objective, desired fusion boundary, and allowed implementation stack. Return 1–3 ranked patterns with their assumptions and fallback.

### Source evidence

Return:

```text
claim → evidence ID → local path/symbol → confidence → direct interpretation
```

Open cited source files before making a strong implementation claim.

### Comparison

Align candidates on the same axes from `ontology.md`. Do not rank incompatible performance numbers. Compare execution semantics and operational fit first.

### Dataset generation

Return example type, prompt/input, compact target, evidence IDs, lineage/split group, and a short reviewable rationale.

## Structured design output

Use this full form for machine-readable output and compress it for ordinary prose:

```yaml
mode:
normalized_context:
assumptions: []
recommendation:
design_spec:
  execution_scope:
  included_stages: []
  excluded_stages: []
  launch_model:
  task_unit:
  scheduler:
  dependency_protocol: []
  worker_roles: []
  memory_and_state: []
  communication: []
  dynamic_behavior:
  topology_assumptions: []
alternatives: []
non_applicable_patterns: []
evidence:
  - evidence_id:
    locator:
    confidence:
open_questions: []
minimal_validation_plan: []
```

## Decision heuristics

- Fixed shapes, regular dependencies, and launch-dominated latency favor a static monolith or instruction stream.
- Dynamic routing/readiness or severe imbalance favors event/counter/ready-queue scheduling, if scheduler and atomic costs are budgeted.
- Multi-XCD hardware requires locality and signal scope in the scheduler; a flat SM round-robin policy is not a portable default.
- Distributed MoE benefits most when dispatch, readiness, expert compute, and combine overlap at token/tile granularity; merely reducing launch count is insufficient.
- If fusion inflates code/resource pressure or does not eliminate the critical barrier, preserve modular kernels and consider PDL, graph replay, or streaming multi-kernel overlap.
- Separate body optimization from schedule optimization. A faster task handler can be an end-to-end null when it is off the critical path or poorly placed.

## Minimal validation plans

Propose the smallest test that resolves the actual uncertainty:

- launch-boundary question: trace launcher and resident body; no GPU required;
- correctness/shape generality: run the project evaluator on representative and adversarial shapes;
- performance transfer: measure inside the real consumer/runtime, not only a standalone green-context kernel;
- B300-specific resource/scheduler claim: use B300 only when source evidence cannot establish the claim or the decision depends on measured Blackwell behavior;
- cross-vendor migration: validate on the target hardware rather than extrapolating from B200/H100.
