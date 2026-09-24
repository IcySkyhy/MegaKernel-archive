# Distillation and episode workflow

## Core rule

Distill atomic evidence before writing a project summary. The minimum useful knowledge unit is one claim about one concrete execution path with a source locator and applicability boundary.

Do not feed all repository text directly into RAG. README-heavy chunking overweights project claims, duplicates forks, hides backend boundaries, and creates misleading performance comparisons.

## Pipeline

### 1. Inventory and lineage

Parse `archive/CATALOG.md` into candidate project records. Normalize upstream URLs and assign human-readable lineage IDs. Record aliases, forks, ports, vendored implementations, backend adapters, and direct runtime dependencies without code hashing.

Required lineage relations include:

- `alias-of`
- `fork-of`
- `port-of`
- `vendored-from`
- `backend-of`
- `uses-runtime`
- `implements-paper`
- `variant-of`
- `alternative-to`

Examples that must remain grouped: Mirage/Fleet, DeepGEMM/TIRx/FlashInfer DeepGEMM backend, model-as-a-kernel/det-infer, Lucebox/redirect, and ThunderKittens/its AMD port.

### 2. Discover semantic entry points

Search for launcher calls, global/device entry points, persistent loops, task/schedule classes, worker roles, event/counter/barrier operations, scratch/page allocators, tests, and benchmark scripts. Segment by symbol or coherent code region, not a fixed token window.

### 3. Extract execution-path records

For each design, answer:

1. Which stages are inside the invocation, and which are outside?
2. What is the task/tile unit?
3. How is work ordered and made ready?
4. How are dependencies represented?
5. How are SMs/warps divided into roles?
6. Where do activations, scratch, KV, and symmetric buffers live?
7. Is cooperative residency or a global barrier required?
8. What dynamism and routing imbalance are supported?
9. What topology and communication assumptions exist?
10. When should the design fall back to multiple kernels, PDL, or CUDA Graph?

### 4. Write evidence cards

Each evidence card contains one transferable claim, not a whole-repository synopsis. Use a stable `path + symbol` locator. An optional line hint is for navigation only.

Performance claims additionally require GPU, count/topology, dtype, shape/workload, baseline, metric, value, and timing scope. Leave a field absent or mark the claim `not-established` instead of inventing it.

### 5. Synthesize wiki cards

Generate four page classes:

- `mechanism`: how a scheduler, dependency, memory, or role protocol works;
- `decision`: conditions → recommended design → costs → fallback;
- `case`: one implementation's exact boundary and evidence map;
- `boundary`: a difficult neighbor or negative and the missing criterion.

Decision pages are the highest-value layer. They should explain when and why to choose a pattern, not only define it.

### 6. Build hard contrasts

Prefer real alternatives over fabricated bad designs:

- model-as-a-kernel `mega_kernel` vs its per-phase launch path;
- FlashInfer mega vs split backend;
- TeraMoE single cooperative kernel vs StreamEP multi-kernel/two-stream pipeline;
- Hazy persistent interpreter vs PDL reconstruction;
- Machete cross-op replay vs a single-op persistent GEMM;
- qwen_megakernel near-whole-model boundary vs model-as-a-kernel whole-generation boundary;
- DeepGEMM MegaMoE vs communication-only DeepEP;
- source-complete Machete vs binary-core TileRT for openness questions.

The negative label must say which necessary condition is missing.

### 7. Generate agent data

Use `references/dataset-contract.md`. Generate retrieval, design, preference, boundary, and episode examples only from curated cards and evidence IDs. Do not export hidden chain-of-thought.

### 8. Human review and promotion

Review boundary flips, lineage changes, cross-hardware generalizations, and performance comparisons. Routine schema/reference checks can be mechanical; this workflow does not need hashes or broad security auditing.

## KDA-style task contract

When the user wants an implementation task rather than a wiki answer, produce:

```yaml
task_name:
objective:
target_execution_boundary:
hardware_and_topology:
exact_shapes_and_dtypes:
current_baseline:
correctness_requirements:
validation_command:
evaluation_command:
promotion_criteria:
candidate_families:
  - hypothesis:
    evidence_ids: []
    expected_effect:
    main_risk:
required_artifacts:
  - docs/draft.md
  - docs/plan.md
  - candidates.jsonl
  - benchmark.csv
  - profile/
```

Keep the reusable workflow separate from the implementation workspace. The downstream task owns its evaluator and hardware-specific rules.

## Episode feedback

Record each meaningful attempt as:

```yaml
episode_id:
task_contract_id:
parent_episode_id:
hypothesis:
evidence_used: []
proposed_change:
expected_effect:
correctness_result:
measurement_context:
observed_result:
decision: promote | revise | reject | inconclusive
transferable_lesson:
```

Promotion requires task-contract correctness and evidence that the target metric improved or was preserved. Rejected candidates remain valuable if the condition, attempted lever, observed result, and plausible mechanism are explicit.

Episodes enter a candidate layer first. Promote them to Wiki knowledge only after the code path, result, and applicability can be traced. A failed attempt without a checkable condition or mechanism is not a reusable anti-pattern.

## Incremental refresh without hashes

Track `snapshot_date`, upstream ref name, last-change date/title, entity version, `valid_from`, `valid_to`, and `supersedes`. Re-extract changed semantic paths; reconsider the launch boundary when launcher/runtime code changes; append new benchmark records instead of overwriting old ones.
