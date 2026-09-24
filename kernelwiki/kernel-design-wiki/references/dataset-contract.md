# Kernel-design dataset contract

## Dataset families

### Retrieval QA

Question → selected evidence cards → concise cited answer. Evaluate both evidence recall and claim support.

### Design SFT

Hardware/workload/shape/goal constraints → structured design specification. Keep source facts separate from transfer inferences.

### Preference data

Pair an applicable evidence-backed proposal with a real but inapplicable alternative. Include a short rejection reason such as topology mismatch, wrong launch boundary, missing dynamic support, or incompatible measurement scope.

### Boundary classification

Classify strict MegaKernel, near-whole-model, operator/subgraph, adjacent infrastructure, or non-MegaKernel. Name the satisfied and missing criteria.

### Episode data

Task contract + incumbent + hypothesis + action + measured observation → promote/revise/reject. Retain null and regression results when they encode a transferable condition and mechanism.

## Record shape

The seed exporter uses:

```json
{
  "id": "sample-id",
  "type": "retrieval|design|preference|boundary|episode",
  "prompt": "...",
  "input": {},
  "target": {},
  "short_rationale": "reviewable explanation, not hidden chain-of-thought",
  "evidence_ids": ["ev-..."],
  "lineage_ids": ["lineage-..."],
  "split_group": "group-name"
}
```

## Leakage control

Never randomly split code chunks. Keep these together:

1. every design, source, benchmark, and generated example from one repository;
2. original implementation, paper, fork, port, vendored copy, and backend adapter;
3. both members of a positive/negative or preference pair;
4. all summaries/questions derived from one benchmark or optimization episode.

Default split unit is `split_group`, normally a lineage. Add evaluation suites for unseen lineage, scheduler paradigm, hardware vendor, future snapshot, and source-visible/no-license boundary cases.

Known leakage groups include:

- DeepGEMM ↔ TIRx ↔ FlashInfer DeepGEMM backend;
- Mirage ↔ Fleet;
- model-as-a-kernel ↔ det-infer;
- Lucebox ↔ luce redirect;
- Hazy ↔ ThunderKittens dependency;
- upstream implementation pages already present in KernelWiki or agent corpora.

## Licensing-aware export

Use openness metadata for data routing, not as a security audit.

- Open-source components may contribute small, attribution-preserving code excerpts when the downstream dataset license permits it.
- Source-visible/no-root-license or restricted vendor sources should default to metadata, paths, and paraphrased design facts rather than redistributed code.
- Binary-only cores contribute boundary and openness examples, not implementation training targets.

## Quality checks

Only perform checks relevant to knowledge quality:

- required fields and unique IDs;
- evidence and relation references resolve;
- local source locators exist;
- performance claims have complete context;
- lineage/split groups are present;
- target claims do not exceed evidence confidence.

No hash, vulnerability scan, dependency audit, or broad boundary-condition audit is part of this contract.
