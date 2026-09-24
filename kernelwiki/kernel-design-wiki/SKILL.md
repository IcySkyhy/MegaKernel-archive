---
name: kernel-design-wiki
description: Retrieve and synthesize evidence-backed MegaKernel design knowledge from a local repository archive. Use for device-resident cross-operator scheduling, whole-model or whole-stage persistent kernels, compute/communication MegaMoE design, launch-boundary verification, architecture comparison, design-task authoring, and kernel-design dataset generation. Do not use for generic CUDA syntax, isolated single-operator tuning, ordinary framework integration, or general serving questions without a device-side multi-stage execution boundary.
---

# Kernel Design Wiki

Use this skill as an evidence router and design-decision aid. It is intentionally narrower than a general CUDA wiki and broader than an individual-kernel optimizer.

## Find the corpus

The skill root is the directory containing this file. The preferred source archive is a sibling workspace containing `archive/CATALOG.md`.

1. From the current directory, search upward for `archive/CATALOG.md`.
2. If it is absent, query the bundled distilled corpus and say that local source tracing is unavailable.
3. Never assume that every directory in `archive/` is a strict MegaKernel or that every path in a repository belongs to its MegaKernel implementation.

## Route the request

- **Design recommendation or migration:** read `references/ontology.md`, then `references/query-and-output.md`. Query 3–5 cards and follow their evidence into the archive before recommending a design.
- **Verify a project claim or launch boundary:** read `references/ontology.md`. Query for the project, then inspect the cited launcher, resident loop/scheduler, and at least two stage handlers.
- **Compare projects or paradigms:** read `references/query-and-output.md`. Put every candidate on the same execution-boundary, scheduler, dependency, worker-role, memory, communication, dynamism, topology, and openness axes.
- **Distill new repositories or optimization traces:** read `references/distillation.md`. Create atomic evidence records first, then synthesize decision/case pages.
- **Generate SFT, retrieval, preference, or boundary data:** read `references/dataset-contract.md`. Keep aliases, forks, ports, vendored code, and derived backends in the same split.
- **Author a KDA-style implementation task:** read `references/distillation.md` under "Episode feedback" and return a task contract with objective, exact shapes, baseline, correctness, evaluation, promotion criteria, candidate families, and evidence IDs.

## Query first

Run from the skill root:

```bash
python scripts/query.py "MI350 XCD-aware scheduler" --follow-evidence
python scripts/query.py "static instruction stream" --kind decision --limit 5
python scripts/query.py "MoE dispatch combine" --mechanism role-specialized --json
python scripts/query.py "persistent GEMM" --kind boundary
```

Useful filters: `--kind`, `--hardware`, `--mechanism`, `--scope`, `--repo`, `--confidence`, `--limit`, `--json`, and `--follow-evidence`.

When source evidence matters, open the returned `archive/...` paths and locate the cited symbol or nearby lines. Treat `path + symbol` as the stable locator; line numbers are only navigation hints.

## Evidence discipline

Distinguish these claim classes:

- `code-confirmed`: visible implementation supports the claim.
- `test-confirmed`: a focused test or benchmark exercises it.
- `source-reported`: upstream documentation or a paper states it.
- `cross-repo-derived`: a design inference synthesized from multiple implementations.
- `not-established`: the available corpus does not establish it.

Prefer device implementation and launcher evidence over README wording. Absence from a README is not evidence of absence. Performance comparisons require matching hardware, device count/topology, model phase, batch/context, dtype/shape set, metric, timing scope, and baseline boundary.

## Output contract

For design and comparison answers, include only the fields useful to the request, but preserve this logic:

1. normalized context and assumptions;
2. recommended execution boundary;
3. task/tile granularity and scheduler;
4. dependency protocol and worker roles;
5. memory/state placement and communication;
6. dynamic behavior and topology assumptions;
7. alternatives and non-applicable patterns;
8. evidence IDs with local path/symbol and confidence;
9. unresolved questions and the smallest useful validation plan.

State clearly which parts are source facts and which are migration/design inferences. Do not turn incompatible upstream benchmarks into a leaderboard.

## Maintain and export

```bash
python scripts/validate.py
python scripts/build_index.py
python scripts/export_dataset.py --output-dir <dir> --mode all
python scripts/ingest_catalog.py --catalog <workspace>/archive/CATALOG.md
```

Validation here is deliberately limited to corpus structure, ID/reference integrity, locator existence, and performance-context completeness. It does not perform security auditing, hashing, or dependency scanning.
