# MegaKernel design ontology

## Working definition

A MegaKernel schedules tiles or tasks that would ordinarily belong to multiple operators, model stages, or communication stages inside one persistent launch or an equivalent device-resident execution substrate.

`persistent launch` means that host code launches a long-lived grid once, and resident CTAs/warps continue fetching or interpreting work on the device instead of returning to the host after every ordinary operator. An equivalent substrate may be a device program, Pallas/NKI invocation, or vendor superkernel with the same device-resident multi-stage semantics.

It is reasonable to paraphrase this as “a pipeline that schedules cross-operator/stage tasks,” provided the pipeline is explicitly device-resident. A host-orchestrated pipeline, CUDA Graph, or two-stream sequence is not automatically a MegaKernel.

## Entity hierarchy

Do not chunk the corpus only as repository/document/text. Use these entities:

1. **Project:** repository or downloadable artifact; carries upstream, lineage, openness, and maintenance metadata.
2. **Design:** one concrete execution path inside a project. A repository may contain mega, split, graph, and ordinary backends.
3. **Execution region:** the exact invocation boundary: whole generation, whole forward, layer, MoE pipeline, operator subgraph, or single operator.
4. **Stage/task:** attention, GEMM, dispatch, combine, all-reduce, page update, token sampling, and their dependencies.
5. **Source fragment:** semantic code unit such as launcher, resident loop, scheduler, handler, signal protocol, or benchmark harness.
6. **Experiment:** correctness/performance observation with full context.
7. **Boundary case:** near neighbor or negative such as CUDA Graph, multiple persistent kernels, single-op persistent GEMM, communication-only library, binary backend, or roadmap-only claim.
8. **Episode:** KDA/agent hypothesis, change, observation, and decision. Episodes are evidence candidates, not source facts.

## Design axes

Every serious design or comparison should answer the same axes:

| Axis | Typical values |
|---|---|
| execution scope | whole-generation, whole-forward, layer, moe-pipeline, operator-subgraph, single-op |
| launch model | cooperative persistent grid, interpreter, task runtime, vendor device program, multi-kernel alternative |
| task unit | op, phase, task, tile, token, expert tile, communication chunk |
| scheduler | static monolith, static instruction stream, counter DAG, dynamic ready queue, hybrid, role-specialized |
| dependency | grid barrier, formula-indexed barrier, monotonic counter, event, scoreboard, token/tile readiness signal |
| worker roles | homogeneous workers, controller/loader/compute/storer, scheduler/worker, dispatch/compute/combine/gather |
| state placement | registers, shared-memory pages, global scratch, KV cache, symmetric memory, remote buffers |
| communication | none, NVLink/NVSHMEM, RDMA/IBGDA, xGMI, HCCL, Pallas collective, NKI collective |
| dynamism | fixed shape, bounded specialization, dynamic ready order, routing imbalance, continuous batching |
| topology | flat single die, multi-XCD, multi-GPU, multi-node, NVL fabric |
| fallback boundary | ordinary kernels, PDL, CUDA Graph, two streams, split backend |

## Main paradigms

### Static cooperative monolith

Best fit: fixed model, fixed shape, small batch, repeated service, minimal interpreter overhead. Main costs: code size, resource pressure, grid barriers, difficult portability.

### Instruction interpreter or replay stream

Best fit: a stable set of handlers with a reusable schedule and scratch lifetime plan. It trades some dispatch/page-management overhead for composability. Machete, Hazy, and model-as-a-kernel are anchor cases.

### Task/event runtime

Best fit: dynamic shapes, routing imbalance, continuous batching, or compute/communication arrival uncertainty. It needs explicit accounting for scheduler SMs, queues, atomics, fences, and topology. Mirage MPK, Fleet, and DITRON are anchors.

### Role-specialized communication/compute kernel

Best fit: MoE pipelines with natural dispatch, expert compute, gather, and combine roles. It can remove stage barriers and overlap fabric traffic with tensor-core work. DeepGEMM MegaMoE, Mixture-of-Kittens, and TeraMoE are anchors.

### Multi-kernel low-overhead alternative

PDL, CUDA Graph, or multiple persistent streams can preserve modular kernels while approximating device-side overlap. These are design alternatives, not failed MegaKernels.

## Positive evidence bar

A strict implementation claim should normally connect:

```text
host launcher/device invocation
    + resident loop, interpreter, or device scheduler
    + at least two stages that would normally launch separately
```

Tests and benchmarks raise confidence but do not replace launch-boundary evidence.

## Claim confidence

- `code-confirmed`: visible implementation supports the claim.
- `test-confirmed`: a focused test/benchmark exercises the path.
- `source-reported`: upstream documentation or a paper reports it.
- `cross-repo-derived`: a design inference synthesized from several projects.
- `not-established`: the local corpus does not establish the claim.

Do not infer “unsupported” from silence. Do not promote a name or README keyword to `code-confirmed`.

## Technical and openness labels are independent

Technical class:

- `strict-megakernel`
- `near-whole-model`
- `operator-or-subgraph`
- `adjacent-infrastructure`
- `non-megakernel`

Openness class reuses the archive catalog:

- `A-OSS`: core relevant implementation is reviewable under a root open-source license.
- `B-OSS`: licensed subgraph, prototype, port, infrastructure, or adjacent implementation.
- `C-PARTIAL`: important backend or path is binary/incomplete.
- `S-SOURCE`: source-visible but no clear root license or restricted vendor license.
- `D-BOUNDARY`: alias, fork, redirect, educational material, alternative, or false positive.

One label must not stand in for the other.

## Performance comparability

Compare performance only when these fields are compatible:

```text
hardware + device count + topology + model/phase + batch/context +
dtype + shape set + metric + timing scope + baseline boundary
```

Keep upstream-reported, locally reproduced, and derived measurements separate. A standalone kernel result is not an end-to-end or in-runtime result unless the measurement boundary says so.
