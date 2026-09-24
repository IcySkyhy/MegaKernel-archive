# Frontier kernel-agent methods and implications

This is a design-oriented reading map, not a claim that every method is a MegaKernel system.

| Work | Reusable idea for MegaKernelWiki | Adaptation needed |
|---|---|---|
| KDA | task contract; draft → plan → candidate → correctness/metric → promote/reject; simple durable artifacts | downstream MegaKernel task must own its faithful runtime evaluator and launch-boundary contract |
| KernelWiki | immutable sources → synthesized wiki → generated query indices; confidence and reproducibility labels | extend beyond NVIDIA kernels to cross-stage runtimes, topology, communication, negative boundaries; omit heavyweight hash/audit machinery |
| ncu-report-skill | profiler evidence decomposed into actionable dimensions | profiling is optional and should follow source/design triage, not gate ordinary wiki queries |
| K-Search | persistent solution database and co-evolving search/world model | store hypotheses and branches as episodes; do not promote agent beliefs directly to facts |
| KernelAgent | profiler/judge/analyzer/orchestrator separation and beam-style exploration | add system-level schedule/critical-path signals, not only individual-kernel counters |
| Atrex | production-trace task distribution, weighted importance, mechanical evaluator ownership, trace-to-strategy/anti-strategy distillation | keep only evidence-relevant gates; require condition + mechanism for negative knowledge; avoid trace-specific complexity in the base skill |
| KernelBench / KernelBench-X | standardized correctness/performance tasks; category-aware failure analysis | MegaKernel evaluation needs launch-boundary, whole-stage, topology, dynamic routing, and runtime-transfer tasks |
| μCUTLASS + SOL | choose an abstraction that exposes high-value levers; use speed-of-light headroom to allocate search | define a MegaKernel schedule IR that exposes task graph, worker roles, memory lifetime, and topology while keeping handler code modular |
| CAKE | compiler-agent co-design; typed hardware-explicit IR; localized verifier/cost diagnostics; recurring failures improve the harness | evolve MegaKernel IR and cost model from episodes; separate single-shape kernel evolution from library/generalization |
| STARK / KernelArc | strategy-specialized parallel agents, compact shared conclusions, plateau-triggered diversification | share evidence IDs, candidate conclusions, and measured outcomes instead of full noisy transcripts |
| CUDA Agent / RLVR | synthesize tasks, expose a skill-augmented environment, use correctness/performance reward | build lineage-clean design tasks and reward faithful runtime transfer, not benchmark-only speed |
| Correctness Illusion / KernelBench-X | fixed-shape checks miss semantic and numerical failures; correctness need not imply efficiency | generate contract-aware shape/dtype tests and separate compilation, semantic correctness, and performance |

Primary local references:

- `archive/agentic-kernel-design/kernel-design-agents/`
- `archive/agentic-kernel-design/KernelWiki-upstream/`
- `archive/agentic-kernel-design/ncu-report-skill/`
- `archive/agentic-kernel-design/K-Search/`
- `archive/agentic-kernel-design/KernelAgent/`
- `archive/agentic-kernel-design/atrex-kernel-agent/`
- `archive/agentic-kernel-design/KernelBench/`
- `archive/agentic-kernel-design/KernelBenchX/`
- `archive/agentic-kernel-design/CUDA-Agent/`

Paper links:

- CAKE: <https://arxiv.org/abs/2608.12629>
- μCUTLASS + SOL: <https://arxiv.org/abs/2603.29010>
- STARK: <https://arxiv.org/abs/2510.16996>
- Atrex: <https://arxiv.org/abs/2607.14541>
- KernelBench: <https://arxiv.org/abs/2502.10517>
- KernelBench-X: <https://arxiv.org/abs/2605.04956>
- Correctness Illusion: <https://arxiv.org/abs/2606.20128>
- CUDA Agent: <https://arxiv.org/abs/2602.24286>
- Agentic Kernel Optimization: <https://arxiv.org/abs/2608.14560>
- KernelArc: <https://arxiv.org/abs/2608.17071>
