# 从 MegaKernel 开源归档到 KernelWiki、Skill 与数据飞轮

日期：2026-09-01

## 一、结论

最值得建设的不是“把所有仓库 README 切块后做 RAG”，也不是立刻拿源码做大规模 SFT/RL，而是四层相互约束的系统：

```text
Evidence Lake
源码符号、launcher、测试、benchmark、论文/文档、谱系、开放状态
        ↓
Typed MegaKernelWiki / Design Graph
机制、执行边界、设计决策、案例、反例、性能上下文
        ↓
Executable Skill
查询、追证、比较、设计 task、数据导出、KDA 风格候选循环
        ↓
Experience Graph
hypothesis/intent、candidate、tool feedback、测量、promote/reject、迁移验证
```

这四层分别解决四个问题：事实是否可信、知识是否可组合、agent 是否会正确使用、优化经验是否能持续回流。

当前工作区已经落成一个可运行的 MVP：

- 一个 `kernel-design-wiki` Skill；
- project、evidence、relation、decision card 和 seed example 的 JSONL 事实层；
- 机制/决策/边界/案例四类 Wiki 页面；
- 查询、目录候选抽取、索引生成、数据导出和轻量验证脚本；
- 面向 source tracing、设计、偏好、边界和 episode 的 seed 数据与评测用例；
- 额外归档的前沿 kernel-agent、benchmark、wiki 和 evaluator 仓库。

机械验证后的当前规模是：107 个 catalog candidate 均能映射到本地目录；curated seed 含 20 个锚点仓库、27 条原子证据、10 条关系、12 张 Wiki 卡和 12 个训练/评测样本。样本按 10 个 `split_group` 导出为 train/validation/test，未发生组间泄漏；Skill 结构也已通过官方快速校验器。

第一版应继续“少而深”：先覆盖 10–15 个锚点实现路径和真实 hard negative，把 schema、检索行为和评测固定，再扩到全部 A-OSS/B-OSS。对 85/107 个仓库各写一篇浅摘要，价值反而更低。

## 二、为什么 repository chunk 不是正确的数据单位

一个仓库通常同时包含普通 kernel、persistent single-op、MegaKernel backend、split backend、测试、框架接线和 vendored code。以仓库作为一个标签会造成四类错误：

1. FlashInfer 因为存在 `moe_ep/backends/mega` 就被整体标成 MegaKernel；
2. DeepGEMM 的普通 GEMM 与 MegaMoE 子路径被混为一谈；
3. qwen_megakernel 因名字被误写成完整 generation single launch；
4. TIRx port、FlashInfer backend 和 DeepGEMM 原实现被当成三个独立发现，并随机分到 train/test 两侧。

正确的最小单位是“某项目中的一个具体 execution path，以及支持该路径某条 claim 的源码符号”。建议七级实体：

```text
project
  └─ design/backend
      └─ execution region
          └─ stage/task
              └─ source fragment
                  └─ benchmark/experiment
                      └─ boundary/negative
```

Agent 的优化尝试另设正交 `episode` 实体，不能与源码事实混在一起。

## 三、persistent launch 的准确含义

这里的 `persistent launch` 是：host 只启动一次长期驻留的 grid，CTA/warp 在设备侧继续取指、取 task、等待 event 或推进 phase，而不是每完成一个普通算子就返回 host，再由 host 启动下一枚 kernel。

因此可以把它概括成“在一个 device-resident pipeline 中调度跨算子/阶段的 tile/task”，但必须保留 **device-resident** 这个限定：host pipeline、CUDA Graph 或两个 stream 串联多枚 kernel，不会因为看起来像 pipeline 就自动成为 MegaKernel。

NKI/Pallas/Ascend device program 只要提供等价的一次设备 invocation 内多阶段常驻执行，也属于“等价设备常驻底座”；定义不应被 CUDA 语法锁死。

## 四、前沿方法图景：应借什么，不应照搬什么

### 4.1 工作流、知识与搜索

| 项目/论文 | 最值得借鉴 | 本项目中的修正 |
|---|---|---|
| [KDA](https://github.com/mit-han-lab/kernel-design-agents) | task contract → draft → executable plan → candidate → correctness/metric → promote/reject；简单持久工件 | KDA 是 workflow/evidence protocol，不是模型或 RL 算法；MegaKernel task 必须自带真实 runtime 的 faithful evaluator |
| [KernelWiki](https://github.com/mit-han-lab/KernelWiki) | `sources → wiki → queries`；受控 tag/alias；confidence/reproducibility | 扩展到 launch boundary、scheduler、worker role、通信/topology、buffer lifetime 和 hard negatives；不照搬其大规模 hash/audit 体系 |
| [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | 将 profiler 证据拆成可行动诊断 | profiler 是可选工具层，不应让普通 Wiki 查询先跑重 profiling |
| [K-Search](https://github.com/caoshiyi/K-Search) | 显式搜索树；将 optimization intent 与具体代码实例分开；保留解空间 | 一次坏 implementation 不应抹掉一个好 hypothesis；raw search trace 仍需迁移验证，不能直接升为 skill |
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent) | profiler/judge/analyzer/orchestrator 分工与并行候选 | 加入 scheduler、critical path、launch coverage、queue imbalance、通信 overlap，而不只看单 kernel counter |
| [Atrex](https://github.com/alibaba/atrex-kernel-agent) | production-trace task 权重、mechanical evaluator ownership、structured memory、optimization dropout、trace→strategy/anti-strategy | 只保留与结果可信度直接相关的 gate；反策略必须有条件、机制和结论，不能从一次失败泛化 |
| [CudaForge](https://github.com/OptimAI-Lab/CudaForge) / CUDAnalyst | raw profile 与 compact bottleneck object 分离；反馈通道解耦；每轮只给少量最相关信号 | 不把某 GPU 上筛出的固定 NCU 指标表硬编码为跨硬件真理；记录 `feedback_used` 以便归因 |
| [STARK](https://arxiv.org/abs/2510.16996) / [KernelArc](https://arxiv.org/abs/2608.17071) | strategy-specialized 并行 agent、compact shared conclusions、plateau 后多样化 | agent 间共享 evidence IDs、候选结论和测量，不共享全部噪声 transcript |

KDA 的本地 Mirage adapter 还给出一个比抽象流程更重要的教训：standalone kernel benchmark 可以把候选排序错；真正的 promotion authority 应是目标 runtime 中的 faithful per-task 或 end-to-end 边界。这个原则应进入所有 episode schema。

### 4.2 任务、数据与 evaluator

| 项目/论文 | 数据价值 | 不能直接当作什么 |
|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) / [论文](https://arxiv.org/abs/2502.10517) | PyTorch reference、生成 contract、correctness + `fast_p`、多轮执行反馈 | 不能作为唯一 evaluator；固定输入/硬件容易被 shape/value 特化，也没有 persistent footprint/通信边界 |
| [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | hidden value distributions、realistic baseline、memory gate、input-blind task | shape 仍可能固定；它是 correctness gate，不是 MegaKernel system eval |
| [KernelBench-X](https://github.com/BonnieW05/KernelBenchX) / [论文](https://arxiv.org/abs/2605.04956) | category-aware failure；揭示 correctness 与效率、硬件迁移的分离 | 类别统计不能替代具体 runtime/拓扑评测 |
| [robust-kbench](https://github.com/SakanaAI/robust-kbench) | 大量 candidate、compile/correctness/runtime/profile 轨迹，适合 preference/repair | 必须在修正 evaluator 下复验；LLM verifier 只能预筛，旧 speedup 不能直接当真值 |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | Definition/Workload/Solution/Trace 分离；动态 workloads；SOL gap；per-workload tolerance | B200 cold-L2 microbenchmark 需补 warm-cache、persistent、通信和 end-to-end 模式；数据许可不能默认训练 |
| [Atrex-Bench](https://arxiv.org/abs/2607.14541) / [FastKernels](https://arxiv.org/abs/2605.23215) | production trace、shape 分布、重要性权重、生产 baseline | operator microbenchmark 不能自动代表 MegaKernel 调度收益 |
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | 18k 级 PyTorch↔Triton 语法/检索冷启动语料 | 没有设计理由和优化轨迹，compiler output 不是性能智慧；还需保留逐源许可 |
| [KernelLLM](https://huggingface.co/facebook/KernelLLM) | 可作为小模型 generator/student baseline | pass@k correctness 不等于速度，官方 card 本身列出 API、shape、precision 等失败 |

MegaKernel evaluator 需要在普通 kernel 指标之外再记录：launch count/coverage、scheduler progress、residency/occupancy、同步流量、queue/role imbalance、topology locality、communication overlap、device span 与 API/e2e span。

### 4.3 训练、skill evolution 与 compiler-agent co-design

| 项目/论文 | 可迁移设计 | 风险/限制 |
|---|---|---|
| [CUDA Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) / [论文](https://arxiv.org/abs/2602.24286) | task synthesis、skill-augmented environment、protected evaluator、correctness-gated performance reward、long-horizon curriculum | 不应在 evaluator 和数据谱系尚未稳定时复制其大规模 PPO/RFT；公开任务集不是完整优化 trajectory/checkpoint |
| [Dr.Kernel / KernelGYM](https://github.com/hkust-nlp/KernelGYM) | 结构化多轮 trajectory、compile/correctness/profile/runtime feedback、分布式 GPU worker | 基础设施重、仍偏 KernelBench/Triton；需增加 runtime/communication schema |
| [daVinci-kernel](https://github.com/GAIR-NLP/daVinci-kernel) | selection/policy/summary 共训；成功 rollout 提炼成五字段 skill，并在执行环境复验 | 只在原任务复验不等于可迁移；必须加 repo/time/family/shape/value/hardware holdout 和负证据 |
| [KernelFoundry](https://github.com/isl-org/kernelfoundry) | MAP-Elites 多样性档案、negative transition、结构策略与参数调优分离 | 现有 descriptor 不理解 persistent state、通信和 topology，需要新的设计维度 |
| [DRTriton](https://arxiv.org/abs/2603.21465) | constrained synthetic DAG、correct/perf 分信号、curriculum、fusion boundary search | 截至本次调研主要为论文设计；dense Triton/H100 和短随机测试不足以覆盖 MegaKernel |
| [μCUTLASS + SOL](https://arxiv.org/abs/2603.29010) | 为 agent 选择“足够高层但仍暴露关键性能杠杆”的 DSL；用 SOL headroom 分配预算 | MegaKernel DSL 必须额外暴露 task graph、role、handoff、state lifetime、topology 和 communication |
| [CAKE](https://arxiv.org/abs/2608.12629) | typed hardware-explicit IR、verifier/cost/local diagnostics；把重复 failure 固化成新 rule/primitive/tactic | compiler/trajectory 尚未完整公开，论文预算和 B200 结果不能写成已复现；single-shape evolution 要与 library/dispatch generalization 分开 |

最直接的长期方向是：`KernelWiki + executable skills + experience graph` 稳定后，才做 CAKE 式 agent-facing schedule IR 与 compiler/verifier co-evolution。

## 五、目标数据模型

### 5.1 Project 与 Design

Project 只表示容器和元数据；Design 才表示具体执行路径：

```yaml
project_id:
lineage_id:
local_root:
upstream:
snapshot_date:
open_status:

design_id:
project_id:
paradigm:
execution_boundary:
launch_semantics:
resident_entity:
task_unit:
schedule_policy:
task_ir:
dependency_mechanisms: []
worker_roles: []
included_stages: []
excluded_stages: []
memory_lifetimes:
communication: []
topology_awareness:
shape_specialization:
source_evidence_ids: []
confidence:
```

### 5.2 Atomic evidence

```yaml
evidence_id:
repo_id:
design_id:
claim:
claim_kind:
locator:
  path:
  symbol:
confidence: code-confirmed | test-confirmed | source-reported |
            cross-repo-derived | not-established
applicability: []
caveats: []
```

稳定定位用可读 ID 与 `path + symbol`；行号只是导航提示。没有必要使用 SHA256 或内容哈希。

### 5.3 Benchmark

```yaml
benchmark_id:
provenance: upstream-reported | locally-reproduced | derived
hardware:
device_count:
topology:
model_phase:
batch_context:
dtype:
shape_set:
baseline_name:
baseline_boundary:
metric:
unit:
value:
timing_scope:
command_or_script_evidence:
caveats: []
```

只有 `hardware + count + topology + phase + batch/context + dtype + shape + metric + timing scope + baseline boundary` 相容时才允许比较。

### 5.4 Candidate、Intent、Skill 与 Episode

需要保留“好 hypothesis、坏 implementation”这种常见情况：

```yaml
intent:
  hypothesis:
  preconditions: []
  expected_effect:
  attempts:
  best_outcome:

candidate:
  parent_id:
  intent_id:
  implementation_space:
  patch_or_code:
  compile_result:
  correctness_trials:
  latency_samples:
  memory:
  profile_raw:
  bottleneck_summary:
  feedback_used: []
  outcome:
  next_plan:

skill:
  scope:
  preconditions: []
  contraindications: []
  hardware_compiler_range:
  schedule_intent:
  roles_and_handoffs: []
  recipe:
  evidence_ids: []
  negative_evidence_ids: []
  verification_matrix:
```

数据只保留可观测的 hypothesis → action/patch → tool evidence → metric delta → decision，不蒸馏私有思维链。

## 六、蒸馏流水线

### 第 1 步：inventory 与 lineage

读取 CATALOG，建立 project candidate；用上游声明和依赖关系记录 alias/fork/port/vendored/backend/uses-runtime，不做代码哈希。

### 第 2 步：semantic entry point discovery

优先找 launcher、`__global__`/device invocation、resident loop、scheduler、queue/event/counter/barrier、task handler、page/scratch allocator、测试和 benchmark。按符号/语义区域切分，不按固定 token 数切。

### 第 3 步：execution-path extraction

每条路径统一抽取十个轴：included/excluded stage、task granularity、scheduler、dependency、worker role、state placement、cooperative residency、dynamism、communication/topology、fallback boundary。

### 第 4 步：evidence card

一个 record 只支持一条可迁移 claim。源码事实优先级：device implementation/launcher > test/benchmark > design doc > README > 二手综述。

### 第 5 步：Wiki synthesis

生成四类页：

- mechanism：技术怎样工作；
- decision：什么条件下选什么、代价和 fallback；
- case：真实仓库边界剖面；
- boundary：相邻路线、缺失条件和反证。

其中 decision page 比 repository summary 更重要。

### 第 6 步：hard pair

首选同数学/同 API、只改变执行边界的真实对比：

- model-as-a-kernel `mega_kernel` vs `phase_kernel`；
- FlashInfer `mega` vs `split` backend；
- TeraMoE vs StreamEP；
- Hazy persistent interpreter vs 81-kernel PDL reconstruction；
- qwen_megakernel vs model-as-a-kernel；
- DeepGEMM MegaMoE vs DeepEP communication-only；
- Machete cross-op stream vs persistent single-op。

### 第 7 步：执行与反馈

先 compile/static contract，再 correctness，最后 performance/profile。每轮只改变一个可归因 hypothesis，并记录原始工件、compact bottleneck、结果和决策。

### 第 8 步：skill promotion

候选经验先在原任务复验，再经过 repo/time/kernel-family/shape/value/hardware holdout。只有跨任务迁移仍成立时，才从 candidate episode 晋级 reusable skill。

## 七、首批高价值 seed

第一批已经围绕以下 15 类路径建立或预留数据轴：

1. AutoMegaKernel：agent-editable task/counter/page/schedule IR；
2. Hazy：per-SM instruction interpreter + paged SMEM；
3. model-as-a-kernel：phase program 与完整 generation loop；
4. Machete：composable Op、formula barrier、readiness-aware static schedule；
5. Mirage MPK：task/event、worker/scheduler queues；
6. Fleet：XCD-aware gang、local/global hierarchical signals；
7. MegaTritonKernel/DITRON：Python graph → Triton schedule/codegen；
8. Lucebox：固定模型 cooperative layer loop；
9. AWS NKI Transformer TKG：非 CUDA 等价设备 invocation；
10. DeepGEMM MegaMoE：dispatch + linear1 + SwiGLU + linear2 + combine；
11. FlashInfer mega/split backend 边界；
12. Mixture-of-Kittens：training forward/backward、communication SM、activation replay；
13. TeraMoE：dispatch/scheduler/compute/combine/gather 五角色；
14. Ascend PTO：AIC/AIV wave 与完整 MoE pipeline；
15. TPU Pallas Fused EP MoE。

第二批应补 mKernel、SGL Kernel NPU、DeepEP、QuACK/persistent GEMM、cuSync、onelaunch、TileRT、StreamEP 和更多 B300/MI350 实测 episode，重点覆盖困难边界与跨硬件迁移。

## 八、可生成的数据视图

机器事实源保持 JSONL/Parquet；Markdown 只是渲染视图。建议物化：

- `evidence.jsonl`
- `projects/designs/relations.jsonl`
- `tasks.parquet`
- `benchmarks.parquet`
- `candidates.jsonl`
- `trajectories.jsonl`
- `skills.jsonl`

训练/评测视图：

1. retrieval QA：问题 → evidence → cited answer；
2. contract → typed design SFT；
3. code + feedback → repair；
4. 同 task/hardware 的 fast/slow 或 applicable/inapplicable preference；
5. state/profile → next intent；
6. strict/near/adjacent/non-MegaKernel boundary classification；
7. evaluator-backed RL task；
8. candidate episode → skill summary，但只导出可审阅短理由。

默认按 `lineage_id/split_group` 切分。DeepGEMM/TIRx/FlashInfer backend、Mirage/Fleet、model-as-a-kernel/det-infer、Lucebox/redirect、Hazy/ThunderKittens 不能跨 split。

## 九、评测与反馈层级

正确性先于性能，但不需要与任务无关的冗余审计。推荐层级：

```text
parse/build
→ static contract / schedule-IR legality
→ runtime progress and memory correctness
→ hidden value + shape correctness
→ faithfulness / no fallback
→ latency distribution and memory footprint
→ profile bottleneck and SOL headroom
→ faithful runtime/device-span transfer
→ API/end-to-end effect
```

MegaKernel 额外观测：

- launch coverage 与 excluded stage；
- scheduler progress/deadlock；
- occupancy/residency；
- barrier/event/fence/atomic traffic；
- queue 和 worker-role imbalance；
- cache/topology locality；
- communication-compute overlap；
- device span 与 API/e2e span 分离。

## 十、实施路线

### P0：Wiki 与可执行底座（现在）

- 固定 ontology、evidence/schema、lineage；
- 深蒸馏 10–15 个 anchor 与 8–10 个 hard negative；
- query/trace/compare/export Skill；
- task/candidate/episode contract；
- evaluator adapter 规范。

### P1：用执行产生数据

- 扩展全部 A-OSS/B-OSS；
- 运行 project-owned test/benchmark；
- 生成 retrieval/design/repair/preference；
- 先做 RAG、SFT 或 rejection fine-tuning，不急于 RL。

### P2：experience/skill co-evolution

- daVinci 式 skill selection/summary；
- K-Search intent tree；
- CUDAnalyst 式 feedback attribution；
- KernelFoundry 式多样性 archive；
- holdout transfer 后晋级 skill。

### P3：compiler-agent co-design

- MegaKernel typed schedule IR；
- explicit stage/task/role/handoff/state/topology；
- compiler verifier、cost model 与 localized diagnostics；
- 从重复失败中增加 IR primitive、rule 与 tactic；
- single-shape search 与 dispatch/library generalization 分开。

## 十一、B300 怎么用

本轮没有使用 B300，因为目标是 schema、证据抽取、边界判断和数据/Skill 设计，源码足以建立这些结论。B300 最有价值的使用点是：

1. B200/H100 证据无法确定 B300 的 residency、寄存器/SMEM 或调度成本；
2. 候选设计需要真实 Blackwell target ranking；
3. standalone 与 persistent-runtime transfer 需要验证；
4. 需要构建 B300-specific correctness/performance episode。

届时先固定 task contract、目标 shape、真实 consumer baseline、correctness 和 promotion metric，再运行；不需要为了“做过验证”而测试与结论无关的仓库。

## 十二、最终建议

1. 将当前 MVP 作为 canonical schema 和 Skill，而不是马上训练模型。
2. 下一轮优先补全 15 个 anchor 的 design/task/source-fragment records，而不是扩大浅层项目数。
3. 先做高质量 boundary 与 preference 数据；它们比重复的正例摘要更能提升 agent 判断。
4. 把 KDA 作为下游执行协议，把 KernelWiki 作为知识层，把 Atrex/K-Search/daVinci 的 episode 与 skill evolution 接到反馈层。
5. 等 faithful evaluator、lineage split 和 holdout transfer 稳定后，再考虑大规模 SFT/RL。
6. 全程只做对知识与评测有直接意义的验证；不引入 SHA256、内容哈希或无关安全审计。
