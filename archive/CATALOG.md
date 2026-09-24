# MegaKernel 归档总索引

MegaKernel 实现图景快照：2026-08-26；agentic kernel design 扩展：2026-09-01。共 107 个本地 Git 工件，其中 85 个用于实现图景，22 个用于知识蒸馏、agent、评测与数据方法研究。

## 标记规则

| 标记 | 含义 |
|---|---|
| A-OSS | 相关核心源码可审阅、根仓有明确开源许可证，并满足跨算子/跨阶段设备常驻执行定义 |
| B-OSS | 有明确开源许可证，但属于子图、原型、移植、基础设施或相邻实现 |
| C-PARTIAL | 公开仓库存在，但关键 backend 为 binary、论文路径未完整发布，或功能尚不完整 |
| S-SOURCE | 源码可读，但根仓无明确许可证，或使用限制用途的厂商许可；不按严格 OSS 计 |
| D-BOUNDARY | 重定向、重复谱系、fork、教学材料、替代设计或关键词误命中 |

“A-OSS”只说明开源性与技术边界，不代表通用性、性能领先或生产成熟。性能数字请回到各项目的具体模型、shape、精度、拓扑和计时边界。

## 1. 编译器与设备常驻运行时（7）

| 本地工件 | 上游 | 标记 | 核心判断与源码证据 |
|---|---|---|---|
| [AutoMegaKernel](compiler-runtimes/AutoMegaKernel) | [RightNow-AI/AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) | A-OSS · MIT | agent-facing schedule IR、自调优和 cooperative CUDA forward；vm、instructions、schedule、models。当前重点是 Llama-family dense forward，动态 batching/MoE 不应写成已完成。 |
| [fleet-chiplet-megakernel](compiler-runtimes/fleet-chiplet-megakernel) | [ROCm/fleet-chiplet-megakernel](https://github.com/ROCm/fleet-chiplet-megakernel) | A-OSS · Apache-2.0 | AMD 官方 Fleet；MI350/gfx950 的 XCD-aware task、分层同步和 Qwen3-8B demo。是 MPK 路线的多 die 拓扑扩展，不再是“论文无代码”。 |
| [luminal](compiler-runtimes/luminal) | [luminal-ai/luminal](https://github.com/luminal-ai/luminal) | B-OSS · MIT/Apache-2.0 | Rust 图编译器；cuda_lite BlockOps 和 symbolic work queue 能组合成 megakernel，路径仍在快速演进，暂不等同成熟 whole-model runtime。 |
| [machete](compiler-runtimes/machete) | [b-albar/machete](https://github.com/b-albar/machete) | A-OSS · Apache-2.0 | CuTe DSL 可组合 Op、静态 instruction stream、paged SMEM、barrier/TMA 和 persistent CTA replay；src/machete/megakernel 与 src/machete/kernels。不要与 vLLM 的同名 mixed-input GEMM 混淆。 |
| [mirage](compiler-runtimes/mirage) | [mirage-project/mirage](https://github.com/mirage-project/mirage) | A-OSS · Apache-2.0 | MPK 编译器/runtime；tGraph、task/event、设备内 worker/scheduler 队列，覆盖单/多 GPU dense 与 MoE。关键实现 include/mirage/persistent_kernel。 |
| [TileRT](compiler-runtimes/TileRT) | [tile-ai/TileRT](https://github.com/tile-ai/TileRT) | C-PARTIAL · MIT 外壳 | 公开 Python、转换、服务工具；核心 libtilert backend 以固定 ABI binary wheel/so 分发。可运行的 8×B200 产品路径不等于完整开源编译器。 |
| [Triton-distributed-ascend](compiler-runtimes/Triton-distributed-ascend) | [Ascend feature/megakernel](https://gitcode.com/Ascend/Triton-distributed-ascend/tree/feature%2Fmegakernel) | B-OSS · MIT | feature/megakernel 分支；已有双 GEMM、MLP、LM-head opgraph 的 Ascend 大核编译测试。Qwen3 当前是 DSL dump/stub，不能声称已完成 whole-model NPU 实跑。 |

## 2. 整模型、整 forward 或整 token 工件（16）

| 本地工件 | 上游 | 标记 | 一次 launch 的真实边界 |
|---|---|---|---|
| [aws-nki-library](core-whole-model/aws-nki-library) | [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library) | A-OSS · Apache-2.0 | 官方 Transformer TKG 在一个 NKI invocation 内执行多层 attention、MLP、residual 与跨层 collective；Trainium/Inferentia 专用。 |
| [bonsai-turbo](core-whole-model/bonsai-turbo) | [RightNow-AI/bonsai-turbo](https://github.com/RightNow-AI/bonsai-turbo) | A-OSS · Apache-2.0 | Bonsai 27B ternary/H100；src/cuda/mega.cu 提供 cooperative whole-model 路径，另有 CUDA Graph 路径。模型特化。 |
| [DSV2-236B-MegaKernel](core-whole-model/DSV2-236B-MegaKernel) | [SwayamInSync/DSV2-236B-MegaKernel](https://github.com/SwayamInSync/DSV2-236B-MegaKernel) | S-SOURCE · 无根许可证 | 8×B200、DeepSeek-V2-236B；单 CuTe DSL launch 覆盖 60 层、TP/EP collective、LM head 和 argmax。技术边界明确，但不按严格 OSS 计。 |
| [HazyResearch-Megakernels](core-whole-model/HazyResearch-Megakernels) | [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels) | A-OSS · MIT | Llama whole-forward 与多 GPU TP demo；per-SM instruction sequence、GPU interpreter、paged shared memory、counter dependency。 |
| [luce-megakernel](core-whole-model/luce-megakernel) | [Luce-Org/luce-megakernel](https://github.com/Luce-Org/luce-megakernel) | D-BOUNDARY | 上游重定向/内容与 Lucebox 同谱系；保留用于历史链接，不作为第二个独立方案计数。 |
| [lucebox](core-whole-model/lucebox) | [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) | A-OSS · Apache-2.0 | Qwen3.5-0.8B 的 24 层 DeltaNet/attention 单 persistent dispatch；prefill 单列、batch 1/模型专用。 |
| [MaruthiV-megakernel](core-whole-model/MaruthiV-megakernel) | [MaruthiV/megakernel](https://github.com/MaruthiV/megakernel) | S-SOURCE · 无根许可证 | Qwen3-0.6B 教学型 whole-model CUDA 实现，体量和成熟度较低。 |
| [MegaQwen](core-whole-model/MegaQwen) | [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen) | S-SOURCE · 无根许可证 | RTX 3090/Qwen3-0.6B；主 cooperative kernel 跨层，但输出阶段仍有独立 kernel。适合研究同步瓶颈，不是严格全链单启动。 |
| [model-as-a-kernel](core-whole-model/model-as-a-kernel) | [HF phanerozoic/model-as-a-kernel](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel) | A-OSS · Apache-2.0 | phase interpreter 可把 embedding、所有层、LM head、greedy argmax、prompt 消费和完整 generation loop 放入一次 launch；batch 1/greedy/模型变体有限。 |
| [model-as-a-kernel-det-infer](core-whole-model/model-as-a-kernel-det-infer) | [HF phanerozoic/det-infer](https://huggingface.co/kernels/phanerozoic/det-infer) | D-BOUNDARY · Apache-2.0 | 同一 model-as-a-kernel 引擎的 deterministic inference 变体，按同一谱系计数。 |
| [Nemotron-3-Embed-Megakernel](core-whole-model/Nemotron-3-Embed-Megakernel) | [LynnAnalytics/Nemotron-3-Embed-Megakernel](https://github.com/LynnAnalytics/Nemotron-3-Embed-Megakernel) | A-OSS · MIT | sm120a/RTX 5090、16 层 embedding 模型、pooling/norm 的 persistent whole-model kernel；非常早期。 |
| [nki-moe-megakernel](core-whole-model/nki-moe-megakernel) | [KevGomes1403/nki-moe-megakernel](https://github.com/KevGomes1403/nki-moe-megakernel) | A-OSS · Apache-2.0 | Trainium2/3；Qwen3.6/Qwen3/GPT-OSS 的 attention、MoE、norm、vocab、greedy argmax 与 TP collective；与 AWS 官方 NKI 库分开记为应用级实验。 |
| [qwen_megakernel](core-whole-model/qwen_megakernel) | [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) | A-OSS · MIT | RTX 5090/Qwen3-0.6B；embedding、28 层和 final norm 在主 kernel，LM head/argmax 分离，属于“近整模”。 |
| [qwen-tts-0.6b-megakernel](core-whole-model/qwen-tts-0.6b-megakernel) | [ckmonish2000/qwen-tts-0.6b-megakernel](https://github.com/ckmonish2000/qwen-tts-0.6b-megakernel) | S-SOURCE · 无根许可证 | Qwen TTS 0.6B 的模型专用单核实验；应与 qwen-tts-turbo 子图优化分开。 |
| [sinter-qwen36-nomtp-tu102](core-whole-model/sinter-qwen36-nomtp-tu102) | [slartibardfast/sinter-qwen36-nomtp-tu102](https://github.com/slartibardfast/sinter-qwen36-nomtp-tu102) | A-OSS · Unlicense | 双 TU102/NVLink、Qwen3.6 decode 的设备常驻实验；真实但高度特化。 |
| [talos-vs-macbook](core-whole-model/talos-vs-macbook) | [AlexCheema/talos-vs-macbook](https://github.com/AlexCheema/talos-vs-macbook) | A-OSS · MIT | microGPT 4192 参数/约 17KB；单 block 完成 forward、sampling 和多 token loop。是机制教学，不是生产 LLM 库。 |

## 3. 分布式与 MoE MegaKernel（12）

| 本地工件 | 上游 | 标记 | 跨阶段边界 |
|---|---|---|---|
| [Alpha-MoE](distributed-moe/Alpha-MoE) | [Aleph-Alpha/Alpha-MoE](https://github.com/Aleph-Alpha/Alpha-MoE) | A-OSS · Apache-2.0 | Hopper W8A8/FP8、tensor-parallel MoE layer megakernel；仓库已归档只读，按历史实现看待。 |
| [DeepGEMM](distributed-moe/DeepGEMM) | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM#mega-moe) | A-OSS · MIT | MegaMoE 子路径在 SM100 单核融合 EP dispatch、FP8×FP4 linear1、SwiGLU、linear2 和 combine；不要把 DeepGEMM 的全部 kernel 都叫 megakernel。 |
| [FastAFD](distributed-moe/FastAFD) | [hao-ai-lab/FastAFD](https://github.com/hao-ai-lab/FastAFD) | A-OSS · MIT | Blackwell NVL72 Attention–FFN disaggregation；EG 侧一个 persistent kernel 跨层/多 microbatch lane 完成 pull、专家计算和 send-back。vendored DeepGEMM 与 FastAFD 新增 M2N 代码需区分。 |
| [flashinfer](distributed-moe/flashinfer) | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | A-OSS · Apache-2.0 | 不能再只列“基础库”：moe_ep/backends/mega 与 cutedsl_megamoe 已含 DeepGEMM、BF16、NVFP4、MXFP8、SM90 pull/push 等 MegaMoE 后端；其余 FlashInfer 算子不因此自动成为 megakernel。 |
| [FlashMoE](distributed-moe/FlashMoE) | [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | A-OSS · BSD-3-Clause | NVSHMEM/CUTLASS/cuBLASDx；dispatch、expert FFN、combine 与 GPU-initiated communication 在一个 persistent kernel。router 仍可位于调用边界外。 |
| [mixture-of-kittens](distributed-moe/mixture-of-kittens) | [cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens) | A-OSS · Apache-2.0 | GB200/GB300 NVL72；deterministic MoE training forward 与 backward megakernels，含通信/计算 SM 分工、CLC 和 activation replay。 |
| [mKernel](distributed-moe/mKernel) | [uccl-project/mKernel](https://github.com/uccl-project/mKernel) | A-OSS · MIT | Hopper/CX7/EFA；AG+GEMM、GEMM+AR/RS、完整 MoE 和 ring attention 等 GPU-driven multi-GPU/multi-node persistent kernels。各 demo 的融合边界不同。 |
| [pto-isa](distributed-moe/pto-isa) | [CANN PTO-ISA](https://gitcode.com/cann/pto-isa) | S-SOURCE · CANN OSL v2.0 | dispatch_mega_combine 在 Ascend A2/A3/A5 的一个混合 AIC/AIV 大核中覆盖 reorder、dispatch、GMM1、SwiGLU、GMM2、combine、unpermute。许可证限制在华为 AI 处理器/软件用途，不按 OSI OSS 计。 |
| [sgl-kernel-npu](distributed-moe/sgl-kernel-npu) | [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | A-OSS · MIT | Ascend A3/A5；FUSED_DEEP_MOE 为 dispatch+2×GMM+activation+combine 单 AscendC kernel，DISPATCH_FFN_COMBINE 进一步包含 routing 与 AllToAll/HCCL。 |
| [sglang-jax](distributed-moe/sglang-jax) | [sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax) | A-OSS · Apache-2.0 | TPU v6e/v7 的 Pallas FusedEPMoE；一次调用覆盖 routing、A2A scatter、expert FFN、A2A gather、可选 shared expert 和累加。v1 谱系来自 vLLM TPU inference。 |
| [TeraMoE](distributed-moe/TeraMoE) | [PFCCLab/TeraMoE](https://github.com/PFCCLab/TeraMoE) | A-OSS · MIT 为主 | SM100 跨节点 MoE 训练；一个 cooperative persistent kernel 内设 dispatch、scheduler、compute、combine、gather 五类 worker。NVSHMEM/DeepEP 派生文件保留各自许可。 |
| [Triton-distributed](distributed-moe/Triton-distributed) | [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) | A-OSS · MIT 为主 | 含 Qwen3 TP MegaTritonKernel、DITRON task-level scoreboard、以及 EP 两段大核示例；NVIDIA/AMD。UniEP 论文与该栈相关，但没有可单独核验的完整独立 artifact。 |

## 4. 子图与算子级大核（12）

| 本地工件 | 上游 | 标记 | 判断 |
|---|---|---|---|
| [ads_model_kernel_library](operator-scale/ads_model_kernel_library) | [facebookresearch/ads_model_kernel_library](https://github.com/facebookresearch/ads_model_kernel_library) | B-OSS · Apache-2.0 | gdpa_megakernel 是 Blackwell/TLX 广义 attention 大核；整仓是广告模型 kernel library，不是 whole-model runtime。 |
| [cherimoya](operator-scale/cherimoya) | [jmschrei/cherimoya](https://github.com/jmschrei/cherimoya) | B-OSS · MIT | 生物序列模型的 Cheri block；no-grad 路径把 conv、norm、MLP 等折叠成 Triton inference megakernel。跨领域 block-fusion 案例。 |
| [ClusterFusion](operator-scale/ClusterFusion) | [xinhao-luo/ClusterFusion](https://github.com/xinhao-luo/ClusterFusion) | S-SOURCE · 无根许可证 | H100/RTX 5090 的 QKV、decode attention、output projection cluster-level 深融合；attention stage，不是整模型。 |
| [deepfusionkernel](operator-scale/deepfusionkernel) | [ZixiBenZhang/deepfusionkernel](https://github.com/ZixiBenZhang/deepfusionkernel) | S-SOURCE · 无根许可证 | Triton 单核 SwiGLU MLP/SGLang 集成；vendored SGLang 文件的 Apache 头不能代表根仓授权。 |
| [flashformer](operator-scale/flashformer) | [cheetah-lang/flashformer](https://github.com/cheetah-lang/flashformer) | C-PARTIAL · 无根许可证 | 论文目标是 whole Transformer forward 单核；公开仓仅 components、同步原语和测试，没有完整 whole-model runner。 |
| [fused-mlp-megakernels-blackwell](operator-scale/fused-mlp-megakernels-blackwell) | [HF flashrt artifact](https://huggingface.co/kernels/flashrt/fused-mlp-megakernels-blackwell) | S-SOURCE · 未声明许可证 | Blackwell fused MLP kernel 工件；可读源码不等于获开源授权。 |
| [megagdn-pto](operator-scale/megagdn-pto) | [huawei-csl/megagdn-pto](https://github.com/huawei-csl/megagdn-pto) | S-SOURCE · 无根许可证 | Ascend 单 launch 融合 chunk-GDN 六阶段，另有 KDA；是 Qwen3.5/3.6 prefill 的层级大核，不是整模型。 |
| [mlp-megakernel](operator-scale/mlp-megakernel) | [Marcelo5444/mlp-megakernel](https://github.com/Marcelo5444/mlp-megakernel) | B-OSS · MIT | Triton/cuTile 把 3/5-layer Softplus MLP 的 forward/backward 融为单 launch。 |
| [qwen-tts-turbo](operator-scale/qwen-tts-turbo) | [Imtoocompedidiv/qwen-tts-turbo](https://github.com/Imtoocompedidiv/qwen-tts-turbo) | B-OSS · MIT | Qwen-TTS predictor 的五层 Transformer 子图大核，替代约数十次 launch；不是完整 TTS/LLM。 |
| [smallm-ffn-megakernels-blackwell](operator-scale/smallm-ffn-megakernels-blackwell) | [HF flashrt artifact](https://huggingface.co/kernels/flashrt/smallm-ffn-megakernels-blackwell) | S-SOURCE · 未声明许可证 | Blackwell small-M FFN 大核工件；按 source-visible 记录。 |
| [sonic-moe](operator-scale/sonic-moe) | [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) | B-OSS · Apache-2.0 | CuTe DSL/Triton 训练 MoE operator 与 persistent tile scheduler；不是完整跨节点 dispatch+FFN+combine 单核。 |
| [TIRx-kernels](operator-scale/TIRx-kernels) | [mlc-ai/TIRx-kernels](https://github.com/mlc-ai/TIRx-kernels) | B-OSS · Apache-2.0 | 含 DeepGEMM SM100 MegaMoE 的 TIRx 完整移植，以及 collective+GEMM/persistent GEMM；实现栈独立但算法谱系同源，不当作第二个原创算法。 |

## 5. 基础设施与直接依赖（22）

这些项目解释了 MegaKernel 如何被编译、同步、通信和集成，但除明确子路径外，不能把整仓都算作 MegaKernel 库。

| 本地工件 | 上游 | 开放状态 | 与 MegaKernel 的关系 |
|---|---|---|---|
| [ark](foundations/ark) | [microsoft/ark](https://github.com/microsoft/ark) | MIT | GPU-driven loop kernel 执行分布式 compute/communication；现代设备内运行时的重要前史。 |
| [cusync](foundations/cusync) | [microsoft/cusync](https://github.com/microsoft/cusync) | MIT · 已归档 | 对不同 kernel 做 tile-level dependency/synchronization；仍是多 kernel，不等于单 MegaKernel。 |
| [DeepEP](foundations/DeepEP) | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | MIT | EP dispatch/combine 通信底座；FFN 不在同一 kernel 时不是 MegaMoE。 |
| [FastMoE](foundations/FastMoE) | [laekov/fastmoe](https://github.com/laekov/fastmoe) | Apache-2.0 | 早期分布式 MoE 框架；模块化 kernel，不是单常驻核。 |
| [flux](foundations/flux) | [bytedance/flux](https://github.com/bytedance/flux) | Apache-2.0 | collective/GEMM 细粒度重叠库；是通信计算 fusion 基元，不是 whole-model runtime。 |
| [MoonEP](foundations/MoonEP) | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) | MIT | H20 zero-copy EP 通信/permute 库；expert FFN 外置，因此不算 MegaMoE。 |
| [neuronx-distributed-inference](foundations/neuronx-distributed-inference) | [aws-neuron/neuronx-distributed-inference](https://github.com/aws-neuron/neuronx-distributed-inference) | Apache-2.0 | AWS 官方 NxDI 框架接线，调用 NKI attention/transformer megakernel；框架整体不等于单核。 |
| [pto-kernels](foundations/pto-kernels) | [huawei-csl/pto-kernels](https://github.com/huawei-csl/pto-kernels) | Clear BSD | Ascend PTO primitive/算子库；MegaGDN 是另仓，不能把本仓整体当 MegaKernel。 |
| [PyPTO](foundations/PyPTO) | [CANN/pypto](https://gitcode.com/cann/pypto) | CANN OSL v2.0 | Ascend Tensor/MPMD 图到设备调度的编译底座；现有 GLM/Qwen 子路径多为局部 fusion，不是完整模型大核。 |
| [quack](foundations/quack) | [Dao-AILab/quack](https://github.com/Dao-AILab/quack) | Apache-2.0 | CuTe DSL kernel 集合、persistent/grouped GEMM 基元；本身不是通用 MegaKernel runtime。 |
| [RustCompute](foundations/RustCompute) | [mivertowski/RustCompute](https://github.com/mivertowski/RustCompute) | Apache-2.0 | Rust GPU compute/task abstraction 实验，列为运行时设计参考。 |
| [sentinel-comm](foundations/sentinel-comm) | [DATGMAC/sentinel-comm](https://github.com/DATGMAC/sentinel-comm) | MIT | 小型 persistent CPU→GPU command bus；说明设备常驻控制面，但不构成模型 MegaKernel。 |
| [sinter](foundations/sinter) | [slartibardfast/sinter](https://github.com/slartibardfast/sinter) | MIT | 为手写/生成式模型 kernel 提供调度与代码生成底座；具体 qwen36 工件另列。 |
| [syncopate](foundations/syncopate) | [tie-pilot-qxw/syncopate](https://github.com/tie-pilot-qxw/syncopate) | MIT | Triton source-to-source chunk-centric compute/communication overlap compiler；重要相邻编译路线，通常仍输出多 kernel。 |
| [ThunderKittens](foundations/ThunderKittens) | [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | MIT | tile primitive、persistent grid 和 PGL；Hazy whole-model megakernel 的直接底座。 |
| [ThunderKittens-AMD-port](foundations/ThunderKittens-AMD-port) | [amdpilot-org/ThunderKittens](https://github.com/amdpilot-org/ThunderKittens) | MIT | AMD port/fork 谱系；不与上游算两个独立 MegaKernel runtime。 |
| [tilelang](foundations/tilelang) | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | MIT | Python tile DSL、PipeThreader/sTask 编译底座，支持 NVIDIA/AMD；整仓不是 whole-model MegaKernel。 |
| [tilelang-ascend](foundations/tilelang-ascend) | [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | MIT | Ascend tile DSL 与共享内存通信基础设施。 |
| [TileOPs](foundations/TileOPs) | [tile-ai/TileOPs](https://github.com/tile-ai/TileOPs) | MIT | SM90 operator library；是优化算子集合，不应因 tile/task 概念自动纳入核心。 |
| [TileScale](foundations/TileScale) | [tile-ai/tilescale](https://github.com/tile-ai/tilescale) | MIT | kernel-side communication、AG+GEMM/GEMM+AR/RS 等分布式 tile 基础设施。 |
| [torchair](foundations/torchair) | [Ascend/torchair](https://github.com/Ascend/torchair) | BSD-3-Clause | TorchAir SuperKernel 的编译与 scope 基础；将 child kernels 组合成大 binary，语义不同于单 persistent source kernel。 |
| [vdcores](foundations/vdcores) | [vdcores/vdcores](https://github.com/vdcores/vdcores) | 无根许可证 | resource-isolated virtual cores + dependency micro-op runtime；是反对单体 orchestrator 的重要替代设计。 |

## 6. 相邻与替代执行路线（5）

| 本地工件 | 上游 | 标记 | 为什么不与经典 MegaKernel 混计 |
|---|---|---|---|
| [hpc-ops](alternatives/hpc-ops) | [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops) | B-OSS · MIT 为主 | FusedMoE 目前用 PDL 异步串联 count/gather、两次 grouped GEMM、activation、reduce 等多个 kernel；README roadmap 仍把 Megakernel 列为未来项。 |
| [pdl-megakernel-reconstruction](alternatives/pdl-megakernel-reconstruction) | [tie-pilot-qxw/pdl-megakernel-reconstruction](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction) | B-OSS · MIT | 用 CUDA Programmatic Dependent Launch/约 81 个 kernel 重构 Hazy 调度效果；host overhead 很低但仍非单 kernel。 |
| [Primus-Turbo](alternatives/Primus-Turbo) | [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) | B-OSS · MIT 为主 | MI355X Mega MoE 实际是 dispatch+GEMM 和 GEMM+combine 两枚 FlyDSL 融合核，中间另有 SwiGLU；是强相邻方案而非整层单核。 |
| [SGLang-FluentLLM-NPU](alternatives/SGLang-FluentLLM-NPU) | [meituan-longcat/SGLang-FluentLLM npu](https://github.com/meituan-longcat/SGLang-FluentLLM/tree/npu) | B-OSS · Apache-2.0 | LongCat 代码接入 TorchAir SuperKernel scope/stream-fusion；child binary 组合且 unsupported op 可拆段，官方脚本也非默认启用。 |
| [StreamEP](alternatives/StreamEP) | [evolutionaryscale/StreamEP](https://github.com/evolutionaryscale/StreamEP) | B-OSS · MIT | 多 persistent kernels + 两条 stream 做 tile streaming；设计目标相同，但不满足单设备执行底座定义。 |

## 7. 长尾、教学、fork 与需谨慎项目（9）

| 本地工件 | 上游 | 标记 | 核验结论 |
|---|---|---|---|
| [flint](long-tail-experimental/flint) | [asmit383/flint](https://github.com/asmit383/flint) | S-SOURCE · 无根许可证 | H100/Granite-4.1-3B 的真实 int4 whole-decode megakernel，并叠加 speculative decode；技术上属于核心整模工件，但因无许可证和项目成熟度放在观察区。 |
| [gemma3-mega-kernel-npu](long-tail-experimental/gemma3-mega-kernel-npu) | [joeldushouyu/gemma3-mega-kernel-npu](https://github.com/joeldushouyu/gemma3-mega-kernel-npu) | S-SOURCE · 无根许可证 | 没有根 README，包含 Gemma3/NPU/AIE/MLIR 实验与大量子模块线索；未形成可核验的完整 MegaKernel 发布面。 |
| [gpt2-megakernel](long-tail-experimental/gpt2-megakernel) | [herocharge/gpt2-megakernel](https://github.com/herocharge/gpt2-megakernel) | D-BOUNDARY | README 信息极少的 GPT-2 个人实验；保留作长尾样本，不列成熟方案。 |
| [grokking-megakernels](long-tail-experimental/grokking-megakernels) | [Infatoshi/grokking-megakernels](https://github.com/Infatoshi/grokking-megakernels) | S-SOURCE · 无根许可证 | 教材 companion，重构 MegaQwen、qwen_megakernel、Hazy 思路；有完整教学代码但不是独立原创 lineage。 |
| [jiange91-megakernel](long-tail-experimental/jiange91-megakernel) | [jiange91/megakernel](https://github.com/jiange91/megakernel) | D-BOUNDARY | Triton 大型 fork/开发副本，根 README 仍是 Triton 通用说明；缺乏独立 MegaKernel 项目边界。 |
| [llama.cpp-megakernel-0.8b](long-tail-experimental/llama.cpp-megakernel-0.8b) | [sandeshrajbhandari/llama.cpp-megakernel-0.8b](https://github.com/sandeshrajbhandari/llama.cpp-megakernel-0.8b) | B-OSS · MIT lineage | llama.cpp 实验 fork，加入 Qwen3.5-0.8B single-block persistent fused kernel；体量大、模型路径实验性。 |
| [Llamagen_megakernels](long-tail-experimental/Llamagen_megakernels) | [ighoshsubho/Llamagen_megakernels](https://github.com/ighoshsubho/Llamagen_megakernels) | D-BOUNDARY | LlamaGen 图像生成项目的 fork；根文档未说明独立 MegaKernel 边界，需按 fork/实验看待。 |
| [onelaunch](long-tail-experimental/onelaunch) | [haregali/onelaunch](https://github.com/haregali/onelaunch) | S-SOURCE · 无根许可证 | 名称称 fused decode，但真实边界是“一组 Triton kernel + 一次 CUDA Graph replay”；是降低 launch gap 的替代路线，不是单 kernel。 |
| [TensorSharp2](long-tail-experimental/TensorSharp2) | [SciSharp/TensorSharp2](https://github.com/SciSharp/TensorSharp2) | B-OSS · 多后端项目 | .NET/GGUF 推理引擎，含 CUDA Graph 与多模型优化；未发现统一跨算子 persistent MegaKernel 路径，属于搜索长尾/误命中。 |

## 8. 目录与课程资料（2）

| 本地工件 | 上游 | 用途 |
|---|---|---|
| [Awesome-MegaKernel](catalogs/Awesome-MegaKernel) | [qhy991/Awesome-MegaKernel](https://github.com/qhy991/Awesome-MegaKernel) | CC0 目录、论文表、中文综述与案例研究；截至 2026-08-14，是很好的基线，但本归档补入了其后公开或尚未收录的 Machete、Fleet 源码、DeepGEMM MegaMoE、FastAFD、TeraMoE、SGLang NPU/TPU 等。 |
| [gpu-megakernel-course-art](catalogs/gpu-megakernel-course-art) | [qhy991/gpu-megakernel-course-art](https://github.com/qhy991/gpu-megakernel-course-art) | 课程/图示资料，用于术语和教学，不计作运行库。 |

## 9. Agentic kernel design、知识库与评测（22）

本组是构建 KernelWiki、可执行 Skill 和数据飞轮的方法参考，不按 A–D MegaKernel 实现标记计数，也不改变前 85 项的实现图景结论。

| 本地工件 | 上游 | 本次采用的设计价值 |
|---|---|---|
| [kernel-design-agents](agentic-kernel-design/kernel-design-agents) | [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) | KDA 的 task contract、候选迭代、可执行验证与 promote/reject 工作流。 |
| [KernelWiki-upstream](agentic-kernel-design/KernelWiki-upstream) | [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) | `sources → wiki → queries` 的知识组织、受控词表与证据页结构。 |
| [ncu-report-skill](agentic-kernel-design/ncu-report-skill) | [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | 将 profiler 原始结果压缩成可行动诊断的 Skill 范式。 |
| [KernelBench](agentic-kernel-design/KernelBench) | [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch reference、生成 contract、correctness gate 与 `fast_p` 基线。 |
| [kernel_bench_verified](agentic-kernel-design/kernel_bench_verified) | [facebookresearch/kernel_bench_verified](https://github.com/facebookresearch/kernel_bench_verified) | hidden values、真实基线、内存 gate 与防输入特化的 evaluator 修正。 |
| [KernelBenchX](agentic-kernel-design/KernelBenchX) | [BonnieW05/KernelBenchX](https://github.com/BonnieW05/KernelBenchX) | 按 kernel 类别与硬件分析 correctness/efficiency 泛化差异。 |
| [robust-kbench](agentic-kernel-design/robust-kbench) | [SakanaAI/robust-kbench](https://github.com/SakanaAI/robust-kbench) | 大规模 compile/correctness/runtime/profile 候选轨迹，可用于 repair 与 preference。 |
| [SOL-ExecBench](agentic-kernel-design/SOL-ExecBench) | [NVIDIA/SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | Definition/Workload/Solution/Trace 分离、动态 workload 与 speed-of-light gap。 |
| [K-Search](agentic-kernel-design/K-Search) | [caoshiyi/K-Search](https://github.com/caoshiyi/K-Search) | 显式 hypothesis/search tree，把设计意图与单次实现结果分离。 |
| [atrex-kernel-agent](agentic-kernel-design/atrex-kernel-agent) | [alibaba/atrex-kernel-agent](https://github.com/alibaba/atrex-kernel-agent) | production trace 权重、mechanical evaluator、structured memory 与 strategy/anti-strategy 蒸馏。 |
| [KernelAgent](agentic-kernel-design/KernelAgent) | [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) | profiler、judge、analyzer、orchestrator 分工以及多候选协作。 |
| [CUDA-Agent](agentic-kernel-design/CUDA-Agent) | [BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) | task synthesis、skill-augmented environment、correctness-gated reward 与长程训练。 |
| [KernelGYM](agentic-kernel-design/KernelGYM) | [hkust-nlp/KernelGYM](https://github.com/hkust-nlp/KernelGYM) | 多轮 trajectory、编译/正确性/profile/runtime 反馈与 GPU worker 环境。 |
| [daVinci-kernel](agentic-kernel-design/daVinci-kernel) | [GAIR-NLP/daVinci-kernel](https://github.com/GAIR-NLP/daVinci-kernel) | skill selection/policy/summary 共训及执行环境中的经验复验。 |
| [kernelfoundry](agentic-kernel-design/kernelfoundry) | [isl-org/kernelfoundry](https://github.com/isl-org/kernelfoundry) | MAP-Elites 式多样性 archive、negative transition 和搜索空间分层。 |
| [CudaForge](agentic-kernel-design/CudaForge) | [OptimAI-Lab/CudaForge](https://github.com/OptimAI-Lab/CudaForge) | raw profile 与 compact bottleneck object 分离、反馈通道归因。 |
| [Apex](agentic-kernel-design/Apex) | [AMD-AGI/Apex](https://github.com/AMD-AGI/Apex) | 面向 AMD GPU 的 agentic kernel 优化、工具闭环与任务组织参考。 |
| [AgentKernelArena](agentic-kernel-design/AgentKernelArena) | [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | kernel agent 竞技/评测环境与跨候选比较参考。 |
| [FACT](agentic-kernel-design/FACT) | [Project-FACT/FACT](https://github.com/Project-FACT/FACT) | 编译器反馈驱动的生成、纠错和可验证任务构造参考。 |
| [kernel-opt-agent](agentic-kernel-design/kernel-opt-agent) | [fmh66/kernel-opt-agent](https://github.com/fmh66/kernel-opt-agent) | 轻量 kernel optimization agent 与迭代工件布局参考。 |
| [mlsys2026-flashinfer-contest](agentic-kernel-design/mlsys2026-flashinfer-contest) | [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) | 真实 serving kernel 任务、提交 contract 与比赛 evaluator 参考。 |
| [fastkernels](agentic-kernel-design/fastkernels) | [Snowflake-AI-Research/fastkernels](https://github.com/Snowflake-AI-Research/fastkernels) | production-shape kernel 优化样本和现实 workload/evaluator 参考。 |

## 10. 已知重复与不应误计的谱系

- Luce-Org/luce-megakernel 与 Lucebox：重定向/同谱系。
- model-as-a-kernel 与 det-infer：同一设备解释器的变体。
- TIRx MegaMoE 与 DeepGEMM MegaMoE：实现栈不同、算法移植关系明确。
- FastAFD 内 vendored DeepGEMM/DeepEP：独立贡献是 M2N/AFD 角色路径，vendored 文件不再计一套方案。
- ThunderKittens AMD port、Hazy 的零散 fork、Triton/llama.cpp 大型 fork：按上游 lineage 标注。
- MegaBlocks、FastMoE、DeepEP、MoonEP：名字或 persistent 通信容易误命中，但整仓不是 MegaKernel。
- CUDA Graph、PDL、TorchAir SuperKernel：都是重要的“低 host overhead”替代方案，但其 kernel 边界不同。

## 11. 仅论文/报告，未找到可核验官方源码

| 工作 | 公开材料 | 当前状态 |
|---|---|---|
| Event Tensor | [MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html) | 动态 event-tensor IR；TVM/TIRx 仅称未来集成，未发现官方实现。 |
| Ada-MK | [arXiv 2605.11581](https://arxiv.org/abs/2605.11581) | 百度广告系统/MLIR DAG search/TRT-LLM decode plugin；未发现百度官方仓库。 |
| ExpertPlex | [arXiv 2607.18002](https://arxiv.org/abs/2607.18002) | disaggregated MoE serving/adaptive persistent kernel；未找到作者代码。 |
| RaMP | [arXiv 2604.26039](https://arxiv.org/abs/2604.26039) | routing-histogram-aware kernel polymorphism；未找到作者代码。 |
| FlashFuser | [arXiv 2512.12949](https://arxiv.org/abs/2512.12949) | Hopper DSM 深融合 compiler；未找到作者仓库。 |
| ClusterFusion++ | [arXiv 2604.23553](https://arxiv.org/abs/2604.23553) | 完整 Transformer block cluster kernel；未发现正式代码。 |
| UniEP 独立 artifact | [arXiv 2604.19241](https://arxiv.org/abs/2604.19241) | 与 Triton-distributed 技术栈相关，但没有可单独核验的完整实现目录。 |
| 多节点 proxy/RDMA fence 工作 | [arXiv 2605.00686](https://arxiv.org/abs/2605.00686) | FlashMoE 通信基础研究，未发现独立开源实现。 |

工业公开思路但未开放核心源码的例子包括 [Kog.ai MI300X single-kernel engine](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)。NVIDIA cuDNN frontend 的 [MegaMoE issue/roadmap](https://github.com/NVIDIA/cudnn-frontend/issues/442) 也不应写成已发布实现。
