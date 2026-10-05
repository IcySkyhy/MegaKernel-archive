# ascend_mega_kernel

面向昇腾 950（Ascend950PR）的 mega kernel 开发：在单卡 950PR + CANN 9.1.0 上探索大算子融合与持久化 kernel 技术，目标是对 Qwen3.8-Flash-Next-MXFP4 做高性能推理。

## 文档索引

| 文档 | 说明 |
|---|---|
| [docs/01-environment.md](docs/01-environment.md) | 本机昇腾环境调研：NPU 硬件、CANN/驱动/ATB 版本、Host 平台、Python 环境、环境变量与工具清单、风险建议 |
| [docs/02-workspace-repos.md](docs/02-workspace-repos.md) | /workspace 仓库与模型目录盘点：六个 CANN 官方算子/知识仓、vLLM 与 vllm-ascend、目标模型 Qwen3.8-Flash-Next-MXFP4、shared_assets |
| [docs/03-ascend950-megakernel.md](docs/03-ascend950-megakernel.md) | 昇腾 950 架构（白皮书/keynote 三级来源标注）、AscendC 编程模型、mega kernel 在昇腾与 CUDA 生态的实践对照 |
| [docs/04-summary.md](docs/04-summary.md) | 探索总结与后续建议：环境现状、关键缺口与风险、技术路线判断、最有价值的现有资产、下一步行动 |
| [docs/05-megakernel-design.md](docs/05-megakernel-design.md) | Mega kernel 设计方案 v1.2：per-layer kernel、MIX_AIC_1_2 全核、BufferID/CrossCore 同步约束、op 模块化契约、硬件依据与实证结论 |
| [docs/09-moe-donor-map.md](docs/09-moe-donor-map.md) | MoE donor 代码图谱：逐 op 可抄文件/函数/行号 + 可抄度评级（M5 survey） |
| [docs/10-gdn-analysis.md](docs/10-gdn-analysis.md) | GDN 线性注意力分析：数学/访存/prefill chunk 扫描/donor/实现建议（M6 survey，36/48 层） |
| [docs/11-attn-analysis.md](docs/11-attn-analysis.md) | Full-attention 层分析：QSA 稀疏注意力发现/FA donor/KV cache paged 设计/资源预算（M7 survey，12/48 层） |
| [docs/12-layer-integration.md](docs/12-layer-integration.md) | Layer kernel 集成设计：算件接口盘点/GM 平面图/全局资源表草案/同步图/小改清单（M18 survey；**M38 按 docs/14 修订 hyper-connection 层界语义、资源表与验收口径**） |
| [docs/13-mx-quant-primitives.md](docs/13-mx-quant-primitives.md) | MXFP4 量化原语与官方实现逐句对照：e8m0 cast 的真实位置、官方 scale/数据转换序列、量化原语总表、与我们的 15 步差异表（M28 survey） |
| [docs/14-hyperconnection-ple-indexer-spec.md](docs/14-hyperconnection-ple-indexer-spec.md) | hyper-connection/PLE/QSA indexer 规格：延迟 combine 结构、每 token 带宽账、PLE 索引算术 bit-exact、QSA 接入点与两个地雷（M31 survey） |
| [docs/15-prefill-design.md](docs/15-prefill-design.md) | prefill m=4097 契约与设计：chunked scan BT=64、prefill/decode 瓶颈相反、共用方案 B、14 条不返工清单、vllm-ascend 接入点（M33 survey） |
| [docs/16-vllm-ascend-qwen4exp-plan.md](docs/16-vllm-ascend-qwen4exp-plan.md) | vllm-ascend 承载 qwen4_exp 的接入方案：版本/可行性裁决、950PR 硬约束、接入点地图（模型注册/QSA/MoE/GDN/hc/PLE/量化）、per-layer 接入形态与最小框架改动清单、MVP 七步、基线口径（M37 survey，**原文 700 行逐字节落库**；**文末 A–D 节为塔台后加**：更正注、上游 `rfc/megakernel` 分支与 PR #15986 现状、厂商镜像取证） |
| [docs/17-verification-standard.md](docs/17-verification-standard.md) | **统一数值验收口径**（全仓库强制）：L0 算子 / L1 段 / L2 层与整网三级标准、判据写法规范、五条反例纪律、非空洞性清单 |
| [docs/18-vector-api-audit.md](docs/18-vector-api-audit.md) | **向量 API 审计**（M44 只读审计，M49 落库）：逐模块判定"计算类是否走 `__VEC_SCOPE__`/register-based"、无寄存器等价物的边界清单（`Sort32`/`MrgSort` 例外、`Extract` 可转）、存量 (a) 清单与改造成本、官方 donor 侧取证 |
| [docs/19-gdn-prefill-scan-selection.md](docs/19-gdn-prefill-scan-selection.md) | GDN prefill chunk 扫描选型：把 m18 的 15.51 ms/层 拆开、与 1.37 ms/层 对齐复算（M104 survey；**"矩阵乘法一律 mmad"的人类裁决覆盖其 §4/§5 原有的"先 VF 后 mmad"排法**） |
| [docs/20-kernel-compliance-sweep.md](docs/20-kernel-compliance-sweep.md) | **M15 段体合规清盘**：矩阵乘法（mmad）清单 + 同族同步竞态（标量写 + 次序 token / 标量落 GM）（M107 只读清盘，落库供塔派活） |

> 探索日期：2026-09-26。

## 工程目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `m0/` | M0 bring-up：mix(1,2) 全核启动 + BufferID/CrossCore 最小验证 | ✅ |
| `m1_mxfp4_gemm/` | 单核 MXFP4 GEMM（basic API 全链路，10 用例逐位 PASS） | ✅ |
| `m2_mxfp4_quant/` | MXFP4 激活量化 AIV kernel（对 quant_ref.py 三层链逐字节一致） | ✅ |
| `m3_grouped_gemm/` | 多核 grouped MXFP4 GEMM（28 AIC 专家槽位×N 切分，mode2+mode0+mode2） | ✅ |
| `m4_gdn_recurrent/` | GDN decode 递推核（48 head，寄存器 VF，state 预取流水） | ✅ |
| `m5_swiglu_quant/` | SwiGLU + MXFP4 量化融合 epilogue | ✅ |
| `m6_rmsnorm/` | Add+RMSNorm 残差融合核（**层边界语义已被 hyper-connection 取代**，见 [docs/14](docs/14-hyperconnection-ple-indexer-spec.md) §8.2：层界 norm 是 `hc_norm [10240]` 4 组×2560 分组 Gemma-RMSNorm；其归一化数学仍可复用为 mixer 子步骤） | ✅ |
| `m7_router_topk/` | Router softmax+top-10 核（Sort32+2路MrgSort，m=1 实测 0.066ms） | ✅ |
| `m8_permute/` | MoE permute/unpermute 核（专家槽位重排，golden 逐位一致） | ✅ |
| `m9_gdn_prolog/` | GDN conv1d+l2norm+gating 融合预处理核（与 m4 递推核契约对齐） | ✅ |
| `m11_bf16_gemm/` | bf16 GEMM（GDN in/out_proj 用，非 MX 路径） | ✅ |
| `m12_rmsnorm_gated/` | RMSNormGated（GDN 输出门控+归一化，per-head 128 维） | ✅ |
| `tools/golden/` | MXFP4 打包/解包 + MoE block golden 数据集（m=1/m=33） | ✅ |
| `tools/weights/` | 真实 checkpoint safetensors 读取 + MoE 权重切片 | ✅ |
| `m13_moe_layer/` | **MoE block 整层 kernel**（mix(1,2) 单次启动跑全链 S1-S10，682 判据 + 118 项 numpy 独立校验全 PASS） | ✅（层界 I/O 待改） |
| `m14_gdn_layer/` | **GDN 整层 kernel**（S1-S7 单次 mix 启动，114 kernel 判据 + 103 numpy 校验 PASS） | ✅（层界 I/O 待改） |
| `m15_layer_loop/` | **48 层循环骨架**（host 循环，每层一次 `__mix__(1,2)` 启动）：36 层 GDN 直接复用 m14 整层 kernel，12 层 full attention 暂为**占位直通**；层间只经 GM bf16 残差流交接，conv/ssm state 常驻 GM。**已并入 main；待接 MoE 段（M40 在建）与 hc 层边界（M36 在建）** | ✅（注意力/GDN 子层链骨架：**本循环 ≠ 整层**，MoE FFN 段与 hc 尚未接入；attention 段待 QSA，判据分账口径见其 README §3.6） |
| `probe_sync_quirks/` | 同步/sort/下标类问题常备复现器（MrgSort 26 变体 + VF idx 矩阵 + 证据归档） | ✅ |

> 上表"层界 I/O 待改"的口径：m6/m13/m14 的 ✅ 指**其自身判据**（锚在 device 字节上，与层界语义无关）仍全部 PASS；但 checkpoint 的真实**层边界结构**是 hyper-connection 的延迟 combine —— 层界残差流是 `[m,10240]`（4×2560）而非 `[m,2560]`，跨层携带 3 个张量，`m6` 不再是层边界 op。判定依据与改造清单见 [docs/14](docs/14-hyperconnection-ple-indexer-spec.md) §8.2 与 [docs/12](docs/12-layer-integration.md) §8.1。
