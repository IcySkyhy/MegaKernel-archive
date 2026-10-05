# 探索总结与后续建议

> 总结日期：2026-09-26。基于同日完成的四份调研（[环境调研](01-environment.md)、[workspace 仓库调研](02-workspace-repos.md)、[950 架构与 mega kernel 技术调研](03-ascend950-megakernel.md)）整理。

## 环境现状

本机为**单卡 Ascend950PR**（板卡型号 A310-50-C00MM304A1，Atlas 950 系列，1P 单芯片封装），128GB HBM（npu-smi 显示 28 个 AI Core，Aicore 频率 1650MHz），Driver/npu-smi 25.7.rc1.6 + **CANN 9.1.0**（内置 bisheng 编译器，clang 15.0.5）+ ATB（nnal）9.1.0.B150。Host 为 2×AMD EPYC 9355（64 核 128 线程）+ openEuler 24.03 SP3，Python 3.12.13（`/usr/local/python3.12.13`，pip 26.2，62 个包全为 CANN 生态，含 modelscope、superkernel、cann_ops_transformer）。环境变量已通过 `/root/.bashrc` 自动加载（ATB 按 cxx_abi=1 部署）。

**容器资源限制（cgroup v1，agent 必读）**：本容器内存上限 **32GB**（memsw 64GB；实测 `memory.usage_in_bytes` ≈ 29.8G/32G，其中 rss 仅 ~1GB、其余为可回收 page cache，大编译需限并行防 OOM）；磁盘平台配额 **总共 300GB**（`free`/`df` 显示的宿主机 754GB 内存/8.7T 磁盘**不可信**，均为宿主视角）。**2026-09-26 复核**：模型 170G 已完整落在 `/workspace` 且容器未受限，故 300GB 配额要么另有口径、要么不含共享存储——写盘仍按「先 `df -h` + `du -sh`」的纪律执行。详见 [01-environment.md](01-environment.md) §3.1。

## 关键缺口与风险

1. **torch / torch_npu / vllm 均未安装**：需按 CANN 9.1.0 对应矩阵安装；注意本机 ATB 按 cxx_abi=1 部署（torch_npu 需选匹配 ABI 的版本），且推理框架要求 Python 3.12（默认 `python3` 为 3.11.6 且无 pip，装包须用 `/usr/local/python3.12.13/bin/pip3`）。当前 `numpy` 在默认解释器下亦不可用（CUDA/CPU 参考实现要么自带纯 Python 实现、要么先装 numpy）。
2. **资源硬限制**：容器内存 cgroup 上限 **32GB**（编译需限并行）。**权重按层流式处理没有内存压力**：单层 ~1.4 GiB（MoE 专家权重 1275 MiB 是主体，最大单张量 800 MiB），但整模型 169.7 GiB 既放不进 32GB host 内存、也放不进 128 GiB HBM，必须逐层 pread/mmap + 双缓冲。**例外**：PLE ngram 表 128 分片 / **95.4 GiB**（占整模型 56%）只能稀疏行查找 + 磁盘驻留。磁盘平台配额标注 **300GB**；模型实测 **170G / 131 分片**（此前按 3.2GB×131 估成 ~400GB 偏高）已完整落地 `/workspace`，`df` 实测可用 473G。
3. **vllm 与 vllm-ascend 版本错配**：/workspace/vllm 检出为 main（v0.30.1rc0-189，2026-09-26），而 vllm-ascend fork（分支 qwen3.8-950 / tag v0.23.0+ascend950）要求 vLLM v0.23.0（pin commit 0fc695fc）并附带 `patches/vllm-v0.23.0-ascend950.patch`（GDN 层 `fuse_in_proj_qkvzba`，当前未应用）。需决策：切 vllm 到 v0.23.0，或验证 main 分支与插件 v0.23.0 基线的兼容性。**尚未决策**。
4. **模型已下载完成**（2026-09-26 复核）：131 个 `.safetensors`（170G）+ `model.safetensors.index.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`chat_template.jinja`、`validation_report.json` 全部就位，无 `.incomplete`，下载进程已退出 ⇒ **软硬件齐备，只剩软件栈安装与 kernel 装配**。
5. **AI Core 数量口径差异**：npu-smi 显示 28 个 AI Core，而《昇腾950 NPU 架构白皮书》记载 36 个 AI 子系统（36 Cube + 72 Vector），存在差异，待核实（可能为板卡裁剪或统计口径不同）。

## 目标模型要点

`lenlrx/Qwen3.8-Flash-Next-MXFP4`：`architectures = ["Qwen4ExpForConditionalGeneration"]`（model_type `qwen4_exp`，上游 vLLM main 已原生支持），48 层——36 层 GDN 线性注意力 + 12 层全注意力（3:1 交替）；MoE 512 专家 top-10 + 1 个共享专家；hidden_size 2560，注意力 24 头 / 2 KV 头 / head_dim 256，GDN 线性注意力 16 key 头 ×128 + 48 value 头 ×128。**仅 MLP 专家/共享专家做 MXFP4 量化**（quant_method="ascend"，format `mxfp4-pack-quantized-e8m0`，e2m1 + e8m0 scale、group_size 32、uint8 nibble 打包），注意力/ngram embedding/MTP 保持 bf16；vocab 248320，MTP 1 层（投机解码草稿）。safetensors 头验证与 config 一致（每专家 down_proj U8 [512, 2560, 320]，scale 20 个 e8m0/行）。

## Mega kernel 技术路线判断

- **官方路径 = torchair SuperKernel**（算子二进制融合，图级）：torchair 图模式 `scope.super_kernel` 把范围内算子编译为一个大 Kernel 顺序调用 + 自动插同步，支持 A2/A3、可融合 AllReduce 等通信算子；DeepSeek-V3 61 层中 58 层融合为 1 个 SuperKernel，**整网收益 10–20%**。
- **自定义路线 = AscendC 大算子**：CANN 9.x `KERNEL_TASK_TYPE`（`KERNEL_TYPE_MIX_AIC_1_2` 混合核调度，1:2 Cube/Vector 配比）+ CV 融合通道（Cube↔Vector 核内直连）+ **RegBase 寄存器级链式向量计算**（中间结果不落 UB，多步 Cast/Mul/Add 融合收益最大）+ NDDMA 多维搬运 + BufferID 同步。
- **探索级 = device 侧跨核调度**：950 的 UB Memory（load/store/atomic，最高 128TB 共享内存语义）/ URMA / CCU 提供硬件原语，但**无公开的一等公民 persistent-kernel API**；公开资料中尚无基于它们的通用 persistent-kernel 框架案例。
- **CUDA 侧对照**（核心思想均为消除 kernel 间空隙、device 侧自调度、通信进 kernel）：HazyResearch ThunderKittens 生态的 H100/B200 Megakernel（单 kernel 整网 forward）、Mirage MPK（多 GPU 单 megakernel）、Cohere Megakernel（单卡 MoE decode serving）、Ada-MK（TRT-LLM MegaKernel 学术版）、AutoMegaKernel (AMK)、UniEP（MoE EP 通信 megakernel）、FlashInfer（激进单算子融合 + JIT 路线）。

## 对 mega kernel 开发最有价值的现有资产

| 资产 | 路径 | 价值 |
|---|---|---|
| MXFP4 matmul 调优样例（950PR/DT，CANN ≥ 9.1.0） | `/workspace/asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_mxfp4_high_performance/` | 与目标模型最直接相关的单文件参考（含 Basic API 版 `matmul_mxfp4_basic_api_high_performance/` 与 Tensor API 版） |
| 全量化 BatchMatmul（W4A8/MXFP4 MoE GEMM 生产级实现，含 arch35 目录） | `/workspace/ops-nn/matmul/quant_batch_matmul_v4/` | mega kernel GEMM 部分的直接对标对象 |
| MegaMoE（Dispatch+Linear1+Activation+Linear2+Combine 全流程单算子融合） | `/workspace/ops-transformer/mc2/mega_moe/` | 全 workspace 形态上最接近 "mega kernel" 的生产实现 |
| A5 MxFP4 全量化 attention | `/workspace/ops-transformer/experimental/attention/quant_flash_attn/`（`op_kernel/arch35/vf/vf_attenOut_dn_mxfp4.h` 等） | 量化 attention 内核参考 |
| 950 平台 369 张算子知识卡 | `/workspace/cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/`（命名 `*_950.md`） | 950 算子 API→Kernel 链路背景知识 |
| DeepSeek-V4 MXFP4 混合量化决策卡 | `/workspace/cannbot-knowledge/knowledge/model/inference/concepts/quantization/hybrid_mxfp8_mxfp4.md` | MXFP4 推理设计的核心参考 |
| 算子开发方法论技能（tiling/性能调优/Catlass 模板等） | `/workspace/cannbot-skills/ops/ascendc-perf-optimize/`、`ascendc-tiling-design/`、`catlass-op-design/`（develop/perf-tune）、`ascendc-regbase-best-practice/` | mega kernel 开发方法论基座 |
| MXFP4 GEMM 在线量化自定义算子 | `/workspace/vllm-ascend/csrc/online_mxfp4_gemm/`、`csrc/mxfp4_nz/` | vllm-ascend 侧 AscendC 自定义算子 |
| MXFP4 量化方法（W4A4/W4A8/W4A16） | `/workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a4_mxfp4.py`、`w4a4_mxfp4_dual.py`、`w4a4_mxfp4_flatquant.py`、`w4a16_mxfp4.py`、`w4a8_mxfp4.py` | 与模型 `quant_method="ascend"` 直接对应（`AscendW4A4MXFP4DynamicFusedMoEMethod`） |
| 融合 GDN 线性注意力内核 | `/workspace/vllm-ascend/vllm_ascend/ops/gdn.py`、`gdn_attn_builder.py` | Qwen3.8 linear_attention 层实现 |
| DFlash2 投机解码 | `/workspace/vllm-ascend/vllm_ascend/spec_decode/dflash2_proposer.py`、模型 `qwen3_dflash2.py` | MTP 草稿模型 + proposer |

## 建议的下一步

1. **安装推理软件栈**：按 vllm-ascend README 安装 PyTorch 2.10.0 + torch_npu 2.10.0.post4（注意 ATB cxx_abi=1 ABI 匹配、Python 3.12），并解决 vllm 版本错配（切 v0.23.0 应用 patch，或验证 main 分支兼容性）后安装 vllm / vllm-ascend。
2. **跑通基线推理**：模型已下载完成（131 分片 / 170G / index json 与 tokenizer 就位），具备条件后用 vllm-ascend 在 950PR 上跑通 Qwen3.8-Flash-Next-MXFP4 基线推理，建立 **decode TPS 性能基线**（mega kernel 的收益衡量基准，越早建立越好）。
3. **第一个自定义 kernel**：从 MXFP4 MoE GEMM 入手——对标 `/workspace/ops-nn/matmul/quant_batch_matmul_v4/` 与 `/workspace/vllm-ascend/csrc/online_mxfp4_gemm/`，结合 asc-devkit 的 MXFP4 matmul 调优样例（MIX_AIC_1_2 混合核 + RegBase 链式向量计算）。**进展（2026-09-26）**：MoE 整层（`m13_moe_layer/`）与 GDN 整层（`m14_gdn_layer/`）单次 mix 启动的 kernel 均已跑通并合并；48 层循环骨架、attention decode FA core、L0 装载几何标定、量化 golden 对齐正在进行。
4. **资源治理**：编译任务控制内存并行度（32GB cgroup 上限）；写大文件前先 `df -h` + `du -sh`；`shared_assets/` 为只读共享资源，不应写入。300GB 配额口径仍无权威结论，但实测写 170G 模型未受限。
