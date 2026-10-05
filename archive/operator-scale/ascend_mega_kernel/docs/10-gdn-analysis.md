# GDN 线性注意力分析（Qwen3.8-Flash-Next，36/48 层）

> 调研：M6 survey（2026-09-26，agent-gdn）。平台基线：950PR（28 AIC + 56 AIV、UB 248KB/AIV、L1 512KB/AICore、L2 128MB、HBM 1.6TB/s，见 docs/05 §6）。

## 1. 数学与形状

- 投影（bf16，`quantize_linear_attn=false` → GDN 线性层不量化）：in_proj_qkvzba 2560→16480（q2048 | k2048 | v6144 | z6144 | b48 | a48）；conv1d depthwise K=4 over 10240ch + bias + SiLU；out_proj 6144→2560。
- **注意：本模型 `output_gate_type="sigmoid"`**（config.json:105；vllm `qwen_gdn_linear_attn.py:490-497` 的 RMSNormGated(norm_before_gate=True, act=sigmoid)）——与 Qwen3-Next-80B 的 silu 不同，golden 生成勿抄错。
- State/请求/层：conv_state (10240,3) bf16 = 60KB；ssm_state (48,128,128) = 786K elems，cache 布局 [block,NV,V,K]（vllm mamba_utils.py:295-300）。`mamba_ssm_dtype=float32` → 3.0MiB/层。
- 递推（v-head hv ← key head hv//3，48=3×16 组从；sigmoid_gating.py:211）：g=−exp(A_log)·softplus(a+dt_bias) fp32（β=1, thr=20）；β=sigmoid(b)；q,k=l2norm(eps 1e-6)，q×=1/√128；顺序（sigmoid_gating.py:115-159）：**S←e^g·S；v←β·(v−S·k)；S←S+k⊗v；o=S·q**，全 fp32 累加。所有 donor 的 gating+l2norm 都在 recurrence kernel 外预计算。

## 2. Decode（m=1）访存特征 / token / 层

- 权重 115.9MB（qkvzba 84.4 + out 31.5）；ssm_state r+w **6.29MB fp32**（bf16 则 3.15MB）；conv_state 0.12MB；激活 ~0.15MB → 合计 ~122MB → **76µs@1.6TB/s；×36 层 = 4.40GB → 2.75ms/token（bs=1 GDN 段 HBM 下限）**。
- state 占 GDN 段流量 5.2%（bs=1）；bs≈18 时 state≈权重。递推计算 ~3.1M lanes-ops/层/token ≈ 0.6µs/56AIV——纯带宽问题，**cube 工作为零**（仅 in/out_proj 两个小 GEMM）。
- UB 放 1 head 的 128×128 fp32 slab（64KB）绰绰有余：48≤56 AIV → 1 head/AIV；state 本体必须 GM（L1 非向量可直接寻址；3MB/层 ≫ UB）。

## 3. Prefill chunk 扫描（m=4097 → 65×BT64）

- 所有公开实现 BT=64（fla / vllm-ascend / ops-transformer arch35 硬编码；chunk_kda_fwd 支持 128）。流水线：chunk 内 g cumsum(fp32) → A=tril_strict(β_i·K_iK_jᵀ·e^{g_i−g_j})(fp32) → T=(I+L)⁻¹（16×16 块前代合并 64×64 [vllm-ascend solve_tril.py] 或 32×32 寄存器 [arch35 vf:80-128]）→ WY：w=T@(βe^g k), u=T@(βv) → **chunk 间 h 串行扫描**：v_new=(u−w@h)·e^{g_last−g}; h=e^{g_last}h+kᵀv_new（h fp32）→ o=scale·(qe^g)@h+scale·tril(QKᵀe^{g_i−g_j})@v_new。
- 依赖链 = 每 head 65 chunk 串行；并行轴 = 48 heads × 所有 chunk 的本地准备。fla fwd_h 把 h 常驻寄存器跨 chunk（chunk_delta_h.py:74-75）；arch35 stage2 把 h 放 GM 每 chunk RMW+atomic-add fixpipe（stage2_arch35.h:148-179,275-280）→ GM 往返 ~400MB/层。**mega kernel 把 h 常驻 UB 可砍到 ~6MB**；w/u 中间量 ~100MB/层 可驻 L2。未融合 GM 激活流量 ~640MB/层（~400µs）。

## 4. Donor 盘点

- host 语义：vllm-ascend `ops/gdn.py` + `gdn_attn_builder.py`（BT=64、solve_tril 块 1216、cumsum 工作集 2^18）。
- vllm-ascend 算子分解：npu_fused_qkvzba_split_gating(B1, 可选) → causal_conv1d（可选 C 相融合 npu_causal_conv1d_qkv：conv epilogue 直写 q/k/v 并内嵌 l2norm，仅 decode；prefill 用 npu_causal_conv1d_custom run_mode=0）→ npu_fused_rearrange_qkv_l2norm(B2, 可选) → **decode 核心 npu_recurrent_gated_delta_rule**（AIV-only AscendC、in-place state、MAX_MTP=16、state 预取流水；**arch35 变体用 RegTensor/MicroAPI 寄存器数学，是 mega kernel 向量代码风格的最佳 donor**）→ **prefill = Triton chunk 流水线 + 2 个 AscendC 自定义 op**（chunk_gated_delta_rule_fwd_h、chunk_fwd_o，在 csrc/moe/）→ RMSNormGated。
- ops-transformer（950/arch35，均有 arch35 内核）：**attention/chunk_gated_delta_rule = 完整 MIX_AIC_1_2 三段式 prefill 实现**（BT=64 硬编码、支持 Nv%Nk==0 组头、**fp32 state 仅 950 支持**）——最贴近的 prefill donor；attention/recurrent_gated_delta_rule（decode，寄存器 VF）；chunk_kda_fwd/recurrent_kda/kda_input_proj（KDA 家族：in-kernel l2norm/gating 开关、chunkSize 64/128、state 布局可选）；mamba/causal_conv1d（aclnn Fn/Update 一体，AIV-only 环形 UB kernel，K∈{2,3,4}）；fused_causal_conv1d 家族（K=3、APC cache）。CANN 自带 `torch_npu.npu_chunk_gated_delta_rule` 可用作 golden（需 CANN≥9.1.0 保精度）。
- 性能数据：**无任何 GDN kernel 级实测**（cannbot 卡内只有 ZECO CP 算子级 2.05–2.48× @64K–128K 与 Qwen3-Next-80B 16 卡 A3 decode 20.6ms）。A3 卡平台数字（UB 192KB / L0C 128KB）面向 910C，勿直接套 950PR。

## 5. Mega kernel GDN 段建议

- **decode m=1**：AIC 跑 in_proj_qkvzba + out_proj（bf16 GEMM，权重流带宽 bound）；AIV 做 conv1d(10240ch 逐元素)+l2norm+gating，然后 48-head 递推 1 head/AIV，h slab fp32 64KB 驻 UB，单遍融合（decay→delta→outer→matvec）寄存器 VF 实现（照抄 arch35 recurrent kernel 的 ProcessKQ 融合写法），in-place GM state；递推 ~µs 级，与下一段权重预取重叠。保留 fp32 state。
- **prefill m=4097**：采用 arch35 三段式结构（BT=64），改造四点：(i) SyncAll→CrossCore mode-0 barrier（项目规则）；(ii) h 按 head 常驻 UB 跨 65-chunk 扫描（省 ~400MB GM/层）；(iii) l2norm+gating 融进 stage A（CANN chunk op 无内嵌 l2norm）；(iv) chunk-group 流水穿过串行扫描。m=4097=64×64+1，注意 ragged 尾块。
- 风险/待查：sigmoid-vs-silu 输出门以 checkpoint 实测为准；fp32-vs-bf16 state 可 A/B；golden = aclnn 算子 + fla Triton 参考。

## 6. 附带发现（integration 坑，来自 vllm-ascend donor 栈）

CANN<9.1.0 prefill 精度问题、graph padding 0 vs -1、mamba_cache_indices==0 污染、beta sigmoid 下沉；vllm-ascend block_size=128 对 qwen3_next 静默失效。
