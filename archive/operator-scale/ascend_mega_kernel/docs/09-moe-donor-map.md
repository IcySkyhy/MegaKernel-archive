# MoE Donor 代码图谱（router/topk · permute · SwiGLU · RMSNorm · MXFP4 量化）

> 调研：M5 survey（2026-09-26，agent-donormap）。范围：/workspace 下 ops-math、ops-transformer、ops-nn、vllm-ascend 四仓（只读）。
> 适用约束：docs/05-megakernel-design.md v1.2 §2/§6——禁 Matmul 高阶 API、禁 TPipe/TBuf/TQue/TBufPool/AllocTensor、禁 set_flag/wait_flag（→BufferID）、裸指针基础 API。
>
> **指令级原语口径（2026-09-26 更新，取代旧"白名单"）**：**只有 `Sort32` 与 `MrgSort`** 属计算路径规范的例外（“Reg 侧无等价物 + 官方 donor 同为 memory-based”，每处使用须附一行注释指明依据）；`Reg::LocalMemBar` 属寄存器侧屏障，在 VF 内使用本就合规。**已移出**：`WholeReduceMax`（改用 `Reg::Reduce`）、`Concat`（3510 经典 `Concat` 是 no-op）、`Extract`（厂商 3510 的经典 `Extract` 本身即 `__simd_vf__`，须改造）、`MrgSort4`（3510 为 deprecated 空函数体，禁用）。**完整规范见 `docs/05` §6.1（四条 + 豁免清单）与 `docs/18-vector-api-audit.md`**。

## 0. 跨 donor 共性事实（先看这个）

1. **全部 donor 无一使用 get_buf/rel_buf**。同步一律 TQue EnQue/DeQue + `SetFlag<HardEvent::…>/WaitFlag`（常经本地 SetWaitFlag 包装）+ 少量 SyncAll/LocalMemBar——**所有同步都要转 BufferID**，但都是 1:1 映射（单缓冲 MTE↔V 顺序对 → 一个 get/rls 握手；DB=2 队列 → BufferID ping-pong）。
2. **向量类 donor 无一使用 Matmul 高阶**。Matmul 高阶只出现在 GEMM 系 donor（quant_batch_matmul_v4、grouped_matmul_swiglu_quant_v2、mega_moe、cgmct），那些的 cube 部分按禁令必须重写为 basic mmad。
3. **arch35 "regbase" 计算核心与目标风格最近**：全部是 `__ubuf__ T*` 裸指针自由函数（RegTensor/LoadAlign/StoreAlign/Reg::Reduce），数学路径零 TQue；只有 wrapper 类（Init/CopyIn/CopyOut）带 TPipe/TQue/TBuf 壳（每处约 50-60 行）需要换成静态 buffer+BufferID。**计算核心可近原样 lift**。
4. **SOC→arch 映射**：`ops-transformer/CMakeLists.txt:59-61`（ascend950:arch35）；950 入口惯例是 `op_kernel/<op>_apt.cpp` 分派 arch35 头；版本守卫 `__NPU_ARCH__==3510`（部分新代码用 5102/310，如 rms_norm.cpp:16-24）。

## 1. Router / softmax / topk

| op | 最佳 donor | 路径 |
|---|---|---|
| 融合 router（softmax+topk+renorm 一核） | **moe_gating_top_k_softmax_v2**（arch35 perf 档） | ops-transformer/moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h |
| 独立 softmax | **softmax_v2**（arch35-only） | ops-nn/activation/softmax_v2/op_kernel/arch35/（基类 softmax_v2_base.h） |
| 独立 topk（备选/大 k） | top_k_v2（ops-math，过重，不推荐做 router） | ops-math/math/top_k_v2/op_kernel/ |

**融合 router 细节**：入口 `moe_gating_top_k_softmax_v2_apt.cpp:59`（AIV-only）。可抄核心：

- `TopKFP32Perf`（perf_arch35.h:387-408）——**手写 topk**：k=1 走 `WholeReduceMax(ORDER_VALUE_INDEX)`(:393)；否则 Rearrange+DuplicatePad(-inf)+Reg::Arange 初始索引(:161-181)→`Sort32`(:187)→`MergeSortFP32Perf`（**`MrgSort` + `MrgSort4Info` 归并树** :255-307，实际调用是 `MrgSort(dst, srcList, params)` `:225`/`:243`，`MrgSort4Info` 只是参数结构体 `:212`/`:233`）→`ExtractKFP32Perf`(:309-343)。正是 router k≤8 场景。**注**：`AscendC::MrgSort4` 在 3510 上是 deprecated 空函数体、**我方禁用**（`docs/05` §6.1 ⓒ）——donor 这里用的是 `MrgSort`，不要误读成 `MrgSort4`。
- `UpdateSoftmax`(:256-279) 纯基础 Adds/Exp/Muls 全行 softmax 输出；`ComputeOut` 的 1/sum renorm（:227-228）。
- 被禁点：TQue DB=2（:563-568）→BufferID ping-pong；3 处 SetFlag/WaitFlag（MTE3_V :426-428、V_MTE3 :551-553）→get_buf/rel_buf；`SoftMax<>`(:504 等)→抄 softmax_v2 的 RegBase Max/Sub/Exp/Div 序列替换。
- 次选 v1（moe_gating_top_k_softmax arch35）：softmax 数学 100% RegBase（:261-312），无高阶 SoftMax，更干净但无 renorm/perf 分档。
- 分组路由（group topk）才选 moe_gating_top_k regbase（:657-1000）；moe_fused_topk（DeepSeek 风 sigmoid+组topk+norm）依赖 `TopK<>`/`adv_api reduce`/12×TBuf+7 处硬 flag，重写量最大，仅作数据流参考。

**softmax_v2 细节（全库与目标约束贴合度最高的 donor）**：`softmax_v2_base.h:58` SoftmaxV2OpsBase——纯 RegBase 裸指针：Cast(:68/:74)、`LastReduceSum`(:437/:493)、`NlastDichotomyAdd`(:575-698)、`FirstNormCompute`（Duplicate(-inf)+Max+Reduce<MAX>+Sub+**Reg::Exp** :176-199）、Normalize(:912)。**全 arch35 目录零 SetFlag/WaitFlag**，同步只靠 TQue EnQue/DeQue（→BufferID 改造最简）。arch35-only（无老架构包袱）。

**routing 扩展（topk 之后）**：moe_init_routing_v3 arch35（v4 复用 v3）是 950 现行主路径（counting sort + quant gather + SIMT topk_weight_out）；SIMT 片段+资源管理需重写。unpermute 阶段用 moe_finalize_routing_v2 arch35（RegBase，SetFlag 密集）。

## 2. Permute / 反重排（专家槽位重排）

**最佳 donor：moe_token_permute_with_routing_map + moe_token_unpermute_with_routing_map**（ops-transformer/moe/）。

- 语义完全对齐设计：routingMap[n,512] 即 top-k 掩码，输出 sortedIndices + 槽位化 permuteTokens（非 pad 模式 sortedIndices=argsort(masked_select(routingMap.T))；pad 模式每专家固定 capacity）——正是"decode 运行时已知 top-10-of-512 → 按专家槽位重排"，**无需自研索引协议**。
- **6 个 permute donor 中唯一带 arch35 专属 kernel**：`op_kernel/arch35/gather_v2_simd_two_dim.h:37`（Gatherv2SimdTwoDim，:148-194 NoSplitColProcess 逐行 gather 主循环 + UB 内 Copy 去重优化；另有 SIMT 版）；`__NPU_ARCH__==3510` 守卫。README 标注 950PR/950DT 支持。
- 反重排抄 `moe_token_unpermute_with_routing_map_not_pad.h:21`（scatter+probs 加权折叠）。
- 被禁点：TPipe/TQueBind/TBuf、AllocTensor、18 处 SetFlag/WaitFlag——全转 BufferID；masked_select/排序部分（计数排序、跨核归并）工程量大，只借语义。
- gmm/ 本身**不含** token 级重排；grouped_matmul_finalize_routing 与 mega_moe(arch22) 的 "permute 融进 GMM prologue/epilogue" 路线依赖 cgmct/Matmul 高阶框架，**不可抄**；但其设计思路（"GEMM epilogue 按索引直接 scatter 写出，免独立 unpermute pass"，`cgmct/epilogue/block_epilogue_finalize_routing.h`）与 mega_moe arch35 dispatch 的"按专家 row 区间给各 AIV 分工"（`mega_moe_token_dispatch.h:47-52`）值得参考。
- **建议：permute 先做独立 phase**（纯 MTE2→MTE3 带宽操作，可用 BufferID 与 GEMM ping-pong 掩盖；≤100 行 decode 规模融合收益有限），GMM 手写流水成熟后再把 gather 下沉为 prologue。

## 3. SwiGLU

**纯计算核心：ops-nn/activation/swi_glu**——`op_kernel/swi_glu_impl.hpp:54-82` Compute：基础 API 五元组 `Muls(β)→Exp→Adds(1)→Div(1/(1+e^x))→Mul(gate)→Mul(up)`，PipeBarrier<PIPE_V> 分隔；fp32 路径零 tmp buffer（Dup 1.0 常量向量做被除数）。arch35 目录全套。**五元组可无条件 lift 为裸指针 basic API**。被禁点仅外壳 TPipe/TQue/TBuf。

**SwiGLU+MXFP4 quant 融合 donor（GEMM#2 激活量化参考）：ops-nn/activation/swiglu_group_quant** arch35 MXFP4 路径（tiling key 3000/3100）：

- `arch35/swiglu_group_quant_base.h`：`VFSwiGlu`(:186-194)、**`VFComputeMaxExpMXFP4Vf`(:281-333)**（32 元素块取指数 max）、**`VFComputeScaleMXFP4Vf`(:350-409)**（e8m0 scale + halfScale + INF/NAN/zero 特判）、**`VFComputeDataMXFP4Vf`(:428-482)**（×halfScale→Cast<fp4x2>→DIST_PACK4_B32），全部 `__simd_vf__`+`__ubuf__` 裸指针，常数表 :26-65 齐全——**可整段 lift**。
- 调度 `swiglu_mxfp4_quant_perf.h:117-192`：SwiGLU 结果留 UB、quant 原地接力、fp4+scale 才拷出（"quant 不进 GM"形态）。
- 被禁点：TBufPool(12)/TQue 外壳；仅 groupIndex 归约一处 V_S flag（host 已知组数可删）。
- **融合位置结论：GEMM#1 epilogue = [dequant→]SwiGLU→MXFP4 quant（写激活），GEMM#2 直接消费**——反向不成立（silu 必须在量化前做）；官方 v2 post 三步顺序（`a8w4_msd_post.h:160-233`）即模板。
- 注意：fp16 输入需先转 bf16 再进 fp4 cast（base.h:310-318 特判）——GEMM#1 epilogue 宜直接产出 bf16/fp32。

## 4. RMSNorm / Residual（层边界）

**最佳 donor：ops-nn/norm/add_rms_norm arch35 regbase**——`op_kernel/arch35/add_rms_norm_regbase.h:50`：

- Compute(:103-136) 恰是层边界数据流：①`CalculateXAdd`(:138) 残差加（fp32 寄存器）→ 写回 residual xOut 且 UB 留 xFp32；②`CalculateSquareReduceSum`（二分 fold 平方和）；③`ComputeRstdNewtonRaphson`（NR rsqrt）；④`CalculateY`(:167) y=(x·rstd)·gamma→bf16。
- 编程模型：RegBase 主（裸指针 GetPhyAddr），DMA 用 TQue；**arch35 文件零 SetFlag**，屏障=LocalMemBar<VEC_STORE,VEC_LOAD>（核内 UB 障碍，与 BufferID 正交）。跨核同步无（行并行）。
- 被禁点：wrapper 的 TPipe::InitBuffer/TQue::AllocTensor/EnQue/DeQue——EnQue/DeQue 点即 BufferID 转换点；计算函数不用动。
- 独立 RMSNorm 用 rms_norm arch35（`rms_norm_regbase_common.h:607/836`）；**共享 reduce 积木在 norm/norm_common/op_kernel/reduce_common_regbase{,_part1,_part2}.h**（ReduceSumRstd:181、NR rstd、二分 fold、cast trait）——mega kernel norm 侧第一复用来源。

## 5. MXFP4 在线/动态激活量化（npu_dynamic_mx_quant 语义）

**最佳 donor：/workspace/vllm-ascend/csrc/online_mxfp4_gemm/**：

- `online_mxfp4_quant.asc:62-128` `MxQuantComputeScaleFused` + :213-265 内联版：bf16 按 32 分组取指数 max（And 0x7F80+Max+ReduceDataBlock<MAX>）→ E8M0：`sharedExp=maxExp-0x0100`、`scale=sharedExp>>7`、Inf→0xFF、halfScale=`0x7F00-sharedExp`（零组→0）。
- `online_mxfp4_quant.asc:130-160` `MxQuantComputeDataFP4`：x×halfScale（DIST_E2B_B16）→`Cast<fp4x2_e2m1_t, CAST_ROUND>`→DIST_PACK4_B32 nibble 打包。
- **头注释(:1-19)明确 bit-exact 对标 `torch_npu.npu_dynamic_mx_quant(dst_type=fp4x2_e2m1, round_mode="round", scale_alg=0/OCP, group=32)`**；`test/quant_ref.py:32-67` 是 CPU 位精确参考（含 e8m0 编码 :40-48），已对 npu_dynamic_mx_quant 验证 bit-exact。
- 纯 `__ubuf__` 裸指针 + RegBase + 静态 UB 偏移常量（:47-51），零资源管理耦合——**整段照抄级**。
- 权威上游实现：ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/add_rms_norm_dynamic_mx_quant_common.h（MxQuantComputeMaxExpOCP:308/ScaleOCP:357/DataFP4:660）。MX 常量与 cast trait 在 norm/norm_common/op_kernel/mx_quant_cast_traits.h。
- 同步重写点：quant.asc 的 SetFlag/WaitFlag(:206/271)→get_buf/rel_buf；**mix.asc 已示范 BufferID 等价范式** `GetBuffImpl/ReleaseBuffImpl`（:308-439），含 AIV→MTE3→CrossCore→AIC 跨核握手（:142-143、:277-278）。

**scale 喂 mmad_mx 的 cube 侧契约**（probe.asc，被 quant_batch_matmul_v4/arch35/matmul_custom_impl.h 印证）：

- 量化输出 qx[M,K/2] uint8（低 nibble=偶数 K）；scale 逻辑 [M,K/32] uint8 e8m0、**物理按 2 字节对打包 [M,K/64] uint16**（低字节=前 32 组）；
- AIC 侧 scale 当 half 做 Dn2Nz（nValue=K/64, dValue=M, dstNzC0Stride=K/64，probe.asc:112-123）进 L1；LoadData 带 `LoadData2DMxParams`（yStep=srcStride=(K+63)/64 :161-168）；
- **1 个 e8m0 对（2×32 组）对应 fp4 的 1 个 C0 组（64 fp4 值=32B）**（:145-148）；
- Mmad k 按 fp4 值计、64 对齐，多 K-chunk 累加 unitFlag=2、末 chunk=3（mix.asc:418-421）。
- 坑：scale <32B 落 GM 用 DataCopyPad Compact（quant.asc:274-278）；fp4x2 GlobalTensor 半字节索引要 ×2（mix.asc:326-344）。

## 6. 汇总速查表

| op | 首选 donor（文件） | 编程模型 | 同步（→BufferID） | arch35 | 可抄度 |
|---|---|---|---|---|---|
| router（softmax+topk 融合） | moe_gating_top_k_softmax_v2 perf_arch35.h:387 | RegBase+TQue 壳 | 3 处 flag 对+TQue | ✅ | 高 |
| softmax（独立） | softmax_v2_base.h + arch35 | 纯 RegBase | 仅 TQue | ✅ only | **最高**（零 flag） |
| permute | moe_token_permute_with_routing_map arch35 gather_v2_simd_two_dim.h:148 | SIMD basic+TQue 壳 | 18 处 flag+TQue | ✅ | gather 主循环高 |
| unpermute | moe_token_unpermute_with_routing_map_not_pad.h:21 | SIMD basic+TQue 壳 | 同上 | 复用通用 | 循环骨架高 |
| SwiGLU | swi_glu impl.hpp:54-82 | basic API | 仅 TQue | ✅ | **最高**（五元组） |
| SwiGLU+MXFP4 quant | swiglu_group_quant base.h:281-482 | RegBase+TBufPool 壳 | 1 处 V_S | ✅ only | 高（三 VF 函数整段 lift） |
| Add+RMSNorm | add_rms_norm_regbase.h:103 | RegBase+TQue 壳 | TQue+LocalMemBar | ✅ | 高（层边界数据流现成） |
| MXFP4 激活动态量化 | online_mxfp4_quant.asc:62-160 | RegBase 裸指针 | 2 处 flag（mix 版已示范 BufferID 范式） | ✅ | **最高**（bit-exact 已验证） |
| GEMM cube 参数化 | online_mxfp4_probe.asc / quant_batch_matmul_v4 matmul_custom_impl.h | basic mmad+壳 | TPipe/TBuf(AIC)+flag | ✅ | mmad/LoadData 参数化参考，壳重写 |

## 7. 遗留事项

1. 指令级原语口径（见文首）：**只有 Sort32/MrgSort 入例外**（附注释指明依据）；WholeReduceMax/Concat/Extract 已移出白名单，MrgSort4 禁用。规范全文见 `docs/05` §6.1、审计见 `docs/18`。
2. moe_gating_top_k_softmax_v2 perf 档的 `SoftMax<>` 需替换（源头 softmax_v2）；moe_init_routing_v3 的 SIMT 片段（asc_vf_call）融合时需评估是否改 SIMD。
