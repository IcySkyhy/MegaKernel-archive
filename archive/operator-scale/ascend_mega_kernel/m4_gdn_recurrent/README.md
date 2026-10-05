# M4：GDN decode 递推核（纯 AIV，寄存器 VF 单遍融合）

Qwen3.8-Flash-Next GDN 线性注意力的 **decode（m=1）递推步**：每 value head 对 GM 中
fp32 state slab (128×128) 做原位读改写，寄存器 VF **单遍融合**
decay→delta→outer→matvec。数值与 numpy float32 参考（同公式）逐 head 比对，
5 组形状（4/8/48 head × 1/2 head-per-AIV/超配 AIV）**全部 PASS**
（maxAbsDiff ≤ 1.8e-7，容差 1e-5 相对 + 1e-6 绝对地板）。

donor：`/workspace/ops-transformer/attention/recurrent_gated_delta_rule` arch35
（`vf_vec_mul_mat.h`/`vf_outer_add.h` 寄存器 VF 写法）+ `/workspace/vllm-ascend`
`csrc/attention/recurrent_gated_delta_rule` 的 `ProcessKQ` 融合结构。

## 递推公式与 state 语义

每 value head `hv`（key head `hk = hv//3`，模型 NV=48=3×NK，NK=16，DK=DV=128；全 fp32）：

```
S ← e^g·S            # g 为 per-head log-decay（输入），e^g 在 kernel 内 VEC 一次算好
v ← β·(v − S·k)      # k 为 key head 的 128 维向量（kernel 外已 l2norm；1/√128 只乘 q）
S ← S + k⊗v          # 外积原位累加（[V,K] 布局：S[v,k] += k[k]·v[v]）
o = S·q              # 输出 128 维
```

- state 语义与 vllm cache 布局 `[block,NV,V,K]` 的 per-head slab 一致：行主序
  `S[v][k]`，v/o 是 V 维（128 行），k/q 是 K 维（128 列）。in-place GM 读改写。
- `g = −exp(A_log)·softplus(a+dt_bias)`、`β = sigmoid(b)`、q/k 的 l2norm（**q 另乘
  1/√128，k 不乘**）均在 **本 kernel 外预计算**（本任务只做递推核）；
  `output_gate_type=sigmoid` 的 RMSNormGated 也在递推核之外（docs/10 §1）。
- 本核为单 token（m=1）decode；MTP 多 token 与 prefill chunk 扫描不在本任务范围。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m4_gdn_recurrent/build -S m4_gdn_recurrent -DCMAKE_BUILD_TYPE=Release
cmake --build m4_gdn_recurrent/build -j4
cd m4_gdn_recurrent/build && ./m4_gdn_recurrent            # dump 各 case 的 .bin 到 cwd（build/ 已被 .gitignore）
/usr/local/python3.12.13/bin/python3 ../check_ref.py       # numpy fp32 参考校验（退出码 0/1）
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入 layout（fp32，kernel 面）

| 张量 | 形状 | 说明 |
|---|---|---|
| q | `[NK,128]` | 每 key head 的 q（已 l2norm **且乘 1/√128**，kernel 外预计算） |
| k | `[NK,128]` | 每 key head 的 k（已 l2norm，**不乘** 1/√128——模型语义，见 docs/10 §1） |
| v | `[H,128]` | 每 value head 的 v（conv1d+SiLU 后） |
| g | `[H,8]` fp32，stride-8 | log-decay；仅 `[h][0]` 有效，stride-8 是为 GM 32B 对齐 |
| β | `[H,8]` fp32，stride-8 | 仅 `[h][0]` 有效 |
| state | `[H,128,128]` fp32 | **in-place** 读改写，行主序 V×K（= 整层 (48,128,128) 的 head 子集） |
| out | `[H,128]` fp32 | o = S_new·q |

`H = numHeads ≤ 64`（运行时），`NK = ceil(H/3)`。head 映射 `hk = hv//3` 硬编码为
GROUP=3（与模型 48=3×16 一致）。dump 时 g/β 另导出紧凑 `[H]` 布局供 Python 参考用。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV-only；docs/05 §6 quirk 要求，否则 launch 失败），
  `blockDim` = AIV 数。head 按 `h = bid, bid+nblk, ...` 条带划分，每 AIV 处理若干
  head；`bid ≥ H` 的 AIV 早退。
- **UB 静态布局**（自管理地址，全部 32B 对齐；总 footprint ≈138KB ≤ 248KB）：

| 偏移 | 内容 |
|---|---|
| 0 / 64KB | state slab ping / pong（128×128 fp32 = 64KB each） |
| 128KB 起 | 每 parity 一组 q/k/v/o（各 512B）× 2 |
| 132KB 起 | g / e^g / β（各 `[64][8]` fp32 = 2KB） |

- **单遍融合递推**（`GdnHeadRecurrence`，`__simd_vf__` 寄存器 VF）：每行 i 的状态
  只 `LoadAlign` 一次、`StoreAlign` 一次，四步全在寄存器完成：
  `Mul`（decay，e^g 广播寄存器）→ `Mul/Add/ReduceSum`（`w=(e^gS)·k`，vcadd）→
  `Duplicate` 广播 w、`DIST_BRC_B32` 取 v[i]、`Sub+Mul(β)` 得 delta →
  `MulAddDst`（`S+=k⊗delta`）写回 → `Mul/Add/ReduceSum`（`o[i]=S_new·q`），
  结果经 `DIST_FIRST_ELEMENT_B32` 单元素落 UB。**无需 donor 的 64KB 中间矩阵**
  （broadTmp），每 head UB 流量从 ~448KB 降到 128KB。
- **跨 head 软件流水**（state 预取流水，照 donor 结构改用 BufferID）：
  `MTE2 预取 head h+nblk（parity 1-p）∥ VEC 计算 head h（parity p）∥ MTE3 写回 head h`。
- **同步只用 BufferID**（`GetBufInternal<pipe,false>` acquire 立即 /
  `RlsBufInternal<pipe,false>` release 阻塞释放（= CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地），m2 同款封装），
  无 set_flag/wait_flag、不挂 PIPE_S。各 pipe 统一按 ST→IV→OV 顺序获取，无死锁：

| BufferID | 交接 |
|---|---|
| 0/1 BUF_ST0/1 | MTE2 装载 slab → VEC 递推 → MTE3 原位写回（token 随 slab 全生命周期） |
| 2/3 BUF_IV0/1 | MTE2 装载 q/k/v → VEC 消费 |
| 4/5 BUF_OV0/1 | VEC 写出 o → MTE3 写 GM |
| 6 BUF_GB | MTE2 装载 g/β → VEC Exp 成 e^g（kernel 开头一次） |

## 向量 API 合规（`__simd_vf__` 裸调用；M48 裁定落地）

本核的计算路径全在寄存器侧：`EgExpAll`（g→e^g）与 `GdnHeadRecurrence`（单 head 四步融合）
都是 `__simd_vf__` 函数，函数体内只用 `AscendC::Reg::` 算子（`LoadAlign`/`Exp`/`Mul`/
`ReduceSum`/`MulAddDst`/`StoreAlign`…），无经典 memory-based vector API；
`DataCopy`/BufferID 属搬运/同步类（`docs/05` §6.1 明文允许）。

**调用形态**：两者都在 `__VEC_SCOPE__` **外**裸调用（本文件全篇 0 个 `__VEC_SCOPE__`）。
按 tower 裁定①（M44 审计 `docs/18-vector-api-audit.md` §9.1 的四条存疑之一，已定）：
**判据落在被调函数上**——必须 `__simd_vf__` 且体内只用寄存器 API；`__VEC_SCOPE__` 是词法
入口、`__simd_vf__` 是编译器函数属性（`__attribute__((cce_simd_vf))`，
`__clang_cce_defines.h:44`）；**不要求**调用点有词法 `__VEC_SCOPE__`，也**不要求**
`asc_vf_call`（CANN 上游用 `asc_vf_call<F>` 调 `__simd_vf__` 时调用点同样不在
`__VEC_SCOPE__` 内）。⇒ 本形态合规，**代码不动**。

**本次改动（M48）**：本模块**无算子代码改动**（无一处算子序列 / 参数 / 布局 / 同步变化）
⇒ 位级影响面为空。改动仅两处**文档口径**：①增本节（裁定①的合规说明）；②按 finding
`20260926-reviewer-gdnprolog-improve-m4-readme-k-l2norm-1-128-q-only-scale` 订正 q/k 的
l2norm 措辞——模型真实语义是 **1/√128 只乘 q**（docs/10 §1；已对照 `m9_gdn_prolog` 的
实现 `QSCALE` 仅作用于 q 块确认），原先 README 的「k 已 l2norm×1/√128」是沿用了 host
测试数据的生成方式而非模型语义；`.asc` 文件头与 `GenQk()` 注释同步注明「host 测试数据对
q/k 都乘了 scale，但本核把 q/k 当不透明输入，故对递推验证无影响」。既有判据（5 组形状
× {state, out} = **10 条**）已复跑，**10/10 PASS** 且数值与改造前一致（零回归）。

## 校验方法与结果

host 侧确定性生成（Hash3 公式，无 rand 状态；q/k 经二分法 l2norm×1/√128，host 无
libm），H2D → launch → D2H → 就地把每 case 的 q/k/v/g/β/state_init/state_out/out
dump 为 `.bin`；`check_ref.py` 用 numpy float32 按同一公式逐 head 计算参考
（`eg=exp(g)`；`w=S@k`；`delta=β(v−w)`；`S+=outer(delta,k)`；`o=S@q`），
state 更新与 o **都验**，容差 `|got−exp| ≤ 1e-5·|exp| + 1e-6`（绝对地板吸收
|exp|≈0 处的相消，物理误差尺度 ~1e-7）。

实测（Ascend950PR，CANN 9.1.0，2026-09-26）：

| case (H, blk) | 每 AIV head 数 | state maxAbsDiff | out maxAbsDiff | 判定 |
|---|---|---|---|---|
| (4, 4) | 1 | 2.98e-08 | 3.96e-09 | PASS |
| (8, 8) | 1 | 5.96e-08 | 1.68e-08 | PASS |
| (8, 4) | 2（流水） | 1.79e-07 | 2.98e-08 | PASS |
| (4, 8) | 1（4 AIV 空闲） | 4.47e-08 | 9.31e-09 | PASS |
| (48, 24) | 2（流水，覆盖 16 个 key head 映射） | 1.19e-07 | 2.24e-08 | PASS |

（脚本打印的 maxRelDiff ~1e-4 量级的分子即上述绝对偏差，分母为 |exp|+1e-6，
出现在 S_new 元素天然相消近零处；判定以组合容差为准。）重复运行逐位一致，退出码 0。

## 已知限制 / 后续

- 单 token（m=1）。MTP 多 token 可在 `ProcessHead` 外加 token 循环扩展（donor
  MAX_MTP 结构），prefill chunk 扫描是另一套（docs/10 §3）。
- `H ≤ 64`（g/β UB 静态缓冲上限）；整层 H=48 已覆盖。H 非 3 倍数时最后一个
  key head 只服务剩余 value head（`NK=ceil(H/3)`），与模型 48=3×16 语义一致。
- 输入/输出全 fp32（与 `mamba_ssm_dtype=float32` 一致）；bf16 输入 cast 未做。
- 正确性优先：未做性能细扣（单行 128 列全寄存器、无列分块双缓冲；递推为 µs 级，
  docs/10 §2 带宽下限分析不变）。核间负载按 head 条带静态划分。

## 与 donor 的差异点

| | donor（ops-transformer arch35 / vllm-ascend） | 本核 |
|---|---|---|
| 资源管理 | TPipe/TBuf/TQue + tiling-key 分发 | 全静态地址 + BufferID，无框架资源管理 |
| decay | gamaK 逐 k 维向量衰减（MatVecMul）；gama 标量走 UB `Muls` | 模型 g 为 per-head 标量：e^g 广播寄存器 `Mul` |
| delta matvec | 单独一遍（`MatVecMul` 写 64KB broadTmp）+ `ReduceSum` | 与 decay/outer/matvec 融进同一行寄存器单遍，无中间矩阵 |
| outer+matvec | `ProcessKQ` 融合（两行寄存器、两次 Store） | 同一行寄存器直接复用（S_new 无需二次 Load） |
| state 驻留 | vStep 分块（32KB）滚动 + RMW 到 finalState | 整 head 64KB slab 驻 UB，GM 原位写回 |
| 同步 | TQue EnQue/DeQue（隐式 flag） | 只用 BufferID（get/rls_buf，阻塞释放） |
| 通用性 | batch/cuSeqlens/ssmStateIndices/MTP/fp32-bf16 state | 单 token、单 layer-slice、fp32 only、head 子集 |
| 输入类型 | bf16 输入 cast | fp32 直收（gating/l2norm 在 kernel 外 fp32 产出） |
