# 收窄读数 r2：**J3 被排除**（故障在 J1+J2），且 m 相关 ⇒ 与"重复使用"有关；另查出两处静态问题

## 三档读数（同一次锁内，`npu-smi` 复查、单进程；日志 `j1_only_r2.log` / `j12_only_r2.log` / `full_r2.log`）

| 档 | m=1 | m=4097 |
|---|---|---|
| `M15GP_J1_ONLY`（只 J1） | **OK** 0.069 ms | **OK**（r1：1.930 ms） |
| `M15GP_J1J2_ONLY`（J1+J2） | **OK** 0.069 ms | **507015 aicore error**（全部 AIC 核：0,1,2,…） |
| 全档（J1+J2+J3） | **OK** 0.073 ms | **507015 aicore error**（同上，**同签名**） |

设备错误报告（逐字摘录，同一次运行内各核）：
```
core id 0: aicore error exception, error code = 0, mte error info: 0x13d10000000202ce,
           vec error info: 0, cube error info: 0, l1 error info: 0x23b00001893
core id 1: error code = 171, mte error info: 0x13d15000000202ce, vec error info: 0, l1 error info: 0x23800001893
core id 2: error code = 0,   mte error info: 0x13d1a000000202ce, vec error info: 0, l1 error info: 0x33000001004
```
**同签名判读（按塔裁 ③ 的提醒）**：`J1J2-only` 与 `全档` 的 `mte error info` **逐位相同**（`0x13d10000000202ce` / `0x13d15000000202ce`），
`vec error info` 都是 **0**、`cube error info` 都是 **0** ⇒ ① **同一类故障、同一处形态**（不是三个独立问题）；
② 故障**不在 J3**（加回 J3 不改变签名）；③ **在 AIC 侧**（`aicore error` + `l1/mte error info` 非零、`vec` 为 0）。

**m 相关性**：m=1（1 个 chunk）三档全 OK；m=4097（65 个 chunk）从 **J1+J2 档**起就抛 507015
⇒ 与"**同一处代码被重复执行**"有关（不是静态地址越界那类一次性的问题）。

## 另查出的两处静态问题（**未取证是否本次故障的根因**，但都应修）
1. **J3 的 A1 张量声明在 256KB 之外**：`Job3` 里 `l1dt` / `l1ab` 声明为 `TPosition::A1`，
   偏移 = `GP_L1_J3_K(327680) + 12288*4 = 376832` / `+8192*4 = 360448`。若 A1 窗口是 **256KB**
   （m11 的布局把 A 区放 [0,256KB)、B 区放 [256KB,512KB) ⇒ 强烈提示 A1/B1 各是 256KB 窗口），
   则这两个声明**超出 A1 窗口**。本段的 L1 布局是"一根扁平 491KB 空间 + 混用 A1/B1 声明"，
   与 m11 的分窗约定不一致。
2. **无任何一处把"每 head 行 stride"写进契约**（本轮已修）：host 按 `Tp = m + 64`（每 head 尾部补齐）
   存放 q/k，而 AIC/AIV 原先按 `hk*m*DK + t0*DK` 读 ⇒ 行 stride 不匹配（m=1 时恰好被掩盖）。
   已加 `GdnPrefillArgs.qkStride`（host 传 `Tp`），修后构建 rc=0；**该修不改故障签名**（见上表，
   说明它不是本次 aivec/507015 的根因，但是真正会算错的 bug）。

## 下一步（未做，按优先级）
1. 按 m11 的分窗约定**重排 L1 布局**（A 侧全部落在 <256KB 的 A1 窗、B 侧落在 ≥256KB 的 B1 窗），
   或把 J3 的张量全部按实际窗口声明；
2. 若仍复现：在 J1+J2 档里把 `Job2` 的两条 Nd2Nz **按 chunk 计数注释**（第 0 个 chunk 跑、
   第 1 个 chunk 起跳过），验证"是否从第 2 个 chunk 起出问题" ⇒ 若是，则锁定 L1 令牌的**跨次复用**。
