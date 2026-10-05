# 静态对照表：`Job1` vs `Job2`（塔裁 ③：零设备、纯读代码；与"按条跳过"互为交叉验证）

**刻度**：`m15_layer_loop/m15_gdn_prefill.h` 的 `GdnPrefillAic::Job1()` / `Job2()`（commit `e121532`）。

| 维度 | `Job1`（k, q） | `Job2`（q 重装 / ST / W） | 差异 |
|---|---|---|---|
| 用的 BufferID | `GP_BUF_L1`（MTE2 段）+ 每个 mmad 内部 `GP_BUF_L1`/`GP_BUF_L0`/`GP_BUF_L0C` | **完全相同** | **无** |
| acquire/release 位置 | `Acq<MTE2>(L1)` → 2×`Nd2Nz` → `Rel<MTE2>(L1)`；每 mmad：`Acq<MTE1>(L0)`、`Acq<MTE1>(L1)` → LoadL0A/B → `Rel<MTE1>(L1)`、`Rel<MTE1>(L0)` → `Acq<M>(L0)`→`Mmad`→`Rel<M>` → `Acq<FIX>(L0C)`→`Fixpipe`→`Rel<FIX>` | **完全相同**（同一 `DoMmad`） | **无** |
| 是否每 chunk 重新 acquire | 是（每 chunk、每 job 各一次 MTE2 轮转） | 是 | **无** |
| L1 目的偏移 / 尺寸 | `L1Kq(h)`（32KB）、`L1Kq(h)+32KB`（32KB） | `L1Kq(h)+32KB`（32KB，**与 Job1 同一地址**）、`L1St(h)`（**64KB**）、`L1W(h)`（32KB） | **有** |
| `Nd2Nz` 参数 | k/q：`rows=64, cols=128, gmRowStride=128, dstNzC0Stride=64` | q：同 Job1；**ST：`rows=128, cols=128, dstNzC0Stride=128`（64KB）**；W：`rows=64, cols=128, dstNzC0Stride=64` | **有** |
| 声明的 `TPosition` | `l1k`=A1、`l1q`=A1 | `l1q`=A1、`l1st`=A1、**`l1w`=B1** | **有** |
| 同一 chunk 内是否重复写同一 L1 地址 | 否 | **是** —— `L1Kq(h)+32KB` 在本 chunk 内已被 Job1 写过一次 | **有** |

## 三条差异各自对应的"按条跳过"预测（**这就是交叉验证的用法**）
| 若"只跳这一条"就恢复 OK | 则静态侧的对应差异就是根因方向 | 下一步 |
|---|---|---|
| **q 重装** | "**同一 chunk 内对同一 L1 地址的第二次 Nd2Nz**"（唯一一处）—— 形态是**同 chunk 重写**，不是跨 chunk 复用 | 与"跨 chunk 复用"区分开：把 Job1 的 q 改成不写（让 Job2 首次写）再跑 |
| **ST** | **64KB 写入 + `dstNzC0Stride=128`** 这一处（唯一的 128 行 / 64KB Nd2Nz） | **L1 窗口探针优先做 ST 那一档**（塔裁 ② 的咬合点） |
| **W** | **`TPosition::B1` 的那次写入**（Job1 全是 A1；`Job3` 的 `l1kt` 也是 B1） | 窗口探针加一档：同一 `off` 用 A1 / B1 两种声明对比 |
| 三条各跳一条仍复现 | 令牌次序（这段 MTE2 与紧邻 MTE1/FIXP 之间），与地址/尺寸无关 | 上窗口探针 + 令牌次序最小复现 |

## 限度（如实）
* 本表**只是静态对照**，不构成任何"哪一条是根因"的结论；三条差异**都可能是**触发器。
* `2.360 ms`（`j12nz_r1.log`）**不是** B1 的性能读数 —— 那是**跳过 `Job2` 的 L1 写入**后的耗时，
  当时 q 重装/ST/W 的数据是**陈旧/错的**（mmad 读的是 chunk 0 留在 L1 的值）。**不得**当作本段体耗时引用。
