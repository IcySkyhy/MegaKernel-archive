# §4f.2 的 flag 用量 static_assert 重推：硬件计数器计的是「未配平累计」（M197）

## 问题（M197 任务）

`m15_layer_resources.h` 的 §4f.2 里，B3（`m15_attn_fa_core.h`）的用量判据是

```cpp
static_assert(FlagEventsMaxSetUse(FLAG_SEQ_ATTN_FA_CORE, FLAG_SEQ_ATTN_FA_CORE_N) <= 15u, ...);
```

而 `FlagEventsMaxSetUse` 只把事件表里每个 (核型, mode, id) 的 **`count` 字段**（按 §4c 的口径 = 单次核体
里的 `CrossCoreSetFlag` **调用点数**，不随循环次数缩放）求和。事件表按 **单 tile**（外加 AIV 的 4 个
循环前 set 预置）登记，而生产档的 tile 数是

```
nTiles = ceil( min(posBase + rowBase + FAC_P(64), ctx) / FAC_SIN(128) )
```

（`m15_attn_fa_core.h` 的 `FacMakeWork`）。4097 行 ctx 下 ≈ 33。于是疑问是：**运行期每个 id 的 set 次数
远超 15，为什么断言还成立？**

## 依据（逐字）

`docs/05-megakernel-design.md` 的 CrossCore flagId 行：

> 每核 16 个（0-15）… 「每 id 计数 15 次」 = 每（核,flagId) 一个 4bit 计数器（0-15），**未配平累计超 15 报错**。

⇒ 硬件约束的量是 **执行过程中「已 set、尚未在同 id 上 wait 掉」的令牌数的峰值**，**不是 set 总次数**。
总 set 数、以及 `FlagEventsMaxSetUse` 这个"调用点代理"，都与硬件计数器不是同一个量。

## 重推：B3 的未配平峰值 = 2（O(1)，与 nTiles 无关）

B3 核体的配对（`m15_attn_fa_core.h` 的 `FacAiv::Run` / `FacAic::RunWorkItem`，tile 循环 `t`，`q = t & 1`）：

- **AIV**（每 tile）：`wait CC_S_RDY[q]` → `set CC_S_FREE[q]` → `set CC_P0+q` → `wait CC_PV_RDY[q]` →
  `set CC_PV_FREE[q]`。循环**之前**一次性 `set CC_S_FREE[0/1]` + `set CC_PV_FREE[0/1]`（4 个预置）。
- **AIC**（每 tile）：`wait CC_S_FREE[q]` → `set CC_S_RDY[q]` → `wait CC_P0+q` → `wait CC_PV_FREE[q]` →
  `set CC_PV_RDY[q]`。循环前无 set。

逐通道的稳态未配平：

| 通道 | 方向 | 未配平峰值 | 为什么 |
|---|---|---|---|
| `CC_S_FREE`/`CC_PV_FREE`（各 2 号 = parity） | AIV→AIC | **2** | 双缓冲槽（S / PV 各 2 个）；循环前给每个槽 1 个空闲令牌 = 初始 credit。AIC 每 tile 消费一个、AIV 消费完槽后补回一个 ⇒ 水平线 ≤ 槽数 = 2 |
| `CC_S_RDY`/`CC_PV_RDY`（各 2 号） | AIC→AIV | **1** | AIC 对某 parity 的下一次 `set` 在该 parity 的上一轮被 AIV `wait` 之后才发（中间隔着对面的 `*_FREE` 握手）⇒ 同一 id 至多 1 个在飞 |
| `CC_P0/+1`（2 号） | AIV→AIC | **1** | 同上：AIC 每 tile `wait CC_P0+q`，AIV 只在 AIC 放行后（`S_RDY` 之后）才 `set` |

⇒ 所有 (核型, mode, id) 的未配平峰值 **= 2 ≤ 15**，且**与 nTiles 成对回到 ≤2 的稳态**，累计量 O(1)。
逐 tile 的 set/wait **成对**：总 set 数（nTiles=33 时 id 1/3 每个 AIV 约 17~18 次）会随 nTiles 增长，但
计数器每轮被 wait 减回，**永远到不了 15**。

## 结论与处理

**不需要改计数模型**：硬件约束（未配平 ≤ 15）在 B3 的严格乒乓协议下满足，且 O(1)、与 nTiles 无关。
`FlagEventsMaxSetUse` 那条断言**保留**，但它的作用范围在源码注释里被明确成「单次调用里同一 id 的 set
调用点数的**保守代理**」——它**不是**硬件计数器本身，也**不按 nTiles 缩放**（按上面推导，需要缩放的量
本就不构成约束）。真实约束的推导即本文件。

## 复现

- 硬件口径：`docs/05-megakernel-design.md` 的 CrossCore flagId 行（搜「未配平累计超 15 报错」）。
- 配对口径：`m15_layer_loop/m15_attn_fa_core.h` 的 `FacAiv::Run` / `FacAic::RunWorkItem`（搜
  `CC_S_FREE` / `CC_PV_FREE` / `CC_S_RDY` / `CC_PV_RDY` / `CC_P0`）。
- 登记表：`m15_layer_loop/m15_layer_resources.h` 的 `FLAG_SEQ_ATTN_FA_CORE`（§4f.2）；
  本文件是它那条用量断言的推导依据。
