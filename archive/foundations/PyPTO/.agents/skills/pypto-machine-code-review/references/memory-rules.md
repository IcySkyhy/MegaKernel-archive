# 内存规则

> 本文件录入机器侧内存相关的检视规则，补充内置清单（`review-checklist.md`）。
> 规则格式见下方模板，直接在对应分类下追加即可。Agent 在阶段 2 会将本文件与内置清单合并执行。

## 使用方法

1. 在对应分类下按模板追加规则行
2. `适用路径` 填 glob 模式（如 `framework/src/machine/runtime/bundle/*`），留空表示全 machine 适用
3. `关键词` 填代码中可用于 grep 定位的标识（如函数名、宏、类名），留空表示靠人工判断
4. Agent 读到本文件后，在阶段 2 对照检查；命中规则时与内置清单同等处理

## 规则模板

```
| U# | 检查项 | 级别 | 适用路径 | 关键词 | 典型问题/判定标准 |
```

- **级别**：Blocker / Major / Minor（定义见 `review-checklist.md`）
- **适用路径**：glob 模式，如 `framework/src/machine/device/dynamic/*`；留空 = 全 machine
- **关键词**：grep 关键字，如 `AOTCodePoolManager`、`rtMemcpy`；留空 = 人工判断

---

## 算法正确性

| U# | 检查项 | 级别 | 适用路径 | 关键词 | 典型问题/判定标准 |
|----|--------|------|----------|--------|-------------------|
| | | | | | |

## 跨核数据一致性安全（AICPU↔AICore）

> U4–U7 四条规则共同覆盖 AICPU 写入、AICore 读取的共享数据一致性问题的完整链路：
> - **U4** 管写入可见性顺序（`__sync_synchronize` 屏障）
> - **U5** 管分配时 cacheline 对齐（dcci 刷新安全基础）
> - **U6** 管后续新增访问时 dcci 地址覆盖性
> - **U7** 管 slab 注册 size 与 alloc size 一致（U5/U6 的前置保障）
>
> 四条规则层层递进：U7 保证对象大小正确 → U5 保证对象地址对齐 → U6 保证 dcci 覆盖新增字段 → U4 保证写入顺序可见。任一环节断裂都会导致 AICore 读到脏数据。

| U# | 检查项 | 级别 | 适用路径 | 关键词 | 典型问题/判定标准 |
|----|--------|------|----------|--------|-------------------|
| U4 | AICPU 侧跨核共享内存写后需 `__sync_synchronize()` 屏障 | Blocker | `framework/src/machine/device/dynamic/*` | `__sync_synchronize`、`SetReadyQueue`、`TryBatchSendTask`、`CloseFastPathReg`、`ResetShakeBuf` | AICPU 写入 ready queue / shake buffer / MAINBASE 寄存器后，在**下发任务 / 关闭寄存器 / 读取打印缓冲**之前必须有 `__sync_synchronize()`，否则其他核可能读到旧值导致任务丢失或 stop 失败。判定标准：存在「跨核共享写 → 后续依赖该写入的操作」序列但中间无屏障即违规。三类典型场景：(1) 写 ready queue → `TryBatchSendTask` 前；(2) 写 MAINBASE reg → `CloseFastPathReg` 前；(3) 写 shake buffer → 读取前（`ENABLE_AICORE_PRINT` 路径）。**以上函数和场景仅为已知示例，并非穷举——Agent 必须实际分析 PR 改动代码中是否存在新的「跨核共享写 → 依赖该写入的后续操作」序列遗漏屏障，不可仅依赖关键词列表做命中判断。** |
| U5 | AICPU→AICore 跨核共享数据结构须 cacheline 对齐（dcci 刷新安全） | Blocker | `framework/src/machine/device/dynamic/*`、`framework/src/machine/utils/dynamic/*` | `slab`、`dcci`、`devtask`、`DevTaskPtr`、`cceBinary`、`cacheline`、`align` | AICPU 写入、AICore 通过 dcci 读取的共享数据结构（如 devtask 对象）必须 **cacheline 对齐**，否则字段跨 cacheline 边界时 dcci 只刷新起始 cacheline，其余 cacheline 未刷新 → AICore 读到脏数据（指针为 0 或旧值）→ 前几轮正常、后续轮次报 AICPU 异常。判定标准：(1) slab / pool 分配的跨核共享对象地址是否 cacheline 对齐（通常 64B）；(2) 对象内被 dcci 读取的字段（指针、flag）是否可能跨越 cacheline 边界；(3) 是否有 `alignas`/对齐分配保证。典型场景：slab 分配 devtask 对象未对齐 → cceBinary 指针跨 cacheline → dcci 未刷新 → AICore 读到 0。**来源：issue #2610（FA 正向算子前两轮正常第三轮 AICPU 异常），修复 PR #4728/#4762。上述场景仅为已知案例——Agent 必须实际分析 PR 改动中所有 AICPU 写、AICore dcci 读的共享数据结构是否 cacheline 对齐，不可仅依赖关键词列表做命中判断。** |
| U6 | AICore 新增读取 AICPU 写入数据时须验证 dcci 地址覆盖该数据 | Blocker | `framework/src/machine/device/dynamic/*`、`framework/include/machine/device/dynamic/*` | `dcci`、`DCCI`、`aicore_entry`、`cceBinary`、`DevTaskPtr`、`cache invalid` | 当 PR 在 AICore 侧**新增**对 AICPU 写入数据的访问时，必须检查上下文中已有的 dcci（cache invalidate）操作地址范围是否覆盖该新数据，保证两者在同一 cacheline 内或 dcci 地址已显式包含新数据地址。判定标准：(1) 定位新增读取的 AICPU 写入字段的地址 `addr`；(2) 找到上下文中该读取之前最近的 dcci 操作，检查其地址 `dcci_addr` 和刷新范围是否覆盖 `addr`（即 `addr ∈ [dcci_addr, dcci_addr + cacheline_size)`）；(3) 若新数据地址与已有 dcci 地址不在同一 cacheline，则必须新增 dcci 操作显式刷新该地址，否则 AICore 会读到未刷新的脏数据。典型场景：已有 dcci 刷新了 devtask 对象起始 cacheline（含旧字段），新增读取同对象内偏移较大的字段（如 cceBinary 指针）落在另一个 cacheline → 未被 dcci 覆盖 → 读到 0 或旧值。**该规则关注「已有 dcci 是否覆盖新增字段」，与 U5（对象本身 cacheline 对齐）互补：U5 管分配时对齐，U6 管后续新增访问时的 dcci 覆盖性。Agent 必须对 PR 中每处 AICore 侧新增读取 AICPU 写入数据的位置逐一验证 dcci 覆盖性，不可仅依赖关键词列表做命中判断。** |
| U7 | 新增 slab 类型时注册 obj 大小须与实际 alloc size 一致 | Blocker | `framework/src/machine/device/dynamic/*`、`framework/src/machine/utils/dynamic/*`、`framework/include/machine/device/dynamic/*` | `WsAicpuSlabMemType`、`slab`、`SlabAddCache`、`SlabAlloc`、`objSize`、`COHERENT_SLAB_MEM_TYPE_BUTT` | 当 PR 新增 `WsAicpuSlabMemType` 枚举值（或新增 slab 注册）时，必须验证 slab 注册时声明的 `objSize` 与实际 `SlabAlloc` / `SlabAddCache` 调用时传入的 size 一致。判定标准：(1) 新增枚举值须插在 `COHERENT_SLAB_MEM_TYPE_BUTT` 之前（coherent 类型之前的 slab 参与 cacheline 对齐保证，见 U5）；(2) 找到该类型的 slab 注册点（`SlabAddCache(type, objSize, ...)`），记录声明的 `objSize`；(3) grep 所有 `SlabAlloc(type, ...)` / `SlabAlloc<type>(...)` 调用点，逐一核对传入 size 与注册 `objSize` 一致；(4) 若 obj 内含 cacheline 敏感字段（被 dcci 读取），`objSize` 还须 ≥ cacheline（64B）或为 cacheline 整数倍。典型问题：注册 `objSize=48` 但实际 alloc `sizeof(DevTask)=72` → slab 返回 48B 槽位 → 对象截断 → 字段越界读脏数据。**该规则是 U5/U6 的前置保障：objSize 错误会导致 slab 分配截断或跨对象污染，进而引发 cacheline 对齐和 dcci 覆盖问题。Agent 必须对 PR 中每个新增/修改的 slab 类型逐一核对注册 size 与 alloc size，不可仅检查枚举定义。** |

---

## 录入示例

以下为示例，展示填写格式（可删除）：

| U# | 检查项 | 级别 | 适用路径 | 关键词 | 典型问题/判定标准 |
|----|--------|------|----------|--------|-------------------|
| U1 | AOT cache 复用需验证 shape 无关 | Blocker | `framework/src/machine/device/dynamic/*` | `AOTCodePoolManager` | 若 CF binary 随 shape 变化，同 hash 复用会导致用错 code |
| U2 | 禁止在 DeviceRunner 析构中新增 rtMemcpy | Major | `framework/src/machine/runtime/runner/*` | `~DeviceRunner` | 析构与 torch_npu context 释放竞争 |
| U3 | bundle TLV value 必须 4KB 对齐 | Major | `framework/src/machine/runtime/bundle/*` | `valueOffset` | `offset % 4096 != 0` 导致设备 page copy 失败 |
