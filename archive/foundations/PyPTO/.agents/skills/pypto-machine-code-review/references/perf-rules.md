# 昇腾 AICPU 侧高性能编码指南

> 本文件为 skill `pypto-machine-code-review` 的详细原则（progressive disclosure）。
> 面向 PyPTO（CANN）动态图在 **AICPU ARMv8** 上运行的 Control-Flow（CtrlCPU）与 Schedule（ScheCPU）热路径。

---

## 1. 适用范围与目标

| 覆盖 | 不覆盖 |
|---|---|
| `framework/src/machine/device/dynamic/**` Ctrl / Sche 运行时 | AICore kernel 算法本身 |
| 热路径依赖的 `framework/src/machine/utils/dynamic/**` 头（`ItemPool` / `Vector` / workspace） | Host encode / Pass 算法重构 |
| Host 启动路径中的 `rtMemcpy` 纪律 | dump / ESL / perf 采集里的 STL（允许保留） |

包括但不限于：

- Control-Flow：task 分配、stitch、slot、CF cache、AOT 映射、表达式求值
- **Schedule**：`device_sche*`、ready 队列、wrap、核间调度与提交

不含：AICore kernel 算法本身；纯 Host encode / UT 冷路径（但 Host 仍须遵守下文「Host 侧约束」）；[`device_sche.cpp`](../../../../framework/src/machine/device/dynamic/device_sche.cpp) 导出符号入口（文件头 DO NOT MODIFY）。

**目标**：少分配、少间接、少不可预测分支、少整结构清零、少热路径日志、少跨 Host/Device 额外拷贝。

**核心检验句**：这段代码在最热循环里，是否「做了但结果当下用不上」？

```mermaid
flowchart LR
  subgraph ctrl [CtrlCPU]
    Init[CONTROL_FLOW_INIT]
    Stitch[FAST_STITCH / UPDATE_SLOT]
    Build[BUILD_TASK_DATA]
  end
  subgraph sche [ScheCPU]
    Resolve[RESOLVE_DEPENDENCE]
    Send[SEND_AIC_TASK]
    Wait[RUN_TASK wait AICore]
  end
  Init --> Stitch --> Build --> Resolve --> Send --> Wait
```

典型 profile（30 连跑稳态）：`EXEC_DYN` 中 Ctrl 侧 `STAGE_STITCH`（`FAST_STITCH` + `UPDATE_SLOT`）占大头；Sche 侧 `STAGE_SCHEDULE` 墙钟含等核，**CPU 忙点看 `RESOLVE_DEPENDENCE` / `SEND_AIC_TASK`**，不要用 `STAGE_SCHEDULE` 当 CPU 指标。

---

## 2. 高性能编码原则

### 2.1 禁止在 AICPU 热路径使用 STL 容器与通用堆分配

| 禁止 | 原因 |
|---|---|
| `std::vector` / `std::deque` / `std::list` | 隐式堆分配、扩容、异常路径 |
| `std::map` / `std::unordered_map` / `std::set` | 节点分配、缓存不友好 |
| `std::string` 动态拼接 | 分配 + 拷贝 |
| `new` / `delete` / `malloc` / `free`（业务热路径） | 与 workspace/slab 生命周期冲突 |
| 带锁容器 / 共享指针链路 | 抖动与优先级反转 |

| 替代 | 场景 |
|---|---|
| 定长数组 `T buf[N]` / `std::array` | 上限已知的小表、LRU 槽、backup |
| 自定义 `Vector<T, WsMemCategory, Allocator>` | 变长且内存走 workspace/slab |
| `ItemPool<T>` | 高频同构对象创建/销毁 |
| `SPSCQueue` | 真单生产者单消费者提交/回收 |
| 栈缓冲 / 引用参数 | 临时量、避免二次分配 |

原则：**内存从哪来、何时还，由 DeviceWorkspace / slab / ItemPool 说了算。**

**反例**：在 `ResolveEarlyDepends` 里临时 `std::vector` 收集后继。
**正例**：用 `predCount` 表 + 定长 ready 队列。

dump / ESL / perf 采集路径允许保留 STL，**禁止**为“看起来快”拆掉可观测性。

---

### 2.2 类型系统：地址必须用显式指针类型

新增 / 改热路径代码时：

| 禁止 | 要求 |
|---|---|
| 用 `uint64_t` / `uintdevptr_t` 冒充「随便什么地址」长期传递 | **显式指针类型**（如 `uint8_t*`、`T*`、`DevXxx*`）表达对象身份 |
| 无注释的整型 ↔ 指针来回 cast 链条 | 在边界一次性转换，内部保持类型 |

`uintdevptr_t` 仅用于 **ABI / 序列化 / 与 AICore 约定的整型地址槽**；业务逻辑层优先指针，需要整型时在接口边界转换。

---

### 2.3 参数传递：局部对象优先按引用，避免取址传递

- 调用热路径函数时，**局部变量优先 `T&` / `const T&`**，避免 `&local` 再以指针传入（易悬空、妨碍优化、语义不清）。
- 需要可空语义再用指针，并在接口上写清 lifetime。
- 出参优先返回值或引用出参，避免「指针指向调用方栈对象」的隐式约定。

---

### 2.4 结构体：平凡默构 + 布局纪律

#### 2.4.1 平凡默构（trivial default constructible）

热路径对象常经 placement-new / 池化 Create。非平凡默构（尤其大数组 NSDMI）会在 `MakeDynDeviceTask` 等路径放大成上千次无用写。

1. 尽量 **POD / trivially_default_constructible**；
2. `static_assert(std::is_trivially_default_constructible_v<T>)` 钉死契约；
3. 初值用 **显式 Init / Fill / Shell**，只写会读字段。

#### 2.4.2 布局（layout）

| 要求 | 说明 |
|---|---|
| 访问时间相近的字段放一起 | 提高 cache 局部性 |
| 避免不必要的 padding 洞 | 合理排布成员，兼顾对齐 |
| 尽量减小热点结构体体积 | 少载入、少拷贝 |
| 冷热分离 | 不常用字段不要塞进最热结构体；可拆旁路/冷路径结构 |

---

### 2.5 禁止整结构 / 整池「为了安心」的初始化

| 反模式 | 正确做法 |
|---|---|
| `memset(整个 DeviceTask, 0, sizeof(...))` | `InitDevTaskShell`：只清会读壳字段 |
| ItemPool 创建时整池串 freelist | virgin bump；freelist 只服务回收 |
| `memset(表, 0xFF, n)` 冒充多字 sentinel | 按 word 写约定常量（如 `AICORE_TASK_INIT`） |
| 按 `MAX_*` 盲拷 | 按 `usedSize` 拷贝 |

---

### 2.6 惰性与按需激活（Lazy / Bump）

未用到的槽、路径、AOT entry：不构造、不拷贝、不扫描。
池化：高水位 bump + 回收 freelist 分离。
路径：encode 期旗标，runtime `if (!flag) return`。

I/O 首次填充空池时可用 `ItemPool::AllocateBump` / `MakeRuntimeOutcastTensorBump`（跳过 freelist 分支），仅限 **刚 Init、freelist 为空** 的路径。freelist 非空时禁止误用 Bump。

---

### 2.7 小固定池 + Flat 数据

优先：`定长数组 + 下标 + 单调时钟/序号`。
忌：热路径 `map` / `list` / 复杂 refcount 图。
命中路径：**禁止**再 `memcpy` / 无条件 clear icache。

---

### 2.8 Hit / 快路径零多余工作

| 场景 | 要求 |
|---|---|
| Hash / 缓存命中 | 禁止再次拷贝、禁止无条件 cache clear |
| Backup / 列表拷贝 | 长度 = 实际使用量 |
| 策略上不该做的事 | Host 提前 return，禁止 Device 中途补丁挽救 |

---

### 2.9 分支与「以存代算」：能 Host 判断的不要放到 AICPU

- **能在 Host encode / 构图期算清的 if**，不要拖到 AICPU 热循环里反复判断。
- 典型手法：**以存代算**——把结果写成 flag、表项、预计算字段，AICPU 只读。
- 例：有无值依赖、有无 partial/incast、是否需要某条 stitch 路径 → encode 置位，runtime 分支变常量/早退。
- 无工作路径早退（如 `stitchIndex == 0`、`totalZeroPredHub == 0`）。

运行时功能 **感知不到** Host codegen 的 CSE 之类优化细节；AICPU 侧评审应看「是否少算、少分支」，而不是要求手写 CSE。

---

### 2.10 循环：边界与循环不变量外提

1. `for (i = 0; i < B; ++i)` 中若 `B` 含计算，**先算到局部再进循环**（含 `x.size()`）。
2. 循环体内与 `i` 无关的计算、加载、条件，**提到循环外**（如 `GetSource()` / `IsMixArch()` / `readyQue` 选择）。
3. 与 Host codegen 的区分：
   - **AICPU / Schedule 运行时**：手写循环外提（本条）。
   - **Control-Flow 代码生成**：循环不变量 `GetInput*` 等由 Host 生成 `CSE_sd[]`（仅 codegen 关注；**不是** AICPU 运行时 API）。

#### 2.10.1 少间接：指针遍历，避开带检查的 `operator[]`

| 容器 | 热路径写法 |
|---|---|
| `DevRelocVector` | `T* p = vec.Data();` + `p[i]`（`operator[]` 每轮带 OOB assert） |
| `DevLocalVector` | `T* p = &src->At(list, 0);` + `p[i]`（无 `Data()`） |
| 裸数组 / `slotList` | 直接下标即可 |

**反例**：

```cpp
for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
    int slotIdx = nextSrc->At(outcast.toSlotList, j);
```

**正例**：

```cpp
const size_t toCnt = outcast.toSlotList.size();
if (toCnt == 0) { continue; }
const int* toSlots = &nextSrc->At(outcast.toSlotList, 0);
for (size_t j = 0; j < toCnt; ++j) {
    const int slotIdx = toSlots[j];
```

---

### 2.11 Host 侧严禁随意新增 rtMemcpy

Host 启动 / capture（含 **aclgraph**）路径：

| 禁止 | 要求 |
|---|---|
| 随意新增 `rtMemcpy` / `RuntimeMemcpyDirect`（尤其 H2D） | 尽量避免；若必须，走 **`NormalizedRtMemcpy`**（capture 下自动 RELAXED）或显式 `AclModeGuard(RELAXED)` |
| 在 capture 禁止 H2D 的模式下直拷 | 先切 RELAXED，再拷 |

Device 侧核内拷贝另论；本条约束的是 **Host runtime / launcher** 新增拷贝。

---

### 2.12 Sentinel 语义正确

禁止用 `memset(0xFF)` 冒充多字 sentinel。按约定字面量写，Host/Device 共用同一 `Fill*`。

---

### 2.13 AICPU 加载与内存模型

- 生成的 CF 代码：公共量放栈 / 参数；勿依赖不可靠 namespace 全局 `.bss`。
- 元数据走 workspace / slab / ItemPool，与 task stage 回收绑定。
- 队列空 ≠ 内存已全部可再切页。

---

### 2.14 DFX / 日志与热路径隔离

热头文件少拉日志；dump 默认关；性能计数与业务解耦。

| 场景 | 要求 |
|---|---|
| Init / Stitch / Resolve 热循环 | `DEV_VERBOSE_DEBUG`；禁止 `DEV_INFO` |
| 一次性启动日志（ThreadEnter） | 可保留 `DEV_INFO` |
| 仅 Host 仿真 dump | `DEV_IF_NONDEVICE { … }` |
| Device 侧 debug dump | `DEV_IF_DEBUG { … }` |

#### 2.14.1 耗时 debug 操作的宏隔离

新增耗时 debug 操作（dump 文件、遍历 op/tensor 写 CSV、`std::ofstream`/`std::mutex`/`std::string` 等）时，必须根据其运行场景选择正确的宏隔离：

| 场景 | 宏 | 效果 | 典型用法 |
|---|---|---|---|
| 仅 Host 仿真运行（不需要 device 侧编译） | `DEV_IF_NONDEVICE` | device 侧编译期排除整个代码块，零开销 | `DEV_IF_NONDEVICE { DumpRootMemory(...); }` |
| 需要在 device 侧运行但仅 debug 模式生效 | `DEV_IF_DEBUG` | release 模式编译期排除，debug 模式保留 | `DEV_IF_DEBUG { DumpTensorRange(...); }` |

**判定流程**：
1. 该 debug 操作是否需要在 device 侧（AICPU 运行时）执行？
   - **否**（仅 Host 仿真/编译期 dump）→ 用 `DEV_IF_NONDEVICE` 隔离
   - **是**（device 运行时也需要 dump）→ 进入步骤 2
2. 该操作是否耗时（文件 IO、STL 容器、遍历 op/tensor）？
   - **是** → 用 `DEV_IF_DEBUG` 隔离，确保 release 模式不执行
   - **否**（仅读一个 flag/计数器）→ 可不加宏，但须有 `if (!enabled) return` 运行时守卫

**禁止**：
- 在 device 侧也会编译的函数中直接调用 debug 函数而不加任何宏隔离（即使被调函数有 `#else` 空实现，调用本身仍生成无用开销）
- 用 `#ifndef __DEVICE__` 替代 `DEV_IF_NONDEVICE`（`DEV_IF_NONDEVICE` 是框架标准分流宏，语义更清晰且可能附带额外检查）
- debug 操作只加运行时 `if (!DumpEnabled()) return` 守卫但不加编译期宏隔离（release 模式下仍编译 STL/文件 IO 代码，增加二进制体积和 icache 压力）

---

### 2.15 无锁仅用于真 SPSC

`SPSCQueue` 仅单生产者单消费者；并行回收注意队头 `canFree` 语义。

---

## 3. 原则速查（Do / Don't）

| Do | Don't |
|---|---|
| 定长数组 / 自研 Vector·ItemPool·SPSC | STL 容器、热路径 `new` |
| 显式指针类型表达地址 | 长期用 `uint64_t` 当万能地址 |
| 局部对象按引用传递 | 无必要 `&local` 指针传参 |
| trivial 默构 + 热冷分离布局 | 默构清大数组、热点结构塞冷字段 |
| `InitShell` / bump / 按 `usedSize` 拷 | 整结构 memset、`MAX_*` 盲拷 |
| `Data()` / `&At(list,0)` 指针迭代 | 热循环 `operator[]` / 每轮 `At` |
| `AllocateBump`（空池首次填充） | 在已有 freelist 后误用 Bump |
| Host 预计算 flag / 表（以存代算） | 把本可 Host 完成的 if 丢到 AICPU |
| 循环边界与不变量外提 | 循环条件里重复算 `B` / `.size()` / `GetSource()` |
| Host 慎拷；必须时 `NormalizedRtMemcpy` | Host 随意新增直连 `rtMemcpy` H2D |
| 热循环 `DEV_VERBOSE_DEBUG` | 热循环 `DEV_INFO` |
| Schedule + CF 一并审视 | 只优化 stitch、忽略 `device_sche*` / `aicore_manager` |
| debug 操作按场景加 `DEV_IF_NONDEVICE`/`DEV_IF_DEBUG` 宏隔离 | device 侧函数中裸调 debug 函数仅靠 `#else` 空实现 |

---

## 4. ARMv8 AICPU 特化

| 约束 | 含义 |
|---|---|
| 无可靠业务堆 | 热路径内存只走 workspace / slab / ItemPool |
| icache / dcache 紧 | 少代码膨胀、少整结构清零、热字段聚拢 |
| 多 Ctrl / Sche 线程 | 函数内 `static` 缓存跨 program/线程不安全（需按线程或按 prog 键控） |
| 等核 ≠ CPU 忙 | `STAGE_SCHEDULE` 含 AICore 等待；优化 Sche CPU 看 `RESOLVE_DEPENDENCE` |

**明确禁止**：

- 为“看起来快”到处加 `likely` / `unlikely`（仅保留已有或证据充分处）
- 改 stitch / 调度算法冒充“编码细节”
- 修改 `device_sche.cpp`（导出符号入口，文件头 DO NOT MODIFY）
- 拆 dump / ESL / perf 里的 `std::string` / `std::vector`（伤可观测性）

---

## 5. 典型落地对照

| 优先级 | 区域 | 代表文件 | 典型落地 |
|---|---|---|---|
| P0 Ctrl | stitch / slot | `device_stitch_context.cpp`、`device_slot_context.cpp` | FastStitch / UpdateSlots 指针化；Fill I/O `Data()` + Bump；热路径去 `DEV_INFO` |
| P0 Sche | resolve / send | `aicore_manager.h` | `ResolveDepForAllAiCore` / `BatchSendTask` 外提 `typeInt`、`wrapManager`、`IsMixArch`、readyQue |
| P1 Ctrl | execute / task | `device_execute_context.cpp`、`device_task_context.*` | Init 日志降级；`ResolveEarlyDepends` 不变量外提与早退 |
| P1 utils | workspace / pool | `item_pool.h`、`dev_workspace.*` | `AllocateBump`；slab size 按 prog 缓存（注意线程安全） |
| P2 冷路径 | dump / esl / perf | `dump/`、`eslmodel_manager.h` | **不拆 STL**；补宏隔离即可 |
| 不动 | 入口 | `device_sche.cpp`、弱符号、PMU 头 | 保持原样 |

### 验证指标

- 主：`EXEC_DYN` avg、`STAGE_STITCH` / `FAST_STITCH` / `UPDATE_SLOT` total
- Sche CPU：`RESOLVE_DEPENDENCE` total（勿用 `STAGE_SCHEDULE` 墙钟）
- 精度：用例原有 assert 必须通过

---

*违反 §2.1–§2.5、§2.9、§2.11，以及热循环 `operator[]` / `DEV_INFO`（§2.10.1、§2.14）：即使功能正确，也视为热路径 / Host 启动路径回归。*
