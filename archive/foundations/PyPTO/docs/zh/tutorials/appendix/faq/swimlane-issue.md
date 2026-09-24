# 泳道图问题

## 问题现象描述

泳道图数据采集或展示异常，包括：未生成文件、文件为空、首任务启动过长、ACL Graph模式下无数据、`msprof`数据偏差大。

## 可能原因

- **未生成泳道图文件**：未启用性能数据采集（`runtime_debug_mode`未设为1）。
- **泳道图文件为空**：Profiling功能使能失败。
- **首任务启动过长**：AICPU启动较慢，AICore接收任务被整体延后。
- **ACL Graph模式下无数据**：当前不支持ACL Graph Capture阶段的profiling（Task在Capture阶段下发但profiling仅在Replay阶段开启），后续版本支持。
- **msprof与泳道图数据偏差大**：`msprof`的AICore耗时额外包含了AICPU下发devTask等待时间和退出时间。

## 处理步骤

1. 未生成文件：确认代码中已设置`runtime_debug_mode: 1`。
2. 文件为空：开启DEBUG日志，搜索`aicore profiling is opened` / `aicore profiling is closed`确认使能状态。
3. 首任务启动过长：确认AICPU资源分配和调度配置是否正常。
4. ACL Graph无数据：暂时规避该场景，后续版本支持。
5. msprof偏差大：设置`export DUMP_DEVICE_PERF=true`获取更准确的AICore端到端耗时。

## output目录产物说明

`output/output_时间戳`目录下泳道图相关文件：

| 文件 | 用途 |
|---|---|
| `machine_trace_perf_data*.json` | Machine组件原始Profiling数据 |
| `tilefwk_L1_prof_data_*.json` | Machine组件原始Profiling数据 |
| `merged_swimlane.json` | IDE可视化综合泳道图 |
| `machine_runtime_operator_trace*.json` | AI CPU / AI Core泳道图（联合时序） |

> `machine_trace_perf_data*.json`与`tilefwk_L1_prof_data_*.json`可判断底层采集是否成功；`merged_swimlane.json`与`machine_runtime_operator_trace*.json`用于IDE展示，优先联系IDE负责人。

## IDE参数含义

### CTRL AICPU

| 阶段 | 含义 | 打点位置 |
|---|---|---|
| **DEV_TASK_BUILD** | 构建devTask耗时（stitch耗时） | stitch之后 |
| **Post-process** | 构建完所有DevTask到退出 | AICPU退出时 |
| **Total run time** | 从拉起到退出的总时间 | 启动到退出 |

### SCHED AICPU

| 阶段 | 含义 | 打点位置 |
|---|---|---|
| **ALLOC_THREAD_ID** | 线程分配、绑核耗时 | AllocThreadIdx之后 |
| **INIT** | Sched初始化耗时 | Sched init()之后 |
| **CORE_HAND_SHAKE** | Sched与AICore握手耗时 | 握手之后 |
| **DEV_TASK_RCV** | 接收Ctrl构建的devTask | taskQue读取DevTask之后 |
| **Post-process** | 执行完所有DevTask到退出 | ExecuteTask之后 |
| **Total run time** | 从拉起到退出的总时间 | 启动到退出 |

### AICORE

| 阶段 | 含义 | 打点位置 |
|---|---|---|
| **End-to-End time** | AICore端到端实际执行 | 最早到最晚执行ExecCoreFunctionKernel的AICore |
| **Total run time** | 从拉起到退出的总时间 | 启动到退出 |
