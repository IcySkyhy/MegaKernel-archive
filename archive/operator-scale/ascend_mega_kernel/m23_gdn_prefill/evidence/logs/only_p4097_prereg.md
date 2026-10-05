# 实验预登记：p4097 作为**第一个且唯一一个** kernel task（判读规则**先写后跑**）

## 要分开的东西
本档的 `main()` 有两件事（`m23_gdn_prefill.asc`）：case 表 `{"m1",48,1}` 与 `{"p4097",48,4097}`（`:274-275`），
循环里每个 case **一次** `<<<nblk,0,stream>>>`（`:198`）+ `aclrtSynchronizeStream`（`:209`）
⇒ 同一 stream 上**串行两个 kernel task**（m=1 先、p4097 后，中间有同步）。
⇒「两个 task 并发互扰」已排除；但**从未跑过"p4097 是第一个、也是唯一一个 task"**。

## 改动（最小；**默认行为必须不变**）
在 `main()` 里加一个**运行期**开关（不动 CMake/target）：环境变量 `M15GP_ONLY_P4097=1` 时，
循环起点从 case 0 改成 case 1（只跑 p4097）。**默认（不设该变量）走 `first = 0`，与改动前逐字等价**
（`onlyP` 为 false ⇒ 起止索引与原先完全相同）。

## 判读规则（**先写后跑**，防事后解释）
| 观测 | 判读 |
|---|---|
| **仍挂 + 同签名**（`mte error info: 0x13d10000000202ce` 逐位相同） | 与"m=1 那次留下的设备状态"**无关**；故障属 p4097 这个 shape/这段代码自身 |
| **不挂** | "m=1 那次留下的状态"是**必要条件** ⇒ 新角度，另设实验 |
| **挂但签名不同** | **如实说**，不硬套上面两条 |

## 同时要给出的量（塔实测它们在**同一故障**上会变 12 / 20 / 24 ⇒ **不是稳定签名**）
`multi-bit ECC` 的**行数**、`timeout or trap` 的**行数**；稳定项只有 `mte error info`。

## 命令（逐字；单进程、`flock -w 300`、`timeout`、先读设备错误报告）
```
cd m23_gdn_prefill/build
M15GP_ONLY_P4097=1 timeout 60 flock -w 300 /tmp/npu0.lock ./m23_gdn_prefill > ../evidence/logs/only_p4097_r1.log 2>&1
```
（若失败，再跑一次带 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 的对照。）
