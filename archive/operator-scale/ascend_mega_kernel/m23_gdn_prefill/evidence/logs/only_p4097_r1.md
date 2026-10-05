# 实验读数：p4097 作为**第一个且唯一一个** kernel task（`M15GP_ONLY_P4097=1`）

## 命令（逐字；单进程、`flock -w 300`、`timeout 60`、进锁前 `npu-smi` 复查）
```
cd m23_gdn_prefill/build
npu-smi info | sed -n 3p            # 进锁前复查：NPU 0 可用、空载
export M15GP_ONLY_P4097=1
timeout 60 flock -w 300 /tmp/npu0.lock stdbuf -oL ./m23_gdn_prefill > ../evidence/logs/only_p4097_r1.log 2>&1
# rc=1
```
**默认行为不变的保证**：开关是**运行期环境变量**（`m23_gdn_prefill.asc` 的 `main()`），不设它时
`firstCase = 0` ⇒ 循环起止与改动前**逐字等价**（未动 CMake/target）。本档日志里程序自己打印了
`[M23] M15GP_ONLY_P4097=1 ⇒ 本次跑 case [1, 2)`，作为"确实只跑了 p4097"的见证。

## 读数（逐字）
```
[M23] mix(1,2) launch: nblk(AIC)=28 → AIV 线程数=56；slot=753664 B；scratch=21102592 B
[M23] M15GP_ONLY_P4097=1 ⇒ 本次跑 case [1, 2)
[M23][FAIL] H=48 T=4097 launch/sync error 507015 : EZ9999: Inner Error!
[M23] ===== FAILURES PRESENT =====
```
| 量 | 本档（只有 p4097） | 两 case 档（`full_r2.log`） |
|---|---|---|
| `mte error info`（core 0） | **`0x13d10000000202ce`** | `0x13d10000000202ce`（**逐位相同**） |
| `multi-bit ECC` 行数 | **20** | 20 |
| `aicore error exception` 行数 | **24** | 24 |
| `aivec error` 行数 | **48** | 48 |
| `timeout or trap` 行数 | **52** | （塔实测同一故障上会变 12 / 20 / 24） |
| `retCode=0x26` 行数 | **2** | 2 |

## 判读（**对照预登记表**，`evidence/logs/only_p4097_prereg.md`）
预登记分支之一：**"仍挂 + 同签名 ⇒ 与 m=1 那次留下的设备状态无关；故障属 p4097 这个 shape / 这段代码自身"**
⇒ **本档命中该分支**（仍挂 507015，且 `mte error info` core 0 逐位相同；`multi-bit ECC` / `aicore error exception` /
`aivec error` 三个行数也与两 case 档相同）。

**由此**：**"两个 task 串行 / m=1 先跑留下的设备状态"不是本故障的必要条件。**

**限度（如实）**：① 本档只跑 **1 次**（未做重复性读数）；② `timeout or trap` 的行数在同一故障上**会变**
（塔实测 12 / 20 / 24）⇒ 它**不是稳定签名**，稳定项仍只有 `mte error info`；③ **未做**"带
`ASCEND_SLOG_PRINT_TO_STDOUT=1` 的对照档"（塔已自行跑过一条 SLOG，读数在其队列里）⇒ 本档**不据此下任何结论**。
