# M150 取证：m12 RMSNormGated（S5）在 m=257 / m=4097 的 m 包络设备验证

**Date**: 2026-10-04
**Mission**: M150 —— S5（`m12_rmsnorm_gated`）m 包络设备验证到 4097 + 自述范围订正（M145 finding 的落点）
**Scope**: `m12_rmsnorm_gated/**`；本目录证据 + 该段自述（注释 / README）订正。kernel 计算路径未改。

## 1. 要回答的问题

M145 survey 的结论（本 mission Context 转述）：S5 在**代码层是 m-general**，但 host 测试只覆盖到 m=256，
注释写「1..256」，m=4097 **未在设备验证**。本目录用**现成** `m12_rmsnorm_gated/check_ref.py` 在设备上
把这条包络钉死，并给出负向对照。

## 2. 代码侧逐条回源（任务 1，带 文件:行）

| 问题 | 事实 | 出处 |
|---|---|---|
| `Init` 怎么接 m | `Init(..., uint32_t m)`，入口 `M = m;` | `m12_rmsnorm_gated.asc:395`、`:397` |
| GM buffer 定尺 | `oGm/zGm/outGm` 按 `M*HIDDEN` 运行期定尺 | `:398-401` |
| `Process` 的行循环 | `for (row = 0; row < M; ++row) { CopyInRow; ComputeRow; CopyOutRow; }` | `:410-414` |
| 行寻址 `row*HIDDEN` | `off = (uint64_t)row * HIDDEN`（CopyIn/CopyOut 各一次） | `:452`、`:541` |
| 每行 DataCopy 定长、与 m 无关 | o `24576B = 768×32B`；z / out `12288B = 384×32B` | `:455-456`、`:542` |
| `HIDDEN` 常量 | `HEADS=48, HEAD=128, HIDDEN=HEADS*HEAD=6144` | `:72-74` |
| UB 布局是否随 m 变 | 静态编译期偏移（`UB_O..UB_RSTD`），`static_assert(UB_RSTD+256<=248K)` | `:83-91` |
| host 自测覆盖档 | `ms[] = {1, 2, 17, 64, 256}` × `seeds[] = {0..4}` | `:822-823` |
| 注释里的「1..256」 | 文件头 IO 契约 + `Process` 注释 | `:22`、`:407` |

**「1..256」的性质**：它是**任务书范围的口号**，不是 kernel 的编译期上界 —— m 只进 `Init` 的 `M`
（`:397`）与 `Process` 的行循环上界 / 行基址（`:410`、`:452`、`:541`），UB 静态布局与 m 无关（`:83-91`），
无按 m 定尺的静态资源。"已测范围"当时是 `ms[]`（`:822`）里的 `{1,2,17,64,256}`。

## 3. 结论（设备读数）

**设备实测（单次启动，未改 kernel 计算路径）**：

| 档 | 设备内建自检（device vs C 参考） | `check_ref.py` C-ref vs numpy | `check_ref.py` NPU-dev vs numpy | lock_exit |
| -- | -------------------------------- | ----------------------------- | ------------------------------- | --------- |
| m=257  | PASS（worst row rel 7.14e-3 @row 203） | True（max rel 7.19e-3） | True（max rel 6.33e-3） | 0 |
| m=4097 | PASS（worst row rel 7.75e-3 @row 932） | True（max rel 7.81e-3） | True（max rel 7.75e-3） | 0 |

即：m=257 与 m=4097 在**单次启动**下，设备输出与 numpy RMSNormGated 参考在 bf16 网格 1e-2 相对容差内
一致（判定项），且设备内建自检（device vs 同核 C 参考）同为 PASS。

设备已实测的 m 集合 = **{1, 2, 17, 64, 256（`m12_rmsnorm_gated/README.md` §校验方法与结果，内建对拍）,
257, 4097（本目录}**；m∈(257,4097) 的中间值与 >4097 未逐个实测。

**不改 kernel 计算路径即可支持更大运行期 m 的结构性依据**：m 只用于行基址（`:452`、`:541`）与循环
上界（`:410`），UB 静态布局与 m 无关（`:83-91`）。本次读数与之一致。

**本次为取证所做的 host harness 改动（非 kernel）**：`dump` 模式原为「仅 host 生成输入 + C 参考」，
本次扩展为「在设备上按该 m 档单次启动 kernel，落盘 device 输出并打印 PASS/FAIL」，接口与 m11 的 dump
对齐（m11 dump 本就跑设备，`m11_bf16_gemm/m11_bf16_gemm.asc:418-431`）。改动位置：
`m12_rmsnorm_gated.asc:802-814`（dump 入口）、`:707`（`RunCase` 增 `writeRef` 形参）、`:745-756`（落盘）。
kernel 计算路径（`RmsNormGatedKernel` 类 `:391`、入口 `rmsnorm_gated_kernel` `:558`）一字未改。

## 4. 判据与容差归属（T1 还是 T3）

本目录用现成 `m12_rmsnorm_gated/check_ref.py` 对拍：它对 out 用 **bf16 网格 1e-2 相对容差**
（`m12_rmsnorm_gated/check_ref.py:53`，`tol=1e-2`；`m12_rmsnorm_gated/README.md:97` 亦记「容差 1e-2」）。
按 `docs/17-verification-standard.md:34`（§1.1 容差分档）的 T3 触发条件，RMSNormGated 含 `Exp`（sigmoid）
与 `Rsqrt`（NR rstd）**超越函数近似** + 128 宽 **fp32 累加链** ⇒ 该段属 **T3**；本段现有判据是 bf16 网格上的
1e-2 相对界，**不是 T1 逐位（容差 0）**。因此本目录读数一律记为「容差对拍（1e-2 相对）」，不与 bit-exact 混读。
（T3 的规范式 `≤ ε·Σ|terms| + 0.5·ulp` 及 ε 推导不在本 mission 范围；本目录只如实记录现有判据的容差与出处。）

## 5. 环境与锁纪律

- 设备：NPU 0 = Ascend950PR（`npu-smi` 25.7.rc1.10；快照见 `logs/*.run.log` 的「lock entry」段）；
- CANN：`/usr/local/Ascend/ascend-toolkit` 9.1.0（`source set_env.sh`）；
- **每档各自一次** `flock -w 300 /tmp/npu0.lock`；**进锁先跑 `npu-smi info`**（快照落盘到该档 run.log）；
  `timeout 280` 在锁内；两档 `lock_exit=0`，未出现「等锁未取得读数」。

## 6. 逐字命令

构建（worktree 根）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m12_rmsnorm_gated/build -S m12_rmsnorm_gated -DCMAKE_BUILD_TYPE=Release
cmake --build m12_rmsnorm_gated/build -j4
```

m=257 档（在 `dumps_m257_s0/` 内执行；`lock_exit=0`）：

```bash
flock -w 300 /tmp/npu0.lock bash -c "cd <此目录>/dumps_m257_s0 && \
  echo '--- npu-smi (lock entry) ---' && npu-smi info && \
  echo '--- cmd: m12_rmsnorm_gated dump 257 0 ---' && \
  timeout 280 <repo>/m12_rmsnorm_gated/build/m12_rmsnorm_gated dump 257 0"
```

m=4097 档（在 `dumps_m4097_s0/` 内执行；`lock_exit=0`）：

```bash
flock -w 300 /tmp/npu0.lock bash -c "cd <此目录>/dumps_m4097_s0 && \
  echo '--- npu-smi (lock entry) ---' && npu-smi info && \
  echo '--- cmd: m12_rmsnorm_gated dump 4097 0 ---' && \
  timeout 280 <repo>/m12_rmsnorm_gated/build/m12_rmsnorm_gated dump 4097 0"
```

独立 numpy 交叉校验（分别在对应 dump 目录内执行）：

```bash
python3 <repo>/m12_rmsnorm_gated/check_ref.py 257 0    # dumps_m257_s0/
python3 <repo>/m12_rmsnorm_gated/check_ref.py 4097 0   # dumps_m4097_s0/
```

## 7. 负向对照（判据有牙）

m12 本体没有内建 mutant 模式，故用**数据侧**负向对照。`gamma.bin` 为 uint16 小端 bf16 位型：元素 i 占字节
`[2i, 2i+1]`；`gamma[1] = 0xBF80 = -1.0`（字节 `[2]=0x80, [3]=0xBF`）。复算脚本把对照拆成**三个独立检查**：

- **(a) 参考自检**（未篡改副本必须 True）：C-ref True（max rel 7.81e-3）、NPU-dev True（max rel 7.75e-3），
  rc=0（`logs/negative_control_ref.log`）。
- **(b) gamma[1] 翻 1 个 bf16 ULP**（字节偏移 2 的 bit0，`0xBF80→0xBF81`，M146 同款手法）必须变红：
  `NPU-dev vs numpy: ... False`（max rel 1.54e-2），`total 24058`，首错 `[0][2049]`，rc=1
  （`logs/negative_control_1ulp.log`）。C-ref 侧同向变红（`total 24057`）。
- **(c) gamma[1] 翻符号位**（字节偏移 3 的 bit7，`0xBF80→0x3F80`，该列整体取反）必须强红：
  `NPU-dev vs numpy: ... False`（max rel 2），`total 153264`，首错 `[0][1]`，rc=1
  （`logs/negative_control.log`）。

`gamma.bin` sha256：原始 `78bf5dc5…e32fabc`；1 ULP 篡改 `e7acd18f…d4e122c`；符号位篡改 `1977b103…8e3c306`。
说明：m12 判据是 1e-2 相对容差（非 m11 的 bit-exact），(b) 的 1 ULP 翻转**恰好**越过 1e-2（max rel 1.54e-2，
余量小），故另给 (c) 一个余量大的强红对照作兜底；两者都是用**数据侧篡改 + 同一份 device 输出**读数，
不依赖二次设备运行。

复算脚本对三个检查分别判定，且区分「判据如实变红」与「参考崩溃」：(a) 走失败路径时不会被当成 (b)/(c) 达标。

## 8. 证据文件与哈希

大 dump（`*_o.bin`/`*_z.bin`/`*_y.bin`/`*_y_device.bin`；m=4097 约 251MB、m=257 约 15MB）按本仓既有口径
**不入库**（`m12_rmsnorm_gated/.gitignore` 忽略 `evidence/**/*.bin`），重建方式见 §6 与 `reproduce.sh`。
入库的是日志、哈希与脚本：

- `logs/m257_s0.run.log`、`logs/m4097_s0.run.log` —— 锁内 `npu-smi` + 逐字命令 + device PASS；末行 `lock_exit=0`
- `logs/m257_s0.check_ref.log`、`logs/m4097_s0.check_ref.log` —— numpy 交叉校验（C-ref / NPU-dev 两行）
- `logs/negative_control_ref.log` —— 负控 (a)：参考自检（未篡改副本 True）
- `logs/negative_control_1ulp.log` —— 负控 (b)：1 bf16 ULP 翻转变红读数
- `logs/negative_control.log` —— 负控 (c)：符号位翻转强红读数
- `dumps_m257_s0/sha256sums.txt`、`dumps_m4097_s0/sha256sums.txt` —— 各 5 文件哈希
- `reproduce.sh` —— 复算脚本（默认离线核对哈希 + 两档 check_ref + 负控三检查；失败汇总为 exit 1；
  `M12_DEVICE=1` 重跑两档设备档）

dump 关键文件 sha256（完整见各 `sha256sums.txt`）：

```
# m=257 seed=0
1bc1e5fc…783489d  m257_s0_o.bin
99f40687…40576d73 m257_s0_y_device.bin
# m=4097 seed=0
026ea484…268b8f0c m4097_s0_o.bin
795bec99…3141f69d m4097_s0_y_device.bin
```

## 9. 自述订正（任务 4）

把 m 范围表述改成与设备读数一致、并显式标出实测范围：

- `m12_rmsnorm_gated.asc:22`：IO 契约行由「m 运行时 1..256」改为「m 运行时任意逐行循环（无编译期上界；
  设备实测档见 README）」。
- `m12_rmsnorm_gated.asc:407`：`Process` 注释同步订正（去掉「1..256」，改述为运行期 m 行循环）。
- `m12_rmsnorm_gated/README.md:3`、`:46`、`:55`、「已知限制」节：同步订正；显式列出设备实测范围
  {1,2,17,64,256,257,4097} 与「>4097 未逐个实测」。
- 未改 kernel 计算路径；host `ms[]`（`:822`）保持 `{1,2,17,64,256}` 不变（默认自测矩阵不动），
  本次新增档走 `dump` 逐档路径。

## 10. 边界 / 未测（不把范围写得比证据大）

- 逐档只测了 m=257 与 m=4097（外加 README 既有的 {1,2,17,64,256}）；m∈(257,4097) 的中间值与 >4097
  未逐个实测。
- 单核（`blockDim=1`，`m12_rmsnorm_gated.asc:730`），未做 48 head 分核 / GM 预取；本目录不涉吞吐。
- 本目录只覆盖 m12 本体的数值正确性；**输入布局契约**（`o` fp32 token-major `[m,6144]`、`z` bf16
  token-major，`:398-401`）与上游 `wsO` head-major 之间的**转位**不在本次范围（见 M148 / M149）。
- 校验建立在「核 vs C 参考 vs numpy」两层对拍上；`o` 为 fp32 输入、按 §4 的 T3 容差判定，非 bit-exact。
