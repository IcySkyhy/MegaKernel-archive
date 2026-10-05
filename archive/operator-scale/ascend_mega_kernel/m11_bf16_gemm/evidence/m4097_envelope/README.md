# M146 取证：m11 bf16 GEMM 在 m=4097 的 M 包络（S2 in_proj / S6 out_proj）

**Date**: 2026-10-04
**Mission**: M146 —— m11 m=4097 包络设备验证 + `docs/22` 订正（M145 finding 的落点）
**Scope**: 只读设备读数 + 本目录证据；不改 m11 计算代码。

## 1. 要回答的问题

M145 finding（bug，medium）称 `docs/22-prefill-prolog-epilog-wiring.md` §2.3 G1-a 把 m11 写成
「单次启动的 M 上界是 `BASE_M=64`」，与 kernel 的 M-tile 循环和 m11 README 自述冲突（均为读代码推断，
未跑设备）。本目录用**现成** `m11_bf16_gemm/check_ref.py` 在设备上把这条包络钉死。

**M145 论据逐条回源**：finding 的代码侧引用与源码一致 —— `m11_bf16_gemm.asc:47`（`BASE_M=64`
单 tile 注释）、`:102`（`mLoop=CeilDiv(mTotal,BASE_M)`）、`:120`（`mBlock` 循环）、`:121`（`curM` 尾块）、
`:190`（A 基址按 `mBlock`）、`:254`（C 基址按 `mBlock`）、`:54-69`/`:108-118`（L1/L0/L0C 与 m 无关）、
host 测试档 `:447`（`ms[]` 未含 m>64）。即 finding 成立，本文按设备读数订正 `docs/22`。

## 2. 结论（设备读数）

m11 的 M 方向是**运行时多 tile 循环**：`mLoop = CeilDiv(mTotal, BASE_M)`
（`m11_bf16_gemm/m11_bf16_gemm.asc:102`，循环 `:120`，尾块 `:121`）。`BASE_M=64`（`:47`）是
**M tile 边长**，不是单次启动的 M 上界。本次设备实测（单次启动，未改代码）：

| 形状 | K | N | m | host fp32 参考（RNE bf16，逐位）| `check_ref.py` numpy fp32 参考（逐位）| lock_exit |
| ---- | - | - | - | ----------------------------- | ------------------------------------ | --------- |
| in_proj | 2560 | 16480 | 4097 | PASS (bit-exact) | `numpy fp32 ref == device C: True` | 0 |
| out_proj | 6144 | 2560 | 4097 | PASS (bit-exact) | `numpy fp32 ref == device C: True` | 0 |

即：m=4097 在**两个形状、单次启动**下，硬件 cube 输出与 host fp32 参考（RNE 到 bf16 网格）逐比特相等，
且与独立 numpy fp32 参考一致。设备已实测的 m 集合 = {1, 2, 17, 33, 64（`m11_bf16_gemm/README.md:105-108`）,
4097（本目录）}；更大的 m 未逐个实测。

**不改代码即可支持 m>64 的结构性依据**：`mLoop` 只进 `mBlock` 地址算式
（A 基址 `:190`、C 基址 `:254`），L1/L0/L0C 静态布局与 m 无关（`:54-69`、`:108-118`），
无按 m 定尺的静态资源。本次读数与之一致。

## 3. 环境与锁纪律

- 设备：NPU 0 = Ascend950PR（`npu-smi` 25.7.rc1.10；快照见 `logs/*.run.log` 的「lock entry」段）；
- CANN：`/usr/local/Ascend/ascend-toolkit` 9.1.0（`source set_env.sh`）；
- 每档独立一次 `flock -w 300 /tmp/npu0.lock`；**进锁先跑 `npu-smi info`**（快照落盘到该档 run.log）；
  `timeout 280` 在锁内；两档 `lock_exit=0`，未出现等锁未取得读数。

## 4. 逐字命令

构建（worktree 根）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m11_bf16_gemm/build -S m11_bf16_gemm -DCMAKE_BUILD_TYPE=Release
cmake --build m11_bf16_gemm/build -j4
```

in_proj m=4097（在 `dumps_in_proj_m4097/` 内执行；`lock_exit=0`）：

```bash
flock -w 300 /tmp/npu0.lock bash -c "cd <此目录>/dumps_in_proj_m4097 && \
  echo '--- npu-smi (lock entry) ---' && npu-smi info && \
  echo '--- cmd: m11_bf16_gemm dump 2560 16480 4097 ---' && \
  timeout 280 <repo>/m11_bf16_gemm/build/m11_bf16_gemm dump 2560 16480 4097"
```

out_proj m=4097（在 `dumps_out_proj_m4097/` 内执行；`lock_exit=0`）：

```bash
flock -w 300 /tmp/npu0.lock bash -c "cd <此目录>/dumps_out_proj_m4097 && \
  echo '--- npu-smi (lock entry) ---' && npu-smi info && \
  echo '--- cmd: m11_bf16_gemm dump 6144 2560 4097 ---' && \
  timeout 280 <repo>/m11_bf16_gemm/build/m11_bf16_gemm dump 6144 2560 4097"
```

独立 numpy 交叉校验（分别在对应 dump 目录内执行）：

```bash
python3 <repo>/m11_bf16_gemm/check_ref.py 2560 16480 4097   # dumps_in_proj_m4097/
python3 <repo>/m11_bf16_gemm/check_ref.py 6144  2560 4097   # dumps_out_proj_m4097/
```

## 5. 负向对照（判据有牙）

m11 本体未读到内建 mutant 模式（`docs/22` §7.4 已登记），故用**数据侧**负向对照：在临时目录把
in_proj 的 `b.bin` **第 2 个 bf16 元素**（`b.bin` 为 uint16 小端 bf16 位型，元素 i 占字节 `[2i,2i+1]`
⇒ 第 2 个元素的低字节在**字节偏移 2**）的 bit0 翻转 1 ULP，`check_ref.py` 必须变红。
（offset 0 = 第 1 个元素与 offset 2 不等价：前者 `mismatches: 475`、首错 `C[29][0]`；入库读数对应 offset 2。）

- 原始 `b.bin` sha256：`352bb009057e788f292b98daf8e9b50f015434f9efb29ed12c73bca00cbbf356`
- 篡改 `b.bin` sha256：`cd68db3d77f491c6c04066fb08d21af850eaa45118d87e2c99edde1e83ccbc2c`
- 读数（`logs/negative_control.log`）：`numpy fp32 ref == device C: False`，`mismatches: 624`，
  进程退出码 1。篡改只动 1 个权重元素（影响 C 的第 0 列 = 4097 行），仍有 624 行跨过 bf16 舍入网格 ⇒ 变红。

复算脚本把负向对照拆成**两个独立检查**（区分「判据变红」与「参考崩溃」）：
(a) 未篡改副本必须 `device C: True`（证明参考自身跑通，读数见 `logs/negative_control_ref.log`）；
(b) 篡改 offset 2 后必须出现 `device C: False` 且 `mismatches` 为正。若参考因缺文件崩溃，(a) 走失败路径，
不会被当成「判据如期变红」（见 `reproduce.sh` 的 `need_bins` / `fail` 汇总）。

即该判据能区分「m=4097 逐位一致的 device 输出」与「权重错 1 ULP 的输出」，不是空洞判据。

## 6. 证据文件与哈希

大 dump（`a.bin`/`b.bin`/`c_device.bin`，in_proj 约 230MB、out_proj 约 99MB）按本仓既有口径**不入库**
（`m11_bf16_gemm/.gitignore` 忽略 `evidence/**/a.bin`/`b.bin`/`c_device.bin`），重建方式见 §4 与 `reproduce.sh`。
入库的是日志与哈希：

- `logs/in_proj_m4097.run.log` —— 锁内 `npu-smi` + 逐字命令 + device PASS；末行 `lock_exit=0`
- `logs/out_proj_m4097.run.log` —— 同上
- `logs/in_proj_m4097.check_ref.log`、`logs/out_proj_m4097.check_ref.log` —— numpy 交叉校验
- `logs/negative_control.log` —— 负向对照 (b)：篡改后变红读数
- `logs/negative_control_ref.log` —— 负向对照 (a)：参考自检（未篡改副本 `device C: True`）
- `dumps_in_proj_m4097/sha256sums.txt`、`dumps_out_proj_m4097/sha256sums.txt` —— 三文件哈希
- `reproduce.sh` —— 复算脚本（默认离线核对哈希 + 两档 check_ref + 负向对照两检查；失败汇总为 exit 1；
  `M11_DEVICE=1` 重跑设备档）

dump 三文件 sha256：

```
# in_proj  (K=2560 N=16480 m=4097)
b844552807df0a5726a646d43ce49d7cb14ea58c16ae66ddae1d2a3a74360a75  a.bin
352bb009057e788f292b98daf8e9b50f015434f9efb29ed12c73bca00cbbf356  b.bin
26cb4ff0f4a7266bddaebdb5c7196e1914655e148666277b88a2d1886e43da67  c_device.bin

# out_proj (K=6144 N=2560 m=4097)
79c50185abf6b709fb5c848b30ecdb597659142f93c2b678880986511a1ea47d  a.bin
8ecf496815d407047f5df925f8e4a52f508415bb42f49f879e99525ab98cf233  b.bin
ea39be5682facdae4079a50666686e3483a30330ed2e6c4ea6a2417289df196c  c_device.bin
```

## 7. 边界 / 未测（不把范围写得比证据大）

- 逐档只测了 m=4097（外加 README 既有的 {1,2,17,33,64}）；m∈(64,4097) 的中间值与 >4097 未逐个实测。
- 本目录只覆盖 m11 两形状的 GEMM 本体（S2/S6）；prolog 的 S3（conv1d/l2norm/gating，m=1 段体）与
  S2→扫描→S5→S6→S7 的**接线**不在本次范围（`docs/22` §2.3 / §8）。
- 校验建立在 kernel 的「精确整数域」数据（小整数 bf16，fp32 累加无舍入）上（`m11_bf16_gemm/README.md:95-98`）；
  真实权重/激活的一般域为容差校验，未在本次覆盖。
