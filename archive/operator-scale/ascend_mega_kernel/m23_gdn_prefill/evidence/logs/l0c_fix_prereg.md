# 预登记：L0C 的 M↔FIXP 成对交接（修）与验证（**改动+判据+预期先写后跑**）

## 改动（最小 diff，只动 `m15_layer_loop/m15_gdn_prefill.h` 的 `GdnPrefillAic::DoMmad`）
在既有 `MmadF32` 前后**净增两条 M 侧 BufferID**（禁用 set/wait flag）：
`BufAcquire<PIPE_M>(GP_BUF_L0C)` 在 `MmadF32` 之前（WAR：挡上一 tile 的 Fixpipe 读）；
`BufRelease<PIPE_M>(GP_BUF_L0C)` 在 `BufRelease<PIPE_M>(GP_BUF_L0)` 之后、FIX 侧 acquire 之前（RAW）。
并修正 `DoMmad` 顶部注释使其与实现一致。**不动** L1 布局/资源记账；**不动**复审登记的 P2-2（只登记不修）。

## 预期（先写后跑）
| 项 | 做法 | 预期 |
|---|---|---|
| **A（决定性）** | `M15GP_ONLY_M=4097` × 10 | **10/10 OK**；`multi-bit ECC` 行数 **0**；无 `mte error info`（改前 10/10 FAIL） |
| **B（回归）** | `M15GP_ONLY_M=1` × 10 | 10/10 OK（改前 50/50 OK） |
| **C（顺带）** | `M15GP_ONLY_M=17` × 10 | 不再挂（改前 26/30 OK，即 4/30 挂） |
| **D（数值判据）** | `m=4097` 能跑完 ⇒ 跑 `check_ref.py` | 拿到**合格数值读数**（改前因异常从来没有可判对象） |
| **E（签名）** | 若有失败档 | 仍按稳定签名比对（`mte error info` 低 8 位 `…0202ce` + `vec`/`cube` 全 0）；**签名不同如实区分** |

## 命令（逐字；每次 run 一次独立进锁、`flock -w 300`、`timeout 25`、进锁前 `npu-smi` 复查）
```
M15GP_ONLY_M=<m> timeout 25 flock -w 300 /tmp/npu0.lock stdbuf -oL ./build/m23_gdn_prefill \
    > evidence/logs/fix_M<m>_<i>.log 2>&1; rc=$?
/usr/local/python3.12.13/bin/python3 check_ref.py            # D：判 m23_m4097_* 与 m23_m1_*
```
**若 A 仍复现**：如实报，保留这一对（仓内 10 处 + M106 实测都说明它必需），并说明"还有别的洞"，不改判据。
