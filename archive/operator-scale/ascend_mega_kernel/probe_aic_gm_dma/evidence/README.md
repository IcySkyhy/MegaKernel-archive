# M141 `probe_aic_gm_dma` 证据索引与结论段

> 本文件只做**索引与措辞**：它**不新增、不改动**任何读数 / 日志 / dump / 哈希。
> 所有读数以本目录下已归档文件为准（原样保留）。工程说明与变体表见上一级 `../README.md`。

## 结论段（权威口径）

**根因（人类逐字纠正，2026-10-04）**：
「**AIC 就访问不了 UB，当然不能 UB->GM**」＋「**这是L1 to UB，不是直接访问UB**」——
即 **AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**。
这是**架构层定性**（人类给的常识）；本档读数与它**一致**，是对它的**后果 / 一致性证据**。

**先前口径的归位**：本档早先把根因写成「编译期 raw MTE3 被拒 + 软件层 AIV 门控成空」。
按本次纠正，那两条是**后果证据**（保留为读数），**不是根因**。

**分层口径（人类逐字，见 `../README.md` §1.5）**：人类逐字「**这是L1 to UB，不是直接访问UB**」。
本档根因＝**AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**；AIC 侧 `L1→UB` 的 MTE1 分支
（真实 `copy_cbuf_to_ubuf` 调用，头文件逐字见 §1.5）是 **pipeline 搬移**、不是直接访问 UB，
**层级不同、不构成反证**；其运行期效果未隔离验证（限度）。

## 主证据（根因最直接的观测证据）

| 读数（原样） | 出处（文件:行） |
|---|---|
| AIC **标量**访问 UB → `launch=FAILED err=507015` | `logs/aic_scalar_ub.log:18,21,24` |
| plog `error code = 271` | `logs/aic_scalar_ub.plog_excerpt.txt:4` |
| plog `errorStr: The address for scalar to access the internal buffer is out of bounds` | `logs/aic_scalar_ub.plog_excerpt.txt:5` |
| plog `rtStreamSynchronize:ErrCode=507015, desc=[aicore exception]` | `logs/aic_scalar_ub.plog_excerpt.txt:12` |

「internal buffer 地址越界」是**地址空间层面**的报错，与「AIC 不能**直接**访问 UB」**逐字同向**。

## 后果证据（一致性读数，不等同根因）

| 读数（原样） | 出处（文件:行） |
|---|---|
| raw MTE3 `copy_ubuf_to_gm`（UB→GM）在 AIC 编译期被拒 | `compile_reject/raw_ub2gm_aic.log:1,4,5`；`compile_reject/compile_matrix.txt:7` |
| 基础 API `DataCopy(GM,UB)` 在 AIC 上 `landed=0` | `logs/ub2gm_api.log:18,21,24` |
| C API `asc_copy_ub2gm` 在 AIC 上 `landed=0` | `logs/ub2gm_capi.log:18,21,24` |

## 对照读数

| 读数（原样） | 出处（文件:行） |
|---|---|
| AIC FIXP `L0C→GM` 落盘一致 | `logs/fixp_l0c2gm.log:18,21,24` |
| AIV `UB→GM`（MTE3）落盘一致（正对照） | `logs/aiv_ub2gm.log:18,21,24` |
| 负对照 `neg_fixp_short` 判据变红 | `logs/neg_fixp_short.log:18,21,24` |

## 证据索引

| 路径 | 内容 |
|---|---|
| `compile_reject/` | 零设备编译期取证：raw/API 小探针、`*.log`、`compile_matrix.txt`、`repro.sh` |
| `logs/` | 每档 `<variant>.log`（锁内 `npu-smi` + 逐次 `SUMMARY` + `exit=` + `AGG`）、`matrix.txt`、`check_ref.log`、`commands.txt`、`aic_scalar_ub.plog_excerpt.txt` |
| `dumps/` | `out_<variant>_run<N>.bin`（GM 出口 16 KB dump） |
| `npu_smi_snapshots.txt` | 锁内 `npu-smi` 快照 |

> M153 只改措辞（上一级 `README.md` 与本文件）；`compile_reject/`、`logs/`、`dumps/` 与源码/脚本哈希均未改动。
