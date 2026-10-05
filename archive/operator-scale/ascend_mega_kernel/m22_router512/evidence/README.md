# M22 evidence 索引与复跑方法

本目录是 `m22_router512` 的归档证据。判据口径见 `docs/17-verification-standard.md`；
结论与判据清单见 `../README.md` §6。

## 1. 目录

| 路径 | 内容 |
|---|---|
| `mode2/<case>/` | **2 路归并树**（`m22_router512`，默认目标）的 10 档 dump |
| `mode4/<case>/` | **4 路归并树**（`m22_router512_m4`）的同 10 档 dump |
| `dump_sha256.txt` | 所有 dump 的 sha256（含未入库的两个大张量） |
| `determinism_rerun.txt` | clean rebuild + 新进程重跑 vs 首次归档的逐字节对照（0 处不同） |
| `run_mode2.log` / `run_mode4.log` | 机内自检完整输出（判定 100/100、档内 guard 40/40 + 3 条非空洞性 guard） |
| `check_ref_mode2.log` / `check_ref_mode4.log` | 离线 `check_ref.py` 完整输出（每棵树判定 99/99、guard 29/29，逐项判定/报告/guard 分栏）。**M56 已按当前代码重跑刷新**（此前是 pre-M46 快照：报告项 R0 在 m=4097 档原为 `125 槽 / 20 行`，现全档 `0 槽 / 0 行`；语义解释见 `../README.md` §5 的「R0 在 M46 之后语义改变」表） |
| `ftz_modeling_scan.txt` | 设备 softmax 次正规 FTZ 口径扫描（`../README.md` §5 的表）。**FTZ 影响的原始证据冻结在此**（扫描表首行 125 槽 / 20 行），不随日志重跑改变 |
| `cost_model.txt` | 归并树每级代价 + GEMV 指令/流量模型（`../README.md` §3.3/§4.3） |
| `merge_mode_equiv.txt` | 2 路 vs 4 路归并树的逐字节等价性 |

每档 dump 的 9 个张量：`router_logits.bin`(fp32)、`topk_ids.bin`(int32)、`topk_weights.bin`(bf16)、
`expert_counts.bin`(int32)、`expert_slot_base.bin`(int32)、`perm_src_token.bin`(int32)、
`perm_expert.bin`(int32)、`group_list_i64.bin`(int64)、`route_scalars.bin`(int32[4])，
每个附一个同名 `.json`（shape/dtype/字节序/size）。

**未入库的两个大张量**（sha256 见 `dump_sha256.txt`）：
`real_m4097/x.bin`（21MB）、`real_m4097/router_logits.bin`（8.4MB）。
这两个张量**没有**对应的 `.json` 清单文件（避免目录里出现"有清单无数据"的误读）；
它们的 shape/dtype 与同档同名张量一致：`x.bin` = `[4097, 2560]` bf16、
`router_logits.bin` = `[4097, 512]` fp32。

## 2. 复跑

```bash
cd m22_router512

# 一条命令重生成两份归档日志（10 档 × 2 棵树；m=4097 档自动取 x 的 sha256）
tools/rerun_check_ref_logs.sh          # rc=0 表示 20 档全部 rc=0

# 或逐档手动：
# 有 x.bin 的 9 档（含 4 个受控非均匀档）
/usr/local/python3.12.13/bin/python3.12 check_ref.py evidence/mode2/real_m33 \
      --w data/router_weight.bin
/usr/local/python3.12.13/bin/python3.12 check_ref.py evidence/mode2/nu_m64 --w onehot

# m=4097 档：x.bin 未入库 → 用 seed 重生成并核对 sha256；logits 未入库 → J3 转报告项
X=$(awk '/\/real_m4097\/x.bin/{print $1; exit}' evidence/dump_sha256.txt)
/usr/local/python3.12.13/bin/python3.12 check_ref.py evidence/mode2/real_m4097 \
      --w data/router_weight.bin --x-seed 5 --x-sha256 $X --no-logits

# 报告项复现
/usr/local/python3.12.13/bin/python3.12 tools/cost_model.py
/usr/local/python3.12.13/bin/python3.12 tools/ftz_scan.py evidence/mode2/real_m4097 \
      --w data/router_weight.bin --x-seed 5
```

`mode4/` 的同名档用同样命令（把 `mode2` 换成 `mode4`）即可，两侧结论一致。
`tools/rerun_check_ref_logs.sh` 的档序与归档日志一致（`real_m*` → `real_m4097` → `nu_m*`），
便于 `git diff` 逐档对读。

## 3. 判定计数（可复算）

| 校验链 | 判定项 | guard |
|---|---|---|
| 机内自检 `run_mode2.log` / `run_mode4.log` | **100/100**（10 档 × 10 项） | 档内 40/40（10 档 × 4）+ 确定性 2 + 输入敏感性 1 |
| 离线 `check_ref_mode2.log` / `_mode4.log` | **99/99**（5 档 ×10 + m=4097 档 9 + 4 档 ×10） | **29/29**（9 档 × 3 + m=4097 档 2） |

m=4097 档判定为 9 项（其余档 10 项），因为它的 `router_logits` dump 未入库 ⇒ J3（**T3 逐元素界**）
转为报告项；该档 logits 的 T3 界占用与越界元素数在 `run_mode2.log` 的 `T3(logits)` 行
（越界 0，界占用 0.2299%，**对机内 C++ host 参考**）。同理该档只有 2 条 guard：`G4x` 已升级为
§3.1 的**硬闸门**（不再计入 guard 栏），只剩 `G2`/`G3`。

### 3.1 x 重建的硬闸门（tower 条件 ①）

归档必须足以**确定性重建** x，且该校验是**强制**的：

| 闸门 | 条件 | 行为 |
|---|---|---|
| X1 | 走 `--x-seed` 重建路径时**必须**给 `--x-sha256` | 缺失即 `FAIL` + 退出码 1，不进入判定 |
| X2 | 重建 x 的 sha256 必须等于 `--x-sha256` | 不符即 `FAIL` + 退出码 1 |
| X3 | 同时给 `--x-seed` 与已存在的 `x.bin` 时，重建 x 必须与 dump x 逐字节一致 | 不符即 `FAIL` + 退出码 1 |

（此前 X1/X2 只打印一条 guard、不影响退出码，已在 review round-1 的 P2-1 修正。）

`check_ref.py` 的 J1..J10 与机内自检的 J1..J10 是同一批判据的不同排列（机内顺序为
logits/ids/weights/counts/base/permSrc/permExp/g64/scalars/Σ；离线的 J1=ids、J2=weights、
J3=logits、J4..J10 同名），两者都跑、都归档；档位（T1/T3）与 §6.1 的分档表一致。

## 4. 环境

| 项 | 值 |
|---|---|
| 设备 | Ascend950PR，单卡 28 AIC + 56 AIV。**本交付只用 1 个 AIV block（`<<<1,0,stream>>>`），未做 `mix(1,2)` 全核启动**，也不满足用户对层 kernel 的全核要求；多核设计见 `../README.md` §9.1（未实现） |
| CANN | 9.1.0（`/usr/local/Ascend/ascend-toolkit/set_env.sh`） |
| 编译 | `find_package(ASC)` + `--npu-arch=dav-3510`，`CMAKE_BUILD_TYPE=Release` |
| python | `/usr/local/python3.12.13/bin/python3.12`（numpy 只在此解释器下） |
| 权重 | `data/router_weight.bin`，`mlp.gate.weight` [512,2560] bf16 逐字节切片，sha256 `966e4d1ca7c18994a9311e82f0074584fe09e65e81eb8ddefe0a4dceebfc6ac6`（与 `m7_router_topk/data/router_weight.bin` 相同，脚本断言） |

注意：本机与 10+ 个 agent 共用，`run_*.log` 里的 kernel 时间（`t=` 列）是报告项；
首档 `real_m1` 含设备初始化开销（4.6s 级）不代表 kernel 时间。

## 5. R0 的语义变化（M56）

`r0_semantics_recheck.txt` 归档了一次「R0 非空洞性对照」：同一份 m=4097 设备档数据上，
**当前 golden（已建模 FTZ）→ R0 = 0 槽 / 0 行**，而**把 golden 的 FTZ 建模退回（= pre-M46 行为）
→ R0 = 125 槽 / 20 行**（与 M46 commit message 一致）。⇒ 重跑后归档里 R0 全档 `0/0` 是
**M46 补上 golden 侧缺口后的预期值**，不是匹配器失效、也不是档位退化。复算：
`/usr/local/python3.12.13/bin/python3 tools/r0_semantics_check.py`。
