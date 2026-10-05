# M121 r2 自审表 —— 计数类陈述 与 "有产物"类断言 的逐条对照

**范围（写清覆盖什么、不覆盖什么）**：
- **覆盖**：`probe_ub2l1/README.md` 的正文与 `probe_ub2l1/evidence/**`（`version_diff.sh`/`slot.sh`/`retake.sh` 的脚本头、
  `logs/*.log`、`logs/commands.txt`、`logs/device_batch.log`、`logs/attempts.log`、`logs/retake_attempts.log`、
  `dumps/` 目录）里**属于"计数"或"声称有产物"**的陈述。
- **不覆盖**（避免被读成"全仓都已核"）：
  ① §A 第 1 层对**外部快照**的引用（`/workspace/asc-devkit` 的文档与样例、rev `648a601…`）；
  ② §A 第 2 层与 §4/§6.3 对**本机 CANN 头文件/实现**的 `文件:行` 引用（如 `kernel_operator_data_copy_impl.h:582`、
  `kernel_utils.h:36`、`asc_sync_block_arrive_impl.h:32`、`asc_copy_ub2l1_impl.h:31`、`cmath.h`）；
  ③ §8 里 CANN plog 的**原始件**（仓外，`~/ascend/log/debug/plog/`；本仓只保留抄录件）。
  这三类由本表之外单独核对（r3 复审已独立抽核 ② 与 ①，全部属实）。
- **判定口径**：`一致`（产物逐字/逐项支撑该陈述）／`已改`（本轮发现不符并已改陈述，附原措辞）／
  `不适用`（背景陈述等非计数结论）。
**方法**：先在 README 里把两类候选句扫出来（计数：`[0-9]+ ?(次|档|个|条|步|处|支)`、`×[0-9]`、`[0-9]+/[0-9]+`、`共 [0-9]+`；
产物：`见 \``、`记在`、`转录`、`落在`、`log 内`、`evidence/…`），再逐条去产物里核对。
**判定用到的可复算命令**（都在 `probe_ub2l1/` 下执行）：

```bash
# 逐档 log 的批次与判定
for f in evidence/logs/*.log; do printf '%-34s %s %s\n' "$(basename $f)" "$(grep -m1 'at=' $f|sed 's/.*at=//')" "$(grep -oE 'VERDICT=[A-Z_]+|maxAbs=[0-9.e+-]+|^### rc=[0-9]+' $f|tr '\n' ' ')"; done
# 只追加的尝试流水 / 转录
cat evidence/logs/attempts.log; cat evidence/logs/retake_attempts.log
# run_probes 那批每档一行 rc=
grep -nE '^rc=' evidence/logs/device_batch.log
# 源码内容寻址
diff <(sed -n '/内容指纹/,/^$/p' evidence/logs/commands.txt|grep "  "|sort) <(sha256sum probe_ub2l1.asc CMakeLists.txt run_probes.sh check_ref.py evidence/version_diff.sh evidence/slot.sh evidence/retake.sh|sort)
# README 里出现的 evidence/ 路径与逐档 log 名是否存在
grep -oE "evidence/[A-Za-z0-9_./,{}-]+" README.md | sed 's/[.,;)]*$//' | sort -u | while read -r p; do [ -e "$p" ] || echo "缺失 $p"; done
```

---

## 表 1 —— 计数类陈述（逐条）

| # | 位置 | 陈述 | 支撑产物 | 判定 |
|---|---|---|---|---|
| 1 | §0 开头（README:4） | 往返 `8192/8192` 逐字节相同 | `evidence/logs/capi_rt_k0_default.log`（`same=8192 diff=0`）+ `dumps/capi_rt/rt_in.bin`==`rt_out.bin`（`check_ref.py` J4） | 一致 |
| 2 | §0 开头（README:5） | `32×48×64`、`1536 个元素`、`6.18e-07`、全在 `[0,1e-6)` | 几何常量在 `probe_ub2l1.asc`（`M_DIM=32,N_DIM=48,K_DIM=64` ⇒ `C_ELEMS=1536`）；数值见 `basic_mmad_k0_default.log` | 一致 |
| 3 | §0 第 2 行（README:22） | 两构建 `maxAbs=6.183982e-07`、`exact=944/1536` | `basic_mmad_k0_default.log`、`basic_mmad_k0_ssbuf.log` | 一致 |
| 4 | §0 第 4 行（README:24） | `1536 个元素全对` | 同 #3（误差分桶 `[0,1e-6)=1536`） | 一致 |
| 5 | §0 第 4b 行（README:25） | opt=0/opt=2 各 3 次；opt=1 错 | `capi_mmad_{k0,a,b}_opt{0,1,2}.log` 共 9 个档（实测 3/3/3） | 一致 |
| 6 | §0 第 6 行（README:27） | 64KB → `230.75 / 231.24` GB/s，`85.02–85.21 µs` | `capi_bw_64k_300.log`(230.7470/85.205)、`basic_bw_64k_300_default.log`(231.2410/85.023)、`basic_bw_64k_300_ssbuf.log`(231.2382/85.024) | 一致 |
| 7 | §0 第 7 行（README:28） | `1KB→146.32`、`8KB→212.34`、`64KB→230.75` | `capi_bw_1k_2k.log`、`capi_bw_8k_1k.log`、`capi_bw_64k_300.log` | 一致 |
| 8 | §0 第 8 行（README:29） | kill=1：8 次样本 6 次坏 | 逐档：`capi_rt_k1_r1..r4.log`(MISMATCH×4)、`capi_rt_k1_nowait.log`(BYTE_EXACT)、`basic_mmad_k1_r1/r2.log`(错×2)、`basic_mmad_k1_nowait.log`(09-27 读对)——即 §6.4 的两张表；`device_batch.log:79-95` 只覆盖其中 6 条（r1..r4 + basic r1/r2）的 `rc=`，10-04 的 `capi_rt_k1_nowait` 在 `:133`，09-27 的 `basic_mmad_k1_nowait` **不在** `device_batch.log` 里（只在各自 log） | 已改（原写 7/5；r2 表此处曾误串到"#21"，r3 修正引用） |
| 9 | §0 第 8 行（README:29） | kill=2 两支各 1 次挂死；kill=5 两支各 1 次挂死 | `device_batch.log:97,100`（k2/k5 的 `rc=124 … executed=yes`）、`capi_rt_k5_halfarrive.log`/`basic_mmad_k5_halfarrive.log`（`### rc=124`） | 一致 |
| 10 | §0 第 9 行（README:30） | kill=1：8 次里 6 坏 2 对 | 同 #8（8 个逐档 log + §6.4 两张表） | 已改（原写 7/5） |
| 10b | §0 第 9 行（README:30） | "（**9 个** log 名见 §6.4 小表）" | `ls evidence/logs/{capi_rt,basic_mmad}_k1_*.log` = **8** 个；§6.4 小表也是 8 行 | 已改（改为"8 个 log 名"；r2 表曾漏掉这一条，r3 补上） |
| 11 | §0 第 10 行（README:31） | 官方样例 `6 处` 编译错 | `evidence/logs/version_diff.log` §2（`6 errors generated.`）+ `bisheng rc=1` | 一致 |
| 12 | §0 第 11 行（README:32） | `capi-step` 9 步都取到 `NO_FAULT` | `capi_step1..9.log` 九个档都有 `VERDICT=NO_FAULT`（§B.4 表）；其中 **`capi_step8.log` 内含两次运行**（`grep -c '^\[PUB2L1\] mode=' = 2`，两次都 `NO_FAULT`），其余八档各 1 次 | 一致（r2 表此处写"各 1 个"，r3 改准） |
| 13 | §3.1 引用块（README:104-108） | `hostWall=1.895 ms`、`same=8192 diff=0`，原件 `at=2026-10-04T09:13:03` | `capi_rt_k0_default.log`（入库版）逐字：`at=09:13:03`、`hostWall=1.895`、`same=8192 diff=0` | 已改（r2 表误判"一致"：当时陈述写的是 `hostWall=1.901` + `at=08:20:xx`，而入库 log 已是 r1-fix 重采后的 `1.895`/`09:13:03`；`1.901`/`08:48:24` 只存在于 `ae5c25d` 版） |
| 14 | §3.2 引用块（README:120-122） | `M=32 N=48 K=64`、`maxAbs=6.183982e-07`、`exact=944/1536` | `basic_mmad_k0_default.log` 逐字 | 一致 |
| 15 | §5.2 引用块（README:172-174） | `p50=5.308539e-08` 等三行 | `basic_mmad_k0_default.log` 逐字 | 一致 |
| 16 | §5.3（README:186） | ND→NZ：A 的 `4 个 C0 列块`、B 的 `3 个列块` | 由 `probe_ub2l1.asc` 的 `K_DIM/16=4`、`N_DIM/16=3` 得出（`UbNdToNz` 按列块循环） | 一致（源码推导，非读数） |
| 17 | §5.4 表（README:202-208） | opt 各 3 次；opt=1 的 `1.135639e+01`（2 次）/`1.276375e+01`（1 次） | 9 个 opt log：opt1 三次为 1.135639e+01、1.135639e+01、1.276375e+01 | 一致 |
| 18 | §5.4（README:223） | opt=0/1/2 **各跑 3 次**稳定 | 同 #17 | 已改（原写"各跑 2 次"） |
| 19 | §5.5（README:231） | 各 dump 的 `a.bin`/`b.bin` 逐位相同 | `md5sum evidence/dumps/*/a.bin` 与 `*/b.bin` 各只有 1 个唯一值 | 一致 |
| 20 | §5.5（README:233） | `p50=5.355105e-08 p90=1.788139e-07 p99=3.479421e-07`、`relative max=1.677e-05` | `evidence/logs/check_ref.log` 第 1 段（`capi_mmad`）逐字 | 一致 |
| 21 | §6.1（README:245） | 带宽档 L1 目的 `4 个槽` | `probe_ub2l1.asc` 的 `BW_SLOTS=4`；log 内 `slots=4` | 一致 |
| 22 | §6.1 表（README:249-253） | 5 行带宽值（146.32/212.34/230.75/231.24/231.24） | 5 个 `*_bw_*.log` 的 `GBps=` | 一致 |
| 23 | §6.1（README:256） | 三条 64KB 落在 `230.75–231.24`、差 `<0.3%` | 同 #22 的 64K 三行（230.7470 / 231.2410 / 231.2382） | 一致 |
| 24 | §6.1（README:260） | 09-27 三值 `230.44 / 231.20 / 231.31` | `git show 21fb0aa:probe_ub2l1/evidence/logs/capi_bw_64k_300.log` 等三档的 `GBps=`（231.3144） | 已改（原写 231.28） |
| 25 | §6.1（README:271） | 4KB 操作数 "150–200 GB/s、20–27 µs" | **推算**（由 #22 的表线性内插）——文中已改标"（推算，不是实测）" | 已改 |
| 26 | §6.4 表（README:298-300） | kill=1 的 `4/4`、`1/1`、`2/2` | `capi_rt_k1_r1..r4.log`（MISMATCH×4）、`capi_rt_k1_nowait.log`（BYTE_EXACT）、`basic_mmad_k1_r1/r2.log`（错×2） | 一致 |
| 27 | §6.4 样本表（README:310-313） | `×4 / ×1 / ×2 / ×1` | 同 #26（`basic_mmad_k1_nowait.log` 为 09-27 无 `at=`，其 `tf=Sep 27 2026`） | 一致 |
| 28 | §6.4（README:315） | 共 8 次：6 坏 2 对 | 4+1+2+1 = 8，其中坏 = 4+2 = 6 | 已改（原写 7/5） |
| 29 | §6.4（README:322、325） | `8 次里 6 次`、`8 次里 6 次读错` | 同 #28 | 已改 |
| 30 | §6.4（README:323-324） | 10-04 内 `4 次 MISMATCH` + `1 次读对`；09-27 `1 次读对` | 同 #27 | 一致 |
| 31 | §7（README:343） | 基础样例 `6 处`、`rc=1` | `version_diff.log` §2（`6 errors generated.` + `bisheng rc=1`） | 一致 |
| 32 | §7（README:349） | 同形行号 `84/85/86/95/96` | 同上逐字诊断里的行号（第 83 行也出现，共 6 处） | 一致（举例非穷举，已写"同形"） |
| 33 | §8（README:390-395） | `9 步` 二分；step5..8 首轮、step1..4,9 重采 | `capi_step1..9.log`（§B.4 表逐档 `at=`） | 一致 |
| 34 | §B.2（README:518） | `8 次 = 6 错 2 对` | 同 #28 | 已改 |
| 35 | §B.4 表（README:541-547） | 各步"重采前未取得读数的尝试"= 0/0/1/0/2/1/0 | `evidence/logs/retake_attempts.log` 的转录（step3×1、step9×2、k4×1） | 已改（原写"已记 log"但无产物） |
| 36 | §B.4（README:555） | 本轮 `4 次`未取得读数的尝试 | 同 #35 | 已改 |
| 37 | §B.5 表（README:572-573） | 头条两档重采的时间与结果 | `capi_rt_k0_default.log`(09:13:03 BYTE_EXACT)、`basic_mmad_k0_default.log`(09:13:52 maxAbs=6.183982e-07) | 一致 |
| 38 | §B（README:493） | "11 个 agent 抢同一把锁" | 环境观察：塔状态的 roster 规模 + `npu-smi`/`ps -ef \| grep flock` 当时可见的多家进程；**不是读数**，写的是运行背景 | 一致（背景陈述，非计数结论） |

---

## 表 2 —— "有产物"类断言（逐条）

| # | 位置 | 断言 | 核对 | 判定 |
|---|---|---|---|---|
| 1 | §1（README:45） | "命令 + 逐字输出在 `evidence/logs/`" | 目录存在，逐档 log 齐全（见下 #6/#7） | 一致 |
| 2 | §2（README:94） | "读数落在 `evidence/logs/`，check_ref 的输入/输出在 `evidence/dumps/`" | 两目录存在；`check_ref.log` 6 段与 6 个 dump 目录一一对应 | 一致 |
| 3 | §3.1（README:103-108） | "入库 log 的节选（逐字原件 `capi_rt_k0_default.log`，`at=2026-10-04T09:13:03`）" | 引用块四行与入库 log 逐字相同（`hostWall=1.895 ms`、`same=8192 diff=0`、`VERDICT=BYTE_EXACT`），首行按 `...` 省略、已在正文标明 | 已改（r2 表误判"一致"：当时陈述写 `hostWall=1.901` + `at=08:20:xx`，与入库 log 不符；已按入库 log 逐字改） |
| 4 | §3.1（README:111） | "原始读数所在的 log：`capi_rt_k0_default.log`、`capi_rt_k0_ssbuf.log`" | 两文件存在 | 一致 |
| 5 | §5.4（README:223） | "逐档 `capi_mmad_k0_opt{0,1,2}.log` + `capi_mmad_{a,b}_opt{0,1,2}.log`" | 9 个文件都存在 | 一致 |
| 6 | §6.1（README:247-248） | 五个带宽档的 `at=` | 五个 `*_bw_*.log` 内确有该 `at=` | 一致 |
| 7 | §6.4（README:333-334） | "**多数档** log 内 `### <name> at=` 自证批次；09-27 旧格式档以 `tf=` 为准" | 10-04 那批档普遍有 `at=`；**`basic_mmad_k1_nowait.log`（09-27 旧格式）无 `at=`**（`grep -c at= = 0`，只有 `tf=Sep 27 2026`）⇒ 原措辞"每档都有"不成立；另 09-27 的 `capi_rt_k1_nowait` 工作树副本已被 10-04 覆盖（旧版在 `git show 21fb0aa:` 里） | 已改（原写"每档 log 都有"，r3 收窄） |
| 8 | §7（README:341） | "逐字诊断全部在 `evidence/logs/version_diff.log`" | 文件存在，四段 rc 与诊断都在（rc 取自 bisheng） | 一致 |
| 9 | §8 / §0-11 | "runtime 逐字日志留在 `evidence/logs/`"（r1 版措辞） | **r1 版不成立**：runtime 行在仓外 plog，`evidence/logs/*.log` 里 grep 不到 | 已改（现指向新归档 `m121_aicore_exception_507015.plog.txt`，并注明原始件在 `~/ascend/log/debug/plog/`） |
| 10 | §9 目录树（README:417,421） | 列出的 `version_diff.sh`/`slot.sh`/`retake.sh`/`attempts.log`/`retake_attempts.log` | 六个路径都在 | 一致 |
| 11 | §B.1（README:508） | "09-27 的 5 份仍可从 git 历史 `21fb0aa` 取回" | `git show 21fb0aa:probe_ub2l1/evidence/logs/capi_step{1,2,3,4,9}.log` 有内容 | 一致 |
| 12 | §B.4（README:556） | "转录在 `evidence/logs/retake_attempts.log`" | 文件存在（表头标明是转录，非 slot.sh 写入） | 已改（r1 版称"已记 log") |
| 13 | §B.4（README:561-562） | "run_probes 那批未执行的尝试记在 `device_batch.log`；补读数那批记在 `retake_attempts.log`" | `device_batch.log` 4 条 LAF；`retake_attempts.log` 4 条 LAF | 已改（r1 版一句笼统说"逐条记在 device_batch.log"） |
| 14 | §9/§B（README:419-421, 555-558） | 新增 `attempts.log`（slot.sh 只追加流水） | 文件存在，含机制自测 2 行（表头已注明） | 一致（本轮新建） |
| 15 | 全文 | README 提到的每个 `evidence/…` 路径 | 逐个 `[ -e ]` 检查：无缺失（命令见表头） | 一致 |
| 16 | 全文 | README 提到的每个逐档 log 名与 dump 目录 | 41 个 log 名 + 6 个 dump 目录逐个检查：无缺失 | 一致 |

---

## 表 3 —— 本轮（r2-fix）改动清单

| 项 | 改法 |
|---|---|
| P2-1 计数 | `7 次/5 次` → `8 次/6 次`，共 6 处（§0 第 8、9 行；§6.4 表下两处与第 1/2 条；§B.2） |
| P2-2 记录 | 新增 `evidence/logs/retake_attempts.log`（4 次未取得读数尝试的逐字转录，表头写清来源与"非 slot.sh 写入"）；§B.4 表加"重采前未取得读数的尝试"列并写明记录在哪；`slot.sh` 新增只追加的 `attempts.log`（已用两次真实调用自测） |
| 自查新发现 A | §0 第 11 行的"逐字 runtime 日志留在 `evidence/logs/`"不成立 ⇒ 新增归档 `evidence/logs/m121_aicore_exception_507015.plog.txt`（逐字抄录 plog，标明来源），§0/§8 改指向它 |
| 自查新发现 B | §5.4"opt=0/1/2 各跑 2 次"与表内"各 3 次"冲突 ⇒ 改"各跑 3 次" |
| 自查新发现 C | §6.1"4KB → 150–200 GB/s、20–27 µs"是推算 ⇒ 标明"（推算，不是实测）" |
| 纯文字 | 09-27 SSBUF 带宽 `231.28` → `231.31`（log `GBps=231.3144`）；§B 多余空行与重复 `---` 清理 |

### 表 3b —— r3-fix 改动清单（本轮）

| 项 | 改法 |
|---|---|
| r3-P2-1 引文不符 | `README.md:108`/`:296` 的 `hostWall=1.901` → **`1.895`**；`:104` 的 `at=2026-10-04T08:20:xx` → **`at=2026-10-04T09:13:03`**（按入库 `capi_rt_k0_default.log` 逐字；`1.901`/`08:48:24` 是 `ae5c25d` 版） |
| r3-P2-2 计数 | `README.md:30` 的"9 个 log 名" → **"8 个 log 名"**（其余 5 处本就是 8） |
| 表 1 第 13 行 / 表 2 第 3 行 | 判定由 **一致 → 已改**，并写明原措辞为何与产物不符 |
| 表 1 新增第 10b 行 | 补上"9 个 log 名"这条（r2 表漏项），判定"已改" |
| 表 1 第 8 行 | 去掉串号的"#21"交叉引用，改为指向 §6.4 两张表 + 8 个逐档 log；并把 `device_batch.log:79-95` 的覆盖范围写准（6 条）+ 指出 `:133` 与"09-27 那次不在 device_batch" |
| 表 1 第 12 行 | `capi_step1..9.log`"各 1 个" → 写准 **`capi_step8.log` 内含两次运行**（都 `NO_FAULT`） |
| 表 2 第 7 行 | "每档 log 都有 `at=`" → **"多数档有；09-27 旧格式档以 `tf=` 为准"**；`README.md:333-334` 与 `:527` 同步收窄 |
| 表头 | 写明**覆盖范围**与**不覆盖的三类**（§A 外部快照引用、§A/§4/§6.3 的 `*.h:行` 引用、§8 的仓外 plog 原始件），避免被读成"全仓都已核" |
