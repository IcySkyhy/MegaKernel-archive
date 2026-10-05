# M178 evidence —— kv 几何 doc sync 与 DECODE_CTX 留档（M184 已修）

本目录是 **M178**（`kv geometry doc sync and decode ctx sizing`）的设备见证与一键复算。它把
`m15_layer_loop` 的 attention cache 文档几何对齐到**权威头文件** `m15_attn_kv.h`，并留档
`DECODE_CTX` 的算术裁定。**M184** 在同一目录续上 `DECODE_CTX` 的**修复**与改前/改后设备读数。

## 0. 对应 commit 与哈希

| 项 | 值 |
|---|---|
| M178 分支 | `feat/m178-kv-geometry-doc-sync-and-decode-ctx` |
| M178 代码同步 commit（README 几何 + 两处头注释） | `a1941c2` — *M178: sync m15 README KV geometry to authoritative header; document DECODE_CTX decode ctx gap* |
| M184 修复分支 | `feat/m185-decode-ctx-4097-sizing-fix`（已 `git merge main` @ `70fc00b`，M181/M183 在树内） |
| M184 修复 commit（源码 + 文档 + 本证据） | 本目录所在 tip（`git log -1 --oneline`） |
| M184 改前/改后设备日志 | `logs/m184_before_{kv,prefill}.log` / `logs/m184_runs_{kv,prefill}.log`（**均在合并后的树上重取**） |
| M184 二进制 sha256（改后，`m15_layer_loop/build/m15_layer_loop`） | `1595ee63903ac4226b9533e1bff1283b01e71a5ca812456307bb0cfe9f69b424` |
| M184 同步的文档 | `m15_layer_loop/README.md`、`docs/15-prefill-design.md`、`docs/19-gdn-prefill-scan-selection.md` |
| M178 二进制 sha256 | `d2cc02d9b82e86f29c87c3daa57c4b7c5fd676ad443cd12f04f3b068bd7cdb1a` |
| 设备日志（`logs/m178_runs_kv.log`） | 由 `reproduce.sh` 每次重跑重生成；脚本内 `sha256sum` 会把当次二进制 sha 写进日志头（日志自身 sha 不作稳定断言） |

## 1. 目录内容

| 路径 | 说明 |
|---|---|
| `reproduce.sh` | 一键复算：构建 rc=0 → README/header 几何一致性 → `runs=kv`（锁内 `npu-smi` + 逐字命令 + `exit=`）→ 断言设备日志。任一步失败**传播非零退出** |
| `logs/m178_runs_kv.log` | M178 的 `runs=kv` 设备档日志（`=== lock acquired ===`、锁内 `npu-smi` 快照、逐字命令、`exit=0`、`ALL PASS（checks=215, guards=191, fails=0）`） |
| `logs/m178_build.log` | M178 构建命令与输出（rc=0） |
| `logs/m184_before_kv.log` | **M184 改前** `runs=kv`（`DECODE_CTX = 4096`、host 打印按字面 4096） |
| `logs/m184_runs_kv.log` | **M184 改后** `runs=kv`（`DECODE_CTX = 4097`、host 打印从 `DECODE_CTX` 派生） |
| `logs/m184_before_prefill.log` | **M184 改前** `runs=prefill`（旧 `.asc` 打印） |
| `logs/m184_runs_prefill.log` | **M184 改后** `runs=prefill`（新打印，含 `DECODE_CTX=4097`） |
| `logs/m184_build.log` | **M184 改后**构建日志（rc=0，21 warning） |
| `.gitignore` | 挡 dump / 大二进制（`*.bin`/`*.dump`/`*.npy`/`out/`…），只入日志与脚本 |

**一键复算**（仓库根目录下）：

```
bash m15_layer_loop/evidence/m178_kv_geom/reproduce.sh          # 期望末行：REPRODUCE OK（0 failure）
```

脚本把**期望几何值内建**（header 常量与 README 文本），因此以后 README 再漂移会自动变红；同时它
断言 README 里**不再出现**旧值（见 §3 的旧值列）。

## 2. 权威判定（已核实）

权威侧 = `m15_attn_kv.h`（**设备读数与它逐条一致**，不是靠谁的描述更晚）：

| 量 | 权威值 | 依据 | 设备读数 |
|---|---|---|---|
| page bytes | 32,768 B | `m15_attn_kv.h:122,129` | `Kv.align … 主 KV 页 32768`（`logs/m178_runs_kv.log`） |
| block_size（页内 token） | 16 | `m15_attn_kv.h:111` | — |
| 同 head 内 token 步长 | 1,024 B（= K512+V512） | `m15_attn_kv.h:120,132` | `Kv.align … 页内 token 步长 1024` |
| head 平面 | 16,384 B | `m15_attn_kv.h:121,132` | `Kv.align … head 平面 16384 / head 槽内 K‖V = 512|512` |
| head（层）stride | 8,421,376 B（257 页） | `m15_attn_kv.h:211,224` | `Kv.cap 主 KV: … 257 页 × 32768 B/页 = 8421376 B/层` |
| 主 KV 平面 | 101,056,512 B（96.38 MiB） | `m15_attn_kv.h:214,228` | `×12 层 = 101056512 B` |
| raw ring 行宽 | 140（280 B/行） | `m15_attn_kv.h:143,145` | `raw ring: 128×2 + 24 = 280 B/行` |
| DECODE_CTX（M184 修后） | 4097（⇒ DECODE_BLOCKS 257、DECODE_COMP_ROWS 1028，见 §4） | `m15_attn_kv.h:192,202,203` | `decode(1,ctx=4097)：主 KV 257 页 = 8421376 B/层`（`logs/m184_runs_kv.log`） |

README 侧（`m15_layer_loop/README.md`）原先把主 KV 印成**旧半几何**，已在 `a1941c2` 更正为上表值。

## 3. 改动清单（旧值 → 新值，依据）

`m15_layer_loop/README.md`（Part D · M82）：

| 行 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| 1050 | 视图 `[blocks,2,16,256]`、head 平面 `各16×256`、token 步长 `512 B`、页 `16,384 B`、token `1,024 B`、层 stride `4,210,688 B` | `[blocks,2,16,512]`（H2,N16,C512，C=K256‖V256）、各 `16×512`、**同 head 内** token 步长 `1,024 B`、页 `32,768 B`、token `2,048 B`（2 头合计）、层 stride `8,421,376 B`（257 页 × 32,768） | `m15_attn_kv.h:111-124,211` |
| 1055-1057 | 两个 head 平面 `8,192 B`、token 的 `1,024 B` 不连续、两次 `512 B` 传输 | 两个 `16,384 B` 平面、token 的 `2,048 B` 不连续、两次 `1,024 B` 传输 | `m15_attn_kv.h:121,119,124` |
| 1059 | naive 落点 `+ 512` | `+ KV_TOKEN_STRIDE`（= +1,024 B） | `m15_attn_kv.h:120`；实现 `m15_attn_kv_host.h:424` |
| 1065 | 对齐表 `16,384 / 8,192 / 512` | `32,768 / 16,384 / 1,024` | `m15_attn_kv.h:122,121,120` |
| 1149 | `257 页 × 16,384 = 4,210,688 B/层`；decode `256 页 = 4,194,304 B/层` | `257 页 × 32,768 = 8,421,376 B/层`；decode `257 页 × 32,768 = 8,421,376 B/层`（**M184 更新**） | `m15_attn_kv.h:200,202,211` |
| 1154 | 12 层合计 `50,528,256 B（48.19 MiB）` | `101,056,512 B（96.38 MiB）`（= 101,056,512 / 2²⁰ = 96.375） | `m15_attn_kv.h:214,228` |
| 1173-1174 | H_Alloc 示例 `4.211 MB/层 × 16,384 B` | `8.421 MB/层 × 32,768 B` | 与当前二进制实打印一致（见 `logs/m178_runs_kv.log` 的 H_Alloc 行） |

另两处**头文件内注释**（同 commit）：
- `m15_attn_kv_host.h:202`：一个 token 的 `1,024 B` 不连续 → `2,048 B`（2 head × 1,024 B）。
- `m15_attn_kv.h:228`：`static_assert` 消息 `96.40 MiB` → `96.38 MiB`（96.40 是错的舍入）。

`reproduce.sh` 的静态检查把这批值**双向钉住**：header 侧 `static_assert` 文本 + README 侧文本，
并断言 README 内旧值（`4,210,688`/`50,528,256`/`4,194,304`/`4.211`/`48.19`/`各 16×256`/
`token 步长 512`/`两个 8,192 B`/`两次 512 B`）**已清除**。

## 4. `DECODE_CTX` 裁定与修复（M184：**已修 + 设备读数**）

### 4.1 算术推导（M178 已核实，M184 沿用，结论未变）

- 页分配 `BlocksFor(m) = (m + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS = ceil(m/16)` —— `m15_attn_kv.h:195`。
- 每个 token 写 1 页的 1 个 slot：`block = pos/16`、`slot = pos%16` —— `m15_attn_kv.h:246-247`。
- 4097-token prefill 后，cache 覆盖位置 `0..4096`（共 4097 token）；decode 要读全部 ⇒ 需
  `ceil(4097/16) = 257` 页。**token 4096 → page 256（0-based）、slot 0**，即第 257 页。
- 旧 `DECODE_CTX = 4096`（`:192`）⇒ `DECODE_BLOCKS = BlocksFor(4096) = 256`（`:202`）—— **少 1 页**。

### 4.2 影响面：容量定尺**不变小**

分配侧由 `PREFILL_BLOCKS = 257`（主 KV）与 `PREFILL_COMP_ROWS = 1028`（compressed）定尺：
`KV_LAYER_STRIDE = 257 × 32,768 = 8,421,376`（`m15_attn_kv.h:211`）、
`COMP_LAYER_STRIDE = 1,028 × 256 = 263,168`（`:212`）。M184 只改 decode 派生的
`DECODE_BLOCKS`/`DECODE_COMP_ROWS`（`:202-203`），容量**不变、也不变小**。

### 4.3 M184 实际改动（旧 → 新）

| 文件:行 | 旧 | 新 |
|---|---|---|
| `m15_attn_kv.h:192` | `DECODE_CTX = 4096` | `DECODE_CTX = 4097`（= `PREFILL_M`） |
| `m15_attn_kv.h:202` | `DECODE_BLOCKS = 256` | `DECODE_BLOCKS = 257`（`BlocksFor(4097)`） |
| `m15_attn_kv.h:203` | `DECODE_COMP_ROWS = 1024` | `DECODE_COMP_ROWS = 1028`（`CompRowsFor(4097)`） |
| `m15_attn_kv.h:220,222` | `static_assert` 期望 256 / 1024 | 期望 257 / 1028 |
| `m15_layer_loop.asc:4856` | `DECODE_CTX + 1u == PREFILL_M` | `DECODE_CTX == PREFILL_M` |
| `m15_layer_loop.asc:4849-4853` | 打印 prefill 页容量 | 打印 `DECODE_CTX` / `DECODE_BLOCKS` / `DECODE_COMP_ROWS` |
| `m15_attn_kv_host.h:155-164` | decode 打印按字面 `4096` 独立复算（256 页） | 从权威 `DECODE_CTX` 派生（257 页） |
| `m15_layer_loop/README.md:1148-1165` | decode 列 `256 页 / 8,388,608 B`＋M178「改不改待裁定」 | `257 页 / 8,421,376 B`＋「M184 已修」 |
| `docs/15:1414-1422`、`docs/19:1173` | 登记 `DECODE_CTX = 4096` | 标注 `M184 已修` / 改为 4097 |

### 4.4 设备读数（本 mission 实跑；每档 `flock -w 300 /tmp/npu0.lock`、锁内先 `npu-smi`、`timeout` 在锁内）

构建：`m15_layer_loop` rc=0、0 error、21 warning（改前/改后**同数**）；二进制 sha256 见 §0。

`runs=kv`（`logs/m184_before_kv.log` vs `logs/m184_runs_kv.log`）—— **本修复在 KV 几何档上的见证**
（塔裁定「必须收」的那条）：逐条比对**只有 decode 行变化**，其余是计时与 npu-smi 噪声：

| | decode 行 | tally |
|---|---|---|
| 改前 | `decode(1,ctx=4096)：主 KV 256 页 = 8388608 B/层；compressed 1024 行 = 262144 B/层` | `ALL PASS（checks=215, guards=191, fails=0）` |
| 改后 | `decode(1,ctx=4097)：主 KV 257 页 = 8421376 B/层；compressed 1028 行 = 263168 B/层` | `ALL PASS（checks=215, guards=191, fails=0）` |

`runs=prefill`（`logs/m184_before_prefill.log` vs `logs/m184_runs_prefill.log`）—— `.asc` 打印的见证：

| | decode 行 | tally |
|---|---|---|
| 改前 | `…decode 档的 ctx 容量…：257 页/层 × 16 token/页 = 4112 token ≥ 4097；…` | `ALL PASS（checks=149, guards=57, fails=0）` |
| 改后 | `…decode 档的 ctx 容量…：DECODE_CTX=4097 ⇒ 257 页/层 × 16 token/页 = 4112 token；PREFILL_M=4097、DECODE_COMP_ROWS=1028；…` | `ALL PASS（checks=149, guards=57, fails=0）` |

两档前后 tally 均未变（0 FAIL），即本改动**无回归**。

### 4.5 M178「不改、只报」的处置已被本 mission 取代

M178 因 `m15_layer_loop.asc` 被在飞 **M175** 占用而只报不改；M175 已合入。M184 先把 `main`
（已前进到 `70fc00b`，含 M181/M183）合入本分支，再在**合并后的树**上落地 §4.3 的全部改动并重取读数
（`git diff e5c0940 main -- m15_attn_kv.h m15_layer_loop.asc` 为空 ⇒ 无源码冲突）。§4.1 的算术推导保留。

## 5. 负重对照（判据有判别力）

设备日志里旧几何只作为**负向对照**出现，两条都 PASS（`logs/m178_runs_kv.log`）：
- `Kv.neg.oldgeom`（host 算式）：旧式与新式在 28 个采样点上不同；
- `Kv.neg.oldgeom.dev`（device 字节）：用旧几何落页、按新几何读回 ≠ seed。

即：若把权威几何改回旧值（页 16,384 / head 平面 8,192 / token 步长 512），上面两条判据会变红——
这正是「静默错瓦片」方向上有判别力的证据。`m15_attn_kv.h:334-347` 的 `KV_OLD_*` / `KvOldGeomByteOffset()`
是它们的比对面。

## 6. 边界（如实）

- M184 只改 scope 内文件：`m15_attn_kv.h`、`m15_layer_loop.asc`、`m15_attn_kv_host.h`、
  `m15_layer_loop/README.md`、`docs/15-prefill-design.md`、`docs/19-gdn-prefill-scan-selection.md`、
  本证据目录；**未动** `evidence/ple_wire/**`（M171）与其余文件。
- 设备档为**权重档**（真实 checkpoint 切片，`weights_manifest.txt`），非合成档。
- M184 首轮报出的三条 out-of-scope follow-up 已在塔扩 scope 后**全部收掉**：`m15_attn_kv_host.h:155-164`
  改为从 `DECODE_CTX` 派生；`m15_layer_loop/README.md:1148-1165` 与 `docs/15`/`docs/19` 已同步。
- M178 本 mission 之外的三条 finding（历史，延用原措辞）：`m15_loop_layout.h:32` 与
  `evidence/attn_cache/**`（塔另行安排）、`m15_layer_loop.asc:808`（已转 M175）。
