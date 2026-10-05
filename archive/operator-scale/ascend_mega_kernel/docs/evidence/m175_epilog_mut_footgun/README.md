# M175 —— `M15_PREFILL_EPILOG_MUT` 越界「假绿」脚枪：修前 / 修后设备读数

> **性质与边界（先读）**
> - 本目录是 **M175** 的证据（该 mission 的 scope 含 `docs/**`，故落这里）。
> - 被测对象 = `m15_layer_loop/m15_layer_loop.asc` 对 `M15_PREFILL_EPILOG_MUT` 的解析 + prefill epilog 挂载档。
> - 逐档配置（除 `M15_PREFILL_EPILOG_MUT` 外相同；真 checkpoint 权重，`M15` 二进制就地构建）：
>   ```
>   M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=7 \
>   M15_PREFILL_PROLOG=1 M15_PREFILL_EPILOG=1 M15_PREFILL_M=1
>   ```
> - 设备纪律（塔口径）：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`、`timeout` 在锁内、
>   一次进锁一条短命令；**未出现「等锁未取得读数」**。

## 0. 脚枪是什么

`M15_PREFILL_EPILOG_MUT` 只有 **0..4** 有定义（0=干净 / 1=转位 headstride / 2,3=S5 / 4=S6 K−1）。
修前解析写成 `H_EnvU32("M15_PREFILL_EPILOG_MUT", 0u) & 7u` ⇒ 取 **5..7** 时被静默折到「clean」
（`H_PfEpilogRun` 里 `transMut/s5Mut/s6Mut` 全 0），却照样报 `ALL PASS`。想跑负向对照的人会以为
该变异真的跑了 —— 这是「假绿」。

## 1. 修前（base 二进制）

- 源码 tip 的 `m15_layer_loop.asc:291`（修前）：`O.pfEpilogMut = H_EnvU32("M15_PREFILL_EPILOG_MUT", 0u) & 7u;`
- 二进制 sha256 = `d2cc02d9b82e86f29c87c3daa57c4b7c5fd676ad443cd12f04f3b068bd7cdb1a`

| `MUT` | 进程内判定（日志末行） | 进程退出码 |
|---|---|---|
| 0 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 1 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 2 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 3 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 4 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| **5** | `ALL PASS（checks=89, guards=9, fails=0）` | **0** |
| **6** | `ALL PASS（checks=89, guards=9, fails=0）` | **0** |
| **7** | `ALL PASS（checks=89, guards=9, fails=0）` | **0** |

⇒ `MUT=5/6/7` 与 `MUT=0`（干净）读数逐字相同、且**日志里没有一行异常** —— 脚枪现场（`MUT` 0..7 全档
读数见 `logs/pre_mut*.log`）。

## 2. 修做了什么（`文件:符号`，行号随 tip 漂移、以符号为准）

- `m15_layer_loop/m15_layer_loop.asc` 的 `Opts::pfEpilogMutOor`（新增字段，注释在其声明处）。
- `H_ParseOpts`：把 `& 7u` 换成**越界守卫** —— `epiMutRaw > 4` ⇒ 打印
  `[m15][WARN] M15_PREFILL_EPILOG_MUT=%u 越界（…）→ 取 0；挂 epilog 的档将计为失败`、`pfEpilogMut` 取 0、
  `pfEpilogMutOor` 记原始越界值（0 = 未越界）。形态对齐既有的 `M15_PREFILL_PHASES` 越界守卫。
- `H_PfGdnWired`（epilog 挂载点）：`epilogOn && pfEpilogMutOor != 0` ⇒ 打印 `[m15][FAIL] … 本档按失败处理`、
  `C.fails++`、`return false` ⇒ 该臂失败，最终判定**不再是** `ALL PASS`（进程退出码 1）。

## 3. 修后（fix 二进制）

- 二进制 sha256 = `a03baa5e5d7c47da241188a404751a9db23ca9fa5dcb8741f48b779c2ee28b9c`
- 构建：`cmake --build m15_layer_loop/build -j4 --target m15_layer_loop` ⇒ **rc=0**；编译告警数与 base 逐条相同
  （**各 21 条，`grep "warning:"` 逐行 diff 无差异**）。

| `MUT` | 进程内判定（日志末行） | 进程退出码 |
|---|---|---|
| 0 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 1 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 2 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 3 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| 4 | `ALL PASS（checks=89, guards=9, fails=0）` | 0 |
| **5** | `FAILURES PRESENT（checks=2, guards=9, fails=5）` | **1** |
| **6** | `FAILURES PRESENT（checks=2, guards=9, fails=5）` | **1** |
| **7** | `FAILURES PRESENT（checks=2, guards=9, fails=5）` | **1** |

`MUT=5` 档的异常行（逐字，`logs/fix_mut5.log`）：

```
[m15][WARN] M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义：0=干净 / 1=转位 / 2,3=S5 / 4=S6）→ 取 0；挂 epilog 的档将计为失败
[m15][FAIL] Pf.gdn：M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义）⇒ 本档按失败处理
[m15][FAIL] Pf.gdn_mut1：M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义）⇒ 本档按失败处理
[m15][FAIL] Pf.prolog_mut1：M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义）⇒ 本档按失败处理
[m15][FAIL] Pf.prolog_mut2：M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义）⇒ 本档按失败处理
[m15][FAIL] Pf.prolog_mut3：M15_PREFILL_EPILOG_MUT=5 越界（只有 0..4 有定义）⇒ 本档按失败处理
```

（WARN 1 行 + FAIL 5 行：本档有 5 个挂 epilog 的臂各自失败；`MUT=6/7` 同形态。`MUT=5` ⇒ `checks` 从 89
降到 2，因为该臂在 epilog 之前就 `return false`、后续判据未跑。）

## 4. `MUT=0..4` 行为一字不变的证据

对每个 `MUT∈{0,1,2,3,4}`，把修前 / 修后两份日志去掉「权重装载耗时」行后逐行 `diff`：

```
MUT=0: 判据读数逐行相同（差异 0 行）
MUT=1: 判据读数逐行相同（差异 0 行）
MUT=2: 判据读数逐行相同（差异 0 行）
MUT=3: 判据读数逐行相同（差异 0 行）
MUT=4: 判据读数逐行相同（差异 0 行）
```

逐档 `diff` 原样归档在 `logs/normdiff_mut{0..4}.txt`（**均为空**）。唯一可出现的差异是三条
`权重装载完成 … 耗时 N ms` 的**墙钟**数字（归一化时已剔除）。关键判据行（如
`Pf.gdn：epilog 出口 hcAttnOut[0,1) ⇒ 0xCD 残渣 0/2560、非零 2560、非有限 0（输入 wsO 非有限 0 ⇒ 全有限）；
尾行 [1,4097) 0xCD 10485760/10485760`）在两版之间逐字节相同。

## 5. 复现（一条命令；设备；单档 ~14 s）

`reproduce.sh` **同时取得修前与修后两个二进制**并跑完整矩阵（MUT 0..7 × 两个二进制 = 16 档）：

- **修前**：`git archive <base>` 解出 base 源码（缺省 rev = 本 mission 的 base
  `449c5ea9f5ed5b887c241adba535d043e7e820a9`，可用 `M175_BASE_SHA` 覆盖），在该树里
  `cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release` +
  `cmake --build … --target m15_layer_loop`；
  期望 sha256 = `d2cc02d9b82e86f29c87c3daa57c4b7c5fd676ad443cd12f04f3b068bd7cdb1a`。
- **修后**：在本分支 tip 上同法构建；期望 sha256 =
  `a03baa5e5d7c47da241188a404751a9db23ca9fa5dcb8741f48b779c2ee28b9c`。

```bash
cd docs/evidence/m175_epilog_mut_footgun
bash reproduce.sh                 # 构建 + 跑 16 档 + 逐字比对；任何不符 / 未取到读数即非零退出
bash reproduce.sh --no-build      # 复用已有二进制（M175_BASE_BIN / M175_FIX_BIN）
```

**内建期望（不符即 rc=1）**：
- 修前 MUT 0..7 → `ALL PASS`、rc=0（脚枪现场：越界值被 `& 7u` 静默当 clean）；
- 修后 MUT 0..4 → `ALL PASS（checks=89, guards=9, fails=0）`、rc=0；MUT 5..7 →
  `FAILURES PRESENT（checks=2, guards=9, fails=5）`、rc=1，且进程内先有一行越界 `[m15][WARN]`；
- 修前/修后 MUT 0..4 剔墙钟「耗时」行后 diff 必须为空。

**归档**：逐档日志 `logs/pre_mut*.log` / `logs/fix_mut*.log`、逐档归一化 diff `logs/normdiff_mut{0..4}.txt`
（期望为空）、`logs/summary.txt` 由脚本落在本目录（已把临时路径改写成 `WORK` / `REPO` 前缀）；
临时构建树与原始日志在 `M175_WORK`（缺省 `mktemp -d`，不入仓）。设备纪律同 m164：每档各自进一次
`flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（写进该档日志）、`timeout` 在锁内；等锁 / 读数未取得
即非零退出（不写成"未复现"）。

## 6. 纪律自查

- **只改 scope 内文件**：本 mission 的改动集 = `m15_layer_loop/m15_layer_loop.asc`、
  `m15_layer_loop/evidence/m164_epilog_mount/CONTRACT.md`、`m23_gdn_prefill/README.md`、
  `docs/17-verification-standard.md`、本目录。
- **未改**任何既有已提交读数 / 日志 / dump：MUT 矩阵的逐档日志由 `reproduce.sh` 生成在 `logs/`（本目录，入仓）；
  临时构建树 / 原始日志在 `M175_WORK`（缺省 `mktemp -d`，不入仓）。
- **可离线/设备复现**：`bash reproduce.sh` 一次构建修前+修后两个二进制、跑 16 档、内建期望与逐字比对，
  失败即非零退出（见 §5）。
- **禁用词**：本 mission 新增/改动行对六词表的逐字扫描结果见 review-request；清单与命令已落 `docs/17` §9.15。
- 设备档纪律：每档各自进锁、锁内 `npu-smi`、`timeout` 在锁内；无「等锁未取得读数」。
