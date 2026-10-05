# M177 证据：PLE 装载门控加固（`H_PleArgsOf`）

本目录是 mission M177（PL E 段「已分配 / 已装载」门控分离）的**可复算证据**。
来源：M158 只读 survey（`.tower/comms/inbox/20261004-agent-plesurvey-tower-survey-summary-m158-ple-wiring-readiness-tip-ac46c32-zero-de.md` §1.4）标记的潜伏脆弱点。

| 项 | 值 |
|---|---|
| 代码 commit（tip） | `ff916fb`（分支 `feat/m177-ple-load-gating-hardening`；base `main` = `c318895`）|
| 修复版二进制 sha256 | `5f656445f06abdd795dd0c737b5ff82efe81bc0e4446d1051b892c7c54e5b407` |
| 负向对照（回退门控）二进制 sha256 | `fa73875325fff4784b9280caa93db3d60f881acbd67678c55983489efd86d0f5`（由 `reproduce.sh --with-negctl` 的临时 patch 重建）|
| 变更文件 | `m15_layer_loop/m15_hc_host.h`（+21/−4，唯一源码改动）|

负向档的二进制 sha 是「把门控那一行临时改回 `C.pleWDev == nullptr`」这一最小 patch 的产物；任何
让门控退化成「分配即放行」的等价改法，设备行为相同（`aicore error exception` + 非零退出，见 §2 ③）。

## 0. 变更一句话

`m15_hc_host.h::H_PleArgsOf` 的门控从「`pleWDev != nullptr`（分配即放行）」改为要求**装载见证
`C.pleTableDev != nullptr`**；缺见证时 17 个 PLE 实参整组置空（kernel 内 `A.pleW == nullptr`
⇒ 挂载点退化为空操作），并在被判层打一条 `[m15][WARN]` —— 是**响亮跳过**，不是静默送垃圾。
未新增字段、未碰任何 `.asc`、未改同步、未加探针。

## 1. 可达性论证（文件:行）

| 事实 | 落点 |
|---|---|
| `H_Alloc` **无条件**分配 `pleWDev`/`pleScrDev` | `m15_layer_loop/m15_layer_loop.asc:2056-2057` |
| `pleTableDev` 只在 `H_PleWindowSetup` 末尾置位 | `m15_layer_loop/m15_layer_loop.asc:2830`（`C.pleTableDev = dev;`）|
| `H_PleWindowSetup` 只被 `H_PleWireRun` 调用（在**权重 H2D 之后**）| `m15_layer_loop/m15_layer_loop.asc:2937`（权重 H2D）→ `:2970`（窗口）|
| `H_PleWireRun` 是唯一装载路径，只挂在 `runs=plewire` | `m15_layer_loop/m15_layer_loop.asc:5077-5078`；「故意不进 `runs=all`」`:5074-5076` |
| `pleWire` 仅由 env `M15_PLE_WIRE` 解析（默认 0）| `m15_layer_loop/m15_layer_loop.asc:261` |
| 层 1 打断点 `pleBreak = (L==1) ? C.O.chainPle : 0`（`chainPle` 默认 1）| `m15_layer_loop/m15_chain_host.h:365` |
| `runs=chain` / `runs=x` / `runs=all` 都会跑到链上 | `m15_layer_loop/m15_layer_loop.asc:5054` |

⇒ **可达组合**：`M15_PLE_WIRE=1` 配 `runs=chain`（或 `runs=all`）时，`H_PleArgsOf` 旧门控会放行
「`pleWDev` 已分配但权重未 H2D、`pleTableDev == nullptr`」的实参组 ⇒ 设备侧拿到**未初始化权重 +
空表基址**。故本缺陷**可达**，不是「不可达」。

## 2. 三档读数（由 `reproduce.sh` 生成，均为锁内逐字读数）

每份日志自含：`=== lock acquired ===`、锁内 `npu-smi info` 快照、逐字命令、进程输出、`exit=`。

### ① `T1_bad_config_fixed.log` —— 坏配置（修复后）
命令（逐字）：`timeout 240 env M15_LAYERS=2 M15_PLE_WIRE=1 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt chain`
- 实读：`[m15][WARN] H_PleArgsOf：M15_PLE_WIRE=1 但 PLE 装载路径未跑过（pleWDev=0x120077600000 pleTableDev=(nil)）⇒ 层 1 的 PLE 接线被门控关闭（退化为空操作）；M15_PLE_WIRE 只应在 runs=plewire 档打开`
- 实读：`===== ALL PASS（checks=48, guards=6, fails=0）=====`、`exit=0`
- 行为 = **响亮跳过**（WARN + 空组），设备不再收到垃圾实参。

### ② `T2_plewire_fixed.log` —— 合法档 `runs=plewire`
命令（逐字）：`timeout 280 env M15_PLE_WIRE=1 M15_PLE_STAGE=31 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt plewire`
- 归档 `m15_layer_loop/evidence/ple_wire/B_full_stage31.log` 的 `Pw.*` 行（**11 行** = 10 条判定项 + 1 条 `wired.delta` 读数行）**逐字复现**（脚本逐行 `grep -Fqx` 比对）；
- 关键行：`Pw.emb.T1 PASS bad=0 miss=0/16`、`Pw.ids.T1 PASS bad=0/16`、`Pw.wired.delta 读数：5636/10240`；
- 实读：`===== ALL PASS（checks=144, fails=0）=====`、`exit=0`。

### ③ `T3_negctl_oldgate.log` —— 负向对照（门控回退到「分配即放行」）
命令（逐字）：与 ① **同一条**。
- 实读：`[m15][FAIL] sync after ch_layer failed (err 507015)`；
- `aicore error exception` 28 行（core 0..27，`errorStr: timeout or trap error`）；
- 实读：`===== FAILURES PRESENT（checks=7, fails=1）=====`、`exit=1`。
- ⇒ **旧门控确实会咬**：坏配置把垃圾实参送进设备并打出 aicore 507015 异常。对照后源码无条件还原、重建，二进制 sha 回到 `5f656445…`。

## 3. 已知偏差（如实记录；不为对齐 mission 文本而少列）

mission 文本的验收写「与归档 `B_full_stage31.log` 逐字相同（`Pw.emb.T1 …`、**10/10**）」。
本 tip 上 `runs=plewire` 实际是 **12 条判据 / 12 PASS**：

- 多出的两条是 `Pw.kv.T3`、`Pw.kv.nonvac`，由 **M124**（commit `6a8fee3`，已合入 `main`）新增到
  `m15_layer_loop/m15_layer_loop.asc`；
- 归档 `B_full_stage31.log` 出自 **M111** 批次（其记录二进制 `m15_layer_loop/evidence/ple_wire/binary_sha256.txt` = `aa3652f1…`），**早于 M124**，故只有 10 条；
- 归档那 10 条本身在本次**逐字复现**（见 §2 ②）。12 vs 10 是「归档批次旧了」，不是本次修复引入的差异。

（同一过时计数也出现在 `m15_layer_loop/evidence/ple_wire/WITNESS.md:45/167/216`。该文件属 M171
在飞 scope，**本 mission 不改**，已作为 finding 报塔另派。）

## 4. Follow-up（超出本 mission scope，未做）

1. 更稳健的装载见证是在 `H_PleWireRun`（`m15_layer_loop/m15_layer_loop.asc`，属在飞 M175）里加一个
   **只在装载路径置位**的 `bool pleLoaded`，届时 `H_PleArgsOf` 改读它。现用 `pleTableDev` 的语义耦合是
   「装载必然建窗口」；若将来出现「只装权重不建窗口」的新路径需一并改。
2. `runs=all`（默认 `M15_PLE_WIRE=0`）在新门控前**早退**（`m15_layer_loop/m15_hc_host.h:187-189`），
   行为不变 —— 这是代码路径论证，本 mission 未单独跑设备档复验零回归。

## 5. 怎么复算

```bash
# 默认：① + ②（约 2 分钟，含一次构建）
bash m15_layer_loop/evidence/m177_ple_gating/reproduce.sh

# 追加负向档 ③（约 4 分钟：多两次构建 + 一档设备）
bash m15_layer_loop/evidence/m177_ple_gating/reproduce.sh --with-negctl
```

预期：全部检查通过则 `exit=0`；任一条不符（缺 WARN、未 ALL PASS、归档判据行未逐字复现、
或负向档竟然 `exit=0`）⇒ 脚本以非零退出。

设备纪律：脚本每一步都 `flock -w 300 /tmp/npu0.lock`，锁内先落 `npu-smi` 快照，`timeout` 在锁内，
一次进锁一条命令。负向档对源码做的临时 patch 由 `trap … EXIT` 保证还原，并在最后重建校验
二进制 sha 与修复版一致（若进程被 `SIGKILL` 打断，请手动 `git checkout -- m15_layer_loop/m15_hc_host.h`）。
