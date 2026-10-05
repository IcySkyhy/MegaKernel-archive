# M136（Wave D）GDN prefill 相位 A —— 设备证据归档

本目录是 M136 的**设备见证入库位**（r1 复审 P2-1 的整改：原始日志/dump/`npu-smi` 原来只在 `/tmp`）。
分支 `feat/m136-wave-d-gdn-prefill-host-wiring`，工作树 `wt-136`。

判据的性质：**KIND_GDN 的相位 A（GDN 扫描段）**，不是"整层"，也不是"prefill 已打通"。范围与限度见文末。

## 1. 归档清单

| 文件 | 内容 |
| --- | --- |
| `logs/run_m1.log` | m=1 设备档：`npu-smi` 快照 + 命令 + 全量输出（`exit=0`，`ALL PASS`） |
| `logs/run_m4097.log` | m=4097 设备档：同上（`exit=0`，`ALL PASS`） |
| `logs/run_wire0.log` | 零回归档（`M15_PREFILL_WIRE=0`，M=1,4097）：同上（`exit=0`，`ALL PASS`，checks=17） |
| `logs/run_before_hang.log` | **改前侧**（把 `if ASCEND_IS_AIV` 临时还原）：m=1 档在 `Pf.gdn` 启动后 `exit=124`（timeout 挂死） |
| `check_ref/clean_m1.txt` / `mut1_m1.txt` | `check_ref.py` m=1 的 clean（PASS）与 `--mutant 1`（FAIL，如预期变红） |
| `check_ref/clean_m4097.txt` / `mut1_m4097.txt` | 同 m=4097 |
| `dumps_m1/` | m=1 的 m23 兼容 dump（q/k/v/g/β/h0/o/ht，clean + mut1，约 15 MB）**入 git** |
| `dumps_m4097/` | m=4097 的 dump：**只入** `m23_*_meta.txt` + `sha256sums.txt`；`*.bin` 约 530 MB **不入 git**（见 `.gitignore`） |
| `reproduce.sh` | 离线复算（check_ref + sha256）；`M136_DEVICE=1` 时先重跑设备档 |

## 2. 设备读数（真权重 checkpoint 切片）

命令（`<bin>=./m15_layer_loop/build/m15_layer_loop`，`<man>=m15_layer_loop/weights_manifest.txt`）：

```
# m=1
M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_M=1 \
  M15_PREFILL_DUMPDIR=m15_layer_loop/evidence/m136_prefill_gdn/dumps_m1  <bin> <man> prefill
# m=4097
M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_M=4097 \
  M15_PREFILL_DUMPDIR=m15_layer_loop/evidence/m136_prefill_gdn/dumps_m4097  <bin> <man> prefill
# 零回归
M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_M=1,4097  <bin> <man> prefill
```

| 档 | `exit` | 判据 | 设备读数 |
| --- | --- | --- | --- |
| m=1 | 0 | ALL PASS（checks=8, fails=0） | o 非有限 0、`max|o|`=2.1381e-02；ht 非有限 0、`max|ht|`=2.4824e-01 |
| m=4097 | 0 | ALL PASS（checks=8, fails=0） | o 非有限 0、`max|o|`=6.7415e-02（零元素 1/25171968）；ht 非有限 0、`max|ht|`=6.3126e-01 |
| wire=0 | 0 | ALL PASS（checks=17, fails=0） | M110 结构性形态（逐行 x→y）逐档全中 |

`M15_SKIP_WCHECK=1` 跳过的是 `H_WeightSourceCheck` 等**来源校验**（不计入判定项），日志里的
"权重装载完成（真实 checkpoint 切片）" 仍是真权重口径。`M15_LAYERS=1` 是"只跑/只装 1 层"（人类口径：
单层测试不必装全部权重）。每次进锁的 `npu-smi` 快照在对应 `logs/*.log` 的 "npu-smi (lock entry)" 段。

## 3. 数值判据（复用 m23 的 numpy float64 逐句参考）

`m23_gdn_prefill/check_ref.py` 从 **dump 的干净输入**复算 `o`/`ht`，逐元素 `|dev-ref| <= rtol*max(1,|ref|)`，rtol=2e-3：

```
Pf.gdn       m=1    PASS  o 超界 0/6144       ht 超界 0/786432
Pf.gdn       m=4097 PASS  o 超界 0/25171968   ht 超界 0/786432
Pf.gdn_mut1  m=1    FAIL  o 超界 4190/6144     （--mutant 1，如预期变红 ✓）
Pf.gdn_mut1  m=4097 FAIL  o 超界 130298/25171968（--mutant 1，如预期变红 ✓）
```

**判据有牙**：`Pf.gdn_mut1` 只把**送设备的** h0 翻倍，host 侧 dump 的参考输入仍是干净副本 ⇒ 参考吃干净、
设备吃搅动 ⇒ 变红。mut1 的 q/k/v/g/β/h0 dump 与 clean 逐字节相同（`sha256sums.txt` 可核），只有设备 `o`/`ht` 不同。

## 4. 复算方法

```
# 离线（用入库的 dumps_m1 + dumps_m4097 的 meta/sha256）
bash m15_layer_loop/evidence/m136_prefill_gdn/reproduce.sh
# 重跑设备档后再复算（需设备；纪律见脚本头）
M136_DEVICE=1 bash m15_layer_loop/evidence/m136_prefill_gdn/reproduce.sh
```

`dumps_m4097/*.bin` 不入 git：按上面的 m=4097 命令重建后，`sha256sum -c dumps_m4097/sha256sums.txt` 应与入库的清单一致。

## 5. 改前 / 改后两侧

- **改后**：`m15_layer_kernel.h` 的 `M15L_PrefillBody` 把三条 `M15L_PhaseBoundaryAiv`（全体 AIV 的
  mode-0 屏障）收进 `if ASCEND_IS_AIV`（与 decode 路径的用法一致）⇒ 上表三档通过。
- **改前**：同一处无条件调那三条屏障 ⇒ AIC 也走 mode-0。`logs/run_before_hang.log` 是**原始读数**：
  程序走到 `Pf.gdn：GDN 段入口启动 …` 后在 `aclrtSynchronizeStream` 上不再前进，`timeout 60` 到点、
  `exit=124`。该档用临时还原的二进制跑（跑完已把源码恢复并重建；恢复后 `git diff` 无差异）。

## 6. 显式未做 / 限度（不得读成"prefill 已打通"）

- 只到 **KIND_GDN 的相位 A（GDN 扫描段）**。四相位的 H1/H2/B 未开：prefill 入口实参表**缺 hc 层界/权重
  字段**（`M15L_LAYER_PREFILL_ARGS_*` = 两相位 30 + M110 13 + M132 段体 13），`M15L_HcPrefillBoundary`
  会读到全 nil ⇒ 用 `pfStageMask=PF_STAGE_A` 只开相位 A；B5 的 5 个指针已分配并传入但未被消费。
  已立 finding：`.tower/comms/findings/20261004-agent-waved1-bug-m136-prefill-hc-gdn-h1-h2-b.md`。
- q/k/v/g/β 由**宿主合成**（L2 归一的确定性输入；段体 B1 的契约是"已过 prolog 的 q/k/v/g/β"）⇒ 判据
  覆盖"扫描段"，**不含** in_proj/out_proj/conv/l2norm。
- 段间数据流未连通（GDN 出口 `wsO` 未接到 hc(mlp) 的 bo；hc(attn) 的 BLK 未接到相位 A 输入）。
- `H_LaunchChainLayer` 的 m 真传参未被任何设备档覆盖（wire=0 只跑 prefill，不驱动 48 层链；默认
  `M15_CHAIN_M=1` 下与旧 `1u` 等价，`chainM>1` 未验）。
- attention 的"响亮失败"仅代码级验证（`H_CmpOk(C,false)` 会 `fails++`），本轮按任务不接 attention，未加设备档。
- **待同步项（README 不在本 mission scope）**：`m15_layer_loop/README.md` 里"验证 Pf"的判据说明仍是
  M110 的"整面毒值"口径，需改成"段体真产出 + m23 参考对拍 + `Pf.gdn_mut1` 负向对照"。需要塔再加宽 scope
  或另派 mission，本目录不代改。
