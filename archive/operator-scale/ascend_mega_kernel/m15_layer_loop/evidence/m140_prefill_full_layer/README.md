# M140 —— GDN prefill 的 hc 层界字段 + 四相位（H1/A/H2/B）设备证据

> **判据的性质**：本目录证明的是 **KIND_GDN 的 prefill 入口在四相位掩码下能跑、四个相位各自的产出面在
> 设备上留下真数值、hc 两个边界与相位 B（MoE）的 router 段有独立 numpy 参考对拍**。
> 它**不**等于「prefill 已打通」：见 §8 的已知缺口（prolog / epilog / attention B2+B3 /
> H2→相位 B 的跨核可见性）。
>
> 分支 `feat/m140-whole-layer-gdn-prefill-hc-fields-a`，工作树 `wt-140`。
> 前身：M136（相位 A 已通，四相位因**入口实参表缺 hc 字段**而无法开，finding
> `20261004-agent-waved1-bug-m136-prefill-hc-gdn-h1-h2-b.md`）。

## 1. 一句话

`M15L_LAYER_PREFILL_ARGS_*` 补上 15 个 hc 层界/权重实参（清单见 `fields.md`），host 侧从
`HC_LAYER_W_STRIDE` 的 layer-0 槽取**真权重**、按 `M_PREFILL` 分配 4 块 hc 激活平面与 E=512 的
MoE 权重区（`moe512_*` 真 shape 段），相位掩码由 `M15_PREFILL_PHASES` 逐位打开。
设备上 m=1 的十一档（p1/p3/p7/p15 + 隔离档 p8/p12 + pad 控制档 ×4 + repeat 基线 ×4）与 m=4097 的
p15 的判定项读数见 §5；
hc 边界（H1/H2）的 `hcp`（H'）与 `blk` 对 `m20_hyperconn/check_ref.py::reference` 的 float64 参考：
m=1 各档**逐位一致**，m=4097 的 `h1.hcp`/`h2.hcp` 逐位 1.0000、`h1.blk` 逐位 0.9949、
`h2.blk` 逐位 0.9797（**一条判定项 FAIL**，见 §6 第 2 条）。
**相位 B（MoE）的数值对拍**：**单独跑（p8）5/5 PASS**；**跟在 hc 段之后**时第一个 arm 错
（`p12` 0/4、`p15` 0/5；m=4097 的最后一块 PASS）—— 见 §8 第 4 条：这是**未闭合**的 in-scope 缺口，
本 mission 未把它当作通过。

## 2. 归档清单

| 文件 | 内容 |
| --- | --- |
| `fields.md` | **任务 1 的交付物**：要补的字段逐条（字段名 → 语义 → 从哪取 → 归哪个相位） |
| `check_full_layer.py` | hc 四相位（H1/H2）的独立 numpy 对拍（**不含第二份 hc 数学**：参考 = m20 的 `reference`） |
| `reproduce.sh` | 一键复跑（`M140_DEVICE=1` 时先跑设备档；每档各自 `flock` + 锁内 `npu-smi`） |
| `logs/run_m1_p{1,3,7,15}.log` | m=1 的四档阶梯设备档（含锁内 `npu-smi`、全量输出、`exit=`） |
| `logs/run_m1_p{8,12}.log` | m=1 的**隔离档**（p8 只开相位 B；p12 = H2+B）—— 缺口 #4 的归因对照 |
| `logs/run_m1_p15_nopad.log` / `run_m1_p12_nopad.log` / `run_m1_p15_pad3c.log` / `run_m1_p12_pad3c.log` | **MT-pad 的单变量 A/B 控制侧**（同一二进制，只差 `M15_PREFILL_NO_PAD` 或 `M15_PREFILL_PAD_BYTE`；r3 判定其结论不足以支撑 pad-耦合，见 §8 第 4 条 (a)） |
| `logs/run_m1_p12_rep{2,3}.log` / `run_m1_p15_rep{2,3}.log` | **repeat-run 方差基线**（同配置重复跑；r3 据此把 p12 组的抖动 2.58e8–4.31e8 量出来，见 §8 第 7 条） |
| `logs/run_m4097_p15.log` | m=4097 的验收档（同上） |
| `logs/baseline_m136_m1.log` | 本 mission 开工前的 M136 基线（相位 A 档，用于确认设备/工具链未变） |
| `logs/h1only_m1_stale_hostptr.log` | **改前侧**：host 误把 host 指针当设备指针传 hc 权重 ⇒ 设备 `aivec error exception`（保留作定位记录） |
| `logs/h2b_sync_experiment_m1.log` + `logs/h2b_sync_experiment_m26_Pf.gdn/` | H2→B 可见性握手的**实验档**（读数变了但没修好；代码已回退，见 §8 第 4 条 (b)） |
| `binary_sha256.txt` | 产生本目录读数的二进制 sha256 |
| `dumps_m1_p{1,3,7,15}/` | m=1 各档的 hc/m23/m26 兼容 dump（小面入 git）；`sha256sums.txt` |
| `dumps_m1_p{8,12}/` | m=1 隔离档的 dump（相位 B 的 m26 兼容 dump 在 `m26_<tag>/`） |
| `dumps_m4097_p15/` | m=4097 档：只入 `sha256sums.txt` + `m26_*/m26_meta.txt`；`*.bin` 不入 git（见 `.gitignore`） |

## 3. 任务 1 的结论（字段清单）

见 `fields.md`。一句话：prefill 入口缺的**不是新东西** —— `LayerArgs` 里 M58 早已有 hc 的
15 个字段（`hcH/hcBo/hcIj/hcAttnOut` + 8 个权重 + `hcIjStride/hcAttnMode/hcMlpMode`），缺的是
**prefill 入口的实参表没有带它们**（入口内 `M15L_LAYER_ARGS_FILL` 把它们全置 nil）。
另需（不在 `LayerArgs`）：相位 B 的 E=512 MoE 权重槽 —— M136 的 prefill 档传的是 decode 的 E=4 槽
（13.09 MB/层），而 B4 段按 `E=512/topk=10` 编译、槽是 `MOE_W_PREFILL_STRIDE` = 1.34 GB/层。

## 4. 改了什么（逐文件）

| 文件 | 改动 |
| --- | --- |
| `m15_layer_kernel.h` | `M15L_LAYER_PREFILL_ARGS_DECL/FILL` 追加 **15 个 hc 实参**（顺序与 `LayerArgs` 的 hc 字段一致）；`M15L_PrefillPhaseB` 的层输入改为 `pfHcBlk1`（H2 的 BLK 平铺面，**仅在 H2 开时**）并加 **MT=64 行的块循环**（m>MT 时）；用法注释同步 |
| `m15_layer_loop.asc` | 新增：4 块 hc 激活平面 + E=512 MoE 权重区（`H_PfSegAlloc`；`pfHcBlk1` 按 `ceil(m/MT)*MT` 行分配）、`H_LoadMoeSlot512`（按 manifest 的 `moe512_*`/`moe_shared_*` 逐段 pread，**无换算**）、`MoeWPrefillPtrs`、hc 合成输入 `H_PfHcFillHost/H_PfHcH2D`、四相位的产出/权重 dump（`H_PfHcDump`）、相位 B 的 m26 兼容 dump（`H_PfMoeDump26`，按 tag 分子目录）、`H_PfFillMoeIn`（隔离档的确定性输入面）；`H_PfGdnWired` 改为按 `M15_PREFILL_PHASES` 驱动并加四相位判据；`H_PfGdnBuildRouterWPad` 改为从 E=512 的 router 平面取源（M136 从 decode E=4 槽读 512 行是越界读，B 未开时没暴露） |
| `m15_layer_resources.h` / `m15_chain_host.h` / `m15_hc_prefill.h` / `m15_hc_host.h` | **本 mission 未改**（登记用） |
| `evidence/m140_prefill_full_layer/**` | 本节的全部证据 |

**r1 复审整改（4 条 P2，逐条落点）**：

| 复审条目 | 落点 |
| --- | --- |
| P2-1「反面判据不存在」 | `.asc` 的 hc 回读改为**无条件**；`if (phH2) … else` 的 H2-关判据读**设备回读值**；p1/p3/p8 的日志显示 H2 面 100% `0xCD`（见 §5） |
| P2-2「洁净检出跑不动」 | `check_full_layer.py` 的权重改为缺则**从 manifest+checkpoint 取**；入库一份 router pad + m1_p15 的 h0/ht；`reproduce.sh` 加守卫（缺输入打印 SKIP 而不抛 traceback）+ 明确的"洁净检动能跑到哪一步"表（见 §9） |
| P2-3「`moe_y` 是上一档的」 | `H_PfHcDump` 移到 `hcAny` 与 `phB` 两块之后；每档起点 `C.pfMoeYH.clear()`；`H_PfMoeDump26` 改为读**实际**输入面（`C.pfMoeInDev`）；`reproduce.sh` 加"纯函数性见证"（clean vs mut1） |
| P2-4「mode 偏差没写进限度」 | 见 §10 第 3 条 |
| 相位 B 的归因 | 见 §8 第 4 条（`m1_p8`/`m1_p12` 隔离档 + MT 零填充实验 + 纯函数性见证） |

**未改的**（`git diff` 可核）：`m23_gdn_prefill/**`、`m26_moe_prefill/**`、`m20_hyperconn/**`
（只被**调用**做参考）、`m25_attn_fa_core/**`、`m27_hc_prefill/**`、`docs/**`、`tools/**`。

## 5. 设备读数（真权重 checkpoint 切片）

命令（`<bin>` = `./m15_layer_loop/build/m15_layer_loop`；`<man>` = `m15_layer_loop/weights_manifest.txt`）：

```bash
# 每档各自进锁（flock -w 300 /tmp/npu0.lock），进锁先 npu-smi，timeout 在锁内
#   掩码位 = H1:1 / A:2 / H2:4 / B:8；p8 与 p12 是**隔离档**（见 §8 第 4 条）
for spec in "1 m1_p1" "3 m1_p3" "7 m1_p7" "15 m1_p15" "8 m1_p8" "12 m1_p12"; do set -- $spec
  M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
    M15_PREFILL_PHASES=$1 M15_PREFILL_M=1 M15_PREFILL_DUMPDIR=<dir> <bin> <man> prefill; done
# 验收档
M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
  M15_PREFILL_PHASES=15 M15_PREFILL_M=4097 M15_PREFILL_DUMPDIR=<dir> <bin> <man> prefill
```

| 档 | 相位掩码 | `exit` | 判据 | 设备读数（每相位产出非毒值/非全零） |
| --- | --- | --- | --- | --- |
| `run_m1_p1` | 0x1（只 H1） | 0 | ALL PASS（checks=23, fails=0） | H1：H' 毒值 **0**/10240、零 0；BLK 毒值 **0**/2560、零 0；**H2：H' 毒值 10240/10240、BLK 毒值 2560/2560**（设备回读，未开相位保持 `0xCD`） |
| `run_m1_p3` | 0x3（H1+A） | 0 | ALL PASS（checks=32, fails=0） | 相位 A：o 非有限 0、`max|o|`=2.1381e-02、零 0/6144；ht 非有限 0、`max|ht|`=2.4824e-01；H1 同上；H2 两块面仍 100% 毒值 |
| `run_m1_p7` | 0x7（H1+A+H2） | 0 | ALL PASS（checks=38, fails=0） | H2：H' 毒值 **0**/10240、零 0；BLK 毒值 **0**/2560、零 0（A 与 H1 读数同 p3） |
| `run_m1_p15` | 0xF（全四相位） | 0 | ALL PASS（checks=44, fails=0） | 相位 B：MoE `y` 毒值 **0**/2560、零 0 |
| `run_m1_p8` | 0x8（**只 B**） | 0 | ALL PASS（checks=17, fails=0） | 隔离档：hc 两块面全部 100% 毒值（H1/H2 都没开）；相位 B 的输入取 `A.xLayer`（host 填的确定性平面） |
| `run_m1_p12` | 0xC（**H2+B**） | **1** | **FAILURES PRESENT（checks=23, fails=3）** | 隔离档：H1 没开 ⇒ H2 的输入是毒值，它仍写了两块面（BLK 毒值 0/2560）但**数值判定项红**（`fails=3`，**如实登记**）；相位 B 照跑。**exit 以日志为准**（`logs/run_m1_p12.log` 末尾 `exit=1`） |
| `run_m4097_p15` | 0xF | 0 | ALL PASS（checks=44, fails=0） | o 非有限 0、`max|o|`=6.7415e-02；ht `max|ht|`=6.3126e-01；H1/H2 的 H' 与 BLK 毒值 0；MoE `y` 毒值 **0**/10488320、零 250/10488320 |
| `run_m1_p15_nopad` / `run_m1_p12_nopad` / `run_m1_p15_pad3c` / `run_m1_p12_pad3c` / `run_m1_p{12,15}_rep{2,3}` | 15 或 12 | **与对应主档相同**（p15 系 = 0；p12 系 = 1） | pad 控制档 + **repeat-run 基线**（r3）：判定模式与对应主档完全一样（p15 系 0/5、5/5、1/5；p12 系 0/4、4/4、4/4）；logits 的 run-to-run 抖动见 §8 第 7 条 |

**非毒值判据的口径**：hc 的平面在 `H_PfSegAlloc` 里被 `0xCD` 污染；判据要求"毒值字数 × 100 < 总字数"
且"零字数 < 总字数"。**反面判据（未开的相位保持毒值）**：`if (phH2) … else …`
（`m15_layer_loop.asc`）在 H2 关的档上要求 `arena1`/`blk1` 的**设备回读值** ≥99% 是毒值/未写 ——
本 mission 复审 r1 P2-1 的问题（旧版 p1/p3 从未回读 H2 面、落盘的是 host 的零初始化缓冲）已修：
回读现在**无条件**做，`run_m1_p1`/`run_m1_p3`/`run_m1_p8` 的 H2 面读数是设备上的 **100% `0xCD`**
（不是零）。**H1 关的档本 mission 没构造**（阶梯从 H1 起）⇒ "H1 未开时 arena0 保持毒值"这一条
**没有设备读数**，已在 §10 登记。

## 6. 数值对拍（有现成参考就用，不自己造第二套数学）

1. **相位 A（GDN 扫描段）** = `m23_gdn_prefill/check_ref.py`（numpy float64 逐句参考，rtol 2e-3）：
   - m=1 clean：`o` 超界 0/6144、ht 超界 0/786432 ⇒ PASS
   - m=4097 clean：`o` 超界 0/25171968、ht 超界 0/786432 ⇒ PASS
   - `--mutant 1`（只把送设备的 h0 翻倍）：m=1 `o` 超界 4190/6144、m=4097 `o` 超界 130298/25171968 ⇒ 如预期变红
2. **hc 边界 H1/H2** = `check_full_layer.py`（参考 = `m20_hyperconn/check_ref.py::reference`，门限 = 同文件的 `judge()`）：
   | 档 | `h1.hcp` | `h1.blk` | `h2.hcp` | `h2.blk` |
   | --- | --- | --- | --- | --- |
   | m=1（p1/p3/p7/p15 四档） | 逐位 1.0000 | 逐位 1.0000 | 逐位 1.0000 | 逐位 1.0000 |
   | m=4097（p15） | 逐位 1.0000（ulpMax 1） | 逐位 0.9949（良态 0.9985，ulpMax 1） | 逐位 1.0000（ulpMax 1） | 逐位 0.9797（良态 0.9903，**ulpMax 11**）⇒ **FAIL** |
   - `h2.blk` 在 m=4097 **有一条判定项红**（ulpMax 11 > 2）：形态与 m27 在 m=4097 档观测到的
     `blk` 边际同类（m27 README §7 第 0 条记的也是 `blk` 的余差）。**如实登记，不当作通过**。
     它与"重定位/相位接线"的关系未在本次归因（可能含 513 块累计的数值路径；m27 对同源的
     `blk` 余差也未排除设备侧因素）。
3. **相位 B（MoE router 段，E=512）** = `m26_moe_prefill/check_ref.py`（复用 m26 的判据，不复制数学）。
   **判定项条数随档而变**：m26 的 `J1`（topk_ids 逐位）只对"参考的 10/11 名间隔 > 4×该行 logit 容差"的
   行生效（`m26_moe_prefill/check_ref.py:207-211` 的 tie-risk 门）；没有这样的行时 **`J1` 不入判定项**
   （该档从 5 条降到 4 条），`ids` 仍由 `J1b`（按集合）判、tie-risk 行数单列成报告项
   （`check_ref.py:271`）。**这是 m26 的既有设计，不是本 mission 引入的缺陷**；本档 `m1_p12` 就是这种情形。
   | 档 | clean | mut1 | mut2 |
   | --- | --- | --- | --- |
   | `m1_p8`（只 B，输入 = `A.xLayer`） | **5/5 PASS** | 5/5 PASS | 5/5 PASS |
   | `m1_p12`（H2+B） | **0/4 FAIL** | 4/4 PASS | 4/4 PASS |
   | `m1_p12_rep2` / `m1_p12_rep3`（**同配置重复跑**） | 0/4 FAIL | 4/4 PASS | 4/4 PASS |
   | `m1_p12_nopad` / `m1_p12_pad3c`（pad 控制） | 0/4 FAIL | 4/4 PASS | 4/4 PASS |
   | `m1_p15`（全四相位） | **0/5 FAIL** | 5/5 PASS | 1/5（仅 J1b） |
   | `m1_p15_rep2` / `m1_p15_rep3`（**同配置重复跑**） | 0/5 FAIL | 5/5 PASS | 1/5 |
   | `m1_p15_nopad` / `m1_p15_pad3c`（pad 控制） | 0/5 FAIL | 5/5 PASS | 1/5 |
   | `m4097_p15`（**最后一块**） | 5/5 PASS | 5/5 PASS | 5/5 PASS |

   ⇒ 判定模式在所有档上**稳定**（clean 一直红、mut1 一直绿）；但 **logits 本身在 `m1_p12` 上
   同一配置重复跑就不一致**（2.58e8–4.31e8，见 §8 第 4 条与第 7 条），`m1_p15` 上逐字节一致。
   ⇒ 相位 B **单独**在本 kernel 里是 5/5 PASS；**hc 段先跑过**时第一个 arm 就错（见 §8 第 4 条）。
   **本 mission 不把它当作通过。**
4. `injw` / `rstd` 两个证据面**未比较**：挂载点按 m27 README §5(b) 的"最小新增"给 `nullptr`（不重定位成
   平铺面）⇒ 本档的覆盖范围 = `hcp` + `blk`（它们已经覆盖 injW→combine→norm→down/inj→silu→up→gate mix 全链）。

## 7. 负向对照（"把被测对象弄坏必变红"）

| 对照 | 弄坏什么 | 读数 |
| --- | --- | --- |
| `Pf.gdn_mut1`（`M15_PREFILL_MUTANT` 语义 = 1） | 只把**送设备的** `h0` 翻倍（host 干净副本仍落盘） | m23 对拍 `Pf.gdn_mut1` FAIL（m=1：o 超界 4190/6144；m=4097：130298/25171968）✓ |
| `Pf.gdn_mut2`（语义 = 2，**M140 新增**） | 只把**送设备的** `hcH` 翻倍（hc 权重与其余输入不变） | `check_full_layer.py` 在 p1/p3/p7/p15 × m=1、m=4097 全部报 `h1.hcp`/`h1.blk` **FAIL**（逐位率掉到 0.0007–0.0041、ulpMax 约 3.2e4）✓ |
| `noreloc` 式（借 m27 的现成对照） | 不属本 mission 的新对照；m27 的 `rowsteal`/`norel`/`sinkreloc` 覆盖 hc 段本身 | 见 `m27_hc_prefill/README.md` §4.4 |

**对照的限度（如实）**：`mut2` 搅动的相位是 H1；H2 侧**没有**一条"只弄坏 H2"的对照 ——
`h2.hcp/h2.blk` 的参考输入取**设备自己**的 H1 产物（`h1_hcp`），所以把 H1 弄坏时 H2 的参考跟着变、
H2 照样绿（读数见 `logs/run_m1_p15.log` 的 mut2 段：H1 红、H2 绿）。H2 判据的"有牙"目前只由
判据本身（与参考逐张量比）承担，**未**有设备侧反向证据。

## 8. 已知缺口（逐条，**不假装接上**）

1. **H1 的 BLK → 相位 A 的输入：缺 **prolog**。** 相位 A（B1，`m15_gdn_prefill.h`）的输入契约是
   "已过 prolog 的 q/k/v/g/β"（m23 `check_ref.py` 的边界声明），而 H1 的产出 `pfHcBlk0` 是
   **子层段的层输入**（`[m, HID]`）。两者之间要 `in_proj`（qkv/a/b/z）+ `conv1d` + `l2norm` —— 这段
   **本仓 prefill 路径上没有实现**。本档的 q/k/v/g/β 由宿主合成（M136 同款口径，见 `.asc` 的
   `H_PfGdnFillHost`）⇒ **判据覆盖"扫描段"，不含 prolog**。卡在哪：需要一个 prefill 的 in_proj/conv
   prolog 段（形状/权重槽已由 M88 的 attn prolog 与 `m15_gdn_layer.h` 的 decode 路径给出）。
2. **相位 A 的出口 → H2 的 `bo`（`hcAttnOut`）：缺 **epilog**。** B1 的出口是 `wsO`
   （`[48, m, 128] fp32`，每 v-head 的 o），而 H2 要的 `hcAttnOut` 是 `[m, HID=2560] bf16`（单流子层出口）
   —— **不同型、不能直接别名**（m27 README §5(e) 的 M138 登记）。两者之间要 `o_proj` + `RMSNormGated` +
   残差 + 门控。本档给 H2 的 `bo` 是**宿主合成**的确定性平面（`H_PfHcFillHost`），**不是**相位 A 的产物。
   卡在哪：需要 B1 的出口投影段，或由塔裁定 `wsO` 与 `hcAttnOut` 的规范平面关系。
   （**这也是"GDN 的 wsO → hc(mlp) 的 bo"这条接线目前的状态：未接**，理由如上。）
3. **H2 的 BLK → 相位 B 的输入：已接（类型/形状一致）**；相位 B 在**单独跑**时（`m1_p8`）对拍通过，
   但**跟在 hc 段之后**时产出与声明输入不符 —— 见第 4 条（未闭合）。
4. **H2 → 相位 B：已接线（类型/形状一致），但"相位 B 跟在 hc 段之后"时 router 产出与输入面不符
   —— 根因仍是**未闭合**的 in-scope 项（复审 r1 要求不得留下未确立的归因）。**
   已在 `M15L_PrefillPhaseB` 把层输入接到 H2 的 BLK 平铺面（`pfHcBlk1`，与 MoE 的 `xLayer` 同型同形）。
   证据（全部在同一二进制 `binary_sha256.txt` 上）：

   | 档（掩码） | phase B 的输入来自 | m26 对拍（clean / mut1 / mut2 三个 arm） |
   | --- | --- | --- |
   | `m1_p8`（只 B） | `A.xLayer`（host 填的确定性平面） | **PASS / PASS / PASS** |
   | `m1_p12`（H2+B） | H2 的 BLK（H2 的输入是毒值） | **FAIL** / PASS / PASS |
   | `m1_p15`（全四相位） | H2 的 BLK | **FAIL** / **PASS** / FAIL(1/5) |
   | `m4097_p15` | H2 的 BLK（**最后一块**） | PASS / PASS / PASS |

   ⇒ 相位 B **单独跑（p8）在同一个 kernel 里是对的**（5/5 PASS）⇒ "MoE 段自身的数学/该段在本 kernel 里
   不可用"这个假设被排除；**只要 hc 段先跑过（p12/p15），clean（第一个 arm）就错**，而后续 arm 大多对
   （p15 的第三个 arm 只过 J1b）。p8 与 m=4097 档的三个 arm 全对。
   （**pad 的 A/B 控制档** `m1_p15_nopad`/`m1_p15_pad3c`/`m1_p12_nopad`/`m1_p12_pad3c` 与
   **repeat-run 基线** `m1_p{12,15}_rep{2,3}` 的判定都与对应主档**完全一样**；其 logits 层面的
   读数与结论见下面 (a) 与 §8 第 7 条。）
   - **纯函数性见证**（`reproduce.sh` 最后一段，逐字节）：比较面是 H2 的输出 `h2_blk`
     （它 = 相位 B 的**输入面**，**仅**在确实跑 H2 的档（p7/p12/p15）成立；p8 档相位 B 的实际
     输入是 `pfXDev`/`m26_x`，**不是** `h2_blk`）。`h2_blk` 在 clean 与 mut1 两档之间**逐字节相同**；
     产出 `moe_y` 的对照读数 = `p8 SAME / p15 DIFF /
     p12 DIFF / m4097 SAME` ⇒ 在 p12/p15 上，相位 B 的产出依赖**声明输入以外**的东西
     （前序段留在共享 UB/L1/L0C/BufferID 里的状态，或跨 launch 的状态）；p8 上是纯函数。
   - **确定性**：`m1_p15` 连跑两次，`m26_Pf.gdn/*` 与 `m140_*.bin` **逐字节相同** ⇒ 不是随机竞态。
     设备 logits 与跑完后回读的**同一块**输入面（`m26_x.bin`，已核对 = `m140_Pf.gdn_h2_blk.bin`，
     毒值 0）对不上：max|Δ| = 6.816e8，而参考 `x@wᵀ` 的量级是 O(1)。
   - **已用读数约束的假设（逐条）**：     (a) **"MoE 输入面没按 `MT=64` 零填充"**（复审 r1 的 in-scope 候选，类比相位 A 的
     `PF_GDN_QK_PAD_ROWS`）—— 填充**已实现**（`pfHcBlk1` 按 `ceil(m/MT)*MT` 行分配 + 起 kernel 前把
     尾块 pad 行置零）。**但 r1 的那次实验不是单变量**（它同时改了分配尺寸），而且当时写的
     "读数逐字节不变"**是错的**（复审 r2 用已提交 dump 实测：`m26_logits` 在 `852dfbc→9130982`
     之间 `max|a-b| = 2.58e8`，两者 maxAbs 都是 6.816e8）—— 那条表述**已撤回**，当时引的
     "`Mmad` 第 0 行只由 x 第 0 行决定"这条理由**没有支撑，一并撤回**。
     **r2 做过一次"干净 A/B"，但它的结论在 r3 被数据反驳 ⇒ 撤回并整体降级（复审 r3 给的选项 (c)）**。
     r2 的控制（保留，代码在 `m15_layer_loop.asc`）：**开关侧** `M15_PREFILL_NO_PAD=1` 跳过"尾块 pad 置零"
     （分配尺寸与其余代码不动）；**内容侧** `M15_PREFILL_PAD_BYTE=60` 把填充字节从 `0x00` 换成 `0x3C`
     （每 16bit lane = `0x3C3C`；按 bf16 位域解码 = `2^(120-127) × (1+60/128)` = **0.0114746**，
     r2 把它标成 "bf16 1.0" 是**错的**，复审 r3 指出、此处已更正）—— memset 尺寸/次数一样，只有内容不同。

     **r3 的 repeat-run 方差基线**（同一二进制、同一档重复跑；`reproduce.sh` 的 `rep_group` 段）：

     | 组 | 基准 | rep2 | rep3 | nopad | pad3c |
     | --- | --- | --- | --- | --- | --- |
     | **p12**（`maxAbs` 基准 6.8160e8） | — | **max\|diff\|=4.31e8** | **2.58e8** | **2.58e8** | **5.13e8** |
     | **p15**（`maxAbs` 基准 6.8160e8） | — | 0.0（逐字节相同） | 0.0（逐字节相同） | 0.0（逐字节相同） | 0.0（逐字节相同） |

     四组对照全部落在**同一二进制**上（r2 的 `pad3c` 档是跨二进制比的，那 3.39e7 是**跨版本假象**；
     同一二进制下 p15 的 pad3c 与基准**逐字节相同**）。`m26_x`（只有第 0 行）在全部对照里逐字节相同。

     ⇒ **本 mission 的 A/B 无法把 pad 的作用与 run-to-run/时序抖动分开**：
     · **p12 组**：同一配置重复跑的抖动就有 **2.58e8–4.31e8**，与 pad/nopad/pad3c 的差值**同量级**
       （`nopad` 的 2.5815e8 与 `rep3` 的 2.5815e8 **恰好相等**）⇒ **pad 的差值完全被 run-to-run
       抖动解释掉**，不能归因于 pad。
     · **p15 组**：pad 写/不写、pad 内容 0x00/0x3C 四种对照**全部逐字节相同** ⇒ 在稳定档上 pad
       这个变量**没有**移动读数。
     · r3 复审点的 `0x00 ≡ 0xCD ≠ 0x3C` 那个模式（r2 的跨版本差 3.39e7）在本轮的同一二进制下**不存在**
       （该对逐字节相同）⇒ 它是跨版本假象，不是内容耦合。**本 mission 不再主张任何 pad-耦合结论**：
       r2 写的"第 0 行 logits 依赖 ≥ m 的行"**撤回**。
     · **保留下来的只有一条代码事实**：`pfHcBlk1` 按 `ceil(m/MT)*MT` 行定尺（这消掉了 m=4097 最后
       一块读越界的问题）——它**不是**从 A/B 得出的因果关系。
     · **另见下面 §8 第 7 条的独立事实**：`m1_p12` 的 clean 输出**不可复现**（同配置重复跑差 2.58e8–4.31e8），
       而 `m1_p15` 的可复现（0.0）。
     (b) **AIV→AIC 的可见性握手缺失**：试过加 AIV `CrossCoreSetFlag<mode2,PIPE_MTE3>` /
     AIC `CrossCoreWaitFlag<mode2,PIPE_MTE2>`（复用 `PFR_FLAG_M2_RING[0]`）—— 读数变了
     （J2 0/512 → 128/512 列过）但**没修好**；代码已回退，读数归档在
     `logs/h2b_sync_experiment_m1.log` + `logs/h2b_sync_experiment_m26_Pf.gdn/`。
     (c) **"等 hc 段的搬运落地"**（p8 vs p12 的对照）—— p8 证明相位 B 单独跑是对的。
   - **卡在哪 / 需要什么**：本轮把两条候选都收掉了（pad 的 A/B 被 run-to-run 抖动混淆、AIV→AIC 握手
     加了没修好），**剩下的最可查线索是"p12 档同配置重复跑就不一致、p15 档可复现"这个对比本身**：
     先要一个能**重复触发/抑制**该抖动的实验（例如固定/交换 arm 顺序、把 clean 放到第 2/3 个 arm、
     或把 `M15_PREFILL_KIND` 的 arm 数改成 1），把"第一个 arm 效应"与"p12 配置特有的不稳定"分开。
     这一层仍在 scope 内（挂载点与 arm 编排都在 `m15_layer_kernel.h`/`.asc`）。
     **本 mission 不把相位 B 的 m=1 数值判据当作通过**，也不再对它主张任何具体机理。
5. **attention 的 prefill（B2/B3）未接**：`KIND_ATTN` 在 `wire=1` 下**响亮失败**（`H_RunPrefill`
   的 `H_CmpOk(C,false)`），本 mission 未动。
6. **MoE 的 `m > MT` 块循环**：`MoePrefillChain` 是 `MT=64` 单块编排，本 mission 在
   `M15L_PrefillPhaseB` 里加了块循环（m26 README §5.4 点名的 "host/挂载点组装项"）⇒ m=4097 全档通过，
   但**判据只对最后一块做了 m26 数值对拍**（`m26_meta.txt` 的 `m = 最后一块行数`），其余 64 块只有
   "非毒值/非全零"级别的判据。
7. **独立事实（比 pad 结论更重要）：`m1_p12` 档的 clean 输出不可复现，而 `m1_p15` 的可复现。**
   同一二进制、同一档、同一配置重复跑（`reproduce.sh` 的 `rep_group` 段，`m26_x` 逐字节相同、
   内核代码未变）：
   · **p12**：基准 vs `rep2` = **4.31e8**、vs `rep3` = **2.58e8**（`max|diff|`，logits 的 `maxAbs`
     都是 6.8160e8）⇒ **同配置重复跑就有 2.58e8–4.31e8 的抖动**。
   · **p15**：基准 vs `rep2`/`rep3` **逐字节相同**（0.0）。
   ⇒ 这解释了 r2 的 p12 两对为何"有差值"（差值等于抖动，不是 pad）；也说明 **`m1_p12` 档的相位 B
   判据在 m=1 上不是一个可复现的判据**，只能当"现象登记"。它与"**clean（第一个 arm）错、后续 arm 对**"
   属同一类现象：相位 B 在 m=1 上的产出依赖**声明输入以外**的东西（进程内状态 / launch 序），
   并且这种依赖在 p12 配置下连**同一配置的重复跑**都不稳定。机理未定位（本 mission 只登记读数）。

## 9. 复算方法（洁净检出能跑到哪一步 —— 复审 r1 P2-2 的口径）

```bash
# 离线复算（用入库的 dumps_*；脚本对缺输入会打印 SKIP 原因，不抛 traceback）
bash m15_layer_loop/evidence/m140_prefill_full_layer/reproduce.sh
# 重跑设备档后再复算（需设备；每档各自进锁、锁内 npu-smi 落盘）
M140_DEVICE=1 bash m15_layer_loop/evidence/m140_prefill_full_layer/reproduce.sh
```

**洁净检出（`git archive` 抽 tip）能跑到哪一步**（逐条）：

| 判据 | 洁净检出上能否跑 | 依赖 |
| --- | --- | --- |
| hc 四相位（`check_full_layer.py`） | **能**（`dumps_m1_p1/p3/p7/p15/p8/p12` 的激活面入库） | hc 权重面不入 git ⇒ 判据改从 `m15_layer_loop/weights_manifest.txt` + checkpoint 取（与 host 装载同源）；checkpoint 不可达则该档 SKIP |
| 相位 A（m23） | **只在 `dumps_m1_p15`** | 该档的 `h0/ht`（各 3.1 MB）与 `q/k/v/g/β/o` 入库；m=4097 档缺 |
| 相位 B（m26） | **只在 `dumps_m1_*` 档** | `router pad` 入库**一份**（`dumps_m1_p15/m26_Pf.gdn/m26_wpad.bin`，3.27 MB，与 tag/m 无关），脚本把它拷进各 tag 子目录 |
| `dumps_m4097_p15` 的任何判据 | **不能**（整档 `*.bin` 不入 git，约 1.9 GB） | 需 `M140_DEVICE=1` 重跑该档（锁内一条命令） |

大 dump 不入 git（`.gitignore` 的反选口径）：hc 权重平面（`*_w_a{1,2}_*.bin`，每档约 27 MB）、
`*moe_wpad.bin` / `*m26_wpad.bin`（除入库的那一份）、`m23_*_h0.bin` / `m23_*_ht.bin`（除 m1_p15 的一份）、
以及 `dumps_m4097_p15/**/*.bin`。重建后 `sha256sum -c sha256sums.txt` 应与入库清单一致。

## 10. 显式未做 / 限度（不得读成"prefill 已打通"）

- 只到 **KIND_GDN 的单层**：`M15_LAYERS=1`（人类口径：单层测试不必装全部权重）。48 层链、
  多类层（attention 层）的 prefill 路径**未验**。
- **激活由宿主合成**（hcH/hcBo/hcIj/hcAttnOut 与 q/k/v/g/β）；**权重是真的**（hc 从
  `HC_LAYER_W_STRIDE` 槽、MoE 从 manifest 的 `moe512_*` 真 shape 段）。
- **只跑了 layer 0**（`M15_LAYERS=1`）⇒ 真链上 `layer 0` 的 hc(attn) 边界 mode 是 `MODE_MIX`，而
  本 harness 对**两个**边界都取 `MODE_COMBINE_MIX`（`fields.md` 行 14 与 `.asc` 的差异）：
  `MODE_MIX` 不物化 `H'`，而边界 #2 的 `hIn` 正需要它（零拷贝）⇒ 这是刻意的 harness 选择，
  但**真链的那个 mode 本档没有跑到**。
- H1/H2 的 `injw`/`rstd` 不比较；`H_PfHcFillHost` 的 `mutant` 参数目前只区分 1/2（H2 的单独
  负向对照**未做**）。
- **H1 未开的档没有构造**（阶梯从 H1 起）⇒ "H1 未开时 `arena0` 保持毒值"这条**没有读数**；
  已执行的只有 H2 关（p1/p3/p8）的反面判据。
- **H2 的 `bo` 是宿主合成（缺口 #2）** ⇒ H2 的数值判据覆盖"给定 bo 的 hc 边界"，**不覆盖**
  "bo 来自子层段"这条链接；相位 B 的 m=1 数值对拍**未通过**（缺口 #4）。
- **相位 B 在 m=1 上的读数不是一个可复现的判据**：`m1_p12` 档同配置重复跑的 logits 就相差
  2.58e8–4.31e8（`m1_p15` 档逐字节一致）⇒ 本 mission 的 pad A/B **无法**把 pad 的作用与
  run-to-run/时序抖动分开，**不主张**任何 pad-耦合结论（见 §8 第 4 条 (a) 与第 7 条）。
- 设备档**未归档设备侧错误报告**（`errStr`/`ECC`/`aivec` 等独立计数）：跑档只落自检行与 dump。
  形态上本次设备的失败档（`logs/h1only_m1_stale_hostptr.log`）确实出现过 `aivec error exception`，
  修复后各档 `exit=0`；但"本档内没有设备侧异常行"这句话**未取证**（同 M27 README §4.6 的口径）。
