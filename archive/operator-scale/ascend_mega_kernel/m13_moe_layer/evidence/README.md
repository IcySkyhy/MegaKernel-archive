# M13 证据归档（M32：MX 量化非有限 parity 修复 / M50：±Inf 参考修复 + Extract VF）

## M50 追加的文件（±Inf/NaN/次正规 参考 parity + Extract 改造；原 M32 文件见下一节）

| 文件 | 内容 |
|---|---|
| `m50_quant_ref_inf_after.log` | **判据 1/2**：`check_ref.py quant-inf`（修复后的 `quant_mxfp4_hw`）对 device dump 的逐字节判据 —— **40 条判定项 / 40 PASS**（+ 2 条报告项单列） |
| `m50_quant_ref_inf_prefix_nofix.log` | **判据 2/2 = 负向对照（必须 FAIL）**：同一条判据换成 **base commit 的原始参考文件**（`git show <base>:m13_moe_layer/check_ref.py`，sha256 记在文件头，不是改动里的开关）—— **40 条判定项 / 27 PASS / 13 FAIL**（A/H 两侧的 inf=1/2/3 六条用例的 qx+scale，以及 A 侧 inf=4 次正规用例的 qx；13 条 FAIL 全在 40 条判定项内） |
| `m50_quant_inf_after.log` | `quant-inf` 设备自检（M50 新增 kind 4 = 次正规/退化组）：**40/40 判据 PASS**（M32 时为 32/32） |
| `m50_zero_regression.log` | 零回归证据（全部机器生成）：数据集 682 条判据与改动前**逐字节相同**；`M13_DUMP=1` 的 **144 个 dump 文件 sha256 全部相同**；140 个张量与 `dump_manifest.md` 0 不符；`check_ref.py` 118 判定日志逐字节相同；quant-inf 32 → 40 |
| `m50_check_ref_tristate.log` | 审校脚本**三态退出码**负向对照（tower 2026-09-26 规则）：正常输入 rc=0（计数正确）、空输入 `SKIPPED` + rc=2（不发合格证）——`quant-inf` 与数据集模式各一组读数 |
| `m50_dataset_after.log` / `m50_check_ref_after.log` | M50 复跑的**原始**数据集全链日志（682 条判据行）与 `check_ref.py` 数据集日志（118 判定 + 32 参考项）——与 M32 归档的 `run_dataset_after.log` / `check_ref_run.log` **逐字节相同**（`diff` 空），故内容上不引入新读数，只为满足 `docs/17` §2.5「M50 自己的原始日志齐备」 |
| `m50_dump_sha256.txt` | M50 复跑落下的 144 个 dump 文件（含 4 个 `*_meta.txt`）的 sha256 |
| `quant_inf_dumps/qs_{A,H}_i4_r1_s0_*.bin` | M50 新增的 kind 4（次正规/退化组）device dump：A 侧输入为 +0/正次正规 0x0001..0x000F/最小正规 0x0080，H 侧 SwiGLU 输出 = 正次正规 2^-130（0x0008）；两组 scale 字节 = 0、data 码全 0 |
| `quant_inf_dumps_sha256.txt` | 上述 dump 清单，现为 **55 条**（44 条 M32 + 11 条 M50 的 kind 4） |

复跑（判据 1/2 与 2/2 都可离线重放）：

```bash
cd <repo> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B /tmp/m13m50/build -S m13_moe_layer -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/m13m50/build -j4
mkdir -p /tmp/m13m50/qs && cd /tmp/m13m50/qs && M13_QS_DUMP=1 /tmp/m13m50/build/m13_moe_layer quant-inf   # 40/40 PASS
cd <repo>
python3 m13_moe_layer/check_ref.py quant-inf m13_moe_layer/evidence/quant_inf_dumps                       # 40 判定项：40 PASS
git show <base>:m13_moe_layer/check_ref.py > /tmp/check_ref_prefix_nofix.py
python3 m13_moe_layer/check_ref.py quant-inf m13_moe_layer/evidence/quant_inf_dumps \
        --quantizer /tmp/check_ref_prefix_nofix.py                          # 40 判定项：27 PASS / 13 FAIL（负向对照）
```

判据口径（`check_ref.py quant-inf`，README **§5.4 末 / §8.1**）：每个用例 4 条判定项 —— ① `qx` 逐字节；
② scale 行内合法区逐字节；③ scale 行内 padding 未被写；④ 结构性（只用 device 字节与输入位型）：
非有限组/退化组确实存在且这些组的 device data 码全 0。**10 用例 × 4 = 40 条判定项**；
另有 **2 条报告项**（`docs/17` §2.1 分栏，不计入 PASS/FAIL）：用例集合完整性 guard 与
`tools/golden` 交叉见证（`moe_block_ref.quantize_ocp` vs device 逐字节，**0 字节不符**）。

## M32 追加的文件（原 `accept_run_m1_m33.log` / `check_ref_run.log` / `dump_manifest.md` 见 README §5.5）

| 文件 | 内容 |
|---|---|
| `run_dataset_after.log` | 修复后数据集全链运行（`m1 m33` × mode 0/1）：**682 条判据行**全 PASS、rc=0（判据行口径：带 `PASS (` 的 678 行 + `routing 内部一致性` 4 行，不含 `ALL PASS` banner） |
| `run_dataset_prefix_nofix.log` | **修复前**（删掉官方 `:413` 那句 `Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);` 后重编）同一命令的日志 —— 与修复后**逐字节相同**（零回归证据：含 Inf 的修复不影响任何既有判据） |
| `quant_inf_after.log` | `quant-inf` 子命令自检：**32/32 判据 PASS**（side A/H × rows {1,64} × seed {0,1} × infKind {0..3}）；**M50 起为 40/40**（新增 kind 4，见上节 `m50_quant_inf_after.log`） |
| `quant_inf_prefix_nofix.log` | 修复前同一自检：**16/32 PASS**（inf=1「整组 +Inf」与 inf=2「组内部分 Inf」各 4/4 FAIL、A/H 两侧合计 16 FAIL；inf=3（NaN）本来 PASS） |
| `quant_inf_dump_run.log` | 带 `M13_QS_DUMP=1` 的自检运行日志（对应下面 dump 的落盘记录） |
| `quant_inf_dumps/qs_{A,H}_i{0..3}_r1_s0_*.bin` | 自检 dump：`x`、`swiglu_device`（仅 H 侧）、`qx_device`、`scale_device`、`qx_ref`、`scale_ref`；`A` 侧 K=2560/SS=80，`H` 侧 K=640/SS=32（M50 又补了 `i4`，见上节） |
| `quant_inf_dumps_sha256.txt` | dump 的 sha256（按统一验收口径入库；M32 时 44 条，M50 后 55 条） |
| `ocp_independent_witness.log` | **独立见证**：`M13_QS_DUMP=1 quant-inf` 的全部 32 个 device dump 与 M26 分支 `6f68f8e` 的 `tools/golden/moe_block_ref.quantize_ocp`（官方 ops-nn 序列的逐句 numpy 转写，另一个 agent 的独立实现）**逐字节一致**：A 侧 16/16、H 侧 16/16 |

## 复跑

```bash
cd <repo> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m13_moe_layer/build -S m13_moe_layer -DCMAKE_BUILD_TYPE=Release && cmake --build m13_moe_layer/build -j2

# 量化段自检（无需数据集）
mkdir -p /tmp/m13qs && cd /tmp/m13qs
<build>/m13_moe_layer quant-inf                                   # 期望 ALL PASS，40 判据 + 总结行（M50 起）
M13_QS_DUMP=1 <build>/m13_moe_layer quant-inf                     # 额外落盘 qs_*.bin 供离线复算

# 数据集全链（消费 tools/golden/data/{m1,m33}）
<build>/m13_moe_layer <repo>/tools/golden/data m1 m33             # 期望 682 条判据行全 PASS（不含 ALL PASS banner）

# 修复前对照（临时副本，只删那一句 Select）
cp -r m13_moe_layer /tmp/m13_prefix
sed -i '/nanRegTensor, cmpResult/d' /tmp/m13_prefix/m13_moe_layer.asc
cmake -B /tmp/m13_prefix/build -S /tmp/m13_prefix -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/m13_prefix/build -j2
/tmp/m13_prefix/build/m13_moe_layer quant-inf                     # 期望 16 PASS / 18 FAIL（含总结行）
/tmp/m13_prefix/build/m13_moe_layer <repo>/tools/golden/data m1 m33  # 与修复后日志逐字节相同
```

## 自检判据口径（`quant-inf`）

* device `qx`（每行 K/2 字节）与 device `scale`（每行前 K/32 字节）对 host C 参考（官方语义，含 `:413`）**逐字节一致**；
* scale 行内 `K/32` 之后（`SS=32 > 20` 的 padding 区）必须**未被写**（预置 0）；
* H 侧（SwiGLU）另比 device vs host bf16 参考：网格 ≤1 ULP + 有限/非有限归类逐元素一致；
* 注入组（`grp%4==1`）的 device nibble 必须**全 0**（修复前为 ±6 饱和码 `0x7`/`0xF`）——M50 的 kind 4（次正规/退化组）同判据：组 amax 指数域 < `0x0100` ⇒ `halfScale = 0` ⇒ 组内码全 0。

`A` 侧对应正式链路的 `quantA`（K=HIDDEN=2560、无 SwiGLU、SS=GU_SCALE_STRIDE=80），`H` 侧对应 `quantH`（K=INTER=640、带 SwiGLU、SS=DN_SCALE_STRIDE=32），两者复用的是**同一份 `VecQuantStage` 设备代码**（同一源码行），只是模板参数与行索引方式不同。
