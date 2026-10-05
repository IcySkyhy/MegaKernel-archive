# M40：MoE FFN 段融合进 per-layer kernel（每层一次启动）

> **M96 增量（2026-09-27）**：**修掉 M90 改写 `check_moe_ref.py` 时引入的三处口径漂移**，让本文件
> 「MoE 段独立 numpy 交叉校验」这一项在 main 上**重新可执行**。M90 之后（`26d644b..0bd80c6`）该判据
> 在 main tip 上**直接崩**（`ValueError: cannot reshape array of size 32 into shape (20)`，rc=1）⇒
> §1 表里列的 `128 + 16` 当时**无法复现**。三处漂移是同一类错误：把「**静态上界 / 槽位几何**」与
> 「**运行期 / 有效量**」绑到同一个裸符号上。
>
> | # | 漂移 | 原写法 | 正确语义 |
> |---|---|---|---|
> | 1 | H(= down 输入) scale 的**槽宽**被当成**有效前缀** | `M17R.dequant(..., INTER // GROUP)`（20） | `scale_stride` 吃**槽宽** = `M15M::DN_SCALE_STRIDE`（32 B/专家行）；20 只用于「每行取多少组」 |
> | 2 | **运行期 topk** 被当成 meta 的 `topk_max`（**模板上界** 4） | `TOPK = int(meta["topk_max"])` | 运行期 topk = 2（与 harness `m15_layer_loop.asc` 的 `M15_TOPK` 同源）；`x_sorted`/`h_swiglu` 的**定尺**才用上界 |
> | 3 | `t3` 的 **1-D/2-D 形状口径**（参考 `(1,N)` vs 设备一维读） | `router_logits` / `GU_shd` / `Y_shd` 三条 T3 变 shape FAIL | `t3` 保持形状严格（不许静默广播），设备读处补成 `(1,N)` |
>
> **修完的读数**（在 main tip 的融合 kernel dump 上实跑，`m15_layer_loop/check_moe_ref.py <dump>`）：
> **`128 条判定项 + 20 条非空洞性，0 FAIL`，`rc=0`**。判定项条数与 M40 归档的 `128` **相同**（验收面
> 没缩）；非空洞性 **16 → 20**，原因是 M90 起每层多一条 T4 guard「路由参考非退化（活跃专家数可变）」
> —— 是**加强**，不是放宽验收面。参考项（print-only）**16 → 8**：M90 去掉了每层两条重复的 print-only
> 报告项（`combine 位级` / `moe_output 绝对残差`），它们比的对象现在由**判定项** `T3 shared_output` /
> `T3 moe_output` 承担。
>
> **语义没被改动的直接见证**：在**同一份 dump**、同一台机器上，`26d644b` 版（M90 之前）与本轮修好的
> 版本都打 `判定项 128 条：128 PASS / 0 FAIL`（`rc=0`）；两份日志的差异只有：① 头部那行的 topk 措辞、
> ② 非空洞性段多出的 4 行新 guard、③ 判定项/报告项的名字（T1/T3 档位标签）与参考项的条数。即
> **「修好」= 按正确的语义取数把原来的 PASS 拿回来，不是把判据改宽**。
>
> **负向对照**（第五变体纪律「把被测对象弄坏，判据必须变红」）：`check_moe_ref.py … --negctl` 在
> **同一份 dump、同一条判据路径**上把取数环节**在内存里**弄坏（不写任何文件），本机实测：
>
> | 变异 | 读数 |
> |---|---|
> | 正对照（不注入，必须绿） | `rc=0`；判定项 128 条 0 FAIL / 非空洞 20 条 0 FAIL |
> | V1 行距按**另一个候选值** 20 传（= M90 回归本体） | `rc=1`；`ValueError: cannot reshape array of size 32 into shape (20)` |
> | V2 槽内取**尾** 20 B（取数区段偏移错） | `rc=1`；12 FAIL = 每层 `T3 Y slot{e}`（活跃槽）+ `T3 Y_shd` |
> | V3 槽内**逆序**（取数区段方向错） | `rc=1`；12 FAIL（同 V2 的判据集合） |
> | V4 运行期 topk 按**模板上界** 4 取（多读 0xCD 污染 id） | `rc=1`；152 条里 84 FAIL，非空洞 20 条里 8 FAIL |
>
> **对象侧对照**（变异只在 `/tmp` 的 dump 副本上，不改仓内工件）：① 把副本 `moe_layout.txt` 的
> `H_scale` 槽宽改成 20 ⇒ 新增的 `_assert_scale_slot()`（把「槽宽」钉在布局表 + `M15M::DN_SCALE_STRIDE`
> 上）**就地报错**，rc=1；② 把副本 `L00_moe_ws.bin` 的 H_scale 槽首字节写成 `0xFF` ⇒
> `T3 Y slot1` 与 `T1 H_scale 逐字节` FAIL（2 条），rc=1；③ **只改槽内 `0xCD` 填充区**（第 20 字节）
> ⇒ 仍然 `128 + 20 / 0 FAIL` —— 判据读的是前 20 B **有效前缀**，没有把整槽当有效数据（这条防的是
> 「靠把判据改宽来把 rc 弄绿」）。
>
> **陈旧读数收口**（扫描口径：`git grep` 全 tip 树，且**覆盖行内复合读数里的数字**——不只看独立列出的条数串；
> M40/M52 归档的**日志文件本身没动**，动的是把它们当「当前读数」的叙述）：
> 本 `README` 内逐条处置：§1 表的「128 + 16」、§5.4 标题（正文仍是 M40 的 m13 参考链描述，M90 已改指
> m17 —— 本 mission 只收口**读数**，不改写 M40 的历史叙述）、§5.7/§5.9 的两处「128 判定项 + 16 非空洞性」、
> §5.6 证据表里两行日志说明 → 按上面的新读数或「M40 当时读数」重新标注。
> **同一轮扫描命中的仓内其它文件，逐条处置**（下列是**本轮扫描命中到的全部** `check_moe_ref` 相关行；
> 不声称「仓内已无其它引用」——只声称「本轮扫到的都按片段归了类并处置」）：
> ① `docs/17-verification-standard.md` §6 的 m15 行里 `MoE 段 numpy 128+16` **是旧条数**（行内复合读数：
> 它跟在前面的 `709 numpy（M25 原有）` 后，按行文属当前复合读数）⇒ 塔已把 `docs/17` 列入本轮写权限，
> **本轮直接改正**为 `128+20` 并注明口径（M96 后 = 128 判定项 + 20 非空洞；M90–M96 之间该判据在 main
> 上不可执行，`128+16` 是 M40/M52 归档读数）；
> ② `docs/17` §1.2 表的 `m15_layer_loop/check_moe_ref.py:138` 是**行号引用**（不是读数），在本轮之前
> 就已过期（M90 已把 `MB.router_topk` 的调用移到别处；行号不是稳定定位依据）—— 属 M90 的行号漂移、
> 与本 mission 的读数口径无关，**只登记**（另有 `TowerFinding` 交塔）；
> ③ `m13_moe_layer/README.md` 里三处「`m15_layer_loop/check_moe_ref.py` 直接 `import` m13 的
> `check_ref.py`（`M13R`）」—— 那是 M40/M50 当时的形态，M90 起该 import 只在 `--selftest` 里作行为
> 保持对照用（不是读数，也不在本轮写权限内，只登记）；
> ④ `m15_layer_loop/evidence/check_moe_ref_run.log` / `check_moe_ref_run_m52.log` 里的
> `判定 128 + 非空洞 16` 是**归档日志自身的字节**（那两个时点的运行记录）⇒ **不动**（改它等于篡改归档），
> 由上文 §1/§5.6 的读数行指明「这是当时读数」；
> ⑤ 其余命中（`check_hc_ref.py`、`m15_chain_host.h`、`m15_layer_loop.asc`、`m15_moe_host.h`、
> `moe_relift/m91_README.md`、`tools/golden/README.md`、`tools/golden/moe_real_accept.py`、
> `evidence/accept_run_m40/m52.log`、`evidence/m52_dump_sha256_ab.log`、`evidence/m52_relift_diff.txt`、
> `moe_relift/m84_dump_run.log`）只提「供 `check_moe_ref.py` 切片 / 复算」、列文件名，或说「与它同款」，
> **不含本脚本的读数**，无需处置；
> ⑥ 按匹配**片段**排除的一条：`m15_layer_loop/check_ref.py:10` 的「Ver B 逐 token 逐层同款（**144 条**）」
> 是 **M25 那份 `check_ref.py` 自己的** Ver B 计数，与本脚本的 `128+16`（两者总数同为 144 是巧合）无关。
> **本 mission 未动 `m17_moe_real/check_ref.py`**（已归档的 donor 参考：它的 `dequant` 的
> `scale_stride` = **槽宽**，与 m13 `dequant_device` 的语义一致，且它自己的设备路径同样传
> `DN_SCALE_STRIDE`）⇒ **kernel 行为不变**：`m15_moe_*.h` / `m15_layer_loop.asc` /
> `weights_manifest.txt` 一行未动，`git diff --name-status main...HEAD` 只有 `check_moe_ref.py` + 本文件
> （第 2 轮按复审 p2-1 另加 `docs/17-verification-standard.md` 的那一行数字改正）。

> **M82 增量（2026-09-27）**：本文件新增 **Part D · M82：attention 三套 cache + packed indices 的
> 布局冻结**（新运行档 `runs=kv`，新头文件 `m15_attn_kv.h` 为**唯一权威布局**）。**§7 的 ★13 已从
> 「未落实」改为「已落实（M82）」**。本 mission **没有**改 attention 段的计算（相位 A 仍是占位直通），
> 交付的是布局 + 长寿 GM + 容量自检 + 门控判据；**prolog / cache 填充的数学**是显式未完成项（见
> M82-6）。**本 mission 改了 `m15_layer_loop.asc` ⇒ 二进制 sha256 变更、M65 归档的四份日志头
> 已过期**，重基线见 **M82-7**。
>
> **M65 增量（2026-09-26）**：本文件新增 **Part C · M65：48 层链切到四相位入口（+ PLE 打断点
> + 末层全局 mixer）**，新运行档 `runs=chain`。**Part B · M58**（其上是单层四相位形态）保持不变。
> **§1-§10 描述的是「两相位」形态**（M40/M52 交付形态，仍是回归基准）。
>
> **M58 增量（2026-09-26）**：本文件新增 **Part B · M58：hc 层边界接进 per-layer kernel**
> （四相位形态 `hc(attn) → 子层段 → hc(mlp) → MoE 段`、新入口 `*_hc`、hc 段的机械抽取与验证），
> 从下面的 **# Part B · M58** 一节开始。
> 两部分的差异登记在 **M58-8/M65-7**；未完成项在 **M58-9/M65-9**。

**载体 = M25 的 48 层循环骨架**（已合 main，commit `2b9cb36`）。本 mission 把 **m13 的 MoE 段
S1-S10** 并进同一个 per-layer kernel：每层仍是一次 `__mix__(1,2)` 启动，但这一次启动里跑
**两个相位**——相位 A = 子层段（GDN S1-S7 或 attention 占位）、相位 B = MoE 段 S1-S10——
段间用 **PipeBarrier + CrossCore mode-0 barrier** 同步，**不再跨 kernel**。

| 量 | 值（真实 checkpoint 档，48 层 × 3 token） |
|---|---|
| 融合 kernel 判定项 | **440 条**（权重来源 84 + 验证 M1 348 + 验证 M2 8），**0 FAIL**（`evidence/accept_run_m40.log`） |
| 独立启动形态回归（M25 的三验证） | **725 条判定项**（A 216 + B 506 + C 3），**0 FAIL**（同一份日志） |
| 判定项合计 | **1165 条**判定项 + **109 条 guard**，`0 FAIL`，`rc=0` |
| 独立 numpy 交叉校验（MoE 段） | **128 条判定项 + 20 条非空洞性**，**0 FAIL**（**M96 在 main tip 的 dump 上重取**；M40 归档 `evidence/check_moe_ref_run.log` 的读数是 128 + **16**，差额 = M90 起每层新增的 1 条 T4 guard；M90 改写曾使本条在 main 上不可执行，见文首 M96 增量） |
| 独立 numpy 交叉校验（M25 原有） | 709 条判定项，0 FAIL（`evidence/check_ref_run.log`） |
| 融合 UB 峰值 | **227328 B / 253952 B（248KB）**，余 26624 B |
| 专家槽布局 | **紧凑 Σt_e**（起点 = `expert_offsets` 前缀和），GEMM 工作项 **(expert, mTile, nBlock)**（§5.7） |
| 每层 workspace | GDN 5,258,240 B + MoE 7,685,472 B = **12,943,712 B** |
| 段级设备耗时（msprof） | 融合 GDN 层 **204.62 µs/层** vs M25 的 141.78；MoE 段净增量 **62.84 µs/层**（§6） |
| 每 step（48 层）设备耗时 | 融合 **8.110 ms** vs M25 基线 5.134 ms（MoE 段 +2.976 ms/step；含紧凑槽的并行度退化，见 §6） |
| MoE 段权重 | 13,091,840 B/层 × 48 层 = **628.4 MB**（真实 checkpoint 4 专家切片 + 真实共享专家） |

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/build -j4
# 融合形态验收（权重来源 + M1 段间零串扰 + M2 多 token）
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt m
# 全量（M25 的独立启动形态回归 + M40 的融合形态）
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all
# M82：attention 三套 cache + packed indices 的布局冻结（容量自检 + device 探针逐字节 + 负向对照）
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt kv
# 无 checkpoint 的冒烟档（合成权重；**MoE 段在合成档会出 NaN，见 §8.6**，仅用于同步/串扰冒烟）
M15_SYNTH=1 M15_LAYERS=8 ./m15_layer_loop/build/m15_layer_loop "" m
```

> **M52 复核（副本跟随 m13/M50 重抽，2026-09-26）**：上表的读数在**重抽后**的构建上逐项复现
> （`checks=1165, guards=109, fails=0`；`check_moe_ref` 128+16；`check_ref` 709）——`check_moe_ref`
> 的那 16 是 M52 当时的归档读数（M96 用修好后的脚本复跑同一套判据是 **128+20**，见文首 M96 增量），且
> **1131 个 dump 张量的 sha256 与重抽前逐位相同** ⇒ 本轮是**纯机械搬运 + 0 处舍入点改动**。
> 新证据：`evidence/accept_run_m52.log`、`check_moe_ref_run_m52.log`、`check_ref_run_m52.log`、
> `m52_dump_sha256_ab.log`、`m52_relift_diff.txt`、`m52_lift_negative_controls.log`、`msprof_m52_*`。
> 逐项「哪些是搬运（位级不变）/ 哪些改了舍入点」见表 **§5.9**；设备侧 A/B（含抖动范围）见 **§6.2**。

---

# Part B · M58：hc 层边界接进 per-layer kernel（第一个里程碑：**单层内的两次 hc mixer 融合**）

> **下面 §1-§10 描述的是 M40/M52 的「两相位」形态**（子层段 + MoE 段）——它仍是
> `m15_layer_kernel_gdn` / `m15_layer_kernel_attn` 两个入口的交付形态，也是 M25/M40 验收链
> （A 216 / B 506 / C 3 / M1 348 / M2 8）的回归基准，**本轮未改动它的任何判据口径**。
> **M58 交付的是「四相位」形态**：`hc(attn) → 子层段 → hc(mlp) → MoE 段` **一次启动**，
> 入口是 `m15_layer_kernel_gdn_hc` / `m15_layer_kernel_attn_hc`。两者的逐项差异见 **M58-8**。

| 量 | 值（真实 checkpoint 档） |
|---|---|
| 融合形态 | **四相位一次启动**（hc(attn) → 子层段 → hc(mlp) → MoE 段）；层链数据流见 M58-1 |
| 新入口符号 | `m15_layer_kernel_gdn_hc` / `m15_layer_kernel_attn_hc`（四相位）、`m15_hc_segment_kernel`（**单边界**对照路 + combine-only 入口） |
| hc 段来源 | `m20_hyperconn/m20_hyperconn.asc` 的 **device 段 833 行**，由 `lift_hc_segment.py` 机械抽取（**6 类替换**）；device 段 sha256 `57c84acab16245ecb62e7e1db0d38fce944ded884593feae195ffb50efebbf52` 是硬断言（**M190 标注（2026-10-05）**：此值是 **device 段（833 行）**的子集 hash，不是整文件 hash —— 由 `m15_layer_loop/lift_hc_segment.py` 的锚点区间（首个顶格 `namespace {` 到入口 kernel 前最后一个 `}  // namespace`）算出；该段在 M181 r4 donor->re-lift（`a0996eb`）后已变，当前段值见该脚本的 `SRC_DEV_SHA256` = `12fe5ca26d6f76e3e138a98405953925a5b912287d285e3ee64af635d80fa9a9`。复算：`python3 m15_layer_loop/lift_hc_segment.py --check`） |
| kernel 内判定项（一次 `runs=all`，48 层 × 3 token） | **1234 条，0 FAIL** = 权重来源 84（GDN+MoE）+ **48（hc）** + A 216 + B 506 + C 3 + M1 348 + M2 8 + **H 21**（`evidence/m58_accept_run_all48.log`）；另有 **157 条 guard**（单列，见 M58-6） |
| 独立 numpy 交叉校验（hc 段，**参考 = m20 的 `reference()`**） | **判定项 64 + guard 15**，`RESULT: OK`（m=1 与 m=2 两档；m=33 档有 11 条越门限，**显式披露**见 M58-9 第 2 条） |
| UB 峰值 | **227328 B / 253952 B** —— 与两相位版**完全相同**（hc 相位 89472 B 更小，峰值取 max 不是求和） |
| L1 / L0C 峰值 | **303104 B / 524288 B**、**65536 B / 262144 B** —— 同上，未变 |
| hc 相位的 UB 静态占用 | **89472 B**（m20 的五个并列段窗，**偏移一字未动**） |
| hc 权重 | 48 层 × 2 边界 × 13,455,360 B = **1.29 GB**（checkpoint 原样字节；唯一 host 加工 = 注入权重补到 16 行、行 `[4,16)` 置零） |
| hc ws | 48 层 × 2 块 × 4,353,024 B = **417 MB**（**按层号编址**；正因为这样，`H''`/injection 跨层传递是**零拷贝**的） |
| 段级设备耗时（msprof 口径） | **同会话 A/B**（`runs=all` 一次采集内两相位与四相位符号同时出现，3 次采样）：`gdn_hc` **336.7–337.9** vs `gdn` **204.8–205.9** ⇒ **hc 段净增量 = 131.0–132.0 µs/层**；attention 层 184.2–182.7 vs 62.6–62.3 ⇒ **120.1–121.6 µs**；每 step **+6.19 ms（+76%）**。单边界符号 **81.7–83.5 µs**（W0+S1-S6 全链）/ **14.3–14.9 µs**（combine-only）。详见 M58-7（含 5 次重复的抖动范围与**三次**共享卡污染现场） |

## M58-1 四相位段序（一次启动）

```
┌─ 相位 H1（hc 边界 #1 = attn_hc）──────────────────────────────────────────┐
│  M15H::HyperConnOp：W0 injW → S1 combine → S2 grouped GemmaRMSNorm →       │
│  S3 down(+inject) GEMM → S4 silu → S5 up GEMM → S6 gate mix                │
│  入口 H(4 路) ⊕ BO_prev ⊕ IJ_prev   →   出口 BLK_attn / H' / IJ'            │
└────────────────────────────────────────────────────────────────────────────┘
        ↓ PipeBarrier<PIPE_ALL> + 全体 AIV mode-0 barrier（id 8）
┌─ 相位 A（子层段）───────────────────────────────────────────────────────────┐
│  KIND_GDN : GdnLayerChain S1-S7（M25/M40 已验收，一字未动）                 │
│  KIND_ATTN: m15_attn_passthrough_body（占位直通，待 QSA）                   │
│  入口 BLK_attn  →  出口 attn_out                                            │
└────────────────────────────────────────────────────────────────────────────┘
        ↓ PipeBarrier<PIPE_ALL> + 全体 AIV mode-0 barrier（id 9）
┌─ 相位 H2（hc 边界 #2 = mlp_hc）────────────────────────────────────────────┐
│  同一个 M15H::HyperConnOp，另一组权重、另一块 ws                            │
│  入口 H'（mode 0 时 = 层输入 H）⊕ attn_out ⊕ IJ'（= H1 的 OH[:,320:324)）    │
│  出口 BLK_mlp / **H''（层出口的多流残差）** / IJ''（= 层出口的 injection）    │
└────────────────────────────────────────────────────────────────────────────┘
        ↓ PipeBarrier<PIPE_ALL> + 全体 AIV mode-0 barrier（id 8，= M40 的相位边界 id）
┌─ 相位 B（MoE 段）───────────────────────────────────────────────────────────┐
│  M15M::MoeLayerChain S1-S10（M40 已验收，一字未动）                          │
│  入口 BLK_mlp  →  出口 yLayer = MoE 段的 S10 输出 = **下一层的 pending BO**  │
└────────────────────────────────────────────────────────────────────────────┘
```

**与 vLLM `Qwen4ExpDecoderLayer.forward` 的逐句对应**：

| vLLM | 本 kernel |
|---|---|
| `attn_hc.combine_and_mix(hidden, prev_block_output, prev_injection)` | 相位 H1（`attn_hc` 权重；`mode = MODE_MIX`（层 0）或 `MODE_COMBINE_MIX`） |
| `attn_out = attention(block_input)` | 相位 A（入口 = H1 的 `BLK`） |
| `mlp_hc.combine_and_mix(hidden, attn_out, injection)` | 相位 H2（`mlp_hc` 权重，`mode = MODE_COMBINE_MIX`；三个输入都是**层内 handoff**） |
| `mlp_out = mlp(mlp_block_input)` | 相位 B（入口 = H2 的 `BLK`，出口 = 本层的 pending BO） |

**三条相位边界的同步为什么是这两条（与 §1.1 的论证同款，M58 只是把它用了三次）**：
① `PipeBarrier<PIPE_ALL>`（核内 6 条 pipe 全 drain）——三段共址叠放 UB/L1/L0C 与 BufferID；
② 全体 AIV 的 mode-0 barrier（set 挂生产 pipe `PIPE_MTE3`、wait 挂最窄的 `PIPE_MTE2`）——
每个相位的出口都是全体 AIV 协作写、下一位相用另一套行切分读，跨核可见性只能靠 CrossCore。
**AIC 侧不需要相位边界同步**（各段的 A 操作数由 AIV 产出、走各段自己的 mode-2 交接），
核内只留 ①。

**层内 handoff 不走 host（本 mission 的关键设计）**：H2 的三个输入直接由上一位相落在 GM 的
张量充当 ——

| H2 的输入 | 来源 | 形态 |
|---|---|---|
| `hIn`（多流残差） | H1 的 `WS_HCP`（`mode 0` 时 = 层输入 `H`，因为该档不写 H'） | 同一块 ws 的偏移 |
| `bo`（pending block output） | 子层段的出口缓冲 `hcAttnOut` | 独立 GM 平面 |
| `ij`（injection logits） | H1 的 `OH[:, 320:324)`，**行距 = `OH_W` = 336** | 由 `lift_hc_segment.py` 的替换类 4 支持 |

`ij` 那一条是本轮新加的机制：m20 是独立 kernel，层内那次 IJ 提取由 host 物化成 `[m,16]` 平面；
融合后必须**在 kernel 内按可变行距直读**，故 `HcPtrs` 增加 `ijStride`，并把 `InjwStage` 的整块
`DataCopy` 换成**按行 32B 搬运**（两种行距都是 32B 的整数倍 ⇒ 不引入 `DataCopyPad`）。

## M58-2 资源表：四相位同址叠放（本 mission 的「核心难点」，结论是**峰值未变**）

`m15_layer_resources.h` 仍是唯一登记处；`m15_hc_resources.h` 是 hc 段自己的表。四个相位的
UB 区**都从 0 起算**（相位互斥 + 每个交界有 `PipeBarrier<PIPE_ALL>`），因此：

### 2.1 UB

| 相位 | 窗 | 起点 | 峰值结束地址 | 内容 |
|---|---|---|---|---|
| **H1 / H2（hc）** | PERSIST+staging+五个并列段窗 | @0 | **89472** | 声明窗 = `UB_NORMF32`(40960)+`UB_PC_STAGE`(20480)+W0/W1/W2/W3/W4 五窗；**实测被写到的只有 `UB_PC_END=61440` 之后的部分**（见下面的口径说明） |
| A（GDN） | PERSIST + 四段窗顺排 | @0 | **227328** | 见 §2.1（M40 原表） |
| A（attention） | 与 GDN 同址 | @0 | ≤ 64256 | 见 §2.1 |
| B（MoE） | PERSIST + 段窗 | @0 | **193600** | 见 §2.1 |
| **四相位峰值** | | | **227328 ≤ 253952（248KB）** | 余 **26624 B**（= 两相位版的余量，**未变**） |

**「三相位同址叠放」的合法性**（编译期只能抓一部分，故逐条登记）：
* **口径说明（必读，避免把「声明窗」读成「实际占用」）**：hc 段的声明窗跨度 89472 B 里有
  `UB_NORMF32`(40960) + `UB_PC_STAGE`(20480) 两段 —— 它们是 **m20 的遗留常量**：当前
  hc 段实现**不写也不读**这两段（`grep -n "UB_NORMF32\|UB_PC_STAGE\|UB_PERSIST\|BUF_AIV_GJ"
  m15_hc_layer.h` → **0 命中**；`NormStage` 是按组直接读 **bf16** 的 `hcNormGm` 再在寄存器里
  Cast）。**实测被写到的 UB 只有 `[61440, 89472)`（= 五个段窗）**，即 28032 B。本表按**声明窗**
  登记（保守、且与 m20 的表对齐）；**峰值结论不受影响**（227328 来自相位 A，与 hc 无关）。
  这个遗留常量是 m20 侧的文档与实现不一致（m20 README §3/§7.3 仍称做 fp32 预转），
  **已在 `.tower` 里作为 finding 上报、不在本 mission 的 scope 内修**（m20_hyperconn 属 M56）。
* 每个相位的 UB 区都从 0 起算 ⇒ hc 的 `[0,89472)` 与 GDN 的 PERSIST `[0, UB_SEG)`、
  MoE 的 `[0, UB_VEC)` **物理重叠**；安全性来自「**没有任何张量跨相位存活**」+ 交界的
  `PipeBarrier<PIPE_ALL>`（每核 6 条 pipe 全 drain）。
* `static_assert(UB_PEAK_FUSED <= UB_TOTAL_BYTES)` + `static_assert(M15H::UB_PEAK == M15H::UB_BYTES_USED)`
  是编译期兜底（抓「hc 表自己的窗越界」这类错）。
* **没有为了绕过预算而把任何段改成 memory-based API 或加 TPipe/TBuf/TQue**：hc 段一字未动、
  GDN/MoE 段一字未动。**可复核的判据（剥掉行尾注释后计数，与匹配器同源）**：
  `for f in m15_layer_loop/m15_hc_layer.h m15_layer_loop/m15_gdn_layer.h m15_layer_loop/m15_moe_layer.h;
   do echo "$f: $(sed 's|//.*||' $f | grep -c 'TPipe\|TBuf\|TQue\|AllocTensor')"; done` → **三个文件都是 0**。
  （`m15_hc_layer.h` 全文里 `TPipe`/`TQue` 各出现 1 次，都在**注释**里 —— 即上面那条 m20 的
  「被禁的仅 TPipe/TQue wrapper」说明；**不剥注释**直接 grep 会读到 1，这不是资源管理函数的用法。）

### 2.2 L1 / L0C

| 资源 | 相位 H1/H2 | 相位 A | 相位 B | 四相位峰值 | 硬限 |
|---|---|---|---|---|---|
| L1（A/B 区） | 303104 B | 303104 B | 296960 B | **303104** | 524288 B |
| L0C（CO1） | 40960 B | 40960 B | 65536 B | **65536** | 262144 B |
| L0A/L0B | 8KB+20KB | 8KB+20KB | 4KB+16KB | 各 ≤32KB | 64KB/64KB |

### 2.3 GM 平面 / 权重槽

| 区 | 内容 | 规模 |
|---|---|---|
| `ws`（每层复用一块） | GDN 段 5,258,240 B + MoE 段 7,685,472 B | 12,943,712 B（**与两相位版相同**） |
| `hcWs[L]`（**按层号编址**） | 每层两块 hc ws（边界 #1 / #2） | 48 × 8,706,048 B = **417 MB** |
| hc 权重区 | 每层两个边界（attn_hc / mlp_hc） | 48 × 26,910,720 B = **1.29 GB** |
| MoE 权重区 | 48 × 13,091,840 B（未变） | 628.4 MB |
| GDN 权重区 / 状态区 | 未变 | 4174 MB / 115.5 MB |

**hc 的 ws 为什么必须按层编址**（与 GDN/MoE 的单块 ws 处置不同）：hc 边界有两个**跨层存活**的
输出 —— `H'`（= 下一层的 `hcH`）与 `OH[:,320:324)`（= 下一层的 `hcIj`）。若与每层复用的单块 ws
共用，它们会在下一层启动时被覆盖。按层编址的代价是 417 MB，收益是**跨层 handoff 零拷贝**
（下一层直接把上一层 hc1 ws 的 `WS_HCP` / `WS_OH+640` 当输入，不需要抽取 stage 或 host 搬运）。
**本 milestone 只交付单层的四相位形态**：48 层链的这条接线**未接**（M58-9 未完成项第 1 条）。

## M58-3 flagId / BufferID 登记（与两相位版的差异只有 hc 段那一格）

`m15_layer_resources.h` §4 的 `FLAG_SEQ[]` 仍是唯一权威，**按执行序**把四相位的同步点排开，
`FlagSeqAdjacentOk()` / `FlagMaxUse()` 两个 `static_assert` 逐对核对。分配（每个 (核型, mode) 子空间）：

| 段 | AIV mode0 | AIC mode0 | mode2（AIC ↔ 配对 AIV） |
|---|---|---|---|
| **hc（H1 与 H2 共用同一组）** | **12/13/14**（原 m20 是 0/1/2） | **0/1/2/3**（原 m20 是 4/5/6/7） | **8/9/10/11**（与 m20 相同） |
| GDN | 10 → 8 → 9 → 11 | 12,13,14,15 | 4,5,6,7 |
| MoE | 12 → 13 → 14 → 15 → 12（4 槽轮转） | 8,9,10,11（4 槽轮转） | 0,1,2,3（4 槽轮转） |
| 相位边界（全体 AIV mode0） | **H1→A = 8、A→H2 = 9、H2→B = 8**（= M40 的 `FLAG_L0_BOUND_AIV`） | —（AIC 无依赖） | — |

**为什么 hc 的 flagId 要重编**：m20 原来把 AIV mode0 分到 0/1/2、AIC mode0 分到 4/5/6/7，而融合后
**MoE 段的 mode2 ring 正好是 0-3、GDN 段的 mode2 正好是 4-7**。跨 mode 复用本可援引 docs/05 §2
那条较弱的前提（「前一 mode 全部 drain」），但本轮选择**完全错开**：hc 用 AIV mode0 = 12/13/14、
AIC mode0 = 0/1/2/3、mode2 = 8/9/10/11（后两者恰好落在 GDN/MoE 用不到的空档里）。于是
**全程不存在「同一 flagId 在同一核上跨模式复用」**，不必援引那条前提。

相邻性与用量的**执行序见证**（KIND_GDN 层）：
`H1 的 12/13/14 → 边界 8 → GDN 的 10/8/9/11 → 边界 9 → H2 的 12/13/14 → 边界 8 → MoE 的 12/13/14/15/12`
（KIND_ATTN 层没有 GDN 那四条，序变成 `…14 → 8 → 9 → 12…`；**两条都逐对满足「相邻必不同」**）。
用量上界：`FlagMaxUse()` 实测 = **4**（AIV mode0 的 id 12 被 hc H1、hc H2、MoE S1 后、MoE S9b 后
各用一次），`static_assert(FlagMaxUse() <= 4)`；硬件 4bit 计数器上限 15。

**BufferID**（`m15_layer_resources.h` §5）：hc 段用 AIV `0..16`（17 个核内 token）与 AIC `0..6`，
与 GDN/MoE **同编号重叠**（相位互斥 + 交界全 drain）；峰值同时占用数保持 **AIC 7 / AIV 17 ≤ 28**
（新增 `static_assert(M15H::BUF_AIV_IDS <= BUF_AIV_PEAK)` 等两条把 hc 的用量也纳入峰值断言）。
**已知弱项不变**：`BUF_AIC_PEAK/BUF_AIV_PEAK` 仍是字面量，抓不住 `BUF_TABLE[]` 填错（§2.5 的口径说明同样适用于 hc 那一行）。

## M58-4 hc 权重落位（与 m20 的抄改关系）

**布局与 checkpoint 逐字节一致**（m20 的「权重表示一行不转」契约）：`hc_norm`、`input_mix_weight_down`、
`input_mix_weight_up`、`block_inject_weight` 原样消费，**零离线转换**。每层的两个边界各占一个
`HC_W_STRIDE = 13,455,360 B` 的槽（`HW_NORM_OFF=0` / `HW_WDOWN_OFF=20480` / `HW_WINJ_OFF=6,574,080` /
`HW_WUP_OFF=6,901,760`，全部 512B 对齐），一层两个槽 = `HC_LAYER_W_STRIDE = 26,910,720 B`。

**唯一 host 侧加工 = 注入权重的补零**：checkpoint 的 `block_inject_weight` 只有 `INJ_N=4` 行，
而 cube 的 N 分形以 16 行为单位 ⇒ 槽里必须是 **16 行**、且**行 `[4,16)` 全零**（否则 L0C 的列
`[4,16)` 取自 L1 残留，`OH` 的 padding 列 `[324,336)` 就不确定，逐字节判据会跨运行抖动）。
这与 m20 §5.1 的 host 契约**逐字相同**；本轮把它写成**设备侧可复核的 guard**
（`Hc.injZero.L%02u`：只看设备槽的字节，不看 manifest；**每层 1 条 guard、不计入判定项**——
review r1 的 P2-1b 指出旧版 README 把它叫「判定项」而代码是 `H_Guard`，两者对不上，已改齐）。

**manifest**（`slice_layer_manifest.py` → `weights_manifest.txt`）：每层新增 **8 行**
（`attn_hc_{norm,down,up,inj}` + `mlp_hc_{norm,down,up,inj}`），48 层 = **384 行**，
manifest 由 **900 → 1284 条张量行**（**口径**：与本 README 其它处一致，只数 `tensor ...` 行，
不含 4 行文件头；`wc -l` 因此是 904 → 1288。`diff` 的 `+384 / -0` 与之相符）；
**原有 900 条一字未改**（纯增量，`evidence/m58_manifest_diff.txt`
给出删除行数 = 0）。角色名与 `m15_layer_resources.h` §1b 的 `HW_ROLE_*` 一一对应；`*_inj` 行记录的
是 **checkpoint 侧的 4 行**（= host 要 pread 的字节数），补零由 host 做。

**与 m20 侧权重切分的抄改关系**：m20 的 host 把 4 件权重当**运行期入参**逐 case 生成/取用
（它的 case 集是合成的 + `real_m1`）；m15 侧改为**按层号的静态槽表 + manifest 驱动 pread**，与
GDN/MoE 权重的处置同款（`H_LoadHcW` 只做「取字节 + 补零」，`H_HcWPtrsOf` 算指针）。

## M58-5 三档 hc 语义在融合形态下都成立 + combine-only 通路的处置

| 档 | 语义 | 融合形态下的入口 | 对拍 |
|---|---|---|---|
| ① `MODE_MIX`（层 0 的 attn 边界，`prev_block_output is None`） | mix-only：不做 combine（`H' ≡ H`），只做 norm→…→BLK | 四相位入口的 `hcAttnMode`（层 0 传 0） | kernel 内 `H.finite`/`H.nontrivial` + numpy：`A1.H'` **显式 SKIPPED**（该档不产出 H'），其余 6 个张量逐段对拍 |
| ② `MODE_COMBINE_MIX`（层内 attn/mlp 两个边界） | combine + mix | 层 >0 的 `hcAttnMode` / 恒定的 `hcMlpMode` | `A1.*` / `A2.*` / `B整层*` 全部逐段与整层对拍 |
| ③ `MODE_FINAL_MIX`（全局 `hyper_connection_mixer`，`use_combine=false`） | 同 ②，但 S3 只取 lora（2 个 N-tile）、不产出 injection | **hc 段支持该档**（`nDtiles = LOWRANK/BASE_N`），但**本 mission 的层内两边界不用它** —— 全局 mixer 是层循环之外的调用点 | **未接**：是**显式披露项**（全局 mixer 属于「48 层链之外」的接入点，见 M58-9 第 3 条） |
| ④ **`MODE_COMBINE_ONLY`（本轮新增）** | combine-only：只跑 W0 + S1（`H' = bf16(H + BO·injW)`），不做 norm、不做 mix；**AIC 不跑任何 GEMM** | **`m15_hc_segment_kernel` 的 `mode = M15H::MODE_COMBINE_ONLY`**（`lift_hc_segment.py` 替换类 5） | ① kernel 内**一条合并判定项** `H.conly`：`H'` 与 combine_and_mix 档的 **逐字节相同** **且** `BLK` 区保持预填 `0xCD`（**确实没写** —— 这半边是判别性的）；② numpy `C combine-only H'<-H,BO,IJ` 与 m20 参考对拍（m=1/m=2 档逐位=1.0000） |

**④ 的接线状态（必须显式披露，不得含糊）**：**入口可用、已与参考对拍，但未接进 PLE 流程**。
理由与后果：`docs/14 §4.1` / `V_N:model.py:293-297` 里 PLE 挂在 0-based 层 1，PLE 直接加到多流
残差态 ⇒ 层 1 的 attn 边界要把 combine 与 mix **拆成两步**（中间插 PLE）。本轮做的是
「**给出口 + 证明它算得对**」（`m15_hc_segment_kernel` 可单独以该档启动，device 侧与参考对拍
通过）；**没有做**的是「让四相位入口在层 1 走 combine-only → PLE → mix-only 这条三段式」
（PLE 本身也尚未实现）。m20 的三档 mode 都不提供本档，这条缺口由本 mission 在 m15 侧补上，
但**只是入口，不是完整通路**。

## M58-6 数值验收：档位（docs/17）、ε 来源、逐元素检查

**档位与理由**（docs/17 §1.1；因为 hc 段的 device 代码与 m20 **逐字节相同**（`m58_lift_diff.txt`
可核：除登记过的 6 类替换外零差异），所以 m20 §7.4 的 ε 推导**原样成立、无需重推**）：

| 张量 | 档 | 触发条件 / ε 各项来源（m20 §7.4 的同一份推导） |
|---|---|---|
| `H'`（`mode 0` 档） | **T1/T2** | `IJ ≡ 0 ⇒ injW = 2·σ(0) = 1.0` 精确；H/BO 为小整数时 `fp32(H+BO)` 与 double 同值 ⇒ **逐位** |
| `H'`（其他档） | T2′ | 一次乘 + 一次加；`injW` 由设备 fp32 sigmoid 得（相对误差 ≤5.2e-8）⇒ 良态元素 ≤1 ulp |
| `XN` | **T3** | NR-rsqrt 相对误差（`~2^-22`，取 1e-7）+ 2 次 fp32 舍入（`2·2^-24`）+ 上游 H′ 的 ≤2 bf16 ulp 一阶映射（`2·2^-8`）；`Σ\|terms\| = \|t\| + \|t·w\| + \|xn\|` |
| `lora`（= `OH[:,0:320]`） | **T3** | mmad 累加 `K/8` 次 fp32 舍入（每 16 元素分形 2 次 ⇒ `n = 2·K/16 = K/8`）+ 上游 XN 的 ≤2 bf16 ulp（`2·2^-8`）；`Σ_k \|xn_k·w_k\|`（**K = 10240**） |
| `injection`（= `OH[:,320:324]`） | **T3** | 同上（K = 10240） |
| `ls` | **T3** | VF `Exp` 精度（≤1.6 ulp ⇒ `1.6·2^-24`）+ 6 次 fp32 舍入（`6·2^-24`）+ 上游 lora 的 ≤1 bf16 ulp 经 silu 导数传播（`1·2^-8`） |
| `gate` | **T3** | mmad 累加 `(K/8)·2^-24` + 上游 LS 的 ≤2 bf16 ulp（`2·2^-8`）；**K = 320** |
| `blk` | **T3** | 4 路 VF `Exp`（`4×1.6·2^-24`）+ 4 项累加与除法 8 次 fp32 舍入（`12·2^-24`）+ 上游 gate/xn 各 ≤2 bf16 ulp；`Σ_s (…) / HC` |
| `rstd` / `injw`（fp32 证据张量） | 相对误差 | ≤1e-4 / ≤1e-6（m20 的同一 tol） |
| FTZ 子条款 | `\|ref\| < 2^-116` 时 T3 界不适用，改要求 `\|out\| ≤ 2^-116` | 与 m20 同一处置（`ls` 的次正规元素数据相关） |

**逐元素检查的实现**：本 mission 的 host 侧判据**直接复用 m20 的 `judge()`**
（`check_hc_ref.py` `import check_ref as m20` 后调用 `m20.judge`），它逐元素算 bf16 位模式上的
ulp 距离，并同时卡三条：`张量尺度归一化绝对误差 ≤1e-2` **且** `良态元素 ulp ≤2` **且**
`良态元素逐位一致率 ≥99%`。这是**一套比 T3 的 ε 界更严的独立保守门限**（它不做 ε 推导，而是
直接卡 ulp 与逐位率）——**两者互为独立见证，本脚本更严，因此不会产生假 PASS**（这个「判据角色
的自我交代」与 m20 的 `check_ref.py` 同款）。

**逐元素读数（m=1 档，`evidence/m58_check_hc_ref_run.log`）**：**53 条由 `m20.judge()` 直接
产出**的判定项里，逐位率中位数 = **1.0000**（多数张量 100% 逐位一致，`ulpMax=0`）；最低的一条是
H1 的 `gate`（逐位 0.9999 / 良态逐位 1.0000 / ulpMax 0）。也就是说 **ε 界根本没有被触及** ——
本档的偏差在 m20 的门限内是「逐位一致」量级。

**非空洞性 / 可区分性 / 确定性三类证据**（task 要求的三类）：

**验证 H 的计数口径（review r1 的 P2-1 修正后）**：**每层 7 条判定项、0 条 guard** ——
判定项 = `H.finite` / `H.nontrivial` / `H.h1ws` / `H.h2ws` / **`H.conly`** / `H.det_h1` / `H.det_h2`；
其中 `H.conly` 是**一条合并判定项、两个面**（`H'` 与 combine_and_mix 档逐字节相同 **且**
`BLK` 区未被写出 —— 后者是判别性的那一半：若 mode 3 没被兑现而走了完整链，`H'` 仍会相同，
只有「BLK 是否被写出」能区分），因此计数只有一处、不存在「同一件事进两个桶」。
`M15_HC_LAYERS=0,1,3` ⇒ **H 21 条判定项 + 0 条 guard**。
**全 `runs=all` 的 guard 分解（157 条，与打印行同源、逐项相加 == `C.guards`）**：sha256 工具自检
**1** + hc 注入行 `[4,16)` 全零 **48** + 参考链锚点非空 **108**（36 GDN 层 × 3 token）+ 其余 **0**
= **157**；汇总行会把这四类与「其余」一起打印（新出现的 guard 不会被静默吸收进任何一类）。

> **口径时点（M82 追加，防"半刷新"）**：上面这个 **157** 是 **M58 那一刻**的读数。
> M65 起同一条 `runs=all` 的 guard 变成 **160**（多 3 条 M65 链 guard），M82 起变成 **290**
> （再多 **130** 条 M82 的 `Kv.*` guard，按函数分组实测 = **容量独立复算 9**（`H_KvCapacityCheck`）
> + **对齐/层槽编址 115**（`H_KvAlignAndSlotCheck`：32B 对齐 11 + ring 第 1/2/3 行 + packed 行 1 +
> head 平面≠token 步长 + 层循环 36×1 + 12×5 + 3） + **负向对照 6**（off-by-one 4 + paged 2）
> = 130）。
> **现况以程序打印行与 `evidence/m82_accept_run_all.log` 为准**；本节其余读数（21 条 H 判定项等）
> 未变。

| 类 | 证据 | 位置 |
|---|---|---|
| **非空洞性**（改输入必改输出） | ① kernel 内 `H.nontrivial`（层出口 H'' ≠ 层输入 H，逐字节，逐层）；② `H.finite`（H'' 无 NaN/Inf）；③ numpy 侧 4 条 guard/层（H''/BLK 非常量、H''≠H、两个边界 blk 互异）+ 每条判据自带「输入整张量 ×1.5 ⇒ 参考任一输出必须显著变化」的 m20 探针（`C 段`） | `m58_accept_run_all48.log` / `m58_check_hc_ref_run.log` |
| **索引/槽位可区分** | ① numpy 侧跨层 `blk` 指纹两两互异（3 层 → 3 条 guard）；② kernel 内 `Hc.ws.L%02u` 逐层 memcmp（48 条：层号↔槽位错位会立刻红）；③ 两个边界的 `blk` 互异（attn 边界 ≠ mlp 边界） | 同上 |
| **多次运行确定性（逐位）** | ① kernel 内 `H.det_h1` / `H.det_h2`（同一二进制重复运行，两个边界的整块 ws 逐字节相同；**注意 GDN 的 conv/ssm 状态必须复位**，否则「输入本就不同」而不是不确定）；② **两个独立进程**各 dump 一次，93 个 `hc_*.bin` 的 sha256 **全等**（`evidence/m58_dump_sha256.txt`） | 同上 + `m58_dump_sha256.txt` |

**m 档覆盖**：`M15_HC_M` 支持 1..64。实测
* **m=1**（decode 交付档）：kernel 内 H 段 **21/21 PASS**（7 条判定项/层 × 3 层 = 21，**无 guard**）；numpy **64 判定项 + 15 guard 全 OK**；
* **m=2**：kernel 内 21/21 PASS；numpy **64 + 15 全 OK**；
* **m=33**：kernel 内 21/21 PASS（含 `H.h1ws`/`H.h2ws` 两个边界的**逐字节** A/B）；numpy **11/64 条越过 m20 的保守门限**（`maxulp=3` vs ≤2、良态逐位率最低 0.9891 vs ≥0.99）——**如实披露、未归因**，见 M58-9 第 2 条。

## M58-7 段级设备侧分解（msprof；**host 墙钟不作为证据**）

按 §6.1 的实测结论（同一二进制 host 墙钟 24× 抖动），本节**只用设备侧 msprof 口径**，
并先给**抖动范围**再谈差异。

**主证据：同一次会话内的 A/B（`runs=all` 各采样里两相位与四相位符号**同时**出现；
3 次采样，`evidence/m58_msprof_all{1,2,3}.csv`）**——这是 review r1 的 P2-2 之后**重做**的取证：
每条命令都记进日志，且两相位/四相位来自**同一次采集**（不再跨会话）：

| 符号 | Count | Min（3 次） | **Avg（3 次）** | Max（3 次） |
|---|---|---|---|---|
| `m15_layer_kernel_gdn_hc`（**四相位**） | 18 | 297.31 / 297.99 / 308.54 | **336.67 / 336.79 / 337.92** | 367.59 / 367.68 / 368.65 |
| `m15_layer_kernel_gdn`（两相位） | 360 | 193.37 / 197.58 / 197.81 | **205.67 / 204.77 / 205.92** | 212.84 / 213.26 / 215.63 |
| `m15_layer_kernel_attn_hc`（**四相位**） | 6 | 145.93 / 145.36 / 145.02 | **184.21 / 182.71 / 182.67** | 222.24 / 222.83 / 221.89 |
| `m15_layer_kernel_attn`（两相位） | 120 | 60.37 / 60.25 / 60.64 | **62.59 / 62.25 / 62.56** | 68.17 / 68.03 / 68.82 |
| `m15_hc_segment_kernel`（单边界，两种档共用符号） | 37 | 14.49 / 14.54 / 14.29 | **46.90 / 47.20 / 46.60** | 83.05 / 82.12 / 82.99 |
| `m15_gdn_layer_kernel`（子层段独立启动） | 1162 | 108.40 / 107.15 / 107.81 | **140.27 / 139.78 / 141.05** | 150.63 / 151.91 / 153.60 |
| `m15_moe_segment_kernel`（MoE 段独立启动） | 192 | 43.21 / 42.46 / 43.11 | **61.74 / 60.97 / 61.69** | 66.44 / 65.89 / 66.64 |
| `m15_attn_placeholder_kernel`（占位直通） | 384 | 2.01 / 1.93 / 2.00 | **2.48 / 2.42 / 2.53** | 3.08 / 2.88 / 3.29 |

**抖动范围（hc 符号单独 5 次重复，同一命令、命令逐条记入 `evidence/m58_msprof_hc.log` §2）**：
`m15_layer_kernel_gdn_hc` 的 Avg = **334.37 / 335.66 / 336.54 / 335.06 / 332.94 µs**（Count 均为 12，
**5 次极差 1.08%**）、Min 列 = 296.61 / 299.51 / 302.76 / 295.48 / 297.45 ⇒ 与上面同会话的 3 次合计
**8 个采样，Avg 全部落在 332.9–337.9 µs（极差 1.5%）** ⇒ **同一二进制同一命令下该符号的重复性 ≤1.5%**。

**段级结论（含口径说明）**：
* **hc 段净增量 = 131.0 / 132.0 / 132.0 µs/层**（GDN 层，三次同会话采样；Avg 口径），
  attention 层 **121.6 / 120.5 / 120.1 µs**。**Δ 的 3 次极差 ≈0.8%**，而两个符号各自的组内极差
  分别 0.37%（hc）与 0.56%（两相位）⇒ **Δ 比噪声大两个数量级**，结论不受测量噪声影响。
  **Min 口径**给出 Δ = 103.9 / 100.4 / 110.7 µs（min 是单次极值、本身抖 3.5%）⇒ 两个统计量都把
  增量定在 **100–132 µs/层**，本 README 的结论统一写作 **≈131 µs/层（Avg 口径）**。
* **单边界符号** `m15_hc_segment_kernel` 的 min/max 分别对应**两种档**（这正是 host 侧合并判定项
  `H.conly` 与 A/B 对照路共用一个符号的结果）：**max 列 81.7-83.5 µs = W0+S1-S6 全链**
  （`MODE_MIX`/`MODE_COMBINE_MIX`）、**min 列 14.3-14.9 µs = combine-only 档**（只 W0+S1）。
  ⇒ 单边界的完整链要 80+ µs，而融合里两条边界只加了 131 µs（< 2×82），说明**融合确实省掉了
  一次启动/收尾**（每条边界摊薄后 ≈65 µs）。
* **每 step（48 层，3:1）**：同一会话的 3 次采样取均值 —— 两相位基线 = 36×205.45 + 12×62.47 =
  **8.15 ms/step**（与 M40 归档的 8.110 一致，差 0.4%），四相位 = 36×337.13 + 12×183.20 =
  **14.33 ms/step** ⇒ hc 段一项 **+6.19 ms/step（+76%）**。**这不是本 mission 的验收项**（性能只作设计输入），
  但它是一条重要设计输入：hc 边界是 K=10240 的 down GEMM + K=320/N=10240 的 up GEMM，
  而 m=1 时 S3 只填 3/28 个 AIC、S2 只有 `m·HC = 4` 个 item ⇒ **融合没有、也不可能改变这一点**
  （hc 段一字未动）——即 m20 §9.2/§9.3 的两条并行度问题在融合形态下**原样存在**（见 M58-9 第 4 条）。
* **共享卡污染的三个现场（已按 tower 规则剔除，并如实记录）**：① 第一次做 5 次重复时，第 1 次里
  `m15_layer_kernel_gdn_hc` 有**一次启动**读到 **88.07 ms**（该次的 Avg 因此虚高到 7.34 ms）——
  但同一批的 **Min 列仍是 309.2 µs**；② 单独的 `runs=p` 采样里 `m15_layer_kernel_gdn` 有一次启动
  读到 **72.7 s**（Avg 虚高到 673.6 ms），同批的 **Min 列仍是 199.5 µs**；③ 重做的 5 次里，
  **`m15_hc_segment_kernel`**（上一段那个符号）在第 2 次被击中：Max 列 2808.5 µs、Avg 158.7 µs，
  而同批的 **Min 列仍是 14.87 µs**，其余 4 次的 Avg 都是 46.4-47.1 µs。
  ⇒ **被放大的是哪个符号、哪一次启动是随机的**（与 §6.1 / M40 r2 的机制一致）；**Avg 会被污染、
  Min 相对稳**，所以本节同时给两个口径。
* **host 侧数字一律不引用**。

## M58-8 与 m20 / m13 / m14 的差异表（含抄改关系登记）

| 文件 | 来源 | 差异（可复核） |
|---|---|---|
| `m15_hc_layer.h`（新，910 行） | `m20_hyperconn/m20_hyperconn.asc` 的 **device 段**（内容锚点：首个顶格 `namespace {` 到入口 kernel 前最后一个 `}  // namespace`，**不写行号**） | **`lift_hc_segment.py` 机械生成，6 类替换**：① 两处匿名 `namespace {` → `namespace M15H {`；② 删 `using namespace M20;`；③ `m20_resources.h` → `m15_hc_resources.h` + `M20`→`M15H`；④ **`HcPtrs.ijStride` + `InjwStage` 的 IJ 源改按行 32B 搬运**（层内 handoff）；⑤ **`MODE_COMBINE_ONLY`（第 4 档）在 `ProcessAiv`/`ProcessAic` 段首各加一个分支**；⑥ 文件头 PROLOGUE。**段序、同步表、tile 常量、UB/L1/L0 偏移、全部向量/矩阵/搬运语句一字未动**。逐字 diff：`evidence/m58_lift_diff.txt`（+91/−… 行，全部落在上述 6 类） |
| `m15_hc_resources.h`（新） | `m20_hyperconn/m20_resources.h` | **人工改抄**（与 `m15_moe_resources.h` 对 `m13_resources.h` 同款）：namespace + §3 的 flagId 重编（12/13/14、0/1/2/3、8/9/10/11）+ 新增 `MODE_COMBINE_ONLY`/`HC_STAGE_LIMIT`/`L1_PEAK`/`L0C_PEAK`/`UB_PEAK`/`BUF_*_IDS` 等登记常量 + `HC_IJ_STRIDE_*`。**UB/L1/L0/ws 的偏移与尺寸一字未改** |
| `lift_hc_segment.py`（新） | 新写（**结构照抄 `lift_moe_segment.py`**） | 抽取 + `--check` + **上游不变量断言**（自动推导「只在 `__VEC_SCOPE__` 内调用」的 helper 集合；断言段内 0 处 `Sort32`/`MrgSort`/经典 `Extract(`；断言两处 bring-up 披露与 docs/05 §6.2 依据随代码搬运；**device 段 sha256 硬断言**）。6 条负向对照见 `evidence/m58_neg_controls.log` |
| `m15_layer_resources.h` | 本仓（M40 版） | **四相位化**：hc 的三段 ws 表（**按层号编址**）、hc 权重槽与 8 个角色名、四相位 UB/L1/L0C 峰值（`UB_PEAK_PHASE_H` 等）、`FLAG_SEQ[]` 新增 hc 两轮与三条相位边界、`BUF_TABLE[]` 新增 hc 行、入口符号表新增 3 个 |
| `m15_layer_kernel.h` | 本仓（M40 版） | `M15L_FusedBody<KIND, HC>` 模板化（`HC=false` 的两相位路径**逐语句不变**）；`M15L_PhaseBoundaryAiv<FLAG>()` 模板化（三条边界复用同一实现）；新增 `M15L_FillHcPtrs()`、`M15L_LAYER_HC_ARGS_*` 宏、`m15_layer_kernel_{gdn,attn}_hc`、`m15_hc_segment_kernel` |
| `m15_layer_loop.asc` | 本仓（M40 版） | 新增 include（`m15_hc_layer.h` + `m15_hc_host.h`）、`Ctx` 的 6 个 hc 缓冲 + `hcArena` + 3 个计数器、`H_Alloc`/`H_LoadWeights`/`H_Cleanup` 的 hc 分支、`runs=h` 分派、`M15_HC_LAYERS`/`M15_HC_M` 两个 env、分账行与 dump 清单。**三处计数公式**（`phaseA/B/C`）加了 `- C.phaseWh` 项（见 M58-9 第 5 条） |
| `m15_hc_host.h`（新） | 新写 | hc 权重装载/补零/槽装配/来源判据、四相位与单边界启动器、**验证 H**（**7 条判定项 / 层，无 guard**，见 M58-6）、hc dump + `hc_layout.txt` |
| `check_hc_ref.py`（新） | 新写（**复用 m20 的参考**） | `import check_ref as m20` 后用它的 `reference()` 与 `judge()`（同一份门限与计数代码）做 **A 逐段 / B 整层 / C combine-only** 三档判据 + 非空洞性 guard；三态退出码 + 覆盖计数；负向对照 4 组 |
| `slice_layer_manifest.py` / `weights_manifest.txt` | 本仓（M25/M40 版） | 新增 `HC_ROLES`（8 角色 × 48 层 = 384 条张量行）；manifest **900 → 1284 条张量行**（`wc -l` 904 → 1288），**diff 删除 0 行**（`evidence/m58_manifest_diff.txt`） |
| `m15_gdn_layer.h` / `m15_moe_layer.h` / `m15_attn_layer.h` / `m15_loop_layout.h` / `check_ref.py` / `check_moe_ref.py` / `parse_msprof.py` | 本仓 | **未改动**（`runs=all` 的 A/B/C/M1/M2 读数与 M40 归档逐项一致：216/506/3/348/8） |

**为什么 hc 段可以用「机械抽取 + 6 类替换」而不是重写**：用户的要求是「所有功能都优先从现有的
代码库中找代码来抄和改」；m20 的 hc 段是本仓**已验收**（kernel 内 93 判定项 + numpy 204 项）
的同一段实现，机械抽取能同时拿到三样东西：位级可复现（`--check`）、差异可枚举（6 类）、
上游变更可发现（device 段 sha256 断言）。

## M58-9 已知限制与本轮未完成项（**显式清单**）

1. **48 层链没有切到四相位形态（本轮最大的未完成项）**。交付形态是 `*_hc` 两个新入口；
   `runs=all` 的 48 层循环仍走**两相位**入口（M25/M40 的验收链没有 hc 的参考模型，
   换主路径就要把那 700+ 条判据重做）。**结构上已经铺好**：hc ws 按层号编址 ⇒ 下一层的
   `hcH`/`hcIj` 可以直接指向上一层 hc1 ws 的 `WS_HCP` / `WS_OH+640`（`ijStride = OH_W`），
   **零拷贝**；缺的是「层循环里把 H/BO/IJ 三个平面接上 + 把 48 层链的参考模型补上」。
   本轮**已交付**的是「单层四相位 + 它的完整验证」。
2. **m=33 档上有 11/64 条 numpy 判定项越过 m20 的保守门限 —— 未归因**。
   读数：`maxulp` 达 3（门限 ≤2）、良态逐位率最低 0.9891（门限 ≥0.99）、逐位率最低 0.9690，
   集中在 `A2.*`（边界 #2）与 `B整层 blk_mlp/injection`；**m=1 与 m=2 档全部逐位一致**。
   已排除的归因：**不是融合引起的** —— 决定性实验（review r1 要求）：**非融合的单边界路
   `m15_hc_segment_kernel` 在同一 m=33 输入下与融合路的 `H.h1ws`/`H.h2ws` **逐字节一致**
   （n=4,353,024/层，本 mission 自测）⇒ **非融合路的 numpy 对拍必然是同一张 `RESULT: FAILED`
   同 11 条**（review r1 独立跑该路确认）⇒ 越界与非融合路的输入分布（本 mission 的
   `H_GenBf16` amp 1.0/0.8/1.5）有关，**与四相位融合无关**
   （证据：`evidence/m58_check_hc_ref_run.log` 里的 **m=33 档**一段 +
   `evidence/m58_accept_run_all48.log` 的
   `H.h1ws`/`H.h2ws` 行）。
   也**不是**本轮的 6 类替换引起的（替换 ④ 只改 IJ 的取数方式、⑤ 只加一档分支；实测最差层 L3 的
   `A2.H'<-ij,hin,bo` 逐位 1.0000 / `ulpMax=0`、`A2.injW` maxRel=1.1e-7 ⇒ 替换 ④ 的那条
   IJ 取数路径上没有偏差）；评审还重建 m20 自跑 `real_m33` = **OK 28/28**，说明 m20 的门限对
   **它自己的** case 集仍成立。
   **未排除的假说**（未做实验，故只作假说）：m20 的门限是对**它自己的 case 集**校准的，而本
   mission 的输入分布不同，经 K=10240 的 down GEMM 链后落点更靠近 bf16 网格 ⇒ 用同一门限量
   同一算法会读出不同结果。
   判据方向：若要在 m>1 档主张「与 m20 同口径通过」，需要按 docs/17 的 T3 ε 逐元素
   **推导界**并逐步定位超界元素（本轮预算内未做）。修复方向与入口已记录在此。
3. **全局 mixer（第 3 档 `MODE_FINAL_MIX`）未接入**：hc 段**支持**该档（代码一字未改），
   但它是「48 层链之外」的调用点（`hyper_connection_mixer.combine_and_mix(use_combine=False)`），
   本轮没有为它做任何接线或验证。
4. **m20 §9.2 / §9.3 的两条并行度问题在融合后原样存在**（本节明确回答 task 书的问题）：
   * §9.2 第 2 条（S2 的 item 数 = `m·HC`，m=1 时只有 4 个 AIV 在关键路径）：**未变**。实测
     hc 段净增量 ≈131 µs/层里就包含这部分损失（S2 在 m=1 时是 4 个 item）。
   * §9.2 第 3 条（S3 的 down+inject 只有 3 个 N-tile ⇒ 3/28 AIC 有活）：**未变**。
   * §9.3（prefill 的 flagId 预算：一圈段序要 11 个 id、16 个 id 只够一轮 ⇒ 只能多次 launch）：
     **更紧了** —— 融合后一圈**四相位**段序要用掉 **AIV mode0 的 4 个 id（hc 两次 + GDN 四条里的
     4 个）+ 3 条相位边界 + MoE ring**，`FlagMaxUse()` 已到 4。prefill（65 轮）**必须多次 launch**
     这一结论不变，但「按 m-tile 多次 launch」时每次 launch 的 id 预算表要重新核算（届时
     hc 也要按 m-tile 切分）。
   * 另：hc 段的 `stageLimit` 折成 `HC_STAGE_LIMIT`（与 MoE 的 `MOE_STAGE_LIMIT` 同款），
     bring-up 截断开关在融合入口上不可用（`m15_hc_segment_kernel` 也传常量 7）。
5. **计数纪律的两处修复（本轮实测踩到；第二处由 review r1 的 P2-1 抓出）**：
   * (i) 新增的 hc 权重来源判据最初同时套了 `H_CmpBytes` 与 `H_CmpOk`，同一件事被计两次；
     而 A/B/C 三段的计数是「`C.checks` 减已归属项」的**增量式**写法，多出来的 96 条因此
     **落到 `phaseA` 上**（312 vs 216）。已修：hc 那道只用 `H_Guard`，并把 `phaseA/B/C` 的
     算式补上 `- C.phaseWh`。
   * (ii) **同类问题在 combine-only 的「不写 BLK」上漏网**（review r1 的 P2-1a）：
     `H_CmpOk(C, H_Guard(...))` 让同一条断言同时进 `checks` 与 `guards`。修法不是二选一降级，
     而是把它与 `H'` 逐字节比较**合并成一条判定项** `H.conly`（同一论断的两个面，其中
     「BLK 未被写出」是 mode 3 是否真的兑现的**判别性**见证）⇒ 计数 1237/160 → **1234/157**
     （H 24 → 21），且该性质仍是**计入失败**的判定项。
   * **教训**：新增判据时若用了增量式分账，必须同步改**所有**下游算式；并且**同一条断言
     只能进一个桶** —— 复核方式是「分账逐项相加 == `C.checks`」+「guard 分解逐项相加 == `C.guards`」
     （后者由 `m15_layer_loop.asc` 的 guard 汇总行按来源打印，见 M58-6）。
6. **`M15_SYNTH=1` 合成档不能作为 hc 的数值证据**（实测读数，`M15_SYNTH=1 M15_LAYERS=1
   M15_HC_LAYERS=0 <bin> "" h`）：**只有 `H.finite` 会红**（10240/10240 个 bf16 是 NaN/Inf），
   其余 6 条判定项（`H.nontrivial` / `H.h1ws` / `H.h2ws` / `H.conly` / `H.det_h1` /
   `H.det_h2`）**都 PASS**（该档只跑 1 层 ⇒ H 段 7 条判定项，其中 1 条红、6 条 PASS；
   权重来源那条因 `M15_SKIP_WCHECK=1` 未跑）。原因与 MoE 段同族但方向不同：hc 段本身不会造 NaN，
   是**子层段用随机权重算出的非有限值**经 `attn_out → hc(mlp) 的 BO` 传了下来。
   ⇒ synth 档对 hc 的**可用范围** = 同步/串扰/（`H.h1ws`/`H.h2ws`/`det` 的）字节等价冒烟；
   **不能**当作任何数值正确性证据 —— 本 mission 的全部数值证据都是**真实 checkpoint 档**。
7. **未做任何性能调优**：hc 段一字未动（没有跨段权重预取、S3 没有按 K 切分、S2 没有两遍归约
   换满核）。M58-7 的 **≈131 µs/层**（Avg 口径；Min 口径 100–111 µs）是**现状**，不是本 mission 的目标值。

## M58-10 文件与证据（本 mission 新增/改动的部分）

| 文件 | 状态 | 说明 |
|---|---|---|
| `m15_hc_layer.h` | 新（**生成物，勿手改**） | hc 段 device 代码（`lift_hc_segment.py` 生成，`--check` 可复核） |
| `m15_hc_resources.h` | 新 | hc 段的全局静态资源表（命名空间 `M15H`） |
| `lift_hc_segment.py` | 新 | 抽取脚本 + 上游不变量断言 + `--check` |
| `m15_hc_host.h` | 新 | host：hc 权重/来源判据/启动器/**验证 H**/dump |
| `check_hc_ref.py` | 新 | 独立 numpy 交叉校验（复用 m20 的 `reference()`/`judge()`） |
| `m15_layer_kernel.h` / `m15_layer_resources.h` / `m15_layer_loop.asc` / `slice_layer_manifest.py` / `weights_manifest.txt` | 改 | 见 M58-8 |
| `evidence/m58_accept_run_all48.log` | 新 | 48 层 `runs=all`：**1234 判定项 + 157 guard，0 FAIL**（二进制 sha256 `c20d73f63ce768afecd8099b630021727122e41af2a4bf962a0073388aeb7666`，见该文件头；guard 分解 = sha 1 + hc 注入行 48 + 参考链锚点 108 + 其余 0） |
| `evidence/m58_check_hc_ref_run.log` | 新（**三档合一**） | hc 段 numpy 交叉校验的 m=1 / m=2 / m=33 三档日志（同一二进制 `c20d73f6…`）：前两档 `RESULT: OK (64+15)`、m=33 档 `FAILED`（11 条越门限，见 M58-9 第 2 条）。**三个档的完整命令都在该文件头**（只差 `M15_HC_M`） |
| `evidence/m58_lift_diff.txt` | 新 | m20 device 段 → `m15_hc_layer.h` 的逐字 diff（6 类替换） |
| `evidence/m58_manifest_diff.txt` | 新 | manifest 的**纯增量**证明（900 → 1284，删除 0 行） |
| `evidence/m58_neg_controls.log` | 新 | 三组负向对照（`--check` 变异 / 6 条上游不变量断言 / 三态退出码 0-1-2） |
| `evidence/m58_dump_sha256.txt` | 新 | 93 个 hc dump 张量的 sha256 + **两个独立进程逐字节相同**的确定性证据 |
| `evidence/m58_msprof_hc.log` + `m58_msprof_all{1,2,3}.csv` + `m58_msprof_rep{1..5}.csv` | 新 | 设备侧段级分解：**同会话 A/B ×3**（每采样里两相位与四相位符号同时出现）+ hc 符号的 **5 次重复**（含抖动范围与三次共享卡污染现场；每条命令逐字记入该 log） |

**取证纪律（写给后续的评审者与接手者；review r2 提出，作者在场外也踩过同类）**：

1. **构造变异、跑派生脚本、任何会落盘的实验，请在 `/tmp` 的工作副本里做，不要直接改 `wt-NN`。**
   review r2 的变异测试脚本第一版相对路径写错，**误改了 `wt-58` 的 `m15_hc_layer.h` 与
   `m20_hyperconn.asc` 各一处**（当场 `git checkout --` 复原并核 `git diff 12b2555` 为空）。
   正确做法：`cp <file> /tmp/<name>_mut.h` 后改副本，或用 `git show HEAD:<path> > /tmp/<name>`
   再在副本上跑断言。这与「不要在**主 checkout** 落盘」是同一条纪律的延伸（主 checkout 的脏
   文件曾被快照进另一个 mission 的分支）。
2. **交卷前把本文引用的每一个路径 `ls` 一遍**（含 `{}`/`{1..N}` 展开形式）——引用了不存在的
   归档文件属于「与事实不符」（docs/17 §2.4/§2.6）。本轮修掉了 `m58_check_hc_ref_{m2,m33}_run.log`
   两处悬空引用（三档日志已合并进 `m58_check_hc_ref_run.log`）。
3. **同一份文档里的同一个量只允许有一个读数**：本轮的 M58-9 第 4/7 条曾残留**已作废的旧口径
   `129 µs/层`**（那是按「跨会话两次采集相减」算的），现已统一到新层表的同会话采样
   **≈131 µs/层（Avg 口径；Min 口径 100–111 µs）**。凡是引用了旧口径的句子都要一起改。

**复现一条命令行**（仓库根目录）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release && cmake --build m15_layer_loop/build -j8
python3 m15_layer_loop/lift_hc_segment.py --check           # hc 副本与上游规则逐字节一致
python3 m15_layer_loop/slice_layer_manifest.py --out m15_layer_loop/weights_manifest.txt --check
M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all
# hc 段的独立 numpy 交叉校验（m=1 / m=2）
mkdir -p /tmp/m58_dump && cd /tmp/m58_dump && M15_DUMP=1 M15_LAYERS=4 M15_HC_LAYERS=0,1,3 <bin> <repo>/m15_layer_loop/weights_manifest.txt h && /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_hc_ref.py; echo rc=$?
```

---

# Part C · M65：48 层链切到四相位入口（+ PLE 打断点 + 末层全局 mixer）

> **Part B · M58** 交付的是「**单层**的四相位形态」与其验证；本 Part 把 **48 层链**切到这个入口上
> （每层一次启动、层间零拷贝 handoff），并补上 M58 显式留下的两个缺口：**① 层 0→1 被 PLE 打断的
> combine-only 通路**（M58-9 第 1 条 / M58-5 第④档）、**③ 末层全局 mixer**（M58-9 第 3 条）。
> **PLE 本体仍未实现**（表 95.37 GiB 落不下，docs/14 §10 第 1 条是阻塞项）——本 Part 交付的是
> **打断点的结构与它的判据**，PLE 段是显式空操作。
>
> 交付形态 = 新运行档 `runs=chain`（`x` 同义）。两相位入口 `m15_layer_kernel_{gdn,attn}` 及其全部
> 判据（A/B/C/M1/M2）**一字未动**，`runs=h`（M58 的验证 H）口径也一字未动，都仍是回归基准。

| 量 | 值（真实 checkpoint 档，`runs=all`，48 层 × 3 token） |
|---|---|
| 判据总数 | **2012 条判定项 + 160 条 guard，`0 FAIL`**（`evidence/m65_accept_run_all.log`；**M82 r2 起同一条命令是 2068 + 290**：+ 验证 Kv 的 56 判定项与 130 guard —— 见 Part D · M82 与 `evidence/m82_accept_run_all.log`） |
| 分账 | 权重来源 84（GDN+MoE）+ 48（hc）+ A 216 + B 506 + C 3 + M1 348 + M2 8 + H 21 + **Ch 778** |
| **形态回归（不得倒退）** | 上列八项与 M58 归档**逐项相同**（84/48/216/506/3/348/8/21）⇒ 换链**没有**碰到两相位形态与验证 H 的任何口径 |
| 48 层链（`runs=chain`，含手工路径） | **778 条判定项 + 3 条 guard，`0 FAIL`**（`evidence/m65_chain_run48.log`；`M15_DUMP=1` 时多 1 条 `Ch.m39.dump` guard ⇒ 4）；48 次四相位启动 + 1 次末层 mixer |
| 与 M39 官方单层参考对拍（④） | `check_chain_ref.py`：**84 条判定项 + 8 条 guard，rc=0**（层 0/1/3 各两边界 + 末层 mixer）；**6 条超 m20 保守门限的读数逐条披露**（M65-3.3） |
| PLE 打断点的舍入点判据 | 层 1 的 **8 个张量**与单次 `combine_and_mix` **逐位一致**（`逐位=1.0000 ulpMax=0`）；**负向对照**：丢掉那次 bf16 物化 ⇒ `XN` **25.66%** / `BLK` **40.59%** 元素改变（M39 的 B4 在同族数据上量到「物化残差」34.25%） |
| 末层全局 mixer | `multi_hidden` / `sample_hidden` 与官方 `final_mixer` **逐位一致**（逐位 1.0000 / ulpMax 0） |
| 三态布局与双缓冲 | H/IJ **零拷贝**（住在按层号编址的 hc ws 里）；pending BO **双缓冲 2 块 × 320 KB** = 0.66 MB |
| 新增 device 缓冲 | hc 权重区 48 → **49 槽**（每槽 `HC_LAYER_W_STRIDE` = **26.91 MB**（= 2 个边界子槽 × 13.46 MB）⇒ **+26,910,720 B = 26.91 MB**，合计 49 × 26.91 = **1318.63 MB**；全局 mixer 槽只用其中 13.13 MB 数据，槽 stride 仍 26.91 MB）、hc ws 51 → **52 槽**（每槽 `HC_WS_LAYER_STRIDE` = **8,706,048 B = 8.71 MB**，合计 452.71 MB）、pending BO 双缓冲 2 × 327,680 B = **0.66 MB** |
| UB / L1 / L0C 峰值 | **未变**：227328 / 303104 / 65536 B（打断点与末层 mixer 都复用 hc 段已有的段窗） |
| flagId 上界 | AIV mode0 的 id 12 用量 **4 → 6**（层 1 把 H1 拆成 H1a/H1b；硬件 4bit 计数器上限 15） |
| 段级设备耗时（msprof） | 见 **M65-8**（含 PLE 打断点的开关 A/B 与「1 次启动/层 vs 4 段独立启动/层」的差） |

**缓冲口径自检（M65 r2 修 ①）**：下面是 `runs=all` 在 48 层真实权重档上的实际打印行
（**逐字、单行**；复核方式 `grep "M58 hc 缓冲" m15_layer_loop/evidence/m65_accept_run_all.log`）：
```
[m15] M58 hc 缓冲：权重 1318.63 MB（49 槽 × 26.91 MB/槽；49 = 48 个层槽（每层 2 个边界子槽 × 13.46 MB）+ 1 个全局 mixer 槽，该槽只用到 13.13 MB（norm+down+up），槽 stride 仍为 26.91 MB）+ hc ws 452.71 MB（52 槽 × 8.71 MB/槽 = 每层 2 块 + 3 块 scratch + 1 块全局 mixer ws）
[m15] hc 段权重装载完成（真实 checkpoint（hc 权重不量化、原样消费））：本 run H2D 49 槽 × 26.91 MB/槽 = 1318.6 MB（48 个层槽，每层 2 个边界子槽 × 13.46 MB，+ 1 个全局 mixer 槽）；权重区分配 49 槽 × 26.91 MB/槽 = 1318.6 MB → device，耗时 477.4 ms
```
算术：`HC_LAYER_W_STRIDE`（一个层槽）= 2 × `HC_W_STRIDE` = 2 × 13,455,360 B = **26,910,720 B = 26.91 MB**；
**49 × 26.91 = 1318.63 MB**（= 1291.71（48 层槽）+ 26.91（1 个全局 mixer 槽））；
`HC_WS_LAYER_STRIDE` = 8,706,048 B = 8.71 MB，52 槽 = 452.71 MB。判据：**「每槽尺寸 × 槽数 = 合计」必须自洽**
（`26.91` 是分配真值，`13.46` 只是层槽里的一个边界子槽）。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release && cmake --build m15_layer_loop/build -j8
# 新增：48 层链（四相位入口 + PLE 打断点 + 末层 mixer）；也包含在 all 里
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt chain
# ④ 与 M39 官方单层参考对拍（需要 torch ⇒ 用 baseline venv；在 dump 目录里跑。
#    **只驱动 M39 的 ref/*.py，不调用其 compare_dumps.py 逐段 dump 通路** —— 见 M65-3.3 的「方法覆盖」）
mkdir -p /tmp/m65_dump && cd /tmp/m65_dump
M15_DUMP=1 <repo>/m15_layer_loop/build/m15_layer_loop <repo>/m15_layer_loop/weights_manifest.txt chain
/workspace/venvs/baseline/bin/python3 <repo>/m15_layer_loop/check_chain_ref.py; echo rc=$?
```

## M65-1 48 层链的段序（每层一次启动）

```
入口 H = embed_tokens(input_ids).repeat(1, hc_count)          # [m,10240] bf16（4 路复制）
for L in 0..47:                                                # 每层**一次** __mix__(1,2) 启动
  ┌─ H1（hc 边界 #1 = attn_hc）──────────────────────────────────────────────────────────────┐
  │  层 0  : MODE_MIX（prev_block_output 为 None；不做 combine，H' ≡ H）                     │
  │  层 1  : **PLE 打断点**（M65-3）combine-only → PLE 占位 → mix-only                       │
  │  层 ≥2 : MODE_COMBINE_MIX（combine_and_mix）                                             │
  │  输入：H（上一层的 H''）⊕ pending BO（上一层的 mlp_out）⊕ IJ（上一层的 injection logits） │
  └───────────────────────────────────────────────────────────────────────────────────────────┘
  ├─ 子层段（GDN S1-S7 / attention 占位直通）：输入 = 本层 H1 的 BLK_attn
  ├─ H2（hc 边界 #2 = mlp_hc，恒 MODE_COMBINE_MIX）：输入 = H'（层 0 用 H）⊕ attn_out ⊕ H1 的 IJ
  ├─ MoE 段 S1-S10：输入 = H2 的 BLK_mlp，出口 `yLayer` = **下一层的 pending BO**
  └─ 出口三态：H''（下一层的 H）、IJ''（下一层的 IJ）、mlp_out（下一层的 BO）
末层之后：m15_final_mixer_kernel（MODE_FINAL_MIX）→ multi_hidden [m,10240] / sample_hidden [m,2560]
```

与 vLLM 的逐句对应（`V-N:model.py:489-590` / `V-N:model.py:276-331`）：

| vLLM | 本 kernel |
|---|---|
| `prefetch PLE(L+1)` | **未实现**（PLE 本体；M65-9 第 1 条） |
| `hidden, block_output, injection = layer_L(...)` | 一次四相位启动的出口三态 |
| `attn_hc.combine_and_mix(hidden, prev_block_output, prev_injection)` | 相位 H1（层 0 = `mix()` / 层 1 = 打断点 / 其余 = combine_and_mix） |
| `attn_out = attention(block_input)` | 子层段（GDN 真段 / attention 占位直通） |
| `mlp_hc.combine_and_mix(hidden, attn_out, injection)` | 相位 H2 |
| `mlp_out = moe(block_input)` | 相位 B（MoE 段 S1-S10） |
| `multi_hidden, sample_hidden, _ = final_mixer.combine_and_mix(...)` | `m15_final_mixer_kernel` |

## M65-2 三条长寿状态在 GM 的布局与双缓冲（**零拷贝 handoff**）

| 状态 | 形状 | 住在哪 | 谁写 / 谁读 | 双缓冲 |
|---|---|---|---|---|
| `H`（多流残差） | `[m,10240]` bf16 | 层 L 的 **hc1 ws** 的 `WS_HCP`（`hcWs[L]+HC_WS1_IN_LAYER+WS_HCP`） | 层 L 的 H2 写；层 L+1 的 H1 直接当 `hcH` 读 | **不需要**：H 天生按层号各占一块（48 块） |
| `pending BO` | `[m,2560]` bf16 | 独立的 **BO 平面** `chainBo[L % 2]` | 层 L 的 MoE S10 写（`yLayer`）；层 L+1 的 H1 当 `hcBo` 读 | **2 块**（0.66 MB） |
| `IJ`（injection logits） | `[m,4]` bf16 | 层 L 的 **hc1 ws** 的 `OH[:,320:324)`（行距 `OH_W=336`） | 层 L 的 H2 写；层 L+1 的 H1 按 `ijStride=OH_W` 直读 | **不需要**（与 H 同块） |
| 入口 H（层 0） | `[m,10240]` bf16 | `hcHDev`（M58 已有的 `[M_MAX,10240]` 平面） | host 写（= 4 路复制）；层 0 的 H1 读 | — |

**为什么这样放**：H''/IJ'' 必须**活到下一层启动**，所以不能落在每层复用的 GDN/MoE 单块 ws 里；
按层号编址的 hc ws（48 层槽 × `HC_WS_LAYER_STRIDE` = 8,706,048 B ⇒ 417.89 MB）让「下一层的 `hcH`/`hcIj` 直接指上一层 ws 的两个偏移」成立，
**不需要任何抽取 stage 或 host 搬运**。代价 417.89 MB（48 个层槽），收益是层间 handoff 零拷贝。
`m15_chain_host.h::H_ChWiredCheck` 把这条接线做成 5 条结构性 guard：48 层的 H'' 平面两两不同、
H'' 不与自己的 H1 输入别名、IJ'' 不与 H'' 重叠、两个 BO 平面不相交、IJ 源 32B 对齐。

**host 侧只搬「记录」不搬「数据」**：`H_ChRunOnce` 每层把三态 D2H 一份用于判据/对拍，但**喂给
下一层的指针**始终是上面那三个 GM 地址（`Ch.*` 的判据就是拿「记录」与「设备实际产出」比）。

## M65-3 PLE 打断点（层 1）：结构、判据、档位理由

### 3.1 结构与语义依据

`V-N:model.py:290-301`（0-based 层 1 = `ple_layer_ids=[2]`（1-based）那一层）：

```python
if prev_block_output is not None:                 # 实际总是成立（PLE 在层 1，不是层 0）
    hidden = attn_hc.combine(hidden, prev_block_output, prev_injection)   # ← 先物化 pending combine
    prev_block_output = prev_injection = None
hidden = ple(hidden, input_ids, query_start_loc, ngram_context)           # ← PLE 直接改 10240 宽的态
(hidden, block_input, injection) = attn_hc.combine_and_mix(hidden, None, None)   # ← 此后才是 mix
```

⇒ 这个边界的 combine 与 mix **必须分成两段**，中间夹 PLE。本 kernel 的做法（一次启动、五段）：

```
H1a: MODE_COMBINE_ONLY（只跑 W0+S1：H' = bf16(H + BO·injW)，AIC 不跑 GEMM，写 hcWs0+WS_HCP）
 ↓ 相位边界（FLAG_PLE_IN_BOUND_AIV，全体 AIV mode-0）
PLE 段：`M15L_PlePhasePlaceholder()` = **显式空操作**（PLE 本体未实现，M65-9 第 1 条）
 ↓ 相位边界（FLAG_PLE_OUT_BOUND_AIV）
H1b: MODE_MIX（读**物化好的 bf16 H'** → 规范化/down/silu/up/gate → BLK_attn / IJ'）
 ↓ 相位边界（= M40/M58 的 H1→A 边界）
```

两条新边界**不占新 flagId**（AIV 核上 0..15 已被三段的 mode0/mode2 用满），直接复用 M40/M58 登记的
同类型全体-AIV mode-0 barrier id 8/9；相邻性由 `FLAG_SEQ[]` 的执行序见证（层 1 的序：
`12 → 8 → 9 → 12/13/14 → 8 → …`，逐对满足「相邻必不同」），用量上界 `FlagMaxUse()` 从 4 升到 **6**
（= H1a/H1b/H2 三次 + MoE 两处 + 单次路径的 H1；硬件上限 15）。

### 3.2 判据：**逐字节**（`Ch.ple.bf16.L01` / `Ch.ple.single.L01`）

层 1 的 H1 分段路径（combine-only → 同一块 GM 的 bf16 `H'` → mix-only）必须与**单次
`combine_and_mix`**（`m15_hc_segment_kernel(mode=MODE_COMBINE_MIX)`、**同输入同权重**）**逐字节
相同**：`Ch.ple.bf16.L01` 比规范化输出 `XN`，`Ch.ple.single.L01` 比 `BLK`，另有 `Ch.h1hcp.L01`
比物化出来的 `H'`、`Ch.h1inj.L01` 比 injection。

**为什么这条判据判的就是「combine→RMSNorm 之间的 bf16 舍入点」**（docs/14 §10 第 3 条）：

* 单次 `combine_and_mix` 的 `CombineStage` 把 `H'` 以 **bf16 写进 GM**（`WS_HCP`），`NormStage`
  再从 GM 读回 —— 这就是 docs/14 §3.2 说的 `V-N:ops/hc.py:325-327` 那次早舍入；
* 分段路径读的是**同一块 GM 的同一批 bf16 字节**；若实现把 combine 的 fp32 结果**不物化**直接喂给
  规范化（省一次 GM 往返），`XN` 就会与这条参考不同。

**档位理由：逐字节（不是 ulp 门限）**。两侧是同一条指令序列作用在同一批 bf16 字节上，唯一可能的分歧
就是「有没有那次 bf16 物化」⇒ 不存在需要用误差吸收的中间态（docs/17 §1.1：良态元素应当逐位一致）。

**判别性（负向对照）**：`check_chain_ref.py` 在**真实层 1 数据**上算「丢掉那次物化」的反事实
（其余完全一样），读数 **`XN` 25.6641% 元素不同、`BLK` 40.5859%（1039/2560）元素不同** ⇒ 判据不空洞。
对照：M39 的 B4 在同族数据上量到「物化残差」**14029/40960 = 34.25%** 元素受影响
（`m21_layer_ref/evidence/selfcheck.log`）—— 口径不同（B4 量的是残差本身，这里量 `XN`/`BLK`），
只作量级对照，不能直接相减。

**这条判据抓什么、不抓什么（口径说明，必须读）**：它能抓**实现错误**（mix 读错平面、combine 没物化、
分段顺序错 —— 见 M65-8 的变异实验 m3），但**抓不住「`pleBreak` 标志被忽略」**：分段路径与单次路径
**按设计逐字节等价**（这正是「PLE 缺席时打断点不改变数学」的语义要求），两种实现的输出本就相同。
而且 **msprof 也不能当这个见证** —— M65-8 的开关 A/B 实测 Δ ≈ **1 µs/层量级**（分段不重复工作：
H1a+H1b 就是 H1 的两个相位，只多两条 mode-0 边界），远在测量分辨率边缘，正反两侧都"看起来一样"。
⇒ 「标志被忽略」这种错**只能**由代码结构见证：M65-1 的段序图 + `FLAG_SEQ[]` 的执行序
（编译期 `static_assert(FlagSeqAdjacentOk())` 只在打断点的相位序列存在时才成立）+ 显式的
`[m15][DISCLOSE]` 打印。**不得**把「输出逐字节一致」读成「打断点一定生效」。

### 3.3 与 M39 官方单层参考的对拍（④）：读数与**如实披露**

`check_chain_ref.py`（torch；`m21_layer_ref/ref` 的官方语义实现，权重自己从 checkpoint 读）在
`runs=chain` 的 dump 上与设备对拍 **层 0（GDN）/ 1（GDN，PLE 层）/ 3（attention）** 的两个边界
+ 末层 mixer，共 **84 条判定项 + 8 条 guard，`rc=0`**：

| 层 | 边界 | 读数（与 M39 官方参考） |
|---|---|---|
| 0 | H1（`MODE_MIX`） | `xn` 逐位 0.9999 / `lora` 0.9969 / `injection` 1.0000；下游 `ls` 0.9938、`gate` 0.8152、`blk` 0.9430 —— **6 条超 m20 保守门限**（见下） |
| 0 | H2 | **全部逐位 1.0000（ulpMax=0）** |
| 1 | H1（**PLE 打断点**） | **8 个张量全部逐位 1.0000（ulpMax=0）**：`H'`/`xn`/`blk`/`injection`/`lora`/`ls`/`gate`/`injW` |
| 1 | H2 | 全部逐位 1.0000 |
| 3 | H1 / H2 | 全部逐位 1.0000（`A2.gate` 逐位 0.9999、ulpMax=0） |
| — | 末层 mixer | `multi_hidden` / `sample_hidden` 逐位 1.0000；`GM.xn` 逐位 1.0000 |

**层 3 是 attention 层，但本 kernel 的 attention 段是占位恒等直通**（`m15_attn_passthrough_body`）：
上表对层 3 只核了 **hc 边界**（边界 #2 的 `bo` 取自设备 dump 的 `attnout`），
**attention 语义（QSA）没有被对拍**（M65-9 第 2 条）。

**方法覆盖**：与 M39 的对拍直接驱动其 `ref/hc.py` / `ref/layer.py`，**未**使用其 `compare_dumps.py` 逐段 dump 通路（GDN/QSA/MoE 段与我们的实现不同源，见 M65-9）。

**披露（不得含糊）**：层 0 的 H1（= 链上唯一的 `MODE_MIX` 边界）有 3 个张量（`ls` 良态逐位 0.9888、
`gate` 0.9607、`blk` 0.9469）**低于 m20 保守门限的「良态逐位率 ≥99%」**，但都**满足本脚本的判定门**
（良态元素 `ulpMax=1` ≤2、张量尺度归一化绝对误差 ≤3.8e-3 ≤1e-2）。判定门的取法写清楚：
hc 段这六个张量的**适用档是 docs/17 §1.1 的 T3**（逐项 ε 推导见 M58-6 表），而 docs/17 §1.1
**明确把 T3 档的「`≤1ulp 比例` / `maxRel` / 位级一致率」降为报告项**，判定看逐元素的界
⇒ 本脚本的判定门 = **m20 保守门限去掉「良态逐位率 ≥99%」那一条**，并把 m20 的逐位率**当报告项**
逐条列出（`[DISCLOSE]` 段）。已收集的证据（**不作结论性断言**）：

1. **两条独立 oracle 给出逐位相同的读数**（M39 的 torch 与 m20 的 fp64 逐句实现）⇒ 不是某一个
   oracle 的实现差异；
2. 越门限的都是**同一个边界**（`MODE_MIX`）的下游张量，其上游 `xn` 只有 **1 个非良态元素**差 1 ulp；
   机理：silu 把「非良态」的相对尺度换了一档 + K=320 的 up GEMM 在近零处放大 ⇒ 1 ulp 级伴生；
3. 这条边界的整块 ws 与 `m15_hc_segment_kernel(mode=MODE_MIX)`（M58 已验收的独立启动路）
   **逐字节相同**（C++ 侧 `Ch.h1blk.L00` / `Ch.h1inj.L00`）⇒ 差**不是链的接线引入的**。

⇒ 这是一条**待归因项**（性质与 M58-9 第 2 条的 m=33 越门限同类：门限是在别的输入分布上校准的），
不是「已解释干净」。见 **M65-9 第 3 条**。

## M65-4 末层全局 mixer（`MODE_FINAL_MIX`）

* 入口：`m15_final_mixer_kernel`（新符号；与 `m15_hc_segment_kernel(mode=MODE_FINAL_MIX)` 是同一段
  device 实现，单独给符号是为了设备侧分解能按符号区分 —— M65-8）。
* 输入 = 层 47 的三态出口（H''/mlp_out/IJ''），输出 = `ws+WS_HCP`（`multi_hidden [m,10240]`）与
  `ws+WS_BLK`（`sample_hidden [m,2560]`），与 `V-N:model.py:581-590` 的两个返回值一一对应。
* 权重 = checkpoint 的 `model.language_model.hyper_connection_mixer.*` **3 个张量**（新增 manifest
  行 `layer=48 role=gmixer_{norm,down,up}`），槽号 = 第 **49** 个 hc 权重槽（槽 stride = `HC_LAYER_W_STRIDE` = 26.91 MB；3 个张量的数据只有 13.13 MB，槽内余量保持全零）。checkpoint **没有**
  `block_inject_weight`（`V-N:model.py:612` 显式丢弃）⇒ host 把注入区 16 行**置零**；而
  `MODE_FINAL_MIX` 的 S3 只跑 lora 的 2 个 N-tile，那 16 行**从不被读**（两条都见证：`Ch.wsrc.gm`
  逐字节 + 注入区全零 guard）。
* 判据：`Ch.gm.multi`（combine 真的发生）、`Ch.gm.sample` + `Ch.gm.sample_nonconst`、`Ch.wsrc.gm`
  + 注入区 guard、`Ch.det.gm_*`（确定性），以及 ④ 的官方 `final_mixer` 对拍（逐位 1.0000）。

## M65-5 判据清单与计数口径（task 4 的 ①②③④⑤）

| # | 判据 | 落点 | 条数（48 层） |
|---|---|---|---|
| ① | 非空洞性 | `Ch.nonid.L*`（H'' ≠ 层输入 H）、`Ch.moe.L*`（mlp_out ≠ BLK_mlp）、`Ch.finite.L*`、`Ch.gm.multi/sample/sample_nonconst`、`Ch.pert.lastH/gm/gmmerge`（入口 H ×1.5 ⇒ 末层三态与 mixer 两个输出都必须变）+ guard `Ch.pert.in` | 4×48 + 6 = 198 |
| ② | 跨层可区分（槽位/层号不错位） | `Ch.blks.L*`（同层两个边界的 BLK 互异，计在 ① 的 4×48 里）、`Ch.fp.L*`（48 层的「三态原始字节指纹」两两互异，**逐字节比、不做哈希** ⇒ 判据精确、不存在碰撞假说） | 48 |
| ③ | 确定性 | `Ch.detH/detBO/detIJ.L*`（同一二进制重跑整条链，三条长寿状态逐字节相同）+ `Ch.det.gm_multi` / `Ch.det.gm_sample` | 3×48 + 2 = 146 |
| ④ | 与 M39 官方单层参考对拍 | **不在本程序的计数里**：`check_chain_ref.py`（torch，独立三态退出码与计数）在 dump 上做；本程序的 guard `Ch.m39.dump` 只保证输入/输出已落盘。**方法覆盖**：与 M39 的对拍直接驱动其 `ref/hc.py` / `ref/layer.py`，**未**使用其 `compare_dumps.py` 逐段 dump 通路（GDN/QSA/MoE 段与我们的实现不同源，见 M65-9）。 | 84（脚本） |
| ⑤ | 与两相位形态回归 | 设备侧：**逐层 4 段独立启动、host 手工串**（hc#1 → 子层段 → hc#2 → MoE）与「一次启动的链」逐字节一致（`Ch.h1*/Ch.sub/Ch.h2*/Ch.bo`）—— 层 0 7 条、层 1 **10 条**（多 `Ch.ple.*` 三条）、层 2..47 每条 8 条；另 `Ch.wsrc.gm` 1 条 = 全局 mixer 权重来源。读数侧：`runs=all` 的八项与 M58 归档逐项相同 | 7 + 10 + 8×46 + 1 = 386 |

**分账自检**：① 198 + ② 48 + ③ 146 + ⑤ 386 = **778**，与程序打印的 `C.phaseCh` 逐项一致（同源计数）。
**计数纪律**（M58-9 第 5 条的教训）：本段所有判据都由**调用点自增** `C.checks`/`C.guards`，
`C.phaseCh`/`C.guardsCh` 是**差分**（`checks` 减段首快照）；guard 分类行新增 **M65 链** 一栏，
`其余` 仍显式打印 ⇒ 新增 guard 不会被静默吸收进任何一类。④ 的计数**不与本程序混计**，
两边分别打印（避免「同一件事进两个桶」）。

## M65-6 与 docs/14 的逐条对账

| docs/14 处 | 本轮处置 |
|---|---|
| §4.1 骨架（层循环 + 3 张量层界） | ✅ 落地：48 层链、每层一次启动、三态零拷贝（M65-1/2） |
| §4.1「层 0→1 被 PLE 打断」 | ✅ 落地为 PLE 打断点（M65-3）；**PLE 本体未实现**（M65-9 第 1 条） |
| §4.1 末层 `combine_and_mix` → `multi_hidden`/`sample_hidden` | ✅ 落地（M65-4），并与官方 `final_mixer` 逐位对拍 |
| §4.2 层界三张量（20,480 / 5,120 / 8 B）与「3 张量 × 双缓冲」 | ✅ H/IJ 零拷贝（天生按层号各占一块，不需要双缓冲），BO 双缓冲 2 块 |
| §6.1/§6.2 PLE 在 0-based 层 1（`ple_layer_ids=[2]` 是 1-based） | ✅ 代码写成 `pleBreak = (L == 1)`，注释里写明两种口径 |
| §9.2 第 3 条「PLE 特例必须显式建模：层 1 的 `attn_hc` 必须先做独立 combine」 | ✅ 落地；两条相位边界由 `FLAG_SEQ[]` 见证（M65-3.1） |
| §10 第 3 条「combine→RMSNorm 的 bf16 舍入点是否必须复刻」 | ✅ **必须复刻**：逐字节判据 + 负向对照（25.66%/40.59%），档位理由见 M65-3.2 |
| §9.3 第 3 条「prefill 的 flagId 预算」 | ⚠ 未变（prefill 不在本轮范围）；本轮把用量上界从 4 提到 6 |
| §10 第 1 条「PLE 表怎么落地」 | ❌ **仍是阻塞项**，与本轮无关；本轮只把打断点铺好 |

## M65-7 与 M58「单层四相位」形态的差异

| 项 | M58 | M65 |
|---|---|---|
| 交付形态 | `m15_layer_kernel_{gdn,attn}_hc` **单层**（`runs=h` / `H_RunH`） | 同一入口 + **48 层链**（`runs=chain` / `H_RunChain`）；`runs=h` 的判据口径一字未动 |
| 层界三态 | 每层都从 `hcHDev/hcBoDev/hcIjDev` 读**同一组**常量输入 | 层 L 的三态指**上一层**的出口（零拷贝）；只有层 0 是入口平面 |
| PLE 打断点 | 只交付 `MODE_COMBINE_ONLY` 的**入口**（`m15_hc_segment_kernel`） | **接进链**：层 1 的 H1 拆成 combine-only → PLE 占位 → mix-only（新 `LayerArgs::hcPleBreak`） |
| 末层 mixer | 未接线（显式披露） | `m15_final_mixer_kernel` + 3 行 manifest + 第 49 个权重槽 |
| 新增 flagId | hc 段重编为 12/13/14 + 0/1/2/3 + 8/9/10/11 | **不新增**：两条 PLE 边界复用 8/9；用量上界 4 → 6 |
| `*_hc` 入口签名 | 17 个 hc 参数 | **多一个** `uint32_t hcPleBreak`（host 侧 `H_LaunchLayerHc` 相应多了 `yLayerDev/hIn/bo/ij/ijStride/pleBreak`） |
| 缓冲 | hc 权重 48 槽（48 × 26.91 = 1291.71 MB）、hc ws 51 槽（51 × 8.71 = 444.01 MB） | **49 槽**（49 × 26.91 = 1318.63 MB；+1 个全局 mixer 槽 = +26.91 MB）、**52 槽**（52 × 8.71 = 452.71 MB；+1 块全局 mixer ws = +8,706,048 B）、BO 双缓冲 2 × 327,680 B = 0.66 MB |
| 新增判据 | H 21 条 | **Ch 778 条** + ④ 的 84 条（独立脚本） |

## M65-8 段级设备侧分解（msprof；host 墙钟不作为证据）

**采集口径与并发背景（必读）**：`msprof --task-time=l1 --ai-core=on --ascendcl=off --runtime-api=off`，
命令逐条记进 `evidence/m65_msprof.log`，原始 CSV 归档为 `evidence/m65_msprof_*.csv`。
采集期间**共享卡上有别人的进程**（实测同时存在 `m14_gdn_layer` / `m20_hyperconn` / `probe_aiv_sync` /
`probe_v_align`）⇒ **单次启动被挤到 4 ms ~ 106 s 的样本真实出现过**（见下表的 Max 列：`gdn_hc` 在 `a3`
有一次读到 **106,507,076 µs ≈ 106.51 s**），所以本节**每条读数都给 Min / 中位 / 稳健 Avg
（只用 ≤3×中位 的样本）/ 原始 Avg / Max 五个口径**，并把「稳健」与「原始」两个量分开标注
（原始 Avg 被污染样本拉到 4 ms ~ 1 s 量级；单次 Max 最高 106 s）。

**下表与「每 step」表都由本仓工具从归档 CSV 直接产出**（`parse_msprof_chain.py --table`，M65 r3 起；
**表不手抄** —— 手抄会让「由归档 CSV 现算」这句话不成立，r2 就是这样在 Max 列掺进了表外系列的值）。
每格形如「值（来源采集）」，读者可当场核出每个格子出自哪份 CSV。
**口径范围（必须一起读）**：本表**只统计下面命令里列出的这 7 份 48 层纯链采集**（`a1..a3`/`b1..b3`/`c1`）；
**不含** 2 层放大系列 `p*`（M65-8③）与 `runs=all` 系列 `t*`（M65-8④）—— 把别的系列并进来必须**重出本表**。

| 符号 | Count/次 | Min | 中位 | 稳健 Avg（≤3×中位） | 原始 Avg | Max |
|---|---|---|---|---|---|---|
| `链·GDN 层（四相位）` | 108 | **338.76（a3） – 348.15（c1）** | 354.86（b3） – 357.28（b2） | 354.71（b3） – 356.31（b2） | 干净 354.71（b3）；含污染样本的采集 a1=4264、a2=983647、a3=986529、b1=957264、b2=985624 | 106507076.70（a3） |
| `链·attention 层（四相位）` | 36 | **210.38（c1） – 212.73（a2）** | 213.10（a1） – 214.47（a3） | 213.29（a1） – 214.25（a2） | 同稳健 | 219.42（b1） |
| `手工·单边界 hc` | 98 | **17.26（c1）** | 77.25（c1） | 78.14（c1） | 同稳健 | 85.45（c1） |
| `手工·GDN 段` | 36 | **133.72（c1）** | 139.77（c1） | 139.85（c1） | 同稳健 | 145.43（c1） |
| `手工·MoE 段` | 48 | **58.96（c1）** | 63.62（c1） | 62.87（c1） | 同稳健 | 66.12（c1） |
| `末层全局 mixer` | 3 | **74.56（b2） – 76.41（a1）** | 75.32（b3） – 77.41（c1） | 75.42（b2） – 77.23（a1） | 同稳健 | 78.18（a1） |
| `手工·attention 占位` | 12 | **2.22（c1）** | 2.77（c1） | 2.78（c1） | 同稳健 | 3.27（c1） |

| 采集 | 每 step 稳健 Avg（µs） | 每 step 中位（µs） | 每 step Min（µs） |
|---|---|---|---|
| a1 | 15445 | 15445 | 15050 |
| a2 | 15453 | 15446 | 14887 |
| a3 | 15467 | 15496 | 14807 |
| b1 | 15430 | 15427 | 14891 |
| b2 | 15469 | 15501 | 14816 |
| b3 | 15411 | 15417 | 14849 |
| c1 | 15438 | 15441 | 15134 |

（口径：本表只统计上面列出的这批 CSV；若要把别的系列（如 2 层放大 `p*`、`runs=all` 的 `t*`）并进来，必须一并重出本表 —— **不要手抄其它系列的值**。）

**① 每 step（48 层 + 末层 mixer）** = 36×GDN + 12×attention + mixer（同一次采集内三个符号取同一口径），
也由同一条命令产出（见下面命令 A 输出里的第二张表）：7 次采样的稳健 Avg 落在 **15411–15469 µs**、中位 **15417–15501 µs**、
Min **14807–15134 µs**；其中干净采样的细算：`c1` 稳健 Avg = 36×355.47 + 12×213.66 + 77.10 = **15438 µs**、
中位 15441、Min 15134；`b3` 稳健 Avg = **15411 µs**、中位 15417、Min 14849 ⇒ 两条干净采样的口径差 ≤0.2%。

**复现（在本仓根目录跑；只读归档 CSV）**：
```bash
# 命令 A —— 产出本节的表：`--table` 打两张 markdown 表（§1 表 + 每 step 表），末尾 sed 截出这一段
/usr/local/python3.12.13/bin/python3 m15_layer_loop/parse_msprof_chain.py --table \
    a1=m15_layer_loop/evidence/m65_msprof_a1.csv a2=m15_layer_loop/evidence/m65_msprof_a2.csv \
    a3=m15_layer_loop/evidence/m65_msprof_a3.csv b1=m15_layer_loop/evidence/m65_msprof_b1.csv \
    b2=m15_layer_loop/evidence/m65_msprof_b2.csv b3=m15_layer_loop/evidence/m65_msprof_b3.csv \
    c1=m15_layer_loop/evidence/m65_msprof_c1.csv | sed -n '/markdown 表/,/不要手抄其它系列的值/p'
# 命令 B —— 同一批 CSV 的逐采集原始行（grep 只留链的三个符号：命令 A 第二张表的每格出处都在这三行里；
#           第一张表的 `手工·*` 四行不在本 grep 内，去掉末尾 grep 即可看到、其值另见下面 ② 手工路径段）
/usr/local/python3.12.13/bin/python3 m15_layer_loop/parse_msprof_chain.py \
    a1=m15_layer_loop/evidence/m65_msprof_a1.csv a2=m15_layer_loop/evidence/m65_msprof_a2.csv \
    a3=m15_layer_loop/evidence/m65_msprof_a3.csv b1=m15_layer_loop/evidence/m65_msprof_b1.csv \
    b2=m15_layer_loop/evidence/m65_msprof_b2.csv b3=m15_layer_loop/evidence/m65_msprof_b3.csv \
    c1=m15_layer_loop/evidence/m65_msprof_c1.csv | grep -E "采集 |链·GDN|链·att|末层"
```
命令 B 的逐字输出（每条数据行的格式：Count / Min / 中位 / Avg / Max / 稳健Avg / 稳健Total / Total）：
```
===== 采集 a1（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    345.68    355.85   4263.78 422419.02    355.79     38069.1    460488.1
链·attention 层（四相位）            36    210.76    213.10    213.29    217.87    213.29      7678.4      7678.4
末层全局 mixer                     3     76.41     77.12     77.23     78.18     77.23       231.7       231.7
===== 采集 a2（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    340.55    355.61 983647.21 106195837.28    355.72     38061.8 106233899.1
链·attention 层（四相位）            36    212.73    214.07    214.25    216.31    214.25      7713.0      7713.0
末层全局 mixer                     3     74.86     75.63     75.75     76.76     75.75       227.2       227.2
===== 采集 a3（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    338.76    356.83 986529.46 106507076.70    356.12     38104.5 106545181.2
链·attention 层（四相位）            36    211.38    214.47    214.23    216.98    214.23      7712.1      7712.1
末层全局 mixer                     3     75.36     76.07     76.30     77.47     76.30       228.9       228.9
===== 采集 b1（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    341.00    355.06 957264.13 103346530.32    355.10     37996.0 103384526.3
链·attention 层（四相位）            36    211.66    214.12    214.22    219.42    214.22      7712.1      7712.1
末层全局 mixer                     3     75.23     75.68     75.86     76.68     75.86       227.6       227.6
===== 采集 b2（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    338.99    357.28 985624.31 106409300.22    356.31     38124.8 106447425.0
链·attention 层（四相位）            36    211.50    213.60    213.84    216.74    213.84      7698.4      7698.4
末层全局 mixer                     3     74.56     75.70     75.42     76.00     75.42       226.3       226.3
===== 采集 b3（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    339.97    354.86    354.71    364.05    354.71     38308.8     38308.8
链·attention 层（四相位）            36    211.26    213.88    213.83    216.00    213.83      7698.0      7698.0
末层全局 mixer                     3     74.93     75.32     75.79     77.12     75.79       227.4       227.4
===== 采集 c1（设备侧 Task Duration，µs）=====
链·GDN 层（四相位）                 108    348.15    355.60    355.47    364.63    355.47     38391.3     38391.3
链·attention 层（四相位）            36    210.38    213.54    213.66    216.59    213.66      7691.6      7691.6
末层全局 mixer                     3     76.19     77.41     77.10     77.68     77.10       231.3       231.3
```
（`a1..c1` 的含义：`a*` = `MANUAL=0 PLE=1`、`b*` = `MANUAL=0 PLE=0`、`c1` = `MANUAL=1 PLE=1`；
每份 CSV 里 `Count=108` 的 GDN 符号 = **3 条链 × 36 个 GDN 层**。）

**② 手工路径（4 段独立启动/层）与链的对照**（`M15_CHAIN_MANUAL=1`，与链**同一次采集**：
两组符号在同一个 CSV 里 ⇒ 不受跨会话差异影响；这四行在 §1 的工具表里也出现 —— **同一份 `c1` CSV 的另一种口径**）：

| 符号 | Count | Min | Avg | Max |
|---|---|---|---|---|
| `m15_hc_segment_kernel`（手工·单边界；**两种档共用一个符号**） | 98（48×2 边界 + 2 条 PLE 对照） | **17.26**（= combine-only 档） | 78.14 | **85.45**（= W0+S1-S6 全链档） |
| `m15_gdn_layer_kernel`（手工·GDN 段） | 36 | 133.72 | 139.85 | 145.43 |
| `m15_moe_segment_kernel`（手工·MoE 段） | 48 | 58.96 | 62.87 | 66.12 |
| `m15_attn_placeholder_kernel`（手工·attention 占位） | 12 | 2.22 | 2.78 | 3.27 |

⇒ **每层「1 次启动（四相位链）」vs「≥4 次启动（手工串）」的设备侧差**：
GDN 层 ≈ `2×78.14 + 139.85 + 62.87 − 355.47` = **≈3.5 µs/层**；attention 层 ≈ `2×78.14 + 2.78 +
62.87 − 213.66` = **≈8.3 µs/层**（Avg 口径）。**结论（含口径说明）**：把一层的 hc#1 → 子层段 →
hc#2 → MoE **并成一次启动**，设备侧每层只省 **≈3.5–8.3 µs（占单层 1–2%）** —— 这说明**本形态下单次
启动/收尾的开销本来就很小**，时间几乎全在段内（hc 段 ≈131 µs/层 + 子层段 + MoE 段，见 M58-7）。
**这不是本 mission 的性能结论**（未做任何调优），而是「层间 handoff 走 GM 零拷贝 + 每层一次启动」
这条设计的**代价上界**：链**不比**手工串慢，且省掉了 host 侧的 4 次下发与 3 次同步。

**③ PLE 打断点的开关 A/B**（`M15_CHAIN_PLE=0/1`）：因为效应只落在 48 层里的 1 层，per-symbol Avg 会被
稀释 1/36 ⇒ 用**只用 2 层**的采集放大杠杆（层 0 = `MODE_MIX`、层 1 = PLE 层，两者**共用同一个符号**）：

| 采集（3 次重复，各 6 次启动） | 中位（µs） | 稳健 Avg | 剔除的污染样本 |
|---|---|---|---|
| `M15_CHAIN_PLE=1` | 351.17 / 352.55 / 352.64 | 350.26 / 347.56 / 347.67 | 1–2 / 6 |
| `M15_CHAIN_PLE=0` | 350.92 / 351.75 / 351.86 | 346.92 / 346.99 / 348.72 | 2 / 6 |

中位口径的 Δ（PLE 开 − 关）= **−0.25 / +0.58 / +0.69 µs**（每层摊到 1/2 ⇒ 该层实差 ≈ ±1.2 µs），
而同一配置的组内极差 ≈0.8 µs ⇒ **打断点的设备侧代价 ≈1 µs/层量级、在测量分辨率边缘**。
**这既是一个正面设计输入**（分段不重复工作：H1a+H1b 就是 H1 的两个相位，只多两条 mode-0 边界，代价
可忽略），**也是一条取证限制**：msprof **不能**用来见证「`pleBreak` 真的生效」（见 M65-3.2）。

**④ 与「两相位」形态的同会话对照**（`runs=all`，`evidence/m65_msprof_t1.csv`：两组符号在**同一个
CSV** 里 ⇒ 不受跨会话差异影响；链的 112 次启动与 A/B/C/M1/M2/H 的启动交错，故这一段的链 Avg 比
§1 的纯链采样略高，Min 更低）：

| 符号（`runs=all`，48 层） | Count | Min | 中位 | Avg |
|---|---|---|---|---|
| `m15_layer_kernel_gdn`（**两相位**，M40 形态） | 360 | 196.47 | 204.95 | 204.79 |
| `m15_layer_kernel_gdn_hc`（**四相位链**） | 112 | 302.81 | 354.23 | 353.82 |
| `m15_layer_kernel_attn`（两相位） | 120 | 60.93 | 62.47 | 62.81 |
| `m15_layer_kernel_attn_hc`（四相位链） | 38 | 147.35 | 213.03 | 211.90 |

⇒ **hc 段净增量 = +149.0 µs/层（GDN，Avg）/ +149.1（attention）**（Min 口径 +106.3 / +86.4）。
M58 归档的同会话 A/B 读数是 **+131.0 ~ +132.0 µs/层**（Avg 口径）；本轮的链 Avg（353.82）比
M58 的 hc 符号 Avg（336.7）高 ≈5%，两者都在共享卡上采（本节 §1 的纯链采样是 354.71–355.47 ⇒
与 353.82 自洽）。**这是设计输入，不是本 mission 的验收项**（未做任何调优）。


## M65-9 未完成项清单（**显式，不得含糊**）

1. **PLE 本体未实现**（ngram 表查找 / key·value 投影 / 门控 / 膨胀卷积）：表 **95.37 GiB**，
   容器 cgroup 32 GB、`/dev/shm` 16 GB、HBM 128 GiB 都装不下（docs/14 §10 第 1 条是**阻塞项**）。
   本轮的 `M15L_PlePhasePlaceholder()` 是**显式空操作**：它的输入（combine-only 物化的 `H'`）与输出
   （mix-only 读到的 `H'`）**逐字节相同**，语义是「PLE 缺席」，**不是**「等价于 PLE」。
   每次走这条路径都会打印一行 `[m15][DISCLOSE]`。
2. **attention 段仍是占位直通**（M25 起的已知项）：链上层 3/7/…/47 的 `attn_out` = `BLK_attn`
   （identity），QSA 未接入 ⇒ ④ 的官方对拍**只覆盖 hc 边界与末层 mixer**，attention 本体不覆盖。
3. **层 0 的 `MODE_MIX` 边界有 3 个张量超 m20 保守门限（T3 界内）—— 未归因到底**：读数与证据见
   M65-3.3。性质与 M58-9 第 2 条（m=33 越门限）同类。判据方向：若要主张「与 m20 同口径通过」，
   需按 docs/17 T3 逐步定位那几个跨界元素（本轮只定位到「上游 `xn` 的 1 个非良态元素」这一层）。
4. **MoE 段是 4 专家缩形档**（checkpoint 512 个 routed 专家的前 4 个），M25/M40 起就如此；
   因此链的出口数值**不是**官方 512 专家形态的数值，④ 的官方对拍只能挂在 hc 边界上。
5. **本轮的判据不能证明「`pleBreak` 生效」**（M65-3.2 的口径说明）：分段与单次路径按设计逐字节
   等价，输出判据抓不住「标志被忽略」；见证靠 msprof 开关 A/B（M65-8）与 `FLAG_SEQ[]` 结构。
6. **未做任何性能调优**：hc 段一字未动、GDN/MoE 段一字未动；M65-8 的数字是**现状**不是目标值。
   用户的目标（decode TPS）还剩 m20 §9.2/§9.3 的两条并行度问题（S2 的 item 数 = `m·HC`、S3 的
   down 只有 3 个 N-tile）**原样存在**。
7. **prefill（`T` 大）未接**：`m15_layer_kernel_{gdn,attn}_prefill` 仍是预留符号；本轮只做 decode
   档（m=1；`M15_CHAIN_M` 支持到 64，但 m>1 时 GDN 的状态语义是「多行共用一个序列状态」，
   链的判据是结构/字节级的）。

## M65-10 文件与证据（本轮新增/改动）

| 文件 | 状态 | 说明 |
|---|---|---|
| `m15_chain_host.h` | **新** | host：48 层链（`H_ChRunOnce`/`H_RunChain`）、PLE 打断点启动、末层 mixer、手工路径对拍、Ch 判据、M39 对拍 dump、全局 mixer 权重装载与来源判据 |
| `check_chain_ref.py` | **新** | 与 **M39 官方单层参考**（torch）对拍（**只覆盖层 0/1/3 的 hc 边界 + 末层 mixer；attention 段是占位直通、其语义不在对拍范围内**）+ 第二 oracle（m20 fp64）+ 舍入点负向对照 + 三态退出码 |
| `parse_msprof_chain.py` | **新** | M65-8 的汇总工具：按符号给 Min/中位/Avg/稳健量、每 step 合计、PLE 开关 A/B、链 vs 手工路径的每层差（只用标准库） |
| `m15_layer_kernel.h` | 改 | `LayerArgs::hcPleBreak`、相位 H1 的五段拆分（`M15L_PlePhasePlaceholder`）、`m15_final_mixer_kernel`、`FillHcPtrs` 的 handoff 例外、`*_hc` 签名 +1 参数 |
| `m15_hc_host.h` | 改 | `H_LaunchLayerHc` 参数化（层界三态 / yLayer / pleBreak 由调用方给） |
| `m15_layer_resources.h` | 改 | PLE 的两条边界 id、`FLAG_SEQ[]` 新增 H1a/PLE/H1b 段、`FlagMaxUse ≤ 6`、`HC_W_SLOTS`/`HC_WS_GMIX_SLOT`/`HW_ROLE_GM_*`、`ENTRY_FINAL_MIX` |
| `m15_layer_loop.asc` | 改 | `runs=chain` 分派、`M15_CHAIN_M`/`M15_CHAIN_PLE`、Ctx 新缓冲与计数器（`chainBoDev`/`phaseCh`/`guardsCh`）、加载第 49 槽、分账与 guard 分解行 |
| `slice_layer_manifest.py` / `weights_manifest.txt` | 改 | 新增 `GM_ROLES`（3 行，`layer=48`）；张量行 **1284 → 1287**，**删除 0 行**（`evidence/m65_manifest_diff.txt`） |
| `evidence/m65_accept_run_all.log` | 新 | `runs=all`：**2012 判定项 + 160 guard，0 FAIL** |
| `evidence/m65_chain_run48.log` | 新 | `runs=chain`（含手工路径）：**778 判定项 + 3 guard，0 FAIL** |
| `evidence/m65_chain_ref_run.log` | 新 | ④ 的三档（正常 / 缺输入 SKIPPED） |
| `evidence/m65_manifest_diff.txt` | 新 | manifest 纯增量证明（1284 → 1287，删除 0 行） |
| `evidence/m65_msprof_*.csv` + `evidence/m65_msprof.log` | 新 | M65-8 的原始读数与命令（含 PLE 开关 A/B、并发背景、以及 r2 的 **c2 补采**（修复后二进制的纯链采样，与 §1 的 7 次同口径）及其 `npu-smi` 快照证据） |
| `evidence/m65_neg_controls.log` | 新 | 负向对照：`check_chain_ref.py` 空目录 → rc=2；`M15_CHAIN_MANUAL=0` 的覆盖下降；**两组变异**（kernel 内 IJ handoff 变异 / 链上错平面变异）在 `/tmp` 副本里构建并确认**判据变红** |

---

---

# Part D · M82：attention 三套 cache + packed indices 的布局冻结（2026-09-27）

> 本 Part 是 **M33 清单 ★13 / docs/15 §5.3 ★13** 与 **docs/17 §7 裁决 ②**（"QSA 的 cache 填充必须
> 首期就做"）的第一段落盘。它**不改 attention 段的计算**（相位 A 仍是 `m15_attn_passthrough_body`），
> 交付的是：**唯一权威的物理布局头文件** + **跨 12 层的长寿 GM 分配** + **容量自检** + **paged /
> off-by-one 契约的实跑判据与两条负向对照**。
>
> 唯一权威 = **`m15_attn_kv.h`**（另有 device 探针 `m15_attn_kv_probe.h`、host 判据 `m15_attn_kv_host.h`）。
> 本节的表是**它的说明**；两者不一致时以头文件为准（头文件里有 `static_assert` 逐条钉住）。

## M82-1 三套 cache + packed indices 的物理字节布局（冻结）

| cache | 物理视图（逐维） | dtype | 每单位字节 | 层 stride（分配用） | 谁写 / 何时 |
|---|---|---|---|---|---|
| **主 KV**（paged） | `[blocks, 2, 16, 512]`（官方 `[blocks, H=2, N=16, C=512]`，C = K(256)‖V(256)）；页内再分 2 个 head 平面（各 16×512），**同一 head 内** token 步长 1,024 B（= K512+V512） | bf16 | 页 **32,768 B**；token **2,048 B**（2 头合计） | **8,421,376 B**（257 页 × 32,768 B） | 相位 A 前端，**逐 token 按 `slot_mapping` 落页**，**必须早于核心 gather** |
| **raw key ring** | `[cap=4, 140]`：列 `[0,128)` = raw k，列 `[128,140)` = 3 个 int64 MRoPE 位置（24 B） | bf16 | 行 **280 B** | **1,120 B**（4 行；每请求终生 1 物理块） | 相位 A pre-indexer，每 token 一行（`slot = pos % 4`）；**不是全历史 cache** |
| **compressed key cache** | `[blocks, 4, 1, 128]`（`block_size/4 = 4` 行/页） | bf16 | 行 **256 B**；**64 B / token-of-context** | **263,168 B**（1,028 行） | 相位 A pre-indexer，**每满 4 token 写 1 行** |
| **packed indices** | `[m, 2052]` int32，行主序；`[0,2048)` = `4b+子槽`（请求内绝对 token 位置）、`[2048,2050)` = open group 的 tail、末列 2051 = **有效数**（不是块数）、其余 `-1` | int32 | 行 **8,208 B** | **33,628,176 B**（4,097 行，**层间复用一块**） | 打分/topk/expand 段产出；核心消费 |

**主 KV 里最容易被写错的一点（已写成判据）**：`[blocks,2,16,512]` 的两个 head 是**两个 16,384 B 的
平面**（各 16 token × 512 元素 K‖V），而**同一 head 内** token 步长只有 **1,024 B** ⇒ 一个 token 的
2,048 B **不是**连续区间，写它必须**两次 1,024 B 传输**。若按"连续 2,048 B"写，第 2 个 1,024 B 会
落到 head0 的**下一个 token 槽**。
见证 = 判据 `Kv.lay.headsplit`（`Kv.neg.offbyone` 那一组里的 `H_CmpMustDiffer`）：它读 naive 落点
（`KvInBlockOffset(slot,0,0) + KV_TOKEN_STRIDE` = +1,024 B）必须**不等于** seed 的 head1（读数 PASS，见 M82-5）。

### 对齐约束表（**判据，不是注释**）

| 结构 | 字节 | `% 32` | 结论 |
|---|---|---|---|
| 主 KV 页 / head 平面 / **同一 head 内** token 步长 | 32,768 / 16,384 / 1,024 | 0 | 可直接 `DataCopy` / `Block1()` |
| 主 KV 层 stride、compressed 行（256）、层 stride（263,168） | — | 0 | 同上 |
| raw ring **整环**（1,120 = 35×32） | 1,120 | 0 | 整环一次拷合法（但语义不是我们要的） |
| raw ring **单行** | **280** | **24** | **非 32 B 整数倍 ⇒ 单行搬运必须 `DataCopyPad`**（`Block1(280)` 会 `Trap`：非整数块） |
| packed **单行** | **8,208** | **16** | 同上：**单行必须 `DataCopyPad`**（行按 16 B 对齐而已） |
| packed 整块（m 行连续） | — | 0 | 整块拷合法 |

⇒ 这三条（`280 ≡ 24`、`8,208 ≡ 16`、`1,120 ≡ 0 (mod 32)`）在头文件里是 `static_assert`，
在 host 里是 guard，并且**在 device 上被实跑见证**：探针用 `DataCopyPad` 完成 280 B 与 8,208 B 的
搬入/搬出，判据 `Kv.ring.row.L*`（n=280）与 `Kv.pack.rows.L*`（n=16,416 = 2 行）逐字节 PASS。

## M82-2 Q1 口径冲突的显式处理：raw ring 行宽 **140 vs 128**（带时点的设计决定）

**冲突事实（如实报）**：`docs/11-attn-analysis.md:25` 把 raw key cache 写成 `[tokens,128]`；
`docs/14:664-666` 明确纠正为"每请求一个环、`head_size=140`、1,120 B/请求/层"；而
`m19_qsa_indexer` 的**实现**是 `ring[4,128]`（无位置尾）。

**M82 的决定（时点 2026-09-27，本条**不是**恒真句）**：

> **默认按规范宽 `head_size = 140`**（128 raw k + 12 个 bf16 = 3 个 int64 MRoPE 位置尾，280 B/行）。
> 分配与所有编址**只用 140**；`RING_HEAD_SIZE_ALT = 128` 只用于"与 m19 的 128 形态对拍"时按元素比较。

理由与代价（逐条）：

1. **推导链已逐环核对（本次直接读 doner 仓库，2026-09-27）**：
   `QSAKeyStateCache`（`/workspace/vllm/vllm/models/qwen4_exp/common/qsa_cache.py:808-826`）里
   `storage_head_size = ceil(key_head_size/4)*4 + (cache_rope_positions ? 3*4 : 0)`；
   `cache_rope_positions = vllm_config.model_config.uses_mrope`
   （`qwen4_exp/nvidia/indexer_qsa.py:167`、`qwen4_exp/amd/indexer_qsa.py:129`）；
   `uses_mrope = _mrope_section(config) is not None`
   （`/workspace/vllm/vllm/transformers_utils/config.py:676-713`），它扫
   `rope_parameters.{mrope_section,xdrope_section}`。
   **本 checkpoint 的 `config.json` 有 `text_config.rope_parameters.mrope_section = [11, 11, 10]`**
   ⇒ `uses_mrope = True` ⇒ `cache_rope_positions = True` ⇒
   `storage_head_size = ceil(128/4)*4 + 12 = 128 + 12 = **140**`。
   （一致性旁证：`mrope_section` 有 3 段 = 3 个轴（与 `_NUM_ROPE_AXES = 3` 吻合）；
   `partial_rotary_factor = 0.25` ⇒ 主 attention rope 维 = 256×0.25 = **64**，与 specs 的 RoPE(64) 吻合。）
   **口径**：这是**源码推导**，**没有**跑 vLLM 去观测那块 cache 的实际 shape（无运行期见证）。
2. **文本-only 下 128 在算数上确实够**：组首位置恒等于 `p-3`，那 3 个 MRoPE 位置**可由位置号推出**
   —— 这正是 m19 现在只存 128 的直接原因（`m19 README:101-102`）。注意 doner 的 128 也不是"裸 128"，
   而是 `ceil(128/4)*4`（按 4 元素向上圆整）。
3. **140 是"不会挡路"的默认**：`140 > 128` ⇒ 只写 128 的实现**不需要改 stride**，仍落在同一块已分配
   的行里（前 256 B/行）。反向（先按 128 分配、后发现要 140）需要**重分配整环**，而环是**跨层长寿**
   的状态 ⇒ 返工面大。
4. **仍留给后续的一条**：若将来要跑**文本-only 且明确不要位置尾**的形态，须显式记一次"退回 128"的
   决定（改 `RING_HEAD_SIZE` 与两处 stride，不涉及主 KV/compressed 的布局）。

## M82-3 paged 契约定死（docs/15:528-531）

- **所有地址都经 `m15_attn_kv.h` 的式子算**：`KvBlockOf/KvSlotInBlock/KvInBlockOffset`（§1b 的宏，
  host 与 device 共用同一份文本）+ `KvPhysBlock = block_table[req][logical_block]`。
- **单请求下 `block_table` 为恒等**（`block_table[r][b] == b`），但这只是**恒等表这一特例**，
  **不是另一套布局**；连续布局 = `KvByteOffsetContig()`，paged = `KvByteOffsetPaged()`。
- **判据（T-KV-PAGED）**：对同一 token 位置，**paged 地址写**、**连续地址读**，读到的
  2×256 维向量**逐字节相同**。实跑：`Kv.kv.paged.L{0,1,2,3,4,7,17,300,4095,4096}` 全 PASS（n=1,024）。
- **负向对照（证明判据有判别力）**：给**非恒等** `block_table`（交换物理页 1↔2）⇒ **两条一起给**：
  ① **host 算式**：`KvPagedContractHolds()` 必须**不成立**（guard：恒等 true / 交换 false）；
  ② **device 字节**：用交换表**真启动一次探针**，host 按连续地址读回的内容必须**不等于** seed
     —— 判据行 `Kv.neg.paged`（`H_CmpMustDiffer` PASS；它在 `runs=kv` 的输出里可见）。
  两条缺一不可：① 只证明算式不同、② 才证明 device 真的读了 `block_table`。
  **口径注（如实）**：r1 复审指出本节的 ② 当时**只有 host 算式比对、没有 device 启动**（它在
  冻结 tip `03bf0ff` 上复跑确实看不到 `Kv.neg.paged` 这一行）——**那条判 findings 属实**；
  M82 r2 补上了启动与判据，现在 ① ② 都在 `runs=kv` 里可复现。
- 为什么现在就要按 paged 写死（而不是"先连续、以后再改"）：`docs/15:528-531` 的要求，且 prefill 与
  decode **必须写同一份 cache**；接口一经固化，先连续后改会把 12 层的状态布局全部返工。

## M82-4 off-by-one 契约（本 mission 的命根子）与容量账

**写入门控**（`M15KV_COMP_ROW_WRITTEN`，与 m19 的 `closes` 同源）：

```
压缩行 = (pos + 1) % 4 == 0   ⟺   pos ≡ 3 (mod 4)   ⟹   pos ≥ 3
```

⇒ 文档里的「且 p ≥ 3」是**被蕴含的条件**（`uint32` 无下溢路径），不是额外限制。
⇒ **pos 4096：`(4096+1)%4 = 1` ⇒ 属"开放组" ⇒ 不写 compressed**，只作 causal 尾部 token
（`docs/17:286`、`docs/11:158`）。4097 上下文下压缩行出现在 **3,7,…,4095 共 1,024 行**。
⇒ 可见块数上界 **不含**正在累积的 open group：`visible_blocks = max(0, min((p+1)//4, seq//4))`
（`m19 README:121-123`、`V-C:qsa_cache.py:268-275`）。

**容量账（`runs=kv` 每次打印；数值由"只用规范原始数字"的独立复算得出）**：

| 量 | prefill m=4097 单序列 | decode m=1 / ctx=4097 |
|---|---|---|
| 主 KV 页数 / 字节 | 257 页 × 32,768 = **8,421,376 B/层** | 257 页 × 32,768 = **8,421,376 B/层** |
| compressed 容量行 / 字节 | `ceil(4097/16)*16/4 = 1,028` 行 × 256 = **263,168 B/层** | 1,028 行 = **263,168 B/层** |
| compressed **实际写入**行 | **1,024 行**（位置 3,7,…,4095）| 1,024 行 |
| raw ring | 4 行 × 280 = **1,120 B/层** | 同左 |
| packed | 4,097 × 8,208 = **33,628,176 B**（**层间复用一块**） | 1 行 = 8,208 B |
| 12 层合计 | 主 KV **101,056,512 B（96.38 MiB）** + compressed **3,158,016 B（3.01 MiB）** + ring **13,440 B** + packed 33,628,176 B（32.07 MiB） | — |

> **decode 口径（M178 算术 ⇒ M184 已修）**：M178 核出旧 `m15_attn_kv.h:192` 的 `DECODE_CTX = 4096`
> ⇒ `DECODE_BLOCKS = BlocksFor(4096) = 256`（页分配 `BlocksFor(m) = ceil(m/16)`，`:195`）。但
> 4097-token prefill 覆盖位置 0..4096（token 4096 落在 0-based 第 256 页）⇒ 需 `ceil(4097/16) = 257` 页，
> 旧值**少 1 页**。**M184 已修**（M175 合入、`.asc` 释放后）：`DECODE_CTX = 4097`（`:192`）⇒
> `DECODE_BLOCKS = 257`（`:202`）、`DECODE_COMP_ROWS = 1,028`（`:203`）；`m15_layer_loop.asc:4856` 的
> guard 改为 `DECODE_CTX == PREFILL_M`。**分配侧不受影响**（主 KV 容量由 `PREFILL_BLOCKS = 257` 定尺，
> `KV_LAYER_STRIDE = 257 × 32,768`，`:211`）。设备读数：`runs=kv` 的 decode 打印由
> `m15_layer_loop/m15_attn_kv_host.h:155-164` **从 `DECODE_CTX` 派生**，改前 `256 页 = 8388608 B/层`、
> 改后 `257 页 = 8421376 B/层`（`ALL PASS（checks=215, guards=191, fails=0）` 两次）；`runs=prefill`
> 打印 `DECODE_CTX=4097 ⇒ 257 页/层`（读数见 `m15_layer_loop/evidence/m178_kv_geom/`）。

**为什么 packed 层间复用一块**：12 个 attention 层在流内**串行**、host 逐层消费；若按层常驻需要
12×32.07 MiB = 385 MiB（本 mission 不做）。**若**将来要 12 层并发消费，这是必须改的一处。

## M82-5 长寿 GM 分配、容量自检与判据读数

- **按"第 k 个 attention 层"编址**：`m15_loop_layout.h` 新增 `AttnSlot(layer)` —— attention 层返回
  `layer / ATTN_INTERVAL` = 0..11（对应 0-based 层号 3,7,…,47），GDN 层返回 `NO_SLOT`。
  它与 GDN 的 `LayerSlot()` **是两条独立映射**（36 vs 12，混用会静默错位）。`LayerSlot(3)` 仍是
  `NO_SLOT`（M25 语义未变，判据 `Kv.align` 的 guard 见证）。
- 四块长寿平面在 `H_Alloc` **host 一次分配**（与 `chainBoDev` 同款），尺寸**全部取自
  `m15_attn_kv.h` 的 `*_PLANE_BYTES`**（不在 `.asc` 重算）；初值全零（它们是跨 token 常驻状态）。
- **容量自检 = 三件事**（`runs=kv`）：
  1. **独立复算**：只用规范原始数字（16 / 2 / 256 / 4 / 140 / 2052 / 4097）重算每层/每平面字节数，
     与头文件常量逐条 `H_Guard` 核对（**9 条**：`H_KvCapacityCheck` 的 9 个 `H_Guard`）。打印见 M82-9 的证据行。
  2. **层槽边界 round-trip**：4 个平面 × 各层槽的**首字节 + 尾字节**写 8 B 标记（共 74 个位置），
     **全部写完再统一读回**逐字节比对 —— 同时证明"第 k 槽可寻址"与"写第 k 槽不踩第 k+1 槽"。
     判据 `Kv.cap.rt.{main,comp,ring,pack}`（192/192/192/16 B）全 PASS。
  3. **实测打印**：`H_Alloc` 打印四块平面的分配尺寸与逐层算术（主 KV 12 层 × 8.421 MB/层 = 257 页/层
     × 32,768 B …），与上表逐项一致。
- **判据清单与读数**（`runs=kv`，真实 checkpoint 档）：**M82 时点为 56 条判定项 + 130 条 guard，0 FAIL**
  （`evidence/m82_runs_kv.log`）；**现 tip 的同一条命令是 83 条判定项 + 142 条 guard，0 FAIL**
  （M98 的 `Ac.*` 段 26+9 随 M98 合入、M178 之后；读数见 `evidence/m178_kv_geom/logs/m178_runs_kv.log`，
  总量 `checks=215 / guards=191`）。下面按 **M82 时点**分解（按函数实测，不给粗略近似）：
  - 判定项 **56** = `Kv.cap.rt.*` **4**（4 个平面各 1 条）+ 每个 pos 的 **5** 条
    （`Kv.kv.paged` / `Kv.ring.row` / `Kv.comp` / `Kv.pack.rows` / `Kv.flag`）× 10 个 pos = **50**
    + `Kv.lay.headsplit` **1** + `Kv.neg.paged` **1**（device 侧的 paged 负向对照）；
  - guard **130** = **容量独立复算 9**（`H_KvCapacityCheck`）+ **对齐/层槽编址 115**
    （`H_KvAlignAndSlotCheck`：32 B 对齐 11 + ring 第 1/2/3 行 3 + packed 第 1 行 1 +
    head 平面 ≠ 页内 token 步长 1 + 层循环（36 GDN × 1 + 12 attention × 5 = 96）+ 收尾 3）
    + **负向对照 6**（`H_KvNegOffByOne` 4 + `H_KvNegPaged` 2）。
- **两条负向对照（docs/17 §4 硬要求）**：
  1. **off-by-one 注入**：探针的 `mode = MODE_BROKEN_OFFBYONE` 让压缩行**无条件写**。对 pos 4096
     跑**同一条判据**：契约档成立（压缩行**未**写，读回全零）、注入档**被打破** ⇒ 判据对 off-by-one
     有判别力（打印行 `Kv.neg.offbyone.L4096`，两条 `H_Guard`：契约成立 + 注入不成立）。
  2. **paged 非恒等表**：见 M82-3。
- **pos 4096 的实跑读数**：`Kv.comp.L4096` PASS（gate=0，压缩行未写、读回全零）、`Kv.flag.L4096`
  PASS（`gate=0 g=1024 slot=0`）—— 与 `docs/17:286` 的契约一致。

## M82-6 显式未完成项（**不得含糊**）

| # | 未完成项 | 现状 | 卡在哪 |
|---|---|---|---|
| 1 | **attention 前端 prolog**：`qkv_proj` bf16 GEMM、`q/k` RMSNorm(256, eps 1e-6)、partial RoPE(64)、gate split；indexer 路 `index_qk_proj` + indexer q 的 GemmaRMSNorm(128)/RoPE(64) | **未实现** | **Q3 是硬骨头**：主 attention 的 q/k RMSNorm256+RoPE(64) 在 `docs/11 §1.1` **只有规格、仓内没有已验证的 Ascend 实现**，与 indexer 的 GemmaRMSNorm(128) 是两条路径 ⇒ 必须分别取证。可抄改资产：`m19_qsa_indexer` 的 S0/S1/S2、donor `posembedding/kv_rms_norm_rope_cache` |
| 2 | **cache 填充的数学**：raw k → ring；每 4 token 的 `mean 4 raw k → GemmaRMSNorm → RoPE@组首(p-3)` → compressed | **寻址/门控已实现并实跑**（M82-4），**数学未实现** | 依赖第 1 项（要有 raw k / pooled 的**已验证**实现才能谈它的正确性）。本 mission 消费的是"已算好的 raw k / compressed 向量" |
| 3 | **主 KV 的 KV 向量本身**（`v_proj` 输出落页） | **落页寻址已实现并实跑**，向量未接 | 同第 1 项（v_proj 未实现） |
| 4 | **接口接线**（`LayerArgs` 加 attention 权重/cache/pos/layer、AIC 侧 attention arm、入口宏、资源表 flagId 重规划） | **未做**（本 mission 边界） | 见 M82-7 的接口核算；M76 §6 已列逐项清单 |
| 5 | **`weights_manifest.txt` 的 attention 权重 role** | **已做（M82）**：`slice_layer_manifest.py` 新增 `ATTN_ROLES`（9 个 role × 12 层 = 108 行），manifest **张量行 1287 → 1395**（`wc -l` 1291 → 1399）、**删除 0 行**、`--check` 通过（`evidence/m82_manifest_diff.txt`） | 剩下的只是**消费侧**：host 尚未读这些 role（属第 4 项接口接线） |
| 6 | **attention 段的独立数值参考**（自建稠密 causal） | **未做** | 首期验收基准；**本基准不是官方行为**（官方 QSA 每 token 只 attend 约一半历史，`docs/17:281`） |
| 7 | 打分 / topk / expand（packed 的**选择语义**） | **未做** | 本 mission 只判 packed 的**字节布局/行距**，不判它选了谁 |

**要新增的 attention 权重 role（9 个，全部 `self_attn.*`；**M82 已落进 manifest**，checkpoint 实测）**：
`attn_q_proj [12288,2560]`、`attn_k_proj [512,2560]`、`attn_v_proj [512,2560]`、
`attn_o_proj [2560,6144]`、`attn_q_norm [256]`、`attn_k_norm [256]`、
`attn_idx_qk_proj [640,2560]`、`attn_idx_q_norm [128]`、`attn_idx_k_norm [128]`。
（现在 manifest 里 attention 层的权重 role 数 = **0**：M76 实测，本 mission 复核仍为 0。）

### 哪些**旧**判据在真 attention 接入后不再有判别力

现在相位 A 的 attention 段是 `m15_attn_passthrough_body`（`y = x`）。由此：

| 判据 | 现状 | 换真 kernel 后 |
|---|---|---|
| `Ch.sub.L{k}`（k = 12 个 attention 层的层号） | **自比**：链上的 attention 段与手工路径的子层段**都调同一个 `m15_attn_placeholder_kernel`**（`m15_chain_host.h` 的手工路）⇒ 这条对 attention 层**没有判别力** | 变成"同 kernel 两次启动的一致性"判据（**仍不是**独立数值参考） |
| `Ch.nonid.L*` / `Ch.moe.L*` / `Ch.blks.L*` / `Ch.finite.L*` | **仍有**判别力（attention 占位是恒等 ⇒ 变化来自 hc(mlp) 与 MoE 段） | 不变 |
| `A'`（M25 的"attention 占位直通"行，§5.5） | 判的是占位本身的字节直通 | 该行的对象消失，条目应删除或改写 |

⇒ **换真 kernel 时必须同时引入独立参考**（自建稠密 causal；M76 §4 已写入计划），否则 attention 段的
数值正确性只有"自比"这一层。

## M82-7 二进制指纹重基线（**旧值已过期，勿当"当前值"**）

| 项 | 旧值（M65 归档） | 新值（M82） | 时点 |
|---|---|---|---|
| `m15_layer_loop` 二进制 sha256 | `07efbf0364dfd4ceca2f5fcfea36a5e3bc1d30fc44939706bdb302fa036d356d` | `c5268d215d26bf75e3e9bc0f0d0a2a46336bb4fa2da2f8e91f78ecda140d17be`（M82 r2；r1 的 `285923ad…` **已被 r2 取代** —— r2 在 host 侧补了 paged 负向对照的 device 启动） | 2026-09-27（M82 r2 落盘，base = main @ `d6e405d`） |
| 归档日志头（4 份） | `m65_accept_run_all.log` / `m65_chain_run48.log` / `m65_chain_ref_run.log` / `m65_msprof.log` 的头写着旧 sha256 | **未改这 4 份归档**（它们是 M65 那一刻的历史见证，改头会伪造历史） | 同上 |

**口径**：M65 的读数在 M82 的构建上**逐项复现**（`runs=chain` = 778 判定项 + 3 guard；
`runs=all` = 2012 判定项 + 160 guard；均 0 FAIL），但**二进制不同** ⇒ 引用 M65 归档时必须写明
"M65 归档的 sha256 对应 M65 的构建；M82 的重基线见 `evidence/m82_*.log` 的头"。
**新增证据**：`evidence/m82_runs_kv.log`（`runs=kv`）、`evidence/m82_accept_run_all.log`（`runs=all`）、
`evidence/m82_regress_chain.log`（`runs=chain` 零回归对拍）。

## M82-8 接口核算：README §8.4 的「零改动」声明**不成立**（M76 已证伪，本 mission 复核）

`README` §8.4（旧「已知限制」第 4 条）曾写"真 QSA 接入时**只替换 `m15_attn_passthrough_body`**，
层循环与 MoE 段零改动"。**M76 已证伪**，本 mission 复核确认：

- 调用点确实**只有一处**（`m15_layer_kernel.h` 的 `M15L_FusedBody` 里 `m15_attn_passthrough_body(subIn, subOut, HIDDEN*2u*A.m)`）；
- 但**入出参远远不够**：现签名只有 `(xIn, yOut, bytes)`，`LayerArgs` 里**没有**层号、qkv/o/gate/indexer
  权重指针、KV/raw ring/compressed/packed 指针、`pos`/`slot_mapping`；**AIC 侧对 attention 完全缺席**
  （`if constexpr (KIND == KIND_GDN)` 之外没有任何 attention 分支）。
⇒ 真实接入要改接口面（M76 §6 的 8 项）。**本 mission 没有硬塞**：不改 `LayerArgs`、不改入口宏、
不动资源表的 flagId（那是接口接线 mission 的范围）。

## M82-9 文件与证据（本轮新增/改动）

| 文件 | 状态 | 说明 |
|---|---|---|
| `m15_attn_kv.h` | **新** | **唯一权威**：三套 cache + packed 的物理布局、§1b 的寻址宏（host/device 共用）、paged 契约、off-by-one 门控、容量常量 + `static_assert` |
| `m15_attn_kv_probe.h` | **新** | device 探针（`__mix__(1,2)`，单 AIV0）：paged 写→连续读、raw ring 280 B 行、compressed 门控（两档 mode）、packed 两行 8,208 B；UB 窗 [128 KB, 138 KB) |
| `m15_attn_kv_host.h` | **新** | host：容量独立复算、对齐/层槽判据、74 位置边界 round-trip、pos 扫描的逐字节判据、**两条负向对照** |
| `m15_loop_layout.h` | 改 | 新增 `AttnSlot()`（第 k 个 attention 层）+ attention cache 平面在 GM 平面图里的说明 |
| `m15_attn_layer.h` | 改 | 文件头：把悬空的 README 章节引用改成按符号引用，并写明 M76 对「零改动」声明的证伪（M82-8/M82-10） |
| `slice_layer_manifest.py` / `weights_manifest.txt` | 改 | 新增 `ATTN_ROLES`（9 个 role × 12 个 full_attention 层 = 108 行）；张量行 **1287 → 1395**、删除 **0** 行、`--check` 通过 |
| `evidence/m82_manifest_diff.txt` | 新 | manifest 纯增量证明（1287 → 1395，删除 0 行） |
| `m15_layer_loop.asc` | 改 | Ctx 新增 4 块长寿平面 + `block_table`/seed/读回缓冲；`H_Alloc` 按 `m15_attn_kv.h` 的尺寸分配并打印容量；`runs=kv` 分派；判据分账新增 `Kv` |
| `evidence/m82_runs_kv.log` | 新 | `runs=kv`：**56 判定项 + 130 guard，0 FAIL**（含容量复算打印、两条负向对照（各含 host 算式 + device 字节两面）、`Kv.lay.headsplit`） |
| `evidence/m82_accept_run_all.log` | 新 | `runs=all` 全量回归：**2068 判定项 + 290 guard，0 FAIL**（= M65 归档的 2012/160 **逐项不变** + 本 mission 的 Kv 56 判定项 + 130 guard） |
| `evidence/m82_regress_chain.log` | 新 | `runs=chain`：**778 判定项 + 3 guard，0 FAIL**（与 M65 归档逐项一致 ⇒ 36 个 GDN 层 + 48 个 MoE 段的出口未回退） |

## M82-10 顺手收口两条同目录引用缺陷（M76/M77 勘查发现的 finding，未单独立项）

| # | 缺陷 | 处置 | 为什么以后不再漂 |
|---|---|---|---|
| ① | `m15_attn_layer.h` 的文件头注释引了一个**现已不存在的 README 章节名**（原文是「README §〈那个章节名〉」） | 改为**按符号/结构**引用：调用点 = `m15_layer_kernel.h` 的 `M15L_FusedBody`；并把"M76 证伪的零改动声明"写在注释里（见 M82-8） | 引的是**符号与函数名**，不是 README 的章节标题或行号 |
| ② | `README` §8 第 1 条引 **`m15_moe_resources.h:139`** 的 `UB_RT_WF` —— 那是**行号**引用；该常量在本文件里其实在 `m15_moe_resources.h` 的**另一行**（`m15_moe_resources.h` 中 `UB_RT_WF` 的定义行随本文件历史增删而漂移），而上游 `m13_moe_layer/m13_resources.h` 的 `UB_RT_WF` **恰好在 :139** ⇒ 这是「副本加了注释后行号漂移」的典型 | 改为**按符号引用**（`m15_moe_resources.h` 的 `UB_RT_WF = (NUM_EXPERTS+1)*HIDDEN*4`），并在括号里写明"原引 :139 的来历" | 常量名在两端都稳定；行号只在原文件里才成立 |

**核对命令（本轮实跑，LC_ALL=C / C.UTF-8 / en_US.UTF-8 三个 locale 下读数相同）**：

| 扫描 | 读数 | 说明 |
|---|---|---|
| `grep -cE "^#+ .*两个占位点" m15_layer_loop/README.md` | **0**，rc=1 | **判据**：名字为「两个占位点」的**章节**在 README 里不存在（缺陷 ① 的实质） |
| `grep -c "两个占位点" m15_layer_loop/m15_attn_layer.h` | **0** | **判据**：`m15_attn_layer.h` 里的那条悬空引用**已改掉** |
| `grep -c "两个占位点" m15_layer_loop/README.md` | **不给数字** | **不**作为判据，也**不**在这里报数：本节（M82-10）自述这条缺陷时必然把该短语写回 README ⇒ 这个计数**只反映本节自己写了几次**，且**随本节文字而变**（可复核的例：在 r1 冻结 tip `03bf0ff` 上该命令返回 **2** —— `git show 03bf0ff:m15_layer_loop/README.md \| grep -c "两个占位点"`；r2 改写本节后该值又变）。这正是 M82 r1 复审 P2-2 抓到的自指形态。**可复现的用法**：只报上面两条**带作用域**的扫描 |

（口径：**能带时点/带作用域的计数就给**；但**自指的计数**既不稳定也不能证明任何事，故不写。
本节不含「已全部 / 无残留」类断言。）
另外：`grep -rn "UB_RT_WF" m15_layer_loop/m15_moe_resources.h m13_moe_layer/m13_resources.h`
（各自命中定义行；README 现按符号引用）。

---

## 1. 融合 kernel 的段序（一次启动内的两个相位）

| 相位 | 段 | 算子 | 核 | 来源 |
|---|---|---|---|---|
| **A** | GDN S1 | Add+RMSNorm #1（层输入 + 零残差） | AIV 行切分 | m15_gdn_layer.h（← m14） |
| A | GDN S2 | in_proj bf16 GEMM（K=2560 N=16480） | AIC N 条带 | 同上 |
| A | GDN S3 | conv1d+SiLU+q/k l2norm+gating | AIV block 条带 | 同上 |
| A | GDN S4 | 递推（ssm_state in-place RMW） | AIV head 条带 | 同上 |
| A | GDN S5 | RMSNormGated | AIV head 条带 | 同上 |
| A | GDN S6 | out_proj bf16 GEMM（K=6144 N=2560） | AIC N 条带 | 同上 |
| A | GDN S7 | Add+RMSNorm #2（残差 = S1 的 fp32 res1）→ **相位 B 的输入** | AIV 行切分 | 同上 |
| A' | attention 占位 | `y = x`（逐字节；QSA 未通） | AIV 块条带 | m15_attn_layer.h |
| — | **相位边界** | `PipeBarrier<PIPE_ALL>` + 全体 AIV mode-0 barrier | 两侧 | m15_layer_kernel.h |
| **B** | MoE S1 | Add+RMSNorm（输入 = 相位 A 出口，残差 = 零） | AIV 行切分 | m15_moe_layer.h（← m13） |
| B | MoE S2 | router：GEMV → softmax(max-shift) → top-k → renorm；同段算共享门 | AIV0 | 同上 |
| B | MoE S3 | 路由索引生成（计数排序） | AIV0 | 同上 |
| B | MoE S4 | permute（四级流水 gather） | AIV 行切分 | 同上 |
| B | MoE S5 | A 侧 MXFP4 全 VEC 量化（routed + shared） | AIV 行切分 | 同上 |
| B | MoE S6 | grouped gate_up GEMM（5 槽位 = 4 routed + 1 shared） | AIC | 同上 |
| B | MoE S7 | SwiGLU + MXFP4 量化 | AIV 行切分 | 同上 |
| B | MoE S8 | grouped down GEMM（5 槽位） | AIC | 同上 |
| B | MoE S9a | unpermute 加权折叠（topK 模板 1..4） | AIV 行切分 | 同上 |
| B | MoE S9b | combine：`shared = bf16(fp32(sigmoid(sgate)·Y_shd))`、`moe = bf16(fp32(routed + sigmoid·Y_shd))` | AIV 行切分 | 同上 |
| B | MoE S10 | Add+RMSNorm（残差 = S1 的 fp32 res1）→ **层出口 `yLayer`** | AIV 行切分 | 同上 |

**专家槽是紧凑的（Σt_e，tower 对 M40 的硬约束）**：MoE 段的 A/scale/GU/H/Y 都按
「Σt_e 紧凑布局」寻址 —— 第 e 个专家的行起点 = S3 产出的 `expert_offsets[]` 前缀和
（`off[e]`），行数 = 它的 `t_e`；两组 grouped GEMM 的工作项是 **(expert, mTile, nBlock)**
（空专家由 counts 跳过；mTile 数由 `t_e` 推出，本实例 `t_e ≤ BASE_M ⇒ mTiles ≡ 1`）。
**段内没有任何「每专家固定 M_MAX 行 / 定长 topk2 / 编译期 4」的布局假设**：
行数与循环上界由 `counts`/`active_num` 驱动（`VecQuantStage` 的 `ROUTED` 形参、
GEMM 的 `countsGm.GetValue(slot)`），`NUM_EXPERTS`/`TOPK_MAX` 只作为规模参数出现在
缓冲区尺寸与模板实例上。见 §5.7（改造清单 + 证据 + 尚未交付的部分）。

**层链数据流**：`xLayer →（相位 A）→ yLayer →（相位 B）→ yLayer`。MoE 段跑的是 m13 的
`sliceMode=1`「层链模式」：S1 对相位 A 的出口做 `Add+RMSNorm(y + 0)` 得 `x_norm`，
router / permute / 共享专家量化都消费 `x_norm`（m13 已验收的同一条路径）。

**`yLayer` 是相位 A 的出口、也是相位 B 的输入与出口（in-place）**：MoE S1 读它、S10 写回它。
这在 m=1（以及 m>1：S1 逐行读、S10 末尾写）下都安全，由 MoE 段自身的段序保证；副作用是
**启动结束后 yLayer 里是 MoE 段的输出**，相位 A 的出口值不可再从 yLayer 观测——这一点的
验证处置见 §5.2。

### 1.1 相位边界的同步为什么是这两条

1. **`PipeBarrier<PIPE_ALL>`（核内）**：两段的 UB 段窗、L1 区、L0C 区、BufferID 编号都是
   **同址复用**（§2）。BufferID 是核内 token，跨相位复用只有在「上一相位任何 pipe 都不再
   持有该 UB 区」时才安全 —— 这就是 docs/12 §4「段窗叠放以段间 barrier 为前提」的显式版。
2. **全体 AIV mode-0 barrier（`CrossCoreSetFlag<CC_MODE0, PIPE_MTE3>` + `wait<CC_MODE0, PIPE_MTE2>`）**：
   相位 A 的 `yLayer` 由**全体 AIV 协作写**（GDN S7 的 m6 行切分 / attention 占位的 512B 块
   条带划分），而相位 B 的 S1 由另一套行切分读它 —— 跨核可见性必须用 CrossCore
   （BufferID 管不了跨核，docs/05 §6.1）。set 挂 MTE3（排空本核写）、wait 挂最窄的 MTE2
   （挡本核后续读），于是「本核写完成 → 全体到齐 → 本核才开始读」三条同时成立。
   不挂 PIPE_S：这里的 wait 后面**不紧接** CrossCoreSetFlag，不触发 docs/05 §2 的
   「链式 wait→set 必须挂 PIPE_S」例外。
3. **AIC 侧不需要相位边界同步**：AIC 在相位 A 与 B 之间没有数据依赖（MoE 的 A 操作数由 AIV
   产出，走 MoE 段自己的 mode-2 交接）；核内只需第 1 条的 PipeBarrier。

## 2. 资源表（`m15_layer_resources.h` 是融合形态的唯一登记处）

登记制（docs/05 §5.2）：BufferID / CrossCore flagId / UB / L1 / L0C / GM 偏移**全部编译期
静态常量**，模块只声明 footprint、不自持有内存。两段各自的表保持原样
（`m15_gdn_resources.h` = m14 的表、`m15_moe_resources.h` = m13 的表，只重编 flagId），
**融合表只新增「跨段共享的那些平面」**（ws 顺排切分、MoE 权重槽、全局 flagId/BufferID 命名空间图）。

### 2.1 UB 预算（两相位**同址叠放**，峰值取 max 而不是求和）

| 相位 | 窗 | 起点 | 峰值结束地址 | 内容 |
|---|---|---|---|---|
| A（GDN） | PERSIST + 四段窗顺排 | @0 | **227328** | gamma bf16 staging + 3 份 gamma fp32 预转 + m6/m9/m12/m4 四个互不重叠的段窗 |
| A（attention） | 与 GDN 同址 | @0 | ≤ 64256 | 只有 5120B 的搬运缓冲（复用 `UB_M6`） |
| B（MoE） | PERSIST + 段窗 | @0 | **193600** | `UB_ZEROS_B16`(16KB) + gamma1/gamma2 fp32 预转 @16384/@26624 + router/m8/m5/m3 段窗 |
| **融合峰值** | | | **227328 ≤ 253952（248KB）** | 余 **26624 B** |

两相位的窗**故意重叠**（相位 B 的 PERSIST 会覆盖相位 A 的 gamma2_f32 区、相位 B 的段窗与
相位 A 的段窗几乎完全重叠）：这是 docs/12 §4 的「段窗同址叠放、以段间 barrier 为前提」，
合法性由 §1.1 的第 1/2 条保证 —— **没有任何张量跨相位存活**（相位 A 的 gamma 只在 A 内被读；
相位 B 的 gamma 只在 B 内被写与读）。`m15_layer_resources.h` 里的
`static_assert(UB_PEAK_FUSED <= UB_TOTAL_BYTES)` 是编译期兜底。

### 2.2 L1 / L0C 预算（同样按相位取 max）

| 资源 | 相位 A | 相位 B | 融合峰值 | 硬限 |
|---|---|---|---|---|
| L1（A/B 区） | 303104 B（B 区 @262144 ×2 组） | 296960 B | **303104** | 524288 B |
| L0C（CO1） | 40960 B（64×160 fp32） | 65536 B（64×256 fp32） | **65536** | 262144 B |
| L0A/L0B | 8KB+20KB（BASE_M=64,BASE_K=64,BASE_N=160） | 4KB+16KB（BASE_K=128,BASE_N=256） | 两段各自 ≤ 32KB/32KB | 64KB/64KB |

### 2.3 GM 平面切分

| 区 | 内容 | 规模 |
|---|---|---|
| `ws[0, 5,258,240)` | GDN 段 ws（13 个中间张量，M15G 的偏移表） | 5.26 MB |
| `ws[5,258,240, 12,943,712)` | MoE 段 ws（32 个中间张量，M15M 的偏移表） | 7.69 MB |
| MoE 权重区 | 48 层 × 13,091,840 B（层号即槽号；attention 层也有 MoE） | 628.4 MB |
| GDN 权重区 | 36 层 × 115.95 MB（M25 原样） | 4174 MB |
| 状态区 | 36 层 × 3.21 MB（conv_state + ssm_state，M25 原样） | 115.5 MB |

**为什么 MoE 权重单独一个区**：attention 层**没有** linear_attn 权重（M25 刻意不给它们占槽，
省 12×116MB），但它们**有** MoE 权重 → 两套槽表按层号各自编址（MoE 槽号 = 层号 0..47）。

### 2.4 CrossCore flagId 登记表（唯一权威；相邻性由编译期断言逐对核对）

每核 16 个 id（docs/05 §6.1：本项目不用 SyncAll/高阶 API → 0-15 全可用）。
核心规则 = **同一核上相邻的两个同步点必须不同 id**。融合后的分配（按 (核型, mode) 子空间）：

| 段 | AIV mode-0 | AIC mode-0 | mode-2（AIC ↔ 配对 AIV） |
|---|---|---|---|
| GDN | 10 → 8 → 9 → 11（S1 后 / S3→S4 / S4→S5 / S5→S7） | 12,13,14,15（in_proj 前/后、out_proj 前/后） | 4,5,6,7 |
| **相位边界** | **8**（复用 GDN 的 S3→S4 id；同核同 mode、其间还隔着 GDN 的 9/11 两次 barrier → 已 drain） | —（无依赖，见 §1.1 第 3 条） | — |
| MoE | 12 → 13 → 14 → 15 → 12（S1 后 / S3 后 / S4 后 / S9a 后 / S9b 后，4 槽轮转） | 8,9,10,11（GateUp 前/后、Down 前/后） | 0,1,2,3 |

**除相位边界有意复用 GDN 的 AIV mode-0 id 8 外（上表 `reuse=true` 那一行），两个段在每一个
(核型, mode) 子空间里占的都是互不相交的 id 集合**（GDN 用不到的 AIV
12-15 给 MoE 的 mode-0 ring、AIC 用不到的 8-11 给 MoE 的 mode-0 ring、两侧都用不到的 0-3 给
MoE 的 mode-2 ring）→ **全程不存在「同一 flagId 在同一核上跨模式复用」**，因此不必援引
docs/05 §2 那条较弱的前提（「跨模式复用须前一模式全部 drain」）。
`FLAG_SEQ[]` 把整个 kernel 的同步点按 (核型, mode) 排成执行序，`FlagSeqAdjacentOk()` /
`FlagMaxUse()` 两个 `static_assert` 逐对核对相邻性与用量（实测最大用量 2 ≪ 15）。

### 2.5 BufferID 登记（每核用户可用 0..27）

| 段 | AIC | AIV |
|---|---|---|
| GDN | 0-6（A/B ping-pong + L0A/L0B + L0C） | 0-4（m6 行窗 0/1 + m9 prolog 2/3/4）、8-17（m4 递推 8-14 + m12 15-17）、21-22（gamma 预转） |
| MoE | 0-6（同上；相位边界后 mmad 复用同编号 token） | 0-7（通用行窗）、15-23（m8#1 四级流水 15-18 + 预取/gamma/rstd 19-23） |
| 峰值同时占用 | 7 ≤ 28 | 17 ≤ 28（编译期 `static_assert`） |

两段各自需要 7/17 个 AIV 编号，**无法完全不相交**；跨相位复用（0-4、15-17、21-22）的
安全性由 §1.1 第 1 条的 `PipeBarrier<PIPE_ALL>`（每核 6 条 pipe 全 drain）给出，
登记表 `BUF_TABLE[]` 里逐 id 标注了 `sharedAcrossPhase`。

> **口径说明（review r1 次要项）**：`static_assert(BUF_*_PEAK <= 28)` 里的 `BUF_AIC_PEAK=7` /
> `BUF_AIV_PEAK=17` 是**字面量**，因此这两个断言只能抓住「有人改了字面量」，**抓不住
> `BUF_TABLE[]` 被填错**（对比 §2.4 的两个 flagId 断言是从 `FLAG_SEQ[]` 真算出来的，
> reviewer 用变异测试证明它们会咬）。BufferID 表本身是人工登记 + reviewer 逐 id 核对，
> 不存在编译期不自洽检测 —— 这是已知的弱项，若要加固需把 `BUF_TABLE[]` 也做成可计算的区间表。

## 3. 段内同步表

**相位 A / 相位 B 的段内同步表与 donor 完全一致、一字未改**：
- GDN 段：`m15_gdn_resources.h` §3（flagId 4-15）—— M25 已归档、本 mission 未触碰；
- MoE 段：`m15_moe_resources.h` §3 —— 与 m13 的表同构，仅按 §2.4 重编号。

规约核对（docs/05 §2 / docs/12 §5）：set 一律挂生产 pipe（MTE3/FIX），**从不挂 PIPE_S/PIPE_ALL**；
wait 用尽可能窄的 pipe，仅「标量值依赖」的段挂 PIPE_S（GDN S1 边界、MoE S3 边界）；
mode 0 只在同类型核之间；mode 2 只做 AIC ↔ 配对 AIV；**相邻同步点 flagId 必不同**（§2.4 断言）。

## 4. 权重与 state 布局

### 4.1 MoE 段权重槽（`m15_layer_resources.h` §1；512B 对齐）

| 偏移 | 张量 | 字节 | checkpoint 来源（真实档） |
|---|---|---|---|
| `MW_ROUTER_OFF` | router | 20,480 | `mlp.gate.weight` **[512,2560] 行 0..3** |
| `MW_SGATE_OFF` | 共享门 | 5,120 | `mlp.shared_expert_gate.weight` [1,2560] |
| `MW_WGU_OFF` | experts gate_up | 6,553,600 | `mlp.experts.gate_up_proj` **[512,1280,1280] 专家 0..3** |
| `MW_SGU_OFF` | ↑ scale | 409,600 | `…gate_up_proj.weight_scale` 专家 0..3 |
| `MW_WDN_OFF` | experts down | 3,276,800 | `mlp.experts.down_proj` 专家 0..3 |
| `MW_SDN_OFF` | ↑ scale | 204,800 | `…down_proj.weight_scale` 专家 0..3 |
| `MW_WGUSHD_OFF` | 共享 gate_up | 1,638,400 | `shared_expert.gate_proj` **行拼接** `up_proj` |
| `MW_SGUSHD_OFF` | ↑ scale | 102,400 | 两个 scale 行拼接 |
| `MW_WDNSHD_OFF` | 共享 down | 819,200 | `shared_expert.down_proj.weight` |
| `MW_SDNSHD_OFF` | ↑ scale | 51,200 | `…down_proj.weight_scale` |
| `MW_G1_OFF/MW_G2_OFF` | 段内 norm gamma | 5,120 ×2 | **checkpoint 无此张量 → 确定性合成**（见 §8.2） |

**权重表示一行不转**：kernel 直接消费 HF 原始 layout（bf16/i8 nibble + e8m0 scale 不做转换）；
**唯一 host 侧组板 = 共享专家 gate/up 的行拼接**（与 m13 口径一致，字节不变）。
manifest 里的 MoE 张量行由 `slice_layer_manifest.py` 生成（**48 层每层 12 行**，专家在最外维
→ 前 4 个专家的字节段连续，offset 不变、`bytes` 换成切片长度）。

### 4.2 state

MoE 段**无跨 token 状态**（m13 亦如此）：`conv_state`/`ssm_state` 只属于 GDN 段，布局与
in-place RMW 语义与 M25 完全一致（`m15_loop_layout.h` 未改动）。

## 5. 验证

判定项分栏（docs/17 §2）：`checks` = 实质判定项（计入失败）；`guards` = 恒真前置断言
（单列、不计入）；**报告项**只打印。`accept_run.log` 末尾有分账行。

| 组 | 条数（真实档，48 层） | 内容 |
|---|---|---|
| 权重来源（GDN） | 36 | 独立 pread + 按 manifest rows 拼接 → 与设备槽位 memcmp + 双路 sha256（M25 原有） |
| 权重来源（MoE） | 48 | 同上，12 个张量/层（**新增**，§5.1） |
| A | 216 | 层间组合正确性（独立启动形态；M25 原有） |
| B | 506 | 多 token 状态连续性 vs host 参考链（独立启动形态；M25 原有） |
| C | 3 | 开销基线 + 无 sync 连发形态数据复核（独立启动形态；M25 原有） |
| **M1** | **348** | **段间零串扰**：融合路 vs 独立启动的两段逐字节（§5.2） |
| **M2** | **8** | **多 token（融合形态）**：融合 == 两次启动的组合、非空洞性、交付形态确定性（§5.3） |
| 合计 | **1165** | |

### 5.1 MoE 段权重来源（48 条，新增）

`H_MoeWeightSourceCheck`：**不复用装载路径**，按 manifest 的 role 自己重新 pread 12 个张量
（共享专家的 gate/up 自己再拼一次行），与**设备 MoE 权重槽回读**逐字节比对，并打印两侧
sha256（`accept_run_m40.log` 每层两列）。它同时兜住：① 层号 ↔ MoE 槽位错位（48 层摘要互不
相同）；② 共享专家 gate‖up 的拼接方向；③ MXFP4 打包字节没被任何中间转换改动。

### 5.2 验证 M1：段间零串扰（348 条判定项，0 FAIL）

**方法**：逐层三路，同一输入、同一状态初值：

| 路 | 形态 |
|---|---|
| P1 融合 | `m15_layer_kernel_gdn` / `_attn`（一次启动，两个相位） |
| P2 子层 | 独立启动 `m15_gdn_layer_kernel`（M25 的层 kernel）或 attention 占位 |
| P3 MoE  | 独立启动 `m15_moe_segment_kernel`，输入 = P2 的出口 |

判据（每层）：

| 判据 | 比对面 | 条数 |
|---|---|---|
| `M.finite` | 融合层出口是有限 bf16（无 NaN/Inf） | 48 |
| `M.nontrivial` | 融合层出口 ≠ 层输入（整层不是恒等映射） | 48 |
| `M.ym` | 融合层出口 == P3 出口（逐字节） | 48 |
| `M.moews` | 融合路 MoE 段 ws 全量 7,685,472 B == P3 的（逐字节） | 48 |
| `M.gdnws` | 融合路 GDN 段 ws 全量 5,258,240 B == P2 的（逐字节，**排除 g/β padding 车道**，见下） | 36 |
| `M.gb_lane0` | g/β 的 `[h][0]`（48 head × 2 数组 × 4B）逐字节 | 36 |
| `M.st_cs` / `M.st_ssm` | conv_state 61,440 B / ssm_state 3,145,728 B 逐字节 | 36 + 36 |
| `M.attn` | attention 占位段「入口 == 出口」逐字节（M25 的性质） | 12 |
| 合计 | | **348** |

> **为什么 `M.gdnws` 要排除 g/β 的 padding 车道（本 mission 实测发现的一处既有 quirk）**：
> GDN 段 S3 的 g/β 出口是 `[48][8] fp32` 的 stride-8 槽位（`m15_gdn_layer.h:941-943` 明确
> 「仅 `[h][0]` 有效」）；生产者每次写**整个 32B 槽**（`:886`），而槽里 8 个 float 只有第 0 个
> 是算出来的，其余 7 个来自 UB staging 的无关内容；消费者（S4 递推）**只读槽位首元素**
> （`:1148`「BRC 按槽位首元素广播」）。于是这 7 个 padding float 取决于**同一 UB 区在本次
> 启动前的残留内容**，而残留内容在「融合 kernel 的 GDN 相位」与「独立启动的 GDN kernel」
> 两种上下文里本来就不同。**不变量（判定项）**：差异**全部**落在 g/β 的 padding 车道内 —— 这
> 由 `M.gdnws`（排除 padding 后逐字节，出现 padding 外的差异即 FAIL）与 `M.gb_lane0`（被消费的
> 48×2 个槽位首元素逐字节）双向夹住。**差异字节数是报告项、且随运行变化**（padding 内容 = UB 残留，
> 取决于启动序列）：本 commit 的 `evidence/accept_run_m40.log` 末尾报告项为 **合计 3344 B / 48 层、
> 单层最大 2439 B**（该口径把 `A.direct_ws` 与 `M.gdnws` 两处一起累计），
> 落在 g/β 两个数组的 padding 车道内；**其余 5.26MB/层逐字节相同**，且 `M.gb_lane0`（被消费
> 的 48×2 个 fp32）逐字节一致。处置不是放宽容差：把「哪些字节被消费」本身写成了判据
> （`M.gdnws` 排除 + `M.gb_lane0` 独立见证 + padding 差异数作为报告项打印）。

> **为什么没有「融合路相位 A 出口 == P2 出口」这条判据**：融合 kernel 的 `yLayer` 同时是相位 B
> 的输入与输出（§1），启动结束后它装的是 MoE 段的输出，相位 A 的出口已被覆盖。相位 A 的
> 正确性由**更强的**三条见证：`M.gdnws`/`M.gb_lane0`（GDN 段 ws 全量逐字节，含 `WS_OPOUT` 与
> fp32 的 `WS_RES2`，而 `y = f(WS_OPOUT, WS_RES1, gamma2)` 是它们的确定函数）、`M.st_*`、
> 以及 `M.ym`（两侧喂同一输入给同一个 MoE 段，出口逐字节相同 ⇒ 两路的相位 A 出口相同）。

> **补一条更强的见证（review r1 指出）**：M1 的 `M.moews` 比的是**整块** MoE 段 ws，其中含
> `res1 = fp32(段输入)`，而 numpy 侧又独立判 `res1 == fp32(layer_in)` 位级 ⇒ 融合路与独立路的
> `res1` 逐字节相等，**这就是「两路相位 A 出口相同」的直接见证**（比下面那段「MoE 是确定函数」
> 的论证更强）。`check_moe_ref.py` 的 `res1 == fp32(层输入)（S1 零残差）位级` 就是它。

> **Ver M1 证明什么 / 不证明什么**：它证明「把两个段放进同一个 kernel（共用 UB/L1/L0C 叠放与
> BufferID 编号、相位边界只靠 PipeBarrier + 一次 mode-0 barrier）**不改变任何一段的任何字节**」
> —— 这是融合最大的风险点（资源叠放没隔离干净、BufferID 复用未 drain、flagId 冲突都会让至少
> 一个相位的 ws 与独立启动不同）。它**不**证明「MoE 段本身算得对」——那是 §5.4 的 numpy 独立
> 校验与 m13 既有验收的事；也不证明「路由规模是真实模型的规模」——见 §8.1。

### 5.3 验证 M2：多 token（融合形态，8 条判定项，0 FAIL）

3 个 token × 48 层，两条链：

| 路 | 形态 |
|---|---|
| 融合 | 每层 **1 次**启动（`m15_layer_kernel_*`） |
| 两次启动组合 | 每层 **2 次**启动（子层段 + MoE 段，同一 stream 串行、靠 kernel 边界交接） |

| 判据 | 口径 |
|---|---|
| `M2.fused_vs_composed_h` | 3 token 末层出口逐字节一致 |
| `M2.fused_vs_composed_state` | 36 层 × (conv_state + ssm_state) = 115,458,048 B 逐字节一致 |
| `M2.state_evolved` | 非空洞性：**同一层**（首个 GDN 层）的 `cs+ssm` 在 token 0 与 token(steps−1) 的快照必须不同（逐字节）；相同即 FAIL。review r1 前这条比的是「层 0 的 conv_state」与「末层 ssm_state」两块无关缓冲 ⇒ 恒 PASS（空洞），已修 |
| `M2.token_sensitive` | 非空洞性：末层出口随 token 变化（输入敏感性） |
| `M2.nosync_vs_sync_{h,state}` | 交付形态（连发不插 host sync）与逐层 sync 逐字节一致 |
| `M2.nosync_repeat_{h,state}` | 连发形态自身可重复（「逐字节」类判据的确定性前提，docs/17 §4 末条） |

第二条链（两次启动的组合）正是 **M25 已对 host 参考链验过**的组成（§5.5 的 Ver A/B），
因此 `M2.fused_vs_composed_*` 把「融合形态」接在那条已验证的链上（传递论证；本 mission
没有重跑 host 参考链，see §8.7）。

### 5.4 独立 numpy 交叉校验（`check_moe_ref.py`，128 + 20 条，0 FAIL）

> **M96 读数更新**：本节的 **16 已改为 20**（M90 起每层多一条 T4 guard「路由参考非退化」）。
> M90 的改写（把参考链改指 m17）曾让本条在 main 上**崩掉**、`128 + 16` 无法复现；M96 修好后在
> main tip 的 dump 上实跑读数为 **128 条判定项 + 20 条非空洞性，0 FAIL**（详见文首 M96 增量）。
> 下面正文里的「复用 m13 的参考链」是 **M40 当时的形态**：M90 起量化器/反量化/T3 界改由
> `m17_moe_real/check_ref.py` 提供，m13 那条链只在 `--selftest` 里作行为保持对照 —— 本 mission
> 只收口读数，不改写 M40 的历史描述。

**复用 m13 的参考链，不另造一套**：脚本 `import` 的是
`m13_moe_layer/check_ref.py`（硬件 floor 指数的 MXFP4 量化器 `quant_mxfp4_hw`、反量化
`dequant_device`、判据工具 `rel_check`、golden 规范对照量化器 `quant_mxfp4_f32`）与
`tools/golden/moe_block_ref.py`（权威编解码与路由语义）；**权重也不是 m13 的合成数据，而是从
真实 checkpoint 按张量名独立 pread**（`mlp.gate.weight` 行 0..3、`mlp.experts.*` 专家 0..3、
`mlp.shared_expert*`）。差异只有「dump 的载体」：融合 kernel 落的是**每层一整块 MoE 段 ws**
+ `moe_layout.txt`（布局表由 C++ 的 M15M 常量算出，所以 python 与 kernel 用的是**同一套偏移**）。

| 组 | 判据 |
|---|---|
| router | `topk_ids` 精确相等；`router_logits` / `topk_weights` 容差（2e-2 / 1e-3 + 半 bf16 ulp） |
| 索引生成 | `expert_token_counts` / `expert_offsets` / `perm_src_token` / `perm_expert` 精确；`inv_slot` 与 perm 互洽（padded 行号 = expert×M_MAX + 专家内局部序号）；`w_tk_packed` 低 16 位 = bf16(topk_weights) |
| permute | `x_sorted` == `x_norm[perm_src]` 的 gather，**逐位** |
| 量化器 | `A_qx` / `A_scale` / `H_qx` / `H_scale`（含共享专家）与 numpy 独立重量化**逐字节**（硬件 floor 指数规范） |
| w4a4 double 链 | `GU` / `H_swiglu` / `Y`（每个非空专家槽）+ 共享专家同款 —— 激活锚点取 **device 自己的量化输出**（逐 op 口径，残差只含本段） |
| combine | `shared` / `moe` 与「未舍入 gated 乘积」的 bf16 位距 —— **实测位级一致 2560/2560** |
| S1/S10 | `res1` == fp32(层输入) 位级、`res2` == fp32(moe_out) + res1、层出口非零且有限、层出口 ≠ moe_output |
| 非空洞性（**20 条**；M40 当时 16 条） | x_sorted 非全零、`topk_ids` 落在 [0,E) 且互不相同、`Σt_e` = m×topk、logits 非常数（**M90 起每层再 +1**：路由参考非退化「活跃专家数可变」）；层出口非零且有限与 `层出口 ≠ moe_output` 现列在**判定项**栏 |

**报告项**（不参与 PASS/FAIL，逐条打印）：① 两套 MXFP4 规范的逐字节差异率（kernel 用硬件
floor 指数规范、golden 权重用 `ceil(log2(amax/6))`；m13 已量化披露的同一现象，实测
`A_qx_shd` 29%–38%、`A_scale_shd` 34%–43%）；② combine 段的 bf16 位距分布与绝对残差
（maxAbs 3.0e-5–6.1e-5 @ |操作数| 均值 ~4–5e-3）。

### 5.5 M25 的三验证（独立启动形态）保留为回归

`m15_layer_loop.asc` 里 M25 的 `H_RunA/B/C` 仍用 `H_Launch`（独立启动形态）：
**A 216 / B 506 / C 3 条判定项**（计数不变）。**唯一改动**（review r1 后）：`A.direct_ws`
的比较从「整块逐字节」改成「**排除 g/β 的 `[h][1..7]` padding 车道**」——与 §5.2 的 `M.gdnws`
同一处置、同一理由（该 7 个 float 由 UB 残留内容决定，消费方按 stride-8 契约只读槽位首元素；
槽位首元素仍在比较范围内）。改前该判据会**跨运行抖动**：实测一次 `runs=all` 里 L00 差 701 B、
L14 差 2077 B，**全部落在 padding 内**，且因为 `H_RunCPath` 以 `C.fails == 0` 为前提，这两个
失败还连带把 Ver C 的 no-sync 复核整段截断（分账行显示 `C 0`）。改动后 `A.direct_ws` 与其
padding 差异数（报告项）一起进 `C.gbPadDiffBytes`，判据数不变。这一步是刻意的：① 它们是与 M25 归档证据可逐条对照的回归；
② 它们对 host 参考链（m14 的 double 参考链）验过的正是「两次启动组合」这条链，M2 的等价性
判据依赖它。**交付形态是融合 kernel**（M1/M2 验它），独立启动形态是它的对照与回归。

### 5.7 紧凑专家槽 / active_num 接口改造（tower 硬约束）与**尚未交付的部分**

tower 对 M40 的裁决：走缩形档 (A)，但**缩形只允许体现在「路由器这一趟吐出几个 id」上，
不许把「4 专家 / 定长 topk2 / 每专家固定槽」的布局假设固化进 MoE 段**。据此本 mission 做了
**纯寻址**的改造（缓冲区尺寸一字未改：本实例 `NUM_EXPERTS*M_MAX = 256` 恰好等于紧凑上界
`TOTAL_MAX = M_MAX*TOPK_MAX = 256`，故 ws 天然够用；真实规模下紧凑布局还能显著缩小它）：

| # | 位置 | 改造 |
|---|---|---|
| 6a | S3 的 `inv_slot` | `e*M_MAX + 局部行号` → **紧凑行号**（= 计数排序产出位置） |
| 6b | S5 A 侧量化 | 目标由 padded 改**紧凑**（`PAD_DST=false`） |
| 6c | S7 H 侧量化 | 源与目标都改**紧凑**（`PAD_SRC=false, PAD_DST=false`） |
| 6d/6e | S6/S8 两组 GEMM | A/H/Y 槽位基址 `slot*M_MAX*…` → **`off[slot]`（前缀和）** |
| 6f | S6/S8 工作项 | 扁平 `(slot, nb)` → **(expert, mTile, nBlock)** 三重循环，空专家由 counts 跳过 |
| 6g | 量化器行数/槽位解码 | 新增 `ROUTED` 形参：行数 = **Σt_e**（counts 驱动，不写 `m*topk`、不依赖 NUM_EXPERTS）；槽位由前缀和解码 |

**证据（本 mission 已交付的）**：
1. 全部改造由 `lift_moe_segment.py` 的规则 6a–6g **机械生成 + `--check` 复核**
   （入库文件与抽取规则逐字节一致；`build()` 里断言「GEMM 区不得再出现 `slot * M_MAX` 寻址」）；
2. `runs=all` 的 1165 条判定项仍 **0 FAIL**（M1 的两路逐字节等价在紧凑寻址下成立）；
3. `check_moe_ref.py` 的 **128 判定项 + 20 非空洞性仍 0 FAIL**（M52 当时是 128+16；M90 起非空洞
   +1/层，M96 修好 M90 的口径漂移后复跑读数；见文首 M96 增量）—— 它按 `expert_offsets[]`
   切片（python 与 kernel 用同一套前缀和），逐字节验 A/H 量化器、逐槽位验两组 GEMM 与
   unpermute 的加权折叠，即**紧凑寻址下的数值链正确**；
4. `×` **未交付：接口契约测试**（tower 裁决里的「关键交付」）——「活跃专家数可变 + 每专家
   token 数不均匀（含空专家/单 token 专家）」的构造用例。当前 `m` 仍是 1（融合路的层链只跑
   decode 单 token），要覆盖 **m>1 的非均匀分布**需要给 `m15_moe_segment_kernel` 开一个
   可传 `m`/`topk` 的用例入口 + 构造输入（按 one-hot router 权重设计每 token 的 top-k 集合），
   本 mission **预算用尽、未实现**。**这一条必须由后续 mission 补**（见 §8.9）。

### 5.8 M44 裁决在本 mission scope 内的两项（review r1 的 P2-4 + tower 指派）

| 项 | 处置 | 位置 |
|---|---|---|
| `m15_gdn_layer.h` 的事实性错误注释「全部在 `__VEC_SCOPE__` 内调用」 | **已改为事实正确的说法**（判据落在被调函数上：函数标 `__simd_vf__` 且体内只用寄存器 API；调用点不要求词法 `__VEC_SCOPE__`；本仓 0 处 `asc_vf_call`）。**代码一字未动** | `m15_gdn_layer.h` 的 VF 段标题注释（依据 tower M44 裁决 ①） |
| 并入的 MoE 段是否含经典 `Extract` | **不含**（M52 重抽后副本与上游**逐行相同**）：m13 的 S2 router（`M15M::RouterStage::SoftmaxTopkRow`）已按 M44 裁决 ① 把经典 memory-based `Extract` 换成厂商 VF 形态 —— `Sort32` 之后一次 `LoadAlign<float, LoadDist::DIST_DINTLV_B32>` 解交织 + 两条掩码 `StoreAlign`（抄 dav_3510 的 `ExtractVf<float>`：`x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h`），**由 M50 在 m13 落地**；本轮重抽即取得该形态。M40 那版曾在本脚本里手写一份等价替换（旧规则 7，产物另有一处「不落 UB 再读回」的形态差），**已被重抽取代**（位级影响面见 §5.9）。**判据 T1 逐位**：选中集合/顺序/权重由 `check_moe_ref.py` 四条判据逐位见证（`topk_ids 精确相等` / `topk_weights` / `inv_slot 与 perm 互洽` / `w_tk_packed 低16位 = bf16(topk_weights)`），复跑 128+16 全 PASS（M52 当时读数；M96 后同一套判据为 128+20，见文首 M96 增量） | `lift_moe_segment.py` 的 `assert_upstream_vf()`、`m15_moe_layer.h` |
| `Sort32` | **保留**（tower M44 裁决 ② 的白名单例外）。**依据注释随上游带进来**（`SoftmaxTopkRow` 的 `Sort32` 调用点：指向 docs/05 §6.1 计算路径规则 ⓒ + 两条实证 =「Reg 侧无等价物（CANN 9.1.0 `reg_compute/**` 穷举）」与「官方 donor 同为 memory-based」）；重抽脚本对**这四条**做**断言**（例外声明 / docs/05 §6.1 / 无 `Reg::` 等价物 / 官方 donor 位置，见 `lift_moe_segment.py` 的 `SORT32_BASIS_MARKS`），缺任一条即失败 | 同上 |
| `MrgSort` / `MrgSort4` | **本段不含**（无任何调用；只有 S2 段首注释里的历史提法） | — |

### 5.9 本次重抽的位级影响面（M52：副本跟随 m13/M50）

**起因**：`m15_moe_layer.h` 是 m13 device 段的机械副本（§9），而 M40 抽取它时 m13 还没有 M50 的
S2 改造。M52 重抽前先修了抽取脚本自身的一处失效：旧「规则 7」（M40 在**脚本里**手写一份
`Extract`→VF 替换）的目标文本随 M50 在 m13 落地而消失 ⇒ 脚本直接 `AssertionError`
（**重抽在 M52 开工时不可行**）。修法 = 把它降级为**断言** `assert_upstream_vf()`：① device 段
不得出现经典 `Extract(` 调用（断言前先剥行尾注释，否则注释里的历史提法会误报）；② 必须存在
`DIST_DINTLV_B32` 解交织的 VF 形态；③ `Sort32` 调用点必须带依据注释的**四段**（例外声明 /
docs/05 §6.1 / 无 `Reg::` 等价物 / 官方 donor 位置，见 `SORT32_BASIS_MARKS`）。**脚本不再做任何
Extract 替换** —— 副本因此恒等于上游的当前形态。

**重抽带进来的逐项差异与位级结论**（`m15_moe_layer.h` 的 **device 段**共改 43 行（+29/−14），
**全部落在 S2 router 内**；相对分支 tip 的全部改动是 54 行（+36/−18），多出的 1 个 hunk 是文件头
PROLOGUE 的说明文字）：

| # | 差异 | 性质 | 位级影响 |
|---|---|---|---|
| 1 | S2 段首注释、`SoftmaxTopkRow` 段首注释 | 纯注释 | **无**（不生成代码） |
| 2 | `Sort32` 调用点的依据注释（docs/05 §6.1 计算路径规则 ⓒ + 无 `Reg::` 等价物 + 官方 donor 位置） | 纯注释 | **无** |
| 3 | `SoftmaxTopkRow` 的 `Extract`→VF **形态**换成上游版：解交织后把解出的两个寄存器落 `UB_RT_OV`/`UB_RT_OI`，再由后一个 `__VEC_SCOPE__` 载回做 renorm。M40 那版是**在同一个** `__VEC_SCOPE__` 内直接 renorm（省掉这一趟 UB 往返），故这是**两种形态**之差 | **指令形态**（每行多 2 次 `StoreAlign` + 2 次 `LoadAlign`，另多一条核内 `LocalMemBar<VEC_STORE,VEC_LOAD>` 给这趟 UB 往返兜顺序；**无算术改动**） | **位级不变**（见下表的三条见证） |
| 4 | 6 类机械替换（namespace / 资源表 include / 命名 / `yLayer` / 紧凑槽 + 工作项） | 本轮**一字未动**（M40 已交付） | 无 |
| 5 | **舍入点** | **本轮 0 处改动**：`Extract` 的 VF 化自始至终不参与算术（纯位重排）；两种形态的 `Reduce<SUM>`/`Div` 次序、`UpdateMask<float>(TOPK)` 掩码、`StoreAlign` 的分布完全一致 | — |

**段序与同步表：一字未动**（这是任务书要求停下来的那条边界，此处明确报告「没有动」）：重抽的全部
影响面就是上面 5 行里的第 1–3 行，**没有触碰任何跨核 flagId / pipe / 段序**；第 3 行新增的那一条
`LocalMemBar<VEC_STORE, VEC_LOAD>` 是**核内** pipe 排空（给本函数内 UB 的 store→load 兜顺序），
不是跨核同步，也不进 §3 的段内同步表 —— 它是 M50 在 m13 上已随其全套判据验收过的形态的一部分。
`m15_moe_layer.h` 的其余部分与改动前**逐字节相同**（相对分支 tip 共改 **54 行 = +36/−18、5 个
hunk**：**43 行（+29/−14、4 个 hunk）在 S2 router 内**，另 1 个 hunk 是文件头 PROLOGUE 的说明
文字 —— 它随脚本头部一并更新；**device 段**的全文差异见 `evidence/m52_relift_diff.txt`）。

**「位级不变」的三条见证**（同一次会话内 A/B：改动前构建 `pre` = 分支 tip，改动后构建 `post`
= 重抽后；两个二进制的 sha256 不同，见 §5.6 证据）：

1. **1131 个 dump 张量 sha256 逐位相同**：两侧各跑一次
   `M15_DUMP=1 M15_STEPS=3 <bin> weights_manifest.txt all`，两份 `sha256sum` 清单 **diff 为空**。
   覆盖面 = M25 的 1117 个张量 + 每层整块 MoE 段 ws（7,685,472 B × 4 层）+ `moe_layout.txt`。
   **这是最强的一条**：它直接证明被 dump 的每一个字节（含 `topk_ids` / `topk_weights` /
   `inv_slot` / `w_tk_packed` / `perm_*`）都没变。
2. `check_moe_ref.py` 的**整份日志逐字节相同**（M52 当时的读数：128 判定项 + 16 非空洞性 + 16
   参考项，连所有报告项数字都一致 —— 该说法指 M52 改前/改后两侧日志相同，不指 M96 后的条数）。
3. `runs=all` 的 **1165 判定项 + 109 guard、0 FAIL**；`check_ref.py` **709** —— 读数与 M40 归档
   一致。**两侧日志的差别只有** host 墙钟类行（共享卡噪声，见 §6.1）与 g/β padding 车道差异
   报告项（按 §5.2 的定义它本来就随 UB 残留变化）。

**两类改动各自的位级要求（按 docs/17 §1.1 写清档位）**：

- **`Extract` VF 化 = T1 逐位**（它的输出是**索引与掩码/位域类**：`topk_ids` 是索引、打包字节
  是位域，按 T1「无例外」）。判据形态 =「**选中集合与掩码逐位一致**」，逐条落到：

| 判据 | 承载 |
|---|---|
| 选中集合逐位 | `check_moe_ref.py`：`topk_ids 精确相等`（有序数组 ⇒ 集合与顺序一次锁住） |
| 权重逐位 | 同上：`topk_weights`；以及 `w_tk_packed` 低 16 位 == `bf16(topk_weights)`（把「权重」与「打包字节」两侧锁在一起） |
| 索引/槽位自洽 | 同上：`inv_slot 与 perm 互洽`、`perm_src_token`/`perm_expert` 精确 |
| 掩码 | 两种形态用的是同一个 `UpdateMask<float>(RT_LANES)`（解交织）与同一个 `UpdateMask<float>(TOPK)`（renorm）；更强的一条是上面第 1 项「整块 ws 逐字节相同」，掩码语义若有差会立刻显形 |

- **`Sort32` 依据注释 = 无位级要求**（纯注释）。它要求的是**注释内容**：必须指向 docs/05 §6.1
  计算路径规则 ⓒ，并给出「Reg 侧无等价物」与「官方 donor 同为 memory-based」两条依据。由
  `assert_upstream_vf()` 断言；负向对照（把它删掉/把 Extract 退回经典形态时断言是否会咬）见 §5.6。

### 5.10 「A 操作数多读的行」契约：读侧 + 结果侧一起定义

`m15_gdn_layer.h` 的 `Cube::Bf16Gemm::RunTile` 里有一条 M22/M25 留下的契约（注释原文：
「计算侧统一提升为 >=2 行（多读的 1 行由 host 保证可读，结果行不写出）」）：3510 实测 `Nd2Nz`
在行数为 1 时不切分（退化为 1D 拷贝 ⇒ m=1 数据错位），故取 `calcM = max(curM, 2)`。
按 docs/05 的规则，这类「**多读的行由 host 保证可读**」式契约**必须同时给出「多算出的列/结果的
确定性」**，否则非确定性会从残留里渗进被消费的结果（M36 的 `down GEMM` 实例：那里的 padding
列**被写出**了，所以必须由 host 把 `wInj` 的多余行置零）。M40 只留了读侧那半句，本条把两侧
**补全**：

**读侧（host 契约）**：A 操作数按 `calcM = max(curM, 2)` 行从 GM 读。m=1 ⇒ **多读第 1 行**；
本段两块 A 操作数都落在 ws 的 `WS_XNORM` / `WS_OPIN` 区，尺寸按 `M_MAX` 行分配
（`m15_gdn_resources.h` 的 `SZ_XNORM = M_MAX*HIDDEN*2`、`SZ_OPIN = M_MAX*V_DIM*2`），而 m=1
只用行 0 ⇒ **多读的那一行在分配区间内、结构上可读**（层间残差流同理：`m15_loop_layout.h` 的
`H_ROWS_BYTES` 按 `M_MAX` 行分配）。

**结果侧（多算出的行去哪了 —— M40 未写完、本条补上的那一半）**：

1. `Mmad` 的 `mmadParams.m = calcM` ⇒ 只对 `calcM` 行做乘加。`LoadA` 虽把 L0A 载到
   `calcMAlign = AlignUp(calcM, CUBE_BLOCK)` 行（m=1 时 = 16 行），但**多载的那些行只是躺在
   L0A 里，不参与任何乘加**（m 轴在 `Mmad` 里已封顶在 `calcM`）；
2. `Fixpipe` 的 `fp.mSize = curM` ⇒ **多算出的第 1 行不写出**（只留在 L0C 内，被下一个 tile 的
   `cmatrixInitVal = (kBlock == 0)` 整块重算）；
3. `Mmad` 的**行间无耦合**（本 tile 的累加只沿 K 轴、不跨 M 轴）⇒ 被写出的 `curM` 行与「多读的
   那一行 / 多算的那一行」的取值**无关**；
4. 因此：即使多读的那一行取自 **UB/ws 残留**（m=1 时 `WS_XNORM` 的第 1 行就是上一次使用该 ws
   的残留内容，**本身不确定**），它也**不可能渗入任何被写出、被下游消费的字节** —— 非确定性被
   关在「不写出的那 1 行」里。**这正是与 M36 实例的差别**：那里 padding 列**被写出**，所以必须
   由 host 置零；这里 padding 行**不写出**，所以 host 只需保证「可读」而**不需要**保证取值。

**见证**：`M1`/`M2` 的全部逐字节判据（`M.ym` / `M.moews` / `M.gdnws` / `M.st_*` /
`M2.fused_vs_composed_*` / `M2.nosync_repeat_*`）—— 任何渗漏都会让某一条 FAIL；其中
`M2.nosync_repeat_*` 另外给出「连发形态自身可重复」，即被写出的结果对非确定残留不敏感。

（§5.2 的 g/β padding 车道是同一族规则的另一个实例：那里「多写的字节」被**排除在判据外**并
由 `M.gb_lane0` 独立见证被消费的槽位首元素 —— 因为消费方按 stride-8 契约只读槽位首元素，
「多写的字节」既不被消费、也不影响被消费的值。）

### 5.6 证据归档（`m15_layer_loop/evidence/`）

| 文件 | 内容 |
|---|---|
| `accept_run_m40.log` | **一次 `runs=all` 的完整日志**：权重来源 84 + A 216 + B 506 + C 3 + M1 348 + M2 8 = 1165 条判定项 + 109 条 guard，`rc=0`；含 M1 的 padding 报告项与分账行 |
| `accept_run_m25.log` | M25 归档的 `runs=all` 日志（761 判定项 + 109 guard；独立启动形态的历史证据，本 mission 未改动它对应的代码路径） |
| `check_moe_ref_run.log` | `check_moe_ref.py` 的日志（**M40 当时的读数**：128 判定项 + 16 非空洞性 + 16 参考项；M96 修好 M90 的口径漂移后同一套判据是 128 + **20**，见文首 M96 增量） |
| `check_ref_run.log` | M25 归档的 `check_ref.py` 日志（709 判定项；消费同一批 dump 的 M25 侧判据） |
| `dump_sha256.txt` | M25 归档的 dump 张量 sha256 + 再生成命令 |
| `msprof_op_statistic.csv` | msprof 原始 per-kernel 表（Count/Total/Min/Avg/Max/Ratio） |
| `msprof_summary.log` | 段级分解的推导（§6）+ host 计时 + 一行式复现命令 |
| `msprof_run.log` | `runs=p`（最小启动集）的完整 stdout。**其 P1 host 墙钟值随运行环境变化（共享卡，见 §6.1），host 侧数字一律只作定性参考** |
| `msprof_r2_op_statistic.csv` | **review r2 追加**：本 commit 构建的设备侧 msprof 真值（`--task-time=l1 --ai-core=on`），五个符号的 Count/Avg —— 用于把 §6 的表从「r1 之前的采样」升级为「对当前构建成立」（review r2 指出共享卡上 host 墙钟不构成证据，设备侧口径才作数） |
| `msprof_r2_run.log` | **review r2 追加**：上面那次设备侧采样的完整 stdout（含 P 形态与 host 计时） |
| `p1_timing_distribution.log` | **review r2 追加**：同一二进制、同一命令连续 5 次 `runs=p` 的 host 墙钟**分布**（§6.1 的抖动实测；按规则必须给范围而不是单点） |
| `accept_run_m52.log` | **M52 追加**：重抽后的完整 `runs=all` 日志 —— `checks=1165, guards=109, fails=0`（与 M40 归档读数一致） |
| `check_moe_ref_run_m52.log` | **M52 追加**：重抽后的 `check_moe_ref.py` 日志（M52 当时读数 128 + 16 + 16 参考项；M96 后为 128 + 20）。**与改动前的同一份日志逐字节相同**（§5.9 见证 ②） |
| `check_ref_run_m52.log` | **M52 追加**：重抽后的 `check_ref.py` 日志（709 判定项）。与 M25 归档的差别只有 dump 目录路径 |
| `m52_dump_sha256_ab.log` | **M52 追加**（§5.9 见证 ①）：改动前/后两个构建各跑一次 dump，1131 个文件 sha256 **逐位相同**；含两侧二进制 sha256、复现命令与**覆盖范围** |
| `m52_relift_diff.txt` | **M52 追加**：**device 段**的 `diff <改动前的 m15_moe_layer.h> <重抽后的>` 全文（4 个 hunk、+29/−14）—— 证明重抽对 device 代码的影响面就落在 S2 router 内（2 处注释 + `SoftmaxTopkRow`）。相对分支 tip 的全部改动是 5 个 hunk、+36/−18（多出的那 1 个 hunk 是文件头 PROLOGUE 的说明文字）。**行号口径见该文件头**：它生成时两侧都还是旧 PROLOGUE，故**本文件的 `+` 侧行号**比入库文件低 3（`-` 侧与 main 版的行号一致） |
| `m52_lift_negative_controls.log` | **M52 追加**：`assert_upstream_vf()` 与 `--check` 的 1 条正对照 + 3 条负向对照 + 1 条匹配器对照（含各条的**覆盖范围**交代） |
| `msprof_m52_op_statistic.csv` | **M52 追加**（§6.2）：重抽后构建的设备侧 msprof 真值（与 `msprof_r2_op_statistic.csv` 同一口径） |
| `msprof_m52_ab.log` | **M52 追加**（§6.2）：`pre`/`post` 两个构建**交错** `runs=p` 的设备侧 A/B 读数（各含 host 计时行，仅作定性；本节的结论只依据设备侧列） |
| `msprof_m52_run.log` | **M52 追加**（§6.2）：post 那次采样的完整 stdout |

dump 原始字节（每层 MoE ws 7.69MB × 4 层 + M25 的 1117 个张量 ~51MB）未入库：
kernel 已实测**确定性**（M2 的 `nosync_repeat_*` 判据 + M25 的同一结论），需要原始字节时按
`dump_sha256.txt` / `§9` 的命令在空目录复现并按 sha256 逐字节核对。

**M82 追加（2026-09-27，attention 布局冻结；二进制 sha256 `c5268d21…`（r2），M65 归档的 `07efbf03…` 已过期）**：

| 证据 | 内容 |
|---|---|
| `m82_runs_kv.log` | `runs=kv`：**56 判定项 + 130 guard，0 FAIL**。含容量独立复算打印、对齐判据、`Kv.cap.rt.*` 边界 round-trip、10 个 pos 的逐字节判据、`Kv.neg.offbyone.L4096` 与 `Kv.neg.paged`（**device 侧启动**）两条负向对照、`Kv.lay.headsplit` |
| `m82_accept_run_all.log` | `runs=all`：**2068 判定项 + 290 guard，0 FAIL**（M65 的 2012/160 逐项不变 + Kv 56/130） |
| `m82_regress_chain.log` | `runs=chain`：**778 判定项 + 3 guard，0 FAIL**（与 M65 归档逐项一致 ⇒ 36 个 GDN 层 + 48 个 MoE 段出口未回退） |

## 6. 段级 msprof 分解（设计输入，不是调优）

段在同一个 kernel 内，msprof 只能按 **kernel 符号** 统计，所以用**三种形态相减**得到段级分解
（`runs=p` = 「验证 P：最小启动集」，无判据；层间连发不插 host sync）。
下表是**紧凑槽版**（当前交付代码）的设备真值（48 层 × 3 轮；36 GDN + 12 attention）——
**M52 重抽后的设备侧复核见 §6.2**（同口径、含抖动范围）：

| 形态 | 符号 | 次数 | 平均设备耗时 | 扁平槽版（改造前） |
|---|---|---|---|---|
| P1 融合（1 次启动/层） | `m15_layer_kernel_gdn` | 108 | **204.62 µs/层** | 183.87 |
| P2 子层段（M25 的层 kernel） | `m15_gdn_layer_kernel` | 108 | **141.78 µs/层** | 141.93 |
| P1 融合（attention 层） | `m15_layer_kernel_attn` | 36 | **61.93 µs/层** | 43.31 |
| P2 占位直通 | `m15_attn_placeholder_kernel` | 36 | **2.46 µs/层** | 2.43 |
| P3 MoE 段独立启动 | `m15_moe_segment_kernel` | 144 | **58.96 µs/层** | 39.06 |

**段级结论（紧凑槽版）**：
- GDN 层：**MoE 段净增量 = 204.62 − 141.78 = 62.84 µs/层**（占融合层的 30.7%）；
- attention 层：MoE 段净增量 = 61.93 − 2.46 = **59.47 µs/层**，与 P3 的 58.96 吻合（互证）；
- 每 step（48 层，3:1）：融合 36×204.62 + 12×61.93 = **8.110 ms** vs M25 基线 5.134 ms
  ⇒ MoE 段一项 **+2.976 ms/step（+58%）**。

> **紧凑槽改造带来的性能退化（已知、已定位、未修）**：MoE 段从 39.06 → 58.96 µs/层（**+51%**）。
> 根因不是算术变多，而是**并行度**：改造前扁平工作项 `item = (slot, nb)` 按 `bid` 条带摊到 28 个
> AIC 上（每个 AIC ≤1 个 item）；改造后 `(expert, mTile, nBlock)` 三重循环里**内层 `nb` 也按
> `bid` 条带**，而 `GU_NBLK = 5`（down 为 10）< 28 ⇒ 只有 `bid < 5` 的 AIC 分到活。
> **修法（一行级）**：把「一个专家内的 `(mTile, nb)` 组合」先展平成一个长度 `mTiles*NBLK` 的
> item 序号再按 `bid` 条带（`it = bid; it < mTiles*NBLK; it += numBlocks`），即可同时保住
> `(expert, mTile, nTile)` 的语义与 28 核并行度。**M40 预算耗尽、未做**（用户明确现阶段
> 只求正确性）；完整修法与原因已记录在此，供后续立项。
> **本条已登记进 §8 的编号限制清单（第 12 条）**，免得只出现在本节而被漏读。

本节 §6 表**对当前交付构建同样成立**（review r2 重测，见 §6.1）。

host 侧（同一次运行内 print）：P1 0.16 / P2 0.11 / P3 0.06 ms/层量级，Δ 与设备侧同向。
**host 侧墙钟在本卡上不构成证据**（review r2 实测同一二进制 24× 抖动，`evidence/p1_timing_distribution.log`）——本行的量级只作**定性**参考；要作数的性能数字一律取本表的设备侧 msprof 口径。

> ## 6.1 融合符号的 host 计时在共享设备上不稳定（review r2 已归因：**测量环境，不是代码**）
>
> **现象**：`runs=p`（`M15_REPS=3 M15_LAYERS=48`）的 host 侧「每层一次启动」计时，
> **同一二进制、同一命令、连续 5 次**实测为 0.1773 / 0.4911 / 2.6948 / 3.6654 / 4.3321 ms/层
> （24× 抖动）；同一批里**代码逐字节未变的** `m15_gdn_layer_kernel` 也出现 0.4339 ms/层（4×）。
>
> **归因（reviewer r2 实测，2026-09-26）**：
> 1. 这三个 host 计时是「**连发 N 次 + 一次 sync 的墙钟**」；共享 NPU 上其他进程的突发会把
>    某个形态的排空时间整体推后 ⇒ 该形态墙钟被放大，**与被测代码无关**（代码未变的 P2 同样
>    抖动即为证；抖动出现在哪个形态是随机的）。
> 2. **设备侧真值不受影响**：本交付构建重新 msprof 采样 = `m15_layer_kernel_gdn` 205.23 µs /
>    `m15_gdn_layer_kernel` 142.12 / `m15_layer_kernel_attn` 61.99 /
>    `m15_attn_placeholder_kernel` 2.43 / `m15_moe_segment_kernel` 59.18 —— 与 §6 的归档表
>    （204.62 / 141.78 / 61.93 / 2.46 / 58.96）**全部在 0.3% 以内** ⇒ **§6 的表对当前构建成立**。
> 3. 静场下把「r1 之前的 device 代码」与「本构建」两个二进制**交错各测 4 次**：
>    0.1733–0.2359 vs 0.1736–0.2682 ms/层 ⇒ **两条分布重合**。
> ⇒ **不存在与本轮改动相关的性能退化**。§6.1 原稿把 4.7× 归因到「两条链 inline 进同一 kernel
> 后的寄存器压力/指令调度交互」并建议后续 mission 专项目定位 —— 该归因**已被实测推翻**，
> **该 follow-up 不必立**。
>
> **方法学教训（与 M24 的 507015 同型）**：单次 host 墙钟在共享设备上**不足以**支撑「某符号变慢
> N×」这类结论。要下这种结论必须同时满足：(a) 同一二进制重复采样并给出分布；(b) 用设备侧符号
> 耗时（msprof）交叉核对；(c) 在无其他负载的窗口内**交错**比较两个二进制。本次三件事都没做，
> 于是把环境噪声写成了代码现象。
>
> **复现命令**（任一空目录）：
>   A) 抖动：`for i in 1 2 3 4 5; do M15_REPS=3 M15_LAYERS=48 <bin> <repo>/m15_layer_loop/weights_manifest.txt p | grep 每层一次启动; done`
>   B) 设备真值：`msprof --output=prof --task-time=l1 --ai-core=on --ascendcl=off --runtime-api=off env M15_REPS=3 M15_LAYERS=48 <bin> <repo>/m15_layer_loop/weights_manifest.txt p`（再用本目录的聚合一行式）
>   C) 交错对比：两个构建（仅 `m15_moe_layer.h` 不同）轮流各跑 A 的 4 轮
> 本节数据由 reviewer-m40 在 `80769d0..ab746c5` 评审时于 `/tmp` 构建实测，非本 mission 归档；
> 本轮修复请把 A) 与 B) 的日志落到 `evidence/`（如 `evidence/p1_timing_distribution.log`、
> `evidence/msprof_r2_op_statistic.csv`），使修正后的本节有同 commit 证据。
>
> **本 commit 的复现（同 commit 证据，agent-moefuse 补；上述 reviewer 文字保持原样）**：
> `evidence/p1_timing_distribution.log` = 同一二进制连续 5 次 `runs=p`：P1 = 0.1734 / 0.1895 /
> 0.1738 / 0.1734 / 0.1760 ms/层，而**第 3 次的 `P3`（MoE 段——与融合路径无关的符号）从
> 0.0591 跳到 0.3819（6.5×）** ⇒ **独立复现了「被放大的是哪个形态是随机的」这一机制**
> （reviewer 那次被放大的是 P2，这次是 P3；两次都不是被断言的符号）。
> 设备侧真值 `evidence/msprof_r2_op_statistic.csv`：`m15_layer_kernel_gdn` **205.20** /
> `m15_gdn_layer_kernel` 141.79 / `m15_layer_kernel_attn` 62.31 / `m15_attn_placeholder_kernel`
> 2.47 / `m15_moe_segment_kernel` 59.31 µs —— 与 §6 归档表（204.62 / 141.78 / 61.93 / 2.46 /
> 58.96）**全部在 0.8% 以内** ⇒ **§6 的表对当前交付构建成立**。
>
> **本节结论**：不存在与本轮改动相关的性能退化；原 §6.1 的归因与它建议的 follow-up（查寄存器
> 压力/局部内存）**已撤回，不必立项**。

### 6.2 M52 重抽后的设备侧复核 + 形态 A/B（同 commit 证据）

本轮改了 MoE 段 S2 的 `Extract` **形态**（§5.9），需要确认它有没有可测的设备侧代价。按项目规则
**只用设备侧 msprof 口径**，并且**先给抖动范围再谈差异**。

**A/B 两个构建**（sha256 不同 ⇒ 确实是两份不同的 device 代码）：
`pre` = 分支 tip（M40 那版形态）、`post` = 重抽后（m13/M50 那版形态）。
同一次会话内**交错**采集 5 次（`runs=p`，`M15_REPS=3 M15_LAYERS=48`，顺序 post#1 → pre#1 →
post#2 → pre#2 → post#3）：

| 符号（Avg Time, µs） | pre 范围（2 次） | post 范围（3 次） | 5 次极差 | pre 两样本取小者 → post 中位（3 样本） | §6 归档表 |
|---|---|---|---|---|---|
| `m15_layer_kernel_gdn` | 205.305–205.455 | 204.245–205.435 | 0.59% | 205.305 → 204.915（−0.19%） | 204.62 |
| `m15_gdn_layer_kernel` | 142.430–142.625 | 141.498–142.534 | 0.80% | 142.430 → 142.162（−0.19%） | 141.78 |
| `m15_moe_segment_kernel` | 59.219–59.320 | 58.949–59.327 | 0.64% | 59.219 → 59.181（−0.06%） | 58.96 |
| `m15_layer_kernel_attn` | 62.002–62.011 | 61.713–62.020 | 0.50% | 62.002 → 61.776（−0.36%） | 61.93 |
| `m15_attn_placeholder_kernel` | 2.461–2.518 | 2.434–2.499 | 3.45% | 2.461 → 2.457（−0.16%） | 2.46 |

**结论（含抖动上界）**：同一形态的重复采样极差 **≤0.80%**（placeholder 是 2.4 µs 的极小符号，
极差 3.45% = 0.08 µs，属采样粒度）；第 5 列 `pre 两样本取小者 → post 中位（3 样本）` 的最大 |Δ| = **0.365%**
（表中按 2 位小数显示为 0.36%），落在重复采样极差之内 ⇒ **两个形态在设备侧不可区分**
（该列口径：pre 只有 2 次样本、无中位，故取两值中的**小者** —— 方向是**保守**的，它只会让 post
显得更快，而本节**不主张变快**）。与 §6 归档表（另一 commit、另一次会话）的偏差：四个实质符号
**≤0.62%**（gdn +0.41 / gdn_layer +0.60 / moe +0.62 / attn −0.35，按 5 次样本对归档值的最大
绝对值）；`m15_attn_placeholder_kernel` 是 2.4 µs 量级的极小符号，偏差 ≤2.36%（2.434~2.518 vs
2.46），属采样粒度。
⇒ **本轮 S2 形态改动没有可测的设备侧代价**（每行多的 2 次 UB `StoreAlign` + 2 次 `LoadAlign`
在 59 µs/层的尺度上低于采样分辨力）。**口径注意**：本节只说「无可测差异」，**不主张更快**。

**段级分解沿用 §6 的表**（它对本构建成立：四个实质符号最大偏差 0.62%）——**§6 的 +51% 退化结论不变**。

**顺带再复现一次 §6.1 的 host 抖动机制**（不是本节的判据，只是又一条现场）：同一次 A/B 里
`pre#1` 的 **host** P1 读到 0.3720 ms/层、P3 读到 0.4538 ms/层，而**同一二进制**的 `pre#2` 是
0.1744 / 0.0597 ms/层 —— 而这一对的**设备侧**读数完全重合。即被放大的只是 host 那一趟排空。

证据：`evidence/msprof_m52_ab.log`（A/B 全量读数 + 范围 + 结论）、`msprof_m52_op_statistic.csv`
与 `msprof_m52_run.log`（post#3 那次的原始 CSV / 完整 stdout）、`evidence/m52_dump_sha256_ab.log`
（位级见证）。

## 7. M33（docs/15 §5.3）不返工清单逐条处置

| # | 条目 | 处置 | 说明 |
|---|---|---|---|
| ★1 | GEMM 工作项升级为 `RunTile(mTile, nBlock)` 二维 | **MoE 段已落实（本 mission）；GDN 段待后续** | MoE 段的两组 grouped GEMM 现在是 **(expert, mTile, nBlock)** 三重工作项，`Run(nb, rows)` 的 `rows` 就是本 mTile 的行数（§5.7 的 6f）——即 MoE 侧已经按 mTile 收尾；**GDN 段（m14 复制）仍是 `RunTile(nBlock, curM)` 一维**，未改（scope 内但会动 M25 已验证的代码，本 mission 未做） |
| 2 | A/B 操作数统一 `(M,K)×(K,N)`、允许 K 尾块 | **已落实（接口层）** | MoE 段的 GEMM 已按 `(M,K)×(K,N)` 描述并遍历 NB/K；`static_assert` 只锁「整除」不锁死上界 |
| ★3 | MoE 每专家固定槽 → `Σt_e` 紧凑 + count 模式 `group_list` | **接口已按真实规模设计（本 mission 交付）** + **路由规模待移植（后续 mission）** | 寻址已全部改成紧凑 Σt_e（§5.7 的 6a–6g：`inv_slot` 存紧凑行号、量化的源/目标都紧凑、两组 GEMM 的 A/H/Y 基址用 `expert_offsets` 前缀和、工作项是 `(expert, mTile, nBlock)`、量化器行数由 counts 驱动）。`group_list` 的 **count 模式 int64[512]** 契约由 M42（`m22_router512`）提供，本段消费的是 `counts` + `expert_offsets`（等价语义，见 M42 的接口确认请求）；**真实 512 专家的路由器不在本 mission**（§8.1） |
| ★4 | 路由数组 `TOTAL_MAX` 定长 → `active_num`；计数排序单核 → 多核 | **接口部分落实（本 mission 交付）+ 待后续** | 段内**行数与循环上界已由 counts/Σt_e 驱动**（§5.7 的 6g/6f），不再依赖 `m*topk` 的定长常量；但①索引生成仍是 AIV0 单核标量 counting sort（decode m=1 下 active_num=2，够用；prefill 的 40970 槽位单核 94ms 是必须解决的）；②`inv_slot`/`perm_*`/`w_tk` 的**缓冲区**仍是 `M_MAX*TOPK_MAX` 定长（本实例恰好等于紧凑上界），prefill 需按 active_num 重新定尺 |
| 5 | 签名保留 `m`/`topk` 为运行期参数 | **已落实** | 融合入口的 `m`/`topk` 是运行期参数（`m15_layer_kernel.h` 的 `LayerArgs`），**没有**像 M25 那样退化成常量；`M15_TOPK` 可覆盖。而 `sliceMode/stageLimit/subLimit`（bring-up 截断开关，非模型参数）折成编译期常量（= m15 对 m14 的同款处理） |
| 6 | ws 用 host 指针 + 编译期偏移表，按 prefill 最大规模定尺寸 | **部分（接口已备）** | ws 仍是 host 指针 + 编译期偏移表（`m15_layer_resources.h` 的 `WS_MOE_OFF` 顺排），但尺寸按**当前实现**定（decode 的 M_MAX=64），不是 prefill 的 ~100MB 级 |
| 7 | UB 布局写成「窗口表 + 每 mode 变体」，每 mode 独立断言峰值 | **已落实（本节形式）** | `m15_layer_resources.h` §2 给出**两相位各自的峰值与融合峰值**，断言是 `max(相位峰值) ≤ 248KB`（不是求和）；§2.4 的 flagId 表也是按 (核型, mode) 分节登记的 |
| ★8 | flagId/BufferID 命名空间按 mode 分节登记；声明「两 mode 互斥、id 可复用」 | **已落实** | `m15_layer_resources.h` §2.4/§2.5：flagId 按 (核型, mode) 子空间登记为**执行序**并用 `static_assert` 逐对核对相邻性/用量；头文件显式写明「两相位互斥执行、BufferID 跨相位复用的前提是相位边界的 `PipeBarrier<PIPE_ALL>`」 |
| 9 | `GetBlockNum() * 2` 的修正在两条路径都保留 | **已保留** | 两段的 AIV 段都取 `GetBlockNum() * 2`（各自 donor 原样） |
| 10 | GDN 的 g/β stride-8 槽位契约与 `{H,1,0,0}` 直读方式冻结 | **已保留** | 一字未改；但它的 padding 车道在融合形态下**被实测证实是不确定的**（§5.2 的 padding 报告项）——prefill 若有别的消费者要读整槽，必须先解决这一点（已记入 §8.5） |
| 11 | conv_state/ssm_state 布局与 in-place 语义不变 | **已保留** | M1 的 `M.st_*` 与 M2 的 `M2.fused_vs_composed_state` 逐字节见证 |
| ★12 | 层边界（残差出口/入口）合约先与 M31 的 HC 规格统一 | **未落实（本 mission 明确不接 hc）** | 融合后层边界 = `Add+RMSNorm(x + 0)` / `Add+RMSNorm(moe_out + res1)` 的**归一化值**，gamma1/gamma2 仍是合成占位；与 M25 的占位口径一致、未固化得更深（接口上 MoE 段的输入/输出都是独立的 GM 指针，换 hc 只改 host 装配） |
| ★13 | attention 侧 packed indices / KV 三套布局定死 | **已落实（M82，2026-09-27）** | 唯一权威 = **`m15_attn_kv.h`**（三套 cache + packed 的物理字节布局 + 寻址宏 + paged 契约 + off-by-one 门控 + 容量 `static_assert`）；长寿 GM 按"第 k 个 attention 层"编址（`m15_loop_layout.h` 的 `AttnSlot()`），容量自检 + device 探针逐字节判据 + 两条负向对照实跑 PASS（新档 `runs=kv`，56 判定项 + 130 guard，见 **Part D · M82** 与 `evidence/m82_runs_kv.log`）。**注**：attention 段的**计算**仍是占位直通；prolog / cache 填充的数学是显式未完成项（M82-6） |
| 14 | 所有新增跨核交接登记 flagId 与 pipe，禁 PIPE_ALL；同 pipe 背靠背复用补 PipeBarrier | **已落实** | §2.4 登记表覆盖两段的全部跨核交接；新增的相位边界 barrier 也登记（`FLAG_L0_BOUND_AIV`）；**未新增任何 `PIPE_ALL` 用于跨核**（相位边界的 `PipeBarrier<PIPE_ALL>` 是核内 drain，不是跨核同步） |

## 8. 已知限制

1. **MoE 段的路由规模是缩形档（4 专家 / topk 2），不是真实模型的 512 / 10** —— 这是本 mission
   最大的范围限制，必须显式披露。原因不是选择，而是 **m13 段的结构限制**：它的 router 把
   **全部专家权重一次性预转进 UB**（`m15_moe_resources.h` 的 `UB_RT_WF = (NUM_EXPERTS+1)*HIDDEN*4`
   —— **本节原写 `m15_moe_resources.h:139`，那是行号引用**；该常量在 `m15_moe_resources.h` 里的真
   定义位置随本文件历史增删而漂移，而同一个常量在**上游** `m13_moe_layer/m13_resources.h` 里恰好
   在第 139 行 ⇒ 一度被引成 `:139`。**M82 改为按符号引用**（`UB_RT_WF`），今后不再随行号漂）
   且 `RT_LANES = 32`（一趟 Sort32 排 32 个 logit）—— 512 专家需要 513×2560×4 = **5.25MB > 248KB**，
   top-10-of-512 还需要归并树。**真实规模的路由器已有资产**：`m7_router_topk`（512 专家 / top-10 /
   m≤64 / 真实 checkpoint 的 `mlp.gate.weight` 逐字节切片 / 对 `moe_block_ref.py` 全 PASS），
   而 m13 的 S2 本就是 m7 的缩形改造。**后续项（已确认，优先级最高）**：把 m7 的流式 router
   移植进 MoE 段 → 真实 512/10；同时要解决「decode 只读被选中的 10 个专家权重（24.6MB/层）、
   而权重槽是按 NUM_EXPERTS 连续编址」的取专家问题（1.26GB/层 × 48 层无法常驻）。
   本 mission 的 MoE 权重是**真实 checkpoint 的前 4 个专家切片 + 真实共享专家**（不是合成数据），
   但**路由语义是 4 选 2**，与真实模型的 512 选 10 不同。
2. **层级 norm gamma 是合成占位**：checkpoint 里没有 `[2560]` 形状的 MoE 段 norm
   （归一化在 `*_hyper_connection.hc_norm.weight`，`hc_count=4`），与 M25 的 GDN gamma
   同一处置（确定性合成、按层加盐）。替换点与 M25 §4 一致。
3. **hyper-connection 未接入（本 mission 明确不做）**：接口给它留了位（相位 A/B 之间只经
   `yLayer` 一个 bf16 缓冲；层边界合约未固化）——见 §7 的 ★12。
4. **attention 仍是占位直通**：12 个 attention 层的相位 A 不贡献任何计算（`y = x`）。
   ~~真 QSA 接入时只替换 `m15_attn_passthrough_body`，层循环与 MoE 段零改动。~~
   **该声明不成立（M76 证伪，M82 复核）**：替换**调用点**确实只有一处，但现接口（`(xIn,yOut,bytes)`
   + `LayerArgs`）不带 attention 需要的任何输入（层号/权重/cache/pos/slot），且 AIC 侧对 attention
   完全缺席 ⇒ 真实接入必须改接口面。逐项清单与已落地的布局部分见 **Part D · M82**（M82-6/M82-8）。
   attention 侧的 cache 布局与门控契约已冻结（`m15_attn_kv.h`）；**prolog 与 cache 填充的数学未实现**。
5. **g/β 的 padding 车道不确定（§5.2 实测）**：不影响本 mission（消费者只读 `[h][0]`），
   但若将来有别的消费者读整槽，必须先把这个 quirk 消掉（生产者只写有效元素或用 `DataCopyPad`）。
6. **合成权重档下 MoE 段会出 NaN**：`M15_SYNTH=1` 生成的随机字节不是合法的 MXFP4（e8m0 scale
   可以是 0xFF=NaN），MoE 段输出 NaN，于是所有涉及 MoE 的判据都会退化成「NaN==NaN 恒真」
   ——**这正是 `M.finite` 非空洞性判据的用途**（它在 synth 档会红）。故 **synth 档只能用于
   同步/串扰冒烟，不能作为任何 MoE 数值证据**；验收一律用真实 checkpoint 档。
7. **接口契约测试未交付（本 mission 最大的未完成项，tower 裁决的关键交付）**：见 §5.7 的
   第 4 条。当前 `m` 在融合路与 MoE-only 路都固定为 1，因此**没有覆盖「m>1 + 非均匀分布 +
   空专家/单 token 专家」**这一将来 512/10 必然遇到的情形。接口本身已按真实规模设计
   （§5.7 的 6a–6g + numpy 在紧凑寻址下的 128+20 条判据），但「接口在非均匀/可变 active_num
   下正确」这一**实证**缺失。构造方法（已设计未实现）：用 one-hot router 权重把
   `logit_e = x_norm[e]` 变成可设计的排序 → 指定每 token 的 top-k 集合 → 得到任意
   counts 分布（含 0），再用 `check_moe_ref.py`（读同一套 `expert_offsets`）逐段验证。
8. **M2 的等价性是传递论证，没有重跑 host 参考链**：融合形态 ~= 两次启动组合（M2 逐字节），
   两次启动组合 ~= host 参考链（M25 的 Ver B，506 条判定项）。本 mission 没有把 host 参考链
   直接接到融合形态上；若要一步到位，需要把 Ver B 的参考链驱动改成从融合路的 dump 取锚点。
9. **prefill 未覆盖**：`m=4097` 的 prefill 是另一个入口符号（`m15_layer_kernel_*_prefill`，
   **已预留未实现**）；当前 kernel 内**没有按 m 的分叉**，但 tiling 是 decode 定死的
   （BASE_M 单 tile、M_MAX=64、每专家固定槽），prefill 需要 ★1/★3/★4 三条一起做。
10. **未做任何性能调优**：段间仍 barrier 串行、无跨段权重预取（与 m13/m14 同一状态）；
   MoE 段的 AIC 端 item 枚举仍是「按槽位扫」而非紧凑枚举。§6 的分解只作设计输入。
11. **复制改造的代价**：`m15_moe_layer.h` 由 `lift_moe_segment.py` 从
   `m13_moe_layer.asc` 的 device 段（**内容锚点** = 首个顶格 `namespace {` 到层 kernel 入口前
   最后一个 `}  // namespace`，**不写行号**：上游每加一行行号就会漂）机械生成，**上游 m13 的
   后续变更不会自动流入**；`m15_gdn_layer.h` 同理（M25 已披露）。`lift_moe_segment.py --check`
   可验证入库文件与抽取规则一致；与上游的差异是逐行可枚举的 **6 类**替换 + **1 条上游不变量
   断言**（见 §9 的差异表与 §5.9）。

   **M32 delta（review r1 的 P1 修复，2026-09-26）**：本分支原来的 donor 是 **pre-M32** 的
   m13（分支点 `2b9cb36`），而 M32（`feat/mx-quant-inf-parity-fix-across-m2-m5-m13`，
   `3fc5322` + `e93f521`，已在 main）恰恰改了**被抽取区间内**的 `MxQuantComputeScale`：
   新增 `NAN_CUSTOMIZATION = 0x7f81` + `Duplicate(nanRegTensor, …)` +
   `Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult)`（与官方
   `add_rms_norm_dynamic_mx_quant_common.h:412-413` 同源：组内非有限时让
   `Mul(±Inf, NaN) = NaN → Cast<fp4> = 0.0`，否则 `Mul(±Inf, 2^-126) = ±Inf → Cast` 饱和成 ±6）。
   本轮已 **rebase 到 main 并重抽**（`m15_moe_layer.h` 现含该修复），同时把
   `SRC_BEGIN/SRC_END` 从**行号锚点**换成**内容锚点**（否则上游每次加行都会静默错位、
   甚至写出截断文件）。
   **对既有判据中性（已实测复跑）**：M32 的 `Select` 只在**组 amax 非有限**时发射，本档
   （真实 checkpoint 4 专家切片）数据全有限 ⇒ 逐字节/位级判据结果不变。

   **M50 delta（M52 重抽，2026-09-26）**：M50 在 m13 上把 S2 的 `Extract` 改成厂商 VF 形态、
   并给 `Sort32` 补了依据注释。本分支那份副本是 M40 在 M50 之前抽的，**故本轮重抽**：
   `m15_moe_layer.h` 的 S2 现在与上游**逐行相同**；M40 那条「在脚本里手写 Extract 替换」的旧
   规则 7 目标文本已失效（脚本自身 AssertionError）⇒ 已改成**断言** `assert_upstream_vf()`
   （**不再做替换**）。替换类别数因此由 7 降到 **6**。位级影响面（含「没有舍入点改动」的三条
   见证）见 §5.9。**这是本项披露的第一条「上游变更已回流」实例**：流程 = 重抽 → `--check` →
   全量判据复跑 → dump sha256 对拍。

12. **紧凑专家槽改造带来 +51% 的设备侧性能退化（已知、已定位、未修；非正确性问题）**：
   MoE 段从 **39.06 → 58.96 µs/层**（设备侧 msprof 口径，§6 表）。**根因是并行度、不是算术**：
   改造前的扁平工作项 `item = (slot, nb)` 按 `bid` 条带摊到 28 个 AIC（每个 AIC ≤1 个 item）；
   改造后 `(expert, mTile, nBlock)` 三重循环里**内层 `nb` 也按 `bid` 条带**，而
   `GU_NBLK = 5`（down 为 10）< 28 ⇒ 只有 `bid < 5` 的 AIC 分到活。**修法（一行级）**：
   把「一个专家内的 `(mTile, nb)` 组合」先展平成一个长度 `mTiles*NBLK` 的 item 序号再按 `bid`
   条带（`it = bid; it < mTiles*NBLK; it += numBlocks`），即可同时保住 `(expert, mTile, nTile)`
   语义与 28 核并行度。**完整修法与原因写在 §6 末的同一段 blockquote 里**（本行只是把它登记进
   编号限制清单，免得只在 §6 而漏掉）。**用户明确现阶段只求正确性 ⇒ 本 mission 未做；
   后续立项请以 §6 那段为准。**

## 9. 与 donor（m13 / m14）的差异表

| 文件 | 来源 | 差异 |
|---|---|---|
| `m15_moe_layer.h`（新） | **main 的** `m13_moe_layer.asc` 的 device 段（**内容锚点**：首个顶格 `namespace {` 到层 kernel 入口前最后一个 `}  // namespace`；**不写行号**，行号随上游变动漂移） | **`lift_moe_segment.py` 机械生成**，共 **6 类**替换 + **1 条上游不变量断言**：① 6 个顶层匿名 namespace → `namespace M15M`（与 GDN 段共享同一 TU，避免同名工具重定义）；② 删 6 行 `using namespace M13;`；③ 资源表 include 改名；④ 注释/行尾 `M13` → `M15M`；⑤ **出口 y 独立成 GM 缓冲**（`MoeLayerPtrs` 加 `yLayer`、S10 的 m6#2 出口由 `ws+WS_YFINAL` 改为 `args.yLayer` —— 与 m15 对 m14 的同一处改造）；⑥ 紧凑专家槽 + `(expert,mTile,nBlock)` 工作项（见 §5.7）。**不再有「Extract → VF」这类替换**：S2 的 VF 形态与 `Sort32` 依据注释**已在上游 m13**（M50 落地），副本逐行取得，脚本只做断言 `assert_upstream_vf()`（见 §5.9）。**段序与同步一字未动**；⑥ 是寻址改造（算件逻辑不变）；`--check` 可复核（含内容锚点） |
| `m15_moe_resources.h`（新） | `m13_resources.h` | namespace 改名 + **§3 的 flagId 槽位重编**（§2.4 的表：按 (核型, mode) 与 GDN 段错开、新增相位边界 id）；其余常量/UB/L1/GM 布局一字未动 |
| `m15_layer_resources.h`（新） | 新写 | 融合形态的跨段登记表（ws 顺排、MoE 权重槽、UB/L1/L0C 峰值、flagId/BufferID 登记、入口符号表、M33 ★8 的「两 mode 互斥」声明） |
| `m15_layer_kernel.h`（新） | 新写 | 融合入口（`m15_layer_kernel_gdn`/`_attn`/`m15_moe_segment_kernel`）+ 相位边界同步 + 相位 B 的指针组装 |
| `lift_moe_segment.py`（新） | 新写 | 抽取脚本 + `--check`（用于证明「与上游的差异只有上面 **6 类** + 1 条断言」。M52 起含 `assert_upstream_vf()`：上游若退回 memory-based `Extract` 或缺 `Sort32` 依据注释，重抽即失败） |
| `m15_attn_layer.h` | M25 | 把 kernel 主体抽成 `m15_attn_passthrough_body()` inline 函数，`__global__` 入口只是薄壳（**数据面/同步/BufferID 用法一字未动**） |
| `m15_layer_loop.asc` | M25 | 新增：MoE 权重装载（`H_LoadMoeW`/`H_PutMoeSlot`）、MoE 权重来源校验、dump（`moe_layout.txt` + 每层 MoE ws）、`runs=m`（M1/M2）、Ctx 新缓冲；**M25 的 Ver A/B/C 函数体未改**（唯一例外：`A.direct_ws` 改 padding 感知，见 §5.5） |
| `m15_moe_host.h`（新） | 新写 | MoE 权重（host 侧）、权重来源判据、验证 M1/M2、padding 感知比较、dump |
| `m15_layer_ref.h` / `m15_gdn_layer.h` / `m15_gdn_resources.h` / `m15_loop_layout.h` | M25 | **未改动** |
| `slice_layer_manifest.py` | M25 | 新增 `MOE_ROLES`（48 层 × 12 行）与专家切片；GDN 部分未改 |
| `weights_manifest.txt` | M25 | 324 行（GDN）→ **900 行**（+576 MoE） |
| `check_moe_ref.py`（新） | 新写（**复用** m13 的 `check_ref.py` 与 `moe_block_ref.py`） | 消费「整块 MoE ws + 布局表」的 dump，权重取真实 checkpoint；判据口径与 m13 一致 |
| `check_ref.py` / `parse_msprof.py` | M25 | 未改动（`check_ref.py` 消费的 M25 dump 命名未变） |

## 10. 文件

| 文件 | 说明 |
|---|---|
| `m15_layer_loop.asc` | 单 TU：融合入口 + host（权重装载/来源校验/三验证/验证 M/dump/计时） |
| `m15_layer_kernel.h` | 融合 kernel 入口（两个相位 + 相位边界） |
| `m15_layer_resources.h` | 融合形态资源登记表（UB/L1/L0C/ws/权重槽/flagId/BufferID/入口符号） |
| `m15_moe_layer.h` | MoE 段 device 代码（机械生成自 m13） |
| `m15_moe_resources.h` | MoE 段资源表（m13 的表的融合版：flagId 重编） |
| `m15_moe_host.h` | MoE 权重（host）+ 权重来源判据 + 验证 M1/M2 |
| `m15_gdn_layer.h` / `m15_gdn_resources.h` | GDN 段 kernel 与资源表（← m14，未改动） |
| `m15_attn_layer.h` | attention 占位直通（主体抽成 inline 函数） |
| `m15_loop_layout.h` | 48 层循环 GM 平面图（未改动） |
| `m15_layer_ref.h` | host 参考链与判据工具（未改动） |
| `lift_moe_segment.py` | m13 → m15 的 device 段抽取脚本（`--check` 可复核差异） |
| `slice_layer_manifest.py` / `weights_manifest.txt` | 权重 manifest（GDN 324 + MoE 576 行） |
| `check_moe_ref.py` | MoE 段的独立 numpy 交叉校验（复用 m13 的参考链 + moe_block_ref） |
| `check_ref.py` / `parse_msprof.py` | M25 的独立校验与 msprof 汇总（未改动） |
| `evidence/` | 验收证据（§5.6） |

---

## 11. 验证 Pf（prefill 判据；M142 追加，口径 = M136）

> 本节由 M142 追加，把 prefill 的「验证 Pf」判据同步到 **M136 已落地并实跑过**的那一套，
> 替换 M110 的「整面毒值」口径（那套只证**结构占位**：段体未落地时 `Pf.gdn` 走逐行 `x→y`，
> 判据是"整面逐字节相等 + 毒值行计数"；历史读数与二进制见
> `evidence/prefill_contract/README.md` §6.1）。
>
> **与 §8 第 9 条（§8.9）的关系（不矛盾）**：第 9 条的「prefill 未接」指**整链**未接
> （48 层链 / 四相位 / `subOut` 数据流）；本节描述的是 **KIND_GDN 相位 A 这一段已通**、
> 其余相位（H1/H2/B）与 attention 仍未接。两条各自成立，互不改写。

**三条判据（缺一不可）**：

1. **段体真产出**：设备上 `m15_layer_kernel_gdn_prefill`（KIND_GDN 相位 A）真跑，出口 `o`/`ht`
   是段体产物 —— 不是逐行 `x→y` 的结构占位，也不是未写毒值。
2. **m23 参考对拍**：`m23_gdn_prefill/check_ref.py` 从 dump 的**干净输入**用 numpy float64 复算，
   逐元素 `|dev − ref| ≤ rtol·max(1, |ref|)`（rtol = 2e-3）。实测 m=1 与 m=4097 都 PASS（`o`/`ht` 超界 0）。
3. **负向对照**：`Pf.gdn_mut1`（`--mutant 1`，只把**送设备的** h0 翻倍、host 侧参考仍吃干净副本）
   **必须变红** —— 实测 m=1 `o` 超界 4190/6144、m=4097 `o` 超界 130298/25171968；
   mut1 的 q/k/v/g/β/h0 dump 与 clean **逐字节相同**，只有设备 `o`/`ht` 不同 ⇒ 判据有判别力。

**覆盖范围（不得读大）**：只到 **KIND_GDN 的相位 A（GDN 扫描段）** —— 不是「整层」，
也不是「prefill 打通」。q/k/v/g/β 为**宿主合成**输入（段体 B1 的契约是"已过 prolog 的 q/k/v/g/β"，
**不含** in_proj / out_proj / conv / l2norm）；四相位的 H1/H2/B 未开；段间数据流未连通；
attention 未接（`KIND_ATTN` 在 `wire=1` 时走 `H_CmpOk(C, false)` 判红 —— 只在代码级核过，未加设备档）。
改前/改后两侧读数、设备命令、`npu-smi` 快照、逐档对拍与复算脚本见
`evidence/m136_prefill_gdn/README.md`（其 §6"显式未做 / 限度"就是本节覆盖范围的口径来源）。

（改前侧：无条件调三条 `M15L_PhaseBoundaryAiv` ⇒ AIC 也走 mode-0，m=1 档在 `Pf.gdn：GDN 段入口启动`
后于 `aclrtSynchronizeStream` 停住、`exit=124`；收进 `if ASCEND_IS_AIV` 后三档 `exit=0`。）

---

## 12. M151：prolog 链路挂进相位 A（q/k/v/g/β 改由设备产出）

相位 A 的 q/k/v/g/β 不再只由宿主合成 —— `M15_PREFILL_PROLOG=1` 时改由设备产出：
H1 起一次产 BLK → S2 in_proj（`m11 bf16_gemm` donor 的 m15 lift，cube）→ S3 prolog
（`m9_gdn_prolog` 的 `gdn_prolog_mt_kernel`，逐字抽取，AIV）→ 相位 A 读设备产出。
宿主合成路径保留为 `M15_PREFILL_PROLOG=0`（缺省）的对照/回归档。

**挂载形态 ≠ 终态（如实登记的差距）**：本节与 §1 描述的是 **host 编排的 4 次 launch/层**
（H1 → S2 → S3 → 其余相位），而同一个 `m15_layer_resources.h` 头文档化的**终态**是
「**每层一次 `__mix__(1,2)` 启动、kernel 内四个相位**」（`m15_layer_resources.h:4`）。
本次是**打通步骤**、不是终态；差距与后果（额外 launch、`pfQkvzbaDev` 135 MB/层的 GM 往返、
prolog 与相位 A 之间无 UB/L1 复用）与后续 build 项见
`evidence/m151_prefill_prolog_wiring/README.md` §7 第 6 条。

**只接 prolog 链路**：**epilog**（`wsO → hcAttnOut`，S5/S6/S7）与 **attention B2/B3 未接**；
不得读成「整层四相位打通」。相位 B m=1、m=4097 `h2.blk` ulpMax 11 两条既有红项**原样保留**。

- 契约（逐条 `文件:行`）：`evidence/m151_prefill_prolog_wiring/CONTRACT.md`
- 设备档 + 离线对拍 + 负向对照 + 复算：`evidence/m151_prefill_prolog_wiring/README.md`
  （`bash evidence/m151_prefill_prolog_wiring/reproduce.sh`）
- 新平面/峰值登记：`m15_layer_resources.h` §3c-bis（钉死 `PfPeakUnfilled()==3` 不变）
