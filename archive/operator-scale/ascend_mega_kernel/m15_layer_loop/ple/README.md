# M85 — PLE 独立 kernel（规格钉死 + 最小基线实现 + 判据骨架）

> 分支 `feat/m85-ple-kernel-spec-pin-and-standalone-i`（**未合入**；引用本文件内容时按 `docs/17 §9.6`
> 标注分支与状态）。**不接线**：`m15_layer_kernel.h` §3b 的挂载点、host、README、`.asc`、manifest
> 全部未动（那些归 M82；接线等 M82 合入后由后续 mission 做）。
>
> 本文件是 **M85 r1 复审（`p1-1items / fix-then-merge`）修完后的第 2 版**：
> 复审的两条 P1/P2 已修，并按复审要求做了一次**完整的上游分歧回扫**（`ple/DIVERGENCES.md`）。

## 0. TL;DR

| 项 | 结论 |
|---|---|
| PLE 规格能否钉死 | **能**。`ple/PLE_SPEC.md`（输入/权重/输出/步序四表）+ `ple/DIVERGENCES.md`（与上游两份实现的**逐项**对照）。规则来源 = `docs/14` + `/workspace/vllm/vllm/models/qwen4_exp/` 的 triton 与 eager 两条实现 + checkpoint header 实测 |
| 独立 kernel 形态 | `m15_ple.asc`：**自己的两个入口符号**（`m15_ple_ids_kernel` =①、`m15_ple_body_kernel` =②③④⑤）+ **自己的 main()**；`m15_layer_loop/CMakeLists.txt` 注册 `m15_ple` 目标 |
| 参考/对拍 | `m15_ple_check.py`（numpy/float64 独立参考 + 分档判据）、`m15_ple_mutants.py`（变异矩阵驱动）、`ple/gen_ple_data.py` + `ple/pick_observable_seed.py`（数据） |
| 判据现状 | **14 条判据 / 0 FAIL**（第 14 条 `B_dev.fail` 由 M111 新增；M85 的 13 条读数在 `ple/logs/run_all.log`，M111 的整批读数在 `evidence/ple_wire/M111_LANDING_PATH.md`，rc=0）；**咬合力被变异演示过 = 14/14**（脚本强制，见下） |
| 变异矩阵 | **15 个变异位**：每个都有指定判据 FAIL、基线全 PASS，且**并集覆盖全部 14 条判据**（`ple/logs/mutants.log` 的 `RESULT\|coverage\|14\|14\|(none)\|OK`，rc=0；第 15 位由 M111 新增，见 `evidence/ple_wire/M111_LANDING_PATH.md §2.3`） |
| 参考的输入 | 按 `docs/17 §1.3` 三分逐条列举（§4.1）：① 声明输入 + ② 上游输出（逐条指明覆盖它的判据）+ **③ 无** |
| 负向对照 | 成立（`ple/logs/neg.log`）：① 取模方向反转后主判据 FAIL（160/160、31/32），且变异产物 == 用反转规则算出的参考 |
| 覆盖的步序 | decode 全链 **① ② ③ ④ ⑤**（含 conv 状态移位、**null 槽位行**） |
| 未覆盖 | prefill/spec 的 short-conv 独立 writeback；与 §3b 挂载点的实际接线（§7） |

## 0.1 M92 补丁（真实表 host-mapped gather）—— 本文件其余部分仍是 M85 的原始交付

> 分支 `feat/m92-ple-ngram-real-table-host-mapped-gat`（**未合入 `main`**）。规格/取证/判据读数见
> **`ple/REAL_TABLE.md`**（本目录），那里是本 mission 的正式交付物；本节只说明**本文件里哪些条目被推进了**。

| 项 | M85 的状态 | M92 之后 |
|---|---|---|
| §7 **U2** 95.37 GiB 真实表 | 阻塞（`docs/14 §10` 第 1 条），用缩减词表合成表 | **阻塞已解除**（前置能力由 M86 实测证实）：`ple/REAL_TABLE.md` 给出落地形态规格 + 真实 checkpoint 取证（8 个真实行 id 逐字节一致；整片 shard_0 的完整行集合 = 762.94 MiB 真实规模跑过一次）。**仍**未做整表/128 片全量遍历（`REAL_TABLE.md` §9 U-H） |
| §7 **U6** 设备侧判据薄弱 | 只有 `dev_range_fail`（①）与 `dev_fail`（只数 null 行）；②③④⑤ 全靠 host 参考咬 | **已补**：`Hd.row_fail`（设备侧**错行计数**，1022 个窗口内 item 全部被设备自己比过）、`Hd.miss`（越窗行计数）、`Hd.cores`/`Hd.nonvac`（非空洞守卫）。负向对照 bit14 实测把 `Hd.row_fail` 打到 1022（`REAL_TABLE.md` §7）。**M111 再补**：`dev_fail` 原先是**空读数**（`bad` 从不自增），现由 `PleGather` 真的数「词表外的 id」并进判据（`B_dev.fail`）；以上三处设备侧落盘都不再走 GM 标量 `SetValue`（见 `evidence/ple_wire/M111_LANDING_PATH.md`） |
| §1 判据表 | 13 条（`B1..B5`） | M92 **不动**：新增的 10 条走**独立的 `H_*` 名字**与 `check_hm`，既有 13 条在本 tip 复跑仍 **13/13 PASS**（`ple/logs/run_all.log`）。**M111 动了**：新增第 14 条 `B_dev.fail`（判 `B_body_meta.txt` 的设备侧 `dev_fail`）—— 理由与「把被测对象弄坏必须变红」的见证（变异 bit15）见 `evidence/ple_wire/M111_LANDING_PATH.md` §3 |
| §7 **U1 / U3 / U7** | 未做 | **仍未做**（M92 的 mission 边界明示不在其内） |

M92 的读数与命令：`ple/logs/{real_table_probe_evidence,real_scale,emit,hm_run,hm_neg,mutants,hm_recount,run_all}.log`
（其中 §6.2/§6.3 的重算由 `ple/recount_hm.py` 给出，**只读、非判据**）。
入口符号仍是 M85 的两个（`m15_ple_ids_kernel` / `m15_ple_body_kernel`），窗口模式由 `M15_PLE_HM=1` 打开，
**默认关** ⇒ 本文件下面所有读数与命令在默认路径上仍然成立。

## 0.2 M124 补丁（③ kv 投影：AIV 逐列 GEMV → **cube `Mmad`**）—— 本文件其余部分仍是 M85/M92/M111 的交付

> 分支 `feat/m124-ple-plegemv-cube-mmad-compliance-re`（**未合入 `main`**）。完整改动清单、判据、真 shape
> 读数、flagId 预算表与「未做」清单见 **`evidence/ple_wire/m124/M124_CUBE_MMAD.md`**；本节只说本文件里被推进的条目。

| 项 | 改前 | M124 之后 |
|---|---|---|
| ③ 的实现 | AIV 向量：每列一次 40 chunk `Mul`+`Add` → `Reduce<SUM>`（56 条 AIV 按列条带） | **AIC cube `Mmad`**（`C[n_tok,KVW] = emb·wcatᵀ`，`Nd2Nz`→`LoadData2D`→`Mmad`→`Fixpipe`，28 条 AIC 按 `BASE_N=160` 的 N-tile 轮转条带，K=2560 分 40 块）。**权重排布不变**（`wcat` 的 `[12800,2560]` 行主序正是 donor `Bf16Gemm` 的 `B[N,K]` 形态，不做在线转换） |
| §1 判据表第 8 行 `B3.kv` | T3，`eps=96·2^-24=5.72e-6` | 档位不变（T3），**界换成 cube 推导式** `EPS_MMAD=2560·2^-24=1.526e-4`（`docs/17 §1.1` 的「mmad 的 `k·2^-24`」）；**改前/改后同一条判据、同一条界**：两版都 `bad=0/25600` |
| §2 的 ε 表 | 只有 `EPS_REDUCE`（③ 的 VF 路径界） | 新增 `EPS_MMAD`（③ 的 cube 界，它**盖住** `EPS_REDUCE`）；`EPS_REDUCE` 保留给 ④ 的判据 |
| ③ 的核间同步 | 全体 AIV 的 mode-0 自封 barrier（`FLAG_B2/B3`） | **跨核型**：`AIVs→AIC`（mode 2）→ 全体 AIC 的 mode-0 对齐 → `AIC→AIVs`（mode 2）。形态照 `m15_attn_prolog_probe.h` / `m15_gdn_layer.h` 的既定先例；`FLAG_B2/B3` 的**核型与 mode 都变了**（登记见 `m15_layer_resources.h` 的 `FLAG_SEQ`） |
| 层路径（`m15_layer_kernel.h`） | ③ 在 AIV 段里跑 | 层 1 的 **AIC 分支新增 `M15L_PleAic`**（挂在 hc(H1a) 与 hc(H1b) 之间）；AIV 段的 `PleGemv` 只做握手 |
| 层路径的 kv 判据 | 无（`Pw.planes.nonvac` 一类判不了 kv 的数值） | **新增 `Pw.kv.T3`**（`m15_layer_loop.asc` 的判据块 3b）：参考**完全取自 host**（窗口字节 + host 的 ids 参考重建 emb，× host 侧 `wcat`，double 下点乘），界 = `k·2⁻²⁴·Σ|terms| + 1.0·ulp(out)`；配 `Pw.kv.nonvac` 非空洞守卫。读数：A/B 档 `bad=0/12800`；**③ 不跑**的负向对照档 ⇒ `bad=12753/12800`（判据变红） |
| §7 **U4** | 「③ 逐列 `DataCopy` + 逐列 `Reduce` 的 GEMV」列为性能未做项 | **该条已消解**（③ 不再是逐列 GEMV、也不再每列一次 `Reduce`）；U4 余下的「段内全 `PipeBarrier<PIPE_ALL>`、未引入 BufferID 流水」仍**未做**，且**新增**「未做设备侧计时」的读数缺口（见 §7 U4 与 M124 报告） |

**与既有读数的关系**：M85 的 13 条 + M111 的第 14 条判据在本 tip 上**逐条复跑**：
`ALL PASS`（14/14，`bad=0`），变异矩阵 15 位**全部 OK**、覆盖 14/14 —— 读数归档在
`evidence/ple_wire/m124/M124_{check_base,check_new,mutants}.log`。改前/改后的设备 dump 逐平面差异见
`evidence/ple_wire/m124/M124_plane_diff.log`（**kv 只有 9/25600 个元素差、最大 1 个 bf16 格点；最终 `out` 平面逐字节相同**）。

## 1. 判据（`docs/17 §1.1` 分档 + 分档理由；单位**统一按元素**，字节差只在 detail 里）

| # | 判据 | 档 | 量（元素） | 结果 | 分档理由 | 咬合力演示（变异位） |
|---|---|---|---|---|---|---|
| 1 | `B1.ids.A` n-gram id（**真实词表常量**；T=10、2 请求 6+4 token、含 EOS 回退与 chunk 边界 `c=1`） | **T1** | 160 | bad=0 | 纯整数域 + 位运算，无浮点 | bit0 / bit1 / bit10 |
| 2 | `B1.ids.A.range` 值域 `[0,320001536)` | T1-结构 | 160 | bad=0 | 结构判据 | **bit10**（只演示**上界**，见 §7 U8） |
| 3 | `B1.ids.B` id（缩减词表档，供 ② 消费） | **T1** | 32 | bad=0 | idem | bit0 / bit10 |
| 4 | `B1.ids.B.range` | T1-结构 | 32 | bad=0 | idem | **bit10**（同上） |
| 5 | `B2.emb` 表行 gather + 展平 | **T1** | 5120 | bad=0 | 索引类 + 无算术字节拷贝 | bit6 |
| 6 | `B5.null.out` **null 槽位行**的 out（纯 bf16 加链） | **T1** | 10240 | bad=0 | 无超越函数、舍入链可完整建模 | bit9 |
| 7 | `B5.null.state` null 行的状态**逐字节不变** | **T1** | 92160 | bad=0 | 纯拷贝 | **bit11** |
| 8 | `B3.kv` kv 投影 GEMM（k=2560 长累加；**M124 起 = cube `Mmad`**） | **T3** | 25600 | bad=0（改后 `max(d/bound)=0.976`；改前重建 `0.979`；界 = `EPS_MMAD`） | 触发条件③「长累加 / cube 累加」 | bit7 |
| 9 | `B4.gated` 门控输出 | **T3** | 20480 | bad=0 | 触发条件①（`Rsqrt`/`Sigmoid`）+ ②（2560 长归约） | bit2 / bit5 |
| 10 | `B4.normed` 分组归一输出 | **T3** | 20480 | bad=0 | idem | bit2 / bit5 |
| 11 | `B5.out` 卷积 + SiLU + 残差加 | **T3** | 20480 | bad=0 | 含 `Sigmoid`（SiLU）+ 4 tap fp32 累加 | bit3 / bit4 / bit5 / bit9 / **bit13** |
| 12 | `B5.state` conv 状态移位 | **T3** | 184320 | bad=0 | 值来自 ④ 的输出（已带超越函数误差） | bit8 / bit11 / bit12 |
| 13 | `B5.state_evolve` 状态演化（**设备产物** `state_out` vs **设备输入** `state_in`，活跃行） | **T4** | 92160 | 变化 92082 元素 | `docs/17 §4`「状态演化」 | **bit12** |

**分档计数**：T1 家族 **7** 条 + T3 家族 **5** 条 + T4 **1** 条 = **13** 条；每条都在
`ple/logs/run_all.log` 里写了「能 FAIL 掉的错误类别」。负向对照的 4 条（`N1_neg.*`）是**对照项**，
不计入 13。

**「有咬合力」是一个被脚本验证的计数，不是标签**（M85 r2 复审 P3）：
`m15_ple_mutants.py` 在跑完 14 个变异位之后，取所有变异体 FAIL 集合的**并集**，要求它覆盖
**每一条**基线判据，并打印 `RESULT|coverage|13|13|(none)|OK`；任一判据没被任何变异体咬到 ⇒ **退出码非 0**。
当前结果：**13 / 13 全部被变异演示过**（上表最后一列逐条给出是哪些 bit）。
判据脚本 `m15_ple_check.py` 自身**不再声明**任何"有咬合力 N 条"（旧版把"非取反行条数"印成该标签）。

**变异矩阵（`ple/logs/mutants.log`，`m15_ple_mutants.py` 一键复现）**

| bit | 变异（故意写错） | 对应分歧 | 期望 FAIL 的判据 | 实测 FAIL 集合 | 结论 |
|---|---|---|---|---|---|
| 0 | ① 取模方向反转 | D5 | `B1.ids.A` | `B1.ids.A,B1.ids.B` | OK |
| 1 | ① ctx 列退回「与 c 无关」 | **D3**（r1 P1） | `B1.ids.A` | `B1.ids.A` | OK |
| 2 | ④ 跳过 dot 和的 bf16 物化 | **D4**（r1 P2） | `B4.gated` | `B4.gated,B4.normed,B5.out,B5.state` | OK |
| 3 | ⑤ tap 归属反转 | D2 家族 | `B5.out` | `B5.out` | OK |
| 4 | ⑤ 残差加分组反转 | D2 家族 | `B5.out` | `B5.out` | OK |
| 5 | ④ value 换成 key 切片 | D6 | `B4.gated` | `B4.gated,B4.normed,B5.null.out,B5.out,B5.state` | OK |
| 6 | ② head 落点顺序反转 | D7 | `B2.emb` | `B2.emb` | OK |
| 7 | ③ key/value 输出块对调（**M124 起 cube 档：N-tile 读取基址 ±HID**） | **D1** | `B3.kv` | `B3.kv` | OK |
| 8 | ⑤ 状态移位方向反转 | D8 | `B5.state` | `B5.state` | OK |
| 9 | ⑤ null 槽位整行跳过（漏写 out） | **N1**（回扫新发现） | `B5.null.out` | `B5.null.out,B5.out` | OK |
| 10 | ① 漏掉取模与偏移（`id = mixed`） | ① 值域判据 | `B1.ids.A.range` | `B1.ids.A,B1.ids.A.range,B1.ids.B,B1.ids.B.range` | OK |
| 11 | ⑤ null 行**也**写状态 | N1 家族 | `B5.null.state` | `B5.null.state,B5.state` | OK |
| 12 | ⑤ 活跃行**不写回**状态 | D8 家族 | `B5.state_evolve` | `B5.state,B5.state_evolve` | OK |
| 13 | ⑤ 跳过 `conv_output` 的 bf16 物化 | **D2 本身**（那个取整点） | `B5.out` | `B5.out` | OK |

> bit13 是 r2 复审 P3 点出的"D2 的取整点自身没有专门变异位"的补位：它跳过的是 **D2 那一次**
> `conv_output = bf16(y)`（`m15_ple.asc` 的 `Adds(cr, cvt, 0f)` 分支），而不是像 bit3/bit4 那样动
> ⑤ 的别处。bit3/bit4 现在在表里改标为「⑤ 的 tap 序 / ⑤ 的残差分组」，不再冒充 D2 的判据。

## 2. T3 的 ε 推导与裕度（`docs/17 §1.1`：界必须推导）

| 符号 | 取值 | 逐项来源 |
|---|---|---|
| `EPS_MMAD` | `2560·2^-24 = 1.526e-4` | **M124 起 ③ 的界**：cube `Mmad` 在 L0C 里的 `K=2560` 次 fp32 乘加累加 ⇒ 按 `docs/17 §1.1` 的「mmad 的 `k·2^-24`」项取 `k=KNORM=2560`。它**盖住**下面的 `EPS_REDUCE`（5.72e-6）⇒ 改前（VF）与改后（cube）用**同一条界**做同一条判据 |
| `EPS_REDUCE` | `96·2^-24 = 5.72e-6` | **④** 的 fp32 逐 chunk 累加：每 64-lane chunk 各 1 次 `Mul`+1 次 `Add`（`-ffp-contract=off`）⇒ 40 chunk/列 ≈ 80 次舍入；`Reduce<SUM>` ≈ 6 次。**M124 之前**它也用于 ③ 的 VF 逐列 GEMV（同一条推导），现在 ③ 不再用它做判定（保留供 ④ 用） |
| `EPS_RSQRT` | `4·2^-24 = 2.38e-7` | `M15H::NormDonor::ComputeRstdNewtonRaphsonReg`（`m15_hc_layer.h:147-203`）：1 `Div`+1 `Sqrt`+2 步 NR |
| `EPS_SIGMOID` | `2·2^-24 = 1.19e-7` | `Exp` 官方规格 ≈1 ulp（`m5_swiglu_quant/README.md:130` 实测与 host `expf` 逐位一致）+1 `Div` |
| 输出网格项 | `1.0·ulp(out)` | 参考本身已量化到 bf16 格点 ⇒ 按 `docs/17 §1.1`「参考已量化」条取 `1.0·ulp`（`BfUlp(v)=2^(e-7)`） |

判据形式：`|out−ref| ≤ ε·Σ|terms| + 1.0·ulp(out)`，**逐元素**；`maxRel` 与 `max(d/bound)` 都是**报告项**。

**`max(d/bound)` 的读法**（M85 r1 复审 P3-4 的口径澄清）：`≈1.000` 表示"某个元素恰好差 **1 个 bf16 格点**"
—— 参考已量化、格点对格点的最大合法差就是 1 ulp，所以这正是 `1.0·ulp(out)` 项**覆盖**的情形，
不是"侥幸擦过"。本轮 `B3.kv = 0.999`（有 1 个元素差 1 格点）、`B4.normed` 修完 D4 后回到 `0.000`。
**M124 之后**：`B3.kv` 的界换成 `EPS_MMAD` 后 `max(d/bound)=0.976`（改后）/`0.979`（改前重建），
两版都 `bad=0/25600`；④ 的两条（`B4.gated`/`B4.normed`）仍是 `0.000`（③ 的 1 格点差被 ④ 的 bf16 链吸收）。

## 3. 复现（命令 → 输出/rc；均在**本 tip** 上实跑，完整转录见 `ple/logs/`）

| # | 命令 | 输出 / rc |
|---|---|---|
| 0 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` | — |
| 1 | `cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release` | `Build files have been written` rc=0 |
| 2 | `cmake --build m15_layer_loop/build -j8 --target m15_ple` | `[100%] Built target m15_ple` rc=0（仅 `-Wcce-compat` 警告） |
| 3 | `/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/gen_ple_data.py` | `ckpt == docs/14 §6.4 recomputation (3/3)`；`rows=1470`；`wrote 26 files` + `写了 SHA256SUMS（25 个输入文件）` rc=0 |
| 4 | `M15_PLE_OUT=m15_layer_loop/ple/out ./m15_layer_loop/build/m15_ple` | `aic=28 aiv=56`；`A/B ① ids: dev_range_fail=0`；`body ②③④⑤: stage_mask=15 n_tok=2 table_rows=1470 dev_fail=0`；`kernel-side OK` rc=0 |
| 5 | `/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_check.py m15_layer_loop/ple/out` | `判据合计 13 条 = 判定项 13 + 对照项 0；FAIL 0 条` / `ALL PASS` rc=0 |
| 6 | 确定性：第二次独立进程 `M15_PLE_OUT=.../out2` 后逐文件 `cmp` | 8/8 个落盘文件 `same`（`ple/logs/run_all.log` §4） |
| 7 | locale 一致性：`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 各跑一次判据（`--brief`） | **整个 `--brief` stdout** 的 md5 两者相同 = `5383b4c9e7ef8d677d2226ce76af77c4`（13 行 `RESULT\|` + 末尾 `===== ALL PASS =====`）；**只取 `RESULT\|` 行**的 md5 = `6b880e4e212d30def63b3a594c1f2d36`（r2 复审 P3：旧版把前者说成「RESULT 行 md5」，措辞错） |
| 8 | `/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_mutants.py` | 基线 13 条 FAIL 0；**14 个变异位全部 OK**；`RESULT\|coverage\|13\|13\|(none)\|OK` rc=0 |
| 9 | `M15_PLE_MUT=1 M15_PLE_OUT=m15_layer_loop/ple/out_neg ./m15_layer_loop/build/m15_ple` | 只跑 ①；`B_ids.bin` 为变异产物 rc=0 |
| 10 | `... m15_ple_check.py m15_layer_loop/ple/out_neg` | `B1.ids.A bad=160/160`、`B1.ids.B bad=31/32` ⇒ `FAILURES PRESENT`（**期望**） |
| 11 | `... m15_ple_check.py m15_layer_loop/ple/out_neg --neg` | `N1_neg.*` 4 条全 PASS（对照成立）rc=0 |
| 12 | `/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/pick_observable_seed.py` | `可观测种子 = 4`，`(t=1,s=2)` 的 `g`: spec=0.2578125 vs 漏取整=0.259765625 rc=0 |
| 13 | `sha256sum m15_layer_loop/ple/data/*`（`docs/17 §1.3` ① 类要求） | 由 `gen_ple_data.py` 直接写出 `ple/data/SHA256SUMS`（25 个输入文件），内容收录进 `ple/logs/run_all.log §6`（含 `ids_A.bin` / `table_red.bin` / `hidden.bin` / `conv_state.bin`） |

## 4. 数据与「独立参考」的来源合规（`docs/17 §1.2 / §1.3`）

- **权重**：**真实 checkpoint** `model.language_model.layers.1.ple.*`（key_proj / value_proj /
  conv1d / 3×norm，全部 BF16）。由 `gen_ple_data.py` 直接从 safetensors 字节抽出，**不经设备中间量**。
- **三个 int64 buffer**：从 checkpoint 读出，并与 `docs/14 §6.4` 的公式**逐值比对通过**；
  判据侧真值只用 checkpoint 的值（`docs/15:458` 要求的"不要自己实现质数搜索"得以遵守，
  素数搜索只作交叉验证）。
- **ngram 表**：真实表 95.37 GiB 落不下本容器（`docs/14 §6.3`）⇒ 用**缩减词表**（16 个小素数、
  `Σsize = 1470` 行）的合成随机 bf16 表；乘子仍用真实值。这是 **本 mission 唯一的合成输入**。
- **参考实现的"规则"来源**：`nvidia/ops/ple.py`（三个核 + writeback）与 `nvidia/ple_layer.py`、
  `nvidia/ngram_embedding.py`，在 `m15_ple_check.py` 里**独立重写**为 numpy/float64（**不 import 上游**）。
- **参考实现的"输入"来源：按 `docs/17 §1.3` 三分逐条列举** —— 见下面的 **§4.1**（M85 r2 复审 P2 指出
  旧版这里写成「不读设备中间量」，与代码相反；现已按事实改写并给出三分表）。
- **hidden 种子是"选"出来的，如实说明**：`--hidden-seed 4`（默认值）由
  `ple/pick_observable_seed.py` 从 `s=0..` 搜索得到 —— 选它的唯一目的是让
  **D4（dot 和的 bf16 物化）在随包数据上可观测**（该分歧只在 ≈2%/gate 项上改变 `g`）。
  **这不是"挑数据让判据好看"**：搜索的目标是"让分歧**暴露**"，不是"让判据通过"；
  基线在选出的种子上同样 13/13 PASS。搜索过程与判据都在 `pick_observable_seed.py` / `mutants.log` 里，
  reviewer 可换任意种子复算（换种子后变异 bit2 可能不可观测 —— 那正是"必须选"的原因，见 §6）。
- **未用官方稀疏 dump**：`m21_layer_ref/` **不含 PLE 段**（其 README §1 明写"不实现 PLE"），
  故不存在"拿官方稀疏 dump 当参考"的问题。

### 4.1 参考的输入三分（`docs/17 §1.3`：① 声明输入 / ② 上游输出 / ③ 被判量自身产物）

**逐条列举**（`①` 必须给路径 + sha256；`②` 必须**逐条指明覆盖它的那条判据**）：

| 参考（判据） | 参考吃的输入 | 类别 | 合法性依据 |
|---|---|---|---|
| `B1.ids.A` / `B1.ids.B` / 两条 `.range` | `ple/data/{ids,qsl,ctx}_{A,B}.bin`、`{m,sizes,offsets}_{full,red}.bin` | **①** | 路径 `m15_layer_loop/ple/data/`；sha256 见 `ple/data/SHA256SUMS`（由 `gen_ple_data.py` 写出）与 `ple/logs/run_all.log §6`。**这些输入只由 host 按 seed 生成**（常数部分直接来自 checkpoint 字节） |
| `B2.emb`（② 的参考） | `table_red.bin`（①）+ **`B_ids_body.bin`（②）** | ①+② | ② 由 **`B1.ids.B`**（T1 逐位；同一组输入、同一组参数的 ① 产物）覆盖。位置：本文件 §1 第 3 行 |
| `B3.kv`（③ 的参考） | `w_key_proj.bin`/`w_value_proj.bin`（①，checkpoint 字节）+ **`B_emb.bin`（②）** | ①+② | ② 由 **`B2.emb`**（T1 逐字节）覆盖。位置：§1 第 5 行 |
| `B4.gated` / `B4.normed`（④ 的参考） | `w_norm_{key,query,conv}.bin`（①）+ `hidden.bin`（①）+ **`B_kv.bin`（②）** | ①+② | ② 由 **`B3.kv`**（**T3**）覆盖。位置：§1 第 8 行 ⇒ **有残差风险，见下** |
| `B5.out` / `B5.state` / `B5.null.out` / `B5.null.state`（⑤ 的参考） | `w_conv1d_sq.bin`、`conv_state.bin`、`state_idx.bin`、`hidden.bin`（全 ①）+ **参考链自身**的 ④ 输出 | ①（参考自身的中间量**不算** ③） | ③ 的定义是"被判量**自己**产出的字节"——参考链的中间量不是设备产物 |
| `B5.state_evolve` | 设备 `B_state_out.bin`（**被判量自身**）vs `conv_state.bin`（①） | 结构性判据 | 它只声明"设备状态相对输入发生了演化"这一**结构性事实**，**不**作数值正确性判据（§1 已如此表述） |
| **③（被判量自身产物当参考输入）** | **没有** | — | 按 §1.3「怎么判」第 1 步把 `m15_ple_check.py` 里所有 `fromfile`/`read_bytes` 逐个归类，结论：**只有 ① 与 ②，没有 ③** |

**③→④ 那条 T3 链接的残差风险（一句话，按要求写明）**：`B3.kv` 是 **T3** 判据，界为
`EPS_MMAD·Σ|terms| + 1.0·ulp(out)`（**M124 起**；改前是 `5.72e-6·Σ|terms| + 1.0·ulp(out)`，见 §2
—— 两个界都是同一形态的推导式，`EPS_MMAD` 盖住旧的 VF 项）；设备 ③ 落在该界内的误差会**被 ④ 的参考继承**，于是 ④ 的判据
对"完全由 kv 误差引起"的那部分偏差**分辨力下降**。当前数据上 `B4.gated`/`B4.normed` 的
`max(d/bound) = 0.000`，说明这份继承在此数据上不显著 —— 但**不能据此推广到别的输入**。
（`B2.emb` 也是 ② 输入，但它是 **T1 逐字节**，不存在这条残差。
M124 的实测旁证：改前/改后的 `B4.gated` 有 15/20480 个元素落在 2 个 bf16 格点内、`B4.normed` 13 个、
而 **`B5.out` 逐字节相同** —— 见 `evidence/ple_wire/m124/M124_plane_diff.log`。）

**为什么这条链的根是干净的**：`B1.ids.A`/`B1.ids.B` 的参考**只吃 ①**（host 生成 + checkpoint 字节），
所以"设备 ①→ 设备 ② → 设备 ③ → 设备 ④"这条依赖链**每一环都被一条判据覆盖**，且链条根部不含设备产物。

## 5. 架构约束的落地（人类裁定逐条，且按事实写）

| 裁定 | 本文件的落地 |
|---|---|
| 不用 `set flag / wait flag` 系列 | 核内只出现 `PipeBarrier<PIPE_ALL>`；**没有** `SetFlag/WaitFlag` |
| 核内 pipeline 用 buffer id | ⚠️ **本文件没有用 `M15H::BufAcquire/BufRelease`**（BufferID 软件流水）：每个 stage 是"搬运→计算→搬运"的短链，用 `PipeBarrier<PIPE_ALL>` 保证正确性。这是"先打通再扣性能"的取舍，**列为未完成项 U4**（M85 r1 P3-1 指出旧 PROLOGUE 声称用过 → 已按事实改写） |
| 核间用 set cross core | `M15H::BarrierAiv<PIPE_MTE2, FLAG_B2/B3/B4>()`（= `CrossCoreSetFlag/WaitFlag` mode-0，set 挂 MTE3 drain 写、wait 挂 MTE2） |
| 不用 AscendC 资源管理函数 | 无 `TPipe/TBuf/TQue/AllocTensor`；UB 地址是本文件 `§UB 布局` 的**编译期常量** |
| buffer id / cross core id / 地址静态分配 | flag id 自定义（`FLAG_B2/B3/B4=12/13/14`）；UB 峰值 `UB_END=52224 B` < 248 KB（`m15_hc_resources.h:159`） |
| 优先抄改现有代码 | `#include "m15_hc_layer.h"` 复用 `NormDonor::{LoadRegForDtype,ComputeRstdNewtonRaphsonReg,SigmoidReg}` / cast traits / `Block1` / `ExtBlock1` / `BarrierAiv`；行 gather 照 `m8_permute.asc:177-193`；conv tap/状态移位照 GDN 的 planar 惯例（`m15_gdn_layer.h:557-563`） |

## 6. 本轮踩到的坑（按"会不会误导后来人"排序）

1. **参考与实现同源 ⇒ 判据是"空过"**（r1 的 P1，D3）：第一版 kernel、参考、SPEC **三处写了同一个
   简化**（跨 chunk 列与 `c` 无关）⇒ `B1.ids.A` 报 PASS 却抓不住任何东西。
   ⇒ 一般化：**参考必须从上游源代码独立推出**，而不是从自己的 SPEC 抄一遍。
   本轮补了变异 bit1，把这条变成"可复现地能咬住"。
2. **`pack_bf16` 取低位 = 全 0**（第 1 版）：`bf16_rne` 已把低 16 位清零，再 `.astype(np.uint16)`
   得到**低 16 位 = 全 0** ⇒ 合成数据整片为 0，②③④⑤ 的 T1/T3 判据**全部"0 vs 0"空过**。
   **是 T4 非空洞判据（`B5.state_evolve`）把它抓出来的**。修法：`>>16`（本仓已有正确实现
   `tools/golden/moe_block_ref.py:455-461` —— 当初"优先抄改现有代码"就不会踩这个坑）。
3. **判据的"可观测性"不是天然的**（D4/P2）：有些分歧（1-ulp 级取整）只在**部分输入**上改变输出
   （≈2%/gate 项）⇒ "改了实现判据却不 FAIL"会被误读成"判据有咬合力"。
   ⇒ 一次性处置：先搜出可观测的数据（`pick_observable_seed.py`），再用变异证明判据 FAIL。
4. **负向对照会污染自己**：`M15_PLE_MUT=1` 时若仍跑 body，body 内部的 ① 复算会把 `B_ids.bin`
   覆盖成**正确**的 ids，使 ① 的对照读数失真。已改为**变异 bit0 只跑 ①**。
   ⇒ 一般化：**变异产物的落盘路径必须与正常路径隔离**（`ple/out` vs `ple/out_neg` / `out_mut_*`）。
5. **判据的单位要写死**（r1 P3-2）：`cmp_bits` 原先报的是**字节**数，README/复审申请里却当成**元素**
   （`1280` vs `160`）。现在统一按元素，字节差单独列。
6. **参考侧的非空洞判据不等于设备侧的非空洞**（r1 P3-5）：`B5.state_evolve` 原先比的是"参考输出 vs 输入"，
   与设备无关。现在比**设备 `state_out` vs 设备 `state_in`**。

7. **标签冒充计数**（r2 的 P3，本队的通病）：判据脚本原先把「非取反行条数」印成「有咬合力 N 条」——
   那是一个**声明**，不是**验证**。现在把两件事分开：
   `m15_ple_check.py` 只报「判定项 N + 对照项 K」；「咬合力被演示」只由
`m15_ple_mutants.py` 的 `RESULT|coverage|` 行给出（并强制：任一判据没被任何变异体咬到 ⇒ 非 0 退出）。
   为此补了 4 个变异位（bit10 值域、bit11 null 状态、bit12 状态不写回、bit13 D2 的取整点本身），
   使覆盖从 9/13 变成 **13/13**。
8. **"参考的规则独立" ≠ "参考的输入链独立"**（r2 的 P2）：第一版把"独立重写、不读设备中间量"
   写得太满 —— 规则确实独立重写了，但 ②③④ 的参考**吃设备上一段的 dump**（属 `docs/17 §1.3` 的 **②**，
   合法但**必须逐条指明覆盖它的判据**）。现在 §4.1 给了完整三分表 + 逐条的覆盖判据 + ③→④
   那条 T3 链接的残差风险说明。
9. **变异位本身可能不安全**：bit10（破坏 ① 的 id）会让 ② 的 gather 用越界 id 去查表 ——
   第一版实现里 body 用的正是"被判的那份 ids"，于是变异跑会越界读 GM。
   现在 body 的 ids 走**独立的 `B_ids_body.bin`**（内部以 `negMask=0` 复算，恒在词表范围内），
   与被判的 `B_ids.bin` 分开；这同时解决了"body 的内部复算覆盖掉变异产物"的老问题。
10. **`mixed ≥ 0` 在本模型下恒成立**（数值实测）：三项乘积的上界分别为
   `5.93e18 / 5.03e18 / 2.01e18`（token id 取 250000），都 `< 2^63 = 9.22e18` ⇒ 它们的 XOR
   符号位恒为 0 ⇒ `mixed` 恒非负 ⇒ **`FloorMod64` 的负余数修正分支在本模型下不可达**。
   后果：`B1.ids.*.range` 的**下界**（`id < 0`）无法用任何"合理的写错方式"演示，
   只能用"漏掉 mod+offset"演示**上界**（bit10）。已记为 U8。

## 7. 未完成 / 未覆盖（逐条列出，不声称完整）

| # | 项 | 卡在哪 |
|---|---|---|
| U1 | 与 `m15_layer_kernel.h` §3b 挂载点的实际接线 | mission 边界：挂载点/host/README 归 M82，须等 M82 合入后由后续 mission 做 |
| U2 | 95.37 GiB 真实 ngram 表 | **M92 已推进**（`ple/REAL_TABLE.md`）：落地形态 + 真实 checkpoint 取证 + 真实规模（整片 shard_0）跑过一次；**未做**跨分片窗口 / 多槽滑窗 / 128 片全量（`REAL_TABLE.md` §9 U-A/U-B/U-H）。本 mission（M85）当时用缩减词表的合成表 |
| U3 | **prefill / spec** 的 short-conv 独立 writeback | `ops/ple.py:489-569`；本 mission 只做 decode（`c≥1` 的多 token chunk **只在 ① 的 id 算术上**被覆盖，②③④⑤ 未跑多 token chunk） |
| U4 | 性能 | **M124 起 ③ 已不是"逐列 `DataCopy` + 逐列 `Reduce` 的 GEMV"**（改成 AIC 上的 cube `Mmad`，N 按 `BASE_N=160` 的 tile 跨 28 条 AIC 轮转）。**仍未做**：段内仍以 `PipeBarrier<PIPE_ALL>` 为主、未引入 BufferID 软件流水；**未做设备侧计时**（共享卡 host 墙钟不构成证据，`docs/17 §9.1`）⇒「改后比改前快多少」**没有读数**，见 `evidence/ple_wire/m124/M124_CUBE_MMAD.md` 的「没做完 / 没取到读数」 |
| U5 | 设备矩阵轴 | 只铺了 T=2（②③④⑤）与 T=10（①）两档 + 14 个变异位；**多请求、多 token chunk（②③④⑤）、多核数（coreDiv）、表行数**的轴未铺开 |
| U6 | **设备侧判据薄弱** | **M92 已补**（见 §0.1 与 `ple/REAL_TABLE.md` §6）：新增 `Hd.row_fail` / `Hd.miss` / `Hd.cores` / `Hd.nonvac`。**M85 原状**：kernel 内建判定量只有 `dev_range_fail`（① 的值域计数器）与 `dev_fail`（null 行计数），②③④⑤ 的判据全部在 host 脚本里 |
| U7 | `has_initial_states` / ETP-DP gather+reduce / dequantize / prefetch | 见 `DIVERGENCES.md §4` G1-G7（本 checkpoint 下为恒等或纯性能机制；decode 路径 `has_init ≡ state_ok`） |
| U8 | `B1.ids.*.range` 的**下界**无法演示 | `mixed ≥ 0` 在本模型参数范围内恒成立（数值见 §6 第 10 条）⇒ `id < 0` 只可能由"不计 mod/offset"之类的破坏产生，而那种破坏同时越界到上界。已用 bit10 演示**上界**；**下界**只能算"结构上不可达"，见 §1 表第 2/4 行的标注 |

## 8. 文件清单

| 路径 | 角色 |
|---|---|
| `m15_layer_loop/ple/PLE_SPEC.md` | **task 1 交付物**：输入/权重/输出/步序四表 + 与上游的分歧记录（D1-D4） |
| `m15_layer_loop/ple/DIVERGENCES.md` | **r1 复审要求**：与上游**两份**实现的逐项对照表 + 「参考与实现同源」清单 + 上游两实现之间的差异 + 未实现项 |
| `m15_layer_loop/m15_ple.asc` | 独立 kernel：`m15_ple_ids_kernel`（①）、`m15_ple_body_kernel`（②③④⑤）+ 自己的 `main()` + 14 个变异位（掩码 `M15_PLE_MUT`） |
| `m15_layer_loop/m15_ple_check.py` | numpy/float64 独立参考 + 13 条分档判据 + 负向对照档 |
| `m15_layer_loop/m15_ple_mutants.py` | 变异矩阵驱动（14 个变异位各跑一次设备 + 要求指定判据 FAIL + **coverage 强制核查：每条基线判据都必须被某个变异体咬到**） |
| `m15_layer_loop/ple/gen_ple_data.py` | 输入/权重生成（真实 checkpoint 权重 + 缩减词表合成表 + null 槽位） |
| `m15_layer_loop/ple/pick_observable_seed.py` | 为"只在部分输入上可观测"的分歧选数据种子（可复现） |
| `m15_layer_loop/ple/logs/{run_all,neg,mutants}.log` | §3 表里每条读数的完整转录 |
| `m15_layer_loop/CMakeLists.txt` | 追加 `m15_ple` 目标（**仅追加**） |
| `m15_layer_loop/ple/.gitignore` | 生成物不入库（`data/`、`out*/`、`build/`） |

## 9. M85 r1 复审意见的处置

| r1 条目 | 处置 |
|---|---|
| **P1** ctx 列（D3） | 三处同改（`PLE_SPEC.md` ①、`m15_ple.asc:173-174`（`col1/col2`）、`m15_ple_check.py:105-107`）；登记为 D3；变异 bit1 证明 `B1.ids.A` 现在能咬住 |
| **P2** dot 和的 bf16 物化（D4） | kernel 补 `Cast<bf16>`（`m15_ple.asc:449-453`）；登记为 D4；选可观测种子 + 变异 bit2 证明 `B4.gated` 能咬住 |
| P3-1 PROLOGUE 夸大 | 按事实改写（§5 表第 2 行） |
| P3-2 单位混淆 | `cmp_bits` 改按元素计；本 README 与复审申请同步 |
| P3-3 分档计数不符 | §1 重新计数（T1 家族 7 / T3 家族 5 / T4 1 = 13） |
| P3-4 容差贴边 | §2 澄清 `max(d/bound)≈1` 的结构含义；D4 修完后 `B4.normed` 裕度回 0.000 |
| P3-5 `B5.state_evolve` 只比参考侧 | 改为设备对设备（§1 第 13 条） |
| P3-6 质数搜索 | 保留（只作与 checkpoint 的对账，判据真值仍取 checkpoint） |
| 「参考与实现同源」清单 | `DIVERGENCES.md §2`（5 条，逐条给出处置） |
| 回扫新发现 | **N1**（null 槽位漏写 out）**已修**；**L1/L2/S1** 布局/接口差异已记并标注为"接线注意"；**G3 prefill** 等未实现项列在 `DIVERGENCES.md §4` 与本文件 U3/U7 |

## 10. M85 r2 复审意见的处置

| r2 条目 | 处置 |
|---|---|
| **P2** `docs/17 §1.3` 的输入三分缺失 + 「不读设备中间量」与代码相反 | 新增 **§4.1**：参考的每一个输入逐条归到 ①/②/③；两个 ② 输入（`B_ids_body.bin`、`B_emb.bin`）与一个 ②（`B_kv.bin`）**逐条指明覆盖它的判据**（← `B1.ids.B` / `B2.emb` / `B3.kv`）；写明 ③→④ 那条 T3 链接的**残差风险**；把「不读设备中间量」那句改成按事实的表述（§4 第 2 条）。同时补进 `DIVERGENCES.md §2` 的「参考与实现同源」清单（新条 S-6）。① 类按 §1.3 要求给「路径 + sha256」：`gen_ple_data.py` 现在写出 `ple/data/SHA256SUMS`，其内容收录进 `ple/logs/run_all.log §6` |
| **P3** 「13 条有咬合力」是标签不是计数 | `m15_ple_check.py` 不再打印该标签（改为「判定项 N + 对照项 K」，并注明咬合力计数在变异脚本里）；`m15_ple_mutants.py` 新增 **coverage 强制核查**：取所有变异体 FAIL 集合的并集，要求覆盖每条基线判据，不覆盖即非 0 退出。补 4 个变异位（bit10 值域 / bit11 null 状态 / bit12 状态不写回 / **bit13 D2 的取整点本身**）⇒ **覆盖 9/13 → 13/13** |
| **P3** bit9 的归属写错 | `DIVERGENCES.md` 的 N1 行改为：`B5.null.out ← bit9`、`B5.null.state ← bit11`（并注明 bit9 不改变 null 态） |
| **P3** D2 的判据归属写错 | `DIVERGENCES.md` 的 D2 行改为 `B5.out ← bit13`（**D2 那个取整点本身**）；bit3/bit4 在两张表里都改标为「⑤ 的 tap 序 / ⑤ 的残差分组」，不再冒充 D2 的判据 |
| **P3** README §3 行的 md5 措辞 | 改为「整个 `--brief` stdout 的 md5 = `5383b4c9…`」，并另给「只取 `RESULT\|` 行的 md5 = `6b880e4e…`」 |
| 依赖闭包 | 引用这两个说法/计数的地方一并改齐：README §0/§1/§3/§6/§7/§10、`DIVERGENCES.md §1 §2`（本表即依赖闭包清单） |
