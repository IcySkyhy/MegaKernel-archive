# M116（Wave B2）—— attention 前端的 prefill 段体 + cache 填（chunk 尺度）

> 权威清单 = `docs/15-prefill-design.md` 的 `## M103 重盘` → `M103-2.2` 的 **B2 行**；本目录是它的
> **独立验证路**（`M103-2.0` P1：Wave B 不碰 `m15_layer_loop/CMakeLists.txt`）。
> 段体头交付在 `m15_layer_loop/`，本目录只 include 它。

## 1. 交付物

| 文件 | 是什么 |
|---|---|
| `../m15_layer_loop/m15_attn_prefill.h` | **段体头**：AIC 的 4 个 `Mmad` GEMM（工作项 = (mTile, nBlock)）+ AIV 的 (row × item) 二维工作项 + cache 填的 chunk 驱动 |
| `../m15_layer_loop/m15_attn_prefill_host.h` | host 侧：**独立 fp64 参考**（GEMM / GemmaRMSNorm+RoPE / 环行 / 主 KV 行 / 池化）+ 发射器 `PfFillArgs` |
| `../m15_layer_loop/m15_attn_cache.h`（**改**） | `AC_MAX_ROWS` 8 → 128、`AC_MAX_GROUPS` 2 → 32；新增 `MainKvFillChunk`（主 KV **逐行**落页）；门控 lane 的落盘通路按 `docs/05 §6.1 ⓔ` 改成「写侧 acquire → 标量写 → 阻塞释放 → 才搬」 |
| `m24_attn_prefill.asc` | 带 `main()` 的**验证工程**：夹具 + 判据 + 变异档 |
| `CMakeLists.txt` / `.gitignore` | 自带构建（`-I ../m15_layer_loop`） |
| `evidence/` | 设备读数（`*.log`，二进制 sha256 见 §5.4） |

## 2. 本段相对 M88 探针改了什么（逐条 = `M103-2.2` 的 B2 行 + 任务书任务 2）

| # | M88（`m15_attn_prolog_probe.h`） | 本 mission |
|---|---|---|
| ① | AIV 按 `bid → head` 一维硬分派（`bid<24` → q 头…，`31` → raw k），**一趟一行**，只有 32 个 AIV 有活 | **`(row × item)` 二维工作项**：`for (i = bid; i < rows*56; i += 2*numBlocks)`；行数与 AIV 数都进网格 |
| ② | AIC 只有 N 维分派；`AttnGemm` 把 `CALC_M = 2` 写死（m=1） | `PfAttnGemm` 带 **m-tile**：工作项 = `(mTile, nBlock)`（一个 chunk = 2 个 m-tile），尾块用 `curM` mask |
| ③ | 入口只有一个 `apPos` 标量 | **per-row 位置**：`PfPosRef::At(绝对行号)`（表 / `posBase + 行号` 二选一，与 Wave A 的 `pfPos`/`pfPosBase` 契约同语义） |
| ④ | — | **矩阵乘法一律 `Mmad`**（人类裁决）：4 个 GEMM 全走 `PfAttnGemm`，无 VF 实现、无选型对比 |
| ⑤ | cache 填是**独立启动的探针档**（`AC_MAX_ROWS = 8`，M98 README §5 自述"真实规模档未跑"） | cache 填进 **chunk 尺度**（128 行 = 32 条压缩行/次），主 KV **逐行**落页；`out` 上的 k/v/rawk 三段不再物化（主 KV 与环就是它们的存储位置），见 §6.1 |

## 3. 判据（口径 + 覆盖粒度 + 变异档）

**真值**全部来自本目录的 fp64 参考，**不读任何设备产物**；`m=4097` **不对官方 QSA 输出**
（`docs/17 §7`：官方 `indexer_budget=2048` 是 token 预算 ⇒ 每行只看得到约一半历史，
稠密路径与它不可能一致）。

**判据的两层分解 + 一处**参考fidelity**的实测修正（M116 设备首跑逼出来的，如实记录）**：

- **两层分解**：① `Pf.y0.*`（T2′，声明行）判**上游 GEMM**（与 fp64 精确积和的差 ≤ `0.5·ulp(got) + ε_acc·Σ|terms|`，
  `ε_acc = 2560·2^-24`）；② 其余判据判**本段**（norm+rope / 池化 / cache 填），其输入口径取**设备 y0 的字节**
  （`ExpectChunk(F, rc, R, devY0, …)`）。y0 是本段自己 AIC 的产物（`docs/17 §1.3` 的 ②上游输出），
  它的正确性由第 ① 条独立承担 ⇒ 两层**分开判、不合并**。
- **参考 fidelity 的修正（M116 实测，第一版参考错了）**：设备的 `AivNormRope` 是**两段式**——
  norm 的结果先 **落 bf16**（`StoreAlign<DIST_PACK_B32>`），下一个 `__VEC_SCOPE__` 的 RoPE 从**那块
  bf16** 读回来旋转（M88 r2 的"两个 scope"结构就是这个顺序）。第一版参考在**未舍入**的 fp64 norm
  结果上旋转 ⇒ 差 ~1.2e-4（半个 bf16 ulp 级，`|Δ|` 与 `ulp(ref)` 比可达 5.6×），K 槽上 60%+ 的行"越界"。
  **修正参考（norm 先落 bf16 再旋转）之后，同一个紧界（`slack = 0`）下 m64 档 23 条判据全过**。
  ⇒ 这**不是**放松判据，是"参考的算术序列要和设备一致"这条基本要求。
- **`PF_T3_SLACK_ULP = 0`**：修正后余量回到 0（最紧的 1 ulp 界）。它保留成常量只是为了让这条口径
  显式可改（改大必须重新推导，见 `m24_attn_prefill.asc` 的 `ChkT3Tag` 注释）。
- **单次发射档里中间 chunk 的设备 y0 已被覆写** ⇒ 那一档的**终态内容**判据（`Ac.mkv.final` /
  `Ac.comp.final` / `Ac.ring.final`）**不适用**（代码里只在逐 chunk 档执行；结构判据 `Ac.*.nospur`
  与末 chunk 的 per-chunk 判据照做）。内容判据由**逐 chunk 档**承担（同一批数据、每轮都有设备 y0）。

档位：

| 档 | 判据 | 用在哪 |
|---|---|---|
| **T1 逐字节** | ① 抄写段**与设备 y0 的内部一致性**（gate 段 / V 槽 / raw k 段——设备内部走位，与 GEMM 无关）；<br>② 位置尾 `3×int64(pos)`（标量算出来的值，与参考逐字节）；<br>③ 未写区仍是毒值（`nospur`）；④ 门控 8 个 lane | `Pf.out.gate.*`、`Pf.in.rawk.*`（raw k 段）、`Pf.in.kv` 的 V 槽、`Ac.mkv` 的 V 槽、`Ac.ring.*`、`Ac.pack.*`、`Ac.flag.*`、`Ac.*.nospur` |
| **T2′ 元素级** | `\|got − exact\| ≤ 0.5·ulp(got) + ε_acc·Σ\|terms\|` | `Pf.y0.r*`（整行 13952 列，声明行） |
| **T3 元素级** | `\|got − ref\| ≤ 14·2^-24·Σ\|terms\| + 6·ulp(ref)`（输入口径 = 设备 y0） | `Pf.out.q.*`、`Pf.out.qidx.*`、`Pf.in.kv`（K 槽）、`Ac.comp.*`、`Ac.mkv`（K 槽）、`Ac.mkv.final` |

**覆盖粒度（如实划界，不含糊）**：

- **全行全覆盖**：主 KV 的 **每一行**（`Ac.mkv` 按 chunk 逐行 + `Ac.mkv.final` 逐行）、环行与位置尾、
  压缩行（每条候选逐条）、pooled、packed、门控 lane、in-plane 的 raw k / K / V 槽。
- **声明行**（首 chunk 与尾 chunk 的 `r ∈ {0, rows-1}`，chunk ≥ 128 行时另加 `{63, 64}` 两条 m-tile 边界）：
  `Pf.y0`（整行 T2′）+ `Pf.out.q/qidx`（T3）—— 因为它们要付"整行 fp64 GEMM"的代价。
  ⇒ **未覆盖的部分写窄**：q/gate/idxq 三个**段**的"非声明行"只有间接覆盖（gate 有内部一致性判据
  `Pf.out.gate.*` 覆盖全部声明行；非声明行的 q/idxq 没有元素级判据）。

**变异档（"把被测对象弄坏必变红"，`docs/17 §4` 的第五变体纪律）**：

| 档 | 打哪一条判据 |
|---|---|
| `AP_MODE_PLAIN_NORM` / `AP_MODE_ROPE_SIGN` / `AP_MODE_NO_ROPE` | `Pf.out.q.*`（R3/R9 三个方向） |
| `PF_MUT_NO_GEMM` | `Pf.y0.*`（AIC 不跑 GEMM） |
| `PF_MUT_NO_KV` | `Ac.mkv.*`（k/v 不落 cache 输入面） |
| `PF_MUT_NO_RAWK` | `Pf.in.rawk.*` / `Ac.ring.*` |
| `PF_MUT_NO_QIDX` | `Pf.out.qidx.*` |
| `AC_MODE_NO_RING_STORE` | `Ac.ring.*` |
| `AC_MODE_NO_COMP_STORE` | `Ac.comp.*` |
| `AC_MODE_SUM_NOT_MEAN` | `Ac.pool.*` / `Ac.comp.*` |
| `AC_MODE_OFFBYONE` | 候选/门控（`Ac.comp.*` / `Ac.flag.*`） |
| `AC_MODE_NO_RING_READ`（在 `rag65` 上跑，chunk 起点 mod 4 走遍 0/1/2/3） | `Ac.comp.*`（跨 chunk 成员） |

参考**恒按契约语义**算（`apMode = 0`、池化除 4、RoPE 用组首）⇒ 变异档必然与参考不同 ⇒ 判据
只要"没咬住"就是本档 FAIL。

**非空洞 guard（`Pf.guard.*` / `Ac.guard.*`，只跑在契约档）**：本档的 T3 判据的**输入**取自设备 y0，
所以必须有一道闸防"设备 y0 全毒 ⇒ 两边一起毒 ⇒ 判据空过"（复审 P2-1）。已实现的 guard（`CheckChunk`
里的 `runGuards` 块，`!cs.expectRed` 时为真）：
① `Pf.guard.y0live.<c>`：本 chunk **每一行**的 y0 都必须是"活"的（`0xCDCD` 的 bf16 值 = −1.6015625×2²⁸
≈ −4.29916e8；用幅值上界 1e6 判，抽前 64 个元素）—— **逐行**断言（"至少一行"太松，128 行的档只 1 行活就过）；
**绿方向**在设备上实测（§4-A：`poisonRows=0/64`）、**红方向**也在设备上实测（§4-C/D 的 `MUT_guard_nogemm`：
`poisonRows=64/64`），且`guardMutant` 档的 PASS **断言** `guardFails > 0`（§4-D，不是快照）；
② `Ac.guard.poollive.<c>`：池化行的参考不得是毒值面（4 个毒值成员的在 fp64 下的均值舍回 bf16 仍是 `0xCDCD`）。
（r1 曾有第三条 `Pf.guard.reflive`，r2 复审指出它**恒真**——K 槽参考是 norm+rope 的输出、不可能整段等于 `0xCD`——已删。）
guard 失败计入 `guards/guardFails` 并让该档 FAIL（与"判定项"分账，`docs/17 §1.1` 的口径）。

## 4. 设备读数

**二进制谱系（逐行标出"哪条读数属于谁"；**不写数词**，以表格行为准）**

| 二进制 | sha256 | 读数状态 |
|---|---|---|
| **读数所属二进制**（= `064410d` 的源码） | `c0839e66b2ab7808fcf3c4b25316d0e39b2c378c163f334bf7aa8cb9f82a58ae` | 下面表里的**设备读数全部属于它**（复审在 tip 上重编并逐字节对齐过） |
| 更早一版 m24 二进制 | `5839b401a4e855dbe4f716f13bab270b5680501c48ed9f2f5fb3eaf52e271ee6` | **有 2 次设备读数**：§4-A（guard 契约档）与 §4-B（Trap 档） |
| 上一版 m24 二进制 | `f046f2bb17e4eaa7a6c484bd254bacbb527eb22e420128264532bd63b316151f` | **有 1 次设备读数**：§4-C（guard 红方向负向对照 `MUT_guard_nogemm`） |
| **当前 tip 的 m24 二进制** | `9fe07a3b303b3518a12baadb853e58aac849222798461d036e7f645a733d73bc` | **有 1 次设备读数**：§4-D（同上一档、在**硬化断言**之后复跑） |
| **融合 TU 二进制** | `aa01a4fb543214e1528d5ebf2b9177e729131455442ffd19393883f3af720317` | M98 的零回归读数（215/191/0, rc=0）**属于它**；本 tip 与它是同一个二进制（段体在融合 TU 里没有被引用，Wave A 的挂载点那一支仍是 `return;`） |

`c0839e66` → `5839b401` 之间源码改了这些（都不动 `nChunks = 1` 的已验证路径）：
① 段体 `AttnPrefillPhaseA` 对 `nChunks > 1` 加 `Trap`（只在该参数下触发）；
② harness 加真 `Pf.guard.*`（**只加 host 侧前置断言**）与"已知缺陷档（期望响亮失败）"；
③ `m15_attn_cache.h` 的 5 处过期字面量注释改派生式（纯注释；融合 TU 二进制不变即其独立读数）；
④ r2 复审后：`Pf.guard.y0live` 提成**逐行**断言、删掉恒真的 `Pf.guard.reflive`、`big4097_seq_refused`
移到 case 列表**末尾**、`expectLaunchFail` 分支打印 `aclGetRecentErrMsg()`；
⑤ 本轮（复审再授权去跑设备）：guard 的**绿也打印**一行（带 `poisonRows` 之类的 detail，让日志能看出
这道闸被执行了）+ 新增 **Trap 之后的可用性探针**（同一进程/stream 里再发一个 2 行 1 chunk 的契约档，
`st.Guard("posttrap.context.usable", …)`）。

**r3/r4 授权读数（最终取到；A/B/C/D 四条）**：先按"`flock -n` 探锁"试了 **18 次非阻塞探测、全部返回忙**
（`-n` 是插队式轮询，拥挤时排不上）；随后按塔的裁决改用**一次有界等待** ——
`flock -w 180 /tmp/npu0.lock <一条短命令>`（锁内先 `npu-smi` + `df -h`、命令 `timeout 110`；
**A/B/C/D 四条各一次独立取锁**）。四次都在窗口内进锁并跑完；读数逐字如下 —— **A/B 属于 `5839b401…`、
C 属于 `f046f2bb…`、D 属于当前 tip `9fe07a3b…`**（与上面谱系表逐行一致）：

**A) `M24_ONLY=m64`（契约档，验逐行 guard）** —— `evidence/m24_guardrun_m64.log`，rc=0：
```
[M24][GUARD-OK] Pf.guard.y0live.c0 poisonRows=0/64(阈值 1e6)
[M24][GUARD-OK] Ac.guard.poollive.c0 池化行参考 != 毒值面
[M24]   m64：checks=23 fails=0 guards=2(guardFails=0)
```
⇒ 逐行 `Pf.guard.y0live` 在真机契约档上**被执行且绿**（日志里有两行 `[M24][GUARD-OK]` 带 detail，
不是只靠汇总的 `guards=2`）；m64 档全绿。

**B) `M24_ONLY=big4097_seq_refused`（Trap 那档）** —— `evidence/m24_trap_probe.log`，rc=0：
```
[M24]   big4097_seq_refused 已知缺陷档：launch 非正常结束 err=507015 msg=EZ9999: Inner Error!
…（设备错误报告，CANN 打印；**28 条 `aicore`（AIC）+ 56 条 `aivec`（AIV）= 84 条**，覆盖 28+56 个物理核，全部同形）：
  error code = 286 … The extend info: errcode:(286) errorStr: The trap instruction reports an error. subErrType: 0x4.
[M24]   big4097_seq_refused：Trap 之后的可用性探针（2 行 1 chunk 契约档）err=0 msg=(空) ⇒ context 仍可用
[M24][GUARD-OK] posttrap.context.usable Trap 后同一 stream 仍能跑契约档
[M24]   big4097_seq_refused：checks=0 fails=0 guards=1(guardFails=0)
[M24] ===== ALL PASS =====
```
⇒ ① 那道硬拦在**设备侧**被确认为 **trap 指令**异常（`errcode 286` / `errorStr: The trap instruction
reports an error.`），不只是 host 侧看到的通用 aicore 异常码 `507015`；② **28 AIC + 56 AIV = 全部 84 个
物理核都撞上了** `AttnPrefillPhaseA` 的第一条语句（`Trap()` 在核型分支之前）⇒ 与"硬拦拦住了这次启动"
自洽；③ **这一次读数里 context 仍可用**（Trap 之后同一 stream 再发契约档 `err=0`）⇒ "连坐"没有出现；
④ **写窄**：这是**一次**读数，不构成"任何 aicore 异常之后 context 都可用"的普遍结论。

**C) `M24_ONLY=MUT_guard_nogemm`（guard 的**红方向**负向对照）** —— `evidence/m24_guard_red_mut.log`，rc=0：
```
[M24][GUARD-FAIL] Pf.guard.y0live.c0 poisonRows=64/64(阈值 1e6)
[M24][GUARD-FAIL] Ac.guard.poollive.c0 池化行参考 != 毒值面
[M24]   MUT_guard_nogemm：checks=23 fails=2 guards=2(guardFails=2)
[M24]   MUT_guard_nogemm 变异档：PASS（判据咬住了）（变红判据数=2）
```
⇒ `Pf.guard.y0live` 的**红方向在真机上被实测到**（不是只有构造性论证）：这一档跳过 AIC 的 GEMM ⇒
设备 y0 停在本档开始时铺的毒值上 ⇒ `poisonRows=64/64` 红、`Ac.guard.poollive` 亦红。
（这一档是复审 r3 的新增：`Case::guardMutant = true` 把 `runGuards` 也打开 —— 否则 `MUT_nogemm`
属 `expectRed`、`runGuards=false`，这道闸只会被看到"绿"。）

**D) `M24_ONLY=MUT_guard_nogemm`（B-4 硬化断言后的复跑）** —— `evidence/m24_guard_red_hardened.log`，rc=0：
```
[M24][GUARD-FAIL] Pf.guard.y0live.c0 poisonRows=64/64(阈值 1e6)
[M24][GUARD-FAIL] Ac.guard.poollive.c0 池化行参考 != 毒值面
[M24]   MUT_guard_nogemm：checks=23 fails=2 guards=2(guardFails=2)
[M24]   MUT_guard_nogemm 变异档：PASS（判据咬住了）（变红判据数=2，guardFails=2，guardMutant 档额外要求 guardFails>0）
```
⇒ 该档的 PASS **包含** `guardFails > 0` 这个条件（复审 r3 的口径：红方向不仅要记录、还要断言）：
`expectRed` 档若 `Case::guardMutant`，则 `red = red && (st.guardFails != 0u)` —— 将来 guard 被静默改坏时
这一档会变 `FAIL（判据没咬住）`，不再只是一张"当时红过"的快照。

**四次读数的限度（集中一行）**：每个档**各跑一次**；除 A/B/C/D 之外的档**没有**在 `9fe07a3b…` / `f046f2bb…` 这两个
二进制上重跑（表里其余读数属 `c0839e66…`）；**没有**重复运行读数；**没有** SLOG / 性能 / msprof 读数
（本 mission 不做性能）。
**四条读数各自的日志见 §4-A（`evidence/m24_guardrun_m64.log`）/ §4-B（`evidence/m24_trap_probe.log`）/
§4-C（`evidence/m24_guard_red_mut.log`）/ §4-D（`evidence/m24_guard_red_hardened.log`）；A/B 两条命令的
经过另记在 `evidence/m24_r2_readings_pending.log`**（那是"尚未取到时"的经过记录，含先前 18 次 `-n`
探测全忙的历史）。

**m24 自己的 rc**：`run_evidence.sh` 从 r1 起把 `m24 rc=` 写进日志（字段名固定）；`c0839e66` 那次
**未归档**它（当时命令末尾的 `grep` 只截了 case 行），按 `big4097_seq` 的 11 条红**当时必然是 rc=1**；
改档之后（`big4097_seq_refused` 期望 Trap）预期为 rc=0 —— **未取到复跑读数**。

命令与输出：`evidence/run_evidence.sh` / `evidence/m24_readings_r1.log`（原始 `m24_all.log` 被后续一次
已回退的实验截断覆盖，副本在 `evidence/m24_aborted_1set_pairing.log`；参考 fidelity 的诊断为
`evidence/m24_diag_m64_reffix.log`）。入口 `flock` 内先 `df -h` + `npu-smi` 复查：那次入口时 NPU 0
无进程、util 0%，`/` 余 308 G。

| 档（真 shape） | checks | fails | 备注 |
|---|---|---|---|
| `big4097_step`（4097 行，128×33，**逐 chunk 发射**） | 327 | **3** | 3 条 = **同一处**：K 槽 1 个元素超出 1-ulp 界（1.33×，2,097,152 个元素里 1 个）⇒ 被 `Pf.in.kv` / `Ac.mkv` / `Ac.mkv.final` 三条判据同时看见。按"参考 fp64 / 设备 fp32 在格点**平局**上的舍入方向差异"登记为**已接受偏差**（`docs/17 §7.1` 的 FTZ 同款口径）；`PF_T3_SLACK_ULP` 置 1 即可清掉（该常量与其推导留在源码里，本档刻意保持 0） |
| `big4097_seq_refused`（同序列，**单次发射 33 个 chunk**：去打段体对 `nChunks > 1` 的 `Trap` 硬拦） | — | — | **已知缺陷档（期望响亮失败，排在 case 列表末尾**：aicore exception 有污染 stream/context 的风险，按仓内先例（M98 的 `Ac.dcpad.required`）把期望 Trap 的档放末位）。段体加硬拦之后该档预期 launch 非正常结束（不产出判据读数）；分支现在把 `err` 与 `aclGetRecentErrMsg()` 一起记进日志（两者合起来才是"确实是那道 Trap"的指纹）。**加拦之前的同档读数**：20 checks / 11 fails（y0 整行毒值、门控 8 lane 全毒、packed 却对）—— 见下 |
| `m1ctx4097`（1 行 @ pos 4096，ctx=4097） | 23 | 0 | m=1 档 ✓ |
| `rag65_step`（393 行，chunk 65 行 × 7） | 93 | 0 | chunk 起点 mod 4 走遍 0/1/2/3 ⇒ **真跑到"跨 chunk 成员从环里读"** ✓ |
| `m128`（1 chunk = 2 m-tile） | 31 | 0 | m-tile 边界 ✓ |
| `m64`（1 chunk = 1 m-tile） | 23 | 0 | — |
| `m2`（最小档） | 23 | 0 | `calcM = max(curM,2)` 的 3510 契约 ✓ |
| 12 个变异档 | 23 | 2~14（各自） | **8/12 有归档读数**（`m24_readings_r1.log` 49 行，在 `MUT_ac_nocompstore` 的**档头之后**截断 ⇒ 缺 `ac_nocompstore` / `ac_sumnotmean` / `ac_offbyone` / `ac_noringread` 四档的**归档读数**，其余 4 档读数**未取到**）：`AP_MODE_PLAIN_NORM` 7、`AP_MODE_ROPE_SIGN` 5、`AP_MODE_NO_ROPE` 5、`PF_MUT_NO_GEMM` 14、`PF_MUT_NO_KV` 3、`PF_MUT_NO_RAWK` 6、`PF_MUT_NO_QIDX` 2、`AC_MODE_NO_RING_STORE` 3 —— 每档"至少一条判据变红"这一条在这 8 档上有归档支撑；另 4 档（README 早先记的红数 4/2/4/43）**只出现在被截断的 stdout 里，本仓无归档**。⚠ 读"红数"时还要扣掉 1 条：`Ac.flag` 的期望把 lane4（mode）**硬钉 0**，而 cache 变异档本身就传 `acMode ≠ 0` ⇒ 每个 cache 变异档至少有 1 条红是 **harness 自己造的**（与被测段无关） |

**M98 探针的零回归档**（`m15_attn_cache.h` 是 M116 改过的既有文件）：
`./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt kv` →
**ALL PASS（checks=215, guards=191, fails=0）rc=0** ⇒ M116 对该文件的改动（`AC_MAX_ROWS` 8→128、
`AC_MAX_GROUPS` 2→32、门控 lane 的标量落盘通路改成写侧 acquire + 阻塞释放、主 KV 落页抽成
`MainKvFillChunk(..., rows = 1, ...)`）**没有动 M98 的任何读数**。

**两处失败/未修的诊断（如实写窄）**：

1. **`big4097_seq`（单次 launch 跑 33 个 chunk）不产出**：终态 D2H 显示
   `Pf.y0.r0.c32` 的整行 13952 列都是**毒值**（`bf16(-4.29916e8)` = `0xCDCD` 的位型）⇒ **AIC 的 GEMM 产物
   一次都没落到 y0**；门控 8 个 lane 也全是毒值（`-1`，cache 体没跑到第 7 段）；而 packed 的 seed 行
   **是对的**（说明 AIV 段与 cache 体至少跑到过第 5 段）。⇒ 形态是"多 chunk 单次发射时，AIC 臂与 AIV 臂
   在若干轮之后失步"，不是数据通路错。**同一个段体的逐 chunk 形态（`big4097_step` / `m128` / `rag65`）
   全绿** ⇒ **本 mission 只把 `nChunks = 1`（每个 chunk 一次启动）作为已验证形态交付**（见 §5.1/§6.1）。
2. **试过并回退的一个修法（**未取得读数**）**：把 mode-2 反向信号从"2 set 配 1 wait"改成"只在每个
   AICore 的 AIV0 侧 set 一次（1:1 配平）"⇒ **该次尝试没有取得读数**：归档 `evidence/m24_aborted_1set_pairing.log`
   里那一行是 `timeout 900 …` + `m24 rc=137`（**137 = SIGKILL / Killed**），且该次日志 10:23:54→10:31:49
   **不足 900 s** ⇒ **不是内层 timeout 收回**（timeout 正常收回是 rc=124）、日志里也没有 NPU util 读数。
   按塔的口径（"外部杀进程 ≠ 复现死锁"；rc=137/Killed 的不返回要先读设备/系统错误报告）⇒ **这条只记
   "未取得读数（进程被 Killed）"**，**不作**"1:1 会挂"的结论，也不作根因证据。**已回退**到有归档读数的
   2:1 形态（§5.5）。⇒ 由此，"多 chunk 单次发射失步"的**根因**目前**没有**归档的设备错误报告支撑，
   §6 第 1 条给的建议（4 槽 id 轮转）只是**待验证的假设**，下游不得当结论引用。

   **另一条需要解释的读数差异（如实标"未解释"）**：`evidence/m24_m2_first.log`（10:00 那次）里
   `Pf.out.gate.r0.c0` / `Pf.out.gate.r1.c0` 各差 1/12288。这条判据比的是**设备 `out` vs 设备 `y0`**
   （设备内部一致性，与 host 参考无关），而 10:12 那次跑（`PF_T3_SLACK_ULP=0`、同一段体）它全绿。
   两次之间**只改过 host 侧参考**（`m15_attn_prefill_host.h` 的 norm 先落 bf16）⇒ host 侧改动**不可能**
   翻转这条判据。⇒ 记为「**设备行为在两次读数之间变过，未解释**」（该文件对应的是 10:00 的二进制，其 sha 未归档。⚠ **当时 `Pf.y0` 与 `Ac.flag` 判据是"绿"的**
   —— 10:00 那份日志的失败清单按 `Stats::Note` 的插入序是 `Pf.in.kv.c0` / `Ac.mkv.c0` /
   `Pf.out.gate.r0.c0` / `Pf.out.q.r1.c0` / `Pf.out.gate.r1.c0` / `Pf.out.qidx.r1.c0` / `Ac.mkv.final`
   共 7 条，不含 `Pf.y0.*` 与 `Ac.flag.*`；而 `Pf.y0` 绿说明当时的 GEMM 产物是好的、`Ac.flag` 绿
   说明当时的寻址/门控也对 ⇒ **不能用"那次运行的输入面处于未定态"来解释这份 gate 差异**）。⇒ 结论仍是「未解释」，但理由换成上面这条事实。

## 5. 融合清单（`M103-2.7` 第 2 条的固定小标题；给 Wave C）

### 5.1 挂载点

| 项 | 内容 |
|---|---|
| 接进哪个相位/入口 | `m15_layer_kernel.h` §3d 的 `M15L_PrefillPhaseA<KIND_ATTN>` 里 `if (A.pfWired != 0u)` 那一支。⚠ **一次调用只跑一个 chunk**：Wave C 要按 chunk 循环启动（`n = ceil(A.m / 128)` 次），或在段里改好"多 chunk 单次发射"那条（§6 第 1 条） |
| 消费的 GM 平面 | `A.xLayer`（[m,2560] bf16）、`A.apW`（attention 8 role 权重平面，`M15AP` §3）、`A.apCs`（cos/sin 表）、`A.pfPos`/`pfPosBase`（位置） |
| 生产的 GM 平面 | `out` 平面（q / gate / idx q 三段）、主 KV（`pfKv` + `KvLayerOffset(pfLayerK)`）、compressed（`pfComp`）、raw ring（`pfRing`）、packed 的 seed 行原样抄写（`pfPack`） |
| **不生产** | k/v/raw k 的 `out` 段（它们的存储位置就是主 KV 与环，见 §6.1）；packed 的**内容**（那是 B3 的 indexer）；attention 核心与 `o_proj`（B3）；`subOut` 的生产者（N1，塔裁归 Wave C） |

### 5.2 `LayerArgs` 需要的字段

Wave A 已给的 13 项里，本段用 **9** 项：`pfKv` / `pfComp` / `pfRing` / `pfPack`（四个 cache 基址，
attention 层要各自加 `KvLayerOffset(pfLayerK)` / `CompLayerOffset(pfLayerK)` / `RingLayerOffset(pfLayerK)`）、
`pfPos` / `pfPosBase`（per-row 位置）、`pfLayerK`（层序号 k）、`pfWired`（开关）、`pfMutant`（变异/负向对照掩码）。
**未用**：`pfBlockTable`（**必须为 `nullptr`**：本段只支持恒等表/单请求）、`pfCounts` / `pfExpertOffsets`（MoE 的）、
`pfStageMask`（本段不分段截断；如需 bring-up 定位，可用它把 chunk 数截断，需 Wave C 加两行）。

**本段还需要 5 个 scratch 指针（Wave A 的 13 项里没有）** —— 二选一：

- **（建议，§5.8 的补丁按这一支写）在 `LayerArgs` 追加 5 个字段** `pfY0` / `pfOut` / `pfIn` /
  `pfPooled` / `pfFlag`，尺寸按**真实 shape**：

  | 字段 | 尺寸（字节） | 出处 |
  |---|---|---|
  | `pfY0` | **3,571,712** = `PF_ROWS × PF_Y0_STRIDE × 2` = 128 × 13952 × 2 | 本段 §1 的 `PF_Y0_STRIDE = M15AP::Y0_N` |
  | `pfOut` | **3,571,712** = `PF_ROWS × PF_OUT_STRIDE × 2`（同值，因为 `Y0_N == AP_OUT_N`） | 同上 |
  | `pfIn` | **299,264** = `M15AC::AC_IN_BYTES` = 128×144×2 + 256 + 128×2048 | `m15_attn_cache.h` §1a 的派生式 |
  | `pfPooled` | **8,192** = `M15AC::AC_POOLED_BYTES` | 同上（诊断面） |
  | `pfFlag` | **32** = `M15AC::AC_FLAG_LANES × 4` | 同上（门控 lane） |

  合计 **7,450,912 B ≈ 7.11 MiB**（= 上面 5 项之和：3,571,712×2 + 299,264 + 8,192 + 32）。⇒ 补丁（§5.8）引用的是这 5 个字段（Wave C 仍需先按本表把它们加进 `LayerArgs`）。
- 或者由 Wave C 从 prefill 的 `ws` 平面里切出同样大小的 5 块（**偏移由 Wave C 定**，本段只吃指针）——
  此时把 §5.8 的 `pfa.pooled/pfa.flag` 两行换成对应的切片式子即可（其余一字不改）。

**一个字段承载两种档**（免得再加字段）：`pfMutant` 的位约定 =
`[0,12)` 前端掩码（低 8 位 = `M15AP::AP_MODE_*`）、`[12,16)` cache 档（`M15AC::AC_MODE_*`）；
`M15PF::PfUnpackMut/PfUnpackAcMode` 是它的可执行形式（`m15_attn_prefill.h` §4）。

### 5.3 资源窗与峰值（**填 `m15_layer_resources.h` §3c 的 `PF_*` 槽用**）

| 槽 | 值 | 出处 |
|---|---|---|
| `PF_UB_BYTES_ATTN` | **117,312 B** | `M15PF::PF_UB_PEAK_BYTES` = `AC_UB_END`；相位 A 的三段窗（M15AP `[0,14336)` / 前端 `[14336,16768)` / cache `[32768,117312)`）**不在同一时刻**（AIV 项 → cache 填）⇒ 峰值取段窗顶的最大值 |
| `PF_L1_BYTES_ATTN` | **294,912 B** | `M15PF::PF_L1_PEAK_BYTES` = `M15AP::L1_B1 + AP_L1_B_ELEMS*2`（4 个 GEMM 的 A/B ping-pong） |
| `PF_L0C_BYTES_ATTN` | **32,768 B** | `M15PF::PF_L0C_PEAK_BYTES` = `M15AP::L0C_BYTES`（单 m-tile 的 64×128 fp32 tile） |

三个量都有各自的 `static_assert`（≤ 248 KB / 512 KB / 256 KB）。

### 5.4 BufferID 清单（核内；**编译期静态分配**）

| id | 核型 | 用途 | 备注 |
|---|---|---|---|
| 0 / 1 / 2 | AIV | `M15AP` 的 `AivCopy`（两段式）/ `AivNormRope`（三段式：`AP_BUF_IN`+`AP_BUF_OUT`） | M88 既有 |
| **14** | AIV | **前端新增** `PF_BUF_ROW`：环行组装（MTE2 读 raw k 256 B → **S 写位置尾 24 B** → MTE3 落 280 B） | 每 chunk 3 次 acquire/3 次 release，**成对** |
| **15** | AIV | **前端新增** `PF_BUF_KV`：主 KV 一行的暂存（MTE2 → MTE3，`MainKvFillChunk` 内逐行） | — |
| 10 / 11 / 12 / **13** | AIV | cache 段的 `AC_BUF_IN` / `AC_BUF_OUT` / `AC_BUF_PACK` / **`AC_BUF_FLAG`（M116 新增）** | 13 专给门控 lane 的标量落盘（写侧 acquire → 写 → 阻塞释放 → 才搬），避免 S 的阻塞释放把整条 MTE2 流水也等上 |
| 0..6 | AIC | `M15G::BUF_AIC_A0/A1/B0/B1/L00/L01/L0C`（4 个 GEMM 的 L1/L0/L0C ping-pong） | 与 GDN 段同号：两者在**不同层**（KIND_ATTN / KIND_GDN）从不并发 |

AIV 峰值 id = 15（≤ 27 ✓）、AIC 峰值 id = 6 ✓。

### 5.5 flagId 清单（核间；`(核型, mode, id, pipe)`）

| (核型, mode, id) | 符号 | set / wait 的 pipe | 语义 |
|---|---|---|---|
| `(AIC, 0, 1)` | `M15PF::PF_AIC_M0_BAR` = `AP_AIC_M0_OUT` | set `PIPE_FIX` / wait `PIPE_S` | 4 个 GEMM 的 FIXP 写 GM 排空 + 全体 AIC 对齐（**M97 已登记**） |
| `(both, 2, 5)` | `M15PF::PF_A2V_TILE` = `AP_A2V_GEMM` | AIC set `PIPE_MTE2` / AIV wait `PIPE_MTE2` | AIC → 配对 2 AIV「本 chunk 的 y0 全量可见」（**M88 已用**） |
| `(both, 2, 4)` | `M15PF::PF_V2A_DONE` = `AP_V2A_READY` | AIV set `PIPE_MTE3` / AIC wait `PIPE_S` | **本 mission 起启用**：AIV → 配对 AIC「y0 已消费，可覆写」。**没有它就有真竞态**（AIC 的下一个 chunk GEMM 会覆写在 AIV 还在读的 y0 行上）。AIV 侧**全体各 set 一次**（`m15_attn_prefill.h` 的 `AttnPrefillChunkAiv`：无 `bid` 判定）⇒ **2 set 配 1 wait**（`docs/05 §2` 明确把这一形态列为合法配对之一，与 `m10_attn_decode` 的 `CC_AIVDONE` 同形）。⚠ 与之相邻的另一个形态（只在每个 AICore 的 **AIV0** 侧 set 一次 = 1:1 配平）本 mission **试过但未取得读数**（该次运行被 Killed，`rc=137`；见 §4 第 2 条）⇒ **交付的是 2:1 这一形态**（它在 `big4097_step` 的 33 次逐 chunk 启动上有归档读数） |
| `(AIV, 0, 10)` | `M15PF::PF_AIV_BAR_ITEMS` = `M15G::FLAG_AIV_SEG_S1` | set `PIPE_MTE3` / wait `PIPE_MTE2` | B1：`in` 输入面（环行 + K/V 行）全体就位 |
| `(AIV, 0, 11)` | `M15PF::PF_AIV_BAR_CACHE` = `M15G::FLAG_AIV_SEG2` | set `PIPE_MTE3` / wait `PIPE_MTE2` | B2：block 0 读完 `in` 之前，别的 AIV 不得开下一个 chunk 的项（`in` 是每 chunk 覆写的 scratch） |

**相邻性**（`m15_layer_resources.h` §4e 的口径）：

- **AIC mode-0**：本段只有 id 1；KIND_ATTN 层的下一个 AIC mode-0 逻辑点是 core 的 `CC_BAR(8)` ⇒ `1 ≠ 8` ✓（Wave A 已把这条订正成 `static_assert(AP_AIC_M0_OUT != ATTN_CORE_M0_AIC_BAR)`）。
- **mode-2**：本段用 4 与 5，都在 Wave A §4d 的 `{4..7}` 窗内，且 `4 ≠ 5` ✓（本文件有 `static_assert`）。
- **AIV mode-0**：本段用 10 与 11，`10 ≠ 11` ✓（本文件有 `static_assert`）。
- ⚠ **同步点的循环复用（报塔项，含未证部分）**：本段**交付的形态是"每个 chunk 一次启动"** ⇒
  每个 launch 里同一 `(核型, mode, id)` **只 set/wait 一次**，不存在"跨 chunk 累计"的问题 ✓
  （`big4097_step` 的 33 次启动、`rag65_step` 的 7 次启动都是这个形态的实跑证据）。
  在**多 chunk 单次发射**形态下，每次迭代会给 `(both,2,4)` 留下 **2 set 配 1 wait**（净 +1）——
  这与该项目观测到的失步**可能**有关，但**没有归档的设备错误报告**（§4 第 2 条）⇒ **只是待验证的假设**，
  下游不得当结论引用。若将来要打开多 chunk 形态，建议同时做两件事：① 把 chunk 循环里的 mode-0 barrier
  与 mode-2 信号改成 **4 槽 id 轮转**（照 MoE 段 `FLAG_*_RING[4]`）；② 先读设备/系统错误报告再谈根因
  （Wave A §4e 的相邻性判据是按"逻辑点序列"写的，**表达不了循环** —— 那一条仍归 Wave C 收口）。
- **Wave C 要补登记**：把 `(both,2,4)`、`(AIV,0,10)`、`(AIV,0,11)` 三条加进
  `FLAG_SEQ_PREFILL_ATTN[]`（三个都在 decode 的 `FLAG_SEQ[]` 里已登记过 ⇒ 复用见证成立），
  并把 `PfAttnReuseOk()` 的 mode-2 窗检查覆盖到 id 4。

### 5.6 相位边界

本段**自带**两条 AIV mode-0 barrier（B1/B2，见 §5.5）与 AIC 的每 tile mode-0 barrier；
**不需要**额外的跨相位边界（相位 A 自包含）。若 Wave C 把本段与 B3 的 core 串在同一个相位里，
B3 用 mode-4 之前必须先 drain mode-0/mode-2（Wave A §4d 末尾的使用契约）。

### 5.7 `m` / `pos` 语义

- **每次调用只处理一个 chunk**（`nChunks = 1`）：`rowStart` = **本 chunk 的绝对首行号**，
  `seqRows` = 本次要处理的行数（= 本 chunk 的行数，≤ 128 = `M15AC::AC_MAX_ROWS`），`chunkRows` = 同值；
  4097 行的序列按 `off = 0, 128, …, 4096` 启动 33 次（尾 chunk 1 行 ⇒ 走 `calcM = max(curM,2)` 的契约）。
- `pos(row)`：`posTbl != nullptr` ⇒ 按**绝对行号**取表；否则 `posBase + 绝对行号`；两者都给以表为准。
- **契约（写窄）**：① 每个 chunk 内的位置**必须连续**（`pos(r+1) = pos(r) + 1`）——cache 的门控/组号由
  `(chunkStart, chunkRows)` 推出来，不逐行读表；设备在给表时**逐行复核**，不连续就**跳过 cache 填**并把
  `AC_FLAG_FE_CONTRACT`（门控 lane 7）置 1（响亮失败，不静默错）。② **只支持恒等 `block_table`（单请求）**：
  所有 cache 地址由 `pos` 经 `M15KV_KV_*` 算出（Wave A 的字段契约 + `m15_attn_kv.h` 的 D2）。
- `x` 平面尾部**必须留 1 行可读余量**（`calcM = max(curM, 2)` 的 3510 Nd2Nz 契约：尾 chunk 只有 1 行时会多读 1 行）。

### 5.8 挂载点补丁文本（**可直接用的形状**；Wave C 照抄）

`m15_layer_kernel.h` §3d 的 `M15L_PrefillPhaseA<KIND>` 里，把

```cpp
    if (A.pfWired != 0u) {
        // ... 现有注释 ...
        return;                       // 段体未落地 ⇒ 什么都不写（响亮失败）
    }
```

换成（**只动这一支**，其余一字不改）：

```cpp
    if (A.pfWired != 0u) {
        if constexpr (KIND == KIND_ATTN) {
            // ---- M116（Wave B2）：attention 前端的 prefill 段体 ----
            // ⚠ **逐 chunk 调用**：段体一次只跑**一个** chunk（`pfa.nChunks = 1`）；
            //    `nChunks > 1` 在段体里是 `Trap` 硬拦（`AttnPrefillPhaseA` 的防呆，理由见 README §6.1）
            //    ⇒ 下面这层 `off` 循环就是"按 chunk 启动 N 次"的形态（N = ceil(A.m / 128)）。
            const uint32_t k = A.pfLayerK;             // 0..11（= M15Loop::AttnSlot）
            for (uint32_t off = 0u; off < A.m; off += M15PF::PF_ROWS) {
                const uint32_t rows = ((A.m - off) < M15PF::PF_ROWS) ? (A.m - off) : M15PF::PF_ROWS;
                M15PF::PfChunkArgs pfa;
                pfa.x = A.xLayer;                      // **平面基址**（[m, HIDDEN] bf16；尾部留 1 行余量）
                pfa.w = A.apW;                         // attention 的 8 role 权重平面
                pfa.cs = A.apCs;                       // cos/sin 表
                pfa.posTbl = A.pfPos;                  // 可空（按绝对行号取表）
                pfa.posBase = A.pfPosBase;             // 两者都给时以表为准
                pfa.kv = A.pfKv + M15Kv::KvLayerOffset(k);
                pfa.comp = A.pfComp + M15Kv::CompLayerOffset(k);
                pfa.ring = A.pfRing + M15Kv::RingLayerOffset(k);
                pfa.pack = A.pfPack;
                pfa.packSeed = A.pfPack;               // 本段只抄 seed 行（生产者是 B3）
                pfa.y0 = A.pfY0;                       // 3,571,712 B（见 §5.2）
                pfa.out = A.pfOut;                     // 3,571,712 B
                pfa.in = A.pfIn;                       //   299,264 B
                pfa.pooled = A.pfPooled;               //     8,192 B（见 §5.2）
                pfa.flag = A.pfFlag;                   //        32 B（见 §5.2）
                pfa.rowStart = off;                    // **本 chunk 的绝对首行号**（x 的行偏移）
                pfa.seqRows = rows;                    // 本次要处理的行数 = 本 chunk 的行数
                pfa.chunkRows = rows;                  // 逐 chunk：chunkRows == 本 chunk 行数
                pfa.nChunks = 1u;                      // **一次调用只跑一个 chunk**（>1 会被段体 Trap）
                pfa.mut = M15PF::PfUnpackMut(A.pfMutant);
                pfa.acMode = M15PF::PfUnpackAcMode(A.pfMutant);
                M15PF::AttnPrefillPhaseA(pfa);
            }
        }
        return;
    }
```

**前置条件（Wave C 的清单）**：① `LayerArgs` 追加 `pfY0` / `pfOut` / `pfIn` / `pfPooled` / `pfFlag`
（§5.2 给了尺寸；补丁引用的是这些字段与既有常量，Wave C 先按 §5.2 把它们加进 `LayerArgs`）；② `.asc` 里
`m15_attn_cache.h` 已经**自动**把本段带进融合 TU（那个头末尾 include 了 `m15_attn_prefill.h`，
**`.asc` 一字不用改**）；③ `pfBlockTable` 为 `nullptr`（非空要报错，不要静默算错）；
④ `PREFILL_WIRED` 打开前先把 `PF_{UB,L1,L0C}_ATTN` 三个槽填上 §5.3 的实数（Wave A 的合取断言会自动拦）。

### 5.9 需要 Wave A 配合的两件事（**不碰清单外文件的落点**）

1. **`m15_attn_prolog.h` 的 `AP_*` 多行化（M103 的 B2 行划给 B2，但不在本 mission 的 scope）**：
   现状 `AP_*` 的 `apX` / `apY0` / `apOut` 三块平面按**单行**定尺（M88 探针尺度）。本段**不依赖**它
   （位置按行传、平面由调用方给），所以**不阻塞接线**；但若 Wave A 之后把 prolog 探针也抬到 prefill
   尺度（`m = 4097`），那三块平面要按 `PF_ROWS × 各自列宽` 重定尺，**那时才需要**与 Wave A 对齐接口。
   ⇒ **Wave C 只需知道"本段不吃 `AP_*` 的单行假设"，不需要为这条做任何改动。**
2. **`.asc` 的 include 清单**：本段的段体头由 `m15_attn_cache.h` 末尾 include 带进融合 TU（§5.8 的前置
   条件 ②）⇒ **`.asc` 一字不用改**；若 Wave C 更喜欢显式 include，在 `m15_layer_loop.asc` 的 include 块
   里加一行 `#include "m15_attn_prefill.h"` 即可（幂等，卫哨保证不重复展开）。

## 6. 显式未完成项（写窄）

1. **多 chunk 单次发射（`nChunks > 1`）形态未通，**段体现已对它硬拦****（§4 第 1 条）：设备上 AIC 臂与
   AIV 臂会在若干轮后失步（y0 全是毒值、门控 lane 未写、packed 却对）。**本 mission 交付并验证的形态是
   `nChunks = 1`**（一个 chunk 一次启动；`big4097_step` 那 33 次启动就是它的实跑证据）。
   ⇒ **Wave C 接进融合 kernel 时按 chunk 启动 N 次**（N = ceil(m/128)，补丁见 §5.8）。
   `AttnPrefillPhaseA` 现在对 `nChunks > 1` **`Trap` 硬拦**（响亮失败，防"静默跑错"）；
   harness 里 `big4097_seq_refused` 这一档就是去打那道硬拦的**已知缺陷档**（期望 launch 非正常结束）。
   读数（§4 末尾 B）：host 侧 `err=507015 msg=EZ9999: Inner Error!`，**设备侧** `errcode 286 /
   "The trap instruction reports an error."`（**28 条 `aicore`(AIC) + 56 条 `aivec`(AIV) = 84 条 = 全部物理核**、
   同形）⇒ 那道硬拦确认为 trap 指令异常；
   Trap 之后同一 stream 仍能跑契约档 ⇒ 这一次读数里**没有连坐**。它只在 `nChunks > 1` 时触发
   ⇒ 已验证的 `nChunks = 1` 路径不变。**根因仍是假设**（怀疑与 `(both,2,4)` 的 2:1 累计有关），
   没有"根因级"的设备错误报告 ⇒ 不当结论引用；要打开这条形态，先做 §5.5 里那两件事。
2. **cache 填仍是单核（AIV block 0）串行**：B2 交的是"**填对的** cache"（判据在寻址/几何/数学），
   **不是吞吐**（M98 探针同样如此）。多核切分（按行切 + 组内成员跨核的交接）**未做**，
   属于 Wave C/D 的性能收口。
3. **`big4097_step` 的 1 个元素越界**（2,097,152 里 1 个，1.33×界）登记为**已接受偏差**（§4）；
   若要清零，把 `PF_T3_SLACK_ULP` 置 1（推导已在源码注释里）。
4. **`out` 上的 k/v/rawk 三段不物化**（有意，见 §2 的 ⑤ 与文件头）：覆盖由 cache 侧判据接管。
5. **多请求 / 非连续位置 / 非恒等 `block_table`**：**不支持**（显式契约，见 §5.7）。给表时设备会逐行复核
   连续性，违反则**跳过 cache 填**并置 `AC_FLAG_FE_CONTRACT = 1`（响亮失败）。
6. **`m15_attn_prolog.h` 的 `AP_*` 多行化未做**：M103 的 B2 行把该文件划给 B2，但**本 mission 的 scope
   里没有它**。本 mission 用**不碰它的方式**达到了"per-row 位置"这一项目标（`PfPosRef` 在段体内传位置，
   `AP_*` 一个字节没改）。若塔要把它多行化（`apX` / `apY0` / `apOut` 三块平面按 m 定尺），那是 Wave A 的事
   —— 本段**不依赖**它。
7. **不引入 `slot_mapping`**（M103-3 的结论 + M82 的 D2）：扩展点是 Wave A 的 `pfBlockTable`。
8. **性能 / aclgraph / msprof**：不在本 mission（`docs/15` M103-2.2 的 B2 行没有这一项）。
9. **packed 的内容**不做（只抄 seed 行）：生产者是 B3 的 indexer。

## 7. 复现命令

```bash
cd <wt-116>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m24_attn_prefill/build -S m24_attn_prefill -DCMAKE_BUILD_TYPE=Release
cmake --build m24_attn_prefill/build -j8
# 设备槽纪律：flock -w（在锁文件之前）；进锁后再 npu-smi 复查；内层再套一个 timeout 兜底
mkdir -p m24_attn_prefill/evidence
flock -w 1800 /tmp/npu0.lock bash -c 'df -h / | tail -1; npu-smi info | sed -n "2,12p";
  timeout 900 ./m24_attn_prefill/build/m24_attn_prefill' 2>&1 | tee m24_attn_prefill/evidence/m24_all.log
# 单档复跑（诊断用）：M24_ONLY=m64 ... 同上
```
