# M164 —— 预填充 epilog 链路（转位 → S5 RMSNormGated → S6 out_proj）挂进预填充路

> **性质与边界（先读）**
> - 本 mission **只接 epilog 链路**：`相位 A 出口 wsO + prolog S2 的 z → ① 转位/落位 → ② S5
>   RMSNormGated → ③ S6 out_proj → ④ H2 的 bo（hcAttnOut）`。**attention 的 B2/B3 未接**；
>   **不得**读成「整层四相位打通」。`m15_layer_resources.h:503` 的 `PREFILL_WIRED` 仍为 0。
> - 形态 = M151 同款 **host-orchestrated 多 launch**（把 rest 拆成 `A` 与 `H2|B`，中间插 epilog）；
>   不改 `m15_layer_kernel.h` 的 ABI/实参表，不新增 flagId，不用 set flag/wait flag。
> - 契约（带 `文件:行`）见 `CONTRACT.md`；行号基准 = 包含本文件的提交。
> - 既有红项（相位 B m=1 MoE router、m=4097 `h2.blk` ulpMax 11）在本 README **只作回归输入**，
>   **不预判因果**；本 mission 重设了 H2 的数值输入基线（原因见 §7）。

## 0. 一句话

把相位 A 出口 `wsO`（head-major fp32 `[48,m,128]`）与 prolog S2 的 z 段，经**转位 → S5 →
S6** 落成 H2 的 `bo`（`hcAttnOut`，bf16 行主序 `[m,2560]`，行距 5120 B），使 H2 的 combine 吃到
**设备产出**（在此之前它吃的是宿主合成 bo）。

## 1. 数据流与接线契约

```
① H1（产 BLK） → ② S2 in_proj + S3 prolog（产 q/k/v/g/β 与 pfQkvzba）
③ 相位 A（吃设备 q/k/v/g/β，产 wsO = pfGdnODev [48,m,128] fp32）
④ epilog：转位（wsO → oTok fp32 [m,6144]；qkvzba z 段 → zTok bf16）→ S5（→ yTok bf16 [m,6144]）
        → S6（→ hcAttnOut，Fixpipe dstStride=N=2560 elems / F322BF16）
⑤ H2（combine 读新的 bo = hcAttnOut）
```
挂载点与逐条 `文件:行` 见 `CONTRACT.md` §0/§6。关键锚点：`H_PfGdnWired` 的 `epilogOn` 分支、
`H_PfEpilogRun`（`m15_layer_loop.asc`）；三段 device 段体在 `m15_prefill_epilog.h`
（`M15PE::Trans` / `M15PE::Chain`）。

## 2. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `m15_layer_loop/m15_prefill_epilog.h` | **新增**：`M15PE::Trans`（m28_gdn_epilog.asc:69-220 逐字）、`M15PE::Chain`（m28_epilog_chain.asc:91-893 逐字）+ 四个入口壳 `m15_pf_epilog_{transpose,s5,s6,s6_mut}_kernel` |
| `m15_layer_loop/m15_layer_loop.asc` | include；`Opts.pfEpilog/pfEpilogMut` + env；`Ctx` 三块新平面；`H_PfSegAlloc` 按需分配 + 毒 `0xCD`；`H_PfEpilogDump`/`H_PfEpilogRun`；`H_PfGdnWired` 的 launch 拆分与插入；4 条跨文件 `static_assert` |
| `m15_layer_loop/m15_layer_resources.h` | 新增 §3c-ter：epilog 的 UB/L1/L0C 峰值 + GM 台账 + `PfPeakUnfilled()==3`/`PREFILL_WIRED==0` 的 `static_assert` |
| `m15_layer_loop/evidence/m164_epilog_mount/**` | 本目录（契约 / README / 复算脚本 / lift 见证 / 逐档日志） |

## 3. 抽取等价性（逐字节见证）

`verify_lift.py` 从 donor 重切两段并与头文件里两个标记区间逐字节比对：

```
TRANS  donor m28_gdn_epilog/m28_gdn_epilog.asc:69-220  sha256 370e18e226c5750d… 6599 bytes
       header M15PE::Trans                              sha256 370e18e226c5750d… 6599 bytes   OK
CHAIN  donor m28_gdn_epilog/m28_epilog_chain.asc:91-893 sha256 d4637d8ff6447c0b… 36013 bytes
       header M15PE::Chain                              sha256 d4637d8ff6447c0b… 36013 bytes  OK
==== verify_lift: LIFT OK ====
```
**只抽 device 段，不含 donor 的 host `main()` 与 #include**；入口壳是**新写**的（4 个），不在逐字区间内。

## 4. 设备档读数（真权重、`M15_LAYERS=1`、相位掩码 `H1|A|H2=7`）

九档，**每档各自进一次 `flock -w 300 /tmp/npu0.lock`**，进锁先 `npu-smi`（快照落进该档日志），
`timeout` 在锁内；逐档日志见 `logs/run_*.log`。

| 档 | 环境（除公共外） | 进程内判定 | 日志 |
|---|---|---|---|
| `m1_clean` | `M=1 PROLOG=1 EPILOG=1 MUT=0` | ALL PASS（checks=89, fails=0） | `logs/run_m1_clean.log` |
| `m4_clean` | `M=4` 同上 | ALL PASS（89） | `logs/run_m4_clean.log` |
| `m4097_clean` | `M=4097` 同上 | ALL PASS（89） | `logs/run_m4097_clean.log` |
| `m1_epilog0` | `M=1 EPILOG=0`（判据 c 的对照档） | ALL PASS（69） | `logs/run_m1_epilog0.log` |
| `m1_mut1` | `M=1 EPILOG_MUT=1`（转位 headstride） | ALL PASS（89） | `logs/run_m1_mut1.log` |
| `m4_mut1` | `M=4 EPILOG_MUT=1` | ALL PASS（89） | `logs/run_m4_mut1.log` |
| `m1_mut2` | `M=1 EPILOG_MUT=2`（S5 nogamma） | ALL PASS（89） | `logs/run_m1_mut2.log` |
| `m1_mut4` | `M=1 EPILOG_MUT=4`（S6 K−1） | ALL PASS（89） | `logs/run_m1_mut4.log` |
| `m1_hostsynth` | `PROLOG=0 EPILOG=0`（M136 基线回归） | ALL PASS（38） | `logs/run_m1_hostsynth.log` |

epilog 出口的进程内判据读数（`Pf.gdn` arm 首现）：

```
m1_clean    hcAttnOut[0,1)  0xCD 残渣 0/2560、非零 2560、非有限 0（输入 wsO 非有限 0 ⇒ 全有限）
            尾行 [1,4097) 0xCD 10485760/10485760
m4_clean    hcAttnOut[0,4)  0xCD 残渣 0/10240、非零 10240、非有限 10240（输入 wsO 非有限 2048 ⇒ 预期非有限）
            尾行 [4,4097) 0xCD 10478080/10478080
m4097_clean hcAttnOut[0,4097) 0xCD 残渣 0/10488320、非零 10488320、非有限 10488320
            （输入 wsO 非有限 8374400 ⇒ 预期非有限）；尾行 [4097,4097) 0xCD 0/0
```
- m=4097 的 `wsO` 非有限计数 **8374400** 与 M151 登记的相位 A 读数逐字一致 ⇒ **非有限是继承自
  相位 A 出口**（见 §7），不是 epilog 引入的。
- 尾行 `[m, M_PREFILL)` 在 m=1/m=4 档**整片保毒**（`0xCD`），即未写区没有看起来合法的数据。

## 5. 离线对拍（独立参考，锁外）

`check_epilog.py` 用 numpy **独立**重算链：转位与 z 压实按 **T1 逐位**，S5 按 **T3/bf16 相对 1e-2**，
S6 按 **T3**（`|Δ| ≤ 2.5e-3·Σ|terms| + 1e-6`，参考用 BLAS fp32 独立算）。判据的期望结果按
`mut` 逐档判（干净档全绿、负向档对应项必须红），并对非有限位置按「pattern 必须一致」判。

| 档 | ① 转位 bit-diff | ② z 压实 bit-diff | ③ S5 finite-bad / worstRel | ④ S6 finite-bad / worst(tol-ratio) |
|---|---|---|---|---|
| m=1 clean | 0/6144 | 0/6144 | 0/6144 / 0 | 0/2560 / 0.0788 |
| m=4 clean | 0/24576 | 0/24576 | 0/22528 / 0 | 0/0（见 §7） |
| m=4097 clean | 0/25171968 | 0/25171968 | 0/16797568 / 0.00775 | 0/0（见 §7） |

⇒ **m=1**：四段全绿（S5 的 bf16 相对逐位一致 worstRel=0；S6 的 T3 比值 0.0788 ≪ 1）。
⇒ **m>1**：转位与 z 压实逐位一致（含 NaN payload 位）、S5 的**有限位置**全部在容差内；非有限位置
两边的 pattern 一致（`nan-pattern-mismatch 0`）。

**(a) 出处**：`0xCD16` 残渣 0；设备产出 vs 宿主合成 bo **逐面 DIFFER**（m=1 2560/2560、m=4 10240/10240、
m=4097 10488320/10488320）⇒ 数据来自设备，不是宿主合成件的回声。

**(c) H2 真吃到新 bo**：`EPILOG=0/1` 两档的 H2 输出 `m140_Pf.gdn_h2_blk.bin`
sha256 前 16 位 = `e0954ade5a73ed27`（epilog=1）vs `e827954dc0893e45`（epilog=0）⇒ **DIFFER**。

**(d) 相位 A 无副作用**：`m23_gdn_prefill/check_ref.py` 对 m=1 与 m=4097 的 dump 均 PASS：
```
m=1    o 超界 0/6144      max|Δ|=5.267e-09  | ht 超界 0/786432   max|Δ|=1.194e-08 | PASS
m=4097 o 超界 0/25171968  max|Δ|=1.892e-07  | ht 超界 0/786432   max|Δ|=5.543e-07 | PASS
```
（m=4097 档参考侧有 `np.exp` overflow 的 RuntimeWarning，是 fp64 参考本身对 `exp(-Σg)` 的行为，
与设备侧同一 pattern；判据 0 超界。）

## 6. 负向对照（能变红，非空洞）

| 对照 | 期望 | 实测 |
|---|---|---|
| 转位 `headstride`（`MUT_HEADSTRIDE`，源 offset 去 `*m`）m=1 | 退化档：转位恒等 ⇒ o 仍相等 | o bit-diff **0/6144**（分离成立） |
| 同上，m=4 | m>1 必红 | o bit-diff **24064/24576** |
| S5 `nogamma`（gamma 预转表置 1.0） | S5 出口必红 | S5 finite-bad **4932/6144**，worstRel 0.15 > 0.01 |
| S6 `K−1`（经 `M15OP::OProjGemm` 的现成 mutation） | S6 出口必红 | S6 finite-bad **619/2560**，worst(tol-ratio) 5.34 > 1 |

⇒ 三条负向对照各自只打红对应的段（S5 nogamma 档的 S6 判据仍绿：S6 的参考吃的是**设备实际 y**，
所以它正确；这正说明判据是**分段**的，不是一条笼统的"全绿/全红"）。

## 7. 已知限度 / 未接（如实登记）

1. **M159 §2.2 的「两个新 GM 暂存」少了第三个**：`y`（S5 出 = S6 的 A 面）是转位/S5 与 S5/S6 之间
   的第三个 GM 中间面（donor `m28_epilog_chain.asc` README 的中间面带也单列它为「中间 `y`」，
   `RunChain` 亦为其单独分配）。本实配 **三块**：`pfOTokDev`/`pfZTokDev`/`pfYTokDev`。
2. **M159 §2.3 的「H2 读 0xCD 毒值」更正**：当前 tip 上 H2 读的是 **宿主合成 bo**（`H_PfHcH2D`
   把合成面 H2D 进 `pfHcAttnOutDev`），不是毒值。epilog 把它换成**设备产出**；判据 (c) 区分的正是
   「宿主合成」与「设备产出」。
3. **m>1 的非有限是继承的**：`wsO` 在 m>1 含相位 A 递推 `exp(-Σg)` 溢出后的非有限（M151 §5 已登记，
   独立 m23 fp64 参考复现同一 pattern）。epilog 忠实传播它（转位逐位、S5 非有限 pattern 一致）。
   因此：
   - m>1 档的 **S6 数值判据是空过的**（`finite-bad 0/0`：每行都含非有限 ⇒ 无有限位置可比）。
     S6 的**数值 PASS 只在 m=1 档成立**（0/2560，worst ratio 0.0788）。**不得**把这个空过读成
     "S6 在 m>1 数值正确"。
   - 进程内"出口全有限"判据**只在输入全有限时**作为判据；输入非有限时改判"出口确实带非有限"
     （反面对照：不许被静默清成有限值）。
4. **多层链路未接**：epilog 的 gamma/Wout 取层 0（与本 run 的 `M15_LAYERS=1` 一致）；多层的
   权重轮转不在本 mission。
5. **attention B2/B3 未接**；`PREFILL_WIRED` 仍为 0。
6. **既有红项**（相位 B m=1 MoE router、m=4097 `h2.blk` ulpMax 11）本 run 的掩码 `H1|A|H2=7`
   不含相位 B，且 `h2.blk` 的 ulpMax 判据在 m27/m140 的口径里 ⇒ 本 README 不重复判它们，
   只作为**回归输入**登记；epilog 挂载改变了 H2 的 bo 输入 ⇒ 那些读数会移动，须重设基线，
   **不预判因果**。
7. `check_epilog.py` 的 `f32_to_bf16` 第一版把 max-payload NaN（`0x7FFFFFFF`）经 RNE bias 进位成
   `0x8000`（-0.0），曾造成 m>1 的 `nan-pattern-mismatch` 假红；已改为对非有限 lane 显式
   canonicalize（本 README 的读数来自修正后的脚本）。

## 8. 复算命令（可传播失败）

```bash
cd m15_layer_loop/evidence/m164_epilog_mount
bash reproduce.sh                 # 九档设备 + 锁外离线判据（失败即非零退出）
bash reproduce.sh --offline-only  # 只跑离线（lift 见证 / numpy 链 / 判据 c / 判据 d）
```
`reproduce.sh` 的纪律：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（写进该档
日志）、`timeout` 在锁内、一次进锁一条短命令；**等锁未拿到 ⇒ 该档记「未取得读数」并以非零退出**
（不写成"未复现"）。所有离线对拍在锁外做。

## 9. 纪律自检

- 判据非空洞：每条判据都带"若被测对象坏掉它会怎样"的对照（§6）；m>1 的 S6 空过已显式披露（§7.3）。
- 参考独立于实现：numpy 链不是从 kernel 翻译来的；S6 参考用 BLAS 独立算。
- 写窄：只接 epilog 链路；attention B2/B3 未接；既有红项原样保留。
- 六个禁用词在本目录（README/CONTRACT/脚本）逐词计数为 0（`grep -c` 见 §9 脚注命令）。
  为免自命中，此处不逐字复写清单。
- 引用的行号/符号以**本提交 tip** 为准（`CONTRACT.md` 顶部给了重算命令）。
