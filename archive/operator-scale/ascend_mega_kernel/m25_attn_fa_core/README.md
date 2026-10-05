# m25_attn_fa_core —— M117 / Wave-B3：**稠密 causal prefill attention core** 的独立验证路

> # ✅ 挂死已解、数值已收敛（M189，2026-10-05）
> **M185 之前**：`m=64 / 16 / 96 / 128` 四个档全部在设备上挂死（`timeout` 被 kill、**无任何输出文件**）。
> **M185**：核间同步由 **mode 4（AIC↔单个 AIV、带 `+16` 双通道）改成 mode 2**（1 AIC ↔ 配对的
> 2 AIV），并把 P 由 GM 中转改成 **UB→L1 硬通道直写** ⇒ **四个 m 档全部跑完（EXIT=0）、拿到
> PASS/FAIL 读数**；但数值仍 FAIL（`maxratio=195.58`），M185 把失败点缩到 cube 的 S/PV 数据面，
> **未定位**。
> **M189 定位并修复根因**：`L1 -> L0B` 的两条流水里，**生产侧（MTE1）的 BufferID
> `Acq/Rls` 被放在 `ks`（BMM1 的 2×128 k 半块）/ `nh`（BMM2 的 2×128 维半块）循环之外**
> ⇒ M 侧两条 `Mmad` 都等到循环结束才拿到 L0B、**都读最后一次装载**：BMM1 两个 k 半块都用了
> 第二个半块的 K（S 系统性错），BMM2 两个维半块都用了第二个半块的 V（PV 系统性错）。
> 修法 = 把 `Acq/Rls<PIPE_MTE1>(B_L0/B_K|B_V)` 挪进循环、每个半块装载后立即 release
> （与 donor `m10_attn_decode` BMM1 逐字同款）。**只改同步，不改任何装载几何/数值参数。**
> **M189 设备读数**：`check_ref.py` 四档 **core 全 PASS（`over=0`、`maxratio=0.2962`）**；
> 负向对照 `negmask/negshift/negstart` **全 FAIL**；同一 `m` 三跑 sha256 一致。
> 逐档读数与定位过程见 `evidence/M189_readings.txt`（历史负结果资产见 `evidence/M185_readings.txt`）。
> ⇒ 离线判据自证仍在：`evidence/check_ref_selftest.txt`
>   （回灌 fp64 参考 ⇒ PASS `maxratio 0.2203`；×1.05 ⇒ FAIL `over=117926/227328`）。

> # M196：两个**生产形态**验收通过（2026-10-05）
> 在 M189 tip（`f66ad1e`）上，按 harness **默认档**跑 `core`：
> **m=4097**（prefill，`posBase=0,ctx=4097`）`over=0/25171968 maxratio=0.2962`；
> **m=1**（decode，`posBase=4096,ctx=4097`）`over=0/6144 maxratio=0.0262`；
> 同档四跑 `out.bin` sha256 一致。两档的 `nTiles` 上界都 = **33**（按 `FacMakeWork` 自算，
> 见证据 §2）⇒ 设备上首次跑到多 tile。三个负向对照在 **m=4097 均 FAIL**（判据会咬）；
> **在 m=1 上三者不咬（PASS）** —— decode 单行已见全部上下文，机制与读数如实登记在证据。
> `m=4097` 设备核墙钟 ≈ **3.4 ms**（`time_core.py`，只读既有打印）；端到端 ≈ 6.8 s。
> 逐档读数 / 原始日志 / 复现：`evidence/M196_readings.txt`、`evidence/m196_logs/`、
> `reproduce_accept.sh`。**口径**：core 级、合成 bf16、对自建 fp64 dense-causal 参考，
> **不是**真实权重 / KV parity。

> 本目录是 `docs/15-prefill-design.md` §M103-2.2 **B3 行**（已改名 `m15_attn_fa_core*`）的**独立验证路**。
> 并入的段体头是 `m15_layer_loop/m15_attn_fa_core.h`、host 头是
> `m15_layer_loop/m15_attn_fa_core_host.h`（**都是融合形态的那一份，不复制** —— CMake 用
> `-I ../m15_layer_loop` 引用它们，于是「被测对象」与「将来融合用的那一份」不可能漂移）。
> **本目录不碰 `m15_layer_loop/CMakeLists.txt`**（塔裁：Wave B 与那个文件零接触），
> 也**不碰 M101 的 `m15_attn_core.*`**（那是 decode 形状，KEEP OUT）。

## 0. 一句话

单请求、恒等 `block_table` 下，`out[i,h,:] = Σ_{j ≤ posBase+i} softmax(Q·Kᵀ·scale + 掩码) · V`
的**稠密 causal** FlashAttention core：**QKᵀ 与 PV 两个矩阵乘法都走 cube 上的 `Mmad`**，
AIV 只做非矩阵乘的向量/标量运算（缩放 / **核心自带的三角掩码** / online softmax / 行列归一）。

真实 shape：`FAC_P=64`（donor config4 sOuter）、`FAC_SIN=128`（config4 sInner）、
`FAC_HD=FAC_DV=256`（config4 的 D/DV，也是 checkpoint 的 head_dim）、`NH=24`、`NKV=2`。
验收档：prefill `m=4097, posBase=0, ctx=4097`；decode 同档回归 `m=1, posBase=4096, ctx=4097`。

## 1. 构建 / 运行 / 对拍

```bash
# 设备槽（**绝不并发**）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m25_attn_fa_core/build -S m25_attn_fa_core -DCMAKE_BUILD_TYPE=Release
cmake --build m25_attn_fa_core/build -j4

cd /workspace/ascend_mega_kernel && df -h /          # 写文件前先看盘
flock -w 900 /tmp/npu0.lock bash -c '
  npu-smi info -t common -i 0 | head -6                     # 进锁后再复查
  M25FA_OUT=m25_attn_fa_core/out ./m25_attn_fa_core/build/m25_attn_fa_core core'

python3 m25_attn_fa_core/check_ref.py m25_attn_fa_core/out/m4097_core
python3 m25_attn_fa_core/check_ref.py m25_attn_fa_core/out/m1_core
```

环境变量：`M25FA_M="4097,1"`（逗号分隔）、`M25FA_BLOCKS=<n>`、`M25FA_OUT=<dir>`。

## 2. 判据（**自建稠密 causal 参考**）

### 2.1 为什么不能用官方 QSA 输出

`docs/17` §7 / `docs/19` §11 / M103-4 三条：官方 QSA 的 `indexer_budget=2048` 是 **token** 预算，
只覆盖约一半历史；`decode ctx=4096` 同理。**稠密 causal 与官方输出在任何长度下都不可能一致**
（gather 集合与 softmax 分母都不同）⇒ 判据必须是**自建 numpy fp64 稠密 causal 参考**
（`check_ref.py`），本目录**不写**「m=4097 对齐官方输出」这类表述。

### 2.2 判据口径

按 `docs/17` §1.1 的 T3 形态 `|got − exp| ≤ ε·Σ|terms| + 0.5·ulp(out)`：

```
scale_elem[i,h,d] = Σ_j p_ij·|V[j,n2(h),d]|          # 该元素的 Σ|terms|
tol               = EPS·(scale_elem + |ref|) + 0.5·ulp_bf16(ref)
EPS               = 2^-9 × 4                          # 推导见下
```

`ε` 的来历（**推导，不是按结果调参**）：
1. P 落 bf16 ⇒ 逐元素相对量化上界 `2^-9`（尾数 7+1 位）；
2. 分子与分母同时被扰动 ⇒ `|Δout| ≤ 2^-9·(Σp|v|/Σp + |out|)`；
3. 再乘 `SAFETY=4` 吸收 **fp32 累加（≤33 个 kv tile）**、设备 `Exp`、fp32 行规约的松弛。

数字口径：**max 相对该 tol 的比值**（不是「首次不匹配」）。判定项与报告项分栏
（`docs/17` §2.1）：判定项只有「逐元素超界数 = 0」这一条；`maxRel`、`≤1bf16ulp 比例`、
`std(ref)/std(got)`、输出 `sha256` 都是**报告项**。

### 2.3 非空洞 + 把被测对象弄坏必变红

> **现状（2026-10-05，M189 实跑）**：`core` 四档（`m=16/64/96/128`）**全 PASS**（`over=0`、
> `maxratio=0.2962`）；三个负向对照 `negmask / negshift / negstart`（`m=64`）**全 FAIL**
> ⇒ **判据非空洞、且把被测对象弄坏必变红**（设备档，不再只靠离线自证）。
> 逐档读数见 `evidence/M189_readings.txt`。

| 对照 | 怎么弄坏 | 期望 | M189 设备读数 |
|---|---|---|---|
| `core` | — | 判据 **PASS** | **PASS**（四档，`over=0`，`maxratio=0.2962`） |
| `negmask` | 调 `FaCoreBody<false>`（**关掉三角掩码**，未来 token 也能看见） | 判据 **FAIL** | **FAIL** `over=377061/393216, maxratio=72.62` |
| `negshift` | KV 平面指针整体后移 16 个 token（K/V 与位置错位） | 判据 **FAIL** | **FAIL** `over=381380/393216, maxratio=6.9e30` |
| `negstart` | 传 `posBase+1`（因果窗口整体错位） | 判据 **FAIL** | **FAIL** `over=346055/393216, maxratio=134.50` |

三个负向对照都**只动调用点**（`m15_layer_loop/m15_attn_fa_core_host.h` 的注入点），段体一字未改；
它们的 `params.txt` 写**正确**的 `posBase`，于是参考是正确的那份 ⇒ 判据必须变红。
`check_ref.py` 的 guard（报告项）另给 `std(ref)` 与 `std(got)`，防「恒输出常数也能过」。

### 2.4 确定性

`run_checks.sh` 对同一档连跑 `M25FA_REPEAT`（默认 5）次，比较 `out.bin` 的 `sha256`；
再跑三个负向对照并断言它们变红。**不写绝对断言**：脚本打印每次的哈希与判定，结论以打印为准。

## 3. **融合清单**（`docs/15` §M103-2.7 第 2 条的六项）

> ⚠ 数值已收敛（M189 四档 PASS，见文首横幅），但下列挂载点 / 资源窗 / 开关**尚未在融合 TU
> （`m15_layer_loop`）里按相位边界实跑验证** ⇒ 只能当"照此接入"的提案，**不得读成"已可用"**。

### (a) 挂载点

* 入口符号（**11 参**，逐字；`witScratch` **必填、无默认值**）：
  `M15FAC::FaCoreBody<Causal>(q, qStride, kv, out, pScratch, witScratch, m, posBase, ctx, lane, nBlk)`
* 相位：`m15_layer_kernel.h` 的 `M15L_PrefillPhaseA<KIND_ATTN>`（**相位 A**，`pfWired != 0` 分支）。
* 消费的 GM 平面：① Q 平面（= prolog 的 `apOut`，行距 `AP_OUT_N = 13952`）；② 主 KV 平面（本层层基址）。
* 生产的 GM 平面：attention 输出平面 bf16 `[m][NH*HD]`（行距 6144）。
* 跨段：**段起点不 wait 任何段外 flag**；段尾由调用方补 `PipeBarrier<PIPE_ALL>()`。

### (b) `LayerArgs` 需要的字段（Wave A/C 照它加）

**5 个 `__gm__` 指针 + 6 个标量**：
```
__gm__ uint8_t* attnQ;          // Q 平面（= prolog 的 apOut），行距 = M15AP::AP_OUT_N
__gm__ uint8_t* attnKv;         // 主 KV 平面基址（**本层**；M15KV_KV_* 编址，只读）
__gm__ uint8_t* attnOut;        // 输出平面 bf16 [m][NH*HD = 6144]
__gm__ uint8_t* attnPScratch;   // P 中转：nBlk*2*FAC_P*FAC_SIN*2 B（= nBlk*32 KB）
__gm__ uint8_t* attnWitScratch; // **跨核记账见证区**（只被编译期开关 M25FA_WIT 用；默认惰性）
                                // 需要 (28 + 2*nBlk) × 32(id) × 2(set/wait) × 4 B；nBlk=28 时 = 21,504 B
                                // ⚠ 本档测试 host 的 kWitBytes 按 56 槽分配（nBlk ≤ 14 够用）——
                                //    探针构建的已知不足，见 evidence/exclusions.md §8
uint32_t        qStride;        // = M15AP::AP_OUT_N（Q 平面行距，元素）
uint32_t        pfPosBase;      // Q 行 0 的位置（chunked prefill 的起点）
uint32_t        pfCtx;          // KV 长度（可见 token 数）
```
`lane`（attention 层序号 k）现接口里用 `A.layer` 经 `AttnSlot()` 得到。
`m ∈ [1, M15Kv::PREFILL_M]`；**4097 是验收档、不是接口假设**。

### (c) 资源窗

| 类别 | 区间/峰值 |
|---|---|
| UB（AIV） | `FAC_UB_TOTAL = 148,096 B = 144.6 KB` « 248 KB |
| L1（AIC） | `FAC_L1_TOTAL = 196,608 B = 192 KB` « 512 KB |
| L0A | 64 KB（Q@0 32 KB、P@32 KB 16 KB） |
| L0B | 64 KB（用 32 KB） |
| L0C | 4 × 32 KB = 128 KB « 256 KB |
| GM scratch | P 中转 `nBlk × 32 KB`（nBlk=28 时 896 KB） |

UB 逐窗（字节）：`S0[0,16384) S1[16384,32768) PV0[32768,65536) PV1[65536,98304)
PC[98304,114688) ACC[114688,147456) THR[147456,147712) M[147712,147840)
SUM[147840,147968) ED[147968,148096)`。
（`UB_PC` = `FAC_AIV_ROWS*FAC_DV*2` = **16 KB**；P 落盘只用前 8 KB，归一化用满整 16 KB；
**没有独立的 `UB_OUT` 窗** —— 归一化 staging 复用 `UB_PC`。）
L1 逐窗：`Q[0,32768) K[32768,98304) V[98304,163840) P0[163840,180224) P1[180224,196608)`。

**P 走 UB→L1（M185 起，不再经 GM）**：M121 在本平台（3510 真机）实测 UB→L1 硬通道可用
（`docs/evidence/ub_to_l1`）；AIV 用 `DataCopy(dstL1, srcUb, DataCopyParams)` 逐列分形 burst 把
P 半块写进 L1P(q) 的 NZ 槽，AIC 只做 `L1 → L0A`。**实测与旧 GM 路径逐位等价**
（`evidence/M185_readings.txt` §3）。`pScratch` 形参保留、不再使用。

### (d) BufferID 与 flagId

* AIC `MutexID`：`B_Q=0, B_K=1, B_V=2, B_L0=3, B_C0=4, B_C1=5, B_C2=6, B_C3=7, B_PL1=8`
* AIV `MutexID`：`B_S0=0, B_S1=1, B_PV0=2, B_PV1=3, B_PC=4`（`B_ACC=5` **仅登记 UB 窗名、不进令牌**；
  **没有 `B_OUT`**）。
* CrossCore flagId（**mode 2**，每核池 0..15；**每个同步事件只用一个 id**）—— **命名即语义：
  `*_RDY` 只由 AIC `set`、AIV `wait`；`*_FREE` 只由 AIV `set`、AIC `wait`（「单写方」设计，见头文件
  常量的说明）**：`CC_S_RDY=0, CC_S_FREE=1`（S(t) parity0 就绪 / 已消费）、
  `CC_PV_RDY=2, CC_PV_FREE=3`（PV(t) parity0 就绪 / 已消费）、
  `CC_P0=5, CC_P1=6`（P(t) parity **已直写 L1**，{2 AIV}→AIC）；另 parity1 用 `+1`。
  配对口径：AIC `set` 一次 → 配对 2 AIV 各 `wait` 一次；**配对 2 AIV 都 `set`** 后 AIC 的 `wait`
  才放行（严格 2 set ↔ 1 wait）。**M185 删掉了 mode-4 的 `FAC_AIV_CH(=16)` 双通道**。
* **相邻性**：`0/1/2/3` 与 `5/6` 两簇不共享物理槽；同一 (核型, mode) 子空间内同时「在飞」的 id 最多 2 个
  （一 RDY 一 FREE）。
* **每 id 的 set 次数不是常数**（= 该核处理到的 tile 数，`ceil(m/64)` 档 / 负载均衡后）
  ⇒ Wave C 必须**按档**断言 ≤ 15（见 §6 未完成项）。段内 6 个 id × 2 通道 = 12 个物理点。

### (e) 需要相位边界的位置

* **段尾**：`PipeBarrier<PIPE_ALL>()`（挂载点的 `M15L_PrefillBody` 已经带）。
* 段内**不需要**全体 AIV 的 mode-0 barrier：AIC↔AIV 全部走 mode-4 成对 flag。

### (f) `m` / `pos` 语义

* Q 平面：bf16，行距 `qStride`，头 `h` 占列 `[h*256, h*256+256)`（**对齐 prolog 的 out 平面**）。
  ⚠ 调用方必须把 Q 平面**按 `FAC_P=64` 行向上圆整分配**（尾 tile 整块读入）。
* 输出平面：bf16，行距 `FAC_NH*FAC_HD = 6144`；头 `h` 占列 `[h*256, h*256+256)`。
  ⚠ 同样必须**按 `FAC_P=64` 行向上圆整分配**（尾 q tile 会写满 64 行；行号 ≥ m 的走 padding）。
* 主 KV 平面：`M15KV_KV_BYTE_OFF_CONTIG(pos, n2, KV_LANE_K|V, dim)` 编址；**只读**。
  ⚠ 契约（M103-6 第 8 条）：**只支持单请求 / 恒等 `block_table`**；batch>1 会静默错。
  容量 `BlocksFor(ctx)` 页即可（可见页数逐 tile 收紧：`FacBlocksInTile`）。
* `posBase` = Q 行 0 的位置；行 i 位置 `= posBase+i`，只 attend KV 列 `0..posBase+i`。
  decode 档 = `m=1, posBase=ctx-1`。

## 4. **挂载点补丁文本（待收敛后启用的提案）**

完整文本见 `evidence/mount_patch.md`；核心 hunk 如下（`m15_layer_kernel.h` 的
`M15L_PrefillPhaseA<KIND>` 的 `if (A.pfWired != 0u) { ... }` 分支）：

```cpp
    if (A.pfWired != 0u) {
        // ---- M117 / Wave-B3：稠密 causal prefill attention core ----
        // 段体签名只吃「指针 + 标量」（Wave B 共同契约第 1 条）；**不吃 LayerArgs**。
        if constexpr (KIND == KIND_ATTN) {
            M15FAC::FaCoreBody</*Causal=*/true>(
                A.apOut,                                     // Q 平面 = prolog 的 out 平面
                M15AP::AP_OUT_N,                             // Q 平面行距（元素）
                A.attnKv,                                    // 主 KV 平面（本层层基址）
                A.attnOut,                                   // 输出 bf16 [m][NH*HD]
                A.attnPScratch,                              // P 中转（nBlk*32 KB）
                A.attnWitScratch,                            // 跨核记账见证区（默认惰性，不读不写）
                A.m, A.pfPosBase, A.pfCtx,
                M15L::AttnSlot(A.layer),                     // attention 层序号 k ∈ [0,12)
                AscendC::GetBlockNum());                     // = AIC 数（mix(1,2) 下两侧同值）
        }
        return;
    }
```

## 5. 判据脚本 / 证据

| 文件 | 内容 |
|---|---|
| `check_ref.py` | 自建 numpy fp64 稠密 causal 参考 + T3 形态判据（PASS/FAIL + 报告项） |
| `evidence/README.md` | **实跑读数的收口**（含挂死定位、未取到清单） |
| `evidence/M189_readings.txt` | **M189 设备读数**：根因（L0B 生产/消费 BufferID 跨循环持有）+ 四档 PASS + 负向对照 FAIL + 确定性 |
| `evidence/M185_readings.txt` | **M185 设备读数**（历史）：四档跑完（EXIT=0）+ check_ref 四档 FAIL + 单变量定位证据 |
| `evidence/exclusions.md` | **设备端挂死的排除清单**（每条：改动/读数/日志/commit）+ 未触及维度 + 见证状态 |
| `run_checks.sh` | 连跑多次判确定性 + 三个负向对照必须变红（负控默认在 `M25FA_NEG_M=4097` 上断言；m=1 不咬，见 M196） |
| `reproduce_accept.sh` | **M196 一键复跑**：默认两形态 + 三负控；内置期望 + 失败非零传播；flock/npu-smi/timeout 纪律 |
| `time_core.py` | **M196 量设备核墙钟**：只读既有 harness 的 flushed 打印，不新增探针/同步 |
| `evidence/M196_readings.txt` | **M196 生产形态验收读数**：m=4097/m=1 PASS、nTiles=33、负控 m=4097 FAIL（m=1 不咬）、m=4097 核墙钟 |
| `evidence/m196_logs/` | M196 原始 device 日志 + `npu-smi` 快照 + `checks.txt` + 墙钟 |
| `evidence/` | 实跑读数（含失败档与负向对照读数） |

## 6. 未完成项（显式写窄）

0. **端到端数值已收敛（M189）**：`m=16 / 64 / 96 / 128` 四档 `core` **全 PASS**
   （`over=0`、`maxratio=0.2962`）；`negmask / negshift / negstart` 全 FAIL；同档三跑 sha256 一致。
   根因 = `L1 -> L0B` 的两个半块循环里，生产侧 MTE1 的 `Acq/Rls` 被提到循环外，M 侧两条
   `Mmad` 都读到最后一次装载。修法只动同步（每个半块装载后立即 release）。见
   `evidence/M189_readings.txt`。**M196 更新**：`m=1(posBase=4096,ctx=4097)` 与 `m=4097` 两个
   验收档设备读数**已取到**（PASS，见文首 M196 段与 `evidence/M196_readings.txt`）；
   `m=4097` 设备核墙钟 ≈ 3.4 ms **已测**；吞吐 / 深流水分析仍未测。
1. **性能未做**：当前是**逐 tile 锁步**的 AIC↔AIV 流水（每个 tile 一次 S 往返 + 一次 P 往返），
   没有做多 tile 预取 / L0C 深流水。正确性优先；吞吐数字**未测**。
2. **每 id 的 set 次数未按档断言**：段内 flag 的 set 次数 = 该核处理的 tile 数，随 `m`/负载变化。
   Wave C 要把「≤ 15 / id」按档写进 `m15_layer_resources.h` 的断言（本段只给 id 清单与相邻性）。
3. **paged `block_table` 未接**：本段只走恒等表的连续等价式（`M15KV_KV_BYTE_OFF_CONTIG`）。
   非恒等表要走 `KvByteOffsetPaged` + `GetValue`，属 Wave D。
4. **`o_proj` / `×sigmoid(gate)` 不在本段**：本段只到 attention 输出；N1（`subOut` 的生产者）归 Wave C。
5. **多请求 batch>1 不支持**（契约见 (f)）。
6. **未与 `m21_layer_ref` 的 attention dump 对拍**：那份 dump 是**官方稀疏 QSA 输出**，
   按 `docs/17` §7 **不得**当稠密判据（本段的自建参考就是替代品）。
