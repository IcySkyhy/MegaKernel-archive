# M121 —— 3510 的 UB→L1 硬件通道真机探针（`probe_ub2l1`）

> **给 B1–B5 / Wave C 的一句话**：**3510 上「AIV 在 UB 里算出的操作数直接喂给同核组 AIC 的 mmad」这件事，真机实测可用**：
> 数据字节原样落到 L1（8192/8192 逐字节相同），AIC 把它当 mmad 操作数**真做了一次收缩**，
> 32×48×64 的 1536 个元素与 host fp64 参考的最大绝对误差 **6.18e-07**（全部落在 [0,1e-6)）；
> 通道带宽在 64KB/次时实测 **≈231 GB/s（每 AIV 核，设备侧 SYS_CNT 计时）**。
> 两条 API 路径（C API `asc_copy_ub2l1`、基础 API `DataCopy(L1, UB)`）**读数一致**，且**都命中硬件通道**——
> 塔原先「默认基础 API 不可用」的对照假设，在本机 9.1.0 上**不成立**（§4 有逐字的分支判定）。
> 需要 AIC 消费 AIV 刚写进 L1 的数据时，用 **mode 2 成对会合**（官方口径：AIV 两侧 `block_arrive` + 生产者 `intra_arrive`，
> AIC `intra_wait` + `block_wait`）；**去掉 AIV 那一侧会合，AIC 直接挂死（rc=124，两支都复现）**（§6）。
>
> **一句话边界**：以上都是**实测读数**，不是推算；但本报告**不主张**「官方样例可在 9.1.0 直接跑通」
> （≥9.2.0 声明的差异见 §7），也不主张任何绝对化的硬件结论 —— 每条读数都带最小可复现用例与逐字诊断。

---

## 0. 结论速览（每条都给判据/出处）

| # | 结论 | 判据 | 出处 |
|---|---|---|---|
| 1 | **C API `asc_copy_ub2l1` 在本机 9.1.0 真机可用**，且**不需要任何编译选项** | `capi-rt 0` → `same=8192 diff=0`（默认构建与 `-DENABLE_CV_COMM_VIA_SSBUF=true` 构建**读数相同**） | §3.1 |
| 2 | **基础 API `DataCopy(L1, UB)` 在本机 9.1.0 真机可用**，带宏与不带宏读数**逐位相同** | 两者 `maxAbs=6.183982e-07`、`exact=944/1536` 完全一致 | §3.2 |
| 3 | **塔的对照假设"默认基础 API 走 GM+Matmul 注册 ⇒ 不可用"在本机 9.1.0 不成立** | `#if KFC_C310_SSBUF == 1 \|\| __MIX_CORE_AIC_RATION__ != 1`：默认构建 `KFC_C310_SSBUF=0` 但 `__MIX_CORE_AIC_RATION__` **未定义**（`#if` 里当 0）⇒ `0 != 1` 为真 ⇒ 落硬件分支 | §4 |
| 4 | **数据真的落到 L1 了**：AIC 把它当 mmad 操作数做收缩，1536 个元素全对；**两条 API 路径给出逐位相同的 C** | 绝对误差分桶 `[0,1e-6)=1536`、其余桶全 0；独立 `check_ref.py` 复核 rc=0 | §5.2/§5.5 |
| 4b | **C API 路径端到端同样成立**（回写用 `asc_copy_l0c2gm` + `asc_set_l0c2gm_nz2nd(1,2,M*N)`）：`maxAbs=6.183982e-07`、`exact=944/1536`，与基础 API 路径**逐位相同**。（**更正**：09-27 那次"`asc_copy_l0c2gm` 一个字节都不写"是我 probe 自己的地址假设缺陷，已作废 —— 见 §5.4） | `capi-mmad opt=0`（`asc_copy_l0c2gm`+nz2nd）与 `opt=2`（基础 `Fixpipe`）**都与基础路径逐位相同**（各 3 次）；`opt=1` 关 nz2nd → 错 | §5.4 |
| 5 | **ND→NZ 必须在 UB 内自己做**（官方 scenario 2 的写法），硬件不提供随路 ND2NZ 的 UB→L1 | 官方文档逐字：`DataCopy_UBToL1_ND2Z` 为软件仿真实现…硬件本身不支持该能力 | §5.3 |
| 6 | **带宽（本 mission 最载重的读数）**：64KB/次 → **230.75 (C API) / 231.24 (基础 API) GB/s**（每 AIV 核；**2026-10-04 批次**，log 内 `at=` 时间戳） | 每核 19.66 MB / 85.02–85.21 µs，`GetSystemCycle()`（1GHz ⇒ us=cycle/1000） | §6.1 |
| 7 | 带宽随单次长度上升：1KB→146.32、8KB→212.34、64KB→230.75 GB/s（小包有固定开销；**不给出每次固定开销的定量值**） | 同批三档实测（10-04） | §6.1 |
| 8 | **AIC 消费 AIV 刚写入 L1 的数据要用 mode 2 成对会合**；`asc_sync_block_arrive` 内部就是 `ffts_cross_core_sync(mode=0x02)` | 头文件逐字（§6.3）；设备侧（本批 2026-10-04）：**AIV 侧不发 → 两支各 1 次都 rc=124 挂死**；**AIC 侧不等 → 分支里可核对的 8 次样本里有 6 次读到坏数据**（RT 四次 MISMATCH；basic-mmad 两次 `maxAbs=6.5e5 / 2.1e2`；另有 `capi_rt_k1_nowait` 与 09-27 的 `basic_mmad_k1_nowait` 各 1 次恰好读对，共 2 对）；**只让一半 AIV 发 → 两支各 1 次都挂死**（mode 2 要求配对的两个 AIV 都到） | §6.3/§6.4 |
| 9 | `kill=1`（AIC 不等）是**竞态**：分支里可核对的 8 次样本中 **6 次读到坏数据、2 次恰好读对**（读对的两次：10-04 的 `capi_rt_k1_nowait`、09-27 的 `basic_mmad_k1_nowait`）⇒ 本报告只说"**不等就至少有时会读错**" | 逐档 log 的批次时间戳（8 个 log 名见 §6.4 小表） | §6.4 第 2 条 / §B |
| 10 | 官方样例在本机 9.1.0 上**整编不通过**，缺的是 9.2 新增符号/枚举包装 | 6 处 `no member named 'ceil_div' in namespace 'AscendC::Std'` 等逐字诊断 | §7 |
| 11 | 一处**观测到但未复现、未归因**的 aicore exception（首轮 507015 / AIV 错误码 263） | 逐字 runtime 行已从 CANN plog **归档到 `evidence/logs/m121_aicore_exception_507015.plog.txt`**（原二进制未保留；plog 本身在仓外 `~/ascend/log/debug/plog/`）；后一版的 `capi-step` 二分**9 步都取到了 `NO_FAULT`**（step5..8 为 10-04 首轮、step1..4,9 为同日重采，见 §B） | §8/§B |

---

## 1. 背景与边界（这一节决定下面读数怎么用）

M120 用**host 侧三层证据**得到「UB→L1 只能走 GM 中介」这条结论，并说明它**至少过窄**：
`3510_new_features.md` 逐字写着 3510 新增 UB→L1 Buffer 数据通路（C API `asc_copy_ub2l1`「**无需配置编译选项**」，
基础 API 配 `ENABLE_CV_COMM_VIA_SSBUF=true`）。但**真机上的可用性/带宽/同步语义**当时全是空白。
本探针把这三件事补上，并顺手把「ND→NZ 要不要在 UB 内自己做」与「版本差到底卡在哪」钉死。

**本报告主张什么、不主张什么**：

- 主张：§3–§6 的每一条都有**本机真机可复现读数**（命令 + 逐字输出在 `evidence/logs/`）。
- 不主张：「官方样例能在 9.1.0 上跑通」（§7）；不主张任何关于芯片缺陷的判断（§8 那条**未复现、未归因**）；
  不主张带宽是通道的理论上限（我们只报**这个最小用例下**的实测值）。

**纪律**（人类逐字要求 + 塔的补充）：数据自造（LCG 生成 half，**不加载任何模型权重**）；变异只在 `/tmp`；
设备槽按塔 2026-10-04 收紧版纪律：一次 `flock` 只跑一条短命令（`-w ≤300`）、进锁前先 `flock -n` 探锁、
`timeout` 放在锁内、单进程、进锁后先 `npu-smi` 复查；等锁没拿到按「**未取得读数**」记（不写成「未复现」）；
结论不下绝对断言；编译/运行不过就给**逐字诊断 + 代码上下文 + 编译选项 + rc**。

---

## 2. 探针结构与复现命令

同一份源码 `probe_ub2l1.asc` 里有**两个 mix 核**，把同一份数学走两条 API 路径：

| 核 | AIV 侧的 UB→L1 原语 | 其余（AIC 侧） |
|---|---|---|
| `capi_kernel` | **C API `asc_copy_ub2l1`**（+ `asc_copy_ub2ub` 在 UB 内做 ND→NZ） | `asc_copy_l12l0a` / `asc_copy_l12l0b_transpose` / `asc_mmad` / `asc_copy_l0c2gm` |
| `basic_kernel` | **基础 API `DataCopy(l1Local, ubLocal, count)`**（同官方样例 scenario 2 的写法） | `LoadData(2DParamsV2)` / `Mmad` / `Fixpipe` |

两条路径各自沿用**同族 API**，避免把两族混用的风险带进结论。
同一份源码编两个可执行：`probe_ub2l1`（默认）与 `probe_ub2l1_ssbuf`（`-DENABLE_CV_COMM_VIA_SSBUF=true`）。

### 档位（每个 mode 一个独立进程，`timeout` 包住 ⇒「挂死」表现为 rc=124，与 FAIL/正常返回可区分）

| 档 | 做什么 |
|---|---|
| `capi-step <n>` | **二分定位**：n=1..9 累积步进（gm2ub → ub2l1 → 空转 → ub2gm → sync_pipe → mode2 会合 → l12ub → intra；n=9 用声明式 `__cbuf__` 数组替代 raw 地址 cast） |
| `capi-rt <kill> [dir]` | **往返**：UB→(ub2l1)→L1→(AIC `asc_copy_l12ub`)→UB→(AIV `asc_copy_ub2gm`)→GM，逐字节比对 |
| `capi-mmad <kill> [dir] [opt]` | **消费者验证**：ND→NZ（UB 内）→L1→AIC LoadData→Mmad→回写 GM，与 host fp64 逐元素比 |
| `basic-mmad <kill> [dir]` | 同上，走基础 API |
| `capi-bw <bytes> <reps>` / `basic-bw <bytes> <reps>` | **带宽**：连发 reps 次 `<bytes>` 的 UB→L1 拷贝，设备侧 SYS_CNT 计时 |

`<kill>` 是同步语义的变体（§6）：
`0`=官方口径（AIV 全体 block_arrive + 生产者 intra_arrive；AIC intra_wait + block_wait）、
`1`=AIC 两侧 wait 全去掉、`2`=AIV 不发 block_arrive、`3`=只发 block、`4`=只发 intra、
`5`=**只有生产者那一半 AIV 发 block_arrive**（测 mode 2 是否要求配对的两个 AIV 都到）。
`opt` 是 mmad 出参回写变体：`0`=nz2nd 开（经 `asc_set_l0c2gm_nz2nd` 配置）、`1`=nz2nd 关、`2`=改用基础 API `Fixpipe`。

### 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd probe_ub2l1
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
bash evidence/version_diff.sh     # host 侧版本差取证（零设备），输出重定向到 evidence/logs/version_diff.log
bash run_probes.sh                # 整批设备档（在 flock 内）+ 独立判据 check_ref.py
```

`run_probes.sh` 的读数落在 `evidence/logs/`，`check_ref.py` 的复核输入/输出在 `evidence/dumps/`。

---

## 3. 可用性（任务①：三条路逐条给逐字读数）

### 3.1 C API `asc_copy_ub2l1`（免编译选项）

- **编译**：默认构建与带宏构建都 **rc=0**（`evidence/logs/version_diff.log` §5）。
- **真机**（`capi-rt 0`，UB 8192B → L1 → AIC 读回 → GM）。下面是**入库 log 的节选**（逐字原件：
  `evidence/logs/capi_rt_k0_default.log`，`at=2026-10-04T09:13:03`；省略处标了 `...`）：

```
[PUB2L1] mode=capi-rt kill=0(full(official: block+intra)) ...
[PUB2L1] synchronize ret=0 (0=正常返回) hostWall=1.895 ms
[PUB2L1][RT] bytes=8192 same=8192 diff=0 ; 前 4096B 与哨兵(aGm 前 4096B)相同字节=12
[PUB2L1][RT] VERDICT=BYTE_EXACT
```

- **带 `-DENABLE_CV_COMM_VIA_SSBUF=true` 的构建里跑同一档**：`same=8192 diff=0` ⇒ **C API 这条路与编译选项无关**，
  与官方「无需配置编译选项」的措辞一致。
- **这是原始读数所在的 log**：`evidence/logs/capi_rt_k0_default.log`、`evidence/logs/capi_rt_k0_ssbuf.log`。

### 3.2 基础 API `DataCopy(L1, UB)`（带宏，官方配置）

```
[PUB2L1][MMAD] M=32 N=48 K=64 elems=1536
[PUB2L1][MMAD] maxAbs=6.183982e-07 meanAbs=7.704330e-08 maxRel=2.259530e-05 exact_vs_fp64=944/1536
[PUB2L1][MMAD] FINITE_AND_MATCHED=yes
```

### 3.3 对照组：**默认**基础 API（塔预期"走 GM+Matmul 注册 ⇒ 不可用"）

**实测：可用，且与 §3.2 逐位相同**（`maxAbs=6.183982e-07`、`exact=944/1536` 完全一致）。
⇒ 塔那条对照假设在本机 9.1.0 上**不成立**，机制见 §4。
**注意口径**：这只说明**本机 9.1.0 + `__mix__(1,2)`** 下默认构建走硬件分支；
「换别的 mix 配比 / 别的工具链版本会怎样」本探针**没有覆盖**，不外推。

---

## 4. 分支判定：默认构建为什么也走硬件分支（逐字）

被判定的就是基础 API 实现自己的条件（`.../impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:582`）：

```c
#if KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1
    CopyUbufToCbuf(dst, src, intriParams.blockCount, intriParams.blockLen, ...);   // 硬件通道
#else
    ... ScmDataCopyMsg(...) / GetKfcClient()->AllocUB(...)                          // GM + Matmul 注册
#endif
```

探针（`evidence/version_diff.sh` §6，**只编不跑**）在 mix kernel **体内**抄同一个 `#if`：

```
[选项: （无）]      SCOPE_BODY_BRANCH_HARDWARE ; BODY_MIX_RATION_UNDEFINED
[选项: -DENABLE_CV_COMM_VIA_SSBUF=true] SCOPE_BODY_BRANCH_HARDWARE
宏取值探针（static_assert 故意写错，让诊断把值打出来）：
  默认构建： static assertion failed due to requirement '0 == 1'   ⇒ KFC_C310_SSBUF == 0
  带宏构建： static assertion failed due to requirement '1 == 0'   ⇒ KFC_C310_SSBUF == 1
```

**机制**：默认构建里 `KFC_C310_SSBUF = 0`（`kernel_utils.h:36` 的 `#if ENABLE_CV_COMM_VIA_SSBUF != 0 && __MIX_CORE_AIC_RATION__ != 1 → 1 else 0`），
但 `__MIX_CORE_AIC_RATION__` 在**本机 9.1.0 的 mix kernel 体内未定义**，`#if` 里当 0 用 ⇒ `0 != 1` 为真 ⇒ 条件成立 ⇒ **落硬件分支**。
带宏时则是 `KFC_C310_SSBUF == 1` 成立。两条路因此都走同一条硬通道 —— 与 §3.3 的设备侧读数**互相印证**。

---

## 5. 正确性（任务②：消费者验证 + ND→NZ）

### 5.1 消费者验证的做法

不是"搬运没报错就算对"，而是把搬进 L1 的数据**当真 mmad 操作数做一次收缩**：
`A[32,64] × B[64,48] → C[32,48]`（half 操作数、fp32 累加、`Mmad`），GM 取回后与 host 侧**独立 fp64 参考**逐元素比。

### 5.2 读数（基础 API 路径，两构建一致）

```
[PUB2L1][MMAD] maxAbs=6.183982e-07 meanAbs=7.704330e-08 maxRel=2.259530e-05 exact_vs_fp64=944/1536
[PUB2L1][MMAD] absErr p50=5.308539e-08 p90=1.788139e-07 p99=3.455207e-07 p100=6.183982e-07
[PUB2L1][MMAD] 绝对误差分桶(每桶计数): [0e+00,1e-06)=1536 [1e-06,1e-05)=0 [1e-05,1e-04)=0 [1e-04,1e-03)=0 [1e-03,1e+30)=0
```

独立复核（`check_ref.py`，numpy，另一套实现）：`[J1] OK / [J2] OK / [J3] OK` + `[PASS] 全部判据成立（rc=0）`。
误差形态符合"half 操作数 + fp32 累加"该有的样子（量级 ≈ 1e-7，且 944/1536 与 fp64→fp32 舍入逐位相等）——
**数据确实是原样进了 L1 又原样被 L0 取走的**；排布错位、中途截断或读到脏值，与这个误差分布（1536 个元素全落在 [0,1e-6)、944/1536 与 fp64→fp32 舍入逐位相等）都不符。

### 5.3 ND→NZ：**必须在 UB 内自己做**

官方 3510 文档逐字（`DataCopy_UBToL1_ND2NZ.md:37`）：本接口为软件仿真实现…数据先搬入 GM 再搬入 L1 Buffer，
需要先 `REGISTER_MATMUL`；样例 README 更直白：**硬件本身不支持该能力**。
所以本探针按官方 scenario 2 的做法：**先在 UB 内逐 C0 列块重排成 NZ，再用连续搬运送进 L1**
（`UbNdToNz()` / `BasicUbToL1Nd2Nz()`，见源码注释）。A（32×64，4 个 C0 列块）与 B（64×48，3 个列块）都算对。

### 5.4 C API 路径的端到端：**成立**（与基础 API 路径逐位相同）+ 两处**我自己 probe 的坑**

**正确读数**（2026-10-04，当前提交的源码；`capi-mmad 0 <dir> 0`：C API 路径，回写用
`asc_copy_l0c2gm` + 先 `asc_set_l0c2gm_nz2nd(1,2,M*N)`）：

```
[PUB2L1][MMAD] maxAbs=6.183982e-07 meanAbs=7.704330e-08 maxRel=2.259530e-05 exact_vs_fp64=944/1536
绝对误差分桶: [0,1e-6)=1536  其余桶全 0
check_ref.py → [PASS] 全部判据成立（rc=0）
```

与基础 API 路径（§5.2）**逐位相同**。⇒ **C API 路径同样完成了消费者验证**：
`asc_copy_ub2l1` 搬进 L1 的操作数被 AIC 当 mmad 操作数真做了收缩，且结果与另一族 API 完全一致。

**三个 `opt` 变体（各 3 次，用来界定"哪一步是必需的"）**：

| `opt` | 回写方式 | 结果（10-04，各 3 次） |
|---|---|---|
| `0` | `asc_copy_l0c2gm` + `asc_set_l0c2gm_nz2nd(1,2,M*N)` | **对** ×3（`maxAbs=6.183982e-07`，`exact=944/1536`） |
| `1` | `asc_copy_l0c2gm` 且 `nz2nd_en=false` | **错** ×3（`maxAbs=1.135639e+01`（2 次）/ `1.276375e+01`（1 次），`exact=30/1536`）⇒ **NZ→ND 那条配置是必需的** |
| `2` | 改用基础 API `Fixpipe` 去读同一块 CO1 视图 | **对** ×3（与 opt=0 逐位相同） |

（逐档 log：`capi_mmad_k0_opt{0,1,2}.log` + `capi_mmad_{a,b}_opt{0,1,2}.log`，`at=` 为 2026-10-04。）

**一条重要的自我更正（两次，都必须说清，免得被当成硬件结论）**：
我在 2026-09-27 的构建里，用的是"mmad 写一个**独立声明的** `__cc__ float cL0[]`，回写时再用
`LocalTensor<float>(TPosition::CO1, 0, C_ELEMS)` 去读它"——**这里隐含了"`cL0` 恰好在 L0C 偏移 0"这个假设**。
后来为了别的事加了几行代码（`kill=5`），这个假设失效，**同一份 mmad 代码的读数就从"对"变成"错"**。
于是 09-27 那次"`asc_copy_l0c2gm` 一个字节都不写"的结论，
**是我 probe 自己的缺陷（未声明的地址假设），不是硬件行为，已作废**。
现在改成**写出与读出显式引用同一块 CO1 缓冲**（`LocalTensor<float> cT(TPosition::CO1,0,C_ELEMS);
__cc__ float* cL0 = (__cc__ float*)cT.GetPhyAddr();`），假设消失；
`opt=2` 那条"跨族混用不行"的结论同样是这个缺陷的产物，**也一并作废**。
此外 L1→L0、M→FIX 之间原先用 `asc_sync_pipe(PIPE_MTE1/PIPE_M)` 代替事件配对，
也出现过"同一二进制时对时错"；现在改成 `SetFlag/WaitFlag<MTE1_M>` 与 `<M_FIX>` 之后，
opt=0/1/2 **各跑 3 次**读数完全稳定（逐档 `capi_mmad_k0_opt{0,1,2}.log` + `capi_mmad_{a,b}_opt{0,1,2}.log`）。

**教训（对 B1–B5 有用）**：① 同一块 L0C 的"写"与"读"**必须用同一个显式引用**（别一边用声明的数组、
一边用"偏移 0"的 `LocalTensor`）；② **跨流水（MTE1→M、M→FIX）用显式事件配对**，
不要用 `asc_sync_pipe` 顶替 —— 这两条都是本探针踩过、并且靠"同一二进制重复跑"才暴露出来的。

### 5.5 两条 API 路径互证

同一个 32×48×64 问题、同一份输入 dump（`a.bin`/`b.bin` 逐位相同），
**C API 路径（`asc_copy_ub2l1`）与基础 API 路径（`DataCopy(L1,UB)`）给出逐位相同的 C 矩阵**：
max=6.183982e-07、p50=5.355105e-08、p90=1.788139e-07、p99=3.479421e-07、exact=944/1536、相对误差 max=1.677e-05。
两条互不相干的 API 族得到同一结果，说明**UB→L1 通道本身（而不是某个 API 的包装）在起作用**。

---

## 6. 带宽与同步（任务③④）

### 6.1 带宽（读数方式一并给出）

**方法**：AIV 上把源数据用 `GM→UB` 装好，预热一次，然后 `GetSystemCycle()` 夹住
`for (reps) asc_copy_ub2l1(L1 槽轮转, UB, bytes);` 再 `asc_sync_pipe(PIPE_MTE3)` 排空后取第二个 stamp。
`us = cycle/1000`（本机 950PR 文档逐字：1GHz ⇒ `time = cycle/1000` us）。
L1 目的在 4 个槽间轮转（每槽 512B 对齐、总占用 ≤384KB，不越 L1 的 512KB）。**两个 AIV 子块跑同一循环**，
因此下面报的是**每核自己的循环吞吐**（单核口径）。

下表**全部取自同一批（2026-10-04）**；每行的 `at=` 取自对应 log，用来核对批次
（`capi_bw_1k_2k.log` 08:39:41 / `capi_bw_8k_1k.log` 08:39:46 / `capi_bw_64k_300.log` 08:49:31 /
`basic_bw_64k_300_default.log` 08:49:36 / `basic_bw_64k_300_ssbuf.log` 08:39:37）：

| 档 | 单次字节 | reps | 总搬运 | device us | **GB/s（每核）** |
|---|---|---|---|---|---|
| `capi-bw 1024 2000` | 1 KB | 2000 | 2.048 MB | 13.997 | **146.32** |
| `capi-bw 8192 1000` | 8 KB | 1000 | 8.192 MB | 38.579 | **212.34** |
| `capi-bw 65536 300` | 64 KB | 300 | 19.66 MB | 85.205 | **230.75** |
| `basic-bw 65536 300`（默认） | 64 KB | 300 | 19.66 MB | 85.023 | **231.24** |
| `basic-bw 65536 300`（SSBUF） | 64 KB | 300 | 19.66 MB | 85.024 | **231.24** |

（2026-09-27 同批档给出的 64KB 读数是 230.44 / 231.20 / 231.31 GB/s（log 内 `GBps=231.3144`），与本批差 <0.5%，
两批的逐档 log 可对比；**本报告引用的数字统一用上表这一批**。）

读法：
- 三条 64KB 读数（C API / 基础 API 两构建）落在 **230.75–231.24 GB/s**，**差 <0.3%** ⇒ 两条 API 走的是同一条通道。
- 小包有固定开销：1KB→146.32、8KB→212.34、64KB→230.75 GB/s。**观测**是"吞吐随单次长度上升并趋向 ~231 GB/s"；
  三档不足以把 `t = a + b·bytes` 的两个系数定下来，所以**本报告不给"每次固定开销 = 多少 ns"这种定量主张**。
- **自检**：host 墙钟（0.45–0.50 ms，含 launch）> device 计时（85 µs），差值 ≈0.37–0.41 ms 稳定，
  与"launch/同步开销占大头"一致 ⇒ 1GHz 换算没有明显错误。**host 墙钟不参与带宽数值本身。**

**给 B1–B5 的用法**：要把 AIV 产出的操作数喂 mmad，一次搬 64KB 时**每核可得 ~231 GB/s**；
若你只要搬一个 32×64 的 half 操作数（4KB），**按本表线性内插（推算，不是实测）**落在 150–200 GB/s 量级、单次约 20–27 µs 的**固定开销主导**区，
**不要按 231 GB/s 去估小块延迟**。

### 6.2 数据真的落到 L1（不是"搬进去没人用"）

见 §5.2：AIC 把它当 mmad 操作数真做了一次收缩，1536 个元素全在 [0,1e-6) 内且独立复核通过。

### 6.3 同步语义：该用哪种 CrossCore 配对

**硬规矩（人类逐字）**：成对同步才用 mode 2；GM/非成对中介用 mode 0。
本场景是**一个 AIC 与它那两个 AIV**之间的交接 ⇒ **成对 ⇒ mode 2**，且实例是"1 个生产者 AIV"。

**代码级证据**（本机 9.1.0 头文件逐字）：
`asc_sync_block_arrive(pipe, flag)` → `ffts_cross_core_sync(pipe, GetfftsConfig(flag))`，而
`GetfftsConfig` 里 `uint16_t mode = 0x02;` ⇒ **`asc_sync_block_arrive` 就是 CrossCore mode 2 的 set**；
`asc_sync_block_wait(pipe, flag)` → `wait_flag_dev(flag_id)`（mode 2 的 wait）。
基础 API 一侧的等价物就是 `CrossCoreSetFlag<0x2, PIPE_MTE3>` / `CrossCoreWaitFlag<0x2, PIPE_MTE1>`（仓内 m0 的既有用法）。
官方 C API 样例还额外用了 `asc_sync_intra_arrive/wait`（`set_intra_block`/`wait_intra_block`）做**生产者 AIV→AIC 的点对点**交接。

### 6.4 设备侧见证

**本批（2026-10-04 用当前提交的源码重采）**：

| 档 | 变体 | 结果 |
|---|---|---|
| `capi-rt 0`（默认构建） | 官方口径：block(mode2) + intra | **`BYTE_EXACT`**（`same=8192 diff=0`，log 内 `hostWall=1.895 ms`） |
| `capi-rt 0`（SSBUF 构建） | 同上 | **`BYTE_EXACT`**（`same=8192 diff=0`） |
| `capi-rt 1` × 4（`capi_rt_k1_r1..r4`） | **AIC 两侧 wait 全去掉** | **4/4 `MISMATCH`**（`same` 远小于 8192） |
| `capi-rt 1` × 1（`capi_rt_k1_nowait`，08:48:40） | 同上 | **1/1 `BYTE_EXACT`**（这一档是竞态，恰好读对） |
| `basic-mmad 1` × 2（`basic_mmad_k1_r1/r2`） | 同上（基础 API 一侧） | **2/2 结果错**：`maxAbs=6.500610e+05`（`exact=0/1536`）与 `maxAbs=2.110441e+02`（`exact=879/1536`） |
| `capi-rt 2` | **AIV 不发 block_arrive** | **rc=124 挂死** |
| `basic-mmad 2` | 同上 | **rc=124 挂死** |
| `capi-rt 5` | **只有生产者那一半 AIV 发 block_arrive** | **rc=124 挂死** |
| `basic-mmad 5` | 同上 | **rc=124 挂死** |

**`kill=1` 的样本逐条列全**（这是竞态档，所以计数和批次归属要说准）：

| 样本 log | 批次（`at=`） | 结果 |
|---|---|---|
| `capi_rt_k1_r1..r4` | 2026-10-04 | `MISMATCH` ×4 |
| `capi_rt_k1_nowait` | 2026-10-04 08:48:40 | `BYTE_EXACT` ×1 |
| `basic_mmad_k1_r1` / `_r2` | 2026-10-04 | 错 ×2（`maxAbs=6.500610e+05` / `2.110441e+02`） |
| `basic_mmad_k1_nowait` | 2026-09-27 | 读对 ×1（`maxAbs=6.183982e-07`） |

⇒ **分支里可核对的 `kill=1` 样本共 8 次：6 次读到坏数据、2 次恰好读对。**
（另有一次 09-27 的 `capi_rt_k1_nowait = BYTE_EXACT` 只存在于 git 历史 `21fb0aa`，
其工作树同名 log 已被 10-04 的运行覆盖 —— 那次不计入上表。）

**几条合起来的结论（说清限度）**：

1. **AIC 侧那个 wait 不是装饰，AIV 侧那个 set 不能省**：`kill=2`（AIV 不发）两批各档都出现挂死；
   `kill=1`（AIC 不等）在 8 次可核对样本里 6 次读到坏数据。
2. **`kill=1` 是竞态，单次采样不足以定性**：同一批（10-04）里就既有 4 次 `MISMATCH`、
   也有 1 次恰好读对，09-27 还各有 1 次读对（basic 一侧）⇒ 本报告主张的是
   "**不等就至少有时会读错**"（8 次里 6 次读错），**不主张**"不等必然读错"。
3. **mode 2 要求配对的两个 AIV 都到**：只让生产者那一半发 `block_arrive`（`kill=5`），
   两支各 1 次都挂死。**对 B1–B5 是个真实的坑**：`CrossCoreSetFlag<0x2, PIPE_MTE3>` 不宜放进
   `if (is_producer)` 里 —— 否则 AIC 可能永远等不齐。
4. `kill=0` 是官方口径；`kill=3`（只 block）与 `kill=4`（只 intra）在**单生产者 AIV** 场景下
   读到的都是 `BYTE_EXACT`（`capi_rt_k3_blockonly` / `capi_rt_k4_intraonly`，均为 10-04）：
   两条会合路各自都能把数据交对，差别在"哪一对是必需"——见第 1、3 条的负向对照。

**批次归属**：**多数档**的 log 内有 `### <name> at=<时间戳>` 自证批次；09-27 那批是旧格式（无 `at=`，
   以首行的 `[PUB2L1] tf=Sep 27 2026` 与文件本身为准，例如 `basic_mmad_k1_nowait.log`）。
   同名 log 被后一次运行覆盖时，以 log 内时间戳/`tf=` 为准。**09-27 与 10-04 两批的逐档读数都留在 `evidence/logs/`（除上面注明被覆盖的那一条）。**

---

## 7. 版本差（任务⑤：9.1.0 vs 官方声明的 ≥9.2.0）

官方样例 README 逐字：`| Ascend 950PR/Ascend 950DT | >= CANN 9.2.0 |`；本机 `version.info` = **9.1.0**。
逐字诊断全部在 `evidence/logs/version_diff.log`（脚本 `evidence/version_diff.sh`，**只编不跑**）。

**官方【基础 API】样例 scenario 1**（6 处，rc=1）：

```
data_copy_ub2l1.asc:83:46: error: no member named 'ceil_div' in namespace 'AscendC::Std'
        loadDataParams.mStep = AscendC::Std::ceil_div(m, 16);
                               ~~~~~~~~~~~~~~^
...（84/85/86/95/96 同形）
6 errors generated.
```

本机 `utils/std/cmath.h` 只提供 `Std::ceil_division` / `ceil_align`，**没有 `ceil_div`**。

**官方【C API】样例 scenario 1**：先卡在它自己不 include `kernel_operator.h`（`ASCENDC_HOST_AICORE` 未定义），
再卡同一批 `ceil_div`；此外还用到 9.1.0 **不存在**的枚举包装 `asc_unit_flag_mode` / `asc_store_l2_cache_mode` /
`asc_relu_pre_mode`（底层枚举在编译器自带头 `cce_aicore_intrinsics.h` 里，所以**传裸值等价**），
以及 `asc_set_l0c_copy_nz_para`（9.1.0 的对应物是 `asc_set_l0c2gm_nz2nd`）。

**最小绕开探针**：把 `AscendC::Std::ceil_div(x,y)` 换成等价的本地 `constexpr`（`(a+b-1)/b`），
基础 API 样例**其余一字不改即编过（rc=0）**；本探针源码本体也是 `rc=0`。

**绕开是否影响结论外推？** —— 分清两件事：

1. 关于「**硬通道在本机能不能用**」：**不受影响**。本探针真机跑的那一行是 9.1.0 头里**本来就有**的
   `asc_copy_ub2l1` / 基础 API `DataCopy`，写法与官方样例的对应行同形；绕开只动了 L0/L0C 那几行的**参数写法**
   （本地 ceil+裸值），与 UB→L1 通道本身无关。
2. 关于「**官方样例能不能在 9.1.0 跑通**」：**不能由此推断**。我们改了源码才编过，
   所以本报告不主张官方样例可用 —— 只主张"接口与通道可用"。

---

## 8. 一处未归因、未复现的事件（如实记录，不作为证据）

**首轮**（另一版二进制，10:11 的那一批）**所有 `capi-*` 档**都以
`ACL_ERROR_RT_AICORE_EXCEPTION (507015)` 失败（同批 `basic-*` 档正常）。runtime 逐字日志：

```
[ERROR] ... PrintTaskErrorMsg:Task run failed, stream_id=61, pos=0, task_sn=2, sqe_type=0(aic), errType=0x1(task exception)
[ERROR] ... ProcessDavidStarsCoreErrorInfo:... there is an aicore error exception, core id is 0, error code = 0,
        dump info: pc start: 0x120041000000, current: 0x1200410000cc, ... l1 error info: 0x29700001810 ...
        The extend info: errcode:(0) errorStr: timeout or trap error. subErrType: 0x4.
[ERROR] ... there is an aivec error exception, core id = 0, error code = 263, ...
        errorStr: The address for scalar to use is unaligned or out of bounds
                  The GM address exceeds 48 bits, or the on-chip buffer address exceeds the size of the buffer.
[DFX_INFO]AI Core kernel execution failed ... fault kernel_name=_Z11capi_kernelPDhS_PfPhS1_Pliijj
```

然后我加了一个**只影响新档**的代码块（二分定位用的 `capi-step`）并重建，**同一批指令在下一轮通过了**；
为了归因，我在新构建里跑 9 步累积二分。**这批读数的批次要说准**（初稿这里写过"9 步每一步都 `NO_FAULT`"，
但按当时的分支状态那句话不成立 —— 见 §B）：
`capi_step5/6/7/8` 是 2026-10-04 首轮的 `VERDICT=NO_FAULT`；
`capi_step1/2/9/3/4` 在同一批里因**等锁超时、命令没跑**（`rc=124`、log 无输出）而没有读数，
后来用改版的 `slot.sh`（`timeout` 放进锁内）**重采**才拿到 `NO_FAULT`。
⇒ **现在 9 步都有 `NO_FAULT` 读数**，但它们是同一天的两小批拼起来的，不是"一次跑完 9 步"。

⇒ 正确处置是：**记录现象、说明未能复现、不归因**。本报告**不**把它当作"芯片缺陷"，也**不**把它当作通道能力的证据；
首轮那个二进制未保留（源码随后被改动），所以无法再取舍。

**上面这段逐字行在哪**：原始件是 CANN 的 plog（仓外，`/root/ascend/log/debug/plog/plog-466880_20260927101105492.log` 与
`plog-466944_20260927101107710.log`）；为便于复核，已把相关行**逐字抄录入库**到
**`evidence/logs/m121_aicore_exception_507015.plog.txt`**（文件头写了来源路径、抄录命令与"未改字"的声明）。
**给后续的提示**：若再遇到 `507015` + AIV `code=263`，先看 `~/ascend/log/debug/plog/plog-<pid>_*.log`。

---

## 9. 目录与交付物

```
probe_ub2l1/
├── probe_ub2l1.asc          两个 mix 核 + host 侧（档位/参考/误差分布/落盘）
├── CMakeLists.txt           同一源码编两个目标（默认 / SSBUF）
├── run_probes.sh            一键复现：构建 → 逐条短命令取设备槽（可反复重跑补齐）→ check_ref.py
│                            M121_ONLY=all|core|none 选集，M121_WAIT 控有界等待（≤300，默认 0）
├── check_ref.py             独立判据（numpy：尺寸/有限性/误差分布/往返逐字节）
├── README.md                本文
├── .gitignore               只忽略构建产物与临时文件；**evidence/ 是要交付的，不忽略**
└── evidence/
    ├── version_diff.sh      host 侧版本差取证（零设备；rc 取 bisheng 自己的退出码）
    ├── slot.sh              取一个设备槽跑**一条**短命令（flock -n 探锁 → 有界 flock -w → **timeout 在锁内**）
    ├── retake.sh            锁紧俏时逐条补读数（witness=同步见证重复采样 / extra=消费者+带宽 / steps=二分 9 步）
    ├── logs/                每档原始 stdout（逐字诊断在这里）+ commands.txt（源码 sha256 内容寻址）
    │                        + device_batch.log（run_probes 那批每档一行 rc=）
    │                        + attempts.log（slot.sh 只追加的尝试流水：时间/档名/结果）
    │                        + retake_attempts.log（本轮补读数时"未取得读数"尝试的逐字转录）
    └── dumps/               各档落盘的 a/b/c.bin 与往返 rt_in/rt_out.bin（check_ref.py 的输入）
```

**设备纪律（本 README 的读数是在这条纪律下取的）**：一次 `flock` 只跑一条短命令（≤120s）、`flock -w ≤300`、
**进锁前先 `flock -n` 探锁**、**`timeout` 放在锁内**（这样 `rc=124` 只表示"命令自己超时"= 挂死，
不会把"等锁超时"混进来）、单进程、进锁后 `npu-smi` 复查并把结果原样记进该档 log。
等锁没拿到记 `LOCK_ACQUIRE_FAILED` —— 按塔的口径记为「**未取得读数**」（不是「未复现」）。
`run_probes.sh` 因此**可反复重跑**补齐被跳过的档。

---

## 10. 给 B1–B5 的接口建议（可直接照办）

1. AIV 产出的 mmad 操作数 **可以直接走 UB→L1**（两条 API 任选，读数一致）；不需要 GM 往返，也不需要为 GM 中介重排资源窗。
2. 操作数通路**抽成可替换的一环**（换 `asc_copy_ub2l1` / `DataCopy(L1,UB)` 即切换），别把某一条写进结构假设。
3. 交接同步用 **mode 2 成对**：AIV 侧 `CrossCoreSetFlag<0x2, PIPE_MTE3>`（或 `asc_sync_block_arrive`），
   AIC 侧 `CrossCoreWaitFlag<0x2, PIPE_MTE1>`（或 `asc_sync_block_wait`）；**AIV 侧那个 set 不能省（省了 AIC 直接挂）**。
4. ND→NZ **在 UB 内自己做**（逐 C0 列块重排再连续搬运），别指望随路 ND2NZ。
5. 带宽按 **每核 ~231 GB/s @64KB/次** 折算自己的搬运窗口；小块请按"固定开销主导"估，别用 231 GB/s 外推。
6. **L0C 出参回写**：C API 路径用 `asc_copy_l0c2gm` + **先** `asc_set_l0c2gm_nz2nd(1,2,M*N)`（本探针实测正确，§5.4）；
   基础 API 路径用 `Fixpipe`（仓内既有口径）。
   两条硬规矩（本探针踩过）：① 同一块 L0C 的"写"与"读"用**同一个显式引用**（别一边声明数组、
   一边按"偏移 0"建 `LocalTensor`）；② MTE1→M、M→FIX 之间用**显式事件配对**，别用 `asc_sync_pipe` 顶替。

---

## A. 三层证据索引（官方文档层 / 本机头文件层 / 编译与真机层）

外部快照的取值（读的人自行核对）：CANN = `/usr/local/Ascend/cann-9.1.0`（`version.info` = 9.1.0）；
asc-devkit = `/workspace/asc-devkit`，rev `648a6018207d75af44c6865f96511bafadd90630`。

### 第 1 层：官方文档（asc-devkit 的 SDD 与样例）

| 文档:行 | 逐字要点 |
|---|---|
| `docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md:38-53` | 「新增UB到L1 Buffer搬运数据通路」；**:46** 「对于C API，使用 `asc_copy_ub2l1` 接口实现UB到L1 Buffer数据搬运，**无需配置编译选项**」；**:47-49** 基础 API「开启该特性需要配置编译选项 `ENABLE_CV_COMM_VIA_SSBUF`：开启后…通过硬件通道直接搬运…**未开启该编译选项时，数据需经由GM搬运至L1 Buffer。在此场景下，UB到L1 Buffer搬运接口采用软件仿真实现**」 |
| `docs/zh/api/SIMD-API/basic_api/.../DataCopy_UBToL1_continuous.md:37` | 「本接口为软件仿真实现，是在Matmul高阶API的基础上，利用…workspace GM空间作为数据中转空间…因此，在使用本接口时，需要先使用 `REGISTER_MATMUL` 注册高阶API」 |
| 同上 `:112` | 「…可以通过配置编译选项 `ENABLE_CV_COMM_VIA_SSBUF` 来选择两种搬运通路，**当…为 true 时，使用 SSBuffer 进行通信，数据通过 UB->L1 Buffer 之间的硬件通道进行搬运（推荐）**；当…为 false 时，数据搬运到 L1 Buffer 经过 GM…需要借助Matmul高阶API进行注册操作」 |
| `docs/zh/api/SIMD-API/basic_api/.../DataCopy_UBToL1_ND2NZ.md:37,116` | 该接口为**软件仿真实现**（先转分形再搬） |
| `docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2l1.md` | 产品支持：Ascend 950PR&950DT「**支持**」（A2/A3 不支持）；**流水 PIPE_MTE3**；`size ∈ [32, 4095×32]` 且为 32 的整数倍；`dst`/`src` 需 32B 对齐；**本接口仅在 AIV 上生效** |
| `examples/.../data_copy_ub2l1/README.md:11` | 「Ascend 950PR/Ascend 950DT | **>= CANN 9.2.0**」 |
| 同上「关于场景2实现方案的说明」 | 「`DataCopy(dst, src, Nd2NzParams)` 接口为软件仿真实现，**硬件本身不支持该能力**」⇒ ND→NZ 要在 UB 内自己做 |

### 第 2 层：本机 CANN 9.1.0 的头文件/实现（逐字）

| 文件:行 | 逐字要点 |
|---|---|
| `x86_64-linux/asc/include/c_api/vector_datamove/vector_datamove.h:468-474` | `asc_copy_ub2l1(dst, src, size)` 与 6 参高维切分版、`_sync` 版 |
| `.../impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h:30-31` | `if ASC_IS_AIV { copy_ubuf_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap); }` ⇒ **直下硬件 intrinsic，不碰 TPipe/KFC/Matmul** |
| `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:1031` | `copy_ubuf_to_cbuf` 的 clang builtin 别名（硬件指令本体） |
| `.../impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:582` | `#if KFC_C310_SSBUF == 1 \|\| __MIX_CORE_AIC_RATION__ != 1` → `CopyUbufToCbuf(...)`（硬通道）；`#else` → `ScmDataCopyMsg` / `GetKfcClient()->AllocUB`（GM + 注册） |
| `.../impl/basic_api/kernel_utils.h:36-39` | `#if ENABLE_CV_COMM_VIA_SSBUF != 0 && __MIX_CORE_AIC_RATION__ != 1` → `KFC_C310_SSBUF 1` / `else 0` |
| `.../impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h`（同族） | UB→UB：`asc_copy_ub2ub` 有 3510 实现（ND→NZ 在 UB 内重排要用它） |
| `.../impl/c_api/instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12ub_impl.h:27-31` | `if ASC_IS_AIC { copy_cbuf_to_ubuf(...) }` ⇒ L1→UB 是 **AIC** 侧指令（往返档用它读回 L1） |
| `.../c_api/sync/sync.h:44,56,62,64` + `.../npu_arch_3510/sync_impl/asc_sync_block_arrive_impl.h:32` | `asc_sync_block_arrive` → `ffts_cross_core_sync(pipe, cfg)`，其中 **`uint16_t mode = 0x02`** ⇒ 它就是 **CrossCore mode 2 的 set**；`asc_sync_block_wait` → `wait_flag_dev`；`asc_sync_intra_arrive/wait` → `set_intra_block/wait_intra_block` |
| `.../npu_arch_3510/sync_impl/asc_sync_inter_arrive_impl.h:32` | `asc_sync_inter_arrive` 用的是 `mode = 0x00`（mode 0），与上一条对照 ⇒ 同一文件里两种 mode 的区分是明确的 |
| `.../include/c_api/sys_var/sys_var.h:63,75` + `.../sys_var_impl/asc_get_phy_buf_addr_impl.h:23-26` | `asc_get_phy_buf_addr(off)` 实现是 `return get_imm(off);` ⇒ 返回的是**立即数偏移**，由指针类型（`__ubuf__`/`__cbuf__`）决定落在哪个存储空间的基址上 |
| `.../include/utils/base/sys_constants.h:81-83` | 3510：`ASC_UB_SIZE = 248*1024 + …`、`ASC_L1_SIZE = 512*1024`（带宽档的 L1 目的槽按这个上限算槽数） |
| `.../include/utils/std/cmath.h:23-27` | `AscendC::Std` 只前向声明了 `sqrt`/`abs`，实际 impl 目录里**没有 `ceil_div`**（9.2 才有的符号） |
| `.../include/c_api/cube_compute/cube_compute.h:131` | `asc_set_l0c2gm_nz2nd(nd_num, src_nd_stride, dst_nd_stride)`（9.1.0 里对应官方样例的 `asc_set_l0c_copy_nz_para`） |

### 第 3 层：编译与真机（本探针的读数）

全部在 `evidence/logs/`（逐字）与 `evidence/dumps/`（原始二进制）；
**§0 的每条结论后面都标了出处小节**，§3–§8 给逐字输出。三层里前两层是**别人写的文档/头文件**，
第三层才是**我们自己的读数** —— 本报告凡下结论都落在第三层（或明确标注是哪一层）。

---

## B. 读数批次与「未取得读数」的交代（r1 复审后重写）

**设备环境**：11 个 agent 抢同一把 `/tmp/npu0.lock`，锁多数时间被别的档持有。按塔 2026-10-04 收紧版纪律：
一次 `flock` 只跑一条短命令（`-w ≤300`）、进锁前先 `flock -n` 探锁、**`timeout` 放在锁内**、进锁后 `npu-smi` 复查。
等锁没拿到就记 `LOCK_ACQUIRE_FAILED` —— 按塔的口径（`docs/17 §9.8`）这记「**未取得读数**」，
**不是**「未复现」，也**不是**「挂死」（后者是 `timeout` 杀掉命令、`rc=124`）。
`evidence/slot.sh` 会把这条结论**写进该档的 log 本身**，免得后人把空 log 误读成"跑过/挂死"。

### B.1 一条必须说清的更正（r1 复审 P1-1）

初稿的 §B/§8 写过"`capi-step1..4,8,9` 的 log 仍是 09-27 的、证据本身没丢"、"9 步累积二分每一步都 `NO_FAULT`"。
**这两句按复审时的分支状态都不成立**，已改：

- 09-27（`21fb0aa`）那批 `capi_step1/2/9/3/4` 确有 `VERDICT=NO_FAULT` 的有效读数；
  但 10-04 首轮用**旧版 `slot.sh`**（`timeout 40 flock -w 300`，`timeout` 在 **flock 外面**）重跑时，
  这 5 档是**等锁超时、命令根本没跑**（`rc=124`、log 无输出），而 `slot.sh` 的 `> "$LOG"` 把 09-27 那份
  **工作树副本覆盖**了 ⇒ 分支里当时只有 `capi_step5/6/7/8` 有 `NO_FAULT`。
  （09-27 的那 5 份仍可从 git 历史 `21fb0aa` 取回，但它**不在分支的工作树里**，不能拿它当"入库证据"。）
- 更严重的口径错误是：把"等锁超时（`rc=124`）"与"挂死（`rc=124`）"混为一谈 —— 二者同码不同因。
  现已在 `slot.sh` 里把 `timeout` 放进锁内，**`rc=124` 从此只表示"命令自己超时未返回"**；
  等锁失败记 `LOCK_ACQUIRE_FAILED`。
- 重采结果见 B.3；`capi_rt_k3_blockonly` / `capi_rt_k4_intraonly` 初稿也误标成"仍是 09-27"，
  实际它们的入库 log 是 **10-04、`rc=0`、`BYTE_EXACT`**，已更正。

### B.2 `kill=1` 的计数（r1 复审 P1-2）

初稿写"本批 6/6 读错"、把唯一"读对"归给 09-27 的 `capi-rt`，都不准。
逐条样本与批次归属见 **§6.4** 的小表：**分支里可核对 8 次 = 6 错 2 对**
（读对的两次是 10-04 的 `capi_rt_k1_nowait` 与 09-27 的 `basic_mmad_k1_nowait`）。
结论方向不变（"不等就至少有时会读错"），限度按证据写。

### B.3 批次归属一览（多数档 log 内 `### <name> at=<时间戳>` 自证；09-27 旧格式档以 `tf=` 为准）

| 档 | 入库 log 的批次 | 结果 |
|---|---|---|
| `capi_rt_k0_default` / `capi_rt_k0_ssbuf` / `capi_rt_k1_r1..r4` / `capi_rt_k1_nowait` / `capi_rt_k2_noset` / `capi_rt_k3_blockonly` / `capi_rt_k4_intraonly` / `capi_rt_k5_halfarrive` | **2026-10-04** | 见 §6.4 表 |
| `basic_mmad_k0_default` / `_ssbuf` / `_k1_r1` / `_k1_r2` / `_k2_noset` / `_k5_halfarrive` | **2026-10-04** | 见 §6.4 表 |
| `basic_mmad_k1_nowait` | **2026-09-27** | 读对 ×1（`maxAbs=6.183982e-07`） |
| `capi_mmad_k0_opt{0,1,2}` + `capi_mmad_{a,b}_opt{0,1,2}` | **2026-10-04** | 见 §5.4 表 |
| `capi_bw_1k_2k` / `_8k_1k` / `_64k_300` / `basic_bw_64k_300_default` / `_ssbuf` | **2026-10-04** | 见 §6.1 表 |
| `capi_step1/2/3/4/9` | 见 B.4（本轮重采） | 见 B.4 |
| `capi_step5/6/7/8` | **2026-10-04**（首轮） | 全部 `VERDICT=NO_FAULT` |

### B.4 `capi-step` 9 步二分的读数状态

本轮（r1 复审后）用改版 `slot.sh` 重采了 `capi_step1/2/3/4/9`，避免"9 步全部 `NO_FAULT`"这句话
在分支状态上不成立。逐档结果（"重采前未取得读数的尝试"一列的口径见下方说明）：

| 步 | 入库 log（`at=`） | 结果 | 重采前未取得读数的尝试 |
|---|---|---|---|
| `capi_step1` | 2026-10-04 08:58:49 | `rc=0` `VERDICT=NO_FAULT` | 0 次 |
| `capi_step2` | 2026-10-04 08:58:57 | `rc=0` `VERDICT=NO_FAULT` | 0 次 |
| `capi_step3` | 2026-10-04 09:06:40 | `rc=0` `VERDICT=NO_FAULT` | 1 次（08:59 那次 `-w 120` 未拿到锁） |
| `capi_step4` | 2026-10-04 09:01:13 | `rc=0` `VERDICT=NO_FAULT` | 0 次 |
| `capi_step9` | 2026-10-04 09:10:33 | `rc=0` `VERDICT=NO_FAULT` | 2 次（`-w 120` 与 `-w 180` 各一次未拿到锁） |
| `capi_rt_k4_intraonly` | 2026-10-04 09:10:41 | `rc=0` `VERDICT=BYTE_EXACT` | 1 次（`-w 120` 未拿到锁） |
| `capi_step5/6/7/8` | 2026-10-04 08:16–08:17（首轮） | `rc=0` `VERDICT=NO_FAULT` | 0 次 |

⇒ **9 步现在都有 `NO_FAULT` 读数**，但它们是同一天两小批拼起来的（step5..8 首轮、step1..4,9 本轮重采），
**不是一次跑完 9 步**。重采用的就是改版后的 `slot.sh`（`timeout` 在锁内），命令见 `evidence/retake.sh steps`。

**关于"未取得读数的尝试"的记录在哪（r2 复审 P2-2 的更正）**：这些尝试**不在** `device_batch.log` 里 ——
那个文件只收 `run_probes.sh` 驱动的那批；本轮的补读数走的是 `retake.sh` / 临时驱动 + `slot.sh`。
而每档 log 是 `>` 截断写的，所以同一档"先失败、后成功"时失败那次会被覆盖。因此：
- 本轮 4 次"未取得读数"的尝试（`capi_step3`×1、`capi_step9`×2、`capi_rt_k4_intraonly`×1）
  逐字转录在 **`evidence/logs/retake_attempts.log`**（表头写清它是转录、不是 `slot.sh` 写的）；
- 从本轮起 `slot.sh` 另建**只追加**的 **`evidence/logs/attempts.log`**（每次调用记一行：时间/档名/结果），
  这类"失败尝试被覆盖"的问题以后不会再出现。

**没有"完全没取到"的档**：本 mission 结论引用的每一档都至少有一个批次的有效读数。
`run_probes.sh` 驱动那批里因锁被占而未执行的尝试记在 `evidence/logs/device_batch.log`
（`rc=1 … LOCK_ACQUIRE_FAILED`）；补读数那批记在 `retake_attempts.log`。

### B.5 内容寻址（哪些读数绑在哪个源码版本上）

`evidence/logs/commands.txt` 是 `run_probes.sh` **最近一次运行时刻**写的源码 sha256（内容寻址）。
本次 r1-fix 对 `.asc` 的改动是**文件头/分支里的注释**与**无参提示的 host `printf`**，
没有改任何设备侧代码路径；改完之后用最终二进制**重采了两个头条读数**，把绑定落实到证据上：

| 档 | log（`at=`） | 结果 | 说明 |
|---|---|---|---|
| `capi_rt_k0_default` | 2026-10-04 09:13:03 | `BYTE_EXACT`（`same=8192 diff=0`） | 用 **r1-fix 后的最终源码**重采 |
| `basic_mmad_k0_default` | 2026-10-04 09:13:52 | `maxAbs=6.183982e-07`、`exact=944/1536` | 同上 |

其余档仍是 r1-fix **之前**那一版 `.asc` 的读数（两版之间只差注释与 host `printf`）。
逐档 log 里的 `[PUB2L1] tf=<日期>` 与 `### <name> at=<时间戳>` 给出"哪份源码 / 哪一刻"的核对线索；
要更紧的绑定，跑一次 `bash run_probes.sh`（它会顺带刷新 `commands.txt` 与 `check_ref.log`）。

`check_ref.log` 里 `capi_mmad_opt1` 那一段 **rc=1 是预期的**（它是负向对照档：关掉 nz2nd、结果本来就错），
其余五段 rc=0 才是判据成立 —— 循环里已按这个口径加了注。
