# M117 / B3 —— 设备端挂死的**排除清单**（负结果核心资产）

> ## ⚠ 后续更新（M189，2026-10-05）：挂死与数值都已收敛
> 本清单记录的是 **M117 挂死期**（`timeout`、无输出文件、根因未定位）的排除史。
> **现状 = 不再挂死（M185）+ 数值已收敛（M189）**：
> **M185** 把跨核 mode 4 → mode 2 后四档跑完（EXIT=0）；**M189** 定位并修复数值根因 ——
> `L1 -> L0B` 的生产侧（MTE1）`Acq/Rls` 被放到 BMM1 的 `ks` / BMM2 的 `nh` 循环之外，
> M 侧两条 `Mmad` 都读到循环最后一次装载。修后四档 **PASS**（`over=0`、`maxratio=0.2962`），
> 负向对照全 FAIL。逐档读数与定位见 **`M189_readings.txt`**。
>
> **下方 14 条（历史）与本现状的关系**：它们都是**挂死期**的排除项（排除的是"挂死原因"）。
> 挂死既已由 M185 消除，这些条目**不再是对当前四档 PASS 的瓶颈**；其中：
>   · **第 12 条（K/V 装载几何多矩阵 vs 单矩阵）**与 **第 13 条（BMM1 的 k 拆分）**
>     在 M189 里被**重新按数值判定**：两种装载形态修同步后**逐位一致**（`M189_readings.txt` §4.1）
>     ⇒ 二者均**不是**失败原因。
>   · 第 1–11、14 条是挂死期的设备读数/静态对账，**按当时口径保留**（未在 M189 重跑）；
>     M189 的根因（L0B 逐半块交接）**不在任何一条里** —— M185 交付时它们也没有覆盖这里。
>   · 本文件的历史读数**一字未改**，只加了这段现状说明。

> # ⚠ 最显眼处的一句话
> **（M117 当时）本档至今没有任何 `m` 档拿到 PASS/FAIL 数值读数。**
> `m = 64 / 16 / 96 / 128` 四个档全部在设备上挂死（`timeout` 到点被 kill、**无任何输出文件**）；
> 三个负向对照（`negmask` / `negshift` / `negstart`）的设备读数、确定性读数、运行时长**同样未取到**。
> 下表是**已用读数（或已登记为静态对账）排除**的候选 —— **根因未定位**，本档按塔裁
> **blocked 收口**（2026-10-04），分支入库**不是**把未验证当完成。

---

## 0. 挂死的稳定签名（所有档共有）

* `aclrtSynchronizeStream` 不返回；`timeout` 到点 kill；**无任何输出文件**
  （`m25_attn_fa_core/evidence/out*/` 下不生成 case 子目录）。
* 最小复现（逐字）：
  ```bash
  cd /workspace/ascend_mega_kernel/.tower/worktrees/wt-117
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cmake -B m25_attn_fa_core/build -S m25_attn_fa_core -DCMAKE_BUILD_TYPE=Release
  cmake --build m25_attn_fa_core/build -j4        # -> [100%] Built target m25_attn_fa_core
  flock -w 300 /tmp/npu0.lock bash -c '
    npu-smi info -t common -i 0 | sed -n "3,5p"
    M25FA_M=64 M25FA_BLOCKS=1 M25FA_OUT=<dir> timeout 25 \
      ./m25_attn_fa_core/build/m25_attn_fa_core core; echo "EXIT=$?"'
  ```
  实测读数：`EXIT=124`（`run_m64_clean.log` 里 `timeout 60` 那次是 `EXIT_C=124`）。
* 探针口径（编译期开关 `M25FA_PROBE`，**默认关闭、现已不在库里**；结论只留日志）定位到两个阻塞点：
  * **AIV：`Acq<PIPE_MTE3>(B_PC)`** —— 同一轮的 `Acq<PIPE_V>(B_PC)` 与 `Rls<PIPE_V>(B_PC)`(drain) 都已完成；
  * **AIC：`(B)` 段 P 装载组里某个 `Acq`**（`Acq<PIPE_MTE2>(B_PL1)` 或 `Acq<PIPE_MTE1>(B_L0|B_PL1)`）——
    **受设备 printf 每核缓冲滞后影响（本次每核 17~20 行即打住），这条只算"疑似"**。
* 范围收窄：`M25FA_BLOCKS=28`（24 个工作项 ⇒ 每块恰好 1 个工作项）**也挂**
  ⇒ 挂死**不需要跨工作项轮转**、发生在**单个工作项内部**、且是**概率性**的
  （`nBlk=1` 时曾连过 2 个工作项才挂；并发块多时命中更快）。

---

## 1. 排除项总表（**14 行**；第三方可逐条复核）

| # | 被排除的候选 | 改动（**一次只动一个变量**） | 读数 | 日志文件 | commit |
|---|---|---|---|---|---|
| 1 | 同 pipe 背靠背复用同一 buffer（M126 §3 的规则） | 逐处加 `PipeBarrier<该 pipe>`：AIV 的 `B_PC`；AIC 的 MTE2 `B_Q/B_K/B_PL1/B_V`、MTE1 `B_L0/B_K/B_PL1/B_V`、M 侧 ks 与 nh 两循环的 `B_L0/B_K/B_V/bC`、FIX 的 `bC` —— 共 **18** 条（`grep -c "PipeBarrier<PIPE" m15_layer_loop/m15_attn_fa_core.h` 在 tip 上 = **18**；
其中 **16 条**由 `aeb5f51` 那一轮加的，另 **2 条**由 `b4c578a` 的 AICDMA 探针（`M25FA_AICDMA` 分支）加的；
未全铺） | **仍挂**：`EXIT_PB=124`，无输出文件 | `run_m64_pb.log` | `aeb5f51` |
| 2 | 收尾 DataCopy 的**形态** | 四种都试：32 行逐行 / 单次带 `dstStride` / 两个 16 行半块 / 一轮 32 行（`UB_PC` 8 KB→16 KB） | 各档**全挂**（`EXIT_1R=124` 等） | `run_m64_1r.log`、`run_m64*.log`、`probe_logs/` | `b15509b`…`aeb5f51` |
| 3 | **跨工作项轮转 / per-iteration 计数耗尽** | `M25FA_BLOCKS=28`（24 项 ⇒ 每块恰 1 项）与 `=24` | **仍挂**：`EXIT28=124`、`EXIT24=124` | `run_m64_28.log` | `aeb5f51` |
| 4 | 全 pipe 排空不够（S/PV 可见性、收尾取令牌前） | AIV 消费 S 前 / 消费 PV 前 / 收尾 `Acq<PIPE_MTE3>` 前各加一条 `PipeBarrier<PIPE_ALL>` | **仍挂**：`EXIT_DIAG28=124`、`EXIT_DIAG1=124` | `run_m64_diag.log` | `aeb5f51` |
| 5 | Fixpipe `dualDstCtl` 的 `mSize` | `FAC_P` 64→32（⇒ `mSize` 64→32、每 AIV 分 16 行），其余一字未改 | **仍挂**：`EXIT_P32_1=124`、`EXIT_P32_28=124` | `run_m64_p32.log` | `2a148a1` |
| 6 | BufferID 封装 API | `GetBuffImpl/ReleaseBuffImpl` → 仓库既有 `GetBufInternal<p,false>` / `RlsBufInternal<p,true>`（= `m15_gdn_layer.h` 的 `BufAcquire/BufRelease`） | 挂死点位**不变**（仍停在 `t=0` 的 `AIV PV_RDY ok` / `AIC PV_RDY set` 之后） | **`probe_logs/run_m64d.log`**（nBlk=4、逐阶段探针）；⚠ 该轮另有一份 `run_m64b.log` **已不在库中**（`git ls-tree -r HEAD` 无此文件）⇒ **本条读数以 `probe_logs/run_m64d.log` 为准** | `b15509b` |
| 7 | 核间旗标「同 id 双向令牌」 | 改成**单写方**（`CC_S_RDY/CC_S_FREE`、`CC_PV_RDY/CC_PV_FREE`）；顺带修掉一处**真竞态**（旧写法下 AIC 可能消费掉自己留下的令牌、在 AIV 读完 `S[q]` 前覆写它） | **仍挂** | **`run_m64_clean.log`**（干净版 `EXIT_C=124`）+ **`probe_logs/run_m64i.log`**（逐工作项探针：`WI 0/1 done`，wi=2 的 `PV_RDY ok` 之后无 `datacopy done`） | `b15509b` |
| 8 | 行状态初值用经典 API | `Duplicate(...)`（经典 API）→ RegBase `FacInitStateVf`（`Duplicate`+`StoreAlign`，`__VEC_SCOPE__`） | 挂死点位**不变** | **`probe_logs/run_m64j.log`** | `b15509b` |
| 9 | 静态 `Acq/Rls` 不平衡 | 逐 `(pipe,id)` 计数脚本（含循环体内的 `bC`；**静态对账、非设备读数**） | **未发现不平衡**（两侧各自配平） | 方法与结果见本文 §9 | —— |
| 10 | **AIC `Fixpipe` 直写对侧 AIV 的 UB**（跨核数据面） | 整条换成「AIC 写 GM（`dualDstCtl=0`，`Fixpipe<..., kFixGm>`）→ CrossCore 通知 → AIV 用 MTE2 读回 UB」（顺带用上原本声明未用的 `B_S0/B_S1/B_PV0/B_PV1`）；**该版更差（多 11 MB fp32 scratch + 多一趟 DMA）且非根因 ⇒ 已回退** | **仍挂**：`EXIT_GM1=124`、`EXIT_GM28=124` | `run_m64_gm.log` | `bec205f` |
| 11 | **`FacSoftmaxTileVf` 掩码阈值基的 scalar 算术**（塔裁从严的那处） | VF 签名改传 `posLo/colBase/colLimit` 三个下标量；VF 内用 `Arange/Adds/Muls` 造负阈值、`Mins` 夹取改成行无关掩码 `lane < colLimit` ⇒ **语义等价、VF 外无算术** | **3/3 挂**（`EXIT=124`，与改前同） | `run_vf64.log` | `63a37b6`（+ 第六轮读数） |
| 12 | **L1→L0 装载几何**（AIC 最后一条打印「P in L0A」所在那一步） | K/V 的 `FacGmToL1Nz`（`ndNum = nb` 的多矩阵形态，**本档自己拼的**）→ 循环 `nb` 次 `ndNum = 1` 的**单矩阵形态**（M101 唯一实证过的形态；目标元素偏移 `256*b`） | **3/3 挂** | `run_geo.log` | `c8b7b66`（宏 `M25FA_GEO_ONEMAT`，默认关、两边代码保留） |
| 13 | **MMAD 的 k 拆分**（BMM1 的 2×128） | BMM1 由「2×128 两次 mmad」改为「**1×256 一次 mmad**」（L0B 一次装 256 列：`mStep=SIN/16`、`kStep=HD/16`、`srcStride=dstStride=SIN/16`；`Mmad(m=FAC_P, n=SIN, k=FAC_HD, init=true)`） | **3/3 挂**：`EXIT=124` × 3 | `run_k1mad.log` | `b4c578a`（宏 `M25FA_K1MAD`，默认关） |
| 14 | **AIC 侧 UB→GM 的 DMA 落盘**（作为 ⓔ 落盘手段） | AIC 目标里做 `GM→UB → PipeBarrier<PIPE_MTE2> → UB→GM`；host 预填 `0x5A`、跑挂后经另一条 stream 读回核对 | **编译面通过**；**落盘面本形态下未见落盘：`wit[4096..5120)` 上 0x5A 字节数 = 0/1024**（详见 §7） | `run_aicdma.log` | `b4c578a`（宏 `M25FA_AICDMA`，默认关） |

> 表 1–8、10–14 都是**设备读数**；**只有第 9 行是静态对账**，已单列其性质，不当作设备读数。

---

## 2. M126（独立探针）的近邻读数 —— **不是本档的读数，转引**

M126（`probe_ub_bufid_tail/**`，塔派）把本档收尾用的交接形态缩成不含算法的最小循环，给出：
* 「`Acq<PIPE_V>(id) → 写 UB → Rls<PIPE_V>(id)`(drain) → `Acq<PIPE_MTE3>(id) → DataCopy` → `Rls<PIPE_MTE3>(id)`」
  在 `N = 1/3/8/16/64` 逐档**不挂死、内容逐元素正确**（含复刻本档「每工作项 3 次同 id 交接」的两档）
  ⇒ **该形态不构成挂死的充分条件**；
* 该交接的**正确性依赖 V 与 MTE3 共用同一个 BufferID**（拆两个 id 时内容错，`c_two_ids`）；
* 去掉 V 侧 release 会永久阻塞（可控挂死对照）；
* **同 pipe 背靠背复用同一 buffer 不保序**（`PIPE_MTE2`，低频间歇，失败签名 = 退回上一次 op）
  —— 本档照抄其规则后仍挂（表第 1 行）。

---

## 3. **尚未被任何读数触及**的维度（无读数 ⇒ **不得**当结论）

1. **BufferID 令牌的逐 `(pipe,id)` 动态计数** —— 见 §6（(b) 未实现，并给出"计数未必是那个工具"的分析）。
2. **跨核 flag 的动态记账** —— 见 §6（通道已验证可读回，但计数落盘未取到）。
3. 其它未被单独动过的量：`FacLoadL0_2D` 的字段组合（表 12 只动了 `FacGmToL1Nz` 的形态）、
   `FacSoftmaxTileVf` 里除阈值基以外的写法、`m=1(posBase=4096,ctx=4097)` 与 `m=4097` 两个验收档本身
   （只跑过 `m=64/16/96/128`，`ctx = m`）。

---

## 4. (a) AIC（cube 核）UB→GM DMA —— **结论写窄**

**做了**（宏 `M25FA_AICDMA`，默认关闭；代码在 `m15_attn_fa_core.h` 的 `FacAic::Run` 开头，
由 `FaCoreBody` 的 `if ASCEND_IS_AIC` 分支调用 ⇒ **只在 AIC 目标编译**）。
host 侧把 P scratch 预填 `0x5A`；跑挂后经另一条 stream 读回 `wit[4096..5120)` 核对。

读数（`run_aicdma.log`，逐字）：
```
[M25FA][WIT] drain copy err=0 sync err=0
[M25FA][WIT] nonzero entries=0
[M25FA][AICDMA] wit[4096..5120) == 0x5A 的字节数 = 0 / 1024
```
* **编译面：通过** —— `DataCopy(UB→GM)` 在 **AIC 目标上能编译**。
* **落盘面：本最小形态下未见落盘**（0/1024 字节）。
* ⚠ **本档只核对了终点，没有分别核对 UB 侧** ⇒ **是哪一条腿（`GM→UB` 还是 `UB→GM`）没走通，本档未分辨**
  ⇒ **不得**从这条读数推成「AIC 不能做 UB→GM DMA」；只能说「**本形态下未见落盘**」。

---

## 5. (c) MMAD 的 k 拆分 —— **3/3 挂**
见表第 13 行。宏 `M25FA_K1MAD` **默认关闭**，两边代码都保留；CMake 里没有留任何 `M25FA_*` define。

---

## 6. (b) BufferID 逐 `(pipe,id)` 动态计数 —— **未实现**（不是"读不回"）

**未做。** 两条原因，都写清：
1. **预算**：本轮在 (a) + (c) 之后已尽，没有余量实现完整链路
   （UB 累加 → 写出侧 drain release → MTE3 DMA 落盘 → 挂死中经另一条 stream 读回）。
2. **更重要的一条是分析（不是读数）**：对 `B_PC` 这个 id 而言，**计数未必能定位** ——
   它的"两侧"是**同一个 AIV** 的 `PIPE_V` 与 `PIPE_MTE3`；而 AIV 每工作项对 `B_PC` 的
   `Acq/Rls` 是**无分支直落代码**（tile: `V-acq, V-rls, E3-acq, E3-rls`；tail 同），
   每项 `2+2+2+2`、严格交替 ⇒ **计数只可能显示"除被卡住的那一次 `Acq` 之外全部配平"**，
   即"令牌就是不在那儿"，**并不能说明令牌去了哪**。
   ⇒ 要定位"漏 release / release 未生效"，需要的是**令牌去向**（谁持有），**那需要比计数更强的插桩**。

**"另一条 stream 读回 GM"这条通道本身已被证明可行**（这是本档的一条正面方法读数）：
`M25FA_WITDRAIN=1` ⇒ `[M25FA][WIT] drain copy err=0 sync err=0`（`run_wit.log`）；
它把 kernel 挂死中的 GM 内容读回来了 —— 只是那次读到的计数全是 0（见 §7 的 ⓔ 说明）。

---

## 7. ⓔ（标量落盘）与 flagId 计数：为什么那次读到全 0
第一个计数版本用 `GlobalTensor::SetValue` 在 GM 上自增；读数 = **通道通、`nonzero entries=0`**。
这与 **rule ⓔ 已知的坑**一致：**标量写 GM 不经「UB 写 → 写出侧 drain release → DMA」就落不下去**
（人类原话「本来就不应该用 scalar pipe 写 GM」）。⇒ 该读数**既不能说"配平"也不能说"不配平"**，
本档如实记成「**通道通、计数读回全 0（与 ⓔ 的标量落盘坑一致）**」。

另：**目标修正已被塔采纳** —— 两个已定位阻塞点**都是 BufferID 的 `Acq`、不是 CrossCore flag 的 `Wait`**
⇒ flagId 计数**逻辑上解释不了**它们；与阻塞点对得上的见证是 §6 那个（且 §6 说明了它也不够）。

---

## 8. 探针开关与默认值（**默认惰性**）

| 开关 | 形态 | **默认** | 作用 | 关闭时的行为 |
|---|---|---|---|---|
| `M25FA_WIT` | 编译期宏 | **关** | 把 19 个 cross-core 调用点包进 `FacCcSet/FacCcWait` 并在 GM 上自增计数 | `FacCcSet/FacCcWait` 退化为直接调用原生 `CrossCoreSetFlag/WaitFlag`（同一 pipe、同一 id），**生产路径零开销** |
| `M25FA_AICDMA` | 编译期宏 | **关** | §4 的 AIC 侧 UB→GM 往返探针 | 不编译该探针 |
| `M25FA_K1MAD` | 编译期宏 | **关** | BMM1 改成 1×256 一次 mmad | BMM1 保持 2×128 两次 mmad |
| `M25FA_GEO_ONEMAT` | 编译期宏 | **关** | K/V 装载改成逐矩阵 `ndNum=1` | K/V 装载保持多矩阵 `ndNum=nb` |
| `M25FA_WITDRAIN` | 环境变量 | **关** | 启动后 sleep 6 s、经另一条 stream D2H 读回见证区后 `_exit(0)` | 走正常的 `aclrtSynchronizeStream` |
| `M25FA_M` / `M25FA_BLOCKS` / `M25FA_OUT` | 环境变量 | `4097,1` / 取设备 AIC 数 / `m25_attn_fa_core/out` | 只影响**测试参数**（跑哪些 m、blockDim、落盘目录） | 默认档 |
| `M25FA_REPEAT` | 环境变量 | **5** | 只被 `run_checks.sh` 用：core 档连跑几次比 `sha256`（确定性检查） | 默认 5 次 |

**核过的两条**（在该文件与该 pattern 范围内）：
* `grep -n "M25FA" m25_attn_fa_core/CMakeLists.txt` ⇒ **无命中**（即 CMake 不定义任何 `M25FA_*`）；
* 段体头里 4 个编译期开关全部写成 `#if defined(M25FA_*)` ⇒ **默认关闭**。
* 名字 `PREFILL_WIRED` **不在本档**（那是 Wave A 侧 `LayerArgs::pfWired` 的开关，本档只在
  `evidence/mount_patch.md` 的**提案文本**里引用它，未在本档代码里定义或依赖）。
* **探针构建的两条已知不足**（只在开关打开时才有影响，默认路径不受影响）：
  ① 见证区大小：段体按 `(28 + 2*nBlk) × 32 × 2 × 4 B` 索引（`nBlk=28` 需 **21,504 B**），
     而 host 的 `kWitBytes` 按 **56 槽**（14,336 B）分配 ⇒ **`nBlk > 14` 时会越界**；
  ② `:9` 的 `M25FA_PROBE` 探针曾用于定位（已从库里移除），其读数只留在
     `evidence/probe_logs/` 的日志里。

---

## 9. 第 9 行（静态对账）的方法与结果
把段体里**每一处** `Acq<PIPE_X>(id)` / `Rls<PIPE_X>(id)` 按 `(pipe, id)` 归类计数（含循环体内的 `bC`），
`FacAic` / `FacAiv` 两侧分别对账。结果：**每一对 `Acq/Rls` 在两侧各自的数量都相等，未发现不平衡**。
⚠ 这是**静态**对账（源码里出现的次数），**不等于动态见证**；且它按"每工作项执行一次"的假设成立 ——
`FacAic` 的 `PIPE_M B_K` 等出现在 `ks` 循环体内，静态计数相等只说明"循环次数相同则动态也相等"。

---

## 10. 本档**未取到**的读数（收口时如实列出）
* `m=64` / `m=1(ctx=4097)` / `m=4097` 的 **PASS/FAIL 数值读数** —— 一个都没有；
* 三个负向对照（`negmask` / `negshift` / `negstart`）的设备读数 —— 没有；
* 确定性（连跑多次比 sha256）、运行时长 / 性能 —— 没有；
* BufferID 逐 `(pipe,id)` 动态计数、flagId 动态记账 —— 没有（§6 / §7）。
* **已取到的正面读数只有一条**：离线判据自证（`check_ref_selftest.txt`：
  回灌 fp64 参考 ⇒ `PASS maxratio 0.2203`；×1.05 ⇒ `FAIL over=117926/227328 maxratio 3.0760`）。
