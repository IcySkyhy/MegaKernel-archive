# M141 —— AIC（cube 核）UB→GM 能力的最小可证伪探针（`probe_aic_gm_dma`）

> 独立工程（自带 `CMakeLists.txt` + 含 `main()` 的独立 `.asc`，不碰仓库任何既有 CMakeLists）。
> **只回答一个可证伪问题**：在 dav-3510 上，**AIC（cube 核）能否把数据从 UB 搬到 GM**？
>
> **根因（权威口径，人类逐字纠正，2026-10-04）**：
> 「**AIC 就访问不了 UB，当然不能 UB->GM**」。
> 「**这是L1 to UB，不是直接访问UB**」（人类对人类这句话的**再纠正**）。
> 归一为**分层口径**：**AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**。
> ⚠ **层级不同，不是冲突**：AIC 侧存在 **L1→UB 的 MTE1 pipeline 搬移**（`copy_cbuf_to_ubuf`，`if ASC_IS_AIC`）——
> 那是 **pipeline 传输**，**不是"直接访问 UB"**，故 **不构成对根因的反证**（事实与判定见 **§1.5**）。
> 这是**架构层定性**（人类给的常识），**不是**本档设备读数反推出来的；本档读数与它**一致**（见 §0 / §1）。
> ⚠ 本档早先把根因写成「**编译期 raw MTE3 被拒 + 软件层 AIV 门控成空**」——按本次纠正，
> 那是这条根因的**后果 / 一致性证据**，不是根因；它们**保留**（是读数），但**不再冒充根因**（§1 / §3 已改写）。
>
> 触发（人类对 M136/M134 方向的既有纠正，逐字）：
> 「**AIC写GM用的是FIXP，而不是MTE3，所以你这个同步本来就是错的呀**」。
> M134 的合成探针只证明了 AIC 经 Fixpipe/`dualDstCtl` 直写**对侧 AIV 的 UB**可行；
> M136 的读数同向但未单独隔离「**AIC 侧 UB→GM**」这条腿。本档就补这条腿。

> ⚠ 未定界纪律（同 M126/M134）：本 README 只写**设备读数直接支持**的东西。
> 本档是**合成形状**（固定 64×64 fp32、单 AICore、单次搬运），与业务核的 shape / tiling / 多个核不同 ⇒
> **不外推**到业务核；也不写任何芯片级绝对判断。

---

## 0. 结论速览（TL;DR，每条带判据/出处）

| # | 结论 | 判据（可复现） | 出处 |
|---|---|---|---|
| 1 | **根因（架构层）：AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**（人类逐字：「AIC 就访问不了 UB，当然不能 UB->GM」；「这是L1 to UB，不是直接访问UB」）。本档读数都与它一致；**主证据（最直接的观测证据）** = AIC 标量访问 UB 越界。下面第 2–3 行是**后果 / 一致性证据**；AIC 侧 `L1→UB` 的 MTE1 搬移是 **pipeline 传输**、不是直接访问 UB，**不构成反证**（§1.5） | 人类逐字纠正（文首）；`aic_scalar_ub` 3/3 `launch=FAILED err=507015`；plog `error code = 271` / `errorStr: The address for scalar to access the internal buffer is out of bounds` | §5（主证据）、§1.0、§1.5 |
| 2 | 后果证据①（编译期）：raw MTE3 intrinsic（UB→GM）在 AIC 分支被 bisheng 拒绝 | raw 探针编译 rc=1（逐字报错见 §1.1） | §1.1 |
| 3 | 后果证据②（软件层）：基础 API `DataCopy(GM,UB)` 与 C API `asc_copy_ub2gm` 能编译，但函数体对 AIC 是 `if ASCEND_IS_AIV` 门、被常量折叠成空 ⇒ AIC 上不落盘 | `ub2gm_api` / `ub2gm_capi` 各 3/3 返回、GM 出口 4096/4096 仍是 sentinel、`landed=0` | §1.2、§2 |
| 4 | 对照腿：AIC 写 GM 的正路是 **FIXP（L0C→GM）**（根因的直接推论：AIC 不经 UB，从 L0C 出门） | `fixp_l0c2gm` 3/3 rc=0，GM 出口 4096/4096 落盘、逐元素 == 128.0f | §2、§4 |
| 5 | 正对照：AIV 的 UB→GM（MTE3）可用 —— 该指令与本档 harness 本身没问题，AIC 之所以不通是因为 AIC 不能直接访问 UB（UB→GM 不可路由） | `aiv_ub2gm` 3/3 rc=0，GM 出口 4096/4096 落盘、逐位 == `1.0f + i` | §2 |
| 6 | 负对照有牙：同一 FIXP 路径只改 `nSize`（64→32，故意写半宽）判据变红 | `neg_fixp_short` 3/3 rc=1，`mismatched=2048 / landed=2048`（未写的半宽仍是 sentinel） | §4 |

> 判据一律走 **DMA 落盘 + host 侧逐元素对拍**：device 只把被测腿的结果写进 GM 出口，
> host 读回后与**另写一份**的期望逐元素比；`check_ref.py` 再用 Python 独立重算一遍（见 §8）。
> 本档所有运行档各 3 次、读数一致（3/3）。

---

## 1. 根因与读数怎么对齐（主证据 + 后果证据）

### 1.0 根因（架构层，人类逐字）与主证据

**根因**（人类逐字，分两层）：「**AIC 就访问不了 UB，当然不能 UB->GM**」＋「**这是L1 to UB，不是直接访问UB**」——
即 **AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**。
AIC 侧的 **L1→UB（MTE1）** 是 **pipeline 搬移**、不是"直接访问 UB"，与本条根因**层级不同**（§1.5）。
这是**架构层定性**（人类给的常识），本档不在设备上"复测架构"；本档能做的是给出与它**一致**的读数。

**主证据（根因最直接的观测证据）＝ `aic_scalar_ub`**（§5 详列）：设备上 AIC 对 UB 做**标量**访问，直接以
**aicore 异常**收场——`launch=FAILED err=507015`，plog `error code = 271` /
`errorStr: The address for scalar to access the internal buffer is out of bounds`
（`probe_aic_gm_dma/evidence/logs/aic_scalar_ub.log:18,21,24`；
`probe_aic_gm_dma/evidence/logs/aic_scalar_ub.plog_excerpt.txt:4,5,12`）。
「internal buffer 地址越界」是**地址空间层面**的报错，与「AIC 不能**直接**访问 UB」**逐字同向**；
它把原来的「附带读数」提升为**主证据**。限度：这条只证明**本档这个字节向量下** AIC 标量访问 UB 越界（§7）。

**后果证据（两条一致性读数，原先被误当根因）**：见 §1.1（编译期）、§1.2（软件层）。它们与根因同向、互为旁证，
但**不是根因**——根因是「AIC 不能**直接**访问 UB」，不是「编译器拒了」或「AIV 门拦了」。

**边界（层级不同，不是冲突 —— 见 §1.5）**：本档给 UB 装数据走的是 AIC 侧 `asc_copy_l12ub`（MTE1 `L1→UB`，§1.2）。
它的 AIC 分支**不是空实现**，而是一条真实的 `copy_cbuf_to_ubuf` pipeline 搬移调用（头文件逐字见 §1.5）；该调用在 AIC 上编译被接受。
⇒ 这是 **L1→UB 的 pipeline 传输**、**不是"直接访问 UB"**（人类逐字：「这是L1 to UB，不是直接访问UB」），
故**不构成对根因的反证**；"这条 MTE1 写 UB 的运行期效果未隔离验证"仅作**限度**保留（§7）。

### 1.1 后果证据①：编译期（零设备；`evidence/compile_reject/`）

命令（与 `probe_crosscore_tail/CMakeLists.txt:22-27` 同源的编译参数；逐字见各 `*.log`）：

```
/usr/local/Ascend/cann-9.1.0/bin/bisheng -DNPU_ARCH_DAV_3510 -std=c++17 \
    --npu-arch=dav-3510 -O3 -DNDEBUG --asc-aicore-lang -c -o /tmp/x.o <probe>.asc
```

| 小探针（`evidence/compile_reject/*.asc`） | 腿 | bisheng 结果 | 首条诊断（截断） |
|---|---|---|---|
| `raw_ub2gm_aic.asc` | AIC **UB→GM**（raw `copy_ubuf_to_gm`） | **REJECT (rc=1)** | `function type '...copy_ubuf_to_gm...' does not support the given target feature`（`evidence/compile_reject/raw_ub2gm_aic.log:1`、汇总 `compile_matrix.txt:7`） |
| `raw_ub2l1_aic.asc` | AIC UB→L1（raw `copy_ubuf_to_cbuf`） | **REJECT (rc=1)** | `...copy_ubuf_to_cbuf... does not support the given target feature`（`compile_matrix.txt:8`） |
| `raw_gm2ub_aic.asc` | AIC GM→UB（raw `copy_gm_to_ubuf`） | **REJECT (rc=1)** | `function type '...copy_gm_to_ubuf... does not support the given target feature`（`compile_matrix.txt:9`） |
| `raw_l12ub_aic.asc` | AIC L1→UB（raw `copy_cbuf_to_ubuf`） | **ACCEPT (rc=0)** | ——（AIC 的 UB 写侧归 MTE1；见 §1.0 边界） |
| `api_ub2gm_aic.asc` | AIC UB→GM（基础 API `DataCopy`） | **ACCEPT (rc=0)** | 编译通过 ≠ 能搬：运行期见 §1.3 |

**读法**：`ACCEPT` 只表示「编译通过」。`api_ub2gm_aic.asc` 编译过，但它的函数体在 AIC 上被折叠成空（§1.2）。
上表是**与根因一致的后果证据**：AIC 侧的 UB→GM（MTE3）通道不存在，raw intrinsic 因此在编译期就被拒。

### 1.2 后果证据②：软件层（CANN 9.1.0 头文件，AIC/AIV 分派）

CANN 头文件根：`/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/`（下称 `<asc>/`）。

| 腿 | 实现位置 | AIC 上的行为 |
|---|---|---|
| UB→GM（MTE3） | `<asc>/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:97`（`CopyUbufToGmAlignV2`；`:94` 注明 `only support VecCore PIPE_MTE3`），函数体 `:101 if ASCEND_IS_AIV {` | AIC 上条件为假 ⇒ **空操作** |
| UB→GM（C API） | `<asc>/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2gm_impl.h:27-33`（`:30 if ASC_IS_AIV {`，内层 `:31 copy_ubuf_to_gm_align_v2`） | AIC 上条件为假 ⇒ **空操作** |
| GM→UB（MTE2） | `.../kernel_operator_data_copy_impl.h:54`（`CopyGmToUbufAlignV2`；`:50` 注明 `only support VecCore PIPE_MTE2`），函数体 `:60 if ASCEND_IS_AIV {` | AIC 上条件为假 ⇒ **空操作** |
| L1→UB（MTE1） | `.../kernel_operator_data_copy_impl.h:188`（`CopyCbufToUbuf`；`:185` 注明 `only support CubeCore PIPE_MTE1`），函数体 `:191 if ASCEND_IS_AIC {` | AIC 上执行——这是 **L1→UB 的 pipeline 搬移**、不是"直接访问 UB"，不提供 UB→GM 出口（本档用它给 UB 装已知数据；见 §1.0 边界 / §1.5） |

`ASCEND_IS_AIC` / `ASCEND_IS_AIV` 在设备编译时是 **`constexpr`**（`<asc>/impl/utils/sys_macros.h:79-80`），
所以上面这些 `if ASCEND_IS_AIV {}` 在 AIC 目标上会被**常量折叠成空**——这正是「编译过、运行期不落盘」的机制。
`<asc>/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:59-61`（`SoftSyncAllImpl` 在 AIC 上直接 `return`）是同一模式的旁证。

**与根因的关系（口径归位）**：这些接口只对 **AIV** 定义不是"另加的软件门"，而是因为 **UB 是 AIV 侧的资源**；
AIC **不能直接访问 UB**，UB→GM 这条腿对 AIC 无从路由（§1.0 / §1.5）。⇒ 门控成空是**根因的后果**，不是根因。

### 1.3 后果证据③：运行期（设备读数）

见 §2 变体表。关键：`ub2gm_api` / `ub2gm_capi` 在 AIC 上**返回**（rc=0），但 GM 出口 4096/4096 仍是 sentinel（`landed=0`）。

### 1.4 读数 × 根因 对齐表（带 文件:行）

| 读数（原样保留，未改动） | 出处（文件:行） | 与根因的关系 |
|---|---|---|
| raw MTE3 `copy_ubuf_to_gm`（UB→GM）在 AIC 编译期被拒 | `probe_aic_gm_dma/evidence/compile_reject/raw_ub2gm_aic.log:1,4,5`；`probe_aic_gm_dma/evidence/compile_reject/compile_matrix.txt:7` | **后果证据**：AIC 侧 UB→GM 的 MTE3 通道不存在（与根因同向） |
| 基础 API `DataCopy(GM,UB)` / C API `asc_copy_ub2gm` 在 AIC 上折叠成空 ⇒ `landed=0` | `probe_aic_gm_dma/evidence/logs/ub2gm_api.log:18,21,24`；`probe_aic_gm_dma/evidence/logs/ub2gm_capi.log:18,21,24`；`probe_aic_gm_dma/evidence/logs/matrix.txt:3,4` | **后果证据**：接口只对 AIV 定义（`kernel_operator_data_copy_impl.h:101`、`asc_copy_ub2gm_impl.h:30`），UB 是 AIV 侧资源 |
| AIC **标量**访问 UB → `launch=FAILED err=507015`；plog `error code = 271` / `The address for scalar to access the internal buffer is out of bounds` | `probe_aic_gm_dma/evidence/logs/aic_scalar_ub.log:18,21,24`；`probe_aic_gm_dma/evidence/logs/aic_scalar_ub.plog_excerpt.txt:4,5,12` | **主证据（根因最直接的观测证据）**：地址空间层面的越界，直接印证 AIC 不能**直接**访问 UB |
| AIC FIXP `L0C→GM` 可用 | `probe_aic_gm_dma/evidence/logs/fixp_l0c2gm.log:18,21,24` | 根因的**直接推论**：AIC 不经 UB，从 L0C 出门 |
| AIV `UB→GM`（MTE3）可用 | `probe_aic_gm_dma/evidence/logs/aiv_ub2gm.log:18,21,24` | 根因的**正对照**：指令本身可用，AIC 不通不是因为指令坏 |

> 上表所有 `文件:行` 指向的证据文件、日志与哈希**未改动**（本档只改本 README 与 `evidence/README.md` 的措辞）。

### 1.5 层级定性（人类逐字）：AIC 侧的 `L1→UB` 是 pipeline 搬移，不是"直接访问 UB"

> 目的：把 §1.0 的「边界」写成**人类的分层口径**，并留下头文件事实。已打开已装 CANN 9.1.0 头逐字确认。

**人类逐字（权威口径）**：「**这是L1 to UB，不是直接访问UB**」。

**分层**（本档根因＝第一层）：
- **第一层「直接访问 UB」（本档测到的）**：AIC 侧**标量**访问 UB → `507015` / plog `271`
  「The address for scalar to access the internal buffer is out of bounds」（§5，主证据）；**UB→GM 对 AIC 不可路由**（`landed=0`，§1.3/§2）。
  ⇒ **AIC 不能"直接"访问 UB。**
- **第二层「pipeline 搬移」（不构成反证）**：AIC 侧存在 **L1→UB 的 MTE1 搬移**分支（`copy_cbuf_to_ubuf`，`if ASC_IS_AIC`）——
  那是 **pipeline 传输**、**不是"直接访问 UB"**，故**不与第一层冲突**（层级不同）。

**事实（头文件逐字 + 编译矩阵，保留）**：第二层是**真实的 raw intrinsic 调用**，不是空实现、不是常量折叠：

- C API：`/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12ub_impl.h:25-27`，逐字：
  ```
      if ASC_IS_AIC {
          copy_cbuf_to_ubuf(dst_addr, src_addr, sub_blockid, n_burst, len_burst, src_gap, dst_gap);
      }
  ```
- 基础 API：`/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:191-193`，逐字：
  ```
      if ASCEND_IS_AIC {
          copy_cbuf_to_ubuf((__ubuf__ void*)dst, (__cbuf__ void*)src, static_cast<bool>(subBlockId), blockCount, blockLen, srcStride, dstStride);
      }
  ```
- `copy_cbuf_to_ubuf` 的声明（编译器内建别名）：`/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:963`：
  `__attribute__((clang_builtin_alias(__builtin_cce_copy_cbuf_to_ubuf))) void copy_cbuf_to_ubuf(...);`
- 该调用在 **AIC 目标上编译被接受**（`raw_l12ub_aic`，`evidence/compile_reject/compile_matrix.txt:10`）；框架把它标为 `only support CubeCore PIPE_MTE1`（`kernel_operator_data_copy_impl.h:185`）。
- 对照：UB→GM 的基础 API 是 `if ASCEND_IS_AIV {…}`，在 AIC 上整段折叠成空（§1.2，`…:101`）——两者形态**不同**。

**判定：层级不同，不是冲突**。第二节这条 `L1→UB` 是 **pipeline 搬移**、**不是**对 UB 的"直接访问"，
因此 **不构成对根因的反证**（**不再**写成"冲突 / 未解"）。

**限度（保留，仅此）**：这条 MTE1 写 UB 的**运行期效果**本档**未隔离验证**——
AIC 无法从自己这侧回读 UB（那正是 §5 的主证据 507015 / 271）；
若要坐实需**设备隔离验证**（AIC 用 `asc_copy_l12ub` 写已知数据、**配对 AIV** 读回核对）。

---

## 2. 变体表（每档各 3 次、独立进程、每档各自进一次 `flock`）

证据目录：`evidence/logs/`（每档 `<variant>.log` 含锁内 `npu-smi`、逐次 `SUMMARY`、`exit=`、`AGG`）、
`evidence/logs/matrix.txt`（汇总）、`evidence/logs/check_ref.log`（Python 独立复核）、
`evidence/dumps/out_<variant>_run<N>.bin`（GM 出口 16 KB dump）。

| 变体 | 轴 | 构造 | 3 次读数 | 判定 |
|---|---|---|---|---|
| `fixp_l0c2gm` | 对照腿 | AIC：Mmad（A0=1s,B=1s,K=128）→L0C=128 → FIXP `Fixpipe` L0C→GM | rc0=3；每次 `landed=4096, fixp_hits=4096`（全 128.0f） | **落盘一致** |
| `ub2gm_api` | 被测腿 | AIC：经 L1 把 `inGm` 装进 UB（`asc_copy_gm2l1`+`asc_copy_l12ub`）→ 基础 API `DataCopy(GM, UB)` | rc0=3；每次 `sentinel=4096, landed=0` | **不落盘**（每元素仍为 host 预置 sentinel） |
| `ub2gm_capi` | 被测腿 | 同上，改用 C API `asc_copy_ub2gm` | rc0=3；每次 `sentinel=4096, landed=0` | **不落盘** |
| `aiv_ub2gm` | 正对照 | AIV：`DataCopy(GM→UB)` + `DataCopy(UB→GM)`（含 `Set/WaitFlag<MTE2_MTE3>`） | rc0=3；每次 `landed=4096, pat_hits=4096` | **落盘一致**（harness 与该指令可用） |
| `neg_fixp_short` | 负对照 | 同 `fixp_l0c2gm`，但 FIXP `nSize=32`（只写 32/64 列） | rc1=3；每次 `mismatched=2048, landed=2048` | **判据变红** |
| `aic_scalar_ub` | 隔离（主证据） | AIC **标量**写 UB（`ubF[i]=PatOf(i)`）+ 标量读回 UB→GM | rc2=3；每次 `launch=FAILED err=507015` | **aicore 异常**（plog `error code = 271`） |

注：`rc0` = 返回且与 host 期望逐元素一致；`rc1` = 返回但有差异；`rc2` = launch 失败；`124`（timeout）= 未返回。
`landed` = GM 出口里**非 sentinel** 的元素数（不看 host 期望的纯计数），用来直接读「落没落盘」。

设备只让 **AICore 0 的 AIC** 干活（`probe_aic_gm_dma.asc:194-196` 起）；`aiv_ub2gm` 这一档的 AIV 是**harness/指令正对照**，
其余变体的 AIV 在 `probe_aic_gm_dma.asc:267-273` 立即返回、不参与数据面。

---

## 3. 隔离链（把读数对齐到同一个根因）

```
根因：AIC 不能「直接」访问 UB（人类逐字：「AIC 就访问不了 UB，当然不能 UB->GM」＋「这是L1 to UB，不是直接访问UB」）
        │
        ├─► 后果 A（编译期）：AIC 的 UB→GM MTE3 通道不存在
        │     raw `copy_ubuf_to_gm`（UB→GM）在 AIC 分支被 bisheng 拒绝（§1.1）
        ├─► 后果 B（软件层）：基础 API `DataCopy(GM,UB)` / C API `asc_copy_ub2gm`
        │     对 AIC 折叠成空 ⇒ 运行期不落盘（landed=0，3/3 实测；§1.2/§1.3）
        ├─► 主证据（设备）：AIC 标量访问 UB 直接越界异常 507015 / plog 271（§1.4、§5）
        └─► 直接推论：AIC 写 GM 走 FIXP（L0C→GM），不经 UB（对照腿 3/3；§2/§4）
```

- 三条读数**同向**：它们不是彼此的根因，而是**同一个根因的三处观测**——「AIC 不能直接访问 UB ⇒ UB→GM 不可路由」。
- 「用 FIXP 而非 MTE2/MTE3」这条既有口径在读数上同向：AIC 写 GM 的可用路径是 **FIXP（`fixp_l0c2gm` 3/3）**，
  而 AIC 的 **MTE3（UB→GM）** 与 **MTE2（GM→UB）** 两条 raw 都编译被拒（§1.1）。
- 本档给 UB 装已知数据用的是 AIC 侧的 `asc_copy_gm2l1`（MTE2 GM→L1）+ `asc_copy_l12ub`（MTE1 L1→UB），
  两条都在 `<asc>/impl/c_api/instr_impl/npu_arch_3510/cube_datamove_impl/`（`asc_copy_gm2l1_impl.h:25 if ASC_IS_AIC {`、
  `asc_copy_l12ub_impl.h:25 if ASC_IS_AIC {`）：这是 **L1→UB pipeline 搬移**、不是直接访问 UB、不提供 UB→GM 出口（§1.0 边界 / §1.5）。

---

## 4. 负对照（证明「落盘一致」判据有牙）

- 被测对象：`fixp_l0c2gm` 的 FIXP 落盘路径。
- 注入：**只改 `nSize`（64→32）**，即故意只写 32/64 列（`probe_aic_gm_dma.asc:236-238` 的 `nSize` 分支）。
- 期望：host 判据仍按全宽（64 列）期望 ⇒ 未写的半宽应保持 sentinel。
- 实测（3/3）：`neg_fixp_short` = `mismatched=2048, sentinel=2048, landed=2048, fixp_hits=2048, rc=1`
  （`evidence/logs/neg_fixp_short.log:18,21,24`）；
  而同一路径的 `fixp_l0c2gm` = `mismatched=0, landed=4096, rc=0`。
- **签名**（可核对）：`SUMMARY variant=neg_fixp_short ... mismatched=2048 sentinel=2048 landed=2048 ... rc=1`
  （`evidence/logs/neg_fixp_short.log`）。同一判据在同一注入下由绿转红 ⇒ 判据确实盯着「GM 出口字节」，
  不是空洞通过。
- 另外，「预期不落盘」那两档用的也是同一条判据：它们的 `landed=0` 是**被数出来的非 sentinel 元素数为 0**，
  不是「没测」。

---

## 5. 主证据：AIC 标量访问 UB 越界（`aic_scalar_ub`）

> 本节是**根因（§1.0）最直接的观测证据**——原来记为「附带读数」，现提升为**主证据**。

- 动机：本档想从 AIC 侧回读 UB，以独立证明「UB 里确已有确定数据」。AIC 没有 UB→X 的 DMA 通路（§1、§2），
  于是试了 AIC **标量**读写 UB（`probe_aic_gm_dma.asc:248-260`）。
- 实测（3/3）：`launch=FAILED err=507015`（`evidence/logs/aic_scalar_ub.log:18,21,24`）。
- 设备侧 plog 逐字（`evidence/logs/aic_scalar_ub.plog_excerpt.txt:4,5,12`）：
  `there is an aicore error exception, core id is 0, error code = 271` /
  `errorStr: The address for scalar to access the internal buffer is out of bounds. subErrType: 0x4.` /
  `rtStreamSynchronize:ErrCode=507015, desc=[aicore exception]`。
- 读法：**在本档的字节向量下**，AIC 标量访问 UB **越界**。这个报错发生在**地址空间层面**
  （「internal buffer 的地址越界」），与根因「AIC 不能**直接**访问 UB」**逐字同向** ⇒ 它是根因的**主证据**。
  限度：这条**不代表**业务核上 AIC 标量访问 UB 一定不可用（shape / 地址 / 编译选项可能不同）；它只证明本档这个用例下越界。
- 与装填的关系：正因 AIC 无法回读 UB，本档「UB 已装已知数据」这一步由 AIC 侧可用的装填原语（§1.2、§3 的 L1→UB pipeline 搬移）支持，
  不由运行期回读支持。

---

## 6. 复用了哪些文件与机制（任务①的交底，带 文件:行）

| 复用的东西 | 来源（仓库内既有探针） | 本档落点 |
|---|---|---|
| 独立 `__mix__(1,2)` kernel + 自带 CMakeLists + `find_package(ASC)` + `--npu-arch=dav-3510` 的工程形态 | `probe_crosscore_tail/CMakeLists.txt:1-31` | `CMakeLists.txt:1-40` |
| `run_probe.sh` 的**设备槽纪律**（每档各自进锁 + 进锁先 `npu-smi` + 锁内 `timeout` + `LOCKFAIL` 记「未取得读数」） | `probe_crosscore_tail/run_probe.sh:1-96`；slot 口径另参 `probe_ub2l1/evidence/slot.sh:1-63` | `run_probe.sh:1-104` |
| host 侧 ACL 装载 / `<<<blockDim,0,stream>>>` 启动 / `aclrtSynchronizeStream` / 逐元素对拍 | `probe_crosscore_tail/probe_crosscore_tail.asc:600-740` | `probe_aic_gm_dma.asc:310-423` |
| FIXP L0C→GM 对照腿的小助手 `GmToL1Nz` / `LoadL0Bf16` / `MmadBf16` / `FixpToGm` | `probe_crosscore_tail/probe_crosscore_tail.asc:187-273` | `probe_aic_gm_dma.asc:109-171` |
| L1→UB 的 C API `asc_copy_l12ub` 走法 | `probe_ub2l1/probe_ub2l1.asc:178-179` | `probe_aic_gm_dma.asc:176-184`（`FillUbFromGm`） |
| 常量 sentinel / 逐元素 `mismatched/sentinel/zeros` 计数形态 | `probe_crosscore_tail/probe_crosscore_tail.asc:70-82, 702-737` | `probe_aic_gm_dma.asc:76-77, 396-414` |

> 路径解析说明：`probe_ub2l1/**` 在当前 main 上可解析；**`probe_crosscore_tail/**` 是 M134 的独立探针**，
> 截至本档写作时在 `feat/m134-cross-core-fixpipe-and-flag-account` 分支上、尚未并入 main。本档对它的引用是
> 「机制沿用」，不是「import 它的文件」——`probe_aic_gm_dma.asc` 自带全部算子小助手。

---

## 7. 限度（明确写窄，不外推）

- **区分「本档设备实测」与「架构层常识」**：根因「AIC 不能**直接**访问 UB ⇒ UB→GM 对 AIC 不可路由」是**架构层常识/人类定性**，
  本档不在设备上复测架构；本档实测的是**与之一致的读数**——`aic_scalar_ub` 的 507015 / plog 271（越界异常，**主证据**）、
  raw MTE3 编译被拒、基础 API 门成空（`landed=0`）。**不得**把「本档没有一条 AIC-only 的运行期回读」读成
  「根因未被证据支持」——主证据是那条**越界异常**，不是回读。
- **只测了**：dav-3510 / CANN 9.1.0 / 单 AICore 0 / `__mix__(1,2)`；形状固定 64×64 fp32（16 KB 单次）；
  UB 的装填只走 `asc_copy_gm2l1`+`asc_copy_l12ub` 一条路（MTE2→MTE1）；UB 只用一个 `TPosition::VECIN` 偏移 0 的窗口。
- **没测**：其它 pipe（`copy_ubuf_to_gm_align_v2` 的其余重载、`DataCopyPad` 系列、`asc_copy_ub2gm_align` 系列）；
  其它核型/核数（多 AICore、多 AIV 配对）；其它 `sub_blockid`；其它 dtype（bf16/int8）；UB 的其它偏移/大小；
  编译选项 `-DENABLE_CV_COMM_VIA_SSBUF=true`；CANN 其它版本。
- **不外推**：本档是合成形状，**不主张**任何业务核的结论（`m15_layer_loop` / `m25_attn_fa_core` 等的 UB→GM 走法另有其上下文）；
  **不主张**芯片级判断（本档只报「在这个最小用例下，AIC 的 UB→GM 读数如此」）。
- **未验证项**：AIC 侧 UB 的**内容**在运行期无法从 AIC 侧回读（§5，这本身即主证据）；
  本档用 AIC 侧可用的装填原语从原理上说明「UB 里有数据」这一步，但**没有**一条 AIC-only 的运行期读数直接打印 UB 内容。
  另：**§1.5 的层级项**——AIC 侧 `L1→UB`（`copy_cbuf_to_ubuf` pipeline 搬移）的**运行期效果**本档**未隔离验证**；
  它**不构成对根因的反证**；若要坐实需设备隔离验证（AIC 写 UB、配对 AIV 读回）。
- **对照组边界**：`aiv_ub2gm` 的 AIV 参与数据面，是为验证 harness 与 MTE3 指令本身可用，**不是**被测形态的一部分。

---

## 8. 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd probe_aic_gm_dma
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4

# (a) 编译期取证（零设备）：逐条小探针 + 汇总表
bash evidence/compile_reject/repro.sh

# (b) 设备档：每档各自进一次 flock（进锁先 npu-smi；timeout 在锁内）
REPS=3 bash run_probe.sh

# (c) 独立宿主侧复核（零设备）：对每份 dump 用 Python 重算
for v in fixp_l0c2gm ub2gm_api ub2gm_capi aiv_ub2gm neg_fixp_short; do
  for b in evidence/dumps/out_${v}_run*.bin; do python3 check_ref.py "$v" "$b"; done
done
```

单档：`./build/probe_aic_gm_dma <variant> <runNo> [outdir] [--trace]`；列变体：`./build/probe_aic_gm_dma list`。

**源码指纹**（内容寻址，用来核对「本 log 由哪份源码产出」）：

```
4ad10a1c97f22bb8587912aa0eaffb263f05886a5b45eb6503946c7da95915c4  probe_aic_gm_dma.asc
08eebfc1a15cc360ec5a737b12254c6344deada23b327c4b527160f72db28e74  CMakeLists.txt
4bce12cbeba2f5e2e4bec3e52699f014b82abcceba8d832e4304e51af49f8b48  run_probe.sh
5f65889a2d02e5f6166e1b17586e4ebb2c1bb3f7b518af35e36e3a469510d598  check_ref.py
```

> `run_probe.sh` 生成的 `evidence/logs/commands.txt` 另存了**运行时点**的 sha256（只含 `probe_aic_gm_dma.asc` / `CMakeLists.txt` / `run_probe.sh` 三份，见 `commands.txt:13-15`；`check_ref.py` 不在其中）。
> **可逐字对照的只有两份**：`probe_aic_gm_dma.asc`（`commands.txt:13`）与 `run_probe.sh`（`commands.txt:15`）——它们与本 §8 指纹一致。
> **`CMakeLists.txt` 两边不同、且只差注释**：`commands.txt:14` = `cbb85ee4…`，本 README §8 = `08eebfc1…` —— 本档收尾时对 `CMakeLists.txt` 只改了注释（去掉对未合入分支的绝对路径引用），可执行产物不受影响；**不要**把这两个哈希当成不一致的 bug。
> 本次（M153）只改**措辞**：`README.md` 与 `evidence/README.md`；上列四份源码/脚本一字未动、哈希未变，`evidence/` 下的日志 / dump / 汇总 / plog 摘录亦**未改**。

---

## 9. 纪律自查

- **根因归位**：§0 / §1 / §3 / §5 已按人类逐字纠正改写——根因＝「AIC 不能**直接**访问 UB ⇒ UB→GM 对 AIC 不可路由」；
  AIC 侧 `L1→UB`（MTE1）是 **pipeline 搬移**、不是直接访问 UB、不构成反证（§1.5）；
  原「编译期被拒 + AIV 门控」降为**后果证据**（保留为读数，不再冒充根因）。
- **禁用词**：mission 纪律清单里的 6 条绝对化措辞，在 `probe_aic_gm_dma/` 的新增行上**未命中**。
  为了让本 README 自己不被该自检命中，这里不逐字复写那 6 个词；逐字 pattern 与命令放在交付说明（review-request）里。
- **`文件:行`**：§0–§5 每条结论都给了 `文件:行` 或 evidence 路径（§1.4 是逐条对齐表）。
- **读数/哈希未改**：`evidence/**` 的日志、dump、plog 摘录、汇总与 `probe_aic_gm_dma.asc` 等源码哈希一并未动（§8）。
- **协议目录字样**：新增文件（含本 README）里不含 mission 明令禁止的那个隐藏目录名（逐字 pattern 见交付说明），自检**未命中**。
- **`flock` 纪律**：每档各自进一次 `flock -w 300`、进锁先 `npu-smi`（快照见 `evidence/npu_smi_snapshots.txt` 与各档 log）、
  `timeout` 放在锁内、锁未取得记「未取得读数」（本轮未出现锁未取得的情况）。
- **不外推**：见 §7。
