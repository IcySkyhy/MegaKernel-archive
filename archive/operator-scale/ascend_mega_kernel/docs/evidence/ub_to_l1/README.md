# UB→L1 操作数通路：把 `docs/15 §6.2` / `docs/19 §7 第 4 条` 的前提钉到可复现的证据上（M120）

本目录是 **M120** 的证据包。它不改代码、不上设备；**只回答两个问题**：

1. 「UB→L1 的操作数通路」到底**哪些接口受影响、哪些不受**（逐字引官方文档 + 本机头文件，带 `文件:行` + 版本）；
2. 在「禁高阶 API」的本仓口径下，**还剩哪几条路能用**，各自的代价是什么。

> **结论先行的正确形态是「一条分支受影响、另有硬通道候选」，不是「通路不成立」。**
> 原始前提（M115/B1 的 finding，见 `.tower/comms/findings/20260927-agent-b1gdn-…`）引的是四条基础 API 的
> NOTE「软件仿真实现…需要 `REGISTER_MATMUL`」；但那句 NOTE 只覆盖 **GM 中转那条分支**。同一批官方文档里
> **另有**「3510 新增 UB→L1 Buffer 硬件通道」这一节，且**明确写了它不需要 Matmul 注册**。
> 详见 §1 的分支表与 §5 的实跑读数。

**纪律声明**（本目录遵守本仓写作规范）：

- 凡计数/读数都带命令与范围；**不写「已全部 / 无残留 / 0 命中」这类绝对断言**（能写的是「本目录的读数如此」）；
- `文件:行` 一律钉**不可变锚点**：asc-devkit 钉 commit `648a6018207d75af44c6865f96511bafadd90630`，
  本机 CANN 路径钉 `/usr/local/Ascend/cann-9.1.0`（安装目录名即版本）；
- 引用规则编号前先 grep 过（`docs/05` §6.1 的 ⓐ–ⓔ 用**内容锚点**引，不裸引行号）；
- 本 mission **零设备档**：§5 全是 host 侧编译读数，**没有一条真机读数** ⇒ 凡涉及真机行为一律写「未确定 + 需要什么实验」。

---

## 1. 接口清单：哪些受影响、哪些不受

先说两条**分支**的名字（下文反复用到）：

- **GM 中介分支**（= 官方文档里那句 NOTE 描述的形态）：UB →（MTE3）GM →（MTE2）L1；实现上要
  `TPipe` 的 workspace + KFC 客户端，也就是「Matmul 高阶 API 的软件仿真」。
- **硬件通道分支**：UB →（MTE3 `copy_ubuf_to_cbuf`）L1，不经 GM、不用 Matmul 注册。

| # | 接口（本机 CANN 9.1.0 头文件里是否存在） | GM 中介分支 | 硬件通道分支 | 本仓口径下的结论 |
|---|---|---|---|---|
| 1 | **`asc_copy_ub2l1`**（C API；连续 + 高维切分两种重载） | 文档未给 | **直接就是硬件通道**；「无需配置编译选项」，且「C API 直接使用对应硬件接口，不能套用基础 API 兼容路径的注册要求」 | **候选可用**（§5 探针 3a：本机编过，rc=0） |
| 2 | `DataCopy(dst_L1, src_UB, count)`（基础 API，连续） | 有：`GetTPipePtr` + `GetKfcClient()->AllocUB` + `ScmDataCopyMsg` | 有：`CopyUbufToCbuf`（`-DENABLE_CV_COMM_VIA_SSBUF=true`） | **候选可用**（§5 探针 3b/3c/3d） |
| 3 | `DataCopy(dst_L1, src_UB, DataCopyParams)`（基础 API，高维切分） | 同 2 | 同 2 | 同 2（未单独探针；分支条件与 2 同源，见 §5「覆盖范围」） |
| 4 | `DataCopy(dst_L1, src_UB, Nd2NzParams)`（基础 API，**随路 ND2NZ**） | 有：AIV 侧 `TransND2NZ` 进栈缓冲 + 搬 GM + `ScmDataCopyND2NZMsg` | 950 有：文档写「支持 UB→L1 的 **1:2 硬通道**」；本机另有 `DataCopyUB2L1ND2NZImplV2`（逐行/逐列 `CopyUbufToCbuf`） | **未确定**；官方样例 README 逐字说该接口「为软件仿真实现，**硬件本身不支持该能力**」，并**推荐在 UB 内自行完成 ND→NZ**（见 #7） |
| 5 | `DataCopyPad(dst_L1, src_UB, …)`（基础 API） | 有（文档逐字「实际搬运路径为 UB→GM→L1 Buffer」＋「发送通信消息会有开销，性能会受到影响」） | 文档列为受益于该编译宏的接口之一 | **未确定**（文档两侧都提到，但本目录未单独探针） |
| 6 | **GM 中介自建**（UB→MTE3 GM→MTE2 Nd2Nz L1，自己写） | — | — | **不受影响**（B1 现行路径；代价见 §2） |
| 7 | **UB 内 ND→NZ 再连续 UB→L1**（`asc_copy_ub2ub` 按 C0 列块 + #1/#2 搬进 L1） | — | 用 #1/#2 落地 | **候选可用**（官方样例 scenario 2 的现实做法） |
| 8 | `LoadData`（L1→L0A/L0B）、`Mmad`、`Fixpipe` | — | — | **完全不受影响**（这条链本来就没有 UB→L1 环节） |

**一句话**：受影响的是 **GM/软件仿真那一条分支**（它要 Matmul 注册，本仓禁高阶 API ⇒ 走不了）。
**硬通道有官方文档、有本机头文件实现、有官方样例，三层都在** ⇒ 它在**文档层**是「本仓口径内可用」的候选；
真机可用性本 mission **未取证**（§3）。

### 1.1 逐字证据（含版本与锚点）

**A. 官方文档（asc-devkit，`文件:行` 以 rev `648a6018` 为准）**

- 四条基础 API 的「软件仿真」NOTE（**四条都在**，注意这四条是**同一句**）：
  - `docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopyPad_UBToL1.md:31`
  - `…/DataCopy_UBToL1_ND2NZ.md:37`
  - `…/DataCopy_UBToL1_continuous.md:37`
  - `…/DataCopy_UBToL1_highdim_split.md:37`

    > 本接口为软件仿真实现，是在Matmul高阶API的基础上，利用Matmul高阶API中的workspace GM空间作为数据中转空间，数据先搬入GM，再搬入L1 Buffer。因此，在使用本接口时，需要先使用REGISTER_MATMUL注册高阶API。

- **同一份 ND2NZ 文档**在「约束说明」里另写 950 的硬通道（**这是 M115 那句结论漏掉的一半**）：
  `…/DataCopy_UBToL1_ND2NZ.md:129`

    > 针对Ascend 950PR&950DT系列产品，支持UB->L1 Buffer的1:2硬通道的方式；同时提供1:1的兼容模式，解决A2/A3的代码迁移兼容问题，这种模式通过Global Memory进行中转，效率稍低。

- `DataCopyPad_UBToL1.md:37`（同文档「函数原型」段）与 `:82`（CAUTION）：

    > 通路：Local Memory->Local Memory，实际搬运过程是UB->GM->L1 Buffer（TSCM）。
    > …内部实现涉及AIC和AIV之间的通信，实际搬运路径为UB（VECIN/VECOUT）->GM->L1 Buffer（TSCM），**发送通信消息会有开销，性能会受到影响**。

- `DataCopy_UBToL1_continuous.md:112` 与 `…/highdim_split.md:127`（逐字相同）：

    > 针对Ascend 950PR&950DT系列产品，在UB->L1 Buffer的数据搬运时，可以通过配置编译选项 ENABLE_CV_COMM_VIA_SSBUF 来选择两种搬运通路，当 ENABLE_CV_COMM_VIA_SSBUF 配置为 true 时，使用 SSBuffer 进行通信，数据通过 UB->L1 Buffer 之间的硬件通道进行搬运（推荐）…；当 ENABLE_CV_COMM_VIA_SSBUF 为 false 时，数据搬运到 L1 Buffer 经过 GM，该场景下需要借助 Matmul 高阶 API 进行注册操作。

- `docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md:38,42,46,48,49`
  （小节「新增UB到L1 Buffer搬运数据通路」，锚点 `#section_ub2l1`）：

    > CV融合算子中，UB向L1 Buffer搬运数据不再需要通过GM中转。
    > - 对于C API，使用 asc_copy_ub2l1 接口实现UB到L1 Buffer数据搬运，**无需配置编译选项**。
    > - 对于基础API，使用 DataCopy（UB到L1 Buffer）、DataCopyPad（UB到L1 Buffer）进行搬运，开启该特性需要配置编译选项 ENABLE_CV_COMM_VIA_SSBUF：
    >   - 开启该编译选项后，数据通过UB到L1 Buffer之间的硬件通道直接搬运。该方式无需经过GM中转，性能较高。
    >   - 未开启该编译选项时，数据需经由GM搬运至L1 Buffer。在此场景下，UB到L1 Buffer搬运接口采用软件仿真实现。

- `docs/zh/guide/programming_guide/advanced_programming/inter_core_communication/ssbuffer_inter_core_memory_feature.md:289`：

    > 基础API的UB到L1 Buffer硬通道路径与GM软件仿真路径不同：前者在相应配置下直接搬运，后者需要借助Matmul注册及GM中转空间。**C API直接使用对应硬件接口，不能套用基础API兼容路径的注册要求。**数据搬运能力也不意味着每次调用都要求用户另写一条SSBuffer消息，是否交换地址和参数取决于算子设计。

- `docs/zh/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.md:299`（`ENABLE_CV_COMM_VIA_SSBUF` 定义处）与 `:305`（那条 bullet）：

    > ENABLE_CV_COMM_VIA_SSBUF 用于控制是否使用SSBuffer以及**UB到L1 Buffer的硬通道**……默认开关关闭；设置为true后，表示开关打开。
    > - 在该平台新开发的算子，以下场景需要打开：……使用DataCopy接口从UB拷贝数据到L1 Buffer。

  **取证陷阱（M120 r1 复审 P2-1；本行号曾错引为 `:301`，已订正）**：定义句里的宏名是**转义下划线**——
  源文件逐字是 `ENABLE\_CV\_COMM\_VIA\_SSBUF`，而 `:301` 那条 bisheng 命令示例与 `:305` 的 bullet 是不转义的。
  ⇒ 直接 grep 字面量 `ENABLE_CV_COMM_VIA_SSBUF` **匹配不到 `:299`**，只会命中 `:301` —— 于是
  **「grep 命中」≠「命中定义行」**，而归档 log 里也看不出这个差（本目录第一版就是这么错引的）。
  `repro.sh` §1d 的 pattern 已改成让反斜杠可选（`ENABLE\\?_CV\\?_COMM\\?_VIA\\?_SSBUF`），
  现在会把 **`:299` 与 `:301`** 一并打出来（归档读数见 `logs/repro_C.log` 的 §1d 段）。
  另注：**`:305`（那条 bullet）里不含宏名**，同一条 pattern 匹配不到它 —— 引它时要单独 grep 它的正文
  （「在该平台新开发的算子……使用DataCopy接口从UB拷贝数据到L1 Buffer」）。

**B. 本机 CANN 9.1.0 的头文件/实现（`/usr/local/Ascend/cann-9.1.0`）**

- C API 公开声明：`x86_64-linux/asc/include/c_api/vector_datamove/vector_datamove.h:468-474`
  （`asc_copy_ub2l1(__cbuf__ void*, __ubuf__ void*, …)` 两个重载 + `asc_copy_ub2l1_sync`）。
- C API 实现（**直接硬件、不碰 TPipe/KFC/Matmul**）：
  `x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h:30-31`

    ```cpp
    if ASC_IS_AIV {
        copy_ubuf_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap);
    }
    ```

- 硬件 intrinsic 本体：`tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:1031`
  （`__attribute__((clang_builtin_alias(__builtin_cce_copy_ubuf_to_cbuf))) void copy_ubuf_to_cbuf(...);`，
  **无 `#if` 包裹** ⇒ 不是靠哪个编译宏才存在的）。
- 基础 API 的两条分支：`x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h`
  - 连续搬运 `DataCopyUB2L1Impl`：`:582` 起

    ```cpp
    #if KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1
        CopyUbufToCbuf(dst, src, intriParams.blockCount, …);          // 硬件通道
    #else
        … GetTPipePtr() … GetKfcClient()->AllocUB(…) …
        CopyUbufToGmAlignV2(…); ScmDataCopyMsg(…);                    // GM + KFC（= 软件仿真）
    #endif
    ```

  - 随路 ND2NZ `DataCopyUB2L1ND2NZImpl`：`:712` 同一条件；`:742` 走 `CopyUbufToCbuf`（先在 UB 栈缓冲里
    做 `TransND2NZ`），`:765`/`:777` 走 `AllocUB` + `ScmDataCopyND2NZMsg`。
  - 950 专用 `DataCopyUB2L1ND2NZImplV2`：`:783-805`，注释逐字「Ascend950 MTE3 UB->L1, but it requires the
    srcDValue must be 32B aligned」，**通体 `CopyUbufToCbuf`、无 TPipe/KFC**。
- `KFC_C310_SSBUF` 的定义：`x86_64-linux/asc/impl/basic_api/kernel_utils.h:36-40`

    ```cpp
    #if ENABLE_CV_COMM_VIA_SSBUF != 0 && __MIX_CORE_AIC_RATION__ != 1
    #define KFC_C310_SSBUF 1
    #else
    #define KFC_C310_SSBUF 0
    #endif
    ```

- 官方 donor（同一条 `asc_copy_ub2l1` 的**生产级使用例**）：
  `opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/blaze/gemm/tile/arch35/copy_weight_ub_to_l1.h`
  的 `CopyUB2L1Weight8Bit::Copy` —— 8bit 权重 UB→L1（NZ 分形），直调 `asc_copy_ub2l1`。
  （`ops_transformer/…` 下有同名副本；两处都是**工具链自带的官方算子库**，非本仓。）

**C. 官方样例（asc-devkit `examples/`）**

- `examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_ub2l1/`：Mmad 场景下的 UB→L1，
  `CMakeLists.txt` 用 `ascendc_compile_definitions` 配 `-DENABLE_CV_COMM_VIA_SSBUF=true`；
  `README.md:68` 逐字「核函数声明为 `__mix__(1, 2)` 混合核（1个AIC + 2个AIV）」，数据流
  「GM→UB→L1 Buffer→L0A/L0B→Mmad→L0C→GM」。
- `examples/02_simd_c_api/03_c_api/00_data_movement/data_copy_ub2l1/data_copy_ub2l1.asc`：C API 版，
  `__mix__(1,2)` 里 `asc_copy_ub2l1(a_l1, a_ub, A_L1_BYTES)`，配套 `asc_sync_block_arrive`（模式 2）
  与 `asc_sync_intra_arrive/wait`（模式 4）。
- **样例 README:11 声明「>= CANN 9.2.0」**，本机是 9.1.0 —— §3 的 U2。

---

## 2. 替代路径逐条 + 代价

| 路径 | 依据（`文件:行`） | 代价 | 状态 |
|---|---|---|---|
| **(a) C API 硬通道**：`asc_copy_ub2l1(dst_L1, src_UB, bytes)` | §1.1-B 头文件；§1.1-A 的「无需配置编译选项」 | **无 GM 往返**；但：① 只在 AIV 上生效（`ASC_IS_AIV`），AIC 侧要另配同步；② 流水类型 `PIPE_MTE3`（C API 文档）；③ **真机带宽/同步语义未标定** | **候选**（§5 探针 3a 编过） |
| **(b) 基础 API 硬通道**：`DataCopy(dst_L1, src_UB, …)` + `-DENABLE_CV_COMM_VIA_SSBUF=true` | §1.1-B `kernel_operator_data_copy_impl.h:582`；§1.1-A `3510_new_features.md:46-49` | 同 (a)；额外要确认本仓构建是否愿意加这个编译宏（本机 `__mix__(1,2)` 下**不加**也落硬件分支，见 §5 探针 3d —— **该现象依赖未定义宏的比较，属"观察到"，不当保证**） | **候选** |
| **(c) GM 中介自建**：UB→MTE3 GM→MTE2 Nd2Nz L1 | `docs/19 §7` 的原始设计；B1 已落地（§4） | **多一次 GM 往返**：操作数字节 ×2 过 GM（写+读），另占 GM scratch 与 GM 带宽；若交接跨 op，同步按核规则要 **mode 0**（`docs/05` §2，内容锚点 `mode 0` 那两条） | **本机/仓内看起来可落地，但真机读数未取到**（B1 已实现、构建 rc=0；**其设备档未返回** —— 是"挂"不是 FAIL，见 §4 的未返回记录） |
| **(d) UB 内 ND→NZ 再连续搬**：`asc_copy_ub2ub` 按 C0 列块重排 → (a)/(b) 搬进 L1 | 官方样例 README:42-49（scenario 2）；ND2NZ 文档 `:117-121` 给临时 UB 空间公式 | **多一次 UB 读+写**（操作数在 UB 里过一遍）+ **一块临时 UB**（公式：`((dValue×sizeof(T)/32 − 1) × dstNzC0Stride + (nValue−1) × dstNzNStride + 1) × 32` 字节）；好处是不碰 GM | **候选**（官方推荐的 ND2NZ 绕法） |

> **真机读数从哪来（M120 r1 复审的建议，2026-09-27 补）**：本表 (a)/(b) 两条**硬通道候选**的真机可用性，
> 塔已另立 **M121**（`feat/m121-ub2l1-hard-channel-real-machine-pro`，wt-121）做最小真机实验；
> 本 mission 是**零设备档**，不重复做，也不替 M121 预判结论。(c) 的真机读数则取决于 B1 的设备档
> （未返回，见 §4）。**本表没有任何一条带真机读数**。

**（不确定的，按纪律不猜）**

- `DataCopyPad_UBToL1`（#5）与「随路 Nd2NzParams」（#4）的**硬通道分支在真机上是否可用**：两条文档的证据方向不一致
  （一边列它们受益于该宏，另一边官方样例说 Nd2NzParams 版「硬件本身不支持」）⇒ **未确定**，见 §3 的 U3。
- 硬通道的 **AIV/AIC 同步配对**：官方 C API 样例用了模式 2 + 模式 4；本仓 `docs/05` §2 把「cube 输入的
  UB→L1」列为 **mode 2** 的点对点场景 ⇒ 形态是一致的，但**本 mission 未在真机上验过**。

---

## 3. 未确定 + 需要什么实验

| # | 未确定 | 需要什么实验 | 能不能用本目录回答 |
|---|---|---|---|
| **U1** | 硬通道（(a)/(b)）在 **Ascend 950 真机**上的可用性、带宽、以及 AIC 如何等到数据 | **一次最小 mmad 实验**：AIV 把一块已知数据写 UB → `asc_copy_ub2l1` 搬进 L1 → AIC `LoadData`+`Mmad` 出数 → 与 host 参考逐元素比；**并配一条负向对照**（把 AIV 那次搬运改成不执行，判据必须变红 —— `docs/17` §4「什么算非空洞」的**负向对照**条：「把该算件的规则/方向故意反过来跑一次，凡被声明为『有咬合力』的判定项必须至少有一条 FAIL」） | ✗（零设备档） |
| **U2** | 本机 CANN **9.1.0** 上这条硬通道是否已就绪（官方样例声明 **>= 9.2.0**） | 上一条实验同时回答；若为否，是"升级 CANN"还是"回 GM 中介"由塔裁 | 部分：§5 探针 3a 证明**编译层**就绪（rc=0），但样例整编不通过（`AscendC::Std::ceil_div` 在 9.1.0 不存在，§5 §4 段） |
| **U3** | ND→NZ 归谁做：随路 `Nd2NzParams` 硬通道 vs UB 内自做（(d)） | 两条各跑一次，比时间与逐元素正确性 | ✗ |
| **U4** | GM 中介（(c)）与硬通道（(a)）的**端到端代价差**（本仓口径下） | 同一 shape 两条路径各测一次（含 L1 窗占用、GM 往返字节、同步开销） | ✗ |

**这些都不能在本 mission 里做**（零设备档）。塔已把其中 **U1** 收成一个 **"UB→L1 硬通道最小真机实验"** 派单：
**M121**（`feat/m121-ub2l1-hard-channel-real-machine-pro`，wt-121，2026-09-27 已 active）—— U2/U3/U4 可并入
它的后续。**在 M121 的读数出来之前，本目录没有任何一条通路可以被写成"不成立"或"已选型"**；
届时本目录与其派生结论（§2 的状态列、两份文档的更正块）都应以 M121 的读数为准复核一遍。

---

## 4. 本 mission 之前已落地的实现（引用事实，不改其文件）

B1 把 GDN prefill 的 mmad 操作数走成了 **GM 中介 (c)**：

- 分支 `feat/m115-b1-gdn-prefill-chunk-scan`，commit `8ca32c5` 与 `1f7b630`（**该分支未合入 main**，故这里只写
  分支名与 commit，不写路径 —— 路径式引用会随合并状态失效）；
- 它的 B1 报告自述「GM 往返偏高」的原因就是这条；它的设备档**未返回**（两次上限 120s / 280s 内都没回来，
  是"挂"不是"FAIL"），根因其自述为**未取证**。
- **本目录不对那个挂死下任何结论**；只登记一条**相关性**供后续实验设计参考：若走 GM 中介，按本仓核规则
  前后 op 跨 GM 交接要用 **mode 0**，用 mode 2 是错的（`docs/05` §2 逐字）。**这是待验假设，不是归因。**

---

## 5. 复现命令与读数（本 mission 实跑）

全部命令都由 `docs/evidence/ub_to_l1/repro.sh` 一键复现（**host 侧，零设备**）：

```bash
bash docs/evidence/ub_to_l1/repro.sh
```

归档读数：`logs/repro_C.log`（`LC_ALL=C`）与 `logs/repro_C_UTF8.log`（`LC_ALL=C.UTF-8`）——两份**逐字符相同**
（归一化掉 `mktemp` 的随机目录名后 `diff` 为空）。归一化命令：

```bash
sed -E 's#/tmp/m120_repro\.[A-Za-z0-9]+#<TMPDIR>#g'
```

关键读数（逐条对应上面的探针编号）：

| 探针 | 命令（公共 flags: `--npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c`） | 读数 |
|---|---|---|
| 3a | `bisheng … -c probe_capi_ub2l1.asc` | **rc=0**（stderr 空）⇒ C API 的 UB→L1 硬通道在本机 9.1.0 上**编得过** |
| 3b | `bisheng … -c probe_basic_ub2l1.asc`（无编译宏） | **rc=0** |
| 3c | `bisheng … -DENABLE_CV_COMM_VIA_SSBUF=true -c probe_basic_ub2l1.asc` | **rc=0** |
| 3d | `bisheng … -c probe_branch.asc`（把 `DataCopyUB2L1Impl:582` 的 `#if` 抄成 `#error`） | 有/无编译宏**都**打印 `BRANCH_HARDWARE_CopyUbufToCbuf` ⇒ 本机工具链给 `__mix__(1,2)` 编的是**硬件分支** |
| 3e | `bisheng … -c probe_mixmacro.asc`（`#ifdef __MIX_CORE_AIC_RATION__` 直接打出来） | `MIX_RATIO_NOT_DEFINED` ⇒ 本机这次编译里该宏**未定义**（3d 的结论由它支撑，不是反推） |
| §4 段 | 官方 basic API 样例（scenario 1）整编 | `cmake rc=0` / **`make rc=2`**，首个 error：`no member named 'ceil_div' in namespace 'AscendC::Std'`（样例源码 `:83`）⇒ 样例整体对 **CANN ≥9.2.0** 的声明与本机 9.1.0 有版本差 |

**覆盖范围（别把这几条读成更多）**：

- 探针 3a/3b/3c **只证明"编得过"**，**不证明**真机跑得对；
- 探针 3d 只证明**预处理器走了哪一支**。`probe_branch.asc` 里的 `__MIX_CORE_AIC_RATION__` 在本机这次编译里
  **未被定义**（探针 3e 用 `#ifdef` + `#error` 直接打出 `MIX_RATIO_NOT_DEFINED`，不是反推），所以 `!= 1` 为真；
  `__mix__(1,2)` 若在其它编译阶段被定义成别的值，结论要重取 —— **故这里写"本机这次编译如此"，不写成通则**；
- 本目录**未**对 #3（高维切分）与 #5（`DataCopyPad`）单独探针；它们的结论沿用"分支条件同源"这一条，**不是独立读数**。

---

## 6. 目录内容

| 文件 | 说明 |
|---|---|
| `README.md` | 本文件（§1 清单 + §2 代价 + §3 未确定 + §5 读数） |
| `repro.sh` | 一键复现：锚点回显 + 官方文档/本机头文件逐字 + 三个编译探针 + 版本差整编 |
| `probe_capi_ub2l1.asc` | 探针 3a：C API `asc_copy_ub2l1`（`__mix__(1,2)`） |
| `probe_basic_ub2l1.asc` | 探针 3b/3c：基础 API `DataCopy(L1_TSCM, UB_VECIN, count)` |
| `probe_branch.asc` | 探针 3d：把 `DataCopyUB2L1Impl:582` 的分支条件抄成 `#error` |
| `probe_mixmacro.asc` | 探针 3e：`#ifdef __MIX_CORE_AIC_RATION__` → `#error`（3d 结论的前提） |
| `logs/repro_C.log`、`logs/repro_C_UTF8.log` | 双 locale 的归档读数（两份逐字符相同） |
| `.gitignore` | 显式列出本目录**不**入库的编译产物 |

> **外部快照声明**：`/usr/local/Ascend/cann-9.1.0` 与 `/workspace/asc-devkit` 都是**本仓不持有**的工具链/上游
> 工作副本。`文件:行` 引用一律按上面的不可变锚点（CANN 路径含版本；asc-devkit 钉 commit `648a6018`）来判读；
> 上游工作副本前移后行号可能漂，读的人应以 commit 锚点为准。
