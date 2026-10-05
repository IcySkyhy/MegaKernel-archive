# M22：GDN layer kernel 骨架（单一 `__mix__(1,2)` 启动，段序 S1–S7）

图纸 = `docs/12-layer-integration.md` §2/§4/§5 的 GDN 段序 + `docs/10-gdn-analysis.md` §1 的数学/形状；
约束 = `docs/05-megakernel-design.md` §5/§6。算件代码全部**复制改造**自 main 上的
m6_rmsnorm / m11_bf16_gemm / m9_gdn_prolog / m4_gdn_recurrent / m12_rmsnorm_gated（原目录未改动）。

decode（m=1）单 token 全链，数值与 numpy 双实现（kernel 内建 C++ 参考链 + `check_ref.py` 的 numpy
参考；**同一作者的两次转写，不是两个独立来源**，见 §5.6(b)）逐段一致：**138 条 kernel 判定项全
PASS、139 条 numpy 判定项全 PASS、conv_state 位级一致**（另有 11 条 kernel 报告项 / 33 条传播诊断项）。
判据按「参考的输入从哪来」分族（host 全链 / 分段设备锚 / 报告诊断），并逐条写明**咬得住什么、
咬不住什么**：见 §5.2 与 §5.6。

## 1. 链结构与分段

| 段 | 算子 | 来源算件 | 核 | 输入 → 输出（GM） |
|---|---|---|---|---|
| S1 | Add+RMSNorm #1 | m6_rmsnorm | AIV 行切分 | 层输入 `x` + `res` → `x_norm`(bf16) + `res1`(**fp32**) |
| S2 | in_proj bf16 GEMM（K=2560 N=16480，权重 [16480,2560]） | m11_bf16_gemm | AIC N 切分（103 tile / 28 AIC） | `x_norm` + `w_in` → `qkvzba`(bf16[1,16480]) |
| S3 | prolog：conv1d(K=4,10240ch)+bias+SiLU → q/k l2norm（仅 q ×1/√128）→ gating | m9_gdn_prolog | AIV block 条带（80×128ch） | `qkvzba` + `conv_state` → `q/k/v`(fp32) + `g/β`(stride-8 槽) + **`conv_state` in-place** |
| S4 | 递推：decay→delta→outer→matvec 寄存器 VF 单遍融合 | m4_gdn_recurrent | AIV head 条带（48 head） | `q/k/v/g/β` + `ssm_state` → `o`(fp32) + **`ssm_state` in-place** |
| S5 | RMSNormGated：per-head(128) RMSNorm ×gamma ×sigmoid(z) | m12_rmsnorm_gated | AIV head 条带（48 head） | `o` + z 段（`qkvzba[10240,16384)`）→ `opin`(bf16[1,6144]) |
| S6 | out_proj bf16 GEMM（K=6144 N=2560） | m11_bf16_gemm | AIC N 切分（16 tile / 28 AIC） | `opin` + `w_out` → `opout`(bf16[1,2560]) |
| S7 | Add+RMSNorm #2（残差 = S1 的 fp32 `res1`，小改 D） | m6_rmsnorm | AIV 行切分 | `opout` + `res1` → `y_final`(bf16) + `res2`(**fp32**) |

**z 段生命周期**（mission 明确要求）：`z` 是 in_proj 输出的元素区间 `[10240,16384)`（6144 维），
S3（只读 q|k|v 前 10240 通道与 b/a 尾段）与 S4 都不触碰它；S2 写 → S3/S4 期间原址存活 →
S5 读（每 head 取 128 维作 output gate）。它没有独立 buffer，随 `WS_QKVZBA` 一起存活，
不存在与其它张量叠放的风险（`m14_resources.h` §5 记录了全部 13 个张量的 生产→消费 生命周期）。

**常驻状态**：`conv_state[3,10240]bf16`（60KB/层，planar，S3 原位左移）与
`ssm_state[48,128,128]fp32`（3MB/层，S4 原位 RMW）由 host 分配、跨 `aclrtLaunch` 调用持续存在；
本工程用 `chain2`（两次启动两个 token）实测了「跨调用 in-place」语义（见 §5）。

数据面：全部中间张量落在一段连续 workspace（偏移见 `m14_resources.h` §5，32B 对齐，
`WS_BYTES = 5258240` ≈ 5.0MB）。权重直接消费 HF 原始 layout（bf16 行主序 `[N,K]`），host 不做离线转换。

## 2. 全局资源表（`m14_resources.h`，对应 docs/12 §4）

| 资源 | 落地 |
|---|---|
| AIC BufferID | 0-1 A ping/pong、2-3 B、4-5 L0A/L0B、6 L0C（编号与 m1/m3/m11 一致）；7-13 预留（跨段权重预取 / FIXP→UB） |
| AIV BufferID | 0-1 m6 行窗；2-4 m9 prolog；8-14 m4 递推（§4 原样）；15-17 m12；21-22 gamma 预转 |
| flagId | **4-7 = GDN 段边界**（S1 后 in_proj 输入就绪 / in_proj 后 / S5 后 / out_proj 后）；8-11 = AIV 段内 mode-0 barrier（S1 后、S3→S4、S4→S5、S5 后）；12-15 = AIC 段内 mode-0 对齐（in_proj 前/后、out_proj 前/后，**前/后各占不同槽**）；**本 mission 实占 4-15 全部槽位，只剩 0-3 留给 MoE/attention** |
| UB | PERSIST @0（gamma bf16 staging + 三份 gamma fp32 预转，末址 @26112）；**四个互不重叠的段窗** @32768 起：m6 31488B / m9 13312B / m12 2560B / m4 141312B（= §4 名义的 ~138KB SEG-GDN 峰值），末址 `UB_RC_END = 227328`，UB 上限 248KB = 253952，**余 26624B（≈26KB）** |
| L1 | A 区 @0 ×2 组、B 区 @262144 ×2 组（与 m11 一致）；@330KB 权重预取滚动窗 §4 已留边界，本 mission 未用 |
| GM workspace | 13 张量（`SZ_*/WS_*` 编译期常量），总量 5.0MB |

**UB 为什么不做同址叠放**（与 docs/12 §4「段内活跃 buffer 地址可叠放」的差别）：§4 允许叠放的前提是
段间 barrier 分隔；m13 实测记录了两条跨 pipe 交接坑（同一 UB 区被不同 BufferID 保护会失效；
同 pipe drain-release 后立刻重新 acquire 会自锁）。本 mission 选择**互不重叠**以构造性消除该类风险，
代价是窗内余量从 ~100KB 降到 ~20KB（m4 的 138KB slab 窗是硬需求，与 §4 数值一致）。

## 3. 同步表

段边界跨核走 docs/12 §5 的标准 mix 序列（**下表与代码逐行一致**：两个 AIV→AIC 边界都先做一次
全体 AIV 的 mode-0 barrier 再 set mode2）：

| 方向 | 序列 |
|---|---|
| AIV → AIC | 全体 AIV 先 `mode-0 barrier`（set 挂 `PIPE_MTE3` 排空本核写、wait 挂 `PIPE_S`，见下）→ 每个 AIV `set mode2(PIPE_MTE3)` → 每个 AIC `wait mode2(PIPE_S)` |
| AIC → AIV | 全体 AIC `set mode0(PIPE_FIX)`（FIXP 写 GM 排空）+ `wait mode0(PIPE_S)`（全体 AIC 对齐）→ 每个 AIC `set mode2(PIPE_MTE2)` → 每个 AIV `wait mode2(PIPE_MTE2)` |
| AIV ↔ AIV（段内交接） | 全体 AIV `set mode0(PIPE_MTE3)` + `wait mode0(PIPE_MTE2)` |

| 同步点 | flagId | 作用 / 依据 |
|---|---|---|
| S1 后 AIV 全体对齐 | 10（mode 0） | **防御层**（对称起见 S5 后也有同款，见下一行）。真正保证「AIC 不会在 `x_norm` 写出前搬 A」的是 AIC 侧的对齐：每个 AIC 先 `wait mode2`（只覆盖配对 2 个 AIV）再做全体 AIC 的 mode-0 barrier——mode2 AIV→AIC 是 2:1 rendezvous，56 AIV ÷ 2 = 28 AIC，故 28 个 AIC 全部越过该 barrier ⇔ 全部 56 个 AIV 都已 set（= 都已过 S1） |
| S5 后 AIV 全体对齐 | 11（mode 0） | 与上一行对称的防御层；真正保证「opin 全部落 GM 后才起 S6」的同样是 AIC 侧的 `wait mode2(6)` + mode-0 barrier（28 个 AIC 的 rendezvous ⇔ 56 个 AIV 都已 set） |
| S1 → AIC（in_proj 输入） | 4（mode 2） | AIC 侧 mode2 wait 见上 |
| in_proj → AIV | 5（mode 2） | AIC 的 FIXP 排空 + 全体 AIC 对齐（12/13）之后才 set：保证 103 个 N tile 全部落 GM |
| S3 → S4 | 8（mode 0） | v head 的 `v/β` 由某个 AIV 产出、可能被另一个 AIV 在 S4 消费（head 条带不同）；g/β 同理。**纯数据等待**（后面紧接 MTE2 装载）→ wait 挂 `PIPE_MTE2` 排空 MTE2 队列 |
| S4 → S5 | 9（mode 0） | 同 8（相邻段共用 MTE2 队列，需排空后才装载 o） |
| S5 → AIC（out_proj 输入） | 6（mode 2） | 同 4 |
| out_proj → AIV（S7） | 7（mode 2） | 同 5 |

> 关于两处边界 barrier 的定位（一处口径澄清）：它们**不是**正确性的必要项——
> 正确性由 AIC 侧「先 wait mode2 再 mode-0 barrier」给出的 full rendezvous 保证（CANN 9.1 mode-2
> 语义：两个 AIV 都 set 之后 AIC 上的 wait 才放行）。S1 处原本就有一层（作者保守加的防御），
> S5 处补上同款（flagId 11，原为空），使两处边界对称、表与代码一致；两处 barrier 的 wait 都挂
> `PIPE_S`（因为它们后面紧接 set，属 M10 的链式同步规则）。

规则核对：set 一律挂生产 pipe（FIX/MTE3），**从不挂 PIPE_S/ALL**；wait 用最窄 pipe（数据依赖段挂
`PIPE_MTE2`、标量依赖/链式紧接 set 的段挂 `PIPE_S`）；mode 0 只在同类型核之间；**相邻同步点 flagId
必不同**（AIV 侧 10→4→5→8→9→11→6→7；AIC 侧 4→12→13→5→6→14→15→7 全部两两不同）；每 id 每层
使用次数 ≤2 ≪ 15 上限。

另按塔的两条广播逐点自查（`grep -n "CrossCore"` 可复核）：

* **M24「pipe 类必须匹配核型」**：AIC 侧只出现 `PIPE_S / PIPE_MTE2 / PIPE_FIX`（**没有任何
  `CrossCoreSetFlag<..., PIPE_MTE3>`**——在 AIC 上是静默空操作、会让对侧永久挂死）；AIV 侧只出现
  `PIPE_S / PIPE_MTE2 / PIPE_MTE3`。两侧 pipe 类均在允许集合内。
* **M10「wait 后紧接 set 的链式同步，wait 必须挂 PIPE_S」**：S1/S5 两处边界 barrier 后面紧接
  `CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>`（flag 4 / 6），故它们的 wait 都挂 `PIPE_S`
  （不需要 narrow-pipe 排空：可见性由 set 挂 `PIPE_MTE3` 的本地 drain 承担）。段内 AIV-AIV 的两处
  交接（S3→S4、S4→S5）是**纯数据等待**（后面紧接 MTE2 装载），仍挂 `PIPE_MTE2` 排空 MTE2 队列。
* **小改 B（同 pipe 背靠背复用同一 UB 缓冲必须 `PipeBarrier`）**已在两处落实：m6
  `NormStage::CopyInRow` 与新增的 m12 head 行循环 `RmsNormGatedStage::CopyInHead`；donor 自带的
  parity 双缓冲/乒乓写法（m9、m4）保持原样（其 BufferID acquire/release 链已把同 buffer 复用隔开）。

## 4. 小改清单落实（docs/12 §6，GDN 相关项）

| 项 | 落实 |
|---|---|
| **A** 元素计数 DataCopy → 显式参数 | 复制改造时逐点落实：m6 段 `Block1(HIDDEN*2/4)`、m9 段 donor 自带 `DataCopyParams{1,n,0,0}`、m12 段 `Block1(512/256B)`、新增的 gamma 预转 `Block1(NELEM*2)`；`Block1()` 内按字节→32B 块换算并 `Trap()` 防呆 |
| **B** 行循环 MTE2 复用补 PipeBarrier | m6 `NormStage::CopyInRow` 保留 donor 的 `PipeBarrier<PIPE_MTE2>`（m13 同款）；新增的 m12 `RmsNormGatedStage::CopyInHead` 同款补上；m9 段 donor 用 parity 双缓冲天然规避；gamma 预转 `PipeBarrier<PIPE_MTE2>` 前置 |
| **D** m6 残差输入 fp32 | `NormStage<RES_F32>` 模板：S7 消费 S1 的 fp32 `res1`，实测 `res2` 与参考**逐位一致** |
| **G** m9↔m4 的 g/β stride-8 契约 | 两段是同一份 VM 内相邻段，契约即编译期常量（`HEADS*8` 槽位 + `{H,1,0,0}` 直读）；S3 写出、S4 `LoadAlign<DIST_BRC_B32>` 消费，实测 g/β 容差占用 ≤0.155 |
| C（量化器）/E（索引批量）/F（calcM≥2 契约） | GDN 段无 MXFP4 量化、无 topK 索引；**calcM≥2 quirk 已落实**：host 按 `M_MAX` 行分配 `x_norm`/`opin`，GEMM 内 `calcM = max(curM,2)` 多读 1 行、Fixpipe 只写 `curM` 行 |

**与 donor 的差异点**（除 UB/flagId 重映射外只有四处，均为必要改造）：

1. `m11 Bf16Gemm::Process()` → `RunTile(nBlock)`：层内核里 m ≤ 64 时 `mLoop==1`，N 方向由各 AIC 条带划分（`item = bid, bid+numBlocks, ...`），不再在 op 内循环 N。
2. `m9 GdnProlog` / `m4 GdnDecodeRec` 的 `GetBlockNum()` → `GetBlockNum() * 2`：donor 是 `__vector__` 启动（AIV 视角 `GetBlockNum()` = AIV 数），**mix(1,2) 下 AIV 视角 `GetBlockNum()` = AIC 数 28**（docs/05 §6 §"已确认" + m3/m0 同款写法），不改会把 80 个 block 条带划分算错。
3. `m12 RmsNormGatedKernel` 单核整行 → `RmsNormGatedStage` head 切分：o/z/out 按 head 取 512B/256B 块（32B 整数倍），48 head 分到 48 个 AIV；`CalculateGateY` 的乘法次序与 sigmoid 四元组**逐字保留**，gamma 预转挪到 kernel 开头（每核一份）。
4. 段内 BufferID/UB 偏移重映射到全局资源表（`BUF_*`/`UB_*` 常量改为引用 `m14_resources.h`），donor 的同步纪律与流水结构（m4 的 slab 乒乓、m9 的 parity 双缓冲、m6 的 fold reduce + NR rsqrt）保持原样。

**NormDonor 的 reduce dispatch**：`CalculateSquareReduceSum` 现为 donor（m6/m12 同源）的**完整四分支**
dispatch（`≤64 → LessThanVL`、`≤128 → LessThanTwoVL`、`≤8192 → Common<1>`、`else Common<2>`），
其中 S1/S7 的 2560 维走 `Common<1>`（= m6 donor 调用）、**S5 的 128 维走 `LessThanTwoVL`（= m12 donor
路径，无 tmp 往返、无 `LocalMemBar`）**。早前的复核指出 m13 lift 时只搬了 `Common` 一支、
dispatch 被硬编码（S5 因此静默走了 `Common<1>`）——现已把 donor 的 `LessThanVL`/`LessThanTwoVL` 两支
补齐、恢复完整 dispatch，故 §4 的「四处差异」仍然只有四处（该偏离已消除，而不是被申报）。
两支的数值等价性也在实测中得到佐证：切换前后 `opin` 的判据值完全一致（隔离 max ulp 0 / 100% bit）。

## 5. 校验方法与结果（Ascend950PR / CANN 9.1.0 / 28 AIC + 56 AIV）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m14_gdn_layer/build -S m14_gdn_layer -DCMAKE_BUILD_TYPE=Release
cmake --build m14_gdn_layer/build -j4
./m14_gdn_layer/build/m14_gdn_layer        # 四个 case：chain / slice / chain2 / hostchain

# 落盘 + numpy 独立交叉校验
mkdir -p /tmp/m14_dump && cd /tmp/m14_dump
M14_DUMP=1 <repo>/m14_gdn_layer/build/m14_gdn_layer
/usr/local/python3.12.13/bin/python3 <repo>/m14_gdn_layer/check_ref.py .
```

`M14_STAGE_LIMIT=n` 截断段序（1..7 = S1..S7，bring-up 定位用；两核侧门控一致；矩阵 **15 组件无挂死**——
13 个有判定项 + 2 个 0 判定项（后者按 §5.3 的规则**不发合格证**且 `rc=2`），命令与读数见
`evidence/dump_manifest.md` §1(b) 与 §5）。

### 5.1 四个 case

| case | slice_mode | 步数 | 覆盖 |
|---|---|---|---|
| `chain` | 0 | 1 | 设备整链 S1→S7（真实层链：router… 无关；in_proj 消费 S1 的 `x_norm`） |
| `slice` | 1 | 1 | host 直供**任意** bf16 `qkvzba` → device S3→S5（AIC 完全不参与）⇒ 族 H 覆盖 S3→S5 |
| `chain2` | 0 | 2 | 同 chain，但两次 `aclrtLaunch` 两个 token：验证 `conv_state`/`ssm_state` **跨调用 in-place** 语义 |
| `hostchain` | 2 | 1 | **host 声明输入直入 S3 → device S3→S7（S6 在 AIC）**：S1/S2 由 host 算，参考侧无任何设备锚 ⇒ 族 H 覆盖 S3→S7（M62 新增，task 2） |

输入全部公式哈希确定性生成（无 rand 状态）：权重/激活 bf16 用通用尾数（非整数域），
`A_log>0`、`dt_bias` 含 >20 的样本（覆盖 m9 softplus 的 Select 两分支）、b 段含饱和样本
（sigmoid 落 0/1 边界）、`ssm_state` 初值幅度 ≤0.5（两步递推稳定）；判据相关的 **38 个 dump
文件** sha256 在任意重复运行下逐字节一致（见 `evidence/dump_manifest.md` §4 的 M48 订正与 §3.1 的
M62 对照；5 个 `*_ws.bin` 例外——含 97.15% 内核从不写的字节，§6 有定位与替代判据）。

### 5.2 判据分层：按「参考的输入从哪来」分三族（M62 重命名 + 补族 H）

M59 的自指验收审计（R7）指出：本模块原先把 S3–S7 的判据一律标成「全链」，而它们的输入其实是
**设备自己的 bf16 出口字节** ⇒ 标签名不副实（会被读成端到端），且整份参考里没有一条**从 host
声明输入重算**的下游链。M62 按**输入来源**重新分族（`check_ref.py` 文件头有同样口径），并把缺的
那一条补上。下表是唯一口径，标签里不再出现裸的「全链」二字：

| 族 | 标签（判据尾缀） | 参考的输入 | 覆盖 | 自指形态（M59 代号） |
|---|---|---|---|---|
| **H** | `(host 全链)` | **host 声明量**（dump 的 params + host 侧公式重算），**零设备字节** | `chain` 的 S1/S2；`slice` 的 S3→S5；`hostchain` 的 S3→S7 | 不是 S1/S2/S3/S4 |
| **D** | `(分段, 设备锚)` | 设备自己的 bf16 出口字节（`qkvzba`/`opin`/`opout`），段间状态由参考自携带 | `chain` 的 S3–S7 | **S1**（设备产物入参），其合法性逐条见 §5.6 |
| **P** | `(报告/诊断)` | 纯参考链（无锚点、参考在每个 bf16 出口用**自己**算出的舍入值） | 只报数、**不判定** | N4（结构性；不可复现） |

> **口径订正：订正的是 M59 哪一处、为什么**（M59 是已收口的 survey，它的命中表会被后续引用，所以这处
> 必须写在文档里而不只是评审记录里）。M59 对 m14 的那一行（其 inbox survey-summary §3 的 m14 行 + R7）
> 把本模块**所有**标「全链」的行都记成自指形态 **S1**（「全部以设备上一段 dump 为锚…标"全链"的行实际
> 也 re-anchor 到设备字节」）。按 `main` 上 base 版代码逐条看，这句话**只对 `chain` 的 S3–S7 成立**：
> * `chain` 的 `S1 res1 / S1 x_norm / S2 qkvzba (全链)` 的输入是 dump 里的 **host 参数 + host 公式**，零设备字节；
> * `slice` 档 `(全链)` 的 S3–S5 其 `qkvzba` 来自 `m14_param_xqkvzba.bin`（`H_DumpStep` 对
>   `sliceMode != 0` 写的就是 host 文件行；manifest 里 `tensor=qkvzba file=m14_param_xqkvzba.bin` 可直查），
>   也是一个**任意 host bf16 张量**，同样零设备字节。
> ⇒ 真正 re-anchor 的只有 `chain` 的 S3–S7。M62 把这部分改名成 `(分段, 设备锚)`，并保留 `slice`/
> `hostchain` 的 `(host 全链)` 标签——它们名副其实。
>
> **复核出处（用 commit 定位；下面每条命令都已实跑，结果见本节末的「命令-输出自检表」）**：这处
> 订正已由对 **commit `acf3f69`** 的独立复核确认**成立**（复核内容是逐 case 反证，不需要读评审记录
> 也能自证）。三条命令（都在仓库根目录跑）：
> * `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc | grep -n 'tensor=qkvzba file='`
>   ⇒ **2 命中**：`H_DumpStep` 里 mode 1 → `m14_param_xqkvzba.bin`、mode 2 → `m14_param_xqkvzba_h.bin`；
> * `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc | grep -n 'H_WsTensorLine("qkvzba"'`
>   ⇒ **1 命中**：同函数的 `if (sliceMode == 0)` 分支（即 `chain` 档的 `ws_off` 行源，**没有 `file=` 字段**）；
> * `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc | grep -n 'hostQkv'`
>   ⇒ **2 命中**：`H_RefChainStep` 的非 0 分支只用 host 张量。
> ⇒ 逐 case 反证：`chain` 档的 S3 输入指向 workspace 槽位，`slice`/`hostchain` 档指向 host 参数文件。
> M59 的 R7 若读作「没有从 host 声明输入重算的**完整** S1→S7 链」则仍成立，且已由本 mission 的
> `hostchain` 补上。

**补的那一条（本 mission 的 task 2）**：新增 `hostchain` 用例（kernel `sliceMode=2`）——
**host 声明输入直入 S3，device 跑通 S3→S7（S6 在 AIC）**；参考侧整链无任何设备锚（S6 的输入取
**参考自己的** `opin`、S7 的 fp32 残差取 host 自己算的 `res1`）。它把 host 声明输入驱动的链从
`slice` 的 S3→S5 **延伸到 S3→S7**，覆盖「S2 出口 → S7」这条组合链上的每一个 bf16 出口
（S2 出口值 → S3 → S4 → S5 出口 → S6 出口 → S7 出口）。`slice` 用例保留，并把它原来的
`slice qkvzba = host 输入` 换成了**非恒真**的物质见证 `device 消费的 q|k|v == host 输入字节`
（旧判据两侧取同一条 host 指针 = 恒真项，属自指形态 S4；见 §5.6）。

**做不到的那一条，明确披露（缺什么、挡住什么，一条都不省）**：

* **缺哪个输入**：参考缺**设备 S2/S6 在 bf16 舍入之前的那个 fp32 L0C 累加值**。实测：host 侧
  double 累加 → RNE bf16 与设备 `qkvzba` 逐位一致 **16478/16480**，其余 **2 个元素差 1 bf16 ulp**；
  把 fp32 累加换成任何可枚举的次序（顺序累加 / 按 64 / 128 / 160 / 256 分块）都不变成 16480/16480
  ⇒ **设备的舍入决策不可复现**。
* **因此做不到哪条判据**：「从 host 声明输入出发、参考在 S2 出口用**参考自己的**舍入值续算」的
  **端到端数值判据**做不到。族 P 的实测（`evidence/accept_run.log` 的诊断段）：`chain` 档
  `S3 k` maxRel **5.05e-3**、`S4 o` maxRel 0.2999、`S4 ssm_state` maxRel 1.005、`S6 opout`
  maxUlp **117**、`S7 y_final` maxUlp **128**；`chain2_s1` 档 `S3 k` maxRel 2.746e-2、
  `S4 ssm_state` maxRel 1.125、`S7 y_final` maxUlp 7。**全部远超**网格/1e-5 容差 ⇒ 把它当判定项
  等于"对不可复现的量设阈值"，故只作报告项（`check_ref.py` 的 `P:` 行）。
* **它挡住的是哪类错误**：**设备在 bf16 出口上「≤1 ulp 的规则级差」**——例如把 RNE 换成截断（RZ），
  或改了 tie 规则。这类差在**每一条**现有判据上都 PASS：网格判据只量位型距离的**大小**（≤1 ulp 全
  放行），族 D 的每一条都 re-anchor 到设备自己的出口字节，族 P 又不可能设阈值。这一点有**负向对照**
  实测：`check_ref.py` 的「咬合自检 ②」把全体元素统一 ±1 ulp ⇒ 网格判据仍 PASS（自检 ① 同时证明
  这条判据改 2 ulp 就咬得住，即它**活着**、只是盲区恰好在 1 ulp 以内）。
  > **这不等于说设备错了**——正相反，实测逐位一致率（S2 99.99% / S5 99.98% / S6 99.92%）说明设备的
  > 舍入**就是** RNE。这里说的是**判据没有咬合力去证明它**。
* **能补到什么程度**：`chain` 的 `①S2`（网格判据）咬住「S2 出口 >1 ulp 的错」，族 D 的每段咬住
  「给定设备上游字节时本段的错」，新增的 `hostchain` 咬住「**组合路径**」——`hostchain` 把整条
  S3→S7 放在 host 声明输入之下（其间不再 re-anchor），因此若下游任何一段的算术/布局/出口舍入错到
  **可观测**（>1 ulp 级），它会 FAIL。⇒ 咬不住的收窄到**一格**：**设备在某个 bf16 出口上恰好
  ≤1 ulp 的规则级差**。

**命令-输出自检表（本节「复核出处」的三条命令，2026-09-26 在仓库根实跑）**：

| 命令（仓库根目录） | 实跑结果 |
|---|---|
| `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc \| grep -n 'tensor=qkvzba file='` | rc=0，**2 命中**（`:2839` mode 1、`:2842` mode 2） |
| `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc \| grep -n 'H_WsTensorLine("qkvzba"'` | rc=0，**1 命中**（`:2837`） |
| `git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc \| grep -n 'hostQkv'` | rc=0，**2 命中**（`:2494` `/` `:2499`） |

> 这三条就是本节论证的全部依据。`acf3f69` 是不可变 commit ⇒ 行号随之冻结；若日后 `.asc` 重排，行号会
> 变而**符号名不变**（`H_DumpStep` / `H_WsTensorLine` / `H_RefChainStep`），以符号为准。

### 5.3 结果（`evidence/accept_run.log`：**138** 条判定项 PASS / 0 FAIL；另有 11 报告 + 33 诊断 + 3 段头）

> **计数口径订正（顺带抓到的一处旧虚高）**：`grep -c PASS accept_run.log` 给 **139**，但其中 1 条是
> 尾声 banner `===== ALL PASS =====`（也含 `PASS`）——**真正的判定项是 138**。旧版 README/§8 写的
> 「114 条」同样含 banner ⇒ 当时的真实判定项是 **113**。本表按真实判定项计数（每 case 逐条数出：
> `chain` 31 / `slice` 20 / `chain2_s0` 31 / `chain2_s1` 31 / `hostchain` 25 = 138）。

下表是**五个 case-step（chain / slice / chain2_s0 / chain2_s1 / hostchain）逐项取最差**后的结果
（括号内是最差档；数字由 `evidence/accept_run.log` 直接提取，可复核）。第二列是**族 D 或 H 的 ①**
（`chain` 档为 `(分段, 设备锚)`，其余档为 `(host 全链)`），第三列是**② 逐段隔离**——两列都是判定项。

| 输出 | 判据 | ① 参考链最差（族 D 或 H） | ② 逐段隔离最差 |
|---|---|---|---|
| `res1`/`res2` | fp32 **逐位** | 逐位一致（`chain`/`chain2`；`hostchain` 的 res2 是**报告项**，见下） | 逐位一致（五组） |
| `conv_state` | bf16 **位级** | 位级一致 30720/30720（五组） | 位级一致 + 移位自洽 |
| `x_norm` | bf16 网格 ≤1 ulp | max ulp 0，bit 一致 100% | 同左 |
| `qkvzba`(S2 出口) | bf16 网格 ≤1 ulp | max ulp 1，bit 一致 **99.9818%**（chain2_s1） | 同左 |
| `opin`(S5 出口) | bf16 网格 ≤1 ulp | max ulp 1，bit 一致 99.9837%（chain/hostchain） | max ulp 0，100%（五组） |
| `opout`(S6 出口) | bf16 网格 ≤1 ulp | max ulp 1，bit 一致 99.9219%（chain） | 同左；`hostchain` 档 99.9609% |
| `y_final`(S7 出口) | bf16 网格 ≤1 ulp | max ulp 1，bit 一致 99.9609%（**hostchain**）；其余 0/100% | max ulp 0，100%（五组） |
| `S3 q` | fp32 rel 1e-5+1e-6 | 容差占用 0.006（slice） | 0.006 |
| `S3 k` | fp32 rel 1e-5+1e-6 | 0.013（slice/chain2_s1） | 0.013 |
| `S3 v` | fp32 rel 1e-5+1e-6 | **0.095**（chain2_s1） | 0.095 |
| `S3 g` | fp32 rel 1e-5+1e-6 | **0.155**（chain2_s1，log 域 + `exp` 近似） | 0.155 |
| `S3 β` | fp32 rel 1e-5+1e-6 | 0.008（chain） | 0.008 |
| `o`(S4) | fp32 rel 1e-5+1e-6 | **0.027**（chain2_s1） | 0.016（chain2_s1） |
| `ssm_state` | fp32 rel 1e-5+1e-6 | **0.332**（chain2_s1） | **0.187**（chain2_s1） |
| `ssm_state` 位级 | **报告项** | bit 一致 34.80-44.55%，≤1 ulp 77.09-84.22% | bit 一致 64.41-69.08%，≤1 ulp 96.97-98.60% |
| `res2`（`hostchain` ①） | **报告项** | 位型不同 1/2560 —— 参考侧 `opout` 只到 bf16 网格 ⇒ 这个 fp32 和不可能复现；S7 的**算术**精确性由 ② 的 `S7 res2 (隔离, fp32)` 逐位判据给出 | 逐位一致 2560/2560 |

最差档（`chain2_s1`）出现在第二次 kernel 启动，状态已递推过一次；**全部判定项仍有 ≥3× 余量**
（最紧的是 `ssm_state` 占用 0.332）。`hostchain` 档的 `y_final` 出现 max ulp 1 / bit 99.96% 是**预期**
的：参考该档用**自己的** `opout` 续算 S7（不 re-anchor），两者只到 bf16 网格。

**零回归对照（改前 = M48 归档，改后 = 本次复跑）**：

| | 改前 | 改后 |
|---|---|---|
| `accept_run.log` 判定项 PASS / FAIL | 113 / 0（8 报告 + 42 诊断 + 4 段头） | **138 / 0**（11 报告 + 33 诊断 + 3 段头）；banner 自述 `ALL PASS（判定项 138）` |
| `accept_run_stagelimit.log` 组合数 / **判定项** / FAIL | 8 / 155 / 0（旧表把 8 条 banner 也写进 PASS 行） | **15 / 254 / 0**；其中 **13 个组件有判定项且全 PASS**，另 **2 个组件 `hostchain` @ `stage_limit=1,2` 判定项为 0** ⇒ 单独标 `NO-CRITERIA（判定项 0：无判定力，不发合格证）` 且 **`rc=2`**，**不进 PASS 计数** |
| `check_ref_run.log` 判定项 / 报告项 / 未覆盖 | 103 / 8 / —（无一栏） | **139 / 25 / 8** |
| 判定项的**判据值**（不是标签） | — | 既有项**逐项未变**（同一份 device 字节；标签按 §5.2 重命名） |
| 已发布 `*.bin` / 各 case 判据 `.txt` | — | **逐字节未变**（sha256 逐一相同，见 `dump_manifest.md` §3 与 §3.1） |

> 计数变动的来源是**新增**的 `hostchain` 档（kernel +25、numpy +26 判定项）与新增的 host 输入
> **报告项**（kernel +3、numpy +17），以及 `slice` 档 ① 不再打与自身重复的传播诊断（`Rp` 与 `R`
> 同源 ⇒ 诊断 −11 条）；**没有删除任何既有判据的判据值**（`chain`/`slice`/`chain2` 三个档的
> 判定项数逐 case 未变：31/20/31/31），失败计数亦为 0。

> **「0 判定项」不发合格证（`79e085b` 起生效）**：旧口径把 `accept_run_stagelimit.log` 的 15 条 banner
> 也写成「15/15 ALL PASS / 269 PASS 行」，而其中 `hostchain` @ `stage_limit=1,2` 两个组件**一条判定项
> 都没产出**（`stage_limit<3` ⇒ ①S3 起全不产出；mode 2 无 S1/S2 判据）——那是「根本没检查」，不是
> 「检查过并通过」。现在两侧都按三态收尾：kernel 的 `main()` 与 `check_ref.py` 的 `Stat.banner()`
> 在**判定项为 0 时打印 `NO-CRITERIA` / `RESULT: SKIPPED` 且返回 `rc=2`**，banner 一律自述判定项数
> （`ALL PASS（判定项 138）`）。**负向对照已实测两次**（可复跑的命令写在 `dump_manifest.md` §1(f)）：
> kernel 侧把 `M14_STAGE_LIMIT` 设为 `2` 跑 `slice` ⇒ `NO-CRITERIA（判定项 0）`、**`rc=2`**；
> 同一批 dump 喂 `check_ref.py` ⇒ `RESULT: SKIPPED（判定项 0…不发合格证）`、**`rc=2`**。

`ssm_state` 的位级一致率不是 100% 是**预期且不可消除**的：host 参考只能用 float64 作 oracle，
而设备在 fp32 里做递推，且 `e^g` 来自向量 `Exp` 指令的近似（与 host `exp` 差 ~1 ulp），
每个 value head 的 128×128 slab 逐行乘 `e^g` 会把这点差异放大到低位。**conv_state 才是真的位级可验**
（纯 bf16 搬移），已 100% 位级一致；`ssm_state` 用 1e-5 相对容差判定（**最差占用 0.332**）+ 位级统计披露
——占用 0.332 意味着对 |exp| ≫ 1e-6 的元素其相对偏差 ≤ ~3.3e-6（fp32 精度量级）。
若后续要求 `ssm_state` 逐位，需要内核额外 dump `e^g` 并确定向量 `ReduceSum` 的归约树次序（后续项，不在本 mission）。

### 5.4 numpy 交叉校验（`evidence/check_ref_run.log`）

`check_ref.py` 不共用任何 C++ 代码，从 **dump 的原始字节**重算整条 S1→S7（float64 归约、fp32 逐次
乘法、RNE bf16 编码），判据与 host 同口径：**139/139 判定项 PASS + 25 报告项 + 8 未覆盖（跳过）**，
退出码 `0`。同族的判据值与 kernel 内建判据逐项吻合（如 chain 档 `S3 q` 0.003 / `S3 g` 0.077 /
`S4 ssm_state(隔离)` 0.052 与 `accept_run.log` 的 chain 档一致；族 P 的纯参考链诊断也吻合：
`chain` 档 `S7 y_final` maxUlp 128/bit 96.2891%、`chain2_s1` 档 maxUlp 7/bit 98.4766%，两边同值）。

> **「同源转录」声明（M62 订正）**：这两份实现**不是两个独立来源**——`check_ref.py` 的 numpy 参考
> （`grep -c 'def ref_' check_ref.py` = 5：`ref_norm`/`ref_gemm`/`ref_prolog`/`ref_recur`/`ref_gated`）
> 与 `.asc` 的 host C++ 参考（`grep -c 'static void H_Ref' m14_gdn_layer.asc` = 7，其中
> `H_RefNorm`/`H_RefGemm`/`H_RefProlog`/`H_RefRecur`/`H_RefGated` 是五个算件参考，另两个是
> `H_RefAlloc`/`H_RefChainStep`）是
> **同一作者按同一份读法的两次转写**（与各自 donor 的关系同理）。故两者一致只排除「转写笔误」，
> **不构成规则级的独立见证**；规则级的独立性由 §5.6 的外部 pin 承担。（旧版本本节写「两份独立的
> 独立实现」，属过度声明，已改。）

覆盖范围**三态分栏**（`RESULT: OK（判定项 139，报告项 25，未覆盖 8）`）：未覆盖栏由与判定门控
**同一份条件**产出（`Stat.gate()` 既门控又记「未覆盖」），不是手写数字；当前 8 条是
`chain`/`chain2` 的「①S2 出口自洽（仅 hostchain）」与 `slice` 的「①S1/S2、①S6、①S7（device 不跑）」
等模式相关项。**一条负向对照**（`咬合自检`）：① 篡改 1 个元素 +2 ulp ⇒ 网格判据 FAIL（判据是活的）；
② 全体元素统一 ±1 ulp ⇒ 网格判据**仍 PASS** ⇒ 定量给出 §5.2 里「咬不住 ≤1 ulp 的出口规则差」那一条。

另有内部自洽性检查（不依赖参考）：`conv_state` 移位自洽（`new[0]=old[1]`、`new[1]=old[2]`、
`new[2]=x[q|k|v]`）与 mode 1/2 下「device 消费的 q|k|v == host 输入字节」。

**判据非空洞（开发期实证，未放宽任何阈值）**：本 harness 在开发中两次给出确定性 FAIL 并精确定位问题——
(a) 不做 bf16 锚定时 `S7 res2` 报 2/2560 位型不同（差异源即 `opout` 的 1 ulp bf16 网格差，定位后
按 §5.2 的口径改成锚点续算；M62 的 `hostchain` 档又遇到同一现象，这次**如实降为报告项**而不是再
加一个锚点）；(b) dump 的 `case=` 一度误写成 `tag=`，使 numpy 侧把 `chain2_s1` 当成独立 case、用
初值 state 而非上一步设备 state，`check_ref.py` 立刻全段 FAIL（修正 dump 元数据后全 PASS）。
两处都是判据自己发现的，说明这些比对确实在起作用。

### 5.5 证据归档（`m14_gdn_layer/evidence/`）

| 文件 | 内容 |
|---|---|
| `accept_run.log` | 四个 case 的 kernel 内建判据全量日志（**138 条判定项 PASS、0 FAIL**，rc=0；`grep -c PASS` 为 139，多出的 1 条是尾声 banner；另 11 报告 + 33 诊断 + 3 段头） |
| `accept_run_stagelimit.log` | 段序截断矩阵：`M14_STAGE_LIMIT=1..7 × {chain, hostchain}` + `=7 × slice`，**15 组件 = 13 个有判定项（254 条，全 PASS、0 FAIL、rc=0）+ 2 个 0 判定项**（`hostchain` @ `stage_limit=1,2`：`NO-CRITERIA` + **rc=2**，不进 PASS 计数）（无挂死、门控一致） |
| `check_ref_run.log` | `check_ref.py` 的 **139 判定项 + 25 报告项 + 8 未覆盖**（三态汇总 + 咬合自检）；判定项为 0 时打 `RESULT: SKIPPED` 并 `rc=2`（负向对照见 §5.3 末注） |
| `dump_manifest.md` | 二进制/源码 sha256 + **38 个可复现 dump 文件**的 sha256 与字节数 + 再生成命令 + 确定性证据；§3.1 = 改前/改后 sha 对照；§6 = `*_ws.bin` 不可复现的逐字节定位（M48） |
| `ws_never_written_check.py` | §6 的定位脚本：把 workspace 逐字节分成「内核语义写 / g·β 槽尾 / 从不写」三类并核对多次 dump 的差异落点（M48；三态退出码） |

dump 原始字节 ~150MB/次未入库（kernel 已实测确定性），按 `dump_manifest.md` 的命令可复现并逐文件核对。

### 5.6 外部 pin、每条判据的「咬得住 / 咬不住」，与反自指自检（M62）

**(a) 规则的出处（外部 pin）** —— 参考的**规则**必须钉在**不是本设备实现**的工件上，并在代码里写出
`文件:符号`（`check_ref.py` 文件头有同表）：

| 段 | 规则 | 外部 pin（`文件:符号`） | 强度 |
|---|---|---|---|
| S1/S7 | Add+RMSNorm（fp32 残差；NR rsqrt） | `ops-nn/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h`::`CalculateXAdd` / `CalculateSquareReduceSum` / `ComputeRstdNewtonRaphson` / `CalculateY` | 仓外官方算子（N2） |
| S2/S6 | `C = round_bf16(A·Wᵀ)`，**舍入 = RNE（CAST_RINT）** | **规则源（N2）**：`asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.md` §`quantPre` 的 `QuantMode_t::F322BF16 // Float32_2_BFloat16，cast mode 为 CAST_RINT 模式`（`DataCopy_L0CToGM.md` 同表）；旁证 = 官方算子 `ops-nn/conv/common/op_kernel/arch35/conv_instr_impl.h`::`GetQuantPreFp32`（`OutputT=bfloat16_t ⇒ QuantMode_t::F322BF16`） | **规则源 N2；实现触点/结构样例不算**（见下） |
| S3 | conv1d(K=4)+SiLU；q/k l2norm（仅 q ×1/√128）；g/β | `ops-transformer/mamba/causal_conv1d`（arch35 环形 conv）；`vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py`（softplus/gating）；`vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`（l2norm 与 q 的 1/√128） | 仓外官方实现（N2） |
| S4 | 递推 decay→delta→outer→matvec | `ops-transformer/attention/recurrent_gated_delta_rule`（arch35 `vf_vec_mul_mat.h` / `vf_outer_add.h`） | 仓外官方算子（N2） |
| S5 | per-head RMSNormGated（`norm_before_gate=True` + sigmoid） | `vllm/vllm/model_executor/layers/layernorm.py::RMSNormGated`；`output_gate_type="sigmoid"`（checkpoint `config.json`） | 仓外官方实现（N2） |

> **S2/S6 那一行的三种角色要分清（`79e085b` 订正）**：①**规则源**（N2）= 上面那份仓外 API 文档里
> `QuantMode_t::F322BF16` 的 `CAST_RINT` 定义 —— 它是「RNE 舍入」这条规则**可独立复核**的出处；
> ②**实现触点** = 本仓 `m14_gdn_layer.asc` 的 `Cube::Bf16Gemm::CopyOut` 里 `fp.quantPre =
> QuantMode_t::F322BF16`（**同源转录**：它只是把规则抄下来，不算独立来源）；③**结构样例** =
> `asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance/mmad.asc`
> 的 `KernelMmad`（`AscendC::Mmad`/`AscendC::Fixpipe`）与仓内 `m1_mxfp4_gemm` 的流水结构 —— 它们是
> **教学样例/流水骨架**，不是算子规格，**不计入 N2**。
> （旧版本这一行写「`ops-nn` matmul donor（m11；见 `m11_bf16_gemm/README.md` 的 donor 表）」——
> 那个条目**不存在**（`m11_bf16_gemm/README.md` 的 donor 段只列 `m1_mxfp4_gemm` 与 asc-devkit 的
> `mmad.asc`），属不可复核的指针 + 过度声明，已改。）

**同源转录（correlated transcription）显式声明**——下列关系**不得**被写成「两处独立」：

1. `check_ref.py` 的 numpy 参考 ↔ `.asc` 的 host C++ 参考（`H_Ref*`）：同一作者、同一份读法的两次转写 ⇒ 二者一致只排除转写笔误；
2. 本模块的 host 参考 ↔ 它复制改造的 donor 核（m6/m9/m4/m12）：**同源转录**，故 donor 那边的语义错会被继承；
3. §5.6(a) 的外部 pin 是**唯一的规则级独立来源**；m6/m9/m4/m12 各自 README 里引的同一批 pin 与本节同源。

**(b) 每条判据咬得住什么 / 咬不住什么**（自指形态按 M59 的 S1/S2/S3/S4 代号标注）：

| 判据（符号/标签） | 咬得住 | 咬不住 | 自指形态 |
|---|---|---|---|
| `①S1 res1 / x_norm (host 全链)` | S1 的 fp32 残差（逐位）与 x_norm 的 RNE bf16 舍入；输入被改必 FAIL | S1 之后的任何东西 | 否（N1 声明输入 + N2 pin） |
| `①S2 qkvzba (host 全链)` | S2 的 GEMM 输出与 host 参考的 bf16 网格距离 ≤1 ulp；>1 ulp 的错必 FAIL | 出口舍入**方向**（≤1 ulp 内）；S3 之后 | 否（同上） |
| `①S3..S7 (分段, 设备锚)`（`chain`） | 给定**设备自己的** bf16 出口字节时，每段的规则与 bf16 边界 | 上游系统误差；**组合路径**；出口舍入方向 | **S1**（设备产物入参）。**合法性**：`qkvzba` 出口另有 `①S2`（族 H）覆盖 ⇒ 条件成立；`opin`/`opout` 出口**在 `chain` 这一份 dump 之内**没有任何 host 输入判据覆盖 ⇒ 这两处的 S1 为**未被覆盖的**自指。**跨用例看**，`hostchain` 档的 `①S5`/`①S6`/`①S7` 就是覆盖这两个出口的 host 声明输入判据——但那是**另一次运行**，不能给这份 dump 免票（`79e085b` 订正：原句漏了「同一 dump 内」这个限定） |
| `①S3..S7 (host 全链)`（`slice`/`hostchain`） | host 声明输入下的整条组合链（含 S5/S6/S7 三个 bf16 出口）；下游任何**可观测**（>1 ulp 级）的算术/布局/舍入错 | 设备**自己** S2 出口的舍入决策（device 不跑 S2）；≤1 ulp 的出口规则差 | 否（N1 + N2 + N3） |
| `② 逐段隔离` | 「给定设备上游字节，本段算得对不对」；单段错误定位 | 上游系统性错误；组合 | **S1**（同上，`chain` 档） |
| `conv_state 原位更新 / 移位一致性` | 原位左移的**位级**正确性 + 设备消费的字节 | 上游 q/k/v 的值对不对 | 否（N4 结构性） |
| `device 消费的 q\|k\|v == host 输入字节`（mode 1/2） | 设备**真的**按 host 字节消费（落盘见证） | 该字节值对不对（那是 ① 的事） | 否（N4）；**M62 订正**：旧版本两侧取同一条指针 ⇒ 恒真，属 **S4**（同一份值算两遍） |
| `P: 纯参考链（报告项）` | 只给数，不判定 | — | **N4**（不可复现，无咬合力） |
| `ssm_state 位级统计 / res2 报告项` | 只给数，不判定 | — | **N4** |

**(c) 反自指自检（每条判定项都能回答「若本算件有规则级错误，它会不会 FAIL」）**：
`check_ref.py` 的 `咬合自检` 与 §5.2 的「做不到的那一条」合起来给出这个答案——**唯一答不出「会 FAIL」
的格是「设备在 bf16 出口上恰好 ≤1 ulp 的规则级差」**，它以负向对照（全体 +1 ulp 仍 PASS）被定量
钉在案上，而不是留给读者去推。其余每一格都有实测的 FAIL 侧（自检 ①：+2 ulp 即 FAIL）。

## 6. 已知限制 / 后续

1. **只有 decode m=1**：`m9`/`m4` 本身是 m=1 核；prefill（BT=64 chunk 扫描 + `(I+L)⁻¹` 块前代）
   是另一套实现（docs/10 §3 四点改造），不在本 mission。host 侧 `m` 参数虽在签名里，但只接受 1。
2. **段间串行、无跨段预取**：段边界一律 barrier 分隔（骨架正确性优先）。in_proj 权重 84.4MB 在
   m=1 时是纯带宽 bound（N 方向 103 tile 条带 28 AIC），docs/12 §4 的 L1 预取滚动窗
   （@330KB，BufferID 7-10 第二组 ping/pong）**留位未用**——与相邻段重叠预取是下一步主要收益点。
3. **无性能测量**：未接 msprof；验收命令整轮 wall ~7s（含 host 生成 116MB 权重与 4 次 116MB H2D），
   单 kernel 时间未单独测量。README 的定位是正确性 + 同步纪律。
4. **GEMM 单核 tile 结构未改**：m11 的 `BASE_M=64/BASE_K=64/BASE_N=160` 与三级流水原样保留，
   只把 N 循环提出来做条带划分；`m>64`、K/N 尾块仍不支持（静态断言）。
5. **m9/m4 的段内划分粒度未调优**：m9 的 80 个 block 按 `b = bid, bid+56, ...` 静态条带，
   56 AIV 下 24 核 2 块、32 核 1 块（donor 已知负载不匀）；m4 的 48 head 每 AIV 1 个（8 个 AIV 空闲）。
   正确性优先，未做负载均衡。
6. **task 4「state 位级校验」的口径（已经 tower 裁决为有意识 descope）**：
   - `conv_state`：**完全达成**位级校验——四组 case-step 均 30720/30720 位级一致，另有
     `new[0]=old[1] / new[1]=old[2] / new[2]=x[q|k|v]` 自洽性检查（`check_ref.py` 同款）。
   - `ssm_state`：**经 tower 裁决降级为「1e-5 相对容差 + 位级统计披露」口径**，理由是逐位不可复现
     （device 为 fp32 递推、`e^g` 来自向量 `Exp` 近似、`Reg::Reduce` 归约树次序未文档化；host 只能用
     float64 作 oracle）。实测：容差占用最差 0.332（chain2_s1）/ 隔离 0.187；位级一致率全链
     34.80-44.55%、隔离 64.41-69.08%（≤1 ulp 分别为 77.09-84.22% / 96.97-98.60%）。
   - 若后续要求 `ssm_state` 逐位，需要内核额外 dump `e^g` 并确定向量 `ReduceSum` 的归约树次序（后续项）。
7. **A_log/dt_bias 按 `[64]` 传入**（48 有效 + 尾零），与 m9 donor 契约一致；若上游只给 `[48]`
   需补齐或改偏移。
8. **`__simd_vf__` 寄存器代码的 UB 布局耦合**：m9 的 gating 用「对齐整 64-lane 加载 + one-hot 掩码
   Reduce 抽 lane」规避 3510 的 `LoadAlign` 32B 对齐 quirk，这部分依赖 `UB_PL_AST/BST` 的 slack
   字节（见 `m14_resources.h` 注释），改动该窗布局需同步复核。
9. **「设备在 bf16 出口上恰好 ≤1 ulp 的规则级差」没有判据**（M62 的诚实边界，理由与实测见 §5.2 末条、
   逐条见 §5.6(b)）：从 host 声明输入出发的**端到端数值判据做不到**——参考缺设备 S2/S6 在 bf16 舍入
   **之前**的 fp32 L0C 累加值（实测 double 累加 → RNE 只在 2/16480 个元素上与设备差 1 ulp，且换任何
   可枚举的 fp32 累加次序都不复现），而 conv 近相消把这点差放大到下游（`chain` 档 `y_final` maxUlp 128）。
   已补的 `hostchain` 用例把 **S3→S7 的组合链**放在 host 声明输入下跑完（覆盖 S5/S6/S7 三个 bf16 出口），
   并给了负向对照（`check_ref.py` 的咬合自检 ①②）——所以缺口只剩「≤1 ulp」这一格，不再是"不知道有没有咬"。

## 7. 文件

| 文件 | 说明 |
|---|---|
| `m14_resources.h` | 全局静态资源表（形状常量 / AIC·AIV BufferID / flagId / UB 四段窗 / L1 / GM workspace 偏移与生命周期） |
| `m14_gdn_layer.asc` | 单一 `__mix__(1,2)` kernel（S1-S7 段序 + 同步 + 各段算件）+ host（确定性输入生成、double 参考链、判据、dump）；case = `chain`/`slice`/`chain2`/`hostchain`（`sliceMode` 0/1/2） |
| `check_ref.py` | numpy 交叉校验（消费 `M14_DUMP=1` 的 dump 与 manifest；**139 判定项 + 25 报告项 + 8 未覆盖**，三态退出码 0/1/2；与 kernel 内建参考是**同源转录**，见 §5.6） |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`，`-ffp-contract=off` 保证 host 参考与 kernel 同 IEEE 序列） |
| `evidence/` | 验收证据归档（见 §5.5） |

## 8. 向量 API 合规与改动记录（M48，历史）

> 本节是 **M48 当时的历史记录**。其中的下列计数与措辞都**已被 M62 取代**，逐项列出以免误引：
> `accept_run.log` 的 `grep -c PASS` **114**（含尾声 banner，真实判定项 113）、
> `accept_run_stagelimit.log` 的 **8 组合 / 163 PASS 行**（未扣 banner、未区分 0 判定项档）、
> **103 条 numpy 判定项 + 8 报告项**（无「未覆盖」栏）、**33 个可复现产物**、
> 以及「两份**独立**实现」这个措辞。
> 现行口径：kernel 判定项 **138**、矩阵 **13 个有判定项组件（合计 254）+ 2 个 0 判定项（rc=2）**、
> numpy **139 / 25 / 8**、可复现产物 **38 个**、参考与 kernel 内建参考是**同源转录**（非独立来源）。
> 现行口径见 **§5.2 / §5.3 / §5.5 / §5.6 / §9**。

### 8.1 `__simd_vf__` 裸调用合规（tower 裁定①）

本文件的寄存器 VF 算件分两种调用形态：`PrologBlock` 在 `__VEC_SCOPE__` **内**调用
（`GdnProlog::ComputeBlock`）；`EgExpAll` 与 `GdnHeadRecurrence` 在 `__VEC_SCOPE__`
**外裸调用**（`GdnDecodeRec::Process` 与 `GdnDecodeRec::ComputeHead`）。按 tower 裁定①
（M44 §9.1 结案，判据原文见 `docs/18-vector-api-audit.md` 与 M48 mission 的 Context）：
**判据落在被调函数上**——必须 `__simd_vf__` 且函数体内只用寄存器 API；`__VEC_SCOPE__`
是词法入口、`__simd_vf__` 是编译器函数属性（`__clang_cce_defines.h` 里 `__simd_vf__` 的
定义，即 `__attribute__((cce_simd_vf))`：`:44`）；**不要求**调用点有词法 `__VEC_SCOPE__`，也**不要求**
`asc_vf_call`（CANN 上游用 `asc_vf_call<F>` 调 `__simd_vf__` 时调用点同样不在
`__VEC_SCOPE__` 内，本仓 `asc_vf_call` 命中 0）。⇒ 两种形态均合规，**代码不动**；
`m14_gdn_layer.asc` 头部「寄存器 VF 计算」段注释已按本裁定更正（原写「全部在
`__VEC_SCOPE__` 内调用」，与上面两个裸调用点的事实不符）。

设备段共 **9** 处 `__VEC_SCOPE__`（与 M44 §3.8 记的 9 个段一致），逐个落在（**以符号为准**，
行号随代码变动，仅作查阅提示）：`NormDonor::CalculateSquareReduceSumCommon`(:154)、
`NormDonor::CalculateSquareReduceSumLessThanVL`(:217)、
`NormDonor::CalculateSquareReduceSumLessThanTwoVL`(:238)、
`NormDonor::ComputeRstdNewtonRaphson`(:350)、`NormStage::CalculateXAdd`(:428)、
`NormStage::CalculateY`(:457)、`GdnProlog::ComputeBlock`(:849)、
`RmsNormGatedStage::CalculateGateY`(:1284)、`PrecastGammaN`(:1617)；另有 4 处出现是在本次新增
的段注释里，只作文字引用、不是设备代码。`GetValue`/`SetValue` 与经典
`Cast(`/`Sort32`/`MrgSort`/`Reduce*`/`Transpose`/`Concat`/`Gather` 等 memory-based 计算
调用**均 0 命中**，计算路径全部落 `RegTensor` + `AscendC::Reg::`（`Cast<` 的 12 处全部
操作 `RegTensor`，靠 `using namespace AscendC::Reg` 解析到 `Reg::Cast`，无经典
`LocalTensor` 版）。`DataCopy`/`DataCopyPad`/`LoadData`/`Mmad`/`Fixpipe`/BufferID/CrossCore/
`LocalMemBar` 属搬运/矩阵/同步类（`docs/05` §6.1 明文允许）。

### 8.2 本次改动的位级影响面（M48）

| # | 改动 | 性质 | 位级影响 |
|---|---|---|---|
| 1 | 段头注释「寄存器 VF 计算（`__simd_vf__`，全部在 `__VEC_SCOPE__` 内调用）」→ 按裁定①改写（逐调用点列出形态） | **纯注释** | 无 |
| 2 | `Block1()` 前注释：撤回「元素计数 DataCopy 的 `blockCount/srcGap/dstGap` 是未初始化栈垃圾，M16 实测踩雷」 | 纯注释（归因修正，`docs/05` §6.1「元素计数 DataCopy 重载」行；该归因已按源码核实） | 无（显式块参数本来就是**代码规范**，不是修 bug） |
| 3 | `CopyInBlock` 前注释：撤回「blockCount>1 走 NZ 格式重排」 | 纯注释（`docs/05` §6.3 #6：该重载本就是「块 + gap」拷贝语义，「NZ 重排」表述已删） | 无 |
| 4 | UB 布局段注释（`PAR_B`/`ST_OFF`/`X_OFF`/`W_OFF`/`B_OFF` 常量表上方）与 `CopyOutBlock` 前注释：同上，「规避 NZ 重排 quirk」→「单块语义最直接」 | 纯注释 | 无 |

**本模块没有任何代码（算子序列 / 参数 / 布局 / 同步）改动** ⇒ 位级影响面为空；
**不需要**同步更新任何参考（host double 参考链与 `check_ref.py` 均未动）。
`m15_layer_loop/m15_gdn_layer.h` 是 m14 ≤1623 行的逐行同源复制件，**本次 m14 改动全在
注释** ⇒ m15 无代码同步义务；m15 内同名段注释（`m15_gdn_layer.h` 里那个「寄存器 VF 计算」
段头，即本文 §8.1 被更正的同款句子）由 M40 负责。

**复跑证据（零回归）**：

| 判据 | 改前（归档） | 改后（本次复跑） |
|---|---|---|
| kernel 内建判据 `accept_run.log` | 114 PASS / 0 FAIL | **114 PASS / 0 FAIL**，日志与归档**逐字节一致** |
| 段序截断矩阵（`M14_STAGE_LIMIT=1..7 × chain` + `=7 × slice`） | 8/8 ALL PASS | **8/8 ALL PASS**（163 条 PASS 行 / 0 FAIL） |
| `check_ref.py` numpy 独立校验 | 103/103 判定项 + 8 报告项 | **103/103 + 8**，日志与归档**逐字节一致** |

**A/B 位级对拍**：把 HEAD（未改）的 4 个 m14 源文件复制到一个临时目录单独构建并
`M14_DUMP=1` 落盘，与本次改造后的二进制落盘结果逐文件 sha256 比对：**33 个可复现产物
（16 参数 + `*_cs_out.bin`/`*_ssm_out.bin`/`*_meta.txt`/各 case 判据 `.txt`）+ 两份判据日志
逐字节一致**；4 个 `*_ws.bin` **不参与** sha 判据（其 sha256 不是内核输出的性质 —— 见下 ⚠ 与
manifest §6，同一二进制的两次相邻运行也可能差数十字节）。**更强的证据是 `binary sha256` 与改动前
逐字节相同**（manifest §2）⇒ 本次改动对设备输出零影响。

> ⚠ **已定位的一处归档不一致**（`*_ws.bin`；M48 轮次里补做的定位，**不是**本 mission 引入，
> 但已把它查清并留明账）：本次复跑的 `chain/slice/chain2_s0/chain2_s1` 四个 `*_ws.bin`
> （`H_DumpStep` 落盘的**整段** 5,258,240 B workspace）sha256 与
> `evidence/dump_manifest.md` §3 记录不同，其余 33 个文件（含全部被判定的
> `*_ssm_out.bin`/`*_cs_out.bin` 与全部判据日志）与清单**完全一致**。定位结论（已写入
> manifest **§6**，附可复跑脚本 `evidence/ws_never_written_check.py`）：
> ① ws.bin 中内核语义写过的只有 **147,008 B（2.80%）**，另有 2,688 B 是 g/β 32 B 槽尾
> （32 B 单块 store 把 UB 槽 `[h][1..7]` 的残留一并写出），其余 **97.15% 内核从不写**；
> ② 6 次运行（含 A/B 对照与**扰动历史**运行）两两比对：**差异 100% 落在那 2,688 B 槽尾、
> 0 字节落在内核语义写区** ⇒ 这 4 个文件的 sha256 本来就不是内核输出的性质，归档值无法
> 事后复现（manifest §4 的"两次运行 37/37 一致"对 ws 只在同一设备/UB 历史下成立，M48 已
> 用相邻两次普通运行给出反例并订正该口径）；
> ③ A/B 对照中，未改动的 HEAD 二进制与本分支二进制产出**同一套可复现产物**（33 个产物 +
> 两份判据日志；4 个 `*_ws.bin` 不参与 sha 判据），且 manifest §2 的 `binary sha256` 与本
> mission 前**逐字节相同** ⇒ 与本 mission 无因果。
> **复核口径**：以 33 个可复现产物 + 两份判据日志为准；`*_ws.bin` 改按 manifest §6.3 的
> 脚本判据核验（比对"内核真正写过的字节"，比 sha256 更严格）。

## 9. M62：链标签分族、host 输入链补齐、外部 pin 与披露（本次改动）

M59 的自指审计（R7）对本模块的处置意见：「把『全链』改称『分段（设备锚）』，另补一条从 host 输入端
重算的组合判据」。M62 的执行与**对审计的一处订正**如下：

| # | 改动 | 文件 | 性质 |
|---|---|---|---|
| 1 | 判据按输入来源分三族（族 H / 族 D / 族 P），裸「全链」二字从标签里消失；`chain` 的 S3–S7 → `(分段, 设备锚)`，S1/S2 与 `slice`/`hostchain` 的 S3–S7 → `(host 全链)` | `.asc` 判据字符串 + `check_ref.py` 标签 + 本文 §5.2 | 口径/文字 |
| 2 | **订正 M59 的过度概括**：审计把本模块所有「全链」行都记成 S1，但 `chain` 的 S1/S2 与 `slice` 的 S3–S5 **本来就是** host 声明输入驱动的（`slice` 的 S3 输入是 host 参数文件） | 本文 §5.2 的「口径订正」段 | 口径 |
| 3 | 新增 **`hostchain`** 用例（`sliceMode=2`）：host 声明输入 → device S3→S7；kernel 侧把 AIC 参与度拆成 `runS1Inp`（mode 0）与 `runS6Out`（mode 0/2），mode 2 预置 `WS_RES1`（host 的 fp32 `res1`）并把 host 的 S2 出口灌进 `dev[13]`；参考侧 `H_RefChainStep` 对 mode 2 取 host 张量且**不传任何锚点** | `.asc`（device gating + host 链）、`check_ref.py`（`slice_mode==2` 分支） | **代码**（device 行为仅新增 mode，mode 0/1 逐字未变） |
| 4 | `slice` 的 `slice qkvzba = host 输入`（两侧同指针 ⇒ **恒真**，自指形态 S4）换成物质见证 `device 消费的 q|k|v == host 输入字节`（比 `conv_state[2]` 落盘字节，n=10240） | `.asc` + `check_ref.py` | 判据修实 |
| 5 | S7 的 `res2` 在 host 全链档降为**报告项**（参考侧 `opout` 只到 bf16 网格 ⇒ 这个 fp32 和不可能复现；S7 算术精确性由 ② 的逐位隔离项给出）；`chain` 档仍是逐位判定 | `.asc` + `check_ref.py` | 口径（不加锚点凑 PASS） |
| 6 | 族 D 的 S1/S2 与族 H 的 S1/S2 同值（S1 无上游）与 `slice` 档 ①/② 的 S3 同值——**在 §5.6 表里如实标注**，不再让它们看起来是两组独立见证 | 本文 §5.6 | 披露 |
| 7 | `check_ref.py`：补外部 pin 表 + 同源转录声明；三态汇总（判定 / 报告 / 未覆盖）由**同一份门控条件**产出；`咬合自检` 负向对照；退出码 `0/1/2`（`2` = `RESULT: SKIPPED`，不发合格证）；新增族 P 的 numpy 独立复算（`chain` 档 y_final maxUlp 128 / `chain2_s1` 档 7，与 kernel 诊断同值） | `check_ref.py` | 脚本 |
| 8 | `ws_never_written_check.py`：`CASES` 补 `hostchain`；docstring 里的裸行号引用改成符号引用 | `evidence/` | 脚本 |
| 9 | 证据重刷 + 改前/改后对照（§5.3 表与 `dump_manifest.md` §3.1） | `evidence/` | 证据 |
| 10 | **`79e085b`：5 条口径/计数修复**（读数见 §9.1）：① S2/S6 的外部 pin 从**不存在的**「`ops-nn` matmul donor」改成**真实存在**的仓外 API 文档规则源 + 官方算子旁证，并把「实现触点/结构样例」降级、不再标 N2；② §5.6(b) 的 `opin`/`opout` 格补上「**同一 dump 内**」这个限定（跨用例覆盖由 `hostchain` 提供，但那是另一次运行）；③ 「15/15 ALL PASS」改为 **13 个有判定项（254 条）+ 2 个 0 判定项**，kernel 与 `check_ref.py` 都加「0 判定项不发合格证 + `rc=2`」并与负向对照实测；④ mode 2 的物质见证不再沿用 `slice:` 前缀；⑤ `ws_never_written_check.py` 的 case 数改成由 `CASES` 推出（不再写死数字） | README / `check_ref.py` / `.asc` / `evidence/` | 口径 + 计数（含 `.asc` 标签字面量 ⇒ 指纹按 manifest §2 重刷） |
| 11 | **`c23f97a`：§2 指纹记账的两个现值改回实际 sha**（+ 当场核对命令与读数），另修两条引用形态 | `evidence/dump_manifest.md` + README §5.2/§8 | 口径/引用 |

§9.1 —— `79e085b` 之后的证据读数（与 `acf3f69` 的差别**只有口径与标签**，无数值结论变化）：

* kernel：`accept_run.log` 判定项 **138 / 0 FAIL**（banner 自述 `ALL PASS（判定项 138）`）；矩阵 **13 个组件有判定项（254 条，rc=0）+ 2 个 `NO-CRITERIA`（rc=2）**，无挂死；
* numpy：`check_ref_run.log` **139 / 25 / 8**，`rc=0`；
* **负向对照（0 判定项）实测**（命令见 `dump_manifest.md` §1(f)）：`M14_STAGE_LIMIT=2` + `slice` 在 kernel 侧 ⇒ `NO-CRITERIA（判定项 0）`、`rc=2`；同批 dump 喂 `check_ref.py` ⇒ `RESULT: SKIPPED（判定项 0…不发合格证）`、`rc=2`；
* `binary sha256` 因 `.asc` 标签字面量改动而**变化属预期**（manifest §2 已重刷并注明）；
* **已发布 `*.bin` 仍逐字节未变**（`79e085b` 的修复只碰 host 侧标签/文档；dumps 与 §3 清单逐一相同）。

**位级影响面与「不静默重写」的证据**：

* **已发布的 `*.bin` 与各 case 判据 `.txt` 逐字节未变**：38 个可复现 dump 文件里，除新增的
  `m14_param_xqkvzba_h.bin` 与 4 个 `hostchain_*` 文件、以及文本 `m14_params.txt`（多一行 manifest，
  `69d1c894…` → `644e8fd0…`）外，**其余 32 个既有文件的 sha256 与 §3 归档值逐一相同**（`dump_manifest.md`
  §3.1 列了逐行对照：same 32 / diff 1 / new 5）。
  参数生成未改任何 RNG 消耗路径 ⇒ 输入字节不变。
* **device 行为**：mode 0/1 的代码路径逐字未变（只把 `loopAic` 拆成 `runS1Inp`/`runS6Out` 两个恒等
  条件）；`M14_STAGE_LIMIT` 矩阵 15 组件**无挂死**（含 mode 2 的 1..7 全档），其中 13 个有判定项且全 PASS、
  2 个 0 判定项（`hostchain` @ 1,2）按 `79e085b` 起的规则标 `NO-CRITERIA` 且 `rc=2`。
* **未跑性能测量**（§6.3）；host 墙钟未用作任何证据。
