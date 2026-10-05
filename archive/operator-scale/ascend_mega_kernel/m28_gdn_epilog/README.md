# M28：GDN prefill epilog 的转位/落位段（`wsO` head-major → token-major `o`；z 段压实）

M148。M145 survey 钉死的缺口：全仓**未读到** `wsO`（head-major `[48,m,128]` fp32）→
token-major `o [m,6144]` fp32 的转位实现；m=1 时两者退化重合，m>1 才暴露
（`docs/22-prefill-prolog-epilog-wiring.md` §4.2 步骤 1、§3.3 G2-a）。

## 0. 收窄边界（先读）

本段**只做两件纯搬运**：

1. **o 转位**：`wsO [48, m, 128]` fp32（head-major）→ `o [m, 6144]` fp32（token-major）。
2. **z 落位**：把 in_proj 输出 `qkvzba [m, 16480]` bf16 的 z 段 `[10240, 16384)` 压实成
   连续的 `z [m, 6144]` bf16（= S5 `m12_rmsnorm_gated` 的 z 输入契约）。

**不做**（窄口径）：不串 S5/S6/S7，不接进 `m15_layer_loop` 的挂载点，不改 B1 的 `wsO` 契约、
不改 m12 的输入契约，**不主张端到端**。z 段本 mission 用**合成输入**驱动（来源契约见 §1，
生产者仍是未接项）。设备档只覆盖本段自身的位级判据。

> M149 续作在**新的链段 target** `m28_epilog_chain.asc` 上把「转位/落位 → S5 → S6 → `hcAttnOut`」
> 串了起来（见 §7）。`m28_gdn_epilog.asc` 本体与本节边界**不变**、仍单独可跑；M149 仍未接进
> `m15_layer_loop`、`z`/`wsO` 仍为合成输入、仍不主张端到端。

## 1. 契约（逐条带 `文件:行`）

### 1.1 `wsO`：head-major fp32 `[48, m, 128]`

- 字段声明：`LayerArgs.wsO`，注释 `[48, m, 128] fp32（段出口）`（`m15_layer_kernel.h:263`）。
- 存储式（B1 扫描段写出）：`oOff = hv*m_*GP_DV + t0*GP_DV`，`GP_DV = 128`
  （`m15_gdn_prefill.h:1089-1091`）。即元素下标 = `h*m*128 + t*128 + d`。
- 挂载点：`gp.out = A.wsO`（`m15_layer_kernel.h:710`）。
- `m` 是每 head 内 token 维长度（运行期），head 在最外层。

### 1.2 m12（S5）的输入契约：token-major `o` + `z`

- `o fp32 [m, 6144]` 行主序：`oGm.SetGlobalBuffer(..., M*HIDDEN)`（`m12_rmsnorm_gated.asc:398`）。
- `z bf16 [m, 6144]` 行主序：`zGm.SetGlobalBuffer(..., M*HIDDEN)`（`m12_rmsnorm_gated.asc:399`）。
- 常量 `HEADS=48 / HEAD=128 / HIDDEN=HEADS*HEAD=6144`（`m12_rmsnorm_gated.asc:72-74`）。
- 数学：`out = bf16(((o·rstd)·gamma)·sigmoid(z))`，per-head（128 维）RMSNorm
  （`m12_rmsnorm_gated.asc:4-26`）。⇒ 门控 sigmoid 的输入面就是本段落位的 `z`。

### 1.3 z 的来源：`qkvzba [m, 16480]` bf16 的 `[Z_OFF, Z_OFF+Z_DIM)` 段

- `IN_N = 16480 = q2048|k2048|v6144|z6144|b48|a48`（`m15_gdn_resources.h:48`）。
- `Z_DIM = 6144`（`m15_gdn_resources.h:52`）；`Z_OFF = V_OFF + V_DIM = 10240`
  （`m15_gdn_resources.h:51-57`）。
- decode 的 S5 就是从这个平面按 head 取 z：`zGm.SetGlobalBuffer(z, M_MAX*IN_N)`，
  取址 `zGm[Z_OFF + h*128]`（`m15_gdn_layer.h:1240,1259-1265`）。
  ⇒ z 段在源侧是**带行距 `IN_N` 的切片**，m12 要求**连续** `[m,6144]`，故需压实。

### 1.4 g/β 的来源（B1 扫描的另外两个输入，供接口完整性）

- g/β 不是本段产出，是 B1 的输入（`GdnPrefillArgs.g/beta`，`m15_gdn_prefill.h:94-95`）。
- 来源：qkvzba 的 b/a 段（`B_OFF=16384`、`A_OFF=16432`，`m15_gdn_resources.h:58-59`）经 S3
  计算：`g = −exp(A_log)·softplus(a+dt_bias)`、`β = sigmoid(b)`（`m9_gdn_prolog.asc:17-18`，
  读 a/b 于 `m15_gdn_layer.h:790-791`），输出 `g/β fp32 [48,8] stride-8`，仅 `[h][0]` 有效
  （`m15_gdn_layer.h:744-745,891-892`）。本段不触碰 g/β，仅登记其来源。

## 2. 布局与搬运几何

| 侧 | 平面 | 元素下标 | 出处 |
| -- | ---- | -------- | ---- |
| 源 | `wsO` | `h*m*128 + t*128 + d` | `m15_gdn_prefill.h:1090` |
| 目的 | `o` | `t*6144 + h*128 + d` | `m12_rmsnorm_gated.asc:398` |
| 源 | `qkvzba` | `t*16480 + j` | `m15_gdn_resources.h:272`（`SZ_QKVZBA`） |
| 目的 | `z` | `t*6144 + j` | `m12_rmsnorm_gated.asc:399` |

搬运的**原子粒度**是每 head 128 fp32 = 512 B 的连续块（源、目的各自连续）。转位按
`(head, token-tile)` 划分工作项：

- **MTE2**：源 offset `h*m*128 + t0*128` 起**连续** `rows*128` 个 fp32（`blockCount=1`）→ UB。
- **MTE3**：目的 `t0*6144 + h*128` 起 `rows` 个 512 B 块，块间 gap = `6144-128` 个 fp32
  （`srcStride=0, dstStride=752`，单位 32 B）。

z 落位按 token-tile 划分：

- **MTE2**：每行连续 6144 个 bf16（`blockLen=384` 单位 32 B），行间 gap `16480-6144`
  个 bf16（`srcStride=646`）→ UB。
- **MTE3**：连续写 `rows*6144` 个 bf16（`blockCount=rows, blockLen=384`，无 gap）。

**为什么不用 VF / 不用 cube / 不落标量**：整段是纯 DMA 的块拷贝 + stride 寻址 —— 既不是
矩阵乘法（不上 cube），也不需要逐元素运算（不上 VF）；地址、下标、循环边界全是标量控制流
（人类约束「matmul⇒cube，其余⇒VF，scalar 只做控制流」；转位按块搬运，无「数值」参与，
见 §3.2）。donor 形态取自 `m8_permute/m8_permute.asc`（MTE2→UB→MTE3 搬运）。

## 3. Kernel 设计（`m28_gdn_epilog.asc`）

### 3.1 两个 phase、工作项按核 stride 划分

- **phase 1（o 转位）**：工作项 = `ceil(m/T_O) * 48` 个 `(head, token-tile)`，`T_O=192`
  （tile = `192×128×4 = 98304 B`）。`for (it = bid; it < itemsO; it += nblk)`，每个工作项
  一条 MTE2 + 一条 MTE3。
- **phase 2（z 落位）**：工作项 = `ceil(m/T_Z)` 个 token-tile，`T_Z=8`
  （tile = `8×6144×2 = 98304 B`）。
- 尾部 tile 用 `rows = min(T, m - t0)` 收窄 `blockCount` 与 `blockLen`，不越界读源。

### 3.2 同步与 UB

- 只用 BufferID（`GetBufInternal`/`RlsBufInternal`，release 一律阻塞释放 `mode=false` = CANN `ASC_LOCK_BLOCK` 默认、`true`=`NON_BLOCK`，两种模式都等本 pipe 已发射指令落地），
  无 set_flag/wait_flag；核内单核流水，无 CrossCore。
- UB 静态布局：`UB_O @0`（98304 B，`BUF_O`）、`UB_Z @98304`（98304 B，`BUF_Z`），
  两窗不重叠（`static_assert` 峰值 ≤ 248 KB）；两 phase 访问的 GM 区间也不相交
  （`o` vs `z`），不引入额外核间同步。

### 3.3 负向对照（`MUT_HEADSTRIDE`）

phase 1 的源 offset 用错 head-stride（`128` 而非 `m*128`）：
`srcOff = h*128 + t0*128`（正确为 `h*m*128 + t0*128`）。m=1 时两式退化重合，m>1 必错
（`docs/22` §7.4「epilog 不做转位 ⇒ m>1 对拍必红」的同型对照）。该 mutant 对任意 m
都不越界（max 源 offset = `(47+m)*128 ≤ 48*m*128`）。host harness 的 C 参考恒为**正确**
转位式 ⇒ mutant 下设备输出 vs C 参考必红；`check_ref.py` 的 numpy 参考同为正确转位式，
负向档走独立判定。

## 4. 校验方法与结果

三层：

1. **设备 vs 内建 C 参考**（`m28_gdn_epilog.asc` 的 `RefTranspose`/`RefPackZ`，纯位拷贝）：
   `RunCase` 位级比较，device 输出缓冲先行 `0xCD` 污染。
2. **设备 vs numpy 参考**（`check_ref.py`，独立于 kernel 与 C 参考）：
   `o[t,h*128+d]=wsO[h,t,d]`、`z[t,j]=qkvzba[t,10240+j]`，位级（fp32 视 uint32、bf16 视
   uint16），T1 容差 0。
3. **负向对照**：`M28_MUTANT=headstride`，numpy 参考必须判红（`check_ref.py --mode mut`）。

### 4.1 构建

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m28_gdn_epilog/build -S m28_gdn_epilog -DCMAKE_BUILD_TYPE=Release
cmake --build m28_gdn_epilog/build -j4
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。
构建读数见证据 `evidence/build.log`。

### 4.2 设备档读数（2026-10-04；证据 `evidence/`）

```bash
M28_CASES=1,4,4097 M28_MUT_CASES=4,4097 bash m28_gdn_epilog/reproduce.sh /tmp/m28_out
```

| 档 | m | 结果 | 读数出处 |
| -- | - | ---- | -------- |
| 正向 | 1（退化档） | o/z 位级一致（`0 bad`） | `evidence/m28_m1.log` |
| 正向 | 4 | o/z 位级一致（`0 bad`） | `evidence/m28_m4.log` |
| 正向 | 4097 | o/z 位级一致（`0 bad`） | `evidence/m28_m4097.log` |
| numpy 判据 | 1/4/4097 | 三项 `bit-exact=True` | `evidence/check_pos.log` |

设备槽纪律：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照
`evidence/*npu_smi*.txt`，4 档 Health 均 `OK`）、`timeout 180` 放在锁内；本轮锁未出现
等不到的情况（各档 log 首行均有 `[lock] acquired`）。

### 4.3 负向对照读数

`M28_MUTANT=headstride`（§3.3）两档：

| 档 | m | 设备 vs C 参考 | numpy 判据 | 读数出处 |
| -- | - | -------------- | ---------- | -------- |
| mutant | 4 | o `5 bad` / z `0 bad` ⇒ FAIL | o `bit-exact=False`（24064/24576 个不同）、z `bit-exact=True` | `evidence/mut_m28_m4.log`、`evidence/check_mut.log` |
| mutant | 4097 | o `5 bad` / z `0 bad` ⇒ FAIL | o `bit-exact=False`（24647552/25171968 个不同）、z `bit-exact=True` | `evidence/mut_m28_m4097.log`、`evidence/check_mut.log` |

两档均只剩 h=0 列（两式在该列重合）相等，正是「错 head-stride 只扰动 o 路」的指纹；
**z 路未被扰动、仍逐位相等**。⇒ 正向判据确实盯着 o 转位，negative control 非空洞。

## 5. 纪律自查

- **禁用词**：mission 纪律清单里的绝对化措辞，在 `m28_gdn_epilog/` 的新增文件上未命中
  （逐字 pattern 与命令见 review-request，本 README 不复写该清单以免自命中）。
- **`文件:行`**：§1–§3 每条结论都给了 `文件:行` 或 evidence 路径。
- **协议目录字样**：新增文件里不含 mission 明令禁止的那个隐藏目录名。
- **`flock` 纪律**：见 §4.2。
- **不外推**：§0 明确未做端到端，未串 S5/S6/S7。
- **判据非空洞**：负向对照必须变红（§3.3 / §4.3）。

## 6. 未完成 / 后续

- 未接进 `m15_layer_loop` 的 epilog 挂载点；挂载点与实参追加的接线方案见
  `docs/22-prefill-prolog-epilog-wiring.md` §6.1（本 mission 不做）。
- z 平面的**生产者**（in_proj 输出在预填充路的持有者）未接；本段只定义了它的落位形态。
- 未做多 stage 流水（当前 per-item 单缓冲），未做带宽调优；本 mission 以正确性为先。

## 7. M149：epilog 链段（转位/落位 → S5 → S6 → `hcAttnOut`）

新增文件：`m28_epilog_chain.asc`（链段本体 + 宿主 harness）、`check_chain_ref.py`（链参考）、
`reproduce_chain.sh`（复现脚本），证据在 `evidence/chain/`。
`m28_gdn_epilog.asc`（M148 转位段）与 M148 的证据**原样保留、单独可跑**。

### 7.1 契约（逐条带 `文件:行`）

| 平面 | 几何 / 存取式 | dtype | 出处 |
| ---- | ------------- | ----- | ---- |
| 入口 `wsO` | head-major `[48, m, 128]`，元素 = `h*m*128 + t*128 + d` | fp32 | `m15_gdn_prefill.h:1089-1091`；字段 `m15_layer_kernel.h:263` |
| 入口 `qkvzba` | 行主序 `[m, 16480]`；z 段 `[10240, 16384)` | bf16 | `m15_gdn_resources.h:272`，`:51-57` |
| 中间 `o`（段①出、段②入） | token-major `[m,6144]`，元素 = `t*6144 + h*128 + d` | fp32 | `m12_rmsnorm_gated.asc:398` |
| 中间 `z`（段①出、段②入） | token-major `[m,6144]` | bf16 | `m12_rmsnorm_gated.asc:399` |
| 中间 `y`（段②出、段③入=A） | `[m,6144]`，A 行主序（`srcDValue=K`） | bf16 | `m12_rmsnorm_gated.asc:401`；`m11_bf16_gemm.asc:190` |
| 出口 `hcAttnOut` | `[m, 2560]` 行主序，行距 = `HID*2 = 5120 B`（`ROW_HID`） | bf16 | `m15_hc_prefill.h:114`；`HID=2560` `m15_hc_resources.h:40` |

S6（out_proj）：K=6144、N=2560、B=W `[2560,6144] bf16`（`m11_bf16_gemm.asc:11-12`，`:46-51`）；
C 的 `dstStride = N = 2560` 元素（`m11_bf16_gemm.asc:249`）与 `hcAttnOut` 行距逐字节一致
⇒ S6 直接写 `hcAttnOut`，无需额外落位拷贝（`docs/22` §4.2 步骤 4）。
M146 已在设备上把 `bf16_gemm_kernel<6144,2560>` 在 m=4097 单次启动测得逐位一致
（`m11_bf16_gemm/evidence/m4097_envelope/README.md`）。

### 7.2 复用 / 新写

- **逐字复用**：段① = `m28_gdn_epilog.asc` 的 `GdnEpilogKernel`（去掉 M148 的转位 mutant）；
  段② = `m12_rmsnorm_gated.asc` 的 `NormDonor` 与逐行数学（`CalculateGateY` 的 sigmoid 四元组 + 三次 Mul）；
  段③ = `m11_bf16_gemm.asc` 的 `Bf16Gemm<K=6144,N=2560>`（Nd2Nz → LoadData2D → Mmad → Fixpipe）。
- **唯一集成改动**：段② 的 `Process()` 行循环按核 stride 划分（`for(row=bid; row<M; row+=nblk)`），
  逐行数学一字未动（m12 原为单核 `for(row=0..M)`；行间无依赖）。
- **新写**：三段之间的 GM 平面与宿主接线、链级宿主参考、两个 S5 mutant、链判据脚本。

人类约束（逐字）「矩阵乘法 ⇒ cube，其余 ⇒ VF，scalar 只做控制流」的落点：只有段③（K×N 矩阵乘）
走 AIC 的 `Mmad`；段①②走 AIV 的 `DataCopy` 与向量寄存器；地址/下标/循环边界/尾块全是 scalar。

### 7.3 判据与读数

三段判据 + 链端到端判据、容差分档（T1 容差 0；S5 bf16 网格 rel ≤ 1e-2；S6 T3
`|got-ref| ≤ ε·Σ|terms| + 0.5·ulp_bf16`，ε=5e-5，`docs/22:341`）与逐档读数见
`evidence/chain/README.md`。要点（2026-10-04，设备）：

- m=4 与 m=4097：段① T1 逐位一致；S5 `max rel` 0.00559 / 0.00781（≤1e-2）；
  S6 T3 隔离档超界 0 / 0；链端到端 T3 over-frac 0 / 3.81e-07；
  `m12 check_ref`（bf16 网格）与 `m11 check_ref`（精确域逐位）两档皆 PASS。
- 负向：`nogamma` 与 `zhead0` 两 mutant × 两档，段①仍逐位一致，S5 与链出口都变红
  （`nogamma` S5 `max rel` 105；`zhead0` S5 `max rel` 8.39e+06、链 over-frac 0.997/0.998——只剩 h=0 列）。

### 7.4 窄口径（未做 / 未主张）

- **未接进 `m15_layer_loop`** 的 epilog 挂载点；H2 的 `bo` 取值点未动。
- `wsO`、`z` 都仍是**合成输入**（B1 扫描段与 in_proj 的预填充生产者未接，`docs/22` §2.3 G1-c）。
- 未做多 stage 流水 / 带宽调优；**不主张端到端**（S7 残差与 HC 边界 #2 不在本段）。
- 设备已实测 m 集合 = {4, 4097}；更大 m 未逐个实测。

### 7.5 纪律自查（M149）

- **禁用词**：mission 清单的六种绝对化措辞，在 `m28_gdn_epilog/` 的新增/改动文件上逐字普查未发现
  （逐字 pattern 与命令见 review-request，本 README 不复写清单以免自命中）。
- **证据包 `sha256sum -c`**：`evidence/sources_sha256.txt`（M148 的 5 个文件，含此前过期的
  `reproduce.sh` 行，已按当前内容刷新）与 `evidence/chain/sources_sha256.txt`（M149 的 5 个文件）
  均可整体通过（命令：在 `m28_gdn_epilog/` 内 `sha256sum -c evidence/sources_sha256.txt`、
  `sha256sum -c evidence/chain/sources_sha256.txt`）。
- **`文件:行`**：§7.1–§7.3 每条结论都给了 `文件:行` 或证据路径。
- **锁纪律**：`flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照落盘）、`timeout 280` 在锁内
  （6 档读数见 `evidence/chain/`）。
- **不外推**：§7.4 明确未接 `m15_layer_loop` 挂载点、未主张端到端。
