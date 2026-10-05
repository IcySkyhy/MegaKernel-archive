# M98 · attention cache 填数学 —— 证据与口径

本目录是 mission M98（`feat/m98-attention-kv-cache-fill-math`，base `main @ 9603c65`）的段级证据。
交付物：`m15_attn_cache.h`（设备侧填 cache 数学）+ `m15_attn_cache_host.h`（host 独立参考 + 判据 + 负向对照），
由 `m15_attn_kv.h` / `m15_attn_kv_host.h` 在**末尾** include（`.asc` 一字未动，它归 M97 独占）。

## 1. 复现（命令 → 输出）

| 命令 | 输出 / rc |
|---|---|
| `cd <wt-98> && source /usr/local/Ascend/ascend-toolkit/set_env.sh && cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release && cmake --build m15_layer_loop/build -j4` | `build rc=0`（无 `m15_attn_cache*.h` 的 error；`m15_hc_layer.h` 的两条既有 `-Wcce-compat` warning 与本 mission 无关） |
| `./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt kv` | **ALL PASS**（checks=215, guards=191, fails=0），`rc=0` —— 见 `m98_runs_kv.log`（本目录），二进制 `sha256=560be00e2f94…75ad3d9`（**含塔裁 A 的几何更正**）。⚠ 二进制身份史：r2 `56679e94…` → r3 `560be00e…`（r3 只改注释/证据文字，但 ASC 带 `-gline-tables-only`，**注释行数变化会进 line table** ⇒ 行为不变、身份变）。**r4（本轮）的 F-R1 用「行数不变的替换」（被改行仍是一行），故重编后 sha256 **逐字节不变** = `560be00e2f94…75ad3d9` ⇒ 本表两档读数与归档日志**继续有效**，无需重跑（已实测：重编后 sha 与冻结值相同） |
| 同上（`runs=kv` 的整段） | 段级分账（读 `m98_runs_kv.log` 的两行小结）：**验证 Ac（M98）= 26 条判定项 + 9 条 guard**；日志里「验证 Kv」那行的 **83 条**是 `H_RunKv` 的区间计数，它**已经把嵌在 `H_RunKv` 里跑完的 Ac 26 条包含进去**（83 = 基线（base commit 上 M82 自有的）56 条 + 本轮 `Kv.neg.oldgeom.dev` 1 条 + Ac 的 26 条；总数 215 = 132 + 83）。⇒ **83 与 26 不是两笔不相交的计数**，不要相加 |
| `./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all` | **ALL PASS**（checks=2095, guards=302, fails=0），`rc=0` —— 见 `m98_runs_all.log`（本目录，**作者自己实跑**；r1 复审也代跑过同一组数字）。**零回归**与对账：基线（**base commit 上**的 2068/290，来源 = main 的 `evidence/attn_wire/runs_all_r2_summary.log:9`）→ 本 tip 2095/302；判定项 **+27 = Ac 26 + `Kv.neg.oldgeom.dev` 1**（两边分账行 `Kv 56` → `Kv 83`）；guard **+12**（分账前四类相同，差额全在「其余」130 → 142；本 tip 自报 Ac 9 + KV 新增 3）。既有判据无一翻转 |

⚠ **环境硬纪律**（本队 2026-09-27 立）：跑 `m15_layer_loop` **前**必须 `npu-smi info` 确认卡上空闲
（本 mission 的每次跑都由一个 `npu-smi` + `/proc/meminfo` 双闸的等待脚本触发，逐次读数在日志里可见）；
写文件前先 `df -h /`。本 mission 的所有跑都满足这两条。

## 2. 读数清单（`m98_runs_kv.log`，逐条）

| 判据 | 口径 | 读数 |
|---|---|---|
| `Ac.ring.row` | **T1 逐字节**，环平面 1,120 B（门控放行的末尾 4 行原样；见 §4.5 的**可观测性限制**：4 槽环下门控放行的恰是全部 4 个槽，故平面字节上**没有**"不该写的槽"可留毒值） | PASS 逐字节一致 n=1120 |
| `Ac.pack.row` | **T1 逐字节**，packed 两行 16,416 B（行距 8,208 B ≡ 16 (mod 32)） | PASS n=16416 |
| `Ac.mainkv.kv` | **T1 逐字节**，主 KV 平面 **8,421,376 B**（257 页 × 32,768 B；4 个 (head,K/V) 槽写对 + 其余全毒值） | PASS n=8421376 |
| `Kv.neg.oldgeom` / `.dev` | **塔裁 A 的新判据**：M82 的旧主 KV 几何必须被否定（host 算式 28 个采样点全不同；设备用旧几何落页 ⇒ 按新几何读回 ≠ seed） | PASS ×2 |
| `Ac.pool.row.p23` / `.p27` | **T1 逐字节**，设备写出的 pooled 行（bf16）vs host 的 `bf16(精确均值)` | PASS n=256 ×2 |
| `Ac.comp.row.p23` / `.p27` | **T3**（ε=16·2^-24·Σ\|terms\| + 平局歧义项 + 1.0·ulp） | PASS，越界 0/128；报告项 `≤1ulp 128/128`、`maxAbs=0.000e+00`、`maxRel=0.000e+00`（**逐元素全等**） |
| `Ac.comp.nospurious` | 结构性：期望行外必须是毒值、期望行不得为空 | PASS（0 个越界非毒字节） |
| `Ac.gate.ring/.comp/.read/.mask` | 门控账：host 按**同一条官方规则**独立复算 vs 设备 flag | PASS（4/4/2/6 全等） |
| `Ac.det.repeat(.comp)` | 确定性（同档两次跑逐字节一致）—— 拿逐字节当判据的前提 | PASS |
| `Ac.nonvac.expect` | 非空洞（比对面非零非退化、两行互异） | PASS（guard） |
| `Ac.nonvac.sens` | 输入敏感性（换 raw k 的盐 ⇒ 环与压缩平面都变） | PASS |
| `Ac.ob.ring` / `Ac.ob.comp` / `Ac.ob.open` | off-by-one 档（chunk 4093..4096）：环 4 行写满且槽绕回 **1,2,3,0**；组 1023 的行；开放组 1024 的行**必须仍是毒值** | PASS ×3 |
| `Ac.neg.plaimnorm` | 方向级负向：norm 乘 `w`（不是 `1+w`） | 判据**被打破** ✓ |
| `Ac.neg.ropelast` | 方向级负向：RoPE 用组**末**位置 | 判据**被打破** ✓ |
| `Ac.neg.noringread` | 方向级负向：跨 chunk 成员不从环读 | 判据**被打破** ✓ |
| `Ac.neg.nocompstor` | **变体五**：填压缩行那段不执行 | 判据**被打破** ✓ |
| `Ac.neg.noringstor` | **变体五**：填环那段不执行 | 判据**被打破** ✓ |
| `Ac.neg.offbyone` | 负向：边界判据写成 `pos % 4 == 0` | "开放组不得被写"**被打破**，同档 `Ac.ob.comp` 也被打破 ✓ |
| `Ac.neg.ringsplit` | **正对照**：280 B 拆 256 B 块 + 24 B Pad 两次搬 | 与一次 280 B Pad 的环平面**逐字节相同** ✓ |
| `Ac.dcpad.required` | **不用 DataCopyPad 会怎样**：`Block1(280)` | launch **非正常结束**（`aclrtSynchronizeStream` 返回 **rc=507015**）。⚠ 判据只看这个 rc：设备侧的**具体文字未落进归档日志**（host 侧 `m15_attn_cache_host.h` 的 `aclGetRecentErrMsg()` 返回值被丢弃、未打印）⇒ 本文不再转述未归档的设备字符串 ✓ |
| `Ac.rep.sumnomean` | **报告项（不计判定项）**：池化不除 4 | 同一判据**仍成立** ⇒ 该错误对**压缩行判据**不可判别（RMSNorm 对常数尺度不变）；但它对**池化的逐字节判据** `Ac.pool.row.*` 是**可判别**的（见 §4.3） |

判定项构成（`m98_runs_kv.log`）：26 条 = T1/T3/结构性/门控/非空洞 16 条 + 负向对照 7 条（含 off-by-one）
+ 正对照（ringsplit）1 条 + trap 1 条 + 确定性 1 条…（以日志里的逐行 PASS/FAIL 为准，**不在此处另起一套口径**）。

## 3. 这一版抓到的三处**参考侧**缺陷（都不是设备缺陷，逐条留痕）

1. **host 把元素步长当字节步长**（`AC_IN_RAWK_STRIDE=144` 用在 `uint8_t*` 上）：环/压缩行的期望值
   整体取错行 ⇒ `Ac.ring.row` 报 980/1120 字节不同（读数里 `got 0x01 exp 0xfe`；事后用同一份 hash
   独立算出 `0x01` 正是位置 28 的首字节 ⇒ 设备是对的、期望是错的）。修：加
   `AC_IN_RAWK_STRIDE_BYTES` 并在 host 侧只用它。
2. **非空洞 guard 用"非毒字节计数"**：随机字节里恰好等于毒值 `0xCD` 是正常现象（实测环 1,116/1,120、
   主 KV 2,045/2,048）⇒ 改成"每个应写段必须与全毒值平面不同"。
3. **参考漏了官方链的第二个 bf16 舍入点**：官方 `pooled`（bf16）→ `gemma_rmsnorm`（输出沿用输入
   dtype ⇒ **bf16**）→ `apply_qsa_rope`（吃 bf16 值）；设备侧同构（AivNormRope 先写 bf16 到 UB，
   rope 段再读回）。参考原先用未量化的 fp32 y 去旋 ⇒ 2/128 越界（|Δ| = 2–4 个输出格点）。
   补上这个舍入点后 `maxAbs` 由 1.562e-02 变 **0.000e+00**（128/128 逐元素全等）。
   依据：`vllm/models/qwen4_exp/nvidia/ops/qsa.py:1001-1005`（pooled dtype = raw_keys.dtype = bf16）、
   `nvidia/indexer_qsa.py:354-358`（gemma_rmsnorm）与 `:359-367`（apply_qsa_rope）。

## 4. 口径与已知边界

### 4.1 判据的档位（docs/17 §1.1，必须在 README 写明理由）
- **T1 逐字节**用于**整数/位域/搬运**类：环行（raw k 字节 + 3×int64 位置尾）、packed 行、主 KV 的
  K/V 槽、pooled 行（bf16 格点上的值 = 位域）。
- **T3** 用于压缩行：它含 `Sqrt`（Rsqrt 精确式，非近似）+ 两次 bf16 量化 + 一次 RoPE ⇒ 属 T3 档。
  ε 的逐项来源：池化 3 次 fp32 加、Σx² 4 次、ReduceSum 1、rstd 3（Muls/Adds/Div/Sqrt）、y 2、rope 3
  ≈ **16 次 fp32 舍入**，每次 ≤0.5·2^-24 相对 ⇒ `16·2^-24·Σ|terms|`；**两个 bf16 舍入点**
  （pooled→bf16、norm→bf16）在参考里**显式建模**（不是当误差项）；输出侧按"参考已量化到格点"取
  **1.0·ulp(out)**（docs/17 §1.1 的 `0.5 → 1.0` 条款）。平局歧义另列 `flipTerm`（见 §4.2）。
- 报告项（**不进 PASS 计数**）：`≤1ulp 比例`、`maxAbs`、`maxRel`、`最大占用`。

### 4.2 平局（bf16 舍入边界）的处理：**枚举合法候选，而不是放宽 ε**
池化值/中间值若落在 bf16 格点中点上，设备的 fp32 累加与参考的 double 可能各取一侧（两者都合法）。
本段不"把 ε 放宽"，而是把这份**合法歧义**按元素算出来：`flipTerm[c] = Σ (∂o/∂输入) × 该输入的一格`，
只对该输出元素**实际依赖且确实落在中点**（≤32 fp32 ulp）的输入计入。实证：`Ac.comp.row.p23` 在
该机制下 0/128 越界；而把漏掉的舍入点补上后 `flipTerm` 已不再被用到（`maxAbs=0`）。
⚠ 本文**不**声称"flipTerm 一定覆盖所有平局组合"——它是逐元素的一阶项，且中点判定的阈（32 fp32 ulp）
是**推导出的**（设备侧 fp32 累加 ≤3 次舍入）。

### 4.3 `Ac.rep.sumnomean`：报告项**不进判定项**，但它的**理由要收窄**（r1 复审 F5）
- 事实：`AC_MODE_SUM_NOT_MEAN`（池化用 Σ 而非 Σ/4）注入后，**压缩行判据** `Ac.comp.row.p23` 仍成立 ——
  因为官方链 `y = x·rstd·(1+w)`、`rstd = 1/sqrt(mean(x²)+eps)` 对池化**常数因子**尺度不变（被 rstd
  精确抵消），只剩下 eps 项与 pooled 的 bf16 格点这点痕迹。⇒ **对该判据**这条方向级错误不可判别。
- **但这不是"这条错误不可判别"**（本 README r1 版把话说过了）：本段还有 **`Ac.pool.row`** —— 池化落
  bf16 后的**逐字节**判据；Σ 不除 4 会让设备写出 `bf16(4·mean)`，与控制档的 `bf16(mean)` 逐字节不同
  ⇒ **在池化判据上它是可判别的**（这是**推导**：派生自 `Ac.pool.row` 的口径 + 除 4 是精确的 2 的幂
  缩放，本轮**未**实跑该档的 `Ac.pool.row`，见 §5 第 7 条）。
- 处置（**不变**）：按 docs/17 §4「答不出的不得计入 PASS 计数，须单列为报告项」+ §2 第 1 条分栏，
  它**不进判定项**，只列报告项 `Ac.rep.sumnomean`；**是计数口径正确、不是打标豁免**。
- 可选升格（未做）：在该档的报告项里把 `Ac.pool.row` 也 silent 判一次（或直接升格为方向级负向对照，
  打的判据改成 `Ac.pool.row.*`）—— 属实现改进，不在本轮返工范围（复审 F5 的强制项只是理由收窄）。

### 4.4 对齐/边界专项（M82 报的两条关键约束的实测）
- **raw ring 单行 280 B ≡ 24 (mod 32)**：本段**一次 `DataCopyPad(280)`** 落环（含首行 slot 0、末行
  slot 3；off-by-one 档 4 行覆盖槽 1,2,3,**0**（绕回））⇒ `Ac.ring.row`/`Ac.ob.ring` **逐字节一致**；
  正对照 `Ac.neg.ringsplit` 证明"256 B 块 + 24 B Pad"两段式结果**完全相同**；
  **不用 Pad 会怎样：launch 非正常结束**（`Ac.dcpad.required`：`aclrtSynchronizeStream` 返回 rc=507015 ——
  判据只依据这个 rc；设备侧文字未归档，见 §2 该行的说明）。
- **packed 单行 8,208 B ≡ 16 (mod 32)**：两行（行距 8,208 B）逐字节一致 ⇒ 单行 `DataCopyPad` ✓。
- **跨页边界**：压缩行在 M 档落在**页 1 的行 1/2**，在 off-by-one 档落在**页 255 的行 3**
  （`Ac.ob.comp` 逐元素全等）⇒ 页寻址/行内偏移都对。

### 4.5 环写入门控的**可观测性限制**（r1 复审 F3 的口径更正）
- 官方门控（`qsa_cache.py:144-149` / `:283`）只允许写「chunk 末尾 capacity 行」。但环**只有 4 槽、
  capacity = 4** ⇒ 门控放行的 4 行恰好把残留类 0..3（`pos % 4`）**各覆盖一次**；更早的行即使被"错误地
  也写了"，也会在同一次遍历中被这 4 行覆盖。
- 实测：档 M（chunk 22..29）与 off-by-one 档（4093..4096）**都把 4 个槽写满** ⇒ **平面字节上不存在
  "不该写的槽仍留毒值"这种形态**（本 README r1 版的这句话已被本轮的 r1 复审更正）。
- ⇒ 本段对该门控**唯一的观测面**是 `Ac.gate.ring`：host 按同一条官方规则独立复算出的 `ringWritten`
  与设备自报的 flag 相等（读数 4 vs 4）。它是合法判据，但**不是平面/字节判据**，也挡不住
  「按同一个错规则计数」的形态。要做成平面判据需要一条**非恒等/带副作用的形态**（登记在 §5 未做）。

## 5. 未完成项 / 显式未覆盖（逐条）

1. ~~主 KV 几何分歧~~ —— 已由**塔裁 A** 授权更正：`m15_attn_kv.h` 的主 KV 段改为官方几何
   （页 32,768 B、token 步长 1,024 B、head 步长 16,384 B、V = K + 512 B 同槽、轴序 `[blocks,H,N,C]`），
   容量 `KV_LAYER_STRIDE` 4,210,688 → **8,421,376 B/层**、`KV_PLANE_BYTES` 50,528,256 →
   **101,056,512 B（96.38 MiB）**；旧几何作为**负向对照**保留在 `m15_attn_kv.h`
   （`KvOldGeomByteOffset()` + `KV_OLD_*`）与探针（`MODE_BROKEN_OLDGEOM`），判据 `Kv.neg.oldgeom` /
   `Kv.neg.oldgeom.dev` 两条都 PASS（实跑见 `m98_runs_kv.log`）。
   本段的搬运实现改为**直接引用** `M15KV_KV_*`（不再自带第二份数字）。
   ⚠ **仍未做**（不在本 mission scope）：`m15_layer_loop/m15_layer_loop.asc:808` 的注释归 **M175**。
   `m15_layer_loop/README.md` 里的同族旧数字已由 **M178 合入**（`4e626bb`）同步为权威值。
   `.asc` 的**打印值**已由宏自动更新，只有注释/文档文字是旧的。
2. **真实规模档（真实权重切片）未做**：本段用的是 host 生成的声明输入（①）。真实 `attn_idx_k_norm`
   权重切片 + 真实 cos/sin 表 + 真实 4097 全上下文的档**未跑**。
3. **`Ac.comp.row.*` 只覆盖 2 条组**（M 档）与 1 条（OB 档）；**open group 的跨 chunk 池化**（组成员
   同时来自环与本 chunk 的更长形态）只覆盖到 4 成员里 2 个来自环的情形。
4. **packed 的内容来源（选择语义）不在本段**：本段把 packed 行当 ① 声明输入，只判行距/传输/逐字节。
5. **主 KV 的 paged 非恒等 block_table** 未覆盖（归 M82 的 T-KV-PAGED）。
6. **`Ac.nonvac.sens` 只换 raw k 的盐**（位置/权重/表都不换）。
7. **环写入门控只在计数上见证**（r1 复审 F3）：4 槽环下该门控在平面字节上不可判别（§4.5）；
   要做成平面判据需要一条**非恒等/带副作用**的形态（把"早于门控的行"写成可区分的种子值并断言它没被搬）
   —— 现在做不到（槽会被后续 4 行覆盖），**未做**。
8. **`Ac.rep.sumnomean` 的可选升格**（r1 复审 F5）：把该档的池化逐字节判据 `Ac.pool.row` 也判一次
   （或把方向级负向对照的打击面从 `Ac.comp.row` 改到 `Ac.pool.row`）—— **未做**（复审的强制项只是
   把"不可判别"的理由收窄到压缩行判据，已在 §4.3 收窄）。
9. **形态偏差（r1 复审 F6）—— 塔已裁决：豁免（2026-09-27）**。事实：mission 第二步的形态写的是
   「核内 buffer id、**核间 set cross core**」，而本段是**单核 AIV 串行、无跨核同步**。
   **塔裁的理由（照录）**：那条要求是「**若需**核间同步则用 `set cross core`（而不是 set/wait flag）」，
   而**不是"必须造出跨核同步"**；一个单核段没有跨核同步要做 ⇒ 不构成违规。
   本段达成项：`VEC_SCOPE`/寄存器基础 API、核内 buffer id（10/11/12）、`PipeBarrier<PIPE_ALL>` 表达核内
   值依赖、地址自管、编译期静态 UB 窗。
   **⚠ 后续接手人的义务（塔点名要写清）**：**本段为单核 AIV、无跨核同步 ⇒ 不涉及 `set cross core`；
   若后续要并行化（多核分片），同步形态须按核间 `set cross core` 重做**（而不是另起 set/wait flag）。
10. **`m15_attn_prolog*.h`（M88）保持只读**：本段复用其 `AivNormRope`（不重写第二份 norm/rope），
   因此**上游 prolog 的产物在本段是 ① 声明输入**（不是 ② 上游输出）—— 这一点写在两处头文件的文件头里。
