# M85 — PLE：我们的 SPEC/kernel 与**上游两份实现**的逐项对照（D 类回扫）

> 立表原因：M85 r1 复审判 `p1-1items / fix-then-merge`，指出"我们的 PLE 规格相对上游存在**简化**这一类
> **系统性**偏差"（当时已知 D1/D2/D3 三条），要求做一次完整回扫、把每条分歧都配一条**能咬住它的判据**。
> 本表就是那次回扫的产物。分支 `feat/m85-ple-kernel-spec-pin-and-standalone-i`（**未合入**）。
>
> **方法（三向对照）**：每个基本算件都比三份东西 ——
> ① 我们的 `PLE_SPEC.md` / `m15_ple.asc`（`M85P` 命名空间）/ `m15_ple_check.py`；
> ② 上游 **triton 生产路径** `/workspace/vllm/vllm/models/qwen4_exp/nvidia/ops/ple.py` + 其调用方
> `nvidia/ple_layer.py`；③ 上游 **eager CPU 路径** `nvidia/ngram_embedding.py`（`compute_ngram_ids`、
> `_shift_precompute`、`_shift_apply`）与 AMD 变体 `qwen4_exp/amd/ple_layer.py`。
> 表里"我们的位置"与"上游位置"都给 `文件:行`。
>
> ⚠ **`m15_ple.asc` 的行号只作路标，不要当精确锚**：抽查发现这些行号在本表写下之后就已与文件
> 不再是同一处（例：D5 引 `m15_ple.asc:136` 的 `FloorMod64`，写下本表时它已在 `:165`；D4 引
> `:449-453` 的「和的 bf16 物化」已在 `:581`）。M111（落盘通路修复）又在 `m15_ple.asc` 上新增/改写了
> 若干行，使这些数字**进一步**偏移。按**符号名**（`FloorMod64`/`col1`/`gUse`/`vOff`…）在本文件里
> grep 定位，比按行号可靠。该漂移已作为 finding 报给塔，**本文件的一处引用（`evidence/ple_wire/
> E_det_probe_summary.txt` 的 `§3.3`→`§3.4`）已顺手修**，其余未逐条重核。
>
> **每条分歧必须配一条判据**：右侧 `咬它的判据` 列里的名字可在
> `ple/logs/mutants.log`（`m15_layer_loop/m15_ple_mutants.py` 的输出）里逐行核到 ——
> 该脚本对每个变异位跑一次设备，要求**指定判据 FAIL** 且**基线全 PASS**。

---

## 1. 分歧主表（有实现的分歧 = 必须能咬）

| # | 我们的位置 | 上游位置 |差在哪 | 可观测？ | 咬它的判据（变异位） | 状态 |
|---|---|---|---|---|---|---|
| **D1** kv 投影的张量组织与顺序 | `PLE_SPEC.md` §2「key_proj / value_proj 分开」；`m15_ple.asc:974-975`（host 侧 `wcat=[key;value]` 拼） | `model.py:153-154`（`ple.key_proj→kv_proj:0`、`value_proj→kv_proj:1`）、`ple_layer.py:107-115,415-416` | checkpoint **没有** `ple.kv_proj`；融合发生在装载期，**key 在前** | 是（顺序反则全列错） | `B3.kv`（变异 bit7：③ key/value 输出块对调） | ✅ 有 |
| **D2** ⑤ 的 conv_output 取整点 | `PLE_SPEC.md:157-160`；`m15_ple.asc:649-652`（`conv_output = bf16(y)`） | `ops/ple.py:438-449`：`conv=bf16(acc)`→`y=fp32(conv)*sigmoid`→`conv_output=bf16(y)`→`ple_output=bf16(residual+conv_output)` | SiLU 在**取整后的** fp32 值上算 | 是（≤1 ulp 级） | `B5.out` ← **变异 bit13**（跳过 `conv_output = bf16(y)` **这一点本身**；32/20480 元素超界）。bit3/bit4 只演示 ⑤ 的**别处**（tap 序 / 残差分组），不作为 D2 的判据 —— r2 复审 P3 指出旧版那样标注是归属错误 | ✅ 有 |
| **D3** 跨 chunk 的 n-gram 上下文列（`c=1`） | `PLE_SPEC.md:126`（→ 本轮改为 `ctx[r, NC-2+c]`）、`m15_ple.asc:173-174`（`col1/col2`）、`m15_ple_check.py:105-107` | `ops/ple.py:81` `ctx_col = NC - shift + c`；eager `ngram_embedding.py:322`（`context=cat([ngram_context,packed])`）与 `:337`（`adjusted_columns=columns+NC`） | 我们（与 `docs/14 §6.2`）把列写成**与 c 无关** `NC-shift`；上游是 `NC-shift+c` ⇒ **c=1 时 lag-2 必须取 `ctx[r,1]`** | 是（矩阵 A：t=1、t=7 的 head 8-15，16/160 元素） | `B1.ids.A`（变异 bit1：ctx 列退回与 c 无关） | ✅ **本轮修**（复审 P1） |
| **D4** ④ 门控点积和的取整点 | `PLE_SPEC.md:147`（SPEC 本来就对）、`m15_ple.asc:449-453`（本轮补上 `Cast<bf16>`） | `ops/ple.py:221-222`：`dot = tl.sum(products.to(f32)).to(dtype).to(f32)`（同段注释在 `:218`）；AMD `amd/ple_layer.py:1113-1117` | **和先物化到 bf16，再除 `√2560`**；第一版 kernel 直接在 fp32 上除 | 只在部分输入上（复审独立测算：`d` 28% 不同、gate 标量 `g` 落不同 bf16 ≈2%/项） | `B4.gated`（变异 bit2：跳过 dot 和的 bf16 物化） | ✅ **本轮修**（复审 P2）+ 数据按 `ple/pick_observable_seed.py` 选种子使之可观测 |
| **D5** 取模方向 | `PLE_SPEC.md:132`（`mod` 取非负余数）；`m15_ple.asc:136`（`FloorMod64`）+ `:215-217`（负修正为 floor-mod） | `ops/ple.py:104-105`（`mixed % sizes` 后 `where(<0, +sizes)`，即 torch.remainder 语义） | floor-mod（结果非负）而不是 C 的截断余数 | 是（负 `mixed` 时） | `B1.ids.A`（变异 bit0：取模方向反转） | ✅ 有 |
| **D6** ④ 的 value 4 流共享 | `PLE_SPEC.md:149`；`m15_ple.asc:357-359`（`vOff` 默认 = `t*KVW + 10240`） | `ops/ple.py:231`（`value_ptr + t*value_rs + lanes`，与 stream 无关） | 4 个 stream 用**同一段** value | 是 | `B4.gated`（变异 bit5：value 换成 key 切片） | ✅ 有 |
| **D7** ② gather 到 head 槽的映射 | `PLE_SPEC.md:135-136`；`m15_ple.asc:252`（`gUse`）与 `:254-256`（`emb[t, gUse*160 …]`） | `ngram_embedding.py:353-400`（`forward`：`ngram_embedding(ids)` → `[T,16,160]` → `flatten(-2)` | head g 的行落在 `[t, g*160:(g+1)*160)` | 是 | `B2.emb`（变异 bit6：head 落点顺序反转） | ✅ 有 |
| **D8** ⑤ 状态移位方向 | `PLE_SPEC.md:160-161`；`m15_ple.asc:673-688` | `ops/ple.py:472-485`（decode：`new[i]=old[i+1]`，`new[8]=x[t]`） | 新样本进**行 8**（最“新”端） | 是 | `B5.state` ← 变异 bit8（移位方向反转）；`B5.state_evolve` ← 变异 bit12（活跃行不写回） | ✅ 有 |
| **N1** `NULL_STATE_ID` 槽位 | `PLE_SPEC.md` 原**未记**；`m15_ple.asc:575-604`（本轮新增 null 分支） | `ops/ple.py:20,391-405,455-462`：`out_ok=false` ⇒ `conv_output=0`，但**仍然写** `out = outer_residual + bf16(residual + 0)`；状态行**不写** | 第一版 kernel **整行跳过**（= 漏写 out） | 是（在我们自己的数据上就有一条 null 行） | `B5.null.out` ← **变异 bit9**（null 行整行跳过；FAIL 集合 `{B5.null.out, B5.out}`）；`B5.null.state` ← **变异 bit11**（null 行**也**写状态）。注：bit9 **不**改 null 态，所以它咬不到 `B5.null.state` —— r2 复审 P3 指出旧版把两条判据都挂在 bit9 上是归属错误 | ✅ **r1 回扫新发现，已修** |
| **L1** conv 权重的内存布局 | `PLE_SPEC.md` §2 记 `[10240,1,4]`；kernel/harness 用 **tap-major** `wtap[4,10240]`（`m15_ple.asc:226`、`ple/gen_ple_data.py:228` 转置） | `ple_layer.py:373` `conv1d.weight.squeeze(1)` → `[c,k]`；`ops/ple.py:430-434` `w_ptr + c*K + k` | 布局不同、**数值等价**（host 侧转置） | 否（自洽） | 不需要（自洽）；但**接线时**必须做这次转置 —— 记在此处避免遗漏 | ⚠️ 约定差异，已在 spec/kernel 注释写明 |
| **L2** conv 状态的内存布局 | `PLE_SPEC.md` I6 记 `[slots,10240,9]`；kernel/harness 用 `[slot][9][10240]`（`m15_ple.asc:607-612` 读状态、`:685-695` 写回；`gen_ple_data.py:276-278` 生成） | `ple_layer.py:185-191` 的 `short_conv_state_shape` + `ops/ple.py:407-417`（`[slots,channels,window]`） | 转置关系（`[slot][w][c]` vs `[slot][c][w]`） | 否（自洽） | 不需要；同 L1 的接线注意 | ⚠️ 同上 |
| **S1** 状态 in-place vs 单独平面 | `PLE_SPEC.md` O2 写"原地"；kernel 读 `stIn`、写 `stOut`（`m15_ple.asc:1006` 用输入预置输出平面模拟 in-place） | `ops/ple.py:472-485`（decode 直接原地写 `state_ptr`） | 我们用两个平面 + host 预置 ⇒ 语义等价，但**接线时每步之间要保证 stOut 回灌** | 单步内不可观测 | 不需要（用 host 预置 + `B5.null.state`/`B5.state` 间接覆盖） | ⚠️ 接口注意，已记 |

## 2. 「参考与实现同源 ⇒ 判据咬不住」清单（复审的硬要求）

| # | 同源的点 | 为什么咬不住 | 本轮处置 |
|---|---|---|---|
| S-1 | **D3 的 ctx 列**（复审 P1） | 第一版 kernel、`m15_ple_check.py` 的参考、`PLE_SPEC.md` **三处都写了同一个简化** ⇒ `B1.ids.A` 报 PASS 却是**空过** | 三处同时改为 `NC-shift+c`；新增变异 bit1 证明 `B1.ids.A` 现在真的能抓住它（`ple/logs/mutants.log` 的 `RESULT|mut|1`） |
| S-2 | **D4 的 dot 取整**（复审 P2） | 参考本来是对的（SPEC 也对），但**实测数据上不可观测**（≈2%/gate 项）⇒ 旧数据上"改错也不 FAIL" | ① kernel 补齐取整；② 新增 `ple/pick_observable_seed.py` 用**设备 kv** 搜出使该分歧可观测的 hidden 种子（`--hidden-seed 4`，1/8 gate 项不同）；③ 变异 bit2 现在使 `B4.gated` FAIL（1022/20480 元素） |
| S-3 | **L1/L2 布局** | kernel、生成器、判据三方都按同一套（转置后的）布局写 ⇒ 上游布局错也照样 PASS | 不伪装成"有判据"：列入 §1 的"约定差异"，并写清**接线时必须做的转置**（L1/L2 的"状态"列标 ⚠️ 而非 ✅） |
| S-4 | **decode-only 用例面** | 矩阵 B 是 T=2 且每请求 1 token（`c=0`）⇒ 结构性看不到 `c≥1`、prefill、`has_initial_states` | 矩阵 A（T=10、2 请求、6+4 token）覆盖 `c≥1` 与 chunk 边界；null 行由 `state_idx=[-1,1]` 覆盖；**prefill 仍未覆盖**（§4 G3） |
| **S-6** | **参考的输入链**（r2 复审 P2） | 规则独立（S-1 修好后），但 ②③④ 的参考**吃设备上一段的 dump**（`B_ids_body.bin` / `B_emb.bin` / `B_kv.bin`）⇒ 属 `docs/17 §1.3` 的 **②**，必须逐条指明覆盖它的判据 | 已按 §1.3 在 `ple/README.md **§4.1**` 给出三分表 + 逐条覆盖判据（← `B1.ids.B` / `B2.emb` / `B3.kv`）+ ③→④ 的 T3 残差风险；③（被判量自身产物当参考输入）经逐项归类为**无** |
| S-5 | ④ 的 ε 宽度 | 旧 ε（`EPS_RSQRT+EPS_SIGMOID`）在参考侧到边（`max(d/bound)=1.000`）⇒ 掩盖了 P2 一类 1-ulp 差 | 修 D4 后 `B4.normed` 的裕度回到 `0.000`；`cmp_t3` 现在**报告 `max(d/bound)`**（报告项）；`B3.kv` 仍是 `0.999`，原因是"恰好差 1 个 bf16 格点"落在 `1.0·ulp(out)` 项内（**结构性解释**，非掩盖） |

## 3. 两份上游实现之间的差异（我们选哪边）

| # | triton（生产路径） | eager（CPU 参考） | 我们跟随 | 我们的判据能否区分 |
|---|---|---|---|---|
| U1 | 融合 `kv_proj`（key→0、value→1） | 分开的 `key_proj`/`value_proj` | checkpoint 命名 + key 在前 | 值相同；顺序反由 `B3.kv` 抓（变异 bit7） |
| U2 | EOS 回退用 `crossed` 累计位（`ops/ple.py:78-98`） | 用 `prior EOS` 的 cummax + `position_in_segment >= shift`（`ngram_embedding.py:244-261,276`） | **triton** | 回扫时按 3000 组随机用例（多请求、tokens/ctx 内都有 EOS、NGR=3）逐一比过**两种机制**：**0 处不同** ⇒ 机制不同、**取值等价**；若真错，`B1.ids.*`（变异 bit0/bit1）会抓 |
| U3 | prefill/spec 用**独立** writeback kernel（`shift=qlen`、`WRITE_W=STATE_LEN`） | 批量 `F.conv1d`（`cat([state,x])` 后取尾） | 都没实现（只做 decode） | **不能** —— 明确列为未覆盖（G3） |
| U4 | 外层残差**融进**卷积核（`out=outer_residual+bf16(residual+conv)`） | 返回 `gated+conv`，**调用方**再加 `hidden` | triton | 数值等价（舍入点相同）；只有归属差异 |
| U5 | 每个边界都显式 `.to(dtype)` | 靠 bf16 tensor 运算自带的取整 | 一致 | 这条正是 D4 的**第二份**证据：两份上游**都**支持"和先取整" |

## 4. 上游有、我们没有（**未实现 / 未覆盖**，不声称完整）

| # | 项 | 位置 | 为什么现在不做 |
|---|---|---|---|
| G1 | `dequantize` 钩子 | `ple_layer.py:414`、`common/ngram_embedding.py:98-104` | 本 checkpoint 的 PLE 全 BF16 ⇒ 恒等映射（`PLE_SPEC.md §1` 已记） |
| G2 | prefetch（提前一层发起） | `ngram_embedding.py:366-383`、`model.py:482` | 纯性能机制，与数值无关 |
| G3 | prefill / spec 的 short-conv 写回 | `ops/ple.py:489-569` | **未实现**；本 mission 只做 decode（`PLE_SPEC.md §4.1` 已声明） |
| G4 | `has_initial_states` 门 | `ops/ple.py:394-404` | decode 路径上 `has_init ≡ state_ok`（`HAS_INIT=false`）⇒ 与我们的 null 判据等价，未单独实现 |
| G5 | 词表尾部 padding 公式 | `ngram_embedding.py:200-201` `ceil(total/divisor)*divisor` | SPEC 只记了行数 320001536，没记公式；**本轮补记**（`PLE_SPEC.md §2.1` 末尾） |
| G6 | ETP/DP 的 gather+reduce | `common/ngram_embedding.py:106-147,491-500` | ETP=DP=1 时是恒等 |
| G7 | `attn_metadata is None` 的退化路径 | `ple_layer.py:348-349`（`residual.add_(outer_residual)`） | 只在 profiling 下触发 |

## 5. 复查复审提示过的三处（P3）

| # | P3 项 | 处置 |
|---|---|---|
| P3-1 | `m15_ple.asc` 顶部 PROLOGUE 说核内用 `BufAcquire/BufRelease`，实际只有 `PipeBarrier` | 已按事实改写 PROLOGUE（不再声称 BufferID 流水；并把"M15 r1 的旧叙述"写清楚） |
| P3-2 | 判据 `n`/`bad` 单位混（字节 vs 元素） | `cmp_bits` 现在**按元素**计 `n`/`bad`，detail 里另列字节差；README/review-request 的单位随之更正（`B1.ids.A` = 160 元素 / 1280 字节） |
| P3-3 | 复审申请里"T1 4 / T3 6 / T4 1"与 §1 表不符 | README §1 重新计数为 **T1 家族 6 / T3 家族 6 / T4 1 = 13 条**（以脚本打印为准） |
| P3-4 | `B3.kv`/`B4.normed` 贴着容差边 | 见 S-5：`max(d/bound)` 进报告项；D4 修完后 `B4.normed` 裕度 0.000，`B3.kv` 0.999 有结构性解释 |
| P3-5 | `B5.state_evolve` 只比参考侧 | 已改为**设备产物** `state_out` vs **设备输入** `state_in`（活跃行），并把"设备侧同口径"落实（判据文字同步更正） |
| P3-7（r2） | 「13 条有咬合力」是标签 | 判据脚本不再打印该标签；`m15_ple_mutants.py` 加 coverage 强制核查（13/13，未覆盖即非 0 退出）；补 bit10/11/12/13 四个变异位 |
| P3-8（r2） | README §3 的 md5 措辞 | 改为「整个 `--brief` stdout 的 md5」，并另列「只取 `RESULT\|` 行的 md5」 |
| P3-6 | `gen_ple_data.py` 自己实现质数搜索 | 保留：它只用于**与 checkpoint 的值对账**（判据侧的真值仍取 checkpoint），`docs/15:458` 的意图（不要用自算值当判据真值）没有被违反 |
