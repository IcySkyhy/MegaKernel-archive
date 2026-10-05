# M19 QSA indexer 前端 kernel

> # ✅ 选择段已通过（M53 收口——读结论之前先读这里）
>
> **① 数值前端**：T1 投影 bf16 **逐位一致**（4/4 档位差 0）；T3 `q`/`ck`/`分数` 按 `docs/17 §1.1`
> 逐元素判据（`|out−ref| ≤ ε·Σ|terms| + 0.5·ulp(out)`，ε 推导见 §3.2）**四档 0 违反**。
> **② 离散选择段**：默认 4 档计数 **2048 / 2050 / 1024 / 2048**（= 应达值），且**选中集合**
> 与「全排序 + 平局按索引升序」的暴力 oracle 在 **kernel 自身分数**上**逐块完全一致**（4/4 档，§3.4）；
> 小档矩阵 `V∈{256,2048} × budget∈{8,32,512} × coreDiv∈{1,2,4,8,56}` 全部逐块通过，
> 且**同档位跨核数 logits 位同、选中集合同**（多核 == 单核，§3.4）；
> 默认 4 档 **9 次独立进程**计数逐次相同（`evidence/select_single_core.log`）。
> **③ M35 的 partial-landing 横幅与「未标注 `M19_ACK_INCOMPLETE_SELECTION=1` 即拒绝运行」守卫已移除**
> （mission 允许的移除条件已满足：① 默认 4 档达标 ② 9 次确定性；读数即在证据文件里）。
> **为何是「删除」而不是改成「默认放行」**：该守卫的语义是「阻止未验收的选择输出接下游」，
> 验收通过后前提消失；保留一个恒真的开关只会变成误导性的仪式，并让 `exit 2` 与「输入缺失」
> 这一同族三态码混淆。故 host 入口不再携带任何拒绝机制，选择输出**可以**接下游消费者。
>
> **⑤ 证据文件来源**：`evidence/select_single_core.log` 现在是
> `bash m19_qsa_indexer/run_select_matrix.sh` **一次运行**的完整输出（判据退出码逐级传递：
> 任一子判据非零 ⇒ 脚本 `exit 1`；每个 dump 目录先断言 `*_meta.txt` 存在）。
> 更早版本（`149b920`）的该文件曾出现「②③⑤ 段是手工追加的补跑」，`7edd74d` 起不再有这种拼接。
> 因此本模块的**选择输出可以接下游消费者**；`M19_ACK_INCOMPLETE_SELECTION` 不再是必需项。
> **三态退出码（仍有效）**：**0 = 判据比过且通过；1 = 比过但有差异/FAIL；2 = 没有真正执行比较**（如输入缺失）。
> ④ M35 归档的 6 条机理 + 1 条否证 + 1 条教训，逐条复核结论见 **`evidence/MECHANISMS_FOUND.md`**
> （其中 (ceq) 那条**原结论被本轮实测否证**：GM 与读回一致，真正错的是"把 `score ≥ K` 的计数当 tie 用"）。

12 个 `full_attention` 层实际是 QSA 稀疏检索注意力（docs/11 §0）。本模块做**前端**：
`index_qk_proj → q/k 的 GemmaRMSNorm + partial RoPE → 压缩缓存写入 → 打分 → 选 512 个
block → expand 成 packed 索引`，输出与 vLLM packed 契约同形，供 attention 主体的
sparse core 消费。

- 权威源码：`/workspace/vllm/vllm/models/qwen4_exp/`（`nvidia/indexer_qsa.py`、
  `nvidia/ops/qsa_indexer.py`、`nvidia/ops/qsa.py::_compress_qsa_groups_kernel`、
  `common/qsa_cache.py`、`nvidia/ops/qsa_pre_indexer.py`）
- donor：`ops-transformer/attention/lightning_indexer{,_v2}`（radix top-k）、
  `bsa_select_block_mask`（跨核 radix + 精确并列处理）
- 约束（用户定）：`mix(1,2)` 全核；只用基础 API（无 Matmul 高阶 / TPipe / TBuf /
  TQue / AllocTensor）；BufferID / CrossCore / 地址全编译期静态；尽量不挂 PIPE_S
- 本模型常量：hidden 2560、indexer 4 头 × 128 维 + 1 KV 头、`budget=2048`、
  `compress_ratio=4`、`block_topk=512`、`rotary_dim=64`（共用主 attention 的
  rotary，`partial_rotary_factor=0.25`、θ=1e7）、`rms_norm_eps=1e-6`、
  `mrope_section=(11,11,10)`、`mrope_interleaved=true`

---

## 1. 张量级契约（逐条对源码取证）

### 1.1 投影

`ReplicatedLinear(hidden_size, (4+1)*128)`，无 bias（`nvidia/indexer_qsa.py:131-137`）。
输出的列布局 = **先 q 后 k**：`[0,512)` 为 q（4 头 × 128，先头后维），
`[512,640)` 为唯一 raw K（`indexer_qsa.py:277-283`）。checkpoint 张量
`self_attn.indexer.index_qk_proj.weight` = BF16 `[640, 2560]`；`q_layernorm.weight` /
`k_layernorm.weight` = BF16 `[128]`（已实测：值域约 ±0.5，均值 −0.04 ⇒ Gemma 约定
`1+w`，见 §1.2）。

### 1.2 GemmaRMSNorm（q 与 compressed k 都用）

`vllm/model_executor/layers/layernorm.py:140-168` + flashinfer `gemma_rmsnorm`：
`weight = w.float() + 1`，`y = rms_norm(x, weight, eps)`，即

```
rstd = 1 / sqrt( mean_{i=0..127}(x_i^2) + 1e-6 )     # fp32
y_i  = x_i * rstd * (1 + w_i)                        # 再取整到 bf16
```

- 输入 x 是**投影后已取整到 bf16** 的值（`projected_qk` 的 dtype），norm 内部升 fp32。
- eps = `config.rms_norm_eps` = 1e-6。
- 输出取整到 indexer dtype（本模型 bf16）。

### 1.3 Partial NeoX RoPE（rotary_dim = 64 于 128 维头内）

indexer 共用主 attention 的 `rotary_emb`（`nvidia/qsa.py:345-349`）：
`head_size=256`、`rotary_dim=int(256*0.25)=64`、`is_neox_style=True`、θ=1e7。

- 只旋每头 **前 64 维**，`[64,128)` 直通（`apply_qsa_rope`，`indexer_qsa.py:29-60`）。
- neox 配对：`out[j] = x[j]*cos[j] − x[j+32]*sin[j]`，`out[j+32] = x[j+32]*cos[j] + x[j]*sin[j]`，
  j ∈ [0,32)；频率 `cos(p·θ^(−2j/64))`。
- q 用 token 位置 p；compressed k 用**组首位置** `first_position = p − 3 = 4g`
  （`nvidia/ops/qsa.py:435-476`）。
- **MRoPE 在本模型退化为普通 RoPE**（文本-only）：`_triton_mrope_forward` 的
  interleaved 分支对每个频率对只从**同一个 offset** 取一个轴的值
  （`rotary_embedding/mrope.py:66-90`：`h/w` 掩码按 `offset%3` 划分，
  `cos_row = t_cos_row + h_cos_row + w_cos_row`），而三轴位置在纯文本输入下相等
  ⇒ 结果与普通 1D RoPE 逐元素相同。多模态输入（三轴不同）时不等价——见 §5。
- cos/sin 缓存 dtype = 模型 dtype（bf16，`base.py:58-63`）。本 kernel 的 host 直接
  生成 bf16 取整后的表并同时喂给参考，避免两边表精度不一致。

### 1.4 压缩（compress_ratio = 4）与两个 cache

`_compress_qsa_groups_kernel`（`nvidia/ops/qsa.py:330-476`）逐字结论：

```
pooled = bf16_rne( ( Σ_{i=0..3} fp32(raw_k[p−3+i]) ) / 4 )     # 位置序左结合求和，再 bf16
ck     = rope_neox_partial( gemma_norm(pooled, k_layernorm.weight), pos = p−3 )
```

- **均值**（非求和/最大值），4 个成员是**未 norm、未 rope 的原始投影 k**。
- 只有 `(logical_position + 1) % 4 == 0`（组边界）且 `p ≥ 3` 的行**才产生**压缩行；
  未完成的组不产生压缩行。
- **raw key cache 是压缩器状态环**（`QSAKeyStateCache`，`common/qsa_cache.py:838-856`）：
  `CircularBufferSpec`，容量 `4 * ceil((4 + num_spec_tokens)/4)`（无投机采样 = 4），
  槽位 = `physical_block * 4 + (position % 4)`；行内容 = bf16 raw key（+ 可选 int64×3
  MRoPE 位置尾）。**它不是全历史 cache**，只服务"凑齐 4 个组成一行"。
- **compressed key cache**（`QSACompressedKeyCache`）：`MLAAttentionSpec`，
  `tokens_per_state=4`，物理视图 `[blocks, num_states, 1, 128]`，`num_states = block_size/4`
  （本工程 block_size=16 ⇒ 4 行/页）；槽位 = `physical_block * num_states + (g % num_states)`，
  `physical_block = block_table[req][g // num_states]`，其中 `g = position // 4`。
- 本模块的 kernel 用**请求内连续**的等价布局（raw ring 4×128、compressed `[V,128]`），
  paged 间接层是 `block_table` 查表的一层包装（见 §5 与 docs/11 §3）。

### 1.5 打分与可见范围

`nvidia/ops/qsa_indexer.py:94-107`（decode）/`:206-218`（prefill）：

```
score[row, b] = Σ_{h=0..3} max( dot_fp32( ck[b], q[row,h,:] ), 0 )     # 无 1/sqrt(d)、无任何 scale
```

- 列 `b` = **压缩行序号**（= 逻辑组号 g），物理位置经 block table 映射。
- 只在 `b < visible_blocks[row]` 上打分；`visible_blocks = max(0, min((p+1)//4, seq_len//4))`
  （`common/qsa_cache.py:268-275`）——**不含当前正在累积的 open group**：只有
  `p ≡ 3 (mod 4)` 时该组才算可见。
- 打分与 attention 主体的 scale 无关（主体 `256^-0.5`）。

### 1.6 选择与 packed 输出

- `block_topk = indexer_budget // compress_ratio = 512`（严格 top-k，`qsa_indexer.py:471-499`）。
- `visible_blocks ≤ 512` 时 vLLM 走 trivial 路径：`block_indices = [0,1,…,V−1, −1,…]`
  （`csrc/.../persistent_topk.cuh:444-455`）⇒ 等价于"全部可见块都选中"。
- 并列：vLLM 的 top-k 用"fp32 单调 key + histogram/radix"选择，**并列内部次序由
  atomic 决定，无 index tie-break** ⇒ 契约层面只要求**集合**正确。
- `expand`（`qsa_indexer.py:221-276`）产出 `out[m, 2052] int32`：
  - 列 `[0, 2048)`：每个选中 block b 展开为 `4b + {0,1,2,3}`（**请求内绝对 token 位置**）；
  - 列 `[2048, 2048+tail_count)`：open group 的 token `V*4 … p`，`tail_count = (p+1) − 4V ∈ [0,3]`；
  - 其余列 = `-1`；
  - **末列 2051 = 有效项数 = `4·min(V,512) + tail_count`**（sparse attention 的 tile 循环上界，
    绝不是 token 索引；`nvidia/ops/qsa.py:80-134` 消费之）。
- 消费侧（`qsa_sparse_paged_attention`）把这 2051 个 token 位置按 `block_table[req][token//block_size]`
  映射到**主 KV cache**（不是 indexer 的两个 cache）做稠密注意力——即 QSA 只稀疏了"读哪些 token"。

### 1.7 decode 档 vs prefill 档

数学完全一致（同一 score 公式 / 同一 top-k / 同一 expand），差异只在批形态与 tiling：
decode 每 request 的 query 行数相同（`decode_query_len`，纯 decode = 1，投机验证 = 1+spec），
prefill 按 `TILE_R` 切行、`logits_width` 圆整到 64 的倍数（`qsa_indexer.py:573-640`）。
每行**独立**取自己的 top-512。

---

## 2. 本模块实现

### 2.1 文件

| 文件 | 说明 |
|---|---|
| `m19_qsa_indexer.asc` | kernel + host（确定性输入、运行、不变量自检、dump） |
| `check_ref.py` | numpy/double 参考 + 判据（T1/T3 + 离散两条，见 §3.1） |
| `check_select.py` | **选择段逐块 oracle**（全排序 + 平局规则）与 `--cross` 多核一致性对照 |
| `run_select_matrix.sh` | 单核先/多核回归矩阵 + 9 次确定性的可复现证据生成（§4） |
| `probe_aiv_barrier.asc` | AIV 间计数交换的**适用边界**探针（6 变体；"选择段为何单核"的证据） |
| `probe_aic_blockidx.asc` | **已撤回（M156，不构建）**：AIC 分支经 UB 暂存落盘，AIC 访问不了 UB ⇒ 全 FAIL 为伪影；见 §5.4 |
| `check_probe_retraction.sh` | 撤回状态的**可复算守卫**（任一断言不成立即非零退出；§5.4 / `evidence/probe_aic_blockidx_retraction.log`） |
| `extract_indexer_weight.py` | 从 checkpoint 抽 layer3 的 indexer 三个权重到 `data/` |
| `data/layer3_*.bin` | `index_qk_proj[640,2560]`、`q_layernorm[128]`、`k_layernorm[128]`（bf16 原始字节，byte-verbatim） |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`） |

### 2.2 kernel 结构（`__mix__(1,2)`，blockDim=28 ⇒ 28 AIC + 56 AIV）

```
AIC 0        : bf16 GEMM  x[1,2560] @ W[640,2560]^T -> qk[1,640]   （结构 lift 自 m11_bf16_gemm）
               mode0 屏障(全 AIC) → mode2 逐对通知 2 个 AIV
AIV 0..3     : q 头 h = 0..3：GemmaRMSNorm(128) + partial NeoX RoPE(pos=p) → qOut(fp32)
AIV 4        : raw k：写 ring[slot] / kOut；若 p%4==3：pooled=bf16(mean 4 行) →
               GemmaRMSNorm(wk) + RoPE(pos=p-3) → 打包 bf16 → 写 comp[g] / ckOut
AIV 全体     :=B1 屏障=（发布 cache 与输出预填）
               逐 chunk(64 行) 打分：score = Σ_h max(dot(ck,q_h),0)（fp32）
               → DumpLogits：把本核那 64 列分数写进 GM logits[0..V)      ← **S1 并行打分**
               → WriteToken：写 4B 交接记号（与分数同 MTE3 管，顺序保证）
AIV 0..31    :=S2= 全体 barrier(FLAG_HANDOFF) + AIV0 读回记号数到齐     ← 一次交接
AIV 0        :=S3/S4= **单核选择**（读 GM 分块，见 §3.4）：
               radix（1 bit/轮 ×32，逐位贪心求第 blkTopk 大 key）
               → 按列序压实：GT 组 rank=运行序、tie 组 rank=gtTotal+已用 tie 数
               → tail + count 列（倒数第二列 2051）
其余 55 AIV   : 只参与 S1 打分与交接，选择段不参与（空转）
```

**选块算法**：对 fp32 分数的**位型**（score ≥ 0 ⇒ 位型按 int32 单调）做 MSD 逐位二分——
`K` = 满足 `count(key ≥ K) ≥ blkTopk` 的最大 key（32 位精确），再取 `{score > K}` 全部 +
`{score == K}` 中按索引升序补足 `blkTopk`。`V ≤ blkTopk` 时跳过搜索（全部选中，等价 vLLM trivial 路径）。

### 2.3 静态资源表

- **BufferID（AIC）**：0/1 A ping-pong、2/3 B ping-pong、4/5 L0 ping-pong、6 L0C（与 m1/m11 同编号）
- **BufferID（AIV）**：8 GEMM/ring staging、9 权重+cos/sin、10 压缩行 chunk、12 scores、
  13/14 跨核计数交换、15 输出（预填 + packed 写）、16 结果打包
- **CrossCore flag（M53 起只剩两处 AIV 间用法，都是 mode0 单发）**：0 = AIC→AIV（mode 2）；
  9 = B1 屏障（q/k/cache 就绪 + 输出预填完成）；**8 = 分数交接屏障**（每核写完自己的分数与 4B
  交接记号后置位，**单发一次**）；10 = 保留（历史口径：原 radix 的跨核计数交换已随选择段单核化删除）；
  11 = AIC 间 mode0 屏障（4 个 N 块落盘）
- **UB**：`m19_qsa_indexer.asc` 顶部常量表（UB_GSTG … UB_TOT ≈ 35KB « 248KB），全部 32B 对齐
- **GM**：host 侧分配（x、W、wq/wk、cos/sin×2、ring、comp[V_MAX,128]、qk、qOut/kOut/ckOut、
  logits、out[2052]、cnt 交换区）

### 2.4 3510 quirk 规避（沿用 m0/m7/m11 结论）

GM 一律按 bf16 位视图寻址；UB 偏移 32B 对齐；`LocalMemBar` 只在 `__VEC_SCOPE__` 内；
同 pipe 背靠背 DMA 之间插 `PipeBarrier`；`Reg::Arange` 只填 64 lane（全部按 64-lane 分块）；
Nd2Nz 单行退化 ⇒ GEMM 的 calcM ≥ 2；fp32→bf16 用整数 RNE 位打包（不用 Reg Cast）；
UB 标量读（`GetValue`，S 管）前插 `V_S` 事件（有值依赖，属 `docs/05 §2` 允许的 PIPE_S 例外，见 §5.5）；
「标量写 UB → MTE3 搬」走 `BufAcq/BufRel<PIPE_S>` 阻塞释放 + 对侧 `BufAcq<MTE3>`（`docs/05 §6.1 ⓔ`）。

---

## 3. 判据与当前结果

### 3.1 判据（`check_ref.py`，全部为可复现的硬判据）

1. **投影**：kernel 的 bf16 `qk` 与"双精度投影→bf16(RNE)" **逐位一致**（位差 0）。
2. **q / compressed k**：按 `docs/17 §1.1` **逐元素**判据 `|out − ref| ≤ ε·Σ|terms| + 0.5·ulp(out)`
   （ε 逐项推导见 §3.2），**违反元素数必须为 0**。
3. **分数**：同上一口径（`Σ|terms| = Σ_h Σ_d |q_hd·k_bd|`），**违反元素数必须为 0**。
   （旧口径 `≤1e-3×max(1,|ref|)` / `≤2e-4×max(1,max score)` 已被 `docs/17 §1.1` 取代；
   D 档分数 maxRel 达 1e28 正是它不可用的原因。）
4. **【离散判据】选中集合**（M53 起分两条互补判据，逐块/逐元素，见 §3.4）：
   (i) **精确**：与「全排序 + 平局按索引升序」的暴力 top-k 在 **kernel 自身 dump 的分数**上
       **逐块相等**（`check_select.py`；`check_ref.py` 判定项里也跑一遍）；
   (ii) **可证条带**（对 double 参考分数）：`K` = 参考第 `blkTopk` 大、`Bmax = max_b B(b)`
       （`B(b)` 即判据 3 的逐元素界）⇒ `score > K + 2·Bmax` 必选中、`< K − 2·Bmax` 必不选中，
       落在 `±2·Bmax` 内的**不可判**（只报数）。旧口径「miss/extra 必须为 0」在 T3 分数下
       **不可能满足**（实测 A 档 `|score−K| < 0.05` 就有 12 个 block），故降级为**报告项**。
   另有：每块恰 4 个 token（`4b+{0,1,2,3}`）、count 列 = `4·min(V,blkTopk)+tail_count`、
   tail 内容、padding 全 −1。

### 3.2 实测与判据分档（Ascend950PR 真机；`evidence/{accept_run,check_ref_run,probe_run}.log`）

**分档声明（`docs/17 §1.1`，逐项理由）**

| 量 | 档 | 理由与判据 |
|---|---|---|
| `index_qk_proj` bf16 输出 | **T1**（位域） | 纯 k 维累加 + 一次 F322BF16 落盘；判据 = **逐位**（实测位差 0） |
| **选中集合 / expanded token 集合** | **T1 离散** | top-k 索引域；判据 = **集合完全一致**，任何容差档都不可替代 |
| `q` / `ck` | **T3** | 触发条件：含 `sin`/`cos` 超越函数近似 + bf16 落盘量化 + 跨段累加 |
| `score` | **T3** | 同上 + 128 维点积与 4 头求和的长累加链 |

T3 的界（逐项推导，实测**未超界**；`check_ref.py::t3_check` 已按 `docs/17 §1.1` **逐元素**判定
（超界元素逐个打印），下表 §3.2 的 ✅ 是**判定项**；`maxRel`/`≤1ulp 比例` 为**报告项**）：

```
ε ≈ ε_cos + ε_bf16 + ε_domain
  ε_cos    ≈ 2^-9      cos/sin 表按 vLLM 同精度取整到 bf16（host 用 double 计算后 RNE）
  ε_bf16   ≈ 2^-9      每个 norm/rope 结果落盘一次 bf16 量化
  ε_domain ≈ 2·2^-9    **运筹域差异**：融合核在 bf16 上做旋转（y 取整后相乘/相减各一次舍入），
                       本实现按未融合 1D CUDA 路径在 fp32 域旋转、只落盘一次取整
  ⇒ ε ≈ 1.5e-3（相对 Σ|terms| 口径）
判据：|out − ref| ≤ ε·Σ|terms| + 0.5·ulp(out)，逐元素；报告项 = maxRel、≤1ulp 比例
```

**判定项结果（4 case：A 关组 V=2048 / B 开组 V=2048+tail / C V=256 / D V=65536）**

| 判定项 | A | B | C | D |
|---|---|---|---|---|
| 全链路 launch OK（无挂死） | ✅ | ✅ | ✅ | ✅ |
| **T1** 投影 bf16 **逐位** | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| **T3** q（逐元素违反数） | ✅ 0/512 | ✅ 0/512 | ✅ 0/512 | ✅ 0/512 |
| **T3** ck（逐元素违反数） | ✅ 0/128 | — | ✅ 0/128 | ✅ 0/128 |
| **T3** 分数（逐元素违反数） | ✅ 0/2048 | ✅ 0/2048 | ✅ 0/256 | ✅ 0/65536 |
| **T1 离散(i)** 逐块 == 自身分数 top-k | ✅ 0 差集 | ✅ 0 差集 | ✅ 0 差集 | ✅ 0 差集 |
| **T1 离散(ii)** 参考条带（必选/必不选） | ✅ 0 违反 | ✅ 0 违反 | —（V≤blkTopk） | ✅ 0 违反 |
| count 列 | ✅ | ✅ | ✅ | ✅ |

T3 的判定项按 `docs/17 §1.1` 的**逐元素**口径（`|out−ref| ≤ ε·Σ|terms| + 0.5·ulp(out)`，ε 逐项推导见上），
**四档全部 0 违反** ✓；`maxRel`/`≤1ulp 比例` 作报告项列出（`evidence/check_ref_run.log`）。
注意 `maxRel` 在相消元素上可达 1e-1~1e28（D 的分数），这正是 `docs/17` 要求逐元素界而非 maxRel 的原因。

**结论：T1 投影、T3（q/ck/分数）、T1 离散（逐块精确 + 参考条带）全部满足分档要求。**
默认 4 档 token 计数 = **2048 / 2050 / 1024 / 2048**（`evidence/select_single_core.log`）。

### 3.3 本轮的定位与修复记录（都是"实现/交接"类，非数学）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | q 与参考差 4.3 | 多处 UB 生产→消费缺 BufferID 交接（qOut dump / 计数槽→GM / MTE2→V 读回 / tail） | 逐个补齐 acquire/release + `V_S` |
| 2 | ck 出现 NaN | ① 本 token 的 raw k 写 `UB_KR` slot0 后被 i=0 的历史成员 DMA 覆盖；② bf16 落盘用错指令 | ① slot0..2=历史、slot3=本 token；② 改用官方 `Cast`+`StoreAlign<DIST_PACK_B32>` |
| 3 | 分数与参考相关系数仅 0.12 | 每个 AIV 只算自己那 1 个头，扫描却按 4 头读 UB | B1 屏障后 `LoadQ()` 回搬 4 头 q |
| 4 | 计数恒 0 | `CompareScalar<int32_t,…>` 在本机取值不可靠；`ReadGlobalCount` 的 GM 地址把 bf16 元素偏移当字节再除 2 | 计数改"UB 别名往返 → 浮点阈值"比较；地址改为直接用元素偏移 |
| 5 | AIV 前缀 rank 错（aiv=1 得 128） | `UpdateMask(aiv)` 实测选到多于 1 条 lane | 改用"lane 序号浮点常量 < aiv"的 `CompareScalar` |
| 6 | token 全为 `INT32_MAX` | 抽出时对**分数**做 EQ 比较（不可靠）+ `Muls(-4)` 号错 | 改为 Neg+MAX 取**最小索引**、`Muls(+4)`，剔除用 mask+Select |
| 7 | radix（V>512）选块错 | 每个候选的计数**恒写到 candIdx=0 的槽**，读数却按 `(d-1)` 偏 | 写/读槽位对齐（bf16 元素偏移） |

**方法学（记入本仓库经验）**
- **"选错落盘指令" ≠ "指令语义不同"**：本轮 `DeInterleave` 那条结论按"查文档定义 → 试参数空间 →
  看官方 donor 同类位置"三步复评后**撤回**——官方在 norm+RotE 的 bf16 落盘用的是
  `Cast` + `StoreAlign<DIST_PACK_B32>`（`kv_rms_norm_rope_cache_regbase_base.h:226-231`）。
  ⇒ 以后再遇到"语义不符"，先假设是"我们在用非官方惯用法"。
- 绕行方案（压缩行按 fp32 存、值落在 bf16 网格）**保留为后备、明确标注不要启用** —— 官方 `Cast` +
  `StoreAlign<DIST_PACK_B32>`（本表第 2 行、`m19_qsa_indexer.asc:645`）已解决该问题，绕行不再需要。
  **（M83 校正）** 本句原引「§5.9」：它指 M53 之前的旧 README §5「已知限制 / 后续」的第 9 条
  「不要启用压缩行 fp32 缓存绕行」（见 `5225764`）；该清单已在 M53 收口时被 §5.1–§5.3 取代、条目不再存在，
  故此处就地补明引用对象（指向 §3.3 本表），不另立 §5.9 节。

### 3.4 单核 vs 多核的一致性论证（M53 方法硬要求）

**结论先说**：选择段现在只有**一条**逻辑，跑在一个核（`SEL_CORE = AIV 0`）上，读 GM 上的分数向量；
`coreDiv`（核数）只影响**打分的列切分**。因此「多核 == 单核」是**构造性质**，并且有实测：

| 断言 | 判据 | 读数 |
|---|---|---|
| 打分逐列独立于切分 | 同档位、不同 `coreDiv ∈ {1,2,4,8,56}` 的 `logits` **位同** | `check_select.py --cross`：6 档 × 4 组对照全 `CROSS-CONSISTENT` |
| 选择与核数无关 | 同上各档 **选中集合同** | 同上 |
| 单核是第一等公民 | `coreDiv=1` 全档逐块 oracle 通过 | 矩阵 `V∈{256,2048} × budget∈{8,32,512} × coreDiv=1` 全 PASS |
| 选择本身正确 | 与暴力 oracle 逐块相等 | 默认 4 档 + 矩阵 30 组全 0 差集 |

**为何选择段改成单核**（不是「绕过 bug」，是**设计决定 + 实测依据**）：

原实现每轮 radix 都要「56 核各写自己的计数到 GM → 全核读回求和」，共 16 轮**在同一片槽上复用**。
本模块用 `probe_aiv_barrier.asc` 把这条用法的**边界**单独测了（6 个变体并列，都带偏斜以放大弱同步；
每变体每核记「违规次数 / 见到的最少槽数」，读数见 `evidence/probe_aiv_barrier.log`）：

| 变体 | 用法 | 违规核 | 见到的最少槽数 | 判定 |
|---|---|---|---|---|
| v0 | 无屏障（对照，4 轮） | 56/56 | 1/56 | 对照生效 ⇒ 探针敏感 |
| **v1** | **单发 1 轮**：`Set<0,MTE3>(1)` + `Wait<0,MTE2>`（**本 kernel 交接段现在的用法**） | **0/56** | **56/56** | ✅ **成立** |
| v2 | 循环 4 轮**复用同一片槽**：`Set<0,MTE3>(1+r%8)` + `Wait<0,MTE2>`（**原 radix 的用法**） | **每次都违规**（见下） | 见下数字口径 | ❌ 不成立 |
| v3 | 循环 4 轮 + set 前 `MTE3_S` 排空（`docs/06 §5.2 规则 6` 的处方） | **每次都违规**（见下） | 见下数字口径 | ❌ 仍不成立 |
| **v4** | **单发 1 轮 + 官方 `SyncAll<true>()`**（AIV-only 全核屏障，wait 挂 **PIPE_S**） | **0/56** | **56/56** | ✅ **成立** |
| v5 | 循环 4 轮 + 官方 `SyncAll<true>()` | **每次都违规**（见下） | 见下数字口径 | ❌ 仍不成立 |

**数字口径（只覆盖本仓已归档的运行，不作为上下界）**：v2/v3/v5 **每次运行都违规**——本仓已归档的**5 次独立运行**（`evidence/probe_aiv_barrier.log` 1 次 + `evidence/probe_aiv_barrier_runs.log` 4 次）里，违规核数落在 **18~52/56**、单次最少见到 **1 个槽**；**v1/v4 在这 5 次里都是 0/56、v0 都是 56/56**（定性结论每次运行都稳定，跨运行波动的只有 v2/v3/v5 的具体数字）。**未归档的抽跑读数不作为口径依据。**

**适用范围前提**：上述「多轮复用不可证」是在**每轮只挂一道「写完」屏障**（写 → 屏障 → 读回 → 下一轮写）的形态下观测到的——**读回之后到下一轮写之前没有任何屏障**。本探针**不**据此下「就是缺一道屏障」的结论（该假设由另一个 probe mission 在实测，结论未出）；只说明读数成立的前提。

**这条探针排除了一个很自然的错误归因**：官方 `SyncAll<true>()` 的实际形态可直接在工具链里核实——
`asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h`：`constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;`（:125）
与 `wait_flag_dev(PIPE_S, AscendC::SYNC_AIV_ONLY_ALL)`（同文件 SyncAll 实现），即
「mode0 + flagId 14 + wait 挂 **PIPE_S**」，与 v4/v5 的实现一致。而 v4/v5 的读数说明：
**换成官方 `PIPE_S` 形态后，单发仍然 0/56、循环仍然每次都违规（见上数字口径）** ⇒
「本模块的 wait 挂 `PIPE_MTE2` 才是根因」**不成立**（挂 MTE2 的单发 v1 同样 0/56）。

⇒ 正确表述是**适用边界**，不是「原语不可用」：

- **单发一次交接可证**（v1 与官方 v4 都 0/56）——**本 kernel 新的交接段正是这种用法**，所以它的正确性
  其实**不依赖**「复用是否可靠」这个未解问题；
- **同一片槽多轮复用不可证**（v2/v3/v5 都违规；换成官方 `SyncAll<true>` 也救不了 ⇒ 瓶颈是「复用 + 无重校验」，
  不是「缺一个好屏障」）；
- **与本仓既有结论一致**：`docs/06-m0-bringup.md §5.3` 记「CrossCore mode0/mode2（`__mix__(1,2)` 直调）**可用**」
  （m0 就是「56 AIV 写槽 → mode0 屏障 → 读回求和」并通过）；`§5.2 规则 6` 甚至预记了本现象——
  「`CrossCoreSetFlag<0,MTE3>` 发出前必须确认 GM 写已落盘（`MTE3_S` 排空后再 set），否则先到的核对
  等不到全部 56 份数据（**实测只收到 ~20/56**）」，与本仓 v2/v3 的读数同量级（见上数字口径）。
  `MTE3_S` 排空实测**不够**（v3 仍违规）；本轮补测还覆盖了此前从未测过的两个变量——
  **`wait` 挂 `PIPE_S`** 与 **flagId 14（官方 `SyncAll<true>`）**（v4 单发成立、v5 复用不成立）。

主 kernel 上的症状与之一致：**每个 AIV 收敛到的 `K` 随机分裂**（同一 case 连跑 3 次，分裂核与
多数派 `K` 都不同），计数却仍"看起来对"——B 档实测选中集合与自身 logits 的理想 top-512 差 **84 个 block**。
⇒ 只要选择段依赖"多核计数交换"，就没有可用的正确性基础。改为单核后核间只剩**一次交接**，
并且这次交接是**被校验的**：

- 每核写完自己的分数后写 4B「交接记号」（同一 MTE3 管 ⇒ 记号可见即分数已交付）；
- 全体 barrier 后由 AIV0 读回记号，**只有到齐才继续**，读数落 trace（`handoffOK/tokenSeen`）；
- 兜底：分数段在 host 侧整段预填 `0xFFFFFFFF`(NaN)，任何未写的列与阈值比较都为假 ⇒
  若交接真出问题，选中集合会立刻少块，被 §3.1 判据 5 抓住。

**代价（如实记录）**：选择段从「56 核并行」变为「1 核串行读 GM 分块」，D 档（V=65536）增加
约 32 轮 × 16 块 = 512 次 16KB DMA + 一遍压实扫描；选择段是 O(V·32) 而不是并发 O(V·32/56)。
打分（本模块的算力主项）仍然全核并行。**若后续要把选择段并行化**，前置条件是
**一个带「重校验 / 消费复位」语义的多轮交换协议**（不是换屏障原语——v5 已证明官方 `SyncAll<true>`
在复用下同样失败），否则会退回同一个坑。

**packed 布局与 `M19_BUDGET` 覆盖的关系（P2-4 复查发现的真问题，已修）**：packed 的 tail 区与 count 列是
**模型固定列**（`[BUDGET, BUDGET+tail_count)` 与 `OUT_WIDTH=2051`），`M19_BUDGET` 覆盖只改"选多少块"
（`blkTopk = budget/4`）。`149b920` 的 host 自检与两个 checker 都把 tail 当成"紧跟块区"（`4·blkTopk`），
在 `blkTopk < 512` 的测试档上定位错列 ⇒ 新增 tail 覆盖后立刻暴露（`tail [-1,-1,-1] != [...]`）。
已按固定列修正 host 自检 / `check_select.py` / `check_ref.py`，并补了"块区与 tail 区之间必须是 −1"这条检查。

**tail 覆盖**：`M19_V` 覆盖时 host 取 `pos = 4V−1`（保证 visible_blocks 与序列长度自洽），
矩阵要带 tail 必须显式给 pos ⇒ 矩阵新增 `M19_POS` 覆盖，档位为
`(256,8/32/512, pos=1023, tail=0)`、`(2048,8, pos=8190, tail=3)`、`(2048,32, pos=8189, tail=2)`、
`(2048,512, pos=8193, tail=2)`；再加默认 A/B/C/D 里 B 档的 `tail=2` ⇒ tail 在多档多长度都有覆盖，
判据逐元素比对 tail 与 padding（`check_select.py`）。

### 3.5 本次改动的位级影响面（M53）

| 改动 | 是否触到位级 | 说明 |
|---|---|---|
| **选择段改单核 + 交接校验** | **否** | 只改"谁在算/数据怎么到位"；分数向量逐位不变（跨核数位同，实测） |
| tie 口径修正（`eq` 前缀口径，见 §3.4/机理 (ceq)） | 否 | 只改 rank 归属，不改任何数值 |
| UB 布局：`UB_SCORE` 1216→2048 列；删 8 个只写不读死区 | 否 | 地址重排；`UB_SELS` 原本与 `UB_TAIL` 重叠（潜在踩踏源）一并消除 |
| **选择段**的动态掩码改「lane 序号向量比较」 | **否**（但更稳） | 原 `UpdateMask(变量)` 路径的读数不稳定；**数值前端未动**（`ScanChunk` 里仍有 `UpdateMask(nc)`、`nc` 为变量，是 main 上原有代码，属「不动前端」） |
| `logits` 由 host 整段（V_MAX）预填 NaN | 否 | 只影响越界列的计数（原为未初始化内容），前端数值不变 |
| 数值前端（投影 / norm / rope / 压缩 / 打分） | **未改动** | T1 位差 0、T3 四档 0 违反保持 |

### 3.6 本机编译约束（踩到的坑，写下来给后来者）

ASC 编译器对 **AIV 循环体**很敏感：循环体里多一条比较、或多一个动态掩码，就会报
`Unsupported scalar instruction in AIV loop`（同一份代码只要把循环体缩小一点就能过，
`-O0/-O1/-O2/-O3` 都一样 ⇒ 不是优化等级问题，是**循环体内寄存器压力/指令形态**）。
本模块因此把 radix 拆成「1 阈值/轮 × 32 轮」、循环内只用满掩码 `full`、阈值一律在循环外算好。
另：`BufAcq` 同一 buffer id 的 MTE2 侧/V 侧 acq **不能嵌套**（会互等挂死）；跨 pipe 交接
一律照 `ScanChunk` 的「MTE2 acq/rel → PipeBarrier<MTE2> → V acq/rel」顺序写。

## 4. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m19_qsa_indexer/build -S m19_qsa_indexer -DCMAKE_BUILD_TYPE=Release
cmake --build m19_qsa_indexer/build -j4          # 并行度别拉满（共用单卡）
mkdir -p m19_out && ./m19_qsa_indexer/build/m19_qsa_indexer          # 4 个 case
./m19_qsa_indexer/build/m19_qsa_indexer 2        # 只跑 case 2（C_small_256）
/usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_ref.py ./m19_out
/usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_select.py ./m19_out   # 逐块 oracle
```

**单核先 / 多核回归矩阵（M53 的可复现证据生成，一次跑完 30 组 + 9 次确定性）**：

```bash
bash m19_qsa_indexer/run_select_matrix.sh          # 产物在 m19_qsa_indexer/m19_sel/（已 ignore）
#   读数落在 evidence/select_single_core.log
/usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_select.py --cross \
    m19_sel/V2048_B8_C1 m19_sel/V2048_B8_C8 m19_sel/V2048_B8_C56     # 多核 == 单核
```

**同步原语探针（AIV 间计数交换的适用边界，6 个变体；它是"选择段为什么必须是单核"的证据）**：

```bash
./m19_qsa_indexer/build/probe_aiv_barrier        # 读数见 evidence/probe_aiv_barrier.log
```

环境变量：`M19_V`（覆盖 V，pos 默认跟着取 4V−1 保持自洽）、`M19_POS`（显式覆盖 pos ⇒ 造 tail>0 档）、
`M19_BUDGET`（token 预算，blkTopk=budget/4）、`M19_CORES`（打分列切分核数 1/2/4/8/56…）、`M19_SINGLE`（**兼容性别名，不推荐**：语义 = `M19_CORES=1`，**仅当 `M19_CORES` 未设时**生效；推荐改用 `M19_CORES`）、
`M19_OUT`（dump 目录）、`M19_PROBE_NVAR`（探针只跑前 N 个变体）。
注：`M19_ACK_INCOMPLETE_SELECTION` 已随守卫移除，不再需要。

权重抽取（一次性）：

```bash
/usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/extract_indexer_weight.py --layer 3
```

## 5. 收口状态：本 mission（M53）/ 已移交 tower

### 5.1 本 mission 内（M53）

| # | 项 | 状态 |
|---|---|---|
| 1 | 先读 M35 交接物并**逐条自验**（`evidence/MECHANISMS_FOUND.md`） | ✅ 6 条 + 1 否证 + 1 教训逐条复核；其中 **(ceq) 原结论被否证**（读回没错、是口径错）、**(tail) 现象不成立**（B 档少的是 1 个 block 不是 2 个 tail token） |
| 2 | **单核一等公民**：`coreDiv=1` + 小档（V∈{256,2048}、budget∈{8,32,512}）先跑通 | ✅ 选择段改为单核（`SEL_CORE=AIV 0`），单核与多核**同一套逻辑**；矩阵 `coreDiv=1` 全档逐块通过 |
| 3 | 与 numpy 暴力 oracle **逐块/逐元素**一致（不是只比计数） | ✅ `check_select.py`：全排序 + 平局规则，30 组矩阵 + 默认 4 档**差集 0**；`check_ref.py` 也加了该判定项 |
| 4 | 单核通过后才做多核，且多核与单核逐步回归 | ✅ `coreDiv∈{1,2,4,8,56}` 同档位；`check_select.py --cross`：**logits 位同 + 选中集合同**（6 档全 CROSS-CONSISTENT） |
| 5 | 修掉 M35 定位的收尾缺陷 | ✅ tie 边界块：真因是 `eq` 计数口径（GT∪tie）被当 tie 前缀；单核化后无配额、按列序补足，计数 = `blkTopk` 精确。tail：复核为误读（tail 一直对） |
| 6 | 默认 4 档计数 = 2048/2050/1024/2048 | ✅（`evidence/select_single_core.log`） |
| 7 | ≥9 次独立进程确定性 | ✅（同上，9 次逐档计数相同） |
| 8 | 去掉横幅与守卫（附"为何可以去掉"的证据） | ✅ 条件满足后移除；证据 = 上面 6/7 两条读数 |
| 9 | README：单核 vs 多核一致性论证 / 位级影响面 / 机理逐条更新 | ✅ §3.4 / §3.5 / `evidence/MECHANISMS_FOUND.md` |
| 10 | 边界：不动 m10/m13/m15/docs/tools；不引入 TPipe/TBuf/TQue；核数用 API 取；除值依赖不挂 PIPE_S | ✅（`coreNum` 由 host 用 `ACL_DEV_ATTR_VECTOR_CORE_NUM` 传入；新增的跨核点只有分数交接这一处值依赖） |

### 5.2 移交 tower / 后续（不在本 mission）

- **AIV 间「多轮计数交换」的协议**（**不是**「屏障原语不可用」——`149b920` 写成后者属过度概括，`7edd74d` 起已修正）：
  单发一次交接**可证**（`probe_aiv_barrier.log` v1/v4 都 0/56），同一片槽多轮复用**不可证**
  （v2/v3/v5 都违规；连官方 `SyncAll<true>` 也救不了 ⇒ 缺的是"重校验/消费复位"语义，不是屏障）。
  若要让选择段回到并行，需要先设计这样一个协议（`docs/06 §5.2 规则 6` 的 `MTE3_S` 排空实测不够）。
  本仓既有结论 `docs/06 §5.3`（mode0/mode2 直调可用）并未被本轮推翻。
- 三条"用法待复核"API（`CompareScalar<int32_t>` / `UpdateMask(变量)` / 对分数做 `EQ`）的三步取证：
  本轮把 `UpdateMask(变量)` 从**选择段**的所有动态掩码里换掉了（**数值前端未动**，读数见机理 (LANEF)/(BAR)），
  其余两条仍待复核。
- AIC 的 `GetBlockIdx` 探针（`probe_aic_blockidx.asc`）**已撤回、不再构建**（M156，见 §5.4）：
  原通路（AIC 经 UB 暂存落盘）不成立，全 FAIL 为伪影；若要重测须按 Fixpipe L0C→GM 在
  **产品路径**上做（属新 mission，不在本模块）。GEMM N 切分覆盖（`AIC 0 独占`仍存在，
  属"会静默丢结果"类）仍待办。
- paged `block_table` 间接层、prefill 档 per-row 循环、MTP `skip_topk`、fp8 indexer cache、
  **选择段并行化/性能**（本轮选择段单核化，D 档增加约 512 次 16KB DMA，如实记录在 §3.4）。
- MRoPE 多模态（三轴不等）支持。

### 5.3 给接续者的最小定位手段（已就位）

- `evidence/select_single_core.log`：矩阵（30 组）+ 跨核一致（6 组）+ 9 次确定性 + oracle 读数。
- `evidence/probe_aiv_barrier.log`：AIV 间交换 **6 变体**（单发/循环 × 手写 flag/官方 `SyncAll`）的违规核数
  与最少见到槽数；**适用范围前提**见 §3.4。
- `check_select.py`：逐块 oracle（精确）+ `--cross` 多核一致性；`check_ref.py`：T1/T3 + 离散两条判据。
- 主 kernel 诊断 dump：`DumpStats()`（`m19_qsa_indexer.asc`）= 前 32 个 AIV 各 32×int32 → `statGm[aiv*64]`，
  字段表见本文件 §2.2/`evidence/MECHANISMS_FOUND.md`，直接读 `*_stats.bin`。

### 5.4 AIC `GetBlockIdx` 探针的撤回（M156）

**结论**：`probe_aic_blockidx.asc` 原要回答的问题（`__mix__(1,2)` 下 AIC 的
`GetBlockIdx()`/`GetBlockNum()` 取值）**本探针无法回答，已从构建移除**（`CMakeLists.txt`
的 `add_executable` 与 `foreach` 名单均删去该目标；源文件保留为撤回说明）。归档
`evidence/probe_aic_blockidx.log` 的全 FAIL 是**摆位伪影**；**读数本身未改动**。

**已核实的部分（带 `文件:行`）**

- AIC 分支原 `probe_aic_blockidx.asc:46-54`，在 `:49`/`:52` 调 `PutBid`；`PutBid` 原 `:26`
  构造 `LocalTensor<int32_t> rec(TPosition::VECCALC, UB_REC, 64)`、`:27-30` 标量 `SetValue`、
  `:36-37` UB→GM `DataCopyPad` 落盘 ⇒ 该 AIC 分支**确经 UB 暂存**。
- 归档读数 `evidence/probe_aic_blockidx.log` 第 2 行 `launch: OK`、第 3-4 行六槽全
  `(bid=-1,num=-1)`（host 原 `:75-76` 预置 `0xFF`）⇒ 该档**确没有任何字节落盘**
  （无 271、无异常报告）。
- `evidence/dump_manifest.md:14` 原记「AIC 侧 UB→MTE3 落盘读回哨兵（全项目级负结果）」，
  与根因不符；已就地更正（该行读数保留，只改解读）。

**未隔离验证的推断（写成推断，不写成事实）**

- 「UB→GM 基础 API 在 AIC 上被 `if ASCEND_IS_AIV` 常量折叠成空 ⇒ `rec` 无活引用 ⇒ 标量
  SetValue 被 DCE」是 M154 C1 的**最一致解释**，**未**做编译产物（`-S`）或活消费者的隔离
  验证。本 mission 未补做（口径：不为探针加实验），见 `evidence/probe_aic_blockidx.log`。

**为什么撤回而不是改写成「非 UB 的等价形态」**

- 候选一 **裸标量直写 GM**（`__gm__ T*` store / `GlobalTensor::SetValue`，即 M154 finding
  建议的 `m10_attn_decode/m10_probe.asc:48-51` `Mark` 形态）：被 `docs/05-megakernel-design.md:172`
  的硬规则 ⓔ **一律禁用**（「标量直写 GM …… 一律禁用」），且本仓自有实测记录其不可靠 ——
  `m10_attn_decode/logs/m10_probe_20260926.log:2` 逐字「完成标记用裸标量写 GM，按 m0 结论
  不可靠（实测仅 8~14/56 落盘）」；`docs/06-m0-bringup.md:103`、`:116` 同口径。
- 候选二 **Fixpipe L0C→GM**：是 AIC 唯一合法落盘通路（`docs/05-megakernel-design.md:216`），
  但把 bid 值送进 L0C 需先构造 Mmad 乘积 ⇒ 属「为探针新造实验」，与人类口径
  「最好不要用探针」及本 mission「不扩展探针」冲突。
- ⇒ 两条候选都不满足「最小收口」，选移除。

**修正后本探针能/不能回答什么**

- **不能**：它不再产生读数；归档的全 FAIL 不能读成「AIC 的 bid 取错」或
  「mode0 屏障让部分 AIC 不落盘」。
- **未回答**：AIC 的 `GetBlockIdx()` 在 `mix(1,2)` 下的取值 —— 本 mission **未起任何设备档**
  重新取数（探针已移除，重跑它无意义）。该问题应按人类口径在**最终结果 / 产品路径**上判读，
  不由探针承担。

**可复算守卫**：`bash m19_qsa_indexer/check_probe_retraction.sh` —— 断言「目标不在构建 /
源文件不含 UB 暂存 / manifest 的旧措辞恰 1 次且只在该行的更正语境 / 本节标题（含空白边界）
与正文锚点存在」，任一不成立即非零退出；运行记录见 `evidence/probe_aic_blockidx_retraction.log`。

### 5.5 M195：同步卫生清理（冗余 `set_flag/wait_flag` 对内联，2026-10-05）

**来源**：M193 survey §5.1 指出本文件有 `set_flag/wait_flag` 对内联在**已被同处 BufferID 交接承载**的位置。
本 mission 逐对判定，按人类逐字规则（核内只用 BufferID、核外只用 CrossCore、除有值依赖外不挂 PIPE_S）清零。

**改前清点**：文本上共 **10 对** `SetFlag/WaitFlag`（另加 `VSync()` 的 3 个调用点；M193 survey 记「11 对」，其列出的位置为 10 处）。

| # | 事件对（改前） | 语义 | 同处 BufferID 交接 | 判定 | 处置 |
|---|---|---|---|---|---|
| 1 | `VSync()`（调用点 `CountTokens`/`ScanStage1`/`CountSub`）的 `V_S` | V 写完 UB → 标量 `GetValue` 读 | 无（scalar 侧无 acquire；只有生产侧 `BufRel<PIPE_V>` 排空） | **承载**（有值依赖） | **保留**；A/B 消融：删掉后默认 4 档变红（tokens 1266/1278/1024/0） |
| 2 | `QHeadNormRope` 的 `V_MTE3` | V 写 `UB_Q` → MTE3 搬 GM | `BufRel<PIPE_V>(AIV_BUF_RES)` → `BufAcq<PIPE_MTE3>(AIV_BUF_RES)`（同 id） | 冗余 | 删 |
| 3 | `KPath` 的 `V_MTE3` | V 写 `UB_KPK` → MTE3 搬 | 同上（`AIV_BUF_RES`） | 冗余 | 删 |
| 4 | `DumpLogits` 的 `V_MTE3` | `ScanChunk` 写 `UB_SCORE` → MTE3 搬 logits | `BufRel<PIPE_V>(AIV_BUF_LOG)` → `BufAcq<PIPE_MTE3>(AIV_BUF_LOG)` | 冗余 | 删 |
| 5 | `EmitSub` 的 `V_MTE3` | V 写 `UB_OUTS` → MTE3 搬 out | `BufRel<PIPE_V>(AIV_BUF_RES)` → `BufAcq<PIPE_MTE3>(AIV_BUF_RES)` | 冗余 | 删 |
| 6 | `CountTokens` 的 `MTE2_V` | MTE2 写 `UB_CNTR` → V 归约 | `BufRel<PIPE_MTE2>(AIV_BUF_CNT)` → `BufAcq<PIPE_V>(AIV_BUF_CNT)` | 冗余 | 删 |
| 7 | `ScanStage1` 的 `MTE2_V` | `LoadStageOnly` MTE2 写 `UB_CROW` → V 读 | `BufRel<PIPE_MTE2>(AIV_BUF_CROW)` → `BufAcq<PIPE_V>(AIV_BUF_CROW)` | 冗余 | 删 |
| 8 | `WriteToken` 的 `S_MTE3` | 标量写 `UB_TRACE+192` → MTE3 搬 cntGm | **无**（原只有事件） | 缺覆盖 | 改「`BufAcq/BufRel<PIPE_S>(AIV_BUF_CNT)` 阻塞释放 + 对侧 `BufAcq<MTE3>`」（`docs/05 §6.1 ⓔ`） |
| 9 | `DumpStats` 的 `S_MTE3` | 标量写 `UB_TRACE` 23 字段 → MTE3 搬 statGm | **无** | 缺覆盖 | 同上（`AIV_BUF_C2`） |
| 10 | AIC 入口的 `FIX_S` | GEMM Fixpipe → 标量 set mode2 flag | barrier set 与 mode2 set 同挂 `PIPE_FIX` | 冗余（AIC 不访问 UB，无「标量写 UB」形态） | 删 |

**判据（设备）**：改前/改后各跑一次默认 4 档 + `check_ref.py` + `check_select.py`：两遍均 `ALL PASS`，
计数 **2048/2050/1024/2048**；**除 `*_stats.bin` 的保留字段外所有 dump 逐字节相同**
（`evidence/sync_cleanup_m195/dump_compare.txt`＋生成/校验脚本 `evidence/sync_cleanup_m195/compare_dumps.py`（离线可核、失败非零退出）；`*_stats.bin` 仅字段 23–31 有别 —— 代码里该字段标注「保留」、
是 `UB_TRACE` 前 128B 内的未初始化残值，host 判定只读字段 7/8）。`run_select_matrix.sh` 改后重跑读数亦在
`evidence/sync_cleanup_m195/`。**性能**：未做计时；删掉的是冗余的**标量阻塞等待**（`V_MTE3`/`MTE2_V`/`FIX_S`），
预期 AIV 侧 stall 与指令数略降，量级应为小 —— 无实测数据，不估具体百分比。

**裁定落库（M193 (B) 两条）**：survey 另列 `m13_moe_layer.asc:1082-1083`、`m17_moe_layer.asc:1267-1268` 的
`SetFlag/WaitFlag<MTE3_S>`（router 的 MTE3 GM 写排空后再做标量 GM 读）——**有值依赖**，按 `docs/05` §2
PIPE_S 限制的例外**保留不改**；该裁定已写进 `docs/05-megakernel-design.md` §6.1 ⓔ 的新子条「有值依赖例外」。
本文件文件头的同步策略注释也一并改正（原「不挂 PIPE_S（唯一的 S 管事件是核尾 MTE3 排空…）」已改为
「跨 pipe 可见性走 BufferID；唯一保留的 S 管事件是 `V_S` 有值依赖」）。

### 3.7 M35 时点的历史读数（**存档・非当前结论**）

> **读这段之前先读这一句**：本段是 M35 的原始读数与推断，**其中「错在读回路径」的结论已被 M53 实测否证**
> （每核前缀与 GM 计数数组逐核一致；见 §3.4 与 `evidence/MECHANISMS_FOUND.md` 机理 (ceq)）。
> 保留下来的唯一目的是给接续者看当时的证据链。

- **地面真值（Step 1，成立）**：用 host D2H **直读同一片 GM** 得 `ceq = [1,1,0,…]`、合计 **2**（正确）；
  而 kernel 侧读回 + 归约上报 **`eqTotal = 56`（= NAIV，幻影"每核各 1"）** ⇒ **GM 内容对、读回路径错**。
- **假设否证（Step 2）**：按 `docs/06 §4/§5` 在 8 处 MTE2→V 读回点补 `SetFlag/WaitFlag<HardEvent::MTE2_V>`，
  **`eqTotal` 补前补后都是 56** ⇒ "缺 MTE2→V 事件"**不是**（唯一）原因。该 patch 已回退、未入库。
- **已闭合的因果链**：`eqTotal=56` ⇒ `myEqBase = [0,1,2,2]`（幻影前缀）⇒ tie 配额 `need = 1` 落到 **AIV 0**
  （而 AIV 0 `localTie = 0`，没有 tie 元素），**边界块 87（score == K）在 AIV 1**（`myEqBase=1` ⇒ quota 0）
  ⇒ 该块永不写出 ⇒ **恰好少 1 个 block**（与 A 档"只选 [26]、丢 87"吻合）。
- **未解释的一环**：为何同片 GM 的 `ceq` 被读回成 56。已排除：写侧值错、GM 内容错、缺 MTE2→V 事件；
  未排除：该路 reduce/掩码口径、读回槽被他人预写 1.0。**内核侧对拍探针自身不可信**
  （同一 `v` 的归约 4.0 与上报 `gtTotal=1` 矛盾）且引入运行回归 ⇒ 已回退；接续者须先重建探针取址。
