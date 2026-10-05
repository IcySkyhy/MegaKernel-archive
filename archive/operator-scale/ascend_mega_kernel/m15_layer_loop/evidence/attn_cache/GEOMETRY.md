# M98 · attention cache 填数学：几何钉死（第一步交付）

> 本文是 M98 任务第 1 项（「哪些量进哪套 cache、什么几何」）的落盘。**每条都给 `文件:行号`**，
> 且**都是在本 worktree 里直接读代码得到的**（`m15_attn_kv.h` = 本仓唯一布局权威；
> 官方语义 = `/workspace/vllm/vllm/models/qwen4_exp/` 与 `/workspace/vllm/vllm/`）。
> 基线：分支 `feat/m98-attention-kv-cache-fill-math`，base `main @ 9603c65`（M88 已合入）。
>
> ⚠ 本文**不写**任何「已全部 / 无残留 / 0 命中」类绝对断言；未取证的一律标「未取证」。
>
> **引用格式（r2 起；本队第六变体：不钉会移动的 ref）**：`<文件>::<符号>（@<commit> :<行>）`。
> **符号名是内容锚点** —— 行号漂移时**以符号为准**；行号一律锚定在**不可变 commit** 上，本文件的
> 行号基准 = **`fc83496`**（M98 tip，即更正 `m15_attn_kv.h` 主 KV 段的那个 commit）。
> 复现某一行号：`git show fc83496:m15_layer_loop/m15_attn_kv.h | grep -n '<符号>'`。
> 起因：r1 的裸行号在主 KV 段 +31 行后整体失准（r1 复审 F2），故本文件不再用裸行号作主索引。

## 0. 三套（+1）cache 一览

| 段 | 本仓布局权威 | 谁**写** | 谁**读** | 行/页几何 | 32 B 对齐 |
|---|---|---|---|---|---|
| 主 KV（paged） | `m15_attn_kv.h::KV_BLOCK_BYTES` 等（@fc83496 :88-132） | attention 相位的前端（`k`/`v` 落页） | 稀疏 attention 的 gather（`qsa_sparse_paged_attention`） | 官方：页 32,768 B；token 步长 1,024 B；head 步长 16,384 B | ✓ |
| raw key ring | `m15_attn_kv.h::RING_HEAD_SIZE` 等（@fc83496 :134-153） | 同上（raw k ‖ 3×int64 位置尾） | **压缩器池化**（跨 chunk 成员） | 行 280 B；容量 4 行 = 1,120 B/请求/层 | ✗ **280 % 32 = 24** |
| compressed key | `m15_attn_kv.h::COMP_ROW_BYTES` 等（@fc83496 :155-165） | 压缩器池化 + norm + RoPE | indexer 打分（paged MQA） | 行 256 B；页 4 行 = 1,024 B | ✓ |
| packed indices | `m15_attn_kv.h::PACK_COLS` 等（@fc83496 :166-184） | expand（选择语义，不在本 mission） | 稀疏 attention 的循环上界（末列） | 行 8,208 B（2,052 列 int32）；行距 | ✗ **8,208 % 32 = 16** |

## 1. raw key ring

### 1.1 行宽 140 / 280 B（**从代码里读出来的**，不是凭记忆）
- 本仓：`m15_attn_kv.h::RING_HEAD_SIZE`（@fc83496 :135-139） → `RING_KEY_DIM=128`、`RING_TAIL_INT64=3`、`RING_TAIL_BYTES=24`、
  `RING_TAIL_BF16=12`、`RING_HEAD_SIZE=140`、`RING_ROW_BYTES=280`。
- 官方推导链：`common/qsa_cache.py:814-826`
  `rope_position_offset = ceil(key_head_size/4)*4 = 128`；
  `storage_head_size = 128 + 3*4 = 140`（`_BF16_PER_INT64=4`、`_NUM_ROPE_AXES=3`，同文件 `:811-812`）。
- 位置尾**在同一行内**的证明（不是另开一块）：`common/qsa_cache.py:828-836`
  `position_tail = qsa_cache[..., self.rope_position_offset:]`、`self.rope_position_cache = position_tail.view(torch.int64)`
  ⇒ 尾紧跟在 128 个 bf16 之后、且按 int64 解释 ⇒ 24 B。
- 开关：`nvidia/indexer_qsa.py:167` `cache_rope_positions=vllm_config.model_config.uses_mrope`；
  本 checkpoint `config.json` 的 `text_config.rope_parameters.mrope_section = [11,11,10]`
  ⇒ `uses_mrope = True`（`vllm/transformers_utils/config.py:676-713` 扫 `rope_parameters.{mrope_section,xdrope_section}`）。
  ⚠ M82 已声明：这条是**源码推导**，**没有运行期见证**（`m15_attn_kv.h::D1`，@fc83496 :36-46 —— 该段在主 KV 段**之前**，未随 r2 位移）。

### 1.2 容量 4 行 = 1,120 B/请求/层
- 本仓：`m15_attn_kv.h::RING_ROWS_PER_BLOCK`（@fc83496 :140-144）（`RING_ROWS_PER_BLOCK=4`、`RING_NUM_SPEC=0`、`RING_LAYER_BYTES=1120`，
  规则写成 `4*ceil((4+num_spec)/4)` 并有 `static_assert`）。
- 官方：`common/qsa_cache.py:838-856`（`get_kv_cache_spec`）
  `span = compress_ratio + num_speculative_tokens; capacity = compress_ratio * cdiv(span, compress_ratio)`，
  再 `CircularBufferSpec(block_size=capacity, num_kv_heads=1, head_size=140, head_size_v=0)`。
  ⇒ capacity = `4*ceil(4/4)` = 4；页字节 = `4 × 140 × 2` = **1,120**。
- 环是**每请求一个固定物理块**：`common/qsa_cache.py:129` `physical_blocks = block_table[safe_requests, 0]`
  （恒取 block_table 的第 **0** 列，与逻辑位置无关）。

### 1.3 槽位寻址
- 本仓：`m15_attn_kv.h::M15KV_RING_ROW_OFF`（@fc83496 :252-256） `= physBlk*1120 + (pos%4)*280`。
- 官方：`common/qsa_cache.py:131-133` `slots = physical_blocks * compressor_state_size + positions.remainder(compressor_state_size)`；
  同式在 metadata kernel `common/qsa_cache.py:292-294`。
- 单请求下 `physBlk = 0` ⇒ 槽 = `pos % 4`（与 M82 `m15_attn_kv.h::RingSlot` 的注释一致，@fc83496 :353-359）。

### 1.4 **写入门控（容易漏的一条）**：环只写「query chunk 末尾 capacity 行」
- 官方两处一致：
  - `common/qsa_cache.py:143-149`：`rows + compressor_state_size >= request_ends`，不满足的置 `PAD_SLOT_ID`；
  - metadata kernel `common/qsa_cache.py:280-285`：`valid = mapped & ... & (token_idx + circular_buffer_size >= query_end)`。
- 语义解释：环只有 4 行，**中间位置的 raw k 不必进环**（它们只在"凑齐一组"时需要，而同 chunk 的成员
  直接来自本 step 的 raw k；`ops/qsa.py:401-427` 的 `use_raw` 分支就是这条）。
- 本段把它实现成 `q + RING_ROWS_PER_BLOCK >= chunkEnd`（`m15_attn_cache.h` 的写环段）。
  ⚠ **可观测性口径（r1 复审 F3 更正）**：环只有 4 槽、`capacity = 4`，而门控放行的恰好是"最后 4 行"
  ⇒ 残留类 0..3 各被覆盖一次 ⇒ **平面字节上没有"留下毒值的不该写的槽"**（档 M 与 off-by-one 档
  都把 4 槽写满）。更一般地：一个完全不做门控的实现在环平面上**产出逐字节相同的平面**。
  ⇒ 这条门控在本段唯一的观测面是 `Ac.gate.ring`（设备自报的 `ringWritten` 与 host 按同规则独立复算
  的值相等）；它是合法判据，但**不是平面判据**。详见 `README.md` §4.5 与 §5 未完成项。

### 1.5 读面：压缩器池化的成员来源
- 官方 `nvidia/ops/qsa.py:399-427`：成员位置 `position = end_position - 3 + group_offset`；
  `use_raw = position >= chunk_start_position`；为假时从**环**读
  （`compressor_state_cache_ptr + block*stride_block + (position % COMPRESSOR_STATE_SIZE)*stride_token`，`:415-426`）。
- 顺带：位置尾的读面同构（`ops/qsa.py:437-467`，`LOAD_ROPE_POSITIONS` 分支读 `rope_cache` 的同一个槽）。

### 1.6 对齐：单行 **必须** `DataCopyPad`
- 本仓：`m15_attn_kv.h::static_assert(RING_ROW_BYTES % 32u != 0u…)`（@fc83496 :145-151）；整环 1,120 B = 35×32 ✓。
- 单行落点 `(pos%4)*280` 对 `pos%4 ∈ {1,2,3}` **不是 32 B 倍数**（`m15_attn_kv.h::static_assert(KV_LAYER_STRIDE % 32u == 0u…)`，@fc83496 :221-222 的层 stride 对齐判据）。
- 不用 Pad 的后果：`M15G::Block1()`（`m15_gdn_layer.h:91-97`）与 `M15H::Block1()`（`m15_hc_layer.h:87-94`）
  对非 32 B 倍数**直接 `Trap()`**。本 mission 实跑见证见 `../attn_cache_*.log` 的 `Ac.dcpad.required`。

## 2. compressed key cache

- 行：128 bf16 = **256 B**（`m15_attn_kv.h::COMP_ROW_BYTES`，@fc83496 :155-165）。
- 页：4 行 = **1,024 B**（`COMP_ROWS_PER_BLOCK = KV_BLOCK_TOKENS/COMP_TOKENS_PER_STATE = 16/4`，同处）。
  官方：`MLAAttentionSpec(block_size=cache_config.block_size, num_kv_heads=1, head_size=128, tokens_per_state=4)`
  （`common/qsa_cache.py:862-870`）；页字节 = `num_heads × (block_size/tokens_per_state) × head_size × 2`
  = `1 × 4 × 128 × 2 = 1,024`（`vllm/v1/kv_cache_interface.py:506-528`，
  `num_states = get_num_kernel_states(block_size)` 见 `:205-208`）。
- 槽：`slot = physical_block * storage_block_size + (compressed_position % storage_block_size)`
  （`common/qsa_cache.py:102-105` 的 `_logical_to_physical_qsa_slots`，由 `:159-187` 的
  `compressed_qsa_slot_mapping` 调用；同式在 metadata kernel `:296-317`）。
- **门控（off-by-one）**：`valid = (logical_position >= 0) & ((logical_position + 1) % compress_ratio == 0)`
  （`common/qsa_cache.py:179-182`；metadata kernel `:301`）⇒ 本仓 `M15KV_COMP_ROW_WRITTEN(pos)`
  （`m15_attn_kv.h::M15KV_COMP_ROW_WRITTEN`，@fc83496 :260）。
- 写面计数：4097 上下文下**实际写入** 1,024 行（`{3,7,…,4095}`），容量 1,028 行
  （`m15_attn_kv.h::PREFILL_COMP_ROWS_WRITTEN`，@fc83496 :419-423）。

### 2.1 填数学（本 mission 主体）
- 池化：`pooled[c] = (Σ_{i=0..3} raw_k[成员 i][c]) / 4`
  - 官方 kernel `nvidia/ops/qsa.py:397-433`：`accumulator` 是 **fp32**，`tl.store(..., accumulator / COMPRESS_RATIO, ...)`；
  - **结果落 bf16**：`pooled = torch.empty((rows,1,head_dim), dtype=raw_keys.dtype, ...)`
    （`ops/qsa.py:1001-1005`，`raw_keys` 为 bf16 ⇒ **池化后有一次 bf16 舍入**，然后才 norm）。
- 顺序：**池化 → GemmaRMSNorm(k_layernorm) → RoPE@组首**：
  `nvidia/indexer_qsa.py:342-353`（pool）→ `:354-358`（gemma_rmsnorm）→ `:359-367`（apply_qsa_rope）。
- 组首位置：`first_position = end_position - COMPRESS_RATIO + 1`（`ops/qsa.py:436`）⇒ 本仓
  `CompGroupFirstPos(g) = 4g`（`m15_attn_kv.h::CompGroupFirstPos`，@fc83496 :362）。
- **顺序约束**：compress（读环）在 store（写环）**之前**（`indexer_qsa.py:342-353` vs `:373-383`）。
  本段据此把"读环"放在写环之前（否则末尾几行的槽会绕回并盖掉还要用的成员）。

## 3. 主 KV —— **M98 已按塔裁 A 更正为官方几何**（旧值作为负向对照保留）

- **更正后**（`m15_attn_kv.h` 主 KV 段）：页 **32,768 B**、页内 token 步长 **1,024 B**（= K512 + V512）、
  head 步长 **16,384 B**、一个 (slot, head) 槽内 **K‖V**（V = K + 512 B）、轴序官方
  `[blocks, H=2, N=16, C=512]`；容量 `KV_LAYER_STRIDE = 257 × 32,768 = 8,421,376 B/层`、
  `KV_PLANE_BYTES = 101,056,512 B（96.38 MiB）`。
- **旧值**（M82 原文，现只作负向对照：`KV_OLD_*` + `KvOldGeomByteOffset()` + 探针的
  `MODE_BROKEN_OLDGEOM`）：页 16,384 B、head 平面 8,192 B、页内 token 步长 512 B、
  轴序 head 外层、**没有 V 的去处**（旧 `KV_TOKEN_ELEMS = 2×256 = 512`）。
- 官方（`/workspace/vllm`）：
  - spec：`nvidia/qsa.py:434-442` `FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=256,
    **head_size_v=256**, dtype=bf16)`；
  - 写入：`vllm/v1/attention/backends/flash_attn.py:1525-1540`（`do_kv_cache_update` 的真身，
    被 `nvidia/qsa.py:480` 调用）：注释写明 `(B,H,N,2*D) -> ((B,N,H,D),(B,N,H,D))`，
    `key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` + `reshape_and_cache_flash`；
    读侧同式见 `nvidia/qsa.py:215`；
  - 页字节：`vllm/v1/kv_cache_interface.py:506-528` ⇒ `2 × (16/1) × ((256+256)×2)` = **32,768 B**；
  - 独立交叉校验：经典 vLLM 布局 `2(K,V) × 16 token × 2 kv_head × 256 dim × 2 B` = **32,768 B**。
- 分歧记实（**r1 时点**，留作审计轨迹）：当时本仓登记的是页 16,384 B / token 步长 512 B / head 平面
  8,192 B / 轴序 head 外层 / **没有 V 的去处**；与上面四路官方依据逐项不符（页 16,384 vs 32,768、
  token 步长 512 vs 1,024、head 步长 8,192 vs 16,384、无 V 通道、轴序相反）。
- **处置（r2 已收口）**：报告塔后由**塔裁 A** 授权在 mission 内更正 —— `m15_attn_kv.h` 的主 KV 段
  已改为官方几何（见本节的「更正后」一栏），容量同步为 `KV_LAYER_STRIDE = 8,421,376 B/层`、
  `KV_PLANE_BYTES = 101,056,512 B（96.38 MiB）`；**本段的实现直接引用 `M15KV_KV_*`**
  （首版那套独立命名的 `AC_OFFICIAL_KV_*` 已删除，见 `m15_attn_cache.h::KV_BLOCK_BYTES` 的注释）。
  旧几何**不删**，降级为负向对照：`m15_attn_kv.h::KV_OLD_*` + `KvOldGeomByteOffset()`（host）与
  `m15_attn_kv_probe.h::MODE_BROKEN_OLDGEOM`（device），判据 `Kv.neg.oldgeom` / `Kv.neg.oldgeom.dev`
  两条都 PASS（读数见 `m98_runs_kv.log`）。

## 4. packed indices（本段只覆盖"行的落盘/传输"，不覆盖选择语义）

- 本仓：`m15_attn_kv.h::PACK_COLS`（@fc83496 :166-184）（2,052 列 int32 = 8,208 B/行；末列 2,051 = 有效数；`PACK_TAIL_COL=2048`、
  `PACK_TAIL_MAX=2`）。
- 官方：`nvidia/indexer_qsa.py:187-195`（`output_width = token_topk + compress_ratio - 1`、
  `packed_output_width = output_width + 1`，末列是**有效计数**、"never a token index"）；
  缓冲区形状 `[max_num_batched_tokens, packed_output_width]` int32（`nvidia/qsa.py:416-424`）；
  展开语义在 `ops/qsa_indexer.py` 的 `expand_qsa_block_indices`（**不在本 mission**）。
- 对齐：8,208 % 32 = **16** ⇒ 单行必须 `DataCopyPad`（`m15_attn_kv.h::static_assert(PACK_ROW_BYTES % 32u != 0u…)`，@fc83496 :180-182）。

## 5. 输入溯源三分（docs/17 §1.3）

| 类别 | 本 mission 的实际情况 |
|---|---|
| **① 声明输入** | `acInDev`（本 chunk 每行的 raw k 环行形态 280 B / `attn_idx_k_norm` 128 bf16 / 主 KV 的 k,v 行）、`acCsDev`（cos/sin 表）、`acPackSeedDev`（packed 行内容）、`acRingDev` 的初值（= 上一 chunk 的写入）。**无外部文件**：每一段的字节由本段的 `AC_SALT_*` 常量 + 本 TU 的 `H_RandU`/`AcRawKRow`/`AcKvRow`/`AcBuildCs` 唯一决定（`m15_attn_cache_host.h::AC_SALT_RAWK` 等；raw k 行 = `AcRawKRow(pos, salt)`，位置尾 = 该位置的 3 个 int64）。⇒ 复现口径是**代码锚点**（`AC_SALT_*` + 生成函数），不是落盘文件；如需落盘，把 `acInDev` 的内容 dump 出来即可（本段未做） |
| **② 上游输出** | **本段不吃任何设备上游产物**。真实流里 raw k 来自 M88 的 prolog（`m15_attn_prolog.h` 的 `OUT_KRAW`），但本段把它当①由 host 生成，因而与那次启动的运行时产物解耦 |
| **③ 被判量自身的产物** | **无**。所有期望值由 host 侧独立实现（double）从①算出 |

## 6. 未取证 / 未决（逐条）

1. ~~主 KV 的页几何~~ —— **已收口（塔裁 A，r2）**：权威头已更正、实现直接引用 `M15KV_KV_*`、
   旧几何只作负向对照（§3 末；读数见 `README.md` §2 的 `Kv.neg.oldgeom*` 两行）。**仍开着的一条不在本 mission scope**：
   `m15_layer_loop/m15_layer_loop.asc:808` 的注释（旧值「= 50,528,256 B（257 页/层 × 16,384 B）」）归 **M175**。
   `m15_layer_loop/README.md` 的同族旧数字已由 **M178 合入**（`4e626bb`）同步为权威值
   （页 32,768 B / 层 stride 8,421,376 B）；各处的**打印值**早随宏自动更正
   （日志里是 `101.06 MB … 257 页/层 × 32768 B`），现在只有 `.asc` 的同行注释是旧的。
2. **raw ring 的 140 行宽**：源码推导链完整（§1.1），但**没有运行期见证**（M82 已登记，本 mission 未补）。
3. **压缩行的 paged 非恒等 block_table**：本段只用恒等表（`phys = g / COMP_ROWS_PER_BLOCK`）；
   非恒等表的契约由 M82 的 `T-KV-PAGED` 覆盖，本段不重复。
4. **真实规模档**（真实权重切片 / 4097 全上下文）：**未做**（见 `README.md` 的未完成项）。
5. **`pooled` 落 bf16 之后、norm 之前是否还有其它舍入**：官方是 `gemma_rmsnorm(bf16 输入)`（
   `indexer_qsa.py:354-358`），本段按"bf16 入、fp32 累加、bf16 出"建模；未见第三处舍入。
6. **环写入门控只在计数上见证**（r1 复审 F3）：4 槽环下平面字节不可判别（§1.4 的口径）。
   若要给平面判据也咬住它，需要一条**非恒等/带副作用的形态**（例如让"早于门控的行"带可区分标记并
   断言它没被搬）—— 现在做不到，因为槽会被后续 4 行覆盖；如实登记，未做。
7. **`Ac.rep.sumnomean` 的可选升格**（r1 复审 F5）：该报告项现在只判「压缩行判据」，而它对压缩行
   确实不可判别（RMSNorm 对常数尺度不变）；本段的 **`Ac.pool.row` 是逐字节判据**，池化不除 4 会让
   设备写出 `bf16(4·mean)` ≠ 控制档的 `bf16(mean)` ⇒ **这条方向级错误在池化判据上是可判别的**。
   本轮**未**把该档同时把 `Ac.pool.row` 判一次（属"可选升格"，不是必须修）—— 登记为可做未做。
8. **形态偏差（r1 复审 F6，待塔拍板）**：mission 第二步写的形态是「核内 buffer id、**核间 set cross
   core**」，而本段是**单核 AIV 串行、无任何跨核同步**（理由：本档判寻址/几何/填数学，不判吞吐；
   单核下不存在跨核值依赖 ⇒ 也就没有 PIPE_S）。`VEC_SCOPE`/寄存器 API、核内 buffer id、地址自管、
   编译期静态 UB 窗都做到了，缺的只是"核间分工"这一项形态。是否豁免由塔裁。
