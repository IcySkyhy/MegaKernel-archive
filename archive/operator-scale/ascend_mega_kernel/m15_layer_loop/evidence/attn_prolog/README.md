# M88 · attention 前端 prolog 的 oracle 与 device 探针

> 本目录是 **M88** 的交付：M82 在 `m15_layer_loop/README.md` Part D · M82-6 第 1 项显式 descope 的
> 「attention 前端 prolog」（`qkv_proj` + q/k norm + partial RoPE + gate split；以及 indexer 路）
> 的第一段落盘。
>
> **本 mission 的结论分两半，必须分开读**：
> - **① host 侧 oracle：已完成并实跑通过**（真实 checkpoint 权重、逐条 `文件:符号` 依据、非空洞性读数）。
> - **② device 侧探针 + 判据 + 负向对照：代码与结构已完成、能编译、AIC 的 4 个 GEMM 与 H2D 参数面
>   已实测通过，但整条 walk 在 device 上**挂死**（`rc=124`），根因已缩小到 AIC/AIV 的 mode-2
>   cross-core 配对，**未修好** ⇒ 见下面「显式未完成项」。`runs=prolog` **故意不在 `runs=all` 里**，
>   `runs=all` 仍是 **2068 判定项 + 290 guard / 0 FAIL**（与 M82 基线逐项一致，零回归）。

---

## 0. 一句话：本 mission 最先要解决的「可测性」问题

M82 之所以 descope prolog，唯一理由是**「仓内没有已验证的 Ascend 实现，且没有参考」**。
人类口径是「**vllm 官方的 qwen4 exp 尽量和他对齐**」。所以本 mission 第一步**不是写码**，而是
把 vLLM 的对应实现变成一份 **host 侧 numpy/float64 参考 + 逐条 `文件:符号` 的依据表**。

**这一步立刻抓到两条会直接改结果的语义**（下面 §1 有完整依据链）：

| # | mission Context 里的写法 | 源码里的实际语义 | 后果 |
|---|---|---|---|
| **F1** | 「q/k **RMSNorm**(256, eps 1e-6)」 | **GemmaRMSNorm**：`y = x·rsqrt(mean(x²)+eps)·(1+w)` | 按朴素 RMSNorm（乘 `w`）实现，**layer 3 的 q 会整体差 4.5×**（实测 `q_norm.weight` 的 mean = 0.2833） |
| **F2** | （未写）「mrope_interleaved」的语义 | 文本-only 下 **MRoPE 完全退化为普通 NeoX partial RoPE(64)** | 若照 `docs/11-attn-analysis.md:49-52` 去实现「最少频次轮转置换」，**会引入一套参考里不存在的频率重排** |

F1/F2 与 PLE 那次「靠对账发现 `docs/14 §6.2` 有两处错」是同一形态：**规格文本与源码语义不一致，
且不一致的方向是"照抄规格会算错"**。

---

## 1. 规则依据表（R 表；N2 外部权威，docs/17 §1.2）

「仓外工件」= `/workspace/vllm`。**每条都在 `oracle_attn_prolog.py` 的文件头里复述一次**，改一处必须改两处。

| # | 规则 | `文件:符号` |
|---|---|---|
| R1 | `qkv_proj` 输出宽度 = `nH·(1+gate)·hd + 2·nKV·hd` = 12288+512+512 | `vllm/model_executor/models/qwen3_next.py:427-434` |
| R2 | split 顺序 `[q‖gate], k, v`；q 与 gate **按头交织**（头 h 的 **512** 列 = q[256] ‖ gate[256]） | `qwen3_next.py:428-434`；`models/qwen4_exp/nvidia/qsa.py:505-506` |
| R3 | q/k norm = **GemmaRMSNorm**（`(1+w)`） | `qwen4_exp/nvidia/qsa.py:350-351`；语义 `model_executor/layers/layernorm.py:140-168`（`weight = self.weight + 1.0`，初始化 `zeros` 在 `:156`）；别名 `qwen3_next.py:31`（`GemmaRMSNorm as Qwen3NextRMSNorm`） |
| R4 | Qwen4Exp 的 position id **无条件**是三个恒等轴（所以文本路径上 MRoPE 不产生差异；`config.json` 仍有 `vision_config`，本条不是「模型是文本模型」的断言） | `qwen4_exp/nvidia/model.py:846-852`（`positions.unsqueeze(0).expand(3,-1)`） |
| R5 | ⇒ 三轴置换在数值上恒等 ⇒ **退化为普通 NeoX partial RoPE** | `rotary_embedding/mrope.py:236-247`（`x[0]==x[1]==x[2]` 时返回 `x[0]`）；`mrope.py:374-423` |
| R6 | 类别 = `MRotaryEmbedding`（**不是** `MRotaryEmbeddingInterleaved`） | `rotary_embedding/__init__.py:110-121`；后者只在 `scaling_type=='openpangu'` 分支（`:333`） |
| R7 | `rotary_dim = int(256 × 0.25) = 64` | `rotary_embedding/__init__.py:68-71`；本 ckpt `rope_parameters.partial_rotary_factor = 0.25` |
| R8 | inv_freq = `base^(-2j/rotary_dim)`；表按 **query dtype(bf16)** 落地 | `rotary_embedding/base.py:80-99`、`:105-125`（`_match_cos_sin_cache_dtype` → `.to(dtype=query.dtype)`） |
| R9 | NeoX 配对 `(j, j+32)`；`o1=x1·c−x2·s`、`o2=x2·c+x1·s`；`[64,256)` 直通 | `rotary_embedding/common.py:134-173`（`is_neox_style=True`）；`mrope.py:414-434` |
| R10 | indexer `index_qk_proj` 宽度 = (4+1)×128 = **640** | `qwen4_exp/nvidia/indexer_qsa.py:131-137` |
| R11 | indexer q/k norm = `GemmaRMSNorm(128, eps=rms_norm_eps)` | `indexer_qsa.py:138-145`；fused 实证 `nvidia/ops/qsa_pre_indexer.py:69`（`weight = load(...) + 1.0`） |
| R12 | indexer 复用**同一张** cos/sin 表（stride = 64），只旋 128 维的**前 64 维** | `qsa_pre_indexer.py:180`（`cos_sin_stride = D//2`）、`:174-186`（调用点）、`:29-30`（`HALF=D//2, QUARTER=D//4`）、`:72-79`、`:430` |
| R13 | indexer raw k **不 norm 不 rope**，原样进 raw ring | `qsa_pre_indexer.py:367-372` |

### F1 的证据（可复算）

```
# 手动解析 safetensors 头（无 safetensors/torch 依赖，见本目录的工具脚本说明）
model.language_model.layers.3.self_attn.q_norm.weight      (256,) mean=0.2833 std=0.0610 min=-0.3457 max=0.5938 nonzero=256
model.language_model.layers.3.self_attn.k_norm.weight      (256,) mean=0.2734 std=0.1190 min=-0.7891 max=0.7773 nonzero=256
model.language_model.layers.3.self_attn.indexer.q_layernorm.weight (128,) mean=-0.0372 std=0.0651
model.language_model.layers.3.self_attn.indexer.k_layernorm.weight (128,) mean=-0.0410 std=0.0708
```

`(1+w)` 的等效 scale ≈ **1.283**，朴素 `w` 的等效 scale ≈ **0.283** ⇒ **4.5× 的系统性错**，
而且有一批 `q_norm` 为负（`1+w` 可以很小）。**这条直接决定 T3 的判据有没有意义**：
若按朴素 RMSNorm 实现，负向对照 `MODE_PLAIN_NORM` 与契约档的差别会大到任何判据都能抓住 ——
**但"能抓住错"不等于"实现对了"**，所以契约档必须按 Gemma 来做。

### F2 的证据链（三段）

1. `qsa.py:345-349` 用 `get_rope(head_size=256, rope_parameters=config.rope_parameters)`；
   本 ckpt 的 `text_config.rope_parameters = {rope_type:'default', rope_theta:1e7, partial_rotary_factor:0.25,
   mrope_interleaved:true, mrope_section:[11,11,10]}`。
2. `rotary_embedding/__init__.py:110-121`：`rope_type=='default'` + 有 `mrope_section` ⇒ 构造
   **`MRotaryEmbedding`**（`mrope_interleaved` 只作为它的一个属性）。`MRotaryEmbeddingInterleaved`
   （那个做频率置换的类）**只在 `openpangu` 分支被构造**（`:333`）。
3. `qwen4_exp/nvidia/model.py:846-852` **自己覆盖了** `get_mrope_input_positions`：
   `positions = torch.arange(len(input_tokens))` → `return positions.unsqueeze(0).expand(3, -1), 0`
   ⇒ 三轴相同 ⇒ `apply_interleaved_rope`（`mrope.py:236-247`）在 `x[0]==x[1]==x[2]` 时返回 `x[0]`
   ⇒ **正交于 plain RoPE**。

**旁证**：M82 的 D1（raw ring 行宽 140）的理由链仍然成立 —— 位置尾照存（`cache_rope_positions=True`），
只是它不改变数值。**F2 不构成「退回 `head_size=128`」的理由。**

⚠ **F2 的适用范围（硬要求，塔 2026-09-27 裁决 (3b)）**：这条退化**只在文本-only 下成立**。
本 checkpoint 的 `qwen4_exp/nvidia/model.py:846-852` 让三轴恒等；**一旦接入真·多模态 MRoPE**
（三轴 position id 不再相同），`apply_interleaved_rope` 就不再是恒等，主干 RoPE **必须重新实现**
（读 `mrope_interleaved` + `mrope_section` 做三轴选择，indexer 的 raw-key 环里那 3 个 int64 位置尾
也才真正被用到）。**任何地方不得把本节的结论写成「MRoPE 可以删掉」。**

### `docs/11-attn-analysis.md` 的口径差异（如实报）

- `:40` 把 q/k norm 写成「Qwen3NextRMSNorm」——**不算错但会误导**：vLLM 里
  `qwen3_next.py:31` 就是 `GemmaRMSNorm as Qwen3NextRMSNorm`。**照字面按朴素 RMSNorm 实现是错的。**
- `:45-52` 说 `mrope_interleaved=true` 要「按三模态计数生成最少频次轮转置换（`get_mrope_interleaved_id_list`）
  …… kernel 内可实现为查表换序」——**那是 `MRotaryEmbeddingInterleaved`（openpangu 分支）的语义，
  本模型不走那条路**（R6）。文本-only 下正确的结论是**不需要任何置换**。
  ⚠ 这条**没有**在本 mission 里改 `docs/**`（不在 scope）；已按纪律走 `TowerFinding`/报塔。

---

## 2. host 参考（oracle）：`oracle_attn_prolog.py`

纯 numpy（**无 torch / 无 safetensors 依赖**；权重按 `weights_manifest.txt` 的 `file/offset/rows/cols`
从真实 checkpoint 直接读，**与 device 侧同一来源**）。用 `/usr/local/python3.12.13/bin/python3.12`
（系统 python3 没有 numpy）。

```bash
cd m15_layer_loop/evidence/attn_prolog
/usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py selfcheck   # 常量自检，不依赖 checkpoint
/usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py gen         # 生成 device 侧的输入与期望
```

实跑读数（`../attn_prolog_oracle_selfcheck.log`，逐行可复现）：

```
[selfcheck] q+gate 宽度 = 2*nh*hd = 12288: PASS
[selfcheck] idx 宽度 = (4+1)*128 = 640: PASS
[selfcheck] ROT = int(256*0.25) = 64: PASS
[selfcheck] mrope_section 和 == ROT/2: PASS
[selfcheck] Y0_N == 13952 / OUT_N == 13952: PASS
[selfcheck] pos 0 ⇒ cos 全 1 / sin 全 0: PASS
[selfcheck] OK   (rc=0)

[gen] cos/sin 表 4097×64 bf16 = 524416 B，非零 262176/262208，distinct=3323
[gen] x0: y0 非零 13952/13952，out 非零 13952/13952，distinct=2104，|out|max=4.781
[gen] x1: y0 非零 13952/13952，out 非零 13952/13952，distinct=2085，|out|max=5.344
[gen] 输入敏感性：exp_out(x0) vs exp_out(x1) 不同元素 13938/13952
[gen] OK   (rc=0)
```

**⚠ 一条真的踩过的坑（记下来给接续者）**：第一版 `dump_bf16` 写的是 fp32 的**低 16 位**，
而 bf16 取整把有效位放在**高 16 位** ⇒ 落盘文件整片为 **0**，而内存里的数组是非零的。
于是「非零元素数」这条非空洞性判据**在内存上 PASS、在实际比对的字节上完全空过**（
输入敏感性读数 `0/13952`）。这正是塔广播的 **M85 `pack_bf16` 取低位**的同一种形态。
现在的版本：① `dump_bf16` 右移 16 位；② **所有非空洞性断言都从落盘文件读回**（`load_dump_bf16`），
不拿内存数组当比对面。

### `data/` 里有什么

⚠ **`.bin` 不入库**：`m15_layer_loop/.gitignore:2` 有 `*.bin`（本 mission **不改它** —— 不在 scope）。
入库的只有 `data/meta.json`、`data/meta.json.sha256`（各 `*.bin` 的 sha256）与生成脚本；
**跑 device 档 / 复算判据之前必须先跑一次 `oracle_attn_prolog.py gen`**（0.5 s，确定性）。
`data/meta.json.sha256` 的读数（冻结 tip 上实跑 `sha256sum -c`）：**13/13 OK**。
判据的比对面因此**仍然取自落盘文件**（`gen` 刚写出来的那些），不是内存数组。

| 文件 | 内容 |
|---|---|
| `x0.bin` / `x1.bin` | 两个确定性 bf16 输入（`[2560]`，host 生成 ⇒ N1 输入隔离） |
| `cos_sin.bin` | cos/sin 表 `[4097][64]` bf16，行 = `[cos(32) | sin(32)]`（**设备不算超越函数**） |
| `exp_y0_x{0,1}.bin` | 4 个 GEMM 的**期望输出**（bf16，13952 元素） |
| `exp_out_x{0,1}.bin` | prolog 的**期望输出**（bf16，13952 元素） |
| `exp_y0_exact_x{0,1}.bin` | GEMM 的 fp64 精确值（落 fp32；T2′ 判「翻转是否真的贴格点中点」要用） |
| `exp_terms_{y0,out}_x{0,1}.bin` | 逐元素的 `Σ|terms|`（fp32；T3/T2′ 的界要用） |
| `meta.json` | 层号 = 3、`pos` = 4096、各段偏移、形状常量 |

### 段布局（device 与 oracle 共用，`m15_attn_prolog.h` §2 是唯一权威）

```
y0（GEMM 原始输出，13952 bf16）:  [0,12288) q‖gate | [12288,12800) k | [12800,13312) v | [13312,13952) index_qk
out（prolog 输出，13952 bf16）:   [0,6144) q | [6144,12288) gate | [12288,12800) k
                                  | [12800,13312) v | [13312,13824) idx q | [13824,13952) raw k
```

---

## 3. device 侧：`m15_attn_prolog.h` / `m15_attn_prolog_probe.h` / `m15_attn_prolog_host.h`

### 3.1 结构（**不是**占位直通：AIC 与 AIV 两侧都有真活）

- **AIC**：4 个 bf16 GEMM（q_proj `[12288,2560]`、k/v `[512,2560]`、index_qk `[640,2560]`）。
  流水骨架（MTE2 Nd2Nz → MTE1 LoadData → Mmad → FIXP F322BF16）**改自** `m15_gdn_layer.h:1403-1580`
  的 `Cube::Bf16Gemm`；唯一实质改动是 `BASE_N` 160 → **128**。
  **128 不是凑的**：三个 N 全都 16 对齐，取 `gcd(12288, 512, 640) = 128` ⇒ 96/4/4/5 个 N-块，**无尾块**；
  且 `64×128×2 = 16 KB ≤ 32 KB`（L0B 半区）。
- **AIV**：q 的 24 个头各一个 AIV（GemmaRMSNorm(256) + RoPE(64)），k 两个头，indexer q 四个头
  （GemmaRMSNorm(128) + RoPE(64)），外加 gate / v / raw k 的**原样抄写**（R2/R13）。
- **接口**：受 scope 所限（`m15_hc_host.h` 不在本 mission），走的是 **`runs=prolog` 独立探针档**，
  与 M82 的 `runs=kv` 同构；**没有**动 `LayerArgs` / `M15L_LAYER_HC_ARGS_DECL`。

### 3.2 人类结构约束的落实

| 约束 | 落实 |
|---|---|
| 核内用 **buffer id** | AIC：`Mutex::Lock/Unlock<PIPE_*>(M15G::BUF_AIC_*)`（L1/L0/L0C ping-pong）；AIV：`BufAcquire/BufRelease<PIPE_MTE2/V/MTE3>(AP_BUF_IO / AP_BUF_IN / AP_BUF_OUT)`（三段式用**两个** id）。**全部静态编号**，不用 `TQue`/`TBuf`，不用任何 ascendc 资源管理函数 |
| 核间用 **set cross core** | AIC 的 mode-0 全体对齐（`AP_AIC_M0_OUT`）+ AIC→配对 AIV 的 mode-2（`AP_A2V_GEMM`） |
| **除有值依赖外不用 `PIPE_S`** | `PIPE_S` 只出现在**等 cross-core flag 的 wait 侧**（`CrossCoreWaitFlag<..., PIPE_S>`），与 GDN/hc/MoE 段一致。核内数据依赖一律 buffer id，**没有** set flag / wait flag 系列 |
| 地址自己管、尽量静态分配 | UB 段窗（`m15_attn_prolog.h` §7）、L1/L0 段窗（§6）、cos/sin 表地址、权重段偏移全部是 `constexpr` + `static_assert` |

**flagId 预算盘账（M76 提示的那条）**：探针是**独立启动**，与融合 kernel 的 flag 不同时存在，
所以本档用 mode-0 = {1}、mode-2 = {4,5}（避开 0 这个最容易被别处占的号）。
**但**——真正要接进 `M15L_FusedBody` 时，AIV 的 mode-0 号段已满（hc 12-14 / GDN 8-11 / MoE 12-15 /
三条 boundary 8,9，见 `m15_layer_resources.h:350-352`），**AIC 的 mode-0 也满了（GDN 12-15、hc 0-3、
MoE 8-11）**（`m15_layer_resources.h:365-441` 的 `FLAG_SEQ` 表是权威）。⇒ 接线时必须先改
`FLAG_SEQ` 的分节（`m15_layer_resources.h` 在 scope 内，但本 mission 没走到那一步）。

### 3.3 判据设计（`m15_attn_prolog_host.h`）

| 段 | 档 | 判据 |
|---|---|---|
| `y0` 的 4 段（GEMM） | **T2′** | ① 逐元素 `|got−ref| ≤ 1.0·ulp(ref)`（bf16 格点间距，`2^(e-7)`，docs/17 §1.1 口径）；② **每个翻转元素必须**满足 `0.5·ulp(ref) − |exact−ref| ≤ ε_acc·Σ|terms|`，`ε_acc = K·2^-24 = 1.526e-4` —— 即「翻转只允许发生在真正的格点边界上，且幅度在 fp32 累加误差之内」 |
| `out` 的 `q` / `k` / `qidx`（norm+rope） | **T3** | `|got−ref| ≤ ε·Σ|terms| + 1.0·ulp(ref)`，**ε = 14·2^-24 ≈ 8.34e-7** |
| `out` 的 `gate` / `v` / `raw k` | **T1 逐字节** | 这三段是**原样抄写**（R2/R13），必须逐字节相同 |

**ε 的逐项推导（docs/17 §1.1：不得拿「实测最大 X ulp」当界）** —— 单位 ulp = `2^-24`（fp32 的 unit roundoff）：

| 来源 | 项数 |
|---|---|
| norm 的两次乘（`x·rstd`、`·(1+w)`） | 2 |
| `rstd`：`Div` + `Sqrt` 各 ≤1 ulp | 2 |
| 旋转的两个乘积（`x1·cos`、`x2·sin`） | 2 |
| 旋转的一次加法 | 1 |
| **小计** | **7** |
| ×2 安全系数（覆盖 DIV/SQRT 的末位未逐位建模项） | **14** |

⇒ `ε = 14·2^-24`。`+1.0·ulp(out)` 单列，因为**参考本身已被量化到 bf16 格点**（docs/17 §1.1 的
「参考本身已量化」条款）。

**一个关键的工程选择**：设备侧**故意不用** NR 近似（`NormDonor::ComputeRstdNewtonRaphson`），
改用 `Div` + `Sqrt` 的精确式 —— 这样 ε 里**没有 rsqrt 近似项**。将来若换 NR，必须把它的精度档
重新推导进 ε（README 与本文件都写了这一条）。

**非空洞性（docs/17 §4，塔的硬纪律）**：
- 比对面**取自落盘文件**（`data/exp_*.bin`），不取内存数组（M85 的教训）。
- `Ap.nonvac.x0/x1`：期望的 `out` / `y0` 非零元素数 > 半数（guard，整片为 0 会让判据空过）。
- `Ap.nonvac.sens`：`out(x0)` 与 `out(x1)` **必须不同**（改输入必须改输出）。
- `Ap.nonvac.pos`：`pos=4096` 与 `pos=0` 的 `out` **必须不同**（证明旋转真的用了 `pos`）。

**负向对照（三档，方向级改错；`mode` 是 kernel 参数）**：

| mode | 改的是什么方向 | 打的是哪条规则 |
|---|---|---|
| `AP_MODE_PLAIN_NORM` = 1 | 乘 `w` 而不是 `(1+w)` | R3 |
| `AP_MODE_ROPE_SIGN` = 2 | `o1 = x1·c + x2·s`（符号反向） | R9 |
| `AP_MODE_NO_ROPE` = 3 | 完全不旋 | R9 |

每档都要求「**至少一条被声明有咬合力的判据 FAIL**」（host 侧打印被打断的判据条数）。

---

## 4. device 实跑状态（**r3 全面更新；r2 的叙述与读数已作废，见 §4.5**）

**一句话**：`runs=prolog` 现在是 **rc=0 / ALL PASS**，且**判据做过"把被测对象弄坏它必须变红"的对照**。

### 4.0 三轮的根因链（每一条都是"我自己的代码 bug"，不是设备语义谜题）

| 轮 | 现象 | 真根因 | 依据 |
|---|---|---|---|
| r1 | `runs=prolog` **挂死**（rc=124） | `m15_attn_prolog_probe.h` 把 **AIV 那条臂整块写在 `if ASCEND_IS_AIC {` 内部**（全文件只有 `ASCEND_IS_AIC`、没有 `ASCEND_IS_AIV`）⇒ AIV 核上整段被编译掉，而 AIC 会执行属于 AIV 的 mode-2 `CrossCoreWaitFlag` ⇒ **等一个只有 AIV 能置的 flag**。r1 当时把它误写成「mode-2 set 会阻塞 / flag 配对不符」 | r1 复审用 /tmp 副本把 AIV 块归位 ⇒ rc=124 → rc=1 |
| r2 | 归位后 rc=1，但 `Ap.out.*` 大面积 FAIL | **两处 UB 往返没跨 `__VEC_SCOPE__`**（见 §4.1 的 B4/B5） | r3 实测：修后三条路 **差异 0** |
| r2 | **`Ap.out.k` 假绿（PASS 但从未比过 k 段）** | `m15_attn_prolog_host.h` 裸写 `OUT_K` ⇒ 静默绑定 `M15G::OUT_K`（6144）⇒ 判据比的是 **gate 段** | r2 复审三重定位；r3 的审计脚本给出正/负对照（§4.4） |

### 4.1 逐个修掉的缺陷（全部是本 mission 自己的代码错）

| # | 缺陷 | 后果 / 证据 |
|---|---|---|
| **B0** | host 裸符号 `OUT_K`（`M15AP` 只有 `AP_OUT_K`=12288）静默取 `M15G::OUT_K`=6144 | `Ap.out.k` 比的是 gate 段 ⇒ **假绿**；改成 `AP_OUT_K` 后 k 从一开始就是 FAIL ⇒ **q/k/qidx 三条 norm+rope 路都错**（缺陷三条共有，与 r2 复审一致） |
| **B1** | `Block1()` 入参是**字节**（`m15_gdn_layer.h:91-97`，非 32 B 整数倍直接 `Trap()`），首版传的是「32 B 块数」 | AIV 每次搬运都 `Trap`（errcode 286）。**r4 已在冻结 tip 上用一处变异重新观测到（不再是悬空引用）**：把 `Block1(字节)` 改回 `Block1(块数)` ⇒ `dbg=6` = **25** 条、`dbg=7` = **30** 条，与"陷阱判据 + 逐核枚举"的**推导**、以及 r2 的**历史观测**三者一致。命令、变异点、读数、以及「25 而不是 28」的原因（`v` 那一档 `1024/32=32 ⇒ 32&31=0` 不咬）见 **`evidence/attn_prolog_r2_b1_trapcounts.log`**（§0 是 r4 的原始读数） |
| **B2** | 三段式 UB 交接 `MTE2→PIPE_V→MTE3` **必须两个 buffer id**（GDN 用 `BUF_AIV_ROW0`+`BUF_AIV_OUT`），首版用一个 id | V 的写与 MTE3 的读没被排序；`Ap.repeat` 报 6144/6144 不同 |
| **B3** | host 的 `outX0`/`outX1` 都在两次跑**之后**才捕获 ⇒ 恒等 | `Ap.nonvac.sens` 是**恒 FAIL 的假判据** |
| **B4** | **`rstd` 的 UB 往返（`StoreAlign<DIST_FIRST_ELEMENT_B32>` → `LoadAlign<DIST_BRC_B32>`）没跨 `__VEC_SCOPE__`** | 设备输出 = **该头正确值的「每 launch 一个常数倍」**（mode3 比值 −0.03665、mode0 比值 −0.13134，各在 8 个元素上稳定到 4 位有效数字，两次 launch 之间不同）⇒ 正是「rstd 这一个标量被读成垃圾」的指纹。**这是 q/k/qidx 三条路全错的直接原因。** donor 是两段（`ComputeRstdNewtonRaphson` 与 `CalculateGateY`） |
| **B5** | norm 写 UB → rope 读 UB 的两条 `LoadAlign`（`outUb+0` / `outUb+32`）也没跨 `__VEC_SCOPE__` | 同一类 RAW；一并拆开 |
| **B6** | `Ap.out.gate/v/kraw` 用「T1 逐字节 vs oracle」，但设备 y0 自己就可能与 oracle 差 1 ulp（`Ap.y0.qg` 自身报了 2 个**真格点边界**元素） | 把 GEMM 的边界差误判成抄写错 ⇒ **假红**。改为 **T2′ vs oracle**；「抄写恒等」那一层仍由 `Ap.copy.*`（T1 vs **设备自己的 y0**）承担 |
| **B7** | pos=0 档拿 pos=4096 的期望去比 out 段 | 必然 FAIL 的假红；改为该档只比与 pos 无关的 y0 段 |

### 4.2 冻结身份上的真实读数（**r4 为当前**：`evidence/attn_prolog_r4_runs.log`；r3 见 `attn_prolog_r3_runs.log`）

**冻结身份 = 二进制 sha256 `220a2cf24e21b085ca55271ff0f1c3fa29e3889d6ee44031f27a4bbfe423c025`**（**r4 的值**；
同一份读数也在 `evidence/attn_prolog_r4_runs.log` 的头里）。
⚠ **为什么 r4 的 sha 与 r3 不同**：r4 先把分支 `rebase` 到新 main（`40dd6de`，含 M95），
而 `m15_layer_loop.asc` 会 `#include` M95 改过的 `m15_moe_*` ⇒ **二进制随基座而变**；
本 mission 在 r4 的改动**只在 host / oracle 侧**（`m15_attn_prolog_host.h` 没再动、内核行为未变）。
⚠ 这条 sha 只在**当前基座**上有效；换基座要重取（`attn_prolog_r4_runs.log` 头里有取法）。
本 mission 之后只动 `evidence/`，**只动 evidence 的 commit 不改这个 sha**；
手交时的分支 tip 由报塔消息给出，不写在这里 —— 写进来它自己就把 tip 推走一格（「半刷新」）。
**跑设备前 `npu-smi info`：无在跑的进程**（HBM 5244/131072 MB）；本会话两次 OOM（塔口径：6 agent 并行），
所以所有 device 任务都是**串行单发**。

```
runs=all    rc=0   ALL PASS（checks=2068, guards=290, fails=0）     ← 与 M82 基线逐项一致，零回归
runs=prolog rc=0   连跑 3 次 = checks=173, guards=59, fails=0（三次一致）
                   ⚠ **确定性口径（r4 复审的限定，如实写）**：r4 复审自己另跑 **6 次**（1 基线 + 5 压力）
                   也全 0 fail、`Ap.repeat`=0/6144 ⇒ **合计 9 次全绿**。但 B4/B5 的实效是
                   **「概率性的竞争缓解」**：回退任一处得到的是**概率红**（复审：只回退 B4 ⇒ 3 次里
                   2 次三条全红；只回退 B5 ⇒ 3 次里 2 次 qidx 红），**不是"回退 ⇒ 必红"**。
                   所以 **9/9 全绿只构成经验确定性，不构成 p=0 的证明**——本文件**不声称**后者。
                   想要更强的主张，需要更多次（复审建议 15~20 次）device-空闲跑。
  Ap.y0.qg / .k / .v / .idx   T2'    PASS ×3（qg 有 2 个真格点边界元素差 1 ulp，其余逐字节一致）
  Ap.out.q / .k / .qidx       T3     PASS ×3，**差异 0**  ← r2 时三条都 FAIL
  Ap.copy.gate(24 头)/.v/.kraw T1字节 PASS ×3（vs **设备自己的 y0**）
  Ap.out.gate / .v / .kraw    T2'    PASS ×3
  Ap.nonvac.sens / .pos       PASS ×3
  负向对照 mode=1/2/3（改规则方向）：被打破 **7168 / 1692 / 1584** 条
  变异对照 mode=4/5（**破坏被测对象**）：被打破 **6145 / 27884** 条
```

**口径（承接 r2 复审 P2-1/P2-2，不许写成一个固定数）**：
- r2 的 `fails` **不是固定 14** —— 复核实测 **14 / 14 / 12**（q/qidx 的竞争影响 guard 成败）；r3 是 **0 / 0 / 0**。
- r2 的 `Ap.out.gate` **不是「3 次里 2 FAIL」** —— 该 tag 每次跑有 **4 条**判据（3 次契约 + 1 次 pos 档），
  3 次跑合计 **FAIL 9 / PASS 3 ⇒ 每次 3 FAIL + 1 PASS**；「2」是**单次跑内**的计数，两种说法混了。
- 判定项条数：r2 = 54 + 8 guard（fails=14）；r3 = 41 + 10 guard（fails=0）。少 13 条来自 **B7**（pos 档 `cmpOut=false` ⇒ 该档整轮不再计入，正好少一次契约跑 × 13 条）；
  **B6 不改条数**（它只把比较的**档**从 T1 换成 T2′，每段仍是一条）。，多 2 条 guard 来自两档变异对照。

### 4.3b r4 相对 r3 的两处变化（都是 P2 级、**不动内核行为**）

1. **P2-1 修好：`exp_normonly_*.bin` 的 qidx 段原来是"假信号"。** `indexer_prolog` **只在 `mode == 3`
   分支里**填 `normonly`，而 `gen` 调它用的是 `mode=0` ⇒ 该段**恒为全零** ⇒ 每次跑都打印的
   `qidx 不同 512/512` 是**拿设备的非零 qidx 去比全零**，是 **oracle 数据缺口、不是设备读数**。
   修后（`normonly` 无条件填）该行变成**真实设备读数**：
   `[诊断] device(mode=3 NO_ROPE) vs oracle(仅 norm)：q 不同 0/6144、k 不同 0/512、qidx 不同 0/512`
   ⇒ device 的 q/k/qidx 三段在"只做 norm、不做 rope"下与 oracle **逐元素相同**，
   这正是 B4/B5 定位时用的那把尺子，现在**三段都成立**（r4 之前只对 q/k 成立）。
   **自检也加固了**：`gen` 的 normonly 自检改成**读落盘文件**（不是内存数组）—— 因为 r4 我第一次修时
   踩了这一点（见 §4.3c）。
2. **P2-2 已兑**：B1 的 25/30 现在**可从 tip 复核**（见 §4.1 的 B1 行与 `attn_prolog_r2_b1_trapcounts.log`）。

### 4.3c 半刷新的一次自我介绍（r4，刻意的自我举证）

修 P2-1 时我先做了「把缺口改回去、自检必须 FAIL」的**正对照**：跑了一次 `gen`。那次 `gen`
**在 assert 之前就已经把全 0 的 `data/exp_normonly_*.bin` 落盘**；随后我只把代码恢复、
**没有重跑 `gen`** ⇒ **磁盘上是坏数据、代码是好的**。第一次 device 跑因此仍打印 `qidx 512/512`。
⇒ 现在 `gen` 的自检**读落盘文件**（不是内存数组），这类形态会被咬住（与 M85 的 `pack_bf16`
取低位是同一课：**内存非零、文件全 0**）。

### 4.4 反假绿纪律（r2 复审那条 P1 的直接产物，**这是本 mission 最值钱的一条**）

1. **「把你的被测对象弄坏，判据必须变红」**（塔的纪律）。r2 复审正是用这一招一枪打死 `Ap.out.k` 的假绿
   （把 q/k/i 三条 `AivNormRope::Run()` 全改成不执行 ⇒ q/qidx 变红而 k 仍 PASS）。
   本 mission 现在把这条做成**自动 guard**：kernel 新增两个变异档
   `AP_MODE_NO_COPY=4`（AIV 不写三个抄写段）、`AP_MODE_NO_GEMM=5`（AIC 不跑 4 个 GEMM），
   判据必须被打破（实测 **6145 / 27884** 条）。
   ⚠ **前提**：`H_ApPoisonOut()` 在**每次 launch 前**把 y0/out 两块读回平面毒成 `0xCD`。
   没有这一步，「本次什么都没写」会读到**上一次的正确值**而照常 PASS ——
   **变异档第一次跑就是 0 条被打破**，加毒化后才咬住。
   **r4 复审独立复现了这一点**（把 `H_ApPoisonOut` 关掉重 build ⇒ 两个变异档都变成 **0 条被打破**，
   而负向对照仍咬 7168/1692/1584）⇒ 「毒化承重」不是我一个人的说法。
2. **判据引用偏移常量时必须显式限定命名空间**（或至少过一遍审计）。本 TU 里有
   `using namespace M15G;`（来自 `m15_attn_kv.h` 里的 `using namespace M15G;`；**该头的行号会随它增长
   而漂**（M98 更正主 KV 几何后又 +4）⇒ 这里用**符号锚点**：`grep -n 'using namespace M15G;' m15_attn_kv.h`）。
   裸写一个 `M15AP` 里没有的名字会**静默绑到别处**。
   工具：`evidence/attn_prolog/audit_bare_names.py`。
   **正对照**（跑在 r2 被审 tip `3fd8486` 的 /tmp 副本上）：**3 命中，全是 `OUT_K`，rc=1**；
   **负对照**（跑在 r3 冻结 tip 上）：**0 命中，rc=0**。脚本在扫描面为空时**硬失败**（rc=2），
   不给自己留假绿。防呆第二层：`m15_attn_prolog.h` 里段的绝对下标逐个 `static_assert` +
   `static_assert(AP_OUT_K != M15G::OUT_K)`（就是这次假绿的那个形态）。
3. **判据的比对面必须是"本次 launch 产出的字节"**。毒化（第 1 条）同时把这一点钉住了：
   现在每一次 PASS 都证明设备**真的写了**那些字节。

### 4.5 r2 的叙述与读数（**已作废，保留作历史见证**）

r2 的 `evidence/attn_prolog_r2_runs.log` / `attn_prolog_r2_device_bisect.log` 里的 rc 读数本身可复现，
但**对它们的解释与结论已作废**：`Ap.out.k` 当时报 PASS 是**假绿**（比的是 gate 段），
据此写出的「主干 k 的 GemmaRMSNorm(256)+RoPE(64) device 数值正确」、以及 §5 #1 里「缺陷在 q 路特有的
输入面 / AIV 0..23 特有的时序」的**诊断与下一步都建立在假前提上**，r3 已全部重写。

## 5. 显式未完成项（r3 重写；**逐条列，不合并、不含糊**）

| # | 未完成项 | 现状 | 下一步 |
|---|---|---|---|
| **1** | ~~`AivNormRope` 三条路数值不对~~ | ✅ **r3 已修**（B4/B5：两处 UB 往返没跨 `__VEC_SCOPE__`）⇒ `Ap.out.q/k/qidx` 全部 PASS、**差异 0**，连跑 3 次确定。原诊断（「q 路特有的输入面 / AIV 0..23 特有的时序」）**建立在 r2 假绿的前提上，已作废**（§4.5） | — |
| **2** | **把 prolog 接进 `M15L_FusedBody`（层 3 单层）** | 未做。机械清单见 §5b 的 **W1-W9**；`m15_hc_host.h` 不在本 mission scope（塔裁决 A） | 另立 mission，并把 `m15_hc_host.h` 写进 scope。**接线时 flagId 必须先重分节（W1）** |
| 3 | `o_proj` + `×sigmoid(gate)` epilogue | 未做（本段只产出 pre-sigmoid 的 `gate` 向量） | 与 attention 核心一起做 |
| 4 | packed indices 的消费（打分/topk/expand） | 本 mission 边界之外 | — |
| 5 | raw ring / compressed 的填充数学 | 本 mission 只产出 raw k 与 indexer q 两个**向量**，没写 ring/compressed | 依赖 #2 |
| 6 | `m=4097` 的 per-row 化 | 探针是 `m=1`（prolog 的数学与 `m` 无关，但 AIC 的 N-块分派与 AIV 的按头分派在 `m>1` 时要重排） | — |
| 7 | **把 `runs=prolog` 并回 `runs=all`** | 现在**没并**（`runs=all` 仍 2068/290/0）。r3 起 prolog 档已经 rc=0，**并回的阻碍已消失**，剩下的只是"要不要让 48 层验收多花一次 prolog 的时间"的取舍 | 塔裁决 |
| 8 | MRoPE 真多模态路径 | 见 §1 的 F2 硬约束：文本-only 下的"退化"结论**不适用于**接入真多模态 | — |

---

## 5b. 接线需要什么：机械可执行清单（塔 2026-09-27 裁决 (1) 的附加要求）

> **本节由 M97（2026-09-27）从 `e36e1d1` 逐字取回。** M88 在 `6d6451d`（「README 重写 §4/§5」）
> 的那次重写里把这一节整段删掉了，而 §5 的表里仍写着「机械清单见 §5b 的 W1-W9」⇒ 悬空引用。
> 取回命令：`git show e36e1d1:m15_layer_loop/evidence/attn_prolog/README.md`。
> ⚠ 表里「（当前行号）」是 **`e36e1d1` 那棵树上的行号**，基座已变 —— **定位请看同一格里的
> `文件:符号`，不要按行号**（第六变体）。
> ⇒ **W1–W9 已由 M97 落地**（层 3 单层的接线 + device 见证）：见 `m15_layer_loop/evidence/attn_wire/README.md`，
> 那边还记了我在读这份清单时发现的、与现状不符的一处（W1 的「号占满」严格说是 **mode0 ∪ mode2 的并集**
> 占满，不是 mode0 子空间单独占满）以及据此得出的「复用而不是新造号」的分节。

本 mission **没有**动 `m15_layer_kernel.h`（零改动 ⇒ 融合 kernel 的行为一字未变，`runs=all`
的 2068 判定项即为此的读数）。下面是把 `runs=prolog` 的 prolog 接进**层 3 的单层**要改的每一处，
供后续 mission 照做：

| # | 文件:符号（当前行号） | 改什么 |
|---|---|---|
| W1 | `m15_layer_resources.h` 的 `FLAG_SEQ[]` 表（现 `:365-441`）与其 `FlagSeqAdjacentOk()` / `FlagMaxUse()` 校验（`:455-496`） | **先重分节**：AIV mode-0 已被 hc 12-14 / GDN 8-11 / MoE 12-15 / 三条 boundary 8,9 占满（`m15_layer_resources.h:350-352`），AIC mode-0 已被 GDN 12-15 / hc 0-3 / MoE 8-11 占满；attention 段的 mode-0 必须挤进**空闲号**或把某段的同步移到 mode 2。**这是接线前必须先做的事**（不是接线时顺手做） |
| W2 | `m15_layer_kernel.h:87-150` 的 `struct LayerArgs` | 加 attention 字段：`apW`（本 mission 的权重平面基址）、`apY0/apOut`（或直接复用 M82 的 `attnKvDev/attnCompDev/attnRingDev/attnPackDev`）、`apCs`（cos/sin 表）、`apX`、`apPos`、`layer`（层号，现接口**没有**） |
| W3 | `m15_layer_kernel.h:446-458` 的 `M15L_LAYER_HC_ARGS_DECL` / `:416-444` 的 `M15L_LAYER_HC_ARGS_FILL` | 同步加同名位置参数（**顺序必须与 `LayerArgs` 一致**） |
| W4 | `m15_hc_host.h:169` 的 `H_LaunchLayerHc()`（attention 分支 `:196-204`） | 同步补实参（**不在本 mission scope** ⇒ 这就是接线必须另立 mission 的原因） |
| W5 | `m15_layer_kernel.h:342-346`（AIV 分支的 `m15_attn_passthrough_body`） | 换成 `M15AP` 的 AIV 段（本 mission 的 `AivNormRope` + `AivCopy`，见 `m15_attn_prolog_probe.h`） |
| W6 | `m15_layer_kernel.h:356-374`（AIC 分支，**attention 当前完全缺席**） | 加 `else { attn.ProcessAic(); }`：4 个 `AttnGemm` + mode-0 对齐 + mode-2 通知 |
| W7 | `m15_layer_loop.asc:502-600` 的 `Ctx` + `:1363` 的 `H_Alloc` | 已由本 mission 加好 5 个缓冲（`apWDev/apXDev/apCsDev/apY0Dev/apOutDev`）—— 接线时**复用它们**，不要新建 |
| W8 | `m15_layer_loop.asc:2034` 附近的 `manifestPath` → `C.manifestPath` | 已由本 mission 加好（数据目录以 manifest 为锚点） |
| W9 | `m15_layer_resources.h:536-544` 的入口符号表 | 若另建 `m15_layer_kernel_attn_prefill` 之类入口，需同步该表 |

**接线前必须先解决**：本 mission 的 device 档 **rc=124 挂死**（`evidence/attn_prolog_device_bisect.log`），
根因在 AIC/AIV 的 mode-2 配对 ⇒ **W1 的分节要连同"配对取证"一起做**。

> ⚠ **上面这一段是 `e36e1d1` 当时的原话，它的时点必须讲清**：这个「rc=124 挂死」在**同一个 commit
> 之后的 r2 就已经解掉了**，而且真根因**不是** mode-2 配对 —— 是本 mission 自己的一个花括号
> （AIV 那条臂被整块写在 `if ASCEND_IS_AIC {` 内部 ⇒ AIV 核上整段被编译掉、AIC 却等一个只有 AIV
> 能置的 flag）。修后 `runs=prolog` 是 rc=0 / ALL PASS，**逐条见本文件 §4.0 与 §4.2**。
> 保留这段原话是为了让「M88 当时以为接线前要解决什么」可核；**不要**据此以为接线前还欠一道挂死修复。

### 5d. 诊断闸 `M15_AP_DBG` 的**代码口径**（P2-1 更正：r1 那张表与代码相反）

`M15_AP_DBG=<d>` 直接把 `d` 作为 kernel 第 8 个参数 `dbg` 传给**每一次** launch（含判据）。
代码里的编排是（`m15_attn_prolog_probe.h`）：

```
apAicGemm = (dbg == 0 || dbg == 2 || dbg == 6 || dbg == 7)   // AIC 的 4 个 GEMM
apAicFlags = (dbg != 3)                                       // mode-0 barrier + mode-2 set
apAivNone  = (dbg == 3 || dbg == 4)                           // AIV 连 wait 都不做
apAivWork  = (dbg == 0 || dbg == 1 || dbg == 6 || dbg == 7 || dbg == 8)
apGateCopy = (dbg != 8)                                       // AIV 0..23 的 gate 抄写
```

| dbg | AIC GEMM | AIC 两条 flag | AIV wait | AIV norm/rope | AIV 抄写 |
|---|---|---|---|---|---|
| 0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 2 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 3 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 4 | ❌ | ✅ | ❌ | ❌ | ❌ |
| 6 | ✅ | ✅ | ✅ | ❌ | ✅ |
| 7 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 8 | ✅ | ✅ | ✅ | ✅ | ❌（只关 AIV 0..23 的 gate 抄写） |

另有**独立的 `mode` 参数**（负向/变异对照；`H_ApRunOnce(..., mode, ...)` 传入，不经环境变量）：

| mode | 改什么 | 性质 |
|---|---|---|
| 1 PLAIN_NORM | norm 乘 `w` 而不是 `(1+w)` | **改规则方向**（负向对照） |
| 2 ROPE_SIGN | `o1 = x1·c + x2·s` | 改规则方向 |
| 3 NO_ROPE | 完全不旋 | 改规则方向 |
| **4 NO_COPY** | AIV **不写** gate/v/raw k 三个抄写段 | **破坏被测对象**（变异对照，§4.4） |
| **5 NO_GEMM** | AIC **不跑** 4 个 GEMM | 破坏被测对象 |

（r1 那张表把 `dbg=1/2/4` 的「配置」列写反了 —— 本轮按代码重写。）
⚠ **每个取值都保证 set/wait 配平**：`apAicFlags` 为假时 `apAivNone` 也必须为真，否则 AIV 会等一个
没人发的 flag、把「诊断档」退化成死锁（r1 正是在这里踩过）。

## 6. 文件清单（本 mission 新增/改动）

`git diff --name-status main...HEAD` = **16** 个文件（r1 写「11 个」，r1 复审数出 12，r2 加到 14；
r3 再加 `evidence/attn_prolog_r3_runs.log` 与 `evidence/attn_prolog/audit_bare_names.py` ⇒ **16**）。
`git diff --name-status main...HEAD | grep -vE '^m15_layer_loop/(m15_attn[^/]*\.h|m15_layer_loop\.asc|evidence/attn_.*)$'`
= **0 行**（越界检查）。

| 文件 | 动作 | 说明 |
|---|---|---|
| `m15_layer_loop/evidence/attn_prolog/oracle_attn_prolog.py` | 新增 | host 侧 numpy/float64 参考 + 输入/期望生成（R 表在文件头） |
| `m15_layer_loop/evidence/attn_prolog/data/meta.json` | 新增 | 层号 3、`pos`=4096、各段偏移、形状常量 |
| `m15_layer_loop/evidence/attn_prolog/data/meta.json.sha256` | 新增 | 13 个 `*.bin` 的 sha256（`sha256sum -c` 13 OK / 0 不符） |
| `m15_layer_loop/evidence/attn_prolog/README.md` | 新增 | 本文件 |
| `m15_layer_loop/evidence/attn_prolog_oracle_selfcheck.log` | 新增 | oracle `selfcheck` + `gen` + `sha256sum -c` 的实跑读数 |
| `m15_layer_loop/evidence/attn_prolog_r3_runs.log` | 新增 | **r3（当前）**：`runs=all` + `runs=prolog` ×3 + 负向/变异对照 + 审计正负对照的实跑读数 |
| `m15_layer_loop/evidence/attn_prolog/audit_bare_names.py` | 新增 | **反假绿审计**：本 mission 3 个头文件里裸用的全大写常量名若 `M15AP` 没有、别处有 ⇒ 报错（rc=1）；扫描面为空时硬失败（rc=2） |
| `m15_layer_loop/evidence/attn_prolog_r2_runs.log` | 保留 | **r2 读数（解释已作废，见 §4.5）**：rc 可复现，但 `Ap.out.k` 是假绿 |
| `m15_layer_loop/evidence/attn_prolog_r2_device_bisect.log` | 新增 | **r2**：8 档 `dbg` 诊断矩阵的实跑读数（冻结 tip） |
| `m15_layer_loop/evidence/attn_prolog_device_bisect.log` | 保留 | **r1 原始读数**（当时的事实）+ 顶部一段「r1 的解释已作废」的更正说明 |
| `m15_layer_loop/evidence/attn_prolog_run.log` | 保留 | r1 的 `runs=prolog` 行为（rc=124 挂死）——**历史见证，已过期** |
| `m15_layer_loop/evidence/attn_prolog_accept_run_all.log` | 保留 | r1 的 `runs=all`（2068/290/0） |
| `m15_layer_loop/m15_attn_prolog.h` | 新增 | 形状 / 段布局 / 权重平面 / cos-sin 表 / UB / flag / GEMM tiling 的登记表（`static_assert` 钉住） |
| `m15_layer_loop/m15_attn_prolog_probe.h` | 新增 | device 探针（AIC 4 个 GEMM + AIV norm/rope/抄写 + 诊断闸 `dbg`） |
| `m15_layer_loop/m15_attn_prolog_host.h` | 新增 | host 判据（T2′/T3/T1）、非空洞性、三档负向对照 |
| `m15_layer_loop/m15_layer_loop.asc` | 改 | `#include` 三个新头 + `Ctx` 的 5 个缓冲/11 张 host 表 + `H_Alloc` 5 行 + `runs=prolog` 分派（**不进 `all`**） |

`m15_layer_kernel.h` / `m15_layer_resources.h` **零改动**（符合塔裁决 A）。

## 7. 复跑命令（全部实跑过；`<wt>` = `.tower/worktrees/wt-88`）

```bash
# 构建
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B <wt>/m15_layer_loop/build -S <wt>/m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build <wt>/m15_layer_loop/build -j4

# 零回归（应打印 ALL PASS，checks=2068, guards=290, fails=0）
<wt>/m15_layer_loop/build/m15_layer_loop <wt>/m15_layer_loop/weights_manifest.txt all

# host 参考（应 rc=0，且 gen 的输入敏感性读数 ≠ 0）
cd <wt>/m15_layer_loop/evidence/attn_prolog
/usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py selfcheck
/usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py gen

# device 探针（r2：**不再挂死**，rc=1，出 PASS/FAIL 读数）
<wt>/m15_layer_loop/build/m15_layer_loop <wt>/m15_layer_loop/weights_manifest.txt prolog
# 诊断档（把 dbg 灌进每一次 launch，判据照跑；语义表见 §5d）
M15_AP_DBG=3 <wt>/m15_layer_loop/build/m15_layer_loop <wt>/m15_layer_loop/weights_manifest.txt prolog
```


---

## 8. 崩溃后残留的交代（塔要求逐条写清）

本 mission 期间容器**两次被 OOM 杀掉**（塔口径：当时 6 个 agent 并行；主机 cgroup 上限 32 GB）。
复起后我做了两件事：① 从断点继续（先 `git status --porcelain` 核残留）；② 把 device 任务改成**串行单发**、并**分块提交**。

### 8.1 `m15_layer_loop/m15_moe_resources.h`（**不在我的 scope**）——已恢复，未带入交付

**塔看到的**：该文件在 **index** 里有一处改动（`-53/+20`），而工作树对 HEAD 无差异。
**我的处置与结论**：
- 该 index 条目里的内容既不是 HEAD、也不是 main、也不是任何已提交状态（它是 `-53/+20` 的混合体，
  既"少"了 M91 的 `TOTAL_MAX` 那条说明、又把 `UB_RT_END` 写成 `193600`）⇒ 典型的
  **OOM 在 `git` 写 index 的中途把它打断**留下的**半成品索引项**，不是我有意做的改动。
- 处置：`git restore --staged --worktree m15_layer_loop/m15_moe_resources.h`
  （工作树本来就等于 HEAD，所以只清掉那个 stale index 条目，**没有丢任何内容**）。
- **复核**：`git diff --stat main -- m15_layer_loop/m15_moe_resources.h` 与
  `git diff --stat 4e28544 -- m15_layer_loop/m15_moe_resources.h` **都为空** ⇒ 该文件与 main、
  与 M91 的落点**逐字节一致**；`git diff --name-status main...HEAD` 里**没有**它。
- **我的立场**：我不需要改这个文件，也不认为需要改；如果后续有人看到 `193600` 这个值，它**不是**我留下的。

### 8.2 scope 与边界（三点式）

`git diff --name-status main...HEAD` 的 16 个文件**全部**在 `m15_attn*.h` / `m15_layer_loop.asc` /
`evidence/attn_*` / `evidence/attn_*/**` 内；越界过滤 = **0 行**。
`m15_layer_kernel.h` / `m15_layer_resources.h` **零改动**（符合塔裁决 A）。
`docs/**` / `probe_mask_lanes/**` / `ple/**` / `tools/weights/**` / `m15_moe_*.h` /
`lift_moe_segment.py` / `moe_relift/**` 均**未碰**。

### 8.3 `data/*.bin` 仍不入库

`m15_layer_loop/.gitignore:2` 有 `*.bin`（不在我的 scope，**我没有改它**）⇒ 跑 device 档前必须先跑
一次 `oracle_attn_prolog.py gen`。`data/meta.json.sha256` 是它们的可复现锚：实测 **15 OK / 0 不符**。
