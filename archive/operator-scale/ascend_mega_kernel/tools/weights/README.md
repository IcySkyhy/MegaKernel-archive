# tools/weights — checkpoint 权重切片工具

从 Qwen3.8-Flash-Next-MXFP4（`qwen4_exp`）真实 checkpoint 中抽取小规模
MoE 权重集，输出为 **tools/golden 兼容**的 packed bin + json（文件名、布局、
manifest 约定与 `tools/golden/README.md` 完全一致），可直接喂给
`moe_block_ref.reference_moe_block` 等 golden 参考代码。

纯 numpy（2.5.1），无 torch / safetensors 依赖；Python 用
`/usr/local/python3.12.13/bin/python3.12`。

## 文件清单

| 文件 | 说明 |
|---|---|
| `safetensors_reader.py` | 纯 numpy memmap 读 safetensors 分片：8 字节头长 + JSON 头解析；扫描目录全部分片建 tensor→(文件, 偏移) 索引，**无需 index json**；bf16/fp16 视图；切片读取 |
| `extract_moe_slice.py` | 从 checkpoint 抽取一层 MoE 权重切片（默认 layer 0、8 个专家），输出 golden 兼容 bin + manifest |
| `selfcheck.py` | 自检：bin/json 完整性、与源 checkpoint 逐字节一致、MXFP4 往返定点、权重统计 vs M4 sanity 基准、体积预算、golden 参考端到端可运行 |
| `moe_real_scale_audit.py`（M93） | **真规模取证**：只读 safetensors 头，逐张量给出「单专家字节 / 全 512 专家段 / 头部全量」，再算单层与 48 层的 HBM 常驻预算；带负向对照（`--negative-experts`）。输出 `evidence/moe_real_scale.txt` |
| `evidence/moe_real_scale.txt`（M93） | 上面的实跑读数（20 checks / 0 FAIL） |
| `data/layer0_e0-8/` | 已抽取的切片数据集（12 个 bin + manifest，23.5 MB） |

## safetensors_reader.py

```python
from safetensors_reader import ShardReader, bf16_bits_to_f32

rdr = ShardReader("/workspace/Qwen3.8-Flash-Next-MXFP4")   # 扫描 *.safetensors
rdr.names()                        # 全部 tensor 名（头里有就有）
rdr.info("model.language_model.layers.0.mlp.experts.gate_up_proj")
w = rdr.load("...gate_up_proj", slice(0, 8))   # [8,1280,1280] uint8 零拷贝视图
x = rdr.load_f32("...mlp.gate.weight")          # bf16 → float32（RNE 无损上采样）
```

要点：

- **无需 index json**：直接解析每个分片的 JSON 头建索引；`model.safetensors.index.json`
  缺失或滞后（下载中）都不影响。
- **支持部分下载的目录**：仍在增长的分片会被截断到已落盘范围，
  `rdr.incomplete` 列出超出当前文件尾的张量；完整张量照常可读，
  越界读取抛 `ValueError`。
- **bf16 视图为 uint16 位型**（numpy 无原生 bf16），`load_f32` /
  `bf16_bits_to_f32` 做精确上采样；fp16 为原生 `np.float16`。
- **切片读取零拷贝**：`np.memmap`/buffer 视图，例如从 [512, ...] 的
  MoE 权重取 8 个专家只触碰约 8/512 的页。

## extract_moe_slice.py

```bash
PY=/usr/local/python3.12.13/bin/python3.12
$PY extract_moe_slice.py \
    --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4 \
    --layer 0 --expert-offset 0 --num-experts 8 \
    --outdir data/layer0_e0-8        # 默认即此路径
```

抽取的张量（`model.language_model.layers.{L}.mlp.*`，router/shared 为整
张量、routed experts 按第一维切 `[offset, offset+n)`）：

| 输出 bin | checkpoint 张量 | 形状 | dtype |
|---|---|---|---|
| `router_weight.bin` | `gate.weight` | [n, 2560] | bf16 |
| `shared_expert_gate_weight.bin` | `shared_expert_gate.weight` | [1, 2560] | bf16 |
| `experts.gate_up_proj.bin` | `experts.gate_up_proj` | [n, 1280, 1280] | u8(MXFP4) |
| `experts.gate_up_proj.weight_scale.bin` | 同名 | [n, 1280, 80] | u8(E8M0) |
| `experts.down_proj.bin` | `experts.down_proj` | [n, 2560, 320] | u8(MXFP4) |
| `experts.down_proj.weight_scale.bin` | 同名 | [n, 2560, 20] | u8(E8M0) |
| `shared_expert.{gate,up,down}_proj.bin`(+`.weight_scale.bin`) | `shared_expert.*_proj.weight`(+scale) | golden README 表 | u8 |

checkpoint 里的权重**已经是 MXFP4 打包 / bf16**，抽取是逐字节切片，
**不重新量化**；`manifest.json` 的 `source.tensors` 记录每个 bin 的源张量名、
分片文件、源形状与切片区间，可审计。8 专家默认输出 23.5 MB ≪ 2 GB 预算。

## selfcheck.py

```bash
$PY selfcheck.py                 # 默认校验 data/layer0_e0-8
```

1. bin/json 完整性与形状回读一致；
2. 与源 checkpoint 逐字节一致（源目录不可达则 SKIP）；
3. MXFP4 往返定点：`unpack(pack(unpack(p,s))) == unpack(p,s)` 值级精确
   （复用 `moe_block_ref` 的 pack/unpack，逐专家校验）；
4. 解包权重统计 mean/std：全部有限、非零，std 与 M4 sanity 基准同量级
   （专家 0.03、router/shared 0.02，容差 0.1×–10×），mean 相对 std 很小；
5. 输出总大小 < 2 GB；
6. golden 兼容：切片接入 `reference_moe_block` 端到端跑出有限、正常
   量级的 MoE 输出，路由 top-k 语义不变。

## 复现

```bash
$PY extract_moe_slice.py   # 重新抽取（byte-verbatim，输出稳定）
$PY selfcheck.py           # 全 PASS 即一致
```

---

# M93：MoE 真规模（512 专家）取证 + on-disk 布局核对

本段回答两件事：**真规模是多少**（含人类那句「一层的内存应该不多吧」）与
**checkpoint 的 MXFP4 on-disk 布局能不能被现有 kernel 直接消费**（人类原话：
「我们要求权重加载就能直接用，尽量少的做在线权重格式转换」）。

命令（仓库根目录，`PY=/usr/local/python3.12.13/bin/python3.12`）：

```bash
$PY tools/weights/moe_real_scale_audit.py --out tools/weights/evidence/moe_real_scale.txt   # rc=0
$PY tools/weights/moe_real_scale_audit.py --negative-experts 4                              # 必 FAIL，rc=1
```

## 1. 真规模（只读 safetensors 头，不读 payload）

`moe_real_scale_audit.py` 只用 `ShardReader.info()` 的头部字段（`shape` / `dtype` /
`data_begin` / `data_end`），**一个 payload 字节都不读**（32GB cgroup 下的硬约束）。读数见
`evidence/moe_real_scale.txt`（**20 checks / 0 FAIL**）。逐张量单专家字节：

| 张量（每层） | dtype | checkpoint shape | 单专家 B | 全 512 专家 B |
|---|---|---|---|---|
| `mlp.experts.gate_up_proj` | U8 | `[512, 1280, 1280]` | 1,638,400 | 838,860,800 |
| `mlp.experts.gate_up_proj.weight_scale` | U8 | `[512, 1280, 80]` | 102,400 | 52,428,800 |
| `mlp.experts.down_proj` | U8 | `[512, 2560, 320]` | 819,200 | 419,430,400 |
| `mlp.experts.down_proj.weight_scale` | U8 | `[512, 2560, 20]` | 51,200 | 26,214,400 |
| `mlp.gate.weight` | BF16 | `[512, 2560]` | 5,120 | 2,621,440 |
| `mlp.shared_expert_gate.weight` | BF16 | `[1, 2560]` | —（无专家维） | 5,120 |
| `mlp.shared_expert.{gate,up}_proj.weight` | U8 | `[640, 1280]` | — | 819,200 各 |
| `mlp.shared_expert.{gate,up}_proj.weight_scale` | U8 | `[640, 80]` | — | 51,200 各 |
| `mlp.shared_expert.down_proj.weight` | U8 | `[2560, 320]` | — | 819,200 |
| `mlp.shared_expert.down_proj.weight_scale` | U8 | `[2560, 20]` | — | 51,200 |

**「一层的内存应该不多吧」的答案（routed 专家，512 个全算）**：

```
routed 专家（512 专家 × 4 张量）      = 1,336,934,400 B = 1,275.000 MiB
router gate（512×2560 bf16）         =     2,621,440 B =     2.500 MiB
shared expert（gate/up/down + scale）=     2,616,320 B =     2.495 MiB
---- 单层 MoE 合计                   = 1,342,172,160 B = 1,279.995 MiB = 1.2500 GiB
                                       （= HBM 131072 MiB 的 0.977%）
```

**48 层常驻预算**：routed = 61,200.00 MiB（59.766 GiB）；MoE 段全量 = 61,439.77 MiB
（**60.000 GiB**）= **HBM 131072 MiB 的 46.87%**（routed 部分 46.69%）。

⇒ 「一层不多」（1.25 GiB / 0.98% HBM）成立。

**「权重常驻」可行吗（本 mission 的容量结论，`moe_real_scale_audit.py` 的 `C8` 判据现场算）**：

```
checkpoint 全部张量            = 182,234,382,328 B = 169.72 GiB
ngram 表（人类裁定留在 host）  = 102,400,491,776 B =  95.37 GiB（占整模型 56.19%）
---- 非 ngram（需进 HBM）      =  79,833,890,552 B =  74.35 GiB = HBM 128 GiB 的 58.09%
    其中 MoE routed 48 层      =  64,172,851,200 B =  59.77 GiB（= HBM 的 46.69%）
    其中 MoE routed（mtp 段）  =   5,033,164,800 B =   4.69 GiB
    其余（GDN/attn/hc/embed/lm_head/…）        =   9.90 GiB
---- 留给 KV / 激活 / 运行时    =  57,605,062,920 B =  53.65 GiB（HBM 的 41.91%）
```

⇒ **权重常驻是可行的**：按人类约束把 ngram 表留在 host（95.37 GiB），进 HBM 的非 ngram 权重
只有 **74.35 GiB = HBM 的 58.09%**，余 **53.65 GiB** 给 KV / 激活 / 运行时。
**是否再对 MoE 分层或流式，是那 53.65 GiB 余量下的调度权衡，不是本 dataset 的必然结论** ——
本节不主张「必须流式」。
> 注意 `docs/04-summary.md` 里那句「整模型 169.7 GiB …也放不进 128 GiB HBM，必须逐层 pread/mmap」
> 是**含 ngram 表**的口径（169.72 GiB 里 95.37 GiB 是 ngram）；引用它时要连口径一起引，
> 否则会与「ngram 在 host」这个前提打架。

**交叉见证（两个独立来源，非本文件自证）**：

* `tools/golden/data/real/manifest.json`（M90 的真拓扑档，**逐张量 `size_bytes`**）：
  上表 12 个权重张量逐项相等，权重合计 `1,342,172,160 B` —— 与本文件的单层合计**逐字节相同**。
* `m17_moe_real/evidence/weight_bytecheck.log`：`experts.gate_up_proj 838860800 B`、
  `down_proj 419430400 B`、`router_w 2621440 B`（**设备 D2H 与 host 逐字节一致**），
  与头部读数相同；该 log 的「合计 1342837760 B」是**含 4 个激活尺寸缓冲**的口径
  （权重部分恰为上表的 1,342,172,160 B）。

## 2. on-disk 布局能不能被现有 kernel **直接**消费

**引用约定（r3 起）**：本文件引用**代码位置一律以「符号 / 代码片段」为准，不写裸行号**
（依据 `docs/17` §8.2 的写作规则 —— 该节原文：**「只有行号、没有符号 = 不合格」**，
见 `grep -n '只有行号' docs/17-verification-standard.md`；并有 `docs/05` 的引用约定背书）。
**例外**：**仓外不可变工件**（checkpoint 自带的 `README.quant.md`、vLLM 源码）与**历史读数**（如某次崩溃的栈行号）保留行号，因为那些行号不会漂或本身就是要记的时点。
**为什么这条对本 mission 特别必要**：r2→r3 之间 `m15_moe_layer.h` / `m15_moe_resources.h`
被 **M91 与 M95 各推过一次**（本文档里 `p.wGu` 那句的行号已从 1960 → 2076 → **2172**）——
裸行号在合并进 main 后必然指向错行。**下文只给锚点**；当前行号可用这一批命令现取，
**只作查阅提示、不作依据**：

```bash
grep -n 'p\.wGu + slot \* GU_N \* (HIDDEN / 2)'      m15_layer_loop/m15_moe_layer.h
grep -n 'p\.sGu + slot \* GU_N \* GU_SCALE_STRIDE'   m15_layer_loop/m15_moe_layer.h
grep -n 'p\.wDn + slot \* HIDDEN \* (INTER / 2)'     m15_layer_loop/m15_moe_layer.h
grep -n 'par\.srcDValue = PACKED_K'                    m15_layer_loop/m15_moe_layer.h
grep -n 'srcGm\[xOff + K\]'                           m15_layer_loop/m15_moe_layer.h
grep -n 'VecQuantStage<INTER, DN_SCALE_STRIDE'         m15_layer_loop/m15_moe_layer.h
grep -n 'constexpr uint32_t NUM_EXPERTS = 4'           m15_layer_loop/m15_moe_resources.h
grep -n 'constexpr uint32_t TOPK_MAX = 4'              m15_layer_loop/m15_moe_resources.h
grep -n 'MW_WGU_BYTES ='                               m15_layer_loop/m15_layer_resources.h
grep -n 'MOE_W_STRIDE ='                               m15_layer_loop/m15_layer_resources.h
grep -n 'static bool H_LoadMoeW'                       m15_layer_loop/m15_moe_host.h
grep -n '__builtin_memcpy(W.wGuShd'                    m15_layer_loop/m15_moe_host.h
grep -n 'static bool H_LoadManifest'                   m15_layer_loop/m15_layer_loop.asc
grep -n 'static const M15Tensor\* H_FindTensor'       m15_layer_loop/m15_layer_loop.asc
grep -n 'NUM_EXPERTS = 512'                            m17_moe_real/m17_resources.h
grep -n 'TOPK_MAX = 10'                                m17_moe_real/m17_resources.h
```

**as-of 本 tip 的实测行号**（跑上面这批命令的实际读数；**只是提示，会随 M95+ 漂**）：
`m15_moe_layer.h` —— `p.wGu + slot*…` = **2172**、`p.sGu + slot*…` = **2173**、`p.wDn + slot*…` = **2207**、
`par.srcDValue = PACKED_K` = 224 与 **256**（后者在 `MXFP4GemmItem::CopyInB` 内，`CopyInB` 签名在 247）、
`srcGm[xOff + K]` = **1583**、`VecQuantStage<INTER, DN_SCALE_STRIDE` = **2274/2275**；
`m15_moe_resources.h` —— `NUM_EXPERTS = 4` = **70**、`TOPK_MAX = 4` = **71**；
`m15_layer_resources.h` —— `MW_WGU_BYTES =` = 170、`MOE_W_STRIDE =` = 193；
`m15_moe_host.h` —— `H_LoadMoeW` = 89、`__builtin_memcpy(W.wGuShd` = **119/120**；
`m15_layer_loop.asc` —— `H_LoadManifest` = 241、`H_FindTensor` = **300**；
`m17_moe_real/m17_resources.h` —— `NUM_EXPERTS = 512` / `TOPK_MAX = 10` = **53/54**。
（`m15_moe_host.h` / `m15_layer_resources.h` / `m15_layer_loop.asc` / `m17_moe_real/*` 这几份
**main 至今没动过**，所以它们的行号目前与锚点一致；`m15_moe_layer.h` / `m15_moe_resources.h`
**动过两次**，所以只有它们是必须靠锚点的。）

**消费侧契约**（现 device 路径，逐条以**符号 / 代码片段**为据）：

| 项 | 契约 | 依据（符号 / 代码片段，非行号） |
|---|---|---|
| 装载只「取字节」 | `H_LoadMoeW` 只做 pread，**不做任何数值换算** | `m15_layer_loop/m15_moe_host.h` 的 `static bool H_LoadMoeW(…)`（注释「本函数只做『取字节』」） |
| 唯一 host 组板 | 共享专家 `gate‖up` **行拼接**（routed 专家逐字节搬运） | `m15_moe_host.h` 里两句 `__builtin_memcpy(W.wGuShd.data() …)` / `…(W.sGuShd.data() …)`；搬运在 `H_PutMoeSlot` |
| gate_up 逐专家指针 | `p.wGu + slot * GU_N * (HIDDEN / 2)`（步长 = 1280×1280 B） | `m15_moe_layer.h` 里 `shd ? p.wGuShd : (p.wGu + slot * GU_N * (HIDDEN / 2))` |
| gate_up scale 逐专家指针 | `p.sGu + slot * GU_N * GU_SCALE_STRIDE`（80 B/行） | `m15_moe_layer.h` 里 `p.sGu + slot * GU_N * GU_SCALE_STRIDE` |
| down 逐专家指针 | `p.wDn + slot * HIDDEN * (INTER / 2)`（2560×320 B） | `m15_moe_layer.h` 里 `p.wDn + slot * HIDDEN * (INTER / 2)` |
| B 按 `[N, K]` 原始 layout 搬（K 打包在行内） | `par.srcDValue = PACKED_K` | `m15_moe_layer.h` 的 **`MXFP4GemmItem::CopyInB`** 内那句 `par.srcDValue = PACKED_K` |
| gate/up 语义切分 = 行 `[0,640)` gate、`[640,1280)` up | SwiGLU 按 `K=INTER` 读 `srcGm[xOff+K]` | `m15_moe_layer.h` 里 `DataCopy(gateL, srcGm[xOff], …)` / `DataCopy(upL, srcGm[xOff + K], …)`；模板 `VecQuantStage<INTER, DN_SCALE_STRIDE, true, …>` 的 `quantH` |
| 槽尺寸按 4 专家编译 | `MW_WGU_BYTES = NUM_EXPERTS*GU_N*(HIDDEN/2)` 等 | `m15_layer_resources.h` 的 `MW_WGU_BYTES =` / `MOE_W_STRIDE =`（槽位表整体见 `MW_ROUTER_OFF … MW_G2_OFF`） |

**on-disk 布局 vs 契约（逐项核对）**：

| 项 | checkpoint on-disk | kernel 期望 | 需要在线转换？ |
|---|---|---|---|
| 专家维位置 | **最外维** `shape[0]=512`（逐层 48 层一致） | 逐专家槽 `slot` 乘固定步长 | **否** |
| gate_up packed | `[512, 1280, 1280]` u8，K 打包在最后一维（K=2560 → 1280 B/行） | `wGu[slot]` 为 `[GU_N=1280, K/2=1280]` u8 | **否**（同形） |
| gate_up scale | `[512, 1280, 80]` u8（K/32，行距 80 B） | `sGu[slot]` 为 `[1280, 80]` u8 | **否** |
| down packed | `[512, 2560, 320]` u8（K=640 → 320 B/行） | `wDn[slot]` 为 `[2560, INTER/2=320]` u8 | **否** |
| down scale | `[512, 2560, 20]` u8（行距 **20 B**） | `sDn[slot]` 为 `[2560, 20]` u8 | **否** |
| 转置 | **无**（N 为行、K 为打包列，行主序） | 同（`srcDValue = PACKED_K`） | **否** |
| nibble 次序 | `uint8-nibble-lohi`（**低 nibble = 偶数 in 索引**） | 参考侧 `_pack_codes`：`lo = codes[:,0::2]; hi = codes[:,1::2]; (lo \| hi<<4)` | **否**（同约定） |
| group 轴/大小 | group 32 **沿 in_features（K）** | `reshape(n, k//32, 32)`；`SCALE_K = K/GROUP` | **否** |
| e8m0 解码 | `byte = exponent + 127` ⇒ `2**(byte-127)` | `e8m0_decode`：`exp2(b - 127)` | **否** |
| gate/up 行序 | 行 `[0,640)` = gate、`[640,1280)` = up | 同（SwiGLU 按 `K=INTER` 取后半） | **否** |
| 反量化 | 硬件 `MmadMx` 直接吃 packed + e8m0 | 同 | **否** |

**结论：本 mission 的「必须做的在线转换清单」= 空**。routed 专家的 checkpoint 字节已经是
kernel 期望的 `[E, N, K/2]` packed-u8 + `[E, N, K/32]` e8m0 行主序形式；唯一的 host 组板是
**共享专家**（`gate`/`up` 两个独立张量 → 行拼接），与 routed 专家无关，且**已存在**于
`m15_moe_host.h` 的那两句 `__builtin_memcpy(W.wGuShd/W.sGuShd …)`。**不存在需要新增的在线转置 / 重排 / 反量化 / 格式转换。**

发现的**两处规模差（不是布局差）**，逐条登记（都属后续 mission，不在本 mission 写权限内）：

1. **专家数 512 vs 4**：`m15_moe_resources.h` 的 `constexpr uint32_t NUM_EXPERTS = 4`（注释自陈「golden 数据集：
   512 → 4」）；槽宽 `m15_layer_resources.h` 的 `MOE_W_STRIDE` 只装得下 4 专家。
   `m17_moe_real` 已演示怎么改：`m17_resources.h` 的 `NUM_EXPERTS = 512` / `TOPK_MAX = 10`。
2. **top-k 10 vs 2（模板上界 4）**：`m15_moe_resources.h` 的 `constexpr uint32_t TOPK_MAX = 4`（注释「512 → 4 /
   10 → 2」）。config 的真值是 `num_experts_per_tok = 10`。

## 3. manifest 的**加性**扩展（全 512 专家）

`slice_layer_manifest.py` 新增 `MOE512_ROLES`（5 个新 role，前缀 `moe512_`），
**纯追加在 `weights_manifest.txt` 文件尾**：48 层 × 5 行 = **240 个 tensor 行**，+
3 个 `moe512_*` 元数据行 + 3 个汇总行。

* **加性证明**：`diff <改前> <改后>` = **删除 0 行 / 新增 247 行**
  （`evidence/m93_manifest_diff.txt`）。`role=moe_*` 的每个 E=4 行（offset/bytes/role）
  **逐字节不变**。
  **基线钉的是不可变 commit**（本轮 fork 点 `52f46b8693f0292c132312ea1d47e78c5b52888a`），
  **不钉 `main`** —— 否则本分支合入后 `main` 的 manifest 就等于本 tip，`diff` 退化成 0/0、
  这条判据会**静默空洞**。脚本在**三条失败路径**上强制 `rc≠0`（都不许静默 PASS）：
  「基线**读不到/不存在**」⇒ `ADDITIVE-CRIT FAIL：基线… 读不到` + exit 1；
  「基线 == tip」（0 删 0 增）⇒ `ADDITIVE-CRIT **VACUOUS**` + exit 1；
  「删除 != 0」（真非加性）⇒ `ADDITIVE-CRIT FAIL：…有 N 行在 tip 上不存在` + exit 1。
  详见 §8 与 `tools/weights/m93_negative_controls.sh` 头部的「加性基线」段。
* **为什么不改消费者就安全**：`m15_layer_loop.asc` 的 `H_FindTensor` 是
  `T.layer == layer && T.role == role` 的**精确串匹配**；新 role 名与任何现 role 都**不是**
  子串关系（`moe512_experts_gate_up` 不含 `moe_experts_gate_up`），`H_LoadManifest`
  （`m15_layer_loop.asc` 的 `H_LoadManifest`）对非 `tensor ` 前缀行直接忽略 ⇒ 新段对现消费者不可见。
* **量**：`moe512_routed_layer_bytes=1336934400`、`moe512_routed_total_bytes=64172851200`、
  `moe512_num_experts=512`、`moe512_topk=10`。

## 4. 判据与负向对照

| 判据 | 内容 | 读数 |
|---|---|---|
| `moe_real_scale_audit.py` C0/C1/C2/C3/C4/C5/C6/C8 | 分片完整、48 层逐层字节相同、头部 offset 差 == shape×dtype、单专家×512 == 头部全量、K 打包/尺度步长、专家维在最外维、**非 ngram 权重 ≤ HBM（C8，容量账）** | **20 checks / 0 FAIL** |
| `slice_layer_manifest.py --check` | 已生成 manifest 与 checkpoint 现况一致 | rc=0 |
| NC-1 `--negative-experts 4` | 把专家数当 4（当前缩形档） | **5 FAIL / rc=1**（每条给出 `1 专家 X × 4 = Y != 头部全量 Z`） |
| NC-2 `--moe512-neg-experts 4`（生成器侧） | 同上，落在 manifest 生成路径 | **FAIL / rc=1** |
| NC-2b `--moe512-neg-experts 256` | 证明不是特判 4 | **FAIL / rc=1** |
| NC-3 stride 变异（`one = prod(shape)`，仅 `/tmp` 副本） | stride 算错 | **FAIL / rc=1**（`头部算出的单专家 2621440 B != config 几何量期望 5120 B`） |
| NC-4 `NUM_EXPERTS_MOE: 4→8`（仅 `/tmp` 副本） | 改缩形档专家数 | **加性判据 FAIL**（`diff` 出现删除行） |

### 4.1 第五变体纪律：「把被测对象弄坏，判据必须变红」

塔的第五变体纪律（`20260927-tower-all-item.md`）要求**每条 PASS 判据都要给一次「破坏被测对象 ⇒ 必红」的实测读数**。
**本节的映射已覆盖本 mission 的每一条判据**（不在表里的在下面「没有破坏对照的判据」里逐条披露）。
全部实跑读数见 `evidence/m93_negative_controls.txt`（该文件含 NC-5..NC-12 段）。
机制：`build_mirror <dst> <mutation>` 在 `/tmp` 造 131 分片镜像（**真实头 + 稀疏尾**，
`du` 实测**每镜像 644 KB**），所有破坏都是**等宽原地改** ⇒ 头长不变、其他张量偏移不动。

**逐条判据 → 破坏对照**：

| 判据（本 mission 的交付判据） | 怎么弄坏被测对象 | 哪些判据翻红 | 读数 |
|---|---|---|---|
| **C0** 分片完整性 | NC-11：镜像某分片截到只剩头（该分片全部张量变 incomplete） | **C0** | `incomplete shards = 1` / rc=1 / **fails=1**（其余含 C1/C2/C5/C8 全 PASS） |
| **C1** 48 层逐层字节相同 | NC-5：`gate_up` 的 `data_offsets` 末端等宽 −1 | C2、C1、C3/C6、C6 | **4 FAIL / rc=1**；其余 12 张量与 C4/C8 仍 PASS |
| **C2** 头部 offset 差 == shape×dtype | 同 NC-5 | 同上 | `hdr=838860799 shape×dt=838860800` |
| **C3/C6** 单专家 × n == 头部全量 | 同 NC-5（gate_up）；生成器侧另见 NC-2/2b | 同上 | `838860800 vs 838860799` |
| **C4** K 打包 / N 几何 | NC-10：镜像 `gate_up` 的 shape K `1280→1279`（等宽） | C4、C3/C6、C2 | `1279 ×2 = 2558 != 2560` / **3 FAIL / rc=1** |
| **C5** 专家维 `shape[0] == num_experts` | NC-9：镜像 `shape:[512,...]→[256,...]`（等宽） | C5、C2 | `shape[0]=256` / **2 FAIL / rc=1** |
| **C6** 全量/单专家 == num_experts | 同 NC-5 | 同上 | `511 != 512` |
| **C8** 非 ngram 权重 ≤ HBM（容量账） | NC-8：镜像里 **130 个 `ngram` 张量名等宽改成 `xxxxx`**（破坏「ngram 分流」这个被测对象）——**复审 r2 提供的配方，本条照抄并实跑** | **C8** | 非 ngram 变成 **169.72 GiB = HBM 的 132.59%**、**余 −41.72 GiB** ⇒ `C8 FAIL` / rc=1 / **fails=1**（C1/C2/C5 仍 PASS） |
| 生成器 **C7**（E 档 × 倍数 == 真规模） | NC-12：`/tmp` 副本把 `NUM_EXPERTS_MOE` `4→3`（`512 % 3 = 2 ≠ 0`） | 生成器 C7 | `E=3 档 15360 B × (512/3) != 真规模 2621440 B` / rc=1 |
| 生成器 **C4b**（单专家字节 == config 几何量） | NC-3：`/tmp` 副本把 `one = prod(shape[1:])` 改成 `prod(shape)` | 生成器 C4b | `单专家 2621440 B != config 几何量期望 5120 B` / rc=1 |
| 生成器 真规模算式 vs 头部 | NC-2 / NC-2b：`--moe512-neg-experts 4` / `256` | 生成器算式 | `1 专家 5120 × 4 = 20480 != 2621440` / rc=1 |
| 生成器 `--check` | NC-6：入库 manifest 等宽改一字节 | `--check` | `[FAIL] ... 与 checkpoint 现况不一致` / rc=1 |
| 「E=4 行逐字节不变」加性判据 | NC-4：`/tmp` 副本 `NUM_EXPERTS_MOE` `4→8` | 加性判据（`diff` 删除行数） | 删除 **240 行**（要求 0）⇒ FAIL |
| 设备侧 `runs=all` 判据（本 mission 的产物 = manifest） | NC-7：把破坏后的 manifest 喂给设备 | 设备侧 `B.ssm.*` 等 | **451 条 FAIL / rc=1 / `ALL PASS` 行数 0**（未破坏时同配置 = 2068 + 290 / 0 FAIL） |

> **NC-4 与 NC-12 是一对，不能只看一个**：NC-4（4→8）**骗得过**生成器内部算式 —— 倍数比
> `512/NUM_EXPERTS_MOE` 与档宽同源 ⇒ C7 仍绿，它只被**加性判据**咬住；NC-12（4→3）才是能翻
> C7 的变异（`512 % 3 ≠ 0`）。两者一起才覆盖「缩形档专家数被改错」这个形态。

**没有破坏对照的判据（如实披露，不写成「已全部覆盖」）**：以下三条是
`slice_layer_manifest.py` 的**结构性自检**，其被测对象是 `config.json` 与**本文件常量本身**，
本 mission 未为它们造破坏对照：

1. `layer_types` 与 3:1 模式（每 `full_attention_interval` 个的末位必须是 `full_attention`）；
2. 单层 `in_proj` 拼接行数 == `IN_N`(16480)；
3. `config` 几何量 `(hidden, inter, group) == (2560, 640, 32)`（本文件常量）。

理由：要翻红它们必须改 `config.json` 或改本文件常量 —— 那已不是「弄坏被测对象」，而是
**换一个输入 / 换一份实现**，读数的解释会变（与第五变体要证的「判据盯着某个对象」不是同一件事）。

**未做的破坏对照（真实范围披露）**：kernel 侧那 2068 条判据的被测对象**不是本 mission 改的东西**
—— 本 mission 未碰任何 `.asc`/`.h`，改前/改后（新基线 `main` 与 `HEAD`）二进制 **sha256 逐字节相同**
（见 §8）。要对 kernel 做「弄坏实现」对照必须改 `m15_*.h` / `m15_layer_loop.asc`，而那些文件
**不在本 mission 的写权限内**，故**未做**。本 mission 能做的等价对照是 NC-7（弄坏**自己的产物**
manifest ⇒ 设备判据变红），已给读数。

**checks 口径**：NC/破坏对照**不计入** `moe_real_scale_audit.py` 的 checks（那是该工具的
20 条判据）；NC-8 是**给已有判据 C8 补一次破坏读数**，不是新增判据 ⇒ checks 仍为 **20**
（NC-8 读数是 `checks=20 fails=1`，即 20 条里红 1 条）。

**NC-1 的读数与 manifest 里的 E=4 行内部自洽**：`--negative-experts 4` 给出的
`6,553,600 / 409,600 / 3,276,800 / 204,800` 正是 `weights_manifest.txt` 里那 4 个 E=4 行的
`bytes`（真规模 ÷ 128 = 512/4）。**注意这不算「互证」**：两者同读一个 safetensors 头、同用
`prod(shape[1:]) × dtype × 4` 这条算式 ⇒ 是**同源内部自洽**（能抓 stride/专家数写错，抓不住
「共同误读头」）。真正的**独立**互证是 §1 那条：`tools/golden/data/real/manifest.json`
（另一个 mission 的另一条生成路径）与 `m17_moe_real` 的**设备 D2H 逐字节**核对。

## 5. 输入溯源（`docs/17` §1.3 三分法）

本节所有判据吃的输入逐条归类（`docs/17-verification-standard.md` 的
`#### 1.3 参考的输入从哪来` 一节）：

| prov | 类别 | 本节的输入 | 合法性 |
|---|---|---|---|
| **①** | 声明输入 | `/workspace/Qwen3.8-Flash-Next-MXFP4/*.safetensors` 的 **JSON 头**（`shape`/`dtype`/`data_offsets`；容量账读**全部 1898 个张量**的头，判据只读 MoE 那 12 个 role）、同目录 `config.json` 的 `text_config`、checkpoint 自带 `README.quant.md` | 合法（输入隔离 N1） |
| **②** | 上游输出（设备产物） | **无** | — |
| **③** | 被判量自身产物 | **无**（本节无参考实现，只有「头部读数 vs config 几何量」两个来源互核） | — |

## 6. 规则来源（`docs/17` §1.2）

布局约定（nibble 次序、group 轴、e8m0 bias、3D 打包形状）**钉在仓外不可变对象**上：

* `CKPT:/workspace/Qwen3.8-Flash-Next-MXFP4/README.quant.md:4-7`（元素 E2M1 / group 32 沿
  in_features / `scale = 2**(floor(log2(max_abs))-2)` / `packing: uint8-nibble-lohi`）
  与 `:33-35`（`[E, out, in] → packed [E, out, ceil(in/2)] + scale [E, out, in/32]`）。
* 权威参照（L2）：`/workspace/vllm/vllm/models/qwen4_exp/nvidia/model.py:159`
  （`Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)`）与 `:647`
  （`packed_modules_mapping` 的 `"gate_up_proj": ["gate_proj", "up_proj"]` ⇒
  1280 输出行的前 640 = gate、后 640 = up）。
* 仓内已有转写（**同源转录**，不是独立第二来源）：`tools/golden/moe_block_ref.py::_pack_codes`
  （`:270-274`）、`::unpack_mxfp4`（`:427-447`）、`::e8m0_decode`（`:111-118`）；
  `m13_moe_layer/check_ref.py::dequant_device`（`:153-161`）；`m21_layer_ref/ref/mxfp4.py` 的模块 docstring（开头那段 `CKPT:README.quant.md:<行>` 摘录）。

## 7. 显式未完成项（本 mission **没有**做）

逐条列出，均属后续 MoE-A / MoE-B（本 mission 的写权限不含这些文件）：

1. **kernel 侧常量 `NUM_EXPERTS 4→512`**（`m15_moe_resources.h` 的 `constexpr uint32_t NUM_EXPERTS = 4`）与槽尺寸重算
   （`SZ_*` 现按 **`TOTAL_MAX = M_MAX * TOPK_MAX`** 定尺 —— 即 M91 之后的「紧凑 Σt_e」布局，
   不再是 `NUM_EXPERTS * M_MAX * ...`；见 `m15_moe_resources.h` 里 `SZ_AQ =` … `SZ_Y =` 那一段）。
2. **`TOPK_MAX 4→10`**（`m15_moe_resources.h` 的 `constexpr uint32_t TOPK_MAX = 4`，真值 `num_experts_per_tok=10`；
   M96 已把「模板上界」与「运行期 topk」拆成 `TOPK_TMPL` / `TOPK` 两个常量）。
3. **`MOE_W_STRIDE` 扩槽**（`m15_layer_resources.h` 的 `MOE_W_STRIDE =`，现 13,091,840 B/层 ÷ 4 专家；
   512 专家需要 ≈1.25 GiB/层）。
4. **`m15_layer_loop.asc` 的 host arena / MoE host 装载路径**（`m15_layer_loop.asc` 里 `moeArenaBytes = … MOE_W_STRIDE` 到 `H_PutMoeSlot(C.moeArena, …)` 那一段（`moeArena` / `H_LoadMoeW` 调用）
   的 `moeArena`/`H_LoadMoeW` 调用）——本 mission **未改**该文件。
5. **消费 `moe512_*` 新 role 的加载器**：目前**没有任何**消费者读这 5 个新 role
   （`H_FindTensor` 精确匹配 ⇒ 自然忽略）。本 mission 只把真规模**落进 manifest**，
   不接线。
6. **KV / 激活侧的驻留与调度策略** —— 本 mission 只给容量账（§1：非 ngram 权重 74.35 GiB、
   剩余 53.65 GiB 给 KV/激活），**不主张「必须分层或流式」**，也不给调度策略。
7. **设备侧真规模判据读数**：本 mission 的 `runs=all`/`runs=chain` 零回归是在**缩形档（E=4）**
   上取的（见 §8），512 档的 device 判据要等 1-4 落地。
8. **`m17_moe_real` 已有的 512 档设备字节核对**（`evidence/weight_bytecheck.log`）是**只读引用**，
   本 mission 未复跑、未改 m17。
9. **MTP 段的 MoE routed 权重**（`mtp.layers.*.mlp.experts.*`，4.69 GiB）**只进了 §1 的容量账**，
   没有进 manifest（现 manifest 只覆盖 `model.language_model.layers.*`）。要不要为 MTP 段也落
   真规模切片，属后续 mission 的口径决定。
10. **`ngram` 表 95.37 GiB 的 host 侧驻留方式**（mmap / 稀疏行查找）不是本 mission 的范围
    （M86/M92 那条线）。

## 8. 零回归（两个二进制 × 两个 manifest 的四个组合；**基线 = 不可变 fork 点 `52f46b8…`**）

**基线口径（重要）**：r2 复审的对象是 `04b2ea9`（baseline 为当时的 `main` `0bd80c6`）。r3 之前我
按塔的要求 **rebase 到新 main**；r4 又 **rebase 到当时最新的 main `40dd6de`**（其上已并入
**M95**，此前已并入 M91/M96）。**M91 与 M95 都改过 `m15_moe_layer.h` / `m15_moe_resources.h`**
⇒ **二进制 hash 每轮都变**（`c6ee3ca0…` → `2d42ba86…` → `958c6f2a…`，**三次变化全部来自别的
mission，不是本 mission**）。所以本节所有读数都是**在本 tip 的基线上重取的**，不沿用 r1/r2/r3 那批。

**加性判据的基线钉的是「不可变 commit」，不是 `main`**（塔的第六变体：复现命令不要钉会移动的 ref）：
`tools/weights/m93_negative_controls.sh` 的默认 `BASE_COMMIT=` **本轮的 fork 点全 40 位 sha**
`52f46b8693f0292c132312ea1d47e78c5b52888a`（= rebase 时的 `main`，也是 `git merge-base main HEAD`），**不读 `main`**。
（r5→r6 之间 `main` 从 `40dd6de` 前移到 `52f46b8`（M94 并入）；复审核实 `40dd6de..52f46b8`**没动**本 mission 的 manifest 与所引代码文件，两者 manifest 内容相同 ⇒ 基线换钉新的 fork 点，加性读数不变。）
**为什么必须这样**：本分支一合入，`main` 的 `weights_manifest.txt` 就等于本 tip ⇒ 若基线跟着
`main` 走，加性 `diff` 会退化成 **0 删 / 0 增**，归档里记的 **0/247** 便**复现不出** ——
而「证明这次改动是**纯加性**的」正是本 mission 的核心目的，那会把这条判据**静默变成空洞**。
**现在它不再可能静默空洞**：脚本对「基线 manifest == tip manifest」有
`ADDITIVE-CRIT **VACUOUS**` 守卫，命中即**响亮地 `rc=1` 退出**。实测两种情形：
```
$ bash tools/weights/m93_negative_controls.sh            # 默认（钉 fork 点）
删除行数=0  新增行数=247
ADDITIVE-CRIT PASS（删除 0 / 新增 247 > 0 ⇒ 非空洞）      rc=0
$ BASE_COMMIT=$(git rev-parse HEAD) bash tools/weights/m93_negative_controls.sh   # 模拟「已合入 ⇒ 基线==tip」
删除行数=0  新增行数=0
ADDITIVE-CRIT **VACUOUS**：基线（…）与 tip 的 manifest 相同 ⇒ 本判据无信息。   rc=1
```
（第二条就是 r4 复审判为 p2 的那个形态；**它现在是一条会红的守卫**，而不是一个事后才发现的漏洞。）

**并发背景**（纪律 ⑧，且本 mission 已因并发 OOM 过两次）：每次 run **之前**都用
`npu-smi info | grep -q m15_layer_loop` 等 NPU 上没有别的 `m15_layer_loop`，才起我自己的
（一次只跑一个）。实测**单次 `runs=all`（48 层）host 峰值 RSS = 15,982 MB ≈ 15.6 GiB** ⇒ 32 GB
cgroup 下两个并发即超限；本批期间确实等到过兄弟 run（`[wait_idle] 等了 N0s 后空闲`）。

**两个二进制**：

```
$ sha256sum /tmp/m93_base/m15_layer_loop/build/m15_layer_loop ./m15_layer_loop/build/m15_layer_loop
958c6f2a6d699537ad7fd6509bf77cd6dc4abe2fd6266e7e22f8e10e4c1fe117  （改前 = main `40dd6de`，git archive 到 /tmp 后构建）
958c6f2a6d699537ad7fd6509bf77cd6dc4abe2fd6266e7e22f8e10e4c1fe117  （改后 = 本分支）
```

⇒ **两个二进制逐字节相同** —— 本 mission 没有碰任何 `.asc`/`.h`，变化的**只有 manifest**。
（`958c6f2a…` 与 M95 自己归档里的「改后二进制」一致，见 `m15_layer_loop/moe_relift/m95_README.md`
的对照表 ⇒ 我的构建与 M95 的归档互为印证。）

| 组合 | 二进制 | manifest | `runs=all` | `runs=chain` |
|---|---|---|---|---|
| 1 | 改前 | 改前（base） | **2068 checks + 290 guards / 0 FAIL** | **910 + 52 / 0 FAIL** |
| 2 | 改前 | 改后（tip） | **2068 + 290 / 0 FAIL** | **910 + 52 / 0 FAIL** |
| 3 | 改后 | 改前（base） | **2068 + 290 / 0 FAIL** | **910 + 52 / 0 FAIL** |
| 4 | 改后 | 改后（tip） | **2068 + 290 / 0 FAIL** | **910 + 52 / 0 FAIL** |

命令（4 个组合共用；`<bin>`/`<manifest>` 按表替换，`M15_HC_LAYERS=0,1,3`）：

```bash
M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 <bin> <manifest> all
M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 <bin> <manifest> chain
```

日志：`evidence/m93_accept_run_all_*.log`（4 个）、`evidence/m93_accept_run_chain_*.log`（4 个）。
**读数与塔给的基线逐项相同**：`runs=all` = 2068 + 290 / 0 FAIL，`runs=chain` = 910 + 52 / 0 FAIL
⇒ rebase 到新 main（含 M91）之后，这两组计数**没有变**，且本 mission 的 manifest **不改变它们**。

消费者脚本（`check_*.py` 本 mission 未改）：

| 脚本 | 读数（**本 tip 上重取，两份 manifest 各跑一遍、读数逐字相同**） | 命令 |
|---|---|---|
| `check_ref.py` | rc=0，**判定项 709/709 PASS**（另有 36 条参考项） | `M15_DUMP=1 <bin> <manifest> all` 后在该 dump 目录跑 |
| `check_hc_ref.py` | rc=0，`RESULT: OK (79/79：判定项 64 + guard 15；SKIPPED 1)` | `M15_DUMP=1 M15_LAYERS=4 M15_HC_LAYERS=0,1,3 <bin> <manifest> h` |
| `check_chain_ref.py` | rc=0，`RESULT: OK (92：判定项 84 + guard 8；SKIPPED 2)` | `M15_DUMP=1 M15_LAYERS=48 M15_STEPS=1 <bin> <manifest> chain` |
| `check_moe_ref.py` | rc=0，`RESULT: OK (128 条判定项 + 20 条非空洞全部比较通过)` | `M15_DUMP=1 M15_LAYERS=4 M15_HC_LAYERS=0,1,3 <bin> <manifest> all` 后 `check_moe_ref.py <dumpdir> --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4` |

**另外**：`check_ref` 那一档的 dump（`runs=all`）在**两份 manifest 下逐字节相同**（1307 文件、
清单摘要 `dff40e9b…`）⇒ 「manifest 变了但读数没变」在 `check_ref` 这一档是**被 dump 层直接证实**的，
不只是两条计数相同。

日志：`evidence/m93_check_{ref,hc_ref,chain_ref,moe_ref}_tip.log`（`check_ref`/`hc_ref`/`chain_ref`
另各有 `_base.log`，两份 manifest 读数逐字相同）。

> **`check_moe_ref.py` 这条 finding 已结案（不是本 mission 修的，但要点名状态）**：
> r1 时它在 main 上**崩**（`ValueError: cannot reshape array of size 32 into shape (20)`，
> 栈上的 `check_moe_ref.py:357`（**as-of 旧基线 `0bd80c6` 的行号**）→ `m17_moe_real/check_ref.py:219`），我隔离出「同一二进制 + 同一 dump，
> `26d644b` 版 rc=0、`0bd80c6` 版 rc=1」并报 finding / 归档
> `evidence/m93_check_moe_ref_preexisting.txt`。**塔据此开了 M96**，其 commit `eca3a44` 修掉了
> 三处口径漂移（槽宽 `DN_SCALE_STRIDE` vs 有效前缀 `DN_SCALE_EFF`、模板上界 `TOPK_TMPL` vs
> 运行期 `TOPK`、1-D/2-D 形状），**现已随 `004cab4` 合入 main**。上表那行 `128 + 20` 是
> **我在 rebase 后的 tip 上重取的读数**（`evidence/m93_check_moe_ref_tip.log`）⇒ 由红转绿已复核。
> M96 之前那两份读数（旧基线上的红）已改名为 `m93_check_moe_ref_preM96_base.{log,run.log}`，
> 免得被当成当前读数；`m93_check_moe_ref_preexisting.txt` 末尾有一句时点注。

**更强的零回归证据 —— dump 逐字节相同**（`evidence/m93_dump_sha256_identity.txt`）：同一 dump
配置下，base manifest 与 tip manifest 产出的 **全部 dump 文件 sha256 全部相同**（`diff` 0 行）
⇒ 新 manifest 段**对设备侧产物没有任何影响**（与「二进制逐字节相同」互为印证）。

> **为什么不需要给消费者加参数**：`m15_layer_loop.asc` 的 `H_FindTensor` 是
> `(layer, role)` **精确串匹配**，新 role（`moe512_*`）不可能被任何现 role 命中；
> `H_LoadManifest`（`m15_layer_loop.asc` 的 `static bool H_LoadManifest(…)`）对 `#` 开头与非三个已知前缀的行**直接忽略**，
> 对 `tensor ` 行只按 key 取值（多出来的 role 只是躺进 `M.tensors`）。
> 所以「不改消费者 ⇒ 现有判据仍绿」在本 mission 是可证的（二进制相同 + dump 逐字节相同），
> 而不是约定。

### 8.1 复核入口（r6 实跑的**五条**，供 reviewer 照抄）

```bash
# ① 归档复现（含设备档 NC-7；跑前先确认 npu-smi 无兄弟 m15_layer_loop）
M93_NC_DEVICE=1 bash tools/weights/m93_negative_controls.sh \
    > /tmp/nc.txt 2>&1 && diff /tmp/nc.txt tools/weights/evidence/m93_negative_controls.txt
#   ⇒ 我实跑：与归档**逐字节相同**，rc=0
# ② 只重定向 stdout（应差 1 行 —— NC-6 的 [FAIL] 走 stderr）
M93_NC_DEVICE=1 bash tools/weights/m93_negative_controls.sh > /tmp/nc2.txt 2>/dev/null
diff /tmp/nc2.txt tools/weights/evidence/m93_negative_controls.txt | grep -c '^[<>]'
#   ⇒ 我实跑：差异 1 行
# ③ **合并后场景**：基线取成本 tip（模拟「本分支已合入 ⇒ 基线 manifest == tip」）
BASE_COMMIT=$(git rev-parse HEAD) bash tools/weights/m93_negative_controls.sh; echo rc=$?
#   ⇒ 我实跑：`ADDITIVE-CRIT **VACUOUS**…` + rc=1（**响亮失败**，不是静默 0/0 PASS）
#      ——这条正是 r4 复审判为 p2 的形态，现在是一条会红的守卫。
# ④ **基线读不到/不存在**（塔点名的第三场景）
BASE_COMMIT=0000000000000000000000000000000000000000 bash tools/weights/m93_negative_controls.sh; echo rc=$?
#   ⇒ 我实跑：`ADDITIVE-CRIT FAIL：基线 commit 0000… 读不到 m15_layer_loop/weights_manifest.txt`
#      + **rc=1**（修前这里是**错误 PASS**：before.txt 为空 ⇒ del=0/add=1646 ⇒ 报 PASS 且 rc=0）
# ⑤ **真非加性（删除 != 0）** —— 在 `/tmp` 的 `--shared` 克隆里删掉 tip manifest 的**一行基线里存在的行**：
BASE_COMMIT=52f46b8693f0292c132312ea1d47e78c5b52888a bash tools/weights/m93_negative_controls.sh; echo rc=$?
#   ⇒ 我实跑：`删除行数=1 新增行数=247` → `ADDITIVE-CRIT FAIL：基线（52f46b8）里有 1 行在 tip 上不存在`
#      + **rc=1**（修前该分支**不强制 rc≠0**，与 VACUOUS 分支不对称）
```

**四条失败路径的读数汇总**（都在 tip 上实跑）：
| 场景 | 修前 | 修后 |
|---|---|---|
| ① 正常（基线 = fork 点） | `PASS（0/247）` rc=0 | 不变：`PASS（删除 0 / 新增 247 > 0 ⇒ 非空洞）` **rc=0** |
| ② 基线**读不到/不存在** | **错误 PASS（0/1646）rc=0** ❌ | `FAIL：基线…读不到` **rc=1** |
| ③ 基线 == tip（VACUOUS） | rc=1 | rc=1（不变） |
| ④ 删除 != 0（真非加性） | `FAIL` 但 **rc=0** ❌ | `FAIL：…有 N 行在 tip 上不存在` **rc=1** |

**为什么 ③ 重要**：它把「加性判据会不会变空洞」变成**可执行的**检查。默认基线是不可变 fork 点，
所以 ③ 只在基线被人为设成 tip 时才触发；而一旦本分支合入 `main`，默认基线**仍是** fork 点 ⇒
① 依然复现出 **0 删 / 247 增**，不依赖 `main` 当时指哪里。
