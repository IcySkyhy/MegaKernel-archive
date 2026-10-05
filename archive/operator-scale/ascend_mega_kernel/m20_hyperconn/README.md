# M36：Hyper-connection 融合 kernel（单一 `__mix__(1,2)`，decode 档 m=1，bf16）

图纸 = 官方 vLLM 的 `qwen4_exp` 实现（唯一权威规格，本机 `/workspace/vllm/vllm/models/qwen4_exp/`）
+ `docs/05-megakernel-design.md` §2/§5/§6（约束与 quirk 清单）。算件代码全部**复制改造**自 main 上的
m6 / m11 / m14（donor 目录未改动）。

**一个 kernel 完成一次 hyper-connection 边界**（vLLM 侧的模块级算子），覆盖三档语义（runtime `mode`）：

| mode | 对应 vLLM 调用点 | 输入（GM） | 输出（GM） |
|---|---|---|---|
| 0 `MODE_MIX` | 第 0 层 `attn_hc.mix(h)`（无 pending combine） | H, Wdown, Wup, Winj, hc_norm | XN, OH(lora+injection), LS, GATE, BLK |
| 1 `MODE_COMBINE_MIX` | 层内 `attn_hc/mlp_hc.combine_and_mix`（`use_combine=true`） | + BO, IJ | **H'**（4 路残差流出）, XN, OH(含 injection), LS, GATE, BLK |
| 2 `MODE_FINAL_MIX` | 全局 `hyper_connection_mixer.combine_and_mix`（`use_combine=false`） | + BO, IJ | H', XN, OH(仅 lora), LS, GATE, BLK |

`BLK` = 该边界的 block input（喂 attn / MLP / lm_head），`H'` = 更新后的 4 路残差流，
`OH[:,320:324]` = injection logits（**延迟到下一个 hc 边界的 combine 才被使用**）。

**为什么只有三档 mode 就够（`inj_logits is None` 的路径在 48 层 + 全局 mixer 上不可达）**：
`docs/14` §3.2 的 `mixer_combine` 写了「`inj_logits is None` ⇒ `w = 1.0`」（对应
`_hc_combine_norm_kernel` 收到 `inj_ptr == nullptr` 时按单位权重注入 block output）。这在 vLLM 的
实际层序里**不可达**：`model.py` 里每一个「有待注入 block output」的 combine 点，其 `injection`
都来自一个 `use_combine=True` 的 mixer（层内 attn/mlp 都是；最后一个 combine 的 injection 来自最后一层
的 mlp），因此永远非空；而层 0 的 attn 边界根本没有 pending combine（走 `mix()`，即本 kernel 的 mode 0）。
⇒ 若将来 vLLM 引入「unit-weight combine」路径，需要补第 4 档 mode（INJ≡1），接口上只是
`ProcessAic` 里少一个 N-tile、`S1` 里把 `injW[s]` 换成常量 1.0。

验收结果（Ascend950PR / CANN 9.1.0 / 28 AIC + 56 AIV / API 取核数）：
kernel 内 **93 条判定项全 PASS / 0 fails**（12 个 case × 3 档语义；另有 80 条报告项、155 条 guard，见 §7.1），
numpy 独立交叉校验 **12 case 共 204 条判定项 + 109 条 guard 全 PASS**；
`exact0` / `int` / `zerow` 三个档上 `H'`、`XN`、`lora`、`injection`、`ls`、`gate`、`blk` **逐位一致**。

---

## 1. 规格溯源与数学

### 1.1 源码位置与调用链

| 文件 | 作用 |
|---|---|
| `common/hyperconnection.py` | `GatedResidual.mix` / `.combine` 的精确数学（`GroupedGemmaRMSNorm`） |
| `nvidia/ops/hc.py`（+ `amd/ops/hc.py` 同义） | **实际运行时**的 triton 核（逐句依据：`_grouped_gemma_rmsnorm_kernel` / `_hc_silu_kernel` / `_hc_gate_mix_kernel` / `_hc_combine_norm_kernel`） |
| `nvidia/model.py` | 层序：`attn_hc.combine_and_mix → attn → mlp_hc.combine_and_mix → mlp`；末尾 `hyper_connection_mixer.combine_and_mix(use_combine=False)` |
| `nvidia/hyperconnection.py` | `GatedResidual.mix` / `combine_and_mix` / `combine` 的模块级接口 |

层内一次 hc 边界的 vLLM 语义（`model.py` 的 `Qwen4ExpDecoderLayer.forward`）：

```
hidden, block_input, injection = attn_hc.combine_and_mix(hidden, prev_block_output, prev_injection)
attn_out = attention(block_input)
hidden, mlp_block_input, injection = mlp_hc.combine_and_mix(hidden, attn_out, injection)
mlp_out  = mlp(mlp_block_input)          # mlp_out 作为下一个边界的 prev_block_output
return hidden, mlp_out, injection
```

即 **combine（把上一个 block 的输出注入 4 路残差流）+ mix（把 4 路压成一路 block input）** 在一个
边界内连续发生，这正是本 kernel 融合的对象。

### 1.2 checkpoint 张量形状（实测，`/workspace/Qwen3.8-Flash-Next-MXFP4`）

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `layers.L.attn_hyper_connection.hc_norm.weight` | `[10240]` | bf16 | = `hc_count × hidden_size` ⇒ `hc_per_branch_norm=True`（per-branch 仿射，宽 HYPER） |
| `layers.L.attn_hyper_connection.input_mix_weight_down.weight` | `[320, 10240]` | bf16 | 低秩 down（`hc_lowrank=320`），一行 = 一个低秩分量 |
| `layers.L.attn_hyper_connection.input_mix_weight_up.weight` | `[10240, 320]` | bf16 | 低秩 up（把 320 维门控展开回 4×2560） |
| `layers.L.attn_hyper_connection.block_inject_weight.weight` | `[4, 10240]` | bf16 | 每个残差流一个注入 logit |
| `layers.L.mlp_hyper_connection.{同上四件}` | 同形 | bf16 | MLP 边界一份 |
| `hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` | `[10240] / [320,10240] / [10240,320]` | bf16 | 全局 mixer：**没有 `block_inject_weight`** ⇒ `use_combine=False`（与 `nvidia/model.py` 一致） |

`text_config`：`hc_count=4`、`hc_lowrank=320`、`hidden_size=2560`、`rms_norm_eps=1e-6`、`num_hidden_layers=48`、
`layer_types` = 3×`linear_attention` + 1×`full_attention` 循环。**注意：每层权重实测独立**
（`layers.L.*` 各一份，不是共享），所以本 kernel 的权重以「边界」为参数传入。

### 1.3 逐句伪码（本 kernel 与 `check_ref.py` 都按此实现）

设 `HC=4`、`HID=2560`、`HYPER=HC*HID=10240`、`R=320`、`EPS=1e-6`；
输入 `H[m,HYPER]`（4 路残差流，**HC 外层、HID 内层**）、`BO[m,HID]`（pending block output）、
`IJ[m,HC]`（pending injection logits）。`bf16(·)` = 舍入到 bf16 网格（RNE）。

```
# W0 —— 注入权重（nvidia/ops/hc.py: _hc_combine* 里的 2*sigmoid(inj/HC)）
for (mi, s):  injW[mi][s] = 2 · sigmoid( IJ[mi][s] / HC )          # fp32

# S1 —— combine（model.py: hc_combine；把上一个 block 的输出按流注入）
for (mi, d):  s = d / HID ; j = d % HID
              H'[mi][d] = bf16( H[mi][d] + BO[mi][j] · injW[mi][s] )   # mode 0 跳过 → H' = H

# S2 —— grouped GemmaRMSNorm（common/hyperconnection.py: GroupedGemmaRMSNorm，group=HID）
for (mi, s):
    var  = Σ_j H'[mi][s·HID+j]² / HID                                  # 先在 bf16 网格上取值再平方
    rrms = 1 / sqrt(var + EPS)
    t    = H'[mi][s·HID+j] · rrms
    XN[mi][s·HID+j] = bf16( t + t · hc_norm[s·HID+j] )                 # Gemma 的 (1+w) 写成 t + t·w

# S3 —— down + inject（nvidia/hyperconnection.py 把两支并成一个 MergedColumnParallelLinear）
for (mi, n) where n ∈ [0, R+4):
    W = (n < R) ? Wdown[n] : Winj[n-R]
    OH[mi][n] = bf16( Σ_k XN[mi][k] · W[k] )                          # K = 10240
    # n < 320    → lora（喂 S4）
    # n ∈ [320,324) → injection logits（本边界的输出之一）

# S4 —— silu（nvidia/ops/hc.py: _hc_silu_kernel）
for (mi, r): u = OH[mi][r] / HC ;  LS[mi][r] = bf16( u · sigmoid(u) )

# S5 —— up（GATED 展开回 4 路）
for (mi, n) where n ∈ [0, HYPER):  GATE[mi][n] = bf16( Σ_k LS[mi][k] · Wup[n][k] )   # K = 320

# S6 —— gated mean（nvidia/ops/hc.py: _hc_gate_mix_kernel）
for (mi, j):
    acc = 0
    for s in 0..HC-1:  acc += sigmoid( GATE[mi][s·HID+j] ) · XN[mi][s·HID+j]   # 顺序累加
    BLK[mi][j] = bf16( acc / HC )
```

四个「逐句对齐」的细节（都取自 `nvidia/ops/hc.py`，不是我们自己发明的）：

1. **Gemma 仿射写成 `y + y·w`**（`_grouped_gemma_rmsnorm_kernel` 的 `y = x*rrms; y += y*w`），
   而不是 `x*rrms*(1+w)`——两者差一次 fp32 舍入，本 kernel 按前者（= 实际运行路径）。
   （注：`docs/14` §3.2 的伪码写的是数学形式 `xg * rrms * (1 + W_norm)`；两者同值、舍入点相差一次
   fp32 舍入。本 kernel 以**运行时 triton 核**为准，因为验收要和设备实际行为对齐。）
2. **RMS 在 bf16 舍入后的值上算**（`_hc_combine_norm_kernel` 先 `out = (res + block·inj).to(bf16)`
   再 `out*out` 求和），S2 因此消费的是 S1 落盘后的 bf16。
3. **gate mix 的累加顺序 s=0..3 然后统一 `/HC`**（`_hc_gate_mix_kernel` 的 `acc += sigmoid(g)*x` 循环
   → `acc /= HC`）。
4. **`2·sigmoid(·/HC)` 的 2.0 与 /HC 是 vLLM 独有的重缩放**，不许省。

### 1.4 attn / mlp / global 三处的差异

| | attn（层内） | mlp（层内） | global mixer |
|---|---|---|---|
| 数学 | 与 §1.3 完全相同 | 同左 | 同左，但 **S3 只取 lora（R=320，2 个 N-tile），不产出 injection** |
| 权重 | `layers.L.attn_hyper_connection.*` | `layers.L.mlp_hyper_connection.*` | `hyper_connection_mixer.*`（无 `block_inject_weight`） |
| 调用 | `combine_and_mix`（use_combine=True） | 同左 | `combine_and_mix`（use_combine=False） |
| 在层序中 | 消费上层的 `(mlp_out, injection)` | 消费 attn_out + attn 的 injection | 消费最后一层 mlp 的 `(mlp_out, injection)`，产出 `sample_hidden_states` |

三处在数学上**只有「是否产出 injection」这一个差别**，因此本 kernel 用 runtime `mode` 覆盖，
不做三份实现（`mode` 是统一分支，不产生设备端发散）。

---

## 2. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m20_hyperconn/build -S m20_hyperconn -DCMAKE_BUILD_TYPE=Release
cmake --build m20_hyperconn/build -j2                       # 单卡多 worker 共用，并行度别拉满

# 全档验收（12 个 case × 3 档语义；打印「判定项 / 报告项 / guard」三类计数）
./m20_hyperconn/build/m20_hyperconn

# 落盘全部中间张量后用 numpy 独立交叉校验
mkdir -p /tmp/m20_dump && cd /tmp/m20_dump
M20_DUMP=1 M20_CASE=real_m1 <repo>/m20_hyperconn/build/m20_hyperconn
/usr/local/python3.12.13/bin/python3 <repo>/m20_hyperconn/check_ref.py real_m1

# bring-up 定位用（段序截断 1..7；case 过滤）
M20_STAGE_LIMIT=3 M20_CASE=exact0 ./m20_hyperconn/build/m20_hyperconn
```

环境变量：`M20_DUMP=1`（落盘）、`M20_CASE=<name>`（只跑该 case）、`M20_STAGE_LIMIT=1..7`（段序截断）。

> ⚠️ `M20_CASE=<单个 case>` 是 **bring-up 定位用法**，**不能当独立 PASS 命令**：全局判定项
> 「至少一个 case 上建立中间舍入判别性」只有在跑**多个** case 时才可能成立（单跑 `exact0` 这类
> 整数档会因为两条参考无差异而报 `未能建立判别性 FAIL` ⇒ rc=1）。要判定 PASS 请跑全档（不带 `M20_CASE`）。
独立 CMake 工程（不依赖仓库顶层 `CMakeLists.txt`），`--npu-arch=dav-3510` + `-ffp-contract=off`
（后者保证 host C 参考与 kernel 内同一 IEEE 运算序列，`H'` 才能做逐位判据）。

---

## 3. 全局资源表（`m20_resources.h`，对应 docs/05 §5.2）

| 资源 | 落地 |
|---|---|
| AIC BufferID | 0-1 A ping/pong、2-3 B ping/pong、4-5 L0A/L0B ping/pong、6 L0C（与 m1/m11/m14 同编号） |
| AIV BufferID | 0-3 S1（h/bo/out 槽 + **3 声明未用，见下**）、4-5 W0（IJ staging / 表落盘）、6-7+15 S2（H' / XN / hc_norm 组）、**8 声明未用**、9-10 S4、11-14 S6（**13 声明未用**）、16 rstd |
| AIC flagId | 4-7：`FLAG_AC0..AC3`（两段 GEMM 的前对齐 / 后 drain 对齐） |
| AIV flagId | 0-2：`FLAG_AV0..AV2`（S1→S2、S2→AIC、S4→AIC 段内 barrier，set 挂 MTE3） |
| mode-2 flagId | 8-11：`FLAG_A2C_XN`、`FLAG_C2A_OH`、`FLAG_A2C_LS`、`FLAG_C2A_GATE`（一次性分配，不复用） |
| UB | 248KB 可用。**声明足迹 89472 B**，但其中 **61440 B 是「声明窗、非实际占用」**：`[UB_PERSIST, UB_PC_END)`（hc_norm fp32 预转 40960 + 预转 staging 20480）在 `m20_resources.h` 里声明、**kernel 内 0 处引用**（窗 W0 的**声明基准**是 `UB_PC_END`，所以它只是地址空间的保留）。**窗口径工作集 = 89472 − 61440 = 28032 B**（窗 W0..W5，`[61440, 89472)`）。**不使用 TPipe/TBuf/TQue/AllocTensor**，全部编译期偏移 + 32B 对齐 `static_assert`。**M69 再细一层**：那 28032 B 是**窗跨度**，其中旧 W0 staging `[61440, 61728)`（`UB_PW_IJ`/`UB_PW_SIG` = 288 B）与窗内三个无引用槽（`UB_S1_IW` 32 + `UB_S4_ONE` 32 + `UB_S6_AB` 256 = 320 B）同样不被访问 ⇒ **kernel 实际建 `LocalTensor` 覆盖 27424 B**。三个数都只由 `m20_resources.h` §2 顶部的常量算出，§2 已逐段标「⚠ 未使用」 |
| 资源声明自查（M56，随 `c4b7b0e`/`26a3ae4`；**M69 订正分类**） | 逐符号核 `m20_resources.h` 的声明是否真被 kernel 用到。工具 `tools/check_symbol_claims.py`（并排给出**声明处注释原文** + 两栏引用计数；读数归档 `tools/evidence/symbol_claims_check.log` —— **该归档已随 M71 重生成**（工具 `f0cbc7d`）：读数 A/B/D 的源树是 M69 之前（`cecf0ae`/`1f316b6`）、读数 C 的源树是 M69 树（`1c7b36d`），引用其「声明处注释原文」栏前先看该读数标注的**源树 commit**）。**读数与复算命令见表后**：「窗/缓冲/槽/常量」类里 `.asc` 内 0 引用的有 **17 个**（见下表）；M56 当时写「**12 个**」，漏了 `UB_PW_IJ`/`UB_PW_SIG`/`UB_S1_IW`/`UB_S4_ONE`/`UB_S6_AB` 这 5 个**槽**——它们被归进了「窗边界/尺寸类」，但它们给的是 slot 基准而不是边界；M69 按**匹配片段**逐条重数后改正。工具口径死声明 **10 个**（M56 时点基线 `57f896c` 与本轮 tip 上同数；多出来的 1 个是 `.asc` 里的 `struct Judge` 成员 `pass`，不在下表 —— 下表只列 `m20_resources.h` 的符号；复算见下表后 ③④）。**「用途」一栏逐字抄自声明处注释，不自造描述。** |
| 资源声明清理（M69） | 按 tower 裁定「**注释优先、不移动活地址**」处置上表的死声明：**声明一个都没删、值一个都没改** —— 删那 5 个 `UB_*` 就得把窗 W0 的基准从 `UB_PC_END` 改成 0 ⇒ W0 之后所有窗的偏移整体 −61440，会动一个**已验收 kernel 的静态布局**，而收益只是可读性（finding 自己写了「对正确性/性能无影响」）。做法：①每个死声明处标「⚠ 未使用」+ 缘由；②在 `m20_resources.h` 加 §0.0「每个声明至少被 kernel 引用一次」的可复算自查命令；③§2 顶部列全区间口径。**活地址逐个未变的证据见下（复算块，本轮实跑）** |
| L1 | A 区 @0（8KB ×2 ping/pong）、B 区 @262144（20KB ×2）；总占用 303104 B ≤ 512KB |
| L0 | L0A/L0B 各 32KB ping/pong；L0C（CO1）64×160 fp32 = 40KB |
| GM workspace | 8 个中间张量（`H', XN, RSTD, INJW, OH, LS, GATE, BLK`），偏移/尺寸全编译期常量，总量 4353024 B（`m20_resources.h` §4 每个张量带生命周期注释） |
| tile | `BASE_M=64`（m∈[1,64] 单 tile，尾块 curM mask）、`BASE_K=64`、`BASE_N=160` |

核数**一律用 API 取**：host `aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM)` 拿 AIC 数（实测 28）并用它作
`numBlocks`，同时校验 `ACL_DEV_ATTR_VECTOR_CORE_NUM == 2×AIC`；kernel 内 AIV 侧
`GetBlockNum()*2`。**无任何硬编码核数**。

**逐符号核（M56，随 `c4b7b0e`/`26a3ae4`；M69 订正 (a) 的分类、把 (b) 降为历史读数）**：两栏口径不同，分开写清 ——
- **(a) `asc_refs`** = 该符号在 `m20_hyperconn.asc` 内的引用数（**kernel 是否真用到**）。**M69 起本文件以这栏为准**，且计数按**匹配片段**（字面命令：`grep -o "\b<符号>\b" m20_hyperconn.asc | wc -l`，见下面的复算 ②）而不是按行（`grep -c`）——就本表这些符号而言两者同为 0，但按行计数会被「同一行出现两次」掩盖。
- **(b) `decl_refs`** = 该符号在 `tools/check_symbol_claims.py` 的 `--code` 范围内**除自己声明行外**的**代码** token 出现次数（工具判「死声明」的口径：`= 0` 才是死声明）。**M71（`f0cbc7d`）起该工具按字符跨度剥掉注释、只数代码引用，注释里的提及单列且不算引用**；此前那版逐行逐 token 计数、**不区分注释与代码**（下表 (b) 栏就是那版在 **M56 时点**的读数）。

下表「声明处注释原文」栏给出**当前（M69 后）**的注释；M56 那版逐字抄的是**清理前**的注释，故在括号里并排保留以便与 M56 归档对读（`cecf0ae` 那次张冠李戴的教训见 `tools/evidence/symbol_claims_check.log` 的复现）。

**读数与复算**（(a) 当场计数；(b) 为 M56 时点读数，现行工具口径的读数见 ③④）：
- `.asc` 内 0 引用的**资源表符号共 36 个**；其中**多数是窗边界/尺寸类常量**（`UB_W0_END`…`UB_W5_END`、`SZ_*`、`WS_ALIGN`、`WS_AlignUp`、`L1_BYTES_TOTAL`、`L0_PP_BYTES`、`UB_BYTES_TOTAL` —— 它们的用途就是在 header 内算出下一个偏移），其中 **17 个**是"窗/缓冲/槽/常量"类 ⇒ 即下表前 17 行（**M56 当时写 12 行**，漏了 5 个槽基准）。复算（两条给出同一读数 36；**口径先说清**：`m20_resources.h` 共 **134 个 `constexpr` 常量 + 1 个 `constexpr` 函数** `WS_AlignUp` ⇒ 下面两条命令各**按名字遍历 135 个**、其中 36 个在 `.asc` 内 0 引用 —— 措辞与命令指向同一个集合）：
  ```bash
  cd m20_hyperconn
  # ① M56 那版（按行计数；含 constexpr 函数 WS_AlignUp）
  for s in $(grep -oP 'constexpr\s+\w+\s+\K\w+' m20_resources.h); do
    [ "$(grep -c "\b$s\b" m20_hyperconn.asc)" = 0 ] && echo "$s"
  done | wc -l                                  # 36
  # ② M69 起用这版（按匹配片段；含 WS_AlignUp，过滤后 35）
  for s in $(grep -oP 'constexpr\s+\w+\s+\K\w+' m20_resources.h); do
    printf '%-18s %s\n' "$s" "$(grep -o "\b$s\b" m20_hyperconn.asc | wc -l)"
  done | awk '$2 == 0' | wc -l                  # 36
  cd ..                                          # ③④ 在仓根跑（工具按 --root 解析路径，--root 默认 = 本仓）
  # ③ (b) 口径的现行读数（工具 = M71 `f0cbc7d` 起：按字符跨度剥注释后只数代码引用）
  python3 tools/check_symbol_claims.py --code m20_hyperconn --docs m20_hyperconn/README.md | grep 死声明
  #   [coverage] 声明 280 个（来源：m20_hyperconn）；**死声明（剥注释后代码引用 0）10 个**（其中在注释里被点过名的 8 个 —— **注释提及不算引用**，故它们仍是死声明）
  #   [死声明] BUF_AIV_AB, BUF_AIV_GJ, BUF_AIV_IW, DN_KLOOP, MODE_COUNT, NGROUPS, UP_KLOOP, V_LENGTH, gScale, pass
  # ④ 同一工具换到 M56 时点基线（--docs 也随 --root 解析到基线那棵树）
  mkdir -p /tmp/m20_57f896c && git archive 57f896c m20_hyperconn | tar -x -C /tmp/m20_57f896c
  python3 tools/check_symbol_claims.py --root /tmp/m20_57f896c --code m20_hyperconn --docs m20_hyperconn/README.md | grep 死声明
  #   [coverage] 声明 280 个（来源：m20_hyperconn）；**死声明（剥注释后代码引用 0）10 个**（其中在注释里被点过名的 1 个 —— **注释提及不算引用**，故它们仍是死声明）
  #   [死声明] BUF_AIV_AB, BUF_AIV_GJ, BUF_AIV_IW, DN_KLOOP, MODE_COUNT, NGROUPS, UP_KLOOP, V_LENGTH, gScale, pass
  ```
- 按 (b) 口径**（M56 时点，基线 `57f896c`；现行工具 `f0cbc7d`）死声明 10 个** = 下表里 `decl_refs`(M56) = 0 的 **9 行**（`BUF_AIV_IW/GJ/AB`、`NGROUPS/MODE_COUNT/DN_KLOOP/UP_KLOOP`、`V_LENGTH`、`gScale`）+ `.asc` 里的 `struct Judge` 成员 `pass`（1 个，不在下表 —— 下表只列 `m20_resources.h` 的符号）。
- ⚠ **口径 (b) 的已知脆弱性（M69 实测 + M71 口径修正，写下来免得下一个人误用）**：M69 给这些死声明加了「⚠ 未使用」注释（注释里点了它们的名）后，**当时那版工具**（逐行逐 token、**含注释**）在**同一份工作树**上把 (b) 死声明从 **9 个算成 2 个**（`V_LENGTH`、`gScale`，都在 kernel 内）—— 因为注释里的提及也被算成引用（这正是 M71 `f0cbc7d` 改口径的起因：`⚠ 未使用` 注释会把死声明从判据里抹掉）。**M71 起按字符跨度剥掉注释、只数代码引用**：同一份工作树与基线 `57f896c` 都报 **10 个**，不再随加注释而变。⇒ **(b) 仍不能用来判「kernel 用没用」**：它数的是 `--code` 范围内**全部代码 token**，`m20_resources.h` 自己的偏移链（`A = B + 1`）也算引用（见下表 `UB_PERSIST..UB_PC_END` 的 (b) ≥ 1），本文件此后以 (a) 为准。

| 符号 | (a) `asc_refs` | (b) `decl_refs`（M56 时点） | 声明处注释 —— **当前**（M69 后；行号仅查阅提示）／**清理前**（M56 时点，锚在 `57f896c`） |
|---|---|---|---|
| `UB_PERSIST` | 0 | 1 | `⚠ 未使用（保留窗起点）`（当前 :179）／`PERSIST：hc_norm 的 fp32 预转表（每 AIV 一次，整核生命周期）`（清理前 :134 整行注释 / 声明在 :135） |
| `UB_NORMF32` | 0 | 1 | `⚠ 未使用（声明 40960 B：bf16→fp32 [HYPER]）`（当前 :180）／`bf16→fp32 [HYPER] = 40960 B`（清理前 :136） |
| `UB_PERSIST_END` | 0 | 2 | `⚠ 未使用（= 40960）`（当前 :181）／`40960`（清理前 :137） |
| `UB_PC_STAGE` | 0 | 1 | `⚠ 未使用（声明 20480 B：bf16 [HYPER] staging）`（当前 :184）／`预转 staging（整核只用一次，落盘 hc_norm bf16 原始行）`（清理前 :140 整行注释 / 声明在 :141） |
| `UB_PC_END` | 0 | 1 | `= 61440：保留窗末端 = 活区起点（窗 W0 的唯一基准）`（当前 :185）／`61440`（清理前 :142） |
| `UB_PW_IJ` | 0 | 1 | `⚠ 未使用（旧 staging：16 bf16 = 32 B）`（当前 :191）〔M56 未列，本轮按 (a) 补入〕 |
| `UB_PW_SIG` | 0 | 1 | `⚠ 未使用（旧 staging：64 fp32 = 256 B）`（当前 :192）〔同上〕 |
| `UB_S1_IW` | 0 | 1 | `⚠ 未使用（32 B）：已废弃路径的 injW BRC 单值槽`（当前 :199）〔同上〕 |
| `UB_S4_ONE` | 0 | 1 | `⚠ 未使用（32 B）：1.0 在 __VEC_SCOPE__ 内 Duplicate`（当前 :212）〔同上〕 |
| `UB_S6_AB` | 0 | 1 | `⚠ 未使用（256 B）：累加器是 V 内私有 RegTensor`（当前 :218）〔同上〕 |
| `BUF_AIV_IW` | 0 | **0** | `⚠ 未使用：README §4.1(b) 那条已废弃路径（S1 从 GM 取 injW 再 BRC）的 id`（当前 :254）／`S1: injW 32B MTE2 -> V`（清理前 :202） |
| `BUF_AIV_GJ` | 0 | **0** | `⚠ 未使用：那版未实现的「hc_norm 整核预转」的 id`（当前 :260）／`预转: hc_norm bf16 行 MTE2 -> V（整核一次）`（清理前 :208） |
| `BUF_AIV_AB` | 0 | **0** | `⚠ 未使用：S6 累加器是 V 内私有 RegTensor、无跨 pipe 交接 ⇒ 本就不需 BufferID`（当前 :265）／`S6: 累加器 V 内私有（无跨 pipe 交接）`（清理前 :213） |
| `NGROUPS` | 0 | **0** | `每个 token 的 norm 组数 = hc_count`（当前 :93，M69 未改该行注释；清理前 :63） |
| `MODE_COUNT` | 0 | **0** | （该行无注释）（当前 :108；清理前 :78） |
| `DN_KLOOP` | 0 | **0** | `160`（当前 :118；清理前 :88） |
| `UP_KLOOP` | 0 | **0** | `5`（当前 :121；清理前 :91） |
| `V_LENGTH`（kernel 内） | (不适用) | **0** | （无注释）（`m20_hyperconn.asc:145`，该文件本轮未改） |
| `gScale`（kernel 内） | (不适用) | **0** | （无注释）（`m20_hyperconn.asc:1420`，同上） |

> **行号口径（按 `docs/05` §6.1 的「引用约定」）**：上表两栏行号**都只是查阅提示、随代码变动**，**定位一律以符号名（+「声明处注释」原文）为准**；两栏的**锚**不同 —— 「**当前**」栏 = 本轮 tip 上 `m20_resources.h` 的声明行；「**清理前**」栏 = M56 时点的行号，锚在基线 `57f896c`，是历史事实、不再变。
> **这一列漂过**：`m20_resources.h` 每次改注释都会让**其后的所有行号**平移（`//` 注释也算行），而「当前」栏不会自动跟着变。实测的两次平移：commit `1c7b36d` 往 §0.0 加了 2 行正对照 ⇒ **+2**；本轮修 §0.0 的措辞与口径又各加 1 行 ⇒ **再 +2**。故现值 = `fb2555b` 那版的原始值**逐个 +4**（`git show fb2555b:m20_hyperconn/m20_resources.h` 里 `UB_PERSIST` 在 175，现在是 **179**）。**下面这条命令可当场重取这一栏** —— 下次谁再动 `m20_resources.h`，跑它就能立刻看出漂没漂：

```bash
cd m20_hyperconn
grep -nE '^constexpr [a-z0-9_]+ (UB_PERSIST|UB_NORMF32|UB_PERSIST_END|UB_PC_STAGE|UB_PC_END|UB_PW_IJ|UB_PW_SIG|UB_S1_IW|UB_S4_ONE|UB_S6_AB|BUF_AIV_IW|BUF_AIV_GJ|BUF_AIV_AB|NGROUPS|MODE_COUNT|DN_KLOOP|UP_KLOOP) ' m20_resources.h
# 实测（本轮修正后的树，共 17 行；与上表「当前」栏逐行一致）：
#   93:NGROUPS  108:MODE_COUNT  118:DN_KLOOP  121:UP_KLOOP
#   179:UB_PERSIST  180:UB_NORMF32  181:UB_PERSIST_END  184:UB_PC_STAGE  185:UB_PC_END
#   191:UB_PW_IJ  192:UB_PW_SIG  199:UB_S1_IW  212:UB_S4_ONE  218:UB_S6_AB
#   254:BUF_AIV_IW  260:BUF_AIV_GJ  265:BUF_AIV_AB
```

读数小结（**两栏来源不同**：(a) 由上面的 one-liner 当场取；(b) 由 `tools/check_symbol_claims.py` 的 `collect_decls()` 产出 —— 下表的 (b) 栏写的是 **M56 时点那版工具口径**（逐 token 含注释）的值；该归档已随 M71 重生成、旧读数已删，要复现 M56 那版读数得走 `git show`）：
- **前 17 行在 `.asc` 内 0 引用** ⇒ kernel 完全没用到它们（这是 §3 UB 行「声明窗、非实际占用」与「27424 B 实际引用」的依据）。
- 按口径 (b)（M56 时点、基线 `57f896c`），**死声明 10 个** = 上面那 9 个 + `.asc` 里的 `struct Judge` 成员 `pass`（不在下表 —— 下表只列 `m20_resources.h` 的符号）；现行工具（M71 `f0cbc7d` 起的口径）在本轮 tip 上同样是 **10 个**，不再有「加注释就掉到 2 个」的现象（原因见上面的脆弱性一条）。
- **那 5 个 `UB_PERSIST..UB_PC_END` 不是 (b) 意义下的死声明**：它们被 `m20_resources.h` 自己的偏移链互相引用
  （`UB_NORMF32 = UB_PERSIST`、`UB_PERSIST_END = UB_NORMF32 + HYPER*4`、`UB_PC_END = UB_PC_STAGE + HYPER*2`）
  ⇒ `decl_refs` ≥ 1。**但 kernel 侧 0 引用**，所以它们仍是「占地址空间、不占工作集」的声明窗。
  同理，本轮按 (a) 补入的 5 个槽基准（`UB_PW_IJ`/`UB_PW_SIG`/`UB_S1_IW`/`UB_S4_ONE`/`UB_S6_AB`）各自被**下一个偏移**引用（如 `UB_W0_END = UB_PW_SIG + 256`），(b) 也是 1。
- **M69 的处置**：这些死声明**一个都没删、值一个都没改**（只加「⚠ 未使用」注释 + `m20_resources.h` §0.0 的自查命令 + §2 的区间口径）。此前那句「不在本 mission 的写权限内、已 file TowerFinding 报路由」已由 M69 兑现；finding 原文 = `.tower/comms/findings/20260926-agent-m56audit-improve-m20-resources-h-61440-b-ub-3-bufferid-kernel-0-readme.md`。该 finding 的「建议删除五个 `UB_*` 声明并把窗 W0 基准改成 0」**未采纳**，理由见 §3 上表的「资源声明清理（M69）」行。

**活地址未变的复算（M69，本轮实跑；基线 = 父 commit `57f896c`）**：

```bash
cd <repo>
git show 57f896c:m20_hyperconn/m20_resources.h | sed -E 's@//.*@@; s@[[:space:]]+$@@' \
  | grep -v '^[[:space:]]*$' > /tmp/m20_code_before.h
sed -E 's@//.*@@; s@[[:space:]]+$@@' m20_hyperconn/m20_resources.h \
  | grep -v '^[[:space:]]*$' > /tmp/m20_code_after.h
diff /tmp/m20_code_before.h /tmp/m20_code_after.h     # 期望 0 行 ⇒ 全部改动都在 // 注释里
```

实测：`diff` rc=**0**，两侧各 **184** 行（= 剥掉注释与空行后的全部代码行）。端到端见证（三条，本轮都实跑过）：
1. 用改后的 `m20_resources.h` 重建、在空目录重跑全档验收，`diff m20_hyperconn/evidence/accept_run.log <本轮输出>` = **0 行**
   （归档日志逐字节复现，含 `[M20] WS_BYTES=4353024  UB_USED=89472  L1_USED=303104` 那一行 —— 这也是**不把
   `UB_BYTES_USED` 改写成 28032** 的原因之一：改了它，这条打印值就与归档日志不再一致）。
2. `check_ref.py` 12 case 全跑（每个 case 先 `M20_DUMP=1 M20_CASE=<c>` 落盘）：12 个 case 全部
   `RESULT: OK`、`checkref rc=0`，合计 **判定项 204 + guard 109、SKIPPED 9** —— 与 `evidence/check_ref_run.log`
   的逐 case 分布**逐条相同**：对两份日志各跑同一条命令（本轮实跑，输出逐行相同）
   `grep -o 'RESULT: [A-Z]* ([0-9]*/[0-9]* 条比较过：判定项 [0-9]* + guard [0-9]*；SKIPPED [0-9]*)' <日志> | sort | uniq -c`
   ⇒ `2 × (21/21 判 14 + guard 7, SKIP 2)`、`1 × (25/25 18+7, SKIP 3)`、`2 × (26/26 16+10, SKIP 0)`、
   `2 × (27/27 18+9, SKIP 1)`、`5 × (28/28 18+10, SKIP 0)`。注意：单跑 `M20_CASE=<单个 case>` 时 **4 个 case 的 dump 进程 rc=1**
   （`exact0`/`exact0m33`/`int_mix`/`real_mix`，报「中间 bf16 舍入判别…未能建立判别性 FAIL」）—— 这正是 §2 那条
   ⚠ 说明的已知行为（单 case 建不起判别性），不是 kernel 失败；dump 文件照常落盘，`check_ref.py` 随后全 PASS。
3. **正对照（先做，免得把「恒 0 的哑扫描」当成发现）**：上面那条逐符号计数命令对活符号会报非 0 ——
   `UB_S2_WB` **1**、`BUF_AIV_IJ` **4**、`UB_IW_TAB` **2**（凭空名字 `UB_NOT_EXIST` 报 **0**）；
   注释剥离那条同法做了正对照：把 `UB_S1_IW = UB_S1_OB + 128` 改成 `+ 129` 后 `diff` rc=**1**
   （**咬住了真改动**），还原后 rc=**0**。

---

## 4. 同步表

段边界跨核一律走 docs/12 §5 / docs/05 §6 的**标准 mix 序列**（本工程实测通过）：

| 方向 | 序列 |
|---|---|
| AIV 段内（S1→S2、S2 后、S4 后） | `BarrierAiv<WAIT_PIPE, FLAG>`：全体 AIV `set mode0(PIPE_MTE3)` + `wait mode0(PIPE_MTE2)`（两 pipe 均 drain；flagId 0/1/2） |
| AIC 段内（GEMM 前后） | `BarrierAic<PIPE_S, {PIPE_MTE2｜PIPE_FIX}, FLAG>`：全体 AIC 对齐；**GEMM 后用 `PIPE_FIX` 作 set pipe**（FIXP 写 GM 排空） |
| AIV → AIC（XN / LS 就绪） | AIV 先 `BarrierAiv`（全体到齐）→ 每个 AIV `set mode2(PIPE_MTE3)` → 每个 AIC `wait mode2(PIPE_S)`（「wait 后紧接 set」→ 挂 PIPE_S） |
| AIC → AIV（OH / GATE 就绪） | AIC 先 `BarrierAic<..., PIPE_FIX>`（FIXP 排空 + 全体到齐）→ 每个 AIC `set mode2(PIPE_MTE2)` → 每个 AIV `wait mode2(PIPE_MTE2)` |

**同步纪律自查**（`grep` 可逐条复核）：

* `CrossCoreSetFlag` 的 pipe 与核型严格匹配——AIV 侧只出现 `PIPE_MTE3`；AIC 侧只出现 `PIPE_MTE2` / `PIPE_FIX`
  （docs/05 §6.2：AIC 上写 MTE3 是静默空操作 → 对侧永久挂死；**AIC 写 GM 走 FIXP，不可能用 MTE3**）。
* 除跨核 wait + BufferID 标量握手外**不挂 PIPE_S**；`wait 后紧接 set` 的点（AIC 的两个 mode2 wait）挂 PIPE_S。
* 相邻同步点 flagId 必不同（本工程 0-11 一次性分配，无复用）。
* 核内用 BufferID（`GetBufInternal` / `RlsBufInternal<pipe,false>` 阻塞释放（`false` = CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地））；
  **同 pipe 复用同一 UB 行缓冲的段循环入口补 `PipeBarrier<PIPE_MTE2>`**（docs/05 §6.2 第 1 条）。
* **不使用** `set_flag`/`wait_flag` 系列、`SyncAll`、`Matmul` 高阶 API、`RemoteFetch` 类资源管理函数。

### 4.1 bring-up 期间定位的两处写法问题（**一处已定性为「误用」，一处尚未隔离**）

按 tower 的裁定（`.tower/comms/inbox/20260926-tower-agent-hcmix-m36-a-b-m45-repro.md`）：
本仓已有**四条**「文档/硬件不一致」结论最终全部以「我们选错用法」收场，因此这两条**先隔离、再谈入库**，
并且不得把「尚未隔离的观测」写成平台行为。

#### (a) `StoreDist::DIST_FIRST_ELEMENT_B32` 只写元素 0 —— **误用，不是平台差异**

* **文档说**：CANN 头文件里 `StoreDist` 枚举（`kernel_reg_compute_utils.h`，`DIST_FIRST_ELEMENT_B32`
  与 `DIST_NORM_*` / `DIST_PACK_*` 并列）**只有名字、没有行为注释**——即**名字就是语义**：「落**第一个元素**」。
  （行号随代码变动，以符号 `StoreDist::DIST_FIRST_ELEMENT_B32` 为准。）
  旁证是既有算件的用法一致：`m6_rmsnorm.asc` 的 `StoreAlign<…, DIST_FIRST_ELEMENT_B32>` 共 **6 处**——
  `CalculateSquareReduceSumLessThanVL` ×1、`CalculateSquareReduceSumLessThanTwoVL` ×1、
  `CalculateSquareReduceSumCommon` ×4——都用
  「`Reduce<SUM>` 得标量 → `CreateMask<float, VL1>` → `DIST_FIRST_ELEMENT_B32`」，即它本来就
  只写元素 0、**不参与「按 lane 掩码选元素」**。（行号随代码变动，以符号为准；6 处的行号提示：
  `m6_rmsnorm.asc:165/:195/:234/:241/:251/:264`。）
* **（当时）实测**：本项目 W0 需要把 4 个流的注入权重分别落到 4 个 32B 槽首；用
  `Compares(cmpS, idx, s) → StoreAlign<DIST_FIRST_ELEMENT_B32>(slot_s, v, cmpS)` 时，
  **4 个槽全被写成 lane 0 的值**（与掩码无关）。`exact0` 档因四流值相等而未暴露，GEN_REAL 档才暴露。
* **结论**：**误用**（不是平台 quirk，不入库为硬件行为）。
  **规避（本项目已采用）**：`Compares(cmpS, idx, s) → Select(sel, v, zero, cmpS) → Reduce<SUM>(red, sel, maskAll)
  → StoreAlign<DIST_FIRST_ELEMENT_B32>(slot_s, red, maskVL1)`——先把要选的 lane 收归到 lane 0，再落单元素。

#### (b) 原 S1「从 GM 取 `injW[s]` 再 BRC 广播」非确定性 —— **尚未隔离，不作为平台行为引用**

* **现象（bring-up 期实测；该版代码因当场修掉而**未进任何 commit**）**：原 W0 把 `injW` 落到 **GM**
  （每 token **4B**，`DataCopyPad`，pipe = **MTE3**，BufferID = `BUF_AIV_SG`）；原 S1 按 item 从 **GM** 取回
  （`DataCopyPad` **4B**，也试过 **32B**，pipe = **MTE2**，BufferID = `BUF_AIV_IW`，写进 32B 对齐 UB 槽）
  再 `LoadDist::DIST_BRC_B32` 广播。160 个 chunk 里 **137 个退化为 `out = h`**（= 广播值读到 0），
  且**每次跑错的位置都不同**（非确定性）。
* **对照（同一份代码的三个变体）**：`Duplicate(1.0f)` 硬编码 → **160/160 正确**；
  普通全宽 `LoadAlign<bfloat16_t, DIST_UNPACK_B16>` 读 MTE2 刚搬进的槽 → **一直正确**；
  `DIST_BRC_B32` 读 **V 自己写出**的槽 → **正确**。
* **该路径同时含两个可疑因子，当时未做隔离**：
  1. **同核 `MTE3 → GM → MTE2` 的 GM 回环没有任何跨 pipe 排序原语**——生产者用 `BUF_AIV_SG`、
     消费者用 `BUF_AIV_IW`，**两个 BufferID**，而生产者的阻塞释放没有任何消费者去 acquire ⇒
     **GM 读可能先于 GM 写落盘**（这本身就是一处漏同步，不是硬件怪癖）；
  2. `DIST_BRC_B32` 与「MTE2 刚搬进的 UB 槽」之间的可见性。
* **我的修法（W0 用 V 写出全表 → S1 从 UB 的 32B 对齐槽 BRC）同时去掉了因子 1 与因子 2 ⇒ 无法区分**
  是哪一个（或两者共同）导致非确定性。**故本条状态 = 尚未隔离，不得当作平台行为引用**。
* 已按 tower 裁定向 **M45（`probe_vf_ldst/**`，agent-vecdist）** 提供最小 repro 的事实（见下"给 M45 的事实"）。
  **M45 的初测在「只做 MTE2→V 握手、完全不绕 GM」的形态下复现不出来**（23 变体 × 每个 ≥5 独立进程
  全部 64/64 正确），与该假说一致。

**给 M45 的事实（可交接）**

| 项 | 原实现 |
|---|---|
| 生产者 | `InjwStage`（每 token 一次）：VF 算出 4 个 `2·sigmoid(IJ/4)` → `StoreAlign<DIST_NORM_B32>(sigUb, v, mask4)` → `BufAcquire<PIPE_MTE3>(BUF_AIV_SG)` → `DataCopyPad(injwGm[mi*INJW_STRIDE], sigL, ExtBlock1(16))`（**每 token 16B / 4 个 fp32**）→ `BufRelease<PIPE_MTE3>(BUF_AIV_SG)`（阻塞释放） |
| 消费者 | `CombineStage`（每 item 一次，共 160 次）：`BufAcquire<PIPE_MTE2>(BUF_AIV_IW)` → `DataCopyPad(iwL, injwGm[mi*INJW_STRIDE + s], ExtBlock1(4), DataCopyPadExtParams<float>{false,0,0,0})`（43B 也试过 32B）→ `BufRelease<PIPE_MTE2>(BUF_AIV_IW)` → VF：`LoadAlign<float, DIST_BRC_B32>(w, iwUb)` |
| UB 槽 | `UB_S1_IW`，32B 对齐，32B 大小（`LocalTensor<float>(VECCALC, UB_S1_IW, 8)`） |
| GM 目标 | `WS_INJW + mi*32`（每 (token,stream) 一个 32B 槽） |
| 同步 | **两侧 BufferID 不同**（5 vs 3）；生产者阻塞释放、消费者立即 acquire；**两者之间无任何 CrossCore / PipeBarrier** |
| 观测 | 160 chunk 中 137 个 `out = h`（广播读到 0），**错的位置每次不同**；修法（W0 用 V 写全表、S1 从 UB 槽 BRC）后 160/160 稳定正确 |

---

## 5. 输入输出 layout 与权重契约

### 5.1 kernel 接口

```cpp
extern "C" __global__ __mix__(1, 2) void m20_hyperconn_kernel(
    __gm__ uint8_t* hIn,      // [M_MAX, HYPER] bf16   4 路残差流输入 H
    __gm__ uint8_t* bo,       // [M_MAX, HID]   bf16   pending block output（mode 0 不读）
    __gm__ uint8_t* ij,       // [M_MAX, 16]    bf16   pending injection logits（前 4 列有效；mode 0 不读）
    __gm__ uint8_t* wDown,    // [LOWRANK, HYPER] bf16 input_mix_weight_down（checkpoint 原样）
    __gm__ uint8_t* wInj,     // [>=16, HYPER]  bf16   block_inject_weight（见下方契约）
    __gm__ uint8_t* wUp,      // [UP_N, UP_K]   bf16   input_mix_weight_up（checkpoint 原样）
    __gm__ uint8_t* hcNorm,   // [HYPER]        bf16   hc_norm
    __gm__ uint8_t* ws,       // workspace（偏移见 m20_resources.h §4）
    uint32_t m,               // token 数 1..M_MAX(64)
    uint32_t mode,            // MODE_MIX / MODE_COMBINE_MIX / MODE_FINAL_MIX
    uint32_t stageLimit);     // 段序截断（1..7，7=全开）
```

**权重契约：直接消费 checkpoint 原始排布，不做任何离线转换**（`hf_to_vllm_mapper` 之外无重排）：
`wDown`/`wUp`/`hcNorm` 与 checkpoint 字节一一对应。

唯一例外（与 m11「m=1 提升为 2 行、多读的行由 host 保证可读」同款契约；**本契约正文即下面这一段，
§9 不含此内容**——先前写的「已在 README §9 披露」是指错了位置，此处更正）：
**`wInj` 必须 ≥16 行可读，且行 `[4,16)` 须为 0**。原因：`block_inject_weight` 只有 4 行，而 cube 的
N 分形/`Nd2Nz` 以 16 行为单位；最后一个 N-tile 读 16 行则 L0C 的列 `[4,16)` 由 `0·x` **精确为 0**，
`OH` 的 padding 列 `[324,336)` 恒为 0（实测：`pad cols all zero = True`）。若只读 4 行，列 4..15 会取自
L1 残留（有限但随调度变化），虽然不影响任何有效列，但会破坏重复运行的逐字节一致性。
checkpoint 的前 4 行按原样读取。

**另一个同类问题（`54e5171` 修）**：workspace 的 `injw` 张量非确定。根因是 W0 的 **UB 全表**
每个 32B 槽只写 4B（`DIST_FIRST_ELEMENT_B32` 只落槽首），槽的 `[4,32)` 保留 UB 残留 ⇒
该张量含未初始化字节 ⇒ 同一二进制重复运行哈希不同、归档 manifest 无法复现。
修法：**在写槽首之前把整表按槽清零**（`VL8` 掩码 × `INJW_SLOT` 步长，256 个槽全写 32B）。
> 注意首版修法本身有 bug：掩码 `VL8` 每次只写 **8** 个 fp32，但循环步长误取 `VL_F32=64`
> ⇒ 只覆盖了槽 0、8、16…共 32/256 个槽，实测表现正是「槽 0 确定、其余槽的尾部非确定」。
> 步长改为 `INJW_SLOT` 后，256 个槽全部由本 kernel 写出，确定性成立（见 §7.7 的 ≥5 次复核）。

### 5.2 GM 输出 layout

| workspace 张量 | 物理行距（元素） | 语义 |
|---|---|---|
| `WS_HCP`（H'） | `HYPER=10240` | 更新后的 4 路残差流（mode 1/2 有效；mode 0 不写） |
| `WS_XN` | `HYPER` | grouped GemmaRMSNorm 输出（总输入 S3 与 S6） |
| `WS_RSTD` | 1（按 `(token,stream)` 展平） | fp32 rstd（证据张量） |
| `WS_INJW` | `INJW_SLOT=8` fp32/槽 | 每 (token,stream) 一个 32B 槽、有效值在槽首（证据张量） |
| `WS_OH` | **`OH_W=336`** | `[0,320)`=lora、`[320,324)`=injection logits、`[324,336)`=padding（不比较） |
| `WS_LS` | `LOWRANK=320` | silu(lora/HC) |
| `WS_GATE` | `UP_N=10240` | up GEMM 输出 |
| `WS_BLK` | `HID=2560` | gate mix 输出 = 本边界的 block input |

`OH` 行距取 336 而非 324：`324 × 2B = 648B` 不是 32B 倍数，会让每一行的 Fixpipe 落盘非对齐；
取 336（`672B = 21×32B`）后每个 N-tile 的落盘地址都 32B 对齐。

---

## 6. Kernel 设计与数据通路

单一 `__mix__(1, 2)` 启动（AIC:AIV = 1:2 **编译期定死**），段序 S1-S6 + W0：

| 段 | 算子 | 来源算件 | 核 | 并行划分 |
|---|---|---|---|---|
| W0 | `injW = 2·sigmoid(IJ/HC)` | 新增（sigmoid 排布抄 m5/m12：`Muls(-1)/Exp/Adds(+1)/Div`） | AIV 全体 | 每 AIV 自己建**全 m** 表（m·HC 次 4-lane 向量） |
| S1 | combine：`H' = bf16(H + BO·injW[s])` | 新增（4 路广播） | AIV 全体 | item = 整行第 c 个 64 元素 chunk（m·160 个） |
| S2 | grouped GemmaRMSNorm | m6（NormDonor：RegBase LoadAlign/Cast/Reduce + NR-rsqrt 逐字 lift） | AIV 全体 | item = (token,stream) 组（m·HC 个），组常驻 UB 两遍 |
| S3 | down+inject bf16 GEMM（K=10240，N=324→3 个 N-tile） | m11 `Bf16Gemm` → `HcBf16Gemm<DN_K, OH_W, true>` | AIC | N-tile 按 bid 条带 |
| S4 | `silu(lora/HC)` | 新增（sigmoid 排布同 m12） | AIV 全体 | item = lowrank 行第 c 个 64 元素 chunk（m·5 个） |
| S5 | up bf16 GEMM（K=320，N=10240→64 个 N-tile） | 同 `HcBf16Gemm<UP_K, UP_N, false>` | AIC | N-tile 按 bid 条带（64/28 ≈ 2.3 个/AIC） |
| S6 | gate mix：`BLK = bf16(Σ_s sigmoid(g_s)·xn_s / HC)` | 新增 | AIV 全体 | item = 输出行第 c 个 64 元素 chunk（m·40 个） |

数据通路（全程基础 API）：

```
AIV: GM --DataCopy(MTE2)--> UB --Reg::LoadAlign/Cast + 向量指令--> 寄存器 --StoreAlign--> UB --DataCopy(MTE3)--> GM
AIC: GM --DataCopy(Nd2Nz, MTE2)--> L1 --LoadData(LoadData2DParamsV2, MTE1)--> L0A/L0B
     --Mmad(M)--> L0C(fp32) --Fixpipe(FIXP, F322BF16)--> GM
```

`HcBf16Gemm` 相对 m11 的两处改造：`Process()` → `RunTile(nt, nSize)`（层内核按 bid 条带 N-tile），
以及 `B_DUAL`（down 与 inject 的 B 来自两个 GM 源：`nt` 落在 `[0, L/160)` 内取 `wDown` 的满满 tile，
否则取 `wInj` 的 16 行）。**不用 Matmul 高阶 API**，cube 侧全基础 API（`Mmad` + `LoadData2DParamsV2` +
`FixpipeParamsArch3510`），所有 params 结构体全字段显式零初始化（M16 加固）。

`sigmoid(x) = 1/(1+exp(-x))` 的向量排布 `Muls(-1)/Exp/Adds(+1)/Div` 逐字取自 m5/m12（本平台已实证）。

---

## 7. 校验方法与结果

三层校验链（每层都有独立见证）：

### 7.1 kernel 内判据（`evidence/accept_run.log`，**判定项 93 / 0 fails，guard 155 / 0 fails**）

12 个 case × 3 档语义，每个输出张量一条判据 + 非空洞性 + 零权重解析判据 + 跨 case 判别力：

| 判据 | 口径 |
|---|---|
| bf16 输出张量 | kernel 内**判定项** = docs/17 §1.1 分档（`blk`/`xn`/`lora`/… 走 `JudgeT3` 的 T3 ε 界逐元素、违反数须为 0；`exact0`/`zerow` 档另加 T1 逐位一致或逐位为 0）。`归一化绝对误差 ≤1e-2`、`良态元素 ulpMax`、`良态逐位一致率` 是 `JudgeReport` 的**报告项**（不参与 PASS/FAIL）。独立 numpy 门限（`check_ref.py`）另是一套（同向、独立见证），其子条件修正在 **§7.8** |
| `rstd` / `injw`（fp32） | 相对误差 ≤1e-4 / ≤1e-6 |
| `H'` 在 `exact0` 档 | **逐位一致**（100%）：该档 `IJ ≡ 0 ⇒ injW ≡ 1.0`（精确 2·sigmoid(0)=2·0.5），且 H/BO 为小整数 ⇒ `bf16(fp32(H+BO))` 在 fp32 与 double 下结果一致，**可证明逐位** |
| `zerow` 档 | `lora`/`injection`/`ls`/`gate` 必须**逐位为 0**（Σ 0·x = 0 可证明），且 `gate ≡ 0 ⇒ sigmoid(0)=0.5` 使 `blk` 走解析路径 |
| 非空洞性（参考侧） | 6 个输入张量各自整张量 ×1.5 后，参考**任一**输出必须显著变化（>1e-3）——证明参考确实消费每个输入 |
| 非空洞性（设备侧） | 每个张量的设备输出在参考非常量时也必须非常量 |
| 跨 case 判别力 | 12 个 case 的设备 `blk` 指纹在不同 `(kind, m)` 组间**两两互异**（同组不同 mode 的 BLK 本就应相同——`injection` 在本 op 内不参与下游计算） |

**计数口径（docs/17 §2.1 分栏 + §2.2 可复算）**：kernel 在末尾打印
`计数：判定项 N checks / M fails；报告项 K 条；guard G checks / H fails`。实测 **93 / 0、80、155 / 0**，构成可逐步复算：

* **判定项 93** = 80（T3：mode 1 的 8 个 case × 7 张量〔`h'` `xn` `lora` `inject` `ls` `gate` `blk`〕+ mode 0 的 2 个 case × 6〔无 `h'`〕+ mode 2 的 2 个 case × 6〔无 `inject`〕）
  + 4（T1：`exact0`/`exact0m33` 的 `h'` 逐位一致各 1 + `zerow` 的 `lora≡0`/`inject≡0` 各 1）
  + 8（中间 bf16 舍入判别性：10 个 `mode≠0` 的 case 中，`exact0`/`exact0m33` 因两条参考无差异而**不适用跳过**）
  + 1（全局：至少一个 case 建立判别性）。
* **报告项 80** = 12 case × 逐张量观测行（mode 1 各 7、mode 0/2 各 6）——**不参与 PASS/FAIL**
  （docs/17 §1.1：`≤1ulp 比例` / `maxRel` / 位级一致率全部降为报告项）。
* **guard 155** = 6（参考侧非空洞性探针）+ 46（设备侧取值多样性：mode 1 各 4、mode 0 各 3、mode 2 各 4）
  + 10（`injw` 中间量合理性，`mode≠0`）+ 12（`rstd` 中间量合理性）+ 80（每条报告行的「无不有限值」）+ 1（跨 case 判别力）。

> **与 `54e5171` 之前那版口径的差异**：此前 README 写「114 条判据」是把打印行数当判据数（含 8 条判别性、
> 把汇总行也计入），且 70 条 T3 行**不打印 `PASS`**、并未包含在 114 里；`guard` 未单列；
> T3 张量的逐位率曾同时充当判定项。现已按 docs/17 的真实口径重排并加了计数器。

实测汇总（12 case）：

| case | kind | m | mode | H' 逐位 | XN 逐位 | lora/inject 逐位 | gate 逐位 | blk 逐位 |
|---|---|---|---|---|---|---|---|---|
| `exact0` / `exact0m33` | IJ=0、全整数 | 1 / 33 | 1 | **100%** | **100%** | **100%** | 99.99% / 100% | 99.96% / 99.99% |
| `int` | 全整数 | 1 | 1 | **100%** | **100%** | **100%** | **100%** | **100%** |
| `int_mix` / `int_final` | 全整数 | 1 | 0 / 2 | — / **100%** | **100%** | **100%** | **100%** | **100%** |
| `real_m1` | 真实量级 | 1 | 1 | 99.98% | 99.99% | **100%** | **100%** | **100%** |
| `real_m2/m33/m64` | 真实量级 | 2/33/64 | 1 | ≥99.7% | ≥99.3% | ≥99.6% | ≥98.7% | ≥99.7% |
| `real_mix` / `real_final` | 真实量级 | 1 | 0 / 2 | — / ≥99.9% | **100%** | **100%** | **100%** | **100%** |
| `zerow` | 权重全零 | 1 | 1 | **100%** | **100%** | **100%** | **100%** | **100%** |

非逐位的元素全部是 **1~2 bf16 ulp** 的舍入边界效应，且已逐条定位到成因（见 §9.1）。

### 7.2 numpy 独立交叉校验（`evidence/check_ref_run.log`，**12 case 全 PASS**）

`check_ref.py` 用 `tools/golden/moe_block_ref.py` 的 bf16 编解码 + **纯 numpy float64 独立实现** §1.3 伪码，
消费 `M20_DUMP=1` 的 dump，做三件事：

* **A. 端到端**：设备 vs 独立 double 参考（与 kernel 内 host 参考同口径但**独立实现**）；
* **B. 分段链**：以**设备自己的上游 dump** 为输入逐段复算下游（`H'←ij,hin,bo`、`XN←H',hc_norm`、
  `lora/inject←XN,W`、`ls←lora`、`gate←ls,Wup`、`blk←gate,xn`）——把端到端残差按段归因，
  并独立见证「设备内部自洽」（例如 B7/B8 在所有 case 上都是**逐位一致**）；
* **C. 非空洞性**：同 §7.1 的参考侧 + 设备侧探针，另加取值多样性对照（dev 取值数 ≈ ref 取值数）。

**三态退出码与覆盖计数**（tower 规则「SKIPPED/0-1-2」+「统计工具必须交代覆盖范围」；随 `4fe9ac8` 落地）：

```
0 = 比过且通过   →  RESULT: OK (N/N 条比较过：判定项 J + guard G；SKIPPED K)
1 = 比过且有差异 →  RESULT: FAILED (X 条判定项不符：[...])
2 = 没得比/缺输入 → RESULT: SKIPPED (缺输入文件：...)     ←「我没比」与「我比过通过」可区分
```

覆盖计数与比较**由同一份代码产出**（每个 `judge()`/`read_bin()` 调用点同时自增计数），
`unverified`（mode 0 的 `injw`、mode 0 不消费的 BO/IJ 探针、输入恒 0 的探针）**单列打印**，不静默吞。
**负向对照**（tower 规则要求：喂它空输入/坏输入，确认它不发合格证）实测归档在
`evidence/check_ref_negative_control.log`：

| 对照 | 输入 | 实测 |
|---|---|---|
| 0 | 正常 dump | `RESULT: OK (28/28 条比较过…SKIPPED 0)`，rc=**0** |
| 1 | 空目录 | `RESULT: SKIPPED (缺输入文件：m20_real_m1_hin.bin)`，rc=**2** |
| 2 | 只有部分输入（缺 device 张量） | `RESULT: SKIPPED`，rc=**2** |
| 3 | 正常输入 + 把 `blk[0]` 的 bf16 尾数高位翻转（相对变化 ~50%） | `RESULT: FAILED (2 条判定项不符：['blk …','B8 blk<-dev gate,xn'])`，rc=**1** |

（对照 3 里被 `tail` 截掉的判定行本身印 `FAIL`；同时命中 A 段端到端与 B 段分段链两条判定项，
说明两条链都在真实比对。另：把同一元素**只翻 1 ulp** 时判据**通过**——这是容差判据的应有行为，
不是漏洞；1 ulp 尺度属 docs/17 的容差范围。）

判定项/guard 分栏（docs/17 §2.1）：`check_ref.py` 每个 case 打印
`覆盖：判定项 N 条 + guard M 条（共比较 N+M 条）；unverified/SKIPPED K 条`。
实测 12 个 case 合计 **判定项 204 条 + guard 109 条，全 `RESULT: OK`**，另有 **9 条 SKIPPED**
（`exact0`/`exact0m33` 各 1：mode 1 下 BO/IJ 探针不适用之外的输入恒 0 探针；
`int_mix`/`real_mix` 各 2：mode 0 的 `injw` + BO/IJ 探针；`zerow` 3：三个恒 0 权重的探针）
——判定项 = A 端到端 + B 分段链，guard = C 非空洞性探针与取值多样性，SKIPPED 单列不计入判定项。

**判据角色的自我交代**（随 `4fe9ac8` 落地；**M174 订正子条件**）：本脚本 A/B 段用**独立门限**
（张量尺度归一化绝对误差 ≤1e-2 + 良态逐位一致率 ≥0.99 + **良态元素 `ulp>2` 占比 ≤1e-3**；
`ulpMax` 仍作为报告项逐张量打印）。它**不是** docs/17 §1.1 意义上的分档判定项——T1/T2′/T3 分档判据在
**kernel 内**实现（`JudgeT3` 的 T3 ε 界 + 三类计数器）。两者互为独立见证；本脚本的门限修正
（`maxulp ≤2` → 分位口径）按 docs/17 §9.11 规则⑪ 走举证/登记/复审，理由、读数与影响面见 **§7.8**。
⚠ 该门限**不代表实现无缺陷**——它只覆盖「良态元素的细小 ulp 尾部」这一形态。

### 7.3 与 `docs/14`（M31 规格）的逐条对齐

`docs/14-hyperconnection-ple-indexer-spec.md` 已落库，逐条核对结果：**数学与三处 `/HC` 语义完全一致，
无冲突**。核对点如下（含本 kernel 相对 docs/14 建议的**主动偏离**及理由）：

| docs/14 条目 | docs/14 说法 | 本 kernel | 一致性 |
|---|---|---|---|
| §3.2 `grouped_gemma_rmsnorm` | 4 组各 2560 独立归约；affine 逐元素 `[10240]` | S2 按 (token,stream) 组处理，取 `hc_norm[s*HID+j]` | ✅ |
| §3.2 `mixer_mix` ② | `[lora\|inj\|pad] = Xn @ [W_dn; W_inj]^T`，336 = 320+4+12 | S3 一条 GEMM，B 双源（Wdown 320 行 + Winj 16 行），**不做 336 行权重打包** | ✅（§11 #4 裁决） |
| §3.2 `mixer_mix` ③ | `lora = (lora/HC)·sigmoid(lora/HC)` | S4 完全一致 | ✅ |
| §3.2 `mixer_mix` ⑤ | `Σ_s σ(gate_s)·xn_s` 后 `/HC` | S6 完全一致（顺序累加 s=0..3 再除） | ✅ |
| §3.2 `mixer_combine` | `w = 2·σ(inj_s/HC)`，缺省为 1 | W0 完全一致；缺省路径不可达（见上方说明） | ✅ |
| §3.2 融合版 | **combine 结果先舍回 bf16 再做 RMSNorm**（`nvidia/ops/hc.py::_hc_combine_norm_kernel` 的 `out = (res + block).to(out_ptr.dtype.element_ty)`，即 "Round the materialized combine result before normalization" 处；行号随代码变动，以符号为准） | S1 落盘 bf16 → S2 在该 bf16 网格上取平方求和 | ✅ 见 §7.5 |
| §9.1 L40 | 建议 HC 段**按 stream 窜流**（每 stream 5×`[2560]`+双缓冲 ≈50KB） | S2 正是「一个 (token,stream) 组常驻 UB，两遍扫」 | ✅ |
| §9.1 L40 | 「`[10240]` bf16 scratch（gate 与 xn）双缓冲 = 40KB」 | S6 按 64 元素 chunk 处理，不需要整行驻留 | ✅ 更省 |
| §9.4 三处 `/HC` | 三处不可互换 | 逐一对应（S4 / S6 / W0） | ✅ |
| §9.5 「down 与 inject 拆成两条 GEMM」 | 省 padding，代价是多一个段边界 | 本 kernel 用**一条 GEMM + B 双源**，既无 padding 也不多边界 | ✅ 更优 |
| §9.2 「层界附带张量：pending block `[m,2560]` + inj `[m,4]`」 | 三张量层界 | 接口正是 `hIn` + `bo` + `ij` 三路进、`H'` + `BLK` + `OH[:,320:324]` 三路出 | ✅（与 M37/docs/16 §6.1 的「层界三张量」契约一致） |

**两处主动偏离（已披露）**：

1. **（`daf3344` 订正：这一条原先写错了，实为「无偏离」）`hc_norm` 的读法**：docs/14 §9.1 建议
   **直接读 bf16**（20 KB）。**本 kernel 就是这么做的** —— `NormStage` 每 (token,stream) 组
   `DataCopy(wL, hcNormGm[s*HID], Block1(HID*2))` 读 **bf16**（5120 B/组）到 `UB_S2_WB`，再在
   `__VEC_SCOPE__` 里 Cast 成 fp32 参与计算 ⇒ **与 docs/14 §9.1 一致，不构成偏离**。
   此前这里写「改为整核一次性预转 fp32（40960 B PERSIST）… 代价是 UB 多占 20480 B」：那段描述的
   是 `m20_resources.h` 里**声明了但 kernel 从未引用**的两个窗（`UB_PERSIST`/`UB_NORMF32` 40960 B +
   `UB_PC_STAGE` 20480 B，`grep -o` 在 `m20_hyperconn.asc` 内**计数为 0**）—— 即**声明窗 ≠ 实际占用**。
   更正后的口径见 §3 的 UB 行：声明足迹 89472 B，**窗口径工作集 28032 B**（`[61440, 89472)`），
   其中 **kernel 实际建 `LocalTensor` 只覆盖 27424 B**（M69 又把区间细分了一层）。
   （另一个后果：那句「把连续两个 Cast 的 quirk 暴露面从 2 次降到 1 次」的理由也随之作废 —— Pass B 里
   本就有 bf16→fp32 的 Cast。死声明本身**已由 M69 处置**：按 tower 裁定「注释优先、不移动活地址」，
   **声明保留、只加「⚠ 未使用」注释**并给出可复算自查命令，理由与证据见 §3 上表与「活地址未变的复算」块。）
2. **`OH` 的行距取 336（而非 324）**：这不是权重打包，而是**输出 buffer 的行对齐**——
   `324 bf16 = 648 B` 不是 32B 倍数，会让每个 N-tile 的 Fixpipe 落盘行地址非对齐；
   取 336（`672 B = 21×32B`）后全部对齐。代价只有 `M_MAX × 12 × 2B = 1536 B`。
   **本 kernel 不读也不写 336 行的权重矩阵**，docs/14 §9.5 那 245,760 B/mixer 的 padding 开销本来就不存在。

### 7.4 验收分档（`docs/17` §1.1）与各档理由

| 张量 | 档 | 理由 | 判定项 |
|---|---|---|---|
| `H'`（`exact0` / `exact0m33` 档） | **T1（在 T3 界之上加严）** | `IJ≡0 ⇒ injW = 2·σ(0) = 1.0` **精确**；H/BO 为小整数 ⇒ `fp32(H+BO)` 与 double 同值 | **逐位一致（100%）**（代码：`JudgeT3("h'")` 照跑，`cs.strictHcp` 档再加逐位判） |
| `H'`（其他档） | **T3** | `h + BO·injW` 两项**量级相当、可相消** ⇒ 结果可远小于 `Σ\|terms\|`，T2′ 的 ulp 判据在相消区不可达（实测 `real_m33` 有 1 个元素 `\|out\|≈3.1e-6` 而 `Σ\|terms\|≈4`，4 bf16 ulp 差异全部来自 fp32 的两次舍入，与 double 参考无关） | **逐元素 `\|out−ref\| ≤ ε·Σ\|terms\| + 1.0·ulp_bf16(ref)`**，`ε = 2·2^-24 + 1.04e-7`（见下 ε 表 `h'` 行），违反数必须 0 |
| `XN` / `lora` / `inject` / `ls` / `gate` / `blk` | **T3** | (a) GEMM 是 K=10240/320 的长 fp32 累加链；(b) `ls`/`blk` 含 VF `Exp` 近似；(c) `XN` 含 NR-rsqrt 近似 | **逐元素 `\|out−ref\| ≤ ε·Σ\|terms\| + 1.0·ulp_bf16(ref)`**，违反数必须 0；`≤1ulp 比例`/`maxRel`/逐位率**降为报告项** |
| `zerow` 档的 `lora`/`inject`/`ls`/`gate` | T1 | `Σ 0·x = 0` 可证明 | **逐位为 0** |
| 索引/槽位/布局类 | T4 | — | 非空洞性（§7.1）、dump 位置/非零、设备 blk 指纹跨 case 互异、确定性（重复运行逐字节一致） |

**T3 的 ε 逐项推导**（`docs/17` 要求「界必须推导，不许用实测最大值充当界」）：

| 张量 | ε 的组成 | 数值 | Σ\|terms\| 的定义 |
|---|---|---|---|
| `h'` | 2 次 fp32 舍入（`2·2^-24`）+ `injW` 的 fp32-vs-double 相对不确定度（设备 fp32 sigmoid 实测 ≤5.2e-8，取 **1.04e-7** 为上界） | `2·2^-24 + 1.04e-7` = 2.23e-7 | `\|h\| + \|bo·w\|`（相消区判据用；代码 `saHcp`） |
| `XN` | NR-rsqrt 相对误差（m6 donor 的 NR 收敛到 ~`2^-22`，取 1e-7）+ 2 次 fp32 舍入（`2·2^-24`）+ 上游 H′ 的 ≤2 bf16 ulp 一阶映射（`2·2^-8`） | `2·2^-8 + 1e-7 + 2·2^-24` | `\|t\| + \|t·w\| + \|xn\|`（t = x·rrms；第三项是「H′ 的 1 ulp 经 rrms 整组缩放」的一阶量级） |
| `lora` / `inject` | mmad 累加 `K/8` 次 fp32 舍入（每个 16 元素分形的点积 + 一次 L0C 累加 = 2 次舍入 ⇒ `n = 2·K/16 = K/8`）+ 上游 XN 的 **≤2** bf16 ulp（`2·2^-8`，实测 XN 的 `maxUlp=2`） | `(K/8)·2^-24 + 2·2^-8` | `Σ_k \|xn_k·w_k\|` |
| `ls` | VF `Exp` 精度（M24 实测 ≤1.6 ulp ⇒ `1.6·2^-24`）+ mul 链 6 次 fp32 舍入（`6·2^-24`）+ 上游 lora 的 ≤1 bf16 ulp 经 silu 导数传播（`1·2^-8`，实测 lora 的 `maxUlp=1`） | `6·2^-24 + 1·2^-8` = 3.91e-3 | `\|d(uσ(u))/du\| · scale(u)`（导数 × 上游张量尺度） |
| `gate` | mmad 累加 `(K/8)·2^-24` + 上游 LS 的 **≤2** bf16 ulp（`2·2^-8`） | `(K/8)·2^-24 + 2·2^-8` | `Σ_k \|ls_k·wup_k\|` |
| `blk` | 4 路 VF `Exp`（`4×1.6·2^-24`）+ 4 项累加 + 除法 8 次 fp32 舍入（`12·2^-24`）+ 上游 gate/xn 各 **≤2** bf16 ulp 经偏导传播（`2·2^-8`，与 `gate`/`xn` 行同档，实测二者 `maxUlp=2`） | `12·2^-24 + 2·2^-8` | `Σ_s (\|σ(g_s)·xn_s\| + \|σ′\|·scale(g)·\|xn_s\| + σ·scale(xn)) / HC` |

**FTZ 子条款**（`docs/05` §6.1 明确为硬件模式）：`|ref| < 2^-116`（fp32 中间量落次正规区）时 T3 界不适用，
改为要求 `|out| ≤ 2^-116`（设备把次正规刷成 0）。实测：`ls` 有 0~12997 个这样的元素（数据相关），
**越界 0 个**。

（表中 ε 与 `m20_hyperconn.asc` 里 `JudgeT3` 的实参逐一对齐；数值按 K=10240/320 代入。）

**舍入项为什么是 `1.0·ulp` 而不是 `0.5·ulp`**：设备把 fp32 结果舍到 bf16（≤0.5 ulp），参考把 double 结果
舍到 bf16（≤0.5 ulp），两者可落在**同一 bf16 中点的异侧** ⇒ 合计 ≤1 ulp。实测 `real_m1` 的 `k=7253`
就是这种贴中点元素：设备侧 fp32 值（`0x3dc08000`）恰在中点上、ties-to-even 向下，参考在
中点上方 1 个 fp32 ulp（`0x3dc08001`）向上舍 ⇒ 二者正好差 1 bf16 ulp。取 `max(ulp(got), ulp(ref))`
覆盖跨 binade 的情形。
（**评审后修正**：此前 `BfUlp()` 误写为 `2^(e-1-8)`，而 bf16 只有 **7 个显式尾数位** ⇒ 真值应为
`2^(e-8)`；旧式给出的 ulp 只有真值的一半，会把 T3 的舍入项低估 2 倍，从而把贴中点元素误报为违反。
修正后 `h'` 等张量的违反数归零。这类元素的最大比值会**接近 1.0**——这是**紧界的必然表现**
（界已不可再收紧），不是异常。）

实测（12 case 合计 **80 条 T3 判定项**）：**违反数全部为 0**。
非贴中点元素的最差比值 ≈0.6；贴 bf16 中点的元素比值 ≈0.999（紧界worst case）。

### 7.5 combine→RMSNorm 的中间 bf16 舍入：判别性判据

`docs/14` §3.2 与 tower 的 M36 硬约束：`hc_combine_norm` 在 combine 结果与组内 RMSNorm **之间**
插了一次 bf16 舍入（`nvidia/ops/hc.py::_hc_combine_norm_kernel` 里把 combine 结果转 bf16 落盘处；
行号随代码变动，以符号为准），**必须复刻**。本 kernel 的 S1 把 H′ 物化为 bf16
落盘、S2 在该 bf16 网格上取平方求和，故天然复刻。

为了「将来谁改了这里会被立刻抓住」，host 判据里**同时算两条参考**：
`Reference()`（复刻舍入）与 `ReferenceNoMidRound()`（把 S1 的 `bf16(·)` 去掉、其余逐字相同），
然后断言三件事：① 两条参考的差异必须实质性（≥5% 元素）；② 设备与「复刻舍入」参考的最大 ulp ≤2；
③ 在两条参考不同的元素上，设备必须 ≥99% 站在「复刻舍入」这一侧。

实测（`evidence/accept_run.log`）：

| case | 两条参考差异元素 | 设备 vs 复刻舍入 maxUlp | 站复刻侧比例 |
|---|---|---|---|
| `exact0` / `exact0m33` | 0/10240、0/337920 | — | 本档两条参考无差异（整数数据下中间舍入常不改变取值）→ **不适用（跳过）** |
| `int` / `int_mix` / `int_final` | 2027/10240 | 0 | **1.0000** |
| `real_m1` / `real_final` / `real_mix` | 2740/10240 | 2 | **0.9996** |
| `real_m2` | 5323/20480 | 2 | 0.9998 |
| `real_m33` | 87786/337920 | 3 | 0.9999 |
| `real_m64` | 170054/655360 | 3 | 0.9999 |
| `zerow` | 2027/10240 | 0 | **1.0000** |

即：**在复刻与不复刻会给出不同结果的 19.8%~26.8% 元素上，设备 99.96%~100% 站在「复刻舍入」一侧**
（与 M39 用真实权重在 m=4 上测得的 34.25% 同量级、同方向）。
若 kernel 去掉这次中间舍入，`站复刻侧比例` 会掉到 ~0 而立刻 FAIL。
最后一条全局判据：`中间 bf16 舍入判别：至少一个 case 上判别性成立 PASS`。

### 7.6 计算路径合规自查（tower 标准：计算一律 `__VEC_SCOPE__` register-based）

| 路径 | 实现 | 合规 |
|---|---|---|
| 全部 AIV 计算（W0 sigmoid、S1 combine、S2 归约+rstd+归一、S4 silu、S6 gate mix） | 均在 `__VEC_SCOPE__` 内、对 `RegTensor` 调 `AscendC::Reg::`（`LoadAlign`/`StoreAlign`/`Cast`/`Mul`/`Add`/`Muls`/`Exp`/`Div`/`Reduce`/`Duplicate`/`Arange`/`Compares`/`Select`/`LocalMemBar`） | ✅ |
| memory-based vector API（classic `Add(LocalTensor,…)` / `Duplicate(LocalTensor,…)` / `Cast(LocalTensor,…)` 等） | **一处未用**（`grep` 可复核：kernel 内所有向量调用都在 `__VEC_SCOPE__` 块里，且首参是 `RegTensor`） | ✅ |
| 搬运类（`DataCopy` / `DataCopyPad` / `Nd2Nz`）与矩阵类（`LoadData` / `Mmad` / `Fixpipe`） | 按标准保留 memory-based | ✅ 允许 |

### 7.7 证据归档（`m20_hyperconn/evidence/`）

| 文件 | 内容 |
|---|---|
| `accept_run.log` | kernel 内全量日志（12 case，rc=0）：**判定项 93 / 0 fails、报告项 80 条、guard 155 / 0 fails**，含 **80 条 T3 推导界判定项**、4 条 T1 逐位判定项、8 条中间舍入判别性判定项、1 条全局判别性判定项 |
| `check_ref_run.log` | `check_ref.py` 12 个 case 的日志（判定项 + guard 全 PASS，含三态覆盖计数） |
| `check_ref_negative_control.log` | `check_ref.py` 的三态退出码负向对照（rc=0/2/2/1 四组读数，证明「没比」不会被报成「比过通过」、且 FAIL 路径是活的） |
| `dump_manifest.md` | 180 个 dump `.bin` 的 sha256（`real_m1` 等 12 case × 15 张量）+ 再生成命令 + 确定性说明 |
| `h2_blk_recalib.py` + `reproduce.sh` + `m174_reproduce.log` | M174 门限口径修正的复算脚本与读数（§7.8）：`h2.blk` 在 M169 m=4097 档的新/旧口径并列 + m1 回归 + 孤立负向对照 |
| `negctl_over2.py` | M174 的孤立负向对照（零 dump）：新子条件 `ulp>2 占比 ≤1e-3` 在 2.0e-03 时把判据打红，另两条子条件均 PASS |
| `m174_inject_controls.log` | 向 `h2.blk` 前 64 行注入真实量级缺陷（scale/zero/shift）后，新口径仍红的读数 |

> **格式时点（M174）**：`check_ref_run.log` 由改动前的代码生成，其判定行**没有** `良态ulp>2占比` 列；
> 但该日志的 12 个 case 全部在旧 `maxulp ≤2` 下 PASS ⇒ 每张量的良好态 `ulpMax ≤2`、over2 = 0
> ⇒ **新口径下判定结论与之一致**（旧日志的结论未变，只是打印列少了新的一项）。
> `check_ref_negative_control.log` 同理由改动前代码生成：对照 0/1/2（OK / SKIPPED）不受影响；
> 对照 3（单元素 ~50% 翻转）的 `|Δ|` 量级约 0.24、该档 `blk` 尺度按 `accept_run.log` 为 4.0
> ⇒ `归一` 约 6e-2（>1e-2，**由 scale 推算、未实跑复核**）⇒ 预计仍红，但触发子条件由 `ulpMax` 变为 `归一`；
> 需重跑刷新打印列。

未把 dump 原始字节（约 12MB/次 × 12）入库：kernel 的 dump 已实测**确定性**——
**5 个 case 各 5 次独立进程，15 个 dump 文件的 sha256 全部两两相等**（`evidence/determinism_run.log`）：

```
real_m1  : 5/5 次运行 15 个 dump sha256 全等
exact0   : 5/5 次运行 15 个 dump sha256 全等
int      : 5/5 次运行 15 个 dump sha256 全等
int_final: 5/5 次运行 15 个 dump sha256 全等
real_mix : 5/5 次运行 15 个 dump sha256 全等
```

故用 sha256 清单 + 再生成命令替代——需要原始字节时按 `dump_manifest.md` 的命令在空目录复现并逐字节核对。

> **确定性声明的闭环（`54e5171`）**：此前该声明是**错的**（`injw` 每次哈希都不同，manifest 里的
> `c7146f64…` 永远复现不出）。根因与修法见 §5.1：W0 的每 32B 槽只写 4B，槽 `[4,32)` 是 UB 残留；
> 修为「整表按槽清零后再写槽首」（首版步长取错，只清零了 32/256 个槽，已修正）。
> **修后 `injw` 的哈希回到 `c7146f64c52d8435…`**——与原 manifest 一致（原哈希那次运行恰好 UB 尾部为 0），
> 这也解释了 reviewer 看到的现象：`int` 档 14 match + `injw` MISMATCH、mode 0 档（W0 不跑、injw 全 0）一致。

### 7.8 独立 numpy 门限的口径修正（M174；docs/17 §9.11 规则⑪ 的举证＋登记＋复审）

**改了什么（`文件:行`）**：`m20_hyperconn/check_ref.py::judge` 的一条子条件由「良态元素 `ulpMax ≤ 2`」
改为「良态元素 `ulp>2` 的**占比** `≤ 1e-3`」（常量 `SIG_ULP_OVER2_MAX` = `check_ref.py:70`；
`归一 ≤1e-2` = `NORM_ABS_MAX` `:72`、良态逐位率 ≥0.99 = `SFRAC_MIN` `:71` 两条**原样保留**）。
旧 `ulpMax` 仍逐张量打印（**报告项**，不删不隐藏）。改动前的 `ok` 表达式见 base `2b8438c` 的
`check_ref.py:121`（子串 `maxulp <= 2`）。**M140/M169 台架的调用点未改**（`check_full_layer.py:167-168`
仍是同一 `m20.judge(...)` 调用，只换了子条件口径）。

**为什么改（举证）**：
- **档位归属**：`blk = Σ_s σ(gate_s)·xn_s / HC` 含 4 路 VF `Exp`（超越函数近似）+ 多项 fp32 累加 + mmad 累加
  ⇒ 按 docs/17 §1.1 的触发条件属 **T3**；T3 之下「`≤1ulp` 比例 / 位级一致率」是**报告项**，且 §1.1 两条护栏
  第 2 条禁止「以实测最大 ulp 当界」。旧 `ulpMax ≤2` 是本仓在 `4fe9ac8` 自设的保守门限（kernel 内对 `blk`
  走的是 T3 ε 界 `JudgeT3`，`blk` 的 `ulpMax` 在 kernel 日志里一直是「报告」行），**不是** docs/17 的分档口径。
- **可达性（m27 独立见证）**：另一 kernel `m27_hc_prefill` 在同一条 `m20.judge`、同一界面（`a2.blk`）也撞到
  `ulpMax ≤2`（m=4097 档 ulpMax 9）。其可达性实验 `m27_hc_prefill/evidence/ulp_envelope.log`
  （工具 `m27_hc_prefill/tools/ulp_envelope.py`）把参考换成「**同一批文档写明的 bf16 落盘点 + fp32 K-分块累加**」
  （工具自述称其为「任何在这批落盘点取整的实现能给出的下界」），实测该「理想同位取整实现」在 m33_a2 / m64_a2
  的 `blk` 上已到 **ulpMax 4**（良态逐位率 0.9969 / 0.9983）⇒ 按这批落盘点取整的实现类在 `ulp ≤2` 上过不去。
  ⚠ 该工具自述 B 是「我方的 fp32 累加仿真，不是设备的真实累加序」，故它支持的是「这一实现类过不了 `≤2`」，
  **不等于**对任意实现的形式化证明。
- **本档实读（M174 从入库 dump 独立复算，不引用 M173 结论）**：从 M169 的 m=4097 dump 复算 `h2.blk`
  （参考 = `m20.reference`；脚本 `evidence/h2_blk_recalib.py`，读数 `evidence/m174_reproduce.log`）：
  良态 `ulp>2` 占比 = **4.673e-05**（nolinks，301/6441452）/ **4.201e-05**（links，254/6046413），均 <1e-3；
  `归一maxAbs` = 5.183e-03 / 4.184e-03（<1e-2）。**新口径的判定差异只有一处**：nolinks 的 `h2.blk` 由 FAIL 转 PASS；
  **links 的 `h2.blk` 仍 FAIL** —— 它的良态逐位率 = **0.989951 < 0.99**，红在**被保留**的那条子条件上。
  ⇒ **新门限不等于「实现无缺陷」**：它只说明「良态 ulp 尾部」这一形态在 T3 下不构成 FAIL，links 档另有红项。

**M173 survey 的两处读数订正（本 mission 复算）**：
- survey 称 links 档「良态逐位率 0.9900（≥0.99）都过」——**不成立**：精确值为 **0.989951 < 0.99**
  （`evidence/m174_reproduce.log`），按 4 位小数打印才显示成 0.9900。这是「把打印精度当读数」的形态。
- survey 称良态 `ulp>2` 元素的 `|ref|/scale`「全在 [0.05,~0.15]」——**不成立**：实测 nolinks 为
  `[0.0501, 0.3020]`（301 个里 14 个 >0.15），links 为 `[0.0502, 0.2333]`；尾部并非紧贴良态门限。
  其余关键读数与 survey 一致（ulpMax 11/9、逐位 0.9797/0.9785、良态逐位 0.9903/0.989951、over2 4.67e-5/4.20e-5）。

**判据仍有牙（负向对照）**：
- `evidence/negctl_over2.py`（零 dump、可入库）：构造只有 over2 超界的输入 —— 良态 `ulp>2` 占比 `5.0e-04`
  （<1e-3）时 PASS，**`2.0e-03`（>1e-3）时 FAIL**，且该输入上 `归一maxAbs` 与良态逐位率**两条都过**
  ⇒ 红的确实是 over2 这条（读数 `evidence/m174_reproduce.log` §①）。
- `h2_blk_recalib.py --inject {scale,zero,shift}`：向 `h2.blk` 的**前 64 行**注入真实量级缺陷
  （`scale`=×1.02 / `zero` / `shift`=+1%·max），新口径**仍红**（over2 1.57e-2、良态逐位率 0.9748），
  读数 `evidence/m174_inject_controls.log`。

**口径的限度（如实披露）**：`over2 ≤1e-3` 意味着良态元素里**至多约 0.1%** 可以越 2 ulp 而不触发这条子条件
⇒ **单个元素**级别的缺陷不再必然被 ulp 子条件捕获（旧 `ulpMax ≤2` 会捕获）；小占比（<0.1%）的稀疏错误
只能靠 `归一 ≤1e-2` / 良态逐位率 ≥0.99 兜底，而这两条对「单个元素偏离很大但绝对值不超尺度」的情形可能都不动。
本 mission 的注入对照刻意取 **64 行块**（占比 1.57e-2，远大于 1e-3）来证明新口径对**成片**缺陷仍有牙；
这是新口径与「分位」语义的固有取舍，**必须与「本档 ulp 尾部合规」一起读**，不得读成「实现无缺陷」。

**影响面（登记；这些读数需重跑确认，本 mission 不改动它们）**：`m20.judge` 被
`m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py`（M140/M169 台架）、
`m15_layer_loop/check_hc_ref.py`、`m15_layer_loop/check_chain_ref.py`、`m27_hc_prefill/check_ref.py`
复用。当前已知会翻转的档：M169 `m4097_p15_nolinks` 的 `h2.blk`（FAIL→PASS，本 mission 已复算）；
M27 `m64_a2` 的 `blk`（其精确逐位率是否 ≥0.99 待重跑，README 只给 4 位小数）；
凡「只差在 ulpMax、逐位率与 norm_abs 都过」的档都可能翻转，逐位率 <0.99 或 norm_abs >1e-2 的档不受影响。
**回归**：入库的 `dumps_m1_p15` 上 `check_full_layer.py`（调用不变）复跑 `RESULT: OK`
（`evidence/m174_reproduce.log` §②）。

**M173 的待验证三项（原样保留，不得当已证）**：① 无设备 H2 gate/xn dump ⇒「余差来自上游」是**推断**而非直测；
② 为何 H2 比 H1 在同 kernel/m=4097 漂得多**未证**；③ 忠实 T3 ε 界（含 M87 指数区修正）是否真绿**未证**
—— M173 survey 自述「用 ref 自身 `Σ|terms|` 试算 T3 ε 界仍见 8500/10488320 越界、worst ratio 43」，
本 mission **未独立复算该数**（其 ε 定义在 M58-6 行内即有「8 次 vs `12·2^-24`」的张力，见 docs/17 §1.1 已登记缺口），
故单列为 finding 交塔，不在本 mission 内处置。

**复现**（逐字命令见 `evidence/reproduce.sh`；零设备）：
`bash m20_hyperconn/evidence/reproduce.sh`（m1 回归 + 孤立负向对照）；
`M174_DUMP_ROOT=<M169 evidence 目录> bash m20_hyperconn/evidence/reproduce.sh`（另加 m4097 复算与 64 行注入对照）。
m4097 dump 不随仓库入库（约 1.9 GB），由 `m15_layer_loop/evidence/m169_whole_layer_4phase/reproduce.sh`
在设备上再生成。

**复审状态**：本口径修正是**提出并交复审/塔复核**，不是单方面落地（docs/17 §9.11 规则⑪ 第 2 条 (iii)）。

## 8. vllm-ascend 接入点

按 tower 的方向指令（最终目标是在 vllm-ascend 支持 `qwen4_exp`；官方 vLLM 实现是唯一权威规格）：

* **算子边界**：本 kernel 对应 `qwen4_exp/nvidia/hyperconnection.py::GatedResidual.combine_and_mix`
  一次调用——即 vLLM 侧**一个模块级算子**，不是私有接口。
* **权重来源**：vLLM 的 `GatedResidual` 已把 down+inject 打包成 `input_mix_weight_down_block_inject`
  （`MergedColumnParallelLinear`，逻辑分片 `[hc_lowrank, hc_count, pad]`）。本 kernel 直接消费 checkpoint 的
  **原始两支**（`input_mix_weight_down` / `block_inject_weight`），因此接入时按 `_EXTRA_WEIGHTS_MAPPER`
  的 `orig_to_new_stacked` 逆映射取权重即可，**不需要额外的离线重排**。
* **dtype/量化契约**：hyper-connection 权重在 checkpoint 中是 **bf16（不量化）**，与 attention / ngram embedding
  一致；`hc_norm` 为 per-branch affine（宽 HYPER），与 `nvidia/hyperconnection.py` 的
  `norm_size = hyper_hidden_size if hc_per_branch_norm` 一致。
* **调用点**：层内两次（attn 边界 mode 1、mlp 边界 mode 1）、层末一次（mode 2）；层 0 的 attn 边界用 mode 0
  （此时 `prev_block_output = None`，`model.py` 走 `attn_hc.mix`）。宿主需按 PP 边界决定是否物化延迟 combine
  （本 kernel 的 mode 1/2 已把 combine 融合在段内，直接产出物化的 4 路残差流）。
* **⚠️ 已知缺口（披露，不要求本 mission 实现）——层 0→1 需要 combine-only 通路，本 kernel 未提供**：
  `docs/14` §4.1 / `nvidia/model.py::Qwen4ExpDecoderLayer.forward` 的 PLE 分支（`self.ple is not None` 下
  `attn_hc.combine(hidden_states, prev_block_output, prev_injection)`；行号随代码变动，以符号为准）
  明确 PLE 挂在 0-based layer 1，PLE 直接加到多流残差态 ⇒
  该层 attn 边界**打断 combine 与 mix 的融合**，需要一次**独立的 combine**
  （`H' = bf16(H + BO·injW)`，不含 norm）。本 kernel 三档 mode 中 mode 0 = mix-only、
  mode 1/2 = combine+norm 融合，**没有 combine-only 档**（也不含 PLE）；该通路列入**后续 hc 层边界接入
  mission**（tower 已登记，本 mission 不实现）。kernel 侧实现代价很小：`ProcessAic` 跳过 S3~S6、
  AIV 只跑 W0+S1+落 `H'`。
* **待确认（留给 M37/docs/16）**：vllm-ascend 侧自定义算子的注册槽位、以及 PP/SP 场景下 4 路残差流
  （`[T, 10240]`）的搬运约定、以及上面这条 combine-only 档由谁提供。另外 `nvidia/ops/hc.py` 的 `hc_combine_norm` 把 combine 与下一段的 RMSNorm
  融合，本 kernel 的 S1+S2 正是这一个融合对——若 vllm-ascend 侧希望按更细的算子粒度注册，
  S1 与 S2 之间已由 `FLAG_AV0` 分隔，可直接拆。

---

## 9. 已知限制 / 后续

### 9.1 精度残差的逐条归因（已定位，非实现缺陷）

| 现象 | 成因 |
|---|---|
| `H'` 在真实量级下 2/10240 元素差 1 ulp（`exact0` 档 0/10240） | `injW`：设备用 fp32 sigmoid（相对误差实测 ≤5.2e-8）、参考用 double；`H + BO·injW` 落在 bf16 舍入边界时翻转 1 ulp |
| `XN` 偶发 1~2 ulp（1/10240） | RMSNorm 的 `rrms`：设备走 NR-rsqrt（RegBase，m6 donor），参考走 double `1/sqrt`；1e-7 级相对误差落在 bf16 边界 |
| `ls` 全量逐位率 96%~99% | 3510 向量单元对 **fp32 次正规结果 FTZ**（docs/05 §6.1 明确为硬件模式，不在规避范围）；参考在 double 下保留次正规 → 这些元素 |Δ| ~1e-38，数值上无意义（归一化绝对误差 ~1e-40） |
| `blk` 偶发 1 ulp | `gate mix` 的 4 项相消：设备 fp32 累加 vs 参考 double；|Δ| ≤ 0.4% 张量尺度 |

### 9.2 覆盖范围与并行度

1. **只覆盖 decode / 小 batch：`m ≤ 64`（`BASE_M=64` 单 tile）**。已验证 m = 1, 2, 33, 64；
   `prefill`（m > 64）的可扩展性见 §9.3。
2. **S2 的 item 数 = `m·HC`**（m=1 时只有 4 个 item → 4 个 AIV 在关键路径上）。
   `m ≥ 14` 时 S2 满核。这是当前 m=1 的最主要并行度瓶颈；解法是把 (token,stream) 组按 HID 再切分、
   用「部分和落 GM + 一次 AIV mode-0 barrier + 复算 rstd」的两遍归约换满核（额外 1 个 flagId + 1 次 barrier），
   本 mission 未实现（正确性优先，且 m=1 时该段总量仅 10240 元素）。
3. **S3 的 AIC 利用率低**：down+inject 的 `N=324` → 3 个 N-tile → m=1 时仅 3/28 AIC 有活。
   S5 的 `N=10240` → 64 个 N-tile → 28 AIC 各 2.3 个 tile（满核）。
   提升 S3 利用率的正解是**按 K 切分 + 跨 AIC 部分和归约**（K=10240 → 160 个 k-tile，每 AIC 5.7 个），
   代价是多一个 GM 部分和缓冲与一次 AIC 归约轮次；本 mission 未实现。
4. **段间串行、无跨段预取**：S1→S2→S3→S4→S5→S6 全部由 barrier 分隔，权重不做预取。
   收益点在「S3 的 B（Wdown/Winj）与 S5 的 B（Wup）跨段预取到 L1」——L1 的 B 区 ping/pong 已就位。
5. **性能数字未测**：本 milestone 的定位是正确性 + 同步纪律；无 msprof 计时。
6. **W0 的 injW 全表按每个 AIV 各自重建**（m·HC 次 4-lane 向量，m=64 时 256 次）；
   当前每 AIV 私有 UB 表避免了跨核可见性问题，代价是重复计算（m=1 时可忽略）。

### 9.3 prefill（m > 64）的可扩展性（本 mission 不实现，仅给接口方案）

段序与同步结构与 m 无关，只差一个 **M-tile 外层循环**：

* **AIV 侧的 m-tile 应按 `docs/15` §2.3 / §4.4 的建议取 8~12 行**（8 行 × 10240 × 2B = 164KB < 248KB UB），
  让 4 条残差流整段驻 UB、中间量不往返 GM（M33 实测口径：每层访存 ≈670MB → 363MB）。
  本 kernel 当前 m=1 的 S1/S2/S6 已经是「按流/按 chunk 处理、不整行驻留」，扩展到 m-tile ≤12 只需把
  item 网格的 token 维换成 m-tile 内 token，**结构与 UB 布局都不用改**。
* **AIC 侧的 GEMM tile 仍是 `BASE_M=64`**（`docs/15` 的 m-tile 约束是 UB 口径，与 cube 的 M-tile 无关）：
  外层 `for (uint32_t mt = 0; mt < CeilDiv(m, BASE_M); ++mt)`，
  每轮内把 `m` 换成 `curM = min(m - mt*64, 64)`（`HcBf16Gemm::RunTile` 已支持 `curM` mask 与
  `calcM = max(curM,2)` 的 3510 quirk 规避），GM 寻址加 `mt*BASE_M*rowStride`。
* **同步**：每轮重复整段序同步。本工程 flagId 0-15 中**实际一次性分配 11 个**
  （`FLAG_AV0..AV2`、`FLAG_AC0..AC3`、mode2 的 `FLAG_A2C_XN`/`FLAG_C2A_OH`/`FLAG_A2C_LS`/`FLAG_C2A_GATE`），
  **12-15 空闲**（3 也未用）——见 `m20_resources.h` §3。一圈段序需要 11 个 id，而 16 个 id **只够一轮**；
  prefill（m>64，需 `⌈m/64⌉` 轮）因此**走「按 m-tile 多次 launch」**（零同步风险，代价是每次 launch 的固定开销）。
  若要单次 launch 跑多轮，必须给出可核算的「轮次 × flagId」分配表，并满足 docs/05 §6.1 的
  「每 (核,flagId) 是 4bit 计数器，单次 launch 用量须 ≪15」——按每轮每 id 用 1 次计，单次 launch 最多
  `⌊15/1⌋ = 15` 轮 ⇒ `m ≤ 15×64 = 960`，而 prefill 是 `m = 4097`（65 轮）⇒ **必须多次 launch**。
  （本 README 早期版本曾写「19 个 flagId」并给出「m∈[65,128] 用 12-15」的分配，两者都与实际不符且算不平，
  已按上面可核算的版本更正。）
* **AIV 段**：S1/S4/S6 的 item 网格天然按 `m` 线性扩展（m=4097 时分别有 655520 / 20485 / 163880 个 item）；
  S2 的 item 数 = `m·HC`，m=4097 时 16388 个 item，**56 AIV 自然满核**（prefill 反而比 decode 高效）。
* **掩码**：`mAlloc = max(m, 2)` 的 host 契约（`Nd2Nz` 行数=1 退化 quirk，m11 同款）需要保留；
  GM 分配按 `M_MAX` 上界，或每轮用独立的 GM 缓冲。
* `docs/15`（M33 prefill 契约）若给出 prefill 的 m-tile 划分与 workspace 约定，本 kernel 按其调整即可——
  **接口上不需要新增参数**（段序、同步表、layout 全部不变）。

## 10. 文件

| 文件 | 说明 |
|---|---|
| `m20_resources.h` | 全局静态资源表：形状常量 / AIC-AIV BufferID / CrossCore flagId / L1 / L0 / UB / GM workspace（全部编译期常量 + `static_assert`） |
| `m20_hyperconn.asc` | 单一 `__mix__(1,2)` kernel（W0 + S1-S6 + 同步）+ host（数据生成、两条 double 参考、T1/T2′/T3/T4 分档判据 + 三类计数器、dump） |
| `check_ref.py` | numpy 独立交叉校验（A 端到端 + B 分段链 = 判定项；C 非空洞性 = guard）；三态退出码 0/1/2 + 覆盖计数 + 负向对照；12 case 合计判定项 204 + guard 109 |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510` + `-ffp-contract=off`） |
| `evidence/` | 验收证据：`accept_run.log`（判定项 93/0、报告项 80、guard 155/0）、`check_ref_run.log`（12 case 判定项 204 + guard 109）、`check_ref_negative_control.log`（三态退出码负向对照）、`determinism_run.log`（5 档 ×5 次全等）、`dump_manifest.md` + `dump_manifest_full.txt`（180 个 dump 张量的 sha256 + 再生成命令） |
