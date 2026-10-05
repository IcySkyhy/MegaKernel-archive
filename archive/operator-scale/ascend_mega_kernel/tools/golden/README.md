# tools/golden — MoE block 数据、float golden 与量化感知（w4a4）golden

面向 Qwen3.8-Flash-Next-MXFP4（`qwen4_exp`）MoE block 纵向切片的确定性测试数据与三份参考
（float slice 参考 / 层链 float 参考 / 量化感知参考）。链路（docs/12-layer-integration §3）：

```
x_res → m6#1(Add+RMSNorm, gamma1) → x_norm1 → router(top-k) → permute
      → A 量化(A_qx/A_scale) → GMM#1(bf16 out) → SwiGLU(bf16 out) → H 量化(H_qx/H_scale)
      → GMM#2(bf16 out) → unpermute 加权 combine（+ sigmoid 门控共享专家）
      → moe_output → m6#2(Add+RMSNorm, gamma2) → y_final(bf16) + res2(fp32)
```

纯 numpy（2.5.1）实现，无 torch 依赖；Python 用 `/usr/local/python3.12.13/bin/python3.12`。
自检启动时会打印实测的解释器 / numpy / BLAS，并与这里 pin 的数值栈比对；不一致只改**退出码**与提示
（判定项 FAIL 退 3 而非 1），**不放松任何判据**（见「环境识别（M75）」）。

## M26 新增（起因：M21/M22 的两条 finding）

`20260926-agent-moelayer-bug-tools-golden-golden` 指出两处缺口，本目录的升级即针对它们：

1. **补 gamma / 层链输入**：新增 `x_res`（层残差输入）、`gamma1` / `gamma2`（m6#1 / m6#2 norm
   权重）与层链 golden（`x_norm1` / `res1` / `y_final` / `res2`），使 norm 段也能端到端对齐。
   另补 `shared_gate_logits`（设备 sgate 裸点积）。
2. **补量化感知参考**：新增 `qaware_ref.py`，用 double 精度重算设备量化路径（A/H 量化 + bf16
   舍入点 + combine），并给出与 float 参考的偏差预算；激活量化规范**以官方 ops-nn 序列为准**
   （`moe_block_ref.quantize_ocp` 逐句转写），`ceil` 作为显式标注的 legacy 选项保留。
3. **新增真实拓扑档**：`data/real`（m=8、E=512、topK=10；hidden=2560 / moe_inter=640 与真实模型
   一致）。该档权重 1.25 GiB（1280.8 MiB），**只入库 manifest（含每个张量的 size_bytes + sha256 + 偏差预算），
   bin 按需生成**（见 `.gitignore` 与"真实档"一节）。

## M46 新增（起因：M42 的 router FTZ finding）

`20260926-agent-router512-bug-tools-golden-moe-block-ref-py-router-topk-fp32-ftz-512-logit`
指出：`moe_block_ref.router_topk` **没有建模设备的 fp32 次正规 FTZ**。真实规模实测（真实
checkpoint 的 `mlp.gate.weight` 切片 + m=4097、512 专家 / topK=10）——未建模时 **20/4097 行**
（125 个槽位）的 top-10 **集合**与设备不同（不是数值差，是**选中集合**差）；按设备口径把次正规
分数 flush 成 0 后 **0 行不同**。本 mission **自己复现**了这两个数字（`evidence/router_ftz_device_repro.log`，
含重生成 `x` 的 sha256 与设备 dump 的对应关系），并把结论落进参考。

**用户既有裁决（docs/05 §6.1，2026-09-26）**：「fp32 次正规 FTZ 是硬件模式，**不在规避范围**」
⇒ **设备行为是对的、golden 缺项**。所以修的是参考侧，不是给设备"绕开"：

- `moe_block_ref.router_topk` 在设备会 materialize fp32 结果的三个点位逐点加上 FTZ
  （`ftz_f32`，阈值 `2**-126`）：**FTZ#1** max-shift `Sub` 结果、**FTZ#2** `Exp` 结果（`e_i`）、
  **FTZ#3** top-k renorm `Div` 结果。FTZ#2 是生效项（排序/并列表决就发生在它上面）；FTZ#1/#3
  是本目录数据上的惰性项，但同属那条设备路径（逐句标注见"FTZ 建模"一节）。
- `selfcheck.py` 新增 **check 10**：一条**不依赖真实权重**却**真的打到次正规区**的构造用例，
  断言「不建模 FTZ 的旧参考 FAIL、建模后的新参考 PASS」——把建模删掉会立刻被抓（另附变异测试
  取证）。
- **FTZ 可达性审计**（`evidence/ftz_reachability_audit.log`）：参考链上其余 fp32 乘法/相加/归约
  路径逐条判定——要么不可达，要么结果不可观测（次正规项被 `1+·`、`eps=1e-6`、fp4 cast 吞掉）；
  不建模的写进「已知限制」。
- **零回归（时点 + 数值栈限定）**：已发布的 `m1/m33` `.bin` 在**M46 时点、README pin 的数值栈**
  （numpy 2.5.1 / scipy-openblas 0.3.33.112.0）下**逐字节未被重写**——判据是自检的确定性重建逐文件
  `filecmp` 通过（`git log --oneline -- tools/golden/data/m33/moe_output.bin` 也只有入库那次
  `a50b7de`，本 commit 实跑）；`real` 档 manifest 与 M26 合并版**逐字节一致**；自检判据数 = 旧基线
  + 新增 check 10 的 9 条（数字见「复现与自检」）。
  ⚠ **这是带时点与环境限定的读数，不是"逐字节未重写"这类恒真句**：逐位判据把**数值环境变更**
  也当成一类触发源——换到仓里的基线 venv（numpy 1.26.4 / openblas64 0.3.23.dev），同一棵
  `569113d` 树就会给出 ulp 级 FAIL（见「环境识别（M75）」）；即数值环境变更同样属于
  「上游变更不自动流入」那一类，必须由一条 mission 显式重跑。

## M75 新增（起因：selfcheck 的失败信号不辨环境）

M74 的只读勘查定性了 `tools/golden/selfcheck.py` 在 `m33 routed_output` 处 FAIL 的根因：**不是**
golden 参考过期、**不是**参考实现缺陷，而是**解释器 / BLAS 数值环境**（同一棵树 `569113d`：pin 的
解释器 0 FAIL，仓里基线 venv 7 FAIL；换树不变、换解释器翻转）。本 mission **不修那个数值差**（它是
环境事实），只做两件事：

- **让失败信号可区分**（`selfcheck.py`）：启动时打印并核对解释器 / numpy / BLAS；非 pin 的栈上判定项
  失败退 **3**（并在 stderr 说明"环境不明、不可归因"），pin 的栈上失败仍退 **1**。**判据一个字未放宽**
  （负向对照：真翻一个 bit 的 `.bin` 在 pin 的栈上照旧 FAIL rc=1）。
- **给两条无限定的绝对断言加时点 / 环境限定**（本 README）：M46「零回归」那条（见上）与
  「已知限制」第 1 条那句"已无差异"。

细节、实测幅度与命令 → 输出/rc 转录见「复现与自检」下的「环境识别（M75）」与
`evidence/selfcheck_env_discrimination.log`。

## 文件清单

| 文件 | 说明 |
|---|---|
| `moe_block_ref.py` | 核心库：E2M1/E8M0 编解码、**激活量化（官方 ops-nn 序列逐句转写 `quantize_ocp` + legacy `quantize_ceil_legacy`）**、MXFP4 打包/解包、bf16 RNE、**设备 fp32 FTZ（`ftz_f32`，`router_topk` 三个点位逐句标注）**、m6 RMSNorm 参考（`rmsnorm_ref`）、bin/JSON 读写（描述子含 sha256）、路由/重排、MoE slice float golden、数据集生成 |
| `qaware_ref.py` | **量化感知 golden**：double 精度重算设备量化路径（A_qx/A_scale/H_qx/H_scale/GU/Y/combine）+ `deviation_report` 偏差预算四块（source_budget / ceiling / measured / f32_chain_gap） |
| `gen_dataset.py` | 数据集生成入口（默认 `m1,m33`；`--groups m1,m33,real`，`--rules floor,ceil`） |
| `selfcheck.py` | 自检 1-10：数值基础、官方规范（含角落）与 legacy 规则、与 m13 独立量化器逐字节一致、数据集逐字节确定性、层链 golden 不变量、量化感知 golden 全链重算 + 预算 0 违反、跨档口径一致、**可选设备对拍**（`--m13-dump`）、**router fp32 FTZ 建模**（check 10，M46）、**环境识别**（启动时打印解释器/numpy/BLAS 并区分环境 FAIL 与真漂移，M75） |
| `moe_real_accept.py` | **M90**：真实规模（E=512/topK=10）MoE 段的验收与对拍骨架 —— `prepare`（生成 → manifest sha256 对拍 → 巨物入仓检查）/ `run`（T1/T3 分档 + m17 独立参考链 + m22 `nu_*` 接口契约 + 5 条负向对照）/ `pending`（待 device 段落地的判据清单）。**不导入 device 源码**；见下节 |
| `data/m1/` `data/m33/` | 缩小档（E=4/topK=2；K/N 与真实模型一致），已入库 |
| `data/real/` | 真实拓扑档（E=512/topK=10），只入库 `manifest.json` 与 `qaware/*/manifest.json` |
| `data/*/qaware/floor/` | **官方 OCP/floor 规范**的量化感知 golden（17-21 个张量 + manifest + 偏差预算）|
| `data/*/qaware/ceil/` | legacy ceil 规范的量化感知 golden（同一套张量，仅规范不同）|
| `evidence/` | 自检日志、设备对拍日志、真实档生成日志（见"证据归档"） |

## 数据集结构

每组一个目录，`manifest.json` 汇总全部张量（形状/精度/**sha256**/路由与量化约定/偏差预算/seed），
每个 `*.bin` 带同名 `*.json` 描述子：

```json
{"tensor": "x.bin", "shape": [1, 2560], "dtype": "bf16",
 "strides": [2560, 1], "elem_size": 2, "byte_order": "little", "size_bytes": 5120,
 "sha256": "…", "kind": "input", "description": "…"}
```

- `strides` 为 C 行主序、以元素为单位；`dtype ∈ {bf16, fp32, int32, uint8}`；所有 bin 为小端原始字节。
- `kind` 区分 `input`（kernel 输入）与 `golden`（基准输出）。
- K/N 一律真实：hidden=2560、moe_intermediate=640、shared_expert_intermediate=640；
  m1/m33 只缩小专家数与 top-k（512/10 → 4/2），real 档保持 512/10。

### 输入 bin（kind=input）

| 文件 | 形状 | dtype | 说明 |
|---|---|---|---|
| `x.bin` | [m, 2560] | bf16 | **MoE 段输入**（链路的 post-norm 激活；`sliceMode=0` 用它逐 op 对齐） |
| `router_weight.bin` | [E, 2560] | bf16 | 路由 gate 权重（checkpoint `mlp.gate.weight`） |
| `shared_expert_gate_weight.bin` | [1, 2560] | bf16 | 共享专家 sigmoid 门权重 |
| `x_res.bin` | [m, 2560] | bf16 | **层残差输入**（m6#1 的 x，残差为 0；也是 MoE block 要加回的残差流） |
| `gamma1.bin` / `gamma2.bin` | [1, 2560] | bf16 | **m6#1（pre-MoE）/ m6#2（post-MoE）RMSNorm 权重**，∈[1.0,1.5) |
| `experts.gate_up_proj.bin` | [E, 1280, 1280] | uint8 | 专家合并 gate_up 权重，MXFP4 打包（K=2560，N=1280） |
| `experts.gate_up_proj.weight_scale.bin` | [E, 1280, 80] | uint8 | gate_up E8M0 scale |
| `experts.down_proj.bin` | [E, 2560, 320] | uint8 | down 权重（K=640，N=2560） |
| `experts.down_proj.weight_scale.bin` | [E, 2560, 20] | uint8 | down E8M0 scale |
| `shared_expert.{gate,up,down}_proj.bin` + `.weight_scale.bin` | 见 manifest | uint8 | 共享专家（checkpoint 为独立 gate/up 张量） |

形状与命名均对齐 HF safetensors 原始排布（已对照真实 checkpoint 头验证）。

### float golden（kind=golden）

| 文件 | 形状 | dtype | 说明 |
|---|---|---|---|
| `router_logits.bin` | [m, E] | fp32 | 路由 logits（softmax 前，已 max-shift） |
| `shared_gate_logits.bin` | [m] | fp32 | 共享专家门裸点积（sigmoid 前，设备 `sgate`） |
| `topk_ids.bin` / `topk_weights.bin` | [m, k] | int32 / fp32 | top-k id（降序）与归一化权重 |
| `perm_src_token.bin` / `perm_expert.bin` | [m*k] | int32 | 按专家分组的重排（source token / 专家 id） |
| `expert_token_counts.bin` | [E] | int32 | 每专家 token 数 |
| `x_sorted.bin` | [m*k, 2560] | bf16 | 重排后的激活（按 perm 展开） |
| `routed_output.bin` / `shared_output.bin` / `moe_output.bin` | [m, 2560] | bf16 | 路由加权 combine（未加共享）/ 共享专家输出（sigmoid 门后）/ 两者之和 |
| `x_norm1.bin` | [m, 2560] | bf16 | **m6#1 输出** = bf16(rmsnorm(x_res, gamma1))，链路的 MoE 段输入 |
| `res1.bin` | [m, 2560] | fp32 | **m6#1 残差出口** = f32(x_res) + 0（逐位） |
| `y_final.bin` | [m, 2560] | bf16 | **m6#2 输出** = bf16(rmsnorm(bf16(moe_output) + res1, gamma2)) |
| `res2.bin` | [m, 2560] | fp32 | **m6#2 残差出口** = f32(moe_output) + res1（层叠加的残差流） |

### `x.bin` 与 `x_norm1.bin` 为什么要分两个

链路里 router 消费的是 `m6#1(x_res, gamma1)`；而 slice 模式（`sliceMode=0`，逐 op 对齐）的段输入
就是 `x.bin` 本身。RMSNorm 对逐 token 缩放不变，任何 per-channel gamma 都无法让
`norm(x_res)·γ == x_res`，所以"同一张量既当 norm 输入又当 MoE 段输入"在数学上不成立（M21 已记）。
本目录因此**同时**提供两份：`x.bin`（段输入，逐 op 严格判据用）与 `x_norm1.bin`（链路的段输入，
`sliceMode=1` 用），两张张量互相独立。

### 层链语义（m6 对齐 m6_rmsnorm）

```
xAdd   = f32(x) + f32(residual)            # 残差加（fp32，逐位可复现）
resOut = xAdd                              # fp32 写回（残差出口）
rstd   = 1 / sqrt(mean(xAdd²) + 1e-6)      # fp32 累加，eps=1e-6
y      = bf16((xAdd·rstd)·gamma)           # bf16 写回
```

`res1 = f32(x_res)`（m6#1 零残差）；`res2 = f32(moe_output) + res1`（MoE block 自身不加残差）；
`y_final` 用 **bf16 后的 `moe_output`**（m6#2 实际消费的那份张量）重算。跨档自检会检查
`res1/res2` 逐位、`rms(y/gamma) == 1`（与实现无关的 RMSNorm 不变量）等。

## 真实拓扑档 `data/real`

| 字段 | 值 |
|---|---|
| m / E / top_k | 8 / 512 / 10（= 真实模型的专家数与 topK） |
| seed | 20260928 |
| 权重体积 | 1.25 GiB = 1280.8 MiB = 1343062304 B（gate_up 838 MB + down 419 MB + scale 78 MB，十进制）|

- **只入库 `data/real/manifest.json` 与 `data/real/qaware/{floor,ceil}/manifest.json`**：里面逐张量记录了
  `size_bytes` + `sha256` + 偏差预算，等于把"字节指纹"入库；`*.bin` 与 `*.bin.json` 被 `.gitignore` 忽略。
- 生成：`gen_dataset.py --groups real`（按 seed 确定性重放）；校验：`selfcheck.py --real`
  （先生成到 `data/real`，再跑全部数据集自检；约 3 分钟 / 1.25 GiB 磁盘）。
- 该档的存在意义：512 专家 + topK=10 的真实路由/索引/权重规模、并有大量专家 `t_e = 0`
  （空槽位）——这是 m1/m33 缩小档覆盖不到的。注意 m13 的信封是 m≤64、topk≤4，读不了这一档
  （面向 m7/m8 规模与 golden 自身的规模自检）。

## M90 新增：真实规模（E=512 / topK=10）MoE 段的验收与对拍骨架 `moe_real_accept.py`

M90（M77 勘查的 §5「验收口径」/§6 mission E 的落地）把上面那份真实拓扑档**用起来**：脚本只做
参考侧与判据侧的事，不碰任何 device 源码。三条腿：

1. **`data/real`（m=8 / E=512 / topK=10）——主腿。** `prepare` 走「生成 → sha256 对拍 →
   不留巨物进仓」；`run` 用 m17 的 numpy/double 参考链（`m17_moe_real/check_ref.py`，只读 import）
   **从声明输入出发**复算整条 MoE 段（自己的 `x_sorted` gather → 自己的 `quant_hw` 字节 →
   自己的 `GU`/`H`/`H_qx`/`Y` → 自己的 `routed`/`shared`/`moe`），与档内 float golden +
   `qaware/floor` golden 按 **T1/T3 分档**比对。
2. **`m22_router512/evidence/mode2/nu_*`（4 个受控非均匀档的**设备 dump**，只读）**——把
   「由 `topk_ids` 到 counts / slot_base / perm_*」这条**接口契约**独立复算一遍，并要求非均匀
   画像（空专家 / 单槽专家）真的出现；另外只读重跑 m22 自己的 checker（`--w onehot`）。
   这 4 档正是 M40 自陈未交付的「活跃专家数可变 + 每专家 token 数不均」接口契约测试。
3. **`m15_layer_loop/check_moe_ref.py` 的设备路径**——512/10 的融合 MoE 段 ws dump 由 device 段
   mission 落地；本脚本把它显式标为 PENDING（`pending` 子命令打印清单）。

### 参考的输入从哪来（`docs/17` §1.3 三分法）

`docs/17` §1.3 要求把参考吃的**每个输入**归入三类；本 harness 每条判据都带一个 `prov` 标记并在
运行时打印。逐条如下：

| prov | 类别 | 本 harness 里的具体输入 | 合法性 |
|---|---|---|---|
| **①** | 声明输入 | `x.bin` / `x_res.bin` / `router_weight.bin` / `shared_expert_gate_weight.bin` / `experts.*` / `shared_expert.*`（真实档，manifest 有逐张量 `sha256`） | 合法（输入隔离 N1） |
| **②** | 上游输出（**设备**产物） | `m22_router512/evidence/mode2/nu_*/topk_ids.bin` | 合法，**因为**它被点名判据覆盖（见下） |
| **③** | 被判量自身的产物 / 与其共享同一推导链 | **参考链本身只消费 ①**；被判据级标 ③ 的只有 `res2` 与 `y_final` —— 它们的参考输入是**被判量侧自己的 `moe_output`**（其正确性由 `T3 moe` 判据覆盖，满足 §1.3 ③ 的「不得作为唯一判据」） | 标为**非独立性判据**，单列计数 |

* **② 的点名覆盖**：`nu_*` 那 4 个档的参考输入是**设备产生的 `topk_ids`**，覆盖它的判据 =
  **`m22_router512/check_ref.py` 的 `J1`**（脚本里的 `J1` 是 `topk_ids`。注意
  `m22_router512/README.md` §6.2 那张表的编号不同 —— 表里 `J1` 是 `router_logits`；
  **引用时必须写清是脚本的 `J1`**）。本 harness 在同一趟命令里用 `--w onehot` **只读重跑**了
  那个 checker 并要求 `rc=0`。
* **③ 的登记**：`inv_slot` 在真实档里**没有被判量侧的对应张量** ⇒ 它既不是 ① 也不是 ②，
  按 `docs/17` §2.1 **不计入判定项**，已列入 guard 侧的 `T4 结构性：索引/槽位可区分`。
* **③ 的判据 id 集合：代码 ↔ 常量 ↔ 三处文档 自动比对**（`run` 每次都会跑那条判据，不一致 ⇒ FAIL ⇒ rc=1）：
  这张表被抄进三处（本 README、`moe_real_accept.py` 的 docstring、`m15_layer_loop/check_moe_ref.py`
  的 docstring）⇒ 就是 §1.2 说的**相关性转写**风险。守卫的**两层**都在 `§1.3 ③ 漂移自检` 里：
  ① **代码层**：运行时被标 ③ 的判据 id ↔ 声明常量 `CHAIN_CRIT_TAGS`；
  ② **文档层**：`run` **真的去读那三个文件**，解析机器可读标记（**HTML 注释形式**，
  每份文件**恰好一处**：缺失、写错、重复、作用域写错都判 FAIL），与运行时集合逐条比对。
  本 README 的标记就是下面这一行（它是**唯一**一处；正文里的其它提及都是散字、不参与解析）：
  `<!-- M90-CHAIN-IDS[mainleg]: res2,y_final -->`
  同时 `run` 打印 `[§1.3 ③ 判据清单] [...]`。**标记周围的散文（这句话怎么写）仍是人写的，
  不在这条守卫范围内 —— 见「已知限制 / 后续」的第 1 条。**

### 规则的来源与「同源转录」caveat（`docs/17` §1.2）

量化字节判据的规则托在 `M17.quant_hw` 与 `tools/golden/moe_block_ref.py::quantize_ocp` 上，两者都是
对**同一份官方头文件**的转写 ⇒ **同源转录（correlated transcription）**，`docs/17` §1.2 明说它
「**仍咬不住共同误读**」。引用时须带该节 caveat：m17 的规则在 **M60 之前无外部 pin**，M60 补的是
**同源转录**（m17 的 `W1`–`W6` 见证），**不是独立第二来源**。因此本 harness 把它算作
「**同规则的两份转写交叉见证**」，**不**声称「两套独立实现」。

同理，主腿是 **golden vs 从 ① 出发的复算**，**不是设备对拍**：它是**参考链的一致性/结构性检查**。
设备证据面只有 `nu_*` 那 4 个档（②）。判据分档、FTZ 口径（`docs/17` §7.1）、T3 网格项取 `BfUlp`
哪一半、与 M77 §5.2 短写的两处对账，逐条写在脚本的模块 docstring 里，此处不重复。

命令（`$PY` = `/usr/local/python3.12.13/bin/python3`）：

```bash
$PY tools/golden/moe_real_accept.py prepare        # 生成真实档 + sha256 对拍 + 巨物入仓检查
$PY tools/golden/moe_real_accept.py run            # 主验收（T1/T3 + nu_* 契约 + 负向对照）
$PY tools/golden/moe_real_accept.py run --tier m1  # 同一条链在缩形档（E=4/topK=2）上交叉检查
$PY tools/golden/moe_real_accept.py pending        # 待 device 段落地后才能跑的判据清单
```

**命令 → 读数**（下列读数都是 2026-09-27 在 M90 的 worktree 上实跑抄出的；`LC_ALL` 取 `C` /
`C.UTF-8` 时输出逐字节相同）：

| 命令 | 读数 |
|---|---|
| `prepare` | rc=0；重放后 `manifest.json` sha256 与生成前一致（`a203ce27b893101e…`）；核对 73 个张量、缺 0、sha256 不符 0；盘上 `.bin` 合计 1283.3 MiB；`git status --porcelain tools/golden/data/real` **0 行**；本轮枚举到的 73 个 `.bin` 均被 `git check-ignore` 命中（不入仓） |
| `run`（real） | **RESULT: OK**；判定项 **52** 条全 PASS（T1 35 / T3 12 / NC 5），其中**输入独立（①+②）50 条**、③ 非独立 2 条；guard 9 条单列（含 `§1.3 ③ 漂移自检` PASS） |
| `run --tier m1 --no-m22` | **RESULT: OK**；判定项 **32** 条全 PASS（T1 15 / T3 12 / NC 5），① 30 条 + ③ 2 条；guard 5 条单列 |
| `run` 的 §1.3 两行（文档引用它们，不另写数字） | `§1.3 ③ 漂移自检：运行时被标 ③ 的判据 == CHAIN_CRIT_TAGS（声明常量）` → `运行时 ③ 判据清单 = ['res2', 'y_final']（2 条）；声明 = ['res2', 'y_final']（2 条）→ 一致`；`[§1.3 ③ 判据清单] ['res2', 'y_final']（2 条）` |
| `run` 里的 nu_* 腿 | 4 个设备档（`nu_m1/m7/m33/m64`）× (4 条契约 T1 + 1 条画像 guard)，且 m22 自己的 checker 在同样 4 个 dump 上 `rc=0` |

**负向对照的读数**（`NC1`–`NC3` 是**产物侧注入**：改被判量侧的字节 → **用同一套判据重跑** →
把**真的变红**的判据逐条列出；没咬住即判 FAIL）：

| 对照 | 注入 | 读数（512/10 档） |
|---|---|---|
| `自检-NC1` 路由配错 | 被判量侧 top-k 由 10 改 9（`top-9` + 重复末位），并按其重建 counts/perm/`x_sorted` | **6 条判据真的变红**：`topk_ids`、`expert_token_counts`、`perm_src_token`、`perm_expert`、`expert_offsets/slot_base`、`x_sorted` |
| `自检-NC2` 专家序错位 1 | 被判量侧 ids = `router_topk(x, roll(rw,1), 10)` 的结果 | ids **8/8 行**不同（churn 80/80）；**4 条变红**：`topk_ids`、`expert_token_counts`、`perm_expert`、`expert_offsets/slot_base`。`perm_src_token`/`x_sorted` **不红是正确读数**（整体标号平移不改变专家间相对次序 ⇒ 每槽的 token 不变） |
| `自检-NC3` 槽位计数 off-by-one | 被判量侧 `expert_token_counts[argmax] += 1` | **2 条变红**：`expert_token_counts`、`expert_offsets/slot_base` |
| `自检-NC4` FTZ 关闭 | 规则级反事实（不吃被判量字节），构造 shifted logits = 0 / −90 / −95 / −100 | 建模 FTZ `ids=[[0,1],[0,1]]` vs 不建模 `ids=[[0,510],[0,511]]`（不同）；**本档真实数据上该反事实差异 0/8 行**（档内 shifted logit 下界 −6.961，未触及 −87.34 的次正规带） |
| `自检-NC5` 量化方向反转 | 规则级反事实（借 m17 的 `quant_rule_reversed`） | 与 A 侧 golden 失配 **101700/108800** 字节（93.5%） |

**抓「路由配错」的是哪条**：`自检-NC1`/`NC2` 的读数显示，抓它的是 **T1 的 `topk_ids`**，
并且**同一批注入连带**把 `expert_token_counts` / `perm_expert` / `expert_offsets`（`NC1` 还含
`perm_src_token`/`x_sorted`）打红 —— 即索引类判据**有咬合力**，不是「只有下游数值红」。

## MXFP4 布局约定

权重逻辑形状 `[N, K]`（out×in），行主序、K 连续；与 checkpoint 一致：

- **E2M1**：1 符号 + 2 指数（bias 1）+ 1 尾数；code 0..7 = `0, 0.5, 1, 1.5, 2, 3, 4, 6`
  （有限值，无 NaN/inf；最大可表示值 6）。level 间距**不均匀**（0.5/1/2），这一点影响误差预算
  （见"偏差预算"）。
- **E8M0 scale**：纯指数、bias 127，字节值 `b` 表示 `2^(b-127)`。
- **group = 32 沿 K**：每行 K 维每 32 个连续元素共用一个 scale；scale 张量形状 `[N, K//32]`。
- **lohi nibble**：`packed[n, j]` 字节 = `code(w[n,2j]) | code(w[n,2j+1]) << 4`；低 nibble 存较小 k。
- 打包张量形状 `[N, K//2]` uint8。激活量化（`quantize_activations`）用完全相同的字节布局，
  所以 kernel 一套解包逻辑可同时消费 golden 权重与 golden 激活。

### 激活侧量化规范：以官方实现为准（逐句对照）

用户裁决（2026-09-26，tower 转发）：「参考官方的，和官方一致就行了」。因此本目录的激活量化
**默认规范 = 官方 ops-nn 序列**，`moe_block_ref.quantize_ocp` 是它的**逐句转写**：

权威源：`/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/
add_rms_norm_dynamic_mx_quant_common.h`（与 opp 内置 `.../ops_nn/ascendc/dynamic_mx_quant/
arch35/dynamic_mx_quant_tail_axis.h` 同构），即 op type `DynamicMxQuant`；Qwen3.8 的
`quantization_config.npu_reference` 就是
`torch_npu.npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode="round")`。
完整的 15 步官方-vs-设备差异表见 `docs/13-mx-quant-primitives.md §4/§5`。

| # | 官方步骤（行号 = 官方头文件当前副本，只作查阅提示，见下注） | `quantize_ocp` 里的 numpy 语句 |
|---|---|---|
| 1 | 组内**指数域** max（官方 `MxQuantComputeMaxExpOCP`：`And 0x7F80` ×2 + `Max` + `ReduceDataBlock<MAX>`） | `maxexp = max(f32_to_bf16_bits(x) & 0x7F80)` |
| 2 | emax 常数 `FP4_E2M1_BF16_MAX_EXP = 0x0100`（:71-98） | `FP4_E2M1_BF16_MAX_EXP` |
| 3 | 非有限判定 `Compare NE(maxexp, 0x7F80)`（:401） | `nonfinite = maxexp == 0x7F80` |
| 4 | 下界 clamp `Compare LE` + `Select`（:402-403） | `shared = maximum(maxexp, 0x0100) - 0x0100` |
| 5 | `Sub(shared, maxexp, emax)`（:404） | 同上 |
| 6 | E8M0 字节 `ShiftRights(shared, 7)`（:405） | `byte = shared >> 7` |
| 7 | 非有限 → `0xFF`（`Select`，:406） | `np.where(nonfinite, 0xFF, byte)` |
| 8-9 | `halfScale = 0x7F00 − shared`（:410-412） | `half_bits = 0x7F00 - shared` |
| 10a | 非有限 → halfScale `0x7F81`（:413） | `np.where(nonfinite, 0x7F81, half)` |
| 10b/10c | `shared == 0x7F00` → `0x0040`（:411/:415） | 不可达（`shared` 是 0x80 的倍数，0x7F00 = 254×0x80 不出现） |
| 11 | `shared == 0` → halfScale `0`（:414） | `np.where(shared == 0, 0, half)` |
| 12 | **bf16 域乘**（:753-754） | `h = x * bf16_bits_to_f32(half_bits)` |
| 13-15 | `Interleave` → `Cast<T_Y,T_X,castTraitRM<roundMode>>` → `DIST_PACK4_B32`（:755-765） | `e2m1_encode_away(h)`（CAST_ROUND，平局远离零、±6 饱和）+ lohi 打包 |

> 上表「行号」指官方 `add_rms_norm_dynamic_mx_quant_common.h` 的**当前 ops-nn 副本**
> （`/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/`）—— **只作查阅提示、
> 随代码变动**；随包 CANN 那份同构副本行号不同（`MxQuantComputeMaxExpOCP` :274 /
> `MxQuantComputeScaleOCP` :337 / `MxQuantComputeDataFP4` :688）。定位请以**符号名**为准，
> 不要按区间去找函数 —— 行号随副本与版本漂移，同一个区间可能落到别的函数里。

规范含义：**OCP 语义**，`scale = 2^(floor(log2 amax) − emax)`，`emax = 2`（6.0 的指数）——
是 **floor**，不是 ceil/round（官方三处独立来源一致，docs/13 §4.2）。两个**角落**也在转写里：

- 有限但退化组（指数域被 clamp 到 emax：全零 / bf16 次正规幅值）→ 字节 **0**、halfScale 0 →
  **全 0 code**（:402-403 / :414）；
- 含 ±Inf/NaN 的组 → scale 字节 **0xFF**（E8M0 NaN 码）、halfScale **0x7F81** →
  `Mul(±Inf, NaN) = NaN` → cast → **全 0 code**（:406 / :413）。

> **`ceil` 是 legacy**（`quantize_ceil_legacy` + `pack_mxfp4` 的合成权重打包）：`scale =
> 2^ceil(log2(amax/6))`、最近值/平局取偶 code——这是被 docs/12 §6 小改 C 替换掉的 m3 标量量化器
> 的规则，只用来复现裁决前的 golden（`qaware/ceil`），**不是**给设备判定的参考。两套规则在
> `amax ∈ [4·2^k, 6·2^k]` 完全一致、在 `(6·2^k, 8·2^k]` 相差 1（`ceil` 才是偏差的一方；
> M21 报的 nibble 差异 29~48% 就是这个差）。自检 check 3 显式验证这两条与上面两个角落。
>
> **权重侧不涉及规范**：checkpoint 的打包字节由模型作者的 CPU RTN 工具产出
> （`cpu_rtn_provenance.tool = qwen3.8-flash-next-cpu-rtn-mxfp4`），我们**按给定字节直接消费、
> 从不重新打包**；本目录 `pack_mxfp4`（ceil）只描述*合成*测试权重是怎么造的。
>
> **与设备的关系（现状，核对于 2026-09-26）**：m2/m5/m13 的 VEC 路径**已具备**官方 `:413` 那一句
> `Select`（`halfScale` 非有限组 → bf16 NaN，docs/13 §6）。设备侧的确切写法在
> **`MxQuantComputeScale` 内**：`Duplicate(nanRegTensor, NAN_CUSTOMIZATION)`，其中
> `constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;` —— **小写**，所以**大小写敏感的 `grep 0x7F81` 找不到它**
> （该 grep 每份文件 3 处命中**都不在设备路径上**；`selfcheck.py` 的 check 3 会当场打印这几栏计数）。
> 此前这里曾写「缺该句、对含 Inf 的组设备给 ±6；M32 正在修」，
> 那是**修复前**的状态记录；M32 落地后设备与官方在此角落一致。除此之外逐字节一致
> （见"与设备逐字节对拍"一节，本目录的 golden 数据里没有 Inf/退化组）。

## 量化感知（w4a4）golden `qaware/<rule>/`

`qaware_ref.qaware_moe_block` 用 **float64** 重算设备链路，量化/舍入点与设备逐一对应：

| 阶段 | 参考实现 | 设备对应 |
|---|---|---|
| A 量化 | `quantize_activations(x_sorted / x, rule)`（group 32 沿 K） | m13 S5（routed + shared 两次调用） |
| GMM#1 | `a_dq @ W_gate_up^T`（f64）→ **RNE 到 bf16** | m3/Mmad + FIXP bf16 输出 |
| SwiGLU | 由 **bf16 GU** 算 `silu(gate)·up`（f64）→ **RNE 到 bf16** | m5 SwiGLU（`swigluOut` bf16） |
| H 量化 | `quantize_activations(bf16(h), rule)` | m5 `MxQuant` |
| GMM#2 | `h_dq @ W_down^T`（f64）→ **RNE 到 bf16** | m3/Mmad + FIXP |
| combine | 用 **bf16 top-k 权重**（`w_tk_packed`）折叠 bf16 Y；sigmoid 门取 f32 裸点积；`moe = bf16(f32(routed_bf16 + shared_bf16))` | m8#2 unpermute + combine |
| m6#2 | 同 `rmsnorm_ref`（用 bf16 `moe_output`） | m13 S10 |

张量（均按 **perm 顺序紧凑行**，第 r 行 = 第 r 个排序槽位；设备是逐槽位 MM 行 padded 布局，
映射见下）。路由/重排张量（`router_logits` / `topk_ids` / `topk_weights` / `perm_*` / `x_sorted`）
与 float golden 完全一致（量化不影响路由），因此只在顶层给一份、不重复输出：

| 文件 | 形状 | dtype | 说明 |
|---|---|---|---|
| `a_qx.bin` / `a_scale.bin` | [m*k, 1280] / [m*k, 80] | uint8 | routed A（`x_sorted`）量化结果 |
| `a_qx_shd.bin` / `a_scale_shd.bin` | [m, 1280] / [m, 80] | uint8 | 共享专家 A（`x`）量化结果 |
| `gu.bin` / `gu_shd.bin` | [m*k, 1280] / [m, 1280] | bf16 | GMM#1 输出（gate\|up 拼接，bf16） |
| `h_swiglu.bin` / `h_swiglu_shd.bin` | [m*k, 640] / [m, 640] | bf16 | H 量化器输入（SwiGLU 后 bf16） |
| `h_qx.bin` / `h_scale.bin` | [m*k, 320] / [m*k, 20] | uint8 | routed H 量化结果 |
| `h_qx_shd.bin` / `h_scale_shd.bin` | [m, 320] / [m, 20] | uint8 | 共享专家 H 量化结果 |
| `y_sorted.bin` / `y_shd.bin` | [m*k, 2560] / [m, 2560] | bf16 | GMM#2 输出（bf16） |
| `routed_output.bin` / `shared_output.bin` / `moe_output.bin` | [m, 2560] | bf16 | 量化感知的 combine 输出 |
| `x_norm1.bin` / `res1.bin` / `y_final.bin` / `res2.bin` | 见 manifest | bf16 / fp32 | 层链（用 bf16 `moe_output` 续算） |

### 与设备（m13，`sliceMode=0`）逐字节对拍

`selfcheck.py --m13-dump <dumpdir>`（dump 由 `M13_DUMP=1 ./m13_moe_layer <data> m1 m33` 落在该目录）。
`evidence/m13_device_crosscheck.log` 是实测归档：

| 比较项 | m1 mode0 | m33 mode0 |
|---|---|---|
| A/H 量化字节（`a_qx` `a_scale` `a_qx_shd` `a_scale_shd` `h_qx` `h_scale` `h_qx_shd` `h_scale_shd`） | **0 字节不符**（5100 B） | **0 字节不符**（168300 B） |
| `gu` / `gu_shd` / `h_swiglu` / `h_swiglu_shd` / `y_sorted` / `y_shd` / `shared_output` | **逐位一致** | **逐位一致** |
| `routed_output` | 逐位一致 | ≤1 bf16 ulp（max 1.562e-2） |
| `moe_output` | ≤1 bf16 ulp（1.562e-2） | ≤1 bf16 ulp（3.125e-2） |

设备逐槽位 padded 布局 → 本目录紧凑行的映射：`设备[专家 e][:t_e]` ⇔ `golden[off[e] : off[e]+t_e]`
（`off` = `expert_token_counts` 的前缀和，即 perm 槽位）。注意设备侧 **`h_swiglu` 是紧凑行**
（不入 padded 布局），其余逐槽位张量都是 padded。combine 那 1 ulp 的差来自设备在 **f32** 里做
加权折叠/相加，而本参考用 f64（更精确的一方），m13 的 combine 判据本就是"bf16 位距 ≤2 ulp"。

对拍同时验证**层链验收判据可用**：以 floor 规范 golden 作 `ar`，m1/m33 mode0 的 device vs float
golden 全部满足 `|dev − gold| ≤ 1.5·ar + 0.02·|gold| + 5e-3`（**0 违反**，最差占容差 0.66~0.72）
——即 M21/M22 finding 里缺的那条"w4a4 验收通道"已经打通：量化段可以用量化感知 golden 严格判定，
对 float golden 只用偏差预算解释。

## 偏差预算（与全 float 参考）

`qaware_ref.deviation_report` 给四块（都写进 `qaware/<rule>/manifest.json` 的 `deviation_budget`）：

1. **`source_budget`（可操作、紧）**：量化器本身的逐元素解析上界
   `0.5·(包围该值的两个 E2M1 level 间距) + max(0, |v| − 6·scale)`（E2M1 level 间距不均匀，
   最大间距 2 个 scale 单位；官方 floor 规范还会饱和，故有第二项）。对三档 × 两套规范，
   实测 `max|q−v|` 与上界之比（max_utilization）**正好 1.00、0 违反**——上界可达、不松。
   A 侧上界 1.41、H 侧 5.5（m33，floor）。key 集固定为 `{a_routed, a_shared, h_routed, h_shared}`。
2. **`ceiling`（对 **f64 链**严格；对入库 f32 golden 是经验上界）**：把**实测**量化误差按 `|W|`
   绝对权传播（最坏情况：所有误差同号对齐），再加两个**相对**项（bf16 舍入点 `2^-8·|v|`、
   参考侧 f32/f64 运算差 `F32_REL = 2^-23·|v|`）。这两项都正比于 `|v|`，因此覆盖不了下一种误差。
   - 对 `gu`/`gu_shd`/`y_sorted`/`y_shd`：两侧都在 f64 里算（对比对象是同段 f64 值），**数学上严格**。
   - 对三个输出与 `res2`：对比对象是**入库的 f32 float golden**，而 f32 累加误差不在包络里
     —— 它是"量级正比于求和诸元、与结果幅值无关"的误差，近零元素上可以远超 `2^-8·|golden|`
     （reviewer 实测某元素为 **583×**）—— 所以那里只能称**经验上界**。本目录把该 gap 量出来存进
     `f32_chain_gap`（`max_abs` = max|f32 golden − f64 链|，`max_utilization_vs_ceiling` = 该 gap
     占包络的比例），实测（floor/ceil 同量级）：

     | 档 | f32 gap max_abs（routed / shared / moe / res2） | gap 占包络比例（最大项） |
     |---|---|---|
     | m1 | 1.56e-3 / 5.7e-7 / 1.56e-3 / 8.41e-3 | ≤0.0008（res2） |
     | m33 | 9.27e-3 / 2.3e-6 / 9.27e-3 / 2.08e-2 | ≤0.0018（res2） |

     即 gap 实际只占包络 0.2% 以下，`ceiling` 在这些数据集上仍成立（violations 0），但**严格性只在
     f64 一侧成立**，这一点必须照实说。
   - 保守度：因丢掉 K 求和的符号相消，`ceiling` 对被约束量偏保守，实测 utilization：
     m1/floor **0.075–0.135**（7.4×–13.3×）、m1/ceil 0.070–0.141（7.1×–14.3×）、
     m33/floor 0.109–0.172（5.8×–9.2×）、m33/ceil 0.093–0.180（5.6×–10.7×）、
     real/floor 0.052–0.196（5.1×–19.4×）、real/ceil 0.045–0.161（6.2×–22.3×）；
     总体 **0.045–0.196 ⇒ 余量 5.1×–22×**。口径：各档 `ceiling[*].max_utilization` 的 min–max
     （9 个张量：gu/gu_shd/y_sorted/y_shd/三输出/y_final/res2），可直接从 manifest 复算。
3. **`measured`（判定用的 oracle）**：量化感知 golden 与 float golden 的逐元素偏差，以及各中间段
   （GU / H / Y）相对"同段 float 值"的偏差；层链的 `y_final`/`res2` 也在内（key 集 15 个，
   自检有判定项钉住 key 集，防止静默漏项）。**层链验收判据**（沿用 m13 §5.1 第 3 项，真机全 PASS）：

   ```
   |device − float golden|  ≤  1.5·|qaware − float golden|  +  0.02·|float golden|  +  5e-3
   ```

4. **`f32_chain_gap`（口径披露）**：见上，量化"f32 float golden 自身的累加误差"，供判据留白核算。

实测偏差（`measured.max_abs`，绝对量；与 m13 真机实测同口径）：

| 档 | 规范 | routed_output | shared_output | moe_output |
|---|---|---|---|---|
| m1 | floor | 0.6044 | 0.6267 | 0.7629 |
| m1 | ceil | 0.6242 | 0.6450 | 0.8501 |
| m33 | floor | 0.9628 | 0.8606 | 1.1916 |
| m33 | ceil | 0.9981 | 0.8320 | 1.3247 |
| real | floor | 0.3368 | 0.8453 | 0.8322 |
| real | ceil | 0.3349 | 0.7892 | 0.8797 |

（m13 真机在同一数据集上实测 m1 = 0.6045/0.6282/0.7627、m33 = 0.9609/0.8594/1.1875，
与本参考的一致性在 1% 量级。）相对偏差统计只对"非近零"元素统计（`rel_threshold = 1e-3·max|ref|`），
因为近零元素的相对偏差必然爆表（相消）。

**为什么没有更紧的先验界**：K 维求和里量化误差与权重符号随机，任何不假设分布的严格上界都必然
乘上 `Σ|W| / |ΣW|` 的保守因子；能紧的只有两个量——量化器本身的上界（`source_budget`，紧到 1.0）
与逐元素 oracle（`measured`，即判据里那个 `|qaware − float|`）。这也是 m13 采用"用实测 w4a4 偏差
解释 gap"这条判据的原因；判据自带的 `1.5×` oracle 因子与 `5e-3` 绝对留白，正是用来兜住上面
`f32_chain_gap` 这类未建模项（实测判据最差占容差 0.72 ⇒ 余量 ≥5.1×）。

## 路由语义（Qwen3.8 top-k，对照 vLLM `Qwen3NextSparseMoeBlock`）

1. `logits = x @ router_weight.T`（float32；bf16 输入上采样）
2. `scores = softmax(logits)`（全专家）
3. top-k 按分数降序；精确平局取较小专家 id
4. 路由权重归一化使每 token top-k 权重和为 1（`norm_topk_prob` 默认 True）
5. 共享专家：`sigmoid(x @ shared_expert_gate_weight.T) · MLP_shared(x)`，与路由专家加权和相加
6. 专家内部：`gate_up` 前 640 列为 gate、后 640 列为 up，`silu(gate) · up` 后接 down
7. **设备 fp32 FTZ**（M46 补）：设备 softmax 分数落进 fp32 次正规区时被硬件 flush 成 0，
   参考必须同样建模，否则深尾 logits 下 top-k **集合**都会不同（见下）。

float golden 全程 float32（导出时按 RNE 舍入到 bin 声明的 dtype）；量化感知参考全程 float64、
只在设备会舍入的点舍入到 bf16。

## FTZ 建模（M46）：设备的 fp32 次正规 flush

**口径**：`|结果| < 2**-126`（fp32 最小正规数）的 fp32 算术结果 → `+0`（`ftz_f32`）。
依据 docs/05 §6.1 的用户裁决「fp32 次正规 FTZ 是硬件模式，不在规避范围」——**设备是对的**，
参考缺了这一项 ⇒ 修参考。

### 建模位置（逐句，对照设备 m7/m22 router）

| # | 设备算子（m7 `SoftmaxTopkRenormRow` / m22 同构） | 参考语句（`moe_block_ref.router_topk`） | 实测可达性 | 状态 |
|---|---|---|---|---|
| FTZ#1 | Reg `Sub`（`logits -= max`，结果存 UB 再进 Exp） | `logits = ftz_f32(logits - logits.max(axis=1, keepdims=True))` | 需两个 ~O(1) 的 logits 相差 < 2^-126；真实档 min\|logit\|≠0 = 1.58e-4 ⇒ 不可达 | 建模（惰性） |
| FTZ#2 | Reg `Exp`（`e_i`，**Sort32/归并树比较的就是它**） | `scores = ftz_f32(np.exp(logits))` | **可达**：真实 m=4097 档有 127 个被选中槽位的 `exp` 落次正规区 | **建模（生效项，本 mission 的修正点）** |
| FTZ#3 | Reg `Div`（top-10 renorm，设备**唯一**的除法） | `topk_weights = ftz_f32(topk_weights / topk_weights.sum(...))` | 需 `e_i ∈ [2^-126, Σe·2^-126)`；真实档该带 0 槽 | 建模（惰性） |
| — | 设备**没有** softmax 的"除 sum"：`/sum` 与 renorm 抵消，kernel 直接对 `e_i` 排序再归一 | `scores = scores / scores.sum(...)` | 无对应点位；正数缩放保序，不影响 ids | 不加 FTZ |

三个点位里 FTZ#2 是唯一会改变判据的：设备把深尾 `e_i` 全看成 0，于是**同值并列取小 id**；
参考若保留 1e-38..1e-45 的不同次正规值，就会按"谁更大"排序 ⇒ 选出**另一批**专家。
这就是 M42 的 20/4097 行（复现日志 `evidence/router_ftz_device_repro.log`，两种口径的数字与扫描表
逐项一致）。

### 同一参考链的其余 fp32 乘法/相加/归约路径（审计结论）

审计脚本与原始计数：`evidence/ftz_reachability_audit.{py,log}`（`m1/m33` 全链逐张量数
"0 < |v| < 2^-126" 的个数 + 最小非零 |v|）。判定分三类：

| 路径 | 会碰到次正规？ | 状态 / 理由 |
|---|---|---|
| `np.exp(logits)`（router） | **会**（`logit ≲ -87.3`） | 已建模 = FTZ#2 |
| `logits -= max` / top-k renorm `Div` | 数学上可达、本数据族不可达 | 已建模 = FTZ#1/#3 |
| 参考的 softmax `/sum` | 无设备对应点位 | 不建模（除法保序，ids 不受影响） |
| router GEMV / 专家 gate_up / down / combine 的 fp32 累加链 | 需整条链逐项抵消到 < 2^-126；而 fp32 累加噪声下限为 `0.5·ulp(最大部分和)`（m33 实测 min\|v\|≠0 ≈ 1e-6）⇒ 不可达 | 不建模（已知限制 8） |
| `1 + exp(-v)`（silu / sigmoid） | `exp(-v)` 可次正规，但 `1.0 + 次正规 == 1.0`（fp32 逐位） | 不建模（结果逐位相同，探针证明） |
| 量化器 `h = x(bf16)·halfScale` → fp4 Cast | `h` 可次正规，但次正规 ≪ 0.25（最低 E2M1 台阶）⇒ 仍编码到 **0 码** | 不建模（探针证明） |
| RMSNorm 平方和 `mean(x²)` | 可次正规，但 `+ eps(1e-6)` 完全主导 | 不建模（探针证明） |
| 残差加 / `bf16(...)` 写回 | 仅当输入本身落在 bf16 次正规区（\|x\| < 2^-126）才有差 | 不建模（已知限制 9） |

### 回归用例（防"以后有人把建模删了"）

`selfcheck.py` **check 10**（`check_router_ftz`）用一个**不依赖任何真实权重**的构造输入：
`x[:,0]=1`、`W[e,0]=L_e`（其余为 0）⇒ logits 精确等于设计值（单个 fp32 乘积，无累加噪声）。
一行里专家 0/1 为热点、专家 2 的 `exp = 1.2e-37`（**正好在 flush 线之上**）、专家 3..19 的
`exp ∈ [5.6e-42, 6.05e-39]`（**严格次正规**，且随 id 递增）。于是：

- **旧参考**（保留次正规）选出 `[0,1,2,19,18,17,16,15]`（最大的那几个次正规分数）⇒ **FAIL** 设备语义；
- **新参考**选出 `[0,1,2,3,4,5,6,7]`（次正规全 0 后按**小 id** 并列）⇒ **PASS**；
- 变异测试：只把三处 `ftz_f32` 调用删掉（保留 `ftz_f32` 定义本身），check 10 在
  "`router_topk` ids == 设备语义"这条上立刻 FAIL（三种变异的输出见
  `evidence/router_ftz_mutation.log`，其中第三种是 M26 N1 防漏项机制的复检）。

## 复现与自检

```bash
PY=/usr/local/python3.12.13/bin/python3.12
cd tools/golden
$PY gen_dataset.py                     # 重建 data/m1 data/m33（含 qaware/floor,qaware/ceil）
$PY gen_dataset.py --groups real       # 真实拓扑档（E=512/topK=10，1.25 GiB，按需）
$PY selfcheck.py                       # 全部 PASS 即一致（1-8 项 + check 10）
$PY selfcheck.py --real                # 额外生成并校验真实档
$PY selfcheck.py --m13-dump DIR        # 可选：与 m13 真机 dump 对拍（mode 0）
$PY selfcheck.py --env-ok              # 显式认当前数值栈（见下「环境识别（M75）」）

# 可选取证（M46；只读，不写任何 dump/数据集）
$PY evidence/router_ftz_device_repro.py --dump <m22_router512 dumpdir> \
      --w <router_weight.bin> --x-seed 5 --x-sha256 <dump 记录的 x sha256>
$PY evidence/ftz_reachability_audit.py
```

### 环境识别（M75）

**问题**：本目录对 `data/**.bin` 的判定是**逐位**的，而参考链里 fp32 GEMM 的累加次序是 numpy 所链
BLAS 的属性 ⇒ 换一条数值栈就会翻转极少数 bf16 码，**代码一行没改而判定项变红**；原先它与
「`.bin` 真漂移」给出同一个 `[FAIL]` + rc=1，读者分不出两类原因（定性见 M74 的勘查结论，本 mission
只做「让信号可区分」）。

**实测幅度**（2026-09-26 本 commit，`data/` 一个字节没动，只换解释器；判据 = 逐元素比 bf16 码）：

| 解释器（numpy / BLAS） | m33 `routed_output` | m33 `shared_output` | m33 `moe_output` |
|---|---|---|---|
| pin：`/usr/local/python3.12.13/bin/python3.12`（2.5.1 / scipy-openblas 0.3.33.112.0） | 0/84480 | 0/84480 | 0/84480 |
| 仓里基线 venv：`/workspace/venvs/baseline/bin/python`（1.26.4 / openblas64 0.3.23.dev） | 17/84480（最大 2 ulp） | 15/84480（最大 4 ulp） | 11/84480（最大 2 ulp） |

（m1 三张在两栈下都是 0/2560——不是"m1 免疫"，是 2560 个元素按 0.02 % 的越界率期望不到。）

**做法**：`selfcheck.py` 在任何数据集判据之前打印两行 `env:`（解释器 / numpy / BLAS），与上面 pin 的栈
比对，然后**只改退出码，不改任何判据**：

| 情形 | 输出（节选） | 判定项失败时 rc |
|---|---|---|
| 读到 pin 的栈 | `[ENV] canonical numeric environment: numpy 2.5.1 / scipy-openblas 0.3.33.112.0 …` | **1**（真漂移/回归，照旧） |
| 读到别的栈 | `[ENV] non-canonical numeric environment: numpy 1.26.4 != 2.5.1; BLAS openblas64 != scipy-openblas` | **3**（环境不明 ⇒ 不可归因；stderr 末尾给出复跑命令） |
| 别的栈 + `--env-ok` | `[ENV] --env-ok: this run declares its numeric stack authoritative …` | **1** |

- 两条 `env:` 行与 `[ENV]` 提示**不计入** `判定项 / guard` 计数（只披露、不判任何东西）⇒ pin 的栈上
  基线仍是 `644 判定项 + 6 guard 项`。
- **判据一个字未放宽**：负向对照——在 `/tmp` 副本里把 `m33/routed_output.bin` 翻 1 个 bit，canonical 栈
  仍 `[FAIL] … rc=1`，真漂移照抓，不会被环境识别豁免。
- 失败点落位：**check 7** 的 float-golden 从 bin 重算循环（`--no-regen --groups m33` 的第一条 FAIL 是
  `m33: recomputed 'routed_output' matches stored bin bit-exactly`）。**check 6 与本现象无关**——那是与
  `m13_moe_layer/check_ref.py` 两个独立量化器的逐字节对拍。不带 `--no-regen` 时第一条 FAIL 更靠前，
  是 `m1/manifest.json deterministic regeneration`（重生成 bin 的 sha256 随 BLAS 变）。
- 命令 → 输出/rc 的完整转录：`evidence/selfcheck_env_discrimination.log`。
- 数值环境变更属于「上游变更不自动流入」那一类（见文末「已知限制」10）。

自检覆盖：数值基础（E2M1/E8M0/官方规范及其角落与 legacy 规则/解析界/bf16）、与 **m13 独立量化器逐字节一致**
（floor ↔ `quant_mxfp4_hw`、ceil ↔ `quant_mxfp4_f32`，后者在真机上验证过设备字节）、数据集
**逐字节确定性重建**（含描述子与 manifest）、float golden 由 bin 重算逐位一致、层链 golden
（`res1/res2` 逐位、`rms(y/γ)==1`、`x_norm1 != x`）、量化感知 golden 全链重算逐位一致 +
预算 0 违反 + manifest 里的预算可复现（含 **key 集判定项**：`measured` 15 项 / `ceiling` 9 项 /
`source_budget` 4 项 / `f32_chain_gap` 4 项，少一项即 FAIL——这条是为修 reviewer N1 那条
"层链条目被静默跳过"而加的）、跨档口径一致（张量集/dtype/形状模式/规范集）、**router fp32 FTZ
建模**（check 10，M46：构造次正规输入，断言旧参考 FAIL / 新参考 PASS，删掉建模即 FAIL）。

**判定项与 guard 分栏**（docs/17 §2.1）：恒真/自洽类（harness 与元数据不变量）单独打印 `[GUARD]`
并单独计数，不进判定项总数：`1×` qaware 默认规范 = floor（官方序列）、每档 `1×` manifest
format_version、每档 `1×` floor 规则已生成、跨档 `1×` floor 规则在列。运行末尾会打印
`all selfchecks passed — N 判定项 + M guard 项`。**M46 修前/修后（本 mission 两次实测，同一份
已发布数据集）**：

| 命令 | 修前（M26 合并版 `55f5e18`） | 修后（M46） | 增量 |
|---|---|---|---|
| `--groups m1,m33` | **635 + 6** | **644 + 6** | +9（check 10） |
| `--groups m1,m33 --real` | **929 + 8** | **938 + 8** | +9（check 10） |
| `--groups m1,m33 --no-regen --m13-dump DIR` | **377 + 6** | **386 + 6** | +9（check 10） |

三档**全是 PASS**（0 FAIL），差异只有新增的 check 10 那 9 条 —— 即冻结的既有判据一条未变。
其中 check 9 的设备对拍覆盖两档 × 2 组非有限角落之外的字节/位级比对与判据可用性。
（M75 加的两行 `env:` 与 `[ENV]` 提示同样**不进任何计数**——它们只披露环境、不判任何东西，
故 `644 + 6` 这个 M46 读数在 M75 的树上原样保持，见「环境识别（M75）」。）

确定性：权重/激活由 `np.random.default_rng(seed)` 生成（m1 seed=20260926、m33 seed=20260927、
real seed=20260928）。**新增张量的 RNG 抽样一律追加在原有抽样之后**，因此 **M26 时点**已发布的
m1/m33 旧 `.bin` 未被重写（判据与数值栈限定见上文「零回归」那条；`.json` 描述子新增 `sha256`
字段、`manifest.json` 新增张量与区块——不影响只按文件名读 bin 的消费者，如 m13）。

真实档的 bin 不入库，`selfcheck.py --real` 会现场生成并做同一套校验（`--no-regen` 可跳过重复
生成）；`data/real/*/manifest.json` 里的 `sha256` 是权威指纹，重放生成后 manifest 逐字节一致即
等价于"全部 bin 一致"。

## 证据归档（`evidence/`）

| 文件 | 内容 |
|---|---|
| `selfcheck_run.log` | **M26 归档（未改动）**：`selfcheck.py --data data --groups m1,m33` 全量 PASS（**635 判定 + 6 guard**） |
| `selfcheck_run_m46.log` | **M46**：同命令 **644 判定 + 6 guard**（= M26 基线 + check 10 的 9 条，0 FAIL） |
| `m13_device_crosscheck.log` | **M26 归档（未改动）**：`selfcheck.py --m13-dump` 设备对拍（真机 dump，mode 0，两档；**377 判定 + 6 guard**） |
| `m13_device_crosscheck_m46.log` | **M46**：同命令 **386 判定 + 6 guard**（+9，0 FAIL） |
| `m13_device_regression.log` | m13 kernel 在升级后数据集上重跑：683 PASS / 0 FAIL，且 140 个 dump 张量与升级前逐字节一致 |
| `m13_dataset_chain_m46.log` | **M46**：m13 kernel 数据集全链重跑（**682 判据行 = 678 PASS + 4 routing 一致性**，rc=0，`===== ALL PASS =====`）+ 144 个 dump/meta 与 M26 批次逐字节一致 + m13 numpy `check_ref.py` **118 判定 + 32 参考项全 PASS** |
| `real_tier_gen.log` | 真实档生成摘要（2m12s 空载 / 2m55s 有并发时；1.25 GiB）+ 全部张量 sha256 清单（无 bin 也可复核）|
| `real_tier_selfcheck.log` | **M26 归档（未改动）**：`selfcheck.py --real`（**929 判定 + 8 guard**）|
| `real_tier_selfcheck_m46.log` | **M46**：`--real`（真实档生成 + 逐字节重放 + 全链重算 + 预算校验）**938 判定 + 8 guard**（+9，0 FAIL）|
| `m13_dump_sha256.txt` | **对拍取证**：140 个 m13 dump 的 sha256（+4 个 meta + accept log）+ 被对拍数据集的 `x`/`x_sorted` sha256 + **m13 源码**的 sha256 与 git blob（base commit 7880e63）+ 生成命令 —— 使"那版源码 + 那批 dump"的对应关系在 m13 被 M32 改动后仍可验证 |
| `router_ftz_device_repro.{py,log}` | **M46 主证据**：对设备 m22 dump 自复现 router FTZ（旧参考 20/4097 行、125 槽位不同、权重最大 96 bf16 ulp；建模后 **0/4097**、最大 1 ulp；FTZ 落点 A/B/C/D 口径与阈值扫描表）+ 重生成 `x` 的 sha256 与设备 dump 对应关系 |
| `router_ftz_device_repro_pre_m46.log` | 同一脚本在 **M26 合并版 module** 上的输出：第二行（本 module 的 `router_topk`）与旧口径一样是 20 行 ⇒ 修前日志 |
| `router_ftz_device_sha256.txt` | 上述复现用到的设备 dump / 真实权重 / M42 finding 与日志的 sha256 + wt-42 分支提交 + 与 M42 记载数字的逐项对照 |
| `ftz_reachability_audit.{py,log}` | **M46 排查证据**：参考链上 fp32 乘法/相加/归约路径的次正规可达性逐张量计数（m1/m33）+ 三条"可达但不可观测"的构造探针 + 不可建模项的量级论证 |
| `router_ftz_mutation.log` | **M46 回归防护取证**：① `ftz_f32` 换恒等 ② 只删三处 `ftz_f32` 调用 ③ `deviation_report` 静默漏 `res2`（M26 N1 防漏项）——三种变异各自被 check 10 / key 集判定项抓住 |
| `numeric_env_probe.py` + `selfcheck_env_discrimination.log` | **M75 环境识别取证**：`numeric_env_probe.py` 是只读复算探针（bf16 码漂移 + qaware `measured.max_abs` 容差，可原样重放）；日志是命令 → 输出/rc 全程转录 —— pin 的解释器（canonical，rc=0）/ 基线 venv（non-canonical，FAIL rc=3）/ 基线 venv + `--env-ok`（rc=1）/ `/tmp` 副本里给 `m33/routed_output.bin` 翻 1 bit 后在两栈下的结果（canonical rc=1 = 真漂移照抓）/ 未知 `--groups` 值（rc=1，不当成环境问题）/ 两栈下的逐元素读数 |

> **上游变更不会自动流入**（M26 先例，M46 重申，M75 扩类）：本目录是**冻结产物**——kernel / 官方
> 实现一侧的行为变化（含任何与 FTZ 有关的后续变化）**不会自动反映**到这里的参考与数据集。
> **数值环境变更（numpy / BLAS 升级）同属这一类**（见「环境识别（M75）」）：逐位判据会把
> 另一条 BLAS 的 fp32 末位差读成 FAIL。每次上游行为变化都必须由一条 mission 显式重跑本目录的
> 判据、说明是否重生成 `.bin`，并在 `evidence/` 留证（`real` 档仍只入库 manifest + sha256 + 预算；
> 已发布 `.bin` 不重写）。

## 已知限制 / 后续

**M90 已知边界：§1.3 三分表被抄进三处（相关性转写风险，部分消）** —— 同一张表同时出现在本 README、
`moe_real_accept.py` 的 docstring、`m15_layer_loop/check_moe_ref.py` 的 docstring。
**已被自动守卫覆盖的部分**：③ 那一行 —— 两层都在 `run` 每次跑的那条 `§1.3 ③ 漂移自检` 里：
① 运行时 ③ 判据 id ↔ 声明常量 `CHAIN_CRIT_TAGS`；② **真的去读那三个文件**，解析 HTML 注释形式的
标记（每份文件恰好一处；本 README 的那处在上面「③ 的判据 id 集合」那条里，
作用域 `mainleg`、值 `res2,y_final`；设备路径那份的作用域 `devpath`、值 `-` 表示空集），
与运行时集合逐条比对。
R4 实测（`/tmp` 副本，`run --tier m1 --no-m22`）：改 README 标记 / 截断 docstring 标记 /
把设备路径标记写非空 / 删标记 / 重复标记 / 换作用域 / 改 `CHAIN_CRIT_TAGS`，**7 种变异全部**让
该判据 FAIL 且 `run` 退 1；恢复后回到 rc=0。
**仍未被机制覆盖的部分**：①/② 两行、以及「② 的点名覆盖判据名」、**标记周围的散文**（句子怎么描述）
—— 这些仍是人工同步（本 README 与两个脚本各自维护一份），改 §1.3 表时三处都要改；
`run` 的 ③ 自检不会为它们变红。**这是本 harness 的已知边界，不写成「三处不会漂」**。
（r2 复审正是抓到这条：`moe_real_accept.py` 的 docstring 一度把主腿 ③ 写成 0，而代码与 README 都是 2。）

1. **设备侧 `:413` halfScale 非有限覆盖**：**时点快照 —— 2026-09-26 / M32 时点的 device 源码上，
   m2/m5/m13 各 1 处覆盖点**（按 `docs/05 §11.5`：这种读数带命令 + 时点，不写成"已无差异"式恒真句）。
   可复算计数（`selfcheck.py` check 3 每跑一次都现场打印这几栏；本 commit 实跑读数：
   `Duplicate(nanRegTensor, NAN_CUSTOMIZATION)` 每份 **1**、`NAN_CUSTOMIZATION` 每份 **3**、
   `0x7F81`（大写）每份 **3**、`0x7f81`（小写）每份 **1**）。
   m2/m5/m13 三份 `.asc` 的 `MxQuantComputeScale` 内都有该句，写法是
   `Duplicate(nanRegTensor, NAN_CUSTOMIZATION)` + `constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;`
   —— **小写**，**大小写敏感的 `grep 0x7F81` 匹配不到它**（该 grep 的 3 处命中都不在设备路径上，
   且**不做逐文件分解**：那种静态分解正是会过期的东西 —— `96e7e96` 那版曾写出这样的分解、由 `daf3344` 删除）⇒ **含 Inf** 的组设备与官方都给 0
   （`Mul(±Inf, NaN) = NaN → Cast = 0`），**NaN 组**两边同样都是 0。此前这里写「缺该句、含 Inf 组给 ±6、
   M32 在修」，那是**修复前**的记录；本目录的 golden 按官方实现，设备侧现已同一口径。
   数据集里仍没有 Inf/退化组（不影响对拍）。
2. **combine 段的 1 ulp 差**：设备在 f32 折叠/相加，参考用 f64；需要逐位对齐时可按 `w_tk_packed`
   协议用 f32 复算。
3. **真实档不入库**：1.25 GiB 不适合入库；manifest 的 sha256 是替代物。若 CI 需要，建议挂
   `selfcheck.py --real`（几分钟）而非常驻二进制。
4. **层链只覆盖 MoE block**：`x_norm1/res1/res2/y_final` 给的是 m6#1/m6#2 的 golden，未含 GDN、
   attention、in_proj/out_proj（属其它 mission 的算件）。
5. **m13 仍自合成 gamma**：m13 host 侧用 `H_GenGamma`（∈[0.5,1.5]）与零残差，因此设备对拍只在
   MoE 段（S2-S9）有效；S1/S10 要端到端对齐需换用本目录 `x_res/gamma1/gamma2`（改动落在
   m13 的 scope，本 mission 只提供数据）。
6. **真实档 m=8**：为覆盖 512 专家/10 topK 与空槽位，token 数取小（golden 体积可控）；
   如需 prefill 形态的真实档可另加 spec（`gen_dataset.py` 的 `SPECS`）。
7. **未做 msprof 计时**：真实档生成耗时见 `evidence/real_tier_gen.log`，未做优化。
8. **（M46）累加链不建模逐级 FTZ**：router GEMV、专家 gate_up/down、combine 的 fp32 累加链在
   设备上是 cube/vector 逐级累加，本参考是 numpy fp32 求和——累加次序不同，**逐级 FTZ 无法在
   本参考里表达**。判据是"部分和落进次正规区"需要整条链抵消到 `< 2^-126`，而同一条链的 fp32
   舍入噪声下限是 `0.5·ulp(最大部分和)`（m33 全链实测 `min|v|≠0 ≈ 1e-6`）⇒ 在本数据族**不可达**；
   该链本身的不确定度已由 T3 界 / 量化预算 / `1.5×oracle + 0.02·|gold| + 5e-3` 容差覆盖。
   若将来出现"整条链都极小"的数据（\|x\|、\|w\| 都在 1e-30 量级），需要显式建模——目前无此数据。
9. **（M46）次正规输入依赖的路径未建模**：残差加 `f32(x)+f32(res)`、`bf16((xAdd·rstd)·gamma)`、
   `bf16` 写回等点位，只有**输入本身**落在 bf16 次正规区（`|x| < 2^-126`，bf16 次正规最小
   `2^-133`）时才会与设备分开；已发布数据集（m1/m33/real）与真实权重档实测该项为 0
   （`evidence/ftz_reachability_audit.log`）。对这类输入，本参考与设备可能有 1~2 个 bf16 subnormal
   码的差；判据侧已有容差覆盖，若要逐位对齐需先给一条真实出现次正规输入的数据。
10. **（M75）数值环境（numpy / BLAS）变更会让逐位判据翻转，且不会自动流入**：见「环境识别（M75）」
    与 `evidence/selfcheck_env_discrimination.log`。实测（2026-09-26，本 commit，`data/` 不变）：
    同一棵树用 pin 的解释器 `selfcheck.py` 得 `644 判定项 + 6 guard 项`、rc=0；同一棵树用
    `baseline_env` 那个 venv（numpy 1.26.4 / openblas64 0.3.23.dev）则 FAIL——`--no-regen`
    时第一条是 **check 7** 的 `m33: recomputed 'routed_output' matches stored bin bit-exactly`，
    另有 qaware `measured.max_abs`（容差 1e-9）4 条；幅度全在 fp32 末位（m33 三个输出共
    17/15/11 个码不同、最大 4 ulp；预算值 Δ≤2.4e-7）。**这不是数据坏了**，因此升级 numpy/BLAS 后
    必须显式重跑本目录并决策是否重生成，不能指望判据自己跟上。
