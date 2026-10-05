# M12：GDN 输出 RMSNormGated（vllm 语义，纯 AIV 核）

单 AIV 核实现 GDN 层递推输出 `o` 与输出门 `z` 的 RMSNormGated（hidden = 6144 = 48 value heads × 128，GDN decode m=1 为主，m 运行时任意、逐行循环；设备实测范围见 §校验方法与结果）：

```
per head h（48 组，各 128 维——vllm variance dim=-1）:
  var   = mean(o[h]²)              # fp32 累加，mean = sum/128
  rstd  = 1 / sqrt(var + 1e-6)     # eps 先进 rsqrt
  out   = bf16( ((o[h]·rstd)·gamma) · sigmoid(z[h]) )   # RNE
```

**norm 维度 = 128/head（48 组），不是整行 6144**——vllm 调用点 `qwen_gdn_linear_attn.py:855`
`self.norm(core_attn_out, z)` 处 `core_attn_out` 形状为 `[tokens, 48, 128]`，
`RMSNormGated(head_v_dim=128, group_size=None)` 的 variance 取 `dim=-1`（128）。
gamma（vllm `norm.weight`）形状 `[128]`，48 头共享。

## vllm 语义对照表（逐字段）

| 字段 | vllm 源码 | 本核 |
|---|---|---|
| 构造 | `qwen_gdn_linear_attn.py:486-497` `RMSNormGated(head_v_dim, eps=config.rms_norm_eps, group_size=None, norm_before_gate=True, activation=config.output_gate_type)` | 同左语义硬化为编译期常量 |
| norm 维度 | `head_v_dim = linear_value_head_dim = 128`（config.json:81），48 head 各自归一 | HEAD=128 × HEADS=48 |
| eps | `base.py:38 eps = config.rms_norm_eps`；`config.json:114 rms_norm_eps = 1e-6` | EPS=1e-6f，先进 rsqrt（donor NR 的 `Adds(var, eps)`） |
| 门序 | `norm_before_gate=True`：先 `x_normed*weight`，后 `* act(z)`（`layernorm.py:285-296`） | `Mul(o,rstd)→Mul(·gamma)→Mul(·sigmoid(z))` 同序 |
| act | `config.json:105 output_gate_type="sigmoid"`（本模型为 sigmoid，勿抄 Qwen3-Next-80B 的 silu） | `sigmoid(z)=1/(1+exp(-z))` fp32（Muls/Exp/Adds/Div，m5 实证排布） |
| weight | `nn.Parameter(torch.empty(128))`，`forward_static` 内 `weight.float()` | gamma bf16 [128] 预转 fp32 驻 UB，48 头共享 |
| 输入 dtype | vllm 中 x=core_attn_out bf16 → `x.float()`；z bf16 → `z.float()` | o 直接 fp32（mega kernel 链 m4 出口契约，等价 `x.float()` 幂等）；z bf16→fp32 |
| 输出 dtype | `out.to(orig_dtype)`，模型 bf16 | bf16 RNE（`RoundMode::CAST_RINT`） |
| vllm-ascend 侧 | `ops/layernorm.py` RMSNormGated 复用上述 native 实现 | 语义一致，无第二口径 |

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m12_rmsnorm_gated/build -S m12_rmsnorm_gated -DCMAKE_BUILD_TYPE=Release
cmake --build m12_rmsnorm_gated/build -j4
./m12_rmsnorm_gated/build/m12_rmsnorm_gated
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入输出 layout

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| o | `[m, 6144]` | `float32` | 递推输出（GDN 链 m4 出口），行主序；m 运行时任意（行循环） |
| z | `[m, 6144]` | `bfloat16` | 输出门（in_proj bf16 段），行主序 |
| gamma | `[128]` | `bfloat16` | vllm `norm.weight`，48 头共享 |
| out | `[m, 6144]` | `bfloat16` | 输出：`(o·rstd)·gamma·sigmoid(z)`（RNE） |

行 stride：o 24576B（fp32）/ z、out 12288B（bf16），均 32B 整数倍；GM buffer 由 `aclrtMalloc` 对齐分配。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV-only，3510 声明规范），blockDim=1（单核，正确性优先）；m 运行时任意、逐行循环，行内 48 head 串行（head 间无依赖，后续可按 head 分核）。
- **donor 复用**：m6_rmsnorm（main 已合并）的 NormDonor 积木逐字复用——`CalculateSquareReduceSum`（128/head 走 `LessThanTwoVL` 分支：2×64 配对平方加+Reduce）与 `ComputeRstdNewtonRaphson`（NR rsqrt，`NEED_MAX/NEED_AVG_FACTOR`，mean 因子 1/128）。sigmoid 用 m5_swiglu_quant 实证排布（`Muls(-1)/Exp/Adds(+1)/Div`）。
- **per-head 流水**：每 head `SquareReduce→NR→GateY`，sumSq/rstd 恒在 UB 偏移 0——`LoadAlign`/`DIST_BRC_B32` 全 32B 对齐（M16 quirk），与 m6 实证排布一致，不引入 lane 抽 Broadcast。
- **数据通路**（全基础 API，无 TPipe/TBuf/TQue/AllocTensor/高阶 API）：

  ```
  GM --DataCopy(MTE2, 显式 DataCopyParams)--> UB(o 行 fp32 / z 行 bf16)
  UB(o) --Reg 平方和 fold--> UB(sumSq) --NR--> UB(rstd)
  UB(o,rstd,gammaF32,z) --Reg Mul/Exp/Div/Cast--> UB(out bf16) --DataCopy(MTE3)--> GM
  ```

  全部 UB buffer `LocalTensor(position, offset, size)` 编译期静态分配（偏移 32B 对齐）；计算全部 `__ubuf__` 裸指针 + RegBase 寄存器 API。
- **同步只用 BufferID**（`GetBufInternal/RlsBufInternal`，release 一律 drain `mode=true`；不用 set_flag/wait_flag，不挂 PIPE_S）：

  | BufferID | 交接 | 含义 |
  |---|---|---|
  | 0 (BUF_X) | MTE2 → V | o/z 行就绪（挡住下一行覆写） |
  | 1 (BUF_OUT) | V → MTE3 | out 行就绪（挡住下一行覆写） |
  | 2 (BUF_G) | MTE2 → V | gamma 就绪（kernel 开头一次） |

- **3510 quirk 遵守**（docs/05 §6）：一律显式 `DataCopyParams{1, blockLen32B, 0, 0}`（M16 实测元素计数重载栈垃圾 + blockCount>1 NZ 重排）；UB 偏移全 32B 对齐（M9）；无相邻 Cast（M10：每 chunk 一次 bf16→fp32（z）+ 一次 fp32→bf16（out），中间隔 sigmoid 四元组 + 三 Mul；gamma 预转各 Cast 间插 StoreAlign+LoadAlign）；Reg LoadAlign 地址 32B 对齐（M16）；无 `__VEC_SCOPE__` 外 LocalMemBar（M15，本核走 LessThanTwoVL 分支本就不含 LocalMemBar）。
- **UB 静态布局**（字节偏移，全部 32B 对齐，共 ~49.5KB « 248KB）：

  | 偏移 | 缓冲 | 大小 |
  |---|---|---|
  | 0 | UB_O（o 行 fp32） | 24576B |
  | 24576 | UB_Z（z 行 bf16） | 12288B |
  | 36864 | UB_Y（out 行 bf16） | 12288B |
  | 49152 | UB_GB（gamma 原始 bf16） | 256B |
  | 49408 | UB_GF（gamma fp32） | 512B |
  | 49920 | UB_TMP（reduce partials，备用） | 256B |
  | 50176 | UB_RED（sumSq） | 256B |
  | 50432 | UB_RSTD（rstd） | 256B |

- **mask 说明**：HEAD=128 = 2×64，恰为向量整数倍，列方向无尾块；行方向由运行时 m 循环、head 方向 48 次固定循环覆盖。MaskReg 语义保留在 lifted donor 函数内。

## 校验方法与结果

host 侧确定性随机生成（公式哈希，无 rand 状态）：o fp32 指数域 ∈ [77,177]（值域 2⁻⁵⁰..2⁵⁰，平方累加不溢出/不过小）；特殊行——`i%11==7` 全零 o 行（var=0 → rstd=1/√eps=1000 路径）、`i%11==5` 全行同值；z 默认 |z| ∈ [2⁻⁷,8]，特殊行——`i%7==3` z=+8192（sigmoid 饱和 1.0）、`i%7==5` z=−8192（exp 上溢 → sigmoid 精确 0）、`i%13==4` z≡0（sigmoid=0.5）；gamma 含 ±1.0/2.0 精确幂与随机值。特殊行模数交叠（如 row 73 = 零 o 行 + z 饱和行）由 m=256 用例自然覆盖。

校验链（四层，含 m 包络设备档）：

1. **NPU kernel vs 内建 C 参考**（numpy 语义逐 head 移植：fp32 累加、`((o·rstd)·gamma)·sigmoid(z)` 乘法次序、RNE bf16）：`m ∈ {1,2,17,64,256} × seed ∈ {0..4}` 共 **25 组全部 PASS**（含任务书 GDN decode m=1），进程退出码 0，连续运行稳定。out 逐行最大相对误差 ≤ 7.6e-3（≤2 个 bf16 ulp，容差 1e-2）。
2. **C 参考 vs numpy 参考**（`check_ref.py`，fp32 累加 + 逐行一致判定）：抽样 m∈{17,256} 用例全部在容差内（max rel 7.6e-3）。
3. **NPU kernel vs numpy 参考**：`M12_DUMP=1` 落盘 device 输出后由同一脚本比对，抽样 m∈{1,17,64,256} 用例全部通过（max rel 7.5e-3）。
4. **m 包络设备档（M150，`evidence/m4097_envelope/`）**：`dump <m> <seed>` 逐档在设备上单次启动并落盘 device 输出，再用 `check_ref.py` 对拍：m=257、m=4097（seed 0）device vs numpy 均 True（max rel 6.33e-3 / 7.75e-3），负向对照（数据侧翻转 1 个 bf16 gamma 元素）变红。设备实测 m 范围 = {1,2,17,64,256,257,4097}。

交叉验证用法（在 build/dump 之类的临时目录运行，避免污染仓库）：

```bash
cd m12_rmsnorm_gated/build && mkdir -p dump && cd dump
../m12_rmsnorm_gated dump 256 0                # 设备上跑 m=256 档，落盘 o/z/gamma、C 参考 y、device y_device
/usr/local/python3.12.13/bin/python3 ../../check_ref.py 256 0
# 其他 m 档同理（如 ../m12_rmsnorm_gated dump 4097 0）；dump 现为设备档。
# M12_DUMP=1 ../m12_rmsnorm_gated 仍可跑默认 ms[] 矩阵并落盘 device 输出。
```

## 已知限制 / 后续

- 单核正确性优先：未做 48 head 多 AIV 切分（head 间天然独立，后续可按 head stride 分核 + 双缓冲流水）；未做 GM 预取。
- m 的上界：kernel 无编译期 m 上界（运行期 m 行循环）；设备实测档 = {1,2,17,64,256,257,4097}（M150，见 `evidence/m4097_envelope/README.md`），m∈(257,4097) 的中间值与 >4097 未逐个实测。HEAD 固定 128（vllm head_v_dim），heads=48 由 HIDDEN/HEAD 编译期固定，扩展需改常量。
- **3510 向量单元 FTZ（flush-to-zero）**：fp32 次正规结果（|x| < 1.18e-38）被冲零，host/numpy 保次正规——深尾带（|z| ∈ [17,88] 使 sigmoid 落次正规区，或等价小积）两侧舍入不同。测试生成规避（默认 |z|≤8、饱和行直接 ±8192 取精确 {0,1}），对真实模型（z 为投影门控 ~O(1-10)）无影响；若后续需要深尾 bit 级一致，得换非 FTZ 实现（查表/分段），不建议。
- 输入避开 NaN（o 全零行 × 任意 z 不产 NaN：sigmoid 有界 ∈ {0}∪[3e-7,1]）；bf16 Inf/NaN z 未生成（NR 的 Inf guard 语义保留自 donor，同 m6）。
- out 误差上界 ~2 个 bf16 ulp：来自 fp32 累加次序（kernel 二分 fold vs 参考顺序求和）与 NR rsqrt 的 1-2 ulp，远优于 bf16 网格容差；若后续要求更严，可把 NR 换成精确 `1/sqrt`（donor `ReduceSumRstd` 的 `Div(1,Sqrt)` 路径）。
