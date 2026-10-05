# M6：Add + RMSNorm 融合（层边界残差归一化，纯 AIV 核）

单 AIV 核融合计算层边界数据流（hidden = 2560，bf16）：

```
xAdd   = f32(x) + f32(residual)          # fp32 寄存器残差加
resOut = xAdd                            # fp32 写回 GM（残差出口）
rstd   = 1 / sqrt(mean(xAdd²) + 1e-6)    # fp32，mean = sum/2560
y      = bf16((xAdd·rstd)·gamma)         # bf16 写回 GM
```

输出两个：`y [m,2560] bf16`（归一化结果）与 `resOut [m,2560] fp32`（残差出口的 fp32 累加值）。`m` 为运行时参数（1..256），逐行循环；gamma `[2560] bf16`。与 numpy 参考（fp32 累加）逐行一致（y 容差 bf16 网格 1e-2 相对，resOut 逐位一致），**25 组用例（m ∈ {1,2,17,128,256} × seed ∈ {0..4}，含任务书要求的 m=1 与 m=256）全部 PASS**。

donor：`/workspace/ops-nn/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h` 的 Compute 数据流（①CalculateXAdd 残差加（fp32）→ 写回 residual xOut 且 UB 留 xFp32；②CalculateSquareReduceSum 二分 fold 平方和；③ComputeRstdNewtonRaphson NR rsqrt；④CalculateY y=(x·rstd)·gamma→bf16）与共享积木 `norm/norm_common/op_kernel/reduce_common_regbase{,_part1,_part2}.h`。donor 被禁点仅 wrapper 的 TPipe/TQue（EnQue/DeQue → BufferID）；arch35 文件零 SetFlag、LocalMemBar 白名单内——本工程同样遵守。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m6_rmsnorm/build -S m6_rmsnorm -DCMAKE_BUILD_TYPE=Release
cmake --build m6_rmsnorm/build -j4
./m6_rmsnorm/build/m6_rmsnorm
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入输出 layout

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | `[m, 2560]` | `bfloat16` | 激活输入，行主序；m 运行时 1..256 |
| residual | `[m, 2560]` | `bfloat16` | 残差输入，行主序 |
| gamma | `[2560]` | `bfloat16` | 缩放向量 |
| y | `[m, 2560]` | `bfloat16` | 输出：`(xAdd·rstd)·gamma`（RNE） |
| resOut | `[m, 2560]` | `float32` | 输出：残差加 `f32(x)+f32(residual)`（fp32 出口） |

行 stride 分别为 5120B（bf16）/ 10240B（fp32），均 32B 整数倍；GM buffer 由 `aclrtMalloc` 对齐分配。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV-only，3510 声明规范），blockDim=1（单核，正确性优先）；m 运行时 1..256，kernel 内逐行循环，行间无依赖。
- **数据通路**（全基础 API，无 TPipe / TBuf / TQue / AllocTensor / 高阶 API）：

  ```
  GM --DataCopy(MTE2)--> UB(x,res 行)
  UB(x,res) --Reg LoadAlign/Cast/Add--> xFp32 --Reg StoreAlign--> UB(xFp32)
  UB(xFp32) --DataCopy(MTE3)--> GM(resOut)                     # 残差出口
  UB(xFp32) --Reg 二分 fold Reduce--> UB(sumSq) --NR rsqrt--> UB(rstd)
  UB(xFp32,rstd,gammaF32) --Reg Mul/Mul/Cast--> UB(y) --DataCopy(MTE3)--> GM
  ```

  全部 UB buffer 用 `LocalTensor(position, offset, size)` 编译期静态分配（偏移 32B 对齐，供 DataCopy）；计算全部 `__ubuf__` 裸指针 + RegBase 寄存器 API。
- **计算核心 lift donor**：`CalculateSquareReduceSum`（二分 fold：2560 → 2×1280 配对平方加 → 20 个 chunk partial → 归并）、`ComputeRstdNewtonRaphson`（sqrt 初值 + 两轮 NR，`NEED_MAX/NEED_AVG_FACTOR`）逐字 lift，仅适配 CANN 9.1.0 Reg API（`ReduceSum→Reduce<ReduceType::SUM>`、reg↔UB `DataCopy→LoadAlign/StoreAlign`、`Mula→Mul+Add`、`CompareScalar→Compares`）。`CalculateXAdd`/`CalculateY` 按 donor `add_rms_norm_regbase.h` 排布（XAdd 的两个 bf16→fp32 Cast 之间隔一次 LoadAlign）。
- **gamma 预转 fp32**：kernel 开头一次性把 gamma bf16 转 fp32 存 UB（2560×4B），之后每行 y 循环少 40 次 Cast，且每 chunk 只剩一次 fp32→bf16 Cast。
- **3510 连续 Cast bug 规避（docs/05 §6，M10 实证）**：任何两个 Cast 指令之间都隔着 LoadAlign/StoreAlign——XAdd 内 `Cast,LoadAlign,Cast`；gamma 预转与 y 循环内每 chunk `Cast` 后紧跟 `StoreAlign`、与下一 chunk 的 Cast 之间隔 `StoreAlign+LoadAlign+Mul+Mul`。全 kernel 无相邻 Cast。
- **同步只用 BufferID**（`GetBufInternal/RlsBufInternal` = get_buf/rls_buf；不用 set_flag/wait_flag，不挂 PIPE_S）。生产者 release 一律阻塞释放（`mode=false` = CANN `ASC_LOCK_BLOCK` 默认；`true` = `NON_BLOCK`；两种模式都等本 pipe 已发射指令落地，`true` 额外等此前同 id 的释放 ⇒ 更保守），保证 UB 写对跨 pipe 读可见：

  | BufferID | 交接 | 含义 |
  |---|---|---|
  | 0 (BUF_X) | MTE2 → V | x/res 行就绪（同时挡住下一行覆写） |
  | 1 (BUF_OUT) | V → MTE3 | xFp32(resOut)/y 行就绪（同时挡住下一行覆写） |
  | 2 (BUF_G) | MTE2 → V | gamma 就绪（kernel 开头一次） |

  核内 UB 障碍仅用白名单内的 `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>`（lifted 二分 fold 内，store→load 同址之间），与 BufferID 正交。
- **UB 静态布局**（字节偏移，全部 32B 对齐，共 ~41KB « 248KB）：

  | 偏移 | 缓冲 | 大小 |
  |---|---|---|
  | 0 | UB_X1（x 行 bf16） | 5120B |
  | 5120 | UB_X2（residual 行 bf16） | 5120B |
  | 10240 | UB_XF（xFp32 行） | 10240B |
  | 20480 | UB_Y（y 行 bf16） | 5120B |
  | 25600 | UB_GB（gamma 原始 bf16） | 5120B |
  | 30720 | UB_GF（gamma fp32） | 10240B |
  | 40960 | UB_TMP（reduce 二分 partials） | 256B |
  | 41216 | UB_RED（sumSq） | 256B |
  | 41472 | UB_RSTD（rstd） | 256B |

- **mask 说明**：hidden=2560 = 40×64，恰为向量整数倍，列方向无尾块；行方向由运行时 m 循环覆盖。MaskReg 语义保留在 lifted donor 函数内（scalar lane 的 `UpdateMask(1)`、`UpdateMask(20)`、`MaskPattern::VL1`）。

## 校验方法与结果

host 侧确定性随机生成（公式哈希，无 rand 状态）：元素符号/尾数随机、指数域 ∈ [77,177]（值域 2⁻⁵⁰..2⁵⁰，平方累加不溢出/不过小）；特殊行覆盖角落——`row%7==3` 精确抵消行（xAdd≡0，走 rstd=1/sqrt(eps) 路径）、`row%7==5` 全行同值；gamma 含 ±1.0/2.0 精确幂与随机值。

校验链（三层）：

1. **NPU kernel vs 内建 C 参考**（numpy 语义逐行移植：fp32 累加、`(x·rstd)·gamma` 乘法次序、RNE bf16）：`m ∈ {1,2,17,128,256} × seed ∈ {0..4}` 共 **25 组全部 PASS**，进程退出码 0，连续运行稳定。y 逐行最大相对误差 ≤ 7.6e-3（≤2 个 bf16 ulp，容差 1e-2），resOut **逐位一致**（0 mismatch）。
2. **C 参考 vs numpy 参考**（`check_ref.py`，fp32 累加 + 逐行一致判定）：抽样 m∈{1,17,256} 用例 y 全部在容差内、resOut 逐位一致。
3. **NPU kernel vs numpy 参考**：`M6_DUMP=1` 落盘 device 输出后由同一脚本比对，抽样用例全部通过（m=256 seed=0：y max rel 5.9e-3，resOut bit-exact）。

任务书两项要求：**m=1 PASS、m=256 PASS**（每个 seed 下均 PASS）。

交叉验证用法（在 build/dump 之类的临时目录运行，避免污染仓库）：

```bash
cd m6_rmsnorm/build && mkdir -p dump && cd dump
../m6_rmsnorm dump 256 0                 # 生成 m256_s0_*.bin（输入 + C 参考 y/resout）
M6_DUMP=1 ../m6_rmsnorm                  # 跑全部用例并落盘 *_y_device.bin / *_resout_device.bin
/usr/local/python3.12.13/bin/python3 ../../check_ref.py 256 0
```

## 已知限制 / 后续

- 单核正确性优先：未做多 AIV 行切分（行间天然独立，后续可按行 stride 分核 + 双缓冲流水）；未做 GM 预取。
- m > 256 未验证（任务书范围 1..256）；hidden 仅支持 2560（donor 函数本身泛型，扩展需补尾块 mask 路径——2560 无尾块，未走到 `LessThanVL/LessThanTwoVL` 分支）。
- 输入避开 bf16 Inf/NaN（NR 的 Inf guard 路径未生成数据覆盖，语义保留自 donor）；fp16 输入路径未做（模型为 bf16）。
- y 误差上界 ~2 个 bf16 ulp：来自 fp32 累加次序（kernel 二分 fold vs 参考顺序求和）与 NR rsqrt 的 1-2 ulp，远优于 bf16 网格容差；若后续要求更严，可把 NR 换成精确 `1/sqrt`（donor `ReduceSumRstd` 的 `Div(1,Sqrt)` 路径）。
