# M1：MXFP4 GEMM 单 AIC 核移植

单 AIC 核 MXFP4 GEMM，直接消费模型原始 layout 的 nibble-packed MXFP4 权重，数值与 host double 参考**逐位一致**（容差 1e-2 内，实测 maxRelDiff = 0）。

donor：`/workspace/asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_mxfp4_basic_api_high_performance/mmad_mx.asc`（950PR 官方基础 API 样例，Nd2Nz + LoadData(MX) + Mmad + Fixpipe 全链路，本工程按其结构抄改）。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m1_mxfp4_gemm/build -S m1_mxfp4_gemm -DCMAKE_BUILD_TYPE=Release
cmake --build m1_mxfp4_gemm/build -j4
./m1_mxfp4_gemm/build/m1_mxfp4_gemm
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入 layout（与 Qwen3.8-Flash-Next-MXFP4 safetensors 原始排布一致）

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| A | `[m, K]` | `fp4x2_e2m1_t` | 激活，uint8 nibble 打包：`byte = lo \| (hi << 4)`，lo=偶数 k、hi=奇数 k |
| scaleA | `[m, K/32]` | `fp8_e8m0_t` | 每 32 个 K 元素共享 1 个 scale，`value = 2^(bits-127)` |
| B | `[N, K]` | `fp4x2_e2m1_t` | 权重原始 layout（**行主序 [N, K]**，每行 K/2 字节），kernel 内按 B^T 消费 |
| scaleB | `[N, K/32]` | `fp8_e8m0_t` | 同上 |
| C | `[m, N]` | `bfloat16_t` | 输出，行主序 |

e2m1 值表（bias=1）：`0, ±0.5, ±1, ±1.5, ±2, ±3, ±4`，`0x7/0xF` 为 NaN（生成侧避开）。

编译期模板实例化模型两个真实形状：

| 场景 | K | N |
|---|---|---|
| down_proj | 640 | 2560 |
| gate_up_proj | 2560 | 1280 |

## Kernel 设计

- **核型**：`__global__ __cube__`，blockDim=1（单 AIC 核）；m 为运行时参数（1..256）。
- **Tile（静态）**：baseM=256 × baseK=128 × baseN=256。L0A/L0B ping-pong 各 32KB（占满 64KB），L0C 256×256 fp32（占满 256KB）。
- **数据通路**（全基础 API，无高阶 API / TPipe / TBuf / TQue）：
  `GM --DataCopy(Nd2Nz 数据 + Dn2Nz scale, MTE2)--> L1 --LoadData(LoadData2DMxParams, MTE1)--> L0(MX) --MmadMx(M)--> L0C --Fixpipe(FIXP, F322BF16)--> GM`
- **地址自管理**：所有 L1/L0/L0C buffer 用 `LocalTensor(position, 字节偏移, 大小)` 编译期静态分配（资源表见 `m1_mxfp4_gemm.asc` 头部常量）。
- **同步只用 BufferID**（`Mutex::Lock/Unlock`，不用 set_flag/wait_flag，不挂 PIPE_S）：

| BufferID | 交接 | 含义 |
|---|---|---|
| 0/1 (BUF_A0/1) | MTE2 → MTE1 | A1+scaleA1 ping/pong full/empty |
| 2/3 (BUF_B0/1) | MTE2 → MTE1 | B1+scaleB1 ping/pong |
| 4/5 (BUF_L0_0/1) | MTE1 → M | L0A/L0B ping/pong |
| 6 (BUF_L0C) | M → FIXP | L0C 累加结果就绪；M 侧在 tile 开头先 Lock，挡住下一 tile 的 Mmad 覆盖 |

  该结构使 MTE2(k+2) ∥ MTE1(k+1) ∥ M(k) 三级流水重叠。实测 CANN 9.1.0 上 AIC 各 pipe（MTE2/MTE1/M/FIX）的 BufferID 获取/释放均正常（docs/05 §10.4 遗留项的部分实证）。
- **动态 m**：M 方向 tile 循环 + 尾块 `curM` mask——Nd2Nz/Dn2Nz 只搬 curM 行、LoadData mStep 用 align16、Mmad m=curM、Fixpipe mSize=curM（不越界写 C）。m∈[2,256] 全部按此路径验证通过。

### 已知硬件 quirk（m=1）

实测 3510 上 `Nd2Nz/Dn2Nz` 行数为 1 时不做 NZ 切分（退化为 1D 拷贝），A 行 64B 中的后 32B 落点错误，导致 m=1 结果错乱；m=2..15（同走单分形路径）正常。处理：计算行数统一提升 `calcM = max(curM, 2)`，仅在 Fixpipe 用 curM 写出。因此 **m=1 时 kernel 会多读 A/scale 的第 2 行**（不写出），host 侧按 `max(m,2)` 行分配 A/scaleA 即可（本工程 host 代码已如此分配）。

## 校验方法与结果

host 侧确定性随机输入（公式哈希，无 rand 状态），解包 MXFP4 为 double 做参考 matmul；参考值先 round 到 bf16 网格（与硬件 F322BF16 同网格），逐元素比对：

```
tol = 1e-2 × |expect| + 1e-3 × √K × max|A_i| × max|B_j|
```

（第二项为 fp32 累加在随机符号相消下的物理误差尺度；数据 scale 取 2^-1..2^2。）

实测（Ascend950PR，CANN 9.1.0）：

| 场景 | m=1 | m=2 | m=17 | m=128 | m=256 |
|---|---|---|---|---|---|
| down_proj K=640 N=2560 | PASS | PASS | PASS | PASS | PASS |
| gate_up K=2560 N=1280 | PASS | PASS | PASS | PASS | PASS |

全部 `maxRelDiff = 0.00e+00`（与 bf16 网格化的 double 参考逐位一致），进程退出码 0。

## 已知限制 / 后续

- 单核正确性优先，未做核间切分与性能优化（M1.5 的多核 grouped GEMM 在此基础上扩展）。
- m > 256 时 mLoop>1，代码路径已具备但未验证（验收只要求 1..256）。
- K 尾块（K % 128 ≠ 0）不支持（donor 同限制）；模型两形状 K=640/2560 均可整除。
- 输入生成避开了 e2m1 NaN 编码与 e8m0 的 0/0xFF；真实量化权重若含 NaN 需另行定义语义。
