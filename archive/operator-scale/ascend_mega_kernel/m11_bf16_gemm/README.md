# M11：bf16 GEMM 单 AIC 核（GDN in_proj / out_proj）

单 AIC 核 bf16 GEMM，填 GDN 层缺口算件（docs/12 §8）：`quantize_linear_attn=false` 不量化，in_proj
2560→16480 与 out_proj 6144→2560 两档 bf16 权重流（84.4MB + 31.5MB/层）。流水结构复用
m1_mxfp4_gemm（main 已合并），改非 MX 路径；数值与 fp32 参考（RNE round 到 bf16 网格）**逐位一致**
（精确整数域，非容差），并与 numpy fp32 参考交叉验证通过。

donor：
- `m1_mxfp4_gemm/m1_mxfp4_gemm.asc`（流水结构/同步/资源表，BufferID 编号一致）
- `/workspace/asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance/mmad.asc`
  （非 MX 路径参数：3510 分支 Nd2Nz/LoadData2DParamsV2/Mmad/Fixpipe）

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m11_bf16_gemm/build -S m11_bf16_gemm -DCMAKE_BUILD_TYPE=Release
cmake --build m11_bf16_gemm/build -j4
./m11_bf16_gemm/build/m11_bf16_gemm
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入 layout

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| A | `[m, K]` | `bfloat16_t` | 激活，行主序 |
| B | `[N, K]` | `bfloat16_t` | 权重原始 layout（**行主序 [N, K]**），kernel 内按 B^T 消费 |
| C | `[m, N]` | `bfloat16_t` | 输出，行主序 |

编译期模板实例化模型两个真实形状（任务书：K/N 形状两实例）：

| 场景 | K | N | B 权重量 |
|---|---|---|---|
| in_proj | 2560 | 16480 | 84.4MB |
| out_proj | 6144 | 2560 | 31.5MB |

## 与 m1（MXFP4 GEMM）的差异表

| 项 | m1（MXFP4） | m11（bf16） |
|---|---|---|
| A/B 数据类型 | fp4x2_e2m1（nibble 打包） | bfloat16_t |
| scale 流 | Dn2Nz（fp8 按 b16 视图） | **无**（非 MX 路径） |
| Nd2Nz `dValue` | `BASE_K/2`（字节） | `BASE_K`（元素，bf16） |
| L1→L0 | `LoadData` + `LoadData2DMxParams`（MX） | `LoadData` + `LoadData2DParamsV2`（非 MX） |
| K 轴 `kStep` | `BASE_K/64`（32B 段） | `BASE_K/16`（16 元素分形 = 32B） |
| 累加指令 | `MmadMx` | `Mmad`（L0C fp32 累加） |
| quantPre | F322BF16 | F322BF16（同） |
| BASE_M × BASE_K × BASE_N | 256 × 128 × 256 | **64 × 64 × 160** |
| m 范围 | 1..256 | 1..64（验收）；M146 设备实测 m=4097，见 §校验 |
| BufferID | 7 个（0-6） | 同编号 7 个 |
| params 初始化 | 成员逐一赋值 | 全字段 NSDMI 零初始化 + 无 NSDMI 成员显式清零（M16 加固，见下） |

tile 选型依据：L0B ping-pong 各 32KB（bf16 2B/元素）→ `BASE_K×BASE_N ≤ 16384` 元素；
`BASE_N` 须为 16 倍数且整除两形状 N（gcd(16480, 2560)=160）→ 取 160。

## Kernel 设计

- **核型**：`__global__ __cube__`，blockDim=1（单 AIC 核）；m 为运行时参数（1..64）。
- **数据通路**（全基础 API，无高阶 API / TPipe / TBuf / TQue）：
  `GM --DataCopy(Nd2Nz, MTE2)--> L1 --LoadData(LoadData2DParamsV2, MTE1)--> L0A/L0B --Mmad(M)--> L0C --Fixpipe(FIXP, F322BF16)--> GM`
- **地址自管理**：所有 L1/L0/L0C buffer 用 `LocalTensor(position, 字节偏移, 元素数)` 编译期静态分配
  （注意 `addr` 单位字节、`size` 单位元素，docs/05 §6 API 语义）。
- **同步只用 BufferID**（`Mutex::Lock/Unlock`，不用 set_flag/wait_flag，不挂 PIPE_S），编号与 m1 一致：

| BufferID | 交接 | 含义 |
|---|---|---|
| 0/1 (BUF_A0/1) | MTE2 → MTE1 | A1 ping/pong full/empty |
| 2/3 (BUF_B0/1) | MTE2 → MTE1 | B1 ping/pong |
| 4/5 (BUF_L0_0/1) | MTE1 → M | L0A/L0B ping/pong |
| 6 (BUF_L0C) | M → FIXP | L0C 累加结果就绪；M 侧在 tile 开头先 Lock，挡住下一 tile 的 Mmad 覆盖 |

  三级流水 MTE2(k+1) ∥ MTE1(k) ∥ M(k-1) 重叠（m1 同款结构）。
- **动态 m**：`curM` 尾块 mask——Nd2Nz 只搬 calcM 行、LoadData mStep 用 align16、Mmad m=calcM、
  Fixpipe mSize=curM（不越界写 C）。

### M16 latent 加固（code review 点）

- kernel 内**不存在任何元素计数 DataCopy 重载**（输入 Nd2Nz、输出 Fixpipe，均不经 DataCopyParams），
  M16 的栈垃圾路径在本 kernel 构造性缺席；
- 全部 params 结构体显式零初始化：`Nd2NzParams`/`LoadData2DParamsV2`/`MmadParams` 所有字段在
  CANN 9.1.0 均带 NSDMI 默认值 0，逐一显式赋值；`FixpipeParamsArch3510` 仅 `reluScalar`/`vectorRelu`/
  `deqScalar` 三个成员无默认初始化，均显式清零（其余字段 NSDMI 为 0/false，`params=Nz2NdParams`
  默认 ndNum=1 即单矩阵语义）。源码注释标注了各结构体的零初始化依据。

### 已知硬件 quirk（m=1，契约同 m1）

3510 上 `Nd2Nz/Dn2Nz` 行数为 1 时不做 NZ 切分（退化为 1D 拷贝），m=1 数据错位。处理：计算行数统一提升
`calcM = max(curM, 2)`，仅在 Fixpipe 用 curM 写出。**m=1 时 kernel 多读 A 的第 2 行**（不写出），
host 侧按 `max(m,2)` 行分配 A（本工程 host 代码已如此分配与生成）。

## 校验方法与结果

host 侧确定性生成（公式哈希，无 rand 状态），A/B 取 **[-8,8] 小整数**（bf16 精确表示）：乘积 ≤16 位
尾数，K≤6144 的累加和 |sum| ≤ 393216 < 2^24，**fp32 任意累加次序无舍入**——硬件 cube 累加、host fp32
参考、numpy fp32 参考三者逐位同值，F322BF16（RNE）后比对 bf16 比特即严格逐位校验（无容差）：

- 主校验：device C vs host fp32 参考（RNE round bf16 网格），逐比特相等；
- 交叉校验：`./m11_bf16_gemm dump <K> <N> <m>` 写出 a.bin/b.bin/c_device.bin，
  `python3 check_ref.py <K> <N> <m>` 以 numpy fp32 matmul 独立重算参考并比对（工具/接口同 m2）。

实测（Ascend950PR，CANN 9.1.0，全 10 例 bit-exact，进程退出码 0，全量约 10s）：

| 场景 | m=1 | m=2 | m=17 | m=33 | m=64 |
|---|---|---|---|---|---|
| in_proj K=2560 N=16480 | PASS | PASS | PASS | PASS | PASS |
| out_proj K=6144 N=2560 | PASS | PASS | PASS | PASS | PASS |

numpy 交叉校验（check_ref.py）：(2560,16480,1)、(2560,16480,33)、(6144,2560,1)、(6144,2560,33)
四档全部 `numpy fp32 ref == device C: True`。

**M146 追加档（m=4097，设备实测）**：in_proj (2560,16480) 与 out_proj (6144,2560) 两形状**单次启动**
（不改代码）均逐位一致，`check_ref.py` numpy 交叉校验均为 `numpy fp32 ref == device C: True`；
读数、锁内 npu-smi 与三文件哈希见 `evidence/m4097_envelope/README.md`。负向对照：把 in_proj 的 `b.bin`
翻 1 个 bf16 ULP 后 `check_ref.py` 变红（`mismatches: 624`，退出码 1；原始 `b.bin` sha256
`352bb009…f356` → 篡改 `cd68db3d…bc2c`），证明该判据能区分正确与错权重输出。

## 已知限制 / 后续

- 单核正确性优先，未做核间切分与性能优化；GDN decode m=1 为带宽瓶颈，层内集成时按 docs/12 §4
  的 L1 权重预取滚动窗规划多核切分。
- m > 64 走 `mLoop = CeilDiv(m, BASE_M)` 的多 tile 路径；M146 已设备实测 m=4097 两形状逐位通过
  （`evidence/m4097_envelope/README.md`）。设备已实测的 m = {1,2,17,33,64,4097}，更大 m 未逐个实测。
- K 尾块（K % 64 ≠ 0）与 N 尾块（N % 160 ≠ 0）不支持（静态断言）；模型两形状均可整除。
- 逐位校验建立在"精确整数域"数据上（任意累加次序无舍入）；真实 bf16 权重激活的一般域为容差校验，
  fp32 累加次序差异在 L0C 设计内，属预期行为。
