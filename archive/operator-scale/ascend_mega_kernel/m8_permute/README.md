# M8：MoE 专家槽位重排 / 反重排（permute / unpermute，纯 AIV 核 ×2）

Mega kernel MoE block 纵向切片的重排段（docs/05-megakernel-design.md §5：decode 静态 schedule
按专家槽位划分，运行时按路由结果重排）。两个 `__vector__ __global__` AIV 核：

```
#1 permute   : x_sorted[i, :] = x[perm_src_token[i], :]        i ∈ [0, Σt_e)
#2 unpermute : out[t, :] = bf16( Σ_k fp32(bf16 w[t,k]) * fp32(y_sorted[inv[t,k], :]) )
```

- `Σt_e = m*topK` 运行时由每核自 GM `expert_token_counts` 标量读入求和（任务书要求）。
- 槽位布局与 golden 生成器 `tools/golden/moe_block_ref.py moe_permute` 完全一致：
  sorted slot 按（专家 id, token id）分组排列，`perm_expert/perm_src_token` 为 per-slot
  专家/源 token，`expert_token_counts[e]` 为专家 e 的槽位数。
- `inv[t,k]`（host 预计算）= token t 的第 k 个 topk 专家在 sorted 序列中的槽位；
  `w[t,k] = bf16_RNE(topk_weights[t,k])`（bf16 权重，donor probs 语义）。
- donor（docs/09 §2）：permute 主循环 lift 自 `gather_v2_simd_two_dim.h:148`
  （NoSplitColProcess 逐行 gather）；unpermute 语义对齐
  `moe_token_unpermute_with_routing_map_not_pad.h:21`（scatter+probs 加权折叠）——
  实现为按输出行 gather（每输出行 disjoint，免 GM 原子加）；折叠统一按 k 升序、与参考同序——
  topK=2 时由 fp32 加法交换律与任意 scatter 次序逐位等价，topK>2 的次序契约即 k 升序。

**校验结果：任务书 m=1（m1）与 m=33（m33）两组，permute 输出 vs golden x_sorted 逐位一致，
unpermute 输出 vs numpy 参考（gather/scatter+加权公式）逐位一致**（另加合成 y 变体，
共 4 项 bit-exact 判定/组）；`M8_STRESS=60` 压测 + 15 轮进程级复跑 0 失败。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m8_permute/build -S m8_permute -DCMAKE_BUILD_TYPE=Release
cmake --build m8_permute/build -j4
./m8_permute/build/m8_permute [tools/golden/data] [case ...]   # 默认 m1 m33
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。
环境变量：`M8_DUMP=1` 落盘 device 输出（供 check_ref.py）；`M8_STRESS=N` 每变体重复 N 次压测。

## 输入输出 layout

kernel #1 permute：

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | `[m, 2560]` | `bfloat16` | 激活输入，行主序 |
| perm_src_token | `[Σt_e]` | `int32` | 每个 sorted slot 的源 token（sortedIndices） |
| expert_token_counts | `[E]` | `int32` | 每专家槽位数（kernel 内求 Σt_e） |
| x_sorted | `[Σt_e, 2560]` | `bfloat16` | 输出：按专家分组重排后的激活 |

kernel #2 unpermute：

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| y_sorted | `[Σt_e, 2560]` | `bfloat16` | 专家计算输出（按槽位序） |
| out_token_slot (inv_slot) | `[m, topK]` | `int32` | 每 (token, rank) 的源槽位 |
| w_tk_packed | `[m, 16]` | `int32` | 低 16 位 = bf16 权重位（行距 64B，随 y 行 MTE2 拷入） |
| out | `[m, 2560]` | `bfloat16` | 输出：加权折叠写回 |

行 5120B（bf16）均 32B 整数倍；设计包络（docs/05 §5 decode 规模）：`m ≤ 64`、`topK ≤ 10`、
`Σt_e ≤ 640`、`E ≤ 512`（host 校验，UB 静态分配的前提）。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV-only，3510 声明规范）；行按核 stride 划分
  （`row = bid + r*nblk`），行间无依赖；blk = min(行数, 28)。
- **数据通路**（全基础 API，无 TPipe/TBuf/TQue/AllocTensor/高阶 API；无 Matmul）：

  ```
  #1: GM --DataCopy(MTE2)--> UB 行缓冲(4 级流水) --DataCopy(MTE3)--> GM   # 纯带宽操作
  #2: GM --DataCopy(MTE2)--> UB(y 行×topK + 权重行 64B, 双缓冲)
      UB --Reg LoadAlign/Cast/ShiftLefts/Mul/Add--> fp32 acc(寄存器)
      --Cast--> UB(out 行) --DataCopy(MTE3)--> GM
  ```

- **索引读取**：`GlobalTensor::GetValue` 标量 GM 读（m3_grouped_gemm countsGM 同款）——
  permute 读 `perm_src_token`/counts，unpermute 读 inv_slot；仅用于标量上下文/MTE2 地址。
- **topK 模板实例化**（1..10，host switch 分发）：k 链 `if constexpr` 全展开后所有下标为
  编译期常量——向量循环内禁止动态下标标量加载（bisect 实证 backend 报
  "Unsupported Inst must be hoisted"）。
- **权重送达（bisect 后的关键设计）**：`w[t,k]` bf16 位由 host 打包进 int32 低 16 位
  （`[m,16]` 行距 64B），随 y 行同批 MTE2 拷入 UB；kernel 内
  `LoadAlign<int32_t, DIST_BRC_B32>` 广播 + `ShiftLefts<<16` 位精确还原 fp32（m6 同款
  位技巧）。**kernel 内零标量权重读取**——标量→向量跨 pipe 数据流在 3510 上实测
  间歇性丢写/错值（~1/30 迭代），改为纯 MTE2→V 握手 + 全向量指令后 600+ 次压测 0 失败。
- **同步只用 BufferID**（`GetBufInternal/RlsBufInternal`，release 一律 drain mode=true）：

  | BufferID | 核 | 交接 | 含义 |
  |---|---|---|---|
  | 0-3 (BUF_P0+) | #1 | MTE2 → MTE3 | 行缓冲流水级（4 级 ping-pong） |
  | 0-1 (BUF_U0+) | #2 | MTE2 → V → MTE3 | stage 0/1 双缓冲（y 行+权重行 / out 行） |

- **3510 连续 Cast bug 规避**：#2 计算循环内任意两个 Cast 之间都隔着
  LoadAlign/Mul/Add/ShiftLefts，全 kernel 无相邻 Cast。
- **UB 静态布局**（字节偏移，全部 32B 对齐）：#1 = 4×5120B 行缓冲（20KB）；
  #2 stage 步长 56384B（y 行 10×5120 + out 5120 + 权重行 64）×2 = 110KB « 248KB。

## 校验方法与结果

三层校验链（任务书要求两组全部满足）：

1. **NPU kernel vs 内建 C 参考**（unpermute 折叠公式逐位移植，`-ffp-contract=off` 禁 FMA
   融合，与 kernel 独立 Mul/Add 同一 IEEE 序列）：`m1`/`m33` × {permute vs golden,
   unpermute-goldy, unpermute-synth} **全部逐位一致**，进程退出码 0。
2. **NPU kernel vs numpy 参考**（`check_ref.py`：同一公式 numpy 实现 + moe_block_ref
   RNE bf16 工具）：`M8_DUMP=1` 落盘后逐位比对，m1/m33 共 8 项 bit-exact 判定全过。
3. **golden 自洽**：golden x_sorted vs `x[perm_src_token]` 逐位一致（生成器语义确认）。

交叉验证用法（在临时目录运行，避免污染仓库）：

```bash
REPO=/workspace/ascend_mega_kernel/.tower/worktrees/wt-15   # 本工程所在 checkout 根
mkdir -p /tmp/m8_dump && cd /tmp/m8_dump
M8_DUMP=1 $REPO/m8_permute/build/m8_permute $REPO/tools/golden/data m1 m33
/usr/local/python3.12.13/bin/python3 $REPO/m8_permute/check_ref.py m1 /tmp/m8_dump
/usr/local/python3.12.13/bin/python3 $REPO/m8_permute/check_ref.py m33 /tmp/m8_dump
```

稳定性：`M8_STRESS=60`（每变体 60 次重复 launch+比对）+ 15 轮进程级复跑（m1+m33 全量）
0 失败；输出缓冲先行 0xCD 污染，防假 PASS。

## 已知限制 / 后续

- 行级并行正确性优先：未做列切分（2560 = 40×64 向量整数倍，单行单核 V 计算 ~µs 级）；
  未做 GM 预取深度调优。
- 包络外（m>64 / topK>10 / Σt_e>640）host 直接报错（UB 静态分配与 decode 设计前提）；
  更大规模需扩 UB 分块或改行-列二维切分。
- unpermute 输入 y_sorted 由上游 grouped GEMM#2 epilogue 产生（docs/09 §2 的
  "epilogue 直接 scatter"融合路线）——本核保持独立 phase，融合时 #2 可被吸收。
- permute 的 GM 标量索引读（GetValue 逐行）在 Σt_e=640 全量时约占 ~640 次标量读，
  后续可改 inv 协议为 UB 批量索引（MTE2 一次拷入）进一步掩盖。
