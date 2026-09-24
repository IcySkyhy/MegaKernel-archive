# mega_kernel_ascend / dev_tools

**开发过程辅助工具**，用来把本包装到旁边的昇腾（blade）工程里。
**不是** megakernel 运行时路径，`ModelBuilder` 不会 import 这里。

English: [README.md](README.md)

| 文件 | 作用 |
|------|------|
| `sync_td_headers.py` | 把 OpGraph 头文件 / golden 拷进对方 `TD/`，并打 GraphBuilder / 示例补丁 |
| `td_patches/` | 上述脚本注入的 C++ 模板 |

运行时 ops 测试见 [`../test/ops/README.md`](../test/ops/README.md)。

## OpGraph / TD 同步（`sync_td_headers.py`）

把剥离后的 OpGraph wire 头和平台 golden 拷到另一份昇腾工程的 `TD/`，再改示例 /
GraphBuilder / Qwen 源码去 load 这些 bin。补丁幂等，可重复跑。

### 目录约定

```text
<parent>/
  Triton-distributed-ascend/          # 本仓
  <other-ascend-project>/             # 目标工程
    TD/
      OpGraph.h  TileGraph.h  WireEndian.h
      two_matmul_opgraph.expected.bin
      lmhead_opgraph.expected.bin
    examples/operators/matmul/matmul.cpp           # -p A3
    examples/operators/matmulA5/matmulA5.cpp       # -p 950
    examples/models/qwen3-30B/qwen3_model.cpp     # buildLMHead
    blade/builder/include/graph/graph.h
    blade/builder/src/graph/graph.cpp
```

### 用法

在仓库根目录（`PYTHONPATH` 含 `python/`，或已安装本包）：

```bash
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers /path/to/other-ascend-project
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p A3
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p 950
```

| `-p` | Matmul 示例 | two_matmul / lmhead golden 源 |
|------|----------------|-------------------------------|
| `A3`（默认） | `examples/operators/matmul/matmul.cpp` | `*.expected.bin` |
| `950` | `examples/operators/matmulA5/matmulA5.cpp` | `*.950.expected.bin` |

Golden 在 [`../test/ops/`](../test/ops/)。

### 脚本会做什么

1. 写 `TD/OpGraph.h`、`TileGraph.h`、`WireEndian.h`
   - `namespace triton_dist` → `namespace mk`
   - WireEndian include 改成 `"WireEndian.h"`
   - 剥掉主机侧数据结构（`TYPES_TO_STRIP`），只留 serialize / 文件 I/O
2. 按平台把 `two_matmul` + `lmhead` golden 拷进 `TD/`（目标文件名固定）
3. 改 matmul cpp：`gb.buildOpGraph()` → `load_opgraph_from_file(<TD 绝对路径>)`
   - `-p 950` 还会改 GraphTensor `input` 注释、`mm1`→`mm2`、`512`→`65536`
4. 给 GraphBuilder（`graph.h` / `graph.cpp`）打上 `loadOpGraph` / `mergeOpGraph` /
   `findTensorId` 等
   - 跳过 `class GraphBuilder;` 前向声明
5. 把 `qwen3_model.cpp` 的 `buildLMHead` 改成 merge lmhead bin + `mergeOpGraph`
   （`add_rms_norm` StaticOpResult<3> 为 `y[M,H]`、`x[M,H]`、`rstd[M,1]`；
   `ranks>1` 仍在 merge 后走 `all_gather`）
6. 在 `examples/operators`、`examples/models`、`blade/builder` 的 CMakeLists 里，
   对含 `/blade/runtime/include` 的行在下方插入 `/TD` 行

打进 cpp 的 bin 路径是目标工程 `TD/*.bin` 的**绝对路径**，与编译时 cwd 无关。

### 补丁模板（`td_patches/`）

| 文件 | 作用 |
|------|------|
| `graph_builder_methods.h.inc` | GraphBuilder 公有声明 |
| `graph_builder_private.h.inc` | 私有辅助 |
| `graph_builder_methods.cpp.inc` | 类外 `loadOpGraph` / `mergeOpGraph` |
| `buildLMHead.cpp.inc` | 替换 `buildLMHead` 函数体 |

占位符：`__GB__` → `GraphBuilder`，`__CLASS__` → 模型类名，
`__LMHEAD_BIN_PATH__` → lmhead bin 绝对路径。

### 对昇腾工程的要求

- GraphBuilder 补丁用到的成员：`g_`、`tensorIdByName_`、`fusionGroups_`
- `GraphTensor::id()`，用来接 `mlp_output` / `residual`
- `buildLMHead` 用到的模型字段：`gb_`、`opsArgs_.ranks` / `rankId`、
  `vocabSize_`、`weightCache_`、`finalNormOutput_`、`lmHeadMatmulOutput_`、
  `allGatherOutput_`
- 产品路径：**merge** `TD/lmhead_opgraph.expected.bin`。bin 必须按 Ascend
  `ops::` 导出 `add_rms_norm` **StaticOpResult&lt;3&gt;**：
  **`y[M,H]`**、**`x[M,H]`**、**`rstd[M,1]`**。tiling 的 dimExtents 来自
  `op.outputs` → `desc.shape`。`matmul_lmhead` 和 `finalNormOutput_` 用
  **`y`**（`outputs[0]`）。`mergeOpGraph` 对每个新 op 调
  `inferOpTilingFromSchema`（与 `ops::` 同一套 schema tiling）。`ranks>1`
  仍在 merge 后跑 Ascend `ops::all_gather`。

可选路径缺失（例如没有 `qwen3_model.cpp`）会打日志并跳过；
当前 `-p` 所需的 matmul cpp 缺失会直接失败。
