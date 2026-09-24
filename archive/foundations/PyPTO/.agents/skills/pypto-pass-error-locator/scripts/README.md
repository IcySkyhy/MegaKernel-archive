# PyPTO IR 分析工具

本目录包含用于分析 PyPTO IR（中间表示）文件的工具集，主要用于 Pass 精度问题定位和调试。

## 工具列表

### 1. get_op_info.py - OP 信息查询工具

**功能**：快速查询指定 OP 的详细信息，包括输入输出 tensor 的 shape、validshape、内存类型、属性等。

**使用场景**：
- 精度工具报错时，快速查看 OP 信息
- 与精度工具报错信息进行比对
- 定位 Pass 修改的 OP 信息

**基本用法**：

```bash
# 查询特定 OP 的信息（文本格式）
python3 get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic 10003

# 查询特定 OP 的信息（JSON 格式，便于脚本处理）
python3 get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic 10003 \
    --format json

# 列出所有 OP
python3 get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --list-ops
```

**命令行参数**：

| 参数 | 必需 | 说明 |
|------|------|------|
| `--ir-file` | 是 | IR 文件路径 |
| `--op-magic` | 否 | OP 的 magic ID（与 --list-ops 二选一） |
| `--list-ops` | 否 | 列出所有 OP（与 --op-magic 二选一） |
| `--format` | 否 | 输出格式，可选 `json` 或 `text`，默认 `text` |

**输出示例（文本格式）**：

```
=== Operation Information ===
OP Magic: 10003
Opcode: TILE_ADD
Line: 14

=== Output Tensor ===
Logic Tensor ID: 14
Raw Tensor ID: 14
Shape: [16, 128]
Valid Shape: [16, 128]
Data Type: DT_FP32
Memory Type: MEM_DEVICE_DDR::MEM_DEVICE_DDR
Subgraph ID: -1

=== Input Tensors ===
[0] Logic ID: 6, Raw ID: 10
    Shape: [16, 128]
    Valid Shape: [16, 128]
    Data Type: DT_FP32
    Memory Type: MEM_DEVICE_DDR
    Subgraph ID: -1

[1] Logic ID: 8, Raw ID: 12
    Shape: [16, 128]
    Valid Shape: [16, 128]
    Data Type: DT_FP32
    Memory Type: MEM_DEVICE_DDR
    Subgraph ID: -1

=== Attributes ===
op_attr_reverseOperand: 0

=== Context ===
Graph ID: -1
Scope ID: -1
```

### 2. ir_parser.py - IR 解析器

**功能**：提供 IR 文本文件的解析能力，支持解析 .tifwkgr 格式的 IR 文件。

**主要类**：

- `IRParser`：IR 文本格式解析器
- `IRFile`：IR 文件完整信息
- `Operation`：操作节点信息
- `RawTensor`：RAWTENSOR 信息
- `Cast`：INCAST/OUTCAST 信息
- `IRHeader`：IR 文件头信息

## 精度问题定位流程

### 步骤 1：运行精度验证

```bash
python3 test_case.py
```

如果出现精度问题，日志会输出类似以下信息：

```
[operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op TILE_ADD
op_id=10003, shape=<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32>
```

### 步骤 2：查询 OP 信息

```bash
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_latest/Pass_06_SplitReshape/After_06_SplitReshape_funcname.tifwkgr \
    --op-magic 10003 \
    --format json > op_info.json
```

### 步骤 3：比对分析

将 `op_info.json` 中的信息与精度工具报错信息进行比对：

- **Shape 比对**：检查 IR 中的 shape 与报错信息中的 shape 是否一致
- **属性比對**：检查 IR 中的属性是否完整
- **内存类型比对**：检查内存类型是否正确

### 步骤 4：定位问题

根据比对结果定位问题：

| 比对结果 | 可能原因 | 处理方法 |
|-----------|---------|---------|
| Shape 一致 | 问题不在 shape | 检查属性或其他方面 |
| Shape 不一致 | Pass 错误修改了 shape | 检查 Pass 的 shape 计算逻辑 |
| 属性缺失 | Pass 删除了属性 | 检查 Pass 的属性处理逻辑 |
| 内存类型错误 | Pass 错误分配了内存类型` | 检查 Pass 的内存分配逻辑 |

## 测试

运行测试用例：

```bash
cd pypto/.agents/skills/pypto-pass-error-locator
python3 tests/test_get_op_info.py
```

测试覆盖以下场景：

1. 查询存在的 OP
2. 查询不存在的 OP
3. 列出所有 OP
4. 文本格式输出
5. JSON 格式输出
6. 处理重复 OP ID 的文件
7. 处理 shape 不匹配的文件

## IR 文件格式参考

详细的 IR 文件格式说明请参考：
- `../references/ir-analysis-guide.md`

## 常见问题

### Q: 如何找到 IR 文件？

A: IR 文件位于 Pass 执行后的输出目录中：

```
output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr
```

### Q: OP magic 是什么？

A: OP magic 是操作节点的唯一标识符，格式为 `!` 后跟数字，如 `!10003`。在 IR 文件中，每个操作节点都有唯一的 OP magic。

### Q: 如何从精度工具日志中提取 OP magic？

A: 精度工具日志中会包含 OP 的信息，例如：

```
[operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op TILE_ADD
```

从日志中提取 opcode（如 `TILE_ADD`），然后使用 `--list-ops` 查找对应的 OP magic。
