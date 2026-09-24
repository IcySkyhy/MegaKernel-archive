# PyPTO IR分析指导文档

## 文档目的

本文档旨在指导AI Agent理解和分析PyPTO的.tifwkgr中间表达（IR）文件，用于：
- 辅助问题定位和调试
- 检查Pass模块是否存在逻辑错误
- 验证数据流和内存访问的正确性
- 评估IR转换的合理性

## IR概述

### 什么是IR

IR（Intermediate Representation）是PyPTO编译器在各个Pass阶段生成的中间表达，用于表示计算图、数据流和内存布局。每个Pass都会生成Before和After两个IR文件，用于对比Pass执行前后的变化。

### IR的作用

1. **问题定位**：通过IR变化定位Pass执行中的错误
2. **逻辑验证**：检查数据流、内存访问、依赖关系的正确性
3. **性能分析**：评估优化Pass的效果
4. **调试辅助**：理解编译器内部状态和转换过程

## IR文件结构

### 文件格式

.tifwkgr文件采用文本格式，包含以下主要部分：

```
-------------
Function {函数名}[function_magic] {hash} {函数类型} {图类型} {
  RAWTENSOR声明
  INCAST/OUTCAST声明

  操作节点定义
}
```

### 文件头

```
Function TENSOR_TENSOR_update_kernel_loop_Unroll1_PATH0_hiddenfunc0_5[5] 311644491877735055 DYNAMIC_LOOP_PATH TILE_GRAPH {
```

**字段说明：**
- `Function`: 固定关键字
- `函数名`: 当前处理的函数名称，命名规则为 `{输入类型}_{输出类型}_{基础名称}[可选_loop]_Unroll{unroll值}_PATH{路径}_hiddenfunc{隐藏函数}_{函数id}`
  - `update_kernel_loop`：基础名称（若为循环函数，recorder 会在原始名称后附加 `_loop`，见 `recorder.cpp:67`）
  - `Unroll1`：unroll值，如前端设置 `unroll_list=[16, 1]`，IR文件名中会有 `Unroll1` 和 `Unroll16`
  - `PATH0`：循环中的分支路径，if/else 分别对应 PATH0 和 PATH1
  - `hiddenfunc0`：隐藏函数编号
  - `_5`：函数ID（来自 `GetFuncMagic()`，与方括号中的 function_magic 值相同）
- `[function_magic]`: 函数magic（唯一标识符），与函数名末尾的 `_{函数id}` 值相同，均来自 `GetFuncMagic()`
- `{hash}`: 函数hash值
- `{函数类型}`: 函数类型，如 DYNAMIC_LOOP_PATH
- `{图类型}`: TENSOR_GRAPH 或 TILE_GRAPH

**函数类型说明：**
- `STATIC`: 静态函数（编译时确定）
- `DYNAMIC`: 动态函数（运行时确定shape）
- `DYNAMIC_LOOP`: 动态循环函数（包含循环结构）
- `DYNAMIC_LOOP_PATH`: 动态循环路径函数（循环展开后的路径）
- `DYNAMIC_IF`: 动态条件分支函数
- `DYNAMIC_IF_PATH`: 动态条件分支路径函数

**图类型说明：**
- `TENSOR_GRAPH`: 高层张量图，表示原始的计算逻辑，未进行tiling优化
- `TILE_GRAPH`: 分块图，已进行tiling优化，包含内存访问细节

### 注释

```
/* /mnt/workspace/gitCode/cann/pypto/test_scatter_update.py:20 */
```

注释标记了原始Python代码的行号，用于IR与源代码的对应关系。

## IR元素详解

### 1. RAWTENSOR（原始张量）

#### 语法格式

```
RAWTENSOR[索引] <shape> @{编号}"{名称}"
```

#### 示例

```
RAWTENSOR[  0] <1 x 16 x DT_INT32> @10"TENSOR_3"
RAWTENSOR[  1] <8 x 128 x DT_FP32> @12"TENSOR_2"
RAWTENSOR[  2] <16 x 128 x DT_FP32> @14"TENSOR_1"
```

#### 字段说明

- **索引**: 张量的索引（0, 1, 2, ...）
- **shape**: 张量的形状和数据类型
  - 格式：`dim1 x dim2 x ... x DT_数据类型`
  - 数据类型：DT_INT32, DT_FP32, DT_FP16等
- **编号**: 张量的编号，唯一标识（@10, @12, @14等）
- **名称**: 张量的名称，可能为空

#### 分析要点

1. **索引唯一性**: 确保每个RAWTENSOR索引唯一
2. **形状合理性**: 检查shape的维度和大小是否合理
3. **数据类型**: 验证数据类型是否匹配计算需求

### 2. Tensor类型说明

PyPTO IR中存在两种tensor类型：

#### Logic Tensor（逻辑张量）

- **标识符**: `%` 开头，如 `%6`, `%84`, `%TENSOR_1`
- **含义**: 表示逻辑上的tensor，用于计算图中的数据流
- **特点**: 可以有名称或编号
- **作用**: 在IR中表示数据流和计算依赖

#### Raw Tensor（原始张量）

- **标识符**: `@` 开头，如 `@10`, `@50`, `@TENSOR_2`
- **含义**: 表示物理存储空间
- **特点**: 在RAWTENSOR声明中定义
- **作用**: 表示实际的内存分配

### 3. INCAST/OUTCAST（输入tensor/输出tensor）

#### 语法格式

```
INCAST[索引] <shape / valid_shape> %{logic tensor}@{raw tensor}#(子图ID) fromSlot[槽位列表]
OUTCAST[索引] <shape / valid_shape> %{logic tensor}@{raw tensor}#(子图ID) toSlot[槽位列表]
```

#### 示例

```
INCAST[  0]  <1 x 16 x DT_INT32 / 1 x 16 x DT_INT32> %6@10#(-1) fromSlot[2]
OUTCAST[  0]  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %14@16#(-1) toSlot[3]
```

#### 字段说明

- **索引**: 索引位置
- **shape**: 张量的形状
- **valid_shape**: 张量的有效形状
- **logic tensor**: 标识符（%6, %9, %12等）
- **raw tensor**: 标识符（@6, @9, @12等）
- **子图ID**: 子图标识符，用于标识tensor所属的子图
- **fromSlot/toSlot**: 槽位列表，用于追踪INCAST/OUTCAST的slot映射关系

### 4. 操作节点

#### 语法格式

```
<shape / valid_shape> %{输出logic tensor}@{输出raw tensor}#(子图ID){读内存类型}::{写内存类型} = !{op_id} {opcode}(g:{子图ID}, s:{作用域ID}) {输入参数} {属性}
```

#### 示例

```
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %84@50#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10022 TILE_VIEW(g:-1, s:-1) %12@14#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR from offset:[  0,  0] dynoffset:[  0,  0] to MEM_DEVICE_DDR dynvalidshape:[ 16,128]

<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %1@6#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10014 TILE_INDEX_OUTCAST(g:-1, s:-1) %88@52#(-1)MEM_UB::MEM_UB, %0@5#(-1)MEM_UB::MEM_UB, %84@50#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR #CACHE_MODE{PA_BSND} #PA_NZ_BLOCK_SIZE{1} #axis{0}

<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %76@17#(0)MEM_UB::MEM_UB = !10015 TILE_ADDS(g:0, s:-1) %87@52#(0)MEM_UB::MEM_UB #SCALAR{1.000000} #op_attr_reverseOperand{0}
```

#### 字段说明

- **shape**: 张量的形状
- **valid_shape**: 张量的有效形状
- **输出logic tensor**: 标识符（%6, %9, %12等）
- **输出raw tensor**: 标识符（@6, @9, @12等）
- **子图ID**: 子图标识符，用于标识tensor所属的子图
- **读内存类型::写内存类型**: 双冒号分隔的内存类型
  - 格式：`MEM_TYPE::MEM_TYPE`
  - 例如：`MEM_UB::MEM_UB`, `MEM_DEVICE_DDR::MEM_UB`
- **op_id**: 操作的唯一标识符（!10005, !10001等）
- **opcode**: 操作的名称
- **g:{子图ID}**: 操作所属的子图ID，用于图分区管理
  - 默认值：`NOT_IN_SUBGRAPH = -1`
  - 代码位置：`framework/src/interface/operation/operation.h` 中 `Operation::GetSubgraphID()` 返回 `subgraphID_` 成员变量

- **s:{作用域ID}**: 操作的作用域ID，用于标识操作的执行阶段
  - 默认值：`-1`
  - 代码位置：`framework/src/interface/operation/operation.h` 中 `Operation::GetScopeId()` 返回 `scopeId_` 成员变量
- **输入参数**: 操作的输入参数（变量列表、常量等）
- **属性**: 使用空格分隔的键值对
  - 格式：`#属性名{属性值}`
  - 例如：`#CACHE_MODE{PA_BSND} #PA_NZ_BLOCK_SIZE{1}`

#### 内存类型说明

| 内存类型 | 说明 |
|---------|------|
| MEM_UNKNOWN | 未知内存类型 |
| MEM_DEVICE_DDR | 设备DDR内存 |
| MEM_UB | Unified Buffer（统一缓冲区） |
| MEM_L1 | L1缓存 |
| MEM_L0A | L0A缓存（矩阵A输入） |
| MEM_L0B | L0B缓存（矩阵B输入） |
| MEM_L0C | L0C缓存（矩阵C输出） |

#### 动态参数说明

在 `TILE_VIEW` 和 `TILE_ASSEMBLE` 等操作中，会出现以下动态参数：

- **dynoffset（动态偏移量）**：运行时确定的偏移量，与静态 `offset` 不同，`dynoffset` 的值在编译时无法确定，需要在运行时根据动态 shape 或循环变量计算。例如 `dynoffset:[ 0, 0]` 表示各维度的动态偏移均为 0。
  - 使用场景：动态 shape 场景下，tensor 的实际访问位置依赖于运行时输入 shape
  - 与 `offset` 的区别：`offset` 是编译时确定的固定偏移，`dynoffset` 是运行时计算的动态偏移

- **dynvalidshape（动态有效形状）**：运行时确定的有效形状，与静态 `shape` 不同，`dynvalidshape` 表示在动态 shape 场景下 tensor 实际有效的维度大小。例如 `dynvalidshape:[ 16, 128]` 表示运行时有效形状为 `[16, 128]`。
  - 使用场景：当输入 shape 在运行时变化时，需要记录实际有效的计算范围
  - 与 `valid_shape` 的区别：`valid_shape` 是编译时可推断的有效形状，`dynvalidshape` 是运行时才能确定的有效形状

这些动态参数是 PyPTO 支持动态 shape 计算的关键机制，确保在运行时 shape 不确定的情况下仍能正确访问内存。

#### 常见操作类型

| 操作类型 | 说明 |
|---------|------|
| TILE_VIEW | 张量视图操作 |
| TILE_COPY_IN | 从DDR拷贝到UB |
| TILE_COPY_OUT | 从UB拷贝到DDR |
| TILE_ADDS | 张量加法（标量） |
| TILE_ADD | 张量加法（张量） |
| TILE_MUL | 张量乘法 |
| TILE_MATMUL | 矩阵乘法 |
| TILE_INDEX_OUTCAST | 索引输出操作 |
| TILE_ASSEMBLE | 张量组装操作 |
| TILE_RESHAPE | 张量重塑操作 |
| TILE_TRANSPOSE | 张量转置操作 |
| TILE_BROADCAST | 张量广播操作 |

#### 分析要点

1. **变量定义**: 确保输出变量只被定义一次
2. **变量引用**: 确保输入变量都已被定义
3. **shape匹配**: 验证输入输出shape的合理性
4. **内存类型**: 检查内存类型转换的正确性
5. **操作参数**: 验证操作参数的完整性和正确性
6. **属性一致性**: 检查操作属性的合理性

## 实际IR示例分析

### 示例1：TILE_VIEW操作

```
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %84@50#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10022 TILE_VIEW(g:-1, s:-1) %12@14#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR from offset:[  0,  0] dynoffset:[  0,  0] to MEM_DEVICE_DDR dynvalidshape:[ 16,128]
```

**分析**：
- **输出**: logic tensor `%84`，raw tensor `@50`，子图ID `-1`，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **操作**: `TILE_VIEW`，操作ID `10022`，子图ID `-1`，作用域ID `-1`
- **输入**: logic tensor `%12`，raw tensor `@14`，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **参数**:
  - `from offset:[0, 0]`：源偏移量
  - `dynoffset:[0, 0]`：动态偏移量
  - `to MEM_DEVICE_DDR`：目标内存类型
  - `dynvalidshape:[ 16, 128]`：动态有效形状

### 示例2：TILE_INDEX_OUTCAST操作

```
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %1@6#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10014 TILE_INDEX_OUTCAST(g:-1, s:-1) %88@52#(-1)MEM_UB::MEM_UB, %0@5#(-1)MEM_UB::MEM_UB, %84@50#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR #CACHE_MODE{PA_BSND} #PA_NZ_BLOCK_SIZE{1} #axis{0}
```

**分析**：
- **输出**: logic tensor `%1`，raw tensor `@6`，子图ID `-1`，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **操作**: `TILE_INDEX_OUTCAST`，操作ID `10014`，子图ID `-1`，作用域ID `-1`
- **输入**:
  1. `%88@52`：数据tensor，内存类型 `MEM_UB::MEM_UB`
  2. `%0@5`：索引tensor，内存类型 `MEM_UB::MEM_UB`
  3. `%84@50`：输出tensor，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **属性**:
  - `#CACHE_MODE{PA_BSND}`：缓存模式
  - `#PA_NZ_BLOCK_SIZE{1}`：非零块大小
  - `#axis{0}`：操作轴

### 示例3：TILE_ADDS操作

```
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %76@17#(0)MEM_UB::MEM_UB = !10015 TILE_ADDS(g:0, s:-1) %87@52#(0)MEM_UB::MEM_UB #SCALAR{1.000000} #op_attr_reverseOperand{0}
```

**分析**：
- **输出**: logic tensor `%76`，raw tensor `@17`，子图ID `0`，内存类型 `MEM_UB::MEM_UB`
- **操作**: `TILE_ADDS`，操作ID `10015`，子图ID `0`，作用域ID `-1`
- **输入**: logic tensor `%87`，raw tensor `@52`，内存类型 `MEM_UB::MEM_UB`
- **属性**:
  - `#SCALAR{1.000000}`：标量值
  - `#op_attr_reverseOperand{0}`：操作数反转标志

### 示例4：TILE_ASSEMBLE操作

```
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %18@18#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10018 TILE_ASSEMBLE(g:-1, s:-1) %21@19#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR from MEM_DEVICE_DDR to offset:[  0,  0] to dynoffset:[  0,  0]
```

**分析**：
- **输出**: logic tensor `%18`，raw tensor `@18`，子图ID `-1`，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **操作**: `TILE_ASSEMBLE`，操作ID `10018`，子图ID `-1`，作用域ID `-1`
- **输入**: logic tensor `%21`，raw tensor `@19`，内存类型 `MEM_DEVICE_DDR::MEM_DEVICE_DDR`
- **参数**:
  - `from MEM_DEVICE_DDR`：源内存类型
  - `to offset:[0, 0]`：目标偏移量
  - `to dynoffset:[0, 0]`：目标动态偏移量

## 完整IR文件示例

以下是一个小型的完整 `.tifwkgr` 文件示例，展示 IR 文件的整体结构：

```
-------------
Function TENSOR_TENSOR_add_kernel_loop_Unroll1_PATH0_hiddenfunc0_3[3] 1234567890123456789 DYNAMIC_LOOP_PATH TILE_GRAPH {
  /* /mnt/workspace/test_add.py:10 */
  RAWTENSOR[  0] <16 x 128 x DT_FP32> @10"TENSOR_0"
  RAWTENSOR[  1] <16 x 128 x DT_FP32> @12"TENSOR_1"
  RAWTENSOR[  2] <16 x 128 x DT_FP32> @14"TENSOR_2"

  INCAST[  0]  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %6@10#(-1) fromSlot[0]
  INCAST[  1]  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %8@12#(-1) fromSlot[1]
  OUTCAST[  0]  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %14@14#(-1) toSlot[2]

  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10001 TILE_VIEW(g:-1, s:-1) %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR from offset:[  0,  0] dynoffset:[  0,  0] to MEM_DEVICE_DDR dynvalidshape:[ 16, 128]
  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %8@12#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10002 TILE_VIEW(g:-1, s:-1) %8@12#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR from offset:[  0,  0] dynoffset:[  0,  0] to MEM_DEVICE_DDR dynvalidshape:[ 16, 128]
  <16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %14@14#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10003 TILE_ADD(g:-1, s:-1) %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %8@12#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR #op_attr_reverseOperand{0}
}
```

**结构说明**：
1. **分隔线**：`-------------` 标记函数开始
2. **函数头**：包含函数名、magic、hash、函数类型、图类型
3. **RAWTENSOR 声明**：定义 3 个物理张量（2 个输入 + 1 个输出）
4. **INCAST/OUTCAST 声明**：定义 2 个输入 cast 和 1 个输出 cast
5. **操作节点**：
   - `!10001 TILE_VIEW`：对输入 tensor 0 创建视图
   - `!10002 TILE_VIEW`：对输入 tensor 1 创建视图
   - `!10003 TILE_ADD`：执行加法操作，输出到 tensor 2

**数据流**：
```
INCAST[0] (%6@10) → TILE_VIEW(!10001) → %6@10 ─┐
                                                ├→ TILE_ADD(!10003) → %14@14 → OUTCAST[0]
INCAST[1] (%8@12) → TILE_VIEW(!10002) → %8@12 ─┘
```

## IR分析方法

### 1. 文件完整性检查

#### 检查清单

- [ ] 文件头格式正确
- [ ] 所有RAWTENSOR索引唯一
- [ ] 所有INCAST/OUTCAST索引唯一
- [ ] 所有操作节点op_id唯一
- [ ] 所有变量定义和使用匹配

#### 检查方法

```python
# 伪代码示例
def check_file_completeness(ir_file):
    # 1. 检查文件头
    if not ir_file.has_valid_header():
        return False, "Invalid file header"

    # 2. 检查RAWTENSOR索引唯一性
    rawtensor_indices = ir_file.get_rawtensor_indices()
    if len(set(rawtensor_indices)) != len(rawtensor_indices):
        return False, "Duplicate RAWTENSOR indices"

    # 3. 检查变量定义和使用
    defined_vars = ir_file.get_defined_variables()
    used_vars = ir_file.get_used_variables()
    undefined_vars = used_vars - defined_vars
    if undefined_vars:
        return False, f"Undefined variables: {undefined_vars}"

    return True, "File is complete"
```

### 2. 数据流分析

#### 分析目标

- 追踪数据从输入到输出的完整路径
- 验证数据依赖关系的正确性
- 检查是否存在悬空数据或数据泄露

#### 分析步骤

1. **构建数据流图**
   - 以变量为节点
   - 以数据依赖为边

2. **追踪数据路径**
   - 从INCAST开始追踪
   - 沿着操作链追踪到OUTCAST

3. **验证数据完整性**
   - 确保所有输入数据都被使用
   - 确保所有输出数据都有来源

#### 示例分析

```
# 数据流示例
INCAST[0] → %6 → TILE_VIEW → %84 → TILE_INDEX_OUTCAST → %1 → ... → OUTCAST[0]
```

### 3. 内存访问分析

#### 分析目标

- 检查内存访问的合法性
- 验证内存类型转换的正确性
- 检查是否存在内存冲突

#### 分析要点

1. **内存类型转换**
   - MEM_UNKNOWN → MEM_DEVICE_DDR
   - MEM_DEVICE_DDR → MEM_UB
   - MEM_UB → MEM_DEVICE_DDR

2. **内存访问模式**
   - TILE_COPY_IN: DDR → UB
   - TILE_COPY_OUT: UB → DDR
   - 计算操作: UB → UB

3. **内存冲突检查**
   - 同一内存区域不能同时被读写
   - 需要同步机制保证数据一致性

#### 示例分析

```
# 内存访问模式示例
%0 (DDR) → TILE_VIEW → %84 (DDR) → TILE_INDEX_OUTCAST → %1 (DDR) → TILE_VIEW → %91 (UB) → TILE_ADDS → %76 (UB) → TILE_ASSEMBLE → %14 (DDR)
```

### 4. 依赖关系分析

#### 分析目标

- 检查操作之间的依赖关系
- 验证调度顺序的合理性
- 检查是否存在循环依赖

#### 分析要点

1. **数据依赖**
   - 读取-写入依赖（RAW）
   - 写入-读取依赖（WAR）
   - 写入-写入依赖（WAW）

2. **控制依赖**
   - 条件分支
   - 循环结构

3. **同步依赖**
   - 内存屏障
   - 同步操作

### 5. Shape一致性检查

#### 检查方法

1. **操作输入输出shape匹配**
   - 每个操作的输入shape应该符合操作要求
   - 输出shape应该与操作结果一致

2. **张量shape传播**
   - 沿着数据流追踪shape变化
   - 验证shape变化的合理性

3. **动态shape处理**
   - 检查动态shape的处理是否正确
   - 验证运行时参数的使用

## 问题定位技巧

### 1. 对比Before和After IR

#### 对比维度

| 维度 | 说明 | 检查方法 |
|-----|------|---------|
| 操作数量 | 操作节点的增减 | 统计操作数量变化 |
| 变量数量 | 变量的增减 | 统计变量数量变化 |
| 内存类型 | 内存类型的变化 | 对比内存类型 |
| 数据流 | 数据流路径的变化 | 追踪数据流 |
| shape | shape的变化 | 对比shape |

#### 常见问题模式

1. **操作丢失**
   - 症状：After中缺少某些操作
   - 可能原因：Pass错误删除了必要操作
   - 定位方法：对比操作ID和操作名

2. **变量未定义**
   - 症状：After中使用了未定义的变量
   - 可能原因：Pass删除了变量定义但保留了使用
   - 定位方法：检查变量定义和使用

3. **shape不匹配**
   - 症状：操作输入输出shape不一致
   - 可能原因：Pass错误修改了shape
   - 定位方法：检查操作节点的shape

4. **内存类型错误**
   - 症状：内存类型转换不正确
   - 可能原因：Pass错误分配了内存类型
   - 定位方法：检查内存类型转换链

### 2. 错误模式识别

#### 语法错误

```
# 错误示例：缺少等号（操作名应带 TILE_ 前缀，如 TILE_ADDS）
<16 x 128 x DT_FP32> %1@6#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR !10001 TILE_ADDS(g:-1, s:-1) %10@13#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR
```

**定位方法**：
- 检查操作节点格式
- 验证等号存在性

#### 语义错误

```
# 错误示例：使用了未定义的变量
<16 x 128 x DT_FP32> %1@6#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10001 ADDS(g:-1, s:-1) %999@13#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR
# %999 未定义
```

**定位方法**：
- 检查变量定义和使用
- 验证变量索引范围

#### 逻辑错误

```
# 错误示例：数据流断裂
INCAST[0] → %6 → TILE_VIEW → %7
# %7 未被任何操作使用
```

**定位方法**：
- 追踪数据流
- 检查悬空变量

### 3. 性能问题定位

#### 常见性能问题

1. **冗余操作**
   - 症状：存在重复或无用的操作
   - 定位方法：分析操作冗余性

2. **数据拷贝过多**
   - 症状：TILE_COPY_IN/TILE_COPY_OUT过多
   - 定位方法：统计数据拷贝次数

3. **内存访问不连续**
   - 症状：offset不连续，影响性能
   - 定位方法：分析offset模式

4. **并行度不足**
   - 症状：操作串行化，未充分利用并行
   - 定位方法：分析依赖关系

## 逻辑错误检查清单

### 变量相关

- [ ] 所有变量都有唯一定义
- [ ] 所有使用的变量都已定义
- [ ] 变量索引在有效范围内
- [ ] 变量地址标识符一致

### 操作相关

- [ ] 所有操作ID唯一
- [ ] 操作参数数量正确
- [ ] 操作输入输出shape匹配
- [ ] 操作属性合理

### 数据流相关

- [ ] 所有INCAST数据都被使用
- [ ] 所有OUTCAST数据都有来源
- [ ] 数据流路径完整
- [ ] 不存在悬空数据

### 内存相关

- [ ] 内存类型转换正确
- [ ] 内存访问合法
- [ ] 不存在内存冲突
- [ ] 内存复用合理

### 依赖关系相关

- [ ] 不存在循环依赖
- [ ] 数据依赖正确
- [ ] 控制依赖正确
- [ ] 同步依赖正确

### Shape相关

- [ ] 所有shape维度合理
- [ ] shape传播正确
- [ ] 动态shape处理正确
- [ ] shape边界检查正确

## IR分析流程

### 标准分析流程

```
1. 读取IR文件
   ↓
2. 检查文件完整性
   ↓
3. 解析IR元素
   ↓
4. 构建数据流图
   ↓
5. 执行一致性检查
   ↓
6. 分析数据流
   ↓
7. 检查内存访问
   ↓
8. 验证依赖关系
   ↓
9. 生成分析报告
```

### 详细分析步骤

#### 步骤1：读取和解析

```python
def parse_ir_file(file_path):
    """解析IR文件"""
    ir = IRFile()

    # 1. 读取文件头
    ir.parse_header()

    # 2. 解析RAWTENSOR
    ir.parse_rawtensors()

    # 3. 解析INCAST/OUTCAST
    ir.parse_casts()

    # 4. 解析操作节点
    ir.parse_operations()

    return ir
```

#### 步骤2：完整性检查

```python
def check_completeness(ir):
    """检查IR完整性"""
    checks = [
        check_header(ir),
        check_rawtensor_indices(ir),
        check_variable_definitions(ir),
        check_operation_ids(ir),
    ]

    return all(checks)
```

#### 步骤3：数据流分析

```python
def analyze_dataflow(ir):
    """分析数据流"""
    # 1. 构建数据流图
    graph = build_dataflow_graph(ir)

    # 2. 追踪数据路径
    paths = trace_data_paths(graph)

    # 3. 检查数据完整性
    integrity = check_data_integrity(paths)

    return integrity
```

#### 步骤4：内存分析

```python
def analyze_memory(ir):
    """分析内存访问"""
    # 1. 构建内存访问图
    mem_graph = build_memory_graph(ir)

    # 2. 检查内存类型转换
    type_checks = check_memory_types(mem_graph)

    # 3. 检查内存冲突
    conflicts = check_memory_conflicts(mem_graph)

    return type_checks, conflicts
```

#### 步骤5：生成报告

```python
def generate_analysis_report(ir, checks):
    """生成分析报告"""
    report = {
        'file': ir.file_path,
        'completeness': checks['completeness'],
        'dataflow': checks['dataflow'],
        'memory': checks['memory'],
        'dependencies': checks['dependencies'],
        'issues': checks['issues'],
        'recommendations': checks['recommendations'],
    }

    return report
```

## 最佳实践

### 1. 分析策略

#### 自顶向下分析

1. 先检查文件结构和完整性
2. 再分析数据流和依赖关系
3. 最后深入细节检查

#### 关键路径优先

1. 优先分析关键数据流路径
2. 重点检查核心操作节点
3. 验证关键变量的正确性

### 2. 调试技巧

#### 分段验证

1. 将IR分段分析
2. 逐步验证每个部分
3. 定位问题到具体位置

#### 对比验证

1. 对比Before和After IR
2. 识别变化点
3. 验证变化的合理性

### 3. 记录和报告

#### 详细记录

1. 记录分析过程和发现
2. 保存关键IR片段
3. 记录问题和解决方案

#### 结构化报告

1. 使用标准报告格式
2. 包含问题、原因、建议
3. 提供可操作的改进建议

## 常见问题和解决方案

### 问题1：变量未定义

**症状**：IR中使用了未定义的变量

**原因**：Pass删除了变量定义但保留了使用

**解决方案**：
1. 检查变量定义和使用
2. 确保所有变量都有定义
3. 修复Pass逻辑

### 问题2：数据流断裂

**症状**：数据流路径不完整

**原因**：Pass错误删除了操作

**解决方案**：
1. 追踪数据流路径
2. 识别断裂点
3. 恢复必要操作

### 问题3：shape不匹配

**症状**：操作输入输出shape不一致

**原因**：Pass错误修改了shape

**解决方案**：
1. 检查shape传播
2. 验证操作shape要求
3. 修复shape计算

### 问题4：内存类型错误

**症状**：内存类型转换不正确

**原因**：Pass错误分配了内存类型

**解决方案**：
1. 检查内存类型转换链
2. 验证内存访问模式
3. 修复内存类型分配

## 附录

### IR语法总结

```
IR文件 ::= 文件头 RAWTENSOR* INCAST* OUTCAST* operation*

文件头 ::= "Function" 函数名 "[" function_magic "]" hash 函数类型 图类型 "{"

RAWTENSOR ::= "RAWTENSOR[" 索引 "] <" shape "> @" 编号 '"' 名称 '"'

INCAST ::= "INCAST[" 索引 "] <" shape "/" valid_shape "> %" logic_tensor "@" raw_tensor "#(" 子图ID ") fromSlot[" 槽位列表 "]"

OUTCAST ::= "OUTCAST[" 索引 "] <" shape "/" valid_shape "> %" logic_tensor "@" raw_tensor "#(" 子图ID ") toSlot[" 槽位列表 "]"

operation ::= "<" shape "/" valid_shape "> %" 输出logic_tensor "@" 输出raw_tensor "#(" 子图ID ")" 读内存类型 "::" 写内存类型 " = !" operation_id opcode "(g:" 子图ID ", s:" 作用域ID ")" 参数 属性*

logic_tensor ::= "%" 名称 或 "%" 编号

raw_tensor ::= "@" 编号 或 "@" 名称

shape ::= 维度 " x" 维度 " x" ... " x" 数据类型

属性 ::= "#" 属性名 "{" 属性值 "}"
```

**注意**：
- 内存类型使用双冒号 `::` 分隔读写类型
- 属性使用空格分隔，不是逗号
- 操作参数包含 `(g:子图ID, s:作用域ID)` 子图和作用域信息
- 函数类型和图类型是独立的字段
- 子图ID默认值为 `-1`（NOT_IN_SUBGRAPH）
- 作用域ID默认值为 `-1`
