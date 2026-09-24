# 抽象硬件架构

AI Core是AI处理器的计算核心，AI处理器内部包含多个AI Core。本节介绍AI Core的并行计算架构抽象。PyPTO Pro基于该抽象提供Tile、内存空间和计算/搬运接口，屏蔽不同硬件实现的部分差异，开发者无需直接操作底层指令参数即可描述核内计算。

## 概览

AI Core的抽象硬件架构可以分为**计算单元、存储单元、搬运单元**三类核心组件。计算单元负责执行标量、向量和矩阵计算，存储单元负责保存输入、输出和中间数据，搬运单元负责在不同存储层级之间转移数据。下图展示Ascend 950PR/Ascend 950DT中三类组件在AI Core中的位置关系和协同方式。

**图1 AI Core抽象硬件架构**

![AI Core抽象硬件架构](../figures/abstract_hardware_architecture_3510_2.jpg)

Host侧下发的算子指令序列进入AI Core后，由Scalar计算单元负责控制逻辑和指令发射；Vector、Cube等计算单元分别执行向量计算和矩阵计算，DMA搬运单元执行数据搬运。计算数据通常在Global Memory和Local Memory之间流转；当计算和搬运存在依赖时，需要通过同步信号约束不同单元的执行顺序。

在PyPTO Pro中，核函数中的Python控制流对应Scalar侧控制逻辑；`pl.section_vector()`和`pl.section_cube()`分别描述Vector、Cube侧任务；`pl.load`、`pl.move`、`pl.store`等接口描述DMA搬运任务。

## 向量计算方式

当前硬件支持Membase和Regbase两种向量计算方式，PyPTO Pro对这两种计算方式的抽象如下：

| 计算方式 | 数据暂存位置 | PyPTO Pro表达 | 特点 |
|:---|:---|:---|:---|
| Membase | Local Memory（UB） | `Tile` + `pl.add`、`pl.sub`等Tile API | 每步计算结果写回UB |
| Regbase | VF Register File | `@pl.vector_function`中的`RegTensor`/`MaskReg` + `vf.*` API | 中间结果可保留在寄存器，减少UB读写 |

Regbase的寄存器类型和使用约束参见[`vf.reg_tensor`](../../api/SIMD-API/operation/vf_computation/reg_tensor.md)。

## 计算单元

AI Core中的计算单元主要包括Scalar、Vector和Cube三类。

| 组件名称 | 组件功能 | PyPTO Pro中的对应概念 |
|:---|:---|:---|
| Scalar | 执行地址计算、循环控制等标量工作，并把向量计算、矩阵计算、数据搬运和同步任务发射给对应单元。 | 核函数中的Python控制流、标量表达式和`pl.system.*`同步接口 |
| Vector | 负责执行向量运算。 | `pl.section_vector()`中的Tile向量API，以及`@pl.vector_function`中的`vf.*` API |
| Cube | 负责执行矩阵运算。 | `pl.section_cube()`中的`pl.matmul`、`pl.matmul_acc`、`pl.matmul_mx`、`pl.matmul_mx_acc`等矩阵API |

## 存储单元和搬运单元

存储单元按使用位置分为Local Memory和Global Memory。

- **Local Memory**：AI Core片内存储，用于暂存从Global Memory搬入的数据分片，并保存计算输出和中间结果。数据可供Vector、Cube等计算单元访问，继续参与片内计算，或通过搬运单元写回Global Memory。
- **Global Memory**：Device侧全局存储，是Local Memory中数据搬入和搬出的主要来源或目的位置；PyPTO Pro使用`Tensor`表达其中的数据视图。

**图2 SIMD-Reg向量计算内存层级**

![SIMD-Reg向量计算内存层级](../figures/simd_reg_vector_memory_hierarchy.jpg)

PyPTO Pro使用[`TileType.target_memory`](../../api/SIMD-API/basic_data_structures/TileType.md)将Tile映射到不同的片上缓冲区：

| `pl.MemorySpace` | 物理缓冲区 | 典型角色 |
|:---|:---|:---|
| `Vec` | UB（Unified Buffer） | 向量计算的输入、输出和中间结果 |
| `Mat` | L1 Buffer | GM与L0A/L0B之间的矩阵暂存 |
| `Left` | L0A Buffer | `matmul`左操作数 |
| `Right` | L0B Buffer | `matmul`右操作数 |
| `Acc` | L0C Buffer | `matmul`累加结果（通常为FP32/INT32） |
| `Scaling` | Scaling/FBuffer | 量化、反量化参数 |
| `ScaleLeft` | L0A Buffer（scale专用地址域） | A矩阵的E8M0 scale（分组缩放因子） |
| `ScaleRight` | L0B Buffer（scale专用地址域） | B矩阵的E8M0 scale（分组缩放因子） |

完整枚举说明参见[`pl.MemorySpace`](../../api/SIMD-API/basic_data_structures/MemorySpace.md)。

DMA（Direct Memory Access）搬运单元负责Global Memory与Local Memory之间的数据搬入、搬出，以及不同层级Local Memory之间的数据流转。PyPTO Pro中常见路径如下：

| 搬运或计算路径 | Pipe | PyPTO Pro接口 |
|:---|:---|:---|
| GM → L1/UB | MTE2 | `pl.load` / `pl.load_tile` |
| L1 → L0A/L0B | MTE1 | `pl.move` |
| L1 → ScaleLeft/ScaleRight | MTE1 | `pl.move` |
| L0A × L0B → L0C | M | `pl.matmul` / `pl.matmul_acc` |
| MX L0A × L0B → L0C | M | `pl.matmul_mx` / `pl.matmul_mx_acc` |
| UB → GM | MTE3 | `pl.store` / `pl.store_tile` |
| L0C → GM | FIX | `pl.store` / `pl.store_tile` |
| UB ↔ VF Register File | VF load/store | `vf.load*` / `vf.store*` |

## Tile与硬件存储的映射

Tile是PyPTO Pro对片上缓冲区的编程抽象。[`TileType`](../../api/SIMD-API/basic_data_structures/TileType.md)使用`shape`、`dtype`和`target_memory`描述Tile的逻辑形状、数据类型及所在的片上存储空间；矩阵场景还可以通过布局相关属性描述是否转置及内层分型。开发者通常只需选择目标存储空间，布局细节可沿用对应内存空间的默认值。

## 执行流程与同步机制

理解抽象硬件架构时，还需要从三个视角区分各单元之间的关系：

- **异步指令流**：Scalar侧将计算、搬运等任务发射到Vector、Cube、DMA等单元的指令队列，各执行单元在各自Pipe上异步执行。
- **计算数据流**：Vector/Cube访问Local Memory中的数据完成计算，DMA负责Local Memory与Global Memory之间以及各级Local Memory之间的数据流转。
- **同步信号流**：当不同Pipe的异步任务存在数据依赖或顺序依赖时，通过同步信号约束执行先后；同步信号不是数据本身的流向。

PyPTO Pro推荐使用`pl.make_tile_group`配合`@pl.jit(auto_mutex=True)`，由编译器根据Tile的mutex元数据自动插入跨Pipe同步。使用单个`pl.make_tile`并需要手动控制依赖时，可调用`pl.system.sync_src` / `pl.system.sync_dst`。详细说明参见[Tile矢量计算](../operator_development/tile_based_python_programming/Tile_vector_computation.md)和[`sync_src` / `sync_dst`](../../api/SIMD-API/operation/synchronization/sync_src_sync_dst.md)。

## 多核架构

PyPTO Pro采用SPMD编程模型。各逻辑AI Core执行相同程序，并根据全局逻辑索引处理不同数据分片。`pl.get_block_num()`返回启动时配置的逻辑Block数量，`pl.get_block_idx()`返回当前执行域的全局逻辑核索引。

在同时启动AIC与AIV的1:2混合Kernel中，每个逻辑Block对应一个AIC和两个AIV：

```text
block_dim个逻辑Block
├── Cube段：block_dim个AIC，get_block_idx()范围为[0, block_dim)
└── Vector段：2 * block_dim个AIV，get_block_idx()范围为[0, 2 * block_dim)
```

`pl.get_subblock_idx()`用于区分同一逻辑Block内的AIV，返回`0`或`1`；Vector段的`pl.get_block_idx()`是展平后的全局AIV逻辑索引，可直接用于数据分片。

多核和subblock切分方法参见[多核切分与Tiling](../operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md)。
