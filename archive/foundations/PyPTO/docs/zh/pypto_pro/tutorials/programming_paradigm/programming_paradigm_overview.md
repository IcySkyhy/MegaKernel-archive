# 编程范式概述

PyPTO Pro面向昇腾NPU的AI Core算子开发，采用外层多核SPMD并行与内层单核SIMD并行结合的编程范式，并以Tile作为核内计算和数据搬运的主要载体。开发者通过Python接口显式描述多核数据切分、片上数据搬运和计算逻辑，框架负责Kernel编译、加载与任务下发。

## Host与Device协作

一个基于昇腾处理器的异构系统通常包含CPU与昇腾NPU。其中，CPU及其内存称为Host与Host Memory；NPU及其内存称为Device与Device Memory。

基于昇腾的PyPTO Pro应用程序通常包含两部分：一部分运行在Host CPU上，使用Python（PyTorch）编程；另一部分运行在NPU上，使用PyPTO Pro编写[核函数Kernel](../operator_development/kernel_function.md)。Host端通过PyTorch张量在Device Memory上准备输入输出数据，调用Kernel函数触发JIT编译并下发NPU任务，通过`torch.npu.synchronize()`等待核函数执行完成。Host代码与Kernel可写在同一个`.py`文件中。

**图1**Kernel调度示意图

![Kernel调度示意图](../figures/kernel_scheduling_diagram.png)

| 角色 | 职责 |
|:---|:---|
| **Host（CPU）** | 通过PyTorch张量准备输入输出数据；调用Kernel函数触发JIT编译和NPU任务下发；同步等待结果 |
| **Device（NPU）** | 执行编译后的Kernel二进制；在AI Core上完成Tile级别的数据搬运与计算；将结果写回Global Memory |

PyPTO Pro Kernel的典型数据流如下：

1. Host端通过PyTorch张量在NPU上准备输入输出数据。
2. Host端调用Kernel函数，框架完成JIT编译并下发任务。
3. Device端的AI Core通过`load`/`load_tile`将数据从Global Memory搬入片上缓冲区，完成Tile级别的计算，再通过`store`/`store_tile`将结果写回Global Memory。
4. Host端同步后访问结果。

关于Kernel下发、Stream选择和同步方式，请参考[AI Core算子JIT编译基本用法](../operator_development/compilation_and_execution/JIT_compilation.md#kernel下发stream与同步)。

## SPMD嵌套SIMD编程范式

昇腾NPU采用**外层多核SPMD（Single Program Multiple Data）并行 + 内层单核SIMD（Single Instruction Multiple Data）并行**的双层架构。

- **单卡多核层面采用SPMD编程模型**：各逻辑AI Core执行同一份Kernel代码，并根据全局逻辑索引处理不同的数据分片。
- **单核计算层面采用SIMD并行机制**：单个AI Core内部通过一条指令同时处理多个同构数据元素，完成向量、矩阵或融合计算。

在PyPTO Pro中，`block_dim`表示启动时配置的逻辑Block数量。仅启动Cube或仅启动Vector时，逻辑核数与`block_dim`一致；同时启动AIC与AIV时，各执行域的逻辑核数还取决于AIC:AIV比例。相关接口为：

- `pl.get_block_num()`获取本次Kernel启动的Block总数。
- `pl.get_block_idx()`获取当前执行域的全局逻辑核索引；Vector段返回值已经按subblock展平。
- `pl.get_subblock_idx()`获取当前逻辑Block内的subblock索引。
- `pl.get_subblock_num()`获取当前执行域每个逻辑Block对应的subblock数量。

当前1:2混合Kernel中，Cube段的`get_block_idx()`范围为`[0, block_dim)`，Vector段范围为`[0, 2 * block_dim)`。Vector段可直接使用该全局逻辑索引进行数据分片，`get_subblock_idx()`用于区分同一逻辑Block内的AIV。

SIMD是一种数据并行模型，其核心特征包括：

- **单指令驱动**：并行计算单元执行相同的操作。
- **数据同构**：参与计算的数据具有一致的数据类型和操作方式。
- **同步执行**：同一条指令批量处理多个数据元素。

SIMD主要适用于数据密集、操作规整、无分支或分支较少的计算任务，包括图像处理、音频信号处理、矩阵乘法、卷积和逐元素数学运算等场景。

## AI Core算子类型

根据算子计算逻辑使用的硬件单元，AI Core算子可分为三类：

- **矢量类算子**：主要使用Vector计算单元完成元素级、逻辑类和数据重组类计算，在PyPTO Pro中通过`pl.section_vector()`标记执行域。
- **矩阵类算子**：主要使用Cube计算单元完成矩阵乘、张量卷积等计算，在PyPTO Pro中通过`pl.section_cube()`标记执行域。
- **融合类算子**：联动Cube与Vector计算单元协同计算。开发者可以手动插入核间同步并编排流水；在支持自动Pipeline变换的场景中，也可以使用`@pl.pipeline.stage`标记计算阶段，由框架自动插入核间同步并完成Preload核间流水编排。

AI Core中Scalar、Vector、Cube、存储和搬运单元的详细说明，请参考[抽象硬件架构](abstract_hardware_architecture.md)。

## Kernel通用开发流程

PyPTO Pro算子开发通常包含以下四个步骤：

1. **Tiling（分块）设计**：将全局数据划分为数据分片，并为各AI Core分配处理范围。
2. **数据搬入**：调用`pl.load`/`pl.load_tile`等接口，将数据从Global Memory搬入L1 Buffer、UB等片上存储。
3. **数据计算**：在`pl.section_vector()`或`pl.section_cube()`中调用Tile计算API，完成向量、矩阵或融合计算。
4. **数据搬出**：调用`pl.store`/`pl.store_tile`等接口，将计算结果写回Global Memory。

AI Core片上存储空间有限，无法一次性加载超大尺寸Tensor。实际开发中通常迭代完成分块搬运、分批计算和结果写回，并通过TileGroup的N缓冲流水减少数据搬运带来的等待。

多核切分通常采用跨步分配：第`core_id`个AI Core处理序号为`core_id, core_id+num_cores, core_id+2*num_cores, ...`的Tile，使各核处理的Tile数量最多相差1。详细实践请参考[多核切分与Tiling](../operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md)。

## Tile编程模型

PyPTO Pro使用Tile作为NPU核内计算的载体，通过Tile操作描述完整的SPMD计算流程。计算以Tensor作为输入，通过搬运操作将数据从Tensor搬入Tile，经过Tile级别的核内计算后，再将结果从Tile搬回Tensor。

传统NPU SIMD算子编程主要面临以下问题：

1. 不同算子的多核Tiling切分差异较大。
2. 不同算子的缓冲区复用方式差异较大。
3. offset计算和底层指令参数填写复杂且容易出错。
4. 核内、核间流水排布和同步插入较为复杂。

PyPTO Pro保留由开发者控制多核Tiling切分和缓冲区复用的能力，并通过以下机制简化编程：

### Tile抽象

PyPTO Pro使用二维Tile描述核内缓冲区。开发者只需表达Tile在Global Memory中Tensor上的坐标，无需手工将多维坐标转换为一维offset；Tile API同时封装了底层指令参数。

### TileGroup与自动核内同步

对于Cube、Vector核内流水，PyPTO Pro通过`pl.make_tile_group`将同一流水线中轮转使用的多块Tile封装为一组。开发者通过`next()`和`current()`获取当前Tile，并通过`mutex_ids`标识缓冲区。使用`@pl.jit(auto_mutex=True)`编译Kernel时，框架会根据Tile的mutex信息自动插入核内同步。

### 自动核间流水编排

对于支持自动Pipeline变换的Cube、Vector融合算子，stage是流水编排的基本单位。开发者使用`@pl.pipeline.stage`装饰器标记一个计算函数，再在Kernel循环的`pl.section_cube()`或`pl.section_vector()`执行域中调用该函数。`section_cube`和`section_vector`用于指定代码运行在哪类计算单元上，stage则用于划分编译器可以分析和重排的流水阶段。下面仅展示代码组织结构，省略Tensor和Tile声明：

```python
@pl.pipeline.stage
def cube_stage(cube_input, intermediate):
    # Cube阶段
    ...


@pl.pipeline.stage
def vector_stage(intermediate, output):
    # Vector阶段
    ...


@pl.jit(pipeline=pl.pipeline.PipelineConfig(preload=2))
def fused_kernel(cube_input, intermediate, output):
    for i in pl.range(0, NUM_TILES):
        with pl.section_cube():
            cube_stage(cube_input, intermediate)
        with pl.section_vector():
            vector_stage(intermediate, output)
```

`@pl.pipeline.stage`本身只为函数添加stage标记，不改变函数行为。设置`pipeline=pl.pipeline.PipelineConfig(...)`后，编译器识别循环中的stage调用，分析各阶段对Tile的读写依赖，自动插入Cube与Vector之间的核间同步，并将串行阶段转换为Preload流水，使不同迭代的Cube和Vector阶段可以重叠执行。`preload`用于配置稳态流水开始前首个阶段的预执行次数。未启用Pipeline变换的融合Kernel仍可直接使用`pl.system.set_cross_core`和`pl.system.wait_cross_core`手动管理核间同步。

关于Tile和TileGroup的详细使用方法，请参考[基于Tile的Python编程](../operator_development/tile_based_python_programming/Python_programming_overview.md)。

## 矢量计算模式

Ascend 950PR/Ascend 950DT在UB存储体系的基础上提供向量寄存器编程能力，形成“Global Memory → UB → Register”的存储层级。PyPTO Pro相应提供两种矢量计算模式：

- **Memory矢量计算**：通过Tile API在UB上完成数据缓存与计算，数据流为“Global Memory → UB → Global Memory”，适合通用矢量计算场景。
- **Reg矢量计算**：通过`@pl.vector_function`定义VF函数，使用`vf.*`接口在向量寄存器上完成计算，数据流为“Global Memory → UB → Register → UB → Global Memory”，适合需要精细化调优的高性能场景。

Reg矢量计算的详细说明，请参考[Reg矢量计算编程](../operator_development/tile_based_python_programming/Reg_vector_computation.md)。

## PyPTO Pro编程接口

传统裸指针编程需要开发者手工完成内存偏移计算、维度拆分和边界校验。Tensor抽象使用Shape、Stride、数据类型和内存布局等信息描述高维数据；Tile则作为核内计算、存储和数据搬运的物理载体，衔接全局Tensor和底层硬件。

PyPTO Pro基于Tile抽象提供两类核心编程接口：

| API层级 | 编程模式 | 特点 | 主要用途 |
|----------|----------|------|----------|
| **Tile API** | 基于Tile编程 | 通过`make_tile`/`make_tile_group`分配片上缓冲区，使用`auto_mutex=True`自动管理核内同步与N缓冲流水 | 适配大多数算子开发场景，兼顾硬件控制能力和开发效率 |
| **Reg API（VF计算）** | 基于寄存器编程 | 通过`@pl.vector_function`定义VF函数，使用`vf.*`接口直接操作向量寄存器 | 自主管理寄存器数据加载和存储，用于精细化调优与高性能实现 |

此外，PyPTO Pro提供Utils API，包括Python语法糖以及`printf`、`pto_assert`、`dump_data`和`trap`等调试接口。详细接口说明请参考[SIMD API](../../api/SIMD-API/index.md)和[Utils API](../../api/Utils-API/index.md)。

## 控制流

### 循环

Kernel内使用`pl.range(start, end, step)`表达循环：

```python
num_cores = pl.get_block_num()
core_id = pl.get_block_idx()

for i in pl.range(core_id, m_tile_num, num_cores):
    for j in pl.range(0, n_tile_num, 1):
        ...
```

### 条件分支

Kernel支持Python原生`if`、`elif`和`else`：

```python
if tiling.opkind[4] == 0:
    pl.add(tile_c, tile_a, tile_b)
elif tiling.opkind[4] == 1:
    pl.sub(tile_c, tile_a, tile_b)
else:
    pl.mul(tile_c, tile_a, tile_b)
```

## 编程流程

使用PyPTO Pro开发算子的典型流程如下：

1. **定义Kernel函数**：使用`@pl.jit()`装饰器标记Kernel，通过`pl.Tensor`声明输入输出。
2. **定义Tile与TileGroup**：通过`pl.TileType`描述Tile的Shape、数据类型和内存空间，通过`pl.make_tile`或`pl.make_tile_group`分配片上缓冲区。
3. **编写计算逻辑**：在`pl.section_vector()`或`pl.section_cube()`中，使用搬运和计算接口完成Kernel逻辑。
4. **Host端调用**：通过`kernel[stream, block_dim](*args)`启动Kernel。

详细的编程方法请参考：

- [Tile核函数](../operator_development/kernel_function.md)
- [基于Tile的Python编程](../operator_development/tile_based_python_programming/Python_programming_overview.md)

## 小结

PyPTO Pro以SPMD+SIMD编程范式将同一份Kernel分发到多个AI Core执行，并通过Tile和Reg两级接口表达Vector、Cube及融合计算。开发者显式控制多核数据切分和片上缓冲区使用，框架则通过Tile、TileGroup和stage机制简化指令参数、核内同步及核间流水的表达。
