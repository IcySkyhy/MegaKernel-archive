# 简介

## 什么是PyPTO Pro

PyPTO Pro是一种面向Ascend 950PR/Ascend 950DT、以Python为前端的DSL（Domain-Specific Language，领域特定语言）。它采用SPMD（Single Program Multiple Data，单程序多数据）执行模型，并以二维Tile作为Tile API中Cube、Vector计算和数据搬运的主要载体；对于需要寄存器级编程的场景，还提供基于RegTensor和MaskReg的Reg API。PyPTO Pro通过Tile等抽象，在保留硬件控制能力的同时简化算子开发，并可在经过合理的Tiling切分和流水设计后获得较好的性能。

## PyPTO Pro总体架构

PyPTO Pro提供Tile API、Reg API和Utils API，并通过JIT编译与执行链将Python Kernel编译为可在AI Core上运行的二进制。

**图1 PyPTO Pro总体架构**

![PyPTO Pro总体架构](figures/architecture_pypto_pro.png)

PyPTO Pro的编译与执行链包括以下阶段：

1. **前端解析与优化**：`@pl.jit`标记的Python Kernel由前端解析并构建为PyPTO IR，再通过IR Pass完成优化。
2. **代码生成**：CCE CodeGen根据优化后的IR生成Device侧`kernel.cpp`及Tiling相关头文件。
3. **编译与链接**：生成的Device侧代码与Host封装代码`call_kernel.cpp`经毕昇编译器编译、链接，生成JIT产物`call_kernel.so`。
4. **加载与执行**：运行时加载`call_kernel.so`并下发Kernel任务，最终由AI Core执行。

## 适用场景

PyPTO Pro适用于开发高性能深度学习算子，可以较为快速地实现各种神经网络算子，尤其是Cube和Vector都涉及的融合算子，并达到较为理想的性能。

## 异构系统与程序组成

基于昇腾处理器的异构系统由Host和Device协同工作：

| 角色 | 组成 | 职责 |
|------|------|------|
| **Host（主机）** | CPU + Host Memory | 负责通用计算、设备资源管理、数据准备和任务调度 |
| **Device（设备）** | 昇腾NPU + Device Memory | 负责执行深度学习算子等高密度并行计算任务 |

PyPTO Pro应用程序相应地包含Host代码和Device代码。Host侧使用Python和PyTorch在Device Memory上准备输入、输出张量，调用并等待Device任务；Device侧使用PyPTO Pro编写[核函数Kernel](operator_development/kernel_function.md)，在AI Core上完成数据搬运和计算。两部分代码可以写在同一个`.py`文件中，由`@pl.jit()`标记Kernel函数，框架负责JIT编译和任务下发。

## AI Core与并行执行模型

一个昇腾NPU通常包含多个AI Core，每个AI Core内部具有标量、向量和矩阵运算单元以及片上存储。PyPTO Pro使用逻辑Block描述并行任务，启动Kernel时通过`block_dim`配置逻辑Block数量：

- `pl.get_block_num()`获取本次启动的Block总数。
- `pl.get_block_idx()`获取当前执行域中逻辑AI Core的全局索引。仅启动Cube或仅启动Vector时，其范围为`[0, block_num)`；1:2混合Kernel的Vector段中，其范围为`[0, 2 * block_num)`。
- `pl.get_subblock_idx()`获取当前逻辑Block内的subblock索引，仅在混合Kernel需要区分同一Block内的AIV时使用；Vector段的`get_block_idx()`已经包含该信息。

PyPTO Pro采用外层SPMD与内层SIMD结合的并行方式。SPMD（Single Program Multiple Data，单程序多数据）表示多个逻辑AI Core执行同一份Kernel代码，并依据全局逻辑索引处理不同的数据分片；SIMD（Single Instruction Multiple Data，单指令多数据）表示AI Core内部的一条指令同时处理多个同构数据元素，适合矩阵、向量及融合计算。

## Kernel运行流程

PyPTO Pro算子的典型运行流程如下：

1. **准备数据**：Host侧通过PyTorch在Device Memory上创建输入、输出张量。
2. **启动Kernel**：通过`kernel[stream, block_dim](...)`调用Kernel；首次调用触发JIT编译，后续调用可复用编译产物。
3. **多核分块**：各AI Core通过Block编号认领数据分片，通常使用`pl.range(core_id, total, num_cores)`实现跨步分配。
4. **搬入、计算与搬出**：Kernel使用`pl.load`/`pl.load_tile`将数据搬入片上缓冲区，调用`pl.add`、`pl.matmul`等Tile API完成计算，再通过`pl.store`/`pl.store_tile`将结果写回Device Memory。
5. **同步并访问结果**：Kernel相对于Host异步执行，Host在读取结果前应调用`torch.npu.synchronize()`等待任务完成。

## 核心特性

- **SPMD执行模型**：参与执行的AI Core运行同一份Kernel代码，并通过核索引划分各自处理的数据。
- **基于Tile的编程模型**：Tile API使用Tile（硬件感知的数据块）描述片上数据及其计算，通过Tile简化坐标偏移计算和指令参数配置。
- **基于TileGroup的核内流水表达**：使用TileGroup将同一流水线中可轮转复用的多块Tile封装为一组，用户通过`next()`和`current()`等接口表达多缓冲流水。使用`@pl.jit(auto_mutex=True)`开启自动同步后，框架根据每块Tile绑定的`mutex_id`自动插入核内同步。
- **融合场景的自动核间流水编排**：在支持的Cube、Vector融合场景中，用户通过stage机制和相应标签表达计算阶段，框架可自动插入核间同步，并进行Preload流水编排。
- **Python前端API**：提供更友好的Python API前端，贴近算法开发者的思维模式。
- **IR与编译优化**：Python前端代码会被转换为PyPTO IR，在IR层完成通用代码优化，再由CCE CodeGen生成Device侧代码并编译为可在NPU上运行的二进制文件。

## 产品支持情况

PyPTO Pro当前支持以下产品型号：

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 学习路径

建议按照以下路径学习PyPTO Pro：

1. **环境准备**：参考[环境准备](../../install/prepare_environment.md)完成基础环境搭建。
2. **快速入门**：从[HelloWorld](quick_start/HelloWorld.md)开始，了解Kernel函数定义、JIT编译和运行的基本流程；再通过[Add算子快速入门](quick_start/Add_operator_quick_start.md)学习Tile配置、数据搬运和计算。
3. **编程范式**：阅读[编程范式概述](programming_paradigm/programming_paradigm_overview.md)，理解SPMD编程、Tile抽象和流水机制。
4. **算子开发**：深入学习[基于Tile的Python编程](operator_development/tile_based_python_programming/Python_programming_overview.md)，掌握Tensor、Tile、TileGroup等核心数据结构的使用。
5. **API参考**：查阅[SIMD API](../api/SIMD-API/index.md)和[Utils API](../api/Utils-API/index.md)，了解各接口的详细用法。
