# PyPTO文档中心

欢迎使用PyPTO文档。本文档由华为技术有限公司主导编写，并包含来自CANN社区贡献者的贡献。华为技术有限公司和CANN社区贡献者对本文档内容保留所有权利。

PyPTO（发音:pai p-t-o）是CANN推出的一款面向AI加速器的高效编程框架，旨在简化算子开发流程，同时保持高性能计算能力。PyPTO提供PyPTO Tensor与PyPTO Pro两种编程方式：

- **PyPTO Tensor编程**：采用创新的PTO（Parallel Tensor/Tile Operation）编程范式，以基于Tile的编程模型为核心设计理念，通过多层次的计算图表达，将用户通过API构建的AI模型从高层次的Tensor计算图逐步编译成硬件指令，最终生成可在目标平台上高效执行的代码，并由设备侧以MPMD（Multiple Program Multiple Data）方式调度执行。
- **PyPTO Pro编程**：采用以Python为前端的SPMD编程模型，以二维Tile描述片上数据、数据搬运以及Cube和Vector计算。开发者可以显式组织多核切分、片上存储和计算流水，并可使用Reg API进行寄存器级向量计算，适合需要精细控制硬件资源和深度优化算子性能的场景。

您可以根据所需的编程抽象与性能控制粒度，选择对应的编程指南；API参考提供各编程接口的详细说明。

```{toctree}
:maxdepth: 2
:caption: 目录

install/index
PyPTO Tensor算子开发 <tutorials/index>
PyPTO Tensor API <api/index>
PyPTO Pro算子开发 <pypto_pro/tutorials/index>
PyPTO Pro API <pypto_pro/api/index>
