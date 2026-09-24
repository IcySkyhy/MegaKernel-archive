# PyPTO

## 🔥最新动态
- 2026/04/10：0.2.0版本发布，变更前端表达方法，提升易用性、增强功能及性能，完善各模块能力，优化开发体验与运行效率
- 2026/03/30：v0.1.2版本发布，支持集群训练场景、优化框架编译性能与基础性能、修复已知整网集成问题
- 2026/03/09：v0.1.1版本发布，支持新前端、进一步增加API丰富程度、修复一些已知问题
- 2026/01/06：v0.1.0版本发布，PyPTO项目初始版本
- 2025/12：PyPTO项目首次上线。

## 🚀概述

PyPTO（发音：pai p-t-o）是一款面向AI加速器的高性能编程框架，旨在简化复杂融合算子乃至整个模型网络的开发流程，同时保持高性能计算能力。该框架采用创新的**PTO（Parallel Tensor/Tile Operation）编程范式**，以**基于Tile的编程模型**为核心设计理念，通过多层次的中间表示（IR）系统，将用户通过API构建的AI模型应用从高层次的Tensor图逐步编译成硬件指令，最终生成可在目标平台上高效执行的可执行代码。

本仓已集成代码仓库智能体，点击 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/pypto) 徽章，进入其专属页面，开启在线智能代码学习与知识问答体验！

### 核心特性

- **基于 Tile 的编程模型**：所有计算都基于Tile（硬件感知的数据块）进行，充分利用硬件并行计算能力和内存层次结构
- **多层级计算图转换**：通过编译Pass将Tensor Graph转换为Tile Graph、Block Graph和Execution Graph，每一步包括一系列Pass优化流程
- **自动化代码生成**：编译结果通过CodeGen生成底层PTO虚拟指令代码，再通过编译器将虚拟指令代码编译成目标平台的可执行代码
- **MPMD 执行调度**：可执行代码被加载到设备侧，通过MPMD（Multiple Program Multiple Data）的方式调度到设备上的处理器核
- **完整的工具链支持**：全流程的编译中间产物和运行时性能数据可通过IDE集成的工具链可视化识别性能瓶颈，开发者也可以通过工具链控制编译和调度行为
- **Python 友好 API**：提供直观的Tensor级别抽象，贴近算法开发者的思维模式，支持动态Shape和符号化编程
- **分层抽象设计**：对不同开发者暴露不同抽象层次，算法开发者使用Tensor层次，性能专家使用Tile层次，系统开发者使用Block层次

### 目标用户

- **算法开发者**：主要使用Tensor层次编程，快速实现和验证算法，专注于算法逻辑
- **性能优化专家**：可使用Tile或Block层次，进行深度性能调优，以实现极致性能
- **系统开发者**：可在Tensor/Tile/Block和PTO虚拟指令集层次上进行三方框架对接或集成，以及工具链开发

## ⚡️最佳实践样例

PyPTO提供了丰富的示例代码，涵盖从基础操作到复杂模型实现的多个层级。一些最佳实践样例参考:

### 大模型实现样例

- [DeepSeekV3.2 SFA](https://gitcode.com/cann/pypto/blob/master/models/deepseek_v32_exp/deepseekv32_sparse_flash_attention_quant.py) -稀疏Flash Attention量化实现
- [DeepSeekV3.2 MLA-PROLOG](https://gitcode.com/cann/pypto/blob/master/models/deepseek_v32_exp/deepseekv32_mla_indexer_prolog_quant.py) -MLA Indexer Prolog量化实现
- [GLM V4.5 Attention](https://gitcode.com/cann/pypto/blob/master/models/glm_v4_5/glm_attention.py) -GLM注意力机制实现
- [GLM V4.5 ExpertsSelector](https://gitcode.com/cann/pypto/blob/master/models/glm_v4_5/glm_select_experts.py) -GLM专家选择器实现

### 学习路径

在 [examples](https://gitcode.com/cann/pypto/blob/master/examples)目录下，我们规划了多个层级的样例:

- [beginner/](https://gitcode.com/cann/pypto/blob/master/examples/01_beginner)：基础操作示例，帮助初学者快速上手PyPTO编程
- [intermediate/](https://gitcode.com/cann/pypto/blob/master/examples/02_intermediate)：中级示例，包括自定义操作、神经网络模块等
- [advanced/](https://gitcode.com/cann/pypto/blob/master/examples/03_advanced)：高级示例，包括复杂模式和多函数组合

### pypto-gym 样例仓

在 [pypto-gym](https://gitcode.com/cann/pypto-gym)样例仓，我们提供了融合算子开发样例和大模型适配样例，供开发者学习、复用与对比

这些示例可以帮助开发者学习如何编写PyPTO算子，从简单的Tensor操作到复杂的模型网络实现，以便快速实现移植和部署

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitcode标签源码，使用master分支可能存在版本不匹配的风险。

## ⚡️快速入门

若您希望快速体验PyPTO的使用和开发过程，请访问如下文档获取简易教程。

- [环境部署](docs/zh/install/prepare_environment.md)：介绍项目基础环境的搭建，包括软件包和第三方依赖的获取和安装。
- [编译安装](docs/zh/install/build_and_install.md)：环境部署后，介绍如何快速获取或编译PyPTO软件包并安装。
- [样例运行](docs/zh/invocation/examples_invocation.md)：安装PyPTO软件包后，介绍如何快速实现样例运行。

## 📖文档资源

若您希望深入体验项目功能并修改源码，请访问如下文档获取详细教程。
- [文档中心](https://pypto.gitcode.com) ：当前发布版本的详细文档，包括编程指南、API参考，贡献指南等
- [示例代码](https://gitcode.com/cann/pypto/blob/master/examples/)：丰富的示例代码，从基础到高级应用

## 🔍目录结构

关键目录如下:

```
├── docs/                       # 文档资源
│   └── zh/
│      ├── api/                 # PyPTO API参考
│      ├── contribute/          # 贡献指南文档
│      └── tutorials/           # PyPTO算子开发
│
├── examples/                   # 示例代码
│   ├── 01_beginner/            # 初级示例
│   ├── 02_intermediate/        # 中级示例
│   └── 03_advanced/            # 高级示例
│
├── models/                     # 模型实现示例
│
├── python/                     # Python源码
│   ├── pypto/                  # Python包源码根目录
│   ├── src/                    # pybind11源码根目录
│   └── tests/                  # Python测试用例源码（UTest, STest）
│
├── framework/                  # C++源码根目录
│   ├── include/                # C++对外头文件
│   ├── src/                    # C++源码
│   │   ├── codegen/            # 代码生成模块
│   │   ├── passes/             # 编译Pass模块
│   │   └── ...
│   └── tests/                  # C++测试用例源码
│
├── tools/                      # 工具脚本
│
├── cmake/                      # 构建所需的CMake公共配置及脚本
├── build_ci.py                 # CI执行构建、执行UTest、执行STest辅助脚本
├── CMakeLists.txt              # 顶层CMakeLists.txt，定义所有对外公开编译开关
├── pyproject.toml              # Python编译工具配置文件
├── LICENSE                     # 许可证文件
└── setup.py                    # Python编译工具脚本文件(setuptools)
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [安全声明](./SECURITY.md)
- [许可证](./LICENSE)

## 联系我们

- **问题反馈**：通过GitCode【Issues】提交问题
- **功能建议**：通过GitCode【讨论】参与交流
- **技术支持**：参考文档或提交Issue

---

**注意**：本文档会持续更新，请关注最新版本。
