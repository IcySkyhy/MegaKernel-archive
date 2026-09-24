# PyPTO

## Latest Updates
- 2026/04/10: Version 0.2.0 released, changing frontend expression methods, improving ease of use, enhancing features and performance, refining module capabilities, and optimizing development experience and runtime efficiency.
- 2026/03/30: Version v0.1.2 released, supporting cluster training scenarios, optimizing framework compilation performance and basic performance, and fixing known network integration issues.
- 2026/03/09: Version v0.1.1 released, supporting a new frontend, further increasing API richness, and fixing some known issues.
- 2026/01/06: Version v0.1.0 released, the initial version of the PyPTO project.
- 2025/12: PyPTO project first launched.

## Overview

PyPTO (pronounced: pai p-t-o) is a high-performance programming framework for AI accelerators. It aims to simplify the development process of complex fused operators and even entire model networks while maintaining high-performance computing capability. The framework adopts an innovative **PTO (Parallel Tensor/Tile Operation) programming paradigm** with a **Tile-based programming model** as its core design concept. Through a multi-level intermediate representation (IR) system, it gradually compiles AI model applications built through APIs from high-level tensor graphs into hardware instructions, finally generating executable code that runs efficiently on the target platform.

This repository has integrated a code repository agent. Click the [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/pypto) badge to visit its dedicated page and start the online intelligent code learning and knowledge Q and A experience.

### Core Features

- **Tile-based Programming Model**: All computations operate on Tiles (hardware-aware data blocks), fully leveraging hardware parallel computing capabilities and memory hierarchy.
- **Multi-level Computation Graph Transformation**: The compilation Pass transforms a Tensor Graph into a Tile Graph, Block Graph, and Execution Graph. Each step includes a series of Pass optimization processes.
- **Automated Code Generation**: The compilation result generates low-level PTO virtual instruction code through CodeGen, and then the compiler compiles the virtual instruction code into executable code for the target platform.
- **MPMD Execution Scheduling**: Executable code loads onto the device side and schedules to the processor cores on the device through MPMD (Multiple Program Multiple Data).
- **Complete Toolchain Support**: The IDE-integrated toolchain visualizes the full-process compilation intermediates and runtime performance data to identify performance bottlenecks. Developers can also control compilation and scheduling behavior through the toolchain.
- **Python-friendly API**: Provides intuitive Tensor-level abstractions that align with algorithm developers' thinking patterns, supporting dynamic shapes and symbolic programming.
- **Layered Abstraction Design**: Exposes different abstraction levels to different developers. Algorithm developers use the Tensor level, performance experts use the Tile level, and system developers use the Block level.

### Target Users

- **Algorithm Developers**: Primarily use the Tensor level for programming, quickly implementing and verifying algorithms while focusing on algorithm logic.
- **Performance Optimization Experts**: Use the Tile or Block level for deep performance tuning to achieve极致 performance.
- **System Developers**: Work at the Tensor/Tile/Block and PTO virtual instruction set levels for third-party framework integration or toolchain development.

## Best Practice Samples

PyPTO provides a rich set of sample code covering multiple levels from basic operations to complex model implementations. For some best practice samples, refer to the following:

### Large Model Implementation Samples

- [DeepSeekV3.2 SFA](https://gitcode.com/cann/pypto/blob/master/models/deepseek_v32_exp/deepseekv32_sparse_flash_attention_quant.py) - Sparse Flash Attention quantized implementation
- [DeepSeekV3.2 MLA-PROLOG](https://gitcode.com/cann/pypto/blob/master/models/deepseek_v32_exp/deepseekv32_mla_indexer_prolog_quant.py) - MLA Indexer Prolog quantized implementation
- [GLM V4.5 Attention](https://gitcode.com/cann/pypto/blob/master/models/glm_v4_5/glm_attention.py) - GLM attention mechanism implementation
- [GLM V4.5 ExpertsSelector](https://gitcode.com/cann/pypto/blob/master/models/glm_v4_5/glm_select_experts.py) - GLM expert selector implementation

### Learning Path

In the [examples](https://gitcode.com/cann/pypto/blob/master/examples) directory, the following multi-level samples are available:

- [beginner/](https://gitcode.com/cann/pypto/blob/master/examples/01_beginner): Basic operation samples to help beginners quickly get started with PyPTO programming.
- [intermediate/](https://gitcode.com/cann/pypto/blob/master/examples/02_intermediate): Intermediate samples including custom operations, neural network modules, and more.
- [advanced/](https://gitcode.com/cann/pypto/blob/master/examples/03_advanced): Advanced samples including complex patterns and multi-function composition.

### pypto-gym Sample Repository

In the [pypto-gym](https://gitcode.com/cann/pypto-gym) sample repository, fused operator development samples and large model adaptation samples are provided for developers to learn, reuse, and compare.

These samples help developers learn how to write PyPTO operators, from simple Tensor operations to complex model network implementations, enabling rapid porting and deployment.

##  Version Compatibility

The source code of this project is released in conjunction with CANN software versions. For the correspondence between CANN software versions and the tags of this project, please refer to the relevant version notes in the [release repository](https://gitcode.com/cann/release-management).

Please note that to ensure smooth customization and development of the source code, you should select the matching CANN version and the corresponding Gitcode tag source code. Using the `master` branch may introduce version mismatch risks.

## Quick Start

If you want to quickly experience the usage and development process of PyPTO, refer to the following documents for simple tutorials.

- [Environment Setup](docs/zh/install/prepare_environment.md): Introduces setting up the project's basic environment, including obtaining and installing software packages and third-party dependencies.
- [Build and Install](docs/zh/install/build_and_install.md): After environment setup, describes how to quickly obtain or compile the PyPTO software package and install it.
- [Running Samples](docs/zh/invocation/examples_invocation.md): After installing the PyPTO software package, describes how to quickly run samples.

## Documentation Resources

If you want to deeply experience the project features and modify the source code, refer to the following documents for detailed tutorials.
- [Documentation Center](https://pypto.gitcode.com): Detailed documentation for the current release version, including programming guides, API references, contribution guides, and more.
- [Sample Code](https://gitcode.com/cann/pypto/blob/master/examples/): Rich sample code from basic to advanced applications.

## Directory Structure

Key directories are as follows:

```
├── docs/                       # Documentation resources
│   └── zh/
│      ├── api/                 # PyPTO APIs
│      ├── contribute/          # Contribution guide documentation
│      └── tutorials/           # PyPTO Operator Development
│
├── examples/                   # Sample code
│   ├── 01_beginner/            # Beginner samples
│   ├── 02_intermediate/        # Intermediate samples
│   └── 03_advanced/            # Advanced samples
│
├── models/                     # Model implementation samples
│
├── python/                     # Python source code
│   ├── pypto/                  # Python package source root
│   ├── src/                    # pybind11 source root
│   └── tests/                  # Python test case source (UTest, STest)
│
├── framework/                  # C++ source root
│   ├── include/                # C++ external header files
│   ├── src/                    # C++ source code
│   │   ├── codegen/            # Code generation module
│   │   ├── passes/             # Compilation Pass module
│   │   └── ...
│   └── tests/                  # C++ test case source
│
├── tools/                      # Utility scripts
│
├── cmake/                      # CMake shared configuration and scripts for building
├── build_ci.py                 # CI helper script for building, running UTest, and STest
├── CMakeLists.txt              # Top-level CMakeLists.txt defining all public build switches
├── pyproject.toml              # Python build tool configuration file
├── LICENSE                     # License file
└── setup.py                    # Python build tool script file (setuptools)
```

## Related Information

- [Contribution Guide](./CONTRIBUTION_en.md)
- [Security Statement](./SECURITY_en.md)
- [License](./LICENSE)

## Contact Us

- **Issue Feedback**: Submit issues through GitCode [Issues].
- **Feature Suggestions**: Participate in discussions through GitCode [Discussions].
- **Technical Support**: Refer to documentation or submit an Issue.

---

**Note**: This document is continuously updated. For the latest version, refer to the latest release.
