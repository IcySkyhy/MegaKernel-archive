# Triton-distributed-ascend

## 🎯 项目介绍

Triton-distributed-ascend 是基于 Triton 语言扩展的分布式计算框架，专门为昇腾（Ascend）AI处理器优化。该项目提供了在昇腾NPU上进行高效分布式计算的能力，支持多种通信原语和计算-通信重叠优化，旨在提升大规模分布式训练和推理的性能。

该项目利用 MLIR（Multi-Level Intermediate Representation）的多级抽象能力，为昇腾硬件提供完备的表达能力。通过编译优化，将 Triton 语言编写的分布式算子高效映射到昇腾AI处理器，同时提供细粒度的性能控制接口，支持对片上内存、流水同步等进行精准控制。

### 主要特性

- **分布式通信原语支持**：提供 AllReduce、AllGather、ReduceScatter、All2All 等常用分布式通信操作
- **计算-通信重叠优化**：支持计算与通信的重叠执行，提升整体性能
- **基于 AscendNPU IR 的编译优化**：利用昇腾硬件特性进行深度优化
- **Python 友好接口**：提供简洁的 Python API，易于集成到现有深度学习框架

## 🔍 仓库结构

Triton-distributed-ascend 仓库关键目录如下所示：

```
├── 3rdparty              // 第三方依赖
│   └── triton-ascend     // Triton Ascend
├── asset                 // 资源文件
├── docs                  // 文档
├── include               // 头文件
├── lib                   // 库文件
│   ├── Conversion        // 转换层实现
│   │   └── TritonDistributedToFunc
│   │       └── ASCEND   // 昇腾后端转换
├── python                // Python 绑定和接口
│   └── triton_dist       // 主要 Python 模块
│       ├── language      // 语言扩展
│       │   ├── distributed_ops.py
│       │   └── extra
│       │       └── ascend
│       └── test          // 测试用例
├── scripts               // 脚本工具
├── tutorials             // 教程和示例
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## 📝 版本配套说明

### 硬件要求

- 昇腾 AI 处理器（如 Ascend 910B、910C 等）

### 软件要求

- **CANN 版本**：CANN 9.1.0 或更高版本
  - 请参考 [CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)
- **PyTorch**：支持 PyTorch 2.0 或更高版本
- **torch_npu**：根据 PyTorch 版本选择对应的 torch_npu 插件
  - 请参考 [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html) 文档
- **工具链**：
  - cmake ≥ 3.19
  - GLIBC ≥ 2.28
- **Python**：Python 3.8 或更高版本

## ⚡️ 快速上手

### 1. CANN 包安装

参考 [快速安装 CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)，要求CANN >= 8.5.0

配置 CANN 环境变量（默认安装路径）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 其他软件依赖

安装 PyTorch 框架和 torch_npu 插件，根据实际环境选择对应的版本进行安装。

安装必需的 Python 包：

```bash
pip install -r requirements.txt
```

### 3. LLVM构建

Triton-ascend依赖 LLVM 特定版本。

#### 代码准备: `git checkout` 检出指定版本的LLVM

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout f6ded0be897e2878612dd903f7e8bb85448269e5
   wget https://raw.githubusercontent.com/triton-lang/triton-ascend/2e69438a0a41a00ab72c7d46b44d97092cf2362c/third_party/ascend/patch/llvm_patch_f6ded0b.patch
   git apply llvm_patch_f6ded0b.patch
   ```

#### clang构建安装LLVM

- 步骤1：使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
  如未安装，请按下面指令安装clang、lld、ccache：

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX={PATH_TO}/llvm-install
   ```

- 步骤3：执行以下命令进行构建和安装LLVM：

  ```bash
  cd {PATH_TO}/llvm-project # 路径为用户拉取LLVM代码的路径,需根据实际调整
  mkdir build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
  ninja install
  ```

- 步骤4：需要拷贝FileCheck和llvm-lit到目标安装路径：

   ```bash
   cp  {PATH_TO}/llvm-project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   cp  {PATH_TO}/llvm-project/build/bin/llvm-lit ${LLVM_INSTALL_PREFIX}/bin/llvm-lit
   ```

#### 构建安装AscendNPU-IR
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://gitcode.com/Ascend/AscendNPU-IR.git
cd AscendNPU-IR
git submodule update --init --depth=1
mkdir build
./build-tools/build.sh -o ./build -t --build-type Release --apply-patches --bisheng-compile=$ASCEND_HOME_PATH/bin --build-shmem-template
```

### 4. 源码编译

```bash
# 克隆代码仓
git clone https://gitcode.com/Ascend/Triton-distributed-ascend.git
cd Triton-distributed-ascend

# submodule 初始化
git submodule update --init --depth=1
cd 3rdparty/triton-ascend
git submodule update --init --depth=1

cd ../shmem/
# shmem构建命令. 如果需要编译 A5，请额外添加 `-soc_type Ascend950`
bash scripts/build.sh -python_extension
pip install dist/shmem-xxx.whl


cd ../../

# triton-distributed-ascend构建命令
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF pip install ./python --no-build-isolation
```

#### 替代构建命令
```bash
# 方式1：Editable install（开发模式）
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF pip install -e ./python --verbose --no-build-isolation

# 方式2：构建 wheel
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF pip wheel ./python --no-build-isolation
```

**构建参数说明：**

- `TRITON_BUILD_WITH_CLANG_LLD=ON`：使用 Clang 和 LLD 进行链接
- `TRITON_BUILD_PROTON=OFF`：关闭Proton构建
- `TRITON_BUILD_LITTLE_KERNEL=OFF`：关闭mega-kernel相关构建
- `TRITON_PLUGIN_DIRS`：指定 Triton 插件目录路径

## 运行Triton-distributed示例

```bash
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   export PATH={PATH_TO}/AscendNPU-IR/build/bin:$PATH
   # 运行tutorials示例：
   torchrun --nproc-per-node=2 tutorials/ascend/01-ascend-allgather-gemm.py
```

## 🤝 贡献指南

我们欢迎社区贡献！详细的贡献指南请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 贡献流程

1. Fork 本仓库并从 `master` 分支创建您的分支
2. 如果添加了代码，请添加相应的测试
3. 如果修改了 API，请更新文档
4. 确保测试套件通过
5. 确保代码符合 lint 规范
6. 提交 Pull Request

### 代码规范

- **C++ 代码**：遵循 [Google Style Guide](http://google.github.io/styleguide/)，使用 clang-format 和 clang-tidy
- **Python 代码**：遵循 Google Python Style Guide，使用 yapf 和 ruff 进行格式化和检查
- **编译器部分**：遵循 [MLIR Style Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide)

## 📄 许可证书

[Apache License v2.0](LICENSE)

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/Ascend/Triton-distributed-ascend/issues)
- 提交 [Pull Request](https://github.com/Ascend/Triton-distributed-ascend/pulls)

## 🙏 致谢

感谢以下项目的支持：

- [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed.git) - ByteDance Triton-distributed社区
- [Triton](https://github.com/openai/triton) - OpenAI 的 Triton 语言
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) - 昇腾 NPU 中间表示
- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
