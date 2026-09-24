# 概述

本文档介绍如何运行 PyPTO 里的 Dispatch 算子和 Combine 算子。

# 环境准备

## 硬件要求

产品型号：Atlas A2 系列

操作系统：Linux ARM

目前的 Dispatch 和 Combine 的用例要求 4 张卡运行。

## 软件包安装

1. 参考 [PyPTO 的《环境准备》文档](../../../../../../docs/zh/install/prepare_environment.md) ，安装 Python 依赖，安装编译依赖，安装驱动与固件，安装 CANN toolkit 包，安装 CANN ops 包，获取 pto-isa 源码。

2. 安装 3.2.1 版本的 MPICH。

   从 [官网](https://www.mpich.org/static/downloads/3.2.1/) 下载安装包，执行以下命令安装：

   ```shell
   version='3.2.1'
   tar -zxvf "mpich-${version}.tar.gz"
   cd "mpich-${version}"
   ./configure --disable-fortran --prefix=/usr/local/mpich
   make && make install
   ```

3. 获取 PyPTO 源码。

   ```shell
   git clone https://gitcode.com/cann/pypto.git
   cd pypto
   ```

# 设置环境变量

1. 设置 CANN 包的环境变量。

    安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
    上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 指定路径安装
    source ${install_path}/ascend-toolkit/set_env.sh
    ```

2. 设置 pto-isa 的环境变量。

   ```shell
   export PTO_TILE_LIB_CODE_PATH='/path/to/pto-isa' # 根据实际情况设置为 pto-isa 源码的路径

3. 设置 MPICH 的环境变量。

   ```shell
   export PATH="/usr/local/mpich/bin:${PATH}"
   ```

# 运行算子

运行 Dispatch 算子：

```shell
python3 tools/scripts/run_operation_test_with_config.py MoeDispatch --distributed_op
```

运行成功可以看到如下输出：

```
[       OK ] TestMoeDispatch/DistributedTest.TestMoeDispatch/0 (43164 ms)
[----------] 1 test from TestMoeDispatch/DistributedTest (43164 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (43165 ms total)
[  PASSED  ] 1 test.
[       OK ] TestMoeDispatch/DistributedTest.TestMoeDispatch/0 (43173 ms)
[----------] 1 test from TestMoeDispatch/DistributedTest (43173 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (43173 ms total)
[  PASSED  ] 1 test.
[       OK ] TestMoeDispatch/DistributedTest.TestMoeDispatch/0 (43173 ms)
[----------] 1 test from TestMoeDispatch/DistributedTest (43173 ms total)

[----------] Global test environment tear-down
[==========] [       OK ] TestMoeDispatch/DistributedTest.TestMoeDispatch/0 (43173 ms)
[----------] 1 test from TestMoeDispatch/DistributedTest (43174 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (43174 ms total)
[  PASSED  ] 1 test.
1 test from 1 test suite ran. (43174 ms total)
[  PASSED  ] 1 test.
```

运行 Combine 算子：

```shell
python3 tools/scripts/run_operation_test_with_config.py MoeDistributedCombine --distributed_op
```

运行成功可以看到如下输出：

```
[       OK ] TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/1 (30006 ms)
[----------] 1 test from TestMoeDistributedCombine/DistributedTest (30006 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (30007 ms total)
[  PASSED  ] 1 test.
[       OK ] TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/1 (30008 ms)
[       OK ] TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/1 (30008 ms)
[----------] 1 test from TestMoeDistributedCombine/DistributedTest (30008 ms total)

[----------] Global test environment tear-down
[==========] [----------] 1 test from TestMoeDistributedCombine/DistributedTest (30008 ms total)

[----------] Global test environment tear-down
[==========] [       OK ] TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/11 test from 1 test suite ran. (30008 ms total)
[  PASSED  ] 1 test.
 (30008 ms)
[----------] 1 test from TestMoeDistributedCombine/DistributedTest (30008 ms total)

1 test from 1 test suite ran. (30008 ms total)
[  PASSED  ] 1 test.
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (30009 ms total)
[  PASSED  ] 1 test.
```
