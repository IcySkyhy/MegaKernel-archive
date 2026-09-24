# Overview

This document describes how to run the Dispatch operator and Combine operator in PyPTO.

# Environment Setup

## Hardware Requirements

Product model: Atlas A2 series

Operating system: Linux ARM

Current Dispatch and Combine test cases require 4 cards to run.

## Software Package Installation

1. For the PyPTO Environment Preparation document, refer to `docs/zh/install/prepare_environment.md`. Install Python dependencies, install build dependencies, install drivers and firmware, install the CANN toolkit package, install the CANN ops package, and obtain the pto-isa source code.

2. Install MPICH version 3.2.1.

   Download the installation package from the official website. Execute the following commands to install:

   ```shell
   version='3.2.1'
   tar -zxvf "mpich-${version}.tar.gz"
   cd "mpich-${version}"
   ./configure --disable-fortran --prefix=/usr/local/mpich
   make && make install
   ```

3. Obtain the PyPTO source code.

   ```shell
   git clone https://gitcode.com/cann/pypto.git
   cd pypto
   ```

# Set Environment Variables

1. Set the CANN package environment variables.

     After installation, configure the environment variables. For the actual path of set_env.sh, refer to the following command.
     The environment variable configuration above takes effect only in the current window. You can write the commands into an environment variable configuration file (for example, .bashrc) as needed.

     ```bash
     # Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
     source /usr/local/Ascend/ascend-toolkit/set_env.sh

     # Custom path installation
     source ${install_path}/ascend-toolkit/set_env.sh
     ```

2. Set the pto-isa environment variables.

   ```shell
   export PTO_TILE_LIB_CODE_PATH='/path/to/pto-isa' # Set to the actual path of the pto-isa source code
   ```

3. Set the MPICH environment variables.

   ```shell
   export PATH="/usr/local/mpich/bin:${PATH}"
   ```

# Run the Operators

Run the Dispatch operator:

```shell
python3 tools/scripts/run_operation_test_with_config.py MoeDispatch --distributed_op
```

A successful run produces output similar to the following:

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

Run the Combine operator:

```shell
python3 tools/scripts/run_operation_test_with_config.py MoeDistributedCombine --distributed_op
```

A successful run produces output similar to the following:

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
