# Tile Framework Distributed Communication Operator (Distributed OP) Developer Self-Test Engineering Design

## Path Design

```text
.
├── ops
│   ├── include
│   ├── src
│   ├── script
│   └── CMakeLists.txt
├── framework
│   ├── include
│   ├── src
│   └── CMakeLists.txt
├── CMakeLists.txt
```

## ops Directory Description
Provides a collection of test cases for OPs. Note that all code depends only on the frontend (function/operation/and so on) and does not depend on the runtime environment.
This directory is used for UT and simulation environment ST testing.

### include Directory Description
Provides external header files. If you need to add internal private header files, create a new `inner` subdirectory.

### script Directory Description
Scripts for generating test case parameters and golden data. Only the `DistributedTestGolden` class is exposed as an external interface.

### src Directory Description
Test case code.

## framework Directory Description
Provides the test framework code implementation for runtime communication resource initialization, multi-card process launching, and so on.
This directory is used for hardware board environment ST testing.

### include
Provides external header files.

### src
Test framework code.
