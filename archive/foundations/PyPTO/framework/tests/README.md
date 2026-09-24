# Tile Framework 开发者自测试工程设计

## 路径设计

```text
.
├── cmake
├── CMakeLists.txt
├── utils
│
├── ut
│   ├── CMakeLists.txt
│   ├── utils (可选)
│   ├── stubs (可选)
│   │
│   ├── 模块1 (不区分子模块场景)
│   │   ├── CMakeLists.txt
│   │   ├── utils (可选)
│   │   ├── stubs (可选)
│   │   └── src
│   │
│   └── 模块2 (区分子模块场景)
│       ├── CMakeLists.txt
│       ├── utils (可选)
│       ├── stubs (可选)
│       │
│       ├── 子模块 2.1
│       │   ├── CMakeLists.txt
│       │   ├── utils (可选)
│       │   ├── stubs (可选)
│       │   └── src
│       │
│       └── 子模块 2.2
└── st
    ├── CMakeLists.txt
    ├── utils (可选)
    ├── stubs (可选)
    │
    ├── 模块1
    │   ├── CMakeLists.txt
    │   ├── utils (可选)
    │   ├── stubs (可选)
    │   │
    │   └── src   (子模块是否分子路径可选)
    │       ├── 子模块1.1
    │       └── 子模块1.2
    │
    └── 模块2
```

为便于架构管控、双仓解耦, 规划各模块如下:

| 一级模块          | 二级模块                                   |
|:--------------|:---------------------------------------|
| interface     | tensor, function, machine, ops, passes |
| simulation    | simulation                             |
| codegen       | codegen                                |
| runtime       | runtime                                |
