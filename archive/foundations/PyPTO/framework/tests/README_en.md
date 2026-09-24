# Tile Framework Developer Self-Test Engineering Design

## Path Design

```text
.
├── cmake
├── CMakeLists.txt
├── utils
│
├── ut
│   ├── CMakeLists.txt
│   ├── utils (optional)
│   ├── stubs (optional)
│   │
│   ├── Module 1 (no sub-module scenario)
│   │   ├── CMakeLists.txt
│   │   ├── utils (optional)
│   │   ├── stubs (optional)
│   │   └── src
│   │
│   └── Module 2 (sub-module scenario)
│       ├── CMakeLists.txt
│       ├── utils (optional)
│       ├── stubs (optional)
│       │
│       ├── Sub-module 2.1
│       │   ├── CMakeLists.txt
│       │   ├── utils (optional)
│       │   ├── stubs (optional)
│       │   └── src
│       │
│       └── Sub-module 2.2
└── st
    ├── CMakeLists.txt
    ├── utils (optional)
    ├── stubs (optional)
    │
    ├── Module 1
    │   ├── CMakeLists.txt
    │   ├── utils (optional)
    │   ├── stubs (optional)
    │   │
    │   └── src   (sub-module sub-paths are optional)
    │       ├── Sub-module 1.1
    │       └── Sub-module 1.2
    │
    └── Module 2
```

To facilitate architecture management and dual-repository decoupling, the modules are planned as follows:

| Level 1 Module  | Level 2 Module                            |
|:----------------|:------------------------------------------|
| interface       | tensor, function, machine, ops, passes    |
| simulation      | simulation                                |
| codegen         | codegen                                   |
| runtime         | runtime                                   |
