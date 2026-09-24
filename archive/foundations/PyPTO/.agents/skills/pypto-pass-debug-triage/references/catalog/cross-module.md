# 跨模块 Bug 定界

本仓库历史上 1477 个 `fix(*)` 提交分布在多个模块。很多报错虽然堆栈落在 Pass 内，但根因可能在 operation、machine、frontend 等模块。以下定界表基于历史提交统计得出。

## 症状 → 可能模块

| 模式 ID | 症状关键词 | 最可能模块（优先级） | 说明 |
|---------|------------|---------------------|------|
| C001 | spill / L0C / L1 / copyOut / copyIn / dtype / MTE | pass > machine | 数据搬移路径多在 pass，但底层地址/同步问题可能在 machine |
| C002 | reshape / view / assemble / merge / validshape | pass > operation > frontend | 视图处理在 pass，但 op 约束和前端 shape 也会引发 |
| C003 | matmul / conv / transdata / relu / gather / where | operation > pass | 具体算子实现问题优先看 operation |
| C004 | runtime / launch / sync / event / timeout / rtMalloc | machine > pass | 运行时调度、同步、事件、内存分配多在 machine |
| C004 | aicore / cube / vector / mte / fixpipe | machine > operation | 硬件指令相关 |
| C005 | python / jit / frontend / symbolic / scalar | frontend / front | 前端图捕获与 Python 接口 |
| C005 | IR / token / dce / transform | ir > pass | IR 变换与 pass 接口 |
| C006 | distributed / shmem / allreduce / collective | distributed | 分布式通信相关 |
| C007 | codegen / tile / unroll / cce / kernel | codegen > pass | 代码生成与 tile 策略 |
| C003 | precision / accuracy / fp16 | operation > pass | 精度问题可能来自算子实现或 pass 优化 |

## 跨模块典型案例

历史上曾有 27 个非 `fix(pass)` 提交修改了 `framework/src/passes/` 下的文件，说明这些模块的问题会传导到 Pass：

- **C002/C003**：`fix(operation)` 修改 `infer_param_index.cpp`、`infer_tensor_format.cpp`，说明算子 shape/format 推导会影响 Pass。
- **C001/C004**：`fix(machine)` 修改 `insert_sync.cpp`、`loopaxes_proc.cpp`，说明运行时同步问题可能需要在 Pass 中补偿。
- **C006**：`fix(distributed)` 修改 `infer_param_index.cpp`，说明分布式通信 validshape 会影响 Pass。
- **C007**：`fix(codegen)` 修改 `codegen_preproc.cpp`、`dyn_attr_to_static.cpp`，说明代码生成约束需要在 Pass 中处理。

## 定界决策树

```text
报错位置在 framework/src/passes/ 某个 Pass 内
  │
  ├─ [C001] 报错内容涉及 spill/L0C/L1/dtype/地址/内存类型
  │   └─ 先查 Pass 的 AssignMemoryType / SpillBuffer / OoOScheduler
  │   └─ 若涉及运行时同步/事件/launch，同时查 machine 模块
  │
  ├─ [C002] 报错内容涉及 reshape/view/assemble/merge/validshape
  │   └─ 先查 Pass 的 SupernodeGraphBuilder / SplitLargeFanoutTensor
  │   └─ 若 shape 来自前端输入，查 frontend/front 的 shape 推导
  │
  ├─ [C003] 报错内容涉及具体算子（matmul/conv/transdata/...）
  │   └─ 先查 operation/operator 的算子实现与约束
  │   └─ 再查 Pass 是否漏处理该 opcode
  │
  ├─ [C004] 报错内容涉及 runtime/aicore/同步/timeout/memory pool
  │   └─ 先查 machine 模块
  │
  └─ [C003] 报错内容涉及精度
      └─ 先查 operation 算子实现精度
      └─ 再查 Pass 是否错误合并/搬移/转换 dtype
```

---
