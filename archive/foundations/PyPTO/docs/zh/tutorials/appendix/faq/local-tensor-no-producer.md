# local Tensor无producer时被读取触发device侧断言

## 问题现象描述

在算子编写过程中，如果local Tensor仅创建但未写入，就在loop内被读取，编译/执行到设备路径时可能触发设备侧断言或异常。典型报错如下：

```text
ASSERTION FAILED: WORKSPACE_ITER_INVALID
Root[...] incast ... slotIndex ... read from empty address.
```

该现象通常对应读取到的slot没有producer，即Tensor对应slot未被任何operation写入。

最小示例（语义示意）：

```python
def kernel(x):
    t = pypto.tensor((1,), pypto.DT_FP32, "t")
    for i in range(n):
        _ = t[0]
```

## 可能原因

local Tensor（如`t`）仅被创建，但在被读取前没有任何写操作（如切片赋值、`move`、`assemble`或其他能建立producer的写入），导致读取路径访问到空地址并触发断言。

## 处理步骤

可采用以下任一方式规避：

1. 在读取前先对local Tensor执行有效写入（例如切片赋值、`move`、`assemble`等），确保其存在producer。
2. 将该local Tensor写在loop内，作为loop内Tensor使用，使其在当前loop作用域内，PyPTO内存管理策略会为其申请内存地址。
