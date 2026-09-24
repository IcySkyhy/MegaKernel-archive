# kernel函数出参未写回导致计算不生效

## 问题现象描述

当前PyPTO框架用`pypto.frontend.jit`装饰的kernel函数，不支持有返回值，输出需要通过参数的形式传入并使用`[:]`等进行写回操作。如果直接使用等号赋值，无法将数据写入输出Tensor中。

示例代码：

```python
@pypto.frontend.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y = x + 1  # 此处会创建新的Tensor y

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y)  # 输出torch.empty创建的未经初始化的随机值
```

输出数据：

```python
tensor([[2.0703e-19, 7.1833e+22, 1.8502e+28, 6.8608e+22],
        [4.8011e+30, 1.2123e+25, 4.7418e+30, 1.8465e+25],
        [1.2122e+25, 4.6114e+24, 1.7836e+31, 1.7591e+22],
        [1.1306e+24, 4.2245e-39, 6.8664e-44, 0.0000e+00]])
```

## 原因分析

在`add_kernel`函数内部执行`y = x + 1`时，这里的`y`是函数的局部变量（相当于创建了一个新的变量`y`），它会覆盖传入参数`y`的引用。也就是说，这行代码只是让函数内的`y`指向了`x + 1`的新Tensor，并不会修改外部传入的Tensor `y`的内容。

## 解决措施

通过全切片操作符`[:]`，将计算结果写入函数参数`y`的原有内存空间。

示例代码：

```python
@pypto.frontend.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y[:] = x + 1  # 将x+1的结果写入函数参数y的原有内存空间

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y)  # 输出x + 1的结果
```

输出数据：

```python
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
```

其中`y[:] = x + 1`也可以替换为`y.move(x + 1)`或者`y.assemble(x + 1, [0, 0])`。
