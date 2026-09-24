# PyTorch集成和接入

当前PyPTO支持单算子模式（eager）和图捕获模式（aclgraph）两种执行模式：

- 单算子模式（eager）：代码执行方式与普通Python程序一致，即时执行，函数在定义时立即执行，无需构建计算图，开发调试友好，但是会带来Host的任务下发开销，随着性能优化的不断深入，这些Host开销逐渐成为瓶颈，变成不可忽视的问题。
- 图捕获模式（aclgraph）：采用Capture&Replay方式实现任务一次捕获多次执行，Capture阶段捕获Stream任务到Device侧，暂不执行；Replay阶段从Host侧发出执行指令，Device侧再执行已捕获的任务，从而减少Host调度开销，提升性能。

在Kernel函数前添加@pypto.frontend.jit装饰器可默认在PyTorch框架中采用单算子模式执行，如需开启图捕获模式可以参考如下代码：

```python
B = pypto.DYNAMIC
N1, N2, DIM = 32, 1, 256

# enable frontend.jit for softmax
@pypto.frontend.jit()
def softmax_kernel(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
    output_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
):
    ...



@allow_in_graph
def softmax(x: torch.Tensor, dynamic: bool = True) -> torch.Tensor:
    if isinstance(x, FakeTensor):
        return torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')
    # launch the kernel
    out = torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')
    softmax_kernel(x, out)
    return out


class MM(torch.nn.Module):
    def forward(self, x, dynamic):
        out = softmax(x, dynamic)
        return out


def test_softmax_capture(device_id=None, dynamic: bool = True) -> None:
    # prepare data
    ...
    model = torch.compile(MM(), backend="eager", dynamic=True)
    #graph capture
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y = model(x, dynamic)

    #execute graph
    g.replay()
    torch.npu.synchronize()
    ...

if __name__ == "__main__":
    test_softmax_capture()
```

可以查看info级别Host编译日志，搜索capture mode关键字，默认为0表示关闭图捕获模式，如果为1则表示启用图捕获模式。

```text
2025-12-08 20:56:27.043    capture mode[1]
```

完整的样例请参考：[aclgraph.py](../../../../examples/03_advanced/aclgraph/aclgraph.py)。
