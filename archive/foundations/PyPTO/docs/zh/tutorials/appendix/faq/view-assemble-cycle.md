# 同一个Tensor进行View和Assemble导致图成环报错

## 问题现象描述

示例代码如下：

```python
@pypto.frontend.jit
def foo_kernel(x, y):
    pypto.set_vec_tile_shapes(16, 16)
    a = pypto.zeros([32, 32])
    b = a[:16, :16]          # 从a中view获取数据
    a[16:, 16:] = b.exp()    # 计算后assemble写回a
    y[:] = x + a

torch.npu.set_device(0)
x = torch.ones(32, 32, dtype=torch.float32)
y = torch.empty(32, 32, dtype=torch.float32)
foo_kernel(pypto.from_torch(x), pypto.from_torch(y))
```

执行时报错：`ASSERTION FAILED`

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
```

详细报错如下：

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
Operation not fully processed: /* /home/pypto-dev/a.py:9 */
<32 x 32 x DT_FP32 / 32 x 32 x DT_FP32> %0@2#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10000 VEC_DUP(g:-1, s:-1) #SCALAR{0.000000} #op_attr_shape{[32, 32]} #op_attr_validShape{[32,32]}
, func GetSortedOperations, file function.cpp, line 1105
libtile_fwk_interface.so(npu::tile_fwk::Function::GetSortedOperations() const+0xb3c) [0xffff9c2f6650]
libtile_fwk_interface.so(npu::tile_fwk::Function::SortOperations()+0x38) [0xffff9c2f6f28]
libtile_fwk_interface.so(npu::tile_fwk::Function::EndFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&)+0x960) [0xffff9c31b8d0]
libtile_fwk_interface.so(npu::tile_fwk::Program::FinishCurrentFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&, bool)+0x1b0) [0xffff9c532274]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x10c) [0xffff9c536dcc]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndHiddenLoop(npu::tile_fwk::Function*, bool)+0xb0) [0xffff9c537384]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x5c) [0xffff9c536d1c]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::IterationEnd()+0x44) [0xffff9c5399c4]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::Iterator::operator!=(npu::tile_fwk::RecordLoopFunc::IteratorEnd const&)+0xfc) [0xffff9c539ea0]
```

## 原因分析

该报错的原因是内部在对基本算子做拓扑排序时发现存在环路的报错。

这是由于数据从`a`中读取又写回`a`导致的。由于PyPTO描述的是一个图表达，在读取和写入的时候，当前认为`a`是一个整体，因此创建的连接关系会形成一个环路，即`a → b → b.exp() → a`，而PyPTO不允许构造出的图内存在环路，必须为DAG（有向无环图），所以才有这个报错。

![](../../figures/view_assemble_cycle_1.png)

## 解决措施

- 当前需要将读取和写入`a`的逻辑拆分成两个图去定义，避免一个图内存在环路。
- 后续等Assemble的SSA语义上线后使用该写法不会有问题。

![](../../figures/view_assemble_cycle_2.png)
