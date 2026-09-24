# pypto_pro.language.system.dcci

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

清理并失效数据缓存，保证跨核/跨流水数据一致。

## 函数原型

```python
pypto_pro.language.system.dcci(target, offset=None, *, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE, dst=pl.DcciDst.AUTO)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `target` | 输入 | GM Tensor或Vec Tile，指定缓存失效的目标 |
| `offset` | 输入 | 可选的元素偏移 |
| `cache_line` | 输入 | 缓存行范围 |
| `dst` | 输入 | 缓存目标 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `target` | 输入 | GM Tensor（`pypto_pro.language.Tensor`）或位于`MemorySpace.Vec`的Tile |
| `offset` | 输入 | 单位为元素<br>GM Tensor支持各维偏移列表/元组，也支持整型常量或运行时整型标量表达式表示的线性偏移；Vec Tile仅支持整型常量或运行时整型标量表达式表示的线性偏移<br>缺省时使用目标的起始地址 |
| `cache_line` | 输入 | `pl.CacheLine.ENTIRE_DATA_CACHE`（默认，全数据缓存失效）/ `pl.CacheLine.SINGLE_CACHE_LINE`（单缓存行失效）<br>全缓存失效性能开销较大，仅在大范围数据不一致时使用 |
| `dst` | 输入 | `pl.DcciDst.AUTO`（默认，自动选择）/ `pl.DcciDst.CACHELINE_OUT` / `pl.DcciDst.CACHELINE_UB` / `pl.DcciDst.CACHELINE_ALL` / `pl.DcciDst.CACHELINE_ATOMIC`<br>一般场景使用`pl.DcciDst.AUTO`即可 |

## 调用示例

以下为GM Tensor的调用片段。`inp`为二维GM Tensor，`offset=[0, 0]`表示从其起始地址执行单缓存行操作：

```python
pl.system.dcci(
    inp,
    [0, 0],
    cache_line=pl.CacheLine.SINGLE_CACHE_LINE,
    dst=pl.DcciDst.CACHELINE_OUT,
)
```

其他典型用法（节选）：

```python
# 全缓存失效
pl.system.dcci(inp, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE)
```
