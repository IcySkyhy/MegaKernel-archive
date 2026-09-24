# view未传入valid_shape导致精度问题

## 问题现象描述

部分场景使用`view`时未传入`valid_shape`导致精度问题。

## 问题原因

当`view`接口的输入Tensor没有一个正确的validShape时，框架无法正确推导出输出的validShape。

## 处理步骤

当怀疑`view`部分的validShape推导有问题时，首先给`view`传入一个`valid_shape`，观察输出结果是否符合预期。

一个典型的需要传入`valid_shape`的场景：

当输入的validShape依赖别的Tensor标识，必须传入`dynValidShape`。如下所示场景，`q0`的validShape `curSeq`来自于另外一个Tensor，无法通过推导得到：

```python
# 输入input [B, S, H]
# 输入act_seqs [B]
# 输出out [B, S, H]
# 计算过程AddS
# 代码如下：
for b_idx in pypto.loop(B, name="b_loop", idx_name="b"):
    cur_seq = act_seqs[b_idx]
    a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
    a1 = a0 + 1.0
    out[b_idx:, :, :] = a1
```
