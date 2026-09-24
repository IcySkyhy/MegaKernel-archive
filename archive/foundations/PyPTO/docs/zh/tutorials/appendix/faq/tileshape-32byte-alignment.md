# set_xxx_tile_shapes最后一维未32字节对齐

## 问题现象描述

```python
with pypto.function("TENSOR_SUM_FP32", [x], [res]):
    for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
        pypto.set_vec_tile_shapes(4, 8)
        res.move(x.sum())
```

通过`pypto.set_xxx_tile_shapes`设置TileShape大小，最后一维需要32字节对齐，否则会校验报错：

```text
C++ exception with description "ASSERTION FAILED: vecTile[lastDim] % alignNum == 0
Sum op: the tileShape of last axis need to 32Byte align!, func Sum, file reduction.cpp, line 374
libtile_fwk_interface.so(npu::tile_fwk::Sum(npu::tile_fwk::Tensor const&, int, bool)+0x620) [0xffff9ff2e090]
```

## 问题原因

硬件指令限制处理的数据需要32字节对齐。

## 处理步骤

通过`pypto.set_xxx_tile_shapes`设置TileShape大小时，需要将最后一维大小设置成32字节对齐的数，即`TileShape[-1] * sizeof(dtype) % 32 == 0`。
