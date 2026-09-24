# 功能调试

PyPTO Pro采用SPMD执行模型，同一份Kernel代码由多个逻辑AI Core并行执行，各工作单元通过[`pl.get_block_idx()`](../../api/SIMD-API/operation/system_variables/get_block_idx.md)的全局逻辑索引确定数据分片。`block_dim`配置逻辑Block数；混合Kernel中AIC/AIV的实际逻辑核数还取决于两者比例。功能调试应围绕Kernel的编译、下发、多核切分、片上计算和结果写回逐层开展。

PyPTO Pro提供[`pl.printf`](../../api/Utils-API/debugging/printf.md)、[`pl.dump_data`](../../api/Utils-API/debugging/dump_data.md)、[`pl.pto_assert`](../../api/Utils-API/debugging/pto_assert.md)和[`pl.trap`](../../api/Utils-API/debugging/trap.md)等Kernel级调试接口。当Kernel包含`pl.printf`或`pl.dump_data`时，JIT自动启用设备侧调试打印。

## 调试流程

建议按以下顺序定位问题：

1. 构造可稳定复现问题的最小用例，固定输入Shape、数据类型、TilingKey、`block_dim`和随机种子。
2. 判断问题发生在编译阶段、Kernel下发阶段还是执行阶段。
3. 先使用单核和小Shape验证单个数据块的计算逻辑，再恢复多核验证数据切分。
4. 对照可信的CPU或NPU参考实现检查最终结果，并逐步比较中间结果。
5. 修复后恢复完整Shape、全部TilingKey、全部数据类型和生产核数进行回归。
6. 发布前移除调试打印、数据转储、调试断言和主动中止逻辑。

## 定位编译问题

被`@pl.jit()`装饰的Kernel在首次调用时触发编译。首次调用失败时，应首先检查Python异常中的源码位置和错误信息，重点确认以下内容：

- Kernel参数的类型标注、Shape和数据类型是否与实际输入一致。
- Tile的Shape、数据类型和目标内存空间是否满足所调用API的约束。
- `pl.load`、`pl.move`和`pl.store`的源、目的及偏移是否合法。
- TilingData字段的名称、类型和使用方式是否一致。
- TilingKey是否已通过`@pl.jit(tiling_key=...)`绑定，启动时传入的字段是否完整且通过有效性校验。
- `pl.section_vector()`和`pl.section_cube()`中的操作是否属于对应的执行域。
- Tile地址范围、内存占用和mutex编号是否存在冲突。

JIT编译产物的基础目录默认为当前工作目录下的`./build/<kernel_name>__<arch>/`。每个编译实例均使用独立的TilingKey子目录：使用TilingKey时为`tk_<packed>/`，其中`<packed>`是Key的十六进制打包值；未使用TilingKey时为`tk_none/`。使用datatype特化时，TilingKey子目录位于`dt_<hash>/`下；使用静态签名特化时，基础目录名称还会包含对应的签名后缀。

可检查以下生成文件辅助定位：

| 位置 | 文件 | 用途 |
|---|---|---|
| 当前编译实例的`tk_<packed>/`或`tk_none/`目录 | `kernel.cpp` | 检查生成的Device侧Kernel、参数类型、控制流和数据搬运。 |
| 当前编译实例的`tk_<packed>/`或`tk_none/`目录 | `call_kernel.cpp` | 检查Host侧参数打包和Kernel启动代码。 |
| 当前编译实例的`tk_<packed>/`或`tk_none/`目录 | `call_kernel.so` | JIT调用的Host侧共享库。 |
| 当前编译实例的`tk_<packed>/`或`tk_none/`目录 | `*_tiling.h` | 检查生成的TilingData C结构体布局。 |
| 基础目录 | `pipeline_generated.py` | 启用自动流水变换时，检查变换后的Kernel实现。 |

这些文件是编译器生成的诊断产物，不应直接修改。需要修复问题时，应修改Python Kernel或其调用代码后重新编译。

## 定位运行问题

Kernel启动是异步操作。Host侧调用完成表示任务已下发，Kernel执行状态通过同步接口确认。调试时在Kernel启动后调用同步接口，使设备侧错误在当前调用点暴露：

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()
```

出现运行异常时，可按以下方式缩小范围：

- 将`block_dim`临时设为`1`，排除多核切分、重复写和跨核覆盖问题。
- 将输入缩小到一个或少量Tile，减少循环次数和尾块分支。
- 固定TilingKey和数据类型，只保留发生问题的编译期分支。
- 将复杂Kernel按“GM加载、片上计算、GM写回”拆分检查。
- 对动态偏移、循环边界、有效Shape和尾块长度增加标量打印。
- 对关键Tensor或Tile增加小窗口数据转储。

单核验证通过后必须恢复生产使用的`block_dim`，并检查所有数据是否恰好被一个Core处理。单核正确不能证明多核切分正确。

## Kernel调试接口

### 打印标量信息

使用`pl.printf`打印Core编号、循环变量、偏移、Shape和分支标志等标量信息。当Kernel包含`pl.printf`或`pl.dump_data`时，框架自动开启设备侧调试打印，无需手动设置：

```python
import pypto_pro.language as pl


@pl.jit()
def debug_kernel(out: pl.Tensor[[16], pl.DT_INT32]):
    core_id = pl.get_block_idx()
    block_num = pl.get_block_num()
    if core_id == 0:
        pl.printf("core_id=%d, block_num=%d\n", core_id, block_num, loc=True)
```

`pl.printf`的格式串必须是编译时常量，仅支持受限的C `printf`格式。其中`%d`和`%i`用于有符号整数，`%u`用于无符号整数，`%x`用于十六进制，`%p`用于指针，`%f`仅支持FP32。打印内容通过设备侧打印机制输出，具体查看位置由运行环境的日志配置决定。

多核Kernel中的打印会由每个Core分别执行，容易产生大量交错日志。通常应暂时使用单核，或通过`pl.get_block_idx()`仅打印指定Core的信息。`pl.printf`有显著运行时开销，不得用于性能测量。

### 转储Tensor或Tile

使用`pl.dump_data`打印GM Tensor或片上Tile的数据。优先指定`offsets`和`shapes`转储小窗口，避免输出量过大：

```python
# 打印二维Tensor左上角的8 × 8窗口
pl.dump_data(out, offsets=[0, 0], shapes=[8, 8], loc=True)

# 打印当前Vector Tile
pl.dump_data(tile_out)
```

使用时需满足以下约束：

- `offsets`和`shapes`必须同时提供或同时省略，列表长度必须等于数据维数。
- Tensor窗口的最内维stride必须在编译期确定为`1`。
- Vec、Left和Right Tile转储时无需用户提供`workspace`。
- Acc（L0C）Tile必须通过`workspace`传入GM Tensor作为中转空间，`workspace`的数据类型应与Tile相同，Shape不得小于待转储区域。
- 多核场景应限制转储Core和转储窗口，避免不同Core的输出相互干扰。

`pl.dump_data`的内容通过设备侧打印机制输出，具体查看位置由运行环境的日志配置决定。该接口会改变Kernel的执行开销，仅用于功能定位。

### 检查运行条件

使用`pl.pto_assert`检查动态偏移、循环边界和运行时标志。条件必须是BOOL标量：

```python
pl.pto_assert(offset < total_length, "offset=%d, total=%d", offset, total_length, loc=True)
```

`pl.pto_assert`在条件不满足时通过设备侧打印机制记录信息，不会中止Kernel，也不会在Host侧抛出异常。因此，不能依赖`pl.pto_assert`阻止越界访问或保证后续代码安全。需要无条件停止执行时可使用`pl.trap()`，但该接口会直接中止Kernel，仅应在明确的调试分支中临时使用。

## 精度问题定位

### 建立可复现的精度基线

精度比对应保证测试输入、Shape、数据类型和布局一致。随机输入应固定随机种子，并保留问题输入。可使用`torch.testing.assert_close`比较实际结果和参考结果：

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()

torch.testing.assert_close(
    actual.cpu(),
    expected.cpu(),
    rtol=rtol,
    atol=atol,
)
```

`rtol`和`atol`应根据输入数据类型、累加数据类型、计算规模和算法误差确定，不应为了使失败用例通过而任意放宽阈值。比较前还应分别检查NaN、Inf、最大绝对误差、最大相对误差及首个错误位置。

### 逐层比较中间结果

最终结果不一致时，可沿数据流逐层定位：

1. 检查每个Core计算的首个和末个GM偏移，确认多核覆盖范围。
2. 转储GM加载后的Tile，确认Shape、stride、offset和尾块填充值。
3. 转储关键计算步骤后的Tile，确定首次产生误差的位置。
4. 检查写回前的有效Shape和写回偏移，排除尾块越界或结果覆盖。
5. 对Cube计算检查左右矩阵的布局、转置方式、L0A/L0B格式和累加数据类型。

应优先转储一个Core、一个Tile和少量元素。确认当前层正确后再向下一层移动，避免一次加入大量调试代码而改变问题表现。

## 常见问题检查

| 现象 | 重点检查项 |
|---|---|
| 单核正确，多核错误 | 数据分片是否遗漏或重叠，`block_dim`与`pl.get_block_num()`是否一致，尾块是否分配均衡。 |
| 仅部分Shape错误 | Tile数量的向上取整、尾块有效Shape、补齐值、动态stride和边界条件。 |
| 仅某个TilingKey错误 | 启动时Key字段、Key有效性约束、编译期分支以及对应产物目录。 |
| 结果整体错位 | `make_tensor`的Shape和stride、load/store偏移、布局解释和转置设置。 |
| 误差随归约长度增大 | 累加数据类型、计算顺序、类型转换和舍入方式。 |
| 偶发错误或结果不稳定 | Tile复用时的同步、mutex配置、流水间依赖以及跨核写冲突。 |
| Kernel无Host异常但日志有断言失败 | `pl.pto_assert`只记录日志，不会中止Kernel。 |
| 加入打印后性能显著下降 | `printf`和`dump_data`包含设备侧调试开销，属于预期现象。 |

## 调试完成后的检查

问题修复后，应移除或关闭所有`pl.printf`、`pl.dump_data`、`pl.pto_assert`和`pl.trap`调用，并关闭不再需要的设备打印编译选项。随后使用生产配置重新编译，覆盖全部目标Shape、数据类型、TilingKey和`block_dim`进行精度与稳定性回归。
