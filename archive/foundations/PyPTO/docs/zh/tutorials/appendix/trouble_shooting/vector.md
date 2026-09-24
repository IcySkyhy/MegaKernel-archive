# FC0XXX-FC2XXX

## FC0000 ERR_PARAM_INVALID

**错误描述**

Vector入参非法错误，如参数取值、维度、格式等不满足约束。

**可能原因**

- 不满足Shape约束：输入输出Tensor为空Tensor（某维度为0），或Shape Size大于2147483647（即INT32_MAX）。

   ```python
   # 错误示例-Shape Size大于INT32_MAX
   a = pypto.tensor([65536, 65536], pypto.DT_FP16)  # 65536*65536>INT32_MAX
   b = pypto.tensor([65536, 65536], pypto.DT_FP16)
   out = pypto.pow(a, b)  # 触发ERR_PARAM_INVALID
   ```

- 不满足归约轴约束：归约类算子（如amax）的dim超出输入维度范围。

   ```python
   # 错误示例-amax的dim超出输入维度范围
   x = pypto.tensor([2, 3], pypto.DT_FP32)  # 2维
   out = pypto.amax(x, 2)  # dim=2越界，仅支持0、1或负索引
   ```

**处理方式**

1. 查阅对应算子文档（如[pypto.add](../../../api/operation/pypto-add.md)、[pypto.sin](../../../api/operation/pypto-sin.md)、[pypto.cast](../../../api/operation/pypto-cast.md)、[pypto.amax](../../../api/operation/pypto-amax.md)）确认输入输出Shape、维度等满足要求。

   ```python
   # 正确示例-输入为2维Tensor
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP16)
   out = pypto.add(a, b)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC0001 ERR_PARAM_DTYPE_UNSUPPORTED

**错误描述**

Vector入参数据类型不支持：使用了当前算子或硬件不支持的dtype。

**可能原因**

- 不满足算子数据类型约束：输入dtype不在该算子支持集合内。各算子支持的数据类型详见对应算子文档。

   ```python
   # 错误示例-sin仅支持DT_FP32/DT_FP16，传入sin不支持的DT_INT64
   x = pypto.tensor([4], pypto.DT_INT64)
   out = pypto.sin(x)  # 触发ERR_PARAM_DTYPE_UNSUPPORTED
   ```

- 不满足输入数据类型一致性约束：二元运算两输入Tensor数据类型不一致。

   ```python
   # 错误示例-两输入数据类型不一致（DT_FP16+DT_FP32）
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP32)
   out = pypto.add(a, b)
   ```

**处理方式**

1. 查阅对应算子文档（如[pypto.add](../../../api/operation/pypto-add.md)、[pypto.sin](../../../api/operation/pypto-sin.md)、[pypto.cast](../../../api/operation/pypto-cast.md)）确认输入输出数据类型满足要求，切换为兼容的数据类型重试。

   ```python
   # 正确示例-sin使用支持的DT_FP32
   x = pypto.tensor([4], pypto.DT_FP32)
   out = pypto.sin(x)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC0002 ERR_PARAM_SHAPE_DIM_UNSUPPORTED

**错误描述**

Vector入参Shape维度不支持：输入输出Tensor的维度数不在算子支持的范围内，或多输入间维度数不一致。

**可能原因**

- 不满足维度数约束：输入输出Tensor的维度数不在算子支持的范围内（如多数Vector算子仅支持1-4维）。

   ```python
   # 错误示例-输入为5维，超过Vector支持的1-4维范围
   a = pypto.tensor([2, 2, 2, 2, 2], pypto.DT_FP16)  # 5维
   b = pypto.tensor([2, 2, 2, 2, 2], pypto.DT_FP16)
   out = pypto.pow(a, b)  # 触发ERR_PARAM_SHAPE_DIM_UNSUPPORTED
   ```

- 不满足多输入维度一致性约束：参与运算的多个Tensor维度数不一致。

   ```python
   # 错误示例-两输入维度数不一致（2维与3维）
   a = pypto.tensor([4, 4], pypto.DT_FP16)        # 2维
   b = pypto.tensor([2, 4, 4], pypto.DT_FP16)     # 3维
   out = pypto.pow(a, b)
   ```

**处理方式**

1. 查阅对应算子文档（如[pypto.pow](../../../api/operation/pypto-pow.md)、[pypto.gcd](../../../api/operation/pypto-gcd.md)、[pypto.amax](../../../api/operation/pypto-amax.md)）确认支持的维度范围，并保证各输入Tensor维度数一致且在范围内。

   ```python
   # 正确示例-输入为2维Tensor且维度数一致
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP16)
   out = pypto.pow(a, b)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC0003 ERR_PARAM_COUNT_INVALID

**错误描述**

Vector入参操作数（operand）个数非法：算子在执行或输出shape推导时，检测到输入/输出操作数个数与预期不符。

**可能原因**

- 不满足操作数个数约束：传入算子的输入/输出操作数个数与该算子预期不符，错误日志通常形如`iOperands.size() should be ...`。多见于调用方式与算子文档要求的输入输出个数不一致，或框架内部算子接线错误。

**处理方式**

1. 对照打屏日志中的操作数期望个数，查阅对应算子文档（如[pypto.scatter_](../../../api/operation/pypto-scatter_.md)、[pypto.concat](../../../api/operation/pypto-concat.md)、[pypto.where](../../../api/operation/pypto-where.md)）确认输入输出个数与调用方式匹配。
2. 若调用方式正确仍报此错（疑似框架内部接线问题），请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC1000 ERR_CONFIG_TILE

**错误描述**

Vector切分（Tile）配置非法。

**可能原因**

- 不满足Tile值约束：set_vec_tile_shapes传入的某维度小于等于0。

   ```python
   # 错误示例-TileShape第二维为0
   pypto.set_vec_tile_shapes(4, 0)  # 每个维度必须大于0
   a = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.sin(a)
   ```

- 不满足TileShape维度数量约束：set_vec_tile_shapes传入的维度数超过4个。

   ```python
   # 错误示例-TileShape超过4个维度
   pypto.set_vec_tile_shapes(1, 1, 1, 1, 1)  # 最多不超过4个维度
   ```

- 不满足TileShape大小约束：set_vec_tile_shapes设置超过该算子的TileShape大小约束。

   ```python
   # 错误示例-amax的TileShape超过64KB
   pypto.set_vec_tile_shapes(1024, 1024)  # FP16下1024*1024*2B=2MB，超过64KB
   x = pypto.tensor([1024, 1024], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

- 不满足TileShape维度一致性约束：TileShape维度数与输出（或输入）Tensor维度数不匹配。

   ```python
   # 错误示例-TileShape维度与输入维度不一致
   pypto.set_vec_tile_shapes(4, 16)  # 2维TileShape
   a = pypto.tensor([2, 2, 2, 2], pypto.DT_FP16)  # 4维输入
   out = pypto.add(a, a)
   ```

**处理方式**

1. 查阅[pypto.set_vec_tile_shapes](../../../api/config/pypto-set_vec_tile_shapes.md)及对应算子文档确认TileShape取值满足要求。

2. 调用[pypto.set_vec_tile_shapes](../../../api/config/pypto-set_vec_tile_shapes.md)前确认各维度均为正数、维度数不超过4，且与输出维度一致。

   ```python
   # 正确示例-TileShape各维度为正且与输入维度一致
   pypto.set_vec_tile_shapes(4, 16)
   a = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.sin(a)
   ```

3. 通过[pypto.get_vec_tile_shapes](../../../api/config/pypto-get_vec_tile_shapes.md)回读实际生效的TileShape，核对是否符合切分约束：

   ```python
   pypto.set_vec_tile_shapes(4, 16)
   tile_shape_info = pypto.get_vec_tile_shapes()
   print(tile_shape_info)
   # 输出：[4, 16]
   ```

4. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC1001 ERR_CONFIG_ALIGNMENT

**错误描述**

Vector对齐约束不满足：地址或shape未按硬件要求对齐。

**可能原因**

- 不满足尾轴对齐约束：不满足算子（如amax）尾轴32字节对齐的约束。

   ```python
   # 错误示例-amax尾轴未32字节对齐
   pypto.set_vec_tile_shapes(4, 10)  # FP16下10*2B=20B，非32字节对齐
   x = pypto.tensor([4, 10], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

**处理方式**

1. 查阅对应算子文档（如[pypto.amax](../../../api/operation/pypto-amax.md)）及[pypto.set_vec_tile_shapes](../../../api/config/pypto-set_vec_tile_shapes.md)确认对齐要求。关注reshape/view、交换维度（转置）等是否改变内轴对齐要求，必要时调整Tensor形状或TileShape取值。

   ```python
   # 正确示例-amax尾轴32字节对齐（FP16下16元素）
   pypto.set_vec_tile_shapes(4, 16)
   x = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC2000 ERR_RUNTIME_NULLPTR

**错误描述**

Vector运行时空指针：传入的Tensor为空。

**可能原因**

NA

**处理方式**

1. 确认传入Vector接口的输入输出Tensor均非空且已完成地址分配，确认是否存在nullptr。

   ```python
   # 正确示例-输入Tensor均已分配数据
   a = pypto.tensor([16, 32], pypto.DT_FP16, "a")
   b = pypto.tensor([16, 32], pypto.DT_FP16, "b")
   out = pypto.add(a, b)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC2001 ERR_RUNTIME_LOGIC

**错误描述**

Vector运行时逻辑错误：计算流程进入未定义或异常分支。

**可能原因**

NA

**处理方式**

1. 通过相关日志定位异常路径，核对计算流程是否进入未定义/异常分支，确认中间结果、索引值是否符合预期。

2. 检查核心计算逻辑的前置条件（如TileShape设置、上下文与配置句柄初始化）是否满足。

3. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。
