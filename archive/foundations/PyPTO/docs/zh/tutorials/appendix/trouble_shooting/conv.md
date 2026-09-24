# FC6XXX-FC8XXX

## FC6101 OVER_BUFFER_LIMIT

**错误描述**

Operation阶段校验TileShape配置超出硬件缓存空间限制，包括L0A、L0B、L0C和L1缓存。

**可能原因**

- L0A空间超限：`tileH * tileW * tileK * sizeof（dtype） > L0A_size`。

   ```python
   # 错误示例 - L0A空间超限制
   # 例如L0A_size=65536 bytes
   # FP16（sizeof=2）, tileH=16, tileW=16, tileK=256
   # 16 * 16 * 256 * 2 = 131072 > 65536（L0A_size）
   tile_l0_info = pypto_impl.TileL0Info(
       tileH=16, tileW=16, tileK=256, tileN=16
   )
   ```

- L0B空间超限：`tileK * tileN * sizeof（dtype） > L0B_size`。
- L0C空间超限：`tileH * tileW * tileN * sizeof（FP32） > L0C_size`。
- L1空间超限：输入、权重和bias在L1中的总占用超出L1_size，公式：`CeilAlign（hinL1 * winL1 * kAL1 * sizeof（dtype）, 32） + CeilAlign（nL1 * kBL1 * sizeof（dtype）, 32） + CeilAlign（tileN * sizeof（dtype）, 32） > L1_size`。

**处理方式**

1. 根据报错日志中提示的缓存名称（L0A/L0B/L0C/L1）和实际tile值，结合当前芯片型号的buffer大小，计算占用空间是否超限。
2. 减小对应维度的Tile大小，使`tileH * tileW * tileK * sizeof（dtype） ≤ L0A_size`等约束满足。
3. 参考pypto.set_conv_tile_shapes接口TileShape空间约束说明。

   ```python
   # 正确示例-基于test_conv.py: test_conv2d_fp16_basic_with_bias
   # 例如L0A_size=65536, L0B_size=65536, L0C_size=131072
   # FP16（sizeof=2）, tileH=1, tileW=16, tileK=16
   # L0A: 1 * 16 * 16 * 2 = 512 <= 65536
   # L0B: 16 * 16 * 2 = 512 <= 65536
   # L0C: 1 * 16 * 16 * 4 = 1024 <= 131072
   tile_l0_info = pypto_impl.TileL0Info(
       tileH=1, tileW=16, tileK=16, tileN=16
   )
   ```

## FC6201 EXPANDFUNC_TENSOR_OP_NULLPTR

**错误描述**

Tile图切分阶段，在为MMAD（矩阵乘累加）操作设置属性时，fmap/weight/res的tensor图节点指针为空。

**可能原因**

- Conv Operation的输入/输出Tensor未正确传入或被提前释放，导致Tile图展开时tensor节点缺失。

   ```python
   # 错误示例-在动态场景中，view操作的offset或shape传入异常，导致Tile图展开时无法正确构造fmap子tensor节点
   input_a_view = pypto.view(input_a, [tile_batch, 16, 64], [batch_offset, 0, 0])
   # 若input_a本身为None或未正确初始化，Tile图展开时fmapTensorPtr为空
   out = pypto.conv(input_a_view, input_b, dtype, [1], [1, 1], [1], extend_params={}, groups=1)
   ```

- 在动态shape场景下，view/assemble操作构造的子Tensor链路异常，导致传递到MMAD节点的tensor指针丢失。

**处理方式**

1. 开启编译debug模式dump计算图，确认Conv Operation的输入/输出tensor节点是否完整：

   ```python
   @pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
   def conv_kernel():
   ```

2. 在output目录下使用pto-toolkit打开dump的计算图，定位到报错的Conv Operation节点，检查其输入tensor（fmap、weight）和输出tensor（res）是否均存在且非空。
3. 若为动态shape场景，检查`pypto.view`的offset和shape参数是否合法，确认view后的子tensor能正确传递到`pypto.conv`。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带完整报错日志、dump计算图与复现用例。

## FC6202 EXPANDFUNC_TENSOR_ATTR_GET_FAILED

**错误描述**

Tile图切分阶段，Conv Operation的原始fmap/weight shape属性（`CONV_ORI_FMAP_SHAPE_ATTR`/`CONV_ORI_WEIGHT_SHAPE_ATTR`）未设置。该校验在NZ2NZ模式下触发。

**可能原因**

- 在NZ2NZ模式下使用Conv，但Operation构造流程未正确设置原始fmap/weight shape属性。

   ```txt
   # 错误示例-在NZ2NZ模式下，Conv Operation缺少ori_fmap_shape属性
   # 报错日志示例：
   # Conv ori fmapshape should be set when InOut Tensor NZ mode.
   ```

- Conv Operation通过非标准方式构造（如自定义Pass修改了Conv节点属性），导致必要属性丢失。

**处理方式**

1. 确认运行芯片型号，检查是否走了NZ2NZ路径（`ConstructTensorGraphNZ2NZ`）。
2. 开启编译debug模式dump计算图，在dump图中搜索Conv Operation节点，检查其属性列表是否包含`ori_fmap_shape`和`ori_weight_shape`。
3. 确认Conv Operation是否通过标准`pypto.conv`接口构造，若使用了自定义Pass修改Conv节点，检查是否误删了shape属性。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带完整报错日志、芯片型号信息与复现用例。

## FC6203 EXPANDFUNC_TILE_OP_NULLPTR

**错误描述**

Tile图切分阶段，Tile图新生成的节点出现空指针，包括：当前Function指针为空、fmap/weight/bias/res的tile tensor指针为空。

**可能原因**

- Conv Operation未在`@pypto.frontend.jit`装饰的动态函数内调用，导致获取当前Function指针失败（`functionPtr`为空）。

   ```python
   # 错误示例-未在jit动态函数内调用conv，functionPtr为空
   def not_under_jit_example(fmap, weight):
       output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1])
       return output  # functionPtr为空，触发FC6203
   ```

- 配置了`hasBias=True`但未传入bias tensor，或bias tensor在Tile图展开过程中丢失，导致`biasTensorPtr`为空。

   ```python
   # 错误示例-hasBias为True但bias在extend_params中传入异常
   extend_params = {'bias_tensor': None}  # bias为None，hasBias标记为True但biasTensorPtr为空
   output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1], extend_params=extend_params)
   ```

- Tile图展开过程中，L0层级的fmap/weight/res子tensor节点构造失败。

**处理方式**

1. 确认`pypto.conv`调用位于`@pypto.frontend.jit`装饰的函数内部，不能在普通Python函数中直接调用。
2. 若使用了bias，确认`extend_params['bias_tensor']`传入的是有效的Tensor对象，而非None。
3. 开启编译debug模式dump计算图，检查Tile图展开后各节点的tensor连接关系是否完整。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带完整报错日志与复现用例。

## FC6204 EXPANDFUNC_PARAMS_INVALID

**错误描述**

Tile图切分阶段，Conv Operation的输入操作数数量与预期不匹配。预期操作数数量 = 2（fmap + weight） + hasBias（0或1）。

**可能原因**

- Conv Operation的操作数在图传递过程中被异常增减，如自定义Pass错误地添加或删除了Conv的输入边。

   ```txt
   # 错误示例-Conv Operation的操作数数量与hasBias标记不一致
   # 报错日志示例：
   # Operand vector size mismatch: Expected size: 3, actual size: 2, Conv Common Input: 2, hasBias: True
   ```

- bias操作数传递逻辑异常：`hasBias`标记为True但实际操作数中缺少bias，或`hasBias`为False但操作数中多出了bias。

**处理方式**

1. 开启编译debug模式dump计算图，在dump图中定位报错的Conv Operation节点，检查其输入边的数量和类型。
2. 确认输入操作数数量与`hasBias`标记一致：无bias时操作数为2（fmap + weight），有bias时操作数为3（fmap + weight + bias）。
3. 若使用了自定义Pass，检查是否误修改了Conv Operation的输入边。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带完整报错日志、dump计算图与复现用例。

## FC6205 EXPANDFUNC_INNER_STATUS_FAILED

**错误描述**

Tile图切分阶段，内部功能函数返回值异常。该错误码为预留错误码。

**可能原因**

NA

**处理方式**

1. 此错误码为预留码，当前版本未使用。若遇到此报错，请按CONV组件提issue或问题单，附带完整报错日志与复现用例。

## FC6301 CODEGEN_GET_ATTR_FAILED

**错误描述**

Codegen代码生成阶段，获取Conv TileOp的CopyInMode或CopyOutMode属性失败。这两个属性在Tile图展开阶段设置，Codegen阶段读取。

**可能原因**

- Tile图展开阶段未正确设置CopyInMode/CopyOutMode属性，通常由Tile图展开流程异常导致。

   ```txt
   # 错误示例-Codegen阶段读取CopyInMode属性时返回失败
   # 报错日志示例：
   # GenMemL1CopyInConv get CopyInMode failed.
   # GenMemL0CCopyOutConv get CopyOutMode failed.
   ```

- Conv Operation的Load/Store操作节点在Pass阶段被异常修改，导致属性丢失。

**处理方式**

1. 开启编译debug模式dump计算图，在dump图中定位到Conv的Load（CopyIn）/Store（CopyOut）操作节点，检查其属性是否包含`COPY_IN_MODE`/`COPY_OUT_MODE`。
2. 检查是否使用了自定义Pass修改了Conv的Load/Store节点，确认属性未被误删。
3. 在生成的kernel代码文件（`kernel_aicore`目录下的`TENSOR***.cpp`）中搜索对应的TileOp调用，确认参数是否完整。
4. 若上述排查均未发现问题，请按Codegen组件提issue或问题单，附带完整报错日志、dump计算图与复现用例。

## FC6302 CODEGEN_CHECK_ATTR_INVALID

**错误描述**

Codegen代码生成阶段，Conv TileOp的CopyInMode/CopyOutMode属性值不在合法范围内，或cutW属性为0。

**可能原因**

- CopyInMode值不在`[ND2NZ, DN2NZ]`范围内。

   ```txt
   # 错误示例-CopyInMode/CopyOutMode属性值非法或cutW为0
   # 报错日志示例：
   # GenMemL1CopyInConv CopyInMode is invalid: -1
   # GenMemL0CCopyOutConv CopyOutMode is invalid: 99
   # GenMemL0CCopyOutConv cutW should not be 0!
   ```

- CopyOutMode值不在`{NZ2ND, NZ2NZ, NZ2DN}`范围内。
- cutW属性为0，导致L0C到GM的搬出操作无法正确分块。

**处理方式**

1. 开启编译debug模式dump计算图，在dump图中定位到Conv的Load/Store操作节点，检查`COPY_IN_MODE`/`COPY_OUT_MODE`/`CUT_W`属性值是否合法。
2. CopyInMode合法值：`ND2NZ`（1）、`NZ2NZ`（2）、`DN2NZ`（3）；CopyOutMode合法值：`NZ2ND`（0）、`NZ2NZ`（1）、`NZ2DN`（3）。
3. 确认Tile图展开流程未被自定义Pass干扰，这些属性由框架根据输入tensor格式和芯片型号自动推导。
4. 若上述排查均未发现问题，请按Codegen组件提issue或问题单，附带完整报错日志、dump计算图与复现用例。

## FC6303 CODEGEN_CHECK_DIM_INVALID

**错误描述**

Codegen代码生成阶段，Conv TileOp的src shape或offset维度与预期不匹配。2D conv的shape/offset应为4维，3D conv应为5维，L0C的valid shape应为2维。

**可能原因**

- Tile图展开阶段生成的tensor shape维度与卷积类型不一致，如2D conv的fmap shape被错误地生成为5维。

   ```txt
   # 错误示例-shape/offset维度与卷积类型不匹配
   # 报错日志示例：
   # GenMemL1CopyInConv shape should be 4-dim!（2D conv期望4维，实际非4维）
   # GenMemL1CopyInConv offset should be 4-dim!（2D conv期望4维，实际非4维）
   # GenMemL0CCopyOutConv valid shape should be 2-dim!（L0C期望2维，实际非2维）
   ```

- 动态shape场景下，valid shape的维度推导异常。
- L0C tensor的shape不是2维（M，N），可能由Tile图展开时L0C节点构造错误导致。

**处理方式**

1. 开启编译debug模式dump计算图，在dump图中定位到报错的Conv Load/Store节点，检查其src tensor的shape维度。
2. 确认shape维度与卷积类型匹配：1D conv对应3维（NCL），2D conv对应4维（NCHW），3D conv对应5维（NCDHW）。
3. 若为动态shape场景，检查`pypto.view`构造的子tensor shape维度是否与原始输入一致。
4. 检查生成的kernel代码文件（`kernel_aicore`目录下），确认TileOp调用的shape/offset参数维度正确。
5. 若上述排查均未发现问题，请按Codegen组件提issue或问题单，附带完整报错日志、dump计算图与复现用例。

## FC6401 TILEOP_TENSOR_FORMAT_FAILED

**错误描述**

TileOp阶段，tensor硬件FORMAT校验失败。Conv的Load操作要求src为GM格式、dst为L1格式；Store操作要求src为L0C格式、dst为GM格式。

**可能原因**

- Tile图展开阶段为tensor分配了错误的内存层级，如Load操作的dst被分配到L0C而非L1。

   ```txt
   # 错误示例-Load/Store操作的src/dst硬件格式不匹配
   # 报错日志示例（编译期static_assert）：
   # [TLoadConv Error]: Src format should be GM and Dst format should be L1
   # [TStoreConv Error]: Src format should be L0C and Dst format should be GM
   ```

- 自定义Pass修改了tensor的内存类型属性，导致Load/Store操作的src/dst格式不匹配。

**处理方式**

1. 此错误码对应的校验为编译期`static_assert`，会在kernel代码编译阶段报错。在生成的kernel代码文件（`kernel_aicore`目录下）中搜索`TLoadConv`/`TStoreConv`调用，检查传入的tensor类型声明。
2. 确认Load操作的dst tensor声明为L1类型（如`ConvTile<TileType::Mat, ..., Hardware::L1>`），src为GM类型。
3. 确认Store操作的src tensor声明为L0C类型，dst为GM类型。
4. 若未修改Tile图展开逻辑但仍出现此报错，请按CONV组件提issue或问题单，附带kernel代码文件与复现用例。

## FC6402 TILEOP_SHAPE_SIZE_FAILED

**错误描述**

TileOp阶段，L0C tensor的shape size校验失败。Store操作要求L0C的shape为2维（M, N）。

**可能原因**

- Tile图展开阶段为L0C tensor构造了非2维的shape，可能由Conv的MMAD节点输出shape推导异常导致。

   ```txt
   # 错误示例-L0C tensor shape不是2维
   # 报错日志示例（编译期static_assert）：
   # L0C shape size should be 2 Dim
   ```

- 自定义Pass修改了L0C tensor的shape定义。

**处理方式**

1. 此错误码对应的校验为编译期`static_assert`。在生成的kernel代码文件中搜索`TStoreConv`调用，检查L0C tensor的shape声明。
2. 确认L0C tensor的shape为2维，即`（M, N）`，其中`M = tileH * tileW`，`N = tileN`。
3. 开启编译debug模式dump计算图，在dump图中检查MMAD输出到Store操作的L0C tensor节点shape。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带kernel代码文件与复现用例。

## FC6403 TILEOP_STC_SHAPE_INVALID

**错误描述**

TileOp阶段，static shape（编译期常量shape）校验非法。Conv TileOp的部分维度要求为编译期常量，运行时无法修改。

**可能原因**

- TileShape配置中部分维度需要为编译期常量但被设置为动态值，导致`std::tuple_element`无法在编译期提取。

   ```txt
   # 错误示例-TileShape的静态维度配置异常，如tileCinFmap或tileN配置为0，导致bufferSize = 0
   ```

- TileShape的buffer size计算结果为0或负数（如某维度配置为0）。

**处理方式**

1. 检查`pypto.set_conv_tile_shapes`设置的TileShape各维度值是否均为正整数。
2. 确认TileShape中需要为编译期常量的维度（如`tileN`、`tileWout`等）在动态shape场景下使用了`pypto.symbolic_scalar`正确声明。
3. 开启编译debug模式，查看编译日志中TileShape的展开结果是否合法。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带完整报错日志与复现用例。

## FC6404 TILEOP_INDEX_INVALID

**错误描述**

TileOp阶段，获取shape/stride的index校验非法。Conv TileOp的shape/stride访问index必须小于5。

**可能原因**

- Tile图展开阶段尝试访问tensor shape/stride的第5维及以上（index >= 5），可能由tensor shape维度推导异常导致。

   ```txt
   # 错误示例-shape/stride访问index >= 5
   # 报错日志示例（编译期static_assert）：
   # Idx should be less than 5
   ```

- 2D conv的tensor被错误地当作3D conv处理，或反之，导致维度访问越界。

**处理方式**

1. 此错误码对应的校验为编译期`static_assert`。检查报错对应的TileOp调用，确认传入的index参数是否合法（须 < 5）。
2. 开启编译debug模式dump计算图，检查Conv Load/Store节点的tensor shape维度是否与卷积类型匹配（2D conv为4维+pad=5维，3D conv为5维+pad=6维内部处理）。
3. 确认卷积类型判断正确：1D conv输入为3维，2D conv输入为4维，3D conv输入为5维。
4. 若上述排查均未发现问题，请按CONV组件提issue或问题单，附带kernel代码文件与复现用例。
