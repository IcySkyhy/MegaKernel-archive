# AI Core算子离线二进制编译基本用法

PyPTO Pro Kernel可以接入算子工程的离线编译流程，生成AI Core算子二进制并随算子包发布。安装算子包后，可以通过aclnn调用算子；如需通过图模式调用，还需要补充`op_graph`、Graph Infer和GE算子原型注册等交付件。

离线编译主要包括以下步骤：

1. 准备算子工程，完成算子定义、InferShape和Host侧Tiling等Host侧实现。
2. 在`op_kernel`目录下使用PyPTO Pro实现Kernel。
3. 在`op_host/CMakeLists.txt`中配置PyPTO Pro Kernel。
4. 在Host侧Tiling函数中填充TilingData，并设置TilingKey、BlockDim和Workspace。
5. 使用算子工程的构建脚本编译算子包。

## 准备算子工程

算子工程需要包含Host侧实现、Device侧Kernel实现和调用接口。下面展示与PyPTO Pro离线编译直接相关的主要目录；算子在工程中的上级分类目录以实际工程为准。

```text
<operator_project>
├── build.sh
├── <op_class>
│   └── ${op_name}
│       ├── examples
│       │   └── test_aclnn_${op_name}.cpp
│       ├── op_host
│       │   ├── ${op_name}_def.cpp
│       │   ├── ${op_name}_infershape.cpp
│       │   ├── ${op_name}_tiling.cpp
│       │   └── CMakeLists.txt
│       ├── op_kernel
│       │   └── ${op_file}.py
│       └── op_graph                      # 仅图模式需要
└── CMakeLists.txt
```

各部分功能如下：

- `${op_name}_def.cpp`定义算子名称、输入、输出、属性、数据类型和支持的硬件平台。
- `${op_name}_infershape.cpp`实现输出Shape和数据类型推导。
- `${op_name}_tiling.cpp`根据输入Shape、数据类型和硬件资源计算TilingData、TilingKey、BlockDim及Workspace。
- `${op_file}.py`使用PyPTO Pro实现Device侧Kernel。
- `test_aclnn_${op_name}.cpp`通过aclnn接口调用并验证算子。标准工程通常根据算子定义和CMake中的`ACLNNTYPE aclnn`配置自动生成aclnn接口；仅在需要自定义接口逻辑时手工实现`op_api`。
- `op_graph`包含图模式所需的Graph Infer、算子原型注册等交付件，仅使用aclnn调用时不需要。

## 使用PyPTO Pro实现Kernel

Kernel文件放置在`op_kernel/${op_file}.py`。`${op_file}`不包含`.py`后缀，并且需要与CMake配置中的PyPTO Pro Kernel标记保持一致。

参与离线二进制编译的Kernel需要满足以下要求：

- 使用`@pl.jit`定义Kernel。
- 定义TilingKey，并通过`@pl.jit(tiling_key=...)`绑定到Kernel。
- 使用Python `@dataclass`定义TilingData，并将其作为Kernel参数。
- Kernel函数名与算子Kernel入口名称保持一致。
- Kernel业务输入输出参数名称与算子原型一致，形参排列顺序与Host侧Kernel参数下发顺序一致，因为离线交付按位置绑定Host侧下发参数与Device侧Kernel形参。参数可使用[`Tensor`](../../../api/SIMD-API/basic_data_structures/Tensor.md)或[`Ptr`](../../../api/SIMD-API/basic_data_structures/Ptr.md)声明。
- 在所有业务输入输出参数之后声明`workspace`，并将TilingData作为最后一个参数，即参数结尾固定为`workspace, tiling`。
- Kernel需要获取哪些输入或输出参数的数据类型，就在`@pl.jit(datatype=...)`字典中声明哪些参数。字典的key必须与算子原型中的参数名称一致，value必须是合法的Python标识符，且不能与Kernel参数名或TilingKey字段名冲突；声明后的变量可在Kernel中直接使用。

下面以`add_example`为例展示代码结构，省略具体计算逻辑：

```python
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class AddExampleTilingData:
    total_length: int
    tile_num: int


class AddExampleTilingKey:
    sch_mode = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(
    tiling_key=AddExampleTilingKey,
    datatype={
        "x": "data_dtype",
        "y": "data_dtype",
        "z": "data_dtype",
    },
)
def add_example(
    x: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    z: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AddExampleTilingData,
):
    # data_dtype可直接用于构造Tensor、TileType等。
    x_tensor = pl.make_tensor(x, [tiling.total_length], [1], dtype=data_dtype)
    y_tensor = pl.make_tensor(y, [tiling.total_length], [1], dtype=data_dtype)
    z_tensor = pl.make_tensor(z, [tiling.total_length], [1], dtype=data_dtype)
    # 使用tiling.total_length、tiling.tile_num等字段实现Kernel逻辑。
    ...
```

TilingData字段支持`int`、`float`、`bool`及对应的定长数组类型。字段声明顺序决定生成的C++结构体字段顺序和数据布局，修改或新增字段后需要同步修改Host侧Tiling赋值逻辑。

TilingKey用于描述需要生成独立Kernel实例的编译期配置。每个字段通过`TilingKeyField`声明位宽和候选值，构建系统会为合法的TilingKey组合生成对应的算子二进制。

`datatype`是一个描述Kernel所需参数数据类型的字典。key对应算子原型中的输入或输出参数名称，value是用户自定义的dtype变量名。只需添加Kernel计算过程中需要获取数据类型的参数；不依赖某个参数的数据类型时，无需将其加入字典。多个参数的数据类型相同时，可以像示例中的`x`、`y`和`z`一样映射到同一个变量，此时这些参数在编译时传入的实际数据类型必须一致；需要分别使用各参数的数据类型时，则映射到不同的变量。声明后的变量可直接用于`pl.make_tensor`、`pl.TileType`以及其他需要指定`dtype`的位置。

## 配置CMakeLists.txt

算子工程通过`enable_pypto_kernel`接入PyPTO Pro编译脚本。在算子的`op_host/CMakeLists.txt`中调用`enable_pypto_kernel(<op_file>)`，将该算子标记为PyPTO Pro Kernel。该调用需要放在`add_modules_sources`或`add_modules_sources_with_soc`之前。

`<op_file>`必须与`op_kernel/<op_file>.py`的文件名一致。例如：

```cmake
# op_host/CMakeLists.txt
enable_pypto_kernel(add_example)

add_modules_sources(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

使用`add_modules_sources_with_soc`的算子按以下方式配置：

```cmake
enable_pypto_kernel(add_example)

add_modules_sources_with_soc(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

CMake配置阶段会加载`op_kernel/add_example.py`，生成TilingData头文件、TilingKey头文件和Kernel编译所需的中间文件。后续构建过程会根据TilingKey生成Kernel实例，并将Kernel二进制与Host侧实现、aclnn接口一起打包。

## 实现Host侧Tiling

PyPTO Pro根据Kernel侧Python `@dataclass`自动生成Host侧使用的C++ Tiling类。生成类与Python类具有相同的类名、字段名、字段顺序和数据布局。

Kernel侧定义`AddExampleTilingData`后，Host侧Tiling函数可直接使用同名类型：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddExampleTilingData *tiling =
        context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->total_length = total_length;
    tiling->tile_num = tile_num;

    // GET_TPL_TILING_KEY由自动生成的TilingKey头文件提供。
    // 实参按TilingKey字段定义顺序填写候选实际值，宏负责生成64-bit打包值。
    uint64_t tiling_key = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(block_dim);

    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    size_t user_workspace_bytes = 0; // 根据Kernel实际需要设置。
    workspace_size[0] = user_workspace_bytes;
    return ge::GRAPH_SUCCESS;
}
```

构建系统会将生成的TilingData和TilingKey头文件自动提供给当前算子的Host侧Tiling源文件。`op_kernel`目录下无需额外编写`${op_name}_tiling_data.h`或`${op_name}_tiling_key.h`，Host侧也无需重复声明Tiling结构体。

`context->SetTilingKey()`接收的是打包后的64-bit TilingKey，不是某个字段未经编码的实际值。生成头文件中的`GET_TPL_TILING_KEY(...)`会按照字段定义顺序和每个候选值在`values`中的下标完成打包。例如候选值为`[16, 64, 128]`时，实际值`64`对应的字段编码是候选下标`1`，不能直接把`64`作为最终TilingKey。

Host侧Tiling实现需要保证：

- 填充的字段与Kernel侧TilingData定义一致。
- 传给`GET_TPL_TILING_KEY(...)`的字段值和顺序与Kernel侧TilingKey定义一致，并且属于合法组合。
- BlockDim与Kernel的多核切分方式一致。
- Kernel签名中的`workspace`参数位于TilingData之前。
- Workspace大小满足Kernel实际使用的用户Workspace，以及所调用接口要求的系统Workspace。示例未使用需要系统Workspace的接口，因此设置为`0`；需要系统Workspace时，应通过平台接口查询所需大小后与用户Workspace相加，不能写死固定值。

## 编译算子二进制

编译前需要配置CANN及编译工具链环境变量，并确保构建使用的Python环境能够导入与当前源码配套的`pypto_pro`。进入当前算子工程中`build.sh`所在的根目录，编译指定算子：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add_example
```

编译多个算子时，使用英文逗号分隔算子名称：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add_example,other_op
```

指定自定义算子包名称时，增加`--vendor_name`：

```bash
bash build.sh --pkg --soc=ascend950 \
    --vendor_name=${vendor_name} \
    --ops=add_example
```

编译完成后，算子安装包生成在算子工程根目录的`build_out`目录。完成上述CMake配置后，命令行不需要为PyPTO Pro Kernel增加额外构建参数。
