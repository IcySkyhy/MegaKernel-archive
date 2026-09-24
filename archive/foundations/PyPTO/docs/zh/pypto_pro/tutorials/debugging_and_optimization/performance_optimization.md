# 性能调优

PyPTO Pro的性能调优对象是直接运行在AI Core上的Kernel。调优时应先建立可复现的性能基线，再通过Profiling确定瓶颈，最后针对多核切分、数据搬运、片上内存、计算流水和编译期特化逐项优化。

性能优化必须以结果正确为前提。每次修改Tile Shape、流水、同步或多核分片后，都应先完成精度回归，再比较性能变化。

## 调优流程

建议按照以下顺序开展调优：

1. 固定目标Shape、数据类型、TilingKey、`block_dim`、输入数据和执行Stream。
2. 完成JIT预热，建立稳定、可重复的Kernel耗时基线。
3. 使用Profiling数据判断瓶颈属于计算、访存、流水等待、多核负载不均还是启动开销。
4. 每次只调整一类关键因素，并记录修改前后的精度、耗时和资源占用。
5. 对全部目标Shape、数据类型和TilingKey回归，防止局部优化造成其他场景退化。
6. 在算子实际调用链路中复测端到端性能。

## 建立性能基线

### 排除JIT编译开销

Kernel首次调用可能包含JIT编译、Host侧共享库生成和加载开销，不能计入Kernel执行时间。不同静态Shape、TilingKey或数据类型可能对应不同的编译实例，测量每种配置前均应单独预热。

### 正确处理异步执行

Kernel启动是异步操作。计时前后必须执行同步，否则测得的可能只是Host侧任务下发时间。以下代码给出基本测量方式：

```python
import time
import torch


warmup = 10
repeat = 100

# 输入分配、初始化和参考结果计算放在计时区间之外
for _ in range(warmup):
    kernel[None, block_dim](*args)
torch.npu.synchronize()

torch.npu.synchronize()
start = time.perf_counter()
for _ in range(repeat):
    kernel[None, block_dim](*args)
torch.npu.synchronize()
elapsed = time.perf_counter() - start

avg_call_us = elapsed * 1e6 / repeat
print(f"average JIT call time: {avg_call_us:.3f} us")
```

为了获得可比较的结果，应满足以下条件：

- 输入分配、随机数生成、Host到Device拷贝和结果校验不放入Kernel计时区间。
- 测试期间保持Shape、数据类型、TilingKey、`block_dim`和Stream不变。
- 预热次数应足以完成编译并使运行状态稳定。
- 重复多轮测量并报告中位数或稳定区间，不以单次结果判断优化效果。
- 调优版本和基线版本使用相同的同步位置及测量方法。
- 测量前移除`pl.printf`、`pl.dump_data`、`pl.pto_assert`和`pl.trap`等调试代码。

上述方式测量的是连续下发场景下JIT调用链路的稳态平均耗时，包含Host侧逐次下发开销，不等同于Profiling结果中的纯Device Kernel执行时间。需要测量单次同步调用延迟时，应在每次调用后同步，并将同步方式和固定开销纳入基线。算子通过二进制包和aclnn接口交付时，还应在实际aclnn调用路径中测量端到端耗时，以覆盖Host侧Tiling、参数校验、Kernel选择和任务下发开销。

## 使用Profiling定位瓶颈

完成正确性验证和JIT预热后，可以使用`torch_npu.profiler`采集Device侧Kernel耗时、AI Core流水指标和任务时间线。Profiling用于解释性能瓶颈，不能代替上一节的稳态基线测量；采集本身以及额外同步可能改变Host下发节奏。

以下示例假设待测脚本中已经定义`kernel`、`block_dim`和`args`。示例在每次调用后同步，目的是隔离单次Kernel，并使每个`prof.step()`对应一次已经完成的调用：

```python
import torch
import torch_npu


profiler_output = "./profiling_output"
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    experimental_config=experimental_config,
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=1, active=5, repeat=1, skip_first=5
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        profiler_output, analyse_flag=True
    ),
) as prof:
    for _ in range(11):
        kernel[None, block_dim](*args)
        torch.npu.synchronize()
        prof.step()
```

本例依次执行5个`skip_first` step、1个Profiler warmup step和5个active step，因此循环调用次数为11。JIT预热仍应在进入上述`profile`上下文之前完成；`skip_first`和Profiler warmup有各自的采集阶段语义，不应将它们视为已经完成JIT预热的保证。每次Kernel调用后必须调用一次`prof.step()`，否则schedule无法按预期推进。

上述写法用于分析单次Kernel。逐次调用`torch.npu.synchronize()`会阻断Kernel之间的异步下发和重叠，对应时间线不反映实际业务链路中的任务间隔或跨Kernel并行。真实调用链分析应保留业务原有的同步位置，并单独建立端到端基线。

采集结束后，在`profiling_output`目录中递归查找以下文件。不同CANN和`torch_npu`配套版本的目录层级及部分文件名可能不同，应以当前安装版本的Profiler说明和实际产物为准。

| 文件 | 检查内容 |
|---|---|
| `kernel_details.csv` | 在`Name`或`Type`列中定位目标Kernel，检查`Duration(us)`、`Block Dim`等字段。配置`aic_metrics=PipeUtilization`后，该文件会增加当前平台支持的流水指标字段，可用于比较计算和搬运流水的耗时及占比。 |
| `trace_view.json`或`trace_result.json` | 使用当前版本支持的性能分析工具打开，检查应用层、CANN层和NPU任务的时间关系、Stream、Kernel启动间隔及Host与Device之间的空闲。 |

`trace_view.json`展示的主要是应用、CANN和NPU任务级时间线，不等同于Kernel内部的指令级流水图。`PipeUtilization`给出各类流水的累计耗时或占比，也不能单独证明MTE、Vector和Cube指令在时间轴上的具体重叠关系。需要定位Kernel内部的断流、同步等待或指令级重叠时，应进一步使用当前平台支持的核内流水分析或仿真工具。

如果`kernel_details.csv`中没有目标Kernel，依次检查采集循环是否实际执行、`prof.step()`调用次数是否覆盖`skip_first + wait + warmup + active`对应的阶段、`activities`是否包含`ProfilerActivity.NPU`，以及输出目录是否可写。比较不同实现时，应保持输入Shape、数据类型、TilingKey、`block_dim`、Stream、预热方式和采集配置一致。

使用`PipeUtilization`指标时，可按以下顺序缩小瓶颈范围：

1. 根据`kernel_details.csv`定位目标Kernel，检查多个采集样本的`Duration(us)`；最终性能结论仍以独立的多轮稳态基线为准。
2. 比较Cube、Vector和数据搬运流水的耗时及占比，定位关键路径上的最长流水。流水占比表示对应流水的时间覆盖范围；硬件计算效率和带宽利用率需结合指令吞吐、带宽及阻塞指标分析。
3. 根据数据搬运量、计算量以及目标硬件规格估算理论下界，再将实测流水耗时与理论值比较，区分计算量或搬运量本身较大和硬件利用效率不足两类情况。
4. 结合任务级时间线检查Host下发、Stream依赖、Kernel间隔和Device空闲；需要判断Kernel内部流水先后关系时，继续使用核内流水分析工具。
5. 根据证据选择多核切分、Tile Shape、片上复用或多缓冲等单一优化方向。修改后先完成精度回归，再使用相同基线和Profiling配置复测。

`_ExperimentalConfig`是`torch_npu.profiler`中的实验性接口，枚举、字段和产物可能随配套版本变化。使用前应运行最小采集用例，确认当前环境支持相关配置并能正常生成产物。进行性能分析时，应保留Profiler warmup并采集多个active step。Profiling结果主要反映Device侧Kernel及相关调用链情况，评估完整算子调用时，还应单独记录Host侧Tiling、参数校验、Kernel选择和任务下发等端到端耗时。

## 判断性能瓶颈

| Profiling现象 | 可能原因 | 优化方向 |
|---|---|---|
| 计算流水利用率高且持续繁忙 | Kernel接近计算受限 | 提高Cube或Vector指令效率，减少重复计算，选择合适的数据类型和计算布局。 |
| 数据搬运耗时高、计算流水空闲 | GM或片上搬运受限 | 提高连续访问比例，增大有效搬运粒度，复用片上数据，减少重复加载和写回。 |
| 搬运和计算先后串行 | 流水重叠不足 | 使用TileGroup和多缓冲，使下一块数据搬运与当前块计算重叠。 |
| 流水存在较长等待 | 同步过多或依赖关系不合理 | 检查mutex、Tile复用和手动同步，删除不必要的屏障，同时保证真实依赖不被破坏。 |
| 不同Core耗时差异明显 | 多核负载不均或尾块集中 | 调整分片方式，将Tile跨步或均匀分配给各Core。 |
| Core数量增加但性能无提升 | 单Core工作量过小、核数超过可并行任务数或访存带宽饱和 | 根据Tile总数和硬件资源重新选择`block_dim`。 |
| 小Shape中Kernel时间占比低 | Host下发或固定启动开销占主导 | 减少不必要的多次Kernel启动，在实际调用链路评估端到端收益。 |

## 多核切分优化

PyPTO Pro通过[`pl.get_block_idx()`](../../api/SIMD-API/operation/system_variables/get_block_idx.md)和[`pl.get_block_num()`](../../api/SIMD-API/operation/system_variables/get_block_num.md)进行多核分片。合理的多核策略应同时满足完整覆盖、无重复写和负载均衡。

- `block_dim`不应超过可独立执行的任务块数量，否则会产生空闲Core。
- 每个Core的工作量应尽量接近，避免将全部尾块或耗时较高的分支集中到少数Core。
- 规则二维Tile可先线性编号，再按Core编号进行跨步分配，以减小尾部负载差异。
- 输出区域应由唯一Core写入；需要跨Core归约时，应使用明确且受支持的同步与归约方案。
- 对小Shape和大Shape分别测试核数，单一`block_dim`不一定适合全部Shape。

调试阶段使用`block_dim=1`有助于验证逻辑，但性能测试必须恢复目标核数。

## Tile与片上内存优化

Tile Shape决定一次计算的数据量、片上内存占用、循环次数、尾块比例和搬运效率。选择Tile时需要综合考虑：

- Vec/UB、Mat/L1、Left/L0A、Right/L0B和Acc/L0C的容量限制。
- 数据类型和Tile Shape共同决定的实际字节数。
- 数据搬运和计算指令的对齐要求。
- 尾块比例以及有效数据之外的补齐开销。
- 同一份输入数据在片上的复用次数。
- 多缓冲后总内存占用的倍增。

增大Tile可以减少循环和指令下发次数，但也会增加片上内存占用，可能降低双缓冲可行性或减少并行度。缩小Tile可以降低单块资源占用，但会增加循环、搬运和尾块处理开销。应通过实测选择目标场景的平衡点。

GM访问应尽量连续、对齐并合并为较大的有效搬运。对于会被多次使用的数据，应在容量允许时保留在片上，避免在每个计算步骤中重复从GM加载。中间结果能够在片上直接消费时，应避免不必要的GM写回和重新加载。

## 流水与多缓冲优化

[`pl.make_tile_group`](../../api/SIMD-API/operation/resource_management/make_tile_group.md)可为同一逻辑数据声明多块轮转Tile，通过`current()`取得当前Tile、通过`next()`轮转到下一块。配合`@pl.jit(auto_mutex=True)`，编译器根据Tile的mutex信息插入核内流水同步，可用于构建双缓冲或N缓冲。

优化时应重点检查：

- 加载下一块数据能否与当前块的Vector或Cube计算重叠。
- 当前结果写回能否与后续计算重叠。
- 每个轮转Tile是否使用独立且不冲突的mutex编号。
- Tile调用`next()`的次数和位置是否与生产、消费顺序一致。
- 是否存在不必要或未正确配对的手动同步，以及自动mutex和手动同步是否重复保护同一依赖。
- 流水末尾是否正确排空，最后一个结果是否完成写回。

缓冲数量越多，片上内存占用越大，管理开销也可能增加。双缓冲不能掩盖全部延迟时才考虑更多缓冲，并通过Profiling验证收益。

## Cube与Vector协同优化

矩阵类Kernel使用[`pl.section_cube()`](../../api/SIMD-API/operation/controlflow/section_vector_section_cube.md)描述Cube任务，向量类处理使用`pl.section_vector()`描述Vector任务。混合Kernel应尽量让Cube计算、Vector前后处理和DMA搬运并行，同时避免不必要的数据格式转换和跨存储层往返。

- Cube计算应检查M、N、K方向的Tile Shape、左右矩阵布局、转置方式以及L0A/L0B装载格式。
- 归约长度较大时，应在Acc/L0C中完成分块累加，再按需要转换并写回。
- Vector前后处理应尽量与Cube流水重叠，避免形成全局串行阶段。
- 多个连续的细粒度Vector操作产生明显额外开销时，可在确认瓶颈后使用Vector Function表达寄存器级计算，减少不必要的中间Tile读写。
- 调整Cube与Vector协同时必须重新检查mutex和真实数据依赖，不能通过删除同步换取错误的表面性能。

## TilingKey与编译期特化

TilingKey适合表达数量有限、执行路径差异明显且能够带来性能收益的编译期条件。例如，对齐路径与非对齐路径、不同算法模式或有限的数据布局可以分别生成专用Kernel，从而消除热循环中的无效分支。

不应将取值范围很大的运行时Shape直接展开为TilingKey。过多Key会增加二进制数量、编译时间和缓存占用，也会提高测试与交付复杂度。只有当特化收益经过测量且Key集合可控时，才应新增TilingKey。

常用Shape可使用规整、无分支的主路径，尾块和少见场景使用独立分支处理。热循环中应尽量减少取决于运行时数据的分支，但不能省略必要的边界检查。

## 结果验证与交付检查

完成调优后，应执行以下检查：

- 全部目标Shape、数据类型、布局、TilingKey和`block_dim`均通过精度测试。
- 覆盖最小Shape、非对齐Shape、尾块、空闲Core和最大资源占用等边界场景。
- 在相同测量条件下比较基线与优化版本，并保存多轮结果。
- Profiling数据能够解释性能变化，不以偶然波动作为优化结论。
- 已移除调试接口和仅用于定位问题的额外同步。
- JIT直接调用和实际aclnn调用路径均完成性能验证。
- 二进制大小、TilingKey数量和首次编译时间仍处于可接受范围。
