# F4XXXX-F5XXXX

## F40000 TENSOR_NULL_POINTER

**错误描述**

Tensor或其关联的操作存在空指针引用。

**可能原因**

- Tensor的producer为null。
- Tensor的consumer为null。
- Operation的input tensor为null。
- Operation的output tensor为null。
- Tensor的消费者中存在null consumer。
- Tensor的生产者中存在null producer。

**处理方式**

1. 检查报错Tensor的producer和consumer是否正确创建。
2. 检查Operation的输入、输出Tensor是否为空。
3. 检查图构造过程中是否遗漏Tensor与Operation的连接关系。

## F40001 TENSOR_INVALID_MEMORY_TYPE

**错误描述**

Tensor的内存类型配置不合法或不匹配。

**可能原因**

- Tensor的内存类型为无效值或未定义值。
- Tensor的内存类型与所在子图或计算单元要求不兼容。
- 动态形状Tensor使用了不合法的内存类型配置。
- 边界Tensor未使用规定的内存类型。

**处理方式**

1. 检查Tensor的memory type是否为框架支持的合法值。
2. 检查Tensor所在子图、计算单元与memory type是否匹配。
3. 如果Tensor跨子图使用，检查边界Tensor的memory type配置是否符合要求。

## F40002 TENSOR_SUBGRAPH_BOUNDARY

**错误描述**

跨子图使用的Tensor未正确标记边界。

**可能原因**

- DDR tensor未标记为subgraph boundary。
- 跨子图的Tensor未标记为subgraph boundary。
- Tensor的subgraph id为`NOT_IN_SUBGRAPH`。

**处理方式**

1. 检查跨子图Tensor是否设置了subgraph boundary标记。
2. 检查DDR边界Tensor的boundary属性是否完整。
3. 检查Tensor的subgraph id是否已正确归属到有效子图。

## F40003 TENSOR_SHAPE_MISMATCH

**错误描述**

Tensor的shape或memory type不满足Operation输入输出约束。

**可能原因**

- Operation的输入输出Tensor shape不符合约束。
- Operation的输入输出Tensor memory type不符合约束。

**处理方式**

1. 根据报错Operation检查输入、输出Tensor shape是否满足该Operation的语义要求。
2. 检查reshape、view、assemble等视图类操作的输入输出维度是否一致或可推导。
3. 检查Tensor的memory type是否与Operation支持的输入输出路径匹配。

## F40004 TENSOR_UNSUPPORTED_DATATYPE

**错误描述**

Tensor的数据类型不被Operation支持。

**可能原因**

- Operation不支持输入Tensor的dtype。
- Operation不支持输出Tensor的dtype。
- 输入输出Tensor dtype组合不符合Operation约束。

**处理方式**

1. 检查报错Operation支持的数据类型范围。
2. 将输入或输出Tensor dtype调整为Operation支持的类型。
3. 如需要类型转换，请在进入该Operation前显式插入合法的cast或转换逻辑。

## F40005 TENSOR_MEMORY_ALLOCATION

**错误描述**

Tensor的内存大小、地址范围或对齐配置不满足内存分配约束。

**可能原因**

- 同一内存区域被多个Tensor非法重叠占用。
- 内存段划分不合理导致地址越界。
- Tensor内存大小为0或超出合法分配范围。
- 动态内存分配属性缺失或配置非法。
- Tensor内存对齐方式不符合硬件约束。

**处理方式**

1. 检查Tensor的内存地址、offset、size是否存在越界或非法重叠。
2. 检查动态内存相关属性是否完整。
3. 检查Tensor内存大小和对齐方式是否满足硬件及框架约束。

## F40006 TENSOR_DYNAMIC_ATTR

**错误描述**

Operation动态属性或Tensor的dynValidShape缺失、配置错误。

**可能原因**

- Operation的动态相关属性缺失。
- Tensor的dynValidShape为空。

**处理方式**

1. 检查动态shape场景下Tensor的dynValidShape是否已正确设置。
2. 检查相关Operation是否具备动态属性推导所需的输入信息。
3. 如果pass修改了视图类Operation或Tensor shape，检查dynValidShape是否同步更新。

## F41000 OP_INVALID_OPERAND_COUNT

**错误描述**

Operation的输入、输出或边带输入数量不符合该Operation约束。

**可能原因**

- Operation的实际输入Tensor数不合规。
- Operation的实际输出Tensor数不合规。
- 控制依赖或边带输入数量不符合约束。

**处理方式**

1. 根据Operation语义检查输入Tensor数量是否正确。
2. 根据Operation语义检查输出Tensor数量是否正确。
3. 如Operation使用控制依赖或边带输入，检查对应输入数量和顺序是否符合约束。

## F41001 OP_NULL_POINTER

**错误描述**

Operation、Operation属性或输入输出Tensor列表存在空指针引用。

**可能原因**

- Operation为null。
- Operation的op attribute为null。
- Operation的IOperands或OOperands为null。

**处理方式**

1. 检查Operation是否成功创建并加入Function。
2. 检查需要属性的Operation是否设置了op attribute。
3. 检查Operation的输入输出Tensor列表是否为空指针。

## F41002 OP_INVALID_OPCODE

**错误描述**

Operation的opcode不属于当前Function、子图或pass阶段支持的合法类型。

**可能原因**

- Operation类型不合规。
- 当前pass或当前图结构不支持该opcode。

**处理方式**

1. 检查报错Operation的opcode是否为框架支持的合法类型。
2. 检查该opcode是否允许出现在当前Function、子图或pass阶段。
3. 如为新增Operation，检查对应pass是否已补齐支持逻辑。

## F41003 OP_PRODUCER_CONSUMER

**错误描述**

Operation缺少合法生产者、消费者或Tensor双向连接关系不一致。

**可能原因**

- Operation没有生产者。
- Operation没有消费者。
- Operation与输入输出Tensor的producer或consumer关系不一致。

**处理方式**

1. 检查Operation的输入Tensor是否有合法producer。
2. 检查Operation的输出Tensor是否有合法consumer。
3. 检查Tensor双向连接关系是否一致，即Operation引用了Tensor，Tensor也记录了对应producer或consumer。

## F41004 OP_SPECIAL_CONSTRAINT

**错误描述**

Operation的连接关系或目标memType不满足当前Operation的特殊约束。

**可能原因**

- Operation的生产者或消费者Operation类型不合规。
- Operation的to memType类型不合规。

**处理方式**

1. 根据报错Operation检查其生产者和消费者类型是否满足特殊约束。
2. 检查该Operation的目标memory type是否属于允许范围。
3. 对reshape、view、assemble、copy等特殊Operation，检查前后连接是否符合pass约束。

## F41005 OP_NESTING_DEPTH

**错误描述**

Operation嵌套层级超过框架限制。

**可能原因**

- Operation嵌套深度超过框架限制。

**处理方式**

1. 检查报错Operation所在嵌套结构是否过深。
2. 简化嵌套层级，或拆分为多个合法的中间步骤。
3. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## F41006 OP_SEQUENCE_ERROR

**错误描述**

Operation序列包含当前pass或后端执行不支持的操作组合。

**可能原因**

- 存在不允许的Operation。
- 存在不允许的Operation组合。
- Operation排列顺序不符合当前pass或后端执行约束。

**处理方式**

1. 根据报错位置检查相邻Operation的组合是否合法。
2. 调整Operation顺序或拆分不支持的组合。
3. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## F42000 FUNCTION_GRAPH_STRUCTURE

**错误描述**

Function的incast、outcast、Operation或子图拓扑结构不完整或不合法。

**可能原因**

- Function中存在null operation。
- Function的incast为空。
- Function的outcast为空。
- Function中存在循环依赖。
- 子图拓扑结构不正确。
- 子图ID超出范围。
- 存在空子图。

**处理方式**

1. 检查Function是否包含合法incast和outcast。
2. 检查Function内Operation是否存在空指针。
3. 检查Function是否存在循环依赖或非法子图结构。
4. 检查子图ID是否在合法范围内，并确认不存在空子图。

## F42001 FUNCTION_BOUNDARY_COMPLETENESS

**错误描述**

Function的incast、outcast或Operation子图归属不完整。

**可能原因**

- Incast没有consumer。
- Outcast没有producer。
- Operation的subgraph ID为负数且不是NOP操作。

**处理方式**

1. 检查所有incast是否至少连接到一个consumer。
2. 检查所有outcast是否存在合法producer。
3. 检查Operation的subgraph ID是否有效；NOP以外的Operation不应使用非法负数subgraph ID。

## F42002 FUNCTION_GRAPH_CONNECTION

**错误描述**

Function的输入输出图、子图边界或图边索引关系不一致。

**可能原因**

- 输入输出图不匹配。
- 子图边界Tensor未正确标记。
- 边索引超出Operation列表范围。
- 操作的magic number找不到。

**处理方式**

1. 检查Function输入输出图是否一致。
2. 检查跨子图Tensor是否正确设置边界标记。
3. 检查图边索引是否越界。
4. 检查Operation magic number是否唯一且可在Function中找到。

## F42003 FUNCTION_EXPAND_FEATURE

**错误描述**

Function展开状态或展开后的临时Tensor连接关系不正确。

**可能原因**

- ExpandFunctionAccelerate标志未重置为false。
- 局部定义的临时Tensor用作操作输入，但没有producer。

**处理方式**

1. 检查Function expand相关标志是否在展开后恢复到预期状态。
2. 检查局部临时Tensor是否被直接用作Operation输入。
3. 如临时Tensor需要参与计算，请确保其由合法Operation生产。

## F42004 FUNCTION_MEMORY_REACHABILITY

**错误描述**

Function中的内存类型转换不可达。

**可能原因**

- Operation的输入输出memory type不可达。
- 输入输出memory type转换路径不存在。

**处理方式**

1. 检查Operation输入输出memory type是否存在合法转换路径。
2. 检查配置中是否声明了对应memory type的可达关系。
3. 如缺少转换路径，请调整Tensor memory type或补齐合法的转换Operation。

## F42005 FUNCTION_UNIQUENESS

**错误描述**

Function中Operation或Tensor的magic number重复。

**可能原因**

- Operation的magic number重复。
- Tensor的magic number重复。

**处理方式**

1. 检查Function内Operation magic number是否唯一。
2. 检查Function内Tensor magic number是否唯一。
3. 如重复标识由pass新增节点引入，请检查新增Operation或Tensor的创建方式。

## F42006 FUNCTION_SPECIAL_STRUCTURE

**错误描述**

Function中存在不符合拓扑、子图嵌套或特殊算子组合约束的结构。

**可能原因**

- 存在不符合拓扑规范的特殊节点连接方式。
- 子图嵌套结构不符合框架约束。
- Function内存在不允许的特殊算子组合结构。

**处理方式**

1. 检查特殊节点的生产者、消费者和拓扑顺序。
2. 检查子图嵌套结构是否符合框架约束。
3. 拆分或调整不支持的特殊算子组合。

## F43000 GRAPH_LOOP_DETECTION

**错误描述**

Graph或Function中存在无法拓扑排序的循环依赖。

**可能原因**

- OperationLoopCheck失败，存在循环依赖。
- LoopCheck失败，存在循环。

**处理方式**

1. 检查报错Function或Graph中是否存在环状依赖。
2. 根据报错日志定位环上的Operation。
3. 修改计算逻辑或连接关系，消除数据循环。

## F43001 GRAPH_TOPOLOGY_STRUCTURE

**错误描述**

图的子图拓扑、父子图ID关系或边索引不符合拓扑约束。

**可能原因**

- 子图拓扑结构不正确。
- 父子图ID关系不正确，parent subGraphId应小于等于subGraphId。
- 边索引超出Operation列表范围。

**处理方式**

1. 检查子图拓扑顺序是否满足producer在consumer之前的要求。
2. 检查父子图ID关系是否符合约束。
3. 检查图边索引是否在Operation列表有效范围内。

## F43002 GRAPH_SUBGRAPH_EMPTY

**错误描述**

图划分结果中存在空子图。

**可能原因**

- 子图为空。
- 图划分或pass删除节点后留下空子图。

**处理方式**

1. 检查子图划分结果是否存在空子图。
2. 检查删除或消除类pass是否同步清理子图信息。
3. 调整图结构或pass逻辑，避免生成空子图。

## F43003 GRAPH_SUBGRAPH_ID_INVALID

**错误描述**

Operation的subgraph ID为负数或超出子图数量范围。

**可能原因**

- 子图ID为负数且不是NOP操作。
- 子图ID超出totalSubGraphNum范围。

**处理方式**

1. 检查Operation的subgraph ID是否在合法范围内。
2. 检查totalSubGraphNum是否与实际子图数量一致。
3. 对NOP以外的Operation，避免使用非法负数subgraph ID。

## F43004 GRAPH_EDGE_CONSISTENCY

**错误描述**

图的入边、出边或节点索引关系不一致。

**可能原因**

- 入边图和出边图大小不匹配。
- 节点在输入图中的位置超出输出图范围。
- 节点存在于输入图但未在输出图中找到。
- 出边图中存在未被遍历的边。

**处理方式**

1. 检查入边图与出边图是否同步更新。
2. 检查节点索引是否在图结构有效范围内。
3. 检查新增、删除或替换Operation后是否同步维护边关系。

## F43005 GRAPH_COLOR_CONSISTENCY

**错误描述**

图着色输入输出、子图映射或着色边关系不一致。

**可能原因**

- 输入着色图和输出着色图一致性检查失败。
- 输出着色图和输入匹配失败。
- 原始Operation和子图Operation之间的边在输出着色图中缺失。
- 输出着色图中的边在输出图中没有对应边。

**处理方式**

1. 检查图着色输入输出信息是否一致。
2. 检查原始操作与子图操作之间的映射关系是否完整。
3. 检查输出着色图中的边是否都能在输出图中找到。

## F43006 GRAPH_READY_STATE

**错误描述**

图拓扑遍历中的readyState与前驱计数不一致。

**可能原因**

- 拓扑结构中就绪状态不一致。
- readyState与负的前驱计数不匹配。

**处理方式**

1. 检查拓扑遍历过程中节点readyState是否正确更新。
2. 检查节点前驱计数是否与实际输入边数量一致。
3. 检查图构造或修改后是否重新计算就绪状态。

## F43007 GRAPH_AIV_AIC_MIX

**错误描述**

同一子图中混合了不兼容的计算单元或内存类型。

**可能原因**

- 子图中同时存在AIV和AIC操作。
- 子图中同时存在UB和L0/L1内存类型Tensor。

**处理方式**

1. 检查同一子图内是否混合AIV和AIC操作。
2. 检查同一子图内是否混合UB与L0/L1 memory type Tensor。
3. 根据计算单元和memory type约束重新划分子图。

## F44000 CONFIG_MEMORY_TYPE_REACHABLE

**错误描述**

配置中缺少输入输出memory type之间的可达转换路径。

**可能原因**

- 输入输出内存类型不可达。
- 内存类型转换路径不存在。

**处理方式**

1. 检查配置中是否存在输入输出memory type的可达路径。
2. 检查Tensor memory type是否配置错误。
3. 调整memory type配置或插入合法的内存类型转换路径。

## F44001 CONFIG_SUBGRAPH_BOUNDARY

**错误描述**

跨子图的Tensor边界标记缺失。

**可能原因**

- DDR tensor未标记为子图边界。
- 跨子图的Tensor未标记为子图边界。

**处理方式**

1. 检查DDR Tensor是否正确设置子图边界标记。
2. 检查跨子图Tensor是否设置boundary属性。
3. 检查子图划分后边界Tensor信息是否同步更新。

## F44002 CONFIG_TENSOR_MEMORY_TYPE

**错误描述**

配置中的Tensor memory type与子图或Operation输入输出要求不匹配。

**可能原因**

- 内存类型不匹配。

**处理方式**

1. 检查Tensor memory type是否为框架支持的合法值。
2. 检查Tensor memory type与所在子图、Operation输入输出要求是否匹配。
3. 根据报错上下文调整Tensor memory type或配置项。

## F44003 CONFIG_FILE_FAILED

**错误描述**

配置文件读取、解析或加载失败。

**可能原因**

- 配置文件打开失败。
- 配置文件读取失败。
- 配置文件中不存在对应的配置项。
- 配置文件对应tab下不存在指定的配置key。
- 配置项读取失败，不存在指定的配置信息。

**处理方式**

1. 检查配置文件路径是否正确，文件是否存在且可读。
2. 检查配置文件内容格式是否正确。
3. 检查报错配置项、tab和key是否存在。
4. 如配置文件由工具生成，请重新生成配置文件后重试。
