# Non-Pass Source Pattern Index

> 本文件是 `../patterns/non-pass-patterns.json` 的人类可读索引。修改结构化条目时必须同步更新本表。

这些条目用于跨模块定界；无历史提交时统一按待源码核查的风险假设处理。

| ID | 类别 | 可能模块 | 历史提交 | 证据状态 |
|----|------|----------|----------|----------|
| N001 | 非 Pass：View validshape 动态值默认回退 | operation | — | 源码审计假设 |
| N002 | 非 Pass：Tile shape 只校验尾维对齐 | operation | — | 源码审计假设 |
| N003 | 非 Pass：CeilAlign 对 0 除数静默返回 0 | operation | — | 源码审计假设 |
| N004 | 非 Pass：多 bool 组合用位压缩 switch 遗漏状态 | operation | — | 源码审计假设 |
| N005 | 非 Pass：未支持模式静默返回空 Tensor | operation | — | 源码审计假设 |
| N006 | 非 Pass：Workspace 预算乘法溢出/0 当无限制 | machine | — | 源码审计假设 |
| N007 | 非 Pass：Launch 三流同步错误码相加掩盖具体失败流 | machine | — | 源码审计假设 |
| N008 | 非 Pass：malloc 后未判空直接 memset | machine | — | 源码审计假设 |
| N009 | 非 Pass：RawTensor 别名 dtype/size 校验用整数除法 | machine | — | 源码审计假设 |
| N010 | 非 Pass：Interpreter 属性/operand 访问未校验范围 | interpreter | — | 源码审计假设 |
| N011 | 非 Pass：Interpreter 动态 shape/offset 求值缺失回退 | interpreter | — | 源码审计假设 |
| N012 | 非 Pass：Interpreter dtype/format switch 未识别分支 ASSERT | interpreter | — | 源码审计假设 |
| N013 | 非 Pass：IR shape/broadcast 比较过弱 | ir | — | 源码审计假设 |
| N014 | 非 Pass：IRBuilder 上下文嵌套 dynamic_cast 未判空 | ir | — | 源码审计假设 |
| N015 | 非 Pass：SSA/Token 变量映射缺陷 | ir | — | 源码审计假设 |
| N016 | 非 Pass：前端动态维度绑定字符串比较不可靠 | frontend | — | 源码审计假设 |
| N017 | 非 Pass：JIT 输入 dtype/format/shape 校验特例绕开 | frontend | — | 源码审计假设 |
| N018 | 非 Pass：嵌套函数内联识别脆弱 | frontend | — | 源码审计假设 |
| N019 | 非 Pass：Codegen TileTensor 复用键冲突 | codegen | — | 源码审计假设 |
| N020 | 非 Pass：Codegen 属性访问 opAttrs.at(key) 越界 | codegen | — | 源码审计假设 |
| N021 | 非 Pass：Cube matmul 硬编码索引与隐式转置 | codegen | — | 源码审计假设 |
| N022 | 非 Pass：Shmem signal buffer 固定按 maxTileNum=1 分配 | distributed<br>operation | — | 源码审计假设 |
| N023 | 非 Pass：SymbolicScalar rank 求值后未校验范围 | distributed<br>interpreter | — | 源码审计假设 |
| N024 | 非 Pass：集合通信未校验 worldSize 整除性 | distributed<br>operation | — | 源码审计假设 |
| N025 | 非 Pass：AICPU Wait-Until 任务哈希表容量与冲突链脆弱 | distributed<br>machine | — | 源码审计假设 |
| N026 | 非 Pass：IR/JSON 反序列化缺少空值/类型/长度校验 | operation | — | 源码审计假设 |
| N027 | 非 Pass：Call 类 OP 属性转换缺少前置校验 | operation | — | 源码审计假设 |
| N028 | 非 Pass：视图类 OP shape/offset/validshape 长度不一致 | operation | — | 源码审计假设 |
| N029 | 非 Pass：动态 shape / SymbolicScalar 求值与整数溢出 | operation | — | 源码审计假设 |
| N030 | 非 Pass：TileShapeVerifier / InferShape 空容器与越界索引 | operation | — | 源码审计假设 |
| N031 | 非 Pass：Cube matmul tile 与对齐的零除 / 硬编码常量 | operation | — | 源码审计假设 |
| N032 | 非 Pass：分布式 SHMEM 的 group/worldSize/rank 边界与静态状态 | operation<br>distributed | — | 源码审计假设 |
| N033 | 非 Pass：OpAttribute 类型转换与 memory-type / dtype 路径不一致 | operation | — | 源码审计假设 |
| N034 | 非 Pass：VIEW / 动态 shape 运行时评估缺少边界校验 | machine | — | 源码审计假设 |
| N035 | 非 Pass：控制流缓存指针与启动参数直接解引用 | machine | — | 源码审计假设 |
| N036 | 非 Pass：Cell match 表整数溢出与空表首元素访问 | machine | — | 源码审计假设 |
| N037 | 非 Pass：全局静态通信/缓存状态非线程安全 | machine | — | 源码审计假设 |
| N038 | 非 Pass：函数指针排序引入未定义行为 | machine<br>host | — | 源码审计假设 |
| N039 | 非 Pass：错误码被吞并或简单相加 | machine | — | 源码审计假设 |
| N040 | 非 Pass：同步 Event 创建后未释放 | machine | — | 源码审计假设 |
| N041 | 非 Pass：动态循环属性/设备程序裸指针空解引用 | machine<br>host | — | 源码审计假设 |
| N042 | 非 Pass：控制流源码生成中的字符串拼接未做注入校验 | machine<br>host | — | 源码审计假设 |
| N043 | 非 Pass：FunctionIODataPair 越界/漏拷 outcast | interpreter | — | 源码审计假设 |
| N044 | 非 Pass：视图类 OP size 不匹配时静默跳过 | interpreter | — | 源码审计假设 |
| N045 | 非 Pass：用字符串匹配判定 spill tensor | interpreter | — | 源码审计假设 |
| N046 | 非 Pass：EvaluateOpImmediate 越界访问 linearArgList | interpreter | — | 源码审计假设 |
| N047 | 非 Pass：固定 8 维栈缓冲区 | interpreter | — | 源码审计假设 |
| N048 | 非 Pass：PairHash 弱哈希导致字典冲突 | interpreter | — | 源码审计假设 |
| N049 | 非 Pass：mix-split 以 shared_ptr 身份做键 | interpreter | — | 源码审计假设 |
| N050 | 非 Pass：分布式模拟 Wait 永久自旋且无超时 | interpreter<br>distributed | — | 源码审计假设 |
| N051 | 非 Pass：WIN_IN_SIZE/WIN_EXP_SIZE 硬编码且累加无溢出检查 | interpreter<br>distributed | — | 源码审计假设 |
| N052 | 非 Pass：RawTensorData Numel/CalcRequiredSize int64 溢出 | interpreter | — | 源码审计假设 |
| N053 | 非 Pass：ExecuteOpAMulB 按 dtype 顺序盲搜 scale tensor | interpreter | — | 源码审计假设 |
| N054 | 非 Pass：FlowVerifier 阈值魔法公式 | interpreter | — | 源码审计假设 |
| N055 | 非 Pass：ThreadPool 忙等 | interpreter | — | 源码审计假设 |
| N056 | 非 Pass：从 schema name 直接 std::stoll 无异常处理 | interpreter | — | 源码审计假设 |
| N057 | 非 Pass：Element 除法未判除零 | interpreter | — | 源码审计假设 |
| N058 | 非 Pass：kwargs 类型校验把 Python 的 bool 当非法类型 | ir | — | 源码审计假设 |
| N059 | 非 Pass：tensor.view / tensor.assemble 不校验 rank、offset 与边界 | ir | — | 源码审计假设 |
| N060 | 非 Pass：MemRef MayAlias 整数溢出与符号混用 | ir | — | 源码审计假设 |
| N061 | 非 Pass：Codegen 属性读取未校验存在性与类型 | codegen | — | 源码审计假设 |
| N062 | 非 Pass：Codegen GetOpAttr 返回值被忽略 | codegen | — | 源码审计假设 |
| N063 | 非 Pass：动态 UB Copy 形状/偏移宏复制错误 | codegen | — | 源码审计假设 |
| N064 | 非 Pass：Codegen 硬编码维度下标直接访问 shape/offset | codegen | — | 源码审计假设 |
| N065 | 非 Pass：ForBlockManager 对固定 MAX_LOOP_DEPTH=3 的强假设 | codegen | — | 源码审计假设 |
| N066 | 非 Pass：Reshape copy valid shape 未全覆盖 | codegen | — | 源码审计假设 |
| N067 | 非 Pass：分布式 SHMEM 编码越界/截断假设 | codegen | — | 源码审计假设 |
| N068 | 非 Pass：并行任务异常被吞 + 编译命令拼接未转义 | codegen | — | 源码审计假设 |
| N069 | 非 Pass：比较算子只处理 float 标量 | frontend | — | 源码审计假设 |
| N070 | 非 Pass：整型标量通过 numpy 强制转换导致静默回绕 | frontend | — | 源码审计假设 |
| N071 | 非 Pass：Matmul / Conv 扩展参数缺省时默认 scale=0.0 | frontend | — | 源码审计假设 |
| N072 | 非 Pass：scaled_mm scale K1 维度校验逻辑错误 | frontend | — | 源码审计假设 |
| N073 | 非 Pass：tuple_get_item 对索引不做越界检查 | frontend<br>ir | — | 源码审计假设 |
| N074 | 非 Pass：arange 未校验 step 与方向 | frontend | — | 源码审计假设 |
| N075 | 非 Pass：where 对 float 标量强制使用 DT_FP32 | frontend | — | 源码审计假设 |
| N076 | 非 Pass：pass_verify_save 对非字符串 fname 调用正则 | frontend | — | 源码审计假设 |
| N077 | 非 Pass：normal 在 Python 层对 Tensor 做切片 | frontend | — | 源码审计假设 |
| N078 | 非 Pass：PIL Scope 对未定义变量直接抛 KeyError | frontend | — | 源码审计假设 |
| N079 | 非 Pass：PIL 解析器不处理 **kwargs | frontend | — | 源码审计假设 |
| N080 | 非 Pass：PIL for/if 语句对未初始化 loop-carried 变量生成 none | frontend | — | 源码审计假设 |
| N081 | 非 Pass：shmem_view 参数归一化不一致 | frontend<br>distributed | — | 源码审计假设 |
| N082 | 非 Pass：分布式 pred 默认值使用 DT_INT32 Full | frontend<br>distributed | — | 源码审计假设 |
| N083 | 非 Pass：动态循环边界假设整除且未校验零值 | operator | — | 源码审计假设 |
| N084 | 非 Pass：View / Assemble 的 offset 由未经验证的符号表达式构造 | operator | — | 源码审计假设 |
| N085 | 非 Pass：可选张量仅以 GetStorage() 判活 | operator | — | 源码审计假设 |
| N086 | 非 Pass：dtype 快照丢失 & 输入引用被就地修改 | operator | — | 源码审计假设 |
| N087 | 非 Pass：硬编码 tile / alignment 常量且断言条件写错 | operator | — | 源码审计假设 |
| N088 | 非 Pass：硬编码同步事件 ID 全局复用 | operator<br>tileop | — | 源码审计假设 |
| N089 | 非 Pass：Gather / Scatter / PageAttention 索引未校验越界 | operator<br>tileop | — | 源码审计假设 |
| N090 | 非 Pass：Adapter 失败静默 / 错误码丢失 | adapter | — | 源码审计假设 |
| N091 | 非 Pass：分布式通信地址/下标未做边界校验 | tileop<br>distributed | — | 源码审计假设 |
| N092 | 非 Pass：View 直接 reinterpret dtype 未校验字节数与对齐 | operator | — | 源码审计假设 |
