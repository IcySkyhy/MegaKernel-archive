# PyPTO Pass 异常分析流程指导

本文件提供 PyPTO Pass 模块常见异常类型的分析流程，指导AI Agent 针对不同类型的异常，采用不同策略进行分析。

---

## 通用分析总流程

### 步骤 1：定位日志

1. 从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录获取最新 `pypto-*.log`。
2. 先找 `[ERROR]`，再找 `[WARN]`。

```bash
ls "$ASCEND_PROCESS_LOG_PATH/debug/plog"
```

```bash
grep -n "\[ERROR\]\|\[WARN\]" "$ASCEND_PROCESS_LOG_PATH"/debug/plog/pypto-*.log
```

### 步骤 2：提取关键信息

从日志中提取以下信息：

- `op_magic` 或 `tensor_magic`
- Pass 名称
- Before / After 阶段
- `subgraph_id`
- 报错文本
- 文件路径和行号

### 步骤 3：选择分析入口

1. 如果日志里有 `op_magic` 或 `tensor_magic`，优先走计算图 JSON。
2. 如果拿到 `.tifwkgr`，走 IR 分析。
3. 如果两者都有，先用 JSON 定位，再用 IR 验证变化。

### 步骤 4：定位计算图节点

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --tensor-magic <tensor_magic>
```

建议先在 `output/` 下找最近的计算图文件：

```bash
ls -lt output/*/Pass_*/*.json
```

如果需要确认具体文件名，可先找对应 pass 目录：

```bash
ls -lt output/*/Pass_* | head
```

### 步骤 5：定位源码

1. 取 `OperationInfo.file` 和 `OperationInfo.line`。
2. 打开源码上下文，建议看前后各 20 行。

```bash
sed -n '<line-20>,<line+20>p' <source_file>
```

如果已经通过脚本输出了 `file` 和 `line`，可直接回看用户代码位置。

### 步骤 6：查官方文档

1. 算子类问题，先查 `docs/zh/api/operation/`。
2. 参数类问题，先查 `docs/zh/api/config/pypto-set_pass_options.md`。
3. 计算图解析类问题，直接使用本文件的“通用分析总流程”和 `scripts/computation_graph_analyzer.py`。

```bash
grep -R "<opcode>" docs/zh/api/operation/
```

```bash
grep -n "<param_name>" docs/zh/api/config/pypto-set_pass_options.md
```

### 步骤 7：对比 Before / After

1. 对比同一节点的 `shape`、`validshape`、`mem_type`、`ioperands`、`ooperands`。
2. 确认 Pass 是否产生了非预期变更。

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <after_json_path> \
  --op-magic <op_magic>
```

### 步骤 8：输出根因链

按以下格式输出：

`现象(日志) -> 触发位置(源码行) -> 状态异常(计算图 Before/After) -> 违反规则(文档约束)`

---

## 公共参考：`After` 计算图缺失时补打异常前计算图

当出现以下任一情况时，应先参考本节补打异常前最近一份计算图，再进入对应异常类型的专项分析：

- 日志显示 pass 在 `RunOnFunction` / 核心处理逻辑中抛异常或提前返回
- pass 输出目录里只有 `Before` 图，没有对应 `After` 图
- 需要分析的是“执行中断前最后一版图”，而默认 dump 机制拿不到

适用范围：

- 计算图成环
- 拓扑排序失败
- 属性检查失败
- Pass 中途修改依赖、替换 tensor、重建 op 后立即触发异常
- 其他所有“异常发生在 pass 内部，导致 after 阶段文件未生成”的场景

### 步骤 1：确认默认 dump 在哪里中断

1. 先根据日志判断异常发生在 pass 的哪个函数、哪个代码分支。
2. 确认当前 `Before` 文件已生成但 `After` 文件缺失。
3. 不要把“缺失 After 文件”直接当作 pass 未修改图；它更常见的含义是“图已被部分修改，但尚未来得及统一落盘”。

仓内可核实依据：

- `framework/src/passes/pass_interface/pass.cpp`
- `Pass::DumpFunctionJson(...)`
- `framework/src/interface/function/function.cpp`
- `Function::DumpJsonFile(...)`

可以据此判断：默认 `After` 计算图通常在 pass 正常执行到统一 dump 流程后才会落盘；一旦 pass 中途异常，往往需要人工在异常前最近位置追加临时 dump。

### 步骤 2：选择最近的插桩点

插桩位置必须满足“尽量靠近异常点，同时保证 dump 调用本身有机会执行”。优先级如下：

1. 异常日志对应代码行之前最近一次图结构修改完成处
2. 即将执行拓扑检查、成环检查、属性校验、依赖校验的调用前
3. `return FAILED`、`CHECK`、`throw`、`PYPTO_ASSERT`、`GELOGE(...); return ...;` 之前
4. 遍历中可疑分支内部，在关键修改后立即 dump

禁止做法：

- 只在 pass 入口处 dump 一份 `Before` 图，然后声称已经拿到“异常前图”
- 为了避免插桩而跳过问题分支、注释掉异常逻辑或简化代码路径
- 未重新编译执行就声称“已确认中断前图状态”

### 步骤 3：插入临时 `DumpJsonFile` 代码

优先对当前正在被 pass 修改的 `Function` 对象调用：

```cpp
function.DumpJsonFile("/tmp/pass_debug/<pass_name>_before_abort.json");
```

若当前上下文持有的是 `Function*` / `std::shared_ptr<Function>`，使用等价写法：

```cpp
currFunctionPtr->DumpJsonFile("./config/pass/json/<pass_name>_before_abort.json");
```

文件路径要求：

- 使用明确、不覆盖原始 dump 的文件名
- 建议包含 `pass 名称 + 阶段 + before_abort / pre_check / pre_return` 等语义
- 路径父目录必须真实存在，否则 `Function::DumpJsonFile` 打开文件会失败

命名示例：

- `./config/pass/json/cycle_detect_before_abort.json`
- `./config/pass/json/pass_x_pre_topology_check.json`
- `./config/pass/json/pass_x_before_return_failed.json`

插桩示例 1：在校验前补打

```cpp
// Dump the latest graph state before topology validation aborts execution.
function.DumpJsonFile("./config/pass/json/pass_x_pre_topology_check.json");
auto ret = OperationLoopCheck(function);
if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "loop check failed");
    return ret;
}
```

插桩示例 2：在错误返回前补打

```cpp
if (!IsEdgeValid(producer, consumer)) {
    function.DumpJsonFile("./config/pass/json/pass_x_before_return_failed.json");
    GELOGE(INTERNAL_ERROR, "invalid producer-consumer edge");
    return INTERNAL_ERROR;
}
```

### 步骤 4：重新编译并复现

1. 重新编译受影响模块或按用户原命令重新构建。
2. 在同一会话中保留日志环境变量，重新执行复现命令。
3. 确认临时 dump 文件实际生成。
4. 若仍未生成，继续把插桩点向异常前收缩，直到拿到“中断前最后一版图”。

本步骤必须输出以下事实：

- 是否已重新编译
- 是否已重新执行原复现命令
- 临时 dump 文件路径
- 临时 dump 文件是否实际生成

### 步骤 5：基于补打图继续分析

获得补打图后，继续按原异常类型流程分析：

1. 若是成环 / 拓扑失败：
   - 对补打图执行 `--detect-op-cycle` / `--detect-subgraph-cycle`
   - 与当前 pass 的 `Before` 图、上游 pass 输出图对比
2. 若是属性、shape、依赖异常：
   - 将补打图视为“异常前最新状态图”
   - 对比异常前后关键 op、tensor、subgraph 的变化
3. 在最终结论中明确标注该图来源于“临时异常前 dump”，不要误写成框架自动导出的标准 `After` 图

### 步骤 6：调试结束后清理临时插桩

若本轮任务包含代码修复或准备提交变更：

1. 在确认根因后移除临时 `DumpJsonFile` 调试代码
2. 除非用户明确要求保留诊断代码，否则不要把临时 dump 插桩作为正式修复提交
3. 最终报告中保留：插桩位置、dump 文件名、分析结论；不要保留无必要的临时调试改动

---

## 一、参数配置异常

### 日志特征

- `vec_nbuffer_setting`
- `cube_l1_reuse_setting`
- `cube_nbuffer_setting`
- `sg_set_scope`
- 包含 pass 配置参数名关键字

### 分析流程

#### 步骤 1：获取参数约束

```bash
grep -n "<param_name>" docs/zh/api/config/pypto-set_pass_options.md
```

#### 步骤 2：检查用户参数配置

1. 定位用户代码中的参数设置位置。
2. 对照文档确认参数范围、类型和默认值。

#### 步骤 3：检查 pass 逻辑

1. 回看日志里的源码位置。
2. 检查 pass 是否正确处理了该参数。

#### 步骤 4：确定根因

1. 如果是用户配置不满足约束，明确指出参数值和不满足的规则。
2. 如果是 pass 逻辑问题，明确指出源码位置和错误分支。

---

## 二、算子约束违反

### 日志特征

- `opmagic`
- `tensor_magic`

### 分析流程

#### 步骤 1：提取关键信息

从日志中提取 `op_magic` 或 `tensor_magic`。

#### 步骤 2：获取计算图关键信息

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --tensor-magic <tensor_magic>
```

#### 步骤 3：获取算子信息

1. 如果是 `op_magic`，直接定位 Operation。
2. 如果是 `tensor_magic`，先查生产者，再查消费者。

#### 步骤 4：定位用户代码

1. 使用脚本输出的 `file` 和 `line`。
2. 查看源码上下文。

```bash
sed -n '<line-20>,<line+20>p' <source_file>
```

#### 步骤 5：检查算子约束

```bash
grep -R "<opcode>" docs/zh/api/operation/
```

#### 步骤 6：对比 Pass 前后变化

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --op-magic <op_magic>
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <after_json_path> \
  --op-magic <op_magic>
```

#### 步骤 7：确定根因

1. 如果是用户代码不满足约束，说明违反了哪条文档规则。
2. 如果是 pass 处理导致约束破坏，说明具体变更点。

---

## 三、计算图成环

### 日志特征

当日志中出现以下关键字时，应优先按“计算图成环”场景分析：

- `cycle detected`
- `graph cycle`
- `topological sort failed`
- `circular dependency`
- `Loop Detected`
- `OperationLoopCheck`
- `Cycle Detection`

说明：

- 当前报错 pass 不一定是成环的根因 pass。
- 某些 pass 只是首次执行拓扑检查、排序或子图依赖检查，因此会最先暴露问题，但环可能由更早的 pass 引入。

### 输入材料

分析该场景时，至少需要准备以下材料：

- 当前报错日志
- 当前报错 pass 对应的计算图 JSON
- 上游相邻 pass 的计算图 JSON
- 若可获得，当前 pass 的 Before / After 两份图
- 日志中对应的源码文件与行号

常见计算图目录位置：

- `output/output_*/`
- `build/output/bin/output/output_*/`

pass 中间目录通常类似：

- `Pass_{PASS_SEQ}_{PASS_NAME}`

例如：

```bash
ls -lt build/output/bin/output/output_*/Pass_*/*.json
```

```bash
ls -lt output/output_*/Pass_*/*.json
```

### 分析目标

本场景的分析目标包括：

1. 确认当前图中是否存在 op 之间的循环依赖
2. 确认当前图中是否存在 subgraph 之间的循环依赖
3. 判断当前报错 pass 是否为根因 pass
4. 沿 pass 链向前回溯，定位首次引入环的 pass 模块
5. 结合源码分析形成根因链，并给出修复建议

### 分析流程

#### 步骤 1：定位当前报错 pass 和对应图文件

1. 从日志中提取以下信息：
   - pass 名称
   - Before / After 阶段
   - 报错文件和行号
   - 相关 `op_magic`、`tensor_magic`
2. 在 pass 输出目录中定位当前报错 pass 的图文件
3. 同时收集上一个 pass 的输出图，作为回溯起点
4. 如果当前 pass 缺失 `After` 图，必须先参考“公共参考：`After` 计算图缺失时补打异常前计算图”一节，拿到异常前最近一份图后，再继续成环分析

#### 步骤 2：对当前图执行 op 级成环检测

op 级成环检测用于确认计算图中是否存在 operation 之间的循环依赖。

分析口径参考：

- `framework/src/interface/function/function.cpp`
- `Function::OperationLoopCheck(const std::string&)`

其核心逻辑是：

- 遍历所有 operation
- 根据 op 输出 tensor 到 consumer op 的关系建立依赖边
- 对 consumer 链执行 DFS
- 若遍历过程中再次进入 `IN_STACK` 状态的 op，则说明存在回边，图中成环

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --detect-op-cycle
```

需要记录的结果：

- 是否存在 op 级环
- 成环路径上的 `opmagic` 列表
- 每条依赖边对应的 `tensor magic`
- 每个 op 的 `opcode`、`file`、`line`

#### 步骤 3：对当前图执行 subgraph 级成环检测

subgraph 级成环检测用于确认不同 `subgraphid` 之间是否通过 tensor 形成闭环。

分析口径参考：

- `framework/src/interface/function/function.cpp`
- `Function::LoopCheck()`

其核心逻辑是：

- 按 `subgraphid` 聚合 operation
- 根据跨 subgraph 的 tensor 生产消费关系建立依赖边
- 对 subgraph 图执行 DFS
- 若遍历过程中再次进入 `IN_STACK` 状态的 subgraph，则说明 subgraph 之间存在环

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <graph_json_path> \
  --detect-subgraph-cycle
```

需要记录的结果：

- 是否存在 subgraph 级环
- 成环的 `subgraph id` 路径
- 触发该路径的跨 subgraph tensor
- 对应 producer op 和 consumer op

#### 步骤 4：判断当前图成环是否能被脚本证实

根据脚本结果分情况处理：

1. 如果脚本检测到显式环：
   - 继续沿 pass 链向前回溯，定位首次引入环的 pass
2. 如果脚本未检测到显式环，但日志提示 `cycle detected` 或 `topological sort failed`：
   - 不得直接认定“图中无环”
   - 需要进一步怀疑以下隐式依赖：
     - `dependOperand`
     - operation group 顺序约束
     - 调度级或拓扑级附加依赖

说明：

- 当前图 JSON 中已稳定导出 `ioperands`、`ooperands`
- 当前图 JSON 中未直接导出 `dependOperand`
- 因此脚本第一阶段只能稳定分析“显式 tensor 数据流”上的成环问题
- 若日志来自拓扑排序或调度阶段，则可能存在 JSON 未表达的隐式依赖，需要回到源码侧确认
- 若当前 pass 的标准 `After` 图缺失，但临时补打图已经生成，则应以该临时补打图作为“异常前最后状态图”继续本步骤，不得因为缺少标准 `After` 图而中止分析

#### 步骤 5：沿 pass 链向前回溯，定位首次引入环的 pass

当前报错 pass 不一定是根因 pass，需要逐步回溯。

回溯方法：

1. 对当前报错 pass 的输入图执行成环检测
2. 若当前 pass 存在临时补打图，优先对该图执行成环检测
3. 对上一个 pass 的输出图执行成环检测
4. 按 pass 顺序持续向前回溯
5. 找到“前一阶段无环、当前阶段有环”的边界
6. 将该 pass 标记为“首次引入环的根因候选 pass”

建议执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <curr_graph> \
  --detect-op-cycle
```

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <prev_graph> \
  --detect-op-cycle
```

若脚本支持对比模式，也可执行：

```bash
python3 scripts/computation_graph_analyzer.py \
  --before-json <prev_graph> \
  --after-json <curr_graph> \
  --compare-cycle
```

回溯结论必须区分：

- 报错 pass：首次抛出异常的 pass
- 根因候选 pass：首次引入环的 pass
- 根因未定：图文件不完整或隐式依赖无法从 JSON 还原时的结论

#### 步骤 6：结合源码确认根因修改点

对“首次引入环”的 pass，需要重点检查其是否进行了以下操作：

- 新增了错误的 producer-consumer 依赖
- 复用了旧 tensor，导致边关系串错
- 修改 op 输入输出时遗留旧依赖
- 跨 subgraph 迁移 op 后未同步修正边界 tensor
- 引入额外 depend 依赖但未同步维护一致性

需要输出的证据包括：

- 成环路径上的关键 op
- 成环路径上的关键 tensor
- pass 中建立或修改依赖的源码位置
- Before / After 图差异

### 输出要求

最终输出必须明确区分以下概念：

- **报错 pass**：第一个抛出成环异常的 pass
- **当前图证据**：当前图中是否真的检测到显式环
- **根因候选 pass**：首次引入环的 pass
- **能力边界**：是否存在 JSON 未表达的隐式依赖

建议按如下根因链格式输出：

`现象(某 pass 报 cycle) -> 图证据(某 JSON 中的 op/subgraph 环路) -> 首次引入位置(某上游 pass) -> 触发源码(某文件某行的依赖修改逻辑)`

### 注意事项

- 当前报错 pass 不一定是根因 pass
- 脚本未检测到显式环，不等于图中一定无环
- 拓扑排序失败还可能来自 `dependOperand` 或调度顺序约束
- 图文件不完整时，只能给出“根因候选”，不能伪造成确定结论

## 四、段错误

### 日志特征

当日志或终端输出中出现以下关键字时，**即使未明确指定异常类型**，也应自动按本“段错误”场景进行排查：
- `segment fault`
- `Segmentation fault (core dumped)`
- `SIGSEGV`
- 退出码 `139`
- `null pointer dereference`
- 框架底层的 C++ 崩溃堆栈（如 `libpypto.so` 内部报错）

### 分析目标

本场景的分析目标包括：
1. 优先通过框架日志获取崩溃上下文，避免不必要的耗时调试。
2. 在必须使用 GDB 时，确保调试现场（二进制文件）未被清理。
3. 捕获崩溃瞬间的完整 C++ 调用堆栈，定位触发空指针的具体源码文件与行号。
4. 结合 `Before` 计算图，确认该对象缺失的根本原因。

### 分析流程

#### 步骤 1：检查框架运行时异常日志（优先降级排查）

在调用耗时的 GDB 之前，**必须优先检查**常规测试日志和标准输出，看是否能直接定位问题。
1. 检查 `RunOnFunction` 级别是否已通过信号处理器（Signal Handler）或 `try-catch` 拦截并打印了崩溃堆栈。
2. 如果日志中已经明确打印了导致崩溃的 C++ 文件名、行号以及相关的算子信息，**请直接跳过 GDB 调试（跳过步骤 2 和 3）**，直接进入“步骤 4：定位源码与指针来源”。

#### 步骤 2：确认并保护调试现场（GDB 前置卡点）

如果步骤 1 的日志信息不足以定位空指针来源，必须启动 GDB 调试。但在启动前，必须确认调试目标存在。
1. 检查用于复现的测试二进制文件或 Core Dump 文件是否仍然存在于磁盘上。
2. **ST 场景特殊处理**：如果发现二进制文件已被自动化测试框架的清理脚本删除（找不到文件），**严禁直接报错退出**。你必须先执行带有保留环境参数的重跑命令（例如禁用清理的 flag），重新编译/运行以生成并保留可供调试的二进制文件。

#### 步骤 3：使用 GDB 批处理模式抓取崩溃堆栈（兜底排查）

确认二进制文件存在后，使用 GDB 的非交互式批处理模式（Batch Mode）直接运行并抓取堆栈。

执行以下命令（将 `<user_script.py> <args>` 替换为实际复现命令）：

```bash
gdb -batch -ex "run" -ex "bt full" -ex "quit" --args python3 <user_script.py> <args>
```

**需要记录的结果：**
- 崩溃发生的信号类型（通常为 `SIGSEGV`）。
- 堆栈顶层（`#0`, `#1` 等）属于 PyPTO 框架的 C++ 函数名及行号。
- `bt full` 打印出的关键局部变量（如哪个指针变量的值为 `0x0` 或 `nullptr`）。

#### 步骤 4：定位源码与指针来源

根据日志（步骤 1）或 GDB 堆栈（步骤 3）输出的文件和行号，提取源码上下文，重点寻找引发崩溃的指针操作。

```bash
sed -n '<line-15>,<line+15>p' <source_file>
```

**代码级静态分析重点：**
1. **确认崩溃点**：是 `tensor->GetShape()` 还是 `op->GetOpcode()` 触发了空指针？
2. **向上追溯来源**：该指针是如何获取的？常见的危险来源包括：
   - `GetProducers(...)` 或`GetConsumers(...)`未检查返回值是否为空。
   - `GetIOperands(...)` 或`GetOOperands(...)` 未检查返回值是否为 `nullptr`。
   - 动态类型转换 `dynamic_cast<T*>(...)` 失败返回空。

#### 步骤 5：结合 Before 计算图验证图状态

指针为空通常是因为计算图状态与代码预期不符。必须通过图文件寻找根本原因。

1. 找到当前报错 Pass 对应的 `Before` 计算图 JSON。
2. 根据源码中试图获取的对象特征（如特定的 `op_magic`、节点名称或属性名），在图中进行检索。

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --op-magic <expected_op_magic>
```

**交叉验证逻辑：**
- 如果 `Before` 图中**存在**该节点/属性，说明是当前 Pass 在崩溃前的逻辑中将其误删或破坏。
- 如果 `Before` 图中**不存在**该节点/属性，说明是上游 Pass 漏传，或者当前 Pass 缺少必要的判空保护逻辑（防御性编程不足）。

#### 步骤 6：确定根因并输出修复建议

根据上述收集的证据，形成完整的根因链，并给出具体的修复建议。

**常见修复建议模板：**
- **增加判空保护**：如果该节点/属性是可选的，建议在源码 `<line>` 处增加 `if (ptr == nullptr) { return FAILED; }` 或相应的容错逻辑。
- **修复上游逻辑**：如果该节点是必须的但在 `Before` 图中缺失，需指出是哪个上游 Pass 违反了图构建约束。

**根因链输出格式要求：**
`现象(日志拦截或GDB捕获SIGSEGV) -> 触发位置(源码行及空指针变量) -> 状态异常(Before图中缺失预期节点/属性) -> 修复方案(增加判空或修复图结构)`

## 五、属性缺失

### 日志特征

当日志中出现以下关键字时，应按“属性缺失”场景进行专项分析：
- `attribute not found`
- `missing attribute`
- `required attribute`
- `get attribute failed`
- `HasAttr(...) == false`

### 分析目标

本场景的分析目标包括：
1. 明确缺失的具体属性名称及其所属的算子（Operation）或张量（Tensor）。
2. 通过对比 `Before` 计算图，验证该属性是否确实缺失。
3. 沿 Pass 链向上游回溯，查明该属性是“从未被创建”还是“被中途误删”。
4. 结合算子约束文档，给出最终的修复建议。

### 分析流程

#### 步骤 1：提取关键信息

从报错日志中提取出引发异常的核心要素：
- 报错的节点标识：算子标识（如 `opmagic`、算子名称）或张量标识（如 `magic`、`memId`）。
- 缺失的属性名称（Attribute Name）。

#### 步骤 2：定位源码与属性获取逻辑

根据日志输出的文件和行号，提取源码上下文，查看当前 Pass 是如何获取该属性的。

```bash
sed -n '<line-15>,<line+15>p' <source_file>
```

**代码级静态分析重点：**
- 确认获取属性的 API 调用（如 `op->GetAttr("attr_name", value)` 或 `tensor->GetAttr("attr_name", value)`）。
- 检查代码是否缺少对属性存在性的校验（如调用前未执行 `op->HasAttr` 或 `tensor->HasAttr`）。
- 检查是否缺少默认值兜底逻辑。

#### 步骤 3：在计算图中验证属性状态（需注意 Dump 可见性）

当前 Pass 报错说属性缺失，必须通过 `Before` 图来验证图的真实状态。

```bash
# 针对算子或张量进行查询
python3 scripts/computation_graph_analyzer.py \
  --json-path <before_json_path> \
  --target-node <op_magic_or_tensor_name>
```

**交叉验证逻辑与防坑预警：**
- 检查 JSON 图中对应的 Operation 或 Tensor 节点上是否存在该属性。
- **Dump 可见性确认**：并非所有内存态的属性都会被序列化（Dump）到 JSON 中。如果在 JSON 中未找到该属性，不能立刻断定其缺失。需结合源码经验判断该属性是否属于“不参与 Dump 的内部属性”。如果是不可 Dump 属性，必须直接跳到步骤 5 进行纯源码级的追溯。

#### 步骤 4：查阅算子约束文档

查阅文档确认该属性的具体作用和约束条件，辅助判断构图逻辑是否存在缺陷。

```bash
grep -R "<opcode>" docs/zh/api/operation/
```

#### 步骤 5：向上游 Pass 回溯

如果确认属性在进入当前 Pass 前就已经丢失（或属于不落盘的不可 Dump 属性），必须找出是哪个环节弄丢了它。
1. **图状态对比（针对可 Dump 属性）**：获取上一个 Pass 的 `After` 图，检查属性是否存在，以此二分定位弄丢属性的具体 Pass。
2. **源码追溯（针对不可 Dump 属性或已定位的 Pass）**：不断向前回溯，直到找到该算子/张量最初被创建或被替换的那个 Pass。
3. **排查图修改操作**：重点排查上游 Pass 中涉及**算子/张量新增、替换或复制**的源码逻辑。看是否在这些图修改操作中漏掉了**属性继承（拷贝）**或显式设置属性的步骤（例如：新建 Tensor 时忘记拷贝原 Tensor 的属性，或拷贝 Operation 时未拷贝原 Operation 的属性）。

#### 步骤 6：确定根因并输出修复建议

根据上述收集的证据，形成完整的根因链，并给出具体的修复建议。

**常见修复建议模板：**
- **增强防御性编程**：如果在当前逻辑中该属性缺失属于可容忍情况，建议在源码 `<line>` 处增加 `HasAttr` 判断，并在缺失时赋予合理的默认值。
- **修复上游属性设置与拷贝**：指出前端构图漏传，或上游某 Pass 在重建算子时遗漏了属性继承，建议补充属性继承或拷贝的逻辑。

**根因链输出格式要求：**
`现象(日志报 missing attribute) -> 触发位置(获取属性的源码行) -> 状态异常(Before图中缺失) -> 违反规则(算子文档要求/防御性编程规范) -> 修复方案(增加校验/修复上游)`

---

## 输出要求

最终分析结果至少包含：

- 日志证据
- 计算图证据
- 源码位置
- 官方文档约束
- 可执行修复建议

## 六、内存越界

### 日志特征

当日志中出现以下关键字时，按内存越界场景分析：

- `Alloc tensor .* size .* exceeds .* size`
- `OP .* in/output total size .* exceeds .* size`
- `MEM_UB`、`MEM_L1`、`MEM_L0A`、`MEM_L0B`、`MEM_L0C`

### 分析目标

1. 确认越界内存层级（MEM_UB/L1/L0A/L0B/L0C）
2. 定位超限 tensor 及 producer op
3. 结合计算图回溯判断根因（前端配置/当前Pass/上游Pass）
4. 给出 tile_shape 或内存分配修复建议

### 分析流程

#### 步骤 1：从日志提取信息并定位计算图

从日志提取：
- `tensor_magic`：超限 tensor 标识
- `tensor size`：实际大小（bytes）
- `内存层级`：MEM_UB/L1/L0A/L0B/L0C
- `hardware_limit`：日志中的 `exceeds ... size [Y]`
- `producer op`：生产者算子
- `pass 名称`、报错文件和行号

定位计算图：

```bash
ls -lt output/output_*/Pass_*/*.json
ls -lt build/output/bin/output/output_*/Pass_*/*.json
```

同时收集上一个 pass 的输出图作为回溯起点。

#### 步骤 2：在计算图中搜索 tensor 并判断来源

使用计算图分析脚本：

```bash
python3 scripts/computation_graph_analyzer.py \
  --json-path <json_path> \
  --tensor-magic <tensor_magic>
```

兜底：`grep -n "magic.*<tensor_magic>" <json_path>`

**tensor 来源判断**：

| 图和日志证据 | 初步判断 | 后续验证 |
|---|---|---|
| 当前图找不到 `tensor_magic` | pass 新创建或图未落盘 | 检查补打图、上游 pass 输出图 |
| `COPY_IN` 输出，输入可回溯到用户输入 | 前端输入衍生 | 确认是否有 pass 放大 buffer、改 shape |
| `tensor_magic` 在上游 pass 图已存在 | 上游 pass 已生成 | 回溯首次出现边界 |
| 当前 pass 前后图中 size/shape/mem_type 变化 | 当前 pass 改写导致 | 结合源码确认 |

#### 步骤 3：前端配置问题判断

**三种超限场景**：

- 单 tensor 大小超限：日志特征 `Alloc tensor ... size [X] exceeds ... size [Y]`
- op 输入输出总占用超限：日志特征 `OP ... total size [X] exceeds ... size [Y]`
- Pass 使用有效阈值：日志阈值小于配置文件理论值

**公式定义**：

- **tile_shape**：前端设置切分形状
  - Vector：`pypto.set_vec_tile_shapes(128, 512)` → [128, 512]
  - Cube：`pypto.set_cube_tile_shapes([m, k, n])`
- **dtype_size**：单元素字节大小（fp32=4, fp16=2, int8=1）
- **hardware_limit**：硬件物理容量（配置文件 `[AICoreSpec]` section：ub_size、l1_size）
- **effective_threshold**：Pass 实际阈值（从日志获取）

**tensor_size 计算**：`(tile_shape各维度相乘) × dtype_size`

例：tile_shape=[128,512]，dtype=fp32 → 128×512×4 = 262144 bytes = 256 KB

#### 步骤 4：回溯上游 Pass

对上游 pass 计算图执行 tensor_magic 搜索，找到"Before 不存在、After 存在"的边界 pass，即为首次生成该 tensor 的位置。

#### 步骤 5：定位源码并确定根因

根据日志文件名和行号定位源码：

```bash
sed -n '<line-15>,<line+15>p' <source_file>
```

常见 Pass 锚点：

| Pass | 文件 | 搜索锚点 |
|---|---|---|
| OoOSchedule | `schedule_base.h` | `exceeds %s size` |
| ReplaceTensor | `replace_tensor.cpp` | `GetMemoryLimit(MEM_UB)` |
| AssignMemoryType | `assign_memory_type.cpp` | `GetMemoryLimit(MEM_UB)` |

**根因判断**：

| 根因类型 | 判断依据 | 修复建议 |
|---|---|---|
| 前端配置超限 | tensor 可回溯到前端输入，无 pass 改写证据 | 调整 tile_shape 或 dtype |
| 当前 Pass 问题 | pass 前后图或源码指向该 pass 新增 alloc/spill/copy | 修复内存分配、shape 处理逻辑 |
| 上游 Pass 问题 | 回溯发现上游 pass 首次生成或放大超限 tensor | 修复首次引入超限的 pass |

**根因链格式**：

```
现象(日志) -> 触发位置(文件:行号) -> 状态异常(tensor size > limit) -> 根因(前端/当前Pass/上游Pass)
```

### 输出要求

内存越界分析报告必须包含：

- **日志证据**：完整 exceeds 日志
- **tensor 信息**：magic、shape、dtype
- **计算图回溯**：tensor 在各 pass 图中的搜索结果
- **根因定位**：前端/当前Pass/上游Pass
- **修复建议**：参数调整或代码修复方案

---
