# F6XXXX

## F62014 SYMBOL_NOT_FOUND

**错误描述**

Kernel代码生成阶段变量未定义，日志上下文含`UNDEFINED_VAR`关键字。

**可能原因**

- 子图中的Operation缺少`need_alloc`属性。CodeGen依赖该属性生成变量定义语句，缺失导致变量定义缺失。

**处理方式**

1. **设置日志级别为INFO**：

   ```bash
   export ASCEND_PROCESS_LOG_PATH=<用户指定日志路径>
   export ASCEND_GLOBAL_LOG_LEVEL=1  # 0:DEBUG, 1:INFO, 2:WARN, 3:ERROR
   ```

2. **并行编译改为串行**：修改`tile_fwk_config.json`中`"parallel_compile"`选项，将值改为1使codegen串行编译生效，然后重新编译并部署。

3. **执行用例，获取日志及kernel代码**：
   - 日志：`{日志路径}/debug/plog/pypto-log*.log`
   - Kernel代码：搜索`kernel_aicore/`文件夹内`TENSOR*.cpp`

4. **分析日志**：
   - F60XXX / F61XXX类错误 → 一般为上游数据异常，结合PASS日志分析。
   - 其他类型 → 结合上下文分析。

> **示例**：以kernel代码中TileOp调用参数不符合预期为例。

1. 按上述步骤收集日志。
2. 找到不符合预期的TileOp调用，如：

   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```

3. 以上述代码为关键字搜索日志。
4. 往上找第一个`Op CodeGenNPU Start`，即该TileOp生成起点，向后逐行检查。
5. 若怀疑PASS数据问题，搜索`Gen OP IS`获取Operation Dump信息：

   ```log
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32> %152@5#(0)MEM_UB = !10010 TILE_ADD(...) ...
   ```

   其中`!10010`为该OP唯一标识码，可在PASS图或日志中搜索。PASS定位详见 [pass.md](pass.md)。

## F63001 COMPILE_CODE_FAILED

**错误描述**

Kernel代码编译失败。

> 若bisheng编译器报错日志不完整，可在日志中找到`compile cmd is:`后的make命令，手工执行以获取完整错误信息。

**可能原因**

- **堆栈溢出**（`error: stack frame size exceeds limit`）：函数栈帧超过限制。
- **PTO指令数据类型不匹配**（`the 2nd parameter maybe need a type`）：前端接口参数传递错误，或使用了PTO-ISA不支持的数据类型。
- **硬件指令与平台不匹配**（`does not support the given target feature`）：编译参数为Vector但代码含Cube指令（或反之）。
- **变量未定义**（`use of undeclared identifier`）：运行时动态Shape/Offset变量缺失，数据来源于`Function::GetDynParamTable`。

**处理方式**

1. 堆栈溢出：参考 [算子编译报堆栈溢出错误](../faq/stack-overflow-compilation.md#算子编译报堆栈溢出错误)。
2. 数据类型不支持：重新分析算子计算流程，选用硬件支持的数据类型。
3. 指令与平台不匹配：
   - 确认Block子图是否纯Vector或纯Cube（CodeGen不得混用）。
   - 确认PASS对`Function::IsCube()`的设置是否正确。
4. 变量未定义：联合PASS排查`Function::GetDynParamTable`返回的变量集合是否存在遗漏。

## 编译时长统计

### 整体耗时

Compiler Monitor 默认开启（Watchdog 机制）。编译开始时在 INFO 日志中记录总超时阈值及调整方式；编译总耗时超过阈值时在 WARNING 日志中记录超时摘要（瓶颈阶段、各阶段耗时）：

```log
[Compiler Monitor] Threshold: 600s | You can adjust the timeout threshold via pypto.set_host_options(compile_timeout=...)
[Compiler Monitor] Threshold: 600s | DetectedStage: CodeGen | Prepare: 12s | Pass: 45s | HostMachine: 86s | CodeGen: 1712s
```

编译总耗时未超过阈值时，仅在 INFO 日志中记录阈值提示。可通过 `pypto.set_host_options(compile_monitor_enable=0)` 关闭监控，或设置 `compile_timeout` 调整总超时阈值。

主要编译阶段顺序（从前到后）：

1. **Prepare**：前端准备（import 至开始 stash 编译任务）
2. **Pass**：图编译 Pass
3. **HostMachine**：动态场景下的控制流构建/编译、AICore 链接与 Encode 等
4. **CodeGen**：CCE 代码生成及二进制编译（含 FuncToBin 子阶段，不单独展示）

瓶颈阶段判定：将 `compile_timeout` 均分到上述四个阶段，耗时超过该平均阈值的阶段记入 `DetectedStage`。

### 单kernel文件编译时长

1. 在`kernel_aicore/`下找到目标`TENSOR*.cpp`，文件底部复制bisheng编译命令。
2. `cd`到output上一层目录，执行该命令确认可行。
3. 用`time`统计：

   ```bash
   time bisheng -c -O3 -g -x cce ... -o .../TENSOR_xxx.o .../TENSOR_xxx.cpp
   ```

4. 若编译时间较长且单kernel文件超过5K行，联合PASS考虑调整子图切分。
