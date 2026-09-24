# F7XXXX-F8XXXX

## F70001 COMPILE_AICORE_FAILED

**错误描述**

AICore kernel编译阶段失败。

**可能原因**

- CCE源码生成失败（`GEN_AICORE_FILE_FAILED`）。
- CCE编译命令执行失败（`COMPILE_CCEC_FAILED`）。
- 链接失败（`LINK_FAILED`），常见于并行编译符号未处理。

**处理方式**

1. 检查日志中的CCE编译命令和错误输出。
2. 若出现`ld.lld: error: undefined symbol`，修改`tile_fwk_config.json`中`"parallel_compile": 1`改为串行编译。

## F71004 LAUNCH_AICORE_FAILED

**错误描述**

Host侧AICore kernel下发失败。

**可能原因**

NA

**处理方式**

1. 检查编译产物`kernel_aicore/*.o`是否生成成功。
2. 确认环境变量`ASCEND_HOME_PATH`是否正确。

## F71005 RANGE_VERIFY_FAILED

**错误描述**

encode阶段`InitRawTensorAndMemoryRequirement`断言：`Shape size mismatch`或`Data size mismatch`，伴随`actualRawmagic`、`rawShape`、`actualrawShape`输出。

**可能原因**

内存复用链两端（rawTensor与其actualRaw）的rawShapeSize或rawDataSize不一致。常见于reshape / assemble / view等操作的actualRawmagic传递遗漏。

**处理方式**

1. 从断言日志记录`rawMagic`、`actualRawmagic`、`rawShape`。
2. 打开pass计算图dump（`tile_fwk_config.json`中`"dump_graph": true`）。
3. 在`build/output/pass/`下重点排查**pass4 ~ pass27**之间的tilegraph，搜索对应rawMagic。
4. 沿数据流分析是否某个pass修改了一侧rawshape而另一侧未同步。
5. **判断归属**：
   - 用例写法不满足API约束（inplace、valid_shape一致性等）→ 调整用例。
   - 用例正确 → 联系pass同事排查actualRawmagic传递遗漏。

## F71008 MAP_REG_ADDR_FAILED

**错误描述**

Host侧映射寄存器地址失败：`Map reg addr fail, maybe others are using current device`。

**可能原因**

- 当前设备（DEVICE_ID）被其他进程占用，寄存器映射冲突。
- 同一设备上有残留进程未释放资源。
- 多进程/多容器同时映射同一设备寄存器。

**处理方式**

1. 执行`npu-smi info`查看设备占用情况，确认目标DEVICE_ID是否被其他进程占用。
2. 清理占用设备的残留进程：`fuser -v /dev/davinci<DEVICE_ID>`定位并`kill`。
3. 在容器/多进程场景下，确保`ASCEND_DEVICE_ID`或`DEVICE_ID`环境变量设置正确且不冲突。
4. 若问题持续，尝试重启NPU驱动或切换空闲设备。

## F7100B SYNC_FAILED

**错误描述**

Host侧stream同步失败：`LaunchAicoreKernel`中`DynamicLaunchSynchronize`返回非0，schedule stream与aicore stream之间同步异常。

**可能原因**

- AICore执行异常后stream同步失败。
- Debug模式下profiling数据同步后stream同步失败。
- 底层Runtime stream同步超时或状态异常。

**处理方式**

1. 检查是否开启了`runtime_debug_mode`或手动调用了`torch.npu.synchronize()`，暂时关闭以排除同步时机引入的问题。
2. 查看device log中同步失败时刻前后的异常信息（timeout、task abort、stream abort等）。
3. 若为整网场景，检查是否与其他组件存在stream依赖冲突。

## F72002 HANDSHAKE_TIMEOUT

**错误描述**

Schedule AICPU与AICore握手超时，调度线程无法正常启动。

**可能原因**

- NPU设备不可用或驱动异常。
- 当前进程/容器内NPU资源占用过高，多进程争用同一设备。
- 握手超时配置过短与环境不匹配。

**处理方式**

1. 执行`npu-smi info`确认设备与驱动正常。
2. 检查进程/容器内NPU占用是否过高。
3. 查看日志上下文（如`Schedule run init succ`之后、AbnormalStop相关），区分首次握手失败或运行中异常。
4. 使用关联Skill[pypto-environment-setup](../../../../../.agents/skills/pypto-environment-setup/SKILL.md)。

## F73001 CTRL_FLOW_EXEC_FAILED

**错误描述**

Ctrl AICPU控制流执行失败（devTask构建、stitch处理异常）。

**可能原因**

- Root函数allocCtx或stitchCtx为空。
- 控制流初始化失败。
- Task stats异常。

**处理方式**

1. 检查device log确认具体失败节点（`DEV_TASK_BUILD` / `ROOT_STITCH`）。
2. 查看Ctrl AICPU日志上下文，确认是否有`CELL_MATCH`相关异常。
3. 若伴随stitch依赖异常（精度问题、怀疑依赖边丢失），启用`runtime_debug_mode=3`进行运行时依赖校验：

   ```python
   @pypto.frontend.jit(debug_options={"runtime_debug_mode": 3})
   ```

   执行用例后在输出目录运行：

   ```bash
   python tools/verify_dep_correctness.py <dump_dir>
   ```

   校验规则：

   | 规则 | 内容 |
   |---|---|
   | `rule_static_integrity` | 编译期声明的静态后继在运行时是否保留 |
   | `rule_stitch_legality` | stitch边引用的producer/consumer是否合法 |
   | `rule_cell_write_conflict` | 同一cell是否存在并发写冲突 |

   无问题输出`PASS`，有问题输出分类摘要及`dep_check_report.csv`详细定位。

## F73008 CTRL_ALLOC_TIMEOUT

**错误描述**

Ctrl AICPU运行超时或整网环境中AICPU执行超时。

**可能原因**

- 整网场景中，除PyPTO外其他组件也使用AICPU，强制同Cluster分配导致资源不足。
- `launch_sched_aicpu_num`配置不当。

**处理方式**

1. 整网场景下，设置`PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false`，并通过`launch_sched_aicpu_num`配置可用AICPU数量。
2. 注意：同Cluster分配开启时`launch_sched_aicpu_num`不生效。

## F7400B WORKSPACE_CAPACITY_INSUFFICIENT

**错误描述**

内存分配失败，表现为：

- `torch.OutOfMemoryError: NPU out of memory`
- `rtMalloc failed. size:xxx`

**可能原因**

- Workspace预算估算过大（Tensor Workspace / Metadata / AICore Spilled / Debug四大类中某一项异常）。
- 未经Tiling的超大Tensor进入了子图（Inplace/FixedAddress/shmemData等例外场景）。
- boundary Outcast slot数量过多或单体大小异常。

**处理方式**

1. **定位异常大类**：开启INFO日志后，搜索`[workspaceSize]`，对比`Metadata / tensor / aicoreSpillen / debug`各项。
2. **缩小到Root Function**：每个Root有独立日志`MaxRootInnerMem is xxx`，最大者即为问题来源。
3. **定位问题Tensor**：搜索`staticMemReq=[xxx] is too larger`警告，根据rawmagic在pass计算图中定位。
4. **调整配置**：
   - `max_workspace_kb` → **推荐优先配置**，使能内存驱动 stitch 模式；尤其在使用 `unroll_list` 时，按 encode 日志推荐值设置。
   - `stitch_function_max_num` → 未使能内存驱动模式时控制 stitch 深度（按 root function 个数）。
   - `unroll_list` / `max_unroll` → 控制展开次数。
5. **关联Skill**：[pypto-machine-workspace](../../../../../.agents/skills/pypto-machine-workspace/SKILL.md)。

## F7400X Workspace内存重叠 / 精度问题

**错误描述**

算子精度异常，怀疑MACHINE workspace内存复用存在overlap或踩踏。

**处理方式**

1. **检查输入初始化**：保证输入/输出已正确初始化。
2. **检查Tensor连续性**：MACHINE相关算子要求输入/输出连续（`tensor.is_contiguous()`），非连续tensor可能导致异常。
3. **扩大workspace**：在`python/pypto/frontend/parser/entry.py`中将`workspace_tensor`扩大10倍，若问题消失则说明workspace容量估算偏小。
4. **workspace内部自管理**：修改`framework/src/machine/runtime/device_launcher.cpp`的`PrepareDevProgArgs`，禁用外部workspace传入，改为内部`AllocDev`。
5. **Leaf粒度内存重叠检测**：开启`ENABLE_DUMP_OPERATION=1` + `runtime_debug_mode=1`，运行后执行：

   ```bash
    python3 tools/schema/schema_memory_check.py -d <device_log_dir> -t <dyn_topo.txt_path>
    ```
