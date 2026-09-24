# F9XXXX

## F91001 INVALID_CONFIG

**错误描述**

传入的配置参数不符合仿真器要求，包括配置字符串格式不匹配`key=value`规范、配置名称缺少`.`层级分隔符、AICPU核心数量不等于AIC与AIV之和。

**可能原因**

+ 配置字符串未能匹配`key=value`。

   ```json
   # 错误示例-配置字符串缺少=value
   Model.deviceArch=
   ```

+ 配置前缀与子键之间缺少`.`分隔符（如`Pipeline`应为`Pipeline.xxx`）。

   ```json
   # 错误示例-配置名称缺少.分隔符
   deviceArch="xxx"
   ```

+ `coreMachineNumberPerAICPU`不等于`aicNum + aivNum`。

   ```json
   # 错误示例-核心数配置不匹配
   Model.coreMachineNumberPerAICPU=12
   Model.cubeMachineNumberPerAICPU=4
   Model.vecMachineNumberPerAICPU=4
   ```

**处理方式**

1. 确保配置字符串格式为`key=value`，key与value之间用`=`连接，如`Pipeline.enable=1`。

   ```json
   # 正确示例-配置字符串key=value格式
   Model.deviceArch="xxx"
   ```

2. 检查配置名称包含正确的层级前缀和`.`分隔符，参考`framework/src/cost_model/simulation/config/`下各`xxx.h`中的配置定义。

3. 确保`coreMachineNumberPerAICPU`等于`aicNum + aivNum`，重新核对硬件配置参数。

   ```json
   # 正确示例-核心数配置正确
   Model.coreMachineNumberPerAICPU=12
   Model.cubeMachineNumberPerAICPU=4
   Model.vecMachineNumberPerAICPU=8
   ```

## F91002 INVALID_CONFIG_NAME

**错误描述**

配置名称格式正确（含`.`分隔符），但在仿真配置分发器（dispatcher）中找不到对应的处理器，即配置名不在已注册列表中。

**可能原因**

+ 配置名称拼写错误。

   ```json
   # 错误示例-配置名称拼写错误
   Model.devicearch="xxx"
   ```

+ 使用了不存在或尚未在dispatcher中注册的配置名。

   ```json
   # 错误示例-使用未注册的配置名
   Model.notExist="xxx"
   ```

**处理方式**

1. 核对配置名称拼写是否正确。
2. 参考`framework/src/cost_model/simulation/config/xxx.h`中dispatcher注册的合法配置项列表。
3. 如需新增配置项，在对应配置头文件的dispatcher函数中注册新的配置名称。

## F91003 FILE_FORMAT_ERROR

**错误描述**

JSON文件解析失败，内容不符合JSON语法规范。

**可能原因**

+ JSON存在语法错误：缺少引号、逗号多余或缺失、括号不匹配。
+ 文件编码为非UTF-8或包含非法字符。

   ```json
   # 错误示例-JSON语法错误（key无引号、末尾多余逗号）
   {
      key: "value",
      "list": [1, 2,]
   }
   ```

**处理方式**

1. 使用JSON校验工具检查文件格式：

   ```bash
   # 正确示例-校验JSON格式
   python3 -m json.tool my_config.json
   ```

2. 确保文件编码为UTF-8。
3. 确保JSON中所有字符串用双引号`"`包裹，末尾不能有多余逗号。

   ```json
   # 正确示例-合法JSON
   {
      "key": "value",
      "list": [1, 2]
   }
   ```

## F91004 FILE_CONTENT_ERROR

**错误描述**

文件能够打开和解析，但内容不符合约定的字段要求，如配置行缺少`=`分隔符、数值字段超出合法范围。

**可能原因**

+ 配置文件某行未遵循`key=value`格式（缺少`=`）。

   ```txt
   # 错误示例-配置行缺少等号
   Pipeline.enable 1
   ```

+ CSV行中数值超出`uint64_t`范围。

   ```txt
   # 错误示例-CSV行中数值超出uint64_t范围
   18446744073709551616
   ```

**处理方式**

1. 检查配置文件每行是否为`key=value`格式，确保key与value之间用`=`分隔。

   ```txt
   # 正确示例-配置行key=value格式
   Pipeline.enable=1
   ```

2. 检查数值字段是否在合法范围内（如`uint64_t`不超过2^64-1）。

## F91005 INVALID_PATH

**错误描述**

仿真所需的中间产物文件、Python绘图脚本或精度仿真目标文件（`.o`）路径不存在。

**可能原因**

+ 前置仿真步骤（函数构建、调度等）未正常完成，中间文件（`dyn_topo.txt`、`program.json`、`pipe.swim.json`、`swim.json`、`topo.json`等）未生成。

   ```txt
   # 错误示例-中间文件dyn_topo.txt未生成
   [SIMULATION]: dyn_topo.txt does not exist. topo_txt_path: /path/to/output_xxx/CostModelSimulationOutput/dyn_topo.txt
   ```

+ Python绘图脚本（`draw_pipe_swim_lane.py`、`print_swim_lane.py`、`draw_swim_lane.py`）路径不正确。

   ```txt
   # 错误示例-Python绘图脚本路径不存在
   [SIMULATION]: draw_pipe_swim_lane.py does not exist. drawScriptPath: /path/to/draw_pipe_swim_lane.py
   ```

+ 精度仿真编译产物（`.o`文件）缺失。

   ```txt
   # 错误示例-编译产物.o文件不存在
   obj file does not exist. objPath: /path/to/output.o
   ```

**处理方式**

1. 检查日志中报错的文件路径，确认文件是否存在。

   ```bash
   # 正确示例-确认中间文件已生成
   ls output_xxx/CostModelSimulationOutput/dyn_topo.txt
   ls output_xxx/CostModelSimulationOutput/program.json
   ```

2. 确认前置仿真步骤是否全部正常完成。
3. 确认Python绘图脚本在预期路径下，必要时从源码目录复制到输出目录。
4. 确认精度仿真相关组件已编译完成。

## F91006 FILE_OPEN_FAILED

**错误描述**

无法打开指定文件，常见于JSON配置文件、日历文件、拓扑文件等。

**可能原因**

+ 文件不存在或路径错误。

   ```bash
   # 错误示例-指定文件不存在
   $ ls non_exist.json
   ls: cannot access 'non_exist.json': No such file or directory
   ```

+ 当前用户无读取权限。

   ```bash
   # 错误示例-文件无读取权限
   $ ls -l my_config.json
   ---------- 1 root root 1024 Jan 1 12:00 my_config.json
   ```

+ 文件已损坏或被其他进程锁定。

**处理方式**

1. 确认文件路径是否正确。
2. 检查当前用户对文件是否有读取权限：

   ```bash
   # 正确示例-确认文件存在且有读权限
   $ ls -l my_config.json
   -rw-r--r-- 1 user group 1024 Jan 1 12:00 my_config.json
   ```

3. 尝试用相应工具打开文件验证其完整性。

## F91007 PYTHON_CMD_ERROR

**错误描述**

仿真过程中执行Python脚本（如泳道图绘制脚本`draw_pipe_swim_lane.py`、`print_swim_lane.py`、`draw_swim_lane.py`）返回非零退出码。

**可能原因**

+ Python环境缺少依赖（如`matplotlib`、`graphviz`）。

   ```bash
   # 错误示例-缺少matplotlib依赖
   $ python3 draw_pipe_swim_lane.py input.json
   ModuleNotFoundError: No module named 'matplotlib'
   ```

+ Python脚本与当前Python版本不兼容。

   ```bash
   # 错误示例-Python版本不兼容导致脚本语法错误
   $ python3 draw_pipe_swim_lane.py input.json
   SyntaxError: invalid syntax
   ```

+ 脚本的输入文件缺失或格式错误。

   ```bash
   # 错误示例-脚本输入文件不存在
   $ python3 draw_pipe_swim_lane.py missing_input.json
   FileNotFoundError: [Errno 2] No such file or directory: 'missing_input.json'
   ```

**处理方式**

1. 检查Python环境是否可用：

   ```bash
   python3 --version
   ```

2. 安装所需Python依赖：

   ```bash
   pip3 install matplotlib graphviz
   ```

   ```bash
   # 正确示例-确认依赖已安装
   $ python3 -c "import matplotlib; print('OK')"
   OK
   ```

3. 手动执行报错的Python命令，查看详细错误输出，根据提示修复。

## F92006 INVALID_PIPE_TYPE

**错误描述**

仿真中遇到未识别的pipeline类型：`SCHED_CORE_PIPE_TYPE`中缺少对应opcode的映射，或对非cache类型调用了`GetAddress()` / `GetSize()`。

**可能原因**

+ 使用了`SCHED_CORE_PIPE_TYPE`数据结构中尚未注册的新opcode。
+ 对非read-cache / write-cache的pipe类型调用了地址或大小查询方法。

**处理方式**

1. 在`framework/src/cost_model/simulation/common/ISA.h`的`SCHED_CORE_PIPE_TYPE`数据结构中添加新opcode对应的pipe类型映射。
2. 确保`GetAddress()` / `GetSize()`仅对cache类型调用。

## F92007 SHAPE_INVALID

**错误描述**

传入的Tensor shape为空或第一维为空，仿真无法执行。

**可能原因**

+ 输入Tensor的shape未正确初始化或动态shape推导结果为空。

   ```python
   # 错误示例-shape为空列表
   x = pypto.tensor([], pypto.DT_FP32)
   ```

**处理方式**

1. 检查输入数据shape是否有效，确保shape非空且第一维非空。

   ```python
   # 正确示例-shape包含有效维度
   x = pypto.tensor([4, 8], pypto.DT_FP32)
   ```

2. 对于动态shape场景，确认shape推导逻辑正确。

## F92010 DEAD_LOCK

**错误描述**

仿真运行中检测到死锁，某个Machine在某个cycle无法继续推进调度。

**可能原因**

+ 任务依赖图存在循环依赖或资源竞争。

   ```txt
   # 错误示例-仿真死锁日志
   [ReportDeadlock] Machine 0 is deadlock at cycle 12345
   Simulation is deadlock at cycle 12345 !!!!!!!!!
   ```

+ 任务调度逻辑存在缺陷。

**处理方式**

1. 在`output/output_xxx/CostModelSimulationOutput/graphs`目录下找到死锁对应的dot文件。
2. 使用Graphviz渲染dot文件分析任务依赖关系：

   ```bash
   dot -Tpng deadlock.dot -o deadlock.png
   ```

3. 检查任务依赖图中是否存在循环依赖或资源竞争。
4. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## F94001 NO_SO_EXISTS

**错误描述**

精度仿真所需的共享库文件（如`libpem_davinci.so`）加载失败，文件不存在或路径不正确。

**可能原因**

+ 精度仿真 .so文件未编译或未安装到预期路径。
+ 编译产物路径配置不正确。

   ```txt
   # 错误示例-共享库文件加载失败
   can not load library: /path/to/libpem_davinci.so
   ```

**处理方式**

1. 开启仿真日志定位缺失的 .so路径：

   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

2. 在日志中搜索`can not load library:`，找到缺失的 .so文件路径。
3. 确认精度仿真相关组件已编译安装。

   ```bash
   # 正确示例-确认.so文件存在
   ls -l /path/to/libpem_davinci.so
   ```

## F94002 CANN_LOAD_FAILED

**错误描述**

CANN环境未正确加载，`ASCEND_HOME_PATH`环境变量未设置，精度仿真不可用。

**可能原因**

+ 未执行CANN环境初始化脚本。

   ```bash
   # 错误示例-ASCEND_HOME_PATH未设置
   $ echo $ASCEND_HOME_PATH
   (空)
   ```

**处理方式**

1. 执行CANN环境初始化脚本：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 确认环境变量已设置：

   ```bash
   # 正确示例-环境变量已设置
   $ source /usr/local/Ascend/ascend-toolkit/set_env.sh
   $ echo $ASCEND_HOME_PATH
   /usr/local/Ascend/ascend-toolkit/latest
   ```

## F94003 CMD_ERROR

**错误描述**

精度仿真中执行外部命令失败，如命令字符串格式化截断导致`snprintf_s`失败，或`llvm-objcopy`返回非零退出码。

**可能原因**

+ 命令字符串过长导致格式化截断。
+ `llvm-objcopy`工具未安装或版本不兼容。

   ```bash
   # 错误示例-llvm-objcopy未安装
   $ llvm-objcopy --version
   bash: llvm-objcopy: command not found
   ```

+ 目标文件路径异常导致命令执行失败。

   ```txt
   # 错误示例-目标文件路径异常
   cmd error: llvm-objcopy --only-section=.text /path/to/source.o /path/to/nonexistent/target.o
   ```

**处理方式**

1. 检查日志中打印的完整命令字符串是否正确。
2. 确认`llvm-objcopy`工具已安装且版本兼容：

   ```bash
   llvm-objcopy --version
   ```

3. 在终端中手动执行日志中的命令，查看具体错误输出。

## F94005 CANNSIM_FAILED

**错误描述**

未通过`cannsim`方式启动精度仿真，导致精度仿真不可用。

**可能原因**

+ 未使用`cannsim record`命令启动精度仿真。

   ```bash
   # 错误示例-直接用python3启动而未使用cannsim
   python3 my_script.py --run_mode sim
   ```

**处理方式**

1. 使用`cannsim record`方式启动精度仿真：

   ```bash
   # 正确示例-通过cannsim启动精度仿真
   cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950
   ```

## F9FFFF SIM_INNER_ERROR

**错误描述**

NA

**可能原因**

NA

**处理方式**

1. 请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。
