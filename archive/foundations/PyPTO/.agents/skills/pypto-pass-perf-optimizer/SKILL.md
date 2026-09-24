---
name: pypto-pass-perf-optimizer
description: PyPTO Pass 编译性能优化技能。用于分析和优化 Pass 模块的编译性能，当 Pass 编译耗时过长需要优化时使用此技能。
---

# PyPTO Pass 编译性能优化技能

## 概述

本技能用于分析和优化 PyPTO Pass 模块的编译性能，帮助识别性能瓶颈并提供优化建议。

## ⚠️ 执行纪律警告（CRITICAL）

**🛑 严禁跳过任何关键步骤！以下步骤必须严格执行：**

### 必须完成的步骤序列

```
步骤1-5（环境准备+性能测量）
    ↓
步骤6（编译Debug版本） ⚠️ 强制，不可跳过
    ↓
步骤7（perf分析） ⚠️ 强制，不可跳过
    ↓
步骤8（内存分析，可选）
    ↓
步骤9（理解Pass功能）
    ↓
步骤10-13（性能分析） ⚠️ 强制，必须基于perf数据
    ↓
步骤14-18（优化实施）
    ↓
步骤19（UT验证） ⚠️ 强制，不可跳过
    ↓
步骤20-22（验证与迭代）
    ↓
步骤23（优化总结报告） ⚠️ 强制，不可跳过
```

### 禁止行为

❌ **严禁跳过步骤6-7**：Debug编译和perf分析是优化的数据依据
❌ **严禁跳过步骤10-13**：必须基于perf数据进行优化决策，禁止凭猜测优化
❌ **严禁在未完成perf分析前修改代码**：这是盲目优化，可能导致错误的方向
❌ **严禁在UT未通过前进行性能验证**：功能正确性优先于性能优化

### 正确的执行方式

**优化前的准备工作**：
1. ✅ 必须完成步骤6：编译Debug版本（perf需要调试符号）
2. ✅ 必须完成步骤7：使用perf record采集数据
3. ✅ 必须查看perf report，了解热点函数
4. ✅ 必须记录 top 5 热点函数及其 cycles 占比
5. ✅ 必须理解热点函数的来源（查看调用栈）

**只有完成以上准备工作后**，才能进入步骤14进行代码优化。

**优化后的验证工作**：
1. ✅ 必须完成步骤19：运行UT验证功能正确性
2. ✅ UT全部通过后，才能进行性能验证
3. ✅ 如果UT失败，必须回退修改或重新优化

### 常见错误示例

**错误 1：跳过perf分析直接优化**
```
步骤1-5 → ❌跳过步骤6-7 → 步骤14修改代码
后果：缺乏数据支撑，优化方向可能错误
```

**错误 2：仅凭代码审查进行优化**
```
步骤1-5 → ❌跳过步骤6-13 → 步骤14修改代码
后果：无法识别真正的性能瓶颈，浪费时间
```

**错误 3：UT未通过就进行性能验证**
```
步骤14修改代码 → ❌跳过步骤19 → 步骤20性能验证
后果：可能引入功能bug，性能优化无效
```

### 关键提醒

**为什么要使用perf分析？**
1. **精确定位热点**：只凭代码审查无法确定实际的热点函数
2. **量化分析**：需要cycles、cache-misses等量化指标
3. **调用栈分析**：perf -g 可以显示完整调用栈，找出真正的性能瓶颈
4. **避免盲目优化**：基于实际数据而非猜测进行优化

**为什么要编译Debug版本？**
1. Debug版本包含完整的调试符号
2. perf需要调试符号才能正确显示函数名
3. Release版本优化后可能丢失符号信息

**为什么必须运行UT？**
1. 性能优化可能引入功能bug
2. 必须确保功能正确性优先
3. 性能提升不能以牺牲功能为代价

## 核心目标

### 性能目标定义

**基准目标**：
- 200,000 Op 场景下，单个 Pass 平均耗时不超过 20s

**关键术语定义**：
- **Op数量**：单个Function中的Operation总数（从expand_function Pass日志中获取）
- **平均耗时**：Pass在所有Function上执行的平均时间（总耗时 / 执行次数）
- **线性关系**：Op数量增加或减少时，目标时间线性增加或减少

**目标时间计算公式**：
```
目标平均耗时 = (实际Op数量 / 200,000) × 20s
```

**示例**：
- 100,000 Op → 目标平均耗时 = (100,000 / 200,000) × 20s = 10s
- 200,000 Op → 目标平均耗时 = (200,000 / 200,000) × 20s = 20s
- 300,000 Op → 目标平均耗时 = (300,000 / 200,000) × 20s = 30s

**性能判断标准**：
- ✅ **达标**：实际平均耗时 ≤ 目标平均耗时
- ⚠️ **未达标**：实际平均耗时 > 目标平均耗时
- 💡 **优化方向**：降低平均耗时，使其不超过目标值

## 触发机制

当用户提到以下关键词时触发:
- "Pass 编译性能优化"
- "Pass 耗时优化"
- "Pass 性能分析"
- "优化 XXX Pass 模块"

## 使用场景

- 分析 Pass 编译耗时
- 识别性能瓶颈
- 优化 Pass 编译性能
- 验证优化效果

## Bundled Scripts 路径解析（必须先做）

本 skill 自带工具脚本在 `pypto-pass-perf-optimizer/scripts/`，即本 `SKILL.md` 同级目录下，不在 PyPTO 仓库根目录的 `scripts/`。
执行前先设置 `PASS_PERF_SCRIPTS_DIR=<本 SKILL.md 所在目录>/scripts`，后续 helper script 都用 `$PASS_PERF_SCRIPTS_DIR/...` 调用，业务命令仍在 PyPTO 仓库根目录执行。

## 固定步骤

### 阶段一：环境准备

#### 步骤 1：用户指定算子脚本

用户需要指定要执行的算子脚本（用于触发 Pass 编译流程）：
- 脚本路径
- 脚本参数（如有）

#### 步骤 2：设置日志环境

```bash
# 开启 info 级别日志
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志落盘路径（默认为算子脚本同目录下的 logs 文件夹）
export ASCEND_PROCESS_LOG_PATH=$(dirname {user_specified_script})/logs

# 创建日志目录
mkdir -p $ASCEND_PROCESS_LOG_PATH

# 日志文件将落盘到：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-{pid}-{timestamp}.log
# 注意：单个日志文件超过 20M 会自动拆分为多个文件
# 拆分文件命名：每个文件都有独立的时间戳（pypto-log-{pid}-{timestamp}.log）
# 同一次执行的所有拆分文件具有相同的 pid，可通过 pid 识别
```

### 阶段二：性能测量

#### 步骤 3：编译 Release 版本并安装

**⚠️ 重要：确保使用最新编译的 Release 版本进行性能测试**

```bash
# 编译 Python 包（Release 版本，用于性能测试）
python3 build_ci.py -f=python3 --build_type Release --disable_auto_execute

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall

```

#### 步骤 4：运行算子脚本采集 Pass 耗时

```bash
# 执行用户指定的算子脚本（日志自动落盘）
# 使用超时控制，默认 5 分钟，超时后自动中断并继续后续步骤
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 300 python3 {user_specified_script}.py

# 自定义超时时间（例如 10 分钟）
# bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 600 python3 {user_specified_script}.py

# 不使用超时控制直接执行（不推荐，可能长时间等待）
# python3 {user_specified_script}.py

# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：单个日志文件超过 20M 会自动拆分为多个文件
# 同一次执行的所有文件具有相同的 pid
```

#### 步骤 5：运行 Python 脚本分析性能

```bash
# 解析日志文件，生成 Pass 耗时排序报告
# 日志文件位于：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：脚本会自动处理拆分的日志文件（同 pid 的所有文件）

# 查找最新的日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "使用日志文件: $latest_log"

python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l $latest_log

```

#### 步骤 6：编译 Debug 版本

**⚠️ 重要：perf 采样需要 Debug 版本才能正确显示函数名和调用栈**

```bash
# 编译 Python 包（Debug 版本带完整调试符号，用于 perf 采样）
python3 build_ci.py -f=python3 --build_type Debug

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall
```

#### 步骤 7：使用 perf 分析热点

**⚠️ 重要：perf 能够正确分析当前代码的性能优化点**

##### 选项 A：火焰图方式（推荐）

火焰图提供了直观的可视化视图，更容易识别性能热点。

**生成火焰图：**

```bash
# 生成火焰图（默认 5 分钟超时）
# 参数：超时时间(秒) 输出目录 命令...
bash "$PASS_PERF_SCRIPTS_DIR/generate_flamegraph.sh" \
    300 ./flamegraphs python3 {user_specified_script}.py

# 火焰图将保存到 ./flamegraphs/flamegraph_{timestamp}.svg
# 同时生成折叠数据文件 folded_{timestamp}.txt（用于后续对比）

# 使用浏览器打开查看
firefox ./flamegraphs/flamegraph_*.svg
# 或
google-chrome ./flamegraphs/flamegraph_*.svg
```

**🔥 火焰图解读指南：**

**1. 结构说明**
- **X轴**：表示函数调用栈的样本占比（宽度越大，占用 CPU 越多）
- **Y轴**：表示调用栈深度（从下往上，底部是入口，顶部是叶子函数）
- **颜色**：随机分配，仅用于区分不同函数

**2. 识别热点函数**

| 图形特征 | 含义 | 行动建议 |
|---------|------|---------|
| **宽平台** | 顶部宽大的函数块 | 这是性能热点，优先优化 |
| **高塔尖** | 调用栈很深 | 检查是否存在过度封装或递归 |
| **反复出现** | 同一函数多处出现 | 函数被频繁调用，考虑缓存或批处理 |
| **细长条** | 调用栈很窄但很深 | 可能是单线程瓶颈 |

**3. 交互操作**（浏览器中）
- **鼠标悬停**：显示函数名和样本占比百分比
- **点击函数**：放大查看该函数的调用栈细节
- **搜索功能**：按 `Ctrl+F` 搜索特定函数名
- **重置视图**：点击空白处或刷新页面

**4. 常见热点模式及优化方向**

| 热点函数 | 问题类型 | 优化方向 |
|---------|---------|---------|
| `_malloc` / `_free` | 频繁内存分配 | 预分配内存、使用对象池 |
| `std::unordered_map::find` | 哈希表查找 | 优化哈希函数、预分配bucket |
| `std::vector` 扩容 | 动态扩容 | 使用 `reserve()` 预分配 |
| `memcpy` / `memmove` | 大量数据拷贝 | 减少拷贝、使用引用 |
| 字符串操作 | 字符串拼接/转换 | 使用 `std::string_view`、预分配 |
| 循环内函数调用 | 过度循环 | 循环展开、提前计算 |

**5. 火焰图示例解读**

```
                    ┌─────────────────────┐
                    │  hot_function() 15% │  ← 宽平台：性能热点
                    └─────────────────────┘
               ┌────────────────────────────────┐
               │     process_data() 35%         │  ← 宽块：主要耗时函数
               └────────────────────────────────┘
          ┌──────────────────────────────────────────┐
          │          main_loop() 50%                 │  ← 入口函数
          └──────────────────────────────────────────┘
```

- `hot_function()` 占 15% CPU，宽度较宽，是优化重点
- `process_data()` 占 35%，是主要耗时函数
- 点击 `hot_function()` 可以查看其内部调用栈

##### 选项 B：传统 perf report

如果火焰图工具不可用，可以使用传统方式：

```bash
# 使用 perf 采样
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 300 \
    perf record -g -e cycles,instructions,cache-misses -- python3 {user_specified_script}.py

# 查看报告
perf report
```

**perf report 快捷键：**
- `+`：展开调用栈
- `-`：折叠调用栈
- `Enter`：进入函数详情
- `/`：搜索函数名

##### 分析输出要求

完成火焰图分析后，记录以下信息：

```markdown
## 性能分析记录

### 热点函数 Top 5（从火焰图识别）
| 排名 | 函数名 | CPU占比 | 所属模块 | 可能原因 |
|-----|--------|---------|---------|---------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### 性能瓶颈类型
- [ ] 计算密集型
- [ ] 内存密集型
- [ ] IO密集型

### 优化方向
1. ...
2. ...
```

#### 步骤 8：内存分析（可选）

```bash
# 使用超时控制执行内存分析，默认 5 分钟
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 300 \
    valgrind --tool=massif -- python3 {user_specified_script}.py

massif-visualizer massif.out.*
```

### 阶段三：理解 Pass 功能

**⚠️ 重要：优化前必须充分理解 Pass 的功能和业务逻辑**

#### 步骤 9：使用 pypto-pass-module-analyzer 分析 Pass

调用 `pypto-pass-module-analyzer` skill 来深入理解目标 Pass：

```
分析 Pass 模块 {PassName} 的功能和业务逻辑
```

**需要了解的内容：**
- Pass 的整体功能是什么
- Pass 的主要处理流程
- Pass 的核心数据结构
- Pass 的关键算法
- Pass 的输入输出

**理解 Pass 功能的好处：**
- 知道哪些数据结构可以优化
- 知道哪些算法可以改进
- 知道哪些步骤是性能瓶颈
- 避免优化破坏功能正确性

### 阶段四：性能分析

#### 步骤 10：分析热点函数

从 perf report 中识别：
- CPU 密集函数（cycles 占比高）
- Cache miss 严重的函数
- 内存分配热点

#### 步骤 11：分析数据结构

检查以下问题：
- 容器选择是否合理（vector/set/unordered_map/map）
- 是否有频繁的内存分配/释放
- 是否有不必要的拷贝操作
- 是否有可以预分配的容器

#### 步骤 12：分析算法复杂度

- 评估时间复杂度（O(n)/O(n²)/O(n³)/O(n log n)）
- 检查是否有重复计算
- 分析循环嵌套层数
- 检查是否有可以提前终止的循环

#### 步骤 13：识别性能瓶颈类型

| 瓶颈类型 | 症状描述 | 典型原因 |
|---------|---------|---------|
| **计算密集型** | CPU 利用率高，cycles 占比高 | 算法复杂度过高、重复计算 |
| **内存密集型** | cache miss 多，内存访问频繁 | 数据结构不合理、缓存不友好 |
| **IO 密集型** | 文件读写、日志输出多 | 过多的日志、文件操作 |

### 阶段五：优化实施

#### 步骤 14：修改代码

根据性能分析结果修改 Pass 代码，实施优化。

#### 步骤 15：添加关键步骤耗时统计

在优化过程中，为 Pass 内部关键步骤添加耗时统计：

```cpp
#include <chrono>

// 在 Pass 的 RunOnFunction 或关键函数中添加
auto stepStart = std::chrono::high_resolution_clock::now();

// ... 关键步骤代码 ...

auto stepEnd = std::chrono::high_resolution_clock::now();
auto stepDuration = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
PASS_LOGI("[{PassName}] Step {step_name} cost %ld us", stepDuration.count());
```

**建议添加耗时统计的场景：**
- 主要循环遍历
- 图遍历/搜索
- 数据结构构建
- 排序/查找操作
- 内存密集操作

#### 步骤 16：算法优化

**常见优化方法：**
- 降低时间复杂度
- 减少不必要的遍历
- 利用索引/哈希加速查找
- 提前终止减少循环次数

**优化示例：**
```cpp
// 优化前：O(n²) 查找
for (auto& op1 : operations) {
    for (auto& op2 : operations) {
        if (op1.id == op2.parentId) { ... }
    }
}

// 优化后：O(n) 使用哈希表
std::unordered_map<int, Operation*> idToOp;
for (auto& op : operations) {
    idToOp[op.id] = &op;
}
for (auto& op : operations) {
    auto it = idToOp.find(op.parentId);
    if (it != idToOp.end()) { ... }
}
```

#### 步骤 17：数据结构优化

**容器选择指南：**
| 场景 | 推荐容器 | 原因 |
|-----|---------|-----|
| 顺序访问 | `std::vector` | 连续内存，缓存友好 |
| 频繁查找 | `std::unordered_map` | O(1) 查找 |
| 需要排序 | `std::set` / `std::map` | 自动排序 |
| 频繁插入删除 | `std::list` | O(1) 插入删除 |

**内存优化：**
```cpp
// 预分配内存
std::vector<Operation*> ops;
ops.reserve(expected_size);  // 避免多次重新分配

// 使用引用避免拷贝
for (const auto& op : operations) { ... }  // 而不是 for (auto op : ...)

// 使用 emplace_back 避免临时对象
ops.emplace_back(newOp);  // 而不是 ops.push_back(newOp)

// 使用 std::move 转移所有权
auto result = std::move(tempVector);
```

#### 步骤 18：缓存优化

```cpp
// 改善数据局部性
struct OpInfo {
    int id;
    int parentId;
    Operation* op;
};
std::vector<OpInfo> opInfos;  // 连续内存，缓存友好

// 避免"指针追逐"
// 优化前：多层指针
for (auto& op : ops) {
    for (auto& consumer : op->consumers) {
        for (auto& child : consumer->children) { ... }
    }
}

// 优化后：预计算索引
std::vector<std::vector<int>> opToChildren;  // 直接索引访问
```

### 阶段六：验证与迭代

#### 步骤 19：运行 UT 确保功能正确（优化后）

**⚠️ 重要：优化后必须验证目标 Pass 的 UT 通过**

```bash
# 查找 Pass 相关 UT 测试文件
# 路径：framework/tests/ut/passes/src/test_{pass_name}.cpp

# 运行指定 Pass 的所有 UT
python3 build_ci.py -f=cpp -u="{PassName}Test.*" -j=24

# 示例：运行 AssignMemoryType 的 UT
python3 build_ci.py -f=cpp -u="AssignMemoryTypeTest.*" -j=24

# 运行单个测试用例
python3 build_ci.py -f=cpp -u="AssignMemoryTypeTest.AddReshape" -j=24

# 如果 UT 失败：
# 1. 检查优化代码是否引入 bug
# 2. 必要时回退修改
# 3. 重新优化
```

**注意：UT 测试是 C++ 测试，必须使用 `-f=cpp` 参数**

#### 步骤 20：编译和安装（Release 版本）

**⚠️ 重要：最终性能验证必须使用 Release 版本，确保性能数据准确**

```bash
# 编译 Python 包（Release 版本，用于最终性能验证）
python3 build_ci.py -f=python3 --build_type Release

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall
```

#### 步骤 21：对比优化效果

**⚠️ 重要：优化后需要对比验证优化效果**

##### 21.1 重新采集日志数据

```bash
# 重新采集性能数据（日志自动落盘）
# 使用超时控制，默认 5 分钟
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 300 python3 {user_specified_script}.py

# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：每次运行会生成新的日志文件，可通过时间戳区分

# 查找最新的日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "最新日志文件: $latest_log"

# 运行 Python 脚本对比
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" \
    -l $latest_log
```

##### 21.2 生成优化后火焰图（可选，推荐）

```bash
# 生成优化后的火焰图和折叠数据
bash "$PASS_PERF_SCRIPTS_DIR/generate_flamegraph.sh" \
    300 ./flamegraphs python3 {user_specified_script}.py

# 这会生成新的文件，例如：
#   - folded_20260313_143022.txt (优化前，步骤7生成)
#   - folded_20260313_150335.txt (优化后，刚刚生成)
```

##### 21.3 对比火焰图验证优化效果

```bash
# 对比优化前后的火焰图
bash "$PASS_PERF_SCRIPTS_DIR/compare_flamegraphs.sh" \
    ./flamegraphs/folded_20260313_143022.txt \
    ./flamegraphs/folded_20260313_150335.txt

# 差异火焰图将保存到 ./flamegraphs/diff_flamegraph_{timestamp}.svg
```

**差异火焰图颜色说明：**

| 颜色 | 含义 | 解读 |
|------|------|------|
| 🔴 **红色/橙色** | 优化后增加的热点 | ⚠️ 需要关注，可能是新引入的性能问题 |
| 🔵 **蓝色/青色** | 优化后减少的热点 | ✅ 优化有效，这些函数耗时减少 |
| ⚪ **灰色** | 基本不变 | 优化对该函数影响不大 |

**对比分析要点：**
1. **关注红色区域**：优化后新出现或增加的热点
2. **确认蓝色区域**：验证优化是否对目标函数有效
3. **量化对比**：记录主要函数优化前后的占比变化

#### 步骤 22：迭代优化

如果性能未达标：

**⚠️ 重要：重新生成火焰图进行性能瓶颈分析，不要盲目优化**
重复步骤 6-21 直到达标

### 阶段七：优化总结报告

#### 步骤 23：生成优化总结报告

**⚠️ 重要：性能达标后必须生成优化总结报告，记录所有优化措施及其性能影响**

##### 23.1 收集优化数据

在生成报告前，收集以下数据：

```bash
# 1. 查看所有代码变更
git diff --stat HEAD~{n}  # n 为优化过程中的 commit 数量
git log --oneline HEAD~{n}..HEAD

# 2. 查找优化前后的日志文件
# 优化前日志（步骤5采集）
ls -lh $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log | head -5

# 优化后日志（步骤21采集）
ls -lh $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log | tail -5

# 3. 查找火焰图文件
ls -lh ./flamegraphs/*.svg ./flamegraphs/*.txt
```

##### 23.2 生成 Markdown 报告

基于模板文件生成优化总结报告，并保存到 `$ASCEND_PROCESS_LOG_PATH/perf_optimization_report.md`：

```bash
template_path="$PASS_PERF_SCRIPTS_DIR/../templates/perf_optimization_report.md"
report_path="$ASCEND_PROCESS_LOG_PATH/perf_optimization_report.md"
cp "$template_path" "$report_path"
```

模板文件：`templates/perf_optimization_report.md`

生成报告时需要替换模板中的占位符，并保留以下必填章节：

- `## 基本信息`
- `## 优化概览`
- `## 优化详情`
- `## 性能数据汇总`
- `## 经验总结`
- `## 附录`

##### 23.3 报告保存与验证

```bash
# 报告保存路径
report_path="$ASCEND_PROCESS_LOG_PATH/perf_optimization_report.md"

# 验证报告完整性（检查必填章节）
grep -q "## 基本信息" "$report_path" && \
grep -q "## 优化概览" "$report_path" && \
grep -q "## 优化详情" "$report_path" && \
grep -q "## 性能数据汇总" "$report_path" && \
grep -q "## 经验总结" "$report_path" && \
echo "✅ 报告结构完整" || echo "❌ 报告缺少必填章节"

# 查看报告
cat "$report_path"
```

##### 23.4 报告质量检查清单

生成报告后，逐项确认：

- [ ] **基本信息完整**：Pass 名称、Op 数量、目标耗时计算正确
- [ ] **优化概览准确**：优化前后数据来自实际测量，提升幅度计算正确
- [ ] **每项优化详情完整**：包含问题描述、优化方案、代码变更、性能影响
- [ ] **火焰图路径有效**：所有 SVG 路径指向实际存在的文件
- [ ] **perf 数据真实**：表格中的函数名和占比来自实际 perf report
- [ ] **性能汇总一致**：各优化项贡献之和 ≈ 总提升幅度
- [ ] **经验总结有价值**：可复用模式描述清晰，后续建议具体可行

## Python 脚本使用说明

### 脚本位置

```
$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py
```

### 使用方法

```bash
# 基本用法：分析单个日志文件
# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log

# 查找最新日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l $latest_log

# 或者直接指定日志文件路径
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l /path/to/pypto-log-*.log

# 对比两个日志文件
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l pypto-log-after.log --compare pypto-log-before.log

# 指定 Op 数量阈值（默认 200000）
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l pypto-log-*.log --ops-threshold 200000

# 指定时间阈值（默认 20s）
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l pypto-log-*.log --time-threshold 20
```

## 超时控制配置

### 为什么需要超时控制？

某些算子在运行阶段可能执行很长时间（超过 1 小时），但对于 Pass 编译性能分析而言：
- ✅ Pass 编译信息在编译完成后就已保存在日志中
- ✅ 运行阶段的长时间执行对 Pass 性能分析没有帮助
- ✅ 超时中断不会影响 Pass 性能分析结果

### 默认配置

- **默认超时时间**: 300 秒（5 分钟）
- **中断信号**: SIGINT (Ctrl+C)
- **中断后行为**: 继续执行后续分析步骤

### 如何自定义超时时间

```bash
# 设置超时时间为 10 分钟（600 秒）
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 600 python3 test.py

# 设置超时时间为 15 分钟（900 秒）
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 900 python3 test.py

# 设置超时时间为 2 分钟（120 秒）
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" 120 python3 test.py
```

### 适用场景

超时控制适用于所有需要执行算子的步骤：
1. ✅ 步骤 4：运行算子脚本采集 Pass 耗时
2. ✅ 步骤 7：使用 perf 分析热点
3. ✅ 步骤 8：内存分析
4. ✅ 步骤 21：对比优化效果

### 中断后继续执行的原因

当算子执行超时被中断后，脚本会返回退出码 0，允许继续执行后续步骤：
- ✅ Pass 编译日志已保存，可以进行分析
- ✅ 可以继续 perf 分析、内存分析等
- ✅ 可以对比优化前后的性能数据

如果算子执行因其他原因失败（非超时中断），脚本会返回非零退出码，停止后续执行。

### 超时控制脚本说明

**脚本位置**: `$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh`

**参数说明**:
```bash
bash "$PASS_PERF_SCRIPTS_DIR/run_with_timeout.sh" <超时秒数> <命令> [命令参数...]
```

**退出码**:
- `0` - 成功完成或超时中断（允许继续后续步骤）
- `非 0` - 执行失败（停止后续步骤）

**技术细节**:
- 使用 SIGINT (Ctrl+C) 信号，允许进程优雅退出
- 可以正确关闭文件、释放资源
- Python 程序可以捕获并处理该信号

## 功能正确性保障

### 优化后验证

**⚠️ 重要：优化后必须验证目标 Pass 的 UT 通过**

```bash
# 运行目标 Pass 的 UT
python3 build_ci.py -f=cpp -u=Test{PassName}.* -j=24

# 如果 UT 失败：
#    - 检查优化代码是否引入 bug
#    - 使用 git diff 查看修改
#    - 必要时回退修改
#    - 重新优化

# 只有 UT 全部通过才能接受优化
```

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| Pass 日志机制 | framework/src/passes/pass_mgr/pass_manager.cpp |
| 耗时统计函数 | LogPassRuntime() |
| Op 数量日志 | expand_function.cpp 中的 Operations().size() |
| 日志配置 | ASCEND_MODULE_LOG_LEVEL=PASS=1 |
| perf 使用 | perf record -g, perf report |
| valgrind 使用 | valgrind --tool=massif |

## 常用命令

### 查找最新的日志文件

```bash
# 方法 1：使用 find 命令查找最新日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "最新日志文件: $latest_log"

# 方法 2：使用 ls 按时间排序查找
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -t | head -1)
echo "最新日志文件: $latest_log"

# 直接使用最新日志文件进行分析
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l $latest_log
```

### 处理拆分的日志文件

```bash
# 日志文件拆分规则：单个文件超过 20M 自动拆分
# 拆分文件命名：
#   - pypto-log-{pid}-{timestamp1}.log
#   - pypto-log-{pid}-{timestamp2}.log
#   - pypto-log-{pid}-{timestamp3}.log
#   - ...
# 同一次执行的所有文件具有相同的 pid

# 查看同一次执行产生的所有拆分日志文件
# 方法 1：手动提取 pid 查找
log_file="pypto-log-1051473-20260312202107387.log"
pid=$(echo $log_file | sed 's/pypto-log-\([0-9]*\)-.*/\1/')
ls -lh pypto-log-${pid}-*.log

# 方法 2：自动分析（推荐）
# parse_pass_perf.py 会自动检测并处理所有拆分文件
python3 "$PASS_PERF_SCRIPTS_DIR/parse_pass_perf.py" -l pypto-log-1051473-20260312202107387.log
```

### 查看所有日志文件

```bash
# 查看所有日志文件（按时间排序）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -lht

# 查看所有日志文件（按大小排序）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -exec ls -lh {} \; | sort -k5 -h

# 列出所有拆分的日志文件组（按 pid 分组）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -t | head -20
```

### 查找特定进程或时间的日志

```bash
# 查找特定进程ID的所有日志文件
pid="1051473"
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-${pid}-*.log" -type f

# 查找特定时间段的所有日志文件（2026年3月12日 20:21）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*-202603122021*.log" -type f

# 查找特定日期的日志文件（2026年3月12日）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*-20260312*.log" -type f

# 统计每次执行产生的日志文件数量
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | \
    sed 's/pypto-log-\([0-9]*\)-.*/\1/' | sort | uniq -c
```

## 历史优化案例学习

以下是已完成的 Pass 编译性能优化案例，可作为优化参考。

### 案例1：提前终止遍历 + 避免拷贝 (DynAttrToStatic)

**Commit**: 6002c12d

**优化方法**:
- 引用语义消除冗余拷贝：`auto&` 替代 `auto`
- 短路求值优化：条件不满足时立即 `break` 终止遍历

### 案例2：预计算索引 (ReplaceTensor)

**Commit**: 1303180c

**优化方法**:
- 空间换时间：预构建 `tensorToOrderIndex` 索引映射，将 O(n²) 降为 O(1) 查找

### 案例3：缓存 + 数据结构简化 (SplitReshape)

**Commit**: 0f653952

**优化方法**:
- 记忆化搜索：添加 Cache 避免重复计算
- 扁平化数据结构：两层嵌套哈希扁平化为 `map<pair<K1,K2>, V>`

### 案例4：避免临时对象 (SplitLargeFanoutTensor)

**Commit**: 22f7d2aa

**优化方法**:
- 零拷贝设计：提取核心计算逻辑，直接传递 offset/shape 参数，消除临时对象构造开销

### 案例5：Copy-on-Write 思想 (OoOscheduler)

**Commit**: 18046c2b

**优化方法**:
- 写时复制策略：`shared_ptr` 实现数据共享，仅在修改时触发深拷贝，降低内存与时间开销

---

**通用优化模式总结**:
| 模式 | 适用场景 | 示例案例 |
|-----|---------|---------|
| 提前终止 | 条件检查类遍历 | 案例1 |
| 避免拷贝 | 遍历容器时 | 案例1 |
| 预计算索引 | 重复遍历同一数据 | 案例2 |
| 添加缓存 | 重复计算相同结果 | 案例3 |
| 数据结构简化 | 多层嵌套容器 | 案例3 |
| 避免临时对象 | 频繁创建销毁对象 | 案例4 |
| Copy-on-Write | 大数据频繁复制 | 案例5 |
