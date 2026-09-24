---
name: pypto-host-stacktrace-analyzer
description: 分析 host 侧捕获异常后的堆栈信息，通过地址到源码行映射和符号解析，定位问题代码位置。支持 Python traceback、C++ stack trace 和混合堆栈的自动识别与分析。支持编译Debug版本PyPTO包并定位具体代码行。Triggers："堆栈分析"、"堆栈反汇编"、"分析堆栈信息"、"地址到源码行"、"stack trace"、"backtrace"
---

# 堆栈信息分析器

此技能用于分析 host 侧捕获异常后的堆栈信息，通过反汇编和符号解析，定位问题代码位置。

## 工作流程

1. 收集测试用例文件路径
2. 编译并安装Debug版本PyPTO
3. 重新运行测试用例并收集新堆栈信息
4. 使用综合分析器自动分析
5. 输出完整报告

---

## 步骤 1：收集测试用例文件路径

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **test_case_file**: 测试用例文件路径（必填）

将收集的路径转换成绝对路径，收集到信息后才能继续后续步骤。

---

## 步骤 2：编译并安装Debug版本PyPTO

编译Debug版本：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py
```

**编译完成后自动安装**：
- 找到编译生成的wheel文件
- 使用 `pip install build_out/pypto*.whl --force-reinstall --no-deps` 安装
- 显示编译和安装信息

---

## 步骤 3：重新运行测试用例并收集新堆栈信息

重新运行步骤1中提供的测试用例以获取新的堆栈信息：

```bash
# 使用步骤1中收集的测试用例文件路径
python3 <test_case_file> 2>&1 | tee /tmp/new_stack_trace.log
```

**然后从新输出的日志中提取堆栈信息**：

```bash
# 提取 C++ stack trace
grep -A 50 "libtile_fwk_interface.so" /tmp/new_stack_trace.log > /tmp/new_cpp_stack.txt

# 或提取 Python traceback
grep -A 100 "Traceback" /tmp/new_stack_trace.log > /tmp/new_python_stack.txt

# 或提取完整堆栈（包含错误信息）
grep -A 100 "Run pass failed" /tmp/new_stack_trace.log > /tmp/new_stack_trace.txt
```

**⚠️ 注意事项**：
- 将新堆栈信息保存到文件中，供后续分析使用
- 确保新堆栈信息包含完整的错误信息

---

## 步骤 4：使用综合分析器自动分析

运行综合分析脚本，自动完成所有分析步骤：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py <stack_file> -f
```
其中stack_file为步骤3中提取并保存的堆栈信息文件
---

## 步骤 5：输出完整报告

综合分析器会自动生成包含以下内容的完整报告：

### 错误信息
- 错误码
- 错误位置
- 错误消息

### Python Traceback
- 总帧数
- 每帧的详细信息（文件、行号、函数）
- 错误触发点标记

### C++ Stack Trace
- 总帧数
- 每帧的详细信息（二进制、符号、偏移、地址）
- 符号反混淆结果
- **源码行号**
- 错误发生点标记

### 二进制文件
- 找到的二进制文件路径

---
