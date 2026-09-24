# PTO 前端开发者文档

> 说明：本中文文档为英文版 `developer_doc.md` 的翻译版本；若中英文内容存在出入或歧义，请**以英文版为准**。

## 概览

PTO 前端负责解析使用 `@pypto.frontend.jit` 或 `@pypto.frontend.function` 装饰的 Python 函数，并将其转换为 PTO 中间表示（IR）。前端为编写高性能张量计算内核提供了一个高级、Python 原生的接口，同时允许用户对底层执行细节保持完全控制。

本文档为参与 PTO 前端解析器开发、维护或扩展的开发者提供一份全面指南。

## 架构

### 解析流水线

前端采用从源代码到可执行 IR 的多阶段流水线：

**阶段 1：源代码提取** —— `Source` 类使用 Python 的 `inspect` 模块从被装饰的函数中提取源代码。它处理包含 Jupyter notebook、嵌套类等多种边界情况，并保留源位置（source location）信息以支持错误报告。

**阶段 2：Python AST 解析** —— 使用内置的 `ast.parse()` 将源代码解析为 Python 标准抽象语法树（AST）。

**阶段 3：Doc AST 转换** —— 将 Python AST 节点转换为 “doc AST” 节点。doc AST 提供了一个独立于 Python 版本变化的稳定接口。这一抽象层确保跨 Python 版本兼容（最低 3.9+）。

**阶段 4：活性分析（Liveness Analysis）** —— `LivenessAnalyzer` 遍历 doc AST，确定变量的最后一次使用位置，从而通过插入删除点来实现自动内存管理。

**阶段 5：解析与 IR 生成** —— `Parser` 类使用访问者（visitor）模式遍历 doc AST 并生成 PTO IR。在此阶段，解析器通过 `Context` 类维护变量作用域，通过 `ExprEvaluator` 计算表达式，并通过 `Diagnostics` 系统报告错误。

**阶段 6：惰性执行（Lazy Execution）** —— `JitCallableWrapper` 将实际解析延迟到函数第一次调用时执行，以便在编译前进行动态形状绑定与代价模型评估。

## 模块结构与职责

### 核心模块

#### `__init__.py`

对外导出公共 API 的主要入口：
- `jit`：用于 JIT 编译的装饰器
- `function`：用于内联函数展开的装饰器
- `dynamic()`：用于创建符号维度（symbolic dimensions）

#### `parser/entry.py`

提供 JIT 编译的入口函数与包装器类：
- `parse()`：主解析函数，负责编排整体流水线
- `jit()`：JIT 编译的装饰器工厂（带选项）
- `function`：用于嵌套函数内联展开的装饰器
- `JitCallableWrapper`：使已解析函数可用 torch 张量调用
- 张量数据转换与形状匹配的辅助函数

关键职责：
- 将前端解析器与运行时执行桥接起来
- 管理惰性解析与缓存
- 处理动态维度绑定
- 集成代价模型评估
- 在 torch 张量与 PTO 张量数据之间进行转换

#### `parser/parser.py`

前端的核心模块，包含主 `Parser` 类。该类实现访问者模式以遍历 doc AST 并生成 PTO IR。该模块规模最大（约 1558 行）。

关键职责：
- 使用访问者模式遍历 AST
- 通过 Context 进行变量作用域管理
- 通过 ExprEvaluator 进行表达式求值
- 解析语句与表达式
- 处理控制流（if/else、for 循环）
- 提取函数签名（输入/输出）
- 基于活性分析自动插入变量删除点
- 内联展开嵌套函数

解析器维护的关键状态包括：
- `diag`：Diagnostics 实例，用于错误报告
- `context`：Context 实例，用于变量作用域管理
- `delete_after`：从语句 ID 映射到需删除变量的表
- `_signature_cache`：缓存的函数签名（输入/输出）

#### `parser/context.py`

使用基于栈（stack）的方式管理解析期间的变量作用域与生命周期。

关键类：
- `ContextFrame`：表示单个作用域（代码块或函数体）
- `Context`：由多个 frame 组成的栈，维护变量名到值的映射

关键职责：
- 跟踪变量定义与遮蔽（shadowing）
- 提供上下文管理器，用于自动作用域清理
- 支持在作用域内更新变量
- 管理来自活性分析的“标记为删除”变量

Context 系统使解析器能够在嵌套代码块（if、循环、函数体）之间维持正确的变量作用域，同时支持 Python 的遮蔽语义。

#### `parser/diagnostics.py`

完整的错误报告系统，提供丰富且用户友好的错误信息。

关键类：
- `Source`：源代码表示与位置跟踪
- `Span`：源位置范围（文件、行列范围）
- `DiagnosticLevel`：错误级别（BUG、ERROR、WARNING、INFO、DEBUG）
- `DiagnosticItem`：带位置信息的单条诊断消息
- `Diagnostics`：诊断管理器
- `DiagnosticContext`：用于收集诊断信息的上下文

关键职责：
- 带源代码上下文的美观错误输出（含颜色）
- 在错误位置附近显示多行上下文
- 支持不同诊断级别
- 处理特殊情况（Jupyter notebooks、嵌套类）
- 为所有 AST 节点提供位置信息

错误信息包含：
- 颜色区分的严重级别
- 文件名与行/列号
- 源代码上下文（前 2 行、后 4 行）
- 指向错误位置的插入符（^）

#### `parser/error.py`

解析器错误的异常类，并提供可配置的回溯（backtrace）控制。

关键类：
- `ParserError`：基础异常，将错误与 AST 节点关联
- `RenderedParserError`：表示错误已格式化并显示

关键特性：
- 环境变量 `PTO_BACKTRACE` 控制是否展示完整回溯
- 默认行为（`PTO_BACKTRACE=0`）只显示用户友好的错误信息
- 设置 `PTO_BACKTRACE=1` 开启完整 Python 栈追踪，便于调试
- 出错时自动清理多进程子进程

#### `parser/evaluator.py`

在解析期间对 Python 表达式求值，以解析出具体值。

关键类：
- `ExprEvaluator`：使用 Python 的 `compile()` 与 `eval()` 的表达式求值器

关键职责：
- 计算类型注解（例如 `pypto.Tensor((N, M), pypto.DT_FP32)`）
- 解析常量表达式
- 尽可能将 SymbolicScalar 转换为具体值
- 在求值失败时提供正确的错误上下文

求值器基于解析器的变量表运行，因此能够访问用户定义变量以及 `pypto` 模块命名空间。

#### `parser/liveness.py`

实现用于自动内存管理的活性分析。

关键类：
- `LivenessAnalyzer`：用于跟踪变量使用与定义的 AST visitor

关键职责：
- 记录每个变量的所有使用点
- 找出每个变量的最后使用位置以便自动删除
- 处理“循环感知”的分析（变量在循环中的使用）
- 尊重被豁免变量（函数参数、显式删除）
- 生成删除点映射（语句 ID → 变量名）

该分析通过自动删除不再需要的变量来降低内存使用，尤其适用于大型张量运算，而不要求用户手写 `del`。

处理的特殊情况：
- 循环变量不参与自动删除
- 在循环中使用的变量在循环结束后删除（而非每次迭代）
- 在循环内定义的变量可按迭代删除
- 显式 `del` 会将变量标记为不参与自动删除

#### `parser/doc.py`

用于 Python AST 与 doc AST 之间双向转换的注册系统。

关键类：
- `Entry`：存储转换函数的映射条目
- `Registry`：全局注册表，记录所有 AST 节点转换

关键函数：
- `parse()`：将源字符串解析为 doc AST
- `to_doc()`：将 Python AST 节点转换为 doc AST 节点
- `from_doc()`：将 doc AST 节点转换回 Python AST 节点
- `register()`：为新节点类型注册转换函数

关键职责：
- 维护双向 AST 转换映射
- 提供可扩展的注册机制
- 支持 visitor/transformer 模式
- 处理节点类型解析

#### `parser/doc_core.py`

定义核心 AST 节点类，镜像 Python 标准 AST。

关键基类：
- `AST`：所有 doc AST 节点的基类
- `NodeVisitor`：AST 遍历基类
- `NodeTransformer`：AST 变换基类

节点类别：
- `mod`：模块级节点（Module、Interactive、Expression）
- `stmt`：语句节点（FunctionDef、Assign、For、If、Return 等）
- `expr`：表达式节点（BinOp、Call、Name、Constant 等）
- `operator`：算术运算符（Add、Sub、Mult、Div 等）
- `boolop`：布尔运算符（And、Or）
- `unaryop`：一元运算符（Not、UAdd、USub、Invert）
- `cmpop`：比较运算符（Eq、Lt、Gt 等）

该抽象层通过提供稳定接口，确保解析器可跨 Python 版本一致工作，从而不受 Python 标准 AST 变更的影响。

## 关键概念

### 带源位置的诊断系统

每个 AST 节点都携带源位置（行号、列偏移、结束行、结束列）。当发生错误时，`Diagnostics` 系统使用这些信息来：

1. 读取原始源文件
2. 提取上下文行（前 2 行、后 4 行）
3. 用颜色高亮错误行
4. 用插入符（^）指向精确错误位置
5. 输出带严重级别的错误信息

这提供了类似 Rust 或 Clang 等现代编译器的使用体验。

### 用于内存管理的活性分析

`LivenessAnalyzer` 在解析前对 AST 执行静态分析，计算最优删除点：

1. **跟踪定义**：记录每个变量的定义位置
2. **跟踪使用**：按顺序记录每个变量的所有使用点
3. **计算最后使用**：确定每个变量最后一次被使用的语句
4. **生成删除点**：创建映射（语句 ID → 需要删除的变量名集合）

在解析阶段，每执行完一条语句后，解析器会检查是否有变量需要被删除，并将它们标记为待清理。该优化对大型张量操作尤为关键，可显著降低内存占用。

处理的特殊情况：
- 循环变量不参与自动删除
- 在循环中使用的变量在循环结束后删除（而非每次迭代）
- 在循环内定义的变量可按迭代删除
- 显式 `del` 语句会将变量标记为不参与自动删除

### Doc AST 抽象层

doc AST 提供稳定接口，将解析器与 Python AST 的变更隔离开来。其优势包括：

**版本无关性**：Python AST 在不同版本之间可能变化（例如 Python 3.8 增加了位置信息）。doc AST 提供一致接口。

**简化接口**：移除 PTO 脚本不支持的特性（例如 type comments、async/await）。

**可扩展性**：更易于添加自定义 AST 节点或 PTO 特有属性。

**双向转换**：注册系统支持 Python AST 与 doc AST 的双向转换，可用于调试与 AST 操作。

## 使用示例

### 使用 JIT 装饰器的基础内核

```python
import pypto

N = 1024
M = 1024

@pypto.frontend.jit()
def basic_addsub(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    c: pypto.Tensor((N, M), pypto.DT_FP32),
    d: pypto.Tensor((N, M), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)

    c[:] = pypto.add(a, b)
    d[:] = pypto.sub(a, b)
```

**解析器会做什么：**
1. 提取函数签名以确定输入与输出
2. 将函数体解析为 doc AST
3. 为 `c` 与 `d` 创建 PTO 张量 IR 节点
4. 将 `pypto.add(a, b)` 与 `pypto.sub(a, b)` 转换为 PTO 操作 IR
5. 将切片赋值 `c[:] = ...` 处理为张量填充（fill）操作

### 动态维度

```python
import pypto

N = pypto.DYNAMIC  # 符号维度
M = 1024

@pypto.frontend.jit()
def basic_dynamic(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    out: pypto.Tensor((N, M), pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(32, 32)

    # 此处 N 为符号量，在运行时解析
    for bs_idx in pypto.loop(32):
        tile_a = pypto.view(a, (32, 1024), [bs_idx * 32, 0])
        tile_b = pypto.view(b, (32, 1024), [bs_idx * 32, 0])
        out[bs_idx * 32: (bs_idx + 1) * 32, :] = pypto.add(tile_a, tile_b)


```

**解析器会做什么：**
1. 将 `N` 识别为 SymbolicScalar（由 `pypto.DYNAMIC` 标记）
2. 解析期间保持 `N` 为符号量
3. 第一次执行时，将 `N` 绑定为实际输入形状
4. 校验 `N` 的所有使用保持一致
5. 生成含符号维度的 IR，并在运行时解析该符号维度

### 循环结构

```python
@pypto.frontend.jit()
def basic_loop(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    out: pypto.Tensor((N, M), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)

    for _ in pypto.loop(10):
        a[:] = pypto.add(a, b)
        out[:] = a
```

**解析器会做什么：**
1. 将 `pypto.loop(10)` 识别为特殊循环结构
2. 生成迭代次数为 10 的 PTO loop IR
3. 将循环体作为嵌套作用域处理
4. 循环变量 `_` 被标记为不参与自动删除
5. 在循环中使用的变量（a、b）不会被删除，直到循环结束

### 嵌套函数内联

```python
@pypto.frontend.function
def inner_add(
    x: pypto.Tensor((8,), pypto.DT_FP32),
    bias: pypto.Tensor((8,), pypto.DT_FP32),
    out: pypto.Tensor((8,), pypto.DT_FP32),
):
    out[:] = pypto.add(x, bias)


@pypto.frontend.jit
def outer_kernel(
    a: pypto.Tensor((8,), pypto.DT_FP32),
    b: pypto.Tensor((8,), pypto.DT_FP32),
    out: pypto.Tensor((8,), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 16, 32)
    inner_add(a, b, out)  # 解析期间被内联
```

**解析器会做什么：**
1. 用 `NestedFunctionMarker` 标记 `inner_add`（不会单独编译）
2. 解析 `outer_kernel` 时识别对 `inner_add` 的调用
3. 将 `inner_add` 的函数体内联到 `outer_kernel`
4. 处理参数映射（x→a，bias→b）
5. 生成一个包含内联代码的单一 PTO 函数

### 控制流

```python
@pypto.frontend.jit()
def conditional_kernel(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    out: pypto.Tensor((N, M), pypto.DT_FP32),
    flag: bool,
):

    if flag:
        out[:] = pypto.add(a, b)
    else:
        out[:] = pypto.sub(a, b)
```

**解析器会做什么：**
1. 若条件 `flag` 为常量，则在解析期对其求值
2. 如果条件已知，可通过仅生成被选分支的 IR 来进行优化
3. 如果条件由运行时决定，则生成条件分支 IR
4. 正确处理变量作用域（`result` 在两个分支中都可见）

## 最佳实践

### 代码组织

**在 parser.py 中将相关 visitor 方法分组**：
- 语句 visitor（visit_assign、visit_for、visit_if 等）
- 表达式 visitor（visit_call、visit_bin_op、visit_name 等）
- 辅助方法（以下划线 `_` 开头的私有方法）

**保持 visitor 方法聚焦**：每个方法只处理一种节点类型，复杂逻辑委托给辅助函数。

**使用更具描述性的变量名**：优先使用 `input_tensors` 而非 `inputs`，使用 `loop_iterator` 而非 `i`。

### 错误处理

**提供可操作的错误信息：**
- 不佳：“Invalid syntax”
- 更好：“Expected tensor type annotation, got 'list'. Use pypto.Tensor((shape,), dtype) instead.”

**在错误信息中包含上下文：**
- 说明期望内容与实际内容
- 尽可能给出修复建议
- 对复杂特性引用文档

**使用合适的诊断级别：**
- ERROR：会阻止编译的用户错误
- WARNING：可能有问题但仍可编译的代码
- INFO：调试时的提示信息
- BUG：不应该发生的解析器内部错误

### 类型注解

**为公共 API 完整添加注解：**
```python
def parse(program: Source, extra_vars: Optional[dict[str, Any]] = None) -> Any:
    """Parse a PTO script program."""
    pass
```

**为类属性添加注解：**
```python
class Parser:
    diag: Diagnostics
    context: Context
    delete_after: dict[int, set[str]]
```

**正确使用 Optional 与 Union：**
```python
def visit_node(self, node: Optional[doc.AST]) -> Union[str, int, None]:
    pass
```

### 文档

**为所有公共 API 编写 NumPy 风格 docstring。**

**对不直观的逻辑写明“为什么”而不是“做什么”的注释：**
```python
# We need to pop the frame before adding the variable to avoid
# shadowing issues when the variable name conflicts with a loop variable
self.context.frames.pop()
self.context.add(var_name, value)
```

**保持文档更新**：当修改代码时，同步更新对应 docstring 与注释。

## 总结

PTO 前端提供了强大且可扩展的系统，用于将 Python 代码解析为高性能张量计算 IR。理解本文档中的架构、关键概念与扩展机制后，开发者将能更高效地维护、调试并扩展前端解析器。

如有问题，请参考源码注释、测试用例，或联系 PTO 开发团队。
