# PTO Frontend Developer Documentation

## Overview

The PTO Frontend is responsible for parsing Python functions decorated with `@pypto.frontend.jit` or `@pypto.frontend.function` and converting them into PTO intermediate representation (IR). The frontend provides a high-level, Python-native interface for writing optimized tensor computation kernels while maintaining full control over low-level execution details.

This document provides a comprehensive guide for developers working on or extending the PTO frontend parser.

## Architecture

### Parsing Pipeline

The frontend follows a multi-stage pipeline from source code to executable IR:

**Stage 1: Source Extraction** - The `Source` class extracts the source code from a decorated function using Python's `inspect` module. It handles various edge cases including Jupyter notebooks, nested classes, and maintains source location information for error reporting.

**Stage 2: Python AST Parsing** - The source code is parsed into Python's standard Abstract Syntax Tree (AST) using the built-in `ast.parse()` function.

**Stage 3: Doc AST Conversion** - Python AST nodes are converted to "doc AST" nodes, which provide a stable interface independent of Python version changes. This abstraction layer ensures compatibility across Python versions (minimum 3.9+).

**Stage 4: Liveness Analysis** - The `LivenessAnalyzer` walks the doc AST to determine when variables are last used, enabling automatic memory management by inserting deletion points.

**Stage 5: Parsing & IR Generation** - The `Parser` class uses the visitor pattern to traverse the doc AST and generate PTO IR constructs. During this phase, the parser maintains variable scopes using the `Context` class, evaluates expressions using the `ExprEvaluator`, and reports errors using the `Diagnostics` system.

**Stage 6: Lazy Execution** - The `JitCallableWrapper` defers actual parsing until the first function call, allowing for dynamic shape binding and cost model evaluation before compilation.

## Module Structure and Responsibilities

### Core Modules

#### `__init__.py`

The main entry point that exports the public API:
- `jit` decorator for JIT compilation
- `function` decorator for inline function expansion
- `dynamic()` function for creating symbolic dimensions

#### `parser/entry.py`

Provides the entry points and wrapper classes for JIT compilation:
- `parse()`: Main parsing function that orchestrates the pipeline
- `jit()`: Decorator factory for JIT compilation with options
- `function`: Decorator for nested function inline expansion
- `JitCallableWrapper`: Makes parsed functions callable with torch tensors
- Helper functions for tensor data conversion and shape matching

Key Responsibilities:
- Bridging the frontend parser with runtime execution
- Managing lazy parsing and caching
- Handling dynamic dimension binding
- Integrating cost model evaluation
- Converting between torch tensors and PTO tensor data

#### `parser/parser.py`

The heart of the frontend, containing the main `Parser` class that implements the visitor pattern to traverse the doc AST and generate PTO IR. This is the largest module at 1558 lines.

Key Responsibilities:
- AST traversal using visitor pattern
- Variable scope management via Context
- Expression evaluation via ExprEvaluator
- Statement and expression parsing
- Control flow handling (if/else, for loops)
- Function signature extraction
- Automatic variable deletion based on liveness analysis
- Nested function inlining

The parser maintains several important pieces of state:
- `diag`: Diagnostics instance for error reporting
- `context`: Context instance for variable scoping
- `delete_after`: Mapping from statement IDs to variables to delete
- `_signature_cache`: Cached function signature (inputs/outputs)

#### `parser/context.py`

Manages variable scoping and lifetime during parsing using a stack-based approach.

Key Classes:
- `ContextFrame`: Represents a single scope (block or function body)
- `Context`: Stack of frames with variable name-to-value mappings

Key Responsibilities:
- Tracking variable definitions and shadowing
- Providing context managers for automatic scope cleanup
- Supporting variable updates within scopes
- Managing marked-for-deletion variables from liveness analysis

The context system allows the parser to maintain proper variable scoping across nested blocks (if statements, loops, function bodies) while supporting Python's shadowing semantics.

#### `parser/diagnostics.py`

Comprehensive error reporting system with rich, user-friendly error messages.

Key Classes:
- `Source`: Source code representation with location tracking
- `Span`: Source location span (file, line, column ranges)
- `DiagnosticLevel`: Error severity levels (BUG, ERROR, WARNING, INFO, DEBUG)
- `DiagnosticItem`: Individual diagnostic message with location
- `Diagnostics`: Main diagnostics manager
- `DiagnosticContext`: Context for collecting diagnostics

Key Responsibilities:
- Pretty-printing errors with source context and color coding
- Showing multiple lines of context around error locations
- Supporting different diagnostic levels
- Handling special cases (Jupyter notebooks, nested classes)
- Providing location information for all AST nodes

Error messages include:
- Color-coded severity levels
- File name and line/column numbers
- Source code context (2 lines before, 4 lines after)
- Caret indicators (^) pointing to the error location

#### `parser/error.py`

Exception classes for parser errors with customizable backtrace control.

Key Classes:
- `ParserError`: Base exception associating errors with AST nodes
- `RenderedParserError`: Indicates error already formatted and displayed

Key Features:
- Environment variable `PTO_BACKTRACE` controls whether full backtraces are shown
- Default behavior (PTO_BACKTRACE=0) shows only user-friendly error messages
- Setting PTO_BACKTRACE=1 enables full Python stack traces for debugging
- Automatic cleanup of multiprocessing child processes on error

#### `parser/evaluator.py`

Evaluates Python expressions during parsing to resolve concrete values.

Key Class:
- `ExprEvaluator`: Expression evaluator using Python's `compile()` and `eval()`

Key Responsibilities:
- Evaluating type annotations (e.g., `pypto.Tensor((N, M), pypto.DT_FP32)`)
- Resolving constant expressions
- Converting SymbolicScalar to concrete values when possible
- Providing proper error context for evaluation failures

The evaluator operates on the parser's variable table, allowing access to both user-defined variables and the pypto module namespace.

#### `parser/liveness.py`

Implements liveness analysis for automatic memory management.

Key Class:
- `LivenessAnalyzer`: AST visitor that tracks variable uses and definitions

Key Responsibilities:
- Recording all uses of each variable
- Identifying last-use points for automatic deletion
- Handling loop-aware analysis (variables used in loops)
- Respecting exempt variables (function parameters, explicit deletes)
- Generating deletion point mapping (statement ID → variable names)

The analyzer helps reduce memory usage by automatically deleting variables that are no longer needed, without requiring manual `del` statements from users.

#### `parser/doc.py`

Registry system for bidirectional conversion between Python AST and doc AST.

Key Classes:
- `Entry`: Mapping entry storing conversion functions
- `Registry`: Global registry of all AST node conversions

Key Functions:
- `parse()`: Parse source string to doc AST
- `to_doc()`: Convert Python AST node to doc AST node
- `from_doc()`: Convert doc AST node to Python AST node
- `register()`: Register conversion functions for new node types

Key Responsibilities:
- Maintaining bidirectional AST conversion mappings
- Providing extensible registration system
- Supporting visitor and transformer patterns
- Handling node type resolution

#### `parser/doc_core.py`

Defines the core AST node classes that mirror Python's standard AST.

Key Base Classes:
- `AST`: Base class for all doc AST nodes
- `NodeVisitor`: Base class for AST traversal
- `NodeTransformer`: Base class for AST transformation

Node Categories:
- `mod`: Module-level nodes (Module, Interactive, Expression)
- `stmt`: Statement nodes (FunctionDef, Assign, For, If, Return, etc.)
- `expr`: Expression nodes (BinOp, Call, Name, Constant, etc.)
- `operator`: Arithmetic operators (Add, Sub, Mult, Div, etc.)
- `boolop`: Boolean operators (And, Or)
- `unaryop`: Unary operators (Not, UAdd, USub, Invert)
- `cmpop`: Comparison operators (Eq, Lt, Gt, etc.)

This abstraction layer ensures the parser works consistently across Python versions by providing a stable interface independent of changes to Python's standard AST.

## Key Concepts

### Diagnostic System with Source Locations

Every AST node carries source location information (line number, column offset, end line, end column). When an error occurs, the `Diagnostics` system uses this information to:

1. Read the original source file
2. Extract context lines (2 before, 4 after the error)
3. Highlight the error line with color
4. Add caret indicators (^) pointing to the exact error location
5. Print the error message with severity level

This provides a user experience similar to modern compilers like Rust or Clang.

### Liveness Analysis for Memory Management

The `LivenessAnalyzer` performs a static analysis pass over the AST before parsing to determine optimal deletion points:

1. **Track definitions**: Record where each variable is defined
2. **Track uses**: Record all uses of each variable in order
3. **Compute last use**: Identify the statement where each variable is last used
4. **Generate deletion points**: Create a mapping from statement IDs to variables to delete

During parsing, after executing each statement, the parser checks if any variables should be deleted and marks them for cleanup. This optimization is especially important for large tensor operations where memory usage is critical.

Special cases handled:
- Loop variables are exempt from auto-deletion
- Variables used in loops are deleted after the loop exits (not per-iteration)
- Variables defined inside loops can be deleted per-iteration
- Explicit `del` statements mark variables as exempt from auto-deletion

### Doc AST Abstraction Layer

The doc AST provides a stable interface that isolates the parser from Python AST changes. Benefits:

**Version Independence**: Python's AST can change between versions (e.g., Python 3.8 added position information). The doc AST provides a consistent interface.

**Simplified Interface**: Removes unnecessary Python features (e.g., type comments, async/await) that aren't supported in PTO scripts.

**Extensibility**: Makes it easier to add custom AST nodes or attributes specific to PTO.

**Bidirectional Conversion**: The registry system allows converting between Python AST and doc AST in both directions, useful for debugging and AST manipulation.

## Usage Examples

### Basic Kernel with JIT Decorator

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

**What the parser does:**
1. Extracts the function signature to determine inputs and outputs
2. Parses the function body into doc AST
3. Creates PTO tensor IR nodes for `c` and `d`
4. Converts `pypto.add(a, b)` and `pypto.sub(a, b)` into PTO operation IR
5. Handles the slice assignment `c[:] = ...` as tensor fill operations

### Dynamic Dimensions

```python
import pypto

N = pypto.DYNAMIC  # Symbolic dimension
M = 1024

@pypto.frontend.jit()
def basic_dynamic(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    out: pypto.Tensor((N, M), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)

    # N is symbolic here, resolved at runtime
    for bs_idx in pypto.loop(32):
        tile_a = pypto.view(a, (32, 1024), [bs_idx * 32, 0])
        tile_b = pypto.view(b, (32, 1024), [bs_idx * 32, 0])
        out[bs_idx * 32: (bs_idx + 1) * 32, :] = pypto.add(tile_a, tile_b)

```

**What the parser does:**
1. Recognizes `N` as a SymbolicScalar (marked by `pypto.DYNAMIC`)
2. Keeps `N` symbolic during parsing
3. During first execution, binds `N` to the actual input shape
4. Validates that all uses of `N` are consistent
5. Generates IR with symbolic dimension that gets resolved at runtime

### Loop Constructs

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

**What the parser does:**
1. Recognizes `pypto.loop(10)` as a special loop construct
2. Generates PTO loop IR with iteration count of 10
3. Handles loop body as a nested scope
4. The loop variable `_` is marked as exempt from auto-deletion
5. Variables used in loop (a, b) are not deleted until after loop exits

### Nested Function Inlining

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
    inner_add(a, b, out)  # Inlined during parsing

```

**What the parser does:**
1. Marks `inner_add` with `NestedFunctionMarker` (not compiled separately)
2. When parsing `outer_kernel`, recognizes call to `inner_add`
3. Inlines the body of `inner_add` into `outer_kernel`
4. Handles parameter mapping (x→a, bias→b)
5. Generates a single PTO function with the inlined code

### Control Flow

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

**What the parser does:**
1. Evaluates the condition `flag` at parse time if it's a constant
2. If the condition is known, can optimize by only generating IR for the taken branch
3. If the condition is runtime-determined, generates conditional IR
4. Handles variable scoping properly (out is visible in both branches)

## Best Practices

### Code Organization

**Group related visitor methods together** in parser.py:
- Statement visitors (visit_assign, visit_for, visit_if, etc.)
- Expression visitors (visit_call, visit_bin_op, visit_name, etc.)
- Helper methods (private methods with `_` prefix)

**Keep visitor methods focused** - each should handle one node type and delegate to helpers for complex logic.

**Use descriptive variable names** - prefer `input_tensors` over `inputs`, `loop_iterator` over `i`.

### Error Handling

**Provide actionable error messages:**
- Bad: "Invalid syntax"
- Good: "Expected tensor type annotation, got 'list'. Use pypto.Tensor((shape,), dtype) instead."

**Include context in errors:**
- Show what was expected vs. what was found
- Suggest fixes when possible
- Reference documentation for complex features

**Use appropriate diagnostic levels:**
- ERROR: For user errors that prevent compilation
- WARNING: For potentially problematic code that still compiles
- INFO: For informational messages during debugging
- BUG: For internal parser errors that shouldn't happen

### Type Annotations

**Fully annotate public APIs:**
```python
def parse(program: Source, extra_vars: Optional[dict[str, Any]] = None) -> Any:
    """Parse a PTO script program."""
    pass
```

**Annotate class attributes:**
```python
class Parser:
    diag: Diagnostics
    context: Context
    delete_after: dict[int, set[str]]
```

**Use proper Optional and Union types:**
```python
def visit_node(self, node: Optional[doc.AST]) -> Union[str, int, None]:
    pass
```

### Documentation

**Write docstrings for all public APIs** following NumPy style.

**Document non-obvious logic** with inline comments explaining the "why":
```python
# We need to pop the frame before adding the variable to avoid
# shadowing issues when the variable name conflicts with a loop variable
self.context.frames.pop()
self.context.add(var_name, value)
```

**Keep documentation up-to-date** - when changing code, update the corresponding docstrings and comments.

## Conclusion

The PTO Frontend provides a powerful and extensible system for parsing Python code into optimized tensor computation IR. By understanding the architecture, key concepts, and existing extension mechanisms documented here, developers can effectively work with, maintain, and debug the frontend parser.

For questions or issues, please refer to the source code documentation, test cases, or reach out to the PTO development team.
