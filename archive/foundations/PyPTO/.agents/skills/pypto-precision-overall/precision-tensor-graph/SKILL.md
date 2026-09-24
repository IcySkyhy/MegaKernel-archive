---
name: precision-tensor-graph
description: PyPTO 算子精度校验与对比技能。包含 tensor_graph 校验流程和精度工具对比法（文件保存）。先通过 tensor_graph 校验判断前端构图是否正确，校验失败则开启保存 tensor 对比流程定位首个失败 op。触发词：精度工具、精度对比、文件保存、pass_verify_save、tensor_graph 校验。
---

# 精度校验与对比

> **本 skill 是唯一可分析前端 tileop 代码问题并尝试修复的阶段。**

包含三个阶段：
1. **算子写法校验**：检查算子代码是否符合框架规范（前置检查）
2. **tensor_graph 校验**：判断前端构图是否正确
3. **精度工具对比法**：校验失败时，保存中间 tensor 对比定位首个失败 op

---

## 前置检查：算子写法校验

在执行任何精度校验之前，先检查用户的算子代码是否符合 PyPTO 框架规范。逐项检查以下规则，发现违规立即提示用户修正后再进行后续调试。

### 规则 1：kernel 输出必须通过切片写回，禁止直接赋值

> 参考：[kernel-output-not-written-back.md](../../../../docs/zh/tutorials/appendix/faq/kernel-output-not-written-back.md)

`@pypto.frontend.jit` 装饰的 kernel 函数**不支持返回值**，输出必须通过参数传入并用 `[:]` 写回。直接赋值 `=` 会创建新的局部变量，不会修改外部 Tensor。

**检查方法**：扫描 kernel 函数体，找出所有作为输出参数的变量名，检查是否存在直接赋值（`output = ...`）而非切片写回（`output[:] = ...`）。正确写法包括 `[:]`、`.move()`、`.assemble()`。

### 规则 2：view 的 validShape 依赖其他 Tensor 时必须显式传入 valid_shape

> 参考：[view-valid-shape-precision.md](../../../../docs/zh/tutorials/appendix/faq/view-valid-shape-precision.md)

当 `view` 的输入 Tensor 的 validShape 来自另一个 Tensor（无法通过推导得到）时，必须显式传入 `valid_shape` 参数，否则框架无法正确推导输出的 validShape，导致精度问题。

**检查方法**：扫描 kernel 函数体中所有 `pypto.view()` 调用，检查是否存在 validShape 依赖其他 Tensor 但未传入 `valid_shape` 的情况。

```python
# 错误：cur_seq 来自 act_seqs[b_idx]，框架无法推导出 validShape
a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0])

# 正确：显式传入 valid_shape
a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
```

---

## 阶段一：tensor_graph 校验

### 步骤 1：编译安装

执行算子之前，**必须**先进行编译：

```bash
python3 -m pip install . --verbose
```

### 步骤 2：配置 tensor_graph 校验

在 PyPTO 算子实现文件中配置 `verify_options`，开启 tensor_graph 校验：

> **必须使用 `pass_verify_pass_filter: []`（空列表）**，表示只校验 tensor_graph，不校验任何 Pass。即使用户原始代码中配置了 `["all"]`，进入本阶段时也必须改为 `[]`。

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": []     # 空列表表示只校验 tensor_graph，跳过 pass 校验
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(...)
```

### 步骤 3：设置 Golden 数据

**必须严格按照"计算 golden → 设置 golden → 执行算子"顺序**：

```python
# 计算 golden
torch_output = torch.add(input_data0, input_data1)

# 设置 golden（必须放在 pypto 算子运行前！此处 torch 的数据必须在 cpu 上）
pypto.set_verify_golden_data(goldens=[None, None, torch_output.cpu()])

# 执行 pypto 算子
pypto_output = your_kernel(input_data0, input_data1)
```

### 步骤 4：查看校验结果

```bash
# 查看最新 verify 目录
ls -lt output/output_*/verify_* | head -n 1

# 查看 tensor_graph 校验结果
cat output/output_*/verify_*/interpreter.log
```

**常见问题**：

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| **aicore error / MPU 地址访问无效** | NPU 卡资源冲突或状态异常 | 1. 更换 `TILE_FWK_DEVICE_ID` 到空闲卡<br>2. 重新执行 |
| **verify 日志无内容** | Golden 数据未设置或顺序错误 | 确保"计算 golden → 设置 golden → 执行算子"顺序，检查 set_verify_golden_data 输入输出是否匹配 |

### 步骤 5：根据校验结果决策

> **结论必须基于 `interpreter.log` 的实际输出**。禁止仅凭代码注释、变量命名或推测得出结论。报告中对 PASS / FAIL 的判定必须引用 `interpreter.log` 中的具体行（如 `result FAILED` / `result PASS`）。

```
tensor_graph 校验结果
    │
    ├── FAIL → 前端代码问题，进入【阶段二：精度工具对比法】定位首个失败 op
    │
    └── PASS → 精度问题不在前端，进入 precision-pass 进行 Pass 层校验
```

**移除 tensor_graph 校验配置**：

确认有校验结果后，移除校验配置避免影响后续调试：

```python
@pypto.frontend.jit()  # 移除 verify_options 参数
def your_kernel(...)

# 删除: pypto.set_verify_golden_data(...)
```

---

## 阶段二：精度工具对比法（tensor_graph FAIL 时执行）

使用 `pypto.pass_verify_save()` 和 `torch.save()` 保存中间结果到文件，然后使用对比工具分析。

## 核心原理

1. **数据对齐**：golden 和 kernel 的计算逻辑、切块方式、数据维度必须完全一致，如果实现不一致，改写golden函数
2. **kernel 保存**：使用 `pypto.pass_verify_save()` 保存中间结果（循环场景使用 `cond=(idx == batch_size - 1)`，用最后一块数据对比）
3. **golden 保存**：使用 `torch.save()` 保存中间结果（循环场景使用 `if idx == batch_size - 1:`）
4. **数据对比**：使用对比工具检查 kernel 的 `.data` 文件和 golden 的 `.pt` 文件
5. **全量对比**：一次性开启所有关键检查点，对比所有中间结果，定位第一个计算结果不同的 op
6. **数据类型**：保存什么类型就读取什么类型，kernel 和 golden 的数据类型和 shape 必须完全一致，不一致则直接比对失败

## 核心原则

### 原则 1：插入检查点（kernel 函数）

```python
# 必须启用验证选项
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": []
}

@pypto.frontend.jit(verify_options=verify_options)
def kernel(inputs, outputs):
    # 基础场景
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "1_after_op1")

    # 循环场景：只保存最后一块数据，必须用cond = (idx == end)的条件保存
    # 单层循环：cond=(idx == batch_size - 1)
    # 多层循环：cond=((idx1 == end1) * (idx2 == end2))
    for idx in range(batch_size):
        temp = pypto.compute_op(inputs[idx])
        pypto.pass_verify_save(temp, "2_loop_result", cond=(idx == batch_size - 1))
```

### 原则 2：保存 golden 中间结果

```python
import os
import torch

operator_dir = f"{operator}"
os.makedirs(operator_dir, exist_ok=True)
output_dir = os.path.join(operator_dir, "golden_data")
os.makedirs(output_dir, exist_ok=True)

def golden(inputs, outputs):
    # 基础场景
    temp1 = compute_op1(inputs[0])
    torch.save(temp1, f"{output_dir}/golden_1_after_op1.pt")

    # 循环场景：只保存最后一块数据（更能反映问题）
    for idx in range(batch_size):
        temp = compute_op(inputs[idx])
        if idx == batch_size - 1:
            torch.save(temp, f"{output_dir}/golden_2_loop_result.pt")
```

**文件命名约定**：
- golden 文件：`golden_{序号}_{检查点名称}.pt`
- jit 文件：`{序号}_{检查点名称}_{number}.data`（自动生成）
- 检查点名称必须按计算顺序添加数字前缀，确保按顺序对比

### 原则 3：全量对比策略

一次性开启所有关键计算节点的检查点，对比所有中间结果，定位第一个计算结果不同的 op。

## 易错点及修正方案

### 1. 执行路径问题

**问题**：Output 文件和 pt 文件生成在执行路径下

**修正**：用 `-w` 参数指定工作目录
```bash
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py -w /path/to/operator -v
```

### 2. 数据类型读取问题

**问题**：读取数据时要根据保存的类型读取（BF16/FP32/INT32 等）

**修正**：对比工具会根据 CSV 文件中的 dtype 自动判断数据类型：
- dtype=8: BF16 格式（2字节）
- dtype=7: FP32 格式（4字节）

### 3. FP8/BOOL 类型处理

**说明**：FP8 和 BOOL 类型无法直接进行算术运算，对比脚本会自动处理：
- FP8 类型（dtype=5/17/18）：脚本内部自动转 FP32 再计算，容差仍按 FP8 标准（rtol=1e-1, atol=1e-2）
- BOOL 类型（dtype=15）：脚本内部自动转 INT8 再计算，容差仍按 BOOL 标准（rtol=0, atol=1e-4）

**使用方式**：kernel 和 golden 直接保存原始类型即可，无需手动转换：

```python
# kernel 侧：直接保存 FP8 结果
temp_fp8 = pypto.compute_op(...)  # dtype=17/18 FP8 结果
pypto.pass_verify_save(temp_fp8, "checkpoint_name")

# golden 侧：直接保存 FP8 结果
temp = compute_op(...)  # torch.float8_e4m3fn / torch.float8_e5m2
torch.save(temp, f"{output_dir}/golden_checkpoint_name.pt")
```

### 4. 检查点插入位置问题

**原则**：检查点位置要一一对应
- 如果 kernel 在 A→B→C 三个步骤后都插入检查点，golden 也需要在对应步骤保存
- 保持两边计算逻辑和保存时机完全一致

### 4.5. 计算逻辑不一致问题（高频易错）

**问题**：kernel 分步计算有中间结果，golden 连乘缺少中间变量，导致对比错位

**典型场景**：kernel 在 `matmul` 后乘 `scale` 并保存，golden 用连乘 `matmul * scale` 缺少中间变量

**修正方案**：确保 kernel 和 golden 保存的是同一计算阶段的结果（如都保存乘 scale 后的结果）

### 5. 切块计算问题

**原则**：保持保存的数据维度一致

**解决思路**：
1. **golden 保存对应的切片数据**：golden 一次性计算完整数据，只保存与 kernel 对应的切片
2. **golden 改写为和 kernel 实现完全一致**（推荐）：golden 模拟 kernel 的循环结构和分块策略

### 6. 精度标准问题

**修正**：对比工具根据数据类型自动设置容差：

| 数据类型 | dtype | rtol | atol |
|---------|-------|------|------|
| INT16 | 2 | 0 | 1e-4 |
| INT32 | 3 | 0 | 1e-4 |
| INT64 | 4 | 0 | 1e-4 |
| FP8 | 5 | 1e-1 | 1e-2 |
| FP16 | 6 | 1e-3 | 1e-3 |
| FP32 | 7 | 1e-3 | 1e-4 |
| BF16 | 8 | 5e-3 | 5e-2 |

### 7. 量化场景容差设置

**修正**：使用 `--rtol` 和 `--atol` 参数指定自定义容差：
```bash
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py \
    --rtol 0.0078125 \
    --atol 0.001 \
    -v
```

### 8. 对比逻辑问题

**修正**：对比工具使用双阈值方案：
- 警告阈值：`tol_attn = abs_sum * rtol / 2 + atol`，超出此阈值的元素允许少量存在（不超过 `error_threshold` 个）
- 失败阈值：`tol_fail = tol_attn * 128`，超出此阈值的元素不允许存在（必须为 0）
- 判断条件：`warn_count <= error_threshold and fail_count == 0`
- `error_threshold` 基于统计：`max(16, int(sqrt(non_zero_count)) // 2)`，且不超过 `non_zero_count * min(rtol, atol)`

## 完整工作流程

### 步骤 1：插入检查点

在 jit 和 golden 函数中插入对应的检查点（参考原则 1 和 2）。

**循环场景关键点**：
- 使用 `cond=(idx == batch_size - 1)` 只保存最后一块数据（更能反映问题）
- 多层循环使用 `cond=((idx1 == end1) * (idx2 == end2))`
- 确保 kernel 和 golden 保存相同的 idx 数据
- 在检查点名称中包含 idx 信息

### 步骤 2：运行测试生成数据

```bash
python3 test_operator.py
```

### 步骤 3：对比检查点

```bash
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py -v
```

### 步骤 4：分析对比结果

根据对比结果定位问题：
- 查看哪些检查点不匹配
- 不匹配的检查点就是导致精度问题的位置
- 分析不匹配率、最大差异等指标

**⚠️ 重要检查事项**

1. **检查点数据类型一致性**
    - kernel 和 golden 的数据类型和 shape 必须完全一致，不一致则直接比对失败
    - 在 kernel 和 golden 中使用相同的数据类型保存

2. **审查 log 文件内容**
    - 查看生成的 `*_verify_result.log` 文件
    - 如果在量化类场景下中间结果略超过容差阈值，可以适当放宽，这类检查点不视作精度比对失败

### 步骤 5：定位并修复

根据对比结果定位到具体的 op，然后修复问题。

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤，按计算顺序添加数字前缀。

### 2. 循环场景处理

- 使用 `cond=(idx == batch_size - 1)` 确保只保存最后一块数据（最后一块比第一块更能反映问题）
- 多层循环使用 `cond=((idx1 == end1) * (idx2 == end2))`
- kernel 和 golden 必须保存相同的 idx 数据
- 避免生成过多文件

### 3. 全量对比模式

一次性开启所有关键计算节点的检查点，对比所有中间结果。

**使用场景**：
- 算子计算步骤较多，需要全面检查
- 已经知道大概的问题范围，需要精确定位
- 需要分析整个计算流程的精度变化

### 4. 清理调试代码

修复问题后：
```bash
rm -f operator/golden_data/*.pt
rm -rf output/output_*

# 移除调试代码：
# - 删除 pypto.pass_verify_save() 调用
# - 删除 verify_options 参数
# - 删除 golden 中的 torch.save() 调用
```

## 常见问题

### Q1: kernel 和 golden 保存的数据不一致

**原因**：kernel 保存 idx=0，但 golden 保存了其他 idx / golden 没有切块计算

**解决**：确保两者使用相同的条件，参考"易错点 4：切块计算问题"

### Q2: 找不到检查点文件

**检查**：
- jit 代码中是否使用了 `pypto.pass_verify_save()`
- 是否设置了 `verify_options={"enable_pass_verify": True}`
- 文件命名是否符合约定
- 是否在正确的目录下执行对比工具

## 通用对比工具

本方法提供了通用对比脚本，自动完成检查点扫描和对比：

```bash
# 自动检测并对比所有检查点
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py

# 列出所有检查点
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py --list

# 显示详细对比
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py --verbose

# 指定工作目录
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py -w /path/to/operator -v

# 指定 golden 文件所在目录（推荐）
python3 .agents/skills/pypto-precision-overall/precision-tensor-graph/scripts/compare_accuracy.py -w /path/to/operator -g /path/to/operator/golden_data -v
```

**工具功能**：
- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 根据数据类型自动设置容差标准
- ✓ 统计不匹配率而非只看最大差异
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比
- ✓ 按检查点名称开头的数字排序，确保按计算顺序对比

## 检查清单

使用文件保存方法时，确保：

- [ ] **步骤 1**：插入检查点
  - [ ] 设置 `verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True, "pass_verify_pass_filter": []}`
  - [ ] kernel 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] golden 函数中使用 `torch.save()` 保存中间结果
  - [ ] 循环场景使用 `cond=(idx == batch_size - 1)` 和 `if idx == batch_size - 1:`（最后一块比第一块更能反映问题）
  - [ ] 文件命名遵循约定（添加数字前缀）
  - [ ] 检查点插入位置要一一对应
  - [ ] 切块计算要保持一致
  - [ ] golden 和 kernel 的切块逻辑完全一致
- [ ] **步骤 2**：运行测试生成数据
  - [ ] golden 数据保存到独立文件夹
  - [ ] 避免与之前测试的 golden 文件混淆
- [ ] **步骤 3**：对比检查点
  - [ ] 在正确目录下使用通用工具
  - [ ] 如 golden 文件在独立文件夹，使用 `--golden-dir` 参数指定
  - [ ] 查看对比结果和不匹配率
  - [ ] 确认检查点按计算顺序对比
- [ ] **步骤 4**：继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
  - [ ] 保持检查点命名前缀的连续性
- [ ] **步骤 5**：定位并修复
  - [ ] 定位到具体的 op
  - [ ] 修复问题
  - [ ] 重新验证
  - [ ] 用表格展示最后保存点的对比结果
  - [ ] 清理调试代码
