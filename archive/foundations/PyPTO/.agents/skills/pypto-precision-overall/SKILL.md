---
name: pypto-precision-overall
description: PyPTO 算子精度问题调试技能。提供场景路由式排查：根据用户问题自动选择排查路径，支持用例剪枝、前端校验、Pass校验、特定问题排查、上板二分的自由组合。当需要调试 PyPTO 算子精度、定位精度差异来源时使用此技能。
---

# PyPTO 算子精度调试技能

提供场景路由式排查流程，根据用户问题自动选择排查路径，支持各阶段的自由组合。

## 前置环境检查

执行任何步骤前，先验证环境。环境问题会导致算子执行报错而非精度差异，不应进入精度调试流程。

```bash
# 1. CANN 包是否已 source（ASCEND_HOME_PATH 必须存在）
echo $ASCEND_HOME_PATH

# 2. 设备 ID 是否已设置
echo $TILE_FWK_DEVICE_ID

# 3. PTO 库路径是否已配置
echo $PTO_TILE_LIB_CODE_PATH
```

**三项任一为空 → 退出，提示用户先配置环境。** 环境问题不是精度问题。

```bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
arch=$(uname -m)
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH}/${arch}-linux
export TILE_FWK_DEVICE_ID=0
```

> **注**：前端代码（用户编写的 kernel、测试用例）可深入分析写法问题并尝试修正；框架代码（Pass/machine/同步/合轴/合图）仅定位问题归属，不分析框架内部根因。

## 场景路由

### 路由匹配规则

1. 用户描述中**明确包含**某条路由的触发关键词 → 直接按该路由执行
2. 用户描述**无法匹配**或**匹配到多条**路由 → 列出以下场景让用户选择：

```
请选择排查场景：
1. 全自动排查 — 从前端到上板逐层排查
2. 用例剪枝 — 化简用例减少循环次数再调试
3. 只怀疑前端问题 — 只做 tensor_graph 校验
4. 验证VF/同步/合轴 — 直接验证开关配置问题
5. 前端校验后排查特定问题 — tensor_graph 校验后直接查VF/同步/合轴
6. 只跑Pass对比 — 只做 Pass 层校验
7. machine校验 — 怀疑内存重叠时检测 workspace 是否有问题
8. 直接上板二分 — 直接添加检查点二分定位
9. 偶现精度排查 — 无法稳定复现，用独立开关逐项排查
```

> **machine校验 组合规则**：machine校验涉及重新编译 whl，成本较高且覆盖场景较少，**默认不加入任何流程**。仅当用户明确要求时，才在对pass校验流程的末端追加 machine校验 步骤。

### 路由表

| 场景 | 触发关键词 | 路由链 |
|------|-----------|--------|
| 全自动排查 | 精度问题、精度调试、精度不对 | tensor-graph → pass校验 → 特定排查 → 二分 |
| 用例剪枝 | 用例剪枝、剪枝、化简用例、缩减循环、pruning | pruning → 进入选中的调试流程 |
| 只怀疑前端问题 | 前端校验、tensor_graph、构图问题 | tensor-graph → 结束 |
| 验证VF/同步/合轴 | VF融合、同步问题、合轴、开关验证 | 特定问题排查 → 结束 |
| 前端校验后排查特定问题 | 先查前端再用配置定界、先查构图再查开关、tensor_graph后查特定问题 | tensor-graph → 特定问题排查 |
| 只跑Pass对比 | pass校验、pass_compare、Pass层 | pass校验 → 结束 |
| machine校验 | 内存重叠、workspace问题、内存管理异常、内存复用错误 | memory-overlap-detector → 结束 |
| 直接上板二分 | 二分、上板二分、检查点tensor | 二分 → 结束 |
| 偶现精度排查 | 偶现、不稳定、无法复现、随机失败、时好时坏 | 偶现排查（独立开关逐项验证）→ 结束 |

## 路由链详情

### 全自动排查

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── 定位到前端构图错误 / 首个计算结果不同的 op ──→ 结束
       │
       │ tensor_graph PASS
       ▼
[precision-pass]  PreCheck/PostCheck 全链路校验 + pass_compare 逐 Op 对比
       │
       ├── 定位到引入偏差的 Pass 和具体 Op ──→ 结束
       │
       │ 所有 Pass 校验通过
       ▼
[precision-pass]  特定问题排查（同步/VF融合/合轴等开关配置验证）
       │
       ├── 通过开关配置定位到问题 ──→ 结束
       │
       │ 仍未定位
       ▼
[precision-binary-search]  添加检查点 tensor + 二分对比上板真实数据
       │
       └── 定位到上板执行阶段首个出错 op ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程（**必须用 `pass_verify_pass_filter: []` 先做 tensor_graph 校验，禁止直接开 `["all"]`**）
2. 若 tensor_graph FAIL → 阶段二定位问题 op → 结束
3. 若 tensor_graph PASS → 进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"一、Pass 校验"
4. 若 Pass 校验定位到问题 Op → 结束
5. 若 Pass 校验全部通过 → 进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"
6. 若特定问题定位到 → 结束
7. 若仍未定位 → 进入 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md)

> **结论约束**：所有阶段的 PASS / FAIL 判定和根因结论必须基于精度工具实际输出（`interpreter.log`、`pass_compare.py`、`compare_accuracy.py` 等）。禁止仅凭代码注释、变量命名或推测得出结论。

### 用例剪枝

```
[precision-pruning]  分析并缩减循环次数（保留尾块非对齐路径）
       │
       ├── 循环次数已最小 → 无需剪枝 → 进入主调试流程
       │
       └── shape 需要修改 → 修改 shape → 验证精度仍可复现 → 进入主调试流程
```

> 用例剪枝建议在 tensor-graph 校验之前执行，最大化后续步骤收益。

执行步骤：
1. 进入 [precision-pruning/SKILL.md](precision-pruning/SKILL.md) 执行完整流程
2. 剪枝完成后，返回主流程继续调试

### 只怀疑前端问题

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── FAIL → 阶段二定位首个计算结果不同的 op ──→ 结束
       │
       └── PASS → 前端构图正确 ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程
2. 无论结果如何，流程结束

### 验证VF/同步/合轴

```
[precision-pass]  特定问题排查（直接进入，跳过 Pass 校验）
       │
       ├── 通过开关配置定位到问题 ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"章节
2. 跳过"一、Pass 校验"流程

### 前端校验后排查特定问题

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── FAIL → 阶段二定位问题 op ──→ 结束
       │
       │ PASS（跳过 Pass 校验）
       ▼
[precision-pass]  特定问题排查
       │
       ├── 定位到 ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程
2. 若 tensor_graph FAIL → 阶段二定位问题 op → 结束
3. 若 tensor_graph PASS → 跳过 Pass 校验，直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"

### 只跑Pass对比

```
[precision-pass]  PreCheck/PostCheck 全链路校验 + pass_compare 逐 Op 对比
       │
       ├── 定位到问题 Op ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"一、Pass 校验"章节
2. Pass 校验完成后结束，不进入特定问题排查

### machine校验

> 当用户明确提到内存重叠，或全自动排查各阶段均无果但仍存在精度差异时，作为补充检测。

```
[pypto-memory-overlap-detector]  检测 workspace 内存重叠与管理异常
       │
       ├── 检测到内存重叠 → 修复 → 重新验证精度 → 结束
       │
       └── 无内存重叠 → 结束
```

执行步骤：
1. 进入 [../pypto-memory-overlap-detector/SKILL.md](../pypto-memory-overlap-detector/SKILL.md) 执行完整流程

### 直接上板二分

```
[precision-binary-search]  添加检查点 tensor + 二分对比上板真实数据
       │
       └── 定位到上板执行阶段首个出错 op ──→ 结束
```

执行步骤：
1. 直接进入 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md) 执行完整流程

### 偶现精度排查

> **独立场景，不在全自动排查流程中**。仅当精度问题无法稳定复现（偶现、随机失败、时好时坏）时触发。通过独立开关逐项关闭框架功能，重新编译执行后观察精度是否恢复，定位偶现问题的归属组件。

执行步骤：
1. 进入 [precision-occasional/SKILL.md](precision-occasional/SKILL.md) 执行完整流程

> **关键原则**：每个开关独立测试，每次只改一个变量。每个开关修改后必须重新编译安装。每个开关建议运行 3 次以统计复现概率。测试完成后恢复所有代码改动。

## 方法对比

| 特性 | precision-tensor-graph | precision-pass | precision-binary-search | precision-pruning |
|------|------------------------|----------------|-------------------------|-------------------|
| **定位目标** | 前端构图错误 / 首个计算不同的 op | 引入偏差的 Pass 和 Op / 开关配置问题 | 上板执行阶段首个出错 op | 缩减用例规模，加速调试 |
| **实现方式** | `pass_verify_save()` + `torch.save()` 对比 | PreCheck/PostCheck + `pass_compare` + 开关验证 | 检查点 tensor 作为输入参数对比 | 缩减 shape 减少 tile 循环次数 |
| **代码修改** | 添加 `pass_verify_save()` 调用 | 配置 `verify_options` + `tile_fwk_config.json` | 修改 kernel 函数签名，添加检查点参数 | 修改测试入口的 shape 参数 |
| **使用难度** | 简单 | 中等 | 较复杂 | 简单 |

## 参考资料

- [PyPTO API 文档](../../../docs/zh/api/)
- [pass_verify_save API](../../../docs/zh/api/others/pypto-pass_verify_save.md)
