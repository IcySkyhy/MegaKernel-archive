# Pass 编译性能优化报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **Pass 名称** | {PassName} |
| **优化日期** | {YYYY-MM-DD} |
| **算子脚本** | {user_specified_script}.py |
| **Op 数量** | {ops_count} |
| **目标耗时** | {target_time}s（基于 {ops_count}/200000 × 20s 计算）|

## 优化概览

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| **平均耗时** | {before_avg}s | {after_avg}s | **{improvement}%** |
| **最大耗时** | {before_max}s | {after_max}s | {max_improvement}% |
| **最小耗时** | {before_min}s | {after_min}s | {min_improvement}% |
| **达标状态** | 未达标 | 达标 | - |

## 优化详情

### 优化 1：{优化名称}

**优化类型**：{算法优化/数据结构优化/缓存优化/提前终止/避免拷贝/...}

**修改位置**：
- 文件：`{file_path}`
- 函数：`{function_name}`
- 行号：{line_range}

**问题描述**：
{描述优化前存在的性能问题，例如：O(n²) 的重复遍历、频繁的内存分配、cache miss 严重等}

**优化方案**：
{详细描述优化方法，例如：预构建哈希索引将 O(n²) 降为 O(1)、使用 reserve() 预分配内存等}

**代码变更**：
```cpp
// 优化前
{before_code_snippet}

// 优化后
{after_code_snippet}
```

**性能影响**：
- 优化前耗时：{before_time}s
- 优化后耗时：{after_time}s
- **提升：{delta_time}s（{delta_percent}%）**

**火焰图对比**：
- 优化前：`flamegraph_before.svg` — 路径：`{flamegraph_before_path}`
- 优化后：`flamegraph_after.svg` — 路径：`{flamegraph_after_path}`
- 差异图：`diff_flamegraph.svg` — 路径：`{diff_flamegraph_path}`

**perf 数据佐证**：
| 函数名 | 优化前占比 | 优化后占比 | 变化 |
|--------|-----------|-----------|------|
| {hot_function_1} | {before_pct}% | {after_pct}% | {change}% |
| {hot_function_2} | {before_pct}% | {after_pct}% | {change}% |

---

### 优化 2：{优化名称}

{同上格式，逐项列出每个优化}

---

## 性能数据汇总

### 各优化项贡献

| 优化项 | 耗时减少(s) | 贡献占比 | 优化类型 |
|--------|------------|---------|---------|
| {优化1名称} | {delta1}s | {pct1}% | {type1} |
| {优化2名称} | {delta2}s | {pct2}% | {type2} |
| ... | ... | ... | ... |
| **总计** | **{total_delta}s** | **100%** | - |

### 耗时趋势

```text
优化阶段          平均耗时(s)    累计提升
─────────────────────────────────────────
优化前（基线）     {baseline}s      -
优化1完成后        {after1}s       {pct1}%
优化2完成后        {after2}s       {pct2}%
...
最终              {final}s        {total_pct}%
```

## 经验总结

### 可复用模式

| 模式 | 适用场景 | 本次应用 | 预期收益 |
|------|---------|---------|---------|
| {pattern1} | {scenario1} | {application1} | {benefit1} |
| {pattern2} | {scenario2} | {application2} | {benefit2} |

### 注意事项

1. {注意事项1}
2. {注意事项2}
3. {注意事项3}

### 后续优化建议

{如果还有未实施的优化方向，列出建议}

## 附录

### 相关文件

| 文件类型 | 路径 |
|---------|------|
| 优化前日志 | `{before_log_path}` |
| 优化后日志 | `{after_log_path}` |
| 优化前火焰图 | `{before_flamegraph_path}` |
| 优化后火焰图 | `{after_flamegraph_path}` |
| 差异火焰图 | `{diff_flamegraph_path}` |
| Git commits | `{commit_range}` |

### 原始 perf 数据

<details>
<summary>点击展开优化前 perf report top 20</summary>

```text
{perf_report_before_top20}
```

</details>

<details>
<summary>点击展开优化后 perf report top 20</summary>

```text
{perf_report_after_top20}
```

</details>
