# 仿真故障排除

> 遇到仿真失败时按需加载，根据错误特征定位章节。

## 目录

- [checkSqeSupported 断言失败](#checksqesupported-断言失败)
- [报告生成超时](#报告生成超时)
- [仿真器关闭阶段 segment fault](#仿真器关闭阶段-segment-fault)
- [功能仿真退出码为 1 但无报错](#功能仿真退出码为-1-但无报错)
- [找不到算子文件](#找不到算子文件)

---

## checkSqeSupported 断言失败

### 错误特征

```text
[DRVSTUB_WARN] driver_queue.c:163 checkSqeSupported:RTSQ_2 sqe_payload[0]: 0xb, unsupported task type: SDMA, camodel only supports AIC/AIV/PLACE_HOLDER/NOTIFY_REC/NOTIFY_WAIT
python3: driver_queue.c:164: checkSqeSupported: Assertion `0' failed.
```

关键标志为 `driver_queue.c:164: checkSqeSupported: Assertion '0' failed.`，其前的 `DRVSTUB_WARN` 行会指明具体不支持的任务类型（示例中为 SDMA，也可能为其他类型）。

### 原因

camodel 仿真器仅支持 `AIC`/`AIV`/`PLACE_HOLDER`/`NOTIFY_REC`/`NOTIFY_WAIT` 五种 task type，仿真支持的任务类型有限，不支持隐式搬运操作（如 SDMA）。当用例中存在将 tensor 数据留在 NPU 上、并通过 task queue 提交不支持任务类型的操作时，会被 `checkSqeSupported` 拒绝后触发 `__assert_fail` 崩溃。

此问题不限于 PyTorch 标准算子。任何非 PyPTO kernel 的 NPU 操作都可能命中：计算类算子（如 `torch.topk`）、隐式访问 NPU tensor 的操作（如 `print(tensor)`）等，只要底层提交了不支持的 task type 即会触发。

`.cpu()` 不触发此问题：camodel 是 CPU 仿真器，host/device 共享内存，`.cpu()` 底层走 `drvMemcpy`（直接 `memcpy_s`），不经过 task queue。但 `drvMemcpy` 只做内存拷贝不做计算，无法替代 NPU 算子，不能强制走此路径。

### 定位触发操作

在脚本中逐行添加 trace，定位首个触发断言的操作：

```python
print("[TRACE] before torch.topk", flush=True)
result = torch.topk(x, k=8, dim=-1, sorted=False)[1]
print("[TRACE] after torch.topk", flush=True)
```

最后出现的 `before xxx` 之后缺失 `after xxx` 的操作即为触发点。常见触发操作包括 `torch.topk`、`print(tensor)` 及其他底层提交不支持 task type 的 NPU 操作。

### 修复：将用例相关 tensor 数据分配在 CPU

所有非 PyPTO kernel 处理的 tensor 数据，都应分配在 CPU 上执行，避免经过 task queue 提交不支持的任务类型。先将相关 tensor `.cpu()`，后续操作自然在 CPU 上运行：

```python
# 1. 将相关 tensor 搬到 CPU
input_cpu = input_tensor.cpu()
output_cpu = output_tensor.cpu()

# 2. 非 PyPTO kernel 操作基于 CPU tensor 执行，不触发断言
result = torch.topk(input_cpu.to(torch.float32), k=k, dim=-1, sorted=False)[1]

# 3. 若涉及 topk(sorted=False) 等顺序不确定的算子，比较前需统一排序
sort_idx_k = output_cpu.argsort(dim=-1, descending=True)
sort_idx_g = result.argsort(dim=-1, descending=True)
assert_allclose(output_cpu.gather(1, sort_idx_k).cpu().numpy(),
                result.gather(1, sort_idx_g).cpu().numpy(),
                rtol=5e-3, atol=5e-3)
```

---

## 报告生成超时

原因：32 核日志处理慢。

修复：使用 `--core-id 0` 只生成单核报告。

```bash
cannsim report -e <cannsim_dir> -o <cannsim_dir>/report --core-id 0
```

---

## 仿真器关闭阶段 segment fault

### 错误特征

```text
  passed: bs=32
segment fault !!!
libpem_davinci.so(tm_engine::TmSim::post_tick_all_post_observers...)
Simulation FAILED · exit 1
```

### 原因

camodel 仿真器在所有 task 完成、应用正常退出后的关闭阶段崩溃（`libpem_davinci.so` 的 `TmSim` 线程清理逻辑），是仿真器自身 bug，不影响 kernel 执行结果和精度校验。

### 判定方法

日志满足以下**全部**条件时，可判定为仿真器关闭阶段 bug，**kernel 结果有效**，可忽略该 segment fault：

1. 所有 `TASK_BEGIN` 都有对应的 `TASK_DONE`
2. 无 `checkSqeSupported` 断言失败错误
3. 用例输出 `passed` 或精度校验通过
4. `segment fault` 出现在 `passed` **之后**

---

## 功能仿真退出码为 1 但无报错

### 错误特征

```text
[2026-07-25 02:21:29] [ERROR] USER_APP failed after 21.66s (exit=1)
────────────────────────────────────────────────────────────────────────
Simulation FAILED  ·  run time 21.7s  ·  exit 1
```

关键标志为 `Simulation FAILED · exit 1`，但日志中无 `checkSqeSupported` 断言失败、无 `segment fault` 等明确错误，且用例精度校验通过。

### 原因

功能仿真（模式 A）中，用例脚本已执行完成且 kernel 运行正常，但退出码非 0。通常由仿真环境的退出清理逻辑或依赖库（如 torch_npu）的退出钩子触发，不影响 kernel 执行结果和精度校验。

### 判定方法

日志满足以下**全部**条件时，可判定为仿真成功，**kernel 结果有效**，可忽略退出码：

1. 用例精度校验通过（如 `Max difference` 在容差内，或 `assert_close` 未抛异常）
2. 无 `checkSqeSupported` 断言失败错误
3. 无 `segment fault`
4. AICore Prof Summary 正常输出（有 End-to-End Time 和 Utilization）

---

## 找不到算子文件

原因：执行目录错误，未在 pypto 主目录下执行。

修复：在 pypto 主目录下执行所有命令。

```bash
cd /path/to/pypto
cannsim record 'python3 examples/00_hello_world/hello_world.py' -s Ascend950 -o output/
```
