# Pass 修改前安全检查

## 使用时机

当用户要求审查一个 Pass diff、准备修改 Pass、或怀疑某个修复会引入回归时使用本文件。目标是在改动前识别可能触发的历史 bug 模式。

## 使用流程

1. 读取 `patterns/source-patterns.json` 和 `patterns/non-pass-patterns.json`。
2. 根据本次 diff 修改的文件，筛选出 `files` 或 `files_to_check` 命中的模式。
3. 对命中模式，在新代码中搜索 `code_anti_pattern` 描述的特征。
4. 输出 Change Safety Report：
   - 命中模式列表。
   - 代码中是否存在对应反模式。
   - 建议补充的 UT / dump / checker。

## 通用检查清单

### 容器与指针安全

- [ ] 所有 `.front()` / `.begin()` / `[0]` 访问前是否检查了 `empty()`？
- [ ] 多生产者/消费者场景是否只取第一个元素？是否确认唯一性？
- [ ] 所有 `dynamic_cast` / `static_pointer_cast` 是否检查返回值非空？

### 视图类 OP 一致性

- [ ] 是否统一处理 `OP_VIEW` / `OP_RESHAPE` / `OP_ASSEMBLE` / `OP_VIEW_TYPE`？
- [ ] 新增 opcode 特判时，是否检查视图类 opcode 的合并、传播、内存类型逻辑？

### 内存类型与 dtype 路径

- [ ] 是否显式枚举 DDR / L0C / L1 / UB 等内存类型？
- [ ] 强制 fallback DDR 前是否检查并行消费者和 OoO 约束？
- [ ] spill/copy 路径上的 dtype 转换是否发生在正确方向？
- [ ] shape/stride 乘法是否做溢出保护？

### 动态 shape 与 validshape

- [ ] 创建新 tensor 时是否传递 `DynValidShape`？
- [ ] 插入 VIEW/ASSEMBLE/COPY 中间 tensor 后，是否从源 tensor 同步 `DynValidShape`？对应 P016。
- [ ] 动态 offset/validshape 无法求值时，是明确拒绝还是静默回退到 0/1？
- [ ] 合并 view 时 validshape 是否来自 op attribute 的 `ToDynValidShape`？
- [ ] reshape->assemble 重排是否证明所有中间动态 shape 可求值？对应 P017。

### Axis-combine 与 shape-transform

- [ ] `OP_VIEW` / `OP_ASSEMBLE` / `OP_RESHAPE` 是否在 axis-combine pad 逻辑里统一处理？对应 P018。
- [ ] shape-transform op 是否错误落入 elementwise/fallback 分支？
- [ ] rawshape pad 轴是否符合语义，特别是 `[N, 1]` 场景应 pad 倒数第二轴还是尾轴？

### 统计、累加与下溢

- [ ] 统计字段是用 `=` 还是 `+=`？多次调用是否会覆盖？
- [ ] `size_t` 减法前是否确保被减数大于等于减数？

### 返回值与边界检查

- [ ] 调用的 helper 是否返回状态码/offset？是否检查 `FAILED` / `-1`？
- [ ] 循环/索引边界是否统一为半开区间 `[0, size)`？

### 哈希与顺序

- [ ] hash 是否依赖无序容器遍历顺序或指针地址？
- [ ] 是否需要对 key/operand 排序后再 hash？
- [ ] 拆分/克隆后是否重建 offset/dynParam/argList 索引映射？
- [ ] 公共算子消除是否显式跳过语义敏感 op，如 `OP_VIEW`、`OP_VEC_DUP`？对应 P019。
- [ ] clone 后查找原始属性时是否有 `rawMagic` 兜底，而不是只比较 shared_ptr 身份？对应 P020。

### 图关系同步

- [ ] 替换/删除 op 时是否同步更新 producer/consumer 双向关系？
- [ ] 是否使用统一 API, 如 `ReplaceInput`, 而不是直接赋值？
- [ ] 新增边时是否检查 source != target 和成环？
- [ ] ReduceCopy / auto-mix 合并后，内部 tensor 是否仍被外部子图使用？对应 P021。

### 调度与同步

- [ ] 调试建议是否错误要求关闭 `OoOSchedule` / `OoOScheduler`？若是，改为开启 dump/checker/health report。
- [ ] InsertSync 处理 event-id 耗尽时，是否区分“不可放松”和“执行失败”？对应 P023。
- [ ] 遍历依赖容器时，处理函数是否会修改同一个容器？若会，是否先复制快照？对应 P023。
- [ ] 正反 corepair 是否都被尝试，而不是只按单方向放松同步依赖？

### 硬编码常量

- [ ] 是否有平台相关阈值、latency、event id 上限等 magic number？
- [ ] 是否应收敛到平台配置，而不是散落代码中？
- [ ] 默认 tile shape 推导是否处理动态维 `-1`、空 shape、极大 shape 和左移溢出？对应 P022。

### 特殊 copy / raw tensor 属性

- [ ] SplitRaw 后是否同步更新 VIEW、ASSEMBLE 和特殊 copyout producer 的 offset？对应 P024。
- [ ] `OP_SHMEM_GET` 等具备 copyout 语义的非普通 COPY_OUT 是否被纳入处理？
- [ ] COPY_OUT -> HUB -> ASSEMBLE/OUTCAST 级联复用时，是否同时更新 copy attr 的 offset 和 raw_shape？对应 P025。

### 跨 Pass 影响

- [ ] 本次改动是否改变本 Pass 的输出不变量？
- [ ] 下游 Pass 是否依赖这些不变量？是否需要同步更新？
- [ ] 是否已通过 `pass-memory/index.md` 找到并查看本 Pass 的设计记忆？
- [ ] 若建议使用 `disable_pass` 定界，是否已确认该 Pass 不在 required/structural 列表？对应 S045。
- [ ] 是否存在更窄的功能开关可以替代关整个 Pass，例如 `enable_vf`、`auto_mix_partition`、`copyout_resolve_coalescing`、`ooo_sched_mode`？

## 输出格式

```markdown
# Change Safety Report

## 命中模式
- {pattern_id}: {category_cn}, 命中文件 {file}

## 风险判断
- {risk}: {why}

## 建议动作
- 修改前检查: {code_check}
- UT/dump: {verification_hint}
```
