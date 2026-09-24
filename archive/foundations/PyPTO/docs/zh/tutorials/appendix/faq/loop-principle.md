# Loop原理及其描述

在编译阶段会将loop编译为控制流，将loop body转换为计算流，分别对应`kernel_aicpu` / `kernel_aicore`。`kernel_aicpu`负责控制流的执行，创建`kernel_aicore`任务，包括使用到的内存分配以及执行参数准备，同时分析输入输出的依赖将`kernel_aicore`任务合并成一个更大的调度单元，然后提交给调度单元进行调度。

## 原型介绍

```python
def loop(start, end, step=1, name=None, idx_name=None,
         unroll_list=[1], submit_before_loop=False):

def loop_unroll(start, end, step=1, name=None, idx_name=None,
              unroll_list=[1], submit_before_loop=False):
```

## 参数说明

1. `start`, `end`, `step`：分别表示循环的起始值、结束值和步长，类型可以为`SymbolicScalar`或者`int`，和Python中的`range`语法基本保持一致。

2. `name`：循环的名称，默认值为`loop_{id}`，对实际使用运行效果无影响，仅用于调试和生成代码中注释信息。

3. `idx_name`：循环索引的名称，默认值为`loop_idx_{id}`，对于嵌套的Loop使用相同的`idx`会产生覆盖行为，目前会在前端进行检查报错。

4. `unroll_list`：主要用于循环展开，产生更大的loop body，降低调度开销。

   - 对于`unroll_list=2`，`loop`大概会产生如下代码：

     ```python
     new_start = start
     for k in unroll_list:
         left = (stop - start) % k
         for idx in loop(new_start, stop - left, k):
             for i in range(k):
                 body(idx)  # 需要用户一次处理step=1的步长
         new_start = stop - left
     ```

   - `loop_unroll`大概会产生如下代码：

     ```python
     new_start = start
     for k in unroll_list:
         left = (stop - start) % k
         for idx in loop(new_start, stop - left, k):
             body(idx, k)  # 需要用户一次处理k的步长
         new_start = stop - left
     ```

   原则上如果可以一次处理多个`i`，使用`loop_unroll`会更高效；如果一次只能处理1个`i`，则需要使用`loop`。

5. `submit_before_loop`：表示是否在循环开始前提交任务，默认值为`False`。如果设置为`True`，则循环前的任务会先提交到调度队列中，等待后续任务完成后再开始执行。**过多的设置`submit_before_loop`会增加调度开销**，建议仅在必要时设置为`True`。

6. `unroll_list`对`pypto.cond`的影响：通常一个循环中有一个`pypto.cond`，会产生两个分支。当unroll次数为4次时，会产生2⁴ = 16个路径分支，通常每个分支都需要单独编译，因此会大量增加编译时间和编译出的代码量。为了支持关键算子FA的编译优化，提供了两个特殊的函数`pypto.is_loop_begin()`和`pypto.is_loop_end()`用于优化条件分支。

7. 考虑到对外层loop进行`loop_unroll`不能提升loop body的大小，当前仅支持最内侧循环进行unroll。

8. 隐式loop示例：框架会在编译阶段前端隐式地在function开始的位置插入一个`loop(1)`。循环直到下一个循环开始前结束：

   ```python
   @pypto.frontend.jit
   def foo(a, b, c):
       c[:] = a + b
   # 等价于
   @pypto.frontend.jit
   def foo(a, b, c):
       for i in pypto.loop(1):
           c[:] = a + b

   @pypto.frontend.jit
   def foo(a, b, c):
       t = a + 1
       for i in pypto.loop(1):
           c[:] = t + b
   # 等价于
   @pypto.frontend.jit
   def foo(a, b, c):
       for i in pypto.loop(1):
           t = a + 1
       for i in pypto.loop(1):
           c[:] = t + b
   ```

   框架当前不会自动进行`loop(1)`合并，因此在实际使用中，建议用户手动合并`loop(1)`，以提高效率。
