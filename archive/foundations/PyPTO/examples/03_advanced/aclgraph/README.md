# 图捕获模式算子开发示例 （Graph Capturing Mode）

本示例在自定义的`Softmax`算子基础上，展示了如何开启图捕获模式，以及执行捕获图。

## 总览介绍

随着性能优化不断深入，`Eager`模式带来的`Host`侧开销逐渐成为瓶颈。本目录目前聚焦于：
- **ACLGraph**：通过图捕获模式（`ACLGraph`）将相关任务下沉到Device上并执行，减少`Host`侧开销，当捕获图需要多次执行时，无需再下发任务，只需要多次调用`replay`接口即可。

## 样例代码特性

- **装饰器修饰**：PyPTO 自定义算子对接 `torch.compile`，需要使用` @allow_in_graph` 装饰器修饰，避免图分割报错。
- **FakeTensor处理**  PyPTO 自定义算子对接 `torch.compile`，需要判断输入是否为`FakeTensor`，若是则直接返回空`tensor`。
- **编译模型**： 调用` torch.compile` 编译模型`model`。
- **图捕获**：使用` with torch.npu.graph(g)：` 捕获`model`第一次执行的任务。
- **执行捕获图** 调用` replay` 执行捕获图，计算` Softmax` 结果；

## 代码结构

- **`aclgraph/`**:
  - `aclgraph.py`: 包含图捕获模式开启、捕获图执行以及与 PyTorch 原生算子的精度比对。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
# 安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
# 上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
python3 examples/03_advanced/aclgraph/aclgraph.py
```


## 注意事项
- 当捕获图需要多次执行时，需要更新对应输入数据。
- 在捕获过程中，调用内存同步操作类函数是非法的，会检验失败导致捕获失败。
