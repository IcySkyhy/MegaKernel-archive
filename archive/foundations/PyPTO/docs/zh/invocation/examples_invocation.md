# 样例运行

## 仿真环境（无NPU真实硬件）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd examples/00_hello_world
python3 hello_world.py --run_mode=sim
```

## 真实可运行环境（有NPU真实硬件）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

更多示例请参考`examples/`目录下的示例代码。

## 结果查看

该基础样例运行成功后，在`${work_path}/output/`目录下生成编译和运行产物，相关产物包括[计算图](../tutorials/appendix/glossary.md)和[泳道图](../tutorials/appendix/glossary.md)，计算图和泳道图可通过PyPTO配套的ToolKit插件，在VS-CODE中查看并与代码关联，相关ToolKit使用请参考[快速入门-查看计算图](../tutorials/introduction/quick_start.md#查看计算图)、[快速入门-查看泳道图](../tutorials/introduction/quick_start.md#查看泳道图)

## 快速开始

以下是一个简单的PyPTO使用示例，可以通过`--run_mode`参数指定运行仿真示例或者真实环境示例:

```python
import pypto
import torch
import argparse

shape = (1, 4, 1, 64)

# 根据运行模式创建计算内核
def create_add_kernel(run_mode: str):
    mode = pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = x + y

    return add_kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu", "sim"])
    args = parser.parse_args()

    # 准备输入数据
    device = "cpu"
    if args.run_mode == "npu":
        import torch_npu
        torch.npu.set_device(0)
        device = "npu:0"

    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)

    # 执行计算并查看结果
    create_add_kernel(args.run_mode)(x, y, output)
    print(f"Output shape: {output.shape}")
```

- 对于真实环境或者精度仿真，可以直接通过查看输出张量的值查看运行结果
- 对于性能仿真，通过`output/`下的泳道图查看仿真结果

完整样例请参考：[hello_world.py](../../../examples/00_hello_world/hello_world.py)。
