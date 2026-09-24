# 未设置执行算子的设备ID

## 问题现象描述

在昇腾AI处理器上执行算子时，出现失败，报错信息如下：

```text
2025-12-17 14:31:32.491 E | fail get device id, check if set device id
2025-12-17 14:31:32.492 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | aclmdlRICaptureGetInfo failed, return[100000]
```

## 可能原因

用户定义的算子未使用`@jit`进行装饰，且未使用TorchNPU接口显式设置当前算子执行的Device ID。

## 处理步骤

在算子执行前设置Device ID，例如：

```python
def test_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))  # 从环境变量获取期望执行的device id
    torch.npu.set_device(device_id)  # 显式设置device id
    ....
```
