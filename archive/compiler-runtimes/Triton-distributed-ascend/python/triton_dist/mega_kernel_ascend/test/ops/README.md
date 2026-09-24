# mega_kernel_ascend ops 测试指南

本文说明如何运行 `test_two_matmul.py`（以及同目录其它 ops 用例）。

## 结论（先看）

- **`mega_kernel_ascend` 目录本身不需要单独编译**，它是 Python 适配层（stub CUDA/nvshmem、把 `ModelBuilder` 接到 NPU）。
- 需要先按仓库根目录 [README](../../../../../README.md) 装好 **CANN + torch_npu + 本仓 `./python`（含 triton-ascend / shmem）**。
- 测试里的 `builder.compile()` 是**运行时** codegen + Triton JIT，不是预编译 `mega_kernel_ascend` 包。
- 完整用例需要 **Ascend NPU**；仅 OpGraph 字节比对可在有 stub/假设备的环境做（见文末）。

## 依赖

| 项 | 要求 |
|----|------|
| 硬件 | Ascend NPU（如 910B/910C） |
| CANN | ≥ 8.5（仓库 README 亦写 9.1+）；需 `source set_env.sh` |
| Python | ≥ 3.8 |
| PyTorch + torch_npu | 与 CANN 匹配；根目录 `requirements.txt` 含 `torch-npu==2.7.1.post4` |
| 本仓 | `pip install` / `-e` 安装 `./python` |
| 本用例文件 | 同目录 `two_matmul_opgraph.expected.bin`（已入库） |

`test_two_matmul` 额外只用到：`torch`、`triton`（随本仓）、`triton_dist.mega_kernel_ascend`、`mega_triton_kernel.core.op_graph`。

## 环境准备（可复制）

在 **NPU Linux 机器**、仓库根目录执行。路径按本机修改。

```bash
# CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若已按根目录 README 安装 LLVM：
# export LLVM_INSTALL_PREFIX=/path/to/llvm-install

cd /path/to/Triton-distributed-ascend

# 子模块
git submodule update --init --depth=1
(cd 3rdparty/triton-ascend && git submodule update --init --depth=1)

# Python 依赖（PyTorch / torch_npu 请先按机子版本装好）
pip install -r requirements.txt

# shmem（whl 名以 dist 实际产物为准）
(cd 3rdparty/shmem && bash scripts/build.sh -python_extension && pip install dist/shmem-*.whl)

# 安装本仓（开发模式）
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CLANG_LLD=ON \
TRITON_BUILD_PROTON=OFF \
TRITON_BUILD_LITTLE_KERNEL=OFF \
pip install -e ./python --verbose --no-build-isolation
```

自检：

```bash
npu-smi info
python -c "import torch; print('npu', torch.npu.is_available(), torch.npu.device_count())"
python -c "from triton_dist.mega_kernel_ascend import ModelBuilder; print(ModelBuilder)"
```

## 运行 test_two_matmul

```bash
# 仓库根目录，且已 source CANN env
python -m triton_dist.mega_kernel_ascend.test.ops.test_two_matmul
```

可选参数：

```bash
python -m triton_dist.mega_kernel_ascend.test.ops.test_two_matmul --intra_kernel_profile
python -m triton_dist.mega_kernel_ascend.test.ops.test_two_matmul --enable_runtime_scheduler
```

成功时大致输出：

```text
[OK] OpGraph matches two_matmul_opgraph.expected.bin (422 bytes)
[OK] test_two_matmul passed: mega-kernel (A@B)@weight matches torch on NPU (... SMs, 30 iters).
```

### 用例在做什么

1. `declare_tensor` + 两次 `make_matmul`（执行走 `linear`，导出 OpGraph 为 `Matmul`）。
2. `build_opgraph()` 与 `two_matmul_opgraph.expected.bin` **字节一致**。
3. `compile()` → `run()` × 30，与 `torch.matmul` 链式结果对比（`atol=0, rtol=0`）。

形状：`A[128,256] @ B[256,128] @ weight[128,256]`，`bfloat16`，device=`npu:0`。

## 同目录其它用例

```bash
python -m triton_dist.mega_kernel_ascend.test.ops.test_add
python -m triton_dist.mega_kernel_ascend.test.ops.test_mlp_layer
```

`test_mlp_layer` 默认 `torch.npu.set_device(1)`，按机器改卡号。

## 仅验证 OpGraph（无完整 NPU JIT）

完整 `test_two_matmul` 必须在 NPU 上跑。若只想核对导出字节，可用仓库外脚本（开发机 stub）：

```bash
# 在 kernel 工作区根目录（含 .venv-dump 与 dump/scripts）
.venv-dump/Scripts/python.exe verify_two_matmul.py
```

该脚本 stub 掉 `compile`，比对 `two_matmul_opgraph.expected.bin`（422 bytes）。**不能**替代 NPU 上的数值 / JIT 验证。
