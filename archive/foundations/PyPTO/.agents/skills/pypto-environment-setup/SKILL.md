---
name: pypto-environment-setup
description: PyPTO 环境安装与环境问题修复，包括 CANN、torch_npu、编译工具链、第三方依赖和 PyPTO 编译运行等。触发词：PyPTO environment setup, CANN install, torch_npu, NPU environment, Ascend toolkit, compile PyPTO, build PyPTO, NPU driver, prepare_env, diagnose environment, fix import error, torch_npu import fail, DT_FP8E8M0, pto-isa, ASCEND_HOME_PATH, npu-smi, softmax verify, pip dependency conflict
---

# PyPTO Environment Setup

## 约定

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
```

- **默认版本**：CANN 9.0.0 + PyTorch/torch_npu

| CANN 版本 | torch 推荐版本 | torch_npu 推荐版本 |
|-----------|---------------|-------------------|
| 9.0.0 | 2.8.0 | 2.8.0.post4 |

**版本策略**：
- 新环境（未安装 torch/torch_npu）：优先安装推荐版本（torch 2.8.0 + torch_npu 2.8.0.post4）
- 已安装环境：
  - 版本 ≥ 推荐版本：保持不变，无需降级
  - 版本 < 推荐版本：升级至推荐版本

## ⛔ 隐私保护

> ⚠️ **禁止在屏幕、日志、错误信息中打印 `GITCODE_TOKEN` 环境变量**
>
> 克隆私有仓库时使用 Token 认证，请确保 Token 仅存储在安全位置（环境变量或配置文件），不要在终端输出中暴露。

## 工作流程

> ⚠️ **环境保护**：任何安装前必须先检测当前状态，向用户展示已有组件及版本，获得确认后才执行变更。

### 步骤 1：环境检测

```bash
# 步骤 1.1：检查 PYPTO_REPO 环境变量
echo "PYPTO_REPO: ${PYPTO_REPO:-未设置}"

# 步骤 1.2：检查当前目录是否为 PyPTO 仓库
[ -f "pyproject.toml" ] && [ -d "framework" ] && echo "✓ 当前目录是 PyPTO 仓库"
```

**PyPTO 仓库查找逻辑**（按优先级自动执行）：
1. 环境变量 `$PYPTO_REPO`
2. 当前目录（含 `pyproject.toml` + `framework/`）
3. `$PWD` 下搜索（最多 3 层），找到多个时提示用户选择
4. 未找到 → 自动克隆：

```bash
git clone https://gitcode.com/cann/pypto.git "$PWD/pypto"
export PYPTO_REPO="$PWD/pypto"
```

**运行环境诊断**：
```bash
# 步骤 1.3：运行诊断脚本
python3 scripts/diagnose_env.py --checklist
```

**通过标准**：清单所有项 ✅ OK。⚠️/❌ 项需修复后再继续。

**检测 NPU 设备类型**：
```bash
# 步骤 1.4：检测芯片类型（用于确定 --device-type 参数）
# 方法：通过 lspci 查看设备 ID
lspci | grep -i "acc" | grep -oE "d80[236]"

# 设备 ID 对应关系：
# - d802: Ascend 910B (a2)
# - d803: Ascend 910C/Atlas A3 (a3)
# - d806: Ascend 950 - 暂不支持
```

**设备类型映射规则**：
| 检测结果 | 设备类型 | ops 包名称 |
|---------|-------------------|-----------|
| npu-smi 显示 910b/910B | a2 | Ascend-cann-910b-ops |
| npu-smi 显示 910c/910C | a3 | Ascend-cann-A3-ops |
| lspci 显示 d802 | a2 | Ascend-cann-910b-ops |
| lspci 显示 d803 | a3 | Ascend-cann-A3-ops |

> ⚠️ **重要**：设备类型错误会导致 ops 包不匹配，运行时出现 `ZerosLike ADD_TO_LAUNCHER_LIST_AICORE failed` 等错误。

### 步骤 2：决策分支

- 0 个问题 → 跳至步骤 4 验证
- 有问题 → 向用户展示问题清单，获得确认后进入步骤 3
- 用户拒绝 → 停止，报告当前状态

### 步骤 3：按类别修复

只修复缺失项，不改动已正常组件。

**⚠️ 重要：NPU 环境 + CANN 缺失时，必须按顺序执行以下 5 步，全部完成后才能进入步骤 4**

| 问题类别 | 修复操作 | 失败回滚 |
|---------|---------|---------|
| **NPU 环境 + CANN 缺失** | **按顺序执行以下 5 步**：<br><br>**步骤 3.1：安装编译依赖**<br>`cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2\|a3>`<br><br>**步骤 3.2：下载安装第三方源码包**<br>`bash tools/prepare_env.sh --quiet --type=third_party`<br><br>**步骤 3.3：下载 CANN 包**<br>`script -q -c "bash tools/prepare_env.sh --quiet --type=cann --only-download --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.download.log`<br><br>**步骤 3.4：安装 CANN**<br>`script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.install.log`<br><br>**步骤 3.5：验证 CANN 安装**<br>检查 CANN 安装日志，未完成则安装（见下方代码块）<br><br>⚠️ **全部 5 步完成后才能进入步骤 4** | 检查网络连通性和目录写入权限；见 `troubleshooting.md` |
| **编译工具链缺失** (cmake/gcc/make/g++/ninja/pip3) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps` | 回退到手动 `apt-get install`；检查 apt 源配置 |
| **第三方源码包缺失** (json/libboundscheck) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party` | 检查网络对 cann-src-third-party 的可达性 |

**CANN 安装检查脚本**：
```bash
cd ${PYPTO_REPO:-$PWD}
# 检查 CANN 安装日志，未完成则安装
if ! grep -q "Successfully installed CANN packages." prepare_env.cann.install.log 2>/dev/null; then
    echo "CANN 未安装完成，正在安装..."
    script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.install.log
else
    echo "CANN 已安装完成"
fi
```

> `--device-type` 由步骤 1.4 检测确定，必须使用正确的设备类型。
> 手动安装/编译细节见 [📋 prepare_environment.md](references/prepare_environment.md)。
> 遇到报错见 [🔧 troubleshooting.md](references/troubleshooting.md)。
> ```bash
> pip install torch==2.8.0 torch-npu==2.8.0.post4
> ```

### 步骤 4：验证和编译（必须执行）

任何安装/修复后必须通过 softmax 验证。

#### 步骤 4.1：加载 CANN 环境
```bash
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh
```

#### 步骤 4.2：检查可用的 NPU 卡 (务必在加载 CANN 环境后检查)

```bash
# 获取可用卡总数（确保不遗漏）
python3 -c "import torch_npu; print('NPU device count:', torch_npu.npu.device_count())"

# 查看卡详情（注意：输出可能被截断，以上方 device_count 为准）
npu-smi info
```

> ⚠️ `npu-smi info` 在单机多卡场景下输出可能被截断，导致低估可用卡数。以 `torch_npu.npu.device_count()` 结果为准。

使用本 skill 的空闲卡检测脚本 `scripts/list_idle_chip_ids.sh` 确定空闲 chip id。

#### 步骤 4.3：设置环境变量

**步骤 4.3.1：设置 NPU 设备 ID**
```bash
# 设置 NPU 设备 ID（根据步骤 4.2 查找空闲 chip id）
export TILE_FWK_DEVICE_ID=<空闲 chip id>
```

**步骤 4.3.2：设置并验证 PTO-ISA 路径**

> ⚠️ **高频断裂点**：CANN 内置 PTO-ISA 版本可能与 PyPTO 源码不兼容（缺少 `pto::FmodSAlgorithm`/`RemSAlgorithm`/`ExpAlgorithm`/`DivAlgorithm` 等枚举），导致运行时 kernel 编译失败（`no member named 'XXX' in namespace 'pto'`）。**必须执行下方兼容性检查，不可跳过**。

```bash
# 1. 先使用 CANN 内置路径（默认）
arch=$(uname -m)   # 常见值：x86_64 或 aarch64
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/${arch}-linux

# 2. 验证兼容性：检查关键枚举是否存在，任一缺失即需切换到源码方式
check_items=("FmodSAlgorithm" "RemSAlgorithm" "DivAlgorithm" "ExpAlgorithm" "SqrtAlgorithm")
missing_items=""
for item in "${check_items[@]}"; do
    if ! grep -rq "${item}" "${PTO_TILE_LIB_CODE_PATH}/include/pto/" 2>/dev/null; then
        missing_items="${missing_items} ${item}"
    fi
done

if [ -n "${missing_items}" ]; then
    echo "⚠️ PTO-ISA 不兼容，缺少枚举: ${missing_items}，切换到源码方式"
    PTO_ISA_SRC="${PTO_ISA_SRC:-${PYPTO_REPO}/pto-isa}"
    if [ -d "${PTO_ISA_SRC}/include/pto" ]; then
        cd "${PTO_ISA_SRC}" && git pull origin master
    else
        git clone https://gitcode.com/cann/pto-isa.git "${PTO_ISA_SRC}"
    fi
    export PTO_TILE_LIB_CODE_PATH="${PTO_ISA_SRC}"
else
    echo "✓ PTO-ISA 兼容"
fi
```

> ⚠️ **注意**：`pip install pypto` 成功不代表运行时没问题。Host 侧编译和 Device 侧 kernel 编译使用不同的头文件来源，PTO-ISA 版本不匹配只会在 **运行时 kernel 编译** 阶段暴露。

**步骤 4.3.3：生成 env_setup.sh**

将步骤 4.3.1 和 4.3.2 的结果写入 `env_setup.sh`：

```bash
cat > env_setup.sh << "EOF"
#!/bin/bash
# 自动生成的环境配置文件
# 加载 Ascend 基础环境（根据实际安装路径调整）
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh

# 动态获取当前架构（确保生成的脚本在不同机器上仍正确）
arch=$(uname -m)
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/${arch}-linux

echo "env_setup.sh 加载完成：TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID}, PTO_TILE_LIB_CODE_PATH=${PTO_TILE_LIB_CODE_PATH}"
EOF
```
> **必须用 `source` 执行**环境配置脚本：`source env_setup.sh`

> - \$\{arch\}：CPU 架构，如 aarch64、x86_64.
> - 如果运行 softmax 验证时出现 `no member named 'XXX' in namespace 'pto'` 错误，说明 CANN 内置 PTO-ISA 版本过旧，需要切换到源码方式（见步骤 4.6 失败排查）。

#### 步骤 4.4：安装 torch 和 torch_npu
```bash
pip install torch==2.8.0 torch-npu==2.8.0.post4
```

#### 步骤 4.5：编译安装 PyPTO

```bash
# Source 环境变量文件
source env_setup.sh

# 如果有 pypto 先删除本地 pypto
pip uninstall pypto -y
cd ${PYPTO_REPO:-$PWD}
rm -rf build_out
# 编译产出 whl
python3 build_ci.py --clean --no_isolation
# 安装到当前环境
bash build_out/cann-pypto_*.run --full -q --pylocal
```

#### 步骤 4.6：运行测试验证
```bash
# Source 环境变量文件
source env_setup.sh

# NPU 模式（有 NPU 环境时必须使用）
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu

# SIM 模式（非 NPU 环境时使用）
# python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode sim

```

⚠️ **注意**：NPU 环境必须使用 NPU 模式通过验证。若超过 120 秒未完成，可视为卡死，执行时可以加上timeout 120 说明环境配置有问题，参照[troubleshooting.md](references/troubleshooting.md) 排查。

**通过标准**：退出码 `0`，输出 `Softmax test passed`。

**失败排查**（按错误类型对照）：

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `no member named 'XXX' in namespace 'pto'`（如 `FmodSAlgorithm`、`RemSAlgorithm`、`ExpAlgorithm`、`DivAlgorithm`） | PTO-ISA 版本过旧，CANN 内置头文件缺少枚举定义 | 按步骤 4.3.2 克隆 pto-isa 源码并设置 `PTO_TILE_LIB_CODE_PATH`，然后 `source env_setup.sh` 重跑 softmax |
| `COMPILE_CODE_FAILED` / kernel 编译失败 | 多种可能，先看具体编译错误信息 | 检查上方具体错误类型 |
| 其他错误 | 重新运行步骤 1 诊断 | 对照 [🔧 troubleshooting.md](references/troubleshooting.md) 排查 |

> ⚠️ **注意**：`pip install pypto` 成功不代表运行时没问题。Host 侧编译和 Device 侧 kernel 编译使用不同的头文件来源，PTO-ISA 版本不匹配只会在 **运行时 kernel 编译** 阶段暴露。

### 步骤 5：完成报告

环境配置完成后，运行诊断并整理报告。

**运行诊断**：
```bash
cd ${PYPTO_REPO:-$PWD}
python3 scripts/diagnose_env.py --checklist
```

**报告模板**：
```
=====================================
PyPTO 环境配置
=====================================
CANN 版本:  9.0.0
NPU 芯片:   Ascend910 (A2/A3)
Python:     3.10.x
torch:      2.8.0
torch_npu:  2.8.0.post4
pypto:      ✅ 已安装
PTO-ISA:    CANN 内置 / pto-isa 源码 (gitcode.com/cann/pto-isa) [兼容性: ✅/⚠️]
可用 NPU 卡: <device_count> 卡
=====================================

验证结果:   Softmax（NPU 模式）✅ 通过

过程问题总结：
  - <问题> -> <解决方案>

⚠️ `TILE_FWK_DEVICE_ID` 需根据 `npu-smi info` + `torch_npu.npu.device_count()` 输出修改。
⚠️ `npu-smi info` 输出可能被截断，以 `torch_npu.npu.device_count()` 为准。
```

## 📚 参考文件

| 文件 | 内容 |
|------|----------|
| [📋 prepare_environment.md](references/prepare_environment.md) | 安装 CANN/torch_npu/pto-isa、编译 PyPTO |
| [🔧 troubleshooting.md](references/troubleshooting.md) | 安装/导入/运行报错 |
## 外部参考

- PyPTO: https://gitcode.com/cann/pypto
- CANN 安装指南: https://www.hiascend.com/document/redirect/CannCommunityInstSoftware
- Ascend Extension for PyTorch: https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html
