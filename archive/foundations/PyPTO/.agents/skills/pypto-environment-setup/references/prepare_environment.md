# 环境准备与编译安装

## 安装顺序

1. 驱动/固件（见[官方 CANN 安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)，本技能不执行）
2. CANN toolkit + ops 包
3. pto-isa 源码
4. 加载环境变量
5. PyTorch + torch_npu
6. PyPTO 编译安装

## CANN 环境加载（通用模板）

> 以下代码块在本文件中只出现一次，其他文件统一引用此处。

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
CANN_ENV_SH=$(ls -1 "${ASCEND_INSTALL_PATH}"/*/set_env.sh "${ASCEND_INSTALL_PATH}"/*/*/set_env.sh 2>/dev/null | head -1)
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || echo "CANN set_env.sh not found"
```

加载后关键变量：

| 变量 | 说明 |
|------|------|
| `ASCEND_HOME_PATH` | CANN 版本根目录（PyPTO 主要依赖此变量） |
| `ASCEND_OPP_PATH` | OPP 算子包路径 |


> 诊断脚本路径探测优先级：`ASCEND_HOME_PATH` → `ASCEND_OPP_PATH` 反推 → 兜底扫描 `cann-*`

## 使用 prepare_env.sh 安装（推荐）

```bash
cd "$PYPTO_REPO"

> ⏱️ **说明**：该脚本大约需要 15 分钟执行完成。

# 分步安装（禁止 --type=all）
bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2|a3>
bash tools/prepare_env.sh --quiet --type=third_party
# 先下载 CANN 包
script -q -c "bash tools/prepare_env.sh --quiet --type=cann --only-download --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.log
# 再安装 CANN 包
script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.log


# 仅 CANN
# 先下载 CANN 包
script -q -c "bash tools/prepare_env.sh --quiet --type=cann --only-download --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.log
# 再安装 CANN 包
script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.log

# 仅编译工具链
bash tools/prepare_env.sh --quiet --type=deps

# 仅第三方源码包
bash tools/prepare_env.sh --quiet --type=third_party
```

| 参数 | 说明 |
|------|------|
| `--type` | `deps` / `cann` / `third_party` |
| `--device-type` | `a2`(910B) / `a3`(910C) |
| `--install-path` | CANN 安装路径，默认 `/usr/local/Ascend` |
| `--quiet` | 静默模式，强烈建议始终加上 |

> 手动安装 CANN 的详细步骤见 `$PYPTO_REPO/docs/zh/install/prepare_environment.md`

## pto-isa 获取

### CANN 内置方式（默认）

CANN toolkit 安装后自带 PTO-ISA，路径为 `$ASCEND_HOME_PATH/<arch>-linux`（如 `/usr/local/Ascend/cann-9.0.0/aarch64-linux`）。通过 `export PTO_TILE_LIB_CODE_PATH=$ASCEND_HOME_PATH/<arch>-linux` 即可使用。

> ⚠️ CANN 内置 PTO-ISA 版本可能与 pypto 源码（master 分支）不兼容。运行时如果出现 `no member named 'XXX' in namespace 'pto'` 错误，说明内置版本过旧，需切换到源码方式。

### 源码方式（CANN 内置版本过旧时使用）

```bash
PTO_ISA_DIR=${PTO_ISA_DIR:-$PWD/pto-isa}
git clone https://gitcode.com/cann/pto-isa.git "$PTO_ISA_DIR"
export PTO_TILE_LIB_CODE_PATH="$PTO_ISA_DIR"
test -d "$PTO_TILE_LIB_CODE_PATH/include/pto" && echo OK
```

> 克隆后无需编译，直接设置 `PTO_TILE_LIB_CODE_PATH` 指向源码目录即可。切换后需重新运行 softmax 验证确认通过。

## PyPTO 编译安装

> 官方完整文档：`$PYPTO_REPO/docs/zh/install/build_and_install.md`

### 前提

- CANN 已安装且环境已加载
- `pip3 install -r $PYPTO_REPO/python/requirements.txt`
- 第三方源码包已准备（网络可达时自动下载；否则 `export PYPTO_THIRD_PARTY_PATH=<path>` 或 `prepare_env.sh --type=third_party`）

### 安装

```bash
cd "$PYPTO_REPO"

# 编译安装（推荐）
python3 build_ci.py -f python3 --clean --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall -q

# PyPI
pip install pypto
```

> 高级编译选项（Debug、CMake Generator 等）见 `$PYPTO_REPO/docs/zh/install/build_and_install.md`

## 环境变量速查

| 变量 | 设置方式 | 必须性 |
|------|----------|--------|
| CANN 环境 | `source set_env.sh`（见上方"CANN 环境加载"） | 每次新 shell |
| `PTO_TILE_LIB_CODE_PATH` | 默认 `$ASCEND_HOME_PATH/<arch>-linux`（CANN 内置），版本不匹配时使用 pto-isa 源码目录 | 编译/运行 |
| `TILE_FWK_DEVICE_ID` | `export TILE_FWK_DEVICE_ID=1` | NPU 模式 |
| `PYTHONPATH` | `export PYTHONPATH="${PYPTO_REPO}/python:$PYTHONPATH"` | 仅源码调试（不推荐日常使用） |
