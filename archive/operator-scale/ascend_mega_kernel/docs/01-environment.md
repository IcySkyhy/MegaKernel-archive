# 昇腾环境调研报告（方向1：硬件与软件环境清单）

> 调研时间：2026-09-26 · 主机：`09d612de5aab`（容器/overlay 环境）· 工作目录：`/workspace/ascend_mega_kernel`

## 总体结论（TL;DR）

本机为**单卡昇腾 950 系列 NPU（芯片名 Ascend950PR，板卡型号 A310-50-C00MM304A1，Atlas 950 系列）**，128 GiB HBM（`npu-smi -t memory` 报 131072 MB），**28 AIC + 56 AIV**（PG 降频 binning 版；满配 die 为 32 AIC —— **核数必须用 API 取、不许硬编码**，见 §0）；软件栈为 **Driver/npu-smi 25.7.rc1.6 + CANN 9.1.0（内置 bisheng 编译器，基于 clang 15.0.5）+ nnal(ATB) 9.1.0.B150**，Host 为 2 路 AMD EPYC 9355 + openEuler 24.03 SP3。**尚未安装 torch / torch_npu / vllm**，Python 环境仅有 CANN 自带的 3.12.13（62 个包全为 CANN 生态）。环境变量已通过 `/root/.bashrc` 自动加载。

## 0. 给 agent 的硬约束（先读；违反会写出"跑不起来"的方案）

| 约束 | 值 | 取证 | 一句话后果 |
|---|---|---|---|
| **核数** | 单卡 **28 AIC + 56 AIV**（AIC:AIV = 1:2；PG 降频 binning 版，满配 die 为 32 AIC） | `docs/05-megakernel-design.md` §2；本文 §1.3 的 `npu-smi -t common` **只报 AIC 数 28**，AIV 数须由平台/API 给出 | 1:2 mix 的核分工与配对同步数会全错 |
| **核数取值方式** | **必须用 API 取：`PlatformAscendC::GetCoreNumAic()` / `GetCoreNumAiv()`；blockDim 用 `CalcTschNumBlocks(aicCoreNum, aivCoreNum)`。禁止硬编码** | `docs/05` §2 | 硬编码的核数假设在别的 SKU（32 AIC 满配）上**直接跑不起来或大量核空转**；另注意 `__mix__(1,2)` 下 AIV 侧 `GetBlockNum()` 返回的是 AIC 数（`docs/05` §6.1） |
| **HBM** | 128 GiB（`npu-smi -t memory`：131072 MB），无 DDR | 本文 §1.3 | 整模型 **169.7 GiB 放不进 HBM**，必须按层流式 |
| **host 内存** | cgroup 上限 **32 GiB**（cgroup v1，docker `09d612de5aab`；`memory.limit_in_bytes` = 34359738368；memsw 64GB） | 本文 §3.1（含 2026-09-26 复核） | 编译并行度不控制会 **cgroup OOM**；整模型也放不进 host 内存（单层 ~1.4 GiB 可以） |
| **磁盘** | 按 **300 GB** 总量规划（平台配额，**用户口径**）；实测 `/workspace` 724G / 可用 473G 且未被限流 ⇒ 配额口径未明 | 本文 §3.1 | 堆大文件可能撞配额；写前先 `df -h` + `du -sh` 确认口径 |
| **软件栈** | CANN **9.1.0**（bisheng / clang 15.0.5）+ **nnal(ATB) 9.1.0.B150** + `cxx_abi=1`；Driver/npu-smi 25.7.rc1.6 | 本文 §2 | 按别的 CANN 版本的 API 写代码会踩头文件差异（例：`LoadL0_2D` 这类符号在 9.1.0 里**根本不存在**） |

---

## 1. NPU 硬件（来源：`npu-smi` 系列命令）

### 1.1 概览

`npu-smi info`（摘录）：

```text
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1.6                               Version: 25.7.rc1.6                            |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | OK            | 195.6       56                0     / 0             |
|        |                  | 0000:61:00.0  | 0           0    / 0          5239  / 131072        |
+========+==================+===============+=====================================================+
| No running processes found in NPU 0                                                             |
```

### 1.2 板卡信息（`npu-smi info -t board -i 0`，注意：`-t` 子命令必须带 `-i <卡号>`）

```text
NPU ID                         : 0
Product Name                   : A310-50-C00MM304A1
NPU Name                       : 9579
Chip Name                      : Ascend950PR
Chip Version                   : V100
Model                          : A310-50-C00MM304A1
Serial Number                  : 10264L540872
Software Version               : 25.7.rc1.6
Firmware Version               : 9.0.0.105.229
Compatibility                  : OK
Board ID                       : 0x1a
Main Board ID                  : 0x68(1P)          # 1P = 单芯片封装形态
PCIe Bus Info                  : 0000:61:00.0
PCI Vendor ID                  : 0x19E5
PCI Device ID                  : 0xD806
Chip Count                     : 1
```

> 结论：**确为昇腾 950 芯片（Ascend950PR），整卡仅 1 颗芯片**。产品型号 `A310-50-C00MM304A1` 属 Atlas 950 系列板卡。

### 1.3 显存与算力（`-t memory` / `-t common`）

```text
# npu-smi info -t memory -i 0
DDR Capacity(MB)               : 0
HBM Capacity(MB)               : 131072        # 即 128 GB HBM，无 DDR 内存
HBM Clock Speed(MHz)           : 3200
HBM Temperature(C)             : 51

# npu-smi info -t common -i 0
Aicore Usage Rate(%)           : 0
Aicore Freq(MHZ)               : 1650
Aicore Count                   : 28            # 28 个 AI Core（= AIC）
Temperature(C)                 : 56
NPU Real-time Power(W)         : 195.7
```

> **注意（核数）**：`npu-smi` 只报 **AIC** 数（28）。本机核配置是 **28 AIC + 56 AIV**（AIC:AIV = 1:2；PG 降频 binning 版，满配 die 为 32 AIC）——依据 `docs/05-megakernel-design.md` §2。**核数一律用 `PlatformAscendC::GetCoreNumAic()` / `GetCoreNumAiv()` 取，禁止硬编码**（后果见 §0）。

### 1.4 拓扑与备注

- `npu-smi info -l`：`Total Count: 1`，单卡无互联拓扑。
- 注意：`-t cpu` 不是合法子命令（返回 `Error parameter of -t`）；合法 type 列表含 board/common/flash/memory/usages/temp/power/health/ecc/sensors/top 等。
- 设备节点：容器内可见 `/dev/davinci1` 和 `/dev/davinci_manager`（注意编号是 1 而非 0）。
- `lsmod | grep -i ascend` 无输出（容器内看不到宿主机内核模块，属正常现象，npu-smi 工作正常即可）。

---

## 2. CANN 安装与版本（来源：`/usr/local/Ascend` 目录、`*version.info`）

### 2.1 安装布局

`/usr/local/Ascend` 下内容（`ls /usr/local/Ascend`）：

```text
ascend-toolkit   cann -> cann-9.1.0   cann-9.1.0   driver   nnal
```

- CANN 安装根：`/usr/local/Ascend/cann-9.1.0`（`cann` 为符号链接；`ascend-toolkit/latest` 也指向它）
- Driver：`/usr/local/Ascend/driver`
- nnal（ATB/asdsip）：`/usr/local/Ascend/nnal`
- `~/Ascend` 不存在，无用户态第二安装。

### 2.2 版本信息

`/usr/local/Ascend/cann-9.1.0/x86_64-linux/ascend_toolkit_install.info`：

```text
package_name=Ascend-cann-toolkit
version=9.1.0
innerversion=V100R001C11SPC001B243
arch=x86_64
path=/usr/local/Ascend/cann-9.1.0
```

- `compiler/version.info`：`Version=9.1.0`，`timestamp=20260730_231653901`（2026-07-30 构建）
- `opp/version.info`：`Version=9.1.0`

### 2.3 内置编译器（bisheng / ccec）

bisheng 编译器位于 `tools/bisheng_compiler/bin/` 与 `tools/ccec_compiler/bin/`（`x86_64-linux/bin/` 下亦有软链）。`ccec --version` 输出：

```text
2026-07-30T20:53:21+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin
```

> 结论：**ccec/bisheng 实际为基于 clang 15.0.5 的毕昇编译器**（`tools/ccec_compiler/bin/ccec` 与 `bisheng_compiler/bin/ccec` 版本一致）。

相关工具链二进制（`ls /usr/local/Ascend/cann-9.1.0/x86_64-linux/bin/`）：`atc`、`atc.bin`、`bisheng`、`bisheng-tune`、`bishengir-compile(-a5)`、`bishengir-opt(-a5)`、`ccec`、`cce-ld`、`hivmc(-a5)`、`asc_opc`、`ascendc_pack_kernel`、`cannsim`、`lld` 系列等。

### 2.4 HCCL

- 无独立 `hccl` 命令行二进制（HCCL 是通信库而非 CLI）。
- 库文件：`/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libhccl.so`（含 `ops_hccl` 符号，版本随 CANN 9.1.0）。
- Python 包：`hccl 0.1.0`，位于 CANN 自带 `python/site-packages/hccl`（经 PYTHONPATH 导入，3.11/3.12 均验证 `import hccl` 成功）。
- 压测工具源码目录：`/usr/local/Ascend/cann-9.1.0/tools/hccl_test/`（含 Makefile、hostfile、opbase_test）。

### 2.5 nnal（ATB / asdsip）

`/usr/local/Ascend/nnal/atb/9.1.0/version.info`：

```text
Ascend-cann-atb : 9.1.0
Ascend-cann-atb Version : 9.1.0.B150
branch : br_release_cann_9.1.0_20261223
```

`asdsip` 同为 9.1.0.B150。`nnal/atb/9.1.0/whl/` 下有 `torch_atb-0.0.1+abi1-cp310/cp311-none-any.whl`，**未安装**。

### 2.6 Driver

`/usr/local/Ascend/driver/version.info`：`Version=25.7.rc1.6`，`Innerversion=V100R001C10SPC105B220`，构建日期 2026-07-17；`/etc/ascend_install.info` 显示 `Driver_Install_Status=complete`。

---

## 3. Host 平台（来源：`uname -a`、`lscpu`、`/etc/os-release`、`free -g`、`df -h`）

- **内核/架构**：`Linux 6.6.0-132.0.0.111.oe2403sp3.x86_64`，x86_64
- **CPU**：AMD EPYC 9355 32-Core × 2 路，共 64 核 128 线程（SMT2），2 个 NUMA 节点，主频 1.5–4.4 GHz，支持 AVX-512
- **OS**：openEuler 24.03 (LTS-SP3)
- **内存（宿主视角）**：约 754 GB（`free -g` total 754），swap 3 GB
- **磁盘（宿主视角）**：`/` 为 8.7T overlay，已用 93%

⚠️ 以上是宿主机的观测值。**本容器实际受 cgroup 限制，可用资源远小于此**，详见下节。

### 3.1 容器资源限制（重要，agent 必读）

本容器运行在 cgroup（v1，docker 容器 `09d612de5aab`）中，资源上限由平台强制，**`free`/`df` 显示的是宿主机数字，不代表本容器可用额度**：

| 资源 | 容器上限 | 实测/来源 | 说明 |
|---|---|---|---|
| 内存 | **32 GB** | `cat /sys/fs/cgroup/memory/memory.limit_in_bytes` = 34359738368 | rss+page cache 合计计入；memsw（内存+swap）上限 64GB |
| 内存当前占用 | ~29.8 GB / 32 GB | `memory.usage_in_bytes` = 29782708224（2026-09-26 复核） | 其中 **rss 仅 ~1GB**，其余为 page cache（可回收）；cache 顶满时新分配会触发回收抖动，编译/构建大工程要控制并行度 |
| 磁盘 | **总共 300 GB**（平台配额，用户确认） | 容器内无法直接观测配额，df 显示的 8.7T 是宿主机 overlay 池 | **2026-09-26 复核**：`/workspace` 实测 724G（其中模型 170G、shared_assets ~520G），`df /workspace` 可用 473G，**均已超过标注的 300GB 配额**且容器未被限流 ⇒ 配额要么另有口径、要么不含共享存储（仍无权威结论）；写文件前先 `df -h` + `du -sh` 确认目标路径配额口径 |

**对开发活动的约束**：

- 编译 CANN 算子/大工程时限制并行任务数（如 `make -j4` 起步观察内存），避免触发 cgroup OOM。
- 权重按**层**流式处理没问题，不要整模型加载：整模型 **169.7 GiB** 既放不进 32GB host 内存，也放不进 128 GiB HBM。**单层只有 ~1.4 GiB**，最大的单个张量 800 MiB（MoE 专家 `gate_up_proj` U8 `[512,1280,1280]`），所以按层 pread/mmap + 双缓冲完全没有 host 内存压力。
- **但有一块例外必须单独设计**：PLE ngram 表 `layers.1.ple.ple_embedding.ngram_embedding.shard_*` 共 **128 个分片 / 95.4 GiB**（词表 ~2.5M ×160 ×bf16，每个分片 762.9 MiB），占整模型 56%，**不可能常驻**（HBM 也装不下），只能做稀疏行查找 + 磁盘驻留。另 `layers.1.ple.{conv1d,key_proj,value_proj,norm_*}.weight` 与 `ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}` 是小张量。
- 写盘前确认目标路径是否计入 300GB 配额；`/workspace` 下除模型下载外避免堆积大文件。

---

## 4. Python 环境（来源：`python3 --version`、`pip3 list`）

| 解释器 | 版本 | pip | 说明 |
|---|---|---|---|
| `/usr/bin/python3`（默认，`which python3`） | 3.11.6 | **无 pip**（`No module named pip`） | 系统自带 |
| `/usr/local/python3.12.13/bin/python3` | 3.12.13 | pip 26.2，62 个包 | CANN 生态包都装在这里或经 PYTHONPATH 提供 |

`/usr/local/python3.12.13/bin/pip3 list` 关键条目：

```text
hccl 0.1.0            te 0.4.0              auto_tune 0.1.0
cann_ops_transformer 1.0.0   es_hccl/es_transformer/es_nn/es_math/es_cv 1.0.0
superkernel 0.1.0     dataflow 0.0.1        llm_datadist(_v1) 0.0.1
modelscope 1.40.1     modelscope-hub 0.4.5  （可用于拉取模型）
numpy 2.5.1  sympy 1.14.0  Cython 3.2.9  scipy 1.18.0 等
```

**重要缺失**：`pip list` 和全盘 `find` 均未发现 **torch、torch_npu、vllm**；`import torch` 在两个解释器下均报 `ModuleNotFoundError`。要推理 Qwen3.8-Flash-Next-MXFP4 需自行安装 torch / torch_npu（及推理框架），并注意与 CANN 9.1.0 的版本匹配。

- 无 conda（`which conda` 未命中），`/workspace` 下无 venv。

---

## 5. 环境变量与 set_env.sh

### 5.1 自动加载配置（`/root/.bashrc` 摘录）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh   # bishengir PATH
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1                     # ATB，选 cxx_abi_1
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
```

当前 shell 已生效的关键变量（`env | grep -i ascend`）：

```text
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.1.0/toolkit
ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages:...tbe:...
LD_LIBRARY_PATH=...cann-9.1.0/lib64(+plugin/opskernel,nnengine)...driver/lib64{,/common,/driver}...atb/cxx_abi_1/lib...
CMAKE_PREFIX_PATH=.../toolkit/tools/tikicpulib/lib/cmake:.../lib64/cmake   # 方便 find_package 开发 mega kernel
```

### 5.2 set_env.sh 行为要点（`/usr/local/Ascend/ascend-toolkit/set_env.sh`）

- 向 `PATH` 前置：`$version_dirpath/bin`、`tools/ccec_compiler/bin`、`tools/profiler/bin`、`tools/ascend_system_advisor/asys`、`tools/show_kernel_debug_data`、`tools/msobjdump`
- 向 `LD_LIBRARY_PATH` 前置：`lib64`、`lib64/plugin/opskernel`、`lib64/plugin/nnengine`、`opp/.../op_tiling/lib/linux/x86_64`、`/usr/local/Ascend/driver/lib64{,/common,/driver}`
- 导出 `PYTHONPATH=$version_dirpath/python/site-packages:$version_dirpath/opp/.../tbe`、`ASCEND_OPP_PATH`、`ASCEND_AICPU_PATH`、`TOOLCHAIN_HOME`、`ASCEND_HOME_PATH`

---

## 6. 其他昇腾工具清单

| 工具 | 位置 | 状态 |
|---|---|---|
| `npu-smi` | `/usr/local/bin/npu-smi` | 可用（版本 25.7.rc1.6） |
| `msprof` | `/usr/local/Ascend/cann-9.1.0/bin/msprof` | 可用（`msprof --help` 正常，支持 `msprof op`） |
| `msnpureport` | `/usr/local/Ascend/driver/tools/msnpureport` | 可用（`msnpureport --help` 正常；容器内需加 `--docker`） |
| `msdebug` / `msmemscope` / `msopgen` / `msopprof` / `msopst` / `mssanitizer` / `msobjdump` | `cann-9.1.0/bin/` | 存在 |
| `atc` / `pyatc` | `cann-9.1.0/bin/` | 存在（模型转换） |
| `cannsim` + simulator | `cann-9.1.0/bin/`、`tools/simulator` | 存在（NPU 模拟器） |
| `hccl_test` | `cann-9.1.0/tools/hccl_test/` | 源码目录存在（未编译） |
| **`ascend-dmi`** | — | **未安装**（`which` 未命中；`/usr/local/dcmi/` 仅有 `npu-smi`、`libdcmi.so`、`dcmi_interface_api.h`） |

---

## 7. 风险与建议（面向后续 mega kernel / Qwen3.8-Flash-Next-MXFP4 开发）

1. **torch / torch_npu / vllm 全缺**，需按 CANN 9.1.0 对应矩阵安装（注意本机为 x86_64 + openEuler 24.03，且 ATB 侧按 cxx_abi=1 部署，torch 需选匹配 ABI 的 torch_npu 版本）。
2. **容器资源受限**：磁盘配额共 **300GB**（/workspace 为共享存储已占 649G，是否计入配额待确认）；内存 cgroup 上限 **32GB** 且 page cache 已顶满。MXFP4 模型权重（约 400GB）远超磁盘配额，下载与存放方案需与平台确认后再继续。
3. 默认 `python3`（3.11）无 pip，装包请用 `/usr/local/python3.12.13/bin/pip3`。
4. 容器内 NPU 设备节点为 `/dev/davinci1`（非 davinci0），脚本若硬编码设备号需注意；`lsmod` 在容器内不可见属正常。
5. `npu-smi info -t <type>` 必须携带 `-i <卡号>`；`-t cpu` 在该版本不存在。
6. 单机单卡（Chip Count=1，无拓扑），HCCL 多卡通信暂无用武之地，但 `tools/hccl_test` 可用于单卡内验证。

### 未能获取/未验证的信息（如实说明）

- 宿主机内核模块加载列表（容器内 `lsmod` 无输出，无权限查看宿主机）。
- Atlas 950 整机型号（板卡 Product Name `A310-50-C00MM304A1` 即所能确认到的最细粒度；`dmidecode` 等未尝试（容器内通常无权限））。
- `torch_atb` whl 未安装故未验证其可用性。
