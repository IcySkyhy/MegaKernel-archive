# M30 — 推理基线栈安装 + 版本裁决 + 设备自检（Qwen3.8-Flash-Next-MXFP4 @ Ascend 950PR）

本目录是 M30 的交付物：torch 栈安装、torch/torch_npu 与 vllm/vllm-ascend 的版本裁决、
昇腾 950PR 上的设备自检，以及"为什么现在没有 Ascend 框架栈能服务这个 checkpoint"的逐条证据。
所有数字来自本机实测（2026-09-26），原始输出在 `evidence/`，脚本在 `scripts/`。

> **明确区分（避免误读）**
> **已验证**：torch/torch_npu 栈在昇腾 950PR 上真的能用（§5，aclnn 算子往返数值正确）。
> **未做**：TPS / TTFT / 峰值 HBM 基线 —— **用户裁决押后**（"这个不急，先把单层跑通了调好性能再说"）。
> 本目录不含任何基线数字，也**不是**"基线已建立"。

---

## 0. 结论摘要

1. **torch 栈已装好并已在真实 NPU 上跑通**（不是"import 成功就算过"）：venv `/workspace/venvs/baseline`，
   torch 2.10.0+cpu / torch_npu 2.10.0.post4 / triton-ascend 3.2.2；aclnn 算子
   （matmul / softmax / reduce / elementwise）往返数值正确。见 §5、`evidence/05-npu-selfcheck.txt`。
   系统解释器与 `/root/.bashrc` **未被改动**（§4）。
2. **版本裁决：切 vLLM 到 v0.23.0 是死路；上游插件"支持" Qwen3.8-Flash-Next，但**只限 Atlas A3**，
   950DT/950PR 明确不支持。**
   - **vLLM 侧**：模型架构 `qwen4_exp` 最早出现在 **vLLM v0.29.0**；v0.23.0（fork 所 pin 的
     `0fc695fc`）里 `qwen4_exp` / `hc_count` / `ngram_vocab_size_base` 全为 0 命中。
   - **插件 tag 侧**：已发布 tag 最高 `v0.27.1rc1`（配 vLLM v0.27.1，无 `qwen4_exp`）。
   - **插件分支侧（路线所在）**：上游 `main` 是 **vLLM v0.30.0** 车道
     （`Dockerfile:46 ARG VLLM_TAG=v0.30.0`、`.github/vllm-release-tag.commit=v0.30.0`、
     `.github/vllm-main-verified.commit=ced6857a` = vLLM tag v0.30.0），`releases/v0.29.0rc` 是
     **vLLM v0.29.0** 车道；两者 pin 的 vLLM commit **都含 `qwen4_exp`**（34 个文件）。
   - **上游确实支持这个模型，但仅 A3**：support matrix 该行在 **A2/A3 tab**、
     `Supported Hardware = A3`（🔵 实验性、W8A8 ✅）；教程原文：
     "**The current version supports only Atlas A3 series hardware. … Ascend 950DT and 950PR
     products are not yet supported and will be enabled progressively in future releases.**"
     验证配置 = **Atlas 800 A3 64GB×8、DP1×TP8、W8A8 checkpoint**（GPQA 90.4）。
   - **对我们的目标（单卡 950PR + MXFP4 checkpoint）**：① 硬件代际不在支持范围（950PR 明确"not yet"）；
     ② 我们是 **MXFP4 (W4A4)**，上游验证的是 **W8A8**；③ 插件代码树里 `qwen4_exp` **0 命中**
     ⇒ 模型类来自 vLLM 上游、插件只出算子/patch（机制**待核实**，见 §8）。
   - ⇒ **当前没有任何可直接服务该 checkpoint 的 Ascend 组合**，要打通必须自己补
     `qwen4_exp` 在 **A5/950 + MXFP4** 上的实现与算子；方向不变，且适配起点是
     **上游 main（v0.30.0）/ releases-v0.29.0rc（v0.29.0）车道**，不是 fork 的 v0.23.0 线。详见 §3。
3. **显存/内存台账（可复用的设计输入）**：checkpoint 共 **169.72 GiB**，其中
   **n-gram 表 95.37 GiB**、**device 常驻部分 = 74.35 GiB** —— 与项目设计稿"其余权重 device 常驻
   74.3 GiB"**实测吻合**。n-gram 表按用户裁决走 host（PCIe-through MTE2 直读、可异步），
   其**驻留与分页策略待设计（用户已明确押后）**；本容器 cgroup 上限 32 GiB 是这套策略要面对的一个
   已知边界，**不是本 mission 的交付缺口**。详见 §6.2。
4. **未做 TPS 基线**（用户裁决押后），§7 只给"将来怎么测、怎么和 mega kernel 收益对比"的规程。

---

## 1. 环境事实（原始输出 `evidence/01-host-and-npu.txt`）

| 项 | 实测值 |
| --- | --- |
| NPU | 单卡 **Ascend950PR**（torch 报 `Ascend950PR_9579`），npu-smi HBM **总量 131072 MB**（已用量随其它 worker 波动，读数见 `evidence/01` 采集时刻） |
| HBM | npu-smi 131072 MB = 128 GiB；torch `total_memory` = 132,238,016,512 B = **123.16 GiB**（132238016512 / 2^30 = 123.15625；§5 与 `evidence/05` 里脚本按 1 位小数打印为 `123.2`，两者不矛盾） |
| Driver / npu-smi | 25.7.rc1.6 |
| CANN | 9.1.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） |
| ATB (nnal) | 9.1.0.B150，按 **cxx_abi=1** 部署（`ATB_CXX_ABI=1`、`ATB_HOME_PATH=.../atb/cxx_abi_1`） |
| Python | 3.12.13（`/usr/local/python3.12.13`，本机唯一带 pip 的解释器） |
| 容器内存上限 | 34,359,738,368 B = **32 GiB**（cgroup v1 `memory.limit_in_bytes`；宿主 `MemTotal` 754 GiB，容器受限） |
| 磁盘配额（**规划上限，用户口径**） | **300 GB 总量**（平台配额、用户确认；容器内 `df` 的 8.7T 是宿主 overlay 池，**不是**本容器的配额）。来源：`docs/01-environment.md` §3.1、`docs/04-summary.md` |
| 磁盘**实测**可用（**活值**） | **不得作为结论引用**：该值随其它 worker 波动。本 commit 的采集实例是 **426G**（`evidence/01`，采集时刻 2026-09-26T12:13:46Z）；**用前必须现场重采 `df -h /workspace`**，只引用本次现场读数（历史实例见 `evidence/01` 的 git 历史）。审计里这一格判定为 live，不设固定期望值 —— `scripts/audit_numbers.py` 现**从 `evidence/01` 现场解析**该读数，并校验本行写下的采集实例读数与归档**逐字一致**（写错即 FAIL，防的正是「活值被写死成过期字面量」） |
| 模型 | `/workspace/Qwen3.8-Flash-Next-MXFP4`，131 分片 / 1898 tensor / **169.72 GiB (182.23 GB)** |
| 模型架构 | `architectures=[Qwen4ExpForConditionalGeneration]`、`model_type=qwen4_exp`（文本子配置 `qwen4_exp_text`） |

自检时的一条告警（非致命，已 fallback，建议环境负责人评估）：

```
[W...] Warning: AclrtGetDeviceInfo failed to get total global memory, error code is 507899.
       The possible cause is that the driver version is too old or does not match CANN.
       Fall back to aclrtGetMemInfo.
```

---

## 2. torch / torch_npu 版本裁决

### 2.1 依据（两条独立来源）

- vllm-ascend 官方发布兼容矩阵（fork 内 `docs/source/community/versioning_policy.md`）：

  | vLLM Ascend | vLLM | Python | Stable CANN | PyTorch / torch_npu | Triton Ascend |
  | --- | --- | --- | --- | --- | --- |
  | v0.23.0 | v0.23.0 | >=3.10, <3.13 | **9.1.0** | **2.10.0 / 2.10.0.post4** | **3.2.2** |
  | v0.23.0rc1 | v0.23.0 | >=3.10, <3.13 | 9.0.1 | 2.10.0 / 2.10.0.post2 | 3.2.1 |

- fork README「Ascend 950 Quick Start (this fork, tag v0.23.0+ascend950)」自述实测环境：
  **Ascend 950PR / CANN 9.1.0 / PyTorch 2.10.0 / torch_npu 2.10.0.post4 / Python 3.12**。

本机 CANN 9.1.0 + Python 3.12.13 + 950PR 与该行**逐项吻合**，故选定
`torch==2.10.0` + `torch_npu==2.10.0.post4` + `triton-ascend==3.2.2`。
（上游 `main` 分支的 `requirements.txt` 同样是这三个 pin，见 §3.2。）

### 2.2 cp312 / x86_64 可用的 torch_npu wheel（实查镜像，`evidence/02-version-matrix.txt`）

`2.10.0 / 2.10.0.post2 / 2.10.0.post4 / 2.11.0 / 2.12.0 / 2.13.0rc1`（**没有 2.13.0 正式版**）。

- vLLM main 硬 pin `torch == 2.13.0`（`/workspace/vllm/pyproject.toml:10`），
  而对应 torch_npu 只有 **2.13.0rc1**（RC 版）。vllm-ascend 的依赖策略写明 monthly dev / RC 版
  torch-npu 只能用于 RC 版本 ⇒ **"vLLM main + 稳定 torch_npu"今天不存在**。

### 2.3 cxx_abi=1 匹配性（`evidence/08-abi-linkage.txt`）

- `torch._C._GLIBCXX_USE_CXX11_ABI == True`（torch 2.10.0+cpu 是 CXX11 ABI 构建）。
- torch_npu 的 `libop_plugin_atb.so` 在 `ldd` 下解析到
  `.../nnal/atb/latest/atb/**cxx_abi_1**/lib/libatb.so`、`.../cann-9.1.0/lib64/libascendcl.so`；
  `libatb.so` 的 cxx_abi_0 变体 `__cxx11` 符号 0 个、cxx_abi_1 变体 262 个。
- 即 **torch_npu 2.10.0.post4 与本机 ATB cxx_abi=1 部署一致**，并由 §5 的设备自检实测确认。

---

## 3. vLLM / vllm-ascend 匹配裁决：版本对齐有解，缺的是 A5/950 + MXFP4 的实现

### 3.1 "切 vLLM 到 v0.23.0" = 死路（模型架构不存在）

对 /workspace/vllm 各 tag 做 `git grep`（命中文件数，`evidence/02-version-matrix.txt`）：

| vLLM tag | `qwen4_exp` | `Qwen4ExpForConditionalGeneration` | `hc_count` | `ple_embed_dim` | `ngram_vocab_size_base` |
| --- | --- | --- | --- | --- | --- |
| v0.23.0（fork pin 的 `0fc695fc`） | 0 | 0 | 0 | 0 | 0 |
| v0.25.1 | 0 | 0 | 0 | 0 | 0 |
| v0.27.0 | 0 | 0 | 0 | 0 | 0 |
| v0.28.0 | 0 | 0 | 0 | 0 | 0 |
| **v0.29.0** | **23** | **5** | 14 | 3 | 3 |
| v0.30.0 | 27 | 7 | 14 | 4 | 3 |

复现：`git -C /workspace/vllm grep -c qwen4_exp v0.23.0 -- vllm` → 无命中。

**结论**：fork README 要求的 vLLM v0.23.0 是给它自己那个 Qwen3.8-**27B**（Qwen3.5 系 GDN 架构，
`patches/vllm-v0.23.0-ascend950.patch` 改的就是 `qwen_gdn_linear_attn.py`）用的；
Qwen3.8-**Flash-Next** 需要 **vLLM ≥ 0.29.0**。

### 3.2 "验证 main" = 车道存在，但插件里没有 `qwen4_exp` 实现

> **先记住一条**：vllm-ascend 的 `requirements.txt` / `[build-system].requires` 是**构建期依赖**，
> **不能**用它推断"这个插件支持哪版 vLLM"。目标 vLLM 版本要看 `Dockerfile` 的 `ARG VLLM_TAG`
> 与 `.github/vllm-release-tag.commit` / `.github/vllm-main-verified.commit`。
> （round-1 本报告就是因为读了 `requirements.txt` 的 torch pin 而得出错误结论，见 §10。）

1. **范围界定清楚了**：`tag` 与 `分支` 不是一回事。
   - **tag（已发布）**：上游最新 tag = **`v0.27.1rc1`**（配 vLLM v0.27.1，无 `qwen4_exp`）。
   - **分支（在开发）**：上游有 **26 个 head**（`evidence/11` §1，**采集时刻 `2026-09-26T12:03:08Z`**；上游随时可能新增分支，26 是**那一刻的快照、不是实时值** —— 引用前须复核 `git ls-remote --heads`；M51 复核时（同日）计数仍为 26），其中
     `releases/v0.29.0rc`（`a71b766c`，`Dockerfile` `ARG VLLM_TAG=v0.29.0`）与
     `main`（`1e9e03bc`，`Dockerfile:46 ARG VLLM_TAG=v0.30.0`、
     `.github/vllm-release-tag.commit=v0.30.0`、`.github/vllm-main-verified.commit=ced6857a`=vLLM tag v0.30.0）
     分别瞄准 **vLLM v0.29.0 / v0.30.0**，两者 pin 的 commit **都含 `qwen4_exp`**
     （`git -C /workspace/vllm ls-tree -r ced6857a -- vllm/models/qwen4_exp | wc -l` = **34**）。
   - 本机 fork `gitcode.com/liruixin_dvc/vllm-ascend` 的区间更窄：`.github/vllm-release-tag.commit=v0.23.0`、
     `.github/vllm-main-verified.commit=ee0da84a`（= vLLM **v0.24.0**，且**不是** main HEAD 的祖先，
     `git merge-base --is-ancestor` 返回 1）⇒ 该 fork 只覆盖 v0.23.0–v0.24.0，
     且其 README 服务的是 **Qwen3.8-27B**（`lenlrx/Qwen3.8-27B-MXFP4-ascend950`），与目标
     Qwen3.8-**Flash-Next** 不是同一个模型。
   - 原始 ref 列表与上述 pin 文件全文：`evidence/11-upstream-refs.txt`（`scripts/collect_upstream_refs.sh` 可复跑；
     gitcode fork 的 heads 另见 `evidence/02`）。
2. **两处 pin 的 torch 版本不一致，怎么解析属未实测**：vLLM v0.29.0/v0.30.0 的 `pyproject.toml` 写
   `torch == 2.13.0`，而 vllm-ascend main 车道的 `requirements.txt` 写
   `torch==2.10.0 / torch-npu==2.10.0.post4 / triton-ascend==3.2.2`。镜像上 torch_npu 只有
   **2.13.0rc1**（无 2.13.0 正式版，`evidence/02`）。这在 lane 里如何收敛（是否 `VLLM_TARGET_DEVICE=empty`
   构建 + 覆盖版本，如 fork README 对 v0.23.0 的做法）**需要实测确认，本文不下结论**。
   可以确定的只有：**上游 main 车道的构建期 pin 恰好就是 M30 装进 venv 的那一套**
   （`evidence/11-upstream-refs.txt` §4），所以这个 venv 与该车道同版本。
3. **vLLM 自己不带 Ascend 平台**：`vllm/platforms/` 只有
   `cpu / cuda / interface / rocm / tpu / xpu / zen_cpu`（`evidence/02-version-matrix.txt`），
   NPU 路径**完全依赖插件**。
4. **真正缺的是实现**：上游 `main`、`releases/v0.29.0rc`、本机 fork 的 `vllm_ascend/` 里
   `qwen4_exp` 命中**全部为 0**（`evidence/11-upstream-refs.txt` §5）；上游 main 的
   `VLLM_ASCEND_ENABLE_QSA_*` 环境变量也是 0 命中（文档先行、代码未落地，见 §8）。
5. **上游 `qwen4_exp` 没有 Ascend 分支**（`evidence/09-qwen4exp-dispatch.txt`，带行号）：
   `vllm/models/qwen4_exp/__init__.py:22` 的 `__getattr__` 只分三条路——
   `:30-31` xpu/tpu → `NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")`；
   `:32-37` rocm → `.amd.model`；**`:38-43` 其它平台（含 Ascend）一律 fall through 到 `.nvidia.model`**
   （即 CUDA 实现）。注册表入口：`vllm/model_executor/models/registry.py:114 / :602 / :697`。
   ⇒ NPU 实现**只能由插件补**。
6. **但"把 fork 的 v0.23.0 线插件搬到 vLLM main"这条路不成立**：用
   `scripts/check_plugin_vs_vllm_main.py` 拿 fork 插件 449 个 py 文件的 `from vllm... import ...`
   去 vLLM main 解析，1860 条可解析、**73 条已失效**（12 模块 + 31 符号，
   `evidence/04-plugin-vs-vllm-main.txt`）。样例：
   `vllm.v1.attention.backends.utils.CommonAttentionMetadata` → main 已移到 `vllm/v1/attention/backend.py`；
   `vllm.config.compilation.Range` → main 已无该类；
   `vllm.v1.kv_offload.{abstract,spec,mediums,worker.worker}` → main 重组为 `base.py/config.py/cpu/...`。
   **注意方向**：这 73 条是「**v0.23.0 线插件 vs vLLM main**」的差异，**不是**上游 main 车道的适配起点
   （后者要基于 vLLM v0.30.0 + 上游 main 插件）。该脚本是静态启发式，子模块/符号边角有假阳性 ⇒ 当**指标**看。

### 3.3 裁决表

| 方案 | 结果 | 证据 |
| --- | --- | --- |
| vLLM v0.23.0 + patch + vllm-ascend `qwen3.8-950`（fork 方案） | ❌ 模型架构不支持（`qwen4_exp` 不存在） | §3.1 |
| vLLM main(v0.30.1rc0) + fork 的 v0.23.0 线插件 | ❌ 插件 73 处引用失效（且那是 v0.23.0 线 vs main 的差异清单，不是适配起点） | §3.2(6) |
| vLLM + vllm-ascend 已发布 tag（最新 `v0.27.1rc1`） | ❌ vLLM 侧无 `qwen4_exp`（v0.27.1 < v0.29.0） | §3.1 |
| **vLLM v0.30.0 + vllm-ascend 上游 `main` 车道** | ⚠️ **软件栈首选待验证，但硬件不在支持范围**：车道存在且 pin 到含 `qwen4_exp` 的 vLLM commit（`VLLM_TAG=v0.30.0`）；插件里 `qwen4_exp` 0 命中（实现机制待核实）；**上游对该模型的官方支持只到 A3，950DT/950PR 明确 not yet**；运行期 torch 版本（vLLM 要 2.13.0 / 车道构建期 pin 2.10.0）也待实测 | §3.2, §3.4 |
| vLLM v0.29.0 + vllm-ascend `releases/v0.29.0rc` | ⚠️ 同上次选：`VLLM_TAG=v0.29.0`，硬件支持范围同样只到 A3 | §3.2(1), §3.4 |
| **本项目自研 kernel + 把 `qwen4_exp` 支持做进 vllm-ascend（上游 main 车道）+ 打通 A5/950 与 MXFP4** | ✅ 唯一可行方向（用户已定方向） | §3.4 |

### 3.4 上游对该模型的支持现状：支持，但只到 A3（这就是真正的阻塞）

> 交叉印证：M37 的接入方案 `docs/16-vllm-ascend-qwen4exp-plan.md`（已在 main）独立得到同一结论——
> §3「950PR 可行性裁决」、§2.4 版本口径更正（「无对应**已发布 tag**，但 main 分支已对准」）。
> 本节是从 M30（环境/版本/台账）角度给出的独立证据。

上游 vllm-ascend `main` 里**确实有** Qwen3.8-Flash-Next 的教程与支持矩阵行
（`evidence/11-upstream-refs.txt` §9）。逐条摘录：

- 教程 `docs/source/tutorials/models/Qwen3.8-Flash-Next.md` 第 7 行（原文）：
  > The current version supports only Atlas A3 series hardware. Atlas A2 series hardware and
  > **Ascend 950DT and 950PR products are not yet supported** and will be enabled
  > progressively in future releases. This tutorial describes the W8A8 deployment on Atlas 800 A3.
- 支持矩阵 `docs/source/user_guide/support_matrix/supported_models.md:140`（**A2/A3 tab**）：
  `| Qwen3.8-Flash-Next | 🔵 | | ✅ | A3 | ✅(W8A8) | ... | max-model-len 262144 |` ——
  `Supported Hardware` 列只有 **A3**，**不在 950DT tab 里**。
- 验证配置（教程 §3.1/§5.1）：权重 `Qwen3.8-Flash-Next-w8a8-mtp`（**W8A8**）、
  **1× Atlas 800 A3 (64GB × 8)**、**DP1×TP8**；GPQA Diamond 90.4。
- 镜像 tag（文档引用）：`qwen3.8-next-a3`、`qwen3.8-next-a3-openeuler`、`qwen3.8-a3`、`qwen3.8-a2`、
  `qwen3.8-a5`（通用 a5 tag 存在，但**没有** `qwen3.8-next-a5`）⇒ 该模型尚无 950 预置镜像。
- 教程导出的 `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER` / `VLLM_ASCEND_ENABLE_QSA_E3V`
  只出现在**文档**（教程 + 其中文 .po 翻译，共 2 个文件），**代码树 0 命中**
  ⇒ 上游那套 QSA 使能开关是**文档先行**、实现未落地。
- 插件代码树内 `qwen4_exp|Qwen4Exp` **0 命中**；插件的 `register_model()` 列表里也没有该 arch
  ⇒ 模型类来自 **vLLM 上游**，插件只提供算子/patch —— **具体机制本轮未核实**（§8 待核实项）。

**对本项目的三点直接含义**：
1. 阻塞不在"插件版本对不上"，而在 **① 硬件代际（950PR 未支持）② 量化格式（我们 MXFP4/W4A4，
   上游验证 W8A8）③ 单卡 vs 上游 8 卡**。这三条都不是换个版本能解决的。
2. 上游这条 A3 路线可以直接拿来当**数学/流程规格**（教程 + 上游 `qwen4_exp` 代码），但它
   不能作为**性能基线**（硬件与量化都不同）。
3. 教程第 9 行自称"first supported in vLLM-Ascend **0.26.0rc**"，这与 vLLM 侧时间线冲突
   （vLLM v0.26.0/v0.27.0/v0.28.0 都没有 `qwen4_exp`，v0.29.0 才有）⇒ 该句疑为陈旧文本，**待核实**。

---

## 4. 安装（已实测跑通，可复现）

脚本 `scripts/install_torch_stack.sh`（`VENV=/workspace/venvs/baseline`，rc=0）。
**不污染系统解释器、不改 `/root/.bashrc`**：全部装在 `/workspace/venvs/baseline`（仓库外，不入库）。
**该 venv 按用户/ tower 要求保留**，作为后续 vllm-ascend 栈的基础。

```bash
/usr/local/python3.12.13/bin/python3 -m venv /workspace/venvs/baseline
VENV=/workspace/venvs/baseline bash baseline_env/scripts/install_torch_stack.sh   # rc=0

# 复核：系统解释器里依然没有 torch
/usr/local/python3.12.13/bin/python3 -c "import torch"   # ModuleNotFoundError（未被污染）
/workspace/venvs/baseline/bin/python        -c "import torch; print(torch.__version__)"
```

最终版本清单（`evidence/06-pip-list.txt`）：
`torch 2.10.0+cpu` / `torchvision 0.25.0+cpu` / `torchaudio 2.10.0+cpu` / `torch_npu 2.10.0.post4` /
`triton-ascend 3.2.2`（附带 `triton 3.2.0`）/ `numpy 1.26.4` / `sympy 1.14.0` / `setuptools 80.10.2`。

两个坑（已写进脚本）：

1. **`download.pytorch.org` 在小包上会挂死**：torch/vision/audio 三个大 wheel 正常，但 `networkx`
   卡在 0 字节、socket 已断、pip 空等 12 分钟（`do_poll`）。现在小依赖走 huaweicloud 镜像，
   torch 三个包用 `--no-deps` 从 wheel 缓存装。
2. **venv 不是完全隔离**：`PYTHONPATH` 含 CANN 的 `python/site-packages`，所以 venv 里 `import te`
   等 CANN 包可用（编译 AscendC 算子需要），pip 也会因此报"依赖冲突"告警——属预期。

---

## 5. 设备自检（硬证据，非"import 成功"）

`scripts/selfcheck_npu.py`，完整输出 `evidence/05-npu-selfcheck.txt`，退出码 0：

```
torch 2.10.0+cpu / torch_npu 2.10.0.post4 / torch CXX11 ABI True / python 3.12.13
torch.npu.device_count() = 1
device name             : Ascend950PR_9579
device total memory     : 132238016512 bytes (123.2 GiB)
npu tensor created and stream synchronised in 0.092s
matmul (aclnnMatmul)    max_abs_diff=6.25e-02 rel=4.34e-04
softmax (aclnnSoftmax)  max_abs_diff=0.00e+00 rel=0.000e+00
sum (aclnnReduceSum)    max_abs_diff=1.00e+00 rel=4.66e-04
elementwise mul/add     max_abs_diff=0.00e+00 rel=0.000e+00
4 ops (fp16 512x1024x512 matmul + 3 epilogues) round trip in 9.7 ms
memory_allocated = 3.5 MiB, memory_reserved = 28.0 MiB
acl python bindings importable: yes
SELF-CHECK PASSED
```

即：NPU 上真做了 fp16 512×1024×512 matmul + 三个 epilogue，数值与 CPU 参考一致（相对误差 ≤4.7e-4），
`torch.npu.synchronize()` 正常，显存分配器有记账。**未启动任何模型推理**（遵守时间窗要求；
且用户已裁决基线押后）。

---

## 6. 阻塞证据

### 6.1 框架层：vLLM 路径的可复现命令与实测输出

没有"可运行的组合"可供跑出运行时报错——失败发生在**装之前**。每条都可直接复现：

| # | 命令 | 实测输出 | 结论 | 归档 |
| --- | --- | --- | --- | --- |
| 1 | `git -C /workspace/vllm grep -c qwen4_exp v0.23.0 -- vllm` | 无命中（0 文件） | v0.23.0 无该架构 | `evidence/02` §"qwen4_exp per tag" |
| 2 | 同上，`hc_count` 对 v0.28.0 / v0.29.0 | 0 vs 14 | 架构自 v0.29.0 引入 | `evidence/02` |
| 3 | `git ls-remote --heads https://gitcode.com/liruixin_dvc/vllm-ascend.git` | 仅 `refs/heads/qwen3.8-950` | 本机 fork 只覆盖 v0.23.0–v0.24.0 | `evidence/11` §3、§7 |
| 4 | `git ls-remote --tags https://github.com/vllm-project/vllm-ascend.git` | **46 个 tag**（`grep -vc '\^{}'`；命令原始输出 48 行，其中 2 行是 peeled `^{}` 条目），最新 tag `v0.27.1rc1` | 已发布 tag 无 ≥0.29 | `evidence/11` §2（计数行 `count = 46 tags`；`evidence/02` 另存最新 3 个 tag） |
| 5 | `git ls-remote --heads https://github.com/vllm-project/vllm-ascend.git` | **26 个 head，含 `releases/v0.29.0rc`（`a71b766c`）与 `releases/v0.28.0rc`**；`main` = `1e9e03bc` | ⚠️ round-1 此行的口径是错的（见 §10）；**分支已覆盖 v0.29/v0.30 车道** | `evidence/11` §1 |
| 6 | `git show main:.github/vllm-release-tag.commit`；`main:Dockerfile \| grep VLLM_TAG` | `v0.30.0`；`ARG VLLM_TAG=v0.30.0`（`releases/v0.29.0rc` → `v0.29.0`） | 目标 vLLM 由 `VLLM_TAG`/`.github` pin 决定，**不是** requirements | `evidence/11` §4 |
| 7 | `git -C /workspace/vllm ls-tree --name-only 8a2364605c -- vllm/platforms/` | 无 `npu.py`/`ascend.py` | NPU 必须靠插件 | `evidence/02` §"platforms" |
| 8 | `python baseline_env/scripts/check_plugin_vs_vllm_main.py` | 73 / 1860 引用失效 | fork 的 v0.23.0 线插件不能直接搬到 vLLM main | `evidence/04` |
| 9 | `git -C /workspace/vllm show 8a2364605c:vllm/models/qwen4_exp/__init__.py \| cat -n` | `:30-31` xpu/tpu 报错，`:38-43` else→nvidia | 无 Ascend 分支 | `evidence/09` |
| 10 | `/workspace/venvs/baseline/bin/python -c "import vllm"` | `ModuleNotFoundError: No module named 'vllm'` | 未安装 vLLM（**刻意**） | `evidence/11` §8 |
| 11 | `git show main:docs/source/user_guide/support_matrix/supported_models.md \| grep Qwen3.8` | 该行在 **A2/A3 tab**，`Supported Hardware = A3`（950DT tab 无此行） | **上游支持只到 A3** | `evidence/11` §9b |
| 12 | `git show main:docs/source/tutorials/models/Qwen3.8-Flash-Next.md \| sed -n 7p` | "…**Ascend 950DT and 950PR products are not yet supported**…" | **950PR 明确不支持** | `evidence/11` §9a |
| 13 | `git grep -l VLLM_ASCEND_ENABLE_QSA main` | 仅 2 个文件，**都是文档**（教程 + zh_CN .po），代码树 0 命中 | 上游 QSA 使能开关是文档先行 | `evidence/11` §9d |

第 11–13 行是 round-2 补充的证据，也是本轮最重要的新事实：**上游对 Qwen3.8-Flash-Next 的支持
存在但限定 A3**（详见 §3.4）。§6.1 所有行的原始输出都可在 `evidence/` 中离线复现（round-1 第 5 行
不符合该要求，已修，见 §10）。

补充：曾尝试下载 PyPI 上 `vllm==0.23.0` 的 274 MB wheel 做"已发布产物"二次确认，
`pypi.org` CDN 多次超时未成功（0 字节）；该确认与第 1 条覆盖同一 commit，故未阻塞结论。

### 6.2 显存 / 内存台账（`evidence/03-checkpoint-memory.txt`）

`scripts/checkpoint_memory_inventory.py` 只读 safetensors header，逐 tensor 统计。
下面**逐字摘自 `evidence/03-checkpoint-memory.txt`**（不是重排的摘要，便于逐行对账）：

```
shards           : 131
tensors          : 1898
total checkpoint : 169.72 GiB (182.23 GB)

== by role ==
  ngram embedding table                  95.37 GiB    130 tensors  ( 56.2%)
  MoE experts (mxfp4 packed u8)          56.36 GiB    240 tensors  ( 33.2%)
  MoE experts (unquantised bf16)          4.70 GiB     54 tensors  (  2.8%)
  GDN / linear attention                  3.89 GiB    325 tensors  (  2.3%)
  MoE experts (e8m0 scale)                3.52 GiB    240 tensors  (  2.1%)
  embed / lm_head                         2.37 GiB      2 tensors  (  1.4%)
  router / hyper-connection / ple         1.37 GiB    341 tensors  (  0.8%)
  full attention / indexer                1.25 GiB    117 tensors  (  0.7%)
  vision tower                            0.84 GiB    333 tensors  (  0.5%)
  MTP head                                0.06 GiB     16 tensors  (  0.0%)
  norms                                   0.00 GiB    100 tensors  (  0.0%)
```
（其余各项合计 ≈ 3.5 GiB：router/hc/ple 1.37 + attn 1.25 + vision 0.84 + mtp 0.06 + norms 0.00。）

- **device 常驻部分 = 169.72 − 95.37 = 74.35 GiB**，与项目设计稿"其余权重 device 常驻 **74.3 GiB**"
  **实测吻合**（这条可作为设计输入的直接校验）。可用 HBM 123.16 GiB ⇒ 余 ~48.8 GiB（123.15625 − 74.35
  = 48.81）给 KV cache / 激活；按"decode m=1 + ctx 4096 + 无并发"的定档，这部分是够的。
- **host 侧：驻留与分页策略待设计（用户已押后，非本 mission 交付缺口）**。n-gram 表 95.37 GiB 按用户
  裁决放 host、用 **PCIe-through MTE2 直接读 host 内存（可异步）**；decode 阶段每 token 只查
  **≈16 行 × 320 B ≈ 5 KiB**（tower 口径），所以这是**稀疏行查找 + 驻留/预取策略**问题，
  **不是"必须常驻 95.37 GiB RAM"的阻塞**。设计该策略时要考虑的已知边界：本容器 cgroup 上限
  **32 GiB**（cgroup v1 `memory.limit_in_bytes`），而 `free -g` 显示宿主 754 GiB、容器受限；
  若走 mmap + page cache，cgroup v1 下页缓存可回收、不至 OOM，但会被反复淘汰回读。
  **实现与实测留到后续 mission。**
- 参考：vLLM main 的 `vllm/models/qwen4_exp/common/ngram_embedding.py` 提供
  pinned-host + UVA 后端（`is_uva_available()`），那是 CUDA 系路线；我们要走的是 PCIe-through 自研读路径。

### 6.3 未做的部分（明确声明）

- **TPS / TTFT / 峰值 HBM 基线：未做**。用户裁决押后（"先把单层跑通了调好性能再说"），
  且没有可服务该 checkpoint 的 Ascend 框架栈。本目录**不含**任何基线数字。
- 未申请、未使用 NPU 时间窗，未运行任何模型推理；仅跑了一个毫秒级算子自检（§5）。
- 未搭建 v0.23.0 插件栈（tower 裁决：对目标零收益，且会与其它 worker 抢单卡资源）。

---

## 7. 将来怎么测、怎么和 mega kernel 收益对比（规程，本任务未执行）

### 7.1 指标定义（先固定口径才能比）

| 指标 | 定义 | 采集方式 |
| --- | --- | --- |
| **decode TPS** | 单流（concurrency=1, batch=1）稳态解码：`(输出 token 数 − 1) / (末 token 时刻 − 首 token 时刻)`，**排除 prefill** | 流式 `/v1/chat/completions` 打点，或 fork 的 `examples/ascend950_speed_demo.py`（内建 per-request decode tok/s，设计上 concurrency=1） |
| **TTFT** | 首个 token 到达时间（含 prefill/调度） | 同一流式请求 |
| **峰值 HBM** | 稳态解码期 `npu-smi info` HBM-Usage 峰值 + `torch.npu.max_memory_allocated()` | 采样脚本 + 进程内 API 双证 |
| **batch / 并发** | `--max-num-seqs`、并发数、prompt/输出长度分布 | 启动参数与压测配置一并记录，缺一不可比 |
| **算子占比** | 被测 kernel 的 self-time 占 decode 总时间比例 | `torch_npu.profiler` / mspti，用于把 TPS 变化归因到具体 kernel |

### 7.2 复现推理的命令行（条件满足后）

```bash
# 目标组合：vLLM ≥0.29（官方 qwen4_exp 是权威规格）+ 带 qwen4_exp 支持的 vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git   # 或我们自己的适配分支
cd vllm-ascend && MAX_JOBS=32 pip install -e . --no-build-isolation
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
cd ../vllm && VLLM_VERSION_OVERRIDE=<该插件要求的版本> VLLM_TARGET_DEVICE=empty \
  pip install -e . --no-build-isolation

cd /workspace   # 必须在模型目录父级启动，避免源码目录遮蔽已装包
vllm serve Qwen3.8-Flash-Next-MXFP4 --served-model-name qwen38-flash-next \
  --max-model-len 32768 --gpu-memory-utilization 0.9

curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"stream":true}'
```

`/workspace/vllm`、`/workspace/vllm-ascend` 两个既有检出**未被本任务修改**（只做只读版本核对）。

### 7.3 mega kernel 收益对比方法

1. **A/B 同条件**：同 prompt / 同 `max_tokens` / 同 batch 与并发 / 同 CANN 与 driver，
   只切换被测 kernel（`ASCEND_CUSTOM_OPP_PATH` 指向新算子包，或补丁替换算子实现）。
2. **重复与统计**：每侧 warmup ≥3 次、测量 ≥5 次，报中位数与极差；波动 >3% 的结论不采信。
3. **收益表述**：`speedup = TPS_mega / TPS_baseline`，同时报 TTFT 与峰值 HBM 变化
   （只报 TPS 会让"用显存换速度"看起来无成本）。
4. **归因**：用 §7.1 的算子占比确认收益来自被测 kernel，而非 MoE 路由分布等漂移。
5. **基线快照**：跑通后把 `版本清单 + 启动命令 + npu-smi 峰值 + 每次测量原始输出` 存入
   `evidence/`，并把数字写进本 README，作为全项目唯一收益基准。

---

## 8. 待人工决策 / 后续建议

1. **host 侧 n-gram 表的驻留与分页策略**：待设计；用户已裁走 PCIe-through MTE2 直读、可异步，**暂缓**（§6.2）。已知边界仅有容器 cgroup 32 GiB。**非本 mission 交付缺口，不在本轮调度范围内。**
2. **框架路线**（方向已由用户定）：把 `qwen4_exp` 支持做进 vllm-ascend。适配起点是**上游 `main`
   `VLLM_TAG=v0.30.0` 车道**（或 `releases/v0.29.0rc` = v0.29.0），**不要**用 fork 的 v0.23.0 线
   （§3.2(6) 的 73 处失效引用是"v0.23.0 线插件 vs vLLM main"的差异清单，不是起点）。
   **真正的工程量在硬件与量化**：上游对该模型的支持只到 **A3 + W8A8**，我们要的是
   **A5/950PR + MXFP4**（§3.4），这三条（硬件代际、量化格式、单卡）都得自己落。
3. **待核实项（本轮故意不实跑，单卡被多个 worker 共用）**：
   - 上游到底靠什么机制"支持"该模型（插件树 `qwen4_exp` 0 命中 ⇒ 模型类来自 vLLM 上游 + 插件
     算子/patch，具体是哪几个 op/patch 未核实）；
   - vendor 镜像 tag `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`（tower 实测 200，OCI index ~5.9 GB）
     里的实现是否超出文档范围（**未拉取**，等 tower 统一安排）；
   - 教程自称"first supported in vllm-ascend 0.26.0rc"与 vLLM 侧时间线（`qwen4_exp` 自 v0.29.0 才有）
     冲突，疑为陈旧文本；
   - 运行期 torch 版本在哪一侧定（vLLM v0.30.0 `pyproject` 写 2.13.0，车道构建期 pin 2.10.0）；
   - A5/950 上按需 import 扩展是否可行、PLE 行查找实际延迟。
4. **驱动告警**：`AclrtGetDeviceInfo` 507899（driver 25.7.rc1.6 vs CANN 9.1.0）请环境负责人评估是否升级。
5. `scripts/` 下脚本均可复跑（只读、不改环境）：`collect_env_evidence.sh`、`collect_upstream_refs.sh`、
   `checkpoint_memory_inventory.py`、`check_plugin_vs_vllm_main.py`、`selfcheck_npu.py`。

---

## 9. 文件与证据索引

| 文件 | 内容 |
| --- | --- |
| `scripts/install_torch_stack.sh` | torch 栈安装（实测 rc=0），含两个坑的规避 |
| `scripts/selfcheck_npu.py` | 设备自检：aclnn 算子往返 + 数值校验，退出码即结论 |
| `scripts/collect_env_evidence.sh` | 采集环境 / 版本矩阵 / 平台目录证据 |
| `scripts/checkpoint_memory_inventory.py` | 只读 safetensors header 的显存/内存台账 |
| `scripts/check_plugin_vs_vllm_main.py` | 插件 × vLLM main 引用失效统计 |
| `scripts/audit_numbers.py` | **README 全量数字对账**：每个量 → `evidence/` 出处:行号 → 判定；有对不上的条目即退出码非零 |
| `scripts/collect_upstream_refs.sh` | 上游 ref 原始列表 + 各车道 pin + 该模型支持范围（§11 的来源） |
| `evidence/01-host-and-npu.txt` | 主机 / npu-smi / CANN / ATB / cgroup / 磁盘 / 模型 |
| `evidence/02-version-matrix.txt` | vLLM tag × 架构命中、插件分支与 tag、平台目录、可装 wheel |
| `evidence/03-checkpoint-memory.txt` | checkpoint 逐角色字节台账 |
| `evidence/04-plugin-vs-vllm-main.txt` | 插件对 vLLM main 的 73 处失效引用明细 |
| `evidence/05-npu-selfcheck.txt` | 设备自检原始输出（PASSED） |
| `evidence/06-pip-list.txt` | venv 最终包清单 |
| `evidence/07-install-torch-stack.log` | 完整 pip 日志 |
| `evidence/08-abi-linkage.txt` | torch_npu 动态库 → CANN/ATB(cxx_abi_1) 链接关系 |
| `evidence/09-qwen4exp-dispatch.txt` | `vllm/models/qwen4_exp/__init__.py` 全文（带行号） |
| `evidence/10-venv-isolation.txt` | 系统解释器未被污染 + `/root/.bashrc` 未改动的复核输出 |
| `evidence/11-upstream-refs.txt` | **上游 ref 原始列表 / 各车道目标 vLLM / 该模型支持矩阵与教程原文** |
| `evidence/12-number-audit.txt` | **README 全量数字对账表**（58 条 + 覆盖性检查，`scripts/audit_numbers.py` 输出） |
| `evidence/13-number-audit-script-audit.md` | **audit 脚本的硬编码数值字面量处置表**（M51）：逐个判定为「真常量 / 活值 / 归档固定实例」，活值一律改为**运行时从归档现场解析**，附负例与复现命令 |

venv 位置：`/workspace/venvs/baseline`（**仓库外**，不入库，**按要求保留**）。

---

## 10. 更正记录（round-2，回应 review r1 的 P1/P2）

**P1（口径错误，已改）**：round-1 的 §0.2 / §3.2(1) / §3.3 第 4 行 / §6.1 第 5 行 + commit message
写了「没有任何已发布/在开发的 vllm-ascend 配 vLLM ≥ v0.29」——**与上游事实相反**。

- **真因**：我用 `git ls-remote --tags ... | grep -oE 'releases/v[0-9.]+$' | sort -V | tail` 过滤分支，
  而该正则**匹配不到以 `rc` 结尾的分支名**（`releases/v0.29.0rc` 被静默丢弃），
  于是只看到 `releases/v0.13.0 / v0.18.0 / v0.23.0`；又从 `requirements.txt` 的 torch pin
  去推断"目标 vLLM 版本"（那是**构建期**依赖），得出双重错误结论。
- **修正后的事实**（`evidence/11-upstream-refs.txt`，`collect_upstream_refs.sh` 可复现）：
  上游共 **26 个 head**（`evidence/11` §1），含 `releases/v0.29.0rc`（`VLLM_TAG=v0.29.0`）与 `main`
  （`VLLM_TAG=v0.30.0`、`.github/vllm-release-tag.commit=v0.30.0`、`main-verified=ced6857a`=vLLM v0.30.0）；
  **tag** 最高仍只到 `v0.27.1rc1`。目标 vLLM 版本看 `Dockerfile VLLM_TAG` / `.github/*.commit`，
  **不看 requirements.txt**。
- **底线结论未变、且本轮补强**：上游对 Qwen3.8-Flash-Next 的支持**只到 A3**，
  教程明写 **950DT/950PR not yet supported**（§3.4），插件树里 `qwen4_exp` 0 命中 ⇒
  要打通必须自己补 **A5/950 + MXFP4** 的实现与算子。
- **流程教训**：凡是"ref/分支/tag 列表"类结论，必须归档**未过滤的原始输出**（§6.1 现在每行都有
  `evidence/` 归档列）；凡标「实测输出」的行，都要能在 `evidence/` 离线复现（回应 `docs/17` §2.5）。

**P2（过度声明，已改）**：round-1 把「n-gram 表 95.37 GiB vs 容器 cgroup 32 GiB」写成
"bug/high / host 侧读路径无法落地 / 必须先解决"。用户已裁 n-gram 表放 host、PCIe-through MTE2
直读、可异步、暂缓实现，且 decode 每 token 只读约 5 KiB ⇒ 这是**稀疏行查找 + 驻留/分页策略待设计**，
不是阻塞；已在 §0.3 / §6.2 / §8.1 改为该口径，并注明**不是本 mission 的交付缺口**。
（数字本身未变，reviewer 已逐位复核：169.72 / 95.37 / 74.35 GiB。）

### 10.1 round-3：r2 的两条 P2 更正（同一类"改一半"）

review r2 指出两处**口径与自身证据不符**，都是"改了 A 处、忘了 B 处"：

1. **§8.1 残留**：r2 声称 §0.3/§6.2/§8.1 都已改，但 **§8.1 实际仍是 round-1 原文**
   （「唯一待办」「放开容器 32 GiB 上限？」），与 §0.3/§6.2 自相矛盾。现已改为
   「驻留与分页策略待设计（用户已裁 PCIe-through MTE2 直读、可异步、暂缓）；非本 mission 交付缺口」。
   **教训**：文档类修复必须**逐处 grep 复核**（`grep -n "旧措辞"`），不能凭记忆写对账。
2. **计数与归档不符（三处）**：
   - 「上游 25 个 head」→ **26**（README §3.2(1)/§6.1 第 5 行/§10 三处 + commit message；
     `evidence/11` §1（采集时刻 `2026-09-26T12:03:08Z`）为 26；**这是采集时刻的快照，不再写作「与 live 一致」**）。
   - 「48 个 tag ref」→ **46 个 tag**（48 是含 peeled `^{}` 的原始行数；`evidence/11` §2 归档 46）。
   - `evidence/11` §4 标签原写 `content hits : 20` → 那是**命中文件数**，真实**出现次数 37**
     （20 文件 / 37 出现次数）；标签现已拆成两行并标注单位；
     （`git grep -l | wc -l` = 20 vs `git grep -o | wc -l` = 37）。脚本已把两者**分别列出行**并标注单位，
     §5/§9e 同类标签一并改掉。
   - **新增防呆**：`collect_upstream_refs.sh` 现在在 §1/§2/§3 的列表后**直接打印计数行**
     （如 `(count = 26 heads …)`、`(count = 46 tags; raw ls-remote --tags lines including peeled ^{} = 48)`），
     让数字与列表同源、无法再凭记忆写错（回应 `docs/17` §2.2 计数可复算、§2.6 README 与同 commit evidence 一致）。
3. **同类自查（主动）**：§1 的 npu-smi **已用** HBM 读数（原写 `5245 MB`）是随其它 worker 波动的活值，
   与同 commit 重新采集的 `evidence/01` 不一致 ⇒ 已改为只写**总量 131072 MB**，用量指向证据文件的采集时刻。

### 10.2 round-4：全量数字对账 + 「固定量 vs 活值」口径

review r3 又抓到 §1 两处（磁盘可用量、`123.19 GiB`）与同 commit 归档不符，且指出**磁盘行与我上一轮
刚修的 npu-smi 活值属同一类问题——"活值修复只做了一半"**。这已是同类第三次，所以本轮不再只改被点名的格，
而是**做了一次全量对账**：

- 产出 `evidence/12-number-audit.txt`（`scripts/audit_numbers.py` 可复跑）：把 README 里**每一个带单位的
  量**逐个 grep 到归档出处（给出 `evidence/文件:行号`）并判定；未被对账表覆盖的数字会被单独列出，避免漏项。
- **口径分两类，不许混用**：
  - **固定量**（可写进文档、可被引用）：版本号/pin commit、checkpoint 台账字节、`qwen4_exp` 命中数、
    tag/head 计数、HBM 总量 131072 MB、cgroup 上限 34359738368 B、容器内存 32 GiB、
    `total_memory` 132,238,016,512 B（= 123.16 GiB）。
  - **活值**（**禁止**以具体数字形式写进正文当结论；只能写"固定量 + 指向 evidence 采集时刻"）：
    npu-smi 的**已用** HBM、`df` 的**实测可用**磁盘、解码耗时/带宽类实测数字。
    本轮据此改了：`§1` 磁盘行拆成「配额（用户口径 300 GB，规划上限）」与「实测可用（**活值**：只写本次
    `evidence/01` 采集实例的读数 + 采集时刻，并注明"用前现场重采"）」两格；
    并删掉了 §10.1 里引用过的另一个活值读数（避免它随采集时刻再次过期）。
- **纪律**：写"实测值"前先 `grep` 一遍同 commit 的 `evidence/`；取不到出处就改成固定量或现场重采。
  （这条是本 mission 被同一类问题抓三次后的结论。）

### 10.3 M51（agent-sweep）：audit 脚本的「活值字面量」清扫 + 两处文档数字订正

§10.2 那条纪律只落实到了 README 的**正文**，**没有落实到我自己的工具**：`scripts/audit_numbers.py`
本身就在 `live` 行里硬编码了一个过期值（`expect="453G"`，而它引用的 `evidence/01:83` 是 **426G**）
—— 正是它要防的那一类错。本轮按「改机制、不改字面量」收口：

1. **`audit_numbers.py` 的 `live` 类不再带字面量期望值**：该值改为**运行时从归档现场解析**
   （`archive_pattern` 的捕获组），并校验 README 本行写下的采集实例读数与归档**逐字一致**；
   README 若只写「活值 + 指向采集时刻」而不写读数，同样通过。负例实测：把本行的 426G 改成 453G
   ⇒ 脚本 `exit 1` 并报 `LIVE-FAIL(README 读数 453G ≠ 归档 426G)`；把实例读数删掉 ⇒ `LIVE-OK`。
   同时删掉了 `EXTRA_COVERED` 里两条**指向已不存在值的陈旧豁免**（`453G`）与手工豁免（`426G`），
   改由解析值自动进入覆盖集。
2. **同一类清扫**（脚本内所有硬编码数值字面量逐个判定，完整处置表见 `evidence/13`）：
   - `scripts/check_plugin_vs_vllm_main.py` 的 banner 原硬编码检出快照 `v0.30.1rc0-189` ⇒ 改为运行时
     `git describe --tags`（去尾 `-g<sha>`）。**改后输出与归档 `evidence/04` 逐字节相同**（实测 diff 空）。
   - `scripts/collect_upstream_refs.sh` 在「网络不可用」分支里硬编码了「上游最新 tag」⇒ 改为直接打印
     `evidence/02` 里已归档的清单（该分支不参与归档那次采集 ⇒ `evidence/11` 不变）。
   - 保留的字面量全部归入两类：**真常量**（算子/格式/判定阈值参数、版本 pin、不可变 commit 对象 ID）
     与**归档固定实例**（`checkpoint_memory_inventory.py`/`selfcheck_npu.py` 的输出都是运行时从设备或
     header 现读，脚本内没有活值）。
3. **两处文档数字订正**（同属"活值被写死"的后果）：`docs/17` §6 的 `m18_gdn_prefill/` 行判据
   （`2e-5·|exp|+1e-6` → T1 逐位 + T3 逐元素 `ε·Σ|terms| + 0.5·ulp(out)`，ε **5e-5**，依据 κ∞ ≤ 3.212）
   与 §2/§6 的 L2 三项指标标注；`m22_router512/README.md` 的 m=64 计时（1.7~1.8ms → **1.8ms**）与带宽
   （→ **8.3~12.6 GB/s**），并把 `m22_router512/tools/cost_model.py` 里硬编码的实测时间改成**
   从 `evidence/run_mode{2,4}.log` 现场解析**。四项出处汇总见 `docs/17` §8.1 与 `m22_router512/README.md` §11。

复现：`python3 baseline_env/scripts/audit_numbers.py`（`exit 0`；两次运行输出逐字节相同，sha256 见 `evidence/13`）。
