# vllm-ascend 承载 qwen4_exp 的接入方案（M37 只读调研）

> 调研时间：2026-09-26。方式：**静态取证**（git tag/commit 比对、源码阅读、`git ls-remote`、checkpoint 元数据解析、`hasattr` 探测）。
> **未重编译 vLLM / vllm-ascend，未装包，未跑模型，未提交任何变更。** 全部结论标注来源；凡是必须真机才能确认的，一律放 §9「存疑与需实跑验证」，不猜。
>
> 来源代号（下文 `file:line` 均可按此定位）：
>
> | 代号 | 路径 / 版本 |
> |---|---|
> | `[vLLM]` | `/workspace/vllm`，`v0.30.1rc0-189-g8a2364605c`（2026-09-26，main） |
> | `[asc-main]` | **上游** `vllm-project/vllm-ascend` `refs/heads/main` = `1e9e03bc53bdc53f3cc78c7af191794ee7544ad9`（2026-09-26）。取证方式：浅克隆到 `/tmp/va-up/main`（在仓库之外，不污染任何 working tree） |
> | `[asc-main@0.29]` | 上游 `refs/heads/releases/v0.29.0rc` = `a71b766ce6fc0a9669a412e15a5cf53f7f827093` |
> | `[asc-fork]` | `/workspace/vllm-ascend`，分支 `qwen3.8-950` @ `35d606dfe`（本机唯一分支，gitcode `liruixin_dvc/vllm-ascend`） |
> | `[model]` | `/workspace/Qwen3.8-Flash-Next-MXFP4`（131 分片，182.2 GB） |
> | `[ours]` | `/workspace/ascend_mega_kernel`（m0…m14 + docs/01–13） |

---

## 0. 结论速览

| 问题 | 裁决 | 关键证据 |
|---|---|---|
| 有没有「vLLM + vllm-ascend」组合能承载 `qwen4_exp`？ | **有**。`vLLM v0.30.0`（精确 commit `ced6857a`）+ **上游 vllm-ascend `main`**（不是 tag） | 上游 main 的 `.github/vllm-main-verified.commit` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；该 commit 在本机存在且含 34 个 `vllm/models/qwen4_exp/` 文件 |
| 只按「已发布 tag」找，找得到吗？ | **找不到**。最新 tag `v0.27.1rc1` ↔ vLLM `v0.27.1`，而 `v0.27.1` 里 `qwen4_exp` **0 文件** | `git ls-remote --tags` + 本地 `git ls-tree -r <tag> \| grep qwen4_exp` |
| torch/torch_npu 栈对得上本机吗？ | **完全对得上**（CANN 9.1.0 / torch 2.10.0 / torch_npu 2.10.0.post4 / Python 3.12） | `[asc-main] mkdocs.yml:181-187` + `requirements.txt`；本机 `/workspace/venvs/baseline` 实测 |
| 本机 fork（`qwen3.8-950`）能承载吗？ | **不能**。它 pin 的是 vLLM **v0.23.0/0.24.0-line**，该线没有 `qwen4_exp` | `[asc-fork] .github/vllm-main-verified.commit` = `ee0da84a…`（`git describe` = **v0.24.0**）、`.github/vllm-release-tag.commit` = `v0.23.0` |
| 950PR 能跑吗？ | **组件层面可以起步，模型层面上游明确「未启用」**。且上游 main **根本没有该模型的插件实现**（只有文档） | `[asc-main] docs/source/tutorials/models/Qwen3.8-Flash-Next.md:9`「Atlas 950DT and 950PR products are not yet supported」；`grep -i qwen4` 在 `[asc-main]` 代码树 **0 命中** |
| 我们的 mega kernel 怎么接？ | **用户已裁决（2026-09-26）**：「每个 kernel 只跑一层，对 vllm 的修改会比较小」→ **48 层循环留在框架侧，只做 per-layer 算子**；落地点 = 替换层模块的 forward（§6.1），**不需要**注册整套模型类 | 层 forward 跨 HC-mix→GDN/QSA→HC-combine→MoE 四个模块（`[vLLM] nvidia/model.py:276-331`），一次算子调用覆盖整层；层界三张量由框架持有（§6.2 方案 A） |
| 950 上最容易被忽略的硬约束 | **A5 上 `vllm_ascend_kernels`（AscendC 通用算子库）不编译、`enable_custom_op()` 整体返回 False** | `[asc-main] CMakeLists.txt:74-80`；`[asc-main] vllm_ascend/utils.py:511-519`（FIXME，issue #7157）；A5 capability 集合中**没有** `RUNTIME_CUSTOM_OPS`（`device/hardware_profile.py:309-346`） |
| **造模期的第一个硬失败点** | QSA 层构造时直接 `NotImplementedError: Qwen4Exp QSA requires FlashAttention`（NPU 无 flash-attn 实现） | `[vLLM] nvidia/qsa.py:169-170`（pin commit 版为 `:101-104`）；`vllm/v1/attention/backends/fa_utils.py:346-375` 的可用性判定只覆盖 CUDA/XPU/ROCm |
| **950 的 MoE 走不了 CANN MegaMoe** | 官方约束：`hidden` 仅支持 4096/5120/7168、`num_topk` 仅 6/8、`intermediate_hidden` 仅 1024/2048/3072/4096/7168；本模型是 `hidden=2560 / topk=10 / moe_intermediate=640` → **三项全部不符** | 本机 CANN 自带文档 `/usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/docs/zh/mega_moe.md:1072-1082`（950PR/950DT 段） |
| 最大单项缺口 | **PLE ngram 表**（95.4 GiB，单层，必须在 host/稀疏查找）与 **QSA indexer**（CUDA-only topk + 模型包内私有 attention backend） | `[vLLM] nvidia/ops/qsa_indexer.py:488-490`；`[vLLM] nvidia/qsa.py:64/127/252` |

**一句话**：用户原话「参考官方 vllm 实现就行，最终要在 vllm-ascend 支持 qwen4_exp」在**版本层面已经不再是死路**——上游 vllm-ascend main 已对准含 `qwen4_exp` 的 vLLM 精确 commit；但在**模型层面**，上游对该模型只发了 A3 文档、**没有插件实现**，且把 950PR 列为「尚未启用」。所以我们要做的是：**以 main 为基线**，自己补齐 Ascend 侧的缺口（量化入口、QSA、层算子、PLE），把我们的 per-layer kernel 挂在**官方层模块**的位置上——**不是**另起一套模型实现。

---

## 1. 取证方法（可复现）

```bash
# 1) vLLM 侧：哪些 tag 才有 qwen4_exp
cd /workspace/vllm
for t in v0.26.0 v0.27.1 v0.28.0 v0.29.0 v0.30.0 v0.30.1rc0; do
  echo -n "$t: "; git ls-tree -r --name-only $t | grep -c '^vllm/models/qwen4_exp/'
done
# v0.26.0: 0   v0.27.1: 0   v0.28.0: 0   v0.29.0: 31   v0.30.0: 34   v0.30.1rc0: 34

# 2) 上游 vllm-ascend 支持哪个 vLLM（看 main 与 release 分支，而不是只看 tag）
git ls-remote --heads https://github.com/vllm-project/vllm-ascend
#   → refs/heads/main 1e9e03bc…；refs/heads/releases/v0.29.0rc a71b766c…
#     refs/heads/releases/v0.28.0rc df7d511e…（最新 tag 只到 v0.27.1rc1，tags 明显落后于 branches）

# 3) 上游 main 到底对准哪个 vLLM commit
git clone --depth=1 --branch main https://github.com/vllm-project/vllm-ascend.git /tmp/va-up/main
cat /tmp/va-up/main/.github/vllm-main-verified.commit   # ced6857afa0ea7b2e3f0846a62e1394e90f15607
cat /tmp/va-up/main/.github/vllm-release-tag.commit     # v0.30.0
# 4) 该 commit 在本机存在、且含模型
cd /workspace/vllm && git cat-file -t ced6857afa0ea7b2e3f0846a62e1394e90f15607   # commit
git ls-tree -r --name-only ced6857afa0ea7b2e3f0846a62e1394e90f15607 | grep -c '^vllm/models/qwen4_exp/'   # 34
git describe --tags ced6857afa0ea7b2e3f0846a62e1394e90f15607                      # v0.30.0
git rev-list --count ced6857afa0ea7b2e3f0846a62e1394e90f15607..HEAD              # 604（我们的 HEAD 已漂移 604 个提交）
```

> 注：`FetchURL`（raw.githubusercontent）在本环境被网络策略拒绝，`gh` CLI 不存在；因此上游信息全部通过 **git over https** 取证，落在 `/tmp/va-up/`（仓库之外，不作为任何 mission 的产物）。

---

## 2. 版本与可行性裁决（任务 1）

### 2.1 三层决定性事实

**事实 1 — `qwen4_exp` 是 vLLM 的「新模型」**（`[vLLM]` 实测）：

| vLLM tag | `vllm/models/qwen4_exp/` 文件数 |
|---|---|
| v0.26.0 | 0 |
| v0.27.1 | 0 |
| v0.28.0 | 0 |
| **v0.29.0** | **31**（首次出现） |
| v0.30.0 | 34 |
| v0.30.1rc0 | 34 |

**事实 2 — 上游 vllm-ascend 的「已发布 tag」严重落后于其 main/release 分支**：

- 最新 tag：`v0.27.1rc1`（`3b318862…`，2026-09-25）→ `.github/vllm-release-tag.commit` = `v0.27.1` → **vLLM v0.27.1 无 `qwen4_exp`**，此路不通。
- 但上游存在 **release 分支** `releases/v0.28.0rc`、`releases/v0.29.0rc`（无对应 tag），且 `main`（2026-09-26）的 verified vLLM commit = **`ced6857a`**（`git describe` = `v0.30.0`），**含 `qwen4_exp` 34 文件**。
- 只有 `main` 与 `releases/v0.29.0rc` 的文档区含 `docs/source/tutorials/models/Qwen3.8-Flash-Next.md`（其它 tag/分支没有）。

**事实 3 — 本机 fork 是「另一个模型」的资产**：

- `[asc-fork] .github/vllm-main-verified.commit` = `ee0da84ab9e04ac7610e28580af62c365e898389` → 在本机 `[vLLM]` 里 `git describe --tags` = **v0.24.0**，且**不是** HEAD 的祖先；`.github/vllm-release-tag.commit` = `v0.23.0`。
- `[asc-fork]` 的 README 明确写的是 **Qwen3.8-27B**（`README.md:82`「pre-quantized Qwen3.8-27B models」；`:136` 下载 `lenlrx/Qwen3.8-27B-MXFP4-ascend950`）。
- `[asc-fork]` 全树 `grep -i "qwen4|Flash-Next|hc_count|ple_embed_dim"` → **0 命中**。

### 2.2 推荐组合 A（唯一自洽且本机可满足）

| 组件 | 版本 / commit | 证据 |
|---|---|---|
| vLLM | **`ced6857afa0ea7b2e3f0846a62e1394e90f15607`**（tag `v0.30.0`，= 上游 main 的 verified commit） | `[asc-main] .github/vllm-main-verified.commit`；`[vLLM]` 本地存在该 commit 且含模型 |
| vllm-ascend | **上游 `main` @ `1e9e03bc53bdc53f3cc78c7af191794ee7544ad9`**（备选：`releases/v0.29.0rc` @ `a71b766c…`，verified commit `84030bbe…`，同样含 `qwen4_exp` 34 文件） | `git ls-remote --heads`；`[asc-main] docs/.../Qwen3.8-Flash-Next.md` |
| Python | 3.10–3.13（本机 3.12.13 ✅） | `[asc-main] mkdocs.yml:181` |
| CANN | 9.1.0（本机 ✅） | `[asc-main] mkdocs.yml:182` |
| torch / torch_npu | 2.10.0 / 2.10.0.post4（本机 `/workspace/venvs/baseline` 实测 ✅） | `[asc-main] mkdocs.yml:183-184`、`requirements.txt` |
| triton-ascend | 3.2.2（本机 venv 里是 triton 3.2.0，**需替换**） | `[asc-main] mkdocs.yml:187`、`requirements.txt` |

附带好处：上游 main 的 `requirements.txt` 与 M30 已装好的栈**逐项一致**（`torch==2.10.0`、`torch-npu==2.10.0.post4`、`triton-ascend==3.2.2`、`compressed_tensors>=0.11.0`），说明本机基线选型没跑偏。

**必须 pin 到 `ced6857a`，不要用我们 `/workspace/vllm` 的 HEAD**：HEAD 比它多 604 个提交，且仅 `vllm/models/qwen4_exp/` 就漂移了 `19 files changed, 968 insertions(+), 600 deletions(-)`。另有一条硬证据说明「main + 更新的 vLLM」会出错：`[asc-fork] platform.py:816` 用 `getattr(attn_selector_config, "use_compress", False)` 读一个字段，而 `[vLLM]` HEAD 的 `AttentionSelectorConfig`（`vllm/v1/attention/selector.py:21-61`）**没有** `use_compress`；`[asc-fork] platform.py:841` 又对 `(use_mla, use_sparse, use_compress)` 直接索引，未映射组合会 `KeyError`。

### 2.3 不可行组合 B（本机 fork 路线）

`vLLM v0.23.0/v0.24.0` + `[asc-fork] qwen3.8-950`：**vLLM 侧根本没有 `qwen4_exp`**（v0.23/v0.24/v0.26/v0.27/v0.28 全 0）。要在此线上承载，等于把整个模型 + 它的 vLLM 依赖回移（模型包 `<vLLM> vllm/models/qwen4_exp/` 34 文件、`config/model.py` 的 `model_class_overrides`、`config/engram.py`、`config/speculative.py:52` 的 `qwen4_exp_mtp`、`transformers_utils/model_arch_config_convertor.py:682`、`MambaHybridModelState`、`v1/attention/backends/{short_conv_attn,gdn_attn}` 等）。**结论：不做。** fork 的价值在别处——它是**本机唯一一份「950 + MXFP4 + vllm-ascend 实战调通」的资产**（见 §5）。

### 2.4 对既有结论（M30 / agent-baseline）的两处修正

| M30 原话 | 复核结果 |
|---|---|
| 「上游 vllm-ascend 最新已发 tag 是 v0.27.1rc1 … **No released vllm-ascend pairs with vLLM ≥ v0.29** → 无可安装的 Ascend 栈」 | **对 tag 成立，对分支不成立**。上游 `main` / `releases/v0.29.0rc` 都 pin 到含 `qwen4_exp` 的 vLLM commit。措辞应从「无对应版本」改为「**无对应已发布 tag，但 main 分支已对准**」。 |
| 「fork 最新 ref 配 vLLM v0.23.0–v0.25.1」 | 区间应更窄：`main-verified` = `ee0da84a`（= **v0.24.0**）、`release-tag` = **v0.23.0** → **v0.23.0–v0.24.0 line**。 |

M30 的第三点（`vllm/models/qwen4_exp/__init__.py` 只派发 nvidia/amd，xpu/tpu 抛 `NotImplementedError`）**完全正确**，且我在 §4① 补了一条更关键的推论：**NPU 不在这三类里，所以不会报错，而是静默落到 `nvidia/` 实现**。

---

## 3. 950PR 可行性裁决（含硬约束）

### 3.1 上游对 950（= A5）的整体态度：**支持，但在演进中**

- 设备识别：`soc_version == 260` → `AscendDeviceType.A5`（`[asc-main] vllm_ascend/device/hardware.py:70-73`）；`"ascend950" in soc_version` → A5（`:53-57`）；`is_950()` 在 `[asc-fork] device/device_config.py:61`。
- A5 有**专门的能力档位**：`[asc-main] vllm_ascend/device/hardware_profile.py:309-346` 的 `AscendDeviceType.A5` profile，能力集合里包含 `CANN_MEGAMOE`、`CANN_MEGAMOE_MXFP`、`DYNAMIC_MX_QUANT_FUSION`、`SWIGLU_OAI_MX_QUANT`、`GRAPH_NORM_QUANT_FUSION`、`NPUGRAPH_EX`、`FP8_ATTENTION`、`STANDARD_MAMBA_PATCH` 等。
- csrc 里大量算子的 README 明确标注支持 `ascend950`：例如 `csrc/attention/compressor/README.md:7`（950PR&950DT √）、`csrc/attention/fused_sparse_attention_overlap/README.md:7`（√）、`csrc/attention/chunk_kda_fwd/README.md:102`（A2/A3/950PR&950DT 均支持，A5 有 regbase 双发射特化，见 `chunk_kda_fwd/docs/design.md:6-7,111-133`）；而 `csrc/attention/fused_lightning_indexer_manage/README.md:7` 与 `fused_scatter_copy_sparse_flash_attention/README.md:7` 标 **950 = ×**。
- 构建系统支持：`[asc-main] csrc/CMakeLists.txt:37-42` 的 `ASCEND_COMPUTE_UNIT` 含 `ascend950`；`csrc/attention/chunk_kda_fwd/op_host/CMakeLists.txt:30-33` 出现 `COMPUTE_UNIT Ascend950PR_9599`。

### 3.2 硬约束：**A5 上「自定义算子」是被整体关掉的**

| 约束 | 位置 | 影响 |
|---|---|---|
| `vllm_ascend_kernels`（通用 AscendC 算子库）在 `ascend950` **不编译** | `[asc-main] CMakeLists.txt:74-80`（`if(SOC_VERSION MATCHES "ascend310p.*\|ascend950") … skip`）；`[asc-fork] CMakeLists.txt:76-82` 同构 | 新写的 AscendC kernel **没有通用编译槽位** |
| `enable_custom_op()` 在 A5 **直接返回 False** | `[asc-main] vllm_ascend/utils.py:511-519`（注释：*"Currently custom op compilation and execution are partially available in ASCEND950 chip, we temporarily disable all custom ops"*, FIXME issue #7157）；`[asc-fork] utils.py:426` | 所有走 `enable_custom_op()` 门控的路径在 950 上退化 |
| A5 capability **不含** `RUNTIME_CUSTOM_OPS` | `[asc-main] vllm_ascend/device/hardware_profile.py:309-346`（对比 `_310P:326-345` **含**该能力） | 同上，是策略层而非口误 |
| ATB / direct kernel 在 950 关闭；MLAPO 在 950 被排除 | `[asc-main] CMakeLists.txt:178`、`:66-71` | 不要指望 ATB 路径 |
| 替代路径是存在的 | A5 profile 的 `CANN_MEGAMOE` / `CANN_MEGAMOE_MXFP`（`:64-67`）、`ascend_config.py:1059-1061`（950 MegaMoe 只支持 MXFP 量化）；以及 vllm-ascend 自己的 **aclnn vendor 包**机制（`[asc-main] vllm_ascend/_cann_ops_custom/.gitkeep` 说明 + `utils.py` 的 `bootstrap_custom_op_env()` → `ASCEND_CUSTOM_OPP_PATH`） | MoE / MX 量化在这条线上是「CANN 融合算子 + aclnn 包」，不是「我们的 torch 扩展」 |

**fork 已经给出了 950 上可用的绕法**（这是本机最值钱的一条经验）：在 `ascend950` 上**另建专用 `ascendc_library`**，并且**按需 import 扩展**而不是走 `enable_custom_op()`：

- `[asc-fork] CMakeLists.txt:84-98`：`if(SOC_VERSION MATCHES "ascend950")` → 单独构建 `vllm_ascend_block_fp8_gemm`（`csrc/block_fp8_gemm/op_kernel/*`）与 `vllm_ascend_quant_fusion`（`csrc/quant_fusion/rms_norm_dual_mx_quant_kernel.cpp`）。
- `[asc-fork] vllm_ascend/device/mxfp_compat.py:79-93`：注释直接说明「该融合算子在 `vllm_ascend_C` 扩展里，而扩展是被 `utils.enable_custom_op` 惰性导入的，所以这里**按需 import** 再 `hasattr` 判断」——即**绕过 A5 的全局禁用**。
- `[asc-fork] vllm_ascend/ops/mhc.py`、`csrc/mxfp4_nz/`、`csrc/online_mxfp4_gemm/`、`csrc/block_fp8_gemm/` 都是这条路线上的产物。

### 3.3 上游对 Qwen3.8-Flash-Next 的 950 明确「未启用」

`[asc-main] docs/source/tutorials/models/Qwen3.8-Flash-Next.md:9`：

> "The current version supports only Atlas A3 series hardware. **Atlas A2 series hardware and Ascend 950DT and 950PR products are not yet supported** and will be enabled progressively in future releases."

且该文档的验证环境是 **A3 + W8A8**（8×64GB），不是 950PR + MXFP4（本 checkpoint 是 MXFP4）。

**该文档「不是本树代码的说明」还有三条独立旁证**（追加取证，均为「穷尽式缺席」）：

1. 文档里用的三个环境变量 `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER` / `VLLM_ASCEND_ENABLE_QSA_E3V` / `VLLM_ASCEND_FORCE_QSA_REFERENCE`，在 `[asc-main]` 全树只出现在 **该 tutorial 自己**（`:124-126,155-156`）——`vllm_ascend/envs.py` 只定义了 9 个变量，其中没有它们。功能的真实开关位在 `ascend_config.py` 的 `AscendCompilationConfig`/`AscendFusionConfig` 与 `additional_config` 里。
2. **CI 里没有该模型的任何覆盖**：`.github/workflows/configs/nightly_config.yaml:42-44,179-181` 只有 `Qwen3.8-27B-w8a8-A2/A3`；`tests/e2e/nightly/single_node/models/configs/` 只有 `Qwen3.8-27B-w8a8-{A2,A3}.yaml`；`.github/workflows/misc/model_dataset_list.json` 无 Flash-Next 条目。
3. **特性矩阵里没有该模型的行**：`docs/source/user_guide/support_matrix/supported_features.md` 无 Flash-Next。

**结论：模型的 950 支持要我们自己做，上游没有可抄的现成 950 路径；连「A3 上能跑」这件事本身也只在文档里，代码不在公开 main 树**（对策见 §9-①）。950 的「基础设施」（A5 profile、MX 融合、aclnn 包、专用 ascendc_library 模式）倒是齐的。

---

## 4. 接入点地图（任务 2 · 本文档核心）

> 每条格式：**① 接入点** → `[vLLM]` 位置 → `[asc-*]` 对应机制 → 是否已有可复用实现 → 缺口。
> ⚠️ 通读结论：**`[asc-main]` 与 `[asc-fork]` 全树 `grep -i qwen4` 均只有文档命中，代码 0 命中**（`vllm_ascend/models/__init__.py` 的 `register_model()` 里没有任何 `Qwen4Exp*` 条目）。也就是说，**上游对 Qwen3.8-Flash-Next 的支持是「文档先行、代码不在公开树里」**（见 §9-①）。

### ① 模型注册与派发

- `[vLLM] vllm/models/qwen4_exp/__init__.py:22-46`：包级 `__getattr__`。`is_xpu() or is_tpu()` → `NotImplementedError`；`is_rocm()` → `amd/`；**其余一律 `nvidia/`**。
- **关键推论**：NPU 平台既不是 xpu/tpu 也不是 rocm → **不会报错，会静默加载 `nvidia` 实现**，直到撞上 CUDA-only 算子才炸。这对 MVP 是好事（见 §7 Step 3）。
- 注册表：`[vLLM] vllm/model_executor/models/registry.py:113-116`（`Qwen4ExpForCausalLM`）、`:601-604`（`Qwen4ExpForConditionalGeneration`，多模态）、`:697`（`Qwen4ExpMTP`，投机解码）。
- 覆盖机制：`[vLLM] registry.py:1122-1165 register_model()`，同名**静默覆盖**（`:1141-1147` 只打 debug 日志），接受 `"module:class"` 字符串懒加载。
- 插件侧先例：`[asc-main] vllm_ascend/models/__init__.py:4-89`（约 28 条 `ModelRegistry.register_model`，含 `DeepseekV4ForCausalLM`、`MiniMaxM3*`、`KimiK3*` 等）；`[asc-fork] vllm_ascend/models/__init__.py:4-13`（4 条）。入口函数由 entry point 保证被调用：`[asc-fork] setup.py:543-552` 的 `"vllm.general_plugins": [..., "ascend_model = vllm_ascend:register_model"]` → `vllm_ascend/__init__.py:72-75`。
- **是否已有可复用实现**：机制可复用（现成模板），`Qwen4Exp` 的条目**必须新增**。
- 上策（需上游 PR）：把 `ascend/` 作为第三个变体加进 `[vLLM] vllm/models/qwen4_exp/{nvidia,amd,ascend}/`，与既有 `nvidia/amd` 惯例对齐；否则只能用 `register_model` 覆盖。

### ② attention backend（含 QSA indexer）

- **该模型不使用 vLLM 的 attention backend 注册表**。`[vLLM] nvidia/qsa.py:64 Qwen4ExpQSAFlashAttentionBackend` / `:127 Qwen4ExpQSAFlashAttentionImpl` / `:252 Qwen4ExpQSAAttention`，在**构造期**硬绑定：`nvidia/qsa.py:389-401`（`self.attn_backend = …; self.impl = …`），并通过 `AttentionLayerBase.get_attn_backend()`（`:431-432`）暴露。
- 后果：`[asc-fork] vllm_ascend/platform.py:814-841 get_attn_backend_cls()`（`[asc-main] platform.py:238`）**只能决定平台默认 backend，无法自动接管模型包内私有 backend**。NPU 侧必须：改模型文件（patch 或 model copy）把 `attn_backend/impl` 换成 Ascend 实现。
- QSA 侧缓存（不走 paged KV 主链）：`[vLLM] common/qsa_cache.py:756/808/859`（`_QSAStateCache`、`QSAKeyStateCache`、`QSACompressedKeyCache`），metadata 由 triton kernel 生成、带 torch 回退（`:205` / `:464`，选择点 `:567-569`）。
- Indexer top-k：`[vLLM] nvidia/ops/qsa_indexer.py:488-490` 用 **CUDA-only** `torch.ops._C.cooperative_topk` / `torch.ops._C.persistent_topk`（实现在 `[vLLM] csrc/libtorch_stable/cooperative_topk.cu`）→ **NPU 必炸，必须替换**。注意 `use_cooperative_topk` 还附带 `current_platform.has_device_capability(90)` 判断（`:484-485`），NPU 上会落到**没有 guard 的** `persistent_topk` 分支。
- **★ 造模期第一硬失败点（比 topk 更早）**：`[vLLM] nvidia/qsa.py:169-170`（pin commit `ced6857` 版为 `:100-104`）在 `__init__` 里：
  ```python
  if not is_flash_attn_varlen_func_available():
      raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
  ```
  而 `is_flash_attn_varlen_func_available()`（`[vLLM] vllm/v1/attention/backends/fa_utils.py:346-375`）只认 CUDA/XPU/ROCm 三个来源，**NPU 没有来源 → 恒为 False → 第一个 QSA 层构造即抛异常**。也就是说：**不写任何 Ascend 实现、直接把 checkpoint 喂给 NPU 栈，会在造模阶段就停在 QSA 上**（这正是 MVP 探针要观察的第一个点）。
- 附带的两个「好消息」（说明不是所有 CUDA-only 路径都会挡路）：
  - `[vLLM] nvidia/low_latency_gemm.py:215-216`：skinny GEMM plan 不可用时**优雅回退**到 `torch.nn.functional.linear`（NPU 可走）。
  - `[vLLM] nvidia/ngram_embedding.py:291,303-351`：非 CUDA 走纯 torch 参考路径（见 §4⑥）。
- 配置侧参数（来自 `[model] config.json`）：`indexer_n_heads=4`、`indexer_kv_heads=1`、`indexer_head_dim=128`、`indexer_budget=2048`、`indexer_compress_ratio=4`；校验逻辑 `[vLLM] config.py:140-174`。
- 可复用：`[asc-main] csrc/attention/{compressor, compressor_metadata, indexer_compress_epilog_v2, kv_compress_epilog, lightning_indexer, lightning_indexer_v2, quant_lightning_indexer_v2, sparse_attention_score, sparse_flash_attention, kv_quant_sparse_flash_attention}`；backend 家族 `[asc-main] vllm_ascend/attention/{attention_v1,mla_v1,sfa_v1,dsa_v1,fa3_v1}.py`。**950 可用性**：`compressor` √、`fused_sparse_attention_overlap` √、`fused_lightning_indexer_manage` ×、`fused_scatter_copy_sparse_flash_attention` ×。
- **缺口**：Ascend 版 QSA backend/impl（可先做「参考实现」，即纯 torch 路径，`[vLLM] nvidia/qsa.py:370` 已有 `current_platform.is_cuda()` 分支意识）+ indexer topk 替换。

### ③ MoE / MXFP4 专家算子

- `[vLLM] nvidia/model.py:159 Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)` → 父类 `[vLLM] vllm/model_executor/models/qwen3_next.py:131`，其中 `:218-238 self.experts = FusedMoEFactory(...)`。参数：`num_experts=512`、`num_experts_per_tok=10`、`moe_intermediate_size=640`、`shared_expert_intermediate_size=640`（`[model] config.json`）。
- 插件接管方式（现成）：`[asc-fork] patch/platform/patch_fused_moe.py:30-57` 把 `FusedMoE` 工厂重绑成 `AscendMoERunner`（`[asc-main] patch/platform/patch_fused_moe.py`、`vllm_ascend/ops/fused_moe/fused_moe.py`）——**注意这是全局工厂替换，会影响所有 MoE 模型**，是「零模型改动」路线。
- 950 上更可能的走向是 CANN MegaMoe：`[asc-main] hardware_profile.py:64-67`（`CANN_MEGAMOE` / `CANN_MEGAMOE_MXFP`）+ `ascend_config.py:1059-1061`（950 MegaMoe 只支持 MXFP、hidden/intermediate 需落在离散集合）。
- **裁决（已用官方文档落实，不再是存疑）：本模型用不了 950 的 MegaMoe。** 本机 CANN 自带权威文档 `/usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/docs/zh/mega_moe.md:1072-1082`（**Ascend 950PR/Ascend 950DT** 段）规定：`hidden` 仅支持 4096/5120/7168；`num_topk` 仅支持 6、8；`intermediate_hidden` 仅支持 1024/2048/3072/4096/7168；`num_tokens ∈ [1,512]`；`dispatch_quant_mode` 仅支持 4（MXFP）且 `dispatch_quant_out_dtype` 仅支持 fp8_e5m2/e4m3fn。本模型 `hidden=2560`、`num_experts_per_tok=10`、`moe_intermediate_size=640` → **三项全部不符**。上游 `ascend_config.py:1059-1080 _is_a5_megamoe_supported_by_config` 的检查会返回 False（其离散集合与文档一致）。
  → **950 的 MoE 必须走通用路径**（`torch_npu.npu_grouped_matmul` + `npu_swiglu` + `npu_dynamic_quant`，或我们自己的 m13 整层 kernel）。这与我们的 mega kernel 目标反而是好事：没有被 CANN 融合算子锁死布局。
  → 注意：本机 CANN 9.1.0 **自带 `cann_ops_transformer`**（`/usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/`，含 `ops/mega_moe.cpp`），所以 `is_mega_moe_supported()`（= `find_spec("cann_ops_transformer")`）在本机为 **True**；但它的 import 需要 `ninja`（实测 venv 内 `ninja` 未安装会抛 `RuntimeError: Ninja is required to load C++ extensions`），且上面三项约束仍会否决本模型。
- 可复用：`[asc-main] ops/fused_moe/*`（`moe_mlp.py` 走 `torch_npu.npu_grouped_matmul` / `_C_ascend.grouped_matmul_swiglu_quant*`）、`csrc/gmm/`、`csrc/moe/`。注意：`_C_ascend.*` 在 A5 上默认被禁用（见 §3.2），所以 950 首版应优先押 `torch_npu.*` 原生算子。

### ④ GDN 线性注意力（36/48 层）

- `[vLLM] nvidia/model.py:209-215`：`layer_type=="linear_attention"` → `QwenGatedDeltaNetAttention`。层类型分布来自 `[model] config.json`：`full_attention_interval=4`，`layer_types` 48 项中第 4/8/… 为 `full_attention` → **36 层 GDN + 12 层 full attention**（与 `[ours] docs/10`、`docs/11` 的 36/12 口径一致）。
- 可复用（**复用度最高的一条**）：
  - `[asc-main] vllm_ascend/ops/gdn.py:122 AscendGatedDeltaNetAttention(GatedDeltaNetAttention)`
  - `[asc-main] vllm_ascend/ops/gdn_attn_builder.py:317 AscendGDNAttentionMetadataBuilder`、`:971 AscendGDNAttentionBackend`
  - `[asc-main] csrc/attention/recurrent_gated_delta_rule/`（decode 递推，950 √）、`csrc/moe/chunk_gated_delta_rule_fwd_h/`、`csrc/moe/chunk_fwd_o/`、`csrc/moe/causal_conv1d/`、`csrc/attention/inplace_partial_rotary_mul/`
  - `[asc-fork] csrc/attention/{causal_conv1d_qkv, fused_gdn_gating, fused_qkvzba_split_gating, fused_rearrange_qkv_l2norm}`（Qwen 血统的 GDN prolog 融合，与 `[ours] m9_gdn_prolog` 一一对应）
  - 补丁：`[asc-fork] patch/worker/patch_qwen3_5.py`（重绑 `QwenGatedDeltaNetAttention` 等）
- **需核对**：fork 的融合布局是为 Qwen3.8-27B（`in_proj_qkvzba`）做的；Qwen4Exp 是 Qwen3-Next 血统，本 checkpoint 的 GDN 权重名是 `linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d.weight, norm.weight, out_proj.weight, A_log, dt_bias}`（各 36 份，实测来自 `[model] model.safetensors.index.json`）。是否与 Ascend 侧融合入口的入参约定一致，**需实跑核对**。

### ⑤ hyper-connection mixer

- `[vLLM] nvidia/hyperconnection.py:50 GatedResidual`，接口是**延迟合并**三件套：`mix()`（`:128`）、`combine_and_mix()`（`:153`）、`combine()`（`:190`）；每层两个实例 `attn_hyper_connection` / `mlp_hyper_connection`（`[vLLM] nvidia/model.py:259-274`），模型尾部还有 `hyper_connection_mixer`。
- 算子：`[vLLM] nvidia/ops/hc.py:428-453` 注册 5 个 `torch.ops.vllm.qwen4_exp_*`：`grouped_gemma_rmsnorm`、`hc_silu`、`hc_gate_mix`、`hc_combine`、`hc_combine_norm`（**triton 实现**，`[vLLM] nvidia/ops/hc.py:12-292`）。
- 参数：`hc_count=4`、`hc_lowrank=320`（`[model] config.json`）。权重证据：每层 4 个 × 2 组 = 8 个张量 `{attn,mlp}_hyper_connection.{input_mix_weight_down, input_mix_weight_up, block_inject_weight, hc_norm}.weight`。
- Ascend 侧现状：`[asc-main] csrc/moe/hc_pre`、`csrc/moe/hc_post`，但语义属于 **DeepSeek-V4 的 hc_pre/hc_post**（`[asc-main] vllm_ascend/models/deepseek_v4/model.py:745/751` → `torch.ops._C_ascend.npu_hc_pre_v2` / `npu_hc_post`）；`[asc-fork] vllm_ascend/ops/mhc.py` 是 **Sinkhorn 式** hc 混合的参考实现（`hc_split_sinkhorn_ref`），也不是同一套。
- **结论：必须新写**（对应我方 M36 hc mixer）。

### ⑥ PLE ngram embedding（最大单项缺口）

- `[vLLM] nvidia/ple_layer.py:66 Qwen4ExpPLELayer(MambaBase)`；只在 `ple_layer_ids` 指定的层存在 —— `[model] config.json` 是 `ple_layer_ids=[2]`（**全局仅 1 层**，1-based）。
- 表规模（由配置推导）：`ngram_heads=(ngram_size-1)*heads_per_ngram=(3-1)*8=16`，`head_dim=ple_embed_dim/ngram_heads=2560/16=160`，行数 ≈16×2e7 → `16×2e7×160×2B ≈ 95.4 GiB`。checkpoint 里实际存为 **128 个分片**：`model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`（实测 index.json；与 `split_ngram_parts=128` 一致）。
- 存储后端二选一：`[vLLM] common/ngram_embedding.py:323 Qwen4ExpPLEDeviceEmbedding`（显存）vs `:385 Qwen4ExpPLEPinnedHostEmbedding`（host pinned + UVA 查表 + 旁路预取），由 `engram_config.cpu_offload` 决定（`[vLLM] nvidia/ngram_embedding.py:210-215`）。**本机 128 GiB HBM + 182 GB 权重 → 必须 host 侧**。
- 最关键的可行性发现：`[vLLM] nvidia/ngram_embedding.py:291` 按 `input_ids.is_cuda` 分流，**非 CUDA 走纯 torch 参考路径**（`:303-351`）。→ 在 NPU 上 PLE 的 ngram id 计算与查表**可以先用 torch 路径跑通**（慢但不挡 MVP）。
- Ascend 侧现状：**0 实现**（`[asc-main] csrc/attention/ngram_spec_decode` 是**投机解码**用的 ngram op，不是 PLE 表；`vllm_ascend/models/` 下无任何 PLE 相关文件）。
- **结论：必须新写**（稀疏行查找 + host 驻留 + 128 分片映射）。这是全模型唯一「放不进 HBM、必须流式/稀疏」的部分。
  ⚠️ **排期按用户裁决**：现阶段**不实现**（§12.4）——PLE 表放 host、将来走 PCIe-through MTE2 直读；当前只要求「能加载 + 官方 pinned-host 参考路径能跑通」，kernel 后置。上面那句「相对官方实现最有可能做出真实收益」是**性能阶段**的判断，不是本阶段的目标。

### ⑦ 量化方法（`quant_method="ascend"`）

- checkout 侧声明（实测 `[model] config.json`）：
  ```json
  "quantization_config": {
    "quant_method": "ascend",
    "format": "mxfp4-pack-quantized-e8m0",
    "bits": 4, "element": "e2m1", "group_size": 32, "scale_dtype": "e8m0",
    "packing": "uint8-nibble-lohi",
    "rounding": "round-to-nearest, ties away from zero (torch_npu round_mode=round)",
    "scale_convention": "2**(floor(log2(max_abs))-2), saturate at +-6 (OCP); all-zero block -> 0x00",
    "npu_reference": "torch_npu.npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode=\"round\")",
    "quantize_linear_attn": false
  }
  ```
  量化范围 = **MoE experts（`mlp.experts.gate_up_proj` / `down_proj`）+ `mlp.shared_expert.{gate,up,down}_proj.weight`，共 240 个张量**；其余（n-gram embedding / linear_attn / self_attn / router / norms / hyper-connections / embed & lm_head / vision / MTP / ple）**保留 BF16**（`[model] README.quant.md`、`quantization_plan.json`）。
- vLLM 侧：`[vLLM] vllm/model_executor/layers/quantization/__init__.py:50 QUANTIZATION_METHODS`（注册名清单在 `:15-49`）**不含 `"ascend"`** → 不注册就 `ValueError: Invalid quantization method: ascend`（`:113-114`）。模型侧对它**无特殊分支**（`[vLLM] nvidia/model.py:85-91` 只特判 `modelopt_fp4`）。
- vllm-ascend 侧（现成机制）：
  - 常量：`ASCEND_QUANTIZATION_METHOD = "ascend"`（`[asc-main] utils.py:56`；`[asc-fork] utils.py:49`）
  - 注册：`@register_quantization_config(ASCEND_QUANTIZATION_METHOD)` → `[asc-main] vllm_ascend/quantization/configs/modelslim_config.py:336`（类 `AscendModelSlimConfig`）
  - 平台白名单：`[asc-main] vllm_ascend/platform.py:92-… supported_quantization`（含 `"ascend"`）
  - 方案注册表：`[asc-main] vllm_ascend/quantization/methods/registry.py:21-62`（`register_scheme(quant_type, layer_type)`）；`QuantType` 枚举 `quant_type.py:22-33`
  - MXFP4 相关方案（main 与 fork 同名）：`W4A4_MXFP4`（linear `w4a4_mxfp4.py:106`、moe `:211`）、`W4A8_MXFP`（`w4a8_mxfp4.py:40/99`）、`W4A16_MXFP4`（`w4a16_mxfp4.py:55`，仅 moe）、`W8A8_MXFP8`（`w8a8_mxfp8.py:43/194`）
- **两个硬缺口**（本方案最重要的落地风险之一）：
  1. **`AscendModelSlimConfig` 依赖 `quant_model_description.json`**（`[asc-main] vllm_ascend/quantization/utils.py:83-138 detect_quantization_method`，`:115-117` 命中该文件 → `"ascend"`）。而 `[model]` 目录里**没有这个文件**（只有 `quantization_plan.json` / `validation_report.json`）。→ 要么按 240 条量化名单**生成** `quant_model_description.json`，要么**新写一个 AscendQuantConfig** 直接读 `config.json` 的 `quant_method="ascend"` + 按名字规则映射 scheme。
  2. **权重布局是否与既有 scheme 的加载约定一致，尚未证实**。`[model] README.quant.md` 末句自己也写着：「如需 vllm-ascend 直接加载，**需确认该 fork 对 MoE W4A4_MXFP4 的加载约定与本布局一致**」。本 checkpoint 的布局是：量化张量原名存 packed uint8（最后维减半）+ `<name>.weight_scale`（E8M0, group 32）；3D expert 张量 `[E,out,in] → packed [E,out,ceil(in/2)] + scale [E,out,in/32]`。→ 见 §9-②。
- 好处：`[ours]` 的 m1/m2/m3/m5/m13 正是 MXFP4 量化 + GEMM 原语（`docs/13-mx-quant-primitives.md` 已与官方逐句对拍），**可以直接作为该 scheme 的 kernel 后端**。

### ⑧（附加）MTP / 投机解码

- `[vLLM] nvidia/mtp.py:362 Qwen4ExpMTP`，注册在 `registry.py:697`；spec config 重写 `[vLLM] vllm/config/speculative.py:52`（`"qwen4_exp_mtp"`）、`:844-861`。
- 本 checkpoint 自带 MTP 段：`mtp = {hybrid: True, layer_types: ["full_attention"], num_hidden_layers: 1}`、`mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=False`。
- vllm-ascend 侧：`[asc-main] vllm_ascend/spec_decode/__init__.py`（`eagle`/`eagle3`/`mtp` → `AscendEagleProposer`，门控 `speculative_config.use_step3p5_mtp()`）、`[asc-fork] patch/worker/patch_qwen3_5.py`（`qwen3_5_mtp` 相关重绑）。上游 A3 tutorial 用的是 `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'`。
- **注意**：`method` 名与我们的 checkpoint 配置对不上（`qwen3_5_mtp` vs `qwen4_exp_mtp`）→ 哪个生效需实跑（§9-⑦）。MVP 建议**先关 MTP**。

---

## 5. 需要新增/改动的文件清单（任务 3）

### 5.1 A 组：**零改动可复用**（vllm-ascend 已有，直接享受）

| 能力 | 位置 | 备注 |
|---|---|---|
| Hybrid/Mamba KV 与调度适配 | `[asc-main] patch/platform/{patch_mamba_config,patch_mamba_manager}.py`、`core/{kv_cache_interface,single_type_kv_cache_manager}.py` | 本模型 `IsHybrid`+`HasInnerState`，走这条 |
| GDN 层与 metadata builder | `[asc-main] ops/gdn.py:122`、`ops/gdn_attn_builder.py:317/971` | 复用度高，需核布局 |
| GDN/Mamba triton+csrc 算子 | `[asc-main] csrc/attention/recurrent_gated_delta_rule`、`csrc/moe/{chunk_gated_delta_rule_fwd_h,chunk_fwd_o,causal_conv1d}` | decode 递推 + prefill chunk |
| MoE 工厂接管 | `[asc-main] patch/platform/patch_fused_moe.py`、`ops/fused_moe/*` | 全局替换，注意副作用 |
| MXFP4/MXFP8 量化方案框架 | `[asc-main] quantization/methods/{w4a4_mxfp4,w4a8_mxfp4,w4a16_mxfp4,w8a8_mxfp8}.py` | 缺的是「本 checkpoint 的入口」 |
| 稀疏/压缩 attention 算子 | `[asc-main] csrc/attention/{compressor,indexer_compress_epilog_v2,lightning_indexer_v2,...}` | QSA indexer 的落点 |
| 投机解码 proposer | `[asc-main] spec_decode/__init__.py`、`[asc-fork] patch/worker/patch_qwen3_5.py` | MVP 先关 |
| **950 专用构建+加载模式** | `[asc-fork] CMakeLists.txt:84-98`、`vllm_ascend/device/mxfp_compat.py:79-93`、`csrc/block_fp8_gemm/`、`csrc/quant_fusion/` | **A5 上挂我们 kernel 的唯一已验证路径** |
| 950 本机运行脚本/测速工具 | `[asc-fork] README.md:100-165`、`examples/ascend950_speed_demo.py` | 直接用于 §8 基线 |

### 5.2 B 组：**vllm-ascend 侧需新增/改动**

| # | 文件/位置 | 动作 | 理由 / 证据 |
|---|---|---|---|
| B1 | `vllm_ascend/models/__init__.py` | 增 3 行注册：`Qwen4ExpForConditionalGeneration` / `Qwen4ExpForCausalLM` / `Qwen4ExpMTP` | `[asc-main] models/__init__.py:4-89` 现成模板；`[vLLM] registry.py:601/113/697` |
| B2 | `vllm_ascend/models/qwen4_exp/`（新包：`model.py`、`qsa.py`、`indexer_qsa.py`、`hyperconnection.py`、`ple_layer.py`、`mtp.py`、`model_state.py`） | 新写（对齐 `[vLLM] nvidia/` 目录结构；可先「nvidia 结构的 Ascend 变体 + 参考实现」） | `[vLLM] vllm/models/qwen4_exp/nvidia/*` 是唯一可抄的完整实现 |
| B3 | 量化入口：`quant_model_description.json` 生成器 **或** `vllm_ascend/quantization/configs/qwen4exp_config.py`（新 `AscendQuantConfig`） | 新写 | `[asc-main] quantization/utils.py:83-138` 依赖 ModelSlim 描述文件；`[model]` 无该文件 |
| B4 | `vllm_ascend/patch/worker/patch_qwen4_exp_qsa.py`（新） | 替换 `_topk`（CUDA-only）与 QSA backend 绑定 | `[vLLM] nvidia/ops/qsa_indexer.py:488-490`；`[vLLM] nvidia/qsa.py:389-401` |
| B5 | `vllm_ascend/attention/` 新增 QSA/压缩 KV backend 类 | 新写（可继承 `[asc-main] attention/dsa_v1.py` 或 `mla_v1.py` 的稀疏/压缩机制） | `[asc-fork] platform.py:814-841` 的映射表只有 4 个位置，需扩 |
| B6 | `vllm_ascend/patch/__init__.py` | 追加文档块（项目规范强制） | `[asc-fork] patch/__init__.py` 1102 行纯文档，新增补丁必须登记 |
| B7 | `CMakeLists.txt` + `csrc/<新目录>/` | 增加 **A5 专用 `ascendc_library`**（照抄 block_fp8_gemm 模式），把我们的整层 kernel 挂上 | `[asc-fork] CMakeLists.txt:84-98`；A5 上通用 `vllm_ascend_kernels` 不编译（`:76-82`） |
| B8 | PLE 加载/驻留模块（新，如 `vllm_ascend/models/qwen4_exp/ple_ngram.py`） | 新写：host pinned 表 + 128 分片映射 + 稀疏行查找 | `[vLLM] nvidia/ngram_embedding.py:385-448`（分片加载参考）、`common/ngram_embedding.py:385-502`（pinned host 参考） |

### 5.3 C 组：**必须新写的 kernel**，及与我方已合并资产的对应

| 模型算子 | 我方现有资产 | 复用判断 | 对应 Ascend 侧接口 |
|---|---|---|---|
| MoE 整层（router→permute→grouped MXFP4 GEMM→SwiGLU→unpermute） | `m13_moe_layer`（S1–S10 单次 mix 启动，682 判据）+ `m3_grouped_gemm`、`m5_swiglu_quant`、`m7_router_topk`、`m8_permute` | **可直接复用**，需包成 `W4A4_MXFP4` MoE scheme 的后端 | `[asc-main] ops/fused_moe/moe_mlp.py`、`csrc/gmm/` |
| GDN decode 递推 + prolog + 输出门控 | `m14_gdn_layer`（S1–S7 整层）、`m9_gdn_prolog`、`m4_gdn_recurrent`、`m12_rmsnorm_gated`、`m11_bf16_gemm` | **可直接复用**（GDN 是 36/48 层的大头） | `[asc-main] ops/gdn.py:122`、`csrc/attention/recurrent_gated_delta_rule` |
| GDN prefill（chunk 扫描） | M34（GDN prefill，已随 `e035b0f` 合入 main：`m18_gdn_prefill`） | 需与 `csrc/moe/chunk_gated_delta_rule_fwd_h`+`chunk_fwd_o` 对齐契约 | 同上 |
| QSA indexer + 压缩 KV + 稀疏 attention | M35（QSA indexer，已随 `a208849` 合入 main：`m19_qsa_indexer`）、`[ours] docs/11` | 需新写；950 上 compressor 可用、`fused_lightning_indexer_manage` 不可用 | `[asc-main] csrc/attention/compressor`、`indexer_compress_epilog_v2` |
| hyper-connection mixer（mix/combine_and_mix/combine + per-branch norm） | M36（hc mixer，已随 `ea4c818` 合入 main：`m20_hyperconn`；并入 per-layer kernel 的融合见 M58 `7ca5922`）；`m6_rmsnorm` 供 GroupedGemmaRMSNorm | 语义与 DeepSeek 的 hc_pre/hc_post 不同，**必须新写** | `[asc] csrc/moe/hc_pre|hc_post`（仅作结构参考） |
| PLE ngram 稀疏行查找 | 无（全新） | 必须新写，但**现阶段不实现**（§12.4：表放 host、PCIe 直读，后置） | 无对应 |
| RMSNorm / RMSNormGated / Add+RMSNorm | `m6_rmsnorm`、`m12_rmsnorm_gated` | 可复用 | `[asc-main] csrc/moe/{add_rms_norm_bias,rms_norm_cast}`、`csrc/attention/rms_norm_dynamic_quant` |
| MXFP4 量化原语（激活量化 + e8m0 scale） | `m1_mxfp4_gemm`、`m2_mxfp4_quant`、`m5_swiglu_quant`、`docs/13` | 可复用；与 `torch_npu.npu_dynamic_mx_quant` 已对拍 | `[asc-main] csrc/attention/rms_norm_dynamic_quant`、`quantization/methods/*mxfp4*` |
| 48 层循环骨架 | `m15_layer_loop`（已随 `2b9cb36` 合入 main）、`[ours] docs/12` | **不需要**（用户裁决：循环留在框架侧，我们只做 per-layer 算子，§6.1） | vLLM 自身的层循环 |

> 工程落差提醒：我方 kernel 是 `.asc` + `[ours] CMakeLists.txt` 的独立构建；要挂进 vllm-ascend 需按 **B7** 包成 `ascendc_library`（或走 aclnn vendor 包），两套构建系统的 tiling/入参契约需要一层 adapter。

---

## 6. 接入形态（任务 4 · 已被用户裁决收敛）

> **用户裁决（2026-09-26，经 tower 转达）**：「**不，每个 kernel 只跑一层，这个对 vllm 的修改会比较小**」。
> ⇒ 48 层循环、调度、KV cache、采样**全部留在框架侧**；我们只提供 **per-layer 算子**，vllm-ascend 侧改动的目标是「尽可能小」。
> ⇒ 原文「custom op vs 接管整个 forward」的**并列对比已作废**（保留在 §6.4 作决策记录）。本节改为**具体接入设计**。
> ⇒ 同时生效的口径：现阶段**不关心性能**（只做「设计输入」类估算）；`m=4097` 是 prefill、decode 是 m=1 且 context=4096、无 batch>1；PLE/ngram 表**放 host 侧走 PCIe-through MTE2 直读，暂不实现**；其余权重 device 常驻（74.3 GiB）；**层 kernel 必须包含 hc mix**。

### 6.1 per-layer 算子的接入设计（最终形态）

**替换点**：`[vLLM] vllm/models/qwen4_exp/nvidia/model.py:173 Qwen4ExpDecoderLayer`，其 `forward` 在 `:276-331`，由模型循环在 `:535-543` 调用：

```
layer(hidden_states, prev_block_output, prev_injection,
      positions, input_ids, query_start_loc, ngram_context)
   → (hidden_states', mlp_out, injection')
```

这里面按顺序做了四件事（`:287-330`）：① 若有 pending HC state 先 `attn_hc.combine(...)`；② `ple(...)`（仅第 2 层，1-based → 0-based 层 1）；③ `attn_hc.combine_and_mix()` / `mix()` → `(hidden_states, block_input, injection)`；④ 注意力（GDN 或 QSA）→ `mlp_hc.combine_and_mix(...)` → `mlp(block_input)`。**这就是我们要在一次算子调用里跑完的一层**（含用户要求的 hc mix）。

**接入方式（最小改动的一种）**：不新写整模型类，而是**只替换层模块的 forward**：

1. 在 `vllm_ascend/models/` 下新增一个模块（例如 `qwen4_exp_layer.py`），`from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer`，把 `Qwen4ExpDecoderLayer.forward` 重绑为「调用我们的 op」的薄 wrapper。触发时机沿用 general-plugin 的 `register_model()`（`[asc-fork] setup.py:543-552`、`vllm_ascend/__init__.py:72-75`）——**注册函数里不需要 `ModelRegistry.register_model`**，只要保证 patch 在模型导入前生效（参考 `[asc-fork] patch/platform/patch_fused_moe.py:25-28` 的 import-order 说明）。
2. 算子本身按 **模块级边界**注册（与 `[ours] docs/15 §8` 的结论一致）：`vllm_ascend/ops/register_custom_ops.py:222-298` 的 `direct_register_custom_op(op_name, op_func, fake_impl, mutates_args, dispatch_key="PrivateUse1")` 适合 Python 侧复合算子；C++/AscendC 侧走 `torch.ops._C_ascend.<name>`（现例：`npu_causal_conv1d_custom`、`npu_fused_qkvzba_split_gating`、`npu_recurrent_gated_delta_rule`）。
3. **in-place 张量必须申报**：主 KV cache、GDN 的 `ssm_state`/`conv_state`、QSA raw key ring + compressed key cache、PLE conv state、hc 状态都是 in-place（`docs/15 §8.3` 第 2 条）→ 不申报 `mutates_args` 会破坏 vLLM 的图捕获与别名分析。
4. **一个 chunk 的契约**：prefill 是 chunked prefill，算子的入参是「带初态、m ∈ [1, M_MAX] 任意值」（`docs/15 §8.3` 第 3 条）；注意 vllm-ascend 侧 GDN state 布局是 `[N,H,K,V]`，`ops/gdn.py:602` 有 `transpose(-1,-2)`（与 checkpoint/CANN 的 `[Nv,V,K]` 相反）。
5. **权重按 checkpoint 原始排布消费**（用户约束：kernel 直接支持原始 layout）：MoE `experts.gate_up_proj` 融合张量、GDN in_proj 的 `q|k|v|z|b|a` 行拼接、HC 的 down+inject 打包。

**框架侧必须保留什么**：

| 必须保留 | 理由 / 位置 |
|---|---|
| 48 层 host 循环 + PP 分支 + `IntermediateTensors` | `[vLLM] nvidia/model.py:489-590`（首/末 PP rank 行为不同） |
| PLE 预取调度（`layers[start_layer]` 预取下一层的 PLE） | `[vLLM] nvidia/model.py:515-522, 527-534` —— 若将来 PLE 走我们的 kernel，这个调度钩子仍应保留在框架侧 |
| deepstack（VL）注入与「combine 必须落盘」的约束 | `[vLLM] nvidia/model.py:544-566`（这一条直接决定层界张量不能全部藏在算子里，见 §6.2） |
| 尾部 `hyper_connection_mixer.combine_and_mix` + `_mtp_hidden_buffer` 拷贝 | `[vLLM] nvidia/model.py:577-590` |
| KV/状态的内存分配与 block table（KV cache manager、mamba manager） | `[asc-main] patch/platform/{patch_mamba_config,patch_mamba_manager}.py`；我们的算子只**读写框架给的 cache 张量**，不自己分配 |
| 量化方法解析（`quant_method="ascend"`） | 见 §4⑦；层算子只消费反量化/scale 张量 |

**层界语义必须逐字保真**：官方用的是**延迟合并**（`combine_and_mix`，`[vLLM] nvidia/model.py:153/190` 与 `:287-330`），`mlp_out` 与 `injection` 要被**下一层**的 mix 消费。我们的算子返回值的语义若与这三元组不一致，层循环就会静默错——这是验收时必须单独对拍的一条（对应 `docs/17` 的段级判据）。

### 6.2 层界三张量的两种持有方案（用户点名要评估的点）

层界张量（每 token）：`multi_hidden ∈ [T, hc_count·H] = [T, 10240]`、`pending block ∈ [T, H] = [T, 2560]`、`injection ∈ [T, hc_count] = [T, 4]`。两种方案：

| | **方案 A：框架侧持有并逐层传递**（建议） | 方案 B：算子内部保存（跨层状态） |
|---|---|---|
| 做法 | 保持官方层循环签名不变：框架把这三张量（外加 KV/state 句柄、positions、`input_ids`、`query_start_loc`、`ngram_context`）传给我们的算子，算子返回新的三张量 | 算子在插件内维护持久 buffer，层循环只传 `hidden_states` 与输入 |
| 框架改动量 | **最小**：只替换层模块 forward 的**函数体**，不动层循环、不动 PP/deepstack/final-mixer 路径 | 大：必须改 `[vLLM] nvidia/model.py:489-590` 的层循环本身（三张量的传递与 `:544-566` deepstack 注入、`:577-590` 尾部 mixer 都要跟着改） |
| 与官方语义一致性 | 完全一致（就是官方的数据流） | 需要重建官方数据流，且 **deepstack 注入要求 combine 在特定时刻落盘**（`:544-566`），B 方案要额外在算子内暴露「落盘」时机 → 引入框架不认识的私有中间表示（用户明确禁止） |
| PP / 多 rank | 天然支持（首/末 rank 行为由框架处理） | 需要我们自己处理 rank 边界 |
| 层内融合自由度 | 高（算子内部随便融合；三张量只是 ABI） | 更高（连层界往返都省了） |
| 额外开销 | 每层 3 个张量的往返读写 | 无 |
| 结论 | ✅ **首期采用**。用户已明确「暂不关心性能」，而这一方案的框架改动最小、语义最保真 | ❌ 留作后续（若将来层界往返成为瓶颈，可在「层循环仍由框架驱动」的前提下把相邻两层合成一个 op，而不是改成私有状态机） |

> 一句话给决策者：**三张量由框架侧持有**（方案 A）。我们省下的是「层内四次模块调用的融合」，不是「层间数据流」——这与用户「每个 kernel 只跑一层」的裁决完全一致。

### 6.3 最小框架改动清单（接入点只有三个核心项 + 两项绕不开）

按「改动尽量小」重排（与 §5.2 的完整清单对照）：

| 优先级 | 接入点 | 文件量级 | 是否可省 |
|---|---|---|---|
| 1 | **量化方法类**：让 `quant_method="ascend"` 能解析本 checkpoint（生成 `quant_model_description.json` 或写一个 AscendQuantConfig） | 1 个文件 / 或 1 个 json | **绕不开**——不解决连模型都构造不出来（§4⑦） |
| 2 | **层模块替换**：`Qwen4ExpDecoderLayer.forward` → 调我们的 per-layer 算子（§6.1） | 1 个新模块（+1 行 patch 登记） | **绕不开**——这是「我们的 kernel 进 vllm-ascend」的本体 |
| 3 | **算子注册**：AscendC 算子 → `torch.ops._C_ascend.*`（或 aclnn vendor 包）。**950 上必须走专用 `ascendc_library`**（`CMakeLists.txt` 加一段，照抄 `[asc-fork] CMakeLists.txt:84-98`），不能依赖通用 `vllm_ascend_kernels`（A5 不编译） | 1 个 csrc 目录 + CMake 段 | 绕不开 |
| 4 | **QSA 侧**：`is_flash_attn_varlen_func_available()` 那道 guard + indexer topk（CUDA-only） | 1 个 patch 文件 | 绕不开（否则造模即失败，§4②） |
| 5 | **PLE/ngram**：表放 host、PCIe 直读——**用户已明确暂不实现**，但至少要能加载（host 驻留） | 后续 | 现阶段可用官方 pinned-host 路径顶住 |
| — | `ModelRegistry.register_model` 换整套模型类 | — | **不需要**（走 §6.1 的层替换更小） |
| — | `patch/__init__.py` 增加文档块 | 文档 | 项目规范强制 |

### 6.4 旧候选对比与建议（**已作废，仅作决策记录**）

> 下面这张表与理由清单是裁决前的分析，保留用于回溯「为什么最终选了 per-layer 形态」。**不作为行动依据。**

#### 6.4.1 三个候选（历史）

| | R1 细粒度 custom op | R2 完全自接管 forward | **R3 注册 Ascend 模型类 + 层内整层 kernel**（建议） |
|---|---|---|---|
| 做法 | 保留 vLLM 模型结构与 python 层循环，只把个别算子换成 NPU 实现（`direct_register_custom_op` 重注册同名 `torch.ops.vllm.qwen4_exp_*`，或 aclnn vendor 包 + `torch_npu.*`） | 用 `ModelRegistry.register_model` 覆盖 arch，插件侧自己写整模型 forward（调度/KV/采样可仍用 vLLM） | R2 的变体：注册 Ascend 版 `Qwen4ExpForConditionalGeneration`/`DecoderLayer`，但**只替换层内实现**——每层 forward 内部一次 kernel 启动跑完 HC-mix→attn/GDN→HC-combine→MoE |
| 「一层一个 kernel」契合度 | **差**：一层跨 4 个 vLLM 模块（`nvidia/model.py:276-331` 的 forward 链），细粒度替换只能在模块边界切 | 好 | **好**（天然对齐） |
| mix(1,2) 用满核 / 自管同步 | 可以（kernel 自己 launch），但同步点被 vLLM 模块边界切开 | 可以 | **最好**（层内不许插 host 同步，正是我们的设计约束） |
| 与 vLLM 生态兼容（PP/EP/ACLGraph/EPLB/piecewise） | 最好 | 需自己重接 | 好（外层 python 循环、KV、采样、并行策略全部沿用 vLLM） |
| 改动面 | 小（但打不完，QSA/hc/PLE 都要打） | 大 | 中 |
| 风险 | **最大的一种隐性风险**：模型包内私有 attention backend（`nvidia/qsa.py:389-401` 构造期硬绑定）无法靠 `get_attn_backend_cls` 接管；`_topk` 之类的 CUDA-only 调用点很多，逐个打补丁会持续漂移 | 上游模型更新后需同步维护 | 中：模型类是我们自己的，上游更新时按 diff 对齐 |
| 950 现状约束 | A5 上 `enable_custom_op()` 为 False、通用 AscendC 库不编译 | 同左 | 同左（都需走 fork 的专用 `ascendc_library` + 按需 import 模式） |

#### 6.4.2 当时的建议（历史；注意：裁决后落地形态是 §6.1 的「层模块替换」，与当时的 R3 措辞相近但**不需要**注册整套模型类）

理由（按重要性）：

1. **一层一 kernel 是我方硬约束，而 vLLM 的一层在 python 侧是 4 个模块**（`[vLLM] nvidia/model.py:276-331`）。只有替换层 forward 才能把「HC mix → attn → HC combine → MoE」收进一次 launch；R1 做不到。
2. **必须接管的东西本来就「不开放」**：QSA backend 是模型包内私有类（`nvidia/qsa.py:64/127/252`），indexer topk 是 CUDA-only（`qsa_indexer.py:488-490`），PLE 表在 device 与 pinned-host 之间有独立后端选择（`common/ngram_embedding.py:323/385`）。这些都不是「注册一个 op 就能换掉」的点。
3. **R3 保住了 vLLM 的所有工程能力**：PP、EP/EPLB、Mamba/混合 KV 管理（`patch_mamba_config/_manager`）、前缀缓存、ACLGraph、采样与投机解码都在外层，不需要我们重做。
4. **A5 的现实**：`vllm_ascend_kernels` 在 950 不编译、`enable_custom_op()` 为 False，所以无论走 R1 还是 R3，**都要用 fork 已验证的「专用 `ascendc_library` + 按需 import 扩展」模式**（`[asc-fork] CMakeLists.txt:84-98`、`device/mxfp_compat.py:79-93`）。这一点上 R1 并不比 R3 省事。
5. **图捕获**：首版一律 `--enforce-eager`。整层 kernel 若含 host 侧同步/动态 tiling，会打断 ACLGraph 捕获；等 kernel 契约稳定后再上 `FULL_DECODE_ONLY`（参考上游 A3 命令里的 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`）。

**不建议**：纯 R1（补丁会无限增长）；也不建议完全自接管调度/KV（收益与风险不成比例，且 vLLM 的混合 KV 管理我们已经需要复用）。

### 6.5 必须知道的两条「别人已经做过的整模型融合」路线（R4/R5，**仍然有效，仅作参考**）

这两条不走手写 kernel，但会影响我们的取舍，必须写清：

- **R4 — CANN 编译器级 Super Kernel（官方设施，且在 A5 可用）**：`[asc-main] vllm_ascend/ascend_config.py:125-126` 有 `enable_static_kernel` / `enable_super_kernel` 两个开关，依赖关系 `:148-170`（super→static→npugraph_ex，`NPUGRAPH_EX` 不被支持时自动降级）；`compilation/compiler_interface.py:127-135` 打开 `static_kernel_compile` + `super_kernel_optimize`；`compilation/acl_graph.py:212` 把**整张捕获图**包进 `super_kernel_scope("full_model", …)`，`:246-248` 调 `aclgraph.super_kernel_optimize(...)`；底层是 `torch.npu.super_kernel_scope_begin/end`（`utils.py:1028-1037`）。A5 profile **含 `NPUGRAPH_EX`**，所以这条路在 950PR 上开着。
  → 对我们的意义：这是「把整个 forward 融成更少 kernel」的**官方替代方案**，且不需要我们写 kernel。但它是**编译器在图级别做算子融合**，收益来自减少 kernel launch/访存往返，**不改变单个算子的实现质量**——与我们的「一层一 kernel + mix(1,2) 用满核 + 自管同步」是**互补而非替代**：我们的 kernel 决定「层内算得有多快」，Super Kernel 决定「层间能不能再省 launch」。建议：**mega kernel 路线照走；等项目有线后，把 Super Kernel 作为「层间优化」的第二阶段实验**（提示：CANN 8/9 的 Super Kernel 对自定义 AscendC 算子的支持范围需要实跑确认）。
  → 另注：上游确实存在 `refs/heads/rfc/megakernel` 分支（见 §1 的 `ls-remote --heads` 输出），但**不在 main 上**，本调研未取（未下载、未评估内容）。
- **R5 — `xlite` 外部整模型运行时（不适用本模型）**：`[asc-main] vllm_ascend/xlite/` 通过 `from xlite._C import …`（`xlite/xlite.py:36`）加载第三方原生运行时，用已加载的 vLLM 权重**整批执行 forward**，按 token 预算路由（默认 decode-only，`full_mode` 覆盖 prefill+decode；`docs/source/user_guide/feature_guide/graph_mode.md:300-315`）。但它按 `config.json` 的 `architectures` 注册适配器，支持的集合是 `LlamaForCausalLM / Qwen2 / Qwen3 / Qwen3VL / Qwen3Moe / Glm4Moe / DeepseekV3 / DeepseekV32 / GlmMoeDsa / MiniMaxM2` 等（`xlite/xlite.py:317-322,483,500,523,592,639`），**`Qwen4ExpForConditionalGeneration` 不在其中**，未注册的 arch 直接 `raise ValueError(f"{architecture} not supported!")`（`:672-674`）。
  → 结论：**xlite 不能承载本模型**；它同时也说明「替换整个 forward」这件事官方已经有一条成型路径（万一将来对方支持了 Qwen4Exp，可作对照基线）。

---

## 7. MVP：「在 950PR 上跑出一个 token」的最短路径（任务 5）

> 每一步都标：**前置**、**预估代价**、**风险**、以及 `[需实跑验证]`。
> 全程不涉及重编译的说法只在 Step 0 成立；Step 1/2 是重编译，**必须在有内存窗口时再开工**（容器 cgroup 32 GB，单卡被 10+ worker 共用）。

| Step | 动作 | 前置 | 预估代价 | 风险 |
|---|---|---|---|---|
| **S0** | 冻结环境：确认 CANN 9.1.0 + `cxx_abi=1` + venv `torch 2.10.0+cpu` / `torch_npu 2.10.0.post4`；补 `triton-ascend==3.2.2` | 无（今天可做） | 分钟级 | 低。**已实测**：`torch_npu.npu_dynamic_mx_quant / npu_add_rms_norm_dynamic_mx_quant / npu_dynamic_dual_level_mx_quant / npu_quant_matmul / npu_recurrent_gated_delta_rule` 均 `hasattr=True`，`torch.float4_e2m1fn_x2 / float8_e8m0fnu` 存在（**这是本 checkpoint 的 `npu_reference` 所需**）；`npu_causal_conv1d` = **False**（需走自定义/`npu_causal_conv1d_custom`） |
| **S1** | 装 vLLM **精确 commit `ced6857a`**（tag `v0.30.0`）源码 | 内存窗口；`[vLLM] AGENTS.md` 强制用 `uv` + `.venv`（禁系统 pip） | 30–90 min（`MAX_JOBS` 限到 2–4 以适配 32 GB cgroup） | OOM、triton-cpu sleef 子模块（该 commit 的标题就在修它）、gcc 版本 |
| **S2** | 装 vllm-ascend **main** 源码：`python setup.py build_ext`（内部先 `bash csrc/build_aclnn.sh <ROOT> ascend950`，再 cmake `-DSOC_VERSION=ascend950`） | S1 + CANN | 30–90 min（随 aclnn 算子数增长） | 950 上 `VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` 被关（`CMakeLists.txt:178`）、MLAPO 被排除（`:66-71`）→ 少一批算子；**必须用 fork 已验证的 950 构建脚本经验** |
| **S3** | **最小探针（信息量最大、成本最低）**：先什么都不改，直接 `vllm serve /workspace/Qwen3.8-Flash-Next-MXFP4 --language-model-only --enforce-eager --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 1 --quantization ascend`，记录**第一个**异常 | S2 | 分钟级 | 低。预期失败顺序（便于定位）：① 量化入口（`acquire`/resolve：可能因缺 `quant_model_description.json` 而报）→ ② **QSA 层构造：`NotImplementedError: Qwen4Exp QSA requires FlashAttention`**（`[vLLM] nvidia/qsa.py:169-170`，**几乎必然第一个硬错误**）→ ③ PLE 表加载/显存（95.4 GiB）→ ④ indexer `_topk`（CUDA-only，走到 indexer 才会触发）→ ⑤ GDN 权重布局 |
| **S4** | 量化入口：生成 `quant_model_description.json`（按 `[model] quantization_plan.json` 的 240 条 + `W4A4_MXFP4` 约定）**或**写 `AscendQuantConfig` | S3 的 ① | 0.5–2 天（写+核对） | **布局一致性未证实**（§9-②）。核对法：解包 1 个 `experts.gate_up_proj`（packed uint8 + weight_scale），按 `<vLLM>/<asc>` 的读法反量化，与 fp32 原权重比 cosine；golden 可复用 `[ours] tools/golden`、`tools/weights` |
| **S5** | attention：把 `is_flash_attn_varlen_func_available()` 那道 guard 换掉并接上 Ascend QSA 参考实现；替换 `_topk`（用 `torch_npu.npu_moe_gating_top_k` 或我们的 M35） | S3 的 ②④ | 1–3 天 | 上游没有 `VLLM_ASCEND_FORCE_QSA_REFERENCE` 这类开关（**该 env 名在 `[asc-main]` 代码树 0 命中**，只在文档里出现）→ 需自建；注意 `[asc-main] vllm_ascend/envs.py` 只定义了 9 个变量，新开关要按项目规范加进 `env_variables` 字典 |
| **S6** | PLE：先用 pinned-host + `[vLLM] nvidia/ngram_embedding.py:303-351` 的 torch 参考路径，确认能出 token；后续再替换为 kernel | S3 的 ② | 1–2 天 | host 内存/预取；128 分片映射 |
| **S7** | 出一个 token：`--language-model-only --enforce-eager --max-model-len 2048 --max-num-seqs 1`，`curl /v1/completions`（temperature=0） | S3–S6 | 分钟级 | 这是「MVP 达成」判据 |

**必须先完成的**：S1、S2（没有源码栈就没有任何 NPU 路径）；**S3 必须在 S4–S6 之前**，因为它是唯一能告诉我们「真正的第一个坑在哪」的实验，成本却只有分钟级。

---

## 8. 基线怎么建（任务 6）

> **范围提示（用户裁决）**：现阶段**不关心性能**，所以本节是**「将来要测的时候按什么口径测」的备案**，不是现在的任务。现在不要为它抢卡/抢时间；但口径先定下来，避免将来出现不可比的数字。
> 另：**验收口径以 tower 的 `docs/17-verification-standard.md` 为准**；数值一致性判据见 §12.2（QSA 稠密/稀疏差异的强制标注）。

### 8.1 口径（与 mega kernel 收益可比）

**主指标**：单请求 **decode TPS**（= 输出 token 数 ÷ (最后一个 token 到达时间 − 第一个 token 到达时间)，**排除 TTFT**），并发 = 1，取多轮中位数。这正是 `[asc-fork] examples/ascend950_speed_demo.py:129-134` 的口径（该脚本已在 950 上跑过 Qwen3.8-27B，`--preset easy` 标称 ~300 tok/s）。

**辅指标**：TTFT、e2e TPS、每 token 的分层耗时、HBM 峰值。

### 8.2 必须固定的配置（否则数字不可比）

| 维度 | 取值 | 说明 |
|---|---|---|
| TP / DP | TP=1、DP=1 | 单卡 950PR |
| `max-model-len` | 固定（如 4096） | 影响 KV 与 QSA 预算 |
| `max-num-seqs` | 1（单请求口径） | 另做并发扫描 |
| prompt / 输出长度 | 固定同一组 prompt，`max_tokens` 固定 | 建议同时报「长输出（>2k）」与「短输出」 |
| 采样 | `temperature=0` | 去随机性 |
| MTP | **两档**：关 / 开 | 我们目前 MVP 阶段是关 |
| ACLGraph | **两档**：`--enforce-eager` / `FULL_DECODE_ONLY` | 整层 kernel 的图捕获是后续目标 |
| 量化 | 固定同一份 MXFP4 checkpoint | |
| KV dtype | 固定（bf16） | QSA 只接受 bf16/uint8（`[vLLM] nvidia/qsa.py:377-384`） |
| warmup | 固定（丢弃前 N 轮；`[asc-fork] examples/...` 走 cold-L2 首次） | 建议明确区分 cold-L2 与 steady-state |

### 8.3 测法与三档口径

1. **cold-L2 单请求**：直接复用 `[asc-fork] examples/ascend950_speed_demo.py --preset easy|long|gsm8k`（它已经打印 ttft / output_tokens / decode 秒数 / decode tok/s / e2e tok/s）。
2. **steady-state**：同 prompt 连打 N=20 轮，取 decode TPS 中位数（排除第 1 轮）。
3. **吞吐/延迟曲线**：`vllm bench serve`（`[asc-main] docs/source/developer_guide/evaluation/...` 有用法），并发 1/2/4/8 扫 `max-num-seqs`。
4. **分层收益**（与我们 kernel 直接相关）：在层 kernel 内打点或用 profiler 取 op 级耗时，报告 **GDN / QSA / MoE / HC / PLE 各自占比**，并明确区分「我们的 kernel 时间」与「vLLM python/framework + 调度开销」。

### 8.4 复现命令模板

```bash
# 服务端（MVP 档：eager + 关 MTP + 单卡）
export MODEL_PATH=/workspace/Qwen3.8-Flash-Next-MXFP4
vllm serve "$MODEL_PATH" --served-model-name qwen4exp \
  --language-model-only --trust-remote-code \
  --quantization ascend --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --enforce-eager

# 测速（需 pip install openai；脚本来自 [asc-fork] examples/）
python examples/ascend950_speed_demo.py --model qwen4exp --preset long
```

> 参照物：上游 A3 tutorial 用的是 `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'` + `--compilation-config '{"cudagraph_capture_sizes":[4,8,...,32],"cudagraph_mode":"FULL_DECODE_ONLY"}'` + GPQA Diamond 90.4 分（A3, W8A8）。**我们的基线必须标清楚：单卡 950PR + MXFP4 + 无 MTP**，不能与 A3 8 卡 W8A8 直接比。

---

## 9. 存疑与「需实跑验证」（不猜）

| # | 存疑点 | 证据 | 建议的验证实验 |
|---|---|---|---|
| ① | **上游 main 的 Qwen3.8-Flash-Next 支持是「文档先行」**：`[asc-main]` 全树 `grep -i qwen4` 只命中 1 个 md；`vllm_ascend/models/__init__.py` 无该 arch；tutorial 里引用的 `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER` / `VLLM_ASCEND_ENABLE_QSA_E3V` / `VLLM_ASCEND_FORCE_QSA_REFERENCE` 在 `[asc-main]` 代码树 **0 命中**；tutorial 自称「first supported in v0.26.0rc」，而 vLLM v0.26.0 的 `qwen4_exp` 文件数 = **0** | `docs/source/tutorials/models/Qwen3.8-Flash-Next.md:13`；`git ls-tree -r v0.26.0 \| grep -c qwen4_exp` = 0 | 拉取 tutorial 里提到的镜像 `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`，检查其中 `vllm_ascend/models/` 是否含 `qwen4_exp*`；或直接向上游确认代码落在哪个分支/仓库 |
| ② | **checkpoint 的 MXFP4 布局与既有 scheme 的加载约定是否一致** | `[model] README.quant.md` 末句自述「需确认该 fork 对 MoE W4A4_MXFP4 的加载约定与本布局一致」 | 解包 1 个 `experts.gate_up_proj`（packed uint8 + `<name>.weight_scale`）反量化 → 与 fp32 原权重比 cosine / clip_ratio（`validation_report.json` 已给出参考 clip_ratio ≈ 0.018–0.031） |
| ③ | **`npu_causal_conv1d` 在本机 torch_npu 2.10.0.post4 缺失** | 实测 `hasattr(torch_npu, 'npu_causal_conv1d')` = **False** | 走 `[asc] csrc/moe/causal_conv1d` 的自定义/`npu_causal_conv1d_custom`，或在 950 上用 aclnn vendor 包；需实跑确认可用性 |
| ④ | ~~A5 MegaMoe 的 hidden/intermediate 约束与本模型不匹配~~ → **已结案（见 §4③）**：950PR MegaMoe 要求 `hidden ∈ {4096,5120,7168}`、`num_topk ∈ {6,8}`、`intermediate_hidden ∈ {1024,2048,3072,4096,7168}`，本模型 2560/10/640 **三项全不符 → 950 上不能用 MegaMoe**，须走通用 grouped-matmul / 自研 kernel | 本机 CANN 文档 `cann_ops_transformer/docs/zh/mega_moe.md:1072-1082`（权威、随 CANN 9.1.0 发布） | 无需再验；只需确认 `enable_fused_mc2` 不被误设为 2（会自动落到通用路径，但建议显式设 0 并用日志确认） |
| ⑤ | **A5 上能否按需 import 扩展、绕开 `enable_custom_op()`** | fork 的做法（`[asc-fork] device/mxfp_compat.py:79-93`）在 v0.23/0.24-line 上成立；`[asc-main] utils.py:511-519` 的限制仍在 | 在 `[asc-main]` 版本上实跑一次 `import vllm_ascend.vllm_ascend_C` + `hasattr(torch.ops._C_ascend, ...)`，确认 950 上扩展被编译并注册 |
| ⑤b | **`cann_ops_transformer` 的可用性**：本机 CANN 9.1.0 自带该包（含 `ops/mega_moe.cpp`），但其 import 需要 `ninja`，venv 内实测缺失 | `/usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/`；`/workspace/venvs/baseline` 内 `importlib.util.find_spec('ninja')` = False，import 抛 `RuntimeError: Ninja is required…` | 若要走任何 CANN 融合算子路径，先在目标 venv 装 `ninja`；否则保持 `enable_fused_mc2=0` |
| ⑥ | **单卡能否装下**：权重 182.2 GB > 128 GiB HBM；其中 PLE 表 95.4 GiB（1 层）、保留 BF16 117.9 GB | `[model] model.safetensors.index.json` `total_size=182234382328`；`README.quant.md` | 必须把 PLE 走 host（`engram_config.cpu_offload`）+ `--language-model-only` 跳 vision；实跑测 HBM 峰值 |
| ⑦ | **MTP 用哪个 method**：tutorial 用 `qwen3_5_mtp`，而本 checkpoint 自带 `mtp` 段（`hybrid=True`, `layer_types=["full_attention"]`，1 层）且 vLLM 侧还有 `qwen4_exp_mtp`（`[vLLM] config/speculative.py:52`） | `[model] config.json` `text_config.mtp`；`[vLLM] registry.py:697` | MVP 先关 MTP，之后再逐个 method 试 |
| ⑧ | **`[asc-main]` 代码里 `qwen4_exp` 相关 patch 是否存在私有分支** | `git ls-remote --heads` 只见到 main / releases/* / revert-* / rfc/* ；无模型分支 | 若要用上游实现，需向维护者索取；否则按 §5 B 组自建 |
| ⑨ | **`use_compress` 字段的版本错配** | `[asc-fork] platform.py:816/841` vs `[vLLM] vllm/v1/attention/selector.py:21-61`（无该字段） | 只要 **pin 到 `ced6857a`** 就自洽；若用 `[vLLM]` HEAD 需先验证该字段是否存在，否则 `KeyError` |
| ⑩ | `[vLLM] .buildkite/test_areas/lm_eval.yaml:884` 引用 `vllm/v1/spec_decode/qwen4_exp.py`，但该文件**不存在** | A 组调研记录 | 说明模型仍在快速演进；pin commit 是必须的 |

---

## 10. 附录：原始取证输出（精选）

```text
# A. qwen4_exp 首次出现的 vLLM tag
$ for t in v0.26.0 v0.27.1 v0.28.0 v0.29.0 v0.30.0 v0.30.1rc0; do
    echo -n "$t: "; git -C /workspace/vllm ls-tree -r --name-only $t | grep -c '^vllm/models/qwen4_exp/'; done
v0.26.0: 0
v0.27.1: 0
v0.28.0: 0
v0.29.0: 31
v0.30.0: 34
v0.30.1rc0: 34

# B. 上游 vllm-ascend 分支（tag 落后于分支）
$ git ls-remote --heads https://github.com/vllm-project/vllm-ascend | grep -E 'main|releases'
1e9e03bc53bdc53f3cc78c7af191794ee7544ad9  refs/heads/main
df7d511e9282745ccbef73509495637b01e160b4  refs/heads/releases/v0.28.0rc
a71b766ce6fc0a9669a412e15a5cf53f7f827093  refs/heads/releases/v0.29.0rc
$ git ls-remote --tags  https://github.com/vllm-project/vllm-ascend | tail -3
155f68974c802633c19650ece430d41ee6802fa6  refs/tags/v0.26.0rc2
3b31886237ed65c435e0965b10e8002ff5769766  refs/tags/v0.27.1rc1      <-- 最新 tag
f17417f2d5b5b6cbb3c8929c5d826540a811918e  refs/tags/v0.7.1rc1

# C. 各 ref 的 vLLM 版本 pin（关键！）
main              : verified=ced6857afa0ea7b2e3f0846a62e1394e90f15607  release-tag=v0.30.0
releases/v0.29.0rc: verified=84030bbe3d74d99bad477a3d2e37a973ccd8865c  release-tag=v0.29.0
releases/v0.28.0rc: verified=84030bbe3d74d99bad477a3d2e37a973ccd8865c  release-tag=v0.28.0
v0.27.1rc1        : verified=ba07e4a48fc951300d97eb506217dd530583dea3  release-tag=v0.27.1  (qwen4: 0 命中)
v0.26.0rc1        : verified=d02df748bf9efd99022f1a062597dc3cb3808485  release-tag=v0.26.0  (qwen4: 0 命中)

$ git -C /workspace/vllm log -1 --format='%h %cd' <每个 verified>
ced6857a Mon Sep 21 15:32:49 2026   (describe: v0.30.0)   qwen4_exp 文件数 34
84030bbe Fri Sep 11 00:40:10 2026   (describe: v0.28.1rc0-676) qwen4_exp 文件数 34
d02df748 Fri Jul 24 08:23:08 2026   (describe: v0.23.1rc0-1451) qwen4_exp 文件数 0

# D. 本机 fork 的 pin
$ cat /workspace/vllm-ascend/.github/vllm-main-verified.commit   # ee0da84ab9e04ac7610e28580af62c365e898389
$ cat /workspace/vllm-ascend/.github/vllm-release-tag.commit     # v0.23.0
$ git -C /workspace/vllm describe --tags ee0da84ab9e04ac7610e28580af62c365e898389   # v0.24.0
$ git -C /workspace/vllm merge-base --is-ancestor ee0da84a… HEAD ; echo $?            # 1（不是 HEAD 祖先）

# E. 本机 torch 栈与 MXFP4 算子（只读探测，未初始化设备）
$ /workspace/venvs/baseline/bin/python -c "import torch,torch_npu; ..."
python 3.12.13 | torch 2.10.0+cpu | torch_npu 2.10.0.post4 | triton 3.2.0
npu_dynamic_mx_quant True | npu_add_rms_norm_dynamic_mx_quant True | npu_dynamic_dual_level_mx_quant True
npu_quant_matmul True | npu_grouped_matmul True | npu_swiglu True
npu_moe_init_routing_v2 True | npu_moe_token_unpermute True
npu_recurrent_gated_delta_rule True | npu_format_cast True | npu_causal_conv1d **False**
torch.float4_e2m1fn_x2 True | torch.float8_e8m0fnu True

# F. checkpoint 关键元数据
$ python -c "json.load(open('/workspace/Qwen3.8-Flash-Next-MXFP4/config.json'))"
architectures=['Qwen4ExpForConditionalGeneration']  model_type=qwen4_exp
text_config: num_hidden_layers=48  hidden_size=2560  full_attention_interval=4
  hc_count=4  hc_lowrank=320  ple_embed_dim=2560  ple_layer_ids=[2]  ple_conv_kernel_size=4
  ngram_size=3  heads_per_ngram=8  ngram_vocab_size_base=20000000  split_ngram_parts=128
  indexer_n_heads=4 indexer_kv_heads=1 indexer_head_dim=128 indexer_budget=2048 indexer_compress_ratio=4
  num_experts=512  num_experts_per_tok=10  moe_intermediate_size=640  shared_expert_intermediate_size=640
  mtp={hybrid:True, layer_types:['full_attention'], num_hidden_layers:1}
quantization_config.quant_method='ascend'  format='mxfp4-pack-quantized-e8m0'  element=e2m1  group_size=32  scale_dtype=e8m0
  npu_reference='torch_npu.npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode="round")'
  quantized_tensors=240（MoE experts + shared_expert）；quantize_linear_attn=False
index.json: total_size=182234382328（182.2 GB）  1898 tensors
  layers.N.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight  → 128 个分片（仅 layer 2）
  layers.N.{attn,mlp}_hyper_connection.{input_mix_weight_down,input_mix_weight_up,block_inject_weight,hc_norm}.weight ×48
  layers.N.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,conv1d,norm,out_proj,A_log,dt_bias} ×36

# G. 上游 main 对 A5 的硬约束
$ sed -n '74,80p' /tmp/va-up/main/CMakeLists.txt
if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED ${VLLM_ASCEND_CUSTOM_OP})
endif()
$ sed -n '515,518p' /tmp/va-up/main/vllm_ascend/utils.py
# FIXME(linfeng): Currently custom op compilation and execution are partially available
# in ASCEND950 chip, we temporarily disable all custom ops. Please refer to
# https://github.com/vllm-project/vllm-ascend/issues/7157 ...
if envs.VLLM_BATCH_INVARIANT or not get_current_hardware_profile().supports(HardwareCapability.RUNTIME_CUSTOM_OPS):

# H. fork 在 950 上的绕法（可直接抄）
$ sed -n '84,98p' /workspace/vllm-ascend/CMakeLists.txt
# Block FP8 GEMM mix kernel: arch35 (Ascend 950) only. Built separately because
# the generic vllm_ascend_kernels library above is skipped on ascend950.
if(SOC_VERSION MATCHES "ascend950")
    ascendc_library(vllm_ascend_block_fp8_gemm SHARED ...)
    ascendc_library(vllm_ascend_quant_fusion SHARED ...)
endif()
$ sed -n '79,93p' /workspace/vllm-ascend/vllm_ascend/device/mxfp_compat.py
# ... "The fused op lives in the vllm_ascend_C extension, which is lazily imported
# (see utils.enable_custom_op) and therefore may not be loaded yet ... Import it here on demand"

# I. 造模期第一硬失败点（QSA 需要 FlashAttention）
$ sed -n '166,172p' /workspace/vllm/vllm/models/qwen4_exp/nvidia/qsa.py     # 本地 HEAD（8a2364605c）
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
$ git show ced6857afa0ea7b2e3f0846a62e1394e90f15607:vllm/models/qwen4_exp/nvidia/qsa.py | sed -n '98,106p'
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
$ grep -n "def is_flash_attn_varlen_func_available" -A 20 /workspace/vllm/vllm/v1/attention/backends/fa_utils.py
346:def is_flash_attn_varlen_func_available() -> bool:
     # 平台来源只有 CUDA / XPU / ROCm —— 没有 NPU → 恒为 False

# J. CANN 9.1.0 自带 MegaMoe 权威约束（950PR/950DT 段）
$ sed -n '1072,1082p' /usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/docs/zh/mega_moe.md
    - **Ascend 950PR/Ascend 950DT**：
        - `num_tokens`：取值范围 $[1,\ 512]$。
        - `hidden`：仅支持 4096、5120、7168。
        - `num_topk`：仅支持 6、8。
        - `num_experts_per_rank`：取值范围 $[1,\ 16]$，且 `num_experts_per_rank = num_experts / ep_world_size`。
        - `intermediate_hidden`：仅支持 1024、2048、3072、4096、7168。
        - `ep_world_size`：取值范围 $[2,\ 768]$。
        - `dispatch_quant_out_dtype`：仅支持 `torch.float8_e5m2` 或 `torch.float8_e4m3fn`。
        - `dispatch_quant_mode`：仅支持 4（MXFP 量化模式），group size = 32，scale 为 FLOAT8_E8M0。
# 本模型 hidden=2560 / num_topk=10 / moe_intermediate=640 → 三项全不符
$ python -c "import importlib.util;print(importlib.util.find_spec('cann_ops_transformer').origin)"
/usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/__init__.py
$ /workspace/venvs/baseline/bin/python -c "import cann_ops_transformer"
RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)

# K. 上游 main 的「文档先行」旁证
$ grep -rn "QSA" /tmp/va-up/main --include=*.py | wc -l          # 0
$ grep -rn "QSA" /tmp/va-up/main --include=*.md | grep -v tutorial | wc -l   # 0
$ grep -rn "VLLM_ASCEND_FORCE_QSA_REFERENCE" /tmp/va-up/main | grep -v "docs/source/tutorials" | wc -l   # 0
$ grep -c "" /tmp/va-up/main/vllm_ascend/envs.py  # envs.py 只定义 9 个变量，无 QSA 相关
$ ls /tmp/va-up/main/tests/e2e/nightly/single_node/models/configs/ | grep -i flash   # 无
```

---

## 11. 与其它文档的关系 / 后续动作建议

- 本文是 **vllm-ascend 侧**的接入方案；kernel 侧设计见 `docs/05`、`docs/10`、`docs/11`、`docs/12`、`docs/13`。
- 建议立刻可做（不需要重编译、不需要抢卡）：
  1. **核实 §9-①**：拉 `quay.io/ascend/vllm-ascend:qwen3.8-next-a3` 镜像反查上游模型实现到底在哪（这是唯一可能省掉大量自研工作的路径）。
  2. **核实 §9-②**：用 `[ours] tools/` 现有的 safetensors 读取 + golden 工具，解包一个 expert 张量做反量化比对，确认 checkpoint 布局与 `W4A4_MXFP4` 约定是否一致。
  3. 把 §5.3 的 C 组清单与 M34/M35/M36 里程碑对齐（谁提供 QSA indexer、谁提供 hc mixer、GDN prefill 契约与 `chunk_gated_delta_rule_fwd_h` 对齐）—— 三者均已合入 main（`e035b0f`/`a208849`/`ea4c818`）。
- 需要用户/tower 决策的点：**是否 pin `ced6857a` + 上游 main 作为集成基线**（与「保持 /workspace/vllm 为 main HEAD」冲突）；以及是否接受「MVP 阶段先跑通 nvidia 参考路径、慢速，再逐层替换成我们的 kernel」。

---

## 12. 裁决后的补充口径（2026-09-26 第二轮消息触发的修订）

本节是收到 tower 的两条路由消息后补的，**优先级高于前面任何与之冲突的表述**。

### 12.1 indexer 是**接口约束**，不是可选项

- **必须写成约束**：qwen4_exp 的 attention 段官方就是 **QSA（稀疏）**，其中 **indexer（打分 → topk → expand）必须由我们的 per-layer 算子承担，或显式由框架侧保留该能力**；**不允许**在层内核里把它静默省掉后当作「等价实现」。
- 依据：tower 裁决（`20260926-tower-all-qsa.md`）第 ③ 条——「打分/topk/expand 不在首期验收关键路径上，但在『支持 qwen4_exp』之前是必做项，是正确性要求而不是性能选项」；从性能看不划算（省下的算力只占全 pass 下限 ~1.1%），但不做就不是同一个模型。
- 落到本方案的写法：§6.1 的层算子签名里，attention 侧必须能接收/产出 **raw key ring + compressed key cache** 的读写（含 off-by-one：压缩行在 position 3,7,…,4095；pos 4096 属"开放组"只作 causal 尾部 token、不写 compressed cache），并与 M35 的两阶段交付（阶段一缓存填充+prep，阶段二打分+topk+expand）对齐。
- 相关：M33（`docs/15 §8`）已给出建议的 op 边界（op1 = pre-indexer 含缓存写；op2 = 打分+topk+expand；op3 = sparse/dense 二选一核心）。本方案采纳该边界，并明确 **attention 段不得由框架侧的通用 backend 顶替**（因为 `[asc-main] platform.py:814-841` 的映射表里没有 QSA 位置）。

### 12.2 数值口径：**不许写「m=4097 对齐官方输出」**

- 事实：官方 QSA 的 `indexer_budget=2048`、`indexer_compress_ratio=4` ⇒ `block_topk = 512` 块；而 m=4097 的可见块有 ⌈4097/4⌉ = 1025 个 ⇒ **官方只 attend 2048/4096 个历史 token**，稠密 causal 与官方输出**不可能逐位一致**（gather 集合与 softmax 分母都不同）；decode（ctx 4096）同理。
- 因此：所有提到 m=4097 / decode-ctx4096 数值验收的地方，必须标注「**本基准是自建稠密 causal 参考，与官方 vLLM 的 QSA 输出存在固有差异（约一半历史被截断）**」；**任何地方都不许写「m=4097 对齐官方输出」**。
- 相反，**缓存填充层**（raw key ring + compressed key cache）**必须与官方逐位/逐行对拍**——它是可以被严格验证的部分，也是后续接真 QSA 的前提。

### 12.3 与 M33（`docs/15 §8`）的对齐点（我方采纳）

| M33 的结论 | 本方案的处理 |
|---|---|
| Python 侧注册用 `vllm_ascend/ops/register_custom_ops.py:222-298` 的 `direct_register_custom_op(..., mutates_args, dispatch_key="PrivateUse1")`；C++ 侧走 `csrc/` + `torch.ops._C_ascend.*`，**实际走 C++ 侧**（host tiling + `<<<>>>` 直调），Python 只留薄 wrapper | §6.1 第 2 条采纳；但**注意 950 上 `csrc` 扩展默认被 `enable_custom_op()` 关掉**（§3.2），必须走 `[asc-fork] CMakeLists.txt:84-98` 的专用 `ascendc_library` 模式 |
| in-place state（`ssm_state`/`conv_state`/主 KV/raw key ring/compressed key cache/PLE conv state）必须申报 `mutates_args` | §6.1 第 3 条采纳 |
| 只处理「一个 chunk」：契约是「带初态、m ∈ [1, M_MAX] 任意值」，m=4097 只是验收口径 | §6.1 第 4 条采纳 |
| vllm-ascend 侧 GDN state 布局 `[N,H,K,V]`，`ops/gdn.py:602` 有 `transpose(-1,-2)`（与 checkpoint/CANN 的 `[Nv,V,K]` 相反） | 已登记为层算子 ABI 的一部分，接入时必须显式处理 |
| 权重按 checkpoint 原始排布消费；唯一建议偏离是 HC 的 `336 = 324 + 12 pad`（cublas 启发式，非数学要求） | 与用户「kernel 直接支持原始 layout」一致；HC 的 pad 处理已转由 M31/M36 确认 |
| 各段「有无现成槽位」：GDN prefill/decode ✅、MoE ✅、**QSA ❌、HC ❌、PLE ❌** | 与 §5.1/§5.3 一致，无需修改 |

### 12.4 现阶段范围（避免过度设计）

- **不做性能论证/调优**：tile 形状、带宽利用率、时延都不做取舍；但「设计输入类估算」（权重每 token 读多少字节、UB 够不够、放不放得下）仍要做——那是定结构用的。
- **PLE/ngram 表**：放 host 侧、PCIe-through MTE2 直读（可异步），**现在不实现**，先把前面的打通；接口与 prefill 档调用方式写清即可。
- **m 语义**：`m=4097` = prefill（分块）；decode = m=1、context=4096；无 batch>1。基线测量（§8）是**将来**的事。

---

# 塔台后加部分（2026-09-26；**非 M37 原文**，由 M41 落库时追加）

> **完整性声明（先读这条）**：本文件**前 700 行 = M37 原文，逐字节未改**（不重写、不"优化"措辞）。
> - 源：M37 的 worktree 里的 `docs/16-vllm-ascend-qwen4exp-plan.md`
>   （分支 `feat/vllm-ascend-qwen4-exp-integration-plan-s`；worktree 已随交付移除，源文件无仓内路径；该文件在源 worktree 里是
>   **未提交的工作区文件** —— M37 是只读 survey，按规矩零 commit）
> - 源 `wc -l -c` = **700 行 / 83210 字节**；源 `sha256sum` = `59a206ccb3b8f8956953508f059eb57a03de314c93fad1b8761ac575e9fcbe32`
> - 复现「本文件与源逐字节相同」：
>   ```bash
>   head -c 83210 docs/16-vllm-ascend-qwen4exp-plan.md | sha256sum
>   # 59a206ccb3b8f8956953508f059eb57a03de314c93fad1b8761ac575e9fcbe32
>   ```
> - 下面 A–D 四节是**塔台/M41 后加**的内容，与 M37 原文无关；凡与上文冲突处，**以本节为准**（与 M37 §12 同款写法）。

## A. 更正注：M37「找不到可承载 qwen4_exp 的组合」的适用边界

**口径**：M37 速览里「只按**已发布 tag** 找，找不到」**只对 tag 成立**；对**分支**不成立。上游
`main`（`1e9e03bc`，2026-09-26）与 `releases/v0.29.0rc`（`a71b766c`）的
`.github/vllm-main-verified.commit` 都已经 pin 到**含 `qwen4_exp` 的 vLLM commit**
（main → `ced6857a`，`git describe` = **v0.30.0**；releases/v0.29.0rc → `84030bbe`）；两个 ref 的
`vllm/models/qwen4_exp/` 均是 **34 文件**。

**真正的缺口不是版本对齐，而是插件实现**：上游 main 的 `vllm_ascend/models/__init__.py`
`register_model()` 里**没有任何 `Qwen4Exp*` 条目**；tutorial 引用的
`VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER` / `VLLM_ASCEND_ENABLE_QSA_E3V` /
`VLLM_ASCEND_FORCE_QSA_REFERENCE` 在上游**代码树 0 命中**（只出现在那篇 tutorial 自己里）；
CI 与 support matrix 里也没有该模型。⇒ 以后一律写「**无对应已发布 tag（版本层已不再是死路）；
缺的是 Ascend 侧插件实现**」，不要写「vllm-ascend 无对应版本」。

**一处要更正的是塔台的记录，不是 M37 原文**：塔台消息说「M37 写的『vllm-ascend 无对应版本』」，
但 M37 原文里**没有**这个论断 —— 那句话是 **M30** 的，而 M37 **§2.4 已经把它更正了**
（原文：「**对 tag 成立，对分支不成立**……措辞应从『无对应版本』改为『无对应已发布 tag，但 main
分支已对准』」）；M37 §0 速览那行写的是「只按『**已发布 tag**』找 → 找不到」（准确）。
因此：**本更正注是对 M37 §2.4 的重申与前置**（让只读速览的人也不误读），**不是对 M37 的纠错**。

## B. 上游 mega kernel 现状：`rfc/megakernel` 分支 + PR #15986

取证时间 2026-09-26，方式 = `git ls-remote` + 浅克隆该分支到 `/tmp/va-mega`（仓库之外）后读 diff；
命令见 §B.4，全部可复现。

### B.1 分支与 commit（事实）

| 项 | 值 |
|---|---|
| 上游分支 | `refs/heads/rfc/megakernel` = `f75e6fefa720d3364187b2ecc9abf2cc095fafa5` |
| 分支 tip commit | `f75e6fe` **"Mega kernel glm5 2 (#15986)"**，作者 zk123，提交时间 2026-09-08 20:03:41 +0800（`Signed-off-by: keyi-zz`、`Co-authored-by: gcw_w7eh8umq`） |
| 是否在 main | **不在**。main tip = `1e9e03bc`（2026-09-26）：`git merge-base --is-ancestor f75e6fe main` = 假；`vllm_ascend/models/glm_5_2_mega.py` 与字符串 `ENABLE_MEGAKERNEL` 在 main 树**均 0 命中** |
| 分支基线 | **落后于 main**：tip 的父 commit `6ee2227`（2026-09-03）不是当前 main 的祖先；`git diff main..rfc/megakernel --stat` = **1826 文件** ⇒ 该分支 = 「旧 main 快照 + PR #15986」，不是「main + 1 commit」 |
| 交付物 | 3 文件 **+1370 / −1**：`vllm_ascend/models/glm_5_2_mega.py`（**+1331**，`wc -l` = 1331，55012 B）、`vllm_ascend/models/__init__.py`（+18/−1）、`vllm_ascend/worker/worker.py`（+22）。**没有** csrc / CMakeLists / 测试 / 文档改动（PR 正文自述 "No specific tests were added in this PR"；该分支 `docs/` 树 grep `megakernel` = 0 命中） |
| 声明的 vLLM 基线 | PR 正文 "vLLM main: `ba07e4a48fc951300d97eb506217dd530583dea3`"。**注意**：该 commit 正是 vllm-ascend tag `v0.27.1rc1` 的 verified commit（本文件 §2.1 / 附录 C 已记）⇒ 这条 PR 落在 **vLLM v0.27.1 线**，而本方案 §2.2 推荐组合 A pin 的 `ced6857a`（v0.30.0）**比它新** |

### B.2 它做了什么（读 diff 得到的形态）

- **开关**：`ENABLE_MEGAKERNEL`（`vllm_ascend/models/__init__.py:55`，接受 `1/true/True`）。开着且
  模型 arch 是 `glm_moe_dsa`（`GlmMoeDsaForCausalLM`）时，把 `ModelRegistry.register_model` 的目标
  **从** `vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM` **换成**
  `vllm_ascend.models.glm_5_2_mega:AscendGlm52MegaForCausalLM`（同时把 `DeepSeekMTPModel` 换成
  `AscendGlm52MegaMTP`）；关着走原路。TP 里程碑限 **tp_size=1**。
- **粒度 = 整个模型**：`AscendGlm52MegaForCausalLM` 继承官方模型类，`forward` 里的 `_forward_mega()`
  把本步元数据（`seq_lens` / `block_table` / `slot_mapping` / `query_start_loc`）归一化后拷进
  **按 batch-size 预分配的持久 buffer**（地址稳定 ⇒ kernel 侧 DAG graph-cache key 不随 step 变），
  然后**一次** `state.mega.forward(fbi, input_ids, positions, None, None)` 拿回 logits。
  ⇒ **48 层循环在 kernel 里，不在框架侧**。
- **kernel 本体不在这个仓库**：来自外部 python 包 `blockrt`
  （`blockrt.models.glm_5_2.Glm52MegaKernel`、`blockrt.dist.utils.{init,finalize}_shemm`、
  `blockrt.runtime.tracer.DAGCapture`、`blockrt.models.model_cache.ModelCache`）。使用前需
  `MEGA_KERNEL_HOME`/`PYTHONPATH` 可达 `blockrt`，且已跑过 megakernel 自己的 `install.sh`
  完成 CANN 算子注册。**上游仓库里放的只是 adapter/集成层**。
- **复用框架资源**：权重直接吃 vLLM 已加载的张量（`Glm52MegaKernel(..., vllm_ascend_weights=True)`，
  不二次读盘）；KV 直接借用 vLLM 已分配的 paged cache 与 DSA indexer cache（fp8 packed SFA C8 布局）。
- **worker 生命周期**：`vllm_ascend/worker/worker.py` 在 shutdown 路径加 `finalize_shemm()`
  （PR review 要求别只 `except ImportError`）。构造侧仅在 `tp_size > 1` 时做
  `init_shemm(rank, world_size, mem_size=1GiB, ip_port)`，时机在模型 `__init__` 里、设备还空闲时，
  失败即 fail-fast（打 `libshmem.so` 的 `/proc/self/maps` 后 raise）。
- **其它开关**（都在该文件里，非文档）：`MEGA_MAIN_LAYERS`（只加载前 N 层做缩减模型）、
  `MEGA_WEIGHTS_DIR`（权重目录可与 vLLM 服务的路径不同）、`MEGA_MAX_TOKENS`、`MEGA_KV_BLOCKS`、
  `MEGA_SHMEM_SIZE`、`MEGA_SHMEM_IP_PORT`。

### B.3 结论（一句话）

上游在 rfc 分支上确实有一份**已成型的整模 mega kernel 接入**（GLM-5.2 + `glm_moe_dsa`，vLLM
v0.27.1 线，TP=1），但它是「**整个模型 forward 交给外部运行时**」的形态，且 kernel 源码不在公开仓库里
⇒ 与我们的 per-layer 算子路线**不是同一条路，也不是可直接抄的实现**；可借鉴的是集成面与工程细节（§C）。

### B.4 复现命令

```bash
git ls-remote --heads https://github.com/vllm-project/vllm-ascend | grep -E 'rfc/megakernel|refs/heads/main$'
# f75e6fefa720d3364187b2ecc9abf2cc095fafa5  refs/heads/rfc/megakernel
# 1e9e03bc53bdc53f3cc78c7af191794ee7544ad9  refs/heads/main

git clone --depth=50 --branch rfc/megakernel https://github.com/vllm-project/vllm-ascend.git /tmp/va-mega
cd /tmp/va-mega
git show --stat f75e6fe                                  # 3 files, +1370/-1
wc -l vllm_ascend/models/glm_5_2_mega.py                 # 1331
git log -1 --format=%B f75e6fe | grep -A2 "vLLM main"    # ba07e4a4…
git fetch --depth=1 origin main && git merge-base --is-ancestor f75e6fe FETCH_HEAD; echo $?   # 1（不在 main）
```

## C. 与我们的 per-layer 算子方案：关系判断清单

前提：用户已裁决（2026-09-26）**每个 kernel 只跑一层、48 层循环留在框架侧、层 kernel 含 hc、
对 vLLM 改动要小**，验收口径见 `docs/17`。§B 那条路线**不改变这些裁决**，只提供对照。

**可借鉴（与路线无关的工程经验，7 条）**

1. **开关粒度**：整条路径用**一个环境变量**默认关、打开才替换注册（`ENABLE_MEGAKERNEL`）——
   与我们「改动尽可能小 + 可回退」的目标同向；将来若需要灰度，命名上也可参考（别与上游撞车）。
2. **权重零二次读盘**：kernel 直接消费 vLLM 已加载的张量（`vllm_ascend_weights=True`）——
   与用户「kernel 直接吃 checkpoint 原始排布」同向。
3. **cache 借用而非自建**：不复刻 KV 分配，直接把框架已分配的 paged KV / indexer cache 张量交给 kernel。
   我们的层算子同样只读写框架给的 cache（`docs/15` §8.2）。
4. **元数据 ABI 归一化**：`slot_mapping` 一律转 int64、`block_table` 转 int32 并按 `num_blocks` 截断、
   `q_seq_len` 由 `query_start_loc` 差分得到（prefill/decode 通用）。我们的层算子入参设计会遇到同一批
   转换，可直接照这张清单做。
5. **持久地址 ⇒ graph-cache key 稳定**：把每步变化的输入拷进按 batch-size 预分配的固定 buffer，
   使 kernel 侧 DAG graph cache key 不随 step 变。我们首版 `--enforce-eager`，但一旦要上 ACLGraph，
   这就是必须的那套做法。
6. **进程级资源要有 init/finalize 对**：kernel 需要常驻集合通信/共享内存时，init 放在模型构造
   （设备空闲时、失败即 fail-fast），finalize 挂在 worker shutdown，且**不能只 catch ImportError**。
7. **MTP 接缝**：它也把 MTP（`DeepSeekMTP`）一并换成自己的实现 ⇒ 「接管一层/接管整模」的接缝在 MTP 上
   同样会被撞到；我们接 per-layer 算子时，MTP 那一层走哪条路要单独定（本方案 §4⑧ 已列为存疑）。

**不同（明确不走这条路的理由）**

1. **粒度**：它是**整模型一个 op**（一次 `mega.forward` 出 logits，层循环在 kernel 内）；我们是
   **每层一个算子、层循环在 vLLM**（用户裁决）。落在 vllm-ascend 侧的改动面：它 3 文件 1370 行
   **加一个外部运行时**；我们按 §6.1 只替换层模块 forward（外加量化入口与算子注册）。
2. **kernel 归属**：它的 kernel 在 `blockrt`（仓库外，需单独 `install.sh` 注册 CANN 算子）；
   我们的 `.asc` kernel 在自己仓库 + 自己的 CMakeLists，**源码与判据都在自己手里**（可验收、可改）。
3. **目标不同**：它面向 GLM-5.2（`glm_moe_dsa`，DSA 稀疏 attention）的 A2/A3 线、TP=1；
   我们面向 Qwen4Exp on 950PR（QSA indexer + raw key ring / compressed key cache），基线 pin `ced6857a`。
4. **稀疏 attention 的分工形似但不可套用**：它也是「框架给 metadata + cache 句柄、稀疏逻辑在 kernel 里」，
   但那是 DSA / packed-KV fp8 SFA 布局，与我们的 QSA 契约（`docs/17` §7：打分/topk/expand + 缓存填充）
   无对应关系。
5. **上游态度**：落点是 `rfc/*`（不是 main、也不是 release），且分支基线落后 main；PR 无测试、无文档
   ⇒ **不能引用其正确性**，只能引用其集成形态。
6. **对 §6.4 决策记录的意义**：这条分支说明「整模接管 / 外部运行时」在上游是**已被实现且可合入**的
   路线（与 §6.5 的 R2/R4/R5 呼应），代价是要维护「一整份模型实现 + 一个私有运行时」。
   我们据此**不必回头看 R2**，但可把它当作「上 ACLGraph 时的工程参考」。

## D. 待核实项：`quay.io/ascend/vllm-ascend:qwen3.8-next-a3` 镜像（本次部分取证）

§9-① / §11-① 建议「拉这个镜像反查上游的 Qwen3.8-Flash-Next 实现到底在不在」。M41 本次做了能做的
部分（2026-09-26，只读、不拉大 blob）。

**（1）tag 确实存在（比塔台先前「manifest 返回 200」的证据更硬）**

| 检查 | 结果 |
|---|---|
| `GET quay.io/v2/ascend/vllm-ascend/manifests/qwen3.8-next-a3`（`Accept: application/vnd.oci.image.index.v1+json`） | **HTTP 200**，`content-type: application/vnd.oci.image.index.v1+json`，647 B，`Docker-Content-Digest: sha256:2cbb274af050cb5be7db7318e1c461fe22ec0643c54f1ed32aa7266513a7d421` |
| 平台 | `linux/arm64` = `sha256:92ee36ee433551a70af60ee7057b62f849b3d1a40ddb5849b5c213199131cf8c`（3548 B）、`linux/amd64` = `sha256:9bd8c45eed3f73ed5d86d4b1c79793dd68a27b1eddca351df4b087500cae22cc`（3549 B） |
| **阴性对照**（关键） | 伪造 tag `qwen3.8-next-a3-bogus-doesnotexist` → **404**；`zzz-not-a-tag-xyz` → **404** ⇒ 上面的 200 不是「registry 对任意 tag 都返回索引」的假阳性 |
| 旁证（**不可当依据**） | `GET /v2/ascend/vllm-ascend/tags/list`（含 `?n=1000`）**不列**该 tag，列表里也没有任何 `qwen*` —— 该接口有分页上限，**不是**「tag 不存在」的证据 |

**（2）config 层已取证（arm64 config blob，31 KB，经 302 → `cdn01.quay.io`）**

- 基础镜像是 **CANN 9.1.0 + ATB（`ATB_CXX_ABI=1`）**、`python 3.12.13`；
  `ENV SOC_VERSION=ascend910_9391` —— **910 系列（A2/A3 档）SoC，不是 950**。
- 构建 ARG 里 `VLLM_TAG=v0.26.0` —— 是 **ARG 默认值（构建时可被覆盖）**，不能据此断定镜像内的 vLLM 版本。
- history 83 条；其中 `COPY . /vllm-workspace/vllm-ascend/`（arm64 第 12 层，101,970,256 B）
  ⇒ **该镜像的构建上下文就是一份 vllm-ascend 源码树**（层内文件时间戳 2026-09-11 09:53–10:04）。
- 镜像内带 `.agents/skills/{vllm-ascend-model-adapter,vllm-ascend-release}` 这类内部分支资产
  ⇒ 它是内部构建流产物，而不是公开 tag 的同步产物。

**（3）仍缺什么（后续核实项，方法已给）**

- **最关心的一条仍未证实**：那份源码树里**有没有 `vllm_ascend/models/qwen4_exp*`**（若有，可直接省掉
  §5.2 B 组的大部分自研）。原因：该层 blob 从本机拉不动 —— 实测 CDN 侧 **~20 KB/s**
  （`curl -L` 拉到 4.1 MB / 200 s 超时；Range 重试返回 302）。102 MB 的源码层在本机不可得；
  塔台早前记的「CDN 拉取被 reset」是同一现象的另一表现。
- 可行核实方法（换一台有带宽的机器，或走 `docker pull`）：
  ```bash
  docker pull quay.io/ascend/vllm-ascend@sha256:92ee36ee433551a70af60ee7057b62f849b3d1a40ddb5849b5c213199131cf8c
  # 或：curl -L -o layer.tgz <blob URL> && tar -tzf layer.tgz | grep -iE 'qwen4|Flash-Next'
  ```
  判据：看 `vllm-workspace/vllm-ascend/vllm_ascend/models/` 下有无 qwen4_exp 相关文件，以及
  `vllm_ascend/models/__init__.py` 的 `register_model()` 里有无对应 arch。
- 顺带一条**对 §3.3 的旁证**：镜像的 `SOC_VERSION=ascend910_9391` 与 tutorial 自述「仅支持 A3」一致，
  **再次说明 950 支持要我们自己做**。
