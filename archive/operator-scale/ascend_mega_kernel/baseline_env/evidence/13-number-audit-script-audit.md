# audit 脚本硬编码数值字面量审计与处置表（M51）

> 生成者：`agent-sweep`（mission M51），2026-09-26。范围：`baseline_env/scripts/**`（含 `*.sh`）
> 与只读参考 `m18_gdn_prefill/audit_readme_numbers.py`。
> 起因：`baseline_env/scripts/audit_numbers.py:43` 自己硬编码了一个过期活值（`expect="453G"`，
> 而它引用的 `evidence/01:83` 是 **426G**）—— **一个专门抓「数字与归档不一致」的脚本，内部犯了它要防的错**。
> 本轮不只改字面量，而是从机制上消灭这一类（见 §4）。

---

## 1. 分类定义（本表用的三档）

| 类 | 定义 | 允许的写法 |
|---|---|---|
| **(a) 真常量** | 不随「时间 / 设备状态 / 共享机器 / 上游 ref / 本地检出」变的值：算术定数（`2**-24`）、格式常量（dtype 字节宽、safetensors header `<Q`+8B）、单位换算（`1024**3`/`1e9`）、判定阈值与测试参数、**版本 pin 与不可变 commit 对象 ID**、展示条数/列宽、环境路径 | 可以直接写字面量 |
| **(b) 活值** | **会漂**的读数：`df` 实测可用、npu-smi 已用 HBM、解码/带宽类计时、上游 ref 计数与最新 tag、本地 checkout 的版本串 | **禁止字面量**。只能 ① 运行时**从归档现场解析**，或 ② 只写「活值 + 指向采集时刻」，不得当结论 |
| **(c) 归档固定实例** | 断言"归档里就是 X"：对账表里同时钉住 README 行与 `evidence/` 行的期望值；脚本对（或从）归档取回真值再比对 | 可以写字面量（**前提**：两边都对账，且归档在仓内不可变） |

**判定口径的关键区分**（本题最容易搞混的一对）：
- `v0.30.1rc0-189` = **活值**（`/workspace/vllm` 的 `git describe` 快照，`git pull` 或别的 worker 一动就过期）；
- `8a2364605c` / `ced6857af…` = **真常量**（commit **对象 ID** 不可变；它可能不再是某分支的 HEAD，但它指向的对象永远是那一个 —— 所以"研究对象 pin 一个 sha"不是过期值问题）。

---

## 2. 汇总

| 文件 | (a) 真常量 | (b) 活值（改前） | (c) 归档固定实例 | 处置 |
|---|---|---|---|---|
| `baseline_env/scripts/audit_numbers.py` | 排版/正则/退出码/默认路径 | **1 条**：`expect="453G"`（+2 条陈旧豁免） | 58 条 `CLAIMS` 断言 | **已改机制**（§3.1、§4）；改后**无活值字面量** |
| `baseline_env/scripts/check_plugin_vs_vllm_main.py` | 阈值/退出码/默认路径 | **1 条**：banner 里的 `v0.30.1rc0-189` | — | **已改为运行时 `git describe`**；输出与 `evidence/04` 逐字节相同 |
| `baseline_env/scripts/checkpoint_memory_inventory.py` | dtype 字节宽/单位/展示条数/header 格式/`MODEL_DIR` | **0** | 全部输出由 checkpoint header 现算 | 保留 |
| `baseline_env/scripts/selfcheck_npu.py` | 测试规模/阈值/种子/单位 | **0**（计时只在运行输出里，已归档 ⇒ (c) 口径） | 全部输出由设备现读 | 保留 |
| `baseline_env/scripts/collect_env_evidence.sh` | 重试/超时/展示窗口/环境路径/在查 tag 集/`8a2364605c` | **0** | — | 保留 |
| `baseline_env/scripts/collect_upstream_refs.sh` | 车道名/重试/阈值/`ced6857af…` | **1 条**：网络失败分支里硬编码的「最新 tag = v0.27.1rc1」 | — | **已改为打印 `evidence/02` 的归档清单**（§3.5） |
| `baseline_env/scripts/install_torch_stack.sh` | 版本 pin 与超时/退出码 | **0** | — | 保留 |
| `m18_gdn_prefill/audit_readme_numbers.py`（**只读**） | 容差/常数/时钟假设/布局常量 | **2 条**：`eps_used = 2e-5`、`eps_sum = γ₁₂₈·1.05 + …`（废弃的 Neumann 口径，仅在解析不到 A 矩阵时生效） | 76 条断言（**数值比对**式，形态正确） | **未改**（越 scope）→ 见 §5 上报 |

---

## 3. 逐项处置表

### 3.1 `baseline_env/scripts/audit_numbers.py`

| 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `:36-103` `CLAIMS` 58 行 | 每行的 `expect` + `archive_pattern` | (c) | 对账机制本身：`fixed/derived` 要求该值**同时**出现在 README 行与 `evidence/` 行；`live` 走现场解析；`quote/narrative` 按设计豁免 | 保留（58 条逐条即脚本内该表，类别与出处已写在每行的 `note` 里） |
| `:47`（原 `:43`） | `("磁盘实测可用", "live", …, expect="453G")` | **(b)** | 「磁盘实测可用」随其它 worker 波动；写成字面量即过期（归档是 426G） | **已改**：`expect` 清空；值由 `archive_pattern` 捕获组从 `evidence/01` 现场解析；README 写了实例读数则须一致 |
| `:109-111` | `README_LIVE_RX`（新增） | (a) | README 侧实例读数的抽取式样 | 新增 |
| `EXTRA_COVERED`（本次改动前 `:124`） | `"453G"` | **(b) 遗留** | 它豁免的是 README 里**已经不存在**的值（陈旧豁免）；手工豁免会掩盖「值改了但豁免没改」 | **已删**；改由解析值自动进覆盖集。§10.3 需要引用该旧值时才以「更正记录引用」的名义加回（现 `:155`） |
| `EXTRA_COVERED`（本次改动前 `:141`） | `"426G"` | **(b)** | 手工豁免会掩盖「值改了但豁免没改」 | **已删**（改由解析值自动进覆盖集） |
| `:290-308`（`print` 排版段） | `118`、`26/9/8/16/28/12/8` | (a) | 终端排版列宽 | 保留 |
| `:274`、`:277`、`:172-174` | `token_rx`、`LEDGER_ROW`、`EXTRA_COVERED_PATTERNS` 正则 | (a) | 格式/分类式样，非测量值 | 保留 |
| `:315` | `1 if failures else 0` | (a) | 退出码语义 | 保留 |

### 3.2 `baseline_env/scripts/check_plugin_vs_vllm_main.py`

| 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `:76`（改动前） | 打印串里的 `v0.30.1rc0-189` | **(b)** | 那是 `/workspace/vllm` 的 `git describe --tags` 快照；检出一动即过期 | **已改**：新增 `vllm_checkout_version()` 运行时取 `git describe --tags` 并去掉尾部 `-g<sha>`；**输出与 `evidence/04` 逐字节相同** |
| `:87` | `--show` 默认 `25` | (a) | 展示条数 | 保留 |
| `:149` | `return 0 if total_bad == 0 else 2` | (a) | 退出码语义（`2` = 有失效引用，是**结论**不是错误） | 保留 |
| `:85-86` | `/workspace/vllm-ascend`、`/workspace/vllm` | (a) | 环境路径（均可 `--plugin/--vllm` 覆盖） | 保留 |
| 各处 | 计数器 `+= 1`、`sorted(...)[0]` | (a) | 无测量含义 | 保留 |

### 3.3 `baseline_env/scripts/checkpoint_memory_inventory.py`

| 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `:20-23` | `DTYPE_BYTES`（`8/4/2/1/0.5`） | (a) | IEEE/整型字节宽，格式常量 | 保留 |
| `:68` | `struct.unpack("<Q", fh.read(8))` | (a) | safetensors header：8 字节小端长度 | 保留 |
| `:84,:88` | `1024**3`、`1e9` | (a) | GiB / GB 单位换算 | 保留 |
| `:102-103` | `[:10]`、`"10 largest"` | (a) | 展示条数 | 保留 |
| `:18` | `MODEL_DIR` 默认路径 | (a) | 路径常量（env 可覆盖） | 保留 |
| 全部输出 | 台账字节/tensor 数 | (c) | 由 checkpoint header **现算**，不是写死的读数 | 保留（脚本内**无活值**） |

### 3.4 `baseline_env/scripts/selfcheck_npu.py`

| 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `:59-60` | `512, 1024` | (a) | 测试规模（并出现在 `evidence/05` 的打印行里） | 保留 |
| `:51,:54` | `(2, 3)` | (a) | 往返测试形状 | 保留 |
| `:88` | `1e-2` | (a) | 判定阈值（注释写明：fp16 + 1024 深归约） | 保留 |
| `:69,:76` | `1.0`、`0.5` | (a) | 逐元素算子常数 | 保留 |
| `:46,:95-96` | `1024**3`、`1024**2` | (a) | 单位换算 | 保留 |
| `:58` | `manual_seed(0)` | (a) | 确定性种子 | 保留 |
| 输出里的 `round trip in 9.7 ms` 等 | — | (c) | 计时本身是活值，但**只出现在运行输出里**；README 引用的是"`evidence/05` 那一次"的读数（该 claim 的 note 已写明复跑会在 9.6~11.2ms 抖动）。归档若被刷新为新读数，对账会立刻 FAIL ⇒ 强制同步更新 README —— 这正是想要的行为 | 保留 |

### 3.5 `baseline_env/scripts/*.sh`

| 文件 / 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `collect_env_evidence.sh` | 重试 `1 2 3`、`timeout 40`、`sleep 5`、`head -20/-5`、`tail -6`、`sed` 窗口 | (a) | 采集参数与展示窗口 | 保留 |
| 同上 `:46-58` | `/usr/local/Ascend/cann-9.1.0/…`、`/usr/local/python3.12.13` | (a) | 环境路径（被采集的事实，不是读数） | 保留 |
| 同上 `:103` | `for t in v0.23.0 v0.25.1 v0.27.0 v0.28.0 v0.29.0 v0.30.0` | (a) | **在查的 tag 集合**（本 mission 要证的就是这些旧 tag 无 `qwen4_exp`） | 保留 |
| 同上 `:112-125` | `8a2364605c` | (a) | **不可变 commit 对象 ID**（vLLM main 快照的研究对象） | 保留 |
| `collect_upstream_refs.sh:21` | `LANES="releases/v0.29.0rc releases/v0.28.0rc"` | (a) | 研究对象车道 | 保留 |
| 同上 `:27-29,:181` | `timeout 60`、`sleep 8`、`1 2 3`、完整性阈值 `-ge 10`、`head -6/-7`、`cut -c1-60` | (a) | 重试/完整性守卫/展示 | 保留 |
| 同上 `:113` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | (a) | 不可变 commit 对象 ID（main 车道的 `main-verified` pin，同时也是 §6 那条「34 个文件」的研究对象） | 保留 |
| 同上 `:61`（原） | `echo "  newest tag is v0.27.1rc1; …"` | **(b)** | 上游**最新 tag**会随发布变动，写死在脚本里即过期值 | **已改**：删掉字面量，改为打印 `evidence/02-version-matrix.txt` 里已归档的清单（并写明"不凭记忆写"）；该行只在**网络不可用**分支执行 ⇒ 归档 `evidence/11` 不受影响 |
| `install_torch_stack.sh:19-37` | `packaging>=24.2`、`setuptools>=77.0.3,<81.0.0`、`triton-ascend==3.2.2`、`torch==2.10.0`、`torchvision==0.25.0`、`torchaudio==2.10.0`、`torch_npu==2.10.0.post4` | (a) | **版本裁决的结论**（就是本 mission 要装的那一套；不是"读数"） | 保留 |
| 同上 `:30-37` | `--timeout 60 --retries 5`、退出码 `1/10..14` | (a) | 网络策略与失败点编码 | 保留 |

### 3.6 `m18_gdn_prefill/audit_readme_numbers.py`（**只读参考；本 mission 未改**）

| 位置 | 字面量 | 类 | 判定理由 | 处置 |
|---|---|---|---|---|
| `:203` | `c["eps_sum"] = γ₁₂₈·1.05 + c65delta + 3.0e-7` | **(b)** | `1.05` 来自**已被否定**的 Neumann 界口径（M34 round-3 已改判 `κ∞·(γ₁₂₈+γ₆₄)+65δ_exp+3e-7`） | **未改** → 上报见 §5 |
| `:207` | `c["eps_used"] = 2e-5` | **(b)** | README 现为 `ε = 5e-5`；`2e-5` 是修正前的旧值 | **未改** → 上报见 §5 |
| `:101-194` | 76 条断言的 `README 引用值` | (c) | 设计上做的是**数值比对**（README 引用 vs 归档取回，容差 2%）—— 这是正确形态（"取回真值再比"而不是"同现即通过"） | 保留 |
| `:38-40` | `REL_TOL = 0.02`、`U_F32 = 2**-24`、`DELTA_EXP = 2**-23` | (a) | 容差与算术定数 | 保留 |
| `:320` | `1.8e9` | (a) | **唯一显式假设**（1.8GHz 时钟；README §4.7 已标注"含 1.8GHz 假设"） | 保留 |
| `:318,:319,:322` | `65.0`、`2.0e5`、`36` | (a)/(c) | case 配置（65 chunk）、每 chunk 指令数（设计常量）、36 层外推（README §4.7 同口径） | 保留 |
| `:275,:285,:294,:326,:333` | `BT,HE,PH,MC=64,16384,2,128`、`(PH*MC+ch*9+2)*HE` | (a) | 布局/host 侧偏移常量 —— 属真常量，但**其数据来源（`build/*_probe.bin`）不在仓内** | 保留常量；可复现性问题**上报**（§5） |
| `:80,:92,:311` | `1e-12`、`1e-30`、`TRIVIAL` 正则 | (a) | 数值地板与编号规则 | 保留 |
| `:254-259` | `default_rng(0)`、`0.9/0.05/0.1`、`(2,3,5,8)` | (a) | 随机化自检的种子与测试规模 | 保留 |

---

## 4. 本轮实际改动与验证（活值 → 现场解析）

| # | 改动 | 位置 | 验证（命令与结果） |
|---|---|---|---|
| 1 | `live` 类不再带字面量期望值：`expect=""`，值由 `archive_pattern` 捕获组**从 `evidence/01` 现场解析**；README 写了实例读数则须与归档**逐字一致** | `audit_numbers.py:47`（live 行）、`:109-111`（README 侧式样）、`:233-250`（现场解析 + 一致性判定）、`:262`（表格显示）、`:271`（解析值进覆盖集） | `python3 baseline_env/scripts/audit_numbers.py` ⇒ **rc=0**，该行判定 `LIVE-OK（归档现场解析 = 426G；README 读数 426G 与之一致）` |
| 1a | **负例**（证明机制真的能抓这一类错）：把 README 的 `426G` 改成 `453G` 后重跑 | 临时 README（`/tmp`） | ⇒ **rc=1**，`LIVE-FAIL(README 读数 453G ≠ 归档 426G)`，`失败条目: ['磁盘实测可用']` —— 旧的 `live` 实现（只查"标注了活值 + 归档行存在"）**不会**报出这个错 |
| 1b | **正例**（README 只写"活值 + 指向采集时刻"、不写读数） | 临时 README（`/tmp`） | ⇒ rc=0，`LIVE-OK（… README 未写读数）` |
| 2 | 删掉 `EXTRA_COVERED` 里的陈旧豁免 `"453G"` 与手工豁免 `"426G"`，覆盖集改由解析值自动产生 | `audit_numbers.py`（改动前 `:120,:141`） | 覆盖性检查：改前有 1 项（`58 条`）、加上 §10.3 引用的两个 m22 数字后，**现为「（无）」** |
| 3 | banner 的检出快照改为运行时 `git describe --tags` | `check_plugin_vs_vllm_main.py:24-45,93` | `python3 baseline_env/scripts/check_plugin_vs_vllm_main.py \| diff - baseline_env/evidence/04-plugin-vs-vllm-main.txt` ⇒ **diff 空**（归档无需刷新） |
| 4 | 网络失败分支里硬编码的「最新 tag」改为打印归档清单 | `collect_upstream_refs.sh:59-65` | `bash -n` 通过；该分支不参与归档那次采集 ⇒ `evidence/11` 不变 |
| 5 | `m22` 的代价模型：实测时间/带宽改为从 `evidence/run_mode{2,4}.log` 现场解析 | `m22_router512/tools/cost_model.py:27-50,108-124`（M51 同一类错，顺手在**自己的** scope 内清掉） | `python3 tools/cost_model.py` ⇒ 输出与 `m22_router512/evidence/cost_model.txt` **逐字节相同**（已刷新）；两次运行 sha256 相同 |

### 4.1 确定性（tower 要求：两次运行逐字节相同）

```
$ python3 baseline_env/scripts/audit_numbers.py > run1 ; python3 baseline_env/scripts/audit_numbers.py > run2
run1 sha256 = 9032d8391b2b734475081399bb3a49d63ddf4fe1e4e5e446a1026209a3cd4e82
run2 sha256 = 9032d8391b2b734475081399bb3a49d63ddf4fe1e4e5e446a1026209a3cd4e82
cmp: 无差异 ⇒ 两次输出逐字节相同（rc=0/0）
$ sha256sum baseline_env/evidence/12-number-audit.txt
9032d8391b2b734475081399bb3a49d63ddf4fe1e4e5e446a1026209a3cd4e82   (= run1，已刷新)
```

### 4.2 被审对象的 sha256（归档时刻）

```
a1ec1700f91a03f128d3d0b82961d1a05189e4c482d066774feec2ecbebd8ee4  baseline_env/scripts/audit_numbers.py
9c5759c7aa9475b4e777f0369ce5a9fd211886507bfd141642390cef2f211838  baseline_env/scripts/check_plugin_vs_vllm_main.py
89b4699257893852a0ddb84a50507d3a435392a7b1fe44d446c91cfa13909759  m22_router512/check_ref.py
94a4a706ad601e24ec26de54f740d9b11bddc91eb805d5fb2533ef1f8cd05f56  baseline_env/scripts/collect_upstream_refs.sh
878831f8dd72dc1aed181afb9caeb57715b9c13510e5bcf258608a2db08f9f3c  m22_router512/tools/cost_model.py
820b54c916af2da25ebb87a5e1d176e432993fbff44cdf80ea923bb660f5af59  m22_router512/evidence/cost_model.txt
9032d8391b2b734475081399bb3a49d63ddf4fe1e4e5e446a1026209a3cd4e82  baseline_env/evidence/12-number-audit.txt
```

> **M194 漂移标注（2026-10-05）**：上面 §4.2 的标题已写明是「归档时刻」的对象指纹，各行读数一律保留原值、不作更新。
> 经 M194 逐行复算，其中只有 `m22_router512/check_ref.py`（`:147`）的记录值与当前 main 不一致：
> 该记录值等于 M51 合并 `eb915c4`（`e81ab57315`）时的 blob；文件随后在 M56 收口时被改（分支 commit `0c163b5`，
> 合并 `baf355569f`），其当前值自 M56 r2（`96e7e96`）起为：
> `e22f71e9c6d3af7a9b4da54efb1707f99a13ac959736c0b591b3e97659f2b736`
>
> 同代码块其余各行（`baseline_env/scripts/audit_numbers.py`、`check_plugin_vs_vllm_main.py`、
> `collect_upstream_refs.sh`、`m22_router512/tools/cost_model.py`，以及两份 `.txt` 归档产物）
> 经 M194 复算与当前 main 逐位相同。
> 复算：`sha256sum baseline_env/scripts/audit_numbers.py baseline_env/scripts/check_plugin_vs_vllm_main.py m22_router512/check_ref.py baseline_env/scripts/collect_upstream_refs.sh m22_router512/tools/cost_model.py`

### 4.3 复现命令

```bash
# ① 对账（rc=0；两次运行逐字节相同）
python3 baseline_env/scripts/audit_numbers.py   # RESULT: OK (50 compared = 49 归档比对 + 1 live 现场解析 …)

# ② 负例（应 rc=1）：把 README 的采集实例读数改成过期值
sed 's/\*\*426G\*\*（`evidence\/01`/**453G**（`evidence\/01`/' baseline_env/README.md > /tmp/readme_stale.md
python3 baseline_env/scripts/audit_numbers.py --readme /tmp/readme_stale.md

# ③ 枚举脚本里所有含数字的行（本表的枚举方法）
python3 - <<'EOF'
import re, glob
NUM = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
for f in sorted(glob.glob('baseline_env/scripts/*.py')) + ['m18_gdn_prefill/audit_readme_numbers.py']:
    print("=" * 70); print(f)
    for i, line in enumerate(open(f, encoding='utf-8'), 1):
        if NUM.search(line):
            print(f"{i:4d} {line.rstrip()}")
EOF

# ④ 代价模型（输出应与归档逐字节相同）
cd m22_router512 && python3 tools/cost_model.py | diff - evidence/cost_model.txt
```

---

## 5. 上报（越 scope，不在本 mission 内改）

**Finding: `m18_gdn_prefill/audit_readme_numbers.py` 在干净检出上 `exit 1`（6 条 FAIL），归档却声称 0 失败。**
根因两条：① κ∞/‖A‖∞ 的「脚本重算」依赖 `m18_gdn_prefill/build/*_probe.bin`，而 `.gitignore` 忽略
`build/` 与 `*.bin`（`git ls-files m18_gdn_prefill/build` = 0）⇒ 干净检出一律解析为 `None`；
② 解析不到时脚本**静默回落到旧的 Neumann 口径**（`γ₁₂₈·1.05`、`2e-5`），与 README 的 `5e-5` 冲突。
⇒ 归档 `m18_gdn_prefill/evidence/readme_number_audit.md` 的「PART 1 失败条数 0 / κ∞ 计算值=3.21161」
**不可从本 commit 复算**（违反 `docs/17` §2.5/§2.6）。已用 `TowerFinding` 上报（severity=medium），
建议由 m18 的 owner 修：把 A 段证据入库（或改为从归档文本取回 κ∞），并**删掉静默回退**（依赖缺失时判 SKIP + 进 PART 3，而不是套用废弃口径）。

**版本兼容说明**：该脚本在本轮改动前后**都是** 6 条红/`rc=1`（本轮只改了 `baseline_env/scripts/audit_numbers.py`，
对 `m18_gdn_prefill/**` 零改动 —— 我按"只读参考"处理，唯一一次运行把它自己写出的
`evidence/readme_number_audit.md` 覆盖后已立即 `git checkout --` 还原，工作树内该文件无 diff）。

---

## 6. 上一轮的残留观察 —— 本轮（M51 r1）已收口

1. ~~`baseline_env/README.md:141` 与 `:493` 写「上游有 **26 个 head**（`evidence/11` §1 **与 live 一致**）」~~ ——
   **已改（r1）**：两处都改成"**采集时刻 `2026-09-26T12:03:08Z` 的快照、不是实时值**"，
   并注明引用前须复核 `git ls-remote --heads`（`:493` 处写明"不再写作「与 live 一致」"）。
   **现场复核成功**：r1 重跑 `git ls-remote --heads https://github.com/vllm-project/vllm-ascend.git | grep -cE '^[0-9a-f]{40}'`
   = **26**（与 `evidence/11` §1 的采集实例一致），故该读数在改写时**确为真**；但按新措辞，它此后只作快照。
   `audit_numbers.py` 复跑 rc=0，且输出与 `evidence/12` **逐字节相同**（这两处是加说明、不动被对账的数字）。
2. 本文档 §3.6 列出的 `m18` 两类字面量，随 §5 的 finding 一起修最合适（同一处代码）。**仍未修**（越 scope）。


---

## 7. 追加：审校脚本的**三态退出码**（tower 规则 2026-09-26，即刻生效）

规则要点：审校/校验/对账/复现脚本必须 ① 有"实际比较了多少"的计数器并印在 OK 文案里；
② 计数为 0 时**不得**打印 OK，必须 `RESULT: SKIPPED` + 非零退出；③ 退出码三态写进脚本头注释
（`0` = 比过且通过 / `1` = 比过有差异 / `2` = 没得比、输入缺失）；④ 交卷前对脚本做**负向对照**
（喂空输入/缺失输入，确认它不给合格证），两个读数都归档。

本 scope 内三个脚本按此改完并做了负向对照（全部实测）：

| 脚本 | 改动 | 正例（正常输入） | 负例（缺失输入） |
|---|---|---|---|
| `baseline_env/scripts/audit_numbers.py` | 新增 `compared` 计数；README 不存在或"解析不到任何归档值"时 `SKIPPED` + rc 2；头注释写出三态 | `RESULT: OK (50 compared = 49 归档比对 + 1 live 现场解析; 另 4 外部来源、4 豁免)`，**rc=0** | ① `--readme /tmp/nonexistent_readme.md` ⇒ `RESULT: SKIPPED (README not found …)`，**rc=2**；② `--evidence /tmp/emptyev`（空目录）⇒ `RESULT: SKIPPED (0 compared …)`，**rc=2**（此前会打成 50 条 FAIL/rc=1，无法区分"没得比"与"比出错"） |
| `baseline_env/scripts/check_plugin_vs_vllm_main.py` | 有差异由 rc 2 改为 **rc 1**；插件树缺失/扫到 0 个 py 文件 ⇒ `SKIPPED` + rc 2；**RESULT 行走 stderr**（见下） | `RESULT: FAIL (73 broken references out of 1933 resolved+broken; 449 files scanned)`（stderr），**rc=1** | `--plugin /tmp/nope` ⇒ `RESULT: SKIPPED (plugin tree not found …)`（stderr），**rc=2**；此前该情形会打印 0 broken 并 **rc=0（静默合格证）** |
| `m22_router512/check_ref.py` | 输入缺失不再抛 traceback：捕获 `OSError` ⇒ `SKIPPED` + rc 2；判定项为空时**不得** PASS（`all([])` 恒真那条路已堵）；OK 文案带计数 | `RESULT: OK (10/10 判定项通过，guard 3/3)`，**rc=0**（`evidence/mode2/real_m33`） | ① 不存在的 dump 目录 ⇒ `RESULT: SKIPPED (输入缺失 — [Errno 2] No such file or directory: '…/topk_ids.bin'；这不是通过)`，**rc=2** |

**为什么 `check_plugin_vs_vllm_main.py` 的 RESULT 行走 stderr**：它的 stdout 就是归档证据
`evidence/04-plugin-vs-vllm-main.txt`，而 01–11 是 M30 的**原始采集归档**（tower 定为原则上只读，要改先报）。
把判定横幅放 stderr 后：**重跑 stdout 与 `evidence/04` 逐字节相同**（已实测 `diff` 空），归档无需刷新，
而"计数 + 三态退出码"仍然外露、可被 CI/审阅者看到。**若 tower 认为归档可以刷新**，把这两行改回 stdout、
重跑覆盖 `evidence/04` 即可（差异只有这两行）。

复现（全部离线、无需设备）：

```bash
python3 baseline_env/scripts/audit_numbers.py                                    # rc 0
python3 baseline_env/scripts/audit_numbers.py --readme /tmp/nope.md              # rc 2 (SKIPPED)
python3 baseline_env/scripts/audit_numbers.py --evidence /tmp/empty_dir          # rc 2 (SKIPPED)
python3 baseline_env/scripts/check_plugin_vs_vllm_main.py                        # rc 1 (FAIL: 73 条)
python3 baseline_env/scripts/check_plugin_vs_vllm_main.py --plugin /tmp/nope     # rc 2 (SKIPPED)
cd m22_router512 && python3 check_ref.py evidence/mode2/real_m33 --w data/router_weight.bin   # rc 0
cd m22_router512 && python3 check_ref.py /tmp/no_such_dump --w onehot            # rc 2 (SKIPPED)
```

### 7.1 为什么没有重跑 `m22_router512/evidence/check_ref_mode{2,4}.log`

`check_ref.py` 的输出多了一行 `RESULT:` 横幅，按"脚本与归档同 commit 自洽"本应重跑这两份日志。
**但实测重跑会同时改变一个报告项**：M46（提交 `e330796`）修好 golden 的**次正规 FTZ 建模**之后，
m=4097 档的 R0「未建模 FTZ 的 golden 与本参考的 ids 差」由归档里的 **125 槽 / 20 行** 变成 **0 槽 / 0 行**
（判定项 J1–J10 与 guard 逐条不变；mode2 全 10 档除 R0 一列外与归档逐行相同）。
那是一次**语义口径变更**（不只是加一行横幅），已单独报 finding，**本轮不擅自刷新归档**（保留 pre-M46 快照）。
附带发现：归档里 m=4097 那段的**命令回显不完整**（缺 `--w data/router_weight.bin`），照抄该回显复跑会 argparse 报错 rc=2 —— 已写进同一条 finding。

---

## 8. M51 r1（评审 round-1 的两条 P2 + 两笔追加任务）的收口记录

| # | 项 | 处置 | 验证 |
|---|---|---|---|
| 1 | **P2-1** 裸行号清扫不完整（`docs/12:114` 12 个点位无符号且已漂移 +2~+3；`docs/15:500` 与同文件 §3.3 是同一构造却漏改） | 两处都改成符号引用（`ProcessTile`/`CopyInHead`/`Process`/`CopyIn`/`ProcessAiv`/`MXFP4GemmItem::Run`），原行号降级为"审计时口径、不得用行号定位"。**已逐个 grep 确认这些符号在 current main 上存在** | `docs/12` §6 A 项、`docs/15` §9 ★1 |
| 2 | **P2-1 方法学**：`docs/17 §8.2` 的枚举正则要求带扩展名 ⇒ **看不见简写形 `m<编号>:line`**，却写"指向本仓源码 56 处"（覆盖率计数报错了自己的覆盖范围） | §8.2 改为**三形态分列、每个计数紧跟产生它的命令**：形态① `path.ext:line` **518**（本仓 52 / 外部快照 466）；形态② 简写形 **202**（docs/18 149 已有 enclosing-function 列 / docs/13 41 有 API 名+步骤列 / docs/12 6 与 docs/17 6 是**有意的历史口径引述**）；形态③ doc→doc **109**。另把 `docs/05 §6.1 ⓐ` 的 4 处简写形也改成符号（`EgExpAll`/`GdnHeadRecurrence`） | `docs/17 §8.2`；三个计数命令可逐条复跑 |
| 3 | **P2-2** `baseline_env/README.md:141/:493-494` 的「与 live 一致」（活值的实时断言） | 改写成"采集时刻快照 + 指向采集实例 + 引用前须复核"，并写明"不再写作「与 live 一致」" | 见 §6.1；`audit_numbers.py` rc=0、`evidence/12` 逐字节不变 |
| 4 | 追加：`docs/17 §6` 的 m13 `quant-inf` 计数过期（32/32） | 订正为 **40 判定项 + 2 报告项**（M50：10 用例 × 4；用例集合完整性 guard 与 `tools/golden` 交叉见证按 §2.1 单列为报告项，不计入判定） | 与 `m13_moe_layer/README` §5.4/§8.1 的 M50 口径一致 |
| 5 | 追加：`tools/golden/moe_block_ref.py` 的 `quantize_ocp` docstring 写过期断言「device kernels miss the ``:413`` override (M32 is fixing it)」 | 改写成**可核对的现状**：截至 2026-09-26，该 override 已在 **m2/m5/m13 三处** 的 `MxQuantComputeScale` 内（`nanRegTensor = Duplicate(NAN_CUSTOMIZATION = 0x7F81)` + `Select<uint16_t>(…)`，grep `0x7F81` 可见），并注明归档日志/`selfcheck.py` 的注记是"当时那一刻的记录" | 见下回归 |

### 8.1 docstring 改动后的回归（只改 docstring，判据数/PASS 数必须不变）

| 命令 | M46 基线 | r1 实测 | 结论 |
|---|---|---|---|
| `selfcheck.py --data data --groups m1,m33` | 644 判定 + 6 guard | **644 + 6**（rc=0） | 不变 ✅ |
| `selfcheck.py --data data --groups m1,m33 --no-regen --m13-dump /tmp/m13_dump_m26` | 386 判定 + 6 guard | **386 + 6**（rc=0） | 不变 ✅ |
| `selfcheck.py --data data --groups m1,m33 --real` | 938 判定 + 8 guard | **938 + 8**（rc=0；该命令在共享卡上约 5 分钟） | 不变 ✅ |
| m13 数据集链：`M13_DUMP=1 m13_moe_layer <data> m1 m33` | rc=0 / **682 判据行**（678 `PASS (` + 4 routing 一致性）/ 0 FAIL / `ALL PASS` | **682 判据行**（678 + 4）、0 FAIL、`===== ALL PASS =====`、rc=0 | 不变 ✅ |
| 同上：144 个 dump/meta 与 M26/M46 批次逐字节一致 | 一致 | **144/144 逐字节一致**（sha256 集合 diff 为空） | 不变 ✅ |
| m13 数据集链：`check_ref.py . m1 m33` | `ALL PASS （判定 118 项 / 参考 32 项）` | **`ALL PASS （判定 118 项 / 参考 32 项）`**、rc=0 | 不变 ✅ |

### 8.2 同目录（`tools/golden/**`）**未授权、故未改**的同类过期断言（已报 tower）

`tools/golden/README.md:215`、`tools/golden/README.md:475`、`tools/golden/qaware_ref.py:26`、
`tools/golden/selfcheck.py:239` 仍写「M32 正在修 / M32 is fixing」。
授权范围只含 `moe_block_ref.py` 一个文件，故这四处**先报不改**（`selfcheck.py:239` 是**运行时会打印**的注记，
会让新生成的日志继续带过期措辞）。`tools/golden/evidence/*.log` 里的同句是**归档证据**，按规则保留。

> ⚠ **时点（M73 追加；不改上段的记录）**：上面「**仍写**」是 **M51 r1 时点**的观测（所给行号也是当时的）。**其后由 M56 按 `tools/evidence/stale_runtime_print_scan.log` 逐处清理**（该文件 §1 逐条列 4 处的「原句 → 现句」；`docs/17` §8.4 末另有一条同源的 M70 时点注）。按 `main` 现文复算（**M73 实测 @ `569113d`**）：`grep -c 'M32 正在修\|M32 is fixing' tools/golden/qaware_ref.py tools/golden/README.md tools/golden/selfcheck.py` → **0 / 1 / 1**，且那 2 处都是**历史引述**（`tools/golden/README.md:225`「此前这里曾写…」、`tools/golden/selfcheck.py:239` 注释「旧句…在 M32 落地后即成过期断言」）⇒ **不要把「仍写…」读成当前事实**。`tools/**` 不在本 mission 的改动边界内，故这里只加时点与去向、**不动那两份文件**。
