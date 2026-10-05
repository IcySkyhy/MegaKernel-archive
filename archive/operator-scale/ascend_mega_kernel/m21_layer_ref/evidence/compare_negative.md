# Graded comparison — l0 profile

- reference : `reference/layer0_decode_m1` (manifest sha256[:16] = `15d57a497fc9b315` — this covers the run PARAMETERS only; the stable tensor hashes are in `evidence/reference_sha256.txt`)
- test      : `reference/layer0_decode_m1_pending` (manifest sha256[:16] = `8bb9130f2533a991`)
- command   : `compare_dumps.py --ref reference/layer0_decode_m1 --test reference/layer0_decode_m1_pending`

## 1 判定项 (judgement: PASS/FAIL)

| segment | dtype | max abs err | ref max abs | budget abs | usage | max ulp (≥2^-8·max) | 非有限掩码不匹配 | verdict |
|---|---|---|---|---|---|---|---|---|
| `attn_hc.block_input` | bfloat16 | 4.92969 | 4.53125 | 0.0177002 | 27851.03% | 32699 | 0 | **FAIL** |
| `gdn.conv_out` | bfloat16 | 2.87012 | 4.78125 | 0.0186768 | 15367.32% | 32040 | 0 | **FAIL** |
| `moe.topk_ids` | int32 | 295 | 370 | 0 | inf% | 0 | 0 | **FAIL** |

**判定项总数 3；PASS 0；FAIL 3**

判据：`max_abs_err <= atol + rtol*ref_max_abs` 且 `非有限掩码不匹配 == 0`；并按 profile 另加 `max_ulp <= 0`。ulp 只在 ≥2^-8·ref_max_abs 的元素上计算。
误差数字口径：全部为 **max**（不是首次不匹配），绝对/相对已分列，输入档位见 manifest 的 `extra` 字段。
非有限值（如 QSA 用 `-inf` 表示「该列未参与打分」）不参与减法比较，只比较「是否有限」的掩码是否一致。**默认 `--mask-policy strict`**：参考侧非有限而测试侧有限也算不匹配。
若测试侧在这些列 dump 0（官方 kernel 只写 `column < visible_blocks`，其余列是未初始化 buffer，`ops/qsa_indexer.py:101-107`），本报告会判 FAIL。用 `--mask-policy ignore-unscored`（作用于 `--unscored-segments`，默认 `qsa.index_logits`）可把这些位置从**所有**判据里剔除，剔除个数在报告项 `ignored_unscored` 单列。
行内排序后再比较的段（上游 top-k 顺序未定义）：`qsa.block_indices,qsa.token_indices`。

⚠️ **档位提示**：被你比对的段里有 2 个按 `docs/17` §1.1 属 **T3**（超越函数 / 跨 tile·online 重标定 / mmad 累加），而 `l0` 的判据是 `max_ulp=1`。这些段若 FAIL，先确认不是档位问题：`attn_hc.block_input, gdn.conv_out`。用 `--profile t3` 走逐元素 `rtol·|ref| + 0.5·ulp(out)` 口径。（本工具**不会**自动换档 —— 换档必须显式，见 README §6 的档位声明表。）

## 2 报告项 (report only — NOT counted in PASS/FAIL)

| segment | bit-exact | bit-exact rate | ≤1 ulp rate | max rel err (max, relative) | 显著元素/总元素 | ignored_unscored |
|---|---|---|---|---|---|---|
| `attn_hc.block_input` | 1/2560 | 0.04% | 0.47% | 12733.5 | 2559/2560 | 0 |
| `gdn.conv_out` | 163/10240 | 1.59% | 3.54% | 2337.16 | 5050/10240 | 0 |
| `moe.topk_ids` | 0/10 | 0.00% | 0.00% | 42.1429 | 10/10 | 0 |

行内一致性口径：argmax 越界行（全 `-inf`）已跳过；top-k 的 k = min(10, 该行有限元素数)。

