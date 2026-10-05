### --mask-policy ignore-unscored
# Graded comparison — l0 profile

- reference : `reference/layer3_chunk_m64` (manifest sha256[:16] = `7fac03df735b65c9` — this covers the run PARAMETERS only; the stable tensor hashes are in `evidence/reference_sha256.txt`)
- test      : `reference/.tmp_zeroed` (manifest sha256[:16] = `15278457ffefa442`)
- command   : `compare_dumps.py --ref reference/layer3_chunk_m64 --test reference/.tmp_zeroed --mask-policy ignore-unscored`

## 1 判定项 (judgement: PASS/FAIL)

| segment | dtype | max abs err | ref max abs | budget abs | usage | max ulp (≥2^-8·max) | 非有限掩码不匹配 | verdict |
|---|---|---|---|---|---|---|---|---|
| `qsa.index_logits` | float32 | 0 | 111.121 | 0.000106073 | 0.00% | 0 | 0 | **PASS** |

**判定项总数 1；PASS 1；FAIL 0**

判据：`max_abs_err <= atol + rtol*ref_max_abs` 且 `非有限掩码不匹配 == 0`；并按 profile 另加 `max_ulp <= 8`。ulp 只在 ≥2^-8·ref_max_abs 的元素上计算。
误差数字口径：全部为 **max**（不是首次不匹配），绝对/相对已分列，输入档位见 manifest 的 `extra` 字段。
`--mask-policy ignore-unscored`：对 `qsa.index_logits` 的**参考侧非有限位置**（= 官方 kernel 未写过的列）已从所有判据中剔除，剔除个数见报告项 `ignored_unscored`。其余段的掩码仍按 strict 比较。
行内排序后再比较的段（上游 top-k 顺序未定义）：`qsa.block_indices,qsa.token_indices`。

⚠️ **档位提示**：被你比对的段里有 1 个按 `docs/17` §1.1 属 **T3**（超越函数 / 跨 tile·online 重标定 / mmad 累加），而 `l0` 的判据是 `max_ulp=1`。这些段若 FAIL，先确认不是档位问题：`qsa.index_logits`。用 `--profile t3` 走逐元素 `rtol·|ref| + 0.5·ulp(out)` 口径。（本工具**不会**自动换档 —— 换档必须显式，见 README §6 的档位声明表。）

## 2 报告项 (report only — NOT counted in PASS/FAIL)

| segment | bit-exact | bit-exact rate | ≤1 ulp rate | max rel err (max, relative) | 显著元素/总元素 | ignored_unscored |
|---|---|---|---|---|---|---|
| `qsa.index_logits` | 496/1024 | 48.44% | 48.44% | 0 | 491/1024 | 528 |

### 2b argmax / top-k 一致率（`docs/17` §1 L2 指标②，**报告项**）

| segment | argmax 一致行 | top-10 集合完全一致行 | top-10 元素重合率 |
|---|---|---|---|
| `qsa.index_logits` | 61/64 (95.31%) | 59/64 (92.19%) | 66.88% |

行内一致性口径：argmax 越界行（全 `-inf`）已跳过；top-k 的 k = min(10, 该行有限元素数)。

