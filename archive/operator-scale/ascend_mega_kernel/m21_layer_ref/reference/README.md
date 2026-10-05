# reference/ — 参考激活归档

每个 `<tag>/` 是一次参考跑的产物：

```
<tag>/manifest.json    每段的 dtype / shape / sha256(存储字节) + 本次运行的参数（extra）
<tag>/<segment>.npy    bf16 存为 uint16 位型；fp32/int32 原样
```

## 归档契约

* **`.npy` 不入库**（`.gitignore`），`manifest.json` 与 `evidence/reference_sha256.txt` 入库。
  理由：本仓既有约定（`m13_moe_layer/evidence/`、`tools/*/data/`）是"证据 = 日志 + manifest + sha256"，
  二进制 dump 由确定性重建得到。
* **重建是确定性的，已实测**：两次完全独立的 build（不同进程）产出的 **183 个 tensor 的 sha256 逐位相同**。
  `tools/make_evidence.sh` 步骤 3 会把本次清单与基线 `evidence/reference_sha256.first_run.txt` diff 一遍：
  **diff 非空即报错退出，并把差异留在 `evidence/determinism.diff`**；通过时该文件被删除（所以它不在
  commit 里，也不会被引用）。
* **线程数必须固定**：`tools/make_reference.sh` 固定 `M39_THREADS=8`（写进脚本）。
  实测 8 线程 vs 默认线程时，**唯一**会变的是 `moe.routed_out`（CPU GEMM 的分块随线程数变），
  其余 182 个 tensor 逐位不变。⇒ 对 `moe.routed_out` 的位级比较要按 l0 容差（1 bf16 ulp @ max 幅值）判，
  **不要**要求 100% 位级一致（报告项里它的 bit-exact rate 会 < 100%，这是预期的）。
* 重建一条命令；耗时见 `evidence/rebuild_timing.txt`（**host 墙钟、共享机器，只当区间读**，别当单值）：

  ```bash
  bash tools/make_reference.sh       # 内部已固定 M39_THREADS=8
  ```

  跨机器/跨 torch 版本不保证逐位一致；若对不上，**以入库的 sha256 为准**，优先排查 torch 版本与线程数。

## tag 清单（6 个）

| tag | layer | 类型 | m | 输入 | pending | warmup | dump 位置 | 段数 |
|---|---|---|---|---|---|---|---|---|
| `layer0_decode_m1` | 0 | GDN | 1 | `embed`（= 官方真值） | none（= 官方真值） | 8 | pos 8 | 30 |
| `layer0_decode_m1_pending` | 0 | GDN | 1 | `embed` | synth | 8 | pos 8 | 32 |
| `layer0_chunk_m64` | 0 | GDN | 64 | `embed` | synth | 0 | pos 0..63 | 32 |
| `layer3_decode_m1` | 3 | QSA | 1 | synthetic | synth | 2100 | pos 2100 | 31 |
| `layer3_chunk_m64` | 3 | QSA | 64 | synthetic | none | 0 | pos 0..63 | 29 |
| `layer3_chunk_m64_seed1` | 3 | QSA | 64 | synthetic, seed=1 | none | 0 | pos 0..63 | 29 |

`layer3_chunk_m64` 与 `layer3_chunk_m64_seed1` 是**非空洞性孪生对**（同参数换输入种子），
用 `compare_dumps.py --nonhollow-only` 消费。

`warmup` 的语义见 `../README.md` §4：先跑 N 个 token 不进 dump（**一次 T=N 的调用**），
把 GDN 的 `conv_state`/`ssm_state` 和 QSA 的压缩 cache 推到非退化状态
（否则首 token 的 `ssm_state ≡ 0`、`visible_blocks ≡ 0`）。
`layer3_decode_m1` 取 warmup 2100 ⇒ dump 位置 2100、`visible_blocks = 525 > block_topk = 512`，
所以归档里**同时覆盖** QSA 的 block top-k 截断路径。

## 为什么没有整网/多层 dump

整模型 169.72 GiB（PLE 表 95.37 GiB），容器 cgroup 上限 32 GiB；PLE 段按用户裁决不在 scope。
本目录只提供**单层**参考。详见 `../README.md` §1 与 §11。
