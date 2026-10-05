# 最小 AIC 探针：L0A/L0B/L0C 的 16/17 tile 边界（人类：「直接测试硬件呀」）

预登记：`evidence/logs/probe_l0_prereg.md`（两轴 × 两臂 + 判据，**先写后跑**）。
探针路径：`m23_gdn_prefill/probe_l0/probe_l0.asc`；目标 `probe_l0`（`m23_gdn_prefill/CMakeLists.txt` 末段）；
**1 个 AIC**（`__global__ __cube__`、blockDim=1）、fp32、N=K=128、tile=16 行、iters=200。

## 命令（逐字；每档一次独立进锁、`-w 300`、`timeout 25`、进锁前 `npu-smi` 复查）
```
cd m23_gdn_prefill/build
for cfg in "1 200 0" "2 200 0" "2 200 1" "1 200 1"; do set -- $cfg
  timeout 25 flock -w 300 /tmp/npu0.lock stdbuf -oL ./probe_l0 $1 $2 $3 \
      > ../evidence/logs/probe_l0_t$1_a$3.log 2>&1; echo rc=$?
done
```
（arm=0 = 段体现用形态：每 tile 一次 `BUF_L1→BUF_L0→BUF_L0C` BufferID 轮转、tile 之间无额外屏障；
arm=1 = 保守形态：tile 之间加 `PipeBarrier<PIPE_M>` + `PipeBarrier<PIPE_MTE1>`。）

## 读数（逐档）
| tiles | arm | rc | 结果行 | `multi-bit ECC` | `mte error info` |
|---|---|---|---|---|---|
| 1 | 0 | **0** | `[PL0][OK] C[0][0]=128.0` | 0 | **（无）** |
| 2 | 0 | **0** | `[PL0][OK] C[0][0]=128.0` | 0 | **（无）** |
| 2 | 1 | **0** | `[PL0][OK] C[0][0]=128.0` | 0 | **（无）** |
| 1 | 1 | **0** | `[PL0][OK] C[0][0]=128.0` | 0 | **（无）** |

## 判读（对照预登记的三支）
**三支中的第三支**：「**两臂都不挂 ⇒ 探针没复现到触发条件，如实说，别硬凑**」
⇒ **本探针（1 AIC / fp32 / N=K=128 / 200 次 / 1 或 2 个 16 行 tile）未复现 `…0202ce`**，
因此**既不能说**"这是软件同步缺陷（A 挂 B 不挂）"，**也不能说**"这是硬件约束（两臂都挂）"。
指纹比对因此**没有对象**（四档的 `mte error info` 均为空）。

**一条如实记录的观察（不当作结论）**：`tiles=2` 两档读回的 `C[0][0]` 也是 `128.0`，
而我在 host 侧打印的"期望"写的是 `iters×tiles×K`（=51200），**两者不一致** —— 这说明我这条期望式子
与探针实际写回的语义（每次 `Fixpipe` 覆盖 L0C 行 0..15、`mSize=MT=16`）**不匹配**；
**本探针的用途是"看是否报错"，不是数值正确性**，该不一致**未追查**，如实列出。

## 未做 / 限度（如实）
* **未上 mix(1,2) 满核（28 AIC）** —— 预登记的策略是"先 1 个 AIC，复现不了再上"；本轮停在 1 AIC。
* 未试更大的 `iters`、未试 fp32 之外的 dtype、未试 K/N 的其它量级、未试 L1 双缓冲/多槽位形态。
* 未试"两个 tile 累加到**不同** L0C 区"或"L0C 分区写"的形态。
* 未改被测段体 `m15_gdn_prefill.h`；未动 L1 布局/资源记账。
* 搁置说明：M=16/17 的段体统计批（人类改向前启动）**未跑完即停**（完成 M16 30/30、M17 26/30），
  已按"先搁置"的指示**不再继续**；本轮**未**据它下任何结论。
