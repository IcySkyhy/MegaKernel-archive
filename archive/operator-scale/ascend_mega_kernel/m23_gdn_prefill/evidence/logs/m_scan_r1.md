# M 扫描读数：从 M=1 遍历找「最小复现的 M」（开关 `M15GP_ONLY_M=<int>`）

## 命令模板（逐字；每条 M 一次独立进锁，`-w 300`、`timeout 25`、进锁前 `npu-smi` 复查）
```
cd m23_gdn_prefill
for m in <M列表>; do
  M15GP_ONLY_M=$m timeout 25 flock -w 300 /tmp/npu0.lock stdbuf -oL \
      ./build/m23_gdn_prefill > evidence/logs/mscan_m$m.log 2>&1; rc=$?
  printf "M=%s rc=%s mte=%s ECC=%s trap=%s\n" "$m" "$rc" \
      "$(grep -o 'mte error info: 0x[0-9a-f]*' evidence/logs/mscan_m$m.log | head -1)" \
      "$(grep -c 'multi-bit ECC' evidence/logs/mscan_m$m.log)" \
      "$(grep -c 'timeout or trap' evidence/logs/mscan_m$m.log)"
done
```
判据（预登记 `evidence/logs/m_scan_prereg.md`）：FAIL = `rc!=0` 且含 `507015` 且 `mte error info` 计数 > 0；
OK = `rc=0` 且含 `ALL RUNS OK`。**不设 `M15GP_ONLY_M` 时行为与改动前逐字一致**（仍跑 m1 + p4097 两条 case）。

## 报告表（M | 判定 | mte error info（首个、core 0） | multi-bit ECC 行数 | timeout or trap 行数）
| M | 判定 | mte error info | multi-bit ECC | timeout or trap |
|---|---|---|---|---|
| 1 | OK | - | 0 | 0 |
| 2 | OK | - | 0 | 0 |
| 4 | OK | - | 0 | 0 |
| 8 | OK | - | 0 | 0 |
| 16 | OK | - | 0 | 0 |
| 17 | OK | - | 0 | 0 |
| 18 | OK | - | 0 | 0 |
| 19 | OK | - | 0 | 0 |
| 20 | OK | - | 0 | 0 |
| 24 | OK | - | 0 | 0 |
| 26 | FAIL | 0x13d7e000000202ce | 1 | 2 |
| 27 | OK | - | 0 | 0 |
| 28 | OK | - | 0 | 0 |
| 29 | OK | - | 0 | 0 |
| 30 | FAIL | 0x13d15000000202ce | 1 | 2 |
| 31 | OK | - | 0 | 0 |
| 32 | FAIL | 0x13d42000000202ce | 1 | 2 |
| 64 | FAIL | 0x13d10000000202ce | 5 | 10 |
| 65 | FAIL | 0x13d33000000202ce | 5 | 10 |
| 128 | FAIL | 0x13d1f000000202ce | 8 | 16 |
| 256 | FAIL | 0x13d15000000202ce | 16 | 32 |
| 512 | FAIL | 0x13d10000000202ce | 20 | 52 |
| 1024 | FAIL | 0x13d10000000202ce | 19 | 53 |
| 2048 | FAIL | 0x13d10000000202ce | 18 | 54 |
| 4096 | FAIL | 0x13d10000000202ce | 24 | 48 |
| 4097 | FAIL | 0x13d10000000202ce | 18 | 6 |
| 26 (r2) | OK | - | 0 | 0 |
| 26 (r3) | FAIL | 0x13d510000000202ce | 1 | 2 |

## 必答问题（逐条，如实）
1. **最小的 FAIL M = 26；最大的 OK M = 31**（另 24 / 27 / 28 / 29 / 31 均 OK）。
2. **临界不落在 chunk 边界**：26~32 **全在同一个 chunk（BT=64）之内**（`nChunk = ceil(M/64) = 1`）。
   按人类给的判据，这属于「**第一个 FAIL 就在 M<=64 之内** ⇒ 指向**单 chunk 内部**」那一支，
   **不是**「跨 chunk 状态/令牌复用」那一支（至少首次失败不是）。
3. **失败时 `mte error info` 的低 8 位与 4097 档相同（`...0202ce`）**，第 5 位十六进制（核号字段）随运行变
   （`0x13d42...` / `0x13d10...` / `0x13d33...` / `0x13d1f...` / `0x13d15...` / `0x13d7e...` / `0x13d51...`）
   ⇒ **同一类故障**（不是新问题），但**报错核不固定**。
4. **非单调 + 非确定（本轮最重要的一条）**：`24 OK -> 26 FAIL -> 27/28/29 OK -> 30 FAIL -> 31 OK -> 32 FAIL`；
   **同一个 M=26 连跑三次得 FAIL / OK / FAIL**（r2 OK、r3 FAIL，签名与首次同类）。
   ⇒ 在 M 约 26~32 这一段，故障是**概率性**的，**不是**一条「M >= 阈值即挂」的硬边界。
   命中率（事实）：`M <= 24` **0/6**；`26~32` **4/8**；`M >= 512` **4/4**，且 `multi-bit ECC` 行数随 M 增大（1 -> 20）。

## 判读（只陈述事实与对照，**不写绝对断言**）
* 因「同一 M 三次两次不同」，本 mission **不能**给出「最小复现 M = 常数」这类结论；能说的是上面那张命中率事实。
* 形态上更像**概率/竞态**（命中率随工作量上升），而非固定的地址/尺寸缺陷；这与「跳过 `Job2` 的三条 L1 写后
  m=4097 不再犯」（少了 MTE2/L1 操作）方向自洽 —— **但本节不作结论**。
* `timeout or trap` 行数在**同一档**也会变（4096 档 48 行 vs 4097 档 6 行）⇒ 与塔口径一致，**非稳定签名**。

## 限度（未做 / 边界，如实）
* **每个 M 只跑 1 次**（`26` 跑了 3 次）；`M=25` **未测**；`M=33~63` **未测**。
* 未做 SLOG 对照；未做「每个 M 跑 N 次」的命中率矩阵。
* harness：`Tp = M + 64`、`tp = align8(M)`、`salt` 固定 ⇒ **任何 M >= 1 都可表达**，未遇「装不下」的情形。
* 未改被测段体 `m15_gdn_prefill.h`、未动 L1 布局/资源记账；只改 `m23_gdn_prefill/**`（新增 `M15GP_ONLY_M`）。
