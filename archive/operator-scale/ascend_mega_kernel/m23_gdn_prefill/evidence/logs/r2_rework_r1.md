# r2 复审的两条返工项（判据侧）：修 + 设备读数

## (a) `--mutate` 反向对照「咬不住」⇒ 已修，**设备档已红**
**根因**（复审正确）：`MutateInputs` 原本在 **H2D 之前**就被调用（`m23_gdn_prefill.asc:173`），
于是**设备与参考吃的是同一份被搅动的输入** ⇒ 两边一起错 ⇒ 判据 PASS（负向对照没建立）。
**修**：新增设备侧副本 `vDev/gDev/h0Dev`，**只搅动副本**（H2D 用副本，dump 用干净数据）：
```diff
-    MutateInputs(mutate, v, g, h0);
+    // 反向对照：只搅动送设备的副本；host 侧留干净数据用于 dump（参考吃干净、设备吃搅动）
+    std::vector<float> vDev(v), gDev(g), h0Dev(h0);
+    MutateInputs(mutate, vDev, gDev, h0Dev);
...
-    aclrtMemcpy(dv[2], …, v.data(), …)      →  vDev.data()
-    aclrtMemcpy(dv[3], …, g.data(), …)      →  gDev.data()
-    aclrtMemcpy(dv[5], …, h0.data(), …)     →  h0Dev.data()
```
**设备读数（新鲜，`--mutate 1`，一次独立进锁、rc=0）**：`evidence/logs/r3_mut1.log`；
判据（`evidence/logs/r3_check_ref_mut1.log`，**逐字**）：
```
[REF] m1_mut1     | o 超界 2948/6144        max|Δ|=2.120e-02 | ht 超界 708806/786432 max|Δ|=2.855e-01 | FAIL
[REF] p4097_mut1  | o 超界 23266428/25171968 max|Δ|=1.470e-01 | ht 超界 781261/786432 max|Δ|=1.256e+00 | FAIL
[REF] 反向对照（mut1）：如预期变红 ✓
```
（脚本语义：反向对照档**期望 FAIL**；它打印「如预期变红 ✓」并 `rc=0`；若反而 PASS 则打印「判据没咬住 ✗」并 `rc=1`。）

## (b) 根目录陈旧 dump + 缺新鲜度守卫 ⇒ 已修
* **隔离**：根目录（`m23_gdn_prefill/`）的 88 个陈旧 `m23_*.bin` 与全部 `m23_*_meta.txt` 移入 `stale_dumps/`
  ⇒ 根目录残留 **0**；并把 `m23_*_meta.txt`、`stale_dumps/` 加进 `.gitignore`。
* **新鲜度守卫**（`check_ref.py` 新增）：自动定位当前二进制（`<dir>/build/m23_gdn_prefill` 等），
  **判定的每个 dump 的 mtime 必须 ≥ 二进制的 mtime**，否则打印「陈旧 … 拒绝判定」并从判定集里剔除；
  全部陈旧则 `exit 3`；`--allow-stale` 可显式越过。
  **实测（本轮真发生过一次）**：二进制重建后、设备档因抢不到锁没跑成 ⇒ 守卫直接打印
  ```
  [REF][陈旧] 下列 dump 早于当前二进制（…/build/m23_gdn_prefill，mtime 1791101408）⇒ 拒绝判定：m1, m4097
  ```
  并 `exit 3` —— 即"从根目录/旧 dump 误跑给出旧结论"这条路已被堵死。

## 收口读数（新鲜）
* 运行：`M=1 rc=0`、`M=4097 rc=0`、`--mutate 1 rc=0`（三次各一次独立进锁、锁内 `npu-smi` = `Ascend950PR OK`）。
* 判据（`evidence/logs/r3_check_ref.log`，**rc=0**，只判新鲜 dump）：
```
[REF] m1     | o 超界 0/6144      max|Δ|=7.655e-09 | ht 超界 0/786432 max|Δ|=3.313e-08 | PASS
[REF] m4097  | o 超界 0/25171968  max|Δ|=3.042e-08 | ht 超界 0/786432 max|Δ|=1.458e-07 | PASS
```
⇒ 内核两处修复的结论**不变**；本轮只动**判据侧**（`m23_gdn_prefill.asc` 的反向对照位置 + `check_ref.py` 的守卫 + `.gitignore` + 陈旧 dump 隔离），
**未动内核** `m15_layer_loop/m15_gdn_prefill.h`。

## 限度
* 反向对照只跑了 `--mutate 1`（v 翻号）；未跑 mut2/mut3。
* 新鲜判据各档 1 次；未重跑满量级重复统计（M=16/17 那批按"先搁置"停着）。
* 未做 SLOG/性能；仍 `blocked`、不标 completed、**不自行声称 clean**（交复审）。
