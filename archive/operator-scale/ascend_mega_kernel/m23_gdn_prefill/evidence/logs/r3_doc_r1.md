# r3 复审的返工：**只在文字**（代码侧零改动）

## 1) 状态口径自相矛盾 ⇒ 已改到与事实一致
`README.md` 的状态块（第 3 行起）与 §4 的现状行原来仍写「设备档/判据档未取得读数、数值判据尚未跑通」，
而 §4c0 与 `evidence/` 已有新鲜 PASS 与反向对照红读数。现已改成：
* 内核两处缺陷**已修且已被独立复跑验证**（① `DoMmad` 的 L0C `M↔FIXP` 成对交接；② `StoreState` 的 `TransposeVF` R/C）；
* 新鲜判据 `check_ref.py` **rc=0**：`m1` 的 `o` 0/6144、`ht` 0/786432；`m4097` 的 `o` 0/25171968、`ht` 0/786432；
* 反向对照 `--mutant 1` ⇒ `m1_mut1`/`p4097_mut1` 双双 FAIL +「如预期变红 ✓」；
* **读数 ↔ 二进制对应**：`fix2_*` ← commit `51be119` 构建的二进制；`r3_*` ← commit `461bfe0` 构建的二进制；
  当前 `build/m23_gdn_prefill` 的 `sha256` 前 24 位 = `cf001d6af43a00c791f3dfce`（二进制不入库，以源码 commit 为锚）。
改后 README 中「尚未跑通 / 未取得读数」的命中数 = **0**。

## 2) 「（已 gitignore）」措辞不准 ⇒ 已订正为准确措辞（采纳复审的二选一之①）
`.gitignore` 里的 `stale_dumps/` **只对未跟踪文件生效**；该目录下 **11 份 `_meta.txt` 是有意保留 tracked 的
「改前读数」存档**（`git check-ignore` 对它们不命中）⇒ 保留历史的理由已落进 README §4c0 与本节。
误跑风险由三重防护挡住（目录名自解释 + 守卫 mtime 拒绝 `exit 3` + 越过须显式 `--allow-stale`），
r3 复审已零设备实测 5 例：`--dir stale_dumps` 全拒 rc=3；`--allow-stale` rc=1；根目录无 `--dir` rc=2（陷阱消失）；
`--dir build` 新鲜 PASS rc=0；混合只剔陈旧、判新鲜 PASS rc=0。

## 3) 已知残留（登记，不抹掉，本轮不修）
见 `README.md §4x` 四条（两条 P3：守卫告警文本与行为不一致、找不到二进制时静默失效；一条 P3：比较对象是 meta 而非每个 bin；
一条格位不足：反向对照与干净档**没有**"同一 salt、只换一个输入"那一格）。
