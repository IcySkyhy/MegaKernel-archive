# 末态 `ht` 修复 + 验证（新鲜 dump 专判）

## A（零设备审计）判定：**布局/转置 bug**（不是缺同步）
逐字审计见 `ht_audit_r1.md`。要点：
* `StoreState` 的跨 pipe 交接**是齐的**（V→MTE3 用 `GP_BUF_BLK` 成对 drain release + 对侧 get；
  上传缓冲用 `UP` 令牌；寄存器↔UB 有 `MemBarVL`）⇒ **不是"缺同步"**。
* 错在 `TransposeVF(stUb + cb * GP_VL * GP_DK, tmp, GP_DV, GP_VL)` 的 **R/C 与源的 stride 不符**：
  `TransposeVF` 的语义（同文件逐字）是「dst[c][r] = src[r][c]；src=[R,C]、dst=[C,R] 行主序」，
  要求源的行 stride 恰为 C；而 ST 是 `[DV,DK]`、行 stride = **DK=128**，调用却传 `R=DV, C=VL=64`
  ⇒ 源被当成「128 行、行 stride 64」读（实际 stride 128）⇒ **错位读取**。
* **同文件自对照**：`LoadState`（h0→ST）前有带 gap 的 `DataCopy` 把列窗**打包成连续 [128,64]**，
  再 `TransposeVF(…, R=DK, C=VL)`（源 stride=64 ✓ 自洽）⇒ **读入路径本来是对的**，只有写回路径错
  ⇒ 完全解释塔的两条推断（结构性错误、`m=1` 就错、且**只错 `ht`**）。

## 修（最小 diff，BufferID 形态未变，未用 set/wait flag）
```diff
-            TransposeVF(stUb + cb * GP_VL * M15G::GP_DK, tmp, M15G::GP_DV, GP_VL);
+            // ST 的行 stride 是 DK(=128)；本半块取 ST 行 [cb*64, cb*64+64)（[64,128]，stride=DK）
+            // ⇒ 转置成 [128,64] 正好是 ht 的列窗 [cb*64, cb*64+64)。
+            TransposeVF(stUb + cb * GP_VL * M15G::GP_DK, tmp, GP_VL, M15G::GP_DK);
```
构建 rc=0、`error:` 0。**只改 `StoreState` 一处**；未动 L1 布局/资源记账、未动其它文件。

## B（口径修正）：**新鲜/陈旧彻底分开**
`cd build && mkdir -p stale_dumps && mv m23_*_meta.txt m23_*.bin stale_dumps/`（历次扫描的陈旧 dump 全部移走），
再跑新鲜两档 ⇒ `check_ref.py` 判的对象**只有**本次的 `m23_m1_*` 与 `m23_m4097_*`。

## 新鲜读数（`check_ref.py`，rtol=2e-3；逐字 `fix2_check_ref.log`，**rc=0**）
```
[REF] m1     H=48 T=1    nAic=28 | o 超界 0/6144     max|Δ|=7.655e-09 max|ref|=2.292e-02
                                | ht 超界 0/786432   max|Δ|=3.313e-08 max|ref|=2.439e-01 | PASS
[REF] m4097  H=48 T=4097 nAic=28 | o 超界 0/25171968 max|Δ|=3.042e-08 max|ref|=7.500e-02
                                | ht 超界 0/786432   max|Δ|=1.458e-07 max|ref|=5.829e-01 | PASS
```
运行读数（每档一次独立进锁；`fix2_M1.log` / `fix2_M4097.log`）：
`M=1 rc=0 0.099 ms`、`M=4097 rc=0`（均 `ALL RUNS OK`）；**`507015`/`aicore error`/`mte error info`/`multi-bit ECC` 均 0 命中**。

## 判读与限度
* **`o` 与 `ht` 在真实 shape（m=1 与 m=4097）上双双 0 超界**，`max|Δ|` 在 7.7e-09 ~ 1.5e-07（fp32 舍入水平）
  ⇒ 末态那个洞**已收口**（本轮唯一的改动就是这一处转置参数）。
* 未重跑满量级重复统计（M=16/17 那批按"先搁置"停着）；**未做**多次重复（各档 1 次新鲜读数）；
  未做 SLOG 对照；未做性能读数（本段的耗时数字只用于确认跑完）。
* **不自行声称 clean**：按塔的纪律交复审（本文件 + diff + 新鲜判据）。
