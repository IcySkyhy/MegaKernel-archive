# A（零设备）：末态路径审计 —— 判定 = **布局/转置 bug**（不是缺同步）

## 1) 末态路径上的每一个跨 pipe 交接（逐条）
`GdnPrefillAiv::StoreState`（`m15_gdn_prefill.h`）每半块（`cb ∈ {0,1}`）：
| # | 交接 | 写法 | 成对？ |
|---|---|---|---|
| 1 | V(转置写 UB) → MTE3(搬走) | `BufAcquire<PIPE_V>(GP_BUF_BLK)` → `TransposeVF` → `BufRelease<PIPE_V>(GP_BUF_BLK)`；`BufAcquire<PIPE_MTE3>(GP_BUF_BLK)` → `DataCopy` → `BufRelease<PIPE_MTE3>(GP_BUF_BLK)` | **成对**（drain release + 对侧 get）✓ |
| 2 | 上传缓冲生存期（ST / UB_K） | `UpAcqV()/UpRelV()`、`UpAcqMte3()/UpRelMte3()` | 成对 ✓（且本处 ST 只被读） |
| 3 | 寄存器 ↔ UB（`Gather` 读 ST、`StoreAlign` 写 UB_K） | `MemBarVL()`（`LocalMemBar<VEC_STORE,VEC_LOAD>`） | 有 ✓ |
| 4 | UB_K 同 pipe 复用（跨 `cb` 重复写） | 循环内两次写同一 `UB_K`，中间有 MTE3 搬出（不同 pipe）+ 下一轮 V 取得 BLK | 由 BLK 轮转保证 ✓ |
⇒ **末态路径的跨 pipe 交接是齐的**；**没有"缺同步"**（与 §L0C 那次不同）。

## 2) 对照仓内同类站点
`LoadState`（同一文件，h0→ST）用的是**同一套**令牌与 `MemBarVL`，但它的 `TransposeVF` 前面有一次
**带 gap 的 `DataCopy`**（`DataCopyParams{DK, 8, 8, 0}`）把 h0 的列窗**打包成连续 [128,64]**，
再 `TransposeVF(tmp, …, R=DK, C=VL)`（源 stride = 64 ✓ 与 R/C 自洽）⇒ **`LoadState` 是对的**。
⇒ 同文件内的**自对照**：读入路径对、写回路径错 ⇒ 差异只能在写回那一步的**转置参数**上。

## 3) 判定（符号 + 行 + 逐字）
`StoreState` 的 `TransposeVF(stUb + cb * GP_VL * M15G::GP_DK, tmp, M15G::GP_DV, GP_VL);`
* `TransposeVF` 的语义（同文件逐字）：**「dst[c][r] = src[r][c]；src=[R,C]、dst=[C,R] 行主序」**
  ⇒ 它要求源是**行 stride 恰为 C** 的连续矩阵；
* 而 ST 是 `[DV,DK]`（行 stride = **DK = 128**）；调用传 `R=GP_DV=128, C=GP_VL=64`
  ⇒ 源被当成 `[128 行, 行 stride 64]` 读（实际 stride 128）⇒ **每次 Gather 读的是错位的元素**
  （`src[lane*64 + c]` 而非 `src[lane*128 + c]`）；
* 正确调用应为 `R=GP_VL=64, C=GP_DK=128`（源 = ST 的行 `[cb*64, cb*64+64)` 这个 **[64,128]、stride=128** 的
  半块 ⇒ 转置成 `[128,64]` = ht 的列窗 `[cb*64, cb*64+64)`）。
* **这解释了塔的两条推断**：① 结构性错误（98% 元素错、O(1) 量级）——错位读取必然大面积错；
  ② `m=1` 就已经错 —— 与序列长度无关，是**写回路径本身**；
  ③ 而且**只错在 `ht`**：`o` 的输出不走这条路径 ⇒ `o` 全对 ✓。
