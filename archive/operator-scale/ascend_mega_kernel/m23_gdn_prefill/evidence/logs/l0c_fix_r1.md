# L0C 的 M↔FIXP 成对交接：改动 + 验证读数（人类批准「好，请修改并且验证」）

## 1) 逐字 diff（`git diff` 关键 hunk；`m15_layer_loop/m15_gdn_prefill.h`，净 +10/-1 行）
```diff
-    /** 一次 mmad：A/B 已在 L1；L0 与 L0C 由 BufferID 交接，结果 Fixpipe 落 GM。 */
+    /**
+     * 一次 mmad：A/B 已在 L1；**L0 与 L0C 都由 BufferID 成对交接**，结果 Fixpipe 落 GM。
+     * L0C 的交接是 **M ↔ FIXP 成对**：Mmad 前 M 取 L0C 所有权（挡上一 tile 的 Fixpipe 读 = WAR），
+     * Mmad 后 M 释放（RAW：Fixpipe 必须等 Mmad 完成）—— 与仓内既有站点（m1 / m3 / m11 / m13 /
+     * m14 / m15_gdn_layer / m15_hc_layer / m15_moe_layer / m17 / m20，共 10 处）同形；
+     * 只用 PipeBarrier 不足以建立 M→FIX 与 FIX→M 的跨 pipe 依赖（probe_cube_fp32 的实测：整张表滞后一轮）。
+     */
...
+        BufAcquire<PIPE_M>(M15G::GP_BUF_L0C);   // 新增：M 先取 L0C 所有权（挡上一 tile 的 Fixpipe 读 = WAR）
         BufAcquire<PIPE_M>(M15G::GP_BUF_L0);
         MmadF32(l0c, l0a, l0b, mm, nn, kk, true);
         BufRelease<PIPE_M>(M15G::GP_BUF_L0);
+        BufRelease<PIPE_M>(M15G::GP_BUF_L0C);   // 新增：L0C 结果就绪（RAW：Fixpipe 必须等 Mmad 完成）
         BufAcquire<PIPE_FIX>(M15G::GP_BUF_L0C);
```
硬约束核对：① M 侧 acquire 在 `MmadF32` **之前** ✓；② M 侧 release 在 FIX 侧 acquire **之前** ✓；
③ 两条都是 `BufAcquire`/`BufRelease`（drain 模式）**没用 set/wait flag** ✓；④ 顶部注释已改成与实现一致 ✓。
**未动** L1 布局/资源记账；**未动** 复审登记的 P2-2（只登记不修）。构建 rc=0、`error:` 计数 0。

## 2) A/B/C 读数（每档 10 次，每次一次独立进锁；`evidence/logs/fix/summary.txt`）
| 项 | 档 | 改前 | **改后（逐次序列）** |
|---|---|---|---|
| **A** | `M=4097` ×10 | 10/10 **FAIL** | **`PPPPPPPPPP`（10/10 OK）** ✅ 而 `multi-bit ECC` 行数全为 **0**、无 `mte error info` |
| **B** | `M=1` ×10 | 50/50 OK | **`PPPPPPPPPP`（10/10 OK）** ✅ |
| **C** | `M=17` ×10 | 26/30 OK（4/30 挂） | **`PPPPPPPPPP`（10/10 OK）** ✅ |

⇒ **A 的预期 10/10 达成**（改前 10/10 FAIL）；`multi-bit ECC` **0 行**；**E 无对象**（没有任何失败档，
故没有再比对 `…0202ce`）。

## 3) D：**第一次拿到合格数值读数**（`check_ref.py`，rtol=2e-3）
```
[REF] m4097  H=48 T=4097 nAic=28 rtol=2.0e-03 | o 超界 0/25171968 max|Δ|=3.042e-08 max|ref|=7.500e-02
      | ht 超界 778845/786432 max|Δ|=9.151e-01 max|ref|=5.829e-01 | FAIL
```
* **`o` 判定项：超界 0 / 25,171,968，`max|Δ| = 3.042e-08`（`max|ref| = 7.5e-02`）** ⇒
  **输出 `o` 在 fp32 舍入水平上正确**（改前该档**从未跑完**，因此 `check_ref.py` **从来没有可判对象**——
  这是"挂住了"到"算对了"的分界线）。
* **`ht` 判定项：超界 778,845 / 786,432，`max|Δ| = 9.151e-01`** ⇒ **末态仍然错**，且量级是 O(1)。
  **这不是 L0C 同步那件事**（`o` 正确说明跨 chunk 的状态推进在每一 chunk 都对；若状态推进错，`o` 必错），
  嫌疑落在**末态写回路径**（`GdnPrefillAiv::StoreState` 的两块列窗转置/`DataCopy`）或**token 前向未掩码**
  之类，**本轮未查**（本轮只做 L0C 那一件事）。
* **口径警示**：`check_ref.py` 会把 `evidence/` 里**历次扫描留下的陈旧 dump** 一并判（如 `m26/m27/m28/m29/m31`
  是**改前**、且是"概率失败"批次留下的）⇒ 那些 FAIL **不代表改后状态**，报塔时按陈旧读数处理。

## 4) 判读与限度
* **L0C 的 M↔FIXP 成对交接缺失确实是本次故障的根因**（有 A 的前后对照：10/10 FAIL → 10/10 OK；
  且失败签名 `…0202ce` 与设备 errStr「A multi-bit ECC error occurs when fixpipe reads L0C」一致）。
* **但这不等于"段体就此合格"**：`ht` 仍不合格（778,845 元素超界，`max|Δ|` 0.915）⇒ **还有别的洞**，如实保留。
* **未做**：`ht` 缺陷的定位与修复（本轮只改 L0C 一处）；未重跑满量级统计（M=16/17 那批已按"先搁置"停下）；
  未复跑更大 N 的 A 档（10 次）；未做 SLOG 对照。
* 未改被测段体之外的任何文件；仍 `blocked`、不标 completed、不发 review-request、不改 `docs/19`。
