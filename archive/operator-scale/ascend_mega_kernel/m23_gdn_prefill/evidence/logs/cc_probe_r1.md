# 核间握手探针读数（M115，塔裁 ③ 的最小探针）

被测形态 = 本段体实际用的那种交接：**AIV 写 GM（MTE3 DMA）→ set → AIC 等 → 从 GM 读（Nd2Nz）= GM 中介**，
以及反向。探针源码 `m23_cc_probe.asc`（自己的 target `m23_cc_probe`）。

**变量**：`variant`（哪一处形态不同）、`iters`（重复轮数）。
**判据**：不是数值，而是「**返回 / 不返回**」。

| variant | 形态 | 读数 |
|---|---|---|
| 0 | AIV set 挂 `PIPE_MTE3`；**2 个 AIV set(GO) → AIC 等 1 次**；AIC set(DONE) 挂 `PIPE_MTE2`，1 set → 2 个 AIV 各等 1 次 | **返回 OK** @ iters = 4 / 8 / 16 / 24 / 200（0.36–0.46 ms） |
| 1 | 同 0，但 **2 个 AIV set(GO) → AIC 等 2 次** | **不返回（timeout 124）** @ iters = 4 / 8 / 16（每个都挂） |
| 2 | 同 0，但 AIC 侧**真发一次 Fixpipe（L0C→GM）并把 DONE 的 set 挂 `PIPE_FIX`**（= 本段体 AIC 侧的形态） | **返回 OK** @ iters = 8 / 40（0.39–0.49 ms） |
| 3 | 同 0，但 AIV 的 set 挂 `PIPE_V`（= 本段体 GO1 的形态） | **返回 OK** @ iters = 8 / 40（0.38–0.41 ms） |

## 判读（逐条）
1. **本段体用的那套 mode 2 形态是成立的**：GM 中介 + `PIPE_MTE3`/`PIPE_FIX`/`PIPE_V` 三处 set +
   「2 AIV set → 1 AIC wait」+「1 AIC set → 2 AIV wait」在 200 轮重复下都返回正常
   ⇒ **不是**「每 id 4bit 计数器未配平超 15」那类问题（200 轮远超 15）。
2. **配平是严格的**：把「2 set → 1 wait」改成「2 set → 2 wait」**立刻挂**（iters=4 就挂）
   ⇒ 该形态下的配对必须是 **N set ↔ 1 wait**（或 1 set ↔ N wait），多等一次即死锁。
   ⚠ 这是**塔裁①的前提不成立**的直接证据：本段体的 6 个 flag **全部**是「2 AIV set → 1 AIC wait」，
   即**成对同步**，不是 all-to-all；GM 中介只改变数据通路，不改变"AIC 只等它自己那对 AIV"这件事。
   ⇒ 按人类裁决的字面（"prev op 落 GM / next op 从 GM 读 ⇒ 不是特定一个核对一个核"），本段体**不属于**
   那一类（本段体的前后不是两个 op，而是同一段体内固定配对的 AIC 与其 2 个 AIV）。
3. 因此本段体的挂死**另有根因**，三个核间假设（mode 选择 / 计数器 / set 挂的 pipe）**均被这组读数排除**。
