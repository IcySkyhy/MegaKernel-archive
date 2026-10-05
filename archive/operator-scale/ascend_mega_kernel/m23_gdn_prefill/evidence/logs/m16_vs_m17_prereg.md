# 预登记：M=16 vs M=17 对比统计（人类：「我怀疑是 L0A/L0B/L0C 的 buffer 同步没有做好」）**先写后跑**

## A. 第一步（零设备）：**代码事实**（符号 + 行号 + 逐字片段）
`m15_layer_loop/m15_gdn_prefill.h`：

1. **本段的 mmad 没有 m-tile 循环，M 恒为 `GP_BT`(=64)** —— 6 个收缩的调用点（行号）：
   * `:650` `DoMmad(L1Kq(h), …GP_BT*GP_DK, L1Kq(h), …, M15G::GP_BT, M15G::GP_BT, M15G::GP_DK, …)`（M1 kk）
   * `:653` `DoMmad(L1Kq(h)+…, …, L1Kq(h), …, M15G::GP_BT, M15G::GP_BT, M15G::GP_DK, …)`（M2 qk）
   * `:690` `DoMmad(L1St(h), …, L1W(h), …, M15G::GP_DV, M15G::GP_BT, M15G::GP_DK, …)`（M3 (w·S)ᵀ）
   * `:693` `DoMmad(L1Kq(h)+…, …, L1St(h), …, M15G::GP_BT, M15G::GP_DV, M15G::GP_DK, …)`（M4 q·S）
   * `:717` `DoMmad(L1J3(h,ABM), …, L1J3(h,DT), …, M15G::GP_BT, M15G::GP_DV, M15G::GP_BT, …)`（M5 AB·d）
   * `:721` `DoMmad(L1J3(h,DT), …, L1J3(h,KT), …, M15G::GP_DV, M15G::GP_DK, M15G::GP_BT, …)`（M6 (kᵀ·DP)ᵀ）
   ⇒ **每个 mmad 的 M 维只取 `GP_BT`=64 或 `GP_DV`=128，与运行期 `m` 无关**；`m` 只影响 AIV 的装载/清零与 host 的 stride。
2. **L0A / L0B / L0C 是单块、无双缓冲、无槽位切换**（`DoMmad` 内，行号）：
   * `:615` `LocalTensor<float> l0a(TPosition::A2, 0, GP_L0A_MAX_ELEMS);`
   * `:616` `LocalTensor<float> l0b(TPosition::B2, 0, GP_L0B_MAX_ELEMS);`
   * `:617` `LocalTensor<float> l0c(TPosition::CO1, 0, GP_L0C_MAX_ELEMS);`
   ⇒ 三块都是**偏移 0 的单块**（没有 ping/pong、没有 tile index）。
3. **同步是"按 mmad 调用"、用 3 个 BufferID**（同函数，行号）：
   `:619` `Acq<MTE1>(GP_BUF_L0)` → `:620` `Acq<MTE1>(GP_BUF_L1)` → `LoadData` → `:623` `Rel<MTE1>(GP_BUF_L1)`
   → `:624` `Rel<MTE1>(GP_BUF_L0)` → `:625` `Acq<M>(GP_BUF_L0)` → `Mmad` → `:627` `Rel<M>(GP_BUF_L0)`
   → `:628` `Acq<FIX>(GP_BUF_L0C)` → `Fixpipe` → `:630` `Rel<FIX>(GP_BUF_L0C)`。
   ⇒ **每次调用一次完整轮转**；**与"几个 m-tile"无关**（因为恒为一个 tile）。

**由 A 得出的、对本次假设的直接推论（先写下来）**：
人类假设的机制是"**M=17 = 两个 16 行 tile ⇒ 第二个 tile 的 L0A 装载 / L0B / L0C 累加与槽位切换第一次被走到**"。
**本段不存在这个机制**：mmad 的 M 恒为 64（一个 tile），L0 无槽位切换、无 tile 循环。
⇒ 因此若要"L0 同步没做好"这个怀疑成立，它**不可能**表现为"16 挡住、17 开始犯"。

## B. 零设备：**仓内既有标定**（直接引用，不重测）
* `docs/11-attn-analysis.md:103` 逐字：**「m=1 在 mmad 被 pad 成 16（`matmul.h:670-672` "m==1→16"——mega kernel 同样 m 走 16 行矩阵）」**
  ⇒ 16 是 **mmad 的 m 维 pad 粒度**（该结论在 **matmul 高阶 API** 语境下成立；本段用基础 API `Mmad`，M 恒为 64）。
* `m16_load_geom/README.md:169` 逐字：**「（= `N/16`），`mStep`/`kStep` = 源行/列方向的搬运分形个数」**；
  `:175`/`:176`：`mStep` 单位=**源行方向分形个数**、`kStep` 单位=**源列方向分形个数**（实测 conf 18/19/21 的差别即由此）。
  ⇒ **L0 装载的"分形粒度"已被 M27 标定过**，本 mission 不重复测。

## C. 可被推翻的预测（**先写后跑**）
* **预测 P1（人类假设的形态）**：若故障是 **L0A/L0B/L0C 的 16 行 tile 边界同步问题** ⇒
  **命中率(M=17) 应显著高于 命中率(M=16)**（M=16 一个 tile 不触发、M=17 两个 tile 才触发）。
  **推翻条件**：两档命中率**统计上无差别**（见 D 的判据）⇒ P1 **不被**这个切点支持。
* **预测 P2（由 A 的代码事实得出）**：因 mmad M 恒为 64、L0 无双缓冲/无槽位切换，
  **不存在"16→17 才走到第二块 L0"这件事** ⇒ 16 与 17 之间**不应**出现与 L0 同步相关的突变。
  若仍观测到差异，它不能归给"L0 tile 边界"，而应查 AIV 侧 **cv 相关**的路径：
  `:922` `gLen = ((cv + 7) / 8) * 8`、`:938` `ZeroTailVF(scG, cv)`、`:940`/`:941` 尾部 `ZeroVF`、
  **`:948` `BrcScalarVF(scL, scEG + (cv - 1))` —— 该广播源地址在 cv=16 时是字节偏移 60（非 32B 对齐）、
  cv=17 时是 64（32B 对齐）**，这是一个**真实的 16↔17 不对称**（是否违规见 docs/05 §6.1「单元素 load/store 支持非对齐」，
  本 mission 不据此下结论）。

## D. 判据与样本量（先定）
* **FAIL** = `rc != 0` 且 stdout 含 `507015`；**PASS** = `rc == 0` 且含 `ALL RUNS OK`；其余记 OTHER。
* 样本：**M=16 × 30**、**M=17 × 30**（主对比）、**M=24 × 20**（同会话参照锚，上轮 3/20 = 15%）。
* 统计：每档报 **计数 / N / 命中率 / 95% Wilson CI**；两档之间用 **Fisher 精确检验**（双侧，α=0.05）；
  若两档命中率接近，**合并** 60 次算一个 CI，再看能否与 M=24 的 15% 区分。
* **措辞纪律**：某档 30 次全 PASS ⇒ 只能写「**在 30 次里未观察到失败（上界约 9%，95% CI）**」，
  **不得**写"该 M 不会失败"。
* 每次 run **一次独立进锁**（`flock -w 300`、`timeout 25`、进锁前 `npu-smi` 复查、单进程、同二进制/同 `M15GP_ONLY_M` 开关）。
* **本轮只测不改**：不动 `m15_gdn_prefill.h` 的任何同步代码。
