# M19 evidence 清单（每份 log 标注时点/commit；tip = partial landing 131441a）

| 文件 | 时点 / commit | 内容 |
|---|---|---|
| `accept_run.log` | M35 时点（含其 5 连跑段；其中 `round-4` 字样属 M35 归档自身，不改） | 4 档 kernel 自检：A=2044 B=2046 C=1024(PASS) D=2044（应 2048/2050/1024/2048） |
| `check_ref_run.log` | **tip 重跑（P2-4 刷新）** | T1 投影逐位 0（4/4）；T3 q/ck/scores 四档 0 违反；离散：选中 511/511/256/511、异常 block 0 |
| `guard_controls.log` | M35 tip（**历史**） | M35 partial-landing 守卫的正/负向对照。**M53 起守卫已移除**（移除条件与读数见 select_single_core.log），本文件只作历史存档 |
| `MECHANISMS_FOUND.md` | **M53** | M35 的 6 机理 + 1 否证 + 1 教训**逐条复核**（(ceq) 原结论被否证、(tail) 现象不成立）+ M53 新增 3 条机理（DEV/BAR/COMP） |
| `step1_single_vs_multi.log` | 中间轮（c96653f 前后） | 单 AIV 全对 2048/2050 vs 多 AIV 1280 ⇒ 确定性 bug 在多核切分 |
| `step2_Bfix_readings.log`、`step2_percore_prefix.log` | 中间轮 | (B) 修复前后 candGT/candEQ；每核四组数（localCgt/myGtBase/期望前缀/clipped） |
| `step4_lanef_and_1block.log`、`step5_1block_mechanism.log` | 中间轮 | (LANEF) 修复前后默认档计数；丢的是 score==K 的边界块 + ceq 对拍 |
| `stability_9runs.log` | 早期（17a6584 时期） | 9 次运行波动 —— **历史存档**（已被 cb24e57/81628a2 的修复取代） |
| `probe_run.log` | M35 时点（历史） | 探针 S0/S3/S6 PASS、S1/S2/S4/S5 FAIL ⇒ **历史存档，勿与 tip 的 ✅ 并列** |
| `probe_aic_blockidx.log` | tip 附近（**读数未改动**） | **摆位不成立 ⇒ 全 FAIL 为伪影**：探针在 AIC 分支经 UB 暂存落盘（`probe_aic_blockidx.asc` 原 `:46-54`；`PutBid` 原 `:26` 构造 VECCALC 张量、`:27-30` 标量 SetValue、`:36-37` UB→GM），而 **AIC 访问不了 UB ⇒ UB→GM 对 AIC 不可路由**（M153 根因；`docs/05-megakernel-design.md:216` 同口径）。⇒ 归档的全 FAIL（`launch: OK`、六槽全 `-1`）**不构成**关于 AIC `GetBlockIdx` 取值的证据。**原记「AIC 侧 UB→MTE3 落盘读回哨兵（全项目级负结果）」已按 M154 C1 / M156 更正**；探针源已撤回、不再构建（README §5.4）。 |
| `select_single_core.log` | `7edd74d` | 单核→多核矩阵（V∈{256,2048}×budget∈{8,32,512}×coreDiv∈{1,2,4,8,56}）+ 跨核一致性 + 默认 4 档 9 次确定性 + 逐块 oracle 读数 |
| `probe_aiv_barrier_runs.log` | `b2a6f2d` | 探针**再独立跑 4 次**的归档（README §3.4「数字口径」的构成：本仓共 5 次归档运行 = 本文件 4 次 + `probe_aiv_barrier.log` 1 次） |
| `probe_aiv_barrier.log` | `c641b5a` | AIV 间计数交换**适用边界**探针 6 变体（单发/循环 × 手写 mode0 flag / 官方 `SyncAll<true>`；+无屏障对照）的违规核数与最少见到槽数 |
| `dump_sha256.txt` | M35 tip | dump 与源码 sha256（M35 时点；M53 起字段与布局已变） |

复现：
```bash
./m19_qsa_indexer/build/m19_qsa_indexer   # 写 m19_out/
/usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_ref.py ./m19_out
```
`m19_out/`、`m19_probe/` 为运行产物，不入库；下次动代码时可把默认 `M19_OUT` 改到模块内再加一行 ignore（P2-8，不作为返工条件）。
