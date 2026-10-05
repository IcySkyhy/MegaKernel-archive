# M34 README 数字对账表（`audit_readme_numbers.py` 生成，可复算）

> **负向对照产物（不是合格证）**：故意扰动一条断言，用于验证脚本会 FAIL 并点名。

口径：README 里每个数字都要能 grep 回出处。分类定义见脚本 docstring。
**PART 1 每条都把归档真值取回并与 README 引用值做数值比对**（`close` 档容差 2%；
`<=`/`>=` 档按界判），不是「同现即通过」。**退出码三态 0/1/2** 见脚本 docstring。

## 统计

| 项 | 值 |
|---|---|
| README 行数 | 503 |
| **PART 1 断言条数** | **77** |
| ├ 归档取回类 | 50 |
| ├ **真重算类**（脚本独立算出） | **25** |
| └ 常量一致性类（**非独立重算**，只校验 README 与脚本两处字面量一致） | 2 |
| **其中真正比较过（compared）** | **77** |
| **失败（failed）** | **1** |
| **没得比（unverified，本仓输入缺失）** | **0** |
| **没得比（unverified，外部输入缺失）** | **0** |
| PART 2 含数字的行数 | 0 |
| 其中含 ✅归档 的行 | 0 |
| 其中纯编号行（🔢编号） | 0 |
| **PART 3 待裁定行数** | **0** |

## 运行备注（输入可得性 / 交叉见证）

- 负向对照：README 文本被扰动（4640.499 → 9999.999），期望 rc=1 并点名该判定项。

## PART 1：断言（每条都从归档/源文件取回真值 + 数值比对）

| # | 说明 | README 引用 | 出处 | 取回值 | 结论 |
|---|---|---|---|---|---|
| 1 | §4.7 gqa3 run1 | `4640.499` | run_log.txt:2 | 归档值=4640.499 | FAIL |
| 2 | §4.7 gqa3 run2 | `1.272` | run_log_run2.txt:2 | 归档值=1.272 | PASS |
| 3 | §4.7 one run1 | `0.254` | run_log.txt:4 | 归档值=0.254 | PASS |
| 4 | §4.7 one run2 | `0.260` | run_log_run2.txt:4 | 归档值=0.260 | PASS |
| 5 | §4.7 all48 run1 | `0.729` | run_log.txt:6 | 归档值=0.729 | PASS |
| 6 | §4.7 all48 run2 | `0.732` | run_log_run2.txt:6 | 归档值=0.732 | PASS |
| 7 | §4.7 target run1 | `15.514` | run_log.txt:8 | 归档值=15.514 | PASS |
| 8 | §4.7 target run2 | `15.522` | run_log_run2.txt:8 | 归档值=15.522 | PASS |
| 9 | §4.7 target probe-off run1 | `15.494` | run_log.txt:11 | 归档值=15.494 | PASS |
| 10 | §4.7 target probe-off run2 | `15.503` | run_log_run2.txt:11 | 归档值=15.503 | PASS |
| 11 | §4.7 gqa3 run3（离群） | `80044.004` | run_log_run3.txt:2 | 归档值=80044.004 | PASS |
| 12 | §4.7 one run3 | `0.257` | run_log_run3.txt:4 | 归档值=0.257 | PASS |
| 13 | §4.7 all48 run3 | `0.734` | run_log_run3.txt:6 | 归档值=0.734 | PASS |
| 14 | §4.7 target run3 | `15.516` | run_log_run3.txt:8 | 归档值=15.516 | PASS |
| 15 | §4.7 target probe-off run3 | `15.506` | run_log_run3.txt:11 | 归档值=15.506 | PASS |
| 16 | §4.7 AIV 线程数 | `56` | run_log.txt:1 | 归档值=56 | PASS |
| 17 | §4.7 nblk(AIC) | `28` | run_log.txt:1 | 归档值=28 | PASS |
| 18 | §4.4 max|Δ| | `8.799e-07` | check_ref_log.txt:574 | 归档值=8.799e-07 | PASS |
| 19 | §4.4 max|Δ| Σ|terms| | `1.975` | check_ref_log.txt:574 | 归档值=1.975e+00 | PASS |
| 20 | §4.4 maxRel | `1.133e+01` | check_ref_log.txt:437 | 归档值=1.133e+01 | PASS |
| 21 | §4.4 max|exp| | `1.854` | check_ref_log.txt:55 | 归档值=1.854e+00 | PASS |
| 22 | §4.4 maxΣ|terms| | `8.296` | check_ref_log.txt:80 | 归档值=8.296e+00 | PASS |
| 23 | §4.4 T3 最差占用 | `0.0158` | check_ref_log.txt:146 | 归档值=1.579e-02 | PASS |
| 24 | §4.4 相对|out|占用 | `2.26e+05` | check_ref_log.txt:437 | 归档值=2.263e+05 | PASS |
| 25 | §4.4 S 范围下界 | `1.536e-07` | check_ref_log.txt:486 | 归档值=1.536e-07 | PASS |
| 26 | §4.4 S 范围上界 | `2.684e-07` | check_ref_log.txt:451 | 归档值=2.684e-07 | PASS |
| 27 | §4.4 o 行 max|Δ| | `4.737e-08` | check_ref_log.txt:437 | 归档值=4.737e-08 | PASS |
| 28 | §4.4 o 行 T3 占用 | `2.344e-03` | check_ref_log.txt:229 | 归档值=2.344e-03 | PASS |
| 29 | §4.4 o 行 相对|out|占用 | `2.263e+05` | check_ref_log.txt:437 | 归档值=2.263e+05 | PASS |
| 30 | §4.4 o 行 maxΣ|terms| | `2.096e+00` | check_ref_log.txt:437 | 归档值=2.096e+00 | PASS |
| 31 | §4.4 ht 行 max|Δ| | `2.553e-07` | check_ref_log.txt:438 | 归档值=2.553e-07 | PASS |
| 32 | §4.4 ht 行 T3 占用 | `5.682e-03` | check_ref_log.txt:438 | 归档值=5.682e-03 | PASS |
| 33 | §4.4 ht 行 相对|out|占用 | `1.233e+03` | check_ref_log.txt:438 | 归档值=1.233e+03 | PASS |
| 34 | §4.4 S0 行 max|Δ| | `2.302e-07` | check_ref_log.txt:439 | 归档值=2.302e-07 | PASS |
| 35 | §4.4 S64 行 max|Δ| | `2.038e-07` | check_ref_log.txt:503 | 归档值=2.038e-07 | PASS |
| 36 | §4.3 target o 判定 | `0/25171968` | check_ref_log.txt:269 | 归档值=0/25171968 | PASS |
| 37 | §4.3 target ht 判定 | `0/786432` | check_ref_log.txt:174 | 归档值=0/786432 | PASS |
| 38 | §4.3 chunk 状态判定 | `0/16384` | check_ref_log.txt:13 | 归档值=0/16384 | PASS |
| 39 | §4.3 target q 判定 | `0/8390656` | check_ref_log.txt:267 | 归档值=0/8390656 | PASS |
| 40 | §4.3 target k 判定 | `0/8390656` | check_ref_log.txt:267 | 归档值=0/8390656 | PASS |
| 41 | §4.5 G1 q max(target) | `0.186` | check_ref_log.txt:263 | 归档值=1.863e-01 | PASS |
| 42 | §4.5 G1 q std(target) | `0.088` | check_ref_log.txt:263 | 归档值=8.839e-02 | PASS |
| 43 | §4.5 G1 v max(target) | `1.0` | check_ref_log.txt:263 | 归档值=1.000e+00 | PASS |
| 44 | §4.5 G1 v std(target) | `0.577` | check_ref_log.txt:263 | 归档值=5.773e-01 | PASS |
| 45 | §4.5 G2 target 演化 | `128/128` | check_ref_log.txt:600 | 归档值=演化 128/128 | PASS |
| 46 | §4.5 G2 终态≠初态 | `48/48` | check_ref_log.txt:255 | 归档值=48/48 | PASS |
| 47 | §1 donor tiling BT=64 | `64` | chunk_gated_delta_rule_tiling.cpp:120 | 归档值=64 | PASS |
| 48 | §4.1 一层口径 max Σ|terms|(S) | `1.294e+00` | check_ref_log.txt:263 | 归档值=1.294e+00 | PASS |
| 49 | §4.1 递归口径 max（正反馈） | `8.338e+41` | check_ref_log.txt:263 | 归档值=8.338e+41 | PASS |
| 50 | §4.1 递归/一层 比值 | `6.44e+41` | check_ref_log.txt:263 | 归档值=6.44e+41 | PASS |
| 51 | §4.2 γ₁₂₈ | `7.63e-6` | 脚本重算（gamma128，2% 恒等） | 计算值=7.62945e-06 | PASS |
| 52 | §4.2 δ_exp（Exp 1 ulp） | `1.19e-7` | 脚本重算（delta_exp，2% 恒等） | 计算值=1.19209e-07 | PASS |
| 53 | §4.2 65·δ_exp | `7.7e-6` | 脚本重算（c65delta，2% 恒等） | 计算值=7.7486e-06 | PASS |
| 54 | §4.2 ε 合计 | `4.48e-5` | 脚本重算（eps_sum，2% 恒等） | 计算值=4.48028e-05 | PASS |
| 55 | §4.2 γ₆₄ | `3.82e-6` | 脚本重算（gamma64，2% 恒等） | 计算值=3.81471e-06 | PASS |
| 56 | §4.2 κ∞ = ‖(I+A)⁻¹‖∞ | `3.212` | 脚本重算（kappa_inf，2% 恒等） | 计算值=3.21161 | PASS |
| 57 | §4.2 ‖A‖∞（说明 Neumann 界不适用） | `2.31` | 脚本重算（A_inf，2% 恒等） | 计算值=2.31378 | PASS |
| 58 | §4.2 u = 2^-24 | `5.96e-8` | 脚本重算（u_f32，2% 恒等） | 计算值=5.96046e-08 | PASS |
| 59 | §4.4 UB 占用 KiB | `225.5` | 脚本重算（ub_kib，2% 恒等） | 计算值=225.5 | PASS |
| 60 | §4.4 余量 = 1/占用 | `63` | 脚本重算（margin，2% 恒等） | 计算值=63.3312 | PASS |
| 61 | §4.4 T3 占用中位 | `4.60e-03` | 脚本重算（occ_median，2% 恒等） | 计算值=0.004604 | PASS |
| 62 | §4.4 ≤1ulp 中位 | `16.0` | 脚本重算（ulp1_median，2% 恒等） | 计算值=16 | PASS |
| 63 | §4.4 ≤1ulp 最小 | `13.2` | 脚本重算（ulp1_min，2% 恒等） | 计算值=13.2 | PASS |
| 64 | §10 golden 递推=(I+L)^-1 误差（机器精度量级上界） | `4.0e-15` | 脚本重算（golden_err，计算值 ≤ README 上界） | 计算值=1.11022e-15 | PASS |
| 65 | §10 判别：golden ≠ (I−L)^-1（残差 O(1)） | `1.0` | 脚本重算（golden_wrong，计算值 ≥ README 下界） | 计算值=14.9717 | PASS |
| 66 | §4.2 ‖A‖∞（Neumann 界不适用） | `2.31` | 脚本重算（A_inf，2% 恒等） | 计算值=2.31378 | PASS |
| 67 | §4.2 κ∞ | `3.212` | 脚本重算（kappa_inf，2% 恒等） | 计算值=3.21161 | PASS |
| 68 | §2 UB_G 偏移 | `147456` | 脚本重算（UB_G，2% 恒等） | 计算值=147456 | PASS |
| 69 | §2 UB_H 偏移 | `163840` | 脚本重算（UB_H，2% 恒等） | 计算值=163840 | PASS |
| 70 | §2 target chunk 数 | `65` | 脚本重算（nc_target，2% 恒等） | 计算值=65 | PASS |
| 71 | §4.7 每 chunk µs | `238` | 脚本重算（per_chunk_us，2% 恒等） | 计算值=238.677 | PASS |
| 72 | §4.7 G 指令/s/AIV | `0.83` | 脚本重算（instr_per_s，2% 恒等） | 计算值=0.837953 | PASS |
| 73 | §4.7 指令/cycle（含 1.8GHz 假设） | `0.46` | 脚本重算（cyc_per_instr，2% 恒等） | 计算值=0.465529 | PASS |
| 74 | §4.7 单层 ms | `15.51` | 脚本重算（ms_per_layer，2% 恒等） | 计算值=15.514 | PASS |
| 75 | §4.7 36 层外推 | `559` | 脚本重算（prefill_36，2% 恒等） | 计算值=558.504 | PASS |
| 76 | §4.2 ε 取值 | `5e-5` | 常量一致性（eps_used；**非独立重算**，只校验 README 与脚本两处字面量一致） | 脚本常量=5e-05 | PASS |
| 77 | §4.2 Γ/eg 传递项 | `3e-7` | 常量一致性（gamma_transfer；**非独立重算**，只校验 README 与脚本两处字面量一致） | 脚本常量=3e-07 | PASS |
