#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m18_gdn_prefill/audit_readme_numbers.py —— README 数字对账（把每个数字 grep 回出处）。

背景：M34 在 round-1/2/3 被连抓三次「README 数字与同 commit 归档不一致」。本脚本把这件事
**机械化**，产物落到 `evidence/readme_number_audit.md`，任何人可一条命令复算。

三件事：
1. **PART 1 断言表**：每个"声称来自归档"的数字，从归档（evidence/*.txt）按定位器取回真值，
   并**与被引用的数字做数值比对**，不一致即 FAIL。另有若干"脚本可重算"的推导值就地重算比对
   （γ₁₂₈/δ_exp/UB 占用/余量/占用中位/golden↔(I+L)⁻¹/‖A‖∞/κ∞）。
2. **PART 2 逐行对账表**：README 每一行含数字的行逐个列出，给出该行数字的出处分类。
3. **PART 3 待裁定清单**：非平凡数字里找不到独立出处的，单列出来必须人工裁定。

分类（PART 2）：
  ✅归档  由归档日志/清单硬校验通过（PART 1 断言）
  ✅推导  脚本可重算
  ✅常量  仓内源文件里的设计常量（.asc / check_ref.py）
  📚外部  docs/* 、donor 行号、官方 spec、模型 config
  🔁活值  运行时量（计时），归档行号有限且跨运行会变
  🔢编号  纯编号/章节/行号/≤2 位小整数，无需独立出处
  🔵假设  行内注明"假设"的项
  ⚠待裁定 以上都不是

────────────────────────────────────────────────────────────────────────
**退出码（三态；调用方/CI 只看退出码时的契约）**
  0 = 比过且通过   compared > 0 且 failed == 0 且 **本仓输入** 无 unverified
  1 = 比过且有差异 failed > 0（有差异时优先报 1，即使同时有 unverified）
  2 = 没得比/输入缺失 compared == 0，或存在本仓输入缺失导致的 unverified
**没有真比较过就不发合格证**：`RESULT: OK` 只在 (failed == 0 且 local unverified == 0 且
compared > 0) 时打印，且**必须带实际比较计数**。否则打印 `RESULT: NOT CERTIFIED`（rc=2）。

**覆盖范围（脚本自己交代）**：PART 1 分三栏计类 —— **归档取回类 / 真重算类 / 常量一致性类**
（外加三态 compared / failed / unverified），unverified 逐条点名「哪条断言、缺哪个输入」。
外部路径（不在本仓的 donor/spec）单列一栏，不阻塞本仓合格证，但**始终打印在合格证那一行**，不静默。
**常量一致性类不算「可重算」**：`§4.2 ε 取值` 与 `§4.2 Γ/eg 传递项` 的「计算值」就是本脚本里的字面量
常数 ⇒ 恒真（评审观察 1），故单列一栏，只校验 README 与脚本两处字面量一致；真重算类为 25 条。

**负向对照**：`--negative-control` 跑两组对照（"判据自己要会咬"在脚本层的落地）：
  **A 扰动** —— 把一条输入在仓内的断言在 README 文本里改错，要求 rc=1 **并点名该判定项**；
  **B 缺输入** —— 强制几何量归档缺失，要求 rc=2（NOT CERTIFIED）且 ‖A‖∞/κ∞/ε 四条标成
  UNVERIFIED，**不得静默回退到已作废的 Neumann ε**。
  对照自身退出码：0=两组都有效 / 1=脚本没咬住 / 2=无法构造对照。
  `--no-geometry` 是 B 的可复现命令行形态（产物另写，不覆盖正式合格证）。

**依赖**：纯标准库（无 numpy）。本仓输入全部为仓内文件：
  README.md；evidence/{run_log*.txt, check_ref_log.txt, probe_geometry.txt}；
  源文件 m18_gdn_prefill.asc / check_ref.py / CMakeLists.txt。
  可选交叉见证：build/*_probe.bin（存在则从原始字节独立复算 ‖A‖∞/κ∞ 与归档值对照，
  不存在则跳过并标注 —— **不作 FAIL**，因为 dump 本就不入库）。
  **几何量为何有归档**：κ∞/‖A‖∞ 本要从 build/*_probe.bin 复算，而 dump 不入库 ⇒ 干净 checkout
  上拿不到。故把复算结果连 dump 的 sha256 一起归档进 evidence/probe_geometry.txt，使干净
  checkout 也能对账；dump 在时再用独立复算作交叉见证。

**判据模式**：
  close   README 值 ≈ 取回值（容差 2%，允许四舍五入）
  <= / >= README 值是上/下界（随机数据下的**经验上界**必须按界判，不能按 2% 恒等判）
────────────────────────────────────────────────────────────────────────
用法（任意 cwd）：
  /usr/local/python3.12.13/bin/python3 audit_readme_numbers.py                    # 正常对账
  /usr/local/python3.12.13/bin/python3 audit_readme_numbers.py --negative-control # 负向对照 A+B
  /usr/local/python3.12.13/bin/python3 audit_readme_numbers.py --no-geometry      # 缺输入读数
"""
import array
import hashlib
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))       # 仓根（worktree / 主 checkout 都适用）
EV = os.path.join(HERE, "evidence")
README = os.path.join(HERE, "README.md")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:%|×10[-+]?\d+)?")
REL_TOL = 0.02          # 数值比对容差（README 常做四舍五入）
U_F32 = 2.0 ** -24
DELTA_EXP = 2.0 ** -23

# 断言三态
PASS, FAIL, UNVER = "PASS", "FAIL", "UNVERIFIED"
# 本仓输入 vs 外部输入（外部缺失不阻塞本仓合格证，但始终打印）
LOCAL_KIND, EXTERNAL_KIND = "local", "external"

GEOMETRY = "probe_geometry.txt"


def read_text(path):
    with open(path, errors="ignore") as f:
        return f.read()


def resolve(f):
    """把 CLAIMS 里的路径解析到实际文件；从仓根逐级上溯，兼顾 worktree 布局与外部 donor。"""
    if os.path.isabs(f):
        return f if os.path.exists(f) else None
    cands = [os.path.join(EV, f), os.path.join(HERE, f)]
    d = ROOT
    for _ in range(6):
        cands.append(os.path.join(d, f))
        d = os.path.dirname(d)
    cands.append(f)
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def is_external(f):
    return os.path.isabs(f) or f.startswith("..")


def ev_lines(name):
    return read_text(os.path.join(EV, name)).split("\n")


def as_float(s):
    try:
        return float(str(s).replace("%", "").replace(",", ""))
    except ValueError:
        return None


def num_match_any(tok, values, tol=REL_TOL):
    """把 token 当数字，与一组数值逐个比较（四舍五入安全）。"""
    a = as_float(tok)
    if a is None:
        return False
    for v in values:
        b = as_float(v)
        if b is None:
            if str(tok) in str(v):
                return True
            continue
        if b == 0:
            if abs(a) < 1e-12:
                return True
        elif abs(a - b) / abs(b) <= tol:
            return True
    return False


def num_close(lit, val):
    """README 引用值 vs 取回值：能转 float 就按 2% 容差比，否则按字符串包含。"""
    a, b = as_float(lit), as_float(val)
    if a is not None and b is not None:
        if b == 0:
            return abs(a) < 1e-12
        return abs(a - b) / abs(b) <= REL_TOL
    return str(lit) in str(val)


def num_judge(lit, val, mode):
    """按判据模式比较。mode ∈ {close, <=, >=}。返回 (ok, 说明)。"""
    a, b = as_float(lit), as_float(val)
    if a is None or b is None:
        return num_close(lit, val), "字符串包含"
    if mode == "close":
        return num_close(lit, val), "2% 恒等"
    if mode == "<=":
        return b <= a * (1.0 + 1e-9), "计算值 ≤ README 上界"
    if mode == ">=":
        return b >= a * (1.0 - 1e-9), "计算值 ≥ README 下界"
    raise ValueError("unknown mode %r" % mode)


# ------------------------------------------------------------------ PART 1
# (说明, README 引用值, 归档文件, 归档定位器[必须唯一钉住取值], 归档取值的 group)
CLAIMS = [
    # §4.7 计时（活值，逐条对归档行）
    ("§4.7 gqa3 run1", "4640.499", "run_log.txt", r"H=3  T=257  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 gqa3 run2", "1.272", "run_log_run2.txt", r"H=3  T=257  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 one run1", "0.254", "run_log.txt", r"H=8  T=64   blk=28 : ([\d.]+) ms", 1),
    ("§4.7 one run2", "0.260", "run_log_run2.txt", r"H=8  T=64   blk=28 : ([\d.]+) ms", 1),
    ("§4.7 all48 run1", "0.729", "run_log.txt", r"H=48 T=129  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 all48 run2", "0.732", "run_log_run2.txt", r"H=48 T=129  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 target run1", "15.514", "run_log.txt", r"H=48 T=4097 blk=28 : ([\d.]+) ms", 1),
    ("§4.7 target run2", "15.522", "run_log_run2.txt", r"H=48 T=4097 blk=28 : ([\d.]+) ms", 1),
    ("§4.7 target probe-off run1", "15.494", "run_log.txt", r"([\d.]+) ms  \(probeHeads=0", 1),
    ("§4.7 target probe-off run2", "15.503", "run_log_run2.txt", r"([\d.]+) ms  \(probeHeads=0", 1),
    ("§4.7 gqa3 run3（离群）", "80044.004", "run_log_run3.txt", r"H=3  T=257  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 one run3", "0.257", "run_log_run3.txt", r"H=8  T=64   blk=28 : ([\d.]+) ms", 1),
    ("§4.7 all48 run3", "0.734", "run_log_run3.txt", r"H=48 T=129  blk=28 : ([\d.]+) ms", 1),
    ("§4.7 target run3", "15.516", "run_log_run3.txt", r"H=48 T=4097 blk=28 : ([\d.]+) ms", 1),
    ("§4.7 target probe-off run3", "15.506", "run_log_run3.txt", r"([\d.]+) ms  \(probeHeads=0", 1),
    ("§4.7 AIV 线程数", "56", "run_log.txt", r"AIV 线程数=(\d+)", 1),
    ("§4.7 nblk(AIC)", "28", "run_log.txt", r"nblk\(AIC\)=(\d+)", 1),
    # §4.4 全局极值
    ("§4.4 max|Δ|", "8.799e-07", "check_ref_log.txt", r"(8\.799e-07)", 1),
    ("§4.4 max|Δ| Σ|terms|", "1.975", "check_ref_log.txt", r"(1\.975e\+00)", 1),
    ("§4.4 maxRel", "1.133e+01", "check_ref_log.txt", r"(1\.133e\+01)", 1),
    ("§4.4 max|exp|", "1.854", "check_ref_log.txt", r"(1\.854e\+00)", 1),
    ("§4.4 maxΣ|terms|", "8.296", "check_ref_log.txt", r"(8\.296e\+00)", 1),
    ("§4.4 T3 最差占用", "0.0158", "check_ref_log.txt", r"(1\.579e-02)", 1),
    ("§4.4 相对|out|占用", "2.26e+05", "check_ref_log.txt", r"(2\.263e\+05)", 1),
    ("§4.4 S 范围下界", "1.536e-07", "check_ref_log.txt", r"(1\.536e-07)", 1),
    ("§4.4 S 范围上界", "2.684e-07", "check_ref_log.txt", r"(2\.684e-07)", 1),
    # §4.4 target 关键行（逐字取）
    ("§4.4 o 行 max|Δ|", "4.737e-08", "check_ref_log.txt", r"(4\.737e-08)", 1),
    ("§4.4 o 行 T3 占用", "2.344e-03", "check_ref_log.txt", r"(2\.344e-03)", 1),
    ("§4.4 o 行 相对|out|占用", "2.263e+05", "check_ref_log.txt", r"(2\.263e\+05)", 1),
    ("§4.4 o 行 maxΣ|terms|", "2.096e+00", "check_ref_log.txt", r"(2\.096e\+00)", 1),
    ("§4.4 ht 行 max|Δ|", "2.553e-07", "check_ref_log.txt", r"(2\.553e-07)", 1),
    ("§4.4 ht 行 T3 占用", "5.682e-03", "check_ref_log.txt", r"(5\.682e-03)", 1),
    ("§4.4 ht 行 相对|out|占用", "1.233e+03", "check_ref_log.txt", r"(1\.233e\+03)", 1),
    ("§4.4 S0 行 max|Δ|", "2.302e-07", "check_ref_log.txt", r"(2\.302e-07)", 1),
    ("§4.4 S64 行 max|Δ|", "2.038e-07", "check_ref_log.txt", r"(2\.038e-07)", 1),
    # §4.3 判定计数
    ("§4.3 target o 判定", "0/25171968", "check_ref_log.txt", r"(0/25171968)", 1),
    ("§4.3 target ht 判定", "0/786432", "check_ref_log.txt", r"(0/786432)", 1),
    ("§4.3 chunk 状态判定", "0/16384", "check_ref_log.txt", r"(0/16384)", 1),
    ("§4.3 target q 判定", "0/8390656", "check_ref_log.txt", r"(0/8390656)", 1),
    ("§4.3 target k 判定", "0/8390656", "check_ref_log.txt", r"(0/8390656)", 1),
    # §4.5 guard 细节
    ("§4.5 G1 q max(target)", "0.186", "check_ref_log.txt",
     r"\[target\][\s\S]*?G1 输入非空洞:q.*?max\|·\|=([\d.e+-]+)", 1),
    ("§4.5 G1 q std(target)", "0.088", "check_ref_log.txt",
     r"\[target\][\s\S]*?G1 输入非空洞:q.*?std=([\d.e+-]+)", 1),
    ("§4.5 G1 v max(target)", "1.0", "check_ref_log.txt",
     r"\[target\][\s\S]*?G1 输入非空洞:v.*?max\|·\|=([\d.e+-]+)", 1),
    ("§4.5 G1 v std(target)", "0.577", "check_ref_log.txt",
     r"\[target\][\s\S]*?G1 输入非空洞:v.*?std=([\d.e+-]+)", 1),
    ("§4.5 G2 target 演化", "128/128", "check_ref_log.txt", r"(演化 128/128)", 1),
    ("§4.5 G2 终态≠初态", "48/48", "check_ref_log.txt", r"终态≠初态\(all heads\).*?(48/48)", 1),
    # 外部引用存在性
    ("§1 donor tiling BT=64", "64", "../ops-transformer/attention/chunk_gated_delta_rule/op_host/"
                                    "chunk_gated_delta_rule_tiling.cpp", r"int64_t c = (\d+)", 1),
    ("§4.1 一层口径 max Σ|terms|(S)", "1.294e+00", "check_ref_log.txt",
     r"\[target\][\s\S]*?一层\+传递 max Σ\|terms\|\(S\)=([\d.e+\-]+)", 1),
    ("§4.1 递归口径 max（正反馈）", "8.338e+41", "check_ref_log.txt",
     r"\[target\][\s\S]*?递归展开 max=([\d.e+\-]+)", 1),
    ("§4.1 递归/一层 比值", "6.44e+41", "check_ref_log.txt",
     r"\[target\][\s\S]*?比值 ([\d.e+\-]+)", 1),
]

# 脚本可重算的推导值：(说明, README 引用值, 计算键, 判据模式)
CALC_CLAIMS = [
    ("§4.2 γ₁₂₈", "7.63e-6", "gamma128", "close"),
    ("§4.2 δ_exp（Exp 1 ulp）", "1.19e-7", "delta_exp", "close"),
    ("§4.2 65·δ_exp", "7.7e-6", "c65delta", "close"),
    ("§4.2 ε 合计", "4.48e-5", "eps_sum", "close"),
    ("§4.2 γ₆₄", "3.82e-6", "gamma64", "close"),
    ("§4.2 κ∞ = ‖(I+A)⁻¹‖∞", "3.212", "kappa_inf", "close"),
    ("§4.2 ‖A‖∞（说明 Neumann 界不适用）", "2.31", "A_inf", "close"),
    ("§4.2 u = 2^-24", "5.96e-8", "u_f32", "close"),
    ("§4.4 UB 占用 KiB", "225.5", "ub_kib", "close"),
    ("§4.4 余量 = 1/占用", "63", "margin", "close"),
    ("§4.4 T3 占用中位", "4.60e-03", "occ_median", "close"),
    ("§4.4 ≤1ulp 中位", "16.0", "ulp1_median", "close"),
    ("§4.4 ≤1ulp 最小", "13.2", "ulp1_min", "close"),
    ("§10 golden 递推=(I+L)^-1 误差（机器精度量级上界）", "4.0e-15", "golden_err", "<="),
    ("§10 判别：golden ≠ (I−L)^-1（残差 O(1)）", "1.0", "golden_wrong", ">="),
    ("§4.2 ‖A‖∞（Neumann 界不适用）", "2.31", "A_inf", "close"),
    ("§4.2 κ∞", "3.212", "kappa_inf", "close"),
    ("§2 UB_G 偏移", "147456", "UB_G", "close"),
    ("§2 UB_H 偏移", "163840", "UB_H", "close"),
    ("§2 target chunk 数", "65", "nc_target", "close"),
    ("§4.7 每 chunk µs", "238", "per_chunk_us", "close"),
    ("§4.7 G 指令/s/AIV", "0.83", "instr_per_s", "close"),
    ("§4.7 指令/cycle（含 1.8GHz 假设）", "0.46", "cyc_per_instr", "close"),
    ("§4.7 单层 ms", "15.51", "ms_per_layer", "close"),
    ("§4.7 36 层外推", "559", "prefill_36", "close"),
]

# 常量一致性类（**不是独立重算**）：这些键的值就是本脚本里的字面量常数 ⇒ 比对上恒真。
# 单列一栏（评审观察 1），只校验 **README 与脚本两处的字面量是否一致**，不计入「脚本可重算」。
CONST_CLAIMS = [
    ("§4.2 ε 取值", "5e-5", "eps_used"),
    ("§4.2 Γ/eg 传递项", "3e-7", "gamma_transfer"),
]


# ------------------------------------------------------------ 纯标准库线性代数
def mat_inv(M):
    """高斯-约当消元求逆（float64，部分主元）。M 为 n×n 的行主序 list。"""
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]


def max_row_abs_sum(M):
    return max(sum(abs(x) for x in row) for row in M)


def max_abs_diff(A, B):
    return max(max(abs(A[i][j] - B[i][j]) for j in range(len(A[i]))) for i in range(len(A)))


def golden_recurrence():
    """README §10 的数值验证：看 fla 的逐行递推是否等于 (I+L)⁻¹、且**不**等于 (I−L)⁻¹。

    用**固定 seed 的确定性数据集**（`random.Random(0)`，顺序固定），因此不依赖 numpy。
    ⚠ **跨环境可复现的是「判据」，不是「数值末位」**（评审观察 2）：`random.gauss` 的实现随
    Python 版本而变（实测 py3.12.13 → 1.110e-15、py3.11.6 → 3.553e-15），**两次都满足判据**。
    判据是上界式（见 CALC_CLAIMS 的 `<=`/`>=` 模式）：
      `max|golden_T − (I+L)⁻¹| ≤ 4.0e-15`（README 声明的机器精度量级上界）
      `max|golden_T − (I−L)⁻¹| ≥ 1`（判别性：O(1)）
    这两个都是随机数据下的**经验**量，所以按「界/量级」判，不按 2% 恒等判（否则换个 Python
    版本就会因末位不同而误报 FAIL）。
    返回 (err, err_wrong)。
    """
    rng = random.Random(0)
    err = 0.0
    err_wrong = 0.0
    for BT in (2, 3, 5, 8):
        D = 4
        k = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(BT)]
        b = [rng.random() * 0.9 + 0.05 for _ in range(BT)]
        g, s = [], 0.0
        for _ in range(BT):
            s += -rng.random() * 0.1
            g.append(s)
        dec = [[math.exp(g[i] - g[j]) if j <= i else 0.0 for j in range(BT)]
               for i in range(BT)]
        kb = [[k[i][d] * b[i] for d in range(D)] for i in range(BT)]
        kkT = [[sum(k[i][d] * k[j][d] for d in range(D)) for j in range(BT)] for i in range(BT)]
        # attn = −((k_β @ kᵀ) ⊙ decay)，严格下三角（golden 的 mask 连对角一起清零）
        T = [[(-sum(kb[i][d] * k[j][d] for d in range(D)) * dec[i][j]) if i > j else 0.0
              for j in range(BT)] for i in range(BT)]
        # fla 逐行递推：T[i,:i] += T[i,:i] @ T[:i,:i]
        for i in range(1, BT):
            row = T[i][:i]
            for j in range(i):
                T[i][j] = row[j] + sum(row[t] * T[t][j] for t in range(i))
        T = [[T[i][j] + (1.0 if i == j else 0.0) for j in range(BT)] for i in range(BT)]
        L = [[b[i] * kkT[i][j] * dec[i][j] if i > j else 0.0 for j in range(BT)] for i in range(BT)]
        I = [[1.0 if i == j else 0.0 for j in range(BT)] for i in range(BT)]
        IpL = [[L[i][j] + I[i][j] for j in range(BT)] for i in range(BT)]
        ImL = [[-L[i][j] + I[i][j] for j in range(BT)] for i in range(BT)]
        err = max(err, max_abs_diff(T, mat_inv(IpL)))
        err_wrong = max(err_wrong, max_abs_diff(T, mat_inv(ImL)))
    return err, err_wrong


# ------------------------------------------------------- 归档几何量的读/复算
def read_geometry(path=None):
    """从 evidence/probe_geometry.txt 取回 (A_inf, kappa_inf)；缺文件/缺键 ⇒ (None, None)。

    `path` 可覆盖（负向对照用它模拟"输入缺失"）。
    """
    path = os.path.join(EV, GEOMETRY) if path is None else path
    if not path or not os.path.exists(path):
        return None, None
    vals = {}
    for line in read_text(path).split("\n"):
        m = re.match(r"\s*(A_inf|kappa_inf)\s*=\s*([\d.eE+-]+)\s*$", line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    return vals.get("A_inf"), vals.get("kappa_inf")


def recompute_geometry():
    """从 build/*_probe.bin **原始字节**独立复算 (A_inf, kappa_inf)。

    交叉见证（cross-witness）：dump 存在时才跑；先用 dump_sha256.txt 核对 sha256，
    再按与归档同一算法复算。返回 (dict 或 None, 说明字符串)。
    """
    build = os.path.join(HERE, "build")
    files = (("m18_h48_t04097_probe.bin"), ("m18_h48_t00129_probe.bin"), ("m18_h03_t00257_probe.bin"))
    paths = [os.path.join(build, f) for f in files]
    if not all(os.path.exists(p) for p in paths):
        return None, "build/*_probe.bin 不存在（dump 不入库）⇒ 跳过交叉见证"
    BT, HE, PH, MC = 64, 16384, 2, 128
    sha = {}
    src = os.path.join(EV, "dump_sha256.txt")
    if os.path.exists(src):
        for line in read_text(src).split("\n"):
            m = re.match(r"([0-9a-f]{64})\s+(\S+)$", line.strip())
            if m:
                sha[m.group(2)] = m.group(1)
    a_inf, kappa = 0.0, 0.0
    sha_ok, sha_missing = [], []
    for f, p in zip(files, paths):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if f in sha:
            (sha_ok if sha[f] == h else sha_missing).append(f)
        else:
            sha_missing.append(f)
        arr = array.array("f")
        arr.frombytes(open(p, "rb").read())
        for ch in (0, 1, 2):
            off = (PH * MC + ch * 9 + 2) * HE
            A = [list(arr[off + i * BT: off + (i + 1) * BT]) for i in range(BT)]
            a_inf = max(a_inf, max_row_abs_sum(A))
            kappa = max(kappa, max_row_abs_sum(mat_inv(
                [[A[i][j] + (1.0 if i == j else 0.0) for j in range(BT)] for i in range(BT)])))
    note = "交叉见证：3 个 probe dump 全命中 sha256" if not sha_missing else \
           "交叉见证：sha256 未命中/缺条目 %s" % ",".join(sha_missing)
    return {"A_inf": a_inf, "kappa_inf": kappa}, note


def calc(notes, geom_path=None):
    """重算全部推导值。缺输入 ⇒ 对应键为 None（**绝不回退到已作废的 Neumann 口径**）。

    `geom_path` 可覆盖几何量归档路径（负向对照用来模拟"输入缺失"）。
    """
    c = {}
    c["gamma128"] = 128 * U_F32 / (1 - 128 * U_F32)
    c["gamma64"] = 64 * U_F32 / (1 - 64 * U_F32)
    c["delta_exp"] = DELTA_EXP
    c["c65delta"] = 65 * DELTA_EXP
    c["gamma_transfer"] = 3.0e-7
    c["u_f32"] = U_F32

    # UB 总占用：**从 .asc 的 constexpr 表达式求值**（不手抄公式）
    asc = os.path.join(HERE, "m18_gdn_prefill.asc")
    vals = {}
    if os.path.exists(asc):
        for line in read_text(asc).split("\n"):
            m = re.match(r"constexpr uint32_t (\w+) = ([^;]+);", line.strip())
            if m:
                try:
                    vals[m.group(1)] = eval(m.group(2), {}, dict(vals))
                except Exception:
                    pass
    else:
        notes.append("m18_gdn_prefill.asc 缺失 ⇒ UB 占用/偏移类断言 UNVERIFIED")
    for k, v in vals.items():
        if k.startswith("UB_"):
            c[k] = v
    c["nc_target"] = -(-4097 // 64)
    c["ub_bytes"] = vals.get("UB_TOTAL")
    c["ub_kib"] = (c["ub_bytes"] / 1024.0) if c.get("ub_bytes") else None

    # 归档 recompute：占用中位 / ≤1ulp 中位 / 余量
    log = os.path.join(EV, "check_ref_log.txt")
    if os.path.exists(log):
        occ, ulp1, relocc = [], [], []
        for line in ev_lines("check_ref_log.txt"):
            p = line.split()
            if len(p) == 8:
                try:
                    occ.append(float(p[4]))
                    ulp1.append(float(p[3].rstrip("%")))
                    relocc.append(float(p[7]))
                except ValueError:
                    pass
        if occ:
            c["occ_median"] = sorted(occ)[len(occ) // 2]
            c["margin"] = 1.0 / max(occ)
        if relocc:
            c["occ_rel_max"] = max(relocc)
        if ulp1:
            c["ulp1_median"] = sorted(ulp1)[len(ulp1) // 2]
            c["ulp1_min"] = min(ulp1)
    else:
        notes.append("check_ref_log.txt 缺失 ⇒ 占用/≤1ulp 类断言 UNVERIFIED")

    # 几何量：**归档为权威**；dump 在时用独立复算作交叉见证
    a_arch, k_arch = read_geometry(geom_path)
    if a_arch is None:
        notes.append("%s 缺失或无法解析 ⇒ ‖A‖∞/κ∞/ε 断言 UNVERIFIED" % GEOMETRY)
    c["A_inf"], c["kappa_inf"] = a_arch, k_arch
    wit, wit_note = recompute_geometry()
    notes.append(wit_note)
    if wit is not None and a_arch is not None:
        d_a = abs(wit["A_inf"] - a_arch)
        d_k = abs(wit["kappa_inf"] - k_arch)
        c["geo_witness_dA"] = d_a
        c["geo_witness_dK"] = d_k
        notes.append("交叉见证：复算 ‖A‖∞=%.9f / κ∞=%.9f，与归档差 %.3g / %.3g ⇒ %s"
                     % (wit["A_inf"], wit["kappa_inf"], d_a, d_k,
                        "一致" if (d_a <= 1e-9 and d_k <= 1e-9) else "不一致（归档需复核）"))
    elif a_arch is not None:
        notes.append("交叉见证不可得 ⇒ 仅靠 %s 归档值（该值本身已带 dump sha256 出处）" % GEOMETRY)

    # ε = κ∞·(γ₁₂₈+γ₆₄) + 65·δ_exp + Γ/eg 传递
    # **只有 κ∞ 可得时才给 ε**：不得回退到已作废的 Neumann 界口径（1/(1−‖A‖∞)，该界要求 ‖A‖∞<1，
    # 而实测 ‖A‖∞ = 2.31 > 1 不成立）。
    c["eps_sum"] = None
    c["eps_used"] = None
    if k_arch is not None:
        c["eps_sum"] = k_arch * (c["gamma128"] + c["gamma64"]) + c["c65delta"] + c["gamma_transfer"]
        c["eps_used"] = 5e-5

    # golden 递推（纯标准库、固定数据集）
    c["golden_err"], c["golden_wrong"] = golden_recurrence()

    # 派生性能量（全部由归档计时/指令模型推出，唯一假设：时钟 1.8GHz）
    tt = None
    if os.path.exists(os.path.join(EV, "run_log.txt")):
        m = re.search(r"H=48 T=4097 blk=28 : ([\d.]+) ms", read_text(os.path.join(EV, "run_log.txt")))
        tt = as_float(m.group(1)) if m else None
    if tt:
        c["per_chunk_us"] = tt * 1000.0 / 65.0
        c["instr_per_s"] = 2.0e5 / (c["per_chunk_us"] * 1e-6) / 1e9
        c["cyc_per_instr"] = (c["instr_per_s"] * 1e9) / 1.8e9 if c["instr_per_s"] else None
        c["ms_per_layer"] = tt
        c["prefill_36"] = tt * 36
    else:
        notes.append("run_log.txt 缺失/无 target 行 ⇒ §4.7 派生性能量断言 UNVERIFIED")
    return c


def part1(readme_text, c):
    """返回 (rows, stats)。每条断言的三态都记在 stats 里。"""
    rows = []
    failed, unverified = [], []
    compared = 0
    n_archive = n_calc = n_const = 0
    for desc, lit, f, pat, grp in CLAIMS:
        path = resolve(f)
        ext = is_external(f)
        if path is None:
            unverified.append((desc, f, ext))
            rows.append((desc, lit, "%s（输入缺失%s）" % (f, "，外部" if ext else ""), "-", UNVER))
            continue
        txt = read_text(path)
        m = re.search(pat, txt)
        if m is None:
            failed.append((desc, ext))
            rows.append((desc, lit, "%s（有文件但定位器无命中）" % os.path.basename(f),
                         "归档无该行", FAIL))
            continue
        val = m.group(grp)
        ln = txt[:m.start()].count("\n") + 1
        ok = (lit in readme_text) and num_close(lit, val)
        compared += 1
        n_archive += 1
        if not ok:
            failed.append((desc, ext))
        rows.append((desc, lit, "%s:%s" % (os.path.basename(f), ln),
                     "归档值=%s" % val, PASS if ok else FAIL))
    for desc, lit, key, mode in CALC_CLAIMS:
        v = c.get(key)
        if v is None:
            unverified.append((desc, "计算输入缺失（键 %s）" % key, False))
            rows.append((desc, lit, "脚本重算（%s）" % key, "不可得（输入缺失）", UNVER))
            continue
        ok, why = num_judge(lit, v, mode)
        ok = ok and (lit in readme_text)
        compared += 1
        n_calc += 1
        if not ok:
            failed.append((desc, False))
        rows.append((desc, lit, "脚本重算（%s，%s）" % (key, why),
                     "计算值=%.6g" % v if isinstance(v, float) else "计算值=%s" % v,
                     PASS if ok else FAIL))
    # 常量一致性类：这些"计算值"就是**脚本里的字面量常数** ⇒ 恒真，单列一栏（评审观察 1）。
    for desc, lit, key in CONST_CLAIMS:
        v = c.get(key)
        if v is None:
            unverified.append((desc, "常量 %s 未定义" % key, False))
            rows.append((desc, lit, "常量一致性（%s）" % key, "不可得", UNVER))
            continue
        ok = num_close(lit, v) and (lit in readme_text)
        compared += 1
        n_const += 1
        if not ok:
            failed.append((desc, False))
        rows.append((desc, lit, "常量一致性（%s；**非独立重算**，只校验 README 与脚本两处字面量一致）" % key,
                     "脚本常量=%.6g" % v if isinstance(v, float) else "脚本常量=%s" % v,
                     PASS if ok else FAIL))
    return rows, {"compared": compared, "failed": failed, "unverified": unverified,
                  "n_archive": n_archive, "n_calc": n_calc, "n_const": n_const}


TRIVIAL = re.compile(r"^(\d{1,2}|0|1\.0)$")


def part2(c):
    lines = read_text(README).split("\n")
    ev_all = {f: read_text(os.path.join(EV, f)) for f in sorted(os.listdir(EV))
              if f.endswith(".txt")} if os.path.isdir(EV) else {}
    src = {}
    for f in ("m18_gdn_prefill.asc", "check_ref.py", "CMakeLists.txt"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            src[f] = read_text(p)
    docs = {}
    d = os.path.join(ROOT, "docs")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".md"):
                docs[f] = read_text(os.path.join(d, f))
    calc_vals = [v for v in c.values() if isinstance(v, (int, float))]
    ev_nums = []
    for txt in ev_all.values():
        ev_nums += [m.group(0) for m in NUM.finditer(txt)]
    rows = []
    in_code = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        toks = [m.group(0).rstrip("%") for m in NUM.finditer(line)]
        if not toks:
            continue
        uniq, cls = [], set()
        citation = bool(re.search(r"(§|:\d+-?\d*|docs/\d|/[a-z_]+\.(py|h|cpp|md)|m\d+_)", line))
        for t in toks:
            if t in uniq:
                continue
            uniq.append(t)
            if num_match_any(t, calc_vals):
                cls.add("✅推导")
            elif any(t in v for v in ev_all.values()) or num_match_any(t, ev_nums):
                cls.add("✅归档")
            elif re.search(r"假设", line) and not TRIVIAL.match(t):
                cls.add("🔵假设")
            elif TRIVIAL.match(t) or (citation and len(t) <= 4):
                cls.add("🔢编号")
            elif any(t in v for v in src.values()):
                cls.add("✅常量")
            elif any(t in v for v in docs.values()):
                cls.add("📚外部")
            elif re.search(r"(ms|heads/ms|计时|活值)", line):
                cls.add("🔁活值")
            else:
                cls.add("⚠待裁定")
        rows.append((i, " ".join(uniq[:12]) + (" …" if len(uniq) > 12 else ""),
                     "/".join(sorted(cls)), len(uniq)))
    return rows


def status_of(stats):
    """三态判定：返回 (rc, label)。"""
    failed = stats["failed"]
    unver_local = [u for u in stats["unverified"] if not u[2]]
    if failed:
        return 1, "FAILED"
    if stats["compared"] == 0:
        return 2, "SKIPPED"
    if unver_local:
        return 2, "NOT CERTIFIED"
    return 0, "OK"


def certificate_line(stats):
    """合格证/结论行：**必须带实际比较计数**，且外部覆盖范围与**三类计数**都要显式打印。"""
    total = len(CLAIMS) + len(CALC_CLAIMS) + len(CONST_CLAIMS)
    unver = stats["unverified"]
    unver_local = [u for u in unver if not u[2]]
    unver_ext = [u for u in unver if u[2]]
    rc, label = status_of(stats)
    if rc == 0:
        head = "RESULT: OK"
    elif rc == 1:
        head = "RESULT: FAILED"
    elif stats["compared"] == 0:
        head = "RESULT: SKIPPED"
    else:
        head = "RESULT: NOT CERTIFIED"
    body = ("compared %d/%d claims (archive %d / recomputed %d / constants %d), %d failed, "
            "%d unverified (local), %d unverified (external)"
            % (stats["compared"], total, stats.get("n_archive", 0), stats.get("n_calc", 0),
               stats.get("n_const", 0), len(stats["failed"]), len(unver_local), len(unver_ext)))
    out = ["%s (%s)" % (head, body)]
    if rc != 0:
        for d, why, ext in unver:
            out.append("  [unverified%s] %s ← %s" % ("/external" if ext else "", d, why))
        for d, ext in stats["failed"]:
            out.append("  [failed%s] %s" % ("/external" if ext else "", d))
    return out, rc


def write_md(path, rows1, rows2, stats, notes, banner):
    total = len(CLAIMS) + len(CALC_CLAIMS) + len(CONST_CLAIMS)
    n_ev = sum(1 for r in rows2 if "✅归档" in r[2]) if rows2 else 0
    n_triv = sum(1 for r in rows2 if r[2] == "🔢编号") if rows2 else 0
    warn = [r for r in rows2 if "⚠待裁定" in r[2]] if rows2 else []
    out = [
        "# M34 README 数字对账表（`audit_readme_numbers.py` 生成，可复算）",
        "",
        banner,
        "",
        "口径：README 里每个数字都要能 grep 回出处。分类定义见脚本 docstring。",
        "**PART 1 每条都把归档真值取回并与 README 引用值做数值比对**（`close` 档容差 2%；",
        "`<=`/`>=` 档按界判），不是「同现即通过」。**退出码三态 0/1/2** 见脚本 docstring。",
        "",
        "## 统计",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| README 行数 | %d |" % len(read_text(README).split("\n")),
        "| **PART 1 断言条数** | **%d** |" % total,
        "| ├ 归档取回类 | %d |" % stats.get("n_archive", 0),
        "| ├ **真重算类**（脚本独立算出） | **%d** |" % stats.get("n_calc", 0),
        "| └ 常量一致性类（**非独立重算**，只校验 README 与脚本两处字面量一致） | %d |"
        % stats.get("n_const", 0),
        "| **其中真正比较过（compared）** | **%d** |" % stats["compared"],
        "| **失败（failed）** | **%d** |" % len(stats["failed"]),
        "| **没得比（unverified，本仓输入缺失）** | **%d** |" % len(
            [u for u in stats["unverified"] if not u[2]]),
        "| **没得比（unverified，外部输入缺失）** | **%d** |" % len(
            [u for u in stats["unverified"] if u[2]]),
        "| PART 2 含数字的行数 | %d |" % len(rows2),
        "| 其中含 ✅归档 的行 | %d |" % n_ev,
        "| 其中纯编号行（🔢编号） | %d |" % n_triv,
        "| **PART 3 待裁定行数** | **%d** |" % len(warn),
        "",
        "## 运行备注（输入可得性 / 交叉见证）",
        "",
    ]
    out += ["- %s" % n for n in notes]
    out += ["", "## PART 1：断言（每条都从归档/源文件取回真值 + 数值比对）", "",
            "| # | 说明 | README 引用 | 出处 | 取回值 | 结论 |", "|---|---|---|---|---|---|"]
    for n, (d, lit, s, v, ok) in enumerate(rows1, 1):
        out.append("| %d | %s | `%s` | %s | %s | %s |" % (n, d, lit, s, v, ok))
    if rows2:
        out += ["", "## PART 2：逐行数字对账", "",
                "| README 行 | 数字（去重，最多 12 个） | 分类 | 计数 |", "|---|---|---|---|"]
        for ln, toks, cls, n in rows2:
            out.append("| %d | %s | %s | %d |" % (ln, toks, cls, n))
        out += ["", "## PART 3：无独立出处（必须人工裁定）", ""]
        if warn:
            out += ["| README 行 | 数字 | 分类 |", "|---|---|---|"]
            for ln, toks, cls, _ in warn:
                out.append("| %d | %s | %s |" % (ln, toks, cls))
        else:
            out.append("（无）")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def banner_for(rc, stats):
    total = len(CLAIMS) + len(CALC_CLAIMS) + len(CONST_CLAIMS)
    body = ("compared %d/%d（archive %d / recomputed %d / constants %d），failed %d，"
            "unverified(local) %d，unverified(external) %d"
            % (stats["compared"], total, stats.get("n_archive", 0), stats.get("n_calc", 0),
               stats.get("n_const", 0), len(stats["failed"]),
               len([u for u in stats["unverified"] if not u[2]]),
               len([u for u in stats["unverified"] if u[2]])))
    if rc == 0:
        return "> **状态：OK —— 本文件是合格证**（%s）。" % body
    return ("> **状态：%s —— 本文件不是合格证**（%s）。缺输入/失败的条目见下表 `UNVERIFIED`/`FAIL` 行。"
            % (status_of(stats)[1], body))


def negative_control():
    """两组负向对照（判据自己要会咬）：

    **A 扰动**：把一条**输入在仓内**的断言在 README 文本里改错，要求脚本报 rc=1
      并**点名该判定项**（且只点名它）。
    **B 缺输入**：强制几何量归档缺失（模拟干净 checkout 曾出现的情形），要求脚本报 rc=2
      **且**把 ‖A‖∞/κ∞/ε 四条断言标成 UNVERIFIED —— **不得静默回退到已作废的 Neumann ε**。

    退出码：0=两组对照都有效；1=脚本没咬住；2=无法构造对照。
    """
    base = read_text(README)
    target_desc, target_lit, wrong = "§4.7 gqa3 run1", "4640.499", "9999.999"
    if target_lit not in base:
        print("NEGATIVE-CONTROL: 无法构造对照 —— README 里找不到 %r" % target_lit)
        return 2, [], {"compared": 0, "failed": [], "unverified": []}, target_lit, wrong

    # ---- A：扰动（替换**全部**出现处，否则残留的那处会让断言仍 PASS）
    pert = base.replace(target_lit, wrong)
    c = calc([])
    rows1, stats = part1(pert, c)
    rc_a, _ = status_of(stats)
    named = [d for d, _ in stats["failed"]]
    ok_a = (rc_a == 1) and (named == [target_desc])
    print("NEGATIVE-CONTROL A（扰动）: %s（%s → %s）" % (target_desc, target_lit, wrong))
    print("  rc=%d（期望 1）；点名失败项=%s（期望恰为 [%s]）⇒ %s"
          % (rc_a, named, target_desc, "对照有效" if ok_a else "对照失败：脚本没咬住扰动"))

    # ---- B：缺输入（几何量归档缺失）
    c_b = calc([], geom_path=os.path.join(EV, "__no_such_geometry__.txt"))
    rows_b, stats_b = part1(base, c_b)
    rc_b, _ = status_of(stats_b)
    unver_b = [d for d, _why, _ext in stats_b["unverified"]]
    want_unver = ["§4.2 ε 合计", "§4.2 κ∞ = ‖(I+A)⁻¹‖∞", "§4.2 ε 取值", "§4.2 ‖A‖∞（说明 Neumann 界不适用）"]
    eps_row = dict((d, (v, st)) for d, _lit, _s, v, st in rows_b)
    ok_b = (rc_b == 2) and all(u in unver_b for u in want_unver) \
        and eps_row["§4.2 ε 合计"][1] == UNVER
    print("NEGATIVE-CONTROL B（缺输入）: 强制 %s 缺失" % GEOMETRY)
    print("  rc=%d（期望 2，即 NOT CERTIFIED）；ε 合计行=%s（期望 UNVERIFIED，不是 PASS）⇒ %s"
          % (rc_b, eps_row["§4.2 ε 合计"], "对照有效" if ok_b else "对照失败：缺输入仍发合格证/静默回退"))
    print("  未发合格证文案：%s" % ("是" if rc_b != 0 else "否（不合格！）"))

    ok = ok_a and ok_b
    return (0 if ok else 1), rows1, stats, target_lit, wrong


def main():
    if "--negative-control" in sys.argv[1:]:
        rc, rows1, stats, target_lit, wrong = negative_control()
        if rows1:
            write_md(
                os.path.join(EV, "readme_number_audit_negative_control.md"), rows1, [], stats,
                ["负向对照：README 文本被扰动（%s → %s），期望 rc=1 并点名该判定项。"
                 % (target_lit, wrong)],
                "> **负向对照产物（不是合格证）**：故意扰动一条断言，用于验证脚本会 FAIL 并点名。")
        return rc

    # 前置输入检查：**没有可比对象就不发合格证**（也不崩）
    if not os.path.exists(README):
        print("RESULT: SKIPPED (README.md 缺失 ⇒ 无可比较对象；本脚本不发合格证)")
        return 2
    if not os.path.isdir(EV):
        print("RESULT: SKIPPED (evidence/ 缺失 ⇒ 无可比较的归档输入；本脚本不发合格证)")
        return 2

    notes = []
    # --no-geometry：强制几何量归档缺失（可复现的「输入缺失」读数；产物另写，**不覆盖合格证**）
    force_no_geom = "--no-geometry" in sys.argv[1:]
    geom_path = os.path.join(EV, "__forced_missing_geometry__") if force_no_geom else None
    c = calc(notes, geom_path=geom_path)
    readme_text = read_text(README)
    rows1, stats = part1(readme_text, c)
    rows2 = part2(c)
    warn = [r for r in rows2 if "⚠待裁定" in r[2]]
    rc, label = status_of(stats)
    lines, rc = certificate_line(stats)
    dest = os.path.join(EV, "readme_number_audit_no_geometry.md" if force_no_geom
                        else "readme_number_audit.md")
    if force_no_geom:
        notes = ["**`--no-geometry`**：强制几何量归档缺失（负向对照 · 缺输入读数），产物不覆盖正式合格证。"] + notes
    write_md(dest, rows1, rows2, stats, notes, banner_for(rc, stats))
    for n in notes:
        print("  [note] %s" % n)
    print("wrote %s" % dest)
    for l in lines:
        print(l)
    print("PART1 断言 %d 条（归档 %d / 真重算 %d / 常量一致性 %d）/ 比较 %d / 失败 %d / 没得比 %d；"
          "PART2 %d 行；PART3 待裁定 %d 行"
          % (len(CLAIMS) + len(CALC_CLAIMS) + len(CONST_CLAIMS),
             stats.get("n_archive", 0), stats.get("n_calc", 0), stats.get("n_const", 0),
             stats["compared"], len(stats["failed"]),
             len(stats["unverified"]), len(rows2), len(warn)))
    if warn:
        print("待裁定行号：%s" % ", ".join(str(r[0]) for r in warn))
    # PART 3 有未裁定项同样不许发合格证
    if rc == 0 and warn:
        print("RESULT: NOT CERTIFIED (PART 3 待裁定 %d 行 ≥1 ⇒ 需人工裁定)" % len(warn))
        rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
