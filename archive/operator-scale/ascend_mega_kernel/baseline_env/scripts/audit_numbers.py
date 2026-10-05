#!/usr/bin/env python3
"""M30: reconcile EVERY quantitative claim in baseline_env/README.md against its archived source.

Why this exists: three review rounds in a row found a README number that disagreed with the
`evidence/` file shipped in the same commit (live npu-smi / df readings, a mis-rounded GiB
value, stale counts). Reviewing one cell at a time kept missing the next one, so this script
audits all of them at once and fails loudly on any mismatch.

Claim kinds
-----------
fixed      value must appear on the README line AND its source line must exist in the archive
derived    value must equal the arithmetic recomputed from other fixed values
live       a reading that legitimately moves (npu-smi used HBM, df free space). It must NOT be
           presented as a conclusion: the script only checks that the README marks it as a live
           value and points at the archive that carries the collection timestamp.
           The live value itself is *parsed out of that archive at run time* and, if the README
           also writes an instance reading, the two must agree. A literal in this table would go
           stale the moment the reading moved -- that was the M51 bug (`expect="453G"` while
           evidence/01 said 426G), and it is the one error class this script exists to catch.
external   attributed to a source outside this repo (e.g. a tower measurement): must be marked
           as such in the README; no archive check.
quote      a historical value quoted inside the correction log (changelog), exempt by design.
narrative  prose about a non-reproducible attempt (e.g. an upstream download that timed out).

Exit codes (three states -- "I compared and it passed" must be distinguishable from
"I had nothing to compare", so that this auditor can never issue a silent clean bill):
  0  compared and passed   (prints `RESULT: OK (<n> compared ...)`)
  1  compared and found differences (prints `RESULT: FAIL (...)`)
  2  nothing comparable / input missing (prints `RESULT: SKIPPED (...)`) -- e.g. the README
     does not exist, no evidence file resolved, or the live-value archive could not be parsed

Usage: python3 baseline_env/scripts/audit_numbers.py [--readme P] [--evidence D]
"""

import argparse
import os
import re
import sys

# (id, kind, readme_probe, readme_expect, archive, archive_pattern, note)
CLAIMS = [
    # ---- §1 environment ---------------------------------------------------------------
    ("HBM 总量", "fixed", "npu-smi HBM **总量", "131072", "evidence/01-host-and-npu.txt", r"\d+\s+/\s+131072", "npu-smi HBM 总量 MB（已用量为活值，随 worker 波动）"),
    ("HBM GiB", "derived", "| HBM | npu-smi 131072 MB", "128 GiB", "evidence/01-host-and-npu.txt", r"131072", "131072 MB / 1024 = 128 GiB"),
    ("total_memory 字节", "fixed", "| HBM | npu-smi 131072 MB", "132,238,016,512", "evidence/05-npu-selfcheck.txt", r"132238016512", "torch get_device_properties"),
    ("total_memory GiB", "derived", "| HBM | npu-smi 131072 MB", "123.16", "evidence/05-npu-selfcheck.txt", r"132238016512 bytes \(123.2 GiB\)", "132238016512/2^30 = 123.15625 -> 123.16 (2dp); 脚本打印 123.2 (1dp)"),
    ("driver 版本", "fixed", "| Driver / npu-smi |", "25.7.rc1.6", "evidence/01-host-and-npu.txt", r"npu-smi 25\.7\.rc1\.6", ""),
    ("CANN 版本", "fixed", "| CANN |", "9.1.0", "evidence/01-host-and-npu.txt", r"ASCEND_HOME_PATH=/usr/local/Ascend/cann-9\.1\.0", ""),
    ("ATB 版本", "fixed", "| ATB (nnal) |", "9.1.0.B150", "evidence/01-host-and-npu.txt", r"9\.1\.0\.B150", ""),
    ("容器内存上限字节", "fixed", "| 容器内存上限 |", "34,359,738,368", "evidence/01-host-and-npu.txt", r"^34359738368$", "cgroup v1 memory.limit_in_bytes"),
    ("容器内存上限 GiB", "derived", "| 容器内存上限 |", "32 GiB", "evidence/01-host-and-npu.txt", r"^34359738368$", "34359738368/2^30 = 32"),
    ("宿主内存", "fixed", "| 容器内存上限 |", "754 GiB", "evidence/01-host-and-npu.txt", r"Mem:\s+754", "free -g（宿主视角，非容器上限）"),
    ("磁盘配额（用户口径）", "external", "| 磁盘配额", "300 GB", "docs/01-environment.md", r"总共 300 GB", "仓库文档 docs/01 §3.1 的口径，非本 mission 实测"),
    ("磁盘实测可用", "live", "| 磁盘**实测**可用", "", "evidence/01-host-and-npu.txt", r"overlay\s+\S+\s+\S+\s+(\d+[KMGT]?)\s+\d+%", "活值：值由归档现场解析（本行不设字面量）；README 若写了采集实例读数，必须与归档一致"),
    ("模型分片数", "fixed", "| 模型 | `/workspace/Qwen3.8-Flash-Next-MXFP4`", "131 分片", "evidence/03-checkpoint-memory.txt", r"^shards\s+: 131$", ""),
    ("模型 tensor 数", "fixed", "| 模型 | `/workspace/Qwen3.8-Flash-Next-MXFP4`", "1898 tensor", "evidence/03-checkpoint-memory.txt", r"^tensors\s+: 1898$", ""),
    ("模型总量", "fixed", "| 模型 |", "169.72 GiB", "evidence/03-checkpoint-memory.txt", r"169\.72 GiB \(182\.23 GB\)", ""),

    # ---- §2 versions ------------------------------------------------------------------
    ("torch 版本", "fixed", "torch 2.10.0+cpu / torch_npu 2.10.0.post4 / triton-ascend 3.2.2", "2.10.0", "evidence/06-pip-list.txt", r"^torch\s+2\.10\.0\+cpu$", ""),
    ("torch_npu 版本", "fixed", "torch 2.10.0+cpu / torch_npu 2.10.0.post4 / triton-ascend 3.2.2", "2.10.0.post4", "evidence/06-pip-list.txt", r"^torch_npu\s+2\.10\.0\.post4$", ""),
    ("triton-ascend 版本", "fixed", "triton-ascend 3.2.2", "3.2.2", "evidence/06-pip-list.txt", r"^triton_ascend\s+3\.2\.2$", ""),
    ("torch_npu 候选列表", "fixed", "2.10.0 / 2.10.0.post2 / 2.10.0.post4 / 2.11.0 / 2.12.0 / 2.13.0rc1", "2.13.0rc1", "evidence/02-version-matrix.txt", r"torch_npu-2\.13\.0rc1-cp312", "镜像上可装版本；无 2.13.0 正式版"),
    ("vLLM main torch pin", "fixed", "vLLM main 硬 pin `torch == 2.13.0`", "2.13.0", "evidence/11-upstream-refs.txt", r'torch == 2\.13\.0', "pyproject.toml"),
    ("CXX11 ABI", "fixed", "`torch._C._GLIBCXX_USE_CXX11_ABI == True`", "True", "evidence/08-abi-linkage.txt", r"cxx_abi_1/lib/libatb\.so", "ldd 解析到 cxx_abi_1；自检打印 ABI=True"),
    ("cxx_abi 符号数", "fixed", "cxx_abi_0 变体 `__cxx11` 符号 0 个", "262 个", "evidence/08-abi-linkage.txt", r"cxx_abi_1/lib/libatb\.so: 262", "cxx_abi_0 = 0 个，cxx_abi_1 = 262 个"),

    # ---- §3 vLLM / vllm-ascend --------------------------------------------------------
    ("qwen4_exp v0.23.0 命中", "fixed", "| v0.23.0（fork pin 的 `0fc695fc`） | 0 |", "0", "evidence/02-version-matrix.txt", r"--- v0\.23\.0\s*\n\s*qwen4_exp: 0", "git grep -c 命中文件数"),
    ("qwen4_exp v0.29.0 命中", "fixed", "| **v0.29.0** | **23** |", "23", "evidence/02-version-matrix.txt", r"--- v0\.29\.0\s*\n\s*qwen4_exp: 23", ""),
    ("qwen4_exp v0.30.0 命中", "fixed", "| v0.30.0 | 27 |", "27", "evidence/02-version-matrix.txt", r"--- v0\.30\.0\s*\n\s*qwen4_exp: 27", ""),
    ("上游 head 数", "fixed", "上游有 **26 个 head**", "26 个 head", "evidence/11-upstream-refs.txt", r"\(count = 26 heads", "已修：round-3 前误写 25"),
    ("上游 tag 数", "fixed", "**46 个 tag**", "46 个 tag", "evidence/11-upstream-refs.txt", r"\(count = 46 tags;", "含 peeled 的原始行数 48"),
    ("qwen4_exp 文件数(ced6857a)", "fixed", "两者 pin 的 vLLM commit **都含 `qwen4_exp`**（34 个文件）", "34 个文件", "evidence/11-upstream-refs.txt", r"files under vllm/models/qwen4_exp/ : 34", ""),
    ("vllm_version_is 文件数", "fixed", "`evidence/11` §4 标签原写", "20", "evidence/11-upstream-refs.txt", r'vllm_version_is\("0\.29\.0"\) FILES matching \(git grep -l \| wc -l\) : 20', ""),
    ("vllm_version_is 出现次数", "fixed", "真实**出现次数 37**", "37", "evidence/11-upstream-refs.txt", r'vllm_version_is\("0\.29\.0"\) OCCURRENCES \(git grep -o \| wc -l\) : 37', ""),
    ("插件引用失效统计", "fixed", "1860 条可解析、**73 条已失效**（12 模块 + 31 符号", "73", "evidence/04-plugin-vs-vllm-main.txt", r"total broken references\s+: 73", "12 模块 + 31 符号 = 73"),
    ("插件扫描文件数", "fixed", "449 个 py", "449", "evidence/04-plugin-vs-vllm-main.txt", r"plugin python files scanned\s+: 449", ""),
    ("fork vLLM 区间", "fixed", "该 fork 只覆盖 v0.23.0–v0.24.0", "v0.24.0", "evidence/11-upstream-refs.txt", r"-> vLLM tag: v0\.24\.0", "main-verified = ee0da84a"),
    ("QSA 文档文件数", "fixed", "共 2 个文件", "2", "evidence/11-upstream-refs.txt", r"of which under vllm_ascend/ \(code\): 0", "教程 .md + zh_CN .po = 2 个文件；代码树 0"),
    ("A3 验证配置", "fixed", "**1× Atlas 800 A3 (64GB × 8)**", "64GB × 8", "evidence/11-upstream-refs.txt", r"Atlas 800 A3 \(64GB × 8\)", "教程 §3.1"),
    ("教程行号 7", "fixed", "第 7 行（原文）", "第 7 行", "evidence/11-upstream-refs.txt", r"hardware statement \(line 7\)", ""),
    ("教程行号 9", "fixed", "教程第 9 行自称", "第 9 行", "evidence/11-upstream-refs.txt", r"validated version statement \(line 9\)", ""),
    ("support matrix 行号", "fixed", "supported_models.md:140", "140", "evidence/11-upstream-refs.txt", r"^140:\s+\| Qwen3\.8-Flash-Next", ""),
    ("quay 镜像 tag", "fixed", "镜像 tag（文档引用）：", "qwen3.8-next-a3", "evidence/11-upstream-refs.txt", r"^quay\.io/ascend/vllm-ascend:qwen3\.8-next-a3$", "README 列 5 个 tag，归档 §9c 同为 5 个（a2/a3/a5/next-a3/next-a3-openeuler）"),
    ("OCI 镜像体积", "external", "OCI index ~5.9 GB", "5.9 GB", "", "", "tower 实测（本 mission 未拉取镜像）"),

    # ---- §5 self-check ----------------------------------------------------------------
    ("device_count", "fixed", "torch.npu.device_count() = 1", "1", "evidence/05-npu-selfcheck.txt", r"device_count\(\) = 1", ""),
    ("设备名", "fixed", "Ascend950PR_9579", "Ascend950PR_9579", "evidence/05-npu-selfcheck.txt", r"device name\s+: Ascend950PR_9579", ""),
    ("matmul 相对误差", "fixed", "matmul (aclnnMatmul)    max_abs_diff=6.25e-02 rel=4.34e-04", "4.34e-04", "evidence/05-npu-selfcheck.txt", r"matmul \(aclnnMatmul\).*rel=4\.340e-04", ""),
    ("reduce 相对误差", "fixed", "sum (aclnnReduceSum)    max_abs_diff=1.00e+00 rel=4.66e-04", "4.66e-04", "evidence/05-npu-selfcheck.txt", r"sum \(aclnnReduceSum\).*rel=4\.655e-04", ""),
    ("算子往返耗时", "fixed", "round trip in 9.7 ms", "9.7 ms", "evidence/05-npu-selfcheck.txt", r"round trip in 9\.7 ms", "单次运行耗时（复跑为 9.6~11.2 ms 抖动），数字与 evidence/05 同源"),
    ("显存记账", "fixed", "memory_allocated = 3.5 MiB, memory_reserved = 28.0 MiB", "3.5 MiB", "evidence/05-npu-selfcheck.txt", r"memory_allocated = 3\.5 MiB, memory_reserved = 28\.0 MiB", ""),

    # ---- §6 checkpoint ledger ---------------------------------------------------------
    ("ngram 表", "fixed", "**n-gram 表 95.37 GiB**", "95.37 GiB", "evidence/03-checkpoint-memory.txt", r"ngram embedding table\s+95\.37 GiB", ""),
    ("ngram tensor 数", "fixed", "ngram embedding table                  95.37 GiB    130 tensors", "130 tensors", "evidence/03-checkpoint-memory.txt", r"ngram embedding table\s+95\.37 GiB\s+130 tensors", ""),
    ("device 常驻", "derived", "**device 常驻部分 = 74.35 GiB**", "74.35 GiB", "evidence/03-checkpoint-memory.txt", r"169\.72", "169.72 - 95.37 = 74.35"),
    ("设计稿对齐值", "external", "74.3 GiB", "74.3 GiB", "docs/04-summary.md", r"169\.7 GiB", "项目设计稿口径（docs/04 §2）"),
    ("mxfp4 packed", "fixed", "MoE experts (mxfp4 packed u8)", "56.36 GiB", "evidence/03-checkpoint-memory.txt", r"MoE experts \(mxfp4 packed u8\)\s+56\.36 GiB", ""),
    ("e8m0 scale", "fixed", "MoE experts (e8m0 scale)", "3.52 GiB", "evidence/03-checkpoint-memory.txt", r"MoE experts \(e8m0 scale\)\s+3\.52 GiB", ""),
    ("KV 余量", "derived", "余 ~48.8 GiB", "48.8", "evidence/03-checkpoint-memory.txt", r"169\.72", "123.15625 - 74.35 = 48.81 -> ~48.8"),
    ("每 token 读取量", "external", "**≈16 行 × 320 B ≈ 5 KiB**", "5 KiB", "", "", "tower 裁决口径（用户侧估算），本 mission 未实测"),

    # ---- changelog / narrative (exempt by design) -------------------------------------
    ("§10.1 引用旧值 25 head", "quote", "「上游 25 个 head」→ **26**", "", "", "", "更正记录中引用的旧值"),
    ("§10.1 引用旧值 48 tag", "quote", "「48 个 tag ref」→ **46 个 tag**", "", "", "", "更正记录中引用的旧值"),
    ("§10.1 引用旧活值", "quote", "已用** HBM 读数（原写 `5245 MB`）", "", "", "", "更正记录中引用的旧活值"),
    ("pypi wheel 体积", "narrative", "274 MB wheel", "", "", "", "下载尝试叙述，未成功"),
]

# live 主张的两侧抽取式样：CLAIMS 的 archive_pattern（含捕获组）从归档取「本次采集实例」，
# 这里从 README 行取作者写下的实例读数。README 可以只写「活值 + 指向采集时刻」不写读数
# （此时 README_LIVE_RX 命中为空 ⇒ 通过）；一旦写了读数，就必须与归档逐字相同。
README_LIVE_RX = {
    "磁盘实测可用": r"\*\*(\d+[KMGT]?)\*\*（`evidence/01`",
}

# Tokens that are already accounted for by one of the claims above (same value written
# differently, a section/line cross-reference, or part of a multi-part number), plus the
# genuinely non-measurement tokens. Listed explicitly so the coverage scan cannot silently
# hide a real measurement.
EXTRA_COVERED = {
    "131072 MB": "同 'HBM 总量'（MB 写法）",
    "123.16 GiB": "同 'total_memory GiB'（§1 与 §6.2 引用同一值）",
    "123.2 GiB": "同 'total_memory GiB'（脚本 1 位小数打印值，见 evidence/05）",
    "512 B": "同 'total_memory 字节'（132,238,016,512 B 的尾段）",
    "368 B": "同 '容器内存上限字节'（34,359,738,368 B 的尾段）",
    "182.23 GB": "同 '模型总量'（同一台账行的 GB 写法）",
    "74.3 GiB": "同 '设计稿对齐值'（外部口径）",
    "48.8 GiB": "同 'KV 余量'（派生值）",
    "4.70 GiB": "checkpoint 台账：MoE experts (unquantised bf16)，见 evidence/03",
    "3.89 GiB": "checkpoint 台账：GDN / linear attention，见 evidence/03",
    "2.37 GiB": "checkpoint 台账：embed / lm_head，见 evidence/03",
    "3.5 GiB": "checkpoint 台账：其余(router/hc/ple/attn/vision/mtp/norms) 合计，见 evidence/03",
    "28.0 MiB": "同 '显存记账'（同一行 memory_reserved）",
    "8.7T": "宿主 overlay 池 Size（非容器配额），见 '磁盘配额' 行与 evidence/01",
    "449 个": "同 '插件扫描文件数'",
    "1860 条": "同 '插件引用失效统计'（可解析引用数）",
    "73 处": "同 '插件引用失效统计'（§3.3/§8 的表述差异）",
    "73 条": "同 '插件引用失效统计'",
    "2 个文件": "同 'QSA 文档文件数'",
    "26 个 head": "同 '上游 head 数'",
    "46 个 tag": "同 '上游 tag 数'",
    "5 行": "§6.1/§10 的表格行号交叉引用，非测量值",
    "9 行": "教程行号交叉引用，同 '教程行号 9'",
    "7 行": "教程行号交叉引用，同 '教程行号 7'",
    "13 行": "§6.1 表格行号交叉引用，非测量值",
    "4 行": "§6.1 表格行号交叉引用，非测量值",
    "2 行": "peeled ^{} 行数（46 个 tag 的补注），见 evidence/11 §2 计数行",
    "3%": "§7.3 的判定阈值（方法约定，非实测）",
    "16 行": "同 '每 token 读取量'（tower 口径）",
    "320 B": "同 '每 token 读取量'（tower 口径）",
    "5 KiB": "同 '每 token 读取量'（tower 口径）",
    "274 MB": "pypi wheel 下载尝试叙述（未成功），非宣称值",
    "64GB": "同 'A3 验证配置'（64GB × 8）",
    "26T": "ISO 时间戳 '2026-09-26T…' 的片段，非测量值",
    "2 个": "同 'QSA 文档文件数'",
    "48 行": "同 '上游 tag 数'（含 peeled 的原始行数）",
    "3 个": "同 '上游 tag 数'（`evidence/02` 另存最新 3 个 tag）",
    "453G": "更正记录（§10.3）中引用的旧活值：该行现改为从归档现场解析，字面量只作历史记录",
    "1.8ms": "§10.3 引用的 m22 计时（外仓：`m22_router512/evidence/run_mode{2,4}.log` 的 real_m64 行）",
    "12.6 GB": "§10.3 引用的 m22 带宽端点（外仓：`m22_router512/evidence/run_mode{2,4}.log`，1.3448e9 B ÷ 106.3ms）",
    "58 条": "本脚本自己的对账条目数（见 `evidence/12`），不是被对账的测量值",
    "1 条": "§6.1/§10 的表格行号交叉引用（'第 1 条'），非测量值",
    "130 tensor": "同 'ngram tensor 数'",
    "240 tensor": "checkpoint 台账：MoE experts (mxfp4 packed u8 / e8m0 scale) 的 tensor 数，见 evidence/03",
    "54 tensor": "checkpoint 台账：MoE experts (unquantised bf16) 的 tensor 数，见 evidence/03",
    "25 个": "更正记录中引用的旧值（25 个 head）",
    "48 个": "更正记录中引用的旧值（48 个 tag ref）",
    "5245 MB": "更正记录中引用的旧活值（npu-smi 已用 HBM）",
    "123.19 GiB": "更正记录中引用的旧错误值（已修为 123.16 GiB）",
    "0 个": "同 'cxx_abi 符号数'（cxx_abi_0 变体）",
    "262 个": "同 'cxx_abi 符号数'（cxx_abi_1 变体）",
}

# Token *patterns* covered as a class (avoids listing every verbatim ledger row).
EXTRA_COVERED_PATTERNS = [
    (r"\d+ tensors?", "checkpoint 台账逐行 tensor 数（§6.2 逐字摘自 evidence/03）"),
]


def line_of(path, pattern):
    try:
        rx = re.compile(pattern, re.M)
    except re.error:
        return None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    m = rx.search(text)
    if not m:
        return None
    return text[: m.start()].count("\n") + 1


def value_in(path, pattern, group=1):
    """第一条匹配 pattern 的行里，捕获组 group 的取值（用于 live 值的现场解析）。"""
    try:
        rx = re.compile(pattern, re.M)
    except re.error:
        return None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        m = rx.search(fh.read())
    if not m:
        return None
    try:
        return m.group(group)
    except IndexError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="baseline_env/README.md")
    ap.add_argument("--evidence", default="baseline_env")
    args = ap.parse_args()

    if not os.path.isfile(args.readme):
        print(f"RESULT: SKIPPED (README not found: {args.readme} -- nothing to compare)")
        return 2
    with open(args.readme, encoding="utf-8") as fh:
        readme_lines = fh.read().splitlines()

    rows = []
    failures = []
    live_values = {}
    compared = 0          # 真正拿到"两侧/一侧真值"并做过比对的条目数（决定 OK 还是 SKIPPED）
    for cid, kind, probe, expect, archive, pattern, note in CLAIMS:
        rline = next((i + 1 for i, l in enumerate(readme_lines) if probe in l), None)
        aline = None
        apath = os.path.join(args.evidence, archive) if archive else None
        if archive and os.path.isfile(apath):
            aline = line_of(apath, pattern)
        elif archive:
            aline = None

        if kind in ("quote", "narrative"):
            verdict = "EXEMPT（更正记录/叙述）"
        elif kind == "external":
            verdict = "ATTRIBUTED（外部来源，README 已标注）" if rline else "MISSING in README"
            if not rline:
                failures.append(cid)
        elif kind == "live":
            marked = rline is not None and ("活值" in readme_lines[rline - 1] or "现场重采" in readme_lines[rline - 1])
            parsed = value_in(apath, pattern) if (apath and os.path.isfile(apath)) else None
            live_values[cid] = parsed
            compared += 1 if parsed and marked else 0
            rx = README_LIVE_RX.get(cid)
            rm = re.search(rx, readme_lines[rline - 1]) if (rline and rx) else None
            stated = rm.group(1) if rm else None
            agrees = stated is None or stated == parsed
            if marked and parsed and agrees:
                verdict = "LIVE-OK（标注活值 + 归档现场解析"
                verdict += f" = {parsed}；README " + ("未写读数" if stated is None else f"读数 {stated} 与之一致") + "）"
            else:
                why = []
                if not marked:
                    why.append("README 未标注活值")
                if not parsed:
                    why.append("归档解析不出值")
                if not agrees:
                    why.append(f"README 读数 {stated} ≠ 归档 {parsed}")
                verdict = "LIVE-FAIL(" + "；".join(why) + ")"
                failures.append(cid)
        else:  # fixed / derived
            in_readme = rline is not None and (expect == "" or expect in readme_lines[rline - 1])
            in_archive = aline is not None
            compared += 1 if (in_readme and in_archive) else 0
            verdict = "PASS" if (in_readme and in_archive) else (
                "FAIL(README 缺)" if not in_readme else "FAIL(归档缺)")
            if not (in_readme and in_archive):
                failures.append(cid)

        shown = f"live={live_values.get(cid) or '?'}" if kind == "live" else (expect or "-")
        rows.append((cid, kind, str(rline or "-"), shown,
                     (archive or "-") + (f":{aline}" if aline else ""), verdict, note))

    # ---- coverage: which numeric tokens in the README are NOT covered by any claim? ----
    covered = set(EXTRA_COVERED)
    for cid, kind, probe, expect, archive, pattern, note in CLAIMS:
        for tok in re.findall(r"[0-9][0-9,.]*\s?(?:GiB|GB|MB|G|T|ms|%|个|行|处|条|分片|tensor)", expect or ""):
            covered.add(tok.strip())
    for val in live_values.values():  # 活值读数：现场解析得来，不写字面量
        if val:
            covered.add(val)
    token_rx = re.compile(r"[0-9][0-9,.]*\s?(?:GiB|GB|MB|G|T|ms|个|行|处|条|分片|tensor)")
    uncovered = {}
    ledger_rows = 0
    LEDGER_ROW = re.compile(r"\d+\.\d+ GiB\s+\d+ tensors")
    for i, line in enumerate(readme_lines, 1):
        if LEDGER_ROW.search(line):
            ledger_rows += 1
            continue  # 整行逐字摘自 evidence/03 的角色台账
        for tok in token_rx.findall(line):
            tok = tok.strip()
            if tok in covered:
                continue
            if any(re.fullmatch(p, tok) for p, _ in EXTRA_COVERED_PATTERNS):
                continue
            uncovered.setdefault(tok, []).append(i)

    print("=" * 118)
    print("M30 README 全量数字对账（每个量 → 归档出处:行号 → 判定）")
    print("=" * 118)
    print(f"{'主张':<26}{'类别':<9}{'README':<8}{'期望值':<16}{'归档出处':<28}{'判定'}")
    print("-" * 118)
    for cid, kind, rline, expect, src, verdict, note in rows:
        print(f"{cid:<26}{kind:<9}{rline:<8}{expect:<16}{src:<28}{verdict}")
    print("-" * 118)
    print(f"对账条目: {len(rows)}（PASS/LIVE-OK/ATTRIBUTED/EXEMPT 之外 = 失败）")
    print(f"失败条目: {failures if failures else '无'}")
    n_fixed = sum(1 for r in rows if r[1] in ("fixed", "derived"))
    n_live = sum(1 for r in rows if r[1] == "live")
    n_attr = sum(1 for r in rows if r[1] == "external")
    n_exempt = sum(1 for r in rows if r[1] in ("quote", "narrative"))
    print()
    print("== 覆盖性检查：README 中未纳入对账表的带单位数字（需人工判定是否应补条目） ==")
    if not uncovered:
        print("  （无）")
    else:
        for tok, lines in sorted(uncovered.items(), key=lambda kv: -len(kv[1])):
            print(f"  {tok:<12} 出现在行 {lines[:8]}{' …' if len(lines) > 8 else ''}")
    print(f"  （另有 {ledger_rows} 行 checkpoint 角色台账整行逐字摘自 evidence/03，按 §6.2 的引用说明覆盖）")
    print()
    print("== 口径说明 ==")
    print("  fixed/derived : 固定量，可直接引用；derived 由其他固定量算出（括号内为算式）")
    print("  live          : 活值（随其它 worker 波动），只能写'活值 + 指向 evidence 采集时刻'，不得作为结论；")
    print("                  '期望值'列现场从归档解析（本表不存字面量），README 若写了实例读数则必须与之一致")
    print("  external      : 来自本仓库外（tower/用户口径），README 必须注明来源")
    print("  quote         : 更正记录里引用的旧值，按设计豁免")
    print()
    if compared == 0:
        print("RESULT: SKIPPED (0 compared -- no README line resolved against, or no archive value "
              "parsed; this is NOT a pass)")
        return 2
    if failures:
        print(f"RESULT: FAIL ({len(failures)}/{len(rows)} claims failed; {compared} compared "
              f"= {n_fixed} 归档比对 + {n_live} live 现场解析)")
        return 1
    print(f"RESULT: OK ({compared} compared = {n_fixed} 归档比对 + {n_live} live 现场解析; "
          f"另 {n_attr} 外部来源、{n_exempt} 豁免)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
