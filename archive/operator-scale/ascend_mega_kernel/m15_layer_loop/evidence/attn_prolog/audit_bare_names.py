#!/usr/bin/env python3.12
"""M88 反假绿审计：本 mission 3 个头文件里**裸用**的全大写常量标识符，是否定义在 M15AP 里。

为什么需要它：本 TU 里有 `using namespace M15G;` / `using namespace M15Loop;`（分别来自
`m15_attn_kv.h` 与 `m15_loop_layout.h` 里的同名 using 指令；行号会漂 ⇒ 按**符号** grep）。于是**一个裸写的名字若 M15AP 里没有，就会静默
绑定到别处的同名符号**，判据就会瞄错区段而**假绿**。M88 r2 复审抓到的那条正是这个形态：
`m15_attn_prolog_host.h` 里裸写 `OUT_K`（`M15AP` 只有 `AP_OUT_K` = 12288）⇒ 静默取
`M15G::OUT_K` = 6144 ⇒ `Ap.out.k` 比的是 **gate 段**，k 段从未被比较。

用法（在 m15_layer_loop/ 下）：python3.12 evidence/attn_prolog/audit_bare_names.py
rc=0 = 无命中；rc=1 = 有命中（逐条打印）。
"""
import pathlib
import re
import sys

MINE = ['m15_attn_prolog.h', 'm15_attn_prolog_probe.h', 'm15_attn_prolog_host.h']
OTHERS = ['m15_gdn_resources.h', 'm15_hc_resources.h', 'm15_moe_resources.h',
          'm15_layer_resources.h', 'm15_loop_layout.h', 'm15_attn_kv.h']

STR = re.compile(r'"(?:[^"\\]|\\.)*"')


def strip(line: str) -> str:
    return STR.sub('""', line.split('//')[0])


def declared(paths):
    """这些头文件里**声明为常量/枚举**的全大写标识符（= 会被 `using namespace` 带进来的那类）。"""
    out = set()
    for f in paths:
        if not pathlib.Path(f).exists():
            continue
        s = pathlib.Path(f).read_text()
        s = re.sub(r'//[^\n]*', '', s)
        out |= set(re.findall(r'constexpr\s+[\w:<>]+\s+([A-Z][A-Z0-9_]{1,})\s*=', s))
        out |= set(re.findall(r'^\s*([A-Z][A-Z0-9_]{1,})\s*=\s*0x[0-9a-fA-F]+', s, re.M))
        out |= set(re.findall(r'enum\s+class\s+\w+\s*:\s*\w+\s*\{([^}]*)\}', s, re.S)[0].split(',')) \
            if re.search(r'enum\s+class\s+\w+\s*:\s*\w+\s*\{', s) else set()
    return {n.strip() for n in out if n.strip()}


missing = [f for f in MINE if not pathlib.Path(f).exists()]
if missing:
    print(f"⚠ 在错误的目录里跑：找不到 {missing}；本脚本必须在 `m15_layer_loop/` 下执行")
    sys.exit(2)

ap = declared(MINE)
foreign = declared([f for f in OTHERS if pathlib.Path(f).exists()])

hits = []
for f in MINE:
    if not pathlib.Path(f).exists():
        continue
    for ln, raw in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        for m in re.finditer(r'(?<![:\w.])([A-Z][A-Z0-9_]{1,})\b', strip(raw)):
            nm = m.group(1)
            if nm in ap or nm not in foreign:
                continue          # M15AP 里有 → 安全；两边都没有 → 不是常量类，另说
            hits.append((f, ln, nm, raw.strip()[:96]))

print(f"M15AP 常量 {len(ap)} 个；M15G/M15H/M15M/M15Loop/M15Kv 常量 {len(foreign)} 个")
if not ap or not foreign:
    print("⚠ 扫描面为空（M15AP 或 foreign 常量数为 0）⇒ 这次 0 命中**不算数**，脚本按失败退出")
    sys.exit(2)
if not hits:
    print("裸用「别处同名常量」的标识符：0 条")
    sys.exit(0)
print(f"⚠ 裸用但 M15AP 未定义、而别处有同名常量的标识符：{len(hits)} 条（会静默绑错）：")
for f, ln, nm, txt in hits:
    print(f"  {f}:{ln}  {nm}    {txt}")
sys.exit(1)
