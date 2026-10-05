#!/usr/bin/env python3
"""生成 evidence/dump_sha256.txt（docs/17 §2.5 要求的 dump 校验和）。

覆盖范围 = **数据产物**：两个 M24 切片、2D/3D raw dump + sidecar、`m16_pv` 的 .bin 产物。
**不覆盖 `*.log`**（含 reproduce_run.log）：这些日志里含有 §2.3 记录的 9 个非确定 conf 的
越界槽位内容，且 reproduce_run.log 每次跑都会变 ⇒ 纳入校验和会让"校验和"本身不稳定、失去意义。
日志里的稳定数字（如 0/4096、23552/23552、解码命中 4096/4096）由 README §2.3/§4 引用并逐条列出。

用法（在工程根 m16_load_geom/ 下）：
    /usr/local/python3.12.13/bin/python3 tools/make_sha256.py
"""
import hashlib
import os
import sys

SLICES = ["m24_s256_P_unit0_par0.bin", "m24_s256_V_n2_0.bin"]
DATA_SUFFIXES = (".bin", ".tsv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ev = "evidence"
    missing = [s for s in SLICES if not os.path.isfile(os.path.join(ev, s))]
    if missing:
        print("[sha] 缺少跨分支输入切片：%s —— 提取命令见 README §4.1" % missing, file=sys.stderr)
        return 2
    L = ["# M27 归档校验和（复核：cd m16_load_geom && sha256sum -c evidence/dump_sha256.txt）",
         "#",
         "# 覆盖范围 = 数据产物（切片 / raw dump / sidecar / m16_pv 的 .bin 产物）。",
         "# 不含 *.log：日志含 §2.3 记录的 9 个非确定 conf 的越界槽位内容，且 reproduce_run.log",
         "# 每次运行都会变 ⇒ 纳入校验和会让校验和本身不稳定。日志里的稳定数字见 README §2.3/§4。",
         "#",
         "# 两个 M24 切片是唯一的跨分支输入（提取命令见 README §4.1）；wt-24 源文件不变则 sha 恒定。",
         ""]
    L.append("# --- 跨分支输入：M24 切片 ---")
    for s in SLICES:
        L.append("%s  %s" % (sha256(os.path.join(ev, s)), os.path.join(ev, s)))
    L.append("")
    L.append("# --- 归档数据产物 ---")
    for name in sorted(os.listdir(ev)):
        if name in SLICES or name.endswith(".log") or name.endswith(".txt"):
            continue
        if not name.endswith(DATA_SUFFIXES):
            continue
        p = os.path.join(ev, name)
        if os.path.isfile(p):
            L.append("%s  %s" % (sha256(p), p))
    open(os.path.join(ev, "dump_sha256.txt"), "w").write("\n".join(L) + "\n")
    n = sum(1 for l in L if l and not l.startswith("#"))
    print("[sha] 写出 evidence/dump_sha256.txt（%d 个数据产物）" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
