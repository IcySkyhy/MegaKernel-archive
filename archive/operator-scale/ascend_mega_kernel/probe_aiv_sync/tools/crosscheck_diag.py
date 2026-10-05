#!/usr/bin/env python3
"""crosscheck_diag.py —— 双源交叉核对：**设备侧自读数** vs **host 对设备读回快照的逐词归因**

为什么需要（塔 item-3：「同源的地方要引入独立的第二来源」）：
  主探针的关键读数有两个来源——
  源①  设备侧：每核在 VF 里数出「head 命中本轮 ticket 的槽数」，写进 obs 行（`ROUND ... cores_clean=K/56`）；
  源②  host 侧：把设备**读回的那张表**整块搬回 GM，host 逐词判 `TICKET(T,i)` 是否命中（本脚本再数一遍）。
  两者**走的是不同代码路径**（VF Compare/Reduce vs host 逐词比对），但看的必须是同一份数据。
本脚本按轮核对 `cores_clean` 是否逐一相等；不等即报错（说明其中一条路径的计数逻辑有 bug）。

用法：python3 tools/crosscheck_diag.py <diag_run.log> <snap_*.bin>
退出码：0 = 全部轮次一致；1 = 有不一致（或输入缺失）。
"""
import re
import struct
import sys

MAGIC16 = 0x5153


def device_rounds(log):
    """从 diag 日志里取 ROUND 行：round -> cores_clean"""
    out = {}
    for ln in open(log, encoding='utf-8', errors='replace'):
        m = re.match(r'ROUND variant=(\S+) run=(\d+) round=(\d+) .*cores_clean=(\d+)/(\d+)', ln)
        if m:
            out[int(m.group(3))] = (int(m.group(4)), int(m.group(5)))
    return out


def host_from_snap(path):
    """从设备快照按轮数 head 命中（host 独立复算）"""
    with open(path, 'rb') as f:
        slmax, rndmax, naiv, rounds, st, wl, words, magic = struct.unpack('<8i', f.read(32))
        if magic != 0x5A5A5A5A:
            raise SystemExit('bad snapshot magic')
        raw = f.read()
    data = struct.unpack('<%di' % (len(raw) // 4), raw)
    out = {}
    for r in range(rounds):
        T = r + 1
        n = 0
        for c in range(naiv):
            base = (c * rndmax + r) * words
            reg = data[base:base + words]
            if all(reg[i * st] == ((MAGIC16 << 16) | ((T & 0xFF) << 8) | i) for i in range(naiv)):
                n += 1
        out[r] = (n, naiv)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    dev = device_rounds(sys.argv[1])
    host = host_from_snap(sys.argv[2])
    print('# 双源交叉核对（设备侧 VF 计数 vs host 逐词归因）')
    print(f"{'round':>5s} {'设备侧 cores_clean':>18s} {'host 归因 cores_clean':>21s}  一致?")
    bad = 0
    for r in sorted(set(dev) | set(host)):
        d = dev.get(r)
        h = host.get(r)
        ok = (d == h)
        if not ok:
            bad += 1
        print(f"{r:5d} {str(d):>18s} {str(h):>21s}  {'OK' if ok else '不一致 ✗'}")
    print(f"\n不一致轮次：{bad}    （两源不同路径、同一份数据；不一致即其中一条计数路径有 bug）")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
