#!/usr/bin/env python3
"""把 m16_geom 的原始 L0C dump 解成"几何映射表"。

读 m16_geom_2d_raw.bin / m16_geom_3d_raw.bin（conf × 16 × 64 fp32）。

已知（本探针已标定，见 README §3）：
  * mmad 眼里的 L0B 元素下标 = 16*n + k（k<16，n<64），即 L0B 行(=512B 分形槽) R = n//16，
    行内偏移 = (n%16)*16 + k，行内连续 16 个元素是 k。
  * 对落在 L0B 行 R 的 16x16 源分形（源基准 (r0,c0)）：
      ifTranspose=0 ⇒ B[k][n] = src(r0 + n%16, c0 + k)
      ifTranspose=1 ⇒ B[k][n] = src(r0 + k,   c0 + n%16)
  ⇒ 逐行反解 (r0,c0) 即可读出"哪个源分形落在哪个 L0B 槽位"。

源编码：元素 (r,c) 序号 k = r*32+c，bf16 位型 = 0x4000 + k。
"""
import sys
import numpy as np

R, C, M, N = 32, 32, 16, 64
BASE = 0x4000
CODES = BASE + R * C


def load(path, nconf):
    a = np.fromfile(path, dtype=np.float32)
    per = M * N
    a = a[: a.size // per * per].reshape(-1, per)
    return a.reshape(-1, M, N)


def decode(block):
    bits = block.view(np.uint32)
    hi = bits >> 16
    lo = bits & 0xFFFF
    ok = (lo == 0) & ((hi & 0x8000) == 0) & (hi >= BASE) & (hi < CODES)
    ks = np.where(ok, hi.astype(np.int64) - BASE, -1)
    rr = np.where(ok, ks // C, -1)
    cc = np.where(ok, ks % C, -1)
    zero = block == 0.0
    rr = np.where(ok, rr, np.where(zero, -2, -1))
    cc = np.where(ok, cc, np.where(zero, -2, -1))
    return rr, cc


def rows_summary(rr, cc, transposed):
    """逐 L0B 行反解 (r0,c0)；返回 4 行的字符串。"""
    out = []
    for R0 in range(4):
        k0, n0 = 0, R0 * 16
        # 用 k=0 行推 c0，用 n=n0 推 r0
        r_a, c_a = rr[k0, n0], cc[k0, n0]
        r_b, c_b = rr[k0, n0 + 15], cc[k0, n0 + 15]
        r_c, c_c = rr[15, n0], cc[15, n0]
        if transposed:
            # B[k][n] = src(r0+k, c0+n%16)
            if r_a >= 0 and c_a >= 0 and c_b >= 0 and r_c >= 0:
                out.append(f"R{R0}<-src(r0={r_a},c0={c_a})")
            elif r_a == -2:
                out.append(f"R{R0}=zero")
            else:
                out.append(f"R{R0}=junk")
        else:
            # B[k][n] = src(r0+n%16, c0+k)
            if r_a >= 0 and c_a >= 0 and r_b >= 0 and c_c >= 0:
                out.append(f"R{R0}<-src(r0={r_b - 15},c0={c_a})")
            elif r_a == -2:
                out.append(f"R{R0}=zero")
            else:
                out.append(f"R{R0}=junk")
    return out


def parse_tsv(path):
    """读 sidecar conf 表；返回 {idx: (tag, transpose_bool_or_None, fields)}"""
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        idx = int(parts[0])
        tag = parts[1]
        fields = parts[2]
        tr = None
        for tok in fields.split():
            if tok.startswith("T="):
                tr = tok.split("=")[1] == "1"
            if tok.startswith("enTranspose="):
                tr = tok.split("=")[1] == "1"
        out[idx] = (tag, tr, fields)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "m16_geom_2d_raw.bin"
    labels = parse_tsv(sys.argv[2]) if len(sys.argv) > 2 else {}
    blocks = load(path, 0)
    print(f"# {path}: {blocks.shape[0]} confs")
    for i in range(blocks.shape[0]):
        rr, cc = decode(blocks[i])
        nok = int((rr >= 0).sum())
        nz = int((rr == -2).sum())
        nbad = int((rr == -1).sum())
        tag, tr, fields = labels.get(i, ("", None, ""))
        if tr is None:
            tr = True
        s = rows_summary(rr, cc, tr)
        print(f"[{i:02d}] {tag:<52} ok={nok:4d} zero={nz:4d} other={nbad:4d} "
              f"| {'T1' if tr else 'T0'} | " + "  ".join(s))
        print(f"      {fields}")


if __name__ == "__main__":
    main()
