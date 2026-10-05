#!/usr/bin/env python3
"""m16_load_geom 的独立 numpy 复核（不依赖 C++ 侧的任何结论）。

用法：
  # 1) BMM2 数值复核（读 m16_pv 的 dump：P/V/device C/参考 C）
  python3 check_ref.py pv

  # 2) 几何映射表复核（读 m16_geom 的 raw dump + sidecar 字段表，重算"源分形 → L0B 槽位"）
  python3 check_ref.py geom2d [raw.bin [raw.bin.tsv]]
  python3 check_ref.py geom3d [raw.bin [raw.bin.tsv]]

判据：
  * pv   —— device C 与 numpy fp32 参考（RNE 到 bf16 网格）逐位一致；打印 maxRel。
  * geom —— 按 README §3 的映射模型逐 conf 预测"哪个源分形落在哪个 L0B 行"，与实测解码比对。
"""
import sys

import numpy as np

M, N = 16, 64          # 探针 mmad 窗口
ROWS, COLS = 32, 32    # 探针源尺寸
CODE_BASE = 0x4000


def bf16_to_f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) >> 16).astype(np.uint16)


def check_pv():
    p = np.fromfile("m16_pv_p.bin", dtype=np.uint16).reshape(16, 256)
    v = np.fromfile("m16_pv_v.bin", dtype=np.uint16).reshape(256, 256)
    dev = np.fromfile("m16_pv_c_device.bin", dtype=np.uint16).reshape(16, 256)
    ref = bf16_to_f32(f32_to_bf16(bf16_to_f32(p) @ bf16_to_f32(v)))
    devf = bf16_to_f32(dev)
    bad = int((dev != f32_to_bf16(ref)).sum())
    den = np.maximum(np.abs(ref), 1e-6)
    maxrel = float((np.abs(devf - ref) / den).max())
    print(f"[pv] device vs numpy fp32 ref(RNE->bf16): 不等 {bad}/4096, maxRel={maxrel:.3e}")
    print("[pv] PASS" if bad == 0 else f"[pv] FAIL ({bad} 个元素不等)")
    return 0 if bad == 0 else 1


def decode(block):
    bits = block.view(np.uint32)
    hi = bits >> 16
    lo = bits & 0xFFFF
    ok = (lo == 0) & ((hi & 0x8000) == 0) & (hi >= CODE_BASE) & (hi < CODE_BASE + ROWS * COLS)
    ks = np.where(ok, hi.astype(np.int64) - CODE_BASE, -1)
    rr = np.where(ok, ks // COLS, -1)
    cc = np.where(ok, ks % COLS, -1)
    return rr, cc, ok


def predict_grid(fields):
    """按 README §3 的映射模型，独立预测该 conf 的完整 16x64 解码网格。

    模型（全部由 m16_geom 的实测标定得出，写在这里是为了让 check_ref.py 能独立复算）：
      * 源 L1 = Nd2Nz(R_pad=32) 的 Nz：分形 (r1,c1) 在 L1 的单元号 = r1 + c1*2（每单元 256 元素）；
      * mmad (k,n) 落在 L0B 元素下标 16*n + k（L0B 行 = n//16，行内偏移 = (n%16)*16 + k）；
      * 拷贝 (m1,k1) ∈ [0,mStep)x[0,kStep)，拷贝序 k1 外层、m1 内层（后写覆盖先写）：
          ifTranspose=0: 源单元 = m1 + k1*srcStride，目的行 = m1 + k1*dstStride；
                         落入该行的分形 (r0=r1*16, c0=c1*16) ⇒ B[k][n] = src(r0 + n%16, c0 + k)
          ifTranspose=1: 源单元 = m1*srcStride + k1，目的行 = m1*dstStride + k1；
                         ⇒ B[k][n] = src(r0 + k, c0 + n%16)
      * srcOff/dstOff 以 256 元素为单位整体平移源/目的单元号。
    返回 {(k,n): (r,c)}；未写到的位置不在字典里。
    """
    kv = {}
    for tok in fields.split():
        if "=" in tok:
            kk, vv = tok.split("=", 1)
            kv[kk] = int(vv)
    tr = (kv.get("T", int(kv.get("enTranspose", 1)))) == 1
    m_step, k_step = kv.get("mStep", 0), kv.get("kStep", 0)
    src_stride, dst_stride = kv.get("src", 0), kv.get("dst", 0)
    m_start, k_start = kv.get("mS", 0), kv.get("kS", 0)
    src_off = kv.get("srcOff", 0) // 256
    dst_off = kv.get("dstOff", 0) // 256
    grid = {}
    for k1 in range(k_step):          # 拷贝序：k1 外层
        for m1 in range(m_step):      #          m1 内层
            # 源分形单元号（实测：mStart/kStart 是"源分形的起始下标"，单位=源行列方向的分形个数）
            u_src = (m_start + m1) + (k_start + k1) * src_stride + src_off
            if tr:
                u_dst = m1 * dst_stride + k1 + dst_off
            else:
                u_dst = m1 + k1 * dst_stride + dst_off
            r1, c1 = (u_src % 2, u_src // 2) if 0 <= u_src < 4 else (-1, -1)   # (-1,-1) = 越界读
            for k in range(16):
                for nloc in range(16):
                    n = u_dst * 16 + nloc
                    if not (0 <= n < 64):
                        continue
                    if r1 < 0:
                        grid[(k, n)] = (-1, -1)      # 该槽位被越界数据覆盖 ⇒ 实测不应解码出合法源值
                    else:
                        grid[(k, n)] = (r1 * 16 + k, c1 * 16 + nloc) if tr else (r1 * 16 + nloc, c1 * 16 + k)
    return grid


def check_geom(path, tsv, label_name):
    raw = np.fromfile(path, dtype=np.float32)
    raw = raw[: raw.size // (M * N) * (M * N)].reshape(-1, M, N)
    label = {}
    for line in open(tsv):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            label[int(parts[0])] = (parts[1], parts[2])
    tot_pt = tot_ok = tot_bad = 0
    bad_confs = []
    for i in range(raw.shape[0]):
        rr, cc, ok = decode(raw[i])
        tag, fields = label.get(i, ("", ""))
        if "mStep=" not in fields:
            # 3D conf：模型不适用（未标定），只报告写入面积
            print(f"[geom] conf {i:02d} {tag[:52]:<52} 解码面积={int((rr>=0).sum()):4d}")
            continue
        pred = predict_grid(fields)
        nok = nbad = njunk = nfalse = 0
        for (k, n), (pr, pc) in pred.items():
            gr, gc = rr[k, n], cc[k, n]
            if pr < 0:
                # 越界读覆盖的槽位：内容是"未初始化内存的拷贝"，本身非确定性，
                # 不参与 PASS/FAIL（只统计偶尔碰巧解码成合法值的情况）
                njunk += 1
                if gr >= 0:
                    nfalse += 1
                continue
            if gr == pr and gc == pc:
                nok += 1
            else:
                nbad += 1
        tot_pt += nok + nbad
        tot_ok += nok
        tot_bad += nbad
        flag = "" if nbad == 0 else "   <== 与模型不符"
        print(f"[geom] conf {i:02d} {tag[:52]:<52} 模型点={nok + nbad:4d} 命中={nok:4d} 不符={nbad:4d} "
              f"越界槽={njunk:4d}(其中碰巧解码 {nfalse}){flag}")
        if nbad:
            bad_confs.append(i)
    print(f"[geom] 合计（只统计模型可判定的槽位）：预测点 {tot_pt}，命中 {tot_ok}，不符 {tot_bad}")
    if tot_bad == 0:
        print("[geom] PASS：README §3 的映射模型在全部 2D conf 上逐点成立")
        return 0
    print(f"[geom] FAIL：conf {bad_confs} 与模型不符（需在 README 的【仍存疑】里说明）")
    return 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pv"
    if mode == "pv":
        return check_pv()
    if mode == "geom2d":
        p = sys.argv[2] if len(sys.argv) > 2 else "m16_geom_2d_raw.bin"
        t = sys.argv[3] if len(sys.argv) > 3 else p + ".tsv"
        return check_geom(p, t, "2d")
    if mode == "geom3d":
        p = sys.argv[2] if len(sys.argv) > 2 else "m16_geom_3d_raw.bin"
        t = sys.argv[3] if len(sys.argv) > 3 else p + ".tsv"
        return check_geom(p, t, "3d")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
