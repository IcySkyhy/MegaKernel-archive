#!/usr/bin/env python3.12
"""real_table_probe.py —— M92：真实 ngram 表的**落地形态**取证 + 设备 harness 输入生成

被验对象：checkpoint `/workspace/Qwen3.8-Flash-Next-MXFP4` 的
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`
（BF16 `[2500012,160]` ×128 = 320,001,536 行 = 102,400,491,520 B = 95.37 GiB）。

本脚本**不物化整表**（102.4 GB > 本容器 cgroup 32 GiB，也远超 HBM 常驻预算）：
file-backed `mmap`（按需页）+ 匿名 staging 窗口（= 设备侧要一次性注册的那块）。

三种模式：
  `--evidence`   : 行 id → (分片, 片内行, 文件偏移) 换算 + 逐字节对拍（≥3 个真实行 id）
  `--real-scale` : **一个真实分片的完整行集合**（762.94 MiB）走 mmap→staging 搬一遍，抽样逐字节对拍
  `--emit`       : 生成设备 harness 的输入（真实行 id / 期望行字节 / HM 侧输入）

用法：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/real_table_probe.py --evidence
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/real_table_probe.py --real-scale --shard 0
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/real_table_probe.py --emit
"""
import argparse
import json
import mmap
import pathlib
import struct
import sys
import time

import numpy as np

CKPT = pathlib.Path("/workspace/Qwen3.8-Flash-Next-MXFP4")
HERE = pathlib.Path(__file__).resolve().parent          # m15_layer_loop/ple/
DATA_HM = HERE / "data_hm"

SHARDS = 128
ROWS_PER_SHARD = 2_500_012
COLS = 160
ROW_BYTES = COLS * 2                                     # 320 B
TOTAL_ROWS = SHARDS * ROWS_PER_SHARD                     # 320,001,536
TOTAL_BYTES = TOTAL_ROWS * ROW_BYTES                     # 102,400,491,520

# 设备侧 staging 窗口（= 要 aclrtHostRegisterV2(MAPPED) 一次、之后反复复用的那块）
WIN_MIB_DEFAULT = 256
WIN_BYTES_DEFAULT = WIN_MIB_DEFAULT * 1024 * 1024
WIN_ROWS_DEFAULT = WIN_BYTES_DEFAULT // ROW_BYTES        # 838,860 行

# M171：多槽窗口池（每个 slot 是一段连续的**全局行区间**，可以横跨分片边界）
SLOT_MIB_DEFAULT = 4                                     # 每槽 staging 字节数
SLOT_ROWS_DEFAULT = SLOT_MIB_DEFAULT * 1024 * 1024 // ROW_BYTES   # 13,107 行/槽

HID = 2560
HC = 4
W = HID * HC
STLEN = 9


# ---------------------------------------------------------------- 分片几何
def shard_map():
    """扫 safetensors 头，建 shard id -> {file, data_off, nbytes}。

    safetensors 布局 = 8 B 小端头长 + 对齐后的 JSON 头 + 数据段
    ⇒ 数据段起点 = 8 + len(JSON)，张量在文件内的绝对偏移 = 该起点 + data_offsets[0]。
    """
    out = {}
    for f in sorted(CKPT.glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if "ngram_embedding.shard_" in k and k.endswith(".weight"):
                i = int(k.split("shard_")[1].split(".")[0])
                assert v["dtype"] == "BF16", (k, v["dtype"])
                assert tuple(v["shape"]) == (ROWS_PER_SHARD, COLS), (k, v["shape"])
                o0, o1 = v["data_offsets"]
                assert (o1 - o0) == ROWS_PER_SHARD * ROW_BYTES, (k, o1 - o0)
                out[i] = {"file": str(f), "base": base, "tensor_off": o0,
                          "data_off": base + o0, "nbytes": o1 - o0}
    assert len(out) == SHARDS, len(out)
    return out


def locate(gid):
    """全局行 id → (分片, 片内行)。全局 id 不按片索引：id = shard*2500012 + local。"""
    return divmod(gid, ROWS_PER_SHARD)


class MappedTable:
    """file mmap（按需页）+ 匿名 staging 窗口（模拟设备侧一次性注册的那块）。

    设备读的**不是** file-backed mmap（M86 实测 aclrtHostRegisterV2 对 file mmap 报
    `rc=507899`）⇒ 本类把需要的行**搬进**匿名窗口，设备只读窗口。
    """

    def __init__(self, smap, win_bytes):
        self.sm = smap
        self.win_bytes = win_bytes
        self.win_rows = win_bytes // ROW_BYTES
        self.win = mmap.mmap(-1, win_bytes)          # MAP_ANONYMOUS：设备侧就是这块要被注册
        self.win_base = 0
        self.fds = {}
        self.fmaps = {}
        self.bytes_moved = 0
        self.fills = 0

    def _map(self, shard):
        if shard not in self.fmaps:
            info = self.sm[shard]
            fd = os_open_ro(info["file"])
            self.fds[shard] = fd
            # file-backed mmap：按需页（不落常驻，也不允许整表读入）。
            # 起点按页对齐；长度补上页内偏移，使数据段尾部也可见（不越过文件末端）。
            skew = info["data_off"] % 4096
            start = info["data_off"] - skew
            length = info["nbytes"] + skew
            with open(info["file"], "rb") as fh:
                fh.seek(0, 2)
                length = min(length, fh.tell() - start)
            self.fmaps[shard] = mmap.mmap(fd, length, prot=mmap.PROT_READ, offset=start)
        return self.fmaps[shard]

    def stage_block(self, shard, local_start, nrows):
        """把 [local_start, local_start+nrows) 从分片文件搬进匿名窗口（= 一次块填充）。"""
        nrows = min(nrows, self.win_rows, ROWS_PER_SHARD - local_start)
        info = self.sm[shard]
        src = self._map(shard)
        skew = info["data_off"] % 4096
        off = skew + local_start * ROW_BYTES
        nb = nrows * ROW_BYTES
        self.win[:nb] = src[off:off + nb]
        self.win_base = shard * ROWS_PER_SHARD + local_start
        self.bytes_moved += nb
        self.fills += 1
        return nrows

    def row(self, gid):
        """从窗口取一行（窗口外返回 None）；返回 (bytes, 是否为窗口内容)。"""
        rel = gid - self.win_base
        if rel < 0 or rel >= self.win_rows:
            return None
        off = rel * ROW_BYTES
        return bytes(self.win[off:off + ROW_BYTES])


def os_open_ro(path):
    import os
    return os.open(path, os.O_RDONLY)


def direct_row(smap, gid):
    """**直接读该 shard 文件**：定位到 `data_off + local*320` 读 320 B（不经窗口）。"""
    shard, local = locate(gid)
    info = smap[shard]
    with open(info["file"], "rb") as fh:
        fh.seek(info["data_off"] + local * ROW_BYTES)
        return fh.read(ROW_BYTES)


# ---------------------------------------------------------------- M171 多槽窗口池
class Slot:
    """一个槽 = 一段连续的**全局行区间** `[base, base+rows)`，可以横跨分片边界。

    `row_off` = 该槽第一行在注册窗口（平坦 `[pool_rows,160]` bf16）里的行号。
    设备侧用 (shard, local) 分解 + `delta = (shard-slot_shard)*rows_per_shard + local - slot_local`
    在这段区间里定位（见 `m15_ple.asc::PleGather` 的 `nSlots != 0` 分支）。
    """

    __slots__ = ("shard", "local_base", "rows", "row_off")

    def __init__(self, shard, local_base, rows, row_off):
        assert local_base < ROWS_PER_SHARD
        self.shard = int(shard)
        self.local_base = int(local_base)
        self.rows = int(rows)
        self.row_off = int(row_off)

    @property
    def base(self):
        return self.shard * ROWS_PER_SHARD + self.local_base

    @property
    def end(self):
        return self.base + self.rows

    def contains(self, gid):
        return self.base <= gid < self.end

    def row_of(self, gid):
        return self.row_off + (gid - self.base)


def plan_slots(n_slots, slot_rows, smap):
    """排布 `n_slots` 个槽：**槽 0 横跨分片边界**（shard 0 尾 → shard 1 头），其余落在别的分片。

    这样单次发射同时覆盖 U-A（多槽）与 U-B（跨分片窗口）两条：池里既有跨界槽，
    也有彼此不相邻、分属不同分片的槽。
    """
    assert n_slots >= 1
    assert slot_rows >= 4 and slot_rows < ROWS_PER_SHARD
    half = slot_rows // 2
    slots = [Slot(0, ROWS_PER_SHARD - half, slot_rows, 0)]      # ★ 跨界：0 → 1
    row_off = slot_rows
    # 其余槽：挑与槽 0 不同分片（且不含 shard 0/1）的槽，避免覆盖同一个 id 区间
    far = [3, 60, 127, 31, 90, 7, 45]
    for i in range(1, n_slots):
        sh = far[(i - 1) % len(far)]
        local = (i * 977) % (ROWS_PER_SHARD - slot_rows)
        slots.append(Slot(sh, local, slot_rows, row_off))
        row_off += slot_rows
    pool_rows = row_off
    if not all(sh in smap for sh in (s.shard for s in slots)):
        raise SystemExit("[multi-slot] shard_map 缺分片 %s" % [s.shard for s in slots])
    return slots, pool_rows


def fill_segments(slots, smap):
    """把每个槽拆成 host 侧的搬运行（可跨分片）：`(dst_row_off, nbytes, file, byte_off)`。

    C++ harness 只按这张表 open/mmap/memcpy，**所有分片算术都留在 host（本脚本）**——
    设备侧看不到这条路径，它只从窗口取行。
    """
    segs = []
    for s in slots:
        sh, lb, remaining = s.shard, s.local_base, s.rows
        dst = s.row_off * ROW_BYTES
        while remaining > 0:
            take = min(remaining, ROWS_PER_SHARD - lb)
            info = smap[sh]
            byte_off = info["data_off"] + lb * ROW_BYTES
            segs.append((dst, take * ROW_BYTES, info["file"], byte_off))
            dst += take * ROW_BYTES
            remaining -= take
            sh += 1
            lb = 0
    return segs


def serve_id(slots, rng):
    """在池里随机取一个「已被服务」的真实行 id。"""
    s = slots[int(rng.integers(0, len(slots)))]
    return s.base + int(rng.integers(0, s.rows))


# ---------------------------------------------------------------- 模式
def run_evidence(args):
    smap = shard_map()
    mt = MappedTable(smap, 4 * 1024 * 1024)      # 这里的窗口只需覆盖被查的行
    print("== 真实表几何 ==")
    print("分片数            : %d" % SHARDS)
    print("每片行数          : %d（= ceil(320001446/128)*128/128，PLE_SPEC.md §2.1）" % ROWS_PER_SHARD)
    print("行宽              : %d B（160 × bf16）" % ROW_BYTES)
    print("全表              : %d 行 = %d B = %.2f GB = %.2f GiB"
          % (TOTAL_ROWS, TOTAL_BYTES, TOTAL_BYTES / 1e9, TOTAL_BYTES / 2 ** 30))
    print()
    print("== 行 id → 文件位置（本脚本在真实文件上实测）==")
    ids = [0, 1, 2, 12345, ROWS_PER_SHARD - 1, ROWS_PER_SHARD, ROWS_PER_SHARD + 3,
           TOTAL_ROWS - 1]
    print("%-14s %-6s %-10s %-16s %-12s %s" % ("global_id", "shard", "local", "file(切片)", "file_off", "结果"))
    bad = 0
    rows_out = []
    for gid in ids:
        shard, local = locate(gid)
        info = smap[shard]
        foff = info["data_off"] + local * ROW_BYTES
        mt.stage_block(shard, local, 1)            # 只把该行搬进窗口（按需）
        got = mt.row(gid)
        exp = direct_row(smap, gid)
        same = (got == exp)
        if not same:
            bad += 1
        rows_out.append((gid, shard, local, foff, same, exp))
        print("%-14d %-6d %-10d %-16s %-12d %s" % (
            gid, shard, local, pathlib.Path(info["file"]).name, foff,
            "byte-identical" if same else "MISMATCH"))
    print()
    print("== 逐字节证据（≥3 个真实行 id：窗口 gather vs 直读 shard 文件）==")
    for gid, shard, local, foff, same, exp in rows_out:
        got = mt.row(gid)
        print("id=%-12d shard=%-4d file_off=%-12d sha256(320B)=%s  %s"
              % (gid, shard, foff, __import__("hashlib").sha256(exp).hexdigest()[:32],
                 "OK" if same else "FAIL"))
    print()
    print("窗口搬运：fills=%d bytes_moved=%d（窗口 %d B）" % (mt.fills, mt.bytes_moved, mt.win_bytes))
    print("结论：%s" % ("PASS —— 上述 %d 个真实行 id 的 320 B 与直读分片文件逐字节一致" % len(ids)
                      if bad == 0 else "FAIL —— %d 行不符" % bad))
    return 0 if bad == 0 else 1


def run_real_scale(args):
    smap = shard_map()
    shard = args.shard
    nrows = ROWS_PER_SHARD
    stage_bytes = nrows * ROW_BYTES
    print("== 真实规模读数：整片 shard_%d 的完整行集合 ==" % shard)
    print("行数              : %d" % nrows)
    print("数据段            : %d B = %.2f MiB = %.3f GiB"
          % (stage_bytes, stage_bytes / 2 ** 20, stage_bytes / 2 ** 30))
    mt = MappedTable(smap, stage_bytes)
    t0 = time.perf_counter()
    staged = mt.stage_block(shard, 0, nrows)
    t1 = time.perf_counter()
    print("staging 行数      : %d" % staged)
    print("搬运字节          : %d B（file mmap 按需页 → 匿名 staging 窗口，一次搬运）" % mt.bytes_moved)
    print("耗时（host 墙钟，单次，**非带宽证据**，口径见 docs/17 §9.1）: %.3f s ⇒ %.2f GB/s"
          % (t1 - t0, mt.bytes_moved / (t1 - t0) / 1e9))
    rng = np.random.default_rng(20260927)
    sample = [0, 1, 2, 3, nrows - 1, nrows // 2, 12345, 999983]
    sample += [int(v) for v in rng.integers(0, nrows, size=args.samples)]
    t2 = time.perf_counter()
    bad = 0
    for local in sample:
        gid = shard * ROWS_PER_SHARD + local
        if mt.row(gid) != direct_row(smap, gid):
            bad += 1
    t3 = time.perf_counter()
    print("抽样行数          : %d（含边界 0/1/2/3/%d + 随机；每行 320 B 与直读文件逐字节比对）"
          % (len(sample), nrows - 1))
    print("不匹配行数        : %d" % bad)
    print("抽样比对耗时      : %.3f s（host 墙钟）" % (t3 - t2))
    print("结论：%s" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def pack_bf16(x):
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    out = ((u + 0x7FFF + lsb) & np.uint64(0xFFFF0000)).astype(np.uint32)
    return (out >> 16).astype(np.uint16)


def _copy_small_tensors():
    src = HERE / "data"
    need = ["w_key_proj.bin", "w_value_proj.bin", "w_conv1d_tap.bin", "w_norm_key.bin",
            "w_norm_query.bin", "w_norm_conv.bin"]
    missing = [n for n in need if not (src / n).exists()]
    if missing:
        print("[emit] 缺 %s —— 先跑 gen_ple_data.py" % ", ".join(missing))
        return False
    import shutil
    for n in need:
        shutil.copyfile(src / n, DATA_HM / n)
    return True


def run_emit_multislot(args):
    """M171：多槽窗口池 + 跨分片窗口 的 harness 输入。

    与 legacy 单窗口 `run_emit` 的差别：窗口不是「一个连续区间」，而是 `--slots` 个**独立的
    全局行区间（槽）**，其中一个槽**横跨分片边界**（shard 0 尾 → shard 1 头，见 `plan_slots`）。
    ids 全部由池服务 —— 于是基线判据在「多槽 + 跨界」下应全绿；`ids_hm_miss.bin` 是越窗对照。
    """
    smap = shard_map()
    DATA_HM.mkdir(parents=True, exist_ok=True)
    if not _copy_small_tensors():
        return 2
    T = args.tokens
    NG = 16
    slot_rows = args.slot_mib * 1024 * 1024 // ROW_BYTES
    slots, pool_rows = plan_slots(args.slots, slot_rows, smap)
    pool_bytes = pool_rows * ROW_BYTES
    rng = np.random.default_rng(0x5A6B)
    # 固定 id：跨分片边界的两侧 + 每槽首行，全部由池服务
    fixed = [slots[0].base, slots[0].end - 1, ROWS_PER_SHARD - 1, ROWS_PER_SHARD,
             ROWS_PER_SHARD + 1]
    for s in slots[1:]:
        fixed += [s.base, s.base + s.rows // 2, s.end - 1]
    ids = list(fixed)
    while len(ids) < T * NG:
        ids.append(serve_id(slots, rng))
    ids = np.asarray(ids[:T * NG], dtype=np.int64).reshape(T, NG)
    ids.tofile(DATA_HM / "ids_hm.bin")

    exp = np.zeros((T * NG, ROW_BYTES), dtype=np.uint8)
    for i, gid in enumerate(ids.ravel()):
        exp[i] = np.frombuffer(direct_row(smap, int(gid)), dtype=np.uint8)
    exp.tofile(DATA_HM / "exp_rows.bin")
    exp[:, :128].tofile(DATA_HM / "exp64.bin")
    ids.ravel().astype(np.int64).tofile(DATA_HM / "ids_hm_flat.bin")

    # 越窗对照：第 0 个 token 的 16 个 id 换成**池外**的真实行 —— 取 shard 0 的头部，
    # 它落在槽 0 起点之前（槽 0 从 `ROWS_PER_SHARD - half` 起），其余槽分属别的分片 ⇒ 全部越窗。
    miss_base = 12345
    assert all(not s.contains(miss_base + k) for s in slots for k in range(NG))
    miss_ids = ids.copy()
    miss_ids[0, :] = [miss_base + k for k in range(NG)]
    miss_ids.tofile(DATA_HM / "ids_hm_miss.bin")

    hid = (rng.normal(0.0, 0.5, size=(T, W))).astype(np.float32)
    pack_bf16(hid).tofile(DATA_HM / "hidden_hm.bin")
    st = (rng.normal(0.0, 0.3, size=(2, STLEN, W))).astype(np.float32)
    st[0] = 0.0
    pack_bf16(st).tofile(DATA_HM / "conv_state_hm.bin")
    sidx = np.ones(T, dtype=np.int32)
    sidx[0] = -1
    sidx.tofile(DATA_HM / "state_idx_hm.bin")

    # slot 表（device 侧读；每槽 4×u32：shard / local_base / rows / row_off）
    slot_txt = ["n_slots %d" % len(slots), "rows_per_shard %d" % ROWS_PER_SHARD,
                "n_tok %d" % T, "ng %d" % NG, "pool_rows %d" % pool_rows,
                "pool_bytes %d" % pool_bytes]
    for i, s in enumerate(slots):
        slot_txt.append("slot %d %d %d %d %d" % (i, s.shard, s.local_base, s.rows, s.row_off))
    (DATA_HM / "hm_slots.txt").write_text("\n".join(slot_txt) + "\n")

    # 搬运行（host 侧分片算术；可跨分片拆成两段）
    segs = fill_segments(slots, smap)
    seg_txt = ["n_seg %d" % len(segs)]
    for dst, nb, f, bo in segs:
        seg_txt.append("seg %d %d %s %d" % (dst, nb, f, bo))
    (DATA_HM / "hm_fill.txt").write_text("\n".join(seg_txt) + "\n")

    bound_r0 = min(s.base for s in slots)
    bound_rows = max(s.end for s in slots) - bound_r0
    meta = {
        "mode": "multi-slot", "n_slots": len(slots),
        "slots": [{"shard": s.shard, "local_base": s.local_base, "rows": s.rows,
                   "row_off": s.row_off, "base": s.base, "end": s.end} for s in slots],
        "rows_per_shard": ROWS_PER_SHARD, "row_bytes": ROW_BYTES,
        "r0": bound_r0, "win_rows": pool_rows, "win_bytes": pool_bytes,
        "pool_rows": pool_rows, "bound_rows": bound_rows,
        "tokens": T, "items": int(T * NG), "ng": NG, "ids_fixed": [int(v) for v in fixed],
        "n_seg": len(segs),
        "note": "多槽：槽 0 横跨 shard 0→1；ids 全部由池服务；ids_hm_miss.bin 的 token0 是池外对照",
    }
    (DATA_HM / "hm_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (DATA_HM / "hm_meta.txt").write_text(
        "mode multi-slot\nn_slots %d\nrows_per_shard %d\nr0 %d\nwin_rows %d\nwin_bytes %d\n"
        "n_tok %d\nng %d\npool_rows %d\n"
        % (len(slots), ROWS_PER_SHARD, bound_r0, pool_rows, pool_bytes, T, NG, pool_rows))
    print("[emit] multi-slot: slots=%d slot_rows=%d pool=%d rows (%d B) shard0_cross=%d->%d segs=%d"
          % (len(slots), slot_rows, pool_rows, pool_bytes, 0, 1, len(segs)))
    print("[emit] %s: T=%d items=%d bound=[%d,%d)" % (DATA_HM, T, T * NG, bound_r0, bound_r0 + bound_rows))
    for f in sorted(DATA_HM.iterdir()):
        print("        %-22s %d B" % (f.name, f.stat().st_size))
    return 0


def run_emit(args):
    """生成设备 harness 的输入：真实行 id / 期望行首 64 元素 / 窗口元数据 / HM 侧小输入。"""
    if getattr(args, "slots", 0) > 0:
        return run_emit_multislot(args)
    smap = shard_map()
    DATA_HM.mkdir(parents=True, exist_ok=True)
    # PLE 的真实小张量（key/value/conv1d/3×norm）从 ple/data/ 拷来：HM 案例只用 ② 的真表，
    # ③④⑤ 的权重与缩减表案例完全相同，不重复生成（gen_ple_data.py 是它们的唯一来源）。
    src = HERE / "data"
    need = ["w_key_proj.bin", "w_value_proj.bin", "w_conv1d_tap.bin", "w_norm_key.bin",
            "w_norm_query.bin", "w_norm_conv.bin"]
    missing = [n for n in need if not (src / n).exists()]
    if missing:
        print("[emit] 缺 %s —— 先跑 gen_ple_data.py" % ", ".join(missing))
        return 2
    import shutil
    for n in need:
        shutil.copyfile(src / n, DATA_HM / n)
    shard = args.shard
    win_bytes = args.win_mib * 1024 * 1024
    win_rows = win_bytes // ROW_BYTES
    r0 = shard * ROWS_PER_SHARD
    T = args.tokens
    NG = 16

    # 真实行 id（全局 id，全部落在窗口 [r0, r0+win_rows) 内）
    fixed = [0, 1, 2, 3, 4, 5, 12345, 999983, ROWS_PER_SHARD - 1]
    rng = np.random.default_rng(0x5A6B)
    pool = [int(v) for v in rng.integers(0, min(win_rows, ROWS_PER_SHARD), size=T * NG)]
    ids = list(fixed)
    while len(ids) < T * NG:
        ids.append(pool[len(ids) % len(pool)] if len(ids) >= len(pool) else pool[len(ids)])
    ids = ids[:T * NG]
    ids = np.asarray(ids, dtype=np.int64).reshape(T, NG)
    ids.tofile(DATA_HM / "ids_hm.bin")

    # 期望行字节：**直接读分片文件**（与窗口无关的独立来源）
    exp = np.zeros((T * NG, ROW_BYTES), dtype=np.uint8)
    for i, gid in enumerate(ids.ravel()):
        exp[i] = np.frombuffer(direct_row(smap, int(gid)), dtype=np.uint8)
    exp.tofile(DATA_HM / "exp_rows.bin")                       # 全行（320 B），供设备侧 row 校验（前 64 元素）
    exp[:, :128].tofile(DATA_HM / "exp64.bin")                 # 前 64 个 bf16 = 128 B
    ids_flat = ids.ravel().astype(np.int64)
    ids_flat.tofile(DATA_HM / "ids_hm_flat.bin")

    # 越窗行 id（用于设备侧 miss 计数器：真实 id，但落在窗口之外）
    miss_ids = ids.copy()
    miss_ids[0, :] = [r0 + win_rows + 100 + k for k in range(NG)]
    miss_ids.tofile(DATA_HM / "ids_hm_miss.bin")

    # HM 侧小输入（T 行）：hidden / conv_state（2 槽）/ state_idx
    hid = (rng.normal(0.0, 0.5, size=(T, W))).astype(np.float32)
    pack_bf16(hid).tofile(DATA_HM / "hidden_hm.bin")
    st = (rng.normal(0.0, 0.3, size=(2, STLEN, W))).astype(np.float32)
    st[0] = 0.0
    pack_bf16(st).tofile(DATA_HM / "conv_state_hm.bin")
    sidx = np.ones(T, dtype=np.int32)
    sidx[0] = -1                       # 第 0 行留一个 null 槽位（NULL_STATE_ID 路径也要在真实表上跑）
    sidx.tofile(DATA_HM / "state_idx_hm.bin")

    meta = {
        "shard": shard, "file": smap[shard]["file"], "data_off": smap[shard]["data_off"],
        "rows_per_shard": ROWS_PER_SHARD, "row_bytes": ROW_BYTES,
        "r0": r0, "win_rows": int(min(win_rows, ROWS_PER_SHARD)), "win_bytes": win_bytes,
        "tokens": T, "items": int(T * NG), "ng": NG,
        "ids_fixed": fixed,
        "note": "ids 全部落在 [r0, r0+win_rows)；窗口外行数 = 0（ids_hm_miss.bin 是越窗对照）",
    }
    (DATA_HM / "hm_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    # 设备 harness 用的极简 key/value 文本（C++ 侧逐行 sscanf）
    (DATA_HM / "hm_meta.txt").write_text(
        "shard %d\nrows_per_shard %d\nr0 %d\nwin_rows %d\nwin_bytes %d\nn_tok %d\nng %d\n"
        "file %s\ndata_off %d\n"
        % (shard, ROWS_PER_SHARD, r0, meta["win_rows"], win_bytes, T, NG,
           smap[shard]["file"], smap[shard]["data_off"]))
    print("[emit] %s: T=%d items=%d win=%d MiB (%d rows) shard=%d"
          % (DATA_HM, T, T * NG, args.win_mib, meta["win_rows"], shard))
    for f in sorted(DATA_HM.iterdir()):
        print("        %-22s %d B" % (f.name, f.stat().st_size))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--real-scale", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--win-mib", type=int, default=WIN_MIB_DEFAULT)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--slots", type=int, default=0,
                    help="M171 多槽窗口池：槽数（0 = legacy 单窗口；槽 0 横跨 shard 0→1）")
    ap.add_argument("--slot-mib", type=int, default=SLOT_MIB_DEFAULT, help="每槽 staging 字节数（MiB）")
    args = ap.parse_args()
    if args.evidence:
        return run_evidence(args)
    if args.real_scale:
        return run_real_scale(args)
    if args.emit:
        return run_emit(args)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
