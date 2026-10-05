#!/usr/bin/env python3
"""mmap_ngram_plan.py —— M86：ngram 权重「不整体读入内存」的 host 侧 mmap 方案 + 可跑的参考实现

背景（数字全部复核过，见 README §5）：
  ngram 表 = 128 分片 × 2,500,012 行 × 160 列 × 2 B = 102,400,491,520 B = 102.40 GB = 95.37 GiB
  - 装不进 HBM（128 GiB 里还要放别的）—— 更关键：装不进本容器 host 内存（cgroup 32 GiB）
  - 磁盘 ~402 GB 可用 ⇒ 表能常驻磁盘，但**不能**常驻 RAM

本脚本做两件事：
  A. `--real`  ：按真实几何打印方案（换算、窗口大小、驻留预算），不落盘 102 GB
  B. `--demo`  ：用小几何（默认 8 分片 × 1000 行）**真的落盘 → mmap → 滑窗 staging → 逐字节校验**，
                 证明「行 id → (分片, 文件偏移) → 窗口搬运」这条链路可跑通且逐字节正确。

与设备侧的接口（由 probe_host_dma 实测，不是本脚本的猜测）：
  * 设备 kernel 读的**不是** file-backed mmap（实测 aclrtHostRegisterV2 对 file mmap 报 rc=507899，
    file/anon/malloc 三种 mapmode 的对照见 README §4.2）；
  * 设备 kernel 读的是**注册过的匿名 staging 窗口**（mmap anon / posix_memalign / aclrtMallocHost
    三种都实测 rc=0 且逐字节读对，~22 GB/s）。
  因此本脚本的 staging 池就是「要被 ACL 注册一次、之后反复复用」的那块内存。
"""

import argparse
import json
import mmap
import os
import random
import struct
import sys
from collections import OrderedDict

# ---------------------------------------------------------------- 真实几何
SHARDS = 128
ROWS_PER_SHARD = 2_500_012
COLS = 160
ELEM_BYTES = 2                      # bf16 / fp16
ROW_BYTES = COLS * ELEM_BYTES        # 320 B
TOTAL_ROWS = SHARDS * ROWS_PER_SHARD       # 320,001,536
TOTAL_BYTES = TOTAL_ROWS * ROW_BYTES       # 102,400,491,520

CGROUP_BYTES = 32 * 1024**3         # 本容器实测 memory.max = 34359738368
DISK_FREE_BYTES = 402 * 1000**3     # 测得时快照：/workspace、/ 上 Avail ~402G（会随时间变，仅作量级参考）


def row_bytes_value(global_row, j):
    """合成权重模型：给定全局行 id 与行内字节下标，给出确定值（demo 用）。"""
    return (global_row * 131 + j * 7 + (global_row >> 5)) & 0xFF


# ---------------------------------------------------------------- safetensors
def write_safetensors(path, rows, row_bytes, base_row, payload_fn):
    """写一个最小合法 safetensors：8B 头长 + JSON 头 + 数据。

    payload_fn(global_row, j) -> 行内第 j 字节。数据按行连续。
    """
    hdr = {"ngram_embedding.weight": {"dtype": "BF16", "shape": [rows, row_bytes // 2],
                                      "data_offsets": [0, rows * row_bytes]}}
    hdr_bytes = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(hdr_bytes)) % 8
    hdr_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        data_off = f.tell()
        chunk = bytearray()
        for r in range(rows):
            g = base_row + r
            for j in range(row_bytes):
                chunk.append(payload_fn(g, j))
            if len(chunk) >= 1 << 20:
                f.write(chunk)
                chunk = bytearray()
        if chunk:
            f.write(chunk)
    return data_off


def read_safetensors_data_offset(path):
    """从 safetensors 头里取数据段起点（= 8 + 对齐后的头长）。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        _ = f.read(n)
        return f.tell()


# ---------------------------------------------------------------- 映射与 staging
class NgramStore:
    """全局行 id → (分片, 文件偏移)；可选滑窗 staging（模拟「注册窗口 + 按需搬运」）。"""

    def __init__(self, shard_paths, rows_per_shard, row_bytes, window_bytes, n_windows):
        self.row_bytes = row_bytes
        self.rows_per_shard = rows_per_shard
        self.window_bytes = window_bytes
        self.paths = list(shard_paths)
        self.fds = [os.open(p, os.O_RDONLY) for p in self.paths]
        self.data_off = [read_safetensors_data_offset(p) for p in self.paths]
        self.sizes = [os.fstat(fd).st_size for fd in self.fds]
        # 模拟「一次性注册、之后复用」的匿名 staging 窗口池（设备侧由 ACL 注册）
        self.windows = [bytearray(window_bytes) for _ in range(n_windows)]
        self.win_key = [None] * n_windows          # 当前窗口覆盖的 (shard, block_idx)
        self.lru = OrderedDict()                   # (shard, block_idx) -> window slot
        self.hits = 0
        self.fills = 0
        self.bytes_moved = 0

    # --- 行 id → 位置（跨 128 分片）---
    def locate(self, global_row):
        shard = global_row // self.rows_per_shard
        local = global_row % self.rows_per_shard
        return shard, local

    def file_offset(self, global_row):
        shard, local = self.locate(global_row)
        return shard, self.data_off[shard] + local * self.row_bytes

    def _fetch_block(self, shard, block_idx):
        """把 (shard, block_idx) 这一个窗口从文件搬进 staging 槽。"""
        key = (shard, block_idx)
        if key in self.lru:
            self.lru.move_to_end(key)
            self.hits += 1
            return self.lru[key]
        slot = self.lru.popitem(last=False)[1] if len(self.lru) == len(self.windows) else \
            next(s for s in range(len(self.windows)) if s not in self.lru.values())
        start = self.data_off[shard] + block_idx * self.window_bytes
        n = min(self.window_bytes, self.sizes[shard] - start)
        with open(self.paths[shard], "rb") as f:     # 等价于 mmap 后的 memcpy
            f.seek(start)
            self.windows[slot][:n] = f.read(n)
        self.lru[key] = slot
        self.fills += 1
        self.bytes_moved += n
        return slot

    def row(self, global_row):
        shard, off = self.file_offset(global_row)
        block_idx = (off - self.data_off[shard]) // self.window_bytes
        slot = self._fetch_block(shard, block_idx)
        in_win = (off - self.data_off[shard]) % self.window_bytes
        return bytes(self.windows[slot][in_win:in_win + self.row_bytes])


# ---------------------------------------------------------------- 模式
def read_regbench(evidence_dir):
    """从 evidence/logs/regbench*.log **解析**注册代价（不硬编码任何"实测值"）。

    返回 {size_mb: {'median': ms, 'min': ms, 'max': ms, 'n': k}}；无证据则返回空 dict，
    调用方必须如实显示 NO-EVIDENCE。
    """
    import glob
    import re
    samples = {}
    pat = re.compile(r"\[REGBENCH\] size=\s*(\d+)\s*MB register rc=\d+\s+([0-9.]+) ms")
    for f in sorted(glob.glob(os.path.join(evidence_dir, "regbench*.log"))):
        try:
            with open(f, "r", errors="replace") as fh:
                for line in fh:
                    m = pat.search(line)
                    if m:
                        samples.setdefault(int(m.group(1)), []).append(float(m.group(2)))
        except OSError:
            continue
    out = {}
    for sz, vals in samples.items():
        vals.sort()
        out[sz] = {"median": vals[(len(vals) - 1) // 2] if len(vals) % 2 else
                   (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2,
                   "min": vals[0], "max": vals[-1], "n": len(vals)}
    return out


def run_real(args):
    print("== A. 真实几何方案（128 分片 / 2,500,012 行 / 160 列 / 2 B）==")
    print(f"全局行数            : {TOTAL_ROWS:,}")
    print(f"行宽                : {ROW_BYTES} B")
    print(f"表总大小            : {TOTAL_BYTES:,} B = {TOTAL_BYTES/1e9:.2f} GB = {TOTAL_BYTES/2**30:.2f} GiB")
    print(f"单分片数据段        : {ROWS_PER_SHARD*ROW_BYTES:,} B = {ROWS_PER_SHARD*ROW_BYTES/2**20:.1f} MiB")
    print(f"每 token 查 16 行   : {16*ROW_BYTES} B")
    print()
    print("-- 行 id → 文件位置（跨分片）--")
    print("  shard      = id // 2,500,012           # 0..127")
    print("  local      = id %  2,500,012           # 分片内行号")
    print("  file_off   = data_off[shard] + local*320   # data_off 从该分片 safetensors 头解析（8B 头长 + JSON）")
    print()
    print("-- 为什么必须 mmap（且不能物化）--")
    print(f"  host cgroup             : {CGROUP_BYTES/2**30:.0f} GiB（实测 memory.max={CGROUP_BYTES}）")
    print(f"  表 / cgroup             : {TOTAL_BYTES/CGROUP_BYTES:.2f}x  ⇒ 整表**装不进**（差 {TOTAL_BYTES/CGROUP_BYTES:.1f} 倍）")
    print(f"  磁盘可用                : {DISK_FREE_BYTES/1e9:.0f} GB ⇒ 整表 {TOTAL_BYTES/1e9:.1f} GB 放得下（占 {TOTAL_BYTES/DISK_FREE_BYTES*100:.0f}%）")
    print()
    print("-- 建议的驻留/窗口策略（依据 probe_host_dma 的实测代价）--")
    w = args.window_mib
    n = args.windows
    print(f"  窗口大小 {w} MiB × {n} 个 = {w*n} MiB 注册内存（约 {w*n/1024:.2f} GiB，占 cgroup {w*n/1024/(CGROUP_BYTES/2**30)*100:.1f}%）")

    rb = read_regbench(args.evidence)
    if rb:
        parts = []
        for sz in sorted(rb):
            d = rb[sz]
            parts.append(f"{sz}MB median={d['median']:.3f}ms[n={d['n']},range={d['min']:.3f}..{d['max']:.3f}]")
        print(f"  依据① 注册代价（从 evidence/logs/regbench*.log 解析，跨独立进程中位+区间）：")
        print(f"         " + " / ".join(parts))
        # 用实测中位做池启动代价的线性粗估（明确标注为推算）
        val = []
        for sz in sorted(rb):
            d = rb[sz]
            val.append(f"{n}×{sz}MiB≈[{n*d['min']/1000:.1f}, {n*d['max']/1000:.1f}]s(中位{n*d['median']/1000:.1f}s)")
        print(f"        ⇒ 窗口要**一次注册、反复复用**，绝不按次注册")
        print(f"        ⇒ 池启动注册耗时（按上表线性外推，**推算、未端到端实测；且方差极大**）：{'  '.join(val)}")
        print(f"           ⚠ 方差远大于尺寸效应：注册耗时的跨批次离散可达**一个数量级到数十倍**（见 README §3.6 跨批次表）")
        print(f"             ⇒ **池大小必须在目标机上实测确定**，不要照搬上面的外推值")
    else:
        print(f"  依据① 注册代价：**NO-EVIDENCE**（{args.evidence}/regbench*.log 未找到或未解析到）")
        print(f"        ⇒ 仍成立的定性结论：注册有实打实的代价且随 size 增长 ⇒ 一次注册、反复复用，绝不按次注册")
    print(f"  依据② 注册本身相对 touch 不额外增常驻（RSS delta=0 kB；逐条见 README §3.5）")
    print(f"        ⇒ 窗口池的常驻成本 ≈ 池大小 {w*n} MiB；cgroup {CGROUP_BYTES/2**30:.0f} GiB 下要留足余量")
    print(f"  依据③ device 单侧读带宽约 20–22.5 GB/s（host-mapped，由 evidence/logs/read_run*.log 复核）")
    print(f"        ⇒ 搬 {w} MiB 进 staging 约 {w/1024/22*1000:.1f}–{w/1024/20*1000:.1f} ms 量级（另取决于是否命中 page cache）")
    print()
    print("-- 必须避免的两条（错误码为稳定契约，非计时类测量）--")
    print("  1) 直接注册 file-backed mmap 让设备读：aclrtHostRegisterV2 rc=507899（file/filerd/filepriv 全拒；")
    print("     README §3.5 表 + evidence/logs/filemmap_*.log，跨 3 批稳定）")
    print("  2) 未注册指针 / 越界让设备读：kernel aclError=507035（README §3.4 + evidence/logs/{unreg,page}_*.log）")


def run_demo(args):
    tmp = args.dir
    os.makedirs(tmp, exist_ok=True)
    rows = args.rows
    shard_paths = []
    print(f"== B. demo（{args.shards} 分片 × {rows} 行 × {ROW_BYTES} B）建表 + mmap/staging 校验 ==")
    for s in range(args.shards):
        p = os.path.join(tmp, f"shard_{s:03d}.safetensors")
        base = s * rows
        write_safetensors(p, rows, ROW_BYTES, base, row_bytes_value)
        shard_paths.append(p)
    tot = sum(os.path.getsize(p) for p in shard_paths)
    print(f"落盘 {len(shard_paths)} 个分片，共 {tot:,} B（= 每表数据段 {rows*ROW_BYTES:,} B + 头）")

    store = NgramStore(shard_paths, rows, ROW_BYTES,
                       window_bytes=args.window_kib * 1024, n_windows=args.windows)
    total_rows = args.shards * rows
    rng = random.Random(0xC0FFEE)
    sample = list(range(min(total_rows, 64))) + [rng.randrange(total_rows) for _ in range(args.sample)]
    bad = 0
    for g in sample:
        got = store.row(g)
        exp = bytes(row_bytes_value(g, j) for j in range(ROW_BYTES))
        if got != exp:
            bad += 1
            if bad <= 3:
                print(f"  [FAIL] global_row={g} mismatch")
    print(f"抽样行数            : {len(sample)}（含边界 0..{min(total_rows,64)-1} + 随机）逐字节比对")
    print(f"不匹配行数          : {bad}")
    print(f"staging 命中/填充   : hits={store.hits} fills={store.fills} bytes_moved={store.bytes_moved:,}")
    print(f"staging 池大小      : {args.windows} × {args.window_kib} KiB = {args.windows*args.window_kib/1024:.2f} MiB（常驻）")
    print("结论               : " + ("PASS —— 行 id→分片→文件偏移→窗口搬运→逐字节取数 全链路一致"
                                    if bad == 0 else f"FAIL —— {bad} 行不符"))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--dir", default="/tmp/m86_mmap_demo")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--window-kib", type=int, default=64)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--sample", type=int, default=256)
    ap.add_argument("--window-mib", type=int, default=256)
    ap.add_argument("--evidence", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence", "logs"),
                    help="regbench*.log 所在目录（默认 <script_dir>/evidence/logs）；--real 从这里解析注册代价")
    args = ap.parse_args()
    if args.real:
        run_real(args)
        return 0
    if args.demo:
        return run_demo(args)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
