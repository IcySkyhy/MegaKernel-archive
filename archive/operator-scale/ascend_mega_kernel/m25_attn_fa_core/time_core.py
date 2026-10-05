#!/usr/bin/env python3
"""time_core.py —— 量 `m25_attn_fa_core` 的**设备核墙钟**，不新增任何探针/同步。

被测量的是**既有** harness 的 stdout：binary 在 `m25_fa_kernel<<<...>>>` 之后（`launched,
waiting sync`）与 `aclrtSynchronizeStream` 返回之后（`sync done`）各 `fflush` 一次
（`m15_attn_fa_core_host.h`）。本工具只读这两条**已存在**的打印、用 Python 逐行接收并打时间戳，
于是 `kernel_ms = sync_done - launched` 不受 shell `while read` 子进程开销污染
（后者每个 `date` 子进程约 1 ms，会把「核时间」虚报成循环开销）。

用法：
  python3 time_core.py <binary> [mode] [m] [outdir]
返回码 = binary 的返回码（超时/失败向上传播）。
"""
import os
import subprocess
import sys
import time


def run(binary, mode, m, outdir):
    env = dict(os.environ)
    env['M25FA_M'] = str(m)
    env['M25FA_OUT'] = outdir
    t0 = time.time()
    p = subprocess.Popen([binary, mode], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, bufsize=1)
    launched = syncdone = None
    for line in p.stdout:
        t = time.time()
        if 'launched, waiting sync' in line:
            launched = t
        elif 'sync done' in line:
            syncdone = t
    rc = p.wait()
    t1 = time.time()
    kms = (syncdone - launched) * 1000.0 if (launched is not None and syncdone is not None) else float('nan')
    print(f"[time_core] mode={mode} m={m} kernel_ms={kms:.3f} total_s={t1 - t0:.3f} rc={rc}")
    return rc


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: time_core.py <binary> [mode] [m] [outdir]")
        sys.exit(2)
    binary = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'core'
    m = sys.argv[3] if len(sys.argv) > 3 else '4097'
    outdir = sys.argv[4] if len(sys.argv) > 4 else '/tmp/m196_time'
    sys.exit(run(binary, mode, m, outdir))
