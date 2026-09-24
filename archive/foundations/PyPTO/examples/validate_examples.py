#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO Examples 验证执行器

由 build_ci.py --example 调用。

    python3 examples/validate_examples.py -t examples/ -d 0
    python3 examples/validate_examples.py -t examples/ --run_mode sim -w 4
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SKIP_FILES = {"validate_examples.py", "__init__.py"}


def _collect_from_dir(target_dir):
    return [f.resolve() for f in sorted(target_dir.rglob("*.py"))
            if f.name not in SKIP_FILES]


def collect_scripts(paths):
    scripts = []
    for p in paths:
        target = Path(p).resolve()
        if target.is_file() and target.suffix == ".py" and target.name not in SKIP_FILES:
            scripts.append(target)
        elif target.is_dir():
            scripts.extend(f for f in _collect_from_dir(target) if f not in scripts)
    return scripts


def has_run_mode(script):
    try:
        text = script.read_text(encoding="utf-8")
        return "'--run_mode'" in text or '"--run_mode"' in text
    except Exception:
        return False


def run_script(script, run_mode, device_id, timeout):
    env = os.environ.copy()
    env["TILE_FWK_DEVICE_ID"] = device_id
    if has_run_mode(script):
        cmd = [sys.executable, str(script), "--run_mode", run_mode]
    else:
        cmd = [sys.executable, str(script)]

    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        return {"script": script, "rc": r.returncode, "dur": time.time() - t0, "stderr": r.stderr}
    except subprocess.TimeoutExpired:
        return {"script": script, "rc": -1, "dur": time.time() - t0, "stderr": f"TIMEOUT after {timeout}s"}


def _log_result(script, rc, dur, stderr=None):
    rel = script.relative_to(Path.cwd()) if script.is_relative_to(Path.cwd()) else script
    status = "PASS" if rc == 0 else "FAIL"
    logger.info("  [%s] %s (%.1fs)", status, rel, dur)
    if rc != 0 and stderr:
        for line in stderr.strip().splitlines()[-5:]:
            logger.error("        %s", line)


def _run_with_device(script, run_mode, device_queue, timeout, results):
    """从队列获取设备 → 执行 → 归还设备。"""
    dev = device_queue.get()
    try:
        r = run_script(script, run_mode, dev, timeout)
        results.append(r)
        _log_result(r["script"], r["rc"], r["dur"], r["stderr"] if r["rc"] != 0 else None)
    finally:
        device_queue.put(dev)


def _execute_npu(scripts, devices, run_mode, timeout):
    """NPU 多卡并行，设备租赁式调度。"""
    device_queue = queue.Queue()
    for d in devices:
        device_queue.put(d)

    results = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(_run_with_device, s, run_mode, device_queue, timeout, results)
                   for s in scripts]
        for fut in as_completed(futures):
            fut.result()
    return results


def _execute_sim(scripts, run_mode, timeout, workers):
    """SIM 模式并行，不限设备。"""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_script, s, run_mode, "0", timeout): s for s in scripts}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            _log_result(r["script"], r["rc"], r["dur"], r["stderr"] if r["rc"] != 0 else None)
    return results


def main():
    parser = argparse.ArgumentParser(description="PyPTO Examples Validator")
    parser.add_argument("-t", "--test_path", nargs="+", default=["examples/"])
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("-m", "--run_mode", choices=["npu", "sim"], default="npu")
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    scripts = collect_scripts(args.test_path)
    if not scripts:
        logger.warning("No example scripts found.")
        return

    devices = [d.strip() for d in args.device.split(",")]
    logger.info("Found %d example(s), mode=%s, device=%s", len(scripts), args.run_mode, args.device)

    if args.run_mode == "sim":
        results = _execute_sim(scripts, args.run_mode, args.timeout, args.workers or 16)
    else:
        results = _execute_npu(scripts, devices, args.run_mode, args.timeout)

    passed = sum(1 for r in results if r["rc"] == 0)
    failed = len(results) - passed
    logger.info("Result: %d passed, %d failed, %d total", passed, failed, len(results))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
