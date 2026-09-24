#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Performance metrics baseline comparison tool.

Save / compare per-op-type longest-path compute amounts (amountLp) against a baseline file.
When any op type's amountLp value increases beyond the baseline, a warning is printed.

CLI Usage:
    python3 perf_baseline.py save   <case_name> <metrics_json_path> [baseline_file]
    python3 perf_baseline.py check  <case_name> <metrics_json_path> [baseline_file]
    python3 perf_baseline.py update <case_name> <metrics_json_path> [baseline_file]
    python3 perf_baseline.py list   [baseline_file]

Python Usage:
    from perf_baseline import save_baseline, check_baseline
    save_baseline("attention", "/path/to/_simulate.perf.metrics.json")
    regressions = check_baseline("attention", "/path/to/_simulate.perf.metrics.json")
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("perf_baseline")

_RegressionList = List[Tuple[str, int, int, int]]

DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "perf_baseline.json"
)


def load_metrics(metrics_path: str) -> Optional[Dict]:
    with open(metrics_path, "r") as f:
        return json.load(f)


def extract_amount_lp(data: Dict) -> Dict[str, int]:
    result: Dict[str, int] = {}
    tile_op_paths = data.get("summary", {}).get("tileOpLongestPaths", {})
    for op_name, entry in tile_op_paths.items():
        result[op_name] = entry.get("amountLp", 0)
    return result


def load_baseline(baseline_path: str) -> Dict:
    if not os.path.exists(baseline_path):
        return {}
    with open(baseline_path, "r") as f:
        return json.load(f)


def save_baseline_file(baseline_path: str, baseline_data: Dict) -> None:
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f, indent=2)
        f.write("\n")


def compare_amount_lp(
    baseline: Dict[str, int],
    current: Dict[str, int],
) -> _RegressionList:
    regressions = []
    all_ops = sorted(set(list(baseline.keys()) + list(current.keys())))
    for op in all_ops:
        base_val = baseline.get(op, 0)
        curr_val = current.get(op, 0)
        delta = curr_val - base_val
        if delta > 0:
            regressions.append((op, base_val, curr_val, delta))
    return regressions


def save_baseline(case_name: str, metrics_path: str, baseline_path: str = DEFAULT_BASELINE_PATH) -> str:
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    data = load_metrics(metrics_path)
    amount_lp = extract_amount_lp(data)
    baseline_data = load_baseline(baseline_path)
    baseline_data[case_name] = {"amountLp": amount_lp}
    save_baseline_file(baseline_path, baseline_data)
    return metrics_path


def check_baseline(
    case_name: str,
    metrics_path: str,
    baseline_path: str = DEFAULT_BASELINE_PATH,
) -> _RegressionList:
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    data = load_metrics(metrics_path)
    current_amount_lp = extract_amount_lp(data)
    baseline_data = load_baseline(baseline_path)
    if case_name not in baseline_data:
        raise KeyError(f"Case '{case_name}' not found in baseline. Run `save` first.")
    baseline_amount_lp = baseline_data[case_name]["amountLp"]
    regressions = compare_amount_lp(baseline_amount_lp, current_amount_lp)
    if regressions:
        logger.warning("REGRESSION detected for '%s':", case_name)
        for op, base_val, curr_val, delta in regressions:
            pct = (delta / base_val * 100) if base_val > 0 else float("inf")
            logger.warning("  %s: %s -> %s  (+%s, +%.1f%%)", op, base_val, curr_val, delta, pct)
    else:
        logger.info("'%s' passed: no amountLp increase.", case_name)
    return regressions


def update_baseline(case_name: str, metrics_path: str, baseline_path: str = DEFAULT_BASELINE_PATH) -> str:
    return save_baseline(case_name, metrics_path, baseline_path)


def list_baseline(baseline_path: str = DEFAULT_BASELINE_PATH) -> None:
    baseline_data = load_baseline(baseline_path)
    if not baseline_data:
        logger.warning("No baseline file at %s", baseline_path)
        return
    logger.info("Cases in %s:", baseline_path)
    for case_name, entry in sorted(baseline_data.items()):
        amount_lp = entry.get("amountLp", {})
        nonzero = {k: v for k, v in amount_lp.items() if v > 0}
        logger.info("  %s: amountLp=%s", case_name, nonzero)


def main():
    if len(sys.argv) < 2:
        logger.info(
            "Usage:\n"
            "  python3 perf_baseline.py save   <case_name> <metrics_json_path> [baseline_file]\n"
            "  python3 perf_baseline.py check  <case_name> <metrics_json_path> [baseline_file]\n"
            "  python3 perf_baseline.py update <case_name> <metrics_json_path> [baseline_file]\n"
            "  python3 perf_baseline.py list   [baseline_file]"
        )
        sys.exit(1)

    action = sys.argv[1]
    if action == "list":
        bp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASELINE_PATH
        list_baseline(bp)
        return

    if len(sys.argv) < 4:
        logger.error("Error: save/check/update require <case_name> <metrics_json_path>")
        sys.exit(1)

    case_name = sys.argv[2]
    metrics_path = sys.argv[3]
    baseline_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_BASELINE_PATH

    if action == "save":
        path = save_baseline(case_name, metrics_path, baseline_path)
        logger.info("Saved '%s' from %s", case_name, path)
        logger.info("Baseline file: %s", baseline_path)
    elif action == "check":
        regressions = check_baseline(case_name, metrics_path, baseline_path)
        if regressions:
            sys.exit(2)
    elif action == "update":
        path = update_baseline(case_name, metrics_path, baseline_path)
        logger.info("Updated '%s' from %s", case_name, path)
        logger.info("Baseline file: %s", baseline_path)
    else:
        logger.error("Unknown action: %s", action)
        sys.exit(1)


if __name__ == "__main__":
    main()
