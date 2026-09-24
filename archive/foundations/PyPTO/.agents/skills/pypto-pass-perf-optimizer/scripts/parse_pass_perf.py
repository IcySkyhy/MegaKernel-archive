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
"""
Pass Performance Analysis Script

This script parses log files to extract Pass execution time and generates
performance analysis reports.

Supports:
- Single log file analysis
- Automatic detection and parsing of split log files (when file size > 20M)
- Multiple execution statistics (count, average, total)
- Comparison between two log files

Log file format:
- Main file: pypto-log-{pid}-{timestamp}.log
- Split files: pypto-log-{pid}-{timestamp}.log (each has unique timestamp)
- Files with same pid belong to the same execution

Usage:
    python3 parse_pass_perf.py -l /path/to/pypto-log-*.log
    python3 parse_pass_perf.py -l after.log --compare before.log
"""

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import glob
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


def setup_logger(log_dir: str = "./perf_logs") -> logging.Logger:
    """Setup logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"pass_perf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")

    lg = logging.getLogger("PassPerfAnalyzer")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    lg.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    lg.addHandler(console_handler)

    lg.info(f"Log file: {log_filename}")

    return lg


_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def fmt_dur(seconds: float) -> str:
    """Format duration with adaptive unit for readability.

    Avoids misleading '0.000' display for sub-millisecond passes.

    < 1ms  -> 'NNNus'    (e.g. 0.000018s -> '18us')
    < 1s   -> 'N.NNms'   (e.g. 0.000277s -> '0.28ms')
    >= 1s  -> 'NN.NNNs'  (e.g. 77.189s   -> '77.189s')
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


@dataclass
class PassPerfData:
    """Pass performance data with execution count support"""

    name: str
    durations: List[int] = field(default_factory=list)
    ops: int = 0
    durations_before: List[int] = field(default_factory=list)
    total_pass_time: int = 0

    @property
    def count(self) -> int:
        """Number of executions"""
        return len(self.durations)

    @property
    def duration_us(self) -> int:
        """Total execution time (us)"""
        return sum(self.durations)

    @property
    def avg_duration_us(self) -> float:
        """Average execution time (us)"""
        return self.duration_us / self.count if self.count > 0 else 0

    @property
    def duration_s(self) -> float:
        """Total execution time (s)"""
        return self.duration_us / 1_000_000

    @property
    def avg_duration_s(self) -> float:
        """Average execution time (s)"""
        return self.avg_duration_us / 1_000_000

    @property
    def avg_ops(self) -> int:
        """Average operations per execution"""
        return self.ops

    @property
    def time_percent(self) -> float:
        """Time percentage of total pass time"""
        if self.total_pass_time > 0:
            return (self.duration_us / self.total_pass_time) * 100
        return 0.0

    @property
    def us_per_op(self) -> float:
        """Time per operation (based on total time)"""
        return self.duration_us / self.ops if self.ops > 0 else 0

    @property
    def duration_us_before(self) -> int:
        """Total execution time before optimization"""
        return sum(self.durations_before) if self.durations_before else 0

    @property
    def avg_duration_us_before(self) -> float:
        """Average execution time before optimization"""
        count_before = len(self.durations_before)
        return self.duration_us_before / count_before if count_before > 0 else 0

    @property
    def improvement_percent(self) -> float:
        """Performance improvement percentage (based on total time)"""
        if self.duration_us_before > 0:
            return (self.duration_us_before - self.duration_us) / self.duration_us_before * 100
        return 0.0


class PassPerfAnalyzer:
    """Pass performance analyzer"""

    PASS_TIME_PATTERN = re.compile(r'The Runtime of pass (\S+) for program\s+function \S+ is (\d+) us\.')

    OP_COUNT_PATTERN = re.compile(r'Function\[([^\]]+)\] operation size is: (\d+) after expansion\.')

    def __init__(self, time_threshold_s: float = 20.0, ops_threshold: int = 200000, quiet_mode: bool = False):
        self.time_threshold_s = time_threshold_s
        self.ops_threshold = ops_threshold
        self.quiet_mode = quiet_mode
        self.pass_data: Dict[str, PassPerfData] = OrderedDict()
        self.function_ops: Dict[str, int] = {}
        self.total_ops = 0
        self.total_pass_time = 0

    @staticmethod
    def find_split_log_files(log_file: str) -> List[str]:
        """
        Find all split log files for a given log file based on pid.

        Split log files are created when a single log file exceeds 20M.
        Each split file has the format: pypto-log-{pid}-{timestamp}.log
        Files with the same pid belong to the same execution.

        Args:
            log_file: Path to the main log file

        Returns:
            List of log files sorted by filename (timestamp)
        """
        log_dir = os.path.dirname(log_file)
        filename = os.path.basename(log_file)

        match = re.match(r'pypto-log-(\d+)_\d+\.log', filename)

        if not match:
            return [log_file] if os.path.exists(log_file) else []

        pid = match.group(1)

        pattern = os.path.join(log_dir, f"pypto-log-{pid}_*.log")
        all_files = glob.glob(pattern)

        if not all_files:
            return [log_file] if os.path.exists(log_file) else []

        all_files.sort()

        return all_files

    def parse_log_file(self, log_file: str) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
        """
        Parse log file and extract pass timing data.

        If the log file is split (multiple files due to size limit),
        this method will automatically find and parse all split files.

        Returns:
            Tuple of (pass_times dict, function_ops dict)
            pass_times: {pass_name: [duration1, duration2, ...]}
            function_ops: {function_name: ops_count}
        """
        lg = get_logger()
        pass_times: Dict[str, List[int]] = defaultdict(list)
        function_ops: Dict[str, int] = {}

        log_files = self.find_split_log_files(log_file)

        if not log_files:
            raise FileNotFoundError(f"Log file not found: {log_file}")

        if len(log_files) > 1 and not self.quiet_mode:
            first_file = os.path.basename(log_files[0])
            match = re.match(r'pypto-log-(\d+)_', first_file)
            pid = match.group(1) if match else "unknown"

            lg.info(f"Found {len(log_files)} split log files (pid: {pid}):")
            for i, f in enumerate(log_files, 1):
                size_mb = os.path.getsize(f) / (1024 * 1024)
                lg.info(f"  {i}. {os.path.basename(f)} ({size_mb:.1f}M)")

        try:
            for current_file in log_files:
                with open(current_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = self.PASS_TIME_PATTERN.search(line)
                        if match:
                            pass_name = match.group(1)
                            duration_us = int(match.group(2))
                            pass_times[pass_name].append(duration_us)

                        match = self.OP_COUNT_PATTERN.search(line)
                        if match:
                            func_name = match.group(1)
                            ops = int(match.group(2))
                            function_ops[func_name] = ops
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Log file not found: {log_file}") from e
        except Exception as e:
            raise RuntimeError(f"Error parsing log file: {e}") from e

        return pass_times, function_ops

    def analyze(self, log_file: str, compare_file: Optional[str] = None):
        """Analyze log file and optionally compare with another file."""
        pass_times, function_ops = self.parse_log_file(log_file)
        self.function_ops = function_ops
        self.total_ops = sum(function_ops.values()) if function_ops else 0

        pass_times_before = {}
        if compare_file:
            pass_times_before, _ = self.parse_log_file(compare_file)

        for pass_name, durations in pass_times.items():
            durations_before = pass_times_before.get(pass_name, [])
            self.pass_data[pass_name] = PassPerfData(
                name=pass_name, durations=durations, ops=self.total_ops, durations_before=durations_before
            )

        total_pass_time = sum(d.duration_us for d in self.pass_data.values())
        for data in self.pass_data.values():
            data.total_pass_time = total_pass_time

    def get_status(self, data: PassPerfData) -> Tuple[str, float]:
        """Get status string and dynamic threshold for a pass based on average op count."""
        avg_ops = sum(self.function_ops.values()) / len(self.function_ops) if self.function_ops else 0
        base_ops = 200000
        dynamic_threshold = (avg_ops / base_ops) * self.time_threshold_s

        if data.avg_duration_s > dynamic_threshold:
            return f"WARNING (>{dynamic_threshold:.1f}s for {int(avg_ops)} ops)", dynamic_threshold
        return "OK", dynamic_threshold

    def print_report(self):
        """Print performance analysis report with count and average."""
        lg = get_logger()

        if not self.pass_data:
            lg.info("No pass timing data found in log file.")
            return

        sorted_data = sorted(self.pass_data.values(), key=lambda x: x.duration_us, reverse=True)

        lg.info("=" * 140)
        lg.info(" " * 45 + "Pass Performance Analysis Report")
        lg.info("=" * 140)
        lg.info(f"Time Threshold: {self.time_threshold_s}s (for 200K ops scenario)")
        lg.info("")
        if self.function_ops:
            lg.info("Function Operations Statistics:")
            max_name_len = max(len(name) for name in self.function_ops.keys()) if self.function_ops else 40
            col_width = max(max_name_len, 40)
            lg.info(f"  {'Function Name':<{col_width}} | {'Ops':>10}")
            lg.info(f"  {'-' * col_width} | {'-' * 10}")
            for func_name, ops in self.function_ops.items():
                lg.info(f"  {func_name:<{col_width}} | {ops:>10}")
            lg.info(f"  {'-' * col_width} | {'-' * 10}")
            total_ops = sum(self.function_ops.values())
            avg_ops = total_ops / len(self.function_ops) if self.function_ops else 0
            lg.info(f"  Total: {total_ops} | Average: {avg_ops:.1f}")
            lg.info("")

        has_comparison = any(d.durations_before for d in sorted_data)

        if has_comparison:
            header = (
                f"{'Pass Name':<30} | {'Count':<5} | {'Avg':<9} | "
                f"{'Target(s)':<9} | {'Total':<9} | "
                f"{'Time%':<6} | {'Improve%':<8} | {'Status'}"
            )
            lg.info(header)
            lg.info("-" * 160)
        else:
            header = (
                f"{'Pass Name':<30} | {'Count':<5} | {'Avg':<9} | "
                f"{'Target(s)':<9} | {'Total':<9} | "
                f"{'Time%':<6} | {'Status'}"
            )
            lg.info(header)
            lg.info("-" * 130)

        for data in sorted_data:
            status, dynamic_threshold = self.get_status(data)
            time_percent = f"{data.time_percent:.1f}%"

            if has_comparison and data.durations_before:
                improve = f"{data.improvement_percent:+.1f}%"
                row = (
                    f"{data.name:<30} | {data.count:<5} | {fmt_dur(data.avg_duration_s):<9} | "
                    f"{dynamic_threshold:<9.2f} | {fmt_dur(data.duration_s):<9} | "
                    f"{time_percent:<6} | {improve:<8} | {status}"
                )
                lg.info(row)
            else:
                row = (
                    f"{data.name:<30} | {data.count:<5} | {fmt_dur(data.avg_duration_s):<9} | "
                    f"{dynamic_threshold:<9.2f} | {fmt_dur(data.duration_s):<9} | "
                    f"{time_percent:<6} | {status}"
                )
                lg.info(row)

        lg.info("")

        exceeding = []
        for d in sorted_data:
            _, dynamic_threshold = self.get_status(d)
            if d.avg_duration_s > dynamic_threshold:
                exceeding.append((d, dynamic_threshold))

        if exceeding:
            lg.info("=" * 140)
            lg.info("Passes Exceeding Dynamic Threshold:")
            for i, (data, threshold) in enumerate(exceeding, 1):
                msg = (
                    f"  {i}. {data.name}: avg {fmt_dur(data.avg_duration_s)} > "
                    f"target {threshold:.1f}s (ops: {data.ops}, count: {data.count}, "
                    f"time%: {data.time_percent:.1f}%)"
                )
                lg.info(msg)
            lg.info("=" * 140)

        lg.info("")
        lg.info("Summary:")
        lg.info(f"  Total passes analyzed: {len(self.pass_data)}")
        lg.info(f"  Passes exceeding threshold: {len(exceeding)}")
        total_time = sum(d.duration_us for d in self.pass_data.values())
        total_count = sum(d.count for d in self.pass_data.values())
        lg.info(f"  Total pass executions: {total_count}")
        lg.info(f"  Total pass time: {total_time / 1_000_000:.2f}s")

        if has_comparison:
            total_before = sum(d.duration_us_before for d in self.pass_data.values() if d.durations_before)
            if total_before > 0:
                improvement = (total_before - total_time) / total_before * 100
                lg.info(f"  Total improvement: {improvement:+.1f}%")

    def get_exceeding_passes(self) -> List[PassPerfData]:
        """Get list of passes exceeding threshold."""
        return [d for d in self.pass_data.values() if d.duration_s > self.time_threshold_s]


def main():
    parser = argparse.ArgumentParser(
        description='Parse log files and generate Pass performance analysis reports.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -l pypto-log-*.log
  %(prog)s -l after.log --compare before.log
  %(prog)s -l run.log --time-threshold 15 --ops-threshold 300000
        """,
    )

    parser.add_argument('-l', '--log', required=True, help='Path to the log file to analyze')
    parser.add_argument('--compare', help='Path to a previous log file for comparison')
    parser.add_argument('--time-threshold', type=float, default=20.0, help='Time threshold in seconds (default: 20)')
    parser.add_argument(
        '--ops-threshold', type=int, default=200000, help='Operations threshold for scale reference (default: 200000)'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true', help='Suppress file list output, show only analysis results'
    )
    parser.add_argument(
        '--log-dir', default='./perf_logs', help='Directory for output log files (default: ./perf_logs)'
    )

    args = parser.parse_args()

    global _logger
    _logger = setup_logger(args.log_dir)
    lg = get_logger()

    analyzer = PassPerfAnalyzer(
        time_threshold_s=args.time_threshold, ops_threshold=args.ops_threshold, quiet_mode=args.quiet
    )

    try:
        analyzer.analyze(args.log, args.compare)
        analyzer.print_report()
    except FileNotFoundError as e:
        lg.error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        lg.error(str(e))
        sys.exit(1)

    exceeding = analyzer.get_exceeding_passes()
    if exceeding:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
