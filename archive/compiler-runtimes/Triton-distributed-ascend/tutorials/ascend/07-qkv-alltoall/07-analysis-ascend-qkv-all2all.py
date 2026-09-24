#!/usr/bin/env python3
"""Analyze a two- or four-card QKV msprof run with legacy score semantics.

This intentionally preserves the statistics used by
``qkv-fuse-a2a-v1/analyze_qkv_msprof.py``:

* golden time is All2All + the ConcatD/Slice ops assigned to that invocation;
* samples from all profiled cards are mixed before statistics are calculated;
* golden and QKV samples have upper outliers removed independently; and
* ``Speedup_Avg`` is ``Golden_Avg_us / QKV_Avg_us``.

Unlike the legacy analyzer, case labels are validated against each fused QKV
operator's ``Input Shapes``.  Physical card IDs are discovered from the CSVs,
and the supported V20 grids are inferred from the card count: global
``H=2,4,...,14`` for two cards and ``H=4,8,...,28`` for four cards.
"""

import csv
import glob
import math
import os
import re
import sys


GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

KERNEL_PREFIX = "kernel_qkv_fuse_a2a"
A2A_KEYWORD = "hcom_alltoall"
GOLDEN_OTHER_KEYWORDS = ("concatd", "slice")

WARMUP = int(os.environ.get("QKV_PROFILE_WARMUP", "5"))
ITERS = int(os.environ.get("QKV_PROFILE_ITERS", "50"))
TIMED_START = WARMUP
TIMED_END = WARMUP + ITERS
# Old buffertrue has only warmup + timed calls.  The new harness appends one
# fresh Golden/custom correctness call, which must never enter timing stats.
ACCEPTED_CALLS_PER_CASE = (TIMED_END, TIMED_END + 1)
SUPPORTED_CARD_COUNTS = (2, 4)
S_RANGES = (
    list(range(2040, 2049))
    + list(range(4088, 4097))
    + list(range(8184, 8193))
)
LOCAL_H_LIST = (1, 2, 3, 4, 5, 6, 7)
EXPECTED_CASE_COUNT = len(S_RANGES) * len(LOCAL_H_LIST)


def _grid_for_card_count(card_count):
    """Return the V20 global-head grid and ordered 189 test cases."""
    if card_count not in SUPPORTED_CARD_COUNTS:
        raise ValueError(
            f"expected a {SUPPORTED_CARD_COUNTS[0]}- or "
            f"{SUPPORTED_CARD_COUNTS[1]}-card profile, found {card_count} "
            "op_summary CSV files"
        )
    global_heads = tuple(value * card_count for value in LOCAL_H_LIST)
    test_cases = [
        (sequence_length, n_head, 128)
        for sequence_length in S_RANGES
        for n_head in global_heads
    ]
    if len(test_cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"internal V20 grid must contain {EXPECTED_CASE_COUNT} cases"
        )
    return global_heads, test_cases


def visual_len(value):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    plain_text = ansi_escape.sub("", value)
    return sum(2 if "\u4e00" <= char <= "\u9fff" else 1 for char in plain_text)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def remove_upper_outliers(values):
    """Legacy one-sided coarse + IQR filter; keep byte-for-byte math order."""
    if len(values) <= 2:
        return values

    ordered = sorted(values)
    midpoint = ordered[len(ordered) // 2]
    coarse = [value for value in values if value <= max(midpoint * 5.0, 5000.0)]
    if not coarse:
        coarse = values

    ordered = sorted(coarse)
    if len(ordered) <= 2:
        return ordered
    q1 = ordered[int(len(ordered) * 0.25)]
    q3 = ordered[int(len(ordered) * 0.75)]
    upper_bound = q3 + 1.5 * (q3 - q1)
    filtered = [value for value in coarse if value <= upper_bound]
    return filtered if filtered else coarse


def _match_any(op_name, op_type, keywords):
    lower = (op_name + " " + op_type).lower()
    return any(keyword in lower for keyword in keywords)


def _parse_qkv_input_shape(raw_value, csv_path, row_number):
    """Return the common Q/K/V ``(S,H,D)`` or ``None`` for profiler N/A."""
    stripped = (raw_value or "").strip()
    # Different msprof builds spell the missing first-compile metadata as
    # either N/A or NULL.  Both are accepted only where the caller permits a
    # missing shape (normally warmup); measured calls remain strictly checked.
    if stripped.upper() in ("", "N/A", "NULL"):
        return None

    triples = [
        tuple(int(value) for value in match)
        for match in re.findall(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", stripped)
    ]
    if len(triples) < 3:
        raise ValueError(
            f"{csv_path}:{row_number}: cannot parse three Q/K/V shapes from "
            f"Input Shapes={raw_value!r}"
        )
    if triples[0] != triples[1] or triples[0] != triples[2]:
        raise ValueError(
            f"{csv_path}:{row_number}: Q/K/V Input Shapes disagree: "
            f"{triples[:3]}"
        )
    return triples[0]


def _read_records(csv_path):
    """Read the operator records needed for one profiled card."""
    a2a_records = []
    other_golden_records = []
    qkv_records = []
    device_ids = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "Device_id",
            "Op Name",
            "OP Type",
            "Task Start Time(us)",
            "Task Duration(us)",
            "Input Shapes",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path}: missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_device = (row.get("Device_id") or "").strip()
            if raw_device.isdigit():
                device_ids.add(int(raw_device))

            op_name = row.get("Op Name", "")
            op_type = row.get("OP Type", "")
            combined_lower = (op_name + " " + op_type).lower()
            is_a2a = A2A_KEYWORD in combined_lower
            is_qkv = KERNEL_PREFIX in op_name or KERNEL_PREFIX in op_type
            is_other_golden = _match_any(
                op_name, op_type, GOLDEN_OTHER_KEYWORDS
            )
            if not (is_a2a or is_qkv or is_other_golden):
                continue

            try:
                duration = float((row.get("Task Duration(us)") or "0").strip())
                start_time = float(
                    (row.get("Task Start Time(us)") or "0").strip()
                )
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}:{row_number}: invalid start/duration"
                ) from error

            if is_a2a:
                a2a_records.append((start_time, duration))
            elif is_qkv:
                shape = _parse_qkv_input_shape(
                    row.get("Input Shapes", ""), csv_path, row_number
                )
                qkv_records.append((start_time, duration, shape))
            else:
                other_golden_records.append((start_time, duration))

    if len(device_ids) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly one numeric Device_id, got "
            f"{sorted(device_ids)}"
        )
    a2a_records.sort(key=lambda record: record[0])
    other_golden_records.sort(key=lambda record: record[0])
    qkv_records.sort(key=lambda record: record[0])
    return (
        next(iter(device_ids)),
        a2a_records,
        other_golden_records,
        qkv_records,
    )


def parse_card(csv_path, test_cases):
    """Parse and strictly validate one card's complete 189-case profile."""
    device_id, a2a_records, other_golden_records, qkv_records = _read_records(
        csv_path
    )
    num_cases = len(test_cases)
    if len(a2a_records) % num_cases or len(qkv_records) % num_cases:
        raise ValueError(
            f"{csv_path}: record counts are not divisible by {num_cases}: "
            f"a2a={len(a2a_records)}, qkv={len(qkv_records)}"
        )
    a2a_per_case = len(a2a_records) // num_cases
    qkv_per_case = len(qkv_records) // num_cases
    if a2a_per_case != qkv_per_case:
        raise ValueError(
            f"{csv_path}: per-case Golden/QKV counts disagree: "
            f"a2a={a2a_per_case}, qkv={qkv_per_case}"
        )
    if qkv_per_case not in ACCEPTED_CALLS_PER_CASE:
        raise ValueError(
            f"{csv_path}: calls/case must be one of "
            f"{ACCEPTED_CALLS_PER_CASE}, got {qkv_per_case}"
        )
    calls_per_case = qkv_per_case

    results = []
    missing_warmup_shapes = 0
    for case_index, expected_shape in enumerate(test_cases):
        start = case_index * calls_per_case
        end = start + calls_per_case
        a2a_slice = a2a_records[start:end]
        qkv_slice = qkv_records[start:end]

        missing_positions = [
            offset
            for offset, (_, _, shape) in enumerate(qkv_slice)
            if shape is None
        ]
        measured_missing = [offset for offset in missing_positions if offset >= WARMUP]
        if measured_missing:
            raise ValueError(
                f"{csv_path}: case {case_index} {expected_shape} has N/A Input "
                f"Shapes in measured calls {measured_missing}"
            )
        missing_warmup_shapes += len(missing_positions)

        observed_shapes = {
            shape for _, _, shape in qkv_slice if shape is not None
        }
        if observed_shapes != {expected_shape}:
            raise ValueError(
                f"{csv_path}: case {case_index} expected QKV Input Shapes "
                f"{expected_shape}, observed {sorted(observed_shapes)}"
            )

        win_start = a2a_slice[0][0] - 500.0
        win_end = a2a_slice[-1][0] + a2a_slice[-1][1] + 500.0
        other_in_window = [
            (record_start, duration)
            for record_start, duration in other_golden_records
            if win_start <= record_start <= win_end
        ]
        results.append(
            {
                "a2a_times": a2a_slice,
                "other_golden_times": other_in_window,
                "qkv_raw": [duration for _, duration, _ in qkv_slice],
            }
        )

    return {
        "device_id": device_id,
        "cases": results,
        "calls_per_case": calls_per_case,
        "missing_warmup_shapes": missing_warmup_shapes,
    }


def _stats(values):
    return {
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "average": sum(values) / len(values),
    }


def _group_into_invocations(a2a_times, other_times):
    """Apply the legacy assignment of preprocessing ops to the next All2All."""
    if not a2a_times:
        return []

    boundaries = [start for start, _ in a2a_times]
    invocation_sums = [0.0] * len(boundaries)
    for index, (_, duration) in enumerate(a2a_times):
        invocation_sums[index] += duration

    for other_start, other_duration in other_times:
        for index, boundary in enumerate(boundaries):
            if other_start < boundary:
                invocation_sums[index] += other_duration
                break
    return invocation_sums


def analyze(parent_dir):
    pattern = os.path.join(
        parent_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv"
    )
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"no op_summary CSV found under {parent_dir}")
    global_h_list, test_cases = _grid_for_card_count(len(csv_files))

    cards = []
    for csv_path in csv_files:
        card = parse_card(csv_path, test_cases)
        cards.append(card)
        print(
            f"[INFO] card={card['device_id']} {csv_path}: "
            f"a2a/case={card['calls_per_case']}, "
            f"qkv/case={card['calls_per_case']}, "
            f"warmup-shape-N/A={card['missing_warmup_shapes']}"
        )

    actual_card_ids = tuple(sorted(card["device_id"] for card in cards))
    if len(set(actual_card_ids)) != len(cards):
        raise ValueError(
            "each op_summary CSV must belong to a different physical card; "
            f"got card IDs {actual_card_ids}"
        )
    calls_per_card = {card["calls_per_case"] for card in cards}
    if len(calls_per_card) != 1:
        raise ValueError(
            f"cards came from different harness layouts: calls/case="
            f"{sorted(calls_per_card)}"
        )
    calls_per_case = next(iter(calls_per_card))
    print(
        "[VALIDATED] cards="
        + ",".join(str(value) for value in actual_card_ids)
        + " cases=189 "
        f"calls/case={calls_per_case} timed_offsets={TIMED_START}:{TIMED_END} "
        "global_H="
        + ",".join(str(value) for value in global_h_list)
        + " D=128"
    )

    results = []
    for case_index, (sequence_length, global_heads, head_dim) in enumerate(
        test_cases
    ):
        mixed_golden = []
        mixed_qkv = []
        for card in cards:
            case = card["cases"][case_index]
            all_golden = _group_into_invocations(
                case["a2a_times"], case["other_golden_times"]
            )
            # Always keep exactly 50 timed samples.  For the 56-call harness,
            # offset 55 is the fresh correctness call and is deliberately
            # excluded; the legacy 55-call buffertrue ends at this boundary.
            mixed_golden.extend(all_golden[TIMED_START:TIMED_END])
            mixed_qkv.extend(case["qkv_raw"][TIMED_START:TIMED_END])

        golden = _stats(remove_upper_outliers(mixed_golden))
        qkv = _stats(remove_upper_outliers(mixed_qkv))
        results.append(
            {
                "S": sequence_length,
                "H": global_heads,
                "D": head_dim,
                "golden": golden,
                "qkv": qkv,
                "speedup_min": golden["min"] / qkv["min"],
                "speedup_max": golden["max"] / qkv["max"],
                "speedup_median": golden["median"] / qkv["median"],
                "speedup_average": golden["average"] / qkv["average"],
                "num_cards": len(cards),
            }
        )
    return results


def print_report(results):
    header = (
        f"{'S':>6} {'H':>4} {'D':>4}  "
        f"{'Golden min/max/med/avg (us)':>38}  "
        f"{'QKV min/max/med/avg (us)':>35}  "
        f"{'speedup min/max/med/avg':>31}"
    )
    print("\n" + "=" * visual_len(header))
    print(
        f"{BOLD}{GREEN}QKV kernel vs Golden (a2a + ConcatD + Slice) "
        f"({results[0]['num_cards']} cards mixed){RESET}"
    )
    print(header)
    print("-" * visual_len(header))
    for result in results:
        golden = result["golden"]
        qkv = result["qkv"]
        print(
            f"{result['S']:6d} {result['H']:4d} {result['D']:4d}  "
            f"{golden['min']:8.1f}/{golden['max']:8.1f}/"
            f"{golden['median']:8.1f}/{golden['average']:8.1f}  "
            f"{qkv['min']:8.1f}/{qkv['max']:8.1f}/"
            f"{qkv['median']:8.1f}/{qkv['average']:8.1f}  "
            f"{result['speedup_min']:6.2f}x/"
            f"{result['speedup_max']:6.2f}x/"
            f"{result['speedup_median']:6.2f}x/"
            f"{result['speedup_average']:6.2f}x"
        )

    speedups = [result["speedup_average"] for result in results]
    log_sum = sum(math.log(max(value, 1e-9)) for value in speedups)
    geometric_mean = math.exp(log_sum / len(speedups))
    print("=" * visual_len(header))
    print(
        f"SUMMARY cases={len(results)}/{EXPECTED_CASE_COUNT} "
        f"average_speedup={sum(speedups) / len(speedups):.3f}x "
        f"geo_mean={geometric_mean:.3f}x min={min(speedups):.3f}x "
        f"max={max(speedups):.3f}x"
    )


def write_csv(results, output_path):
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "S",
                "H",
                "D",
                "Golden_Min_us",
                "Golden_Max_us",
                "Golden_Med_us",
                "Golden_Avg_us",
                "QKV_Min_us",
                "QKV_Max_us",
                "QKV_Med_us",
                "QKV_Avg_us",
                "Speedup_Min",
                "Speedup_Max",
                "Speedup_Med",
                "Speedup_Avg",
            ]
        )
        for result in results:
            golden = result["golden"]
            qkv = result["qkv"]
            writer.writerow(
                [
                    result["S"],
                    result["H"],
                    result["D"],
                    golden["min"],
                    golden["max"],
                    golden["median"],
                    golden["average"],
                    qkv["min"],
                    qkv["max"],
                    qkv["median"],
                    qkv["average"],
                    result["speedup_min"],
                    result["speedup_max"],
                    result["speedup_median"],
                    result["speedup_average"],
                ]
            )
    print(f"[INFO] wrote {output_path}")


def main():
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} PROFILE_DIR [OUTPUT_CSV]",
            file=sys.stderr,
        )
        return 2
    parent_directory = sys.argv[1]
    output_path = (
        sys.argv[2] if len(sys.argv) == 3 else os.path.join(parent_directory, "ans.csv")
    )
    print(f"[START] scanning {os.path.abspath(parent_directory)}")
    results = analyze(parent_directory)
    print_report(results)
    write_csv(results, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
