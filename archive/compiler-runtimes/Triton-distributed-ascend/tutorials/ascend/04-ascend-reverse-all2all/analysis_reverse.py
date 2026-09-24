#!/usr/bin/env python3
"""Analyze msprof op_summary CSV files for the Reverse-All2All fused kernel (robust v2).

Fairness fixes vs the legacy ``analysis(1).py``:

- **golden includes its transpose work**: the golden ``torch_reverse_a2a`` does
  ``ConcatD + 2×Transpose + hcom_alltoall`` per invocation.  The legacy script
  counted *only* ``hcom_alltoall`` (~884us), dropping ConcatD+Transpose (~278us,
  24% of golden) and unfairly flattering golden.  v2 sums
  ``hcom_alltoall + ConcatD + Transpose`` per invocation (time-window grouping,
  preprocessing assigned to the following a2a).
- **per-card reporting**: computes and prints each card's speedup separately, so
  device asymmetry (e.g. device_2 vs device_4) is visible instead of masked by
  2-card sample mixing.
- **median in headline**: heavy-tailed comm latency → mean >> median; v2 reports
  median (and geo_mean) alongside average, and labels max as stall-affected.
- **fast-fail on misalignment**: record counts must be divisible by num_cases,
  else raise (no silent case-misassignment).
- **group before warmup**: invocations are built before dropping warmup, so
  leaked preprocessing cannot contaminate the first benchmark sample.
- case-insensitive op matching; ``WARMUP`` env-configurable.

Golden path per invocation = ``hcom_alltoall`` + its preceding ``ConcatD``/``Transpose``.
TD path = single ``kernel_hccl_reverse_a2a_pipelined`` op duration.
"""

import csv
import glob
import math
import os
import re
import sys


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Fused TD kernel op name prefix.
KERNEL_PREFIX = "kernel_hccl_reverse_a2a_pipelined"
WARMUP = int(os.environ.get("REVERSE_PROFILE_WARMUP", "3"))

# Shape sweep (216 cases): S_RANGES x H_LIST, head_dim fixed to 128.
S_RANGES = (
    list(range(2040, 2049))
    + list(range(4088, 4097))
    + list(range(8184, 8193))
)
H_LIST = [8, 12, 24, 28, 32, 40, 48, 56]
HEAD_DIM = 128
TEST_CASES = [(s_value, h_value, HEAD_DIM) for s_value in S_RANGES for h_value in H_LIST]

# OP Type substrings. a2a is the invocation anchor; ConcatD/Transpose are the
# golden's transpose work that must be summed in. (torch_reverse_a2a emits
# ConcatD + 2×Transpose + hcom_alltoall per call; no Slice appears in profile.)
A2A_KEYWORD = "hcom_alltoall"
GOLDEN_OTHER_KEYWORDS = ["concatd", "transpose"]


def visual_len(value):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-9;]*[ -/]*[@-~])")
    plain_text = ansi_escape.sub("", value)
    return sum(2 if "一" <= char <= "鿿" else 1 for char in plain_text)


def pad_str(value, width, align="right"):
    padding = " " * max(0, width - visual_len(value))
    return padding + value if align == "right" else value + padding


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def remove_upper_outliers(values):
    if len(values) <= 2:
        return values
    ordered = sorted(values)
    midpoint = ordered[len(ordered) // 2]
    coarse = [v for v in values if v <= max(midpoint * 5.0, 5000.0)]
    if not coarse:
        coarse = values
    ordered = sorted(coarse)
    if len(ordered) <= 2:
        return ordered
    q1 = ordered[int(len(ordered) * 0.25)]
    q3 = ordered[int(len(ordered) * 0.75)]
    upper_bound = q3 + 1.5 * (q3 - q1)
    filtered = [v for v in coarse if v <= upper_bound]
    return filtered if filtered else coarse


def _match_any(op_name, op_type, keywords):
    lower = (op_name + " " + op_type).lower()
    return any(kw in lower for kw in keywords)


def _device_label(csv_path):
    """Extract 'device_X' from the PROF dir, for per-card reporting."""
    prof_dir = os.path.dirname(os.path.dirname(csv_path))  # .../PROF_000001_XXX
    try:
        for name in os.listdir(prof_dir):
            if name.startswith("device_"):
                return name
    except OSError:
        pass
    return os.path.basename(prof_dir)


def _read_records(csv_path):
    """Return (a2a_records, other_golden_records, td_records).

    Each is a list of (start_time_us, duration_us) sorted by start time.
    """
    a2a_records = []
    other_golden_records = []
    td_records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            op_name = row.get("Op Name", "")
            op_type = row.get("OP Type", "")
            try:
                duration = float(row.get("Task Duration(us)", "0").strip())
                start_time = float(row.get("Task Start Time(us)", "0").strip())
            except (TypeError, ValueError):
                continue

            if A2A_KEYWORD in op_name.lower() or A2A_KEYWORD in op_type.lower():
                a2a_records.append((start_time, duration))
            elif KERNEL_PREFIX in op_name or KERNEL_PREFIX in op_type:
                td_records.append((start_time, duration))
            elif _match_any(op_name, op_type, GOLDEN_OTHER_KEYWORDS):
                other_golden_records.append((start_time, duration))

    a2a_records.sort()
    other_golden_records.sort()
    td_records.sort()
    return a2a_records, other_golden_records, td_records


def parse_card(csv_path):
    """Parse one card's op_summary CSV.

    Returns (results, a2a_per_case, td_per_case, label) where *results* is a list
    of per-case dicts:
        a2a_times          — [(start, duration), ...] for a2a ops.
        other_golden_times — [(start, duration), ...] for ConcatD/Transpose in
                             this case's time window.
        td_raw             — [duration, ...] for kernel ops.
    """
    a2a_records, other_golden_records, td_records = _read_records(csv_path)
    num_cases = len(TEST_CASES)
    if not a2a_records or not td_records:
        raise ValueError(
            f"{csv_path}: found a2a={len(a2a_records)}, "
            f"kernel={len(td_records)} records"
        )
    # fast-fail: counts must be divisible by num_cases, else cases misalign
    if len(a2a_records) % num_cases or len(td_records) % num_cases:
        raise ValueError(
            f"{csv_path}: record counts not divisible by {num_cases}: "
            f"a2a={len(a2a_records)}, td={len(td_records)}"
        )

    a2a_per_case = len(a2a_records) // num_cases
    td_per_case = len(td_records) // num_cases

    results = []
    for case_index in range(num_cases):
        as_idx = case_index * a2a_per_case
        ts_idx = case_index * td_per_case
        a2a_slice = a2a_records[as_idx : as_idx + a2a_per_case]
        td_slice = td_records[ts_idx : ts_idx + td_per_case]

        # Per-case time window: only ConcatD/Transpose within this case's golden
        # phase are attributed to it (prevents any cross-case contamination).
        win_start = a2a_slice[0][0] - 500.0
        win_end = a2a_slice[-1][0] + a2a_slice[-1][1] + 500.0
        other_in_window = [
            (s, d) for s, d in other_golden_records
            if win_start <= s <= win_end
        ]

        results.append(
            {
                "a2a_times": a2a_slice,
                "other_golden_times": other_in_window,
                "td_raw": [d for _, d in td_slice],
            }
        )
    return results, a2a_per_case, td_per_case, _device_label(csv_path)


def _stats(values):
    return {
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "average": sum(values) / len(values),
    }


def _group_into_invocations(a2a_times, other_times):
    """Sum a2a + its preceding ConcatD/Transpose per invocation.

    Boundaries are a2a start-times.  Each other op (start < next a2a) is added to
    the following a2a's invocation (golden preprocessing precedes its a2a).
    """
    if not a2a_times:
        return []
    boundaries = [t for t, _ in a2a_times]
    inv_sums = [0.0] * len(boundaries)
    for i, (_, dur) in enumerate(a2a_times):
        inv_sums[i] += dur
    for o_start, o_dur in other_times:
        for i, b in enumerate(boundaries):
            if o_start < b:
                inv_sums[i] += o_dur
                break
    return inv_sums


def _bench_stats(card, case_index):
    """Per-card, per-case: (golden_stats, td_stats) after warmup + outlier removal."""
    a2a_times = card["results"][case_index]["a2a_times"]
    other_times = card["results"][case_index]["other_golden_times"]
    td_raw = card["results"][case_index]["td_raw"]

    all_golden = _group_into_invocations(a2a_times, other_times)
    golden = _stats(remove_upper_outliers(all_golden[WARMUP:]))
    td = _stats(remove_upper_outliers(td_raw[WARMUP:]))
    return golden, td


def analyze(parent_dir):
    pattern = os.path.join(
        parent_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv"
    )
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"no op_summary CSV found under {parent_dir}")

    cards = []
    for csv_path in csv_files:
        try:
            results, a2a_count, td_count, label = parse_card(csv_path)
        except ValueError as error:
            print(f"[SKIP] {error}")
            continue
        cards.append({"results": results, "label": label})
        print(f"[INFO] {label}: a2a/case={a2a_count}, td/case={td_count}")
    if not cards:
        raise RuntimeError("no CSV contains both matching a2a and kernel records")

    results = []
    for case_index, (sequence_length, n_head, head_dim) in enumerate(TEST_CASES):
        # Mixed-card (pool all cards' samples) — the legacy-style aggregate.
        mixed_golden = []
        mixed_td = []
        per_card = []
        for card in cards:
            golden, td = _bench_stats(card, case_index)
            # raw bench samples (post-warmup, pre-outlier) for mixing
            a2a_times = card["results"][case_index]["a2a_times"]
            other_times = card["results"][case_index]["other_golden_times"]
            td_raw = card["results"][case_index]["td_raw"]
            mixed_golden.extend(_group_into_invocations(a2a_times, other_times)[WARMUP:])
            mixed_td.extend(td_raw[WARMUP:])
            per_card.append({
                "label": card["label"],
                "speedup_avg": golden["average"] / td["average"] if td["average"] > 0 else 0.0,
                "speedup_med": golden["median"] / td["median"] if td["median"] > 0 else 0.0,
            })

        if not mixed_golden or not mixed_td:
            continue

        golden = _stats(remove_upper_outliers(mixed_golden))
        td = _stats(remove_upper_outliers(mixed_td))

        results.append(
            {
                "S": sequence_length,
                "H": n_head,
                "D": head_dim,
                "golden": golden,
                "td": td,
                "speedup_min": golden["min"] / td["min"] if td["min"] > 0 else 0.0,
                "speedup_max": golden["max"] / td["max"] if td["max"] > 0 else 0.0,
                "speedup_median": golden["median"] / td["median"] if td["median"] > 0 else 0.0,
                "speedup_average": golden["average"] / td["average"] if td["average"] > 0 else 0.0,
                "per_card": per_card,
                "num_cards": len(cards),
            }
        )
    return results, [c["label"] for c in cards]


def print_report(results, card_labels):
    if not results:
        print("[WARNING] no matching results")
        return

    num_cards = results[0]["num_cards"]
    header = (
        f"{'S':>6} {'H':>4} {'D':>4}  "
        f"{'Golden min/max/med/avg (us)':>38}  "
        f"{'TD min/max/med/avg (us)':>35}  "
        f"{'speedup min/max/med/avg':>31}"
    )
    print("\n" + "=" * visual_len(header))
    print(
        f"{BOLD}{GREEN}Reverse-All2All TD kernel vs Hcom Golden "
        f"(a2a + ConcatD + Transpose) ({num_cards} cards mixed){RESET}"
    )
    print(header)
    print("-" * visual_len(header))
    for result in results:
        g = result["golden"]
        t = result["td"]
        print(
            f"{result['S']:6d} {result['H']:4d} {result['D']:4d}  "
            f"{g['min']:8.1f}/{g['max']:8.1f}/{g['median']:8.1f}/{g['average']:8.1f}  "
            f"{t['min']:8.1f}/{t['max']:8.1f}/{t['median']:8.1f}/{t['average']:8.1f}  "
            f"{result['speedup_min']:6.2f}x/{result['speedup_max']:6.2f}x/"
            f"{result['speedup_median']:6.2f}x/{result['speedup_average']:6.2f}x"
        )

    avgs = [r["speedup_average"] for r in results]
    meds = [r["speedup_median"] for r in results]
    geo = math.exp(sum(math.log(max(v, 1e-9)) for v in avgs) / len(avgs))
    pass_avg = 100.0 * sum(1 for v in avgs if v > 1.0) / len(avgs)
    pass_med = 100.0 * sum(1 for v in meds if v > 1.0) / len(meds)
    print("=" * visual_len(header))
    print(
        f"SUMMARY cases={len(results)}/{len(TEST_CASES)} "
        f"average_speedup={sum(avgs)/len(avgs):.3f}x "
        f"median_speedup={sum(meds)/len(meds):.3f}x "
        f"geo_mean={geo:.3f}x "
        f"pass_rate(avg/med)={pass_avg:.2f}%/{pass_med:.2f}% "
        f"min={min(avgs):.3f}x max={max(avgs):.3f}x"
    )

    # Per-card breakdown — exposes device asymmetry masked by mixing.
    print("-" * visual_len(header))
    print(f"{BOLD}Per-card breakdown (avg / median speedup across cases):{RESET}")
    for ci, label in enumerate(card_labels):
        c_avg = sum(r["per_card"][ci]["speedup_avg"] for r in results) / len(results)
        c_med = sum(r["per_card"][ci]["speedup_med"] for r in results) / len(results)
        c_pass = 100.0 * sum(1 for r in results if r["per_card"][ci]["speedup_avg"] > 1.0) / len(results)
        print(f"  {label}: avg={c_avg:.3f}x  med={c_med:.3f}x  pass={c_pass:.2f}%")
    print("=" * visual_len(header) + "\n")
    print("(max is stall-affected and not meaningful for distributed comms; use median/avg.)\n")


def write_csv(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "S", "H", "D",
            "Golden_Min_us", "Golden_Max_us", "Golden_Med_us", "Golden_Avg_us",
            "TD_Min_us", "TD_Max_us", "TD_Med_us", "TD_Avg_us",
            "Speedup_Min", "Speedup_Max", "Speedup_Med", "Speedup_Avg",
        ]
        for ci in range(len(results[0]["per_card"]) if results else 0):
            header += [f"Card{ci}_Speedup_Avg", f"Card{ci}_Speedup_Med"]
        writer.writerow(header)
        for result in results:
            g = result["golden"]
            t = result["td"]
            row = [
                result["S"], result["H"], result["D"],
                g["min"], g["max"], g["median"], g["average"],
                t["min"], t["max"], t["median"], t["average"],
                result["speedup_min"], result["speedup_max"],
                result["speedup_median"], result["speedup_average"],
            ]
            for ci in range(len(result["per_card"])):
                row += [result["per_card"][ci]["speedup_avg"],
                        result["per_card"][ci]["speedup_med"]]
            writer.writerow(row)
    print(f"[INFO] wrote {output_path}")


if __name__ == "__main__":
    parent_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    output_path = sys.argv[2] if len(sys.argv) > 2 else "reverse_a2a_summary_v2.csv"
    print(f"[START] scanning {os.path.abspath(parent_directory)} (WARMUP={WARMUP})")
    analysis, card_labels = analyze(parent_directory)
    print_report(analysis, card_labels)
    write_csv(analysis, output_path)
