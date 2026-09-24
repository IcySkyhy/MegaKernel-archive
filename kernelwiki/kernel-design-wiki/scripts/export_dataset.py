#!/usr/bin/env python3
"""Export curated Kernel Design Wiki seed examples with split-group isolation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _common import SKILL_ROOT, dump_jsonl, load_jsonl


MODES = ["all", "retrieval", "design", "preference", "boundary", "episode"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--holdout-lineage", action="append", default=[], help="Force examples containing this lineage into test; repeatable")
    args = parser.parse_args()

    rows = load_jsonl(SKILL_ROOT / "data" / "seed_examples.jsonl")
    if args.mode != "all":
        rows = [row for row in rows if row.get("type") == args.mode]
    if not rows:
        parser.error("no examples match the requested mode")

    forced_test = set(args.holdout_lineage)
    groups = sorted({row["split_group"] for row in rows if not forced_test.intersection(row.get("lineage_ids", []))})
    train_count = max(1, int(len(groups) * 0.8)) if groups else 0
    validation_count = 1 if len(groups) - train_count >= 2 else 0
    assignment = {}
    for index, group in enumerate(groups):
        if index < train_count:
            assignment[group] = "train"
        elif index < train_count + validation_count:
            assignment[group] = "validation"
        else:
            assignment[group] = "test"

    partitions = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = "test" if forced_test.intersection(row.get("lineage_ids", [])) else assignment.get(row["split_group"], "test")
        exported = dict(row)
        exported["split"] = split
        partitions[split].append(exported)

    for split, split_rows in partitions.items():
        dump_jsonl(split_rows, args.output_dir / f"{split}.jsonl")
    counts = Counter(row["split"] for split_rows in partitions.values() for row in split_rows)
    print(f"Exported {sum(counts.values())} examples to {args.output_dir}: " + ", ".join(f"{name}={counts[name]}" for name in ("train", "validation", "test")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
