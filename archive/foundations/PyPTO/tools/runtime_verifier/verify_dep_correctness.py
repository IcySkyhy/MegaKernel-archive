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
import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dep_verifier import get_registered_rules
from dep_verifier.data_loader import (
    infer_edge_seq_no,
    load_dyn_stitch_edges,
    load_dyn_topo,
    load_slot_access,
    load_slot_cell_table,
    load_slot_mapping,
    load_static_topo,
)
from dep_verifier.report import ViolationReport
from dep_verifier.rule_base import RuleContext

DEP_VERIFY_DUMP_SUBDIR = "dep_verify_dump"

INPUT_FILES = {
    "static_topo": os.path.join(DEP_VERIFY_DUMP_SUBDIR, "static_topo.csv"),
    "dyn_topo": "dyn_topo.txt",
    "stitch_edges": os.path.join(DEP_VERIFY_DUMP_SUBDIR, "dyn_stitch_edges.csv"),
    "slot_mapping": os.path.join(DEP_VERIFY_DUMP_SUBDIR, "slot_mapping.csv"),
    "slot_cell_table": os.path.join(DEP_VERIFY_DUMP_SUBDIR, "slot_cell_table.csv"),
    "slot_access": os.path.join(DEP_VERIFY_DUMP_SUBDIR, "dyn_slot_access.csv"),
}
REPORT_NAME = "dep_check_report.csv"

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Runtime tensor data-flow verification. Detects three classes of "
            "operator-level problems: (1) concurrent write overlap, "
            "(2) missing producer/consumer dependency, "
            "(3) illegal read/write linkage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The directory should contain dyn_topo.txt at its root and the "
            "dependency-verification dumps under '<dump_dir>/dep_verify_dump/' "
            "(static_topo.csv, dyn_stitch_edges.csv, slot_mapping.csv, "
            "slot_cell_table.csv, dyn_slot_access.csv). On success, prints "
            "'PASS'; otherwise prints a grouped summary and writes "
            "dep_check_report.csv to the same directory."
        ),
    )
    p.add_argument("dump_dir", help="Directory containing runtime dump files")
    return p.parse_args(argv)


def _resolve(dump_dir: str, key: str) -> Optional[str]:
    path = os.path.join(dump_dir, INPUT_FILES[key])
    return path if os.path.isfile(path) else None


def _resolve_required_paths(dump_dir: str) -> Tuple[str, str, str]:
    static_topo = _resolve(dump_dir, "static_topo")
    dyn_topo = _resolve(dump_dir, "dyn_topo")
    stitch_edges_path = _resolve(dump_dir, "stitch_edges")
    if static_topo and dyn_topo and stitch_edges_path:
        return static_topo, dyn_topo, stitch_edges_path
    required = (
        ("static_topo", static_topo),
        ("dyn_topo", dyn_topo),
        ("stitch_edges", stitch_edges_path),
    )
    missing = [INPUT_FILES[k] for k, p in required if not p]
    raise FileNotFoundError(f"missing required files in directory: {', '.join(missing)}")


def _load_all_inputs(dump_dir: str) -> Tuple[RuleContext, Dict[int, str], Dict[int, str]]:
    static_topo, dyn_topo, stitch_edges_path = _resolve_required_paths(dump_dir)
    static_fns = load_static_topo(static_topo)
    dyn_tasks = load_dyn_topo(dyn_topo)
    stitch_edges = load_dyn_stitch_edges(stitch_edges_path)
    slot_roles = {}
    slot_tensor_names = {}
    slot_func_names = {}
    slot_cell_tables = {}
    slot_accesses = []
    slot_mapping_path = _resolve(dump_dir, "slot_mapping")
    slot_cell_table_path = _resolve(dump_dir, "slot_cell_table")
    slot_access_path = _resolve(dump_dir, "slot_access")
    if slot_mapping_path:
        _, slot_roles, slot_tensor_names, slot_func_names = load_slot_mapping(slot_mapping_path)
    if slot_cell_table_path:
        slot_cell_tables = load_slot_cell_table(slot_cell_table_path)
    if slot_access_path:
        slot_accesses = load_slot_access(slot_access_path)
    infer_edge_seq_no(stitch_edges, dyn_tasks)
    ctx = RuleContext(
        static_functions=static_fns,
        dyn_tasks=dyn_tasks,
        stitch_edges=stitch_edges,
        slot_cell_tables=slot_cell_tables,
        slot_accesses=slot_accesses,
        slot_roles=slot_roles,
        slot_tensor_names=slot_tensor_names,
        slot_func_names=slot_func_names,
    )
    return ctx, slot_tensor_names, slot_func_names


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dump_dir = args.dump_dir
    if not os.path.isdir(dump_dir):
        logger.error("ERROR: directory does not exist: %s", dump_dir)
        return 2
    try:
        ctx, slot_tensor_names, slot_func_names = _load_all_inputs(dump_dir)
    except FileNotFoundError as exc:
        logger.error("ERROR: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("ERROR: failed to load input files: %s", exc)
        return 2
    report = ViolationReport(
        slot_tensor_names=slot_tensor_names,
        slot_func_names=slot_func_names,
    )
    for cls in get_registered_rules():
        report.extend(cls().run(ctx))
    report.print_console()
    report.save_csv(os.path.join(dump_dir, REPORT_NAME))
    return 1 if report.has_failure() else 0


if __name__ == "__main__":
    sys.exit(main())
