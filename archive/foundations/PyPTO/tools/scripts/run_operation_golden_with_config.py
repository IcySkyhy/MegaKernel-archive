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
"""Operation golden 生成入口.

复用现有 operation case 工具链, 先执行 json_only 生成范围内用例 JSON,
再按 build_ci 使用的 STest case 命名规则批量调用 golden_ctrl 生成 golden.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

ROOT_PATH = Path(__file__).resolve().parents[2]
GOLDEN_CTRL_SCRIPT = ROOT_PATH / "cmake/scripts/golden_ctrl.py"
DEFAULT_OPERATION_CASE_DIR = ROOT_PATH / "framework/tests/st/operation/test_case"
DEFAULT_GOLDEN_IMPL_DIR = ROOT_PATH / "framework/tests/st/operation/python"
DEFAULT_GOLDEN_OUTPUT_DIR = ROOT_PATH / "build/output/bin/golden"
CASE_FILE_SUFFIX = "_st_test_cases"
BASE_CASE_COLUMNS = {
    "case_index",
    "case_name",
    "operation",
    "input_shape",
    "input_dtype",
    "input_format",
    "input_trans",
    "input_datarange",
    "output_shape",
    "output_dtype",
    "output_format",
    "view_shape",
    "tile_shape",
    "skip",
    "enable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate operation golden without rebuilding.",
        epilog="Best Regards!",
    )
    parser.add_argument(
        "op",
        type=str,
        help="The operation selector. Use a single op, comma-separated ops, or '*'/'all' for batch.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default=None,
        help="The input csv/xlsx file or a directory that contains case files.",
    )
    add_case_range_args(parser)
    add_golden_path_args(parser)
    add_golden_run_args(parser)
    return parser.parse_args()


def add_case_range_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s",
        "--start_index",
        nargs="?",
        type=int,
        default=0,
        help="The inclusive start index of test cases.",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        nargs="?",
        type=int,
        default=-1,
        help="The inclusive end index of test cases. Negative means until the end.",
    )


def add_golden_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json_path", type=str, default=None, help="Path to save the converted JSON files.")
    parser.add_argument(
        "-o",
        "--golden_output",
        type=str,
        default=str(DEFAULT_GOLDEN_OUTPUT_DIR),
        help="Golden output directory. Defaults to build_ci naming path.",
    )
    parser.add_argument(
        "-p",
        "--golden_impl_path",
        action="append",
        default=None,
        help="Golden implementation path. Can be specified multiple times.",
    )


def add_golden_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j",
        "--job_num",
        nargs="?",
        type=int,
        default=1,
        help="Parallel job number passed to golden_ctrl.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the golden output directory before generating.",
    )


def resolve_input_file(op: str, input_file: Optional[str]) -> Path:
    if input_file:
        return Path(input_file).resolve()
    if is_batch_op(op):
        return DEFAULT_OPERATION_CASE_DIR.resolve()
    ops = parse_op_selector(op)
    if len(ops) != 1:
        raise ValueError("Single-file auto discovery only supports one operation.")
    return (DEFAULT_OPERATION_CASE_DIR / f"{ops[0]}_st_test_cases.csv").resolve()


def resolve_json_path(json_path: Optional[str]) -> Path:
    if json_path:
        return Path(json_path).resolve()
    return DEFAULT_OPERATION_CASE_DIR.resolve()


def resolve_impl_paths(impl_paths: Optional[List[str]]) -> List[Path]:
    if not impl_paths:
        return [DEFAULT_GOLDEN_IMPL_DIR.resolve()]
    return [Path(path).resolve() for path in impl_paths]


def is_batch_op(op: str) -> bool:
    return op.strip() in {"*", "all"}


def parse_op_selector(op: str) -> List[str]:
    if is_batch_op(op):
        return []
    ops = [item.strip() for item in op.split(",") if item.strip()]
    if not ops:
        raise ValueError("At least one operation must be specified.")
    return ops


def infer_op_from_case_file(case_file: Path) -> str:
    stem = case_file.stem
    if not stem.endswith(CASE_FILE_SUFFIX):
        raise ValueError(f"Unsupported case file name: {case_file.name}")
    return stem[:-len(CASE_FILE_SUFFIX)]


def resolve_json_file(json_path: Path, case_file: Path, total_case_files: int) -> Path:
    if json_path.suffix:
        if total_case_files != 1:
            raise ValueError("A json file path can only be used with one selected case file.")
        return json_path
    return json_path / f"{case_file.stem}.json"


def collect_case_specs(op: str, input_file: Path, json_path: Path) -> List[Tuple[str, Path, Path]]:
    selected_ops = set(parse_op_selector(op))
    case_files = [input_file]
    if input_file.is_dir():
        case_files = sorted(input_file.glob("*_st_test_cases.csv"), key=lambda path: path.name.lower())
    if not case_files:
        raise FileNotFoundError(f"No operation case files found in: {input_file}")

    specs: List[Tuple[str, Path, Path]] = []
    discovered_ops = set()
    filtered_case_files = []
    for case_file in case_files:
        case_op = infer_op_from_case_file(case_file)
        discovered_ops.add(case_op)
        if selected_ops and case_op not in selected_ops:
            continue
        filtered_case_files.append((case_op, case_file))

    if selected_ops:
        missing_ops = sorted(selected_ops - discovered_ops)
        if missing_ops:
            raise FileNotFoundError(f"Can not find case files for operations: {', '.join(missing_ops)}")
    if not filtered_case_files:
        raise ValueError("No operation case files selected.")

    total_case_files = len(filtered_case_files)
    for case_op, case_file in filtered_case_files:
        specs.append((case_op, case_file, resolve_json_file(json_path, case_file, total_case_files)))
    return specs


def run_cmd(cmd: List[str]) -> None:
    logging.info("Run command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_PATH, check=True)


def is_number(input_str: str) -> bool:
    try:
        value = float(input_str)
        return value == value and value not in {float("inf"), float("-inf")}
    except (TypeError, ValueError):
        return False


def parse_number(input_str: str) -> Union[int, float]:
    try:
        return int(input_str)
    except ValueError:
        return float(input_str)


def parse_list_str(input_str: Optional[str]) -> Any:
    if input_str is None:
        return None
    text = str(input_str).strip().replace(" ", "")
    if text == "":
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text == "":
        return []

    ret_list = []
    element_split_ident = " "
    if "{" in text:
        element_split_ident = "},{"
    if "[" in text:
        element_split_ident = "],["
    if element_split_ident in text:
        for sub_str in text.split(element_split_ident):
            ret_list.append(parse_list_str(sub_str))
    else:
        for sub_str in text.split(","):
            if sub_str == "":
                continue
            if not is_number(sub_str):
                ret_list.append(sub_str)
            else:
                ret_list.append(parse_number(sub_str))
    return ret_list


def parse_dict_str(input_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if input_str is None:
        return None
    text = str(input_str).strip().replace(" ", "")
    if text == "":
        return {}
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if text == "":
        return {}

    key_values = text.split(",")
    result: Dict[str, Any] = {}
    value_index = 0
    while value_index < len(key_values):
        if ":" in key_values[value_index]:
            key, value = key_values[value_index].split(":", 1)
            while value_index + 1 < len(key_values) and ":" not in key_values[value_index + 1]:
                value += "," + key_values[value_index + 1]
                value_index += 1
            result[key] = parse_scalar(value)
        value_index += 1
    return result


def parse_bool_str(input_str: Optional[str]) -> bool:
    if input_str is None:
        return False
    return str(input_str).strip().upper() in {"TRUE", "1"}


def parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, list, dict)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.upper() in {"TRUE", "FALSE"}:
        return parse_bool_str(text)
    if text.startswith("[") and text.endswith("]"):
        return parse_list_str(text)
    if text.startswith("{") and text.endswith("}") and ":" in text:
        return parse_dict_str(text)
    if is_number(text):
        return parse_number(text)
    return text


def normalize_view_shape(view_shape: Any) -> Any:
    parsed = parse_list_str(view_shape)
    is_nested_list = isinstance(parsed, list) and parsed and isinstance(parsed[0], list)
    if is_nested_list and len(parsed[0]) > 1:
        return parsed[0]
    return parsed


def build_input_tensors(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_shapes = parse_list_str(row.get("input_shape"))
    if input_shapes and not isinstance(input_shapes[0], list):
        input_shapes = [input_shapes]
    input_dtypes = parse_list_str(row.get("input_dtype"))
    input_formats = parse_list_str(row.get("input_format"))
    input_ranges = parse_list_str(row.get("input_datarange"))
    if input_ranges and not isinstance(input_ranges[0], list):
        input_ranges = [input_ranges]
    input_trans = parse_list_str(row.get("input_trans")) if row.get("input_trans") else []
    if input_trans and not isinstance(input_trans, list):
        input_trans = [input_trans]

    tensors = []
    for idx, shape in enumerate(input_shapes):
        tensor = {
            "name": f"input{idx}",
            "shape": shape,
            "dtype": input_dtypes[idx],
            "format": input_formats[idx],
        }
        if input_ranges and idx < len(input_ranges) and input_ranges[idx] is not None:
            tensor["data_range"] = {"min": input_ranges[idx][0], "max": input_ranges[idx][1]}
        if input_trans and idx < len(input_trans):
            tensor["need_trans"] = parse_bool_str(str(input_trans[idx]))
        tensors.append(tensor)
    return tensors


def build_output_tensors(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_shapes = parse_list_str(row.get("output_shape"))
    if output_shapes and not isinstance(output_shapes[0], list):
        output_shapes = [output_shapes]
    output_dtypes = parse_list_str(row.get("output_dtype"))
    output_formats = parse_list_str(row.get("output_format"))

    tensors = []
    for idx, shape in enumerate(output_shapes):
        tensors.append(
            {
                "name": f"output{idx}",
                "shape": shape,
                "dtype": output_dtypes[idx],
                "format": output_formats[idx],
            }
        )
    return tensors


def build_params(row: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key, value in row.items():
        if key in BASE_CASE_COLUMNS:
            continue
        params[key] = parse_scalar(value)
    if params.get("func_id") is None:
        params["func_id"] = -1
    return params


def should_skip_row(row: Dict[str, Any]) -> bool:
    if parse_bool_str(row.get("skip")):
        return True
    enable_value = row.get("enable")
    if enable_value is not None and str(enable_value).strip() != "":
        return not parse_bool_str(enable_value)
    return False


def load_csv_rows(case_file: Path, op: str, start_index: int, end_index: int) -> List[Dict[str, Any]]:
    with open(case_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    selected_rows: List[Dict[str, Any]] = []
    normalized_start = max(start_index, 0)
    normalized_end = len(rows) - 1 if end_index < 0 or end_index >= len(rows) else end_index
    if normalized_start >= len(rows):
        logging.info("The start index [%s] exceeds the max index[%s].", normalized_start, len(rows) - 1)
        return []

    for row_index, row in enumerate(rows):
        if row_index < normalized_start or row_index > normalized_end:
            continue
        normalized_row = dict(row)
        normalized_row.setdefault("operation", op)
        if not normalized_row.get("operation"):
            normalized_row["operation"] = op
        if should_skip_row(normalized_row):
            continue
        normalized_row["index"] = row_index
        normalized_row["case_index"] = parse_scalar(normalized_row.get("case_index"))
        if normalized_row["case_index"] is None:
            normalized_row["case_index"] = row_index
        selected_rows.append(normalized_row)
    return selected_rows


def build_test_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    test_cases = []
    for row in rows:
        test_cases.append(
            {
                "index": row["index"],
                "case_index": row["case_index"],
                "case_name": row["case_name"],
                "operation": row["operation"],
                "input_tensors": build_input_tensors(row),
                "output_tensors": build_output_tensors(row),
                "view_shape": normalize_view_shape(row.get("view_shape")),
                "tile_shape": parse_list_str(row.get("tile_shape")),
                "params": build_params(row),
            }
        )
    test_cases.sort(key=lambda item: item["case_index"])
    return test_cases


def write_json_file(json_file: Path, test_cases: List[Dict[str, Any]]) -> None:
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as outfile:
        json.dump({"test_cases": test_cases}, outfile, ensure_ascii=False, indent=4)


def generate_json(case_file: Path, op: str, start_index: int, end_index: int, json_path: Path) -> List[Dict[str, Any]]:
    if case_file.suffix.lower() != ".csv":
        raise ValueError(f"Only csv case files are supported currently: {case_file}")
    canonical_json_path = DEFAULT_OPERATION_CASE_DIR / f"{op}_st_test_cases.json"
    all_rows = load_csv_rows(case_file=case_file, op=op, start_index=0, end_index=-1)
    all_test_cases = build_test_cases(all_rows)
    write_json_file(canonical_json_path, all_test_cases)

    selected_rows = load_csv_rows(case_file=case_file, op=op, start_index=start_index, end_index=end_index)
    selected_test_cases = build_test_cases(selected_rows)
    if json_path != canonical_json_path:
        write_json_file(json_path, selected_test_cases)
    return selected_test_cases


def to_stest_case_names(test_cases: Iterable[dict]) -> List[str]:
    case_names = []
    for test_case in test_cases:
        op = test_case["operation"]
        index = test_case["index"]
        case_names.append(f"Test{op}/{op}OperationTest.Test{op}/{index}")
    return case_names


def generate_golden(case_names: List[str], output_dir: Path, impl_paths: List[Path], clean: bool, job_num: int) -> None:
    cmd = [
        sys.executable,
        str(GOLDEN_CTRL_SCRIPT),
        f"--cases={':'.join(case_names)}",
        f"--output={output_dir}",
        f"--job_num={job_num}",
    ]
    for impl_path in impl_paths:
        cmd.append(f"--path={impl_path}")
    if clean:
        cmd.append("--clean")
    run_cmd(cmd)


def validate_args(input_file: Path, json_path: Path, output_dir: Path, impl_paths: List[Path]) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file or directory does not exist: {input_file}")
    if json_path.suffix:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        json_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for impl_path in impl_paths:
        if not impl_path.exists():
            raise FileNotFoundError(f"Golden implementation path does not exist: {impl_path}")


def main() -> int:
    args = parse_args()
    input_file = resolve_input_file(op=args.op, input_file=args.input_file)
    json_path = resolve_json_path(json_path=args.json_path)
    output_dir = Path(args.golden_output).resolve()
    impl_paths = resolve_impl_paths(args.golden_impl_path)

    logging.info("Operation golden args:")
    logging.info("Op: %s", args.op)
    logging.info("Input file: %s", input_file)
    logging.info("Start index: %s", args.start_index)
    logging.info("End index: %s", args.end_index)
    logging.info("JSON path: %s", json_path)
    logging.info("Golden output: %s", output_dir)
    logging.info("Golden impl paths: %s", ", ".join(str(path) for path in impl_paths))
    logging.info("Job num: %s", args.job_num)
    logging.info("Clean: %s", args.clean)

    validate_args(input_file=input_file, json_path=json_path, output_dir=output_dir, impl_paths=impl_paths)
    case_specs = collect_case_specs(op=args.op, input_file=input_file, json_path=json_path)

    case_names = []
    for case_op, case_file, case_json_path in case_specs:
        logging.info("Process op %s from %s", case_op, case_file)
        test_cases = generate_json(
            case_file=case_file,
            op=case_op,
            start_index=args.start_index,
            end_index=args.end_index,
            json_path=case_json_path,
        )
        logging.info("Write json file: %s", case_json_path)
        case_names.extend(to_stest_case_names(test_cases))

    if not case_names:
        logging.warning("No test cases selected, skip golden generation.")
        return 0
    generate_golden(
        case_names=case_names,
        output_dir=output_dir,
        impl_paths=impl_paths,
        clean=args.clean,
        job_num=args.job_num,
    )
    logging.info("Generate operation golden finish, case count: %s", len(case_names))
    return 0


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    sys.exit(main())
