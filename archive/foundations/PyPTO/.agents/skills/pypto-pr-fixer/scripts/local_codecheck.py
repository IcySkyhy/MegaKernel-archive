#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import TypedDict, TypeGuard, cast

logging.basicConfig(level=logging.INFO, format="%(message)s")

RULE_ORDER = [
    "__new__",
    "__init__",
    "__post_init__",
    "magic methods",
    "@property",
    "@staticmethod",
    "@classmethod",
    "public methods",
    "private methods",
]

AST_RULES = {"G.CLS.06", "G.LOG.02", "G.ERR.04", "G.TYP.04"}

RULE_DESCRIPTIONS = {
    "G.CLS.06": "Class methods should follow a consistent ordering.",
    "G.LOG.02": "Use logging instead of print().",
    "G.ERR.04": "Use exception chaining with raise ... from e.",
    "G.TYP.04": "Prefer `if seq` / `if not seq` over explicit empty-sequence checks.",
    "G.FMT.01": "Use 4-space indentation for code blocks.",
    "G.FMT.02": "Line length should not exceed 120 characters.",
    "G.FMT.03": "Use blank lines appropriately between classes/functions/methods.",
    "G.FMT.04": "Use spaces around keywords and operators consistently.",
    "G.FMT.05": "Imports should appear after module docstring/comments and before code.",
    "G.FMT.06": "Only one module should be imported per line.",
    "G.FMT.07": "Imports should be grouped and sorted.",
    "G.FMT.08": "Write only one statement per line.",
    "G.OPR.02": "Use `is` / `is not` when comparing to None.",
    "G.OPR.03": "Do not use `is` / `is not` for built-in type value comparison.",
    "G.OPR.05": "Use `is not` instead of `not ... is`.",
    "G.OPR.06": "Use `not in` for membership negation checks.",
    "G.EXP.03": "Do not assign lambda expressions to variables; use def instead.",
    "G.PRJ.03": "Do not leave debugger entry points in production code.",
    "G.TYP.08": "Use isinstance() for type checks.",
}

RUFF_RULE_MAP = {
    "E111": "G.FMT.01",
    "E112": "G.FMT.01",
    "E113": "G.FMT.01",
    "E114": "G.FMT.01",
    "E115": "G.FMT.01",
    "E116": "G.FMT.01",
    "E117": "G.FMT.01",
    "E201": "G.FMT.04",
    "E202": "G.FMT.04",
    "E203": "G.FMT.04",
    "E211": "G.FMT.04",
    "E221": "G.FMT.04",
    "E222": "G.FMT.04",
    "E223": "G.FMT.04",
    "E224": "G.FMT.04",
    "E225": "G.FMT.04",
    "E226": "G.FMT.04",
    "E227": "G.FMT.04",
    "E228": "G.FMT.04",
    "E231": "G.FMT.04",
    "E241": "G.FMT.04",
    "E242": "G.FMT.04",
    "E251": "G.FMT.04",
    "E252": "G.FMT.04",
    "E261": "G.FMT.04",
    "E262": "G.FMT.04",
    "E265": "G.FMT.04",
    "E266": "G.FMT.04",
    "E271": "G.FMT.04",
    "E272": "G.FMT.04",
    "E273": "G.FMT.04",
    "E274": "G.FMT.04",
    "E275": "G.FMT.04",
    "E301": "G.FMT.03",
    "E302": "G.FMT.03",
    "E303": "G.FMT.03",
    "E304": "G.FMT.03",
    "E305": "G.FMT.03",
    "E306": "G.FMT.03",
    "E501": "G.FMT.02",
    "E402": "G.FMT.05",
    "E701": "G.FMT.08",
    "E702": "G.FMT.08",
    "E703": "G.FMT.08",
    "E711": "G.OPR.02",
    "E713": "G.OPR.06",
    "E714": "G.OPR.05",
    "E721": "G.OPR.03",
    "E731": "G.EXP.03",
    "E401": "G.FMT.06",
    "I001": "G.FMT.07",
    "I002": "G.FMT.07",
    "T100": "G.PRJ.03",
    "PLC1802": "G.TYP.04",
}

RUFF_MULTI_RULE_MAP = {
    "E721": ["G.TYP.08"],
}

RUFF_TIMEOUT_SECONDS = 180

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "node_modules",
}

ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    rule_id: str
    description: str
    rule_description: str

    def to_dict(self) -> ViolationRecord:
        return {
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
        }


@dataclass(frozen=True)
class TextEdit:
    start: int
    end: int
    replacement: str


class ViolationRecord(TypedDict):
    file: str
    line: int
    description: str
    rule_id: str
    rule_description: str


class ResultRecord(TypedDict):
    total: int
    by_rule: dict[str, int]
    violations: list[ViolationRecord]


class RuffLocation(TypedDict, total=False):
    row: int
    column: int


class RuffDiagnostic(TypedDict, total=False):
    code: str
    filename: str
    message: str
    location: RuffLocation


class Args(argparse.Namespace):
    path: str = ""
    output: str = "json"
    rules: str | None = None
    fix: bool = False


def normalize_rule_token(token: str) -> str:
    value = (token or "").strip().upper()
    if not value:
        return ""
    if value.startswith("RUFF.") and re.fullmatch(r"RUFF\.[A-Z]\d{3}", value):
        return value
    if re.fullmatch(r"[A-Z]\d{3}", value):
        return f"RUFF.{value}"
    if re.fullmatch(r"[A-Z]{3}\.\d{1,2}", value):
        cat, num = value.split(".", 1)
        return f"G.{cat}.{int(num):02d}"
    m = re.fullmatch(r"G\.([A-Z]{3})\.(\d{1,2})", value)
    if m:
        return f"G.{m.group(1)}.{int(m.group(2)):02d}"
    return value


def parse_rule_filter(raw: str | None) -> set[str]:
    if not raw:
        return set()
    tokens = raw.replace(";", ",").split(",")
    return {t for t in (normalize_rule_token(x) for x in tokens) if t}


def is_rule_enabled(rule_id: str, enabled: set[str]) -> bool:
    return not enabled or rule_id in enabled


def should_run_ruff(enabled: set[str]) -> bool:
    return not enabled or not enabled.issubset(AST_RULES)


def line_start_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def to_offset(offsets: list[int], line: int, column: int) -> int:
    if line <= 0:
        return 0
    index = line - 1
    if index >= len(offsets):
        return offsets[-1]
    return offsets[index] + column


def format_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir).as_posix()
    except ValueError:
        return resolved.as_posix()


def collect_python_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target.resolve()] if target.suffix == ".py" else []
    files: list[Path] = []
    for path in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path.resolve())
    return sorted(files)


def read_source_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    except Exception:
        return None


def ruff_binary() -> list[str]:
    if shutil.which("ruff"):
        return ["ruff"]
    return [sys.executable, "-m", "ruff"]


def normalize_ruff_diagnostics(payload: object) -> list[RuffDiagnostic]:
    raw_items_obj: object
    if isinstance(payload, list):
        raw_items_obj = cast(list[object], payload)
    elif isinstance(payload, dict):
        payload_dict = cast(dict[str, object], payload)
        raw_items_obj = payload_dict.get("diagnostics")
    else:
        raise RuntimeError("Unsupported ruff JSON schema")

    if not isinstance(raw_items_obj, list):
        raise RuntimeError("Unsupported ruff JSON schema")

    raw_items = cast(list[object], raw_items_obj)
    diagnostics: list[RuffDiagnostic] = []
    for raw_obj in raw_items:
        if not isinstance(raw_obj, dict):
            continue
        raw = cast(dict[str, object], raw_obj)
        entry: RuffDiagnostic = {}

        code = raw.get("code")
        if isinstance(code, str):
            entry["code"] = code

        filename = raw.get("filename")
        if isinstance(filename, str):
            entry["filename"] = filename

        message = raw.get("message")
        if isinstance(message, str):
            entry["message"] = message

        location_raw = raw.get("location")
        if isinstance(location_raw, dict):
            location_dict = cast(dict[str, object], location_raw)
            location: RuffLocation = {}
            row = location_dict.get("row")
            if isinstance(row, int):
                location["row"] = row
            column = location_dict.get("column")
            if isinstance(column, int):
                location["column"] = column
            if location:
                entry["location"] = location

        diagnostics.append(entry)

    return diagnostics


def run_ruff_check(target: Path, fix: bool) -> list[RuffDiagnostic]:
    cmd = [
        *ruff_binary(),
        "check",
        str(target),
        "--select",
        "E,F,I,T10,PLC",
        "--preview",
        "--line-length",
        "120",
        "--output-format",
        "json",
    ]

    if fix:
        try:
            fix_proc = subprocess.run(
                [*cmd, "--fix"],
                capture_output=True,
                text=True,
                timeout=RUFF_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ruff --fix timed out after {RUFF_TIMEOUT_SECONDS}s") from exc
        if fix_proc.returncode not in (0, 1):
            msg = fix_proc.stderr.strip() or fix_proc.stdout.strip()
            raise RuntimeError(f"ruff --fix failed: {msg}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RUFF_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ruff check timed out after {RUFF_TIMEOUT_SECONDS}s") from exc
    if proc.returncode not in (0, 1):
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"ruff check failed: {msg}")

    payload = proc.stdout.strip()
    if not payload:
        return []
    try:
        parsed = cast(object, json.loads(payload))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ruff JSON output: {exc}") from exc
    return normalize_ruff_diagnostics(parsed)


def map_ruff_rule(code: str) -> tuple[str, str]:
    mapped = RUFF_RULE_MAP.get(code)
    if mapped:
        return mapped, RULE_DESCRIPTIONS.get(mapped, "")
    rule_id = f"RUFF.{code}"
    return rule_id, f"Ruff rule {code}"


def is_property_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "property"
    if isinstance(node, ast.Attribute):
        return node.attr in {"setter", "getter", "deleter"}
    return False


def classify_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    name = node.name
    if name == "__new__":
        return 0
    if name == "__init__":
        return 1
    if name == "__post_init__":
        return 2
    if name.startswith("__") and name.endswith("__"):
        return 3
    if any(is_property_decorator(dec) for dec in node.decorator_list):
        return 4
    if any(isinstance(dec, ast.Name) and dec.id == "staticmethod" for dec in node.decorator_list):
        return 5
    if any(isinstance(dec, ast.Name) and dec.id == "classmethod" for dec in node.decorator_list):
        return 6
    if name.startswith("_"):
        return 8
    return 7


def check_class_method_order(
    class_node: ast.ClassDef,
    file_name: str,
) -> list[Violation]:
    methods = [stmt for stmt in class_node.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(methods) < 2:
        return []

    violations: list[Violation] = []
    max_seen = -1
    order_str = " -> ".join(RULE_ORDER)
    for method in methods:
        group = classify_method(method)
        if group < max_seen:
            description = (
                f"Class '{class_node.name}' method '{method.name}' is out of order. Expected order: {order_str}."
            )
            violations.append(
                Violation(
                    file=file_name,
                    line=method.lineno,
                    rule_id="G.CLS.06",
                    description=description,
                    rule_description=RULE_DESCRIPTIONS["G.CLS.06"],
                )
            )
        if group >= max_seen:
            max_seen = group
    return violations


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def enclosing_except_handler(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.ExceptHandler | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return None
        if isinstance(current, ast.ExceptHandler):
            return current
        current = parents.get(current)
    return None


def raise_needs_chaining(
    node: ast.Raise,
    parents: dict[ast.AST, ast.AST],
) -> tuple[bool, str | None]:
    if node.exc is None:
        return False, None
    if node.cause is not None:
        return False, None
    handler = enclosing_except_handler(node, parents)
    if handler is None:
        return False, None
    alias = handler.name if isinstance(handler.name, str) else None
    if alias and isinstance(node.exc, ast.Name) and node.exc.id == alias:
        return False, alias
    return True, alias


def is_len_call(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "len" and len(expr.args) == 1
    )


def is_zero_int(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and isinstance(expr.value, int) and expr.value == 0


def is_empty_literal(expr: ast.expr) -> bool:
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return len(expr.elts) == 0
    if isinstance(expr, ast.Dict):
        return len(expr.keys) == 0
    return False


def check_typ_04_compare(node: ast.Compare) -> bool:
    if len(node.comparators) != 1 or len(node.ops) != 1:
        return False
    left = node.left
    right = node.comparators[0]
    op = node.ops[0]

    if is_len_call(left) and is_zero_int(right) and isinstance(op, (ast.Eq, ast.NotEq, ast.Gt, ast.LtE)):
        return True
    if is_len_call(right) and is_zero_int(left) and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.GtE)):
        return True

    left_empty = is_empty_literal(left)
    right_empty = is_empty_literal(right)
    if isinstance(op, (ast.Eq, ast.NotEq)):
        if left_empty != right_empty:
            return True
    return False


def has_logging_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "logging" and (item.asname is None or item.asname == "logging"):
                    return True
    return False


def logging_import_insert_offset(
    source: str,
    tree: ast.Module,
    offsets: list[int],
) -> int:
    lines = source.splitlines(keepends=True)
    line = 1
    if lines and lines[0].startswith("#!"):
        line = 2
    if line - 1 < len(lines) and ENCODING_RE.search(lines[line - 1]):
        line += 1

    body_index = 0
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            line = max(line, (tree.body[0].end_lineno or tree.body[0].lineno) + 1)
            body_index = 1

    while body_index < len(tree.body):
        node = tree.body[body_index]
        if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            break
        line = max(line, (node.end_lineno or node.lineno) + 1)
        body_index += 1

    if line - 1 >= len(offsets):
        return len(source)
    return offsets[line - 1]


def is_print_call(node: ast.AST) -> TypeGuard[ast.Call]:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def unique_edits(edits: list[TextEdit]) -> list[TextEdit]:
    seen: set[tuple[int, int, str]] = set()
    out: list[TextEdit] = []
    for edit in edits:
        key = (edit.start, edit.end, edit.replacement)
        if key in seen:
            continue
        seen.add(key)
        out.append(edit)
    out.sort(key=lambda e: (e.start, e.end))
    return out


def apply_edits(source: str, edits: list[TextEdit]) -> str:
    if not edits:
        return source
    cleaned: list[TextEdit] = []
    current_end = -1
    for edit in edits:
        if edit.start < current_end:
            continue
        cleaned.append(edit)
        current_end = edit.end
    result = source
    for edit in reversed(cleaned):
        result = result[:edit.start] + edit.replacement + result[edit.end:]
    return result


def collect_ast_violations(
    tree: ast.Module,
    file_name: str,
    enabled_rules: set[str],
) -> list[Violation]:
    violations: list[Violation] = []

    if is_rule_enabled("G.CLS.06", enabled_rules):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                violations.extend(check_class_method_order(node, file_name))

    if is_rule_enabled("G.LOG.02", enabled_rules):
        for node in ast.walk(tree):
            if is_print_call(node):
                violations.append(
                    Violation(
                        file=file_name,
                        line=node.lineno,
                        rule_id="G.LOG.02",
                        description="Use logging instead of print().",
                        rule_description=RULE_DESCRIPTIONS["G.LOG.02"],
                    )
                )

    if is_rule_enabled("G.ERR.04", enabled_rules):
        parents = build_parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            needed, alias = raise_needs_chaining(node, parents)
            if not needed:
                continue
            if alias:
                description = f"Use `raise ... from {alias}` to preserve traceback."
            else:
                description = "Use exception chaining (`raise ... from e`) in except blocks."
            violations.append(
                Violation(
                    file=file_name,
                    line=node.lineno,
                    rule_id="G.ERR.04",
                    description=description,
                    rule_description=RULE_DESCRIPTIONS["G.ERR.04"],
                )
            )

    if is_rule_enabled("G.TYP.04", enabled_rules):
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and check_typ_04_compare(node):
                violations.append(
                    Violation(
                        file=file_name,
                        line=node.lineno,
                        rule_id="G.TYP.04",
                        description="Prefer `if seq` / `if not seq` over explicit empty-sequence comparisons.",
                        rule_description=RULE_DESCRIPTIONS["G.TYP.04"],
                    )
                )

    return violations


def build_ast_fixes(
    source: str,
    tree: ast.Module,
    enabled_rules: set[str],
) -> list[TextEdit]:
    offsets = line_start_offsets(source)
    edits: list[TextEdit] = []

    replaced_print = False
    if is_rule_enabled("G.LOG.02", enabled_rules):
        for node in ast.walk(tree):
            if not is_print_call(node):
                continue
            start = to_offset(offsets, node.lineno, node.col_offset)
            if source[start:start + 5] != "print":
                continue
            edits.append(TextEdit(start=start, end=start + 5, replacement="logging.info"))
            replaced_print = True

        if replaced_print and not has_logging_import(tree):
            insert_at = logging_import_insert_offset(source, tree, offsets)
            edits.append(TextEdit(start=insert_at, end=insert_at, replacement="import logging\n"))

    if is_rule_enabled("G.ERR.04", enabled_rules):
        parents = build_parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            needed, alias = raise_needs_chaining(node, parents)
            if not needed or not alias:
                continue
            end_line = node.end_lineno if node.end_lineno is not None else node.lineno
            end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
            if node.lineno != end_line:
                continue
            start = to_offset(offsets, node.lineno, node.col_offset)
            end = to_offset(offsets, end_line, end_col)
            segment = source[start:end]
            if " from " in segment:
                continue
            edits.append(TextEdit(start=start, end=end, replacement=f"{segment} from {alias}"))

    return unique_edits(edits)


def run_ast_checks(
    target: Path,
    base_dir: Path,
    enabled_rules: set[str],
    fix: bool,
) -> list[Violation]:
    files = collect_python_files(target)
    violations: list[Violation] = []

    for file_path in files:
        display_file = format_path(file_path, base_dir)
        source = read_source_file(file_path)
        if source is None:
            if not enabled_rules or "PY.IO" in enabled_rules:
                violations.append(
                    Violation(
                        file=display_file,
                        line=1,
                        rule_id="PY.IO",
                        description="Failed to read file content for analysis.",
                        rule_description="File read failure during local pre-check.",
                    )
                )
            continue
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            if not enabled_rules or "PY.SYNTAX" in enabled_rules:
                violations.append(
                    Violation(
                        file=display_file,
                        line=exc.lineno or 1,
                        rule_id="PY.SYNTAX",
                        description=f"Syntax error: {exc.msg}",
                        rule_description="Python syntax error.",
                    )
                )
            continue

        if fix:
            edits = build_ast_fixes(source, tree, enabled_rules)
            if edits:
                fixed_source = apply_edits(source, edits)
                if fixed_source != source:
                    _ = file_path.write_text(fixed_source, encoding="utf-8")
                    source = fixed_source
                    tree = ast.parse(source, filename=str(file_path))

        violations.extend(collect_ast_violations(tree, display_file, enabled_rules))

    return violations


def run_ruff_violations(
    target: Path,
    base_dir: Path,
    enabled_rules: set[str],
    fix: bool,
) -> list[Violation]:
    diagnostics = run_ruff_check(target, fix)
    violations: list[Violation] = []
    for item in diagnostics:
        code = str(item.get("code") or "").upper()
        if not code:
            continue
        mapped_rule, mapped_desc = map_ruff_rule(code)
        raw_rule = f"RUFF.{code}"
        if enabled_rules and mapped_rule not in enabled_rules and raw_rule not in enabled_rules:
            continue

        filename = str(item.get("filename") or "")
        if not filename:
            continue
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        line = int(item.get("location", {}).get("row") or 1)
        message = str(item.get("message") or "")
        violations.append(
            Violation(
                file=format_path(file_path, base_dir),
                line=line,
                rule_id=mapped_rule,
                description=message,
                rule_description=mapped_desc,
            )
        )
        for extra_rule in RUFF_MULTI_RULE_MAP.get(code, []):
            if enabled_rules and extra_rule not in enabled_rules:
                continue
            violations.append(
                Violation(
                    file=format_path(file_path, base_dir),
                    line=line,
                    rule_id=extra_rule,
                    description=message,
                    rule_description=RULE_DESCRIPTIONS.get(extra_rule, ""),
                )
            )
    return violations


def build_result(violations: list[Violation]) -> ResultRecord:
    ordered = sorted(violations, key=lambda v: (v.file, v.line, v.rule_id, v.description))
    by_rule = Counter(v.rule_id for v in ordered)
    return {
        "total": len(ordered),
        "by_rule": dict(sorted(by_rule.items())),
        "violations": [v.to_dict() for v in ordered],
    }


def print_text(result: ResultRecord) -> None:
    logging.info(f"Total violations: {result['total']}")
    if not result["violations"]:
        return
    grouped: dict[str, list[ViolationRecord]] = {}
    for item in result["violations"]:
        grouped.setdefault(item["rule_id"], []).append(item)
    for rule_id, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        logging.info(f"\n## {rule_id}: {len(items)}")
        for item in items:
            logging.info(f"- {item['file']}:{item['line']} {item['description']}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Local CodeCheck pre-checker (ruff + AST rules)",
    )
    _ = parser.add_argument("path", help="Target file or directory")
    _ = parser.add_argument("--output", "-o", choices=["json", "text"], default="json")
    _ = parser.add_argument("--rules", help="Comma-separated rules, e.g. G.CLS.06,G.LOG.02")
    _ = parser.add_argument("--fix", action="store_true", help="Auto-fix where possible")
    return parser.parse_args(namespace=Args())


def main() -> int:
    args = parse_args()
    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        logging.error("Error: path not found: %s", target)
        return 2

    enabled_rules = parse_rule_filter(args.rules)
    base_dir = target if target.is_dir() else target.parent

    violations: list[Violation] = []
    try:
        if should_run_ruff(enabled_rules):
            violations.extend(run_ruff_violations(target, base_dir, enabled_rules, args.fix))
    except RuntimeError as exc:
        logging.error("Error: %s", exc)
        return 2

    violations.extend(run_ast_checks(target, base_dir, enabled_rules, args.fix))

    result = build_result(violations)
    if args.output == "json":
        logging.info(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_text(result)

    return 1 if result["total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
