#!/usr/bin/env python3
"""Shared, dependency-free helpers for Kernel Design Wiki tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


SKILL_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def dump_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_workspace(start: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    for base in (start or Path.cwd(), SKILL_ROOT):
        resolved = base.resolve()
        candidates.append(resolved)
        candidates.extend(resolved.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "archive" / "CATALOG.md").exists():
            return candidate
    return None


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {scalar_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(scalar_text(item) for item in value)
    return str(value)


def tokens(value: str) -> set[str]:
    lowered = value.lower()
    latin = re.findall(r"[a-z0-9_+.-]+", lowered)
    chinese = re.findall(r"[\u3400-\u9fff]{2,}", lowered)
    return set(latin + chinese)


def load_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    repos = load_jsonl(SKILL_ROOT / "sources" / "repos.jsonl")
    evidence = load_jsonl(SKILL_ROOT / "sources" / "evidence.jsonl")
    relations = load_jsonl(SKILL_ROOT / "sources" / "relations.jsonl")
    cards = load_jsonl(SKILL_ROOT / "wiki" / "cards.jsonl")
    return repos, evidence, relations, cards
