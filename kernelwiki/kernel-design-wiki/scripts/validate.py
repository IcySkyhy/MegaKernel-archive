#!/usr/bin/env python3
"""Validate corpus structure, references, source locators, and dataset grouping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import SKILL_ROOT, find_workspace, load_corpus, load_jsonl


def require(row: dict[str, Any], fields: list[str], label: str, errors: list[str]) -> None:
    for field in fields:
        if field not in row or row[field] in (None, "", []):
            errors.append(f"{label}: missing required field {field}")


def unique(rows: list[dict[str, Any]], field: str, label: str, errors: list[str]) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value in seen:
            errors.append(f"{label}: duplicate {field}={value}")
        elif isinstance(value, str):
            seen.add(value)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, help="Root containing archive/CATALOG.md")
    args = parser.parse_args()
    repos, evidence, relations, cards = load_corpus()
    examples_path = SKILL_ROOT / "data" / "seed_examples.jsonl"
    examples = load_jsonl(examples_path) if examples_path.exists() else []
    vocabulary = json.loads((SKILL_ROOT / "data" / "vocabulary.json").read_text(encoding="utf-8"))
    workspace = args.workspace_root.resolve() if args.workspace_root else find_workspace()
    errors: list[str] = []

    repo_ids = unique(repos, "repo_id", "repos", errors)
    evidence_ids = unique(evidence, "evidence_id", "evidence", errors)
    relation_ids = unique(relations, "relation_id", "relations", errors)
    card_ids = unique(cards, "id", "cards", errors)
    unique(examples, "id", "examples", errors)

    for row in repos:
        require(row, ["repo_id", "lineage_id", "name", "local_root", "upstream", "category", "snapshot_date", "open_status", "technical_class"], row.get("repo_id", "repo"), errors)
        if row.get("open_status") not in vocabulary["open_status"]:
            errors.append(f"{row.get('repo_id')}: unknown open_status {row.get('open_status')}")
        if row.get("technical_class") not in vocabulary["technical_classes"]:
            errors.append(f"{row.get('repo_id')}: unknown technical_class {row.get('technical_class')}")
        if workspace and not (workspace / row.get("local_root", "")).exists():
            errors.append(f"{row.get('repo_id')}: local_root not found: {row.get('local_root')}")

    repo_by_id = {row["repo_id"]: row for row in repos if "repo_id" in row}
    for row in evidence:
        require(row, ["evidence_id", "repo_id", "claim", "claim_kind", "locator", "confidence"], row.get("evidence_id", "evidence"), errors)
        if row.get("repo_id") not in repo_ids:
            errors.append(f"{row.get('evidence_id')}: unknown repo_id {row.get('repo_id')}")
        if row.get("confidence") not in vocabulary["confidence"]:
            errors.append(f"{row.get('evidence_id')}: unknown confidence {row.get('confidence')}")
        locator = row.get("locator", {})
        if not isinstance(locator, dict) or not locator.get("path"):
            errors.append(f"{row.get('evidence_id')}: locator.path is required")
        elif workspace and row.get("repo_id") in repo_by_id:
            source_path = workspace / repo_by_id[row["repo_id"]]["local_root"] / locator["path"]
            if not source_path.exists():
                errors.append(f"{row.get('evidence_id')}: source locator not found: {source_path}")
        if "performance" in row:
            perf = row["performance"]
            require(perf, ["hardware", "device_count", "topology", "dtype", "shape", "baseline", "metric", "value", "timing_scope"], f"{row.get('evidence_id')}.performance", errors)

    all_entity_ids = repo_ids | evidence_ids | card_ids
    for row in relations:
        require(row, ["relation_id", "subject", "predicate", "object", "reason"], row.get("relation_id", "relation"), errors)
        if row.get("subject") not in all_entity_ids:
            errors.append(f"{row.get('relation_id')}: unknown subject {row.get('subject')}")
        if row.get("object") not in all_entity_ids:
            errors.append(f"{row.get('relation_id')}: unknown object {row.get('object')}")

    for row in cards:
        require(row, ["id", "kind", "title", "summary", "evidence_ids", "page", "split_group"], row.get("id", "card"), errors)
        if row.get("kind") not in vocabulary["card_kinds"]:
            errors.append(f"{row.get('id')}: unknown kind {row.get('kind')}")
        for evidence_id in row.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                errors.append(f"{row.get('id')}: unknown evidence_id {evidence_id}")
        page = SKILL_ROOT / row.get("page", "")
        if not page.is_file():
            errors.append(f"{row.get('id')}: page not found: {page}")

    for row in examples:
        require(row, ["id", "type", "prompt", "target", "evidence_ids", "split_group"], row.get("id", "example"), errors)
        if "lineage_ids" not in row or not isinstance(row.get("lineage_ids"), list):
            errors.append(f"{row.get('id')}: lineage_ids must be a list (it may be empty for definition-only samples)")
        for evidence_id in row.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                errors.append(f"{row.get('id')}: unknown evidence_id {evidence_id}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print(f"Validation passed: {len(repos)} repos, {len(evidence)} evidence records, {len(relation_ids)} relations, {len(cards)} cards, {len(examples)} examples.")
    if workspace is None:
        print("Source locator existence was skipped because archive/CATALOG.md was not found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
