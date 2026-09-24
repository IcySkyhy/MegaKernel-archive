#!/usr/bin/env python3
"""Build deterministic query indices from curated cards and evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import SKILL_ROOT, load_corpus


def add(mapping: dict[str, set[str]], values: list[str], card_id: str) -> None:
    for value in values:
        mapping[value].add(card_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SKILL_ROOT / "queries" / "index.json")
    args = parser.parse_args()
    repos, evidence, relations, cards = load_corpus()
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    by_kind: dict[str, set[str]] = defaultdict(set)
    by_tag: dict[str, set[str]] = defaultdict(set)
    by_hardware: dict[str, set[str]] = defaultdict(set)
    by_mechanism: dict[str, set[str]] = defaultdict(set)
    by_scope: dict[str, set[str]] = defaultdict(set)
    by_repo: dict[str, set[str]] = defaultdict(set)

    for card in cards:
        cid = card["id"]
        by_kind[card["kind"]].add(cid)
        add(by_tag, card.get("tags", []), cid)
        add(by_hardware, card.get("hardware", []), cid)
        add(by_mechanism, card.get("mechanisms", []), cid)
        add(by_scope, card.get("execution_scopes", []), cid)
        for evidence_id in card.get("evidence_ids", []):
            if evidence_id in evidence_by_id:
                by_repo[evidence_by_id[evidence_id]["repo_id"]].add(cid)

    def freeze(mapping: dict[str, set[str]]) -> dict[str, list[str]]:
        return {key: sorted(values) for key, values in sorted(mapping.items())}

    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "counts": {"repos": len(repos), "evidence": len(evidence), "relations": len(relations), "cards": len(cards)},
        "by_kind": freeze(by_kind),
        "by_tag": freeze(by_tag),
        "by_hardware": freeze(by_hardware),
        "by_mechanism": freeze(by_mechanism),
        "by_scope": freeze(by_scope),
        "by_repo": freeze(by_repo),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(cards)} cards, {len(evidence)} evidence records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
