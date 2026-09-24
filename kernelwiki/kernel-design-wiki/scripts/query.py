#!/usr/bin/env python3
"""Query distilled MegaKernel design cards and optionally follow source evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import SKILL_ROOT, find_workspace, load_corpus, scalar_text, tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="", help="Natural-language query")
    parser.add_argument("--text", default="", help="Additional query text")
    parser.add_argument("--kind", choices=["mechanism", "decision", "case", "boundary"])
    parser.add_argument("--hardware")
    parser.add_argument("--mechanism")
    parser.add_argument("--scope")
    parser.add_argument("--repo")
    parser.add_argument("--confidence")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--follow-evidence", action="store_true")
    parser.add_argument("--archive-root", type=Path, help="Workspace root containing archive/CATALOG.md")
    return parser.parse_args()


def contains(values: Any, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in scalar_text(values).lower()


def main() -> int:
    args = parse_args()
    repos, evidence, _relations, cards = load_corpus()
    repo_by_id = {row["repo_id"]: row for row in repos}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}

    vocabulary = json.loads((SKILL_ROOT / "data" / "vocabulary.json").read_text(encoding="utf-8"))
    raw_query = " ".join(part for part in (args.query, args.text) if part).strip().lower()
    query_terms = tokens(raw_query)
    expanded_terms = set(query_terms)
    for canonical, aliases in vocabulary.get("aliases", {}).items():
        family = [canonical, *aliases]
        if any(str(item).lower() in raw_query for item in family):
            expanded_terms.update(tokens(" ".join(str(item) for item in family)))
            expanded_terms.add(canonical.lower())

    ranked: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for card in cards:
        if args.kind and card.get("kind") != args.kind:
            continue
        if not contains(card.get("hardware", []), args.hardware):
            continue
        if not contains(card.get("mechanisms", []), args.mechanism):
            continue
        if not contains(card.get("execution_scopes", []), args.scope):
            continue

        card_evidence = [evidence_by_id[eid] for eid in card.get("evidence_ids", []) if eid in evidence_by_id]
        if args.confidence and not any(contains(item.get("confidence"), args.confidence) for item in card_evidence):
            continue
        if args.repo:
            matching_repo = False
            for item in card_evidence:
                repo = repo_by_id.get(item.get("repo_id"), {})
                if contains([repo.get("repo_id"), repo.get("name"), repo.get("local_root")], args.repo):
                    matching_repo = True
                    break
            if not matching_repo:
                continue

        haystack = scalar_text(card).lower()
        score = 1 if not raw_query else 0
        if raw_query and raw_query in haystack:
            score += 8
        card_terms = tokens(haystack)
        score += 3 * len(query_terms & card_terms)
        score += len(expanded_terms & card_terms)
        for term in expanded_terms:
            if len(term) > 2 and term in haystack:
                score += 1
        if raw_query and score == 0:
            continue
        ranked.append((score, card, card_evidence))

    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    selected = ranked[: max(args.limit, 0)]
    workspace = args.archive_root.resolve() if args.archive_root else find_workspace()

    output: list[dict[str, Any]] = []
    for score, card, card_evidence in selected:
        item: dict[str, Any] = {
            "score": score,
            "id": card["id"],
            "kind": card["kind"],
            "title": card["title"],
            "summary": card["summary"],
            "page": str(SKILL_ROOT / card["page"]),
            "mechanisms": card.get("mechanisms", []),
            "hardware": card.get("hardware", []),
            "execution_scopes": card.get("execution_scopes", []),
        }
        if args.follow_evidence:
            followed: list[dict[str, Any]] = []
            for ev in card_evidence:
                repo = repo_by_id.get(ev["repo_id"], {})
                locator = ev.get("locator", {})
                relative = Path(repo.get("local_root", "")) / locator.get("path", "")
                followed.append({
                    "evidence_id": ev["evidence_id"],
                    "claim": ev["claim"],
                    "confidence": ev["confidence"],
                    "repo": repo.get("name"),
                    "locator": str(relative).replace("\\", "/"),
                    "absolute_path": str(workspace / relative) if workspace else None,
                    "symbol": locator.get("symbol"),
                    "caveats": ev.get("caveats", []),
                })
            item["evidence"] = followed
        output.append(item)

    if args.as_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    if not output:
        print("No matching design cards.")
        return 1
    for item in output:
        print(f"[{item['score']}] {item['id']} — {item['title']}")
        print(f"  {item['summary']}")
        print(f"  page: {item['page']}")
        if args.follow_evidence:
            for ev in item.get("evidence", []):
                print(f"  - {ev['evidence_id']} ({ev['confidence']}): {ev['locator']} :: {ev.get('symbol') or '-'}")
                print(f"    {ev['claim']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
