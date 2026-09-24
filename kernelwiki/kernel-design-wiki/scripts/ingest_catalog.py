#!/usr/bin/env python3
"""Parse archive/CATALOG.md into candidate project rows without making semantic claims."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _common import find_workspace


ROW4 = re.compile(
    r"^\|\s*\[(?P<name>[^]]+)\]\((?P<local>[^)]+)\)\s*\|\s*"
    r"\[(?P<upstream_name>[^]]+)\]\((?P<upstream>[^)]+)\)\s*\|\s*"
    r"(?P<status>[^|]+?)\s*\|\s*(?P<summary>.+?)\s*\|$"
)
ROW3 = re.compile(
    r"^\|\s*\[(?P<name>[^]]+)\]\((?P<local>[^)]+)\)\s*\|\s*"
    r"\[(?P<upstream_name>[^]]+)\]\((?P<upstream>[^)]+)\)\s*\|\s*"
    r"(?P<summary>.+?)\s*\|$"
)
SECTION = re.compile(r"^##\s+(?P<section>.+)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSONL instead of stdout")
    args = parser.parse_args()
    workspace = find_workspace()
    catalog = args.catalog or (workspace / "archive" / "CATALOG.md" if workspace else None)
    if not catalog or not catalog.is_file():
        parser.error("catalog not found; pass --catalog")

    rows = []
    section = ""
    for raw in catalog.read_text(encoding="utf-8").splitlines():
        section_match = SECTION.match(raw)
        if section_match:
            section = section_match.group("section")
            continue
        match = ROW4.match(raw) or ROW3.match(raw)
        if not match:
            continue
        data = match.groupdict()
        local = data["local"]
        rows.append({
            "candidate_id": "candidate-" + re.sub(r"[^a-z0-9]+", "-", data["name"].lower()).strip("-"),
            "name": data["name"],
            "local_root": f"archive/{local}",
            "upstream": data["upstream"],
            "catalog_section": section,
            "status_raw": (data.get("status") or "").strip(),
            "catalog_summary": data["summary"].strip(),
            "semantic_review_required": True,
        })

    rendered = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(rows)} candidates to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
