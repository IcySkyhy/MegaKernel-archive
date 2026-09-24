#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""从 GitCode PR 评论 JSON 中提取最新 codecheck URL。

说明:
- 本脚本不调用任何 GitCode API。
- 输入必须是已有的评论 JSON 数据（通常为数组，每个元素含 body 和 created_at）。

用法示例:
  python3 /tmp/extract_latest_codecheck_url.py --input comments.json
  cat comments.json | python3 /tmp/extract_latest_codecheck_url.py
  python3 /tmp/extract_latest_codecheck_url.py --input comments.json --evidence
  python3 /tmp/extract_latest_codecheck_url.py --input comments.json --gate-on-latest-ci --format json

退出码:
  0: 成功找到并输出 URL
  3: 最新 CI 失败但失败任务不是 codecheck
  4: 无法判定最新 CI 结果或缺少 codecheck 状态
  1: 输入错误或未找到 URL
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import re
import sys
from typing import Any, Iterable

CODECHECK_URL_RE = re.compile(
    r"https://www\.openlibing\.com/apps/entryCheckDashCode/[^\s'\">]+",
    flags=re.IGNORECASE,
)
CI_ROW_RE = re.compile(
    r"<td><strong>([^<]+)</strong></td>\s*<td>([^<]+)</td>",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 PR 评论 JSON 中提取最新 codecheck URL（不调用 GitCode API）")
    parser.add_argument(
        "--input",
        help="评论 JSON 文件路径；不传时从 stdin 读取",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="输出完整 JSON 证据链（包含所有匹配的 codecheck URL）",
    )
    parser.add_argument(
        "--gate-on-latest-ci",
        action="store_true",
        help="基于最新 CI 报告判断是否由 codecheck 导致失败；仅 codecheck 失败时输出 URL/证据链",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式（默认 text）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息（comment_id、created_at、url）",
    )
    return parser.parse_args()


def load_json(input_path: str | None) -> Any:
    try:
        if input_path:
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError(f"输入不是合法 JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"读取输入失败: {exc}") from exc


def parse_time(ts: Any) -> datetime:
    if not isinstance(ts, str) or not ts.strip():
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def iter_comments(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for key in ("comments", "data", "items"):
            maybe = data.get(key)
            if isinstance(maybe, list):
                for item in maybe:
                    if isinstance(item, dict):
                        yield item
                return
        raise ValueError("JSON 对象中未找到 comments/data/items 数组")
    else:
        raise ValueError("JSON 顶层必须是数组或对象")


def extract_latest_codecheck(comments: Iterable[dict[str, Any]]) -> tuple[int | None, str | None, str]:
    latest_key: tuple[datetime, int] | None = None
    latest_comment_id: int | None = None
    latest_created_at: str | None = None
    latest_url: str | None = None

    for c in comments:
        body = c.get("body")
        if not isinstance(body, str) or "codecheck" not in body.lower():
            continue

        url_matches = list(CODECHECK_URL_RE.finditer(body))
        if not url_matches:
            continue

        created_at_raw = c.get("created_at")
        created_at_dt = parse_time(created_at_raw)
        cid = c.get("id")
        cid_num = cid if isinstance(cid, int) else -1
        key = (created_at_dt, cid_num)

        if latest_key is None or key > latest_key:
            latest_key = key
            latest_comment_id = cid if isinstance(cid, int) else None
            latest_created_at = created_at_raw if isinstance(created_at_raw, str) else None
            latest_url = url_matches[0].group(0)

    if latest_url is None:
        raise ValueError("未在评论中找到 codecheck URL")

    return latest_comment_id, latest_created_at, latest_url


def extract_with_evidence(comments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """提取所有 codecheck URL 并返回完整证据链。

    Returns:
        包含以下结构的字典：
        {
            "total_found": int,  # 共找到多少个 codecheck URL
            "latest": {
                "comment_id": int | None,
                "created_at": str | None,
                "url": str
            },
            "evidence_chain": [
                {
                    "index": int,
                    "comment_id": int | None,
                    "created_at": str | None,
                    "url": str,
                    "is_latest": bool
                },
                ...  # 按时间倒序排列
            ]
        }
    """
    matches: list[tuple[datetime, int, str | None, str]] = []  # (created_at, comment_id, created_at_raw, url)

    for c in comments:
        body = c.get("body")
        if not isinstance(body, str) or "codecheck" not in body.lower():
            continue

        url_matches = list(CODECHECK_URL_RE.finditer(body))
        if not url_matches:
            continue

        created_at_raw = c.get("created_at")
        created_at_dt = parse_time(created_at_raw)
        cid = c.get("id")
        cid_num = cid if isinstance(cid, int) else -1
        created_at_str = created_at_raw if isinstance(created_at_raw, str) else None
        for url_match in url_matches:
            matches.append((created_at_dt, cid_num, created_at_str, url_match.group(0)))

    if not matches:
        raise ValueError("未在评论中找到 codecheck URL")

    # 按时间倒序排序（最新的在前）
    matches.sort(reverse=True, key=lambda x: (x[0], x[1]))

    # 构建证据链
    evidence_chain = []
    for idx, (created_at_dt, cid_num, created_at_raw, url) in enumerate(matches, start=1):
        evidence_chain.append(
            {
                "index": idx,
                "comment_id": cid_num if cid_num >= 0 else None,
                "created_at": created_at_raw,
                "url": url,
                "is_latest": idx == 1,
            }
        )

    # 提取最新的
    latest_match = matches[0]
    latest = {
        "comment_id": latest_match[1] if latest_match[1] >= 0 else None,
        "created_at": latest_match[2],
        "url": latest_match[3],
    }

    return {"total_found": len(matches), "latest": latest, "evidence_chain": evidence_chain}


def _status_is_failed(status: str) -> bool:
    s = (status or "").strip().lower()
    return "failed" in s or "❌" in s


def _status_is_success(status: str) -> bool:
    s = (status or "").strip().lower()
    return "success" in s or "✅" in s


def _extract_ci_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task, status in CI_ROW_RE.findall(body):
        rows.append({"task": task.strip(), "status": status.strip()})
    return rows


def analyze_latest_ci(comments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    latest_key: tuple[datetime, int] | None = None
    latest_comment: dict[str, Any] | None = None

    for c in comments:
        login = str(c.get("user", {}).get("login", "")).lower()
        body = c.get("body")
        if login != "cann-robot" or not isinstance(body, str):
            continue
        if "<td><strong>" not in body:
            continue

        rows = _extract_ci_rows(body)
        if not rows:
            continue

        created_at_raw = c.get("created_at")
        created_at_dt = parse_time(created_at_raw)
        cid = c.get("id")
        cid_num = cid if isinstance(cid, int) else -1
        key = (created_at_dt, cid_num)

        if latest_key is None or key > latest_key:
            latest_key = key
            latest_comment = c

    if latest_comment is None:
        return {
            "kind": "undecidable",
            "reason": "未找到可解析的最新 cann-robot CI 报告评论",
        }

    body = str(latest_comment.get("body", ""))
    rows = _extract_ci_rows(body)
    row_status_map = {r["task"].strip().lower(): r["status"] for r in rows}
    codecheck_status = row_status_map.get("codecheck")
    failed_tasks = [r for r in rows if _status_is_failed(r["status"]) and r["task"].strip().lower() != "codecheck"]
    codecheck_url_match = CODECHECK_URL_RE.search(body)
    codecheck_url = codecheck_url_match.group(0) if codecheck_url_match else None

    result: dict[str, Any] = {
        "latest_comment_id": latest_comment.get("id"),
        "latest_created_at": latest_comment.get("created_at"),
        "codecheck_status": codecheck_status,
        "failed_tasks": failed_tasks,
        "rows": rows,
    }

    if codecheck_status is None:
        result["kind"] = "undecidable"
        result["reason"] = "最新 CI 报告中未找到 codecheck 状态行"
        return result

    if _status_is_failed(codecheck_status):
        if not codecheck_url:
            result["kind"] = "undecidable"
            result["reason"] = "codecheck 失败但未找到 codecheck URL"
            return result
        result["kind"] = "codecheck_failed"
        result["codecheck_url"] = codecheck_url
        return result

    if _status_is_success(codecheck_status) and failed_tasks:
        result["kind"] = "non_codecheck_failed"
        return result

    result["kind"] = "undecidable"
    result["reason"] = "未命中可判定分支（可能 CI 全成功或状态未完整更新）"
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        data = load_json(args.input)

        comments = list(iter_comments(data))

        if args.gate_on_latest_ci:
            ci_result = analyze_latest_ci(comments)
            kind = ci_result.get("kind")
            if kind == "codecheck_failed":
                if args.evidence or args.format == "json":
                    evidence = extract_with_evidence(comments)
                    out = {
                        "kind": "codecheck_failed",
                        "codecheck_url": ci_result.get("codecheck_url"),
                        "latest_comment_id": ci_result.get("latest_comment_id"),
                        "latest_created_at": ci_result.get("latest_created_at"),
                        "codecheck_status": ci_result.get("codecheck_status"),
                        "failed_tasks": ci_result.get("failed_tasks", []),
                        "evidence": evidence,
                    }
                    logging.info(json.dumps(out, ensure_ascii=False, indent=2))
                else:
                    logging.info(ci_result.get("codecheck_url"))
                return 0

            if kind == "non_codecheck_failed":
                if args.format == "json":
                    logging.info(json.dumps(ci_result, ensure_ascii=False, indent=2))
                else:
                    logging.info("latest_ci_result=non_codecheck_failed")
                    logging.info(f"codecheck_status={ci_result.get('codecheck_status')}")
                    for task in ci_result.get("failed_tasks", []):
                        logging.info(f"failed_task={task.get('task')} status={task.get('status')}")
                return 3

            if args.format == "json":
                logging.info(json.dumps(ci_result, ensure_ascii=False, indent=2))
            else:
                logging.info("latest_ci_result=undecidable")
                logging.info(f"reason={ci_result.get('reason')}")
            return 4

        if args.evidence:
            # 证据链模式：输出完整 JSON
            result = extract_with_evidence(comments)
            logging.info(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # 默认模式：仅输出最新 URL
            comment_id, created_at, url = extract_latest_codecheck(comments)
            if args.verbose:
                logging.info("comment_id=%s", comment_id)
                logging.info("created_at=%s", created_at)
                logging.info("codecheck_url=%s", url)
            else:
                logging.info(url)
    except ValueError as exc:
        logging.error("ERROR: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
