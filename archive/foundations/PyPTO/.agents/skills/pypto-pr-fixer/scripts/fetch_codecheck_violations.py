#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""从 openlibing.com CodeCheck 页面提取违规项。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import re
import time
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, Response, sync_playwright

WaitStrategy = Literal["domcontentloaded", "load", "networkidle", "commit"]


logging.basicConfig(level=logging.INFO, format="%(message)s")

VIOLATION_RE = re.compile(r"文件路径:([^\n:]+):(\d+)\s*问题描述[：:]([^\n]+)\s*规则[：:]([^\n]+)")
TASK_API_PATH = "/gateway/openlibing-codecheck/ci-portal/v1/event/codecheck/task"


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    description: str
    rule_id: str
    rule_description: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
        }


@dataclass(frozen=True)
class FetcherConfig:
    """Playwright fetcher configuration."""

    wait_strategies: list[WaitStrategy]
    nav_timeout_ms: int
    selector_timeout_ms: int
    post_wait_ms: int
    debug_dir: str
    api_page_size: int
    max_pages: int


@dataclass(frozen=True)
class TaskApiSeed:
    """CodeCheck task API seed info discovered from SPA requests."""

    api_url: str
    payload_template: dict[str, Any]
    total: int


class Args(argparse.Namespace):
    url: str = ""
    output: str = "json"
    group: bool = False
    retries: int = 3
    wait_strategies: str = "domcontentloaded,load,networkidle"
    nav_timeout_ms: int = 90000
    selector_timeout_ms: int = 15000
    post_wait_ms: int = 5000
    jitter_ms: int = 500
    debug_dir: str = ""
    api_page_size: int = 100
    max_pages: int = 200


def parse_violations_from_text(text: str) -> list[Violation]:
    violations: list[Violation] = []
    for raw_match in VIOLATION_RE.findall(text):
        file_path, line_text, description, rule = cast(tuple[str, str, str, str], raw_match)
        parts = rule.strip().split(" ", 1)
        rule_id = parts[0] if parts else ""
        rule_description = parts[1] if len(parts) > 1 else ""
        violations.append(
            Violation(
                file=file_path.strip(),
                line=int(line_text),
                description=description.strip(),
                rule_id=rule_id,
                rule_description=rule_description,
            )
        )
    return violations


def _dedup_violations(items: list[Violation]) -> list[Violation]:
    seen: set[tuple[str, int, str, str]] = set()
    result: list[Violation] = []
    for item in items:
        key = (item.file, item.line, item.description, item.rule_id)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _parse_wait_strategies(raw: str) -> list[WaitStrategy]:
    allowed: set[WaitStrategy] = {"domcontentloaded", "load", "networkidle", "commit"}
    parsed = [s.strip() for s in raw.split(",") if s.strip()]
    strategies: list[WaitStrategy] = []
    for strategy in parsed:
        if strategy in allowed:
            strategies.append(cast(WaitStrategy, strategy))
    if strategies:
        return strategies
    return cast(list[WaitStrategy], ["domcontentloaded", "load", "networkidle"])


def _save_debug_artifacts(page: object, debug_dir: str, attempt: int, stage: str) -> None:
    if not debug_dir:
        return

    debug_page = cast(Page, page)
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    prefix = debug_path / f"attempt{attempt}_{stage}_{ts}"

    try:
        debug_page.screenshot(path=str(prefix.with_suffix(".png")), full_page=True)
    except Exception as exc:  # pragma: no cover
        logging.debug("save screenshot failed: %s", exc)

    try:
        html = debug_page.content()
        prefix.with_suffix(".html").write_text(html, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logging.debug("save html failed: %s", exc)


def _collect_candidate_texts(page: object) -> list[str]:
    candidate_page = cast(Page, page)
    texts: list[str] = []

    try:
        texts.append(candidate_page.inner_text("body"))
    except Exception as exc:
        logging.debug("collect body text failed: %s", exc)

    try:
        row_texts = candidate_page.locator("tr").all_inner_texts()
        if row_texts:
            texts.append("\n".join(row_texts))
    except Exception as exc:
        logging.debug("collect tr texts failed: %s", exc)

    try:
        texts.append(candidate_page.content())
    except Exception as exc:
        logging.debug("collect page content failed: %s", exc)

    return texts


def _split_rule_text(rule_text: str, fallback_rule_id: str = "") -> tuple[str, str]:
    raw = (rule_text or "").strip()
    parts = raw.split(" ", 1)
    if parts and parts[0].strip():
        rid = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        return rid, desc
    return fallback_rule_id.strip(), raw


def _normalize_file_path(filepath: str, file_name: str) -> str:
    fp = (filepath or "").strip()
    fn = (file_name or "").strip()
    if fn and "/" in fn:
        if fp and fn.startswith(fp):
            return fn
        return fn
    if not fp:
        return fn
    if not fn:
        return fp
    if fp.endswith(fn):
        return fp
    return f"{fp.rstrip('/')}/{fn}"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _violation_from_api_defect(defect: dict[str, Any]) -> Violation:
    filepath = str(defect.get("filepath") or defect.get("filePath") or "").strip()
    file_name = str(defect.get("fileName") or "").strip()
    file_path = _normalize_file_path(filepath, file_name)

    line_number = _to_int(defect.get("lineNumber") or defect.get("lineNo") or 0, default=0)

    checker_name = str(defect.get("defectCheckerName") or "").strip()
    rule_id_fallback = str(defect.get("ruleId") or "").strip()
    rule_id, rule_description = _split_rule_text(checker_name, fallback_rule_id=rule_id_fallback)

    description = str(defect.get("defectContent") or defect.get("ruleName") or "").strip()

    return Violation(
        file=file_path,
        line=line_number,
        description=description,
        rule_id=rule_id,
        rule_description=rule_description,
    )


def _extract_task_seed_from_url(url: str) -> tuple[str, str, str | None]:
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    try:
        idx = segments.index("entryCheckDashCode")
        task_id = segments[idx + 1]
        uuid = segments[idx + 2]
    except Exception as exc:
        raise ValueError(f"无法从 URL 解析 taskId/uuid: {url}") from exc

    query = parse_qs(parsed.query)
    project_id = query.get("projectId", [None])[0]
    return task_id, uuid, project_id


def _capture_task_api_seed(page: object, url: str, config: FetcherConfig) -> TaskApiSeed | None:
    inspect_page = cast(Page, page)
    captured: list[TaskApiSeed] = []

    def _on_response(response: object) -> None:
        resp = cast(Response, response)
        req = resp.request
        if req.method.upper() != "POST":
            return
        if TASK_API_PATH not in resp.url:
            return

        try:
            body = _as_dict(resp.json())
            result = _as_dict(body.get("result"))
            total = _to_int(result.get("count"), default=0)

            payload_template: dict[str, Any] = {}
            raw_post_data = req.post_data
            if raw_post_data:
                payload_template = _as_dict(json.loads(raw_post_data))

            captured.append(TaskApiSeed(api_url=resp.url, payload_template=payload_template, total=total))
        except Exception as exc:  # pragma: no cover
            logging.debug("capture task api seed failed: %s", exc)

    inspect_page.on("response", _on_response)

    try:
        try:
            inspect_page.goto(url, wait_until="domcontentloaded", timeout=config.nav_timeout_ms)
        except Exception as exc:
            logging.debug("seed goto failed: %s", exc)

        try:
            inspect_page.wait_for_selector("body", timeout=config.selector_timeout_ms)
        except Exception as exc:
            logging.debug("seed selector wait skipped: %s", exc)

        inspect_page.wait_for_timeout(config.post_wait_ms)

        try:
            inspect_page.get_by_text("代码问题").first.click(timeout=3000)
            inspect_page.wait_for_timeout(1200)
        except Exception as exc:
            logging.debug("trigger issue tab skipped: %s", exc)

        if captured:
            return captured[-1]
    finally:
        try:
            inspect_page.remove_listener("response", _on_response)
        except Exception as exc:
            logging.debug("remove response listener failed: %s", exc)

    try:
        task_id, uuid, project_id = _extract_task_seed_from_url(url)
        api_url = f"https://www.openlibing.com{TASK_API_PATH}?uuid={uuid}&taskId={task_id}"
        payload_template: dict[str, Any] = {
            "pageNum": 1,
            "pageSize": max(20, config.api_page_size),
            "date": "",
            "defectLevel": "",
            "ruleType": "",
            "filePath": "",
            "fileName": "",
            "defectStatus": "0",
            "checkType": "",
            "trigger": "",
            "defectCheckerName": "",
            "isDelay": "",
            "flag": "1",
        }
        if project_id:
            payload_template["projectId"] = str(project_id)
        return TaskApiSeed(api_url=api_url, payload_template=payload_template, total=0)
    except Exception as exc:
        logging.debug("fallback seed from url failed: %s", exc)
        return None


def _fetch_all_violations_via_task_api(page: object, seed: TaskApiSeed, config: FetcherConfig) -> list[Violation]:
    fetch_page = cast(Page, page)
    all_items: list[Violation] = []
    total_expected = max(0, seed.total)

    for page_num in range(1, config.max_pages + 1):
        payload: dict[str, Any] = dict(seed.payload_template)
        payload["pageNum"] = page_num
        payload["pageSize"] = max(1, int(config.api_page_size))

        response = fetch_page.request.post(seed.api_url, data=payload, timeout=config.nav_timeout_ms)
        if response.status >= 400:
            raise RuntimeError(f"task api status={response.status}, pageNum={page_num}")

        body = response.json()
        result = _as_dict(_as_dict(body).get("result"))
        defects_raw = result.get("defects", [])
        if not isinstance(defects_raw, list) or not defects_raw:
            break

        page_items = [_violation_from_api_defect(item) for item in defects_raw if isinstance(item, dict)]
        all_items.extend(page_items)

        current_total = _to_int(result.get("count"), default=0)
        if current_total > 0:
            total_expected = current_total

        if total_expected > 0 and len(all_items) >= total_expected:
            break

    if total_expected > 0 and len(all_items) < total_expected:
        raise RuntimeError(
            f"task api incomplete: expected {total_expected}, collected {len(all_items)}, max_pages={config.max_pages}"
        )

    if not all_items:
        return []
    return _dedup_violations(all_items)


def _collect_violations_from_dom(page: object) -> list[Violation]:
    collected: list[Violation] = []
    for text in _collect_candidate_texts(page):
        collected.extend(parse_violations_from_text(text))
    return _dedup_violations(collected)


def extract_violations_with_playwright(
    url: str,
    config: FetcherConfig,
    attempt: int,
) -> list[Violation]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 2200})
            last_error: Exception | None = None

            for strategy in config.wait_strategies:
                try:
                    response = page.goto(url, wait_until=strategy, timeout=config.nav_timeout_ms)
                    status = response.status if response else None
                    logging.info("  strategy=%s status=%s", strategy, status)

                    try:
                        page.wait_for_selector("body", timeout=config.selector_timeout_ms)
                    except Exception as exc:
                        logging.debug("selector wait skipped: %s", exc)

                    page.wait_for_timeout(config.post_wait_ms)

                    seed = _capture_task_api_seed(page=page, url=url, config=config)
                    if seed is not None:
                        try:
                            api_violations = _fetch_all_violations_via_task_api(page=page, seed=seed, config=config)
                            if api_violations:
                                logging.info("  task api extracted %d violations", len(api_violations))
                                return api_violations
                        except Exception as exc:
                            logging.warning("  task api fetch failed, fallback to dom parse: %s", exc)

                    dom_violations = _collect_violations_from_dom(page)
                    if dom_violations:
                        logging.warning(
                            "  only dom-parsed violations available (%d), likely partial",
                            len(dom_violations),
                        )
                        return dom_violations

                    _save_debug_artifacts(page, config.debug_dir, attempt, f"empty_{strategy}")
                    logging.warning("  no violations parsed with strategy=%s", strategy)
                except Exception as exc:
                    last_error = exc
                    _save_debug_artifacts(page, config.debug_dir, attempt, f"fail_{strategy}")
                    logging.warning("  strategy=%s failed: %s", strategy, exc)
            raise RuntimeError(f"all wait strategies failed or empty, last_error={last_error}")
        finally:
            browser.close()


def extract_with_retry(
    url: str,
    max_retries: int,
    config: FetcherConfig,
    jitter_ms: int,
) -> list[Violation]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Attempt %d/%d...", attempt, max_retries)
            return extract_violations_with_playwright(url=url, config=config, attempt=attempt)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_s = (2 ** (attempt - 1)) + random.uniform(0, max(0, jitter_ms) / 1000.0)
                logging.warning("  Failed: %s", exc)
                logging.info("  Waiting %.2fs before retry...", wait_s)
                time.sleep(wait_s)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def group_by_rule(violations: list[Violation]) -> dict[str, list[Violation]]:
    by_rule: dict[str, list[Violation]] = {}
    for violation in violations:
        by_rule.setdefault(violation.rule_id, []).append(violation)
    return by_rule


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="从 openlibing.com 提取 CodeCheck 违规列表")
    _ = parser.add_argument("url", help="CodeCheck 报告 URL")
    _ = parser.add_argument(
        "--output",
        "-o",
        choices=["json", "text", "markdown"],
        default="json",
        help="输出格式",
    )
    _ = parser.add_argument("--group", "-g", action="store_true", help="按规则分组输出")
    _ = parser.add_argument("--retries", type=int, default=3, help="提取重试次数（默认 3）")
    _ = parser.add_argument(
        "--wait-strategies",
        default="domcontentloaded,load,networkidle",
        help="导航等待策略，逗号分隔（domcontentloaded,load,networkidle,commit）",
    )
    _ = parser.add_argument("--nav-timeout-ms", type=int, default=90000, help="导航超时毫秒")
    _ = parser.add_argument("--selector-timeout-ms", type=int, default=15000, help="关键选择器等待超时毫秒")
    _ = parser.add_argument("--post-wait-ms", type=int, default=5000, help="页面加载后额外等待毫秒")
    _ = parser.add_argument("--jitter-ms", type=int, default=500, help="重试抖动毫秒上限")
    _ = parser.add_argument("--debug-dir", default="", help="失败时保存截图/HTML的目录")
    _ = parser.add_argument("--api-page-size", type=int, default=100, help="task api 每页拉取条数")
    _ = parser.add_argument("--max-pages", type=int, default=200, help="task api 最大翻页数，防止死循环")
    return parser.parse_args(namespace=Args())


def main() -> int:
    args = parse_args()

    try:
        retries = max(1, int(args.retries))
        wait_strategies = _parse_wait_strategies(args.wait_strategies)
        config = FetcherConfig(
            wait_strategies=wait_strategies,
            nav_timeout_ms=max(1000, int(args.nav_timeout_ms)),
            selector_timeout_ms=max(0, int(args.selector_timeout_ms)),
            post_wait_ms=max(0, int(args.post_wait_ms)),
            debug_dir=args.debug_dir.strip(),
            api_page_size=max(1, int(args.api_page_size)),
            max_pages=max(1, int(args.max_pages)),
        )
        violations = extract_with_retry(
            url=args.url,
            max_retries=retries,
            config=config,
            jitter_ms=max(0, int(args.jitter_ms)),
        )

        if args.output == "json":
            grouped = group_by_rule(violations)
            result: dict[str, object] = {
                "total": len(violations),
                "by_rule": {rule: len(items) for rule, items in grouped.items()},
                "violations": [v.to_dict() for v in violations],
            }
            if args.group:
                result["grouped"] = {rule: [item.to_dict() for item in items] for rule, items in grouped.items()}
            logging.info(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logging.info("Total violations: %d\n", len(violations))
            if args.group:
                grouped = group_by_rule(violations)
                for rule, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    desc = items[0].rule_description if items else ""
                    logging.info("## %s: %d violations", rule, len(items))
                    logging.info("   %s\n", desc)
                    for item in items:
                        logging.info("   - %s:%d", item.file, item.line)
                        logging.info("     %s\n", item.description)
            else:
                for item in violations:
                    logging.info("%s | %s:%d", item.rule_id, item.file, item.line)
                    logging.info("  %s\n", item.description)

        return 0
    except Exception as exc:
        logging.error("Error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
