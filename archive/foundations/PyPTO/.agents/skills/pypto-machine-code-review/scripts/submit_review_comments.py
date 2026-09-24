#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""批量提交 GitCode PR 行级 diff_comment（检视意见显示在代码行旁）。

用法:
  # 方式 1: 命令行逐条指定
  TOKEN=$(git remote get-url origin | sed -nE 's|.*oauth2:([^@]+)@.*|\\1|p')
  python3 submit_review_comments.py --pr 4974 --token "$TOKEN" \\
    --comment "framework/src/a.h:31:**Blocker**: CRC32 多项式错误" \\
    --comment "framework/src/b.cpp:37:**Major**: 碰撞风险"

  # 方式 2: 从 JSON 文件读取
  python3 submit_review_comments.py --pr 4974 --token "$TOKEN" \\
    --file comments.json

comments.json 格式:
  [
    {"path": "framework/src/a.h", "position": 31, "body": "...", "level": "Blocker"},
    {"path": "framework/src/b.cpp", "position": 37, "body": "...", "level": "Major"}
  ]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import subprocess
import sys
from typing import Optional
import urllib.error
import urllib.request

API_BASE = "https://gitcode.com/api/v5"
OWNER = "cann"
REPO = "pypto"


@dataclass
class ReviewComment:
    path: str
    position: int
    body: str
    level: str = ""

    def to_payload(self) -> dict:
        return {
            "body": self.body,
            "path": self.path,
            "position": self.position,
            "need_to_resolve": True,
        }


def extract_token_from_git() -> Optional[str]:
    """从 git remote URL 提取 oauth2 token。"""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "oauth2:" not in url:
        return None
    return url.split("oauth2:", 1)[1].split("@", 1)[0]


def api_request(token: str, method: str, path: str, payload: Optional[dict] = None) -> tuple[int, dict]:
    """执行 GitCode API 请求，返回 (http_code, response_dict)。"""
    url = f"{API_BASE}/repos/{OWNER}/{REPO}/{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("PRIVATE-TOKEN", token)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body}
    except urllib.error.URLError as e:
        return -1, {"error": str(e)}


def verify_token(token: str) -> bool:
    """验证 token 是否可用。"""
    url = f"{API_BASE}/user?access_token={token}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def submit_comment(token: str, pr_number: int, comment: ReviewComment) -> dict:
    """提交一条行级 diff_comment，返回结果 dict。"""
    code, resp = api_request(
        token, "POST",
        f"pulls/{pr_number}/comments",
        comment.to_payload(),
    )
    result = {
        "path": comment.path,
        "position": comment.position,
        "level": comment.level,
        "body_preview": comment.body[:60],
        "http_code": code,
        "comment_id": resp.get("id", resp.get("note_id")),
        "comment_type": resp.get("comment_type", "?"),
        "status": "ok" if code in (200, 201) else f"fail({code})",
    }
    if code not in (200, 201):
        result["error"] = resp.get("error_message") or resp.get("error") or str(resp)[:200]
    return result


def verify_diff_comment(token: str, pr_number: int, comments: list[ReviewComment]) -> list[dict]:
    """验证评论是否真正成为 diff_comment。

    POST 响应只返回 {id(hash), body, note_id}，不含 comment_type/diff_position。
    GET 列表的 id 是数字格式，与 POST 返回的 hash id 不匹配。
    GET 列表也不含 path 字段，因此用 position + body 前缀匹配。
    默认 per_page=20 不够，需拉取足够页。
    """
    results = []
    for page in range(1, 10):
        code, resp = api_request(token, "GET", f"pulls/{pr_number}/comments?per_page=100&page={page}", None)
        if code != 200 or not isinstance(resp, list) or len(resp) == 0:
            break
        target_by_pos = {}
        for c in comments:
            target_by_pos.setdefault(c.position, []).append(c.body[:30])
        for c in resp:
            dp = c.get("diff_position") or {}
            line = dp.get("start_new_line")
            if line in target_by_pos:
                body_prefix = (c.get("body") or "")[:30]
                if any(body_prefix.startswith(prefix) or prefix in body_prefix for prefix in target_by_pos[line]):
                    results.append({
                        "id": c.get("id"),
                        "comment_type": c.get("comment_type"),
                        "diff_position": dp,
                        "body_preview": (c.get("body") or "")[:60],
                        "is_diff_comment": c.get("comment_type") == "diff_comment" and dp is not None,
                    })
        if len(results) >= len(comments):
            break
    return results


def parse_cli_comment(spec: str) -> ReviewComment:
    """解析 'path:position:body' 格式的命令行参数。"""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid --comment format, expected 'path:position:body', got: {spec}")
    path, pos_str, body = parts
    position = int(pos_str)
    level = ""
    for tag in ("Blocker", "Major", "Minor"):
        if body.startswith(tag):
            level = tag
            break
    return ReviewComment(path=path, position=position, body=body, level=level)


def load_comments_from_file(filepath: str) -> list[ReviewComment]:
    """从 JSON 文件加载评论列表。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    comments = []
    for item in data:
        comments.append(ReviewComment(
            path=item["path"],
            position=int(item["position"]),
            body=item["body"],
            level=item.get("level", ""),
        ))
    return comments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量提交 GitCode PR 行级 diff_comment（检视意见显示在代码行旁）"
    )
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument("--token", type=str, default=None, help="GitCode API token (默认从 git remote 提取)")
    parser.add_argument("--comment", action="append", default=[], help="单条评论: 'path:position:body'")
    parser.add_argument("--file", type=str, default=None, help="从 JSON 文件读取评论列表")
    parser.add_argument("--no-verify", action="store_true", help="跳过 diff_comment 验证")
    args = parser.parse_args()

    token = args.token or extract_token_from_git()
    if not token:
        print("ERROR: 无法获取 token，请用 --token 指定或确保 git remote origin 含 oauth2 token", file=sys.stderr)
        return 1

    if not verify_token(token):
        print("ERROR: token 验证失败 (401)", file=sys.stderr)
        return 1

    comments: list[ReviewComment] = []
    for spec in args.comment:
        try:
            comments.append(parse_cli_comment(spec))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    if args.file:
        comments.extend(load_comments_from_file(args.file))

    if not comments:
        print("ERROR: 未提供评论 (--comment 或 --file)", file=sys.stderr)
        return 1

    print(f"=== 提交 {len(comments)} 条行级 diff_comment 到 PR {args.pr} ===\n")

    success = 0
    for i, c in enumerate(comments, 1):
        result = submit_comment(args.token or token, args.pr, c)
        if result["status"] == "ok":
            success += 1
        print(f"  [{i}/{len(comments)}] {c.path}:{c.position} ({c.level or 'N/A'})")
        print(f"         {result['status']}  id={result['comment_id']}  type={result['comment_type']}")

    print(f"\n提交完成: {success}/{len(comments)} 成功\n")

    if not args.no_verify:
        print("=== 验证 diff_comment ===\n")
        verifications = verify_diff_comment(token, args.pr, comments)
        if not verifications:
            print("  ⚠️ 未找到已提交的评论（可能需要等待或翻页）")
        diff_count = 0
        for v in verifications:
            is_diff = v["is_diff_comment"]
            if is_diff:
                diff_count += 1
            dp = v.get("diff_position") or {}
            line = dp.get("start_new_line", "?")
            mark = "✅" if is_diff else "❌ pr_comment"
            print(f"  id={v['id']}  type={v['comment_type']}  line={line}  {mark}")
        print(f"\n验证完成: {diff_count}/{len(verifications)} 为 diff_comment（行级评论）")

    pr_url = f"https://gitcode.com/{OWNER}/{REPO}/merge_requests/{args.pr}/files"
    print(f"\nPR diff 视图: {pr_url}")
    return 0 if success == len(comments) else 1


if __name__ == "__main__":
    sys.exit(main())
