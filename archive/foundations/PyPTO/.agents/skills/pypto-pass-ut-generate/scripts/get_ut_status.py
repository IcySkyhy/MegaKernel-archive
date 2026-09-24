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

"""
获取 PR 的 UT 测试状态（简化版）

用法：
    python3 get_ut_status.py <PR编号>
"""

import logging
import re
import sys

from common_utils import get_gitcode_token, make_api_request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_ut_status_from_comments(comments: list) -> dict:
    """
    从评论列表中解析 UT 状态

    Args:
        comments: PR 评论列表

    Returns:
        UT 状态信息，包含 status, coverage_url, comment_time
    """
    if not comments:
        return None

    # 按时间排序，获取最新的评论
    sorted_comments = sorted(comments, key=lambda x: x.get('created_at', ''), reverse=True)

    for comment in sorted_comments:
        body = comment.get('body', '')
        if 'UT_Test_report' in body:
            # 提取状态
            match = re.search(r'<td><strong>UT_Test_report</strong></td>\s*<td>([^<]+)</td>', body)
            status = match.group(1).strip() if match else None

            # 提取覆盖率链接 - 优先获取 ut_cov.tar.gz
            cov_match = re.search(
                r'https://ascend-ci\.obs\.cn-north-4\.myhuaweicloud\.com/[^\"\'<>\s]*ut_cov\.tar\.gz', body
            )
            coverage_url = cov_match.group(0).split('>')[0] if cov_match else None

            # 如果没有 ut_cov.tar.gz，尝试获取其他覆盖率相关文件
            if not coverage_url:
                for ext in ['whl', 'tar.gz', 'zip']:
                    cov_match = re.search(
                        rf'https://ascend-ci\.obs\.cn-north-4\.myhuaweicloud\.com/[^\"\'<>\s]*\.{ext}', body
                    )
                    if cov_match:
                        coverage_url = cov_match.group(0).split('>')[0]
                        break

            return {'status': status, 'coverage_url': coverage_url, 'comment_time': comment.get('created_at', '')}

    return None


def get_pr_ut_status(owner: str, repo: str, pr_number: int) -> dict:
    """
    获取 PR 的 UT 状态

    Args:
        owner: 仓库所有者
        repo: 仓库名
        pr_number: PR 编号

    Returns:
        UT 状态信息
    """
    token = get_gitcode_token()
    if not token:
        return {'error': '未找到 GitCode Token'}

    url = f"https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}/comments?per_page=100"
    comments = make_api_request(url, token=token)

    if not comments:
        return {'error': '获取评论失败'}

    if isinstance(comments, dict) and 'message' in comments:
        return {'error': comments.get('message', 'API 请求失败')}

    return parse_ut_status_from_comments(comments)


def main(pr_number: int):
    """主函数"""
    logger.info("获取 PR #%d 的 UT 状态...", pr_number)

    result = get_pr_ut_status('cann', 'pypto', pr_number)

    if 'error' in result:
        logger.info("Error: %s", result['error'])
        return

    if result:
        logger.info("\nUT_Test_report 状态: %s", result.get('status', 'N/A'))
        logger.info("覆盖率报告: %s", result.get('coverage_url', 'N/A'))
    else:
        logger.info("未找到 UT_Test_report 信息")


if __name__ == "__main__":
    pr_number = int(sys.argv[1]) if len(sys.argv) > 1 else 2093
    main(pr_number)
