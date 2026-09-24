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
Pass UT 生成工具 - PR 处理模块

功能：
1. 通过 PR 编号获取代码变更内容
2. 解析 PR 评论中的 UT-REPORT，获取覆盖率和未覆盖行
3. 根据未覆盖行分析业务，设计 UT
4. 支持离线分析（用户提供的 diff 文件或 UT-Report 文件）

支持多种输入方式（PR编号、diff文件、diff内容、离线文件）
"""

from dataclasses import dataclass
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Dict, List, Optional, Tuple
import urllib.error
import urllib.request

try:
    from common_utils import safe_extractall
except ImportError:
    from scripts.common_utils import safe_extractall

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


@dataclass
class PRProcessConfig:
    pr_input: str
    output_dir: Optional[str] = None
    repo_dir: Optional[str] = None
    use_api_for_comments: bool = True
    auto_stash: bool = True
    check_build: bool = True


@dataclass
class FetchBranchConfig:
    owner: str
    repo: str
    pr_number: int
    repo_dir: str
    author: str
    author_repo: str
    head_ref: str


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

GITCODE_API_BASE = "https://api.gitcode.com/api/v5"
GITCODE_TOKEN = os.environ.get("GITCODE_TOKEN", "")


def make_gitcode_api_request(endpoint: str) -> Optional[Dict]:
    """通过 GitCode API 获取 JSON 数据"""
    url = f"{GITCODE_API_BASE}/{endpoint}"
    headers = {"Accept": "application/json"}
    if GITCODE_TOKEN:
        headers["PRIVATE-TOKEN"] = GITCODE_TOKEN

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        logger.warning("HTTP 错误: %d - %s", e.code, e.reason)
        return None
    except Exception as e:
        logger.warning("API 请求失败: %s", e)
        return None


def get_pr_info_via_api(owner: str, repo: str, pr_number: int) -> Optional[Dict]:
    """通过 GitCode API 获取 PR 信息"""
    logger.info("通过 GitCode API 获取 PR #%d 信息...", pr_number)
    return make_gitcode_api_request(f"repos/{owner}/{repo}/pulls/{pr_number}")


def get_pr_comments_via_api(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """通过 GitCode API 获取 PR 评论"""
    result = make_gitcode_api_request(f"repos/{owner}/{repo}/pulls/{pr_number}/comments")
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return []


def get_pr_diff_via_api(owner: str, repo: str, pr_number: int) -> Optional[str]:
    """通过 GitCode API 获取 PR diff"""
    try:
        logger.info("通过 GitCode API 获取 PR #%d 的 diff...", pr_number)
        url = f"{GITCODE_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/diff"
        headers = {"Accept": "text/plain"}
        if GITCODE_TOKEN:
            headers["PRIVATE-TOKEN"] = GITCODE_TOKEN

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            diff_content = response.read().decode('utf-8')

        if not diff_content.strip():
            logger.info("⚠️ API 返回的 diff 内容为空")
            return None

        logger.info("API 成功获取 diff，大小: {len(diff_content)} 字节")
        return diff_content

    except urllib.error.HTTPError as e:
        logger.warning(f"API 获取 diff 失败: {e.code} - {e.reason}")
        return None
    except Exception as e:
        logger.warning(f"API 获取 diff 异常: {e}")
        return None


def get_pr_diff_via_git_fetch(
    owner: str, repo: str, pr_number: int, repo_dir: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """通过 git fetch 获取 PR diff，返回 (diff_content, local_branch)"""
    work_dir = repo_dir if repo_dir else PROJECT_ROOT
    local_branch = f"pr{pr_number}_branch"

    try:
        logger.info(f"通过 git fetch 获取 PR #{pr_number} 的 diff...")

        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd=work_dir)

        remotes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    remotes.append(parts[0])

        for remote_name in remotes + ['cann', 'origin', 'bytecode']:
            result = subprocess.run(
                ['git', 'fetch', remote_name, f'pull/{pr_number}/head:{local_branch}'],
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info("成功从 {remote_name} 获取 PR 分支")

                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%H', local_branch], capture_output=True, text=True, cwd=work_dir
                )

                if result.returncode == 0:
                    commit = result.stdout.strip()
                    base_commit = f"{commit}~1"

                    result = subprocess.run(
                        ['git', 'diff', f'{base_commit}..{commit}'], capture_output=True, text=True, cwd=work_dir
                    )

                    if result.returncode == 0:
                        diff_content = result.stdout
                        if diff_content.strip():
                            logger.info("git fetch 成功获取 diff，大小: {len(diff_content)} 字节")
                            return diff_content, local_branch
                break
            else:
                logger.warning(f"从 {remote_name} fetch 失败: {result.stderr.strip()[:100]}")

        return None, None

    except Exception as e:
        logger.warning(f"git fetch 获取 diff 失败: {e}")
        return None, None


def fetch_pr_branch_from_author(config: FetchBranchConfig) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """从 PR 作者的仓库获取分支"""
    work_dir = config.repo_dir
    local_branch = f"pr{config.pr_number}_branch"

    try:
        logger.info("\n尝试从 PR 作者 (%s) 仓库获取分支...", config.author)

        author_remote_name = f"pr_author_{config.author}"

        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd=work_dir)

        remote_url = None
        for line in result.stdout.strip().split('\n'):
            if line and author_remote_name in line:
                parts = line.split()
                if len(parts) >= 2:
                    remote_url = parts[1]
                    break

        if not remote_url:
            author_git_url = f"https://gitcode.com/{config.author}/{config.author_repo}.git"
            logger.info("添加作者 remote: %s", author_git_url)
            result = subprocess.run(
                ['git', 'remote', 'add', author_remote_name, author_git_url],
                capture_output=True,
                text=True,
                cwd=work_dir,
            )

            if result.returncode != 0 and 'already exists' not in result.stderr:
                logger.warning("添加作者 remote 失败: %s", result.stderr.strip()[:100])
                return None, None, None

            remote_url = author_git_url

        logger.info("从 %s fetch 分支 %s...", author_remote_name, config.head_ref)
        result = subprocess.run(
            ['git', 'fetch', author_remote_name, f'heads/{config.head_ref}:{local_branch}'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning("fetch 分支失败: %s", result.stderr.strip()[:100])

            logger.info("尝试通过 merge request refs 获取...")
            result = subprocess.run(
                ['git', 'fetch', remote_url, f'+refs/merge-requests/{config.pr_number}/head:{local_branch}'],
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning("merge request refs fetch 也失败: %s", result.stderr.strip()[:100])
                return None, None, None

        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H', local_branch], capture_output=True, text=True, cwd=work_dir
        )

        if result.returncode != 0:
            logger.warning("无法获取本地分支 %s", local_branch)
            return None, None, None

        commit = result.stdout.strip()
        base_commit = f"{commit}~1"

        result = subprocess.run(
            ['git', 'diff', f'{base_commit}..{commit}'], capture_output=True, text=True, cwd=work_dir
        )

        if result.returncode == 0:
            diff_content = result.stdout
            logger.info("成功获取 PR diff，大小: %d 字节", len(diff_content))
            return diff_content, local_branch, local_branch

        return None, None, None

    except Exception as e:
        logger.warning("从作者仓库获取分支失败: %s", e)
        return None, None, None


def checkout_or_fetch_pr_branch(
    owner: str, repo: str, pr_number: int, repo_dir: str, pr_info: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    获取 PR 分支，返回 (分支名, head分支名, diff_content)
    """
    local_branch = f"pr{pr_number}_branch"

    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--verify', local_branch], capture_output=True, text=True, cwd=repo_dir
        )

        if result.returncode == 0:
            logger.info(f"本地分支 {local_branch} 已存在")

            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H', local_branch], capture_output=True, text=True, cwd=repo_dir
            )

            if result.returncode == 0:
                commit = result.stdout.strip()
                base_commit = f"{commit}~1"

                result = subprocess.run(
                    ['git', 'diff', f'{base_commit}..{commit}'], capture_output=True, text=True, cwd=repo_dir
                )

                if result.returncode == 0 and result.stdout.strip():
                    return local_branch, local_branch, result.stdout
                else:
                    logger.info("⚠️ 本地分支存在但无 diff，删除后重新获取")
                    subprocess.run(['git', 'branch', '-D', local_branch], capture_output=True, cwd=repo_dir)

        remotes_with_urls = {}
        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd=repo_dir)
        for line in result.stdout.strip().split('\n'):
            if line and 'fetch' in line:
                parts = line.split()
                if len(parts) >= 2:
                    remotes_with_urls[parts[0]] = parts[1]

        for remote_name, remote_url in remotes_with_urls.items():
            logger.info(f"尝试从 {remote_name} 获取 PR #{pr_number}...")
            result = subprocess.run(
                ['git', 'fetch', remote_url, f'+refs/merge-requests/{pr_number}/head:{local_branch}'],
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info(f"成功从 {remote_name} 获取 PR 分支 {local_branch}")

                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%H', local_branch], capture_output=True, text=True, cwd=repo_dir
                )

                if result.returncode == 0:
                    commit = result.stdout.strip()
                    base_commit = f"{commit}~1"

                    result = subprocess.run(
                        ['git', 'diff', f'{base_commit}..{commit}'], capture_output=True, text=True, cwd=repo_dir
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        return local_branch, local_branch, result.stdout
                break
            else:
                logger.warning(f"从 {remote_name} fetch 失败: {result.stderr.strip()[:80]}")

        if pr_info:
            head_ref = pr_info.get('head', {}).get('ref', '')
            author = pr_info.get('user', {}).get('login', '')
            author_repo = pr_info.get('head', {}).get('repo', {}).get('name', repo)

            if head_ref and author:
                config = FetchBranchConfig(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    repo_dir=repo_dir,
                    author=author,
                    author_repo=author_repo,
                    head_ref=head_ref,
                )
                diff_content, branch, _ = fetch_pr_branch_from_author(config)
                if diff_content:
                    return branch, branch, diff_content

        return None, None, None

    except Exception as e:
        logger.warning(f"分支操作失败: {e}")
        return None, None, None


def parse_ut_report_from_comments(comments: List[Dict]) -> Dict:
    """
    从评论列表中解析 UT-REPORT 状态

    返回格式:
    {
        'found': bool,
        'status': str,  # SUCCESS, PARTIAL_FAILED, ABORT, FAILED, NOT_FOUND
        'ut_tests': Dict[str, str],  # task_name -> status
        'coverage_url': str,
        'failed_tests': List[str],
        'message': str
    }
    """
    result = {
        'found': False,
        'status': 'NOT_FOUND',
        'ut_tests': {},
        'coverage_url': '',
        'failed_tests': [],
        'message': '',
    }

    latest_pipeline_comment = None
    latest_time = None

    for comment in comments:
        body = comment.get('body', '')
        created_at = comment.get('created_at', '')

        if '流水线任务触发成功' in body or 'UT-REPORT' in body:
            if latest_time is None or created_at > latest_time:
                latest_time = created_at
                latest_pipeline_comment = comment

    if not latest_pipeline_comment:
        result['message'] = '未找到流水线任务评论'
        return result

    result['found'] = True
    body = latest_pipeline_comment.get('body', '')

    matches = re.findall(r'<td><strong>([^<]+)</strong></td>\s*<td>([^<]+)</td>', body)
    for task_name, status in matches:
        result['ut_tests'][task_name.strip()] = status.strip()

    cov_match = re.search(r'https://ascend-ci\.obs\.cn-north-4\.myhuaweicloud\.com/[^"\']*ut_cov\.tar\.gz', body)
    if cov_match:
        result['coverage_url'] = cov_match.group(0)

    failed_tests = [k for k, v in result['ut_tests'].items() if 'FAILED' in v.upper() or '❌' in v]
    aborted_tests = [k for k, v in result['ut_tests'].items() if 'ABORTED' in v.upper() or '⚪' in v]
    success_tests = [k for k, v in result['ut_tests'].items() if 'SUCCESS' in v.upper() or '✅' in v]

    if failed_tests:
        result['status'] = 'PARTIAL_FAILED'
        result['failed_tests'] = failed_tests
        result['message'] = f'UT 测试部分失败: {", ".join(failed_tests)}'
    elif aborted_tests and success_tests:
        result['status'] = 'ABORT'
        result['message'] = f'UT 测试部分通过({len(success_tests)}个)，部分中止({len(aborted_tests)}个)'
    elif all('SUCCESS' in v.upper() or '✅' in v for v in result['ut_tests'].values() if v.strip()):
        result['status'] = 'SUCCESS'
        result['message'] = 'UT 测试全部通过'
    else:
        result['status'] = 'NOT_FOUND'
        result['message'] = '无法确定 UT 状态'

    return result


def download_and_parse_coverage_report(coverage_url: str, pr_number: int) -> Dict:
    """
    下载并解析覆盖率报告

    返回格式:
    {
        'success': bool,
        'uncovered_files': List[str],
        'low_coverage_files': List[Dict],
        'message': str
    }
    """
    result = {'success': False, 'uncovered_files': [], 'low_coverage_files': [], 'message': ''}

    if not coverage_url:
        result['message'] = '没有覆盖率报告链接'
        return result

    extract_dir = None
    try:
        logger.info(f"\n下载覆盖率报告: {coverage_url[:80]}...")

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        req = urllib.request.Request(coverage_url, headers={'Accept': 'application/octet-stream'})
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(tmp_path, 'wb') as f:
                f.write(response.read())

        logger.info(f"覆盖率报告下载成功: {tmp_path}")

        extract_dir = None
        extract_dir = tempfile.mkdtemp(prefix=f'cov_pr{pr_number}_')

        with tarfile.open(tmp_path, 'r:gz') as tar:
            safe_extractall(tar, extract_dir)

        logger.info(f"覆盖率报告解压到: {extract_dir}")

        result['success'] = True
        result['extract_dir'] = extract_dir
        result['message'] = f'覆盖率报告已解压到: {extract_dir}'

        os.unlink(tmp_path)

    except Exception as e:
        result['message'] = f'下载/解析覆盖率报告失败: {e}'
        logger.warning(f"{result['message']}")
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)

    return result


def parse_pr_info(pr_input: str) -> Tuple[str, str, int]:
    """解析 PR 链接，返回 (owner, repo, pr_number)"""
    if pr_input.isdigit():
        return "cann", "pypto", int(pr_input)

    url_match = re.search(r'gitcode\.com/([^/]+)/([^/]+)/(?:pull|-\/merge_requests)/(\d+)', pr_input)
    if url_match:
        return url_match.group(1), url_match.group(2), int(url_match.group(3))

    short_match = re.search(r'([^/]+)/([^/]+)/(\d+)', pr_input)
    if short_match:
        return short_match.group(1), short_match.group(2), int(short_match.group(3))

    hash_match = re.search(r'#(\d+)', pr_input)
    if hash_match:
        return "cann", "pypto", int(hash_match.group(1))

    raise ValueError(f"无法解析 PR 链接：{pr_input}")


def save_diff_to_file(diff_content: str, output_file: str) -> bool:
    """保存 diff 内容到文件"""
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(diff_content)
        logger.info(f"Diff 已保存到: {output_file}")
        return True
    except Exception as e:
        logger.warning(f"保存 diff 文件失败: {e}")
        return False


def apply_diff_to_repo(diff_file: str, repo_dir: str, auto_stash: bool = True) -> Tuple[bool, str]:
    """
    将 diff 文件应用到本地仓库

    Args:
        diff_file: diff 文件路径
        repo_dir: 仓库目录
        auto_stash: 是否自动 stash 未提交的更改（默认 True）

    Returns:
        (是否成功, 消息)
    """
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=repo_dir)

        uncommitted = [line for line in result.stdout.strip().split('\n') if line]
        _stash_applied = False
        stash_message = ""

        if uncommitted:
            logger.info("=" * 60)
            logger.info("工作区有未提交的更改：")
            for line in uncommitted[:10]:
                logger.info(f"  {line}")
            if len(uncommitted) > 10:
                logger.info(f"  ... 还有 {len(uncommitted) - 10} 个文件")

            if auto_stash:
                logger.info("\n自动执行 git stash...")
                result = subprocess.run(
                    ['git', 'stash', 'push', '-m', 'auto-stash-before-apply-diff'],
                    capture_output=True,
                    text=True,
                    cwd=repo_dir,
                )
                if result.returncode == 0:
                    logger.info("✓ stash 成功")
                    _stash_applied = True
                    stash_message = "（已自动 stash 当前更改）"
                else:
                    logger.warning(f"stash 失败: {result.stderr.strip()}")
                    logger.info("将尝试直接 apply...")
            else:
                return False, f"工作区有未提交的更改: {len(uncommitted)} 个文件"

        result = subprocess.run(
            ['git', 'apply', '--3way', '--check', diff_file], capture_output=True, text=True, cwd=repo_dir
        )

        if result.returncode == 0:
            result = subprocess.run(['git', 'apply', '--3way', diff_file], capture_output=True, text=True, cwd=repo_dir)

            if result.returncode == 0:
                return True, f"Diff 已成功应用到本地仓库 {stash_message}"
            else:
                logger.info("=" * 60)
                logger.info("⚠️ Diff 应用发生冲突")
                logger.info("冲突文件:")

                conflict_files = []
                result = subprocess.run(
                    ['git', 'diff', '--name-only', '--diff-filter=U'], capture_output=True, text=True, cwd=repo_dir
                )
                for f in result.stdout.strip().split('\n'):
                    if f:
                        conflict_files.append(f)
                        logger.info(f"  - {f}")

                choice = input(
                    "\n请选择操作：\n"
                    "  1 - 保留本地版本（丢弃 diff 变更）\n"
                    "  2 - 接受 incoming 版本（应用 diff 变更）\n"
                    "  3 - 手动解决冲突\n"
                    "请输入选项 [1/2/3]: "
                ).strip()

                if choice == '1':
                    result = subprocess.run(
                        ['git', 'checkout', '--ours'] + conflict_files, capture_output=True, text=True, cwd=repo_dir
                    )
                    return True, "已保留本地版本，冲突已解决"
                elif choice == '2':
                    result = subprocess.run(
                        ['git', 'checkout', '--theirs'] + conflict_files, capture_output=True, text=True, cwd=repo_dir
                    )
                    return True, "已接受 incoming 版本，冲突已解决"
                else:
                    logger.info("请手动解决冲突后，运行以下命令继续：")
                    logger.info("  git add <resolved-files>")
                    return False, "请手动解决冲突"
        else:
            conflict_info = result.stderr.strip()
            return False, f"Diff 应用会冲突: {conflict_info[:200]}"

    except Exception as e:
        return False, f"应用 diff 异常: {e}"


def check_build_status(repo_dir: str, pass_files: List[str]) -> Tuple[bool, str]:
    """
    检查代码编译状态

    Args:
        repo_dir: 仓库目录
        pass_files: 需要检查的 Pass 文件列表

    Returns:
        (是否成功, 消息)
    """
    if not pass_files:
        return True, "无需检查编译（无 Pass 文件变更）"

    logger.info("=" * 60)
    logger.info("检查编译状态...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ['python3', 'build_ci.py', '-c', '-j', '24', '-f', 'cpp'],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=600,
        )

        if result.returncode == 0:
            return True, "✓ 编译通过"
        else:
            error_output = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
            return False, f"⚠️ 编译失败:\n{error_output[-500:]}"

    except subprocess.TimeoutExpired:
        return False, "⚠️ 编译超时（超过10分钟）"
    except Exception as e:
        return False, f"⚠️ 编译检查异常: {e}"


def analyze_changed_files(diff_content: str) -> Dict:
    """分析 diff 内容，返回修改的文件和类型"""
    result = {'total_files': 0, 'pass_files': [], 'test_files': [], 'other_files': [], 'files_detail': []}

    if not diff_content:
        return result

    current_file = None
    additions = 0
    deletions = 0

    for line in diff_content.split('\n'):
        if line.startswith('diff --git'):
            if current_file:
                result['files_detail'].append({'file': current_file, 'additions': additions, 'deletions': deletions})
                if 'passes' in current_file and current_file.endswith('.cpp'):
                    result['pass_files'].append(current_file)
                elif 'test' in current_file and current_file.endswith('.cpp'):
                    result['test_files'].append(current_file)
                else:
                    result['other_files'].append(current_file)

            match = re.search(r'a/(\S+)', line)
            if match:
                current_file = match.group(1)
                additions = 0
                deletions = 0

        elif line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    if current_file:
        result['files_detail'].append({'file': current_file, 'additions': additions, 'deletions': deletions})
        result['total_files'] = len(result['files_detail'])
        if 'passes' in current_file and current_file.endswith('.cpp'):
            result['pass_files'].append(current_file)
        elif 'test' in current_file and current_file.endswith('.cpp'):
            result['test_files'].append(current_file)
        else:
            result['other_files'].append(current_file)

    return result


def process_pr(config: PRProcessConfig) -> Dict:
    """
    处理 PR 的主函数

    参数:
        config: PRProcessConfig 配置对象

    返回:
        包含处理结果的字典
    """
    work_output_dir = config.output_dir if config.output_dir else PROJECT_ROOT
    work_repo_dir = config.repo_dir if config.repo_dir else PROJECT_ROOT

    result = {
        'pr_info': None,
        'diff_content': None,
        'diff_file': None,
        'diff_applied': False,
        'diff_apply_error': None,
        'analysis': {},
        'ut_report': None,
        'build_status': None,
        'coverage': None,
        'success': False,
        'error': None,
        'method': None,
        'need_design_ut': False,
        'design_ut_reason': "",
    }

    try:
        logger.info("=" * 60)
        logger.info("PR 处理工具 V8")
        logger.info("=" * 60)

        owner, repo, pr_number = parse_pr_info(config.pr_input)
        logger.info(f"PR编号: {pr_number}")
        logger.info(f"仓库: {owner}/{repo}")
        result['pr_info'] = {'owner': owner, 'repo': repo, 'pr_number': pr_number}

        pr_info = get_pr_info_via_api(owner, repo, pr_number)
        if pr_info:
            logger.info(f"PR 标题: {pr_info.get('title', 'N/A')}")
            logger.info(f"PR 状态: {pr_info.get('state', 'N/A')}")
            base_ref = pr_info.get('base', {}).get('ref', 'master')
            head_ref = pr_info.get('head', {}).get('ref', '')
            logger.info(f"基分支: {base_ref}")
            logger.info(f"头分支: {head_ref}")
            result['pr_info']['base_branch'] = base_ref
            result['pr_info']['head_branch'] = head_ref
            result['pr_info']['author'] = pr_info.get('user', {}).get('login', '')
            result['pr_info']['author_repo'] = pr_info.get('head', {}).get('repo', {}).get('name', repo)
        else:
            logger.info("⚠️ 无法通过 API 获取 PR 信息，将尝试其他方式")

        logger.info("-" * 60)
        logger.info("Step 1: 获取 UT-REPORT 状态")
        logger.info("-" * 60)

        ut_result = {
            'found': False,
            'status': 'NOT_FOUND',
            'ut_tests': {},
            'coverage_url': '',
            'failed_tests': [],
            'message': '无法获取 UT-REPORT 评论',
        }

        if config.use_api_for_comments:
            try:
                comments = get_pr_comments_via_api(owner, repo, pr_number)
                if comments:
                    ut_result = parse_ut_report_from_comments(comments)
                    logger.info(f"通过 API 获取到 {len(comments)} 条评论")
                else:
                    logger.info("⚠️ API 返回空评论列表")
            except Exception as e:
                logger.warning(f"API 调用失败: {e}")

        result['ut_report'] = ut_result

        logger.info(f"\nUT 状态: {ut_result['status']}")
        logger.info(f"消息: {ut_result['message']}")

        if ut_result['ut_tests']:
            logger.info("\nUT 测试任务状态:")
            for task, status in sorted(ut_result['ut_tests'].items()):
                if "SUCCESS" in status.upper() or "✅" in status:
                    icon = "✅"
                elif "FAILED" in status.upper() or "❌" in status:
                    icon = "❌"
                else:
                    icon = "⚪"
                logger.info(f"  {icon} {task}: {status}")

        if ut_result['coverage_url']:
            logger.info(f"\n覆盖率报告: {ut_result['coverage_url'][:80]}...")

        logger.info("-" * 60)
        logger.info("Step 2: 获取 PR 代码变更")
        logger.info("-" * 60)

        diff_content = None
        pr_branch = None

        pr_branch, head_branch, diff_content = checkout_or_fetch_pr_branch(
            owner, repo, pr_number, work_repo_dir, pr_info
        )

        if pr_branch and diff_content:
            logger.info(f"找到 PR 分支: {pr_branch}")
            result['method'] = 'git_fetch'
            result['diff_content'] = diff_content

        if not diff_content:
            logger.info("\n尝试通过 API 获取 diff...")
            diff_content = get_pr_diff_via_api(owner, repo, pr_number)
            if diff_content:
                result['method'] = 'api'

        if not diff_content:
            diff_content, branch = get_pr_diff_via_git_fetch(owner, repo, pr_number, work_repo_dir)
            if diff_content:
                result['method'] = 'git_fetch_legacy'

        if not diff_content:
            error_msg = (
                f"无法获取 PR #{pr_number} 的代码变更。\n"
                f"可能的原因：\n"
                f"  1. 网络问题导致无法连接到 GitCode\n"
                f"  2. API Token 未设置或权限不足 (当前: {'已设置' if GITCODE_TOKEN else '未设置'})\n"
                f"  3. PR 作者的仓库不可访问\n"
                f"\n"
                f"建议解决方法：\n"
                f"  1. 设置环境变量: export GITCODE_TOKEN=your_token\n"
                f"  2. 确保网络可以访问 gitcode.com\n"
                f"  3. 手动下载 PR diff 或让作者提供变更内容\n"
            )
            logger.info(f"\n❌ {error_msg}")
            result['error'] = error_msg
            return result

        diff_filename = f'pr_{pr_number}.diff'
        diff_filepath = os.path.join(work_output_dir, diff_filename)
        save_diff_to_file(diff_content, diff_filepath)
        result['diff_file'] = diff_filepath

        logger.info("-" * 60)
        logger.info("Step 3: 分析修改的代码")
        logger.info("-" * 60)

        analysis = analyze_changed_files(diff_content)
        result['analysis'] = analysis

        logger.info(f"总修改文件数: {analysis['total_files']}")
        if analysis['pass_files']:
            logger.info(f"Pass 文件: {len(analysis['pass_files'])}")
            for f in analysis['pass_files'][:5]:
                logger.info(f"  - {f}")
            if len(analysis['pass_files']) > 5:
                logger.info(f"  ... 还有 {len(analysis['pass_files']) - 5} 个")
        if analysis['test_files']:
            logger.info(f"测试文件: {len(analysis['test_files'])}")
            for f in analysis['test_files'][:5]:
                logger.info(f"  - {f}")

        logger.info("-" * 60)
        logger.info("Step 4: 应用 diff 到本地仓库")
        logger.info("-" * 60)

        logger.info(f"自动 stash: {'开启' if config.auto_stash else '关闭'}")
        apply_success, apply_msg = apply_diff_to_repo(diff_filepath, work_repo_dir, auto_stash=config.auto_stash)
        if apply_success:
            logger.info(f"{apply_msg}")
            result['diff_applied'] = True
        else:
            logger.warning(f"{apply_msg}")
            result['diff_applied'] = False
            result['diff_apply_error'] = apply_msg

        logger.info("-" * 60)
        logger.info("Step 5: 检查编译状态")
        logger.info("-" * 60)

        if config.check_build and result['diff_applied']:
            build_success, build_msg = check_build_status(work_repo_dir, analysis['pass_files'])
            if build_success:
                logger.info(f"{build_msg}")
            else:
                logger.warning(f"{build_msg}")
            result['build_status'] = {'success': build_success, 'message': build_msg}
        else:
            logger.info("跳过编译检查")
            result['build_status'] = {'success': None, 'message': '跳过'}

        logger.info("-" * 60)
        logger.info("Step 6: 分析是否需要设计 UT")
        logger.info("-" * 60)

        need_design_ut = False
        design_ut_reason = ""

        if ut_result['status'] == 'SUCCESS':
            design_ut_reason = "UT 测试全部通过，覆盖率满足要求"
            logger.info(f"✅ {design_ut_reason}")
        elif ut_result['status'] == 'PARTIAL_FAILED':
            need_design_ut = True
            design_ut_reason = f"UT 测试部分失败: {', '.join(ut_result['failed_tests'])}"
            logger.error(f"{design_ut_reason}")
        elif ut_result['status'] == 'ABORT':
            need_design_ut = True
            design_ut_reason = "UT 测试部分中止，需要检查覆盖率"
            logger.warning(f"{design_ut_reason}")
        else:
            need_design_ut = True
            design_ut_reason = "无法获取 UT 状态，建议设计 UT"
            logger.warning(f"{design_ut_reason}")

        if analysis['pass_files'] and ut_result['status'] != 'SUCCESS':
            need_design_ut = True
            design_ut_reason += f"\n  修改了 {len(analysis['pass_files'])} 个 Pass 文件，需要设计 UT"
            logger.info("\n⚠️ 修改了 Pass 文件，需要设计 UT 覆盖新代码")

        result['need_design_ut'] = need_design_ut
        result['design_ut_reason'] = design_ut_reason

        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info("=" * 60)
        result['success'] = True
        return result

    except Exception as e:
        result['error'] = str(e)
        logger.info(f"\n❌ 处理失败: {e}")
        import traceback

        traceback.print_exc()
        return result


def process_offline_diff(diff_file: str) -> Dict:
    """
    处理离线 diff 文件

    Args:
        diff_file: diff 文件路径

    Returns:
        包含处理结果的字典
    """
    result = {'diff_content': None, 'analysis': {}, 'success': False, 'error': None, 'method': 'offline_file'}

    try:
        if not os.path.exists(diff_file):
            result['error'] = f"文件不存在: {diff_file}"
            return result

        with open(diff_file, 'r', encoding='utf-8') as f:
            diff_content = f.read()

        if not diff_content.strip():
            result['error'] = "diff 文件内容为空"
            return result

        result['diff_content'] = diff_content
        result['analysis'] = analyze_changed_files(diff_content)
        result['success'] = True

        logger.info("成功解析离线 diff 文件")
        logger.info(f"  文件路径: {diff_file}")
        logger.info(f"  变更文件数: {result['analysis']['total_files']}")
        logger.info(f"  Pass 文件数: {len(result['analysis']['pass_files'])}")

        return result

    except Exception as e:
        result['error'] = str(e)
        logger.warning(f"处理离线 diff 文件失败: {e}")
        return result


def process_offline_ut_report(report_file: str) -> Dict:
    """
    处理离线 UT-Report 文件

    Args:
        report_file: UT-Report 文件路径（HTML 或文本）

    返回:
        包含处理结果的字典
    """
    result = {
        'report_content': None,
        'coverage_info': {},
        'low_coverage_files': [],
        'success': False,
        'error': None,
        'method': 'offline_report',
    }

    try:
        if not os.path.exists(report_file):
            result['error'] = f"文件不存在: {report_file}"
            return result

        with open(report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()

        if not report_content.strip():
            result['error'] = "UT-Report 文件内容为空"
            return result

        result['report_content'] = report_content

        if report_file.endswith('.html') or '<html' in report_content.lower():
            from scripts.ut_coverage import find_low_coverage_files, parse_coverage_html

            result['coverage_info'] = parse_coverage_html(report_content)
            result['low_coverage_files'] = find_low_coverage_files(result['coverage_info'], 80.0)
        else:
            result['coverage_info'] = {'raw_content': report_content[:500]}

        result['success'] = True

        logger.info("成功解析离线 UT-Report 文件")
        logger.info(f"  文件路径: {report_file}")
        logger.info(f"  总体覆盖率: {result['coverage_info'].get('overall_line_coverage', 'N/A')}")
        logger.info(f"  低覆盖率文件数: {len(result['low_coverage_files'])}")

        return result

    except Exception as e:
        result['error'] = str(e)
        logger.warning(f"处理离线 UT-Report 文件失败: {e}")
        return result


def analyze_offline_files(diff_file: Optional[str] = None, report_file: Optional[str] = None) -> Dict:
    """
    综合分析离线 diff 和 UT-Report 文件

    Args:
        diff_file: diff 文件路径
        report_file: UT-Report 文件路径

    返回:
        包含分析结果的字典
    """
    result = {
        'diff_result': None,
        'report_result': None,
        'combined_analysis': {},
        'ut_design_items': [],
        'success': False,
    }

    logger.info("=" * 60)
    logger.info("离线文件分析")
    logger.info("=" * 60)

    if diff_file:
        logger.info(f"\n[1] 处理 Diff 文件: {diff_file}")
        result['diff_result'] = process_offline_diff(diff_file)

    if report_file:
        logger.info(f"\n[2] 处理 UT-Report 文件: {report_file}")
        result['report_result'] = process_offline_ut_report(report_file)

    if result['diff_result'] and result['diff_result']['success']:
        if result['report_result'] and result['report_result']['success']:
            logger.info("\n[3] 关联分析...")

            from scripts.ut_coverage import generate_ut_design_suggestions

            suggestions = generate_ut_design_suggestions(
                result['diff_result']['analysis'], result['report_result']['coverage_info']
            )

            result['ut_design_items'] = [
                {
                    'file_path': s.file_path,
                    'uncovered_lines': s.uncovered_lines,
                    'change_type': s.change_type,
                    'suggestion': s.suggestion,
                    'priority': s.priority,
                }
                for s in suggestions
            ]

            logger.info(f"  ✓ 生成 {len(result['ut_design_items'])} 条 UT 设计建议")

    result['success'] = True
    return result


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='PR UT 生成工具 V8（支持离线分析）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式：
  # 在线：处理 PR
  python pr_utils.py 1856
  python pr_utils.py --pr 1856

  # 离线：分析 diff 文件
  python pr_utils.py --diff ./diff.file

  # 离线：分析 UT-Report 文件
  python pr_utils.py --report ./report.html

  # 离线：综合分析
  python pr_utils.py --diff diff.file --report report.html

  # 输出 JSON
  python pr_utils.py --diff diff.file --json
        """,
    )

    parser.add_argument('pr_input', nargs='?', help='PR 编号或链接')
    parser.add_argument('--pr', dest='pr_number', help='PR 编号')
    parser.add_argument('--diff', dest='diff_file', help='离线 diff 文件路径')
    parser.add_argument('--report', dest='report_file', help='离线 UT-Report 文件路径')
    parser.add_argument('--json', action='store_true', help='输出 JSON 格式')
    parser.add_argument('--output', '-o', help='输出文件路径')

    args = parser.parse_args()

    if args.diff_file:
        result = analyze_offline_files(diff_file=args.diff_file, report_file=args.report_file)

        if args.json or args.output:
            output = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                logger.info(f"\n结果已保存到: {args.output}")
            else:
                logger.info(output)
        else:
            print_summary(result)

        return 0

    if args.report_file:
        result = process_offline_ut_report(args.report_file)
        if args.json:
            logger.info(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    pr_input = args.pr_input or args.pr_number
    if not pr_input:
        logger.info("PR UT 生成工具 V8")
        logger.info("=" * 60)
        logger.info("使用方式：")
        logger.info("  python pr_utils.py <PR编号或链接>")
        logger.info("  python pr_utils.py --diff <diff文件>")
        logger.info("  python pr_utils.py --report <覆盖率报告>")
        logger.info("  python pr_utils.py --help 查看更多选项")
        return 1

    config = PRProcessConfig(
        pr_input=pr_input,
        output_dir=args.output,
        repo_dir=None,
        use_api_for_comments=True,
        auto_stash=True,
        check_build=True,
    )
    result = process_pr(config)

    if args.json or args.output:
        output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"\n结果已保存到: {args.output}")
        else:
            logger.info(output)
    else:
        print_summary(result)

    return 0


def print_summary(result: Dict):
    """打印结果摘要"""
    logger.info("=" * 60)
    logger.info("处理结果汇总")
    logger.info("=" * 60)
    logger.info(f"成功: {result.get('success', False)}")

    if result.get('pr_info'):
        logger.info(f"PR: #{result['pr_info']['pr_number']} ({result['pr_info']['owner']}/{result['pr_info']['repo']})")

    if result.get('method'):
        logger.info(f"处理方式: {result['method']}")

    if result.get('analysis'):
        analysis = result['analysis']
        logger.info(f"修改文件数: {analysis.get('total_files', 0)}")
        logger.info(f"Pass 文件数: {len(analysis.get('pass_files', []))}")

    if result.get('ut_report'):
        ut = result['ut_report']
        logger.info(f"\nUT 状态: {ut.get('status', 'N/A')}")
        logger.info(f"需要设计 UT: {result.get('need_design_ut', False)}")
        if ut.get('failed_tests'):
            logger.info(f"失败测试: {', '.join(ut['failed_tests'])}")

    if result.get('diff_result'):
        logger.info("\nDiff 分析完成:")
        logger.info(f"  Pass 文件数: {len(result['diff_result']['analysis'].get('pass_files', []))}")

    if result.get('report_result'):
        logger.info("\nUT-Report 分析完成:")
        logger.info(f"  低覆盖率文件数: {len(result['report_result'].get('low_coverage_files', []))}")

    if result.get('ut_design_items'):
        logger.info("\nUT 设计建议:")
        for i, item in enumerate(result['ut_design_items'][:3], 1):
            logger.info(f"  {i}. [{item['priority'].upper()}] {item['file_path']}")
        if len(result['ut_design_items']) > 3:
            logger.info(f"  ... 还有 {len(result['ut_design_items']) - 3} 条建议")

    if result.get('error'):
        logger.info(f"\n错误: {result['error']}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
