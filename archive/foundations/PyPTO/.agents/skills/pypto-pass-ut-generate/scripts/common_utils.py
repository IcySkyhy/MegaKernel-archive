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
公共工具模块 - Pass UT 生成工具共用函数

包含：
- GitCode API 请求
- Token 获取
- 文件下载
- HTTP 请求封装
- 日志记录
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

GITCODE_API_BASE = "https://api.gitcode.com/api/v5"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_gitcode_token(print_hint: bool = True) -> str:
    """
    获取 GitCode token（支持环境变量和配置文件）

    Args:
        print_hint: 是否打印配置提示

    Returns:
        GitCode token 字符串
    """
    # 方式一：环境变量
    token = os.environ.get("GITCODE_TOKEN", "")
    if token:
        return token

    # 方式二：从 opencode 配置文件读取
    try:
        config_path = os.path.expanduser("~/.config/opencode/opencode.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            mcp_config = config.get("mcp", {}).get("gitcode", {})
            env_config = mcp_config.get("environment", {})
            token = env_config.get("GITCODE_TOKEN", "")
            if token:
                return token
    except Exception as e:
        logger.debug("读取配置文件失败: %s", e)

    if print_hint:
        logger.warning("未找到 GitCode Token")
        logger.info("请选择以下方式之一配置 Token：")
        logger.info("方式一：在 opencode.json 中配置（推荐）")
        logger.info("  配置文件路径: %s", os.path.expanduser("~/.config/opencode/opencode.json"))
        logger.info('  添加以下内容: {"mcp": {"gitcode": {"environment": {"GITCODE_TOKEN": "your_token_here"}}}}')
        logger.info("方式二：手动提供离线文件")
        logger.info("  - 提供 .diff 文件: python3 scripts/pr_utils.py --diff ./diff.file")
        logger.info("  - 提供覆盖率报告: python3 scripts/ut_coverage.py --report ./coverage.html")

    return ""


def make_api_request(
    url: str, token: Optional[str] = None, headers: Optional[Dict] = None, timeout: int = 30, method: str = "GET"
) -> Optional[Any]:
    """
    通用 API 请求函数

    Args:
        url: 请求 URL
        token: API Token
        headers: 额外请求头
        timeout: 超时时间（秒）
        method: 请求方法

    Returns:
        解析后的 JSON 响应，失败返回 None
    """
    if headers is None:
        headers = {"Accept": "application/json"}

    if token:
        headers["PRIVATE-TOKEN"] = token

    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8')
            if 'application/json' in response.headers.get('Content-Type', ''):
                return json.loads(content)
            return content
    except urllib.error.HTTPError as e:
        logger.error("HTTP 错误: %d - %s", e.code, e.reason)
        return None
    except Exception as e:
        logger.error("API 请求失败: %s", e)
        return None


def make_gitcode_api_request(endpoint: str, token: Optional[str] = None, timeout: int = 30) -> Optional[Any]:
    """GitCode API 请求封装"""
    if not token:
        token = get_gitcode_token()

    url = f"{GITCODE_API_BASE}/{endpoint}"
    return make_api_request(url, token=token, timeout=timeout)


def get_pr_info(owner: str, repo: str, pr_number: int) -> Optional[Dict]:
    """获取 PR 信息"""
    return make_gitcode_api_request(f"repos/{owner}/{repo}/pulls/{pr_number}")


def get_pr_comments(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """获取 PR 评论"""
    result = make_gitcode_api_request(f"repos/{owner}/{repo}/pulls/{pr_number}/comments?per_page=100")
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return []


def get_pr_diff(owner: str, repo: str, pr_number: int) -> Optional[str]:
    """获取 PR diff"""
    if not get_gitcode_token():
        logger.warning("需要 GitCode Token 才能获取 diff")
        return None

    url = f"{GITCODE_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/diff"
    result = make_api_request(url, timeout=60)

    if result and isinstance(result, str) and result.strip():
        return result
    return None


def download_file(url: str, output_path: Optional[str] = None, timeout: int = 120) -> Tuple[bool, str]:
    """
    下载文件

    Args:
        url: 下载 URL
        output_path: 保存路径（可选）
        timeout: 超时时间

    Returns:
        (是否成功, 消息/保存路径)
    """
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/octet-stream'})

        if output_path is None:
            suffix = os.path.splitext(url.split('/')[-1])[-1] or '.tmp'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                output_path = f.name

        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192

            with open(output_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if total_size:
                        progress = int(downloaded * 100 / total_size)
                        logger.info("\r下载进度: %d%%", progress)

        logger.info("下载成功: %s", output_path)
        return True, output_path

    except urllib.error.HTTPError as e:
        logger.error("下载失败: HTTP %d", e.code)
        return False, "下载失败"
    except Exception as e:
        logger.error("下载失败: %s", e)
        return False, "下载失败"


def _is_within_directory(directory: str, target: str) -> bool:
    """
    判断 target 规整后的真实路径是否严格位于 directory 之内

    用于防止 Zip Slip / 路径穿越（CWE-22）。补 os.sep 防止
    /foo/bar-evil 绕过 /foo/bar 这类前缀误判。
    """
    abs_directory = os.path.realpath(directory)
    abs_target = os.path.realpath(target)
    prefix = abs_directory.rstrip(os.sep) + os.sep
    return abs_target == abs_directory or abs_target.startswith(prefix)


def safe_extractall(tar: tarfile.TarFile, extract_dir: str) -> None:
    """
    安全解压 tar：逐个成员校验后再解压，防止 Zip Slip / 路径穿越（CWE-22）

    校验规则：
    - 拒绝绝对路径成员
    - 拒绝解析后越出 extract_dir 的成员（含 .. 穿越）
    - 拒绝指向目录外的 symlink/hardlink 成员

    命中违规时抛出 ValueError（信息含成员名）。

    Args:
        tar: 已打开的 tarfile.TarFile 对象
        extract_dir: 解压目标目录
    """
    extract_dir = os.path.realpath(extract_dir)

    for member in tar.getmembers():
        # 拒绝绝对路径成员
        if os.path.isabs(member.name):
            raise ValueError(f"检测到不安全的 tar 成员（绝对路径）: {member.name}")

        member_path = os.path.join(extract_dir, member.name)
        # 校验成员目标路径必须位于 extract_dir 之内（拦截 .. 穿越）
        if not _is_within_directory(extract_dir, member_path):
            raise ValueError(f"检测到不安全的 tar 成员（路径穿越）: {member.name}")

        # 校验 symlink / hardlink 目标也必须位于 extract_dir 之内
        if member.issym() or member.islnk():
            if os.path.isabs(member.linkname):
                raise ValueError(f"检测到不安全的 tar 成员（链接指向绝对路径）: {member.name} -> {member.linkname}")
            # symlink 目标相对于成员所在目录；hardlink 目标相对于归档根目录
            if member.issym():
                link_target = os.path.join(os.path.dirname(member_path), member.linkname)
            else:
                link_target = os.path.join(extract_dir, member.linkname)
            if not _is_within_directory(extract_dir, link_target):
                raise ValueError(f"检测到不安全的 tar 成员（链接越界）: {member.name} -> {member.linkname}")

    # 只要运行时提供解压过滤器（Python 3.8.17+/3.9.17+/3.10.12+/3.11.4+/3.12+，
    # 以 tarfile.data_filter 是否存在做特性探测，而非版本号判断），就用 'data' 过滤器：
    # 它在解压时对每个成员重新校验，能拦下静态预校验无法发现的逃逸——例如归档内先出现
    # 的 symlink，使后续成员经其解析而越界。
    if hasattr(tarfile, "data_filter"):
        tar.extractall(extract_dir, filter="data")
        return

    # 旧运行时无解压过滤器：上面的静态预校验无法防"经归档内 symlink 解析越界"，
    # 故在该回退路径直接拒绝任何链接成员，仅解压普通条目（路径已逐一校验在 extract_dir 内）。
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"检测到不安全的 tar 成员（当前 Python 不支持解压过滤器，禁止链接成员）: {member.name}")
    tar.extractall(extract_dir)


def extract_tarball(tar_path: str, output_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    解压 tarball

    Args:
        tar_path: tar 文件路径
        output_dir: 输出目录

    Returns:
        (是否成功, 消息/输出目录)
    """
    created_dir = None
    try:
        if output_dir is None:
            created_dir = tempfile.mkdtemp(prefix='extract_')
            output_dir = created_dir

        os.makedirs(output_dir, exist_ok=True)

        with tarfile.open(tar_path, 'r:gz') as tar:
            safe_extractall(tar, output_dir)

        return True, output_dir

    except Exception as e:
        return False, f"解压失败: {e}"
    finally:
        if created_dir and os.path.exists(created_dir):
            shutil.rmtree(created_dir, ignore_errors=True)


def run_git_command(args: List[str], cwd: str, timeout: int = 60) -> Tuple[int, str, str]:
    """
    运行 git 命令

    Args:
        args: git 命令参数列表
        cwd: 工作目录
        timeout: 超时时间

    Returns:
        (返回码, stdout, stderr)
    """
    try:
        result = subprocess.run(['git'] + args, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def find_file_in_dir(directory: str, pattern: str) -> Optional[str]:
    """
    在目录中查找匹配模式的文件

    Args:
        directory: 搜索目录
        pattern: 文件名模式（支持 endswith）

    Returns:
        找到的文件路径，未找到返回 None
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern.replace('*', '')):
                return os.path.join(root, file)
    return None
