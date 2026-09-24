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
UT 覆盖率分析工具 V3

功能：
1. 解析 diff 文件，提取变更的 Pass 文件和代码段
2. 从 URL 下载覆盖率报告并解析
3. 解析本地覆盖率报告，提取低覆盖率文件和未覆盖行
4. 关联 Diff 和覆盖率，识别需要补充 UT 的代码
5. 支持离线分析（用户提供 diff 文件或覆盖率报告）

支持多种输入方式：
- --diff <file>: 解析本地 diff 文件
- --report <file/url>: 解析本地覆盖率报告或从 URL 下载
- --content <string>: 直接传入 diff 内容
"""

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
import re
import shutil
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
class DiffFile:
    """Diff 文件信息"""

    path: str
    additions: int = 0
    deletions: int = 0
    is_pass_file: bool = False
    is_test_file: bool = False
    changed_lines: List[Dict] = field(default_factory=list)


@dataclass
class CoverageFile:
    """覆盖率文件信息"""

    name: str
    path: str
    line_coverage: float
    function_coverage: float
    uncovered_lines: List[int] = field(default_factory=list)
    uncovered_functions: List[str] = field(default_factory=list)


@dataclass
class UTDesignItem:
    """需要设计 UT 的项"""

    file_path: str
    uncovered_lines: List[int]
    change_type: str
    suggestion: str
    priority: str


def parse_diff_file(file_path: str) -> Dict:
    """
    解析 diff 文件

    Args:
        file_path: diff 文件路径

    Returns:
        包含解析结果的字典
    """
    if not os.path.exists(file_path):
        return {'error': f'文件不存在: {file_path}', 'files': []}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return analyze_diff_content(content)


def analyze_diff_content(diff_content: str) -> Dict:
    """
    分析 diff 内容

    Args:
        diff_content: diff 内容字符串

    Returns:
        包含分析结果的字典
    """
    result = {
        'files': [],
        'pass_files': [],
        'test_files': [],
        'other_files': [],
        'total_files': 0,
        'total_additions': 0,
        'total_deletions': 0,
    }

    if not diff_content or not diff_content.strip():
        return result

    current_file: Optional[DiffFile] = None
    current_hunk_start = 0
    current_line_num = 0

    for line in diff_content.split('\n'):
        if line.startswith('diff --git'):
            if current_file:
                result['files'].append(
                    {
                        'path': current_file.path,
                        'additions': current_file.additions,
                        'deletions': current_file.deletions,
                        'is_pass_file': current_file.is_pass_file,
                        'is_test_file': current_file.is_test_file,
                        'changed_lines': current_file.changed_lines,
                    }
                )

                if current_file.is_pass_file:
                    result['pass_files'].append(current_file.path)
                elif current_file.is_test_file:
                    result['test_files'].append(current_file.path)
                else:
                    result['other_files'].append(current_file.path)

            match = re.search(r'a/(\S+)', line)
            if match:
                file_path = match.group(1)
                current_file = DiffFile(
                    path=file_path,
                    is_pass_file='passes' in file_path and file_path.endswith('.cpp'),
                    is_test_file='test' in file_path.lower() and file_path.endswith('.cpp'),
                )

        elif line.startswith('@@'):
            match = re.search(r'@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@', line)
            if match:
                current_hunk_start = int(match.group(1))
                current_line_num = current_hunk_start

        elif current_file is not None:
            if line.startswith('+') and not line.startswith('+++'):
                current_file.additions += 1
                result['total_additions'] += 1
                current_file.changed_lines.append(
                    {'line_num': current_line_num, 'type': 'addition', 'content': line[1:]}
                )
                current_line_num += 1
            elif line.startswith('-') and not line.startswith('---'):
                current_file.deletions += 1
                result['total_deletions'] += 1
                current_file.changed_lines.append(
                    {'line_num': current_line_num, 'type': 'deletion', 'content': line[1:]}
                )
                current_line_num += 1
            elif not line.startswith('\\'):
                current_line_num += 1

    if current_file:
        result['files'].append(
            {
                'path': current_file.path,
                'additions': current_file.additions,
                'deletions': current_file.deletions,
                'is_pass_file': current_file.is_pass_file,
                'is_test_file': current_file.is_test_file,
                'changed_lines': current_file.changed_lines,
            }
        )

        if current_file.is_pass_file:
            result['pass_files'].append(current_file.path)
        elif current_file.is_test_file:
            result['test_files'].append(current_file.path)
        else:
            result['other_files'].append(current_file.path)

    result['total_files'] = len(result['files'])
    return result


def download_coverage_report(url: str, output_dir: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    从 URL 下载覆盖率报告

    Args:
        url: 覆盖率报告 URL
        output_dir: 输出目录（可选）

    Returns:
        (是否成功, 消息, 报告路径/解压目录)
    """
    if not url:
        return False, "URL 为空", ""

    try:
        logger.info(f"下载覆盖率报告: {url[:80]}...")

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='cov_download_')

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        req = urllib.request.Request(url, headers={'Accept': 'application/octet-stream'})
        with urllib.request.urlopen(req, timeout=120) as response:
            total_size = response.headers.get('Content-Length', 0)
            downloaded = 0
            block_size = 8192

            with open(tmp_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if total_size:
                        pass

        logger.info("\n✓ 覆盖率报告下载成功")

        extract_dir = os.path.join(output_dir, 'cov_report')
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(tmp_path, 'r:gz') as tar:
            safe_extractall(tar, extract_dir)

        logger.info("覆盖率报告解压到: {extract_dir}")

        os.unlink(tmp_path)

        html_file = find_html_file(extract_dir)
        if html_file:
            return True, "覆盖率报告已解压", html_file
        return True, "覆盖率报告已解压", extract_dir

    except urllib.error.HTTPError as e:
        return False, f"下载失败: HTTP {e.code} - {e.reason}", ""
    except Exception as e:
        return False, f"下载失败: {e}", ""
    finally:
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)


def is_coverage_html_file(filename: str) -> bool:
    """判断是否为覆盖率报告 HTML 文件"""
    if not filename.endswith('.html'):
        return False
    lower_name = filename.lower()
    return 'coverage' in lower_name or 'index' in lower_name


def find_html_file(directory: str) -> Optional[str]:
    """在目录中查找 HTML 报告文件"""
    html_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_coverage_html_file(file):
                html_files.append(os.path.join(root, file))

    if html_files:
        return html_files[0]
    return None


def parse_coverage_report(report_path: str) -> Dict:
    """
    解析覆盖率报告（支持本地文件和 URL）

    Args:
        report_path: 覆盖率报告路径（HTML 文件、包含 HTML 的目录或 URL）

    Returns:
        包含覆盖率信息的字典
    """
    if not report_path:
        return {'error': '报告路径为空', 'files': []}

    if report_path.startswith('http://') or report_path.startswith('https://'):
        success, msg, path = download_coverage_report(report_path)
        if not success:
            return {'error': msg, 'files': []}
        report_path = path

    if not os.path.exists(report_path):
        return {'error': f'文件不存在: {report_path}', 'files': []}

    if os.path.isdir(report_path):
        html_file = find_html_in_dir(report_path)
        if not html_file:
            return {'error': '目录中未找到 HTML 报告', 'files': []}
        report_path = html_file

    with open(report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return parse_coverage_html(html_content)


def find_html_in_dir(dir_path: str) -> str:
    """在目录中查找 HTML 报告文件"""
    html_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if is_coverage_html_file(file):
                html_files.append(os.path.join(root, file))

    if html_files:
        return html_files[0]
    return ""


def parse_coverage_html(html_content: str) -> Dict:
    """
    解析覆盖率 HTML 内容

    Args:
        html_content: HTML 内容字符串

    Returns:
        包含覆盖率信息的字典
    """
    result = {'overall_line_coverage': '0%', 'overall_function_coverage': '0%', 'files': []}

    line_cov_match = re.search(r'Overall.*?(\d+(?:\.\d+)?)\s*%', html_content, re.DOTALL | re.IGNORECASE)
    if line_cov_match:
        result['overall_line_coverage'] = f"{line_cov_match.group(1)}%"

    func_cov_match = re.search(r'Functions.*?(\d+(?:\.\d+)?)\s*%', html_content, re.DOTALL | re.IGNORECASE)
    if func_cov_match:
        result['overall_function_coverage'] = f"{func_cov_match.group(1)}%"

    file_pattern = r'<tr[^>]*class="[^"]*file[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>.*?(\d+(?:\.\d+)?)%.*?(\d+(?:\.\d+)?)%'
    file_matches = re.findall(file_pattern, html_content, re.DOTALL)

    for name, line_pct, func_pct in file_matches:
        result['files'].append(
            {'name': name.strip(), 'line_coverage': float(line_pct), 'function_coverage': float(func_pct)}
        )

    uncovered_pattern = r'<span[^>]*class="[^"]*uncovered[^"]*"[^>]*>(\d+)</span>'
    for file_info in result['files']:
        uncovered = re.findall(uncovered_pattern, html_content[html_content.find(file_info['name']):], re.IGNORECASE)
        file_info['uncovered_lines'] = [int(n) for n in uncovered[:20]]

    return result


def find_low_coverage_files(coverage_info: Dict, threshold: float = 80.0) -> List[Dict]:
    """
    查找低覆盖率文件

    Args:
        coverage_info: 覆盖率信息
        threshold: 覆盖率阈值（默认 80%）

    Returns:
        低覆盖率文件列表
    """
    low_coverage = []

    for file_info in coverage_info.get('files', []):
        if file_info.get('line_coverage', 0) < threshold:
            low_coverage.append(
                {
                    'name': file_info.get('name', ''),
                    'path': file_info.get('path', ''),
                    'line_coverage': file_info.get('line_coverage', 0),
                    'function_coverage': file_info.get('function_coverage', 0),
                    'uncovered_lines': file_info.get('uncovered_lines', []),
                    'coverage_value': file_info.get('line_coverage', 0),
                }
            )

    low_coverage.sort(key=lambda x: x['line_coverage'])
    return low_coverage


def correlate_coverage_with_diff(diff_info: Dict, coverage_info: Dict) -> List[Dict]:
    """
    将覆盖率信息与 Diff 变更关联

    Args:
        diff_info: Diff 分析结果
        coverage_info: 覆盖率信息

    Returns:
        关联分析结果
    """
    result = []

    changed_files = {f['path']: f for f in diff_info.get('files', [])}

    for file_info in coverage_info.get('files', []):
        file_name = file_info.get('name', '')
        file_path = None

        for path in changed_files.keys():
            if path.endswith(file_name) or file_name in path:
                file_path = path
                break

        if file_path:
            _diff_file = changed_files[file_path]
            change_type = 'modified'
        elif any(file_name.endswith(ext) for ext in ['.cpp', '.h']):
            change_type = 'new'
        else:
            change_type = 'unchanged'

        item = {
            'name': file_name,
            'path': file_path or '',
            'line_coverage': file_info.get('line_coverage', 0),
            'function_coverage': file_info.get('function_coverage', 0),
            'uncovered_lines': file_info.get('uncovered_lines', []),
            'change_type': change_type,
            'need_ut': file_info.get('line_coverage', 0) < 80.0 and change_type in ['modified', 'new'],
        }
        result.append(item)

    return result


def generate_ut_design_suggestions(diff_info: Dict, coverage_info: Dict, threshold: float = 80.0) -> List[UTDesignItem]:
    """
    生成 UT 设计建议

    Args:
        diff_info: Diff 分析结果
        coverage_info: 覆盖率信息
        threshold: 覆盖率阈值

    Returns:
        UT 设计建议列表
    """
    suggestions = []

    correlation = correlate_coverage_with_diff(diff_info, coverage_info)

    for item in correlation:
        if not item['need_ut']:
            continue

        file_path = item['path']
        uncovered_lines = item['uncovered_lines']

        if not file_path or not uncovered_lines:
            continue

        diff_file = None
        for f in diff_info.get('files', []):
            if f['path'] == file_path:
                diff_file = f
                break

        if not diff_file:
            continue

        change_type = 'modified' if diff_file.get('additions', 0) > 0 else 'new'

        suggestion = f"针对 {file_path} 设计 UT，覆盖以下场景：\n"
        suggestion += f"- 覆盖率: {item['line_coverage']}%\n"
        suggestion += f"- 未覆盖行: {uncovered_lines[:5]}"
        if len(uncovered_lines) > 5:
            suggestion += f" 等共 {len(uncovered_lines)} 行"

        if uncovered_lines:
            first_uncovered = min(uncovered_lines)
            priority = 'high' if first_uncovered < 50 else 'medium'
        else:
            priority = 'low'

        suggestions.append(
            UTDesignItem(
                file_path=file_path,
                uncovered_lines=uncovered_lines,
                change_type=change_type,
                suggestion=suggestion,
                priority=priority,
            )
        )

    suggestions.sort(key=lambda x: 0 if x.priority == 'high' else (1 if x.priority == 'medium' else 2))
    return suggestions


def analyze_coverage_for_pr(diff_info: Dict, coverage_url: str, threshold: float = 80.0) -> Dict:
    """
    综合分析 PR 的 Diff 和覆盖率

    Args:
        diff_info: Diff 分析结果
        coverage_url: 覆盖率报告 URL
        threshold: 覆盖率阈值

    Returns:
        分析结果
    """
    result = {
        'diff_info': diff_info,
        'coverage_info': None,
        'low_coverage_files': [],
        'ut_design_suggestions': [],
        'success': False,
        'error': None,
    }

    if not coverage_url:
        result['error'] = '未提供覆盖率报告 URL'
        return result

    coverage_info = parse_coverage_report(coverage_url)
    if 'error' in coverage_info:
        result['error'] = coverage_info['error']
        return result

    result['coverage_info'] = coverage_info
    result['low_coverage_files'] = find_low_coverage_files(coverage_info, threshold)

    suggestions = generate_ut_design_suggestions(diff_info, coverage_info, threshold)
    result['ut_design_suggestions'] = [
        {
            'file_path': s.file_path,
            'uncovered_lines': s.uncovered_lines,
            'change_type': s.change_type,
            'suggestion': s.suggestion,
            'priority': s.priority,
        }
        for s in suggestions
    ]

    result['success'] = True
    return result


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='UT 覆盖率分析工具 V3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 解析 diff 文件
  python ut_coverage.py --diff ./diff.file

  # 解析本地覆盖率报告
  python ut_coverage.py --report ./coverage.html

  # 从 URL 下载并解析覆盖率报告
  python ut_coverage.py --report https://example.com/ut_cov.tar.gz

  # 同时分析 diff 和覆盖率（支持 URL）
  python ut_coverage.py --diff diff.file --report https://example.com/ut_cov.tar.gz

  # 输出 JSON 格式
  python ut_coverage.py --diff diff.file --report report.html --json
        """,
    )

    parser.add_argument('--diff', '-d', type=str, help='Diff 文件路径')
    parser.add_argument('--report', '-r', type=str, help='覆盖率报告文件或 URL 路径')
    parser.add_argument('--content', '-c', type=str, help='Diff 内容字符串')
    parser.add_argument('--json', '-j', action='store_true', help='输出 JSON 格式')
    parser.add_argument('--threshold', '-t', type=float, default=80.0, help='覆盖率阈值 (默认: 80.0)')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径')

    args = parser.parse_args()

    result = {'diff_info': None, 'coverage_info': None, 'low_coverage_files': [], 'ut_design_suggestions': []}

    logger.info("=" * 60)
    logger.info("UT 覆盖率分析工具 V3")
    logger.info("=" * 60)

    if args.diff:
        logger.info(f"\n[1/3] 解析 Diff 文件: {args.diff}")
        diff_content = None
        if os.path.isfile(args.diff):
            with open(args.diff, 'r', encoding='utf-8') as f:
                diff_content = f.read()
        result['diff_info'] = analyze_diff_content(diff_content or args.diff)
        logger.info("  ✓ 解析完成")
        logger.info(f"    总文件数: {result['diff_info']['total_files']}")
        logger.info(f"    Pass 文件数: {len(result['diff_info']['pass_files'])}")
        logger.info(f"    新增行: {result['diff_info']['total_additions']}")
        logger.info(f"    删除行: {result['diff_info']['total_deletions']}")
    elif args.content:
        logger.info("\n[1/3] 解析 Diff 内容...")
        result['diff_info'] = analyze_diff_content(args.content)
        logger.info("  ✓ 解析完成")

    if args.report:
        logger.info(f"\n[2/3] 解析覆盖率报告: {args.report[:60]}...")
        result['coverage_info'] = parse_coverage_report(args.report)
        if 'error' in result['coverage_info']:
            logger.info(f"  ✗ 错误: {result['coverage_info']['error']}")
        else:
            logger.info("  ✓ 解析完成")
            logger.info(f"    总体行覆盖率: {result['coverage_info'].get('overall_line_coverage', 'N/A')}")
            logger.info(f"    总体函数覆盖率: {result['coverage_info'].get('overall_function_coverage', 'N/A')}")
            logger.info(f"    文件数: {len(result['coverage_info'].get('files', []))}")

    if result['diff_info'] and result['coverage_info']:
        logger.info("\n[3/3] 关联分析与建议生成...")

        result['low_coverage_files'] = find_low_coverage_files(result['coverage_info'], threshold=args.threshold)

        suggestions = generate_ut_design_suggestions(
            result['diff_info'], result['coverage_info'], threshold=args.threshold
        )
        result['ut_design_suggestions'] = [
            {
                'file_path': s.file_path,
                'uncovered_lines': s.uncovered_lines,
                'change_type': s.change_type,
                'suggestion': s.suggestion,
                'priority': s.priority,
            }
            for s in suggestions
        ]

        logger.info(f"  ✓ 低覆盖率文件数: {len(result['low_coverage_files'])}")
        logger.info(f"  ✓ UT 设计建议数: {len(result['ut_design_suggestions'])}")

    logger.info("=" * 60)
    logger.info("分析结果")
    logger.info("=" * 60)

    if args.json or args.output:
        output_content = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            logger.info(f"结果已保存到: {args.output}")
        else:
            logger.info(output_content)
    else:
        if result['diff_info']:
            logger.info("\n【Diff 变更摘要】")
            logger.info(f"  Pass 文件 ({len(result['diff_info']['pass_files'])}):")
            for pf in result['diff_info']['pass_files'][:5]:
                logger.info(f"    - {pf}")
            if len(result['diff_info']['pass_files']) > 5:
                logger.info(f"    ... 还有 {len(result['diff_info']['pass_files']) - 5} 个")

        if result['low_coverage_files']:
            logger.info("\n【低覆盖率文件】")
            for i, lcf in enumerate(result['low_coverage_files'][:5], 1):
                logger.info(f"  {i}. {lcf['name']}")
                logger.info(f"     覆盖率: {lcf['line_coverage']}%")
                logger.info(f"     未覆盖行: {lcf['uncovered_lines'][:5]}")
            if len(result['low_coverage_files']) > 5:
                logger.info(f"  ... 还有 {len(result['low_coverage_files']) - 5} 个文件")

        if result['ut_design_suggestions']:
            logger.info("\n【UT 设计建议】")
            for i, sug in enumerate(result['ut_design_suggestions'][:3], 1):
                logger.info(f"  {i}. [{sug['priority'].upper()}] {sug['file_path']}")
                logger.info(f"     {sug['suggestion']}")
            if len(result['ut_design_suggestions']) > 3:
                logger.info(f"  ... 还有 {len(result['ut_design_suggestions']) - 3} 条建议")
                logger.info("\n  使用 --json 选项查看完整建议")

    return 0


if __name__ == "__main__":
    sys.exit(main())
