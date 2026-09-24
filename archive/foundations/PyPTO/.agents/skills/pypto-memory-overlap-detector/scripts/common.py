#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import logging
import os


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def validate_path(path, path_type="路径"):
    if not os.path.exists(path):
        return False, f"错误：{path_type}不存在: {path}"
    return True, None


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def find_trace_log_dir(log_base_path):
    for root, _, files in os.walk(log_base_path):
        for file in files:
            if (file.startswith('device') or file.startswith('pypto-simulation')) and file.endswith('.log'):
                log_file = os.path.join(root, file)
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    if '#trace' in f.read():
                        return root
    return None


def get_latest_output_subdir(output_path, target_file):
    if not os.path.exists(output_path):
        return None, f"错误：output 目录不存在: {output_path}"

    if not os.path.isdir(output_path):
        return None, "错误：路径不是目录"

    subdirs = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith('output_'):
            subdirs.append(item_path)

    if not subdirs:
        return None, "错误：未找到以 'output_' 开头的子文件夹"

    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = subdirs[0]

    target_path = os.path.join(latest_dir, target_file)
    if not os.path.exists(target_path):
        return None, f"错误：{target_file} 不存在于: {target_path}"

    return target_path, None
