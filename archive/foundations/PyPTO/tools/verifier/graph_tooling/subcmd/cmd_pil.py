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
""" """

import ast
import os
import traceback

import pypto

from ..utils.cli import argument, command
from ..utils.io import print_line


def pil_compile(src_filepath, dst_filepath):
    if not os.path.exists(src_filepath):
        raise Exception(f'Input does not exist: {src_filepath}')
    with open(src_filepath) as src_file:
        stmt_list = ast.parse(src_file.read()).body
    pil_stmt_list = pypto.frontend.parser.pil.parse_stmts(stmt_list)
    pil_code = ast.unparse(pil_stmt_list)
    if dst_filepath == '-':
        print_line(pil_code)
    else:
        with open(dst_filepath, 'w') as dst_file:
            dst_file.write(pil_code)


@command(help='PIL transformation')
@argument('input_files', type=str, nargs='+')
@argument('--output', '-o', type=str, default=None)
def pil(input_files, output):
    if len(input_files) != 1:
        if output is not None:
            raise Exception('-o is not allowed for multiple inputs')
        for src in input_files:
            dst = f'{src}.pil.py'
            try:
                pil_compile(src, dst)
            except Exception:
                traceback.print_exc()
    else:
        if output is None:
            output = '-'
        pil_compile(input_files[0], output)
