#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render pil-transformed test sources.

Reads every ``test_*.raw.py`` in this directory, transforms it through the pil
frontend (``ast.parse`` -> ``pil_parser.parse_stmts`` -> ``ast.unparse``) and writes the
result to the matching ``test_*.pil.py``. The pil form is semantically
equivalent to the raw form (it is a 3-address normalisation), so both should
behave identically when executed.
"""

import ast
import glob
import os

import pypto.frontend.parser.pil_parser as pil_parser

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_DATA_DIR, "render_data.txt"), encoding="utf-8") as _f:
    _LICENSE = _f.read()


def render(src):
    """Transform python source string into its pil-normalised form.

    The generated source is prefixed with the license header from
    ``render_data.txt``.
    """
    tree = ast.parse(src)
    transformed = pil_parser.parse_stmts(tree.body)
    code = ast.unparse(ast.Module(body=transformed, type_ignores=[]))
    return _LICENSE + "\n" + code + "\n"


def render_file(raw_path):
    """Render a single ``test_*.py`` file and return its ``test_*.py.pil`` path."""
    with open(raw_path, encoding="utf-8") as f:
        src = f.read()
    out = render(src)
    pil_path = raw_path + ".pil"
    with open(pil_path, "w", encoding="utf-8") as f:
        f.write(out)
    return pil_path


_EXCLUDED = {"test_pil_builder_utils.py"}


def main():
    for raw_path in sorted(glob.glob(os.path.join(_DATA_DIR, "test_*.py"))):
        if os.path.basename(raw_path) in _EXCLUDED:
            continue
        pil_path = render_file(raw_path)
        print("wrote", os.path.relpath(pil_path))


if __name__ == "__main__":
    main()
