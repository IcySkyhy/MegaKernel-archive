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
"""Check every ``test_*.py.pil`` is a faithful render of its ``test_*.py``.

``render.py`` produces a ``.py.pil`` from a ``.py`` (raw) via
``ast.parse`` -> ``pil_parser.parse_stmts`` -> ``ast.unparse`` (prefixed with a license
header). This test re-applies the same transform to the raw file and compares it
against the committed ``.py.pil`` after normalising both sides through
``ast.parse`` -> ``ast.unparse``. The round-trip strips comments (the license
header) and collapses formatting, so the comparison is purely on the code
structure. It deliberately does not execute anything -- the behavioural
equivalence of the raw and pil forms is validated separately by running them
under pytest.
"""

import ast
import glob
import importlib.machinery
import importlib.util
import os
import sys
import types
import unittest

import pypto.frontend.parser.pil_parser as pil_parser

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pil_builder_data")

_NON_DATA_FILES = {"test_pil_builder_utils.py"}


def _render(src):
    """Apply the pil transform directly: ast.parse -> pil_parser.parse_stmts -> ast.unparse."""
    tree = ast.parse(src)
    transformed = pil_parser.parse_stmts(tree.body)
    return ast.unparse(ast.Module(body=transformed, type_ignores=[]))


def _normalize(src):
    """Round-trip source through ast.parse -> ast.unparse to drop comments/format."""
    return ast.unparse(ast.parse(src))


class TestPilBuilderRenderSync(unittest.TestCase):
    """Each committed ``test_*.py.pil`` must equal the pil transform of ``test_*.py``."""

    def test_pil_files_in_sync_with_raw(self):
        raw_paths = sorted(
            p for p in glob.glob(os.path.join(DATA_DIR, "test_*.py"))
            if os.path.basename(p) not in _NON_DATA_FILES
        )
        self.assertTrue(raw_paths, f"no test_*.py found under {DATA_DIR}")
        for raw_path in raw_paths:
            pil_path = raw_path + ".pil"
            stem = os.path.basename(raw_path)
            with self.subTest(file=stem):
                self.assertTrue(
                    os.path.exists(pil_path),
                    f"missing pil render for {stem}; run render.py",
                )
                with open(raw_path, encoding="utf-8") as f:
                    raw_src = f.read()
                with open(pil_path, encoding="utf-8") as f:
                    pil_src = f.read()
                self.assertEqual(
                    _normalize(_render(raw_src)),
                    _normalize(pil_src),
                    f"{stem}.pil is out of sync with {stem}; run render.py",
                )


_PIL_PARENT_PKG = "_pypto_pil_data_pkg"
_PIL_CHILD_PKG = _PIL_PARENT_PKG + ".test_pil_builder_data"


def _ensure_pil_package_context():
    """Register synthetic packages so ``.py.pil`` relative imports resolve.

    Each ``test_*.py.pil`` opens with ``from .test_pil_builder_utils import ...``.
    To make that relative import work the module must live in a package whose
    ``__path__`` is the ``test_pil_builder_data`` directory (where
    ``test_pil_builder_utils`` resides). We register two synthetic packages for
    that purpose: a parent anchor and a child pointing at
    ``test_pil_builder_data``.
    """
    if _PIL_PARENT_PKG not in sys.modules:
        parent = types.ModuleType(_PIL_PARENT_PKG)
        parent.__path__ = [os.path.dirname(DATA_DIR)]
        sys.modules[_PIL_PARENT_PKG] = parent
    if _PIL_CHILD_PKG not in sys.modules:
        child = types.ModuleType(_PIL_CHILD_PKG)
        child.__path__ = [DATA_DIR]
        child.__package__ = _PIL_CHILD_PKG
        sys.modules[_PIL_CHILD_PKG] = child


def _load_pil_module(pil_path):
    """Load a single ``.py.pil`` file as a Python module via the import API."""
    _ensure_pil_package_context()
    stem = os.path.basename(pil_path)[:-len(".py.pil")]
    mod_name = f"{_PIL_CHILD_PKG}.{stem}"
    sys.modules.pop(mod_name, None)
    loader = importlib.machinery.SourceFileLoader(mod_name, pil_path)
    spec = importlib.util.spec_from_file_location(mod_name, pil_path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class TestPilBuilderRunPil(unittest.TestCase):
    """Load every committed ``test_*.py.pil`` via the import API and run its cases."""

    def test_run_pil_files(self):
        pil_paths = sorted(glob.glob(os.path.join(DATA_DIR, "test_*.py.pil")))
        self.assertTrue(pil_paths, f"no test_*.py.pil found under {DATA_DIR}")
        for pil_path in pil_paths:
            stem = os.path.basename(pil_path)
            with self.subTest(file=stem):
                module = _load_pil_module(pil_path)
                suite = unittest.TestLoader().loadTestsFromModule(module)
                result = unittest.TestResult()
                suite.run(result)
                self.assertFalse(result.errors, f"{stem} had errors:\n{result.errors}")
                self.assertFalse(result.failures, f"{stem} had failures:\n{result.failures}")
                self.assertTrue(result.wasSuccessful(), f"{stem} was not successful")


if __name__ == "__main__":
    unittest.main()
