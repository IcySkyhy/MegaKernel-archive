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
"""End-to-end guard for the kernel bundle (.pyptokb): compile+pack -> segment layout -> load -> launch.

Each test walks the whole route on its own, so a single failing case still pins down which half broke:

  1. PACK    subprocess: compile and run the op with PYPTO_ENABLE_KERNEL_BUNDLE=1, producing a .pyptokb.
  2. INSPECT in-process: parse the container header and the TLV table, and assert the segment layout that
             the scenario implies (notably CtrlFlowCache empty vs. non-empty).
  3. CONSUME subprocess: dlopen libtile_fwk_bundle.so and drive PyptoWorkspace / PyptoLaunch against the file
             alone -- a stand-in for a downstream that has no compiler front end -- then check the numerics.

The two phases have to be separate processes: the consume half must not inherit any recorded Program state
from the pack half, otherwise the test would pass even if the bundle carried nothing.

Scenario coverage:

  ============== ================= ==================== =================================================
  test           control flow      workspace            what it pins down
  ============== ================= ==================== =================================================
  static_add     not value-dep     static, 0 B          baseline round trip; both the path and the
                                                        in-memory ABI variants; CtrlFlowCache non-empty
  dyn_cellmatch  not value-dep     dynamic (symbolic)   SymbolMeta symbol trees + dynamic cell-match
                                                        metas: workspace re-evaluated per launch shape
  value_depend   value-dependent   dynamic              CtrlFlowCache segment EMPTY; the same bundle
                                                        launched twice with different tensor VALUES
  standalone     not value-dep     static, 0 B          the self-contained delivery form: one .so + one
                                                        .pyptokb + configs/, no pypto in the process.
                                                        SKIPPED unless the artifact was built
  ============== ================= ==================== =================================================

Note on dyn_cellmatch: the v1 CtrlFlowCache is a single static-shape snapshot, so a non-value-dependent
bundle may only be LAUNCHED at the shape it was packed with -- launching another shape silently produces
wrong results or trips an AICPU fault. The multi-shape assertion therefore covers PyptoWorkspace only
(pure host, no dispatch), and the launch stays on the packed shape. Lift this once the cache goes dynamic.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from conftest import duration_estimate
import pypto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from st.bundle.bundle_abi import (  # noqa: E402
    TLV_AICORE_KERNEL,
    TLV_AICPU_SO,
    TLV_CTRL_FLOW_CACHE,
    TLV_DEV_PROGRAM,
    TLV_SYMBOL_META,
    read_bundle_tlvs,
    read_symbol_meta,
)

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle_cases.py")
_STANDALONE_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standalone_delivery_smoke.py")

# Packing compiles the op from scratch; consuming has to bring the runtime up on the device.
_PACK_TIMEOUT_S = 1800
_CONSUME_TIMEOUT_S = 900


def _pypto_lib_dir() -> str:
    """Locate <pypto>/lib without importing pypto -- find_spec resolves a top-level package without running it."""
    spec = importlib.util.find_spec("pypto")
    assert spec is not None and spec.origin, "pypto is not importable; install the whl first"
    return os.path.join(os.path.dirname(os.path.abspath(spec.origin)), "lib")


def _run_worker(phase: str, case: str, bundle_path: str, *args, extra_env=None, timeout=None) -> str:
    """Drive one phase of a scenario in a fresh interpreter; fail the test with its output on error."""
    env = os.environ.copy()
    env.update(extra_env or {})
    cmd = [sys.executable, _WORKER, phase, case, bundle_path, *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        pytest.fail(
            f"bundle {phase} phase failed for case '{case}' (rc={proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    print(proc.stdout)
    return proc.stdout


def _pack(case: str, bundle_path: str) -> str:
    return _run_worker(
        "pack",
        case,
        bundle_path,
        extra_env={
            "PYPTO_ENABLE_KERNEL_BUNDLE": "1",
            "PYPTO_KERNEL_BUNDLE_PATH": bundle_path,
        },
        timeout=_PACK_TIMEOUT_S,
    )


def _consume(case: str, bundle_path: str, api: str = "path") -> str:
    # Deliberately does NOT set PYPTO_ENABLE_KERNEL_BUNDLE: the consume side must work with packing off.
    return _run_worker("consume", case, bundle_path, "--api", api, timeout=_CONSUME_TIMEOUT_S)


def _assert_common_segments(bundle_path: str):
    """Every bundle must carry the device binary, the aicpu backend .so and the device program."""
    tlvs = read_bundle_tlvs(bundle_path)
    layout = {tlv.name: tlv.length for tlv in tlvs.values()}
    print(f"[ST] bundle segments: {layout}")

    for type_id in (TLV_AICORE_KERNEL, TLV_AICPU_SO, TLV_DEV_PROGRAM):
        tlv = tlvs.get(type_id)
        assert tlv is not None, f"missing TLV segment {type_id}; got {layout}"
        assert tlv.length > 0, f"segment {tlv.name} is empty; got {layout}"
        assert tlv.offset % 4096 == 0, f"segment {tlv.name} is not 4KB aligned (offset={tlv.offset})"
    return tlvs


@pytest.fixture(name="bundle_path")
def _bundle_path():
    """A private directory per test: a .pyptokb is ~100 MB, so do not let pytest retain them."""
    with tempfile.TemporaryDirectory(prefix="pypto_kb_") as tmp:
        yield os.path.join(tmp, "case.pyptokb")


@duration_estimate(40)
@pypto.options(pass_options={"enable_slice": True})
def test_bundle_static_add(bundle_path):
    """Non-value-dependent, static, zero-workspace op: the baseline pack -> load -> launch round trip.

    Also the only case that drives both ABI variants, since the in-memory entry points differ from the
    path ones only in how the image reaches the loader.
    """
    _pack("static_add", bundle_path)

    tlvs = _assert_common_segments(bundle_path)
    cache = tlvs.get(TLV_CTRL_FLOW_CACHE)
    assert cache is not None and cache.length > 0, (
        "a non-value-dependent op is packed through the emulation hook and must carry a control-flow "
        "cache snapshot; an empty segment means the pack hook moved or the op became value-dependent"
    )

    out = _consume("static_add", bundle_path, api="path")
    assert "workspace size = 0" in out
    out = _consume("static_add", bundle_path, api="memory")
    assert "workspace size = 0" in out


@duration_estimate(30)
@pypto.options(pass_options={"enable_slice": True})
def test_bundle_dynamic_cellmatch(bundle_path):
    """Dynamically shaped op: workspace is evaluated from the launch shapes, not read off a baked constant.

    The SymbolMeta segment has to carry both the symbolic workspace trees and the dynamic cell-match launch
    metas -- the load side re-derives the cell-match stride patches from those, and getting it wrong shows
    up as an AICPU execute failure rather than as a wrong number.
    """
    _pack("dyn_cellmatch", bundle_path)

    tlvs = _assert_common_segments(bundle_path)
    assert tlvs[TLV_CTRL_FLOW_CACHE].length > 0, "expected a control-flow cache snapshot for a non-value-dep op"

    meta = read_symbol_meta(bundle_path)
    assert meta, "a dynamically shaped op must carry a SymbolMeta segment"
    assert "cellMatchLaunch" in meta, (
        f"SymbolMeta carries no dynamic cell-match launch metas, so the load side cannot re-patch the "
        f"strides: {meta[:400]}"
    )
    assert "RUNTIME_GetInputShapeDim" in meta, (
        f"SymbolMeta has no shape-derived symbols, so the workspace is effectively baked: {meta[:400]}"
    )

    _consume("dyn_cellmatch", bundle_path)


@duration_estimate(34)
@pypto.options(pass_options={"enable_slice": True})
def test_bundle_value_depend_page_attention(bundle_path):
    """Value-dependent control flow: loop bounds come from tensor VALUES, so there is no cache to bake.

    Such ops set devProg->disableCtrlFlowCache, skip the emulation pack hook entirely and are packed from
    the launch path instead; the bundle must therefore carry an EMPTY CtrlFlowCache segment. The consume
    step launches the one bundle twice with identical shapes but different act_seqs values -- that only
    works if the on-device interpreter resolves the trip counts at launch.
    """
    _pack("value_depend_pa", bundle_path)

    tlvs = _assert_common_segments(bundle_path)
    cache = tlvs.get(TLV_CTRL_FLOW_CACHE)
    assert cache is not None, "expected a present-but-empty CtrlFlowCache segment, not an absent one"
    assert cache.length == 0, (
        f"a value-dependent op must not carry a control-flow cache snapshot (got {cache.length} B); "
        f"either the op stopped being value-dependent or the value-depend pack hook regressed"
    )
    assert TLV_SYMBOL_META in tlvs, "expected a SymbolMeta segment"

    out = _consume("value_depend_pa", bundle_path)
    assert "page-attention packed" in out and "page-attention alt" in out


@duration_estimate(28)
@pypto.options(pass_options={"enable_slice": True})
def test_bundle_standalone_delivery(bundle_path, tmp_path):
    """The self-contained delivery form: ONE .so + ONE .pyptokb + configs/, with no pypto install in reach.

    Skipped unless PYPTO_BUNDLE_STANDALONE_SO points at a built libtile_fwk_bundle_standalone.so, because
    ENABLE_TILE_FWK_BUNDLE_STANDALONE is OFF by default -- it recompiles ~195 upstream sources into its own
    objects and roughly doubles a full build, so ordinary CI has nothing to test here. A pipeline that does
    produce the delivery artifact gets this case for free by exporting the variable.

    Assembles the delivery layout from bundle/E2E_MANUAL.md section 5.1 rather than pointing straight at the
    build tree: configs/ has to sit beside the .so (GetPyptoLibPath uses dladdr on itself), and getting that
    layout wrong is the single most common way the delivery form fails at a customer.
    """
    sa_so = os.environ.get("PYPTO_BUNDLE_STANDALONE_SO", "")
    if not sa_so or not os.path.isfile(sa_so):
        pytest.skip(
            "PYPTO_BUNDLE_STANDALONE_SO is unset or missing; build with "
            "-DENABLE_TILE_FWK_BUNDLE_STANDALONE=ON and point it at libtile_fwk_bundle_standalone.so"
        )

    # The standalone .so consumes the very same bundle the plain form does -- reuse the static add case.
    _pack("static_add", bundle_path)
    _assert_common_segments(bundle_path)

    deliver = tmp_path / "deliver"
    (deliver / "configs").mkdir(parents=True)
    shutil.copy2(sa_so, deliver / "libtile_fwk_bundle_standalone.so")
    src_configs = os.path.join(_pypto_lib_dir(), "configs")
    assert os.path.isdir(src_configs), f"no configs/ in the installed pypto lib dir ({src_configs})"
    for entry in os.listdir(src_configs):
        src = os.path.join(src_configs, entry)
        if os.path.isfile(src):
            shutil.copy2(src, deliver / "configs" / entry)

    # Runs the smoke script, which must not import pypto -- hence a plain subprocess rather than the worker.
    cmd = [sys.executable, _STANDALONE_SMOKE, str(deliver / "libtile_fwk_bundle_standalone.so"), bundle_path]
    proc = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True, timeout=_CONSUME_TIMEOUT_S)
    if proc.returncode != 0:
        pytest.fail(
            f"standalone delivery smoke failed (rc={proc.returncode})\ncmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    print(proc.stdout)
    assert "no pypto shared objects mapped" in proc.stdout, "the smoke test did not reach its self-containment check"
