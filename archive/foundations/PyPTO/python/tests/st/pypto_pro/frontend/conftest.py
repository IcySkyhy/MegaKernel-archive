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

"""Frontend-test conftest with optional parallel pre-compilation.

Normally each frontend test compiles its ``@pl.jit`` kernel lazily on first launch, so the
whole suite pays for kernel compilation (codegen + bisheng ``.so`` build) serially, interleaved
with device launches. The compile step is host/CPU bound and independent of the runtime tensor
values, so it parallelizes well; the launch step must stay serial (one NPU).

Set ``PARALLEL_COMPILE=1`` to split the run into three phases:

  1. discovery — run every test with kernel launches stubbed out, recording each
     ``(kernel, tilingkey, datatype)`` combo;
  2. compile   — compile every recorded specialization in parallel across a thread pool;
     results land in each kernel's in-process cache;
  3. launch    — run the tests normally; every launch now hits the warm cache and only executes
     on device.

Anything the discovery pass misses (e.g. a kernel reached only after an earlier assertion, or a
non-``_TileJitKernel`` path) simply falls back to lazy compilation in phase 3, so behavior is
always correct — parallel compilation is a pure speed-up.

Tunables:
  PARALLEL_COMPILE       — ``1`` to enable the three-phase flow (default: off).
  PARALLEL_COMPILE_JOBS  — thread-pool size (default: min(#combos, os.cpu_count())).
"""

import logging
import os


def _parallel_compile_enabled() -> bool:
    return os.environ.get("PARALLEL_COMPILE") == "1"


def pytest_runtestloop(session):
    """Three-phase loop (discover → parallel compile → serial launch) when enabled.

    Returning ``None`` when disabled defers to pytest's default runtestloop. Any failure in the
    discover/compile phases is swallowed so the run degrades gracefully to lazy compilation.
    """
    if not _parallel_compile_enabled():
        return None  # default behavior

    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            f"{session.testsfailed} error{'s' if session.testsfailed != 1 else ''} during collection"
        )
    if session.config.option.collectonly:
        return None  # let the default loop handle --collect-only

    from concurrent.futures import ThreadPoolExecutor
    import importlib

    from _pytest.runner import runtestprotocol

    # NB: ``pypto_pro.runtime`` re-exports the ``jit`` *function*, which shadows the submodule
    # under ``from pypto_pro.runtime import jit``; import the module object explicitly.
    jit = importlib.import_module("pypto_pro.runtime.jit")

    tr = session.config.pluginmanager.get_plugin("terminalreporter")

    def info(msg):
        if tr is not None:
            tr.write_line(f"[PARALLEL_COMPILE] {msg}")
        else:  # pragma: no cover - only when terminal reporter is absent
            logging.info("[PARALLEL_COMPILE] %s", msg)

    items = session.items

    # ---- Phase 1: discovery -- record (kernel, tilingkey, datatype) combos, launches stubbed.
    records = []  # ordered, de-duplicated list of (kernel, concrete_key, dtype_key)
    seen = set()
    placeholder = jit.CompiledKernel(lib_path="", param_specs=[])
    tile_jit_kernel_cls = getattr(jit, "_TileJitKernel")
    orig_ensure_compiled = getattr(tile_jit_kernel_cls, "_ensure_compiled")
    orig_launch = getattr(jit, "_launch")

    def key_id(key):
        if key is None:
            return None
        return tuple(sorted((name, str(value)) for name, value in key.items()))

    def discover_ensure_compiled(self, args=None, concrete_key=None, dtype_key=None):
        # Record the specialization reached by this test without compiling during discovery.
        # Signature mirrors the real _ensure_compiled(args, concrete_key, dtype_key): `args`
        # (the runtime call tuple) carries the static-shape signature for non-tilingkey kernels,
        # so it must be captured and replayed at compile time.
        dedup = (id(self), key_id(concrete_key), key_id(dtype_key))
        if dedup not in seen:
            seen.add(dedup)
            records.append((self, args, concrete_key, dtype_key))
        return placeholder

    def noop_launch(*_args, **_kwargs):
        return None

    info(f"phase 1/3: discovering kernels across {len(items)} tests...")
    try:
        setattr(tile_jit_kernel_cls, "_ensure_compiled", discover_ensure_compiled)
        setattr(jit, "_launch", noop_launch)
        for i, item in enumerate(items):
            nextitem = items[i + 1] if i + 1 < len(items) else None
            try:
                # log=False: run setup/call/teardown for fixture/param handling without emitting
                # reports, so this phase does not affect the pass/fail tally.
                runtestprotocol(item, nextitem=nextitem, log=False)
            except Exception:
                pass  # partial discovery is fine; missed kernels compile lazily in phase 3
    finally:
        setattr(tile_jit_kernel_cls, "_ensure_compiled", orig_ensure_compiled)
        setattr(jit, "_launch", orig_launch)

    # ---- Phase 2: compile every recorded kernel in parallel.
    if records:
        jobs = int(os.environ.get("PARALLEL_COMPILE_JOBS") or 0)
        if jobs <= 0:
            jobs = min(len(records), os.cpu_count() or 4)
        info(f"phase 2/3: compiling {len(records)} kernel(s) with {jobs} worker(s)...")

        def build(record):
            kernel, args, key, dtype_key = record
            try:
                # Real _ensure_compiled performs codegen + compile once and populates the
                # kernel cache for the serial launch phase.
                orig_ensure_compiled(kernel, args, concrete_key=key, dtype_key=dtype_key)
            except Exception:  # noqa: BLE001 - phase 3 owns test error reporting
                # Pre-compilation is best-effort. The normal test phase retries the kernel
                # and either handles an expected error or reports an unexpected one.
                pass

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            list(pool.map(build, records))
    else:
        info("phase 2/3: no JIT kernels discovered; nothing to pre-compile.")

    # ---- Phase 3: launch tests one by one (default protocol, warm compile cache).
    info(f"phase 3/3: launching {len(items)} tests...")
    for i, item in enumerate(items):
        nextitem = items[i + 1] if i + 1 < len(items) else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)
    return True
