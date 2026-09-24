# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyzer: extract pipeline structure from serial kernel AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ._cross_core_scanner import CrossCoreSyncContext
from ._stage import is_pipeline_stage
from ._validate import validate_structure

# The one fixed ctx field the pipeline transform adds of its own accord — every other ctx
# field comes from a stage argument. It is named here because it is written by the
# transformer (as a plain scalar) and read back through the ctx slot, and the two halves
# live in different modules: this module decides the ctx layout, the transformer generates
# the assignments and the guards that read them. One name, so the halves cannot drift.
#
# It holds which task the data in this ctx slot belongs to. Auto-sync guards compare it
# against an edge's skew to tell whether the partner task exists.
_PL_TASK_ID_FIELD = "_pl_task_id"
# Marks a stage argument that is a whole struct: the ctx slot is passed in its place, so
# there is no single field to name (its fields are ctx fields under their own names).
_PL_STRUCT_ARG = "_pl_struct_arg"
# The validity flag. Prefixed like every framework field so a user struct field named
# `is_valid` cannot collide with it.
_PL_IS_VALID_FIELD = "_pl_is_valid"


@dataclass
class StageCall:
    """Info about a single stage call in the serial loop body."""

    func_name: str  # e.g. "compute_qk"
    section_kind: str  # "cube" or "vector"
    args: list  # list of ast.expr nodes (call arguments)
    delay: int  # derived from call order: 0, 1, 2, ...
    pre_stmts: list = field(default_factory=list)  # statements before stage call in same section
    post_stmts: list = field(default_factory=list)  # statements after stage call in same section
    # Every buffer access this stage makes, one entry per op in source order:
    # [(buffer, role, pipe), ...]. Per-op rather than collapsed to first/last pipe, so the
    # sync graph can see a local access sitting between two cross-core ones, and so every
    # pipe a buffer is touched on gets its own node (and hence its own sync).
    region_access: list = field(default_factory=list)


@dataclass
class PipelineInfo:
    """Extracted pipeline structure from the serial kernel."""

    stages: list[StageCall] = field(default_factory=list)
    ctx_fields: list[str] = field(default_factory=list)
    # {struct variable name: [field names]} for every pl.struct passed to a stage. Its
    # fields become ctx fields under their own names, so a stage body written against the
    # struct (`ri.ki`) reads the ctx slot unchanged — the slot IS a struct.
    struct_args: dict = field(default_factory=dict)
    # {variable name: ctx field name} for scalars. Normally identical, but a scalar whose
    # name collides with a struct field gets a prefixed field: the struct's field name is
    # fixed by the stage body, the scalar's is not, so the scalar yields.
    scalar_ctx_names: dict = field(default_factory=dict)
    # Maps: stage arg position -> (field_name, fill_expr) | None
    stage_arg_mapping: list[list] = field(default_factory=list)
    inner_loop_var: str = ""  # inner loop variable (e.g. "ki")
    inner_loop_range_end: ast.expr | None = None  # e.g. ast node for "skv_tiles"
    # Step of the pipeline loop, i.e. how far the loop variable moves per iteration.
    # None means an implicit 1. It matters whenever a guard has to reason about a task
    # some iterations away: the loop variable is in whatever unit the user chose, so
    # "n iterations later" is `var + n * step`, not `var + n`. The FA cases step by 1
    # and hide the distinction; the sparse kernel steps by TKV over element offsets.
    inner_loop_step: ast.expr | None = None
    # {variable name: IR type class name} from the parser probe (see probe_kernel_facts).
    # The authority on how a stage argument must reach a delayed stage; empty when the
    # caller did not run the probe, in which case everything falls back to pass-through.
    var_types: dict = field(default_factory=dict)
    # {name: section kind} for names bound inside a pl.section_*() block. Their ctx fill
    # must sit in that same section, since the other target cannot see them at all
    # (see _collect_var_sections).
    var_sections: dict = field(default_factory=dict)
    # The pipeline loop itself, so later passes can tell "inside the loop" from "outside".
    pipeline_loop: ast.For | None = None
    # {slot variable: (group name, slot count)} for slots chosen OUTSIDE the pipeline loop
    # and consumed inside it. The chosen index travels through ctx so a delayed stage gets
    # the slot from its own iteration, not whatever the variable was last rebound to.
    outer_slots: dict = field(default_factory=dict)
    # Cross-core sync context (buffers, memory, lifted ids)
    sync: CrossCoreSyncContext = field(default_factory=CrossCoreSyncContext)
    # Reference to closure_vars, so the transform chain need not thread it through.
    closure_vars: dict = field(default_factory=dict)


def analyze_pipeline(func_def: ast.FunctionDef, closure_vars: dict, var_types: dict | None = None) -> PipelineInfo:
    """Analyze a serial kernel function AST to extract pipeline structure.

    Looks for the innermost for-loop that contains stage calls inside
    section blocks, and extracts stage ordering, ctx fields, etc.

    Runs on the AST after dead constant branches are pruned (see
    _prune_const_branches), so it only ever sees plain unconditional stage calls.

    Args:
        func_def: The kernel function's AST node
        closure_vars: Closure variables (module-level constants, stage functions)

    Returns:
        PipelineInfo with all extracted information
    """
    info = PipelineInfo()
    info.closure_vars = closure_vars
    info.var_types = var_types or {}

    # Find all stage functions.
    stage_func_names = set()
    for name, val in closure_vars.items():
        if is_pipeline_stage(val):
            stage_func_names.add(name)

    # Walk the AST to find the main loop structure.
    _find_pipeline_loop(func_def.body, info, stage_func_names, closure_vars)

    # L3/L5: pipeline enabled but no usable stages found.
    if not info.stages:
        if not stage_func_names:
            raise ValueError(
                "pipeline: no @pl.pipeline.stage functions found, but pipeline=... was "
                "set on the kernel. Decorate your stage functions with @pl.pipeline.stage."
            )
        raise ValueError(
            "pipeline: no pipeline loop found — @pl.pipeline.stage functions exist but "
            "none are called inside a for-loop's `with pl.section_*()` blocks. The "
            "pipeline loop must contain stage calls wrapped in section blocks."
        )

    # Which section each name belongs to, so its ctx fill lands in the same one.
    info.var_sections = _collect_var_sections(func_def)

    # Structs passed to stages: their fields become ctx fields (see _collect_struct_args).
    info.struct_args = _collect_struct_args(func_def, info)

    # Every buffer declaration, scanned once and shared by everything that reads them.
    from ._cross_core_scanner import scan_tile_group_decls

    decls = scan_tile_group_decls(func_def)

    # Slots chosen outside the pipeline loop: their index travels through ctx.
    info.outer_slots = _collect_outer_slots(func_def, info, decls)

    # Derive ctx fields from stage arguments
    _derive_ctx_fields(info, closure_vars)

    # One gate for every check that needs only the parsed structure (see _validate).
    validate_structure(info, func_def, stage_func_names)

    # Scan cross-core buffer accesses for auto-sync
    _scan_cross_core(info, func_def, closure_vars, decls)

    # Producer/consumer and address-reuse checks live in validate_sync, which runs once
    # the sync graph exists — see _sync_graph.build_graph.

    return info


def _scan_cross_core(info: PipelineInfo, func_def: ast.FunctionDef, closure_vars: dict, decls: list):
    """Scan cross-core buffers + each stage's accesses, store into info.

    ``decls`` is the shared one-pass result from scan_tile_group_decls. The order of the
    scans below is the order their errors surface in, so it is deliberate: the cross-core
    scan raises on an unusable declaration, the rest only collect what they can resolve.
    """
    from ._cross_core_scanner import (
        detect_addr_overlaps,
        scan_all_buffer_memory,
        scan_all_tile_group_names,
        scan_buffer_addr_ranges,
        scan_buffer_mutex_ids,
        scan_cross_core_buffers,
        scan_kernel_slot_to_buffer,
        scan_stage_accesses,
        scan_tuple_fields,
    )

    cross_buffers, lifted_ids = scan_cross_core_buffers(decls, closure_vars)
    info.sync.buffers = cross_buffers
    info.sync.lifted_ids = lifted_ids
    if not cross_buffers:
        return

    all_mem = scan_all_buffer_memory(decls)

    # Detect address overlaps involving cross-core buffers (for auto-sync of
    # address-reused buffers). Local-local overlaps are ignored.
    info.sync.addr_ranges = scan_buffer_addr_ranges(decls, closure_vars)
    info.sync.mutex_ids = scan_buffer_mutex_ids(decls, closure_vars)
    info.sync.addr_overlaps = detect_addr_overlaps(info.sync.addr_ranges, set(cross_buffers.keys()))

    vf_func_defs = _collect_vf_func_defs(info, closure_vars)

    # Slots taken from cross-core buffers in the kernel body (pipeline loop), for
    # stages that receive a pre-taken slot instead of the buffer group itself.
    # Every declared group, cross-core or not: a local buffer's slot must resolve too, or the
    # pipe of an op that touches both kinds cannot be determined (see build_binding_map).
    group_names = scan_all_tile_group_names(decls)
    kernel_slot_to_buffer = scan_kernel_slot_to_buffer(func_def, group_names)
    # Members of an aggregate passed to a stage: the one hop that rejoins a tile to its
    # declared group when the kernel bundles groups with pl.make_tuple.
    tuple_fields = scan_tuple_fields(func_def)

    # All @stage function defs (name -> FunctionDef), looked up per stage below.
    stage_func_defs = {}
    for s in info.stages:
        sfd = _try_get_funcdef(closure_vars.get(s.func_name))
        if sfd is not None:
            stage_func_defs[s.func_name] = sfd
    # Buffers sharing a physical region with another buffer, computed once for the whole
    # kernel. Not per stage: a region's members may be touched by different stages, and a
    # per-stage view would miss those.
    region_members = {name for pair in info.sync.addr_overlaps for name in pair}
    for stage in info.stages:
        fd = stage_func_defs.get(stage.func_name)
        if fd is None:
            continue
        stage.region_access = []
        scan_stage_accesses(
            fd,
            cross_buffers,
            vf_func_defs,
            all_mem,
            region_members,
            stage.region_access,
            call_args=stage.args,
            kernel_slot_map=kernel_slot_to_buffer,
            tuple_fields=tuple_fields,
            group_names=group_names,
        )

    # Address-reuse sync is not built here. Which edges need it, and which ids they get,
    # both follow from the sync graph (see _sync_graph.allocate_reuse_ids) — deriving that
    # set here as well is what let allocation and emission disagree.


def _collect_used_event_ids(info: PipelineInfo) -> set:
    """Collect all event ids already used by cross-core buffers' fwd/bwd ids.

    The real literal id lists live in info.sync.lifted_ids (fwd/bwd nodes on the
    buffers were replaced with variable names)."""
    used = set()
    for _var, node in info.sync.lifted_ids:
        if isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, int):
                    used.add(e.value)
    return used


def _allocate_overlap_event_ids(info: PipelineInfo, reverse_sync_pairs: list):
    """Allocate event ids (0-15) for each reverse sync, filling each pair's
    'event_ids' key with a list of ints.

    Strategy:
      1. Each pair needs slot_count ids by default.
      2. Allocate from unused ids (0-15) in order.
      3. If not enough: degrade pairs (share 1 id across all slots) from smallest
         slot_count up, logging a warning each time.
      4. If all degraded to 1 id each and still not enough: raise.
    """
    if not reverse_sync_pairs:
        return

    from ._cross_core_scanner import _MAX_EVENT_ID

    max_event_id = _MAX_EVENT_ID + 1
    used = _collect_used_event_ids(info)
    free = [i for i in range(max_event_id) if i not in used]

    # How many ids each pair wants (default: slot_count). May be degraded to 1.
    wants = [p["slot_count"] for p in reverse_sync_pairs]

    # Degrade (smallest slot_count first) until total demand fits in free ids.
    # Sort indices by slot_count ascending for degradation order.
    order = sorted(range(len(reverse_sync_pairs)), key=lambda i: wants[i])
    deg_ptr = 0
    while sum(wants) > len(free) and deg_ptr < len(order):
        idx = order[deg_ptr]
        if wants[idx] > 1:
            import logging

            logging.warning(
                f"pipeline: not enough event ids for address-overlap reverse sync; "
                f"degrading pair ({reverse_sync_pairs[idx]['first_stage']} -> "
                f"{reverse_sync_pairs[idx]['last_stage']}) from {wants[idx]} ids to 1 "
                f"(slots will serialize, correctness preserved)."
            )
            wants[idx] = 1
        deg_ptr += 1

    if sum(wants) > len(free):
        raise ValueError(
            f"pipeline: not enough free event ids (0-{max_event_id - 1}) for "
            f"address-overlap reverse syncs. Need {sum(wants)}, have {len(free)} free "
            f"(used: {sorted(used)}). Reduce cross-core buffer id usage or overlaps."
        )

    # Allocate from free ids in order
    cursor = 0
    for p, n in zip(reverse_sync_pairs, wants):
        p["event_ids"] = free[cursor:cursor + n]
        cursor += n


def _collect_vf_func_defs(info: PipelineInfo, closure_vars: dict) -> dict[str, ast.FunctionDef]:
    """Collect VF helper func defs.

    Includes both @pl.vector_function decorated functions AND plain wrapper
    functions that transitively call a @pl.vector_function (any depth of
    indirection). This ensures the cross-core scanner can see through chains
    like ``stage a → b → c → @pl.vector_function d``.
    """
    stage_names = {s.func_name for s in info.stages}
    vf_func_defs: dict[str, ast.FunctionDef] = {}
    all_func_defs: dict[str, ast.FunctionDef] = {}
    for name, val in closure_vars.items():
        if callable(val) and name not in stage_names and not is_pipeline_stage(val):
            fd = _try_get_funcdef(val)
            if fd is not None:
                all_func_defs[name] = fd
                if _is_vf_function(fd):
                    vf_func_defs[name] = fd
    # Iteratively include plain functions that call any already-known VF function,
    # repeating until no new additions (transitive closure).
    changed = True
    while changed:
        changed = False
        for name, fd in all_func_defs.items():
            if name in vf_func_defs:
                continue
            if _calls_any_vf_function(fd, vf_func_defs):
                vf_func_defs[name] = fd
                changed = True
    return vf_func_defs


def _try_get_funcdef(fn) -> ast.FunctionDef | None:
    """Get the ast.FunctionDef for a Python function object, or None.

    Line numbers are shifted to the ones in the real file. Re-parsing a source snippet
    numbers it from 1, and every diagnostic that names a node inside a stage body reports
    whatever this returns — so without the shift those messages point at a line the user
    cannot find.
    """
    import inspect
    import textwrap

    if fn is None:
        return None
    try:
        lines, start_lineno = inspect.getsourcelines(fn)
        mod = ast.parse(textwrap.dedent("".join(lines)))
        # getsourcelines is 1-based and so is the fresh parse, hence the -1.
        ast.increment_lineno(mod, start_lineno - 1)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                return node
    except (OSError, TypeError, SyntaxError):
        return None
    return None


def _record_pipeline_loop_info(stmt: ast.For, info: PipelineInfo) -> None:
    """Record loop metadata after the pipeline loop has been found."""
    # L7: loop variable must be a simple Name
    if not isinstance(stmt.target, ast.Name):
        raise ValueError(
            "pipeline: the pipeline loop variable must be a simple name "
            "(e.g. `for ki in pl.range(...)`); tuple unpacking is not supported."
        )
    info.pipeline_loop = stmt
    info.inner_loop_var = stmt.target.id
    # L6: loop must be pl.range(...) with extractable end bound
    info.inner_loop_range_end = None
    info.inner_loop_step = None
    if isinstance(stmt.iter, ast.Call):
        args = stmt.iter.args
        if len(args) >= 2:
            info.inner_loop_range_end = args[1]
        elif len(args) == 1:
            info.inner_loop_range_end = args[0]
        if len(args) >= 3:
            info.inner_loop_step = args[2]
    if info.inner_loop_range_end is None:
        raise ValueError(
            f"pipeline: pipeline loop `for {info.inner_loop_var} in ...` must iterate "
            f"over pl.range(start, end[, step]) so the end bound can be extracted "
            f"for the is_valid guard; got an unsupported loop iterable."
        )



def _nested_search_body(stmt: ast.stmt) -> list[ast.stmt] | None:
    """Return the nested body that can contain a pipeline loop."""
    if isinstance(stmt, (ast.For, ast.With)):
        return stmt.body
    return None


def _find_pipeline_loop(
    stmts: list[ast.stmt], info: PipelineInfo, stage_func_names: set, closure_vars: dict
):
    """Recursively find the innermost for-loop containing stage calls."""
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            stages_found = _extract_stages_from_loop(stmt, info, stage_func_names)
            if stages_found:
                _record_pipeline_loop_info(stmt, info)
                return True
        nested_body = _nested_search_body(stmt)
        if nested_body is not None and _find_pipeline_loop(
            nested_body, info, stage_func_names, closure_vars
        ):
            return True
    return False


def _split_stage_section_body(body: list[ast.stmt], stage_func_names: set):
    """Find one stage call in a section body and split pre/post statements."""
    pre_stmts = []
    post_stmts = []
    stage_call = None
    stage_func_name = None

    for body_stmt in body:
        if stage_call is None:
            if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call):
                func_name = _get_call_func_name(body_stmt.value)
                if func_name in stage_func_names:
                    stage_call = body_stmt.value
                    stage_func_name = func_name
                    continue
            pre_stmts.append(body_stmt)
        else:
            post_stmts.append(body_stmt)
    return stage_func_name, stage_call, pre_stmts, post_stmts


def _extract_stages_from_loop(for_stmt: ast.For, info: PipelineInfo, stage_func_names: set) -> bool:
    """Extract stage calls from a for-loop body (expecting interleaved sections)."""
    found_any = False

    for stmt in for_stmt.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fn = _get_call_func_name(stmt.value)
            if fn in stage_func_names:
                info.stages.append(
                    StageCall(
                        func_name=fn,
                        section_kind="",
                        args=list(stmt.value.args),
                        delay=0,
                    )
                )
                found_any = True
                continue
        if isinstance(stmt, ast.With):
            section_kind = _get_section_kind(stmt)
            if section_kind is None:
                _check_unsupported_section(stmt, stage_func_names)
                continue
            stage_call_info = _extract_stage_from_section(stmt, section_kind, stage_func_names)
            if stage_call_info is not None:
                info.stages.append(stage_call_info)
                found_any = True

    return found_any


def _check_unsupported_section(stmt: ast.With, stage_func_names: set):
    """C1: raise if an unsupported section type contains a stage call."""
    for body_stmt in stmt.body:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and _get_call_func_name(body_stmt.value) in stage_func_names
        ):
            raise ValueError(
                f"pipeline: stage call "
                f"'{_get_call_func_name(body_stmt.value)}' is inside an "
                f"unsupported `with` block. Stage calls must be wrapped in "
                f"`with pl.section_cube()` or `with pl.section_vector()`."
            )


def _extract_stage_from_section(stmt: ast.With, section_kind: str, stage_func_names: set) -> StageCall | None:
    """Extract a stage call from a section block, with L2 validation."""
    stage_func_name, stage_call, pre_stmts, post_stmts = _split_stage_section_body(stmt.body, stage_func_names)
    if stage_call is None:
        return None
    # L2: check for a second stage call in same section
    for body_stmt in post_stmts:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and _get_call_func_name(body_stmt.value) in stage_func_names
        ):
            raise ValueError(
                f"pipeline: section block contains multiple stage calls "
                f"('{stage_func_name}' and "
                f"'{_get_call_func_name(body_stmt.value)}'). Each "
                f"`with pl.section_*()` block must contain exactly one stage call."
            )
    return StageCall(
        func_name=stage_func_name,
        section_kind=section_kind,
        args=stage_call.args,
        delay=0,
        pre_stmts=pre_stmts,
        post_stmts=post_stmts,
    )


def _get_section_kind(with_stmt: ast.With) -> str | None:
    """Extract section kind ('cube' or 'vector') from a with statement."""
    if not with_stmt.items:
        return None
    ctx = with_stmt.items[0].context_expr
    if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
        if ctx.func.attr == "section_cube":
            return "cube"
        elif ctx.func.attr == "section_vector":
            return "vector"
    return None


def _is_vf_function(func_def: ast.FunctionDef) -> bool:
    """Check if a function is a VF helper (``@pl.vector_function`` decorated)."""
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "vector_function":
            return True
        if isinstance(dec, ast.Name) and dec.id == "vector_function":
            return True
    return False


def _calls_any_vf_function(func_def: ast.FunctionDef, vf_func_defs: dict[str, ast.FunctionDef]) -> bool:
    """Check if a function body calls any known VF function (one level of indirection)."""
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in vf_func_defs:
                return True
    return False


def _get_call_func_name(call: ast.Call) -> str:
    """Get the function name from a Call node."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""







def _is_ctx_scalar(name: str, info: PipelineInfo) -> bool:
    """True if ``name`` holds a scalar the ctx slot can carry.

    Every scalar a stage takes is carried, whether or not it changes across iterations:
    snapshotting one that never changes costs a field and stores the same value each beat,
    while trying to work out which ones change means reasoning about how each value was
    produced — and getting that wrong drops a value silently. One field per scalar is the
    cheaper mistake.

    Types come from the parser probe (info.var_types). Without it — a caller that skipped
    the probe — nothing is treated as a ctx scalar and every argument passes through, which
    is what the pipeline did before types were available.

    A scalar bound inside a section is still carried; its fill is simply emitted inside that
    section (see _build_ctx_field_fills), which is where the name exists.
    """
    return info.var_types.get(name) == "ScalarType"


def _collect_var_sections(func_def: ast.FunctionDef) -> dict:
    """{name: section kind} for names bound inside a ``pl.section_*()`` block.

    A name bound inside a section exists only for that target: the other target's parse
    skips the whole block. So a ctx field fed from such a name has to be filled inside the
    same section, or the other target would fill it from whatever the name happens to mean
    there — usually a stale initial value, silently overwriting the real one, since both
    targets write the same ctx memory.

    Names absent from this map come from outside any section and are visible to both.
    """
    sections: dict = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.With):
            continue
        kind = _get_section_kind(node)
        if kind is None:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                sections.setdefault(inner.id, kind)
    return sections


def _is_scalar_expr(node: ast.expr, info: PipelineInfo) -> bool:
    """True if this expression is scalar-valued: arithmetic over scalars and constants."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return _is_ctx_scalar(node.id, info)
    if isinstance(node, ast.BinOp):
        return _is_scalar_expr(node.left, info) and _is_scalar_expr(node.right, info)
    if isinstance(node, ast.UnaryOp):
        return _is_scalar_expr(node.operand, info)
    return False


def _slot_index_field(group: str) -> str:
    """Ctx field (and variable) name holding a group's current slot index."""
    return f"_pl_idx_{group}"


def _slot_pick_of(node: ast.expr, groups: set) -> tuple[str, str] | None:
    """``(group, kind)`` if ``node`` selects a slot of a known group, else None.

    ``kind`` is the accessor name for a method call (only ``next`` advances the group's
    cursor, so the others must not be rewritten as an advance) or ``"index"`` for ``g[i]``.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("next", "current", "previous"):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in groups:
                return base.id, node.func.attr
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id in groups:
            return node.value.id, "index"
    return None


def _collect_outer_slots(func_def: ast.FunctionDef, info: PipelineInfo, decls: list) -> dict:
    """{slot variable: (group, slot count)} for slots picked outside the pipeline loop.

    A slot picked inside the loop rotates once per beat and is protected by the sync graph;
    one picked outside advances only per outer iteration, while every beat reads it. A
    delayed stage handling an older beat then sees whatever the variable was last rebound
    to — the wrong slot as soon as the outer iteration turns over. Carrying the INDEX in
    ctx fixes that: each beat snapshots the index it used, and every consumer re-selects
    the slot with its own beat's index.

    Only slots a stage actually consumes are collected; pure bookkeeping slots are left
    alone. Values are ``(group, kind, slot count)``; the rewrite needs the count as its
    modulus, so a group whose mutex_ids do not resolve statically is reported as an error
    rather than silently skipped — skipping it would leave the delayed stage reading the
    wrong slot with no diagnostic.
    """
    # Read off the shared declaration scan; info.sync is not filled until _scan_cross_core.
    from ._cross_core_scanner import scan_all_tile_group_names, scan_buffer_mutex_ids

    groups = scan_all_tile_group_names(decls)
    if not groups or info.pipeline_loop is None:
        return {}
    slot_counts = {name: len(ids) for name, ids in scan_buffer_mutex_ids(decls, info.closure_vars).items()}

    consumed = {
        arg.id for stage in info.stages for arg in stage.args if isinstance(arg, ast.Name)
    }
    inside = {id(node) for node in ast.walk(info.pipeline_loop)}

    result: dict = {}
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in consumed:
            continue
        if id(node) in inside:
            continue  # picked per beat: the sync graph already covers it
        pick = _slot_pick_of(node.value, groups)
        if pick is None:
            continue
        group, kind = pick
        if group not in slot_counts:
            raise ValueError(
                f"pipeline: slot '{target.id}' is taken from tile group '{group}' outside "
                f"the pipeline loop, but '{group}'s mutex_ids could not be resolved "
                f"statically, so the number of slots is unknown. The transform needs it to "
                f"give each stage the slot from its own iteration."
            )
        result[target.id] = (group, kind, slot_counts[group])
    return result


def _collect_struct_args(func_def: ast.FunctionDef, info: PipelineInfo) -> dict:
    """{name: [field names]} for every ``pl.struct`` a stage is called with.

    A struct's fields are carried in the ctx slot under their own names, and the slot is
    passed to the stage in place of the struct. Nothing about the stage body changes: the
    slot is itself a struct, so ``ri.ki`` resolves against the delayed slot's ``ki``. This
    is what lets the framework stay ignorant of HOW the user updates the struct — in a
    branch, in a helper, over several statements — since only the field values are read,
    at a point where all of that has already run.

    ``pl.struct_array`` is rejected: an array of contexts is exactly what this transform
    generates, and a user-managed second one cannot be kept in step with it.
    """
    stage_arg_names = {
        arg.id for stage in info.stages for arg in stage.args if isinstance(arg, ast.Name)
    }
    if not stage_arg_names:
        return {}

    # Names bound inside the pipeline loop cannot be a stable struct: they would be
    # rebound every beat, which is the user-managed ring buffer case above.
    loop_bound: set = set()
    for stage in info.stages:
        for stmt in stage.pre_stmts + stage.post_stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    loop_bound.add(node.id)

    result: dict = {}
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in stage_arg_names:
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr == "struct_array":
            raise ValueError(
                f"pipeline: '{target.id}' comes from pl.struct_array() and is passed to a "
                f"stage. Use pl.struct() instead: the pipeline already gives every stage "
                f"the values from its own iteration, so a hand-rolled array of contexts is "
                f"both unnecessary and impossible to keep in step with it."
            )
        if call.func.attr != "struct":
            continue
        if target.id in loop_bound:
            raise ValueError(
                f"pipeline: struct '{target.id}' is bound inside the pipeline loop. Declare "
                f"it outside the loop and update its fields inside — the pipeline snapshots "
                f"the fields each iteration."
            )
        result[target.id] = [kw.arg for kw in call.keywords if kw.arg]
    return result


def _derive_ctx_fields(info: PipelineInfo, closure_vars: dict):
    """Decide how each stage argument reaches its stage, by the argument's TYPE.

    The type is what matters, not whether the value changes across iterations: a scalar can
    ride in the ctx slot, so it always does; a Tile cannot (the slot holds scalars) and
    travels as the index that selects it; a struct travels as its fields; a tensor or tile
    group is a handle to one object and does not need to travel at all.

    Asking "does this change?" instead means reasoning about how each value was produced —
    through a branch, a helper call, an alias — and a wrong answer there silently feeds a
    delayed stage the current beat's value. Asking "what is this?" has an exact answer from
    the parser probe (info.var_types), and one extra ctx field for a value that never
    changes is the cheaper mistake.

    stage_arg_mapping stores per-stage, per-arg: None (fixed) or (field_name, fill_expr).
    - field_name: the ctx struct field name (e.g. "ki" or "_pl_arg_compute_qk_1")
    - fill_expr: the expression the ctx field is filled FROM, evaluated at the snapshot
      point inside the loop body — after the user's statements for this beat have run.
      For a plain Name (scalar or struct) that is the variable itself; the point of
      reading the variable rather than re-deriving its value is that the framework then
      needs no understanding of HOW the user computed it (a branch, a helper function,
      several statements — all work the same).
    """
    struct_fields = {f for fields in info.struct_args.values() for f in fields}
    # A scalar whose name collides with a struct field is renamed. The struct's field names
    # are dictated by the stage bodies that read them (`ri.ki`), so they cannot move; a
    # scalar's ctx field name is internal to the generated code, so it can.
    info.scalar_ctx_names = {}

    def scalar_field(name: str) -> str:
        if name not in info.scalar_ctx_names:
            info.scalar_ctx_names[name] = f"_pl_{name}" if name in struct_fields else name
        return info.scalar_ctx_names[name]

    ctx_field_set = set()
    info.stage_arg_mapping = []

    for stage in info.stages:
        arg_map = []
        for argpos, arg in enumerate(stage.args):
            if isinstance(arg, ast.Name):
                if arg.id in info.struct_args:
                    # The whole ctx slot stands in for the struct: same field names, so the
                    # stage body needs no rewriting. Fields are snapshotted individually.
                    ctx_field_set.update(info.struct_args[arg.id])
                    arg_map.append((_PL_STRUCT_ARG, ast.Name(id=arg.id, ctx=ast.Load())))
                    continue
                if arg.id in info.outer_slots:
                    # Slot chosen outside the loop: ctx carries the INDEX, and the stage
                    # re-selects the slot with its own beat's index (see _build_stage_args).
                    group = info.outer_slots[arg.id][0]
                    fname = _slot_index_field(group)
                    ctx_field_set.add(fname)
                    arg_map.append((fname, ast.Name(id=fname, ctx=ast.Load())))
                    continue
                if _is_ctx_scalar(arg.id, info):
                    fname = scalar_field(arg.id)
                    ctx_field_set.add(fname)
                    arg_map.append((fname, ast.Name(id=arg.id, ctx=ast.Load())))
                else:
                    arg_map.append(None)
                continue
            # An expression: snapshot whatever it evaluates to, under a name of its own.
            # Scalar-valued expressions are the only ones that can travel in the ctx slot,
            # and a non-scalar one cannot be turned into a slot index either, since the
            # index it was built from is not recoverable from the expression.
            if _is_scalar_expr(arg, info):
                field_name = f"_pl_arg_{stage.func_name}_{argpos}"
                ctx_field_set.add(field_name)
                arg_map.append((field_name, arg))
            else:
                arg_map.append(None)
        info.stage_arg_mapping.append(arg_map)

    # Framework fields all carry the _pl_ prefix so they can never collide with a field
    # name coming from a user struct. The two trailing ones are the transform's own
    # bookkeeping rather than stage data — see their definitions at the top of this module
    # for why each has to travel with the task instead of being read live.
    info.ctx_fields = [_PL_IS_VALID_FIELD] + sorted(ctx_field_set) + [_PL_TASK_ID_FIELD]


def probe_kernel_facts(kernel_def, bound_signature=None) -> tuple[dict, dict]:
    """Probe-parse the kernel once per target and return ``(if_const_map, var_types)``.

    Both come from the real parser rather than being re-derived here: it already folds
    constant conditions and already knows every variable's IR type, and duplicating either
    is how the two drift apart.

    ``var_types`` maps a variable name to its IR type's class name (``ScalarType``,
    ``TileType``, ``TensorType``, ``TupleType``, ...). It decides how a stage argument
    reaches a delayed stage: a scalar can travel in the ctx slot, a Tile cannot (the slot's
    fields are scalars) and instead travels as the index that selects it, and a tensor or
    tile group does not need to travel at all. Deriving this from the AST would mean
    guessing — a helper call's result has no syntactic clue, yet the parser inlines it and
    knows the answer exactly.
    """
    from pypto.pypto_impl import ir
    from pypto_pro.language.parser._ast_parser import ASTParser

    if_const: dict = {}
    var_types: dict = {}
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        parser = ASTParser(
            kernel_def._source_file,
            kernel_def._source_lines,
            target,
            kernel_def._line_offset,
            kernel_def._col_offset,
            strict_ssa=kernel_def._strict_ssa,
            closure_vars=kernel_def._closure_vars,
            auto_mutex=kernel_def._auto_mutex,
            debug_info=ir.IRDebugInfo(),
            tilingkey_consts=kernel_def._tilingkey_consts,
            datatype_consts=kernel_def._datatype_consts,
            bound_signature=bound_signature,
            void_return_only=True,
            void_return_context="@pl.jit/@pl.kernel",
            allow_early_return=True,
        )
        parser.collect_if_const = True
        # define_var is the single point where every name gets bound, so wrapping it here
        # harvests types without touching the parser itself.
        scope = parser.scope_manager
        original_define = scope.define_var

        def _define(name, value, allow_redef=False, span=None, _orig=original_define):
            var_type = getattr(value, "type", None)
            if var_type is not None:
                var_types.setdefault(name, type(var_type).__name__)
            return _orig(name, value, allow_redef=allow_redef, span=span)

        scope.define_var = _define
        parser.parse_function(kernel_def._func_def, func_type=kernel_def._func_type)
        # Merge, preferring a CONSTANT verdict: a compile-time-constant condition is
        # target-independent, so if EITHER the Cube or Vector parse folded it to a
        # constant, keep that. Only when no parse found it constant is it dynamic.
        # (A single target may skip a section — the other target's section holds the
        # real fold; blind update() would let a later (False, None) clobber it.)
        for k, v in parser.if_const_map.items():
            existing = if_const.get(k)
            if existing is None or (not existing[0] and v[0]):
                if_const[k] = v
    return if_const, var_types
