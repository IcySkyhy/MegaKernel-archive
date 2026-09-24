# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-core access scanner for preload pipeline auto-sync.

Pure-AST analysis. Scans the kernel body for cross-core NBuffer declarations
(those configured with cross_core_forward_id / cross_core_backward_id), then
scans each @stage function body to determine, for each cross-core buffer it
touches, the access role (W/R) and the pipe of the op that does the access.

The result drives automatic wait/set_cross_core insertion at stage boundaries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from pypto.pypto_impl.ir import MemorySpace
from pypto_pro.language.parser._op_pipeline import (
    _BLOCK_OP_TILE_ROLES,
    get_move_pipe,
    get_op_pipe,
    get_store_pipe,
)

# Highest usable cross-core event id. A hardware limit: ids run 0..15, and an
# out-of-range id only fails once the kernel runs on device.
_MAX_EVENT_ID = 15

# MemorySpace attribute name (as written in pl.MemorySpace.<X>) -> enum value
_MEMORY_NAMES = {
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Acc": MemorySpace.Acc,
    "ScaleLeft": MemorySpace.ScaleLeft,
    "ScaleRight": MemorySpace.ScaleRight,
}


def _extract_memory_from_make_tile_group(call: ast.Call) -> MemorySpace | None:
    """From a pl.make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X), ...)
    call, extract the MemorySpace X. Returns None if not resolvable."""
    type_node = None
    for kw in call.keywords:
        if kw.arg == "type":
            type_node = kw.value
            break
    if not (isinstance(type_node, ast.Call)):
        return None
    for kw in type_node.keywords:
        if kw.arg == "target_memory":
            v = kw.value
            if isinstance(v, ast.Attribute) and v.attr in _MEMORY_NAMES:
                return _MEMORY_NAMES[v.attr]
    return None


def _is_make_tile_group(call: ast.Call) -> bool:
    """True if call is pl.make_tile_group(...) (or bare make_tile_group(...))."""
    return _get_ctor_name(call) == "make_tile_group"


@dataclass
class CrossCoreBuffer:
    """A cross-core NBuffer's configuration (from kernel-body declaration)."""

    fwd_ids_node: ast.expr | None  # AST node for cross_core_forward_id value (e.g. a Name)
    bwd_ids_node: ast.expr | None  # AST node for cross_core_backward_id value
    fwd_slot_count: int  # len(forward id tuple) - for % N indexing
    bwd_slot_count: int  # len(backward id tuple)


@dataclass
class CrossCoreSyncContext:
    """Buffer declarations and memory layout the sync graph is built from.

    Populated by _scan_cross_core() during analysis.
    """

    # Buffer declarations: name -> CrossCoreBuffer (only cross-core buffers with fwd/bwd ids)
    buffers: dict = field(default_factory=dict)
    # Lifted literal fwd_ids/bwd_ids: [(var_name, ast_literal)] to declare as variables
    lifted_ids: list = field(default_factory=list)
    # Per-buffer slot address ranges: name -> (memory, [(start, end), ...] per slot).
    # Covers ALL buffers (cross-core + local) for address-overlap detection.
    addr_ranges: dict = field(default_factory=dict)
    # buffer name -> tuple of mutex ids. Used to check that co-located buffers hold the
    # same locks (a mutex locks the address, not the variable).
    mutex_ids: dict = field(default_factory=dict)
    # Address-overlapping buffer pairs: [(buf_a, buf_b), ...] (same memory, ranges intersect)
    addr_overlaps: list = field(default_factory=list)
    # Address-reuse sync is derived from the graph on demand, not stored here — see
    # _sync_graph.allocate_reuse_ids.


@dataclass
class TileGroupDecl:
    """One ``x = pl.make_tile_group(...)`` statement, as written.

    Purely syntactic: the keyword nodes are handed over unevaluated, and nothing here is
    validated. Every scan below reads the fields it needs from this and decides for itself
    what to do when a value will not resolve — some skip the buffer, some raise. Keeping
    those decisions in the scans (and their original call order) is what makes the single
    pass a refactor rather than a change in which error a user sees first.
    """

    names: list[str]  # every variable this statement binds (targets that are plain Names)
    call: ast.Call  # the make_tile_group call itself
    kwargs: dict  # {keyword name: unevaluated AST node}


def scan_tile_group_decls(kernel_func_def: ast.FunctionDef) -> list[TileGroupDecl]:
    """Every ``pl.make_tile_group(...)`` declaration in the kernel body, in AST order.

    The one pass over the kernel body that all buffer-declaration scans share. They used to
    walk it once each, repeating the same "is this an Assign of a make_tile_group call, and
    what does it bind" prologue and re-evaluating the same keywords several times over.
    """
    decls: list[TileGroupDecl] = []
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not _is_make_tile_group(call):
            continue
        decls.append(
            TileGroupDecl(
                names=[t.id for t in node.targets if isinstance(t, ast.Name)],
                call=call,
                kwargs={kw.arg: kw.value for kw in call.keywords if kw.arg},
            )
        )
    return decls


def scan_all_tile_group_names(decls: list[TileGroupDecl]) -> set[str]:
    """Every variable bound directly to a ``pl.make_tile_group(...)`` result.

    Deliberately independent of mutex_ids and cross-core ids: a group whose ids are not
    statically resolvable is still a group, and callers that only need to recognise slot
    selection (``g.next()``, ``g[i]``) must not miss it. Aliases are not followed, so
    ``alias = g`` leaves ``alias`` unrecognised.
    """
    return {name for decl in decls for name in decl.names}


def scan_all_buffer_memory(decls: list[TileGroupDecl]) -> dict[str, MemorySpace]:
    """Return every make_tile_group declaration's memory by buffer variable name."""
    result: dict[str, MemorySpace] = {}
    for decl in decls:
        mem = _extract_memory_from_make_tile_group(decl.call)
        if mem is None:
            continue
        for name in decl.names:
            result[name] = mem
    return result


def _eval_const(node: ast.expr, closure_vars: dict):
    """Evaluate a constant expression node using closure_vars as namespace.
    Returns the value, or None if it cannot be evaluated (e.g. references a
    runtime variable)."""
    try:
        code = compile(ast.Expression(body=node), "<addr>", "eval")
        return eval(code, {"__builtins__": {}}, dict(closure_vars))
    except Exception:
        return None


def _tile_type_slot_size(type_call: ast.Call, closure_vars: dict) -> int | None:
    """Compute per-slot byte size from a pl.TileType(shape=..., dtype=...) call.
    Returns None if shape/dtype cannot be resolved.

    slot_size = product(shape dims) * ceil(dtype_bits / 8)  (mirrors _buffer_parser).
    """
    shape_node = dtype_node = None
    for kw in type_call.keywords:
        if kw.arg == "shape":
            shape_node = kw.value
        elif kw.arg == "dtype":
            dtype_node = kw.value
    if shape_node is None or dtype_node is None:
        return None
    shape = _eval_const(shape_node, closure_vars)
    dtype = _eval_const(dtype_node, closure_vars)
    if not isinstance(shape, (list, tuple)) or dtype is None:
        return None
    elems = 1
    for d in shape:
        elems *= int(d)
    try:
        bits = int(dtype.get_bit())
    except Exception:
        return None
    return elems * ((bits + 7) // 8)


def scan_buffer_addr_ranges(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Compute each buffer's per-slot address ranges from its declaration.

    Returns: name -> (memory, [(start, end), ...] per slot).
    Buffers whose addrs/shape/dtype cannot be statically resolved are skipped
    (they simply won't participate in overlap detection).

    addrs handling (mirrors _buffer_parser):
      - single value  -> contiguous slots: base + i*slot_size
      - list/tuple     -> one explicit start address per slot
    slot count comes from depth when present, otherwise len(mutex_ids).
    """
    result: dict = {}
    for decl in decls:
        memory = _extract_memory_from_make_tile_group(decl.call)
        if memory is None:
            continue

        type_node = decl.kwargs.get("type")
        addrs_node = decl.kwargs.get("addrs")
        mutex_node = decl.kwargs.get("mutex_ids")
        depth_node = decl.kwargs.get("depth")
        if not (isinstance(type_node, ast.Call) and addrs_node is not None):
            continue

        slot_size = _tile_type_slot_size(type_node, closure_vars)
        mutex_ids = _eval_const(mutex_node, closure_vars) if mutex_node is not None else None
        depth = _eval_const(depth_node, closure_vars) if depth_node is not None else None
        addrs = _eval_const(addrs_node, closure_vars)
        if slot_size is None or addrs is None:
            continue
        if mutex_ids is not None and not isinstance(mutex_ids, (list, tuple)):
            continue
        if depth_node is not None:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
                continue
            num = depth
        elif isinstance(mutex_ids, (list, tuple)) and mutex_ids:
            num = len(mutex_ids)
        else:
            continue
        if isinstance(mutex_ids, (list, tuple)) and mutex_ids and len(mutex_ids) != num:
            continue  # malformed; parser will raise the real error

        if isinstance(addrs, (list, tuple)):
            if len(addrs) != num:
                continue  # malformed; skip (parser will raise the real error)
            starts = list(addrs)
        else:
            starts = [addrs + i * slot_size for i in range(num)]
        ranges = [(int(s), int(s) + slot_size) for s in starts]

        for name in decl.names:
            result[name] = (memory, ranges)
    return result


def scan_buffer_mutex_ids(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Each buffer's mutex_ids. Returns name -> tuple of ids.

    Needed to check that buffers sharing an address also share their locks: a mutex locks
    the address, so different ids over one region provide no mutual exclusion at all.
    Buffers whose ids are not statically resolvable are omitted rather than guessed.
    """
    result: dict = {}
    for decl in decls:
        mutex_node = decl.kwargs.get("mutex_ids")
        if mutex_node is None:
            continue
        mutex_ids = _eval_const(mutex_node, closure_vars)
        if not isinstance(mutex_ids, (list, tuple)):
            continue
        for name in decl.names:
            result[name] = tuple(mutex_ids)
    return result


def detect_addr_overlaps(addr_ranges: dict, cross_core_names: set) -> list:
    """Detect address-overlapping buffer pairs relevant to cross-core sync.

    Two buffers overlap if they share the same MemorySpace and any of their slot
    ranges intersect. Only pairs where **at least one side is a cross-core buffer**
    are reported — overlaps between two local buffers (e.g. multiple views of the
    same UB region like p_f16_db / p_f16_main_db) are the user's / auto_mutex's
    concern, not the pipeline cross-core sync's. Returns (buf_a, buf_b) pairs
    (names sorted).

    Constraint: only pairwise overlap is supported. If any buffer overlaps with
    more than one other buffer (counting only cross-core-relevant overlaps),
    raises ValueError.
    """
    names = sorted(addr_ranges.keys())
    overlaps = []
    # buffer -> set of buffers it overlaps with (for the 3+ check)
    overlap_partners: dict = {}

    def _ranges_intersect(ra, rb) -> bool:
        for sa, ea in ra:
            for sb, eb in rb:
                if sa < eb and sb < ea:  # half-open interval intersection
                    return True
        return False

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Skip if neither is a cross-core buffer (local-local overlap is not
            # a pipeline sync concern).
            if a not in cross_core_names and b not in cross_core_names:
                continue
            mem_a, ranges_a = addr_ranges[a]
            mem_b, ranges_b = addr_ranges[b]
            if mem_a != mem_b:
                continue
            if _ranges_intersect(ranges_a, ranges_b):
                # Slot count must be the same for overlapping buffers (precise
                # per-slot overlap tracking is not yet supported).
                if len(ranges_a) != len(ranges_b):
                    raise ValueError(
                        f"pipeline: address-overlapping buffers '{a}' and '{b}' have "
                        f"different slot counts ({len(ranges_a)} vs {len(ranges_b)}). "
                        f"Overlapping buffers must have the same number of slots."
                    )
                overlaps.append((a, b))
                overlap_partners.setdefault(a, set()).add(b)
                overlap_partners.setdefault(b, set()).add(a)

    for buf, partners in overlap_partners.items():
        if len(partners) > 1:
            raise ValueError(
                f"pipeline: buffer '{buf}' has overlapping addresses with multiple "
                f"buffers {sorted(partners)}. Only pairwise address overlap is "
                f"supported (at most 2 buffers may share a region)."
            )
    return overlaps


def scan_cross_core_buffers(
    decls: list[TileGroupDecl], closure_vars: dict
) -> tuple[dict[str, CrossCoreBuffer], list]:
    """Pick out the cross-core tile-group declarations (those carrying fwd/bwd ids).

    The only declaration scan that raises: a buffer asking for cross-core sync must have a
    resolvable memory space and valid id tuples, or no sync can be generated for it. The
    scans that merely collect layout information skip what they cannot resolve instead.

    Returns:
        (dict mapping buffer variable name -> CrossCoreBuffer,
         lifted_ids: list of (var_name, ast_literal) for literal fwd/bwd ids
         that need to be declared as variables before the pipeline loop)
    """
    result: dict[str, CrossCoreBuffer] = {}
    lifted_ids: list[tuple[str, ast.expr]] = []

    for decl in decls:
        fwd_node = decl.kwargs.get("fwd_ids")
        bwd_node = decl.kwargs.get("bwd_ids")
        if fwd_node is None and bwd_node is None:
            continue

        bufname = decl.names[0] if decl.names else "<tile_group>"

        _validate_buffer_memory(decl.call, bufname)
        fwd_count, bwd_count = _validate_ids(fwd_node, bwd_node, bufname, closure_vars)
        fwd_node, bwd_node = _lift_literal_ids(fwd_node, bwd_node, bufname, lifted_ids)

        for name in decl.names:
            result[name] = CrossCoreBuffer(
                fwd_ids_node=fwd_node,
                bwd_ids_node=bwd_node,
                fwd_slot_count=fwd_count,
                bwd_slot_count=bwd_count,
            )

    return result, lifted_ids


def _validate_buffer_memory(call: ast.Call, bufname: str) -> None:
    """L8: Validate that a make_tile_group call declares a resolvable memory space."""
    if _extract_memory_from_make_tile_group(call) is None:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' has no resolvable memory space. "
            f"Its make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X)) "
            f"must set target_memory to a literal pl.MemorySpace.<X>."
        )


def _validate_ids(fwd_node, bwd_node, bufname: str, closure_vars: dict) -> tuple[int, int]:
    """L11: Validate fwd_ids/bwd_ids are resolvable non-empty tuples of valid event ids."""
    fwd_count = _resolve_tuple_len(fwd_node, closure_vars)
    bwd_count = _resolve_tuple_len(bwd_node, closure_vars)
    if fwd_node is not None and fwd_count == 0:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' fwd_ids must be a non-empty tuple "
            f"(literal or a module-level tuple constant); could not resolve its length."
        )
    if bwd_node is not None and bwd_count == 0:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' bwd_ids must be a non-empty tuple "
            f"(literal or a module-level tuple constant); could not resolve its length."
        )
    # Cross-core event ids are a hardware resource limited to 0..15. Out-of-range values
    # fail at runtime on device, so reject them here where the source location is known.
    # Only statically resolvable values are checked; anything else is left to the user.
    for label, node in (("fwd_ids", fwd_node), ("bwd_ids", bwd_node)):
        bad = [v for v in _resolve_tuple_ints(node, closure_vars) if not 0 <= v <= _MAX_EVENT_ID]
        if bad:
            raise ValueError(
                f"pipeline: cross-core buffer '{bufname}' {label} contains out-of-range "
                f"event id(s) {bad}; cross-core event ids must be in 0..{_MAX_EVENT_ID}."
            )
    return fwd_count, bwd_count


def _lift_literal_ids(fwd_node, bwd_node, bufname: str, lifted_ids: list) -> tuple[ast.expr, ast.expr]:
    """Lift literal fwd_ids/bwd_ids to variable names for codegen compatibility."""
    if fwd_node is not None and isinstance(fwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_fwd_ids_{bufname}"
        lifted_ids.append((var_name, fwd_node))
        fwd_node = ast.Name(id=var_name, ctx=ast.Load())
    if bwd_node is not None and isinstance(bwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_bwd_ids_{bufname}"
        lifted_ids.append((var_name, bwd_node))
        bwd_node = ast.Name(id=var_name, ctx=ast.Load())
    return fwd_node, bwd_node


def _slot_accessor_group_name(value: ast.expr) -> str | None:
    """Group name for a slot accessor expression, or None if it is not one.

    Both accessor spellings resolve a group handle to one of its tiles:
    ``group.next()/current()/previous()`` and ``group[i]``.
    """
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in ("next", "current", "previous"):
            return func.value.id if isinstance(func.value, ast.Name) else None
        return None
    if isinstance(value, ast.Subscript):
        return value.value.id if isinstance(value.value, ast.Name) else None
    return None


def _get_slot_accessor_assignment(node: ast.AST, param_names: set[str]) -> tuple[str, str] | None:
    """Return slot variable and source buffer for ``slot = group.next()`` / ``slot = group[i]``."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None

    group_name = _slot_accessor_group_name(node.value)
    if group_name is None or group_name not in param_names:
        return None
    return node.targets[0].id, group_name


def scan_kernel_slot_to_buffer(func_def: ast.FunctionDef, buffer_names: set[str]) -> dict[str, str]:
    """Scan the kernel body for `slot = buf.next()` (or .current()/.previous()),
    mapping slot variable -> buffer name.

    This handles the case where the user takes a slot OUTSIDE the stage function
    (in the pipeline loop) and passes the slot tile as a stage argument.

    Covers EVERY declared buffer, not just the cross-core ones: a local buffer's slot has to
    be traceable too, because an op's pipe is decided by the memory spaces of all its tiles
    — the local side of a `pl.move` included.
    """
    slot_to_buffer: dict[str, str] = {}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, buffer_names)
        if acc is not None:
            slot_name, buffer_name = acc
            slot_to_buffer[slot_name] = buffer_name
    return slot_to_buffer


def build_binding_map(
    func_def: ast.FunctionDef,
    call_args: list | None,
    kernel_slot_map: dict,
    group_names: set[str],
    tuple_fields: dict | None = None,
) -> dict[str, tuple[str, bool]]:
    """Build the name -> (buffer, is_group) binding map for a stage function body.

    Resolves all names that reference a cross-core buffer (directly or indirectly)
    within this function scope. Handles:
      (a) Formal params bound via call-site positional args (the actual is a declared tile
          group — cross-core or local — or a slot taken from one in the kernel body).
      (b) .next()/.current()/.previous() calls in the body: slot = group.next().
      (c) Pure Name-to-Name alias assignments: tmp = some_known_name.
      (d) Member reads off an aggregate param: grp = agg.field, where the call site passed a
          `pl.make_tuple(field=...)` — rejoins the chain the aggregate broke.

    Steps (b) through (d) iterate until stable (supports alias chains).

    Args:
        func_def: the stage function AST.
        call_args: list of AST args from the call site. If None, returns empty
            (no name-collision guessing).
        kernel_slot_map: kernel-body slot -> buffer map (from scan_kernel_slot_to_buffer).
        group_names: every declared tile group's name (from scan_all_tile_group_names).
            Cross-core groups are a subset, so one set covers both kinds.
        tuple_fields: {tuple variable: {field: source variable}} (from scan_tuple_fields).

    Returns: {name: (buffer_declared_name, is_group)}
    """
    param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
    result: dict[str, tuple[str, bool]] = {}
    # Aggregate params: {param name: {field: source variable}}. Kept apart from `result`
    # because an aggregate is not a buffer — it is only a route to one.
    aggregates: dict[str, dict[str, str]] = {}

    # (a) Param bindings from call site (positional). Without call args we cannot
    # resolve bindings — name-collision guessing is intentionally NOT done.
    if call_args is None:
        return result
    for pos, arg in enumerate(call_args):
        if pos >= len(param_names) or not isinstance(arg, ast.Name):
            continue
        actual = arg.id
        # A local group is bound under its real name just like a cross-core one. Only
        # cross-core buffers get sync of their own, but a local buffer still has to be
        # traceable: an op's pipe follows the memory spaces of ALL its tiles, so the local
        # side of a `pl.move` needs a name that resolves. Leaving it out here is what made
        # the scan fall back to the FORMAL parameter name, which only ever matched by the
        # convention that call sites name their arguments after the parameters.
        if actual in group_names:
            result[param_names[pos]] = (actual, True)
        elif actual in kernel_slot_map:
            result[param_names[pos]] = (kernel_slot_map[actual], False)
        elif actual in (tuple_fields or {}):
            aggregates[param_names[pos]] = tuple_fields[actual]

    # (b)(c)(d) Propagate slot accessors (.next() / group[i]), aggregate member reads and
    # alias assignments until stable.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func_def):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in result:
                continue  # already resolved

            # (d) grp = agg.field -> the variable the call site put in that field. Recorded
            # even when that variable is a LOCAL tile group: an op's pipe is decided by the
            # memory spaces of ALL its tiles, so the local side of a `move` has to resolve
            # too. Not being a cross-core buffer keeps it out of the access records.
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                fields = aggregates.get(node.value.value.id)
                if fields is not None:
                    source = fields.get(node.value.attr)
                    if source is not None:
                        result[target] = (source, True)
                        changed = True
                    continue

            if isinstance(node.value, (ast.Call, ast.Subscript)):
                group_name = _slot_accessor_group_name(node.value)
                if group_name is not None and group_name in result:
                    src_buf, src_is_group = result[group_name]
                    if src_is_group:
                        result[target] = (src_buf, False)
                        changed = True
                continue

            if isinstance(node.value, ast.Name) and node.value.id in result:
                result[target] = result[node.value.id]
                changed = True

    return result


def _build_slot_to_buffer_from_bindings(
    func_def: ast.FunctionDef, bindings: dict[str, tuple[str, bool]]
) -> dict[str, str]:
    """Build slot_to_buffer from binding map. Covers ALL params (for pipe resolution).

    - Names with is_group=False → directly map to their buffer (they are slots).
    - Names from `slot = param.next()` for ANY param → map to param's resolved name
      (for non-cross-core params, resolved name = param name itself for pipe calc).
    """
    slot_to_buffer: dict[str, str] = {}
    # All resolved slot names (is_group=False) → buffer
    for name, (buf, is_group) in bindings.items():
        if not is_group:
            slot_to_buffer[name] = buf
    # For pipe resolution: also scan .next() on non-cross-core params (local buffers).
    # Their resolved name is just the param name (standard pipe-table lookup).
    all_param_names = {a.arg for a in func_def.args.args if a.arg != "self"}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, all_param_names)
        if acc is not None:
            slot_name, param_name = acc
            if slot_name not in slot_to_buffer:
                # Resolve: if param is in bindings use its buffer, else param name itself
                if param_name in bindings:
                    slot_to_buffer[slot_name] = bindings[param_name][0]
                else:
                    slot_to_buffer[slot_name] = param_name
    return slot_to_buffer


def _scan_stmts(
    stmts: list, slot_to_buffer, cross_buffers, all_buffer_memory, vf_func_defs, region_members, region_access
) -> None:
    """Record every buffer access in these statements, one ``(buffer, role, pipe)`` per op,
    in source order.

    Control flow is deliberately ignored: an op inside a branch is recorded exactly like
    an unconditional one. The graph then syncs every pipe a buffer is touched on, which
    over-approximates when a path skips some of them but is never short of a sync. Working
    out which pipes a given path really uses would need condition reasoning, and getting
    that wrong drops syncs rather than adding them.
    """
    for stmt in stmts:
        for node in _iter_calls_in_order(stmt):
            _scan_call(
                node, slot_to_buffer, cross_buffers, all_buffer_memory, vf_func_defs, region_members, region_access
            )


def _scan_call(
    node: ast.Call, slot_to_buffer, cross_buffers, all_buffer_memory, vf_func_defs, region_members, region_access
) -> None:
    """Record one call's accesses."""
    if not isinstance(node, ast.Call):
        return
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
        _handle_block_op(node, slot_to_buffer, cross_buffers, all_buffer_memory, region_members, region_access)
    elif isinstance(node.func, ast.Name):
        _handle_vf_call(node, slot_to_buffer, cross_buffers, vf_func_defs, region_members, region_access)


def _handle_block_op(
    node: ast.Call, slot_to_buffer, cross_buffers, all_buffer_memory, region_members, region_access
) -> None:
    """Record a single pl.<op>(...) call."""
    op_name = node.func.attr
    roles = _BLOCK_OP_TILE_ROLES.get(op_name)
    if roles is None:
        return
    if op_name == "fillpad":
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "INPLACE":
                roles = _BLOCK_OP_TILE_ROLES["fillpad_inplace"]
                break
    pipe = None
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, slot_to_buffer)
        if buf is None or argpos >= len(roles):
            continue
        role = roles[argpos]
        if role is None:
            continue
        if buf in cross_buffers or buf in region_members:
            if pipe is None:
                pipe = _block_op_pipe(op_name, node, slot_to_buffer, all_buffer_memory)
            _record_region_access(region_access, buf, role, pipe)


def _handle_vf_call(
    node: ast.Call, slot_to_buffer, cross_buffers, vf_func_defs, region_members, region_access
) -> None:
    """Record a single VF helper call, always on pipe V."""
    vf_name = node.func.id
    vf_def = vf_func_defs.get(vf_name)
    if vf_def is None:
        return
    vf_roles = _scan_vf_roles(vf_def, vf_func_defs)
    vf_params = [a.arg for a in vf_def.args.args if a.arg != "self"]
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, slot_to_buffer)
        if buf is None or argpos >= len(vf_params):
            continue
        role = vf_roles.get(vf_params[argpos])
        if role is None:
            continue
        if buf in cross_buffers or buf in region_members:
            _record_region_access(region_access, buf, role, "V")


def _iter_calls_in_order(node: ast.AST):
    """Yield all ast.Call nodes in SOURCE order (DFS, child order)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_calls_in_order(child)


def scan_stage_accesses(
    stage_func_def: ast.FunctionDef,
    cross_buffers: dict[str, CrossCoreBuffer],
    vf_func_defs: dict[str, ast.FunctionDef],
    all_buffer_memory: dict[str, MemorySpace],
    region_members: set | None = None,
    region_access_out: list | None = None,
    call_args: list | None = None,
    kernel_slot_map: dict | None = None,
    tuple_fields: dict | None = None,
    group_names: set[str] | None = None,
) -> None:
    """Scan a stage function body for its buffer accesses, one entry per op.

    Results land in ``region_access_out``; control flow is ignored (see _scan_stmts).

    Args:
        stage_func_def: the @stage function AST
        cross_buffers: name -> CrossCoreBuffer (cross-core buffers)
        vf_func_defs: name -> FunctionDef for VF helper functions (for role scan)
        all_buffer_memory: name -> MemorySpace for ALL buffers (for move pipe)
        region_members: buffer names that share a physical region with another buffer.
        region_access_out: list to receive the op-level (buffer, role, pipe) entries.
        call_args: list of AST args from the call site (None for fallback).
        kernel_slot_map: kernel-body slot -> buffer (from scan_kernel_slot_to_buffer).
        tuple_fields: tuple variable -> {field: source} (from scan_tuple_fields).
        group_names: every declared tile group's name (from scan_all_tile_group_names).
    """
    bindings = build_binding_map(
        stage_func_def, call_args, kernel_slot_map or {}, group_names or set(), tuple_fields or {}
    )
    if not bindings:
        return

    group_param_names = {p for p, (_b, ig) in bindings.items() if ig}
    _validate_slot_accessors(stage_func_def, group_param_names)
    slot_to_buffer = _build_slot_to_buffer_from_bindings(stage_func_def, bindings)

    region_access: list = []
    _scan_stmts(
        stage_func_def.body,
        slot_to_buffer,
        cross_buffers,
        all_buffer_memory,
        vf_func_defs,
        region_members or set(),
        region_access,
    )
    if region_access_out is not None:
        region_access_out.extend(region_access)


def _validate_slot_accessors(stage_func_def: ast.FunctionDef, group_param_names: set[str]) -> None:
    """L10: cross-core buffer slot accessors must be `slot = param.next()` / `slot = param[i]` form.

    group_param_names: formal params that carry a cross-core buffer GROUP (the ones
    the body calls .next()/.current()/.previous() on, or subscripts)."""
    accessor_rhs_ids = set()
    for node in ast.walk(stage_func_def):
        sa = _get_slot_accessor_assignment(node, group_param_names)
        if sa is not None:
            accessor_rhs_ids.add(id(node.value))
    for node in ast.walk(stage_func_def):
        if not isinstance(node, (ast.Call, ast.Subscript)):
            continue
        group_name = _slot_accessor_group_name(node)
        if group_name is None or group_name not in group_param_names:
            continue
        if id(node) in accessor_rhs_ids:
            continue
        accessor = "[...]" if isinstance(node, ast.Subscript) else f".{node.func.attr}()"
        raise ValueError(
            f"pipeline: cross-core buffer group '{group_name}' slot accessor "
            f"`{accessor}` must be assigned to a simple variable "
            f"(`slot = {group_name}{accessor}`); inline/chained/"
            f"tuple-unpack forms are not supported."
        )


def scan_tuple_fields(kernel_func_def: ast.FunctionDef) -> dict[str, dict[str, str]]:
    """``{tuple variable: {field name: the variable it was built from}}`` for each
    ``x = pl.make_tuple(field=var, ...)`` in the kernel body.

    A stage handed an aggregate reads its members back out as ``agg.field``, which breaks the
    chain from a tile back to its declared tile group. This records the one hop needed to
    rejoin it — see rule (d) in build_binding_map.

    Only keyword members bound to a plain name are recorded: those are the ones a field
    access can be traced through.
    """
    result: dict[str, dict[str, str]] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if _get_ctor_name(node.value) != "make_tuple":
            continue
        fields = {kw.arg: kw.value.id for kw in node.value.keywords if kw.arg and isinstance(kw.value, ast.Name)}
        if not fields:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = fields
    return result


def _get_ctor_name(call: ast.Call) -> str | None:
    """Get the constructor name from a call like pl.UBNBuffer(...) -> 'UBNBuffer'."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _resolve_tuple_len(node: ast.expr | None, closure_vars: dict) -> int:
    """Resolve the length of a tuple-valued node (Name referencing a module-level
    tuple, or a literal Tuple).

    Counts ELEMENTS, deliberately not ``len(_resolve_tuple_ints(...))``: this length becomes
    the buffer's slot count, i.e. the modulus of ``task % slot_count``. An id the scan cannot
    resolve to an int still occupies a slot, so dropping it here would shrink the modulus and
    silently misindex every sync on that buffer. _resolve_tuple_ints skips such an element on
    purpose, because it only feeds a range check that has nothing to say about it.
    """
    if node is None:
        return 0
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        val = closure_vars.get(node.id)
        if isinstance(val, (tuple, list)):
            return len(val)
    return 0


def _resolve_tuple_ints(node: ast.expr | None, closure_vars: dict) -> list:
    """Resolve a tuple-valued node to its int values, or [] if not statically known.

    Same shapes as _resolve_tuple_len (literal tuple/list, or a Name bound to a
    module-level tuple), but returns the values so they can be range-checked.
    """
    if node is None:
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        values = []
        for elt in node.elts:
            # A negative literal parses as UnaryOp(USub, Constant), not Constant(-n), so
            # matching only Constant would silently skip negative ids.
            if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                elt, sign = elt.operand, -1
            else:
                sign = 1
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                values.append(sign * elt.value)
        return values
    if isinstance(node, ast.Name):
        val = closure_vars.get(node.id)
        if isinstance(val, (tuple, list)):
            return [v for v in val if isinstance(v, int)]
    return []


def _tile_arg_buffer(arg: ast.expr, slot_to_buffer: dict[str, str]) -> str | None:
    """If arg is a bare slot variable (from group.next()) that maps to a buffer,
    return the buffer name; else None.

    New API: group.next() returns a bare tile, so op args are plain Names
    (e.g. pl.move(qk_left, cur_k)), not `slot.tile` attributes.
    """
    if isinstance(arg, ast.Name):
        return slot_to_buffer.get(arg.id)
    return None


def _block_op_pipe(
    op_name: str, call: ast.Call, slot_to_buffer: dict[str, str], all_buffer_memory: dict[str, MemorySpace]
) -> str:
    """Determine the pipe name for a block op accessing a cross-core buffer.

    Raises when the pipe cannot be determined. There is no safe default: the pipe decides
    which queue the wait/set lands on, so guessing puts the sync on a queue the op never
    runs on — and a section only has some of the pipes, so a guess can even name one that
    does not exist on that core. Either way the dependency goes unenforced, silently.
    """
    if op_name == "move":
        # move(dst, src): pipe depends on src/dst memory
        dst_mem = _arg_memory(call.args[0] if call.args else None, slot_to_buffer, all_buffer_memory)
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, slot_to_buffer, all_buffer_memory)
        if src_mem is None or dst_mem is None:
            unresolved = "destination" if dst_mem is None else "source"
            raise ValueError(
                f"pipeline: cannot determine the pipe of `pl.move` at line {call.lineno}, "
                f"because its {unresolved} tile does not resolve to a declared tile group. "
                f"The move touches a cross-core buffer, so its pipe decides where the sync "
                f"goes. A tile reached through an aggregate (e.g. `tile_groups.x.next()`) is "
                f"not yet traced; pass the tile group to the stage as its own argument."
            )
        return _pipe_name(get_move_pipe(src_mem, dst_mem))
    if op_name in ("store", "store_tile"):
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, slot_to_buffer, all_buffer_memory)
        if src_mem is None:
            raise ValueError(
                f"pipeline: cannot determine the pipe of `pl.{op_name}` at line {call.lineno}, "
                f"because its source tile does not resolve to a declared tile group. The store "
                f"touches a cross-core buffer, so its pipe decides where the sync goes. A tile "
                f"reached through an aggregate (e.g. `tile_groups.x.next()`) is not yet traced; "
                f"pass the tile group to the stage as its own argument."
            )
        return _pipe_name(get_store_pipe(src_mem))
    pipe = get_op_pipe(op_name)
    if pipe is None:
        raise ValueError(
            f"pipeline: `pl.{op_name}` at line {call.lineno} touches a cross-core buffer, but "
            f"no pipe is registered for it, so the sync has nowhere to go. Register the op's "
            f"pipe (see get_op_pipe) or keep the cross-core buffer out of this op."
        )
    return _pipe_name(pipe)


def _arg_memory(arg, slot_to_buffer, all_buffer_memory) -> MemorySpace | None:
    """Get the MemorySpace of a `slot.tile` arg (any buffer, cross-core or local)."""
    buf = _tile_arg_buffer(arg, slot_to_buffer) if arg is not None else None
    if buf is not None:
        return all_buffer_memory.get(buf)
    return None


def _pipe_name(pipe) -> str:
    """PipeType enum -> short name string used in generated pl.PipeType.<NAME>."""
    # pipe is a PipeType enum; its name attribute gives FIX/V/MTE1/...
    return getattr(pipe, "name", str(pipe).split(".")[-1])


def _record_region_access(region_access: list, buf: str, role: str, pipe: str) -> None:
    """Append one op-level access.

    Kept per-op rather than collapsed to (first_pipe, last_pipe): the graph needs each
    access as its own node, and a pair of endpoints cannot express a local access sitting
    BETWEEN two cross-core ones. The role comes straight from the op's argument roles — a
    buffer may legitimately be read and written within one stage.
    """
    region_access.append((buf, role, pipe))


def _root_name(node):
    """Return the root name of a possibly indexed/attributed expression."""
    cur = node
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.BinOp):
            cur = cur.left
            continue
        if isinstance(cur, (ast.Attribute, ast.Subscript)):
            cur = cur.value
            continue
        return None


def _merge_vf_role(result: dict[str, str], name: str, role: str | None) -> None:
    """Merge one VF read/write role into the per-param role map."""
    ex = result.get(name)
    if ex is None:
        result[name] = role
    elif ex != role and role is not None:
        result[name] = "RW"


def _record_vf_call_role(sub: ast.Call, param_names: set[str], result: dict[str, str]) -> None:
    """Record the R/W role represented by a single vf load/store call.

    Handles both statement form (vf.load_align(dst_reg, ptr, ...)) and
    assignment form (ptr_arg = vf.load_align(ptr, ...)).
    """
    if not isinstance(sub.func, ast.Attribute):
        return
    # Only vf.* calls are relevant (guards against pl.load etc. in non-VF helpers)
    if not (isinstance(sub.func.value, ast.Name) and sub.func.value.id == "vf"):
        return
    op = sub.func.attr
    if op.startswith("load") and len(sub.args) >= 1:
        root = None
        if len(sub.args) >= 3:
            root = _root_name(sub.args[1])
            if root not in param_names:
                root = None
        if root is None:
            root = _root_name(sub.args[0])
        if root in param_names:
            _merge_vf_role(result, root, "R")
    elif op.startswith("store") and len(sub.args) >= 1:
        root = _root_name(sub.args[0])
        if root in param_names:
            _merge_vf_role(result, root, "W")


def _scan_vf_roles(vf_func_def: ast.FunctionDef, vf_func_defs: dict | None = None) -> dict[str, str]:
    """Scan a VF helper body for param R/W roles.

    Works for both ``@pl.vector_function`` decorated functions (scans
    ``vf.load_*`` / ``vf.store_*`` calls directly) and plain wrapper functions
    that delegate to a known VF function (propagates roles through the call's
    positional argument mapping, recursively for multi-level chains).

    - vf.load_*(reg, ptr, ...) -> ptr param is "R"
    - vf.store_*(ptr, reg, ...) -> ptr param is "W"
    - both -> "RW"
    """
    param_names = {a.arg for a in vf_func_def.args.args if a.arg != "self"}
    result: dict[str, str] = {}

    for sub in ast.walk(vf_func_def):
        if isinstance(sub, ast.Call):
            _record_vf_call_role(sub, param_names, result)
            # Propagate roles through calls to known VF functions (recursive)
            if vf_func_defs and isinstance(sub.func, ast.Name):
                callee_def = vf_func_defs.get(sub.func.id)
                if callee_def is not None and callee_def is not vf_func_def:
                    callee_roles = _scan_vf_roles(callee_def, vf_func_defs)
                    callee_params = [a.arg for a in callee_def.args.args if a.arg != "self"]
                    for argpos, arg in enumerate(sub.args):
                        if argpos >= len(callee_params):
                            break
                        callee_param = callee_params[argpos]
                        role = callee_roles.get(callee_param)
                        if role is None:
                            continue
                        root = _root_name(arg)
                        if root in param_names:
                            _merge_vf_role(result, root, role)
    return result
