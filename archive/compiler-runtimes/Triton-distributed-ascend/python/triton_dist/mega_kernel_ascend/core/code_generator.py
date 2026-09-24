# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade megakernel source generator.
#
# Only generates source string; exec is done by ModelBuilder.
# Dispatch is if/elif on op_id (Graph node index) calling named compute fns.
# Call args are taken from each fn's own signature (no unified compute ABI).
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, Dict, Mapping, Union

OpTable = Mapping[int, Callable]

_DISPATCH_PLACEHOLDER = "{dispatch_code}"

_MEGA_KERNEL_TEMPLATE = '''
import triton
import triton_dist
import triton.language as tl
import triton_dist.language as dl
from triton_dist.mega_kernel_ascend.kernels.task_context_blade import (
    Context, Scoreboard, global_block_idx_mix,
    global_block_idx, query_ts_condition, wait_ts_ready,
    STAGE_START, STAGE_AICPU_HS, STAGE_READY, STAGE_START_LOOP,
    STAGE_QUIT, STAGE_QUIT_ACK,
    QUIT_OPCODE, TASK_RING_SIZE,
)
from triton_dist.mega_kernel_ascend.kernels import *
import triton.language.extra.cann.extension as al

@triton_dist.jit
def kernel(ctxData):
    # === 1. Init ===
    block_idx = global_block_idx_mix()
    ctx = Context(ctxData)
    lctx = ctx.scoreboard_ref(block_idx)
    lctx.set_core_id(block_idx)
    lctx.set_block_id(global_block_idx())
    lctx.set_sub_block_id(al.sub_vec_id())
    lctx.set_global_block_id(block_idx)
    lctx.reset()

    # === 2. Handshake ===
    lctx.set_status(STAGE_START)
    lctx.wait()
    lctx.set_status(STAGE_READY)
    lctx.set_status(STAGE_START_LOOP)

    # === 3. Wait DATA_MAIN_BASE ===
    wait_ts_ready()
    # SetSyncBaseAddr (TODO)

    # === 4. Main loop ===
    last_cond = tl.cast(0, tl.uint64)
    running = True

    while running:
        dmb = query_ts_condition()
        dmb_low = (dmb & 0xFFFFFFFF).to(tl.int32)

        if last_cond != dmb_low:
            idx = last_cond % TASK_RING_SIZE
            ring = lctx.fetch_task(idx)
            op_id = ring.get_opId()

            if op_id == QUIT_OPCODE:
                running = False
            else:
                # Read dependInfo + scoreboradPtr from TaskEle
                tileid = ring.get_tileid()
                tilecnt = ring.get_tilecnt()
                opctx = ctx.opcontext_ref(op_id)

                # === wait_deps wrapper only for A3 ===
                # lctx.wait_deps(depend_info)

                ios = ctxData + opctx.args_offset()
                data = ring.get_data()
                # Use tl.uint64 to avoid the source and type check of tl.if
                if data != 0:
                    args = data
                else:
                    args = (ctxData + opctx.tiling_offset()).to(tl.uint64)

                # TODO: replace op_id with task_type
                {dispatch_code}

                # === notify wrapper for A3 ===
                # lctx.release_tile(scoreboard, tile_id)

                dl.ascend.set_cond(last_cond + 1)
                last_cond = last_cond + 1

    # === 5. Quit ===
    lctx.set_lastInfo(ctx.opcontext_ref(0).addr())
    lctx.quit()
'''


def _placeholder_indent(template: str, placeholder: str = _DISPATCH_PLACEHOLDER) -> str:
    """Leading whitespace of the template line that contains ``placeholder``."""
    for line in template.splitlines():
        if placeholder in line:
            return line[: line.index(placeholder)]
    raise ValueError(f"{placeholder!r} not found in megakernel template")


def _locals_before_dispatch(
    template: str, placeholder: str = _DISPATCH_PLACEHOLDER
) -> frozenset[str]:
    """Collect kernel params + names assigned before ``{dispatch_code}``."""
    indent = _placeholder_indent(template, placeholder)
    marker = "pass  # __megakernel_dispatch__"
    src = template.replace(f"{indent}{placeholder}", f"{indent}{marker}", 1)
    tree = ast.parse(src)
    dispatch_lineno: int | None = None
    for i, line in enumerate(src.splitlines(), 1):
        if "__megakernel_dispatch__" in line:
            dispatch_lineno = i
            break
    if dispatch_lineno is None:
        raise ValueError("dispatch marker not found while deriving megakernel locals")

    names: set[str] = set()

    def _add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _add_target(elt)

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for a in node.args.args:
                names.add(a.arg)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            if node.lineno <= dispatch_lineno:
                for t in node.targets:
                    _add_target(t)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.lineno <= dispatch_lineno:
                _add_target(node.target)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if node.lineno <= dispatch_lineno:
                _add_target(node.target)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            if node.lineno <= dispatch_lineno:
                _add_target(node.target)
            self.generic_visit(node)

        def visit_withitem(self, node: ast.withitem) -> None:
            if node.optional_vars is not None and getattr(node, "lineno", 0) <= dispatch_lineno:
                _add_target(node.optional_vars)
            self.generic_visit(node)

    Collector().visit(tree)
    if not names:
        raise ValueError("failed to derive megakernel locals from template")
    return frozenset(names)


# Locals available at the dispatch call site (derived from the template).
_MEGA_KERNEL_SCOPE = _locals_before_dispatch(_MEGA_KERNEL_TEMPLATE)


def _normalize_op_table(op_table: Union[OpTable, list]) -> Dict[int, Callable]:
    if isinstance(op_table, Mapping):
        table = dict(op_table)
    else:
        table = {int(op_id): fn for op_id, fn in op_table}
    if not table:
        raise ValueError("op_table is empty: register at least one compute via ModelBuilder.make_*")
    return table


def _unwrap_callable(fn: Callable) -> Callable:
    """Prefer underlying Python fn for ``@triton.jit`` / wraps."""
    if hasattr(fn, "fn") and callable(fn.fn):
        return fn.fn
    if hasattr(fn, "__wrapped__") and callable(fn.__wrapped__):
        return fn.__wrapped__
    return fn


def _call_expr(fn: Callable) -> str:
    """Build ``name(a, b, ...)`` from ``fn``'s signature using megakernel locals."""
    raw = _unwrap_callable(fn)
    name = getattr(fn, "__name__", None) or getattr(raw, "__name__", "compute")
    sig = inspect.signature(raw)
    args: list[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"{name}: *args/**kwargs not supported in megakernel dispatch call")
        if pname not in _MEGA_KERNEL_SCOPE:
            raise ValueError(
                f"{name}: parameter {pname!r} is not available in megakernel scope "
                f"(allowed: {', '.join(sorted(_MEGA_KERNEL_SCOPE))})")
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            args.append(f"{pname}={pname}")
        else:
            args.append(pname)
    return f"{name}({', '.join(args)})"


def _generate_dispatch_code(op_table: Dict[int, Callable]) -> str:
    """Emit if/elif op_id branches; indent follows the ``{dispatch_code}`` line."""
    lines: list[str] = []
    for i, op_id in enumerate(sorted(op_table)):
        kw = "if" if i == 0 else "elif"
        lines.append(f"{kw} op_id == {op_id}:")
        lines.append(f"    {_call_expr(op_table[op_id])}")
    indent = _placeholder_indent(_MEGA_KERNEL_TEMPLATE)
    return textwrap.indent("\n".join(lines), indent)


def make_mega_kernel_src(op_table: Union[OpTable, list]) -> str:
    """Generate megakernel source from ``op_id`` (Graph node index) -> compute_fn.

    Compute symbols come from ``kernels import *``; only if/elif dispatch is filled in.
    """
    table = _normalize_op_table(op_table)
    indent = _placeholder_indent(_MEGA_KERNEL_TEMPLATE)
    # Replace the whole placeholder line so multi-line dispatch keeps one indent source.
    return _MEGA_KERNEL_TEMPLATE.replace(
        f"{indent}{_DISPATCH_PLACEHOLDER}",
        _generate_dispatch_code(table),
        1,
    )


class CodeGenerator:
    """Builds Blade megakernel Triton source from a compute op table."""

    def generate_code(self, op_table: Union[OpTable, list]) -> str:
        """Generate megakernel source from registered compute mapping."""
        return make_mega_kernel_src(op_table)
