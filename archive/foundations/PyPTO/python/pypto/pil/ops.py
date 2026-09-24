# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from contextlib import contextmanager
import functools
import operator
from typing import Optional, Sequence

import pypto
from pypto import SatStatus, SymbolicScalar, ir, pypto_impl

from .dispatcher import dispatch_block
from .op_registry import impl
from .pir import Block, BreakSignal, BuildContext, ContinueSignal, DoubleStarred, InsertPoint, Jump, LoopRange, Scope

_patch_methods = [
    (pypto.Tensor, "move", 0),
    (pypto, "index_put_", 0),
    (pypto, "index_add_", 0),
    (pypto, "atomic_add", 2),
]

_orig_assemble = pypto.assemble

def _store_wrapper(orig, arg_idx):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        Block.mark_store(args[arg_idx])
        return orig(*args, **kwargs)
    return wrapper


@contextmanager
def apply_patches():
    def assemble(*args, **kwargs) -> None:
        if pypto_impl.ir.assemble_new_logical_tensor() and not isinstance(args[0], Sequence):
            Block.mark_store(args[2])
        _orig_assemble(*args, **kwargs)

    try:
        originals = []
        for obj, name, arg_idx in _patch_methods:
            method = getattr(obj, name)
            originals.append((obj, name, method))
            setattr(obj, name, _store_wrapper(method, arg_idx))
        pypto.assemble = assemble
        yield
    finally:
        for obj, name, method in originals:
            setattr(obj, name, method)
        pypto.assemble = _orig_assemble


def has_scalar(values: list) -> bool:
    return any(isinstance(v, pypto.pypto_impl.SymbolicScalar) for v in values)


# Opaque Python values (dataclass instances, ...) are compile-time only:
# leave them in scope (the body's own stores keep them current) and skip
# them from the IR loop-carried / branch-carried yield state
def is_opaque_value(val) -> bool:
    return not (val is None or isinstance(val, (bool, int, float, pypto.SymbolicScalar, pypto.Tensor, list, tuple)))


# ---- Compile-time ops ----


@impl("pil.const")
def const_impl(ctx, value):
    return value


@impl("pil.load")
def load_impl(ctx, name):
    scope = Scope.current()
    return scope[name]


@impl("pil.store")
def store_impl(ctx, name, val):
    if isinstance(val, pypto.SymbolicScalar) and val.is_concrete():
        val = val.concrete()
    Scope.store(name, val)


@impl(operator.sub, partial=True)
@impl(operator.add, partial=True)
@impl(operator.mul, partial=True)
@impl(operator.truediv, partial=True)
@impl(operator.mod, partial=True)
@impl(operator.pow, partial=True)
@impl(operator.floordiv, partial=True)
@impl(operator.or_, partial=True)
@impl(operator.xor, partial=True)
@impl(operator.and_, partial=True)
@impl(operator.lshift, partial=True)
@impl(operator.rshift, partial=True)
@impl(operator.matmul, partial=True)
def binary_impl(ctx, op, x, y):
    return op(x, y)


@impl(operator.neg, partial=True)
@impl(operator.pos, partial=True)
@impl(operator.invert, partial=True)
@impl(operator.not_, partial=True)
def unary_impl(ctx, op, x):
    return op(x)


# ---- Comparison ----


@impl(operator.eq, partial=True)
@impl(operator.ne, partial=True)
@impl(operator.lt, partial=True)
@impl(operator.le, partial=True)
@impl(operator.gt, partial=True)
@impl(operator.ge, partial=True)
def compare_impl(ctx, op, x, y):
    return op(x, y)


# ---- Attribute / index ----


@impl(getattr, partial=True)
def getattr_impl(ctx, op, obj, attr, *args):
    return op(obj, attr, *args)


@impl(delattr, partial=True)
def delattr_impl(ctx, op, obj, attr):
    return op(obj, attr)


# ---- Tensor construction ----

@impl(pypto.Tensor)
def create_tensor(ctx, *args, **kwargs):
    t = pypto.Tensor(*args, **kwargs)
    lt = t.logical_tensor()
    stmt = ctx.create_tensor_op_stmt(
        result=[lt], result_token=None, opcode="TENSOR_ALLOC",
        args=[], tokens=[], attrs={}, span=ctx.span)
    ctx.emit(stmt)
    return t


# ---- Collection construction ----


@impl(list)
def list_impl(ctx, items=None):
    if items is None:
        return []
    return Scope.current().resolve(items)


@impl(tuple)
def tuple_impl(ctx, items=None):
    if items is None:
        return ()
    return tuple(Scope.current().resolve(items))


@impl(set)
def set_impl(ctx, items=None):
    if items is None:
        return set()
    return set(Scope.current().resolve(items))


@impl(dict)
def dict_impl(ctx, items=()):
    scope = Scope.current()
    result = {}
    for k, v in items:
        if isinstance(v, DoubleStarred):
            result.update(scope.resolve(v.value))
        else:
            result[scope.resolve(k)] = scope.resolve(v)
    return result


@impl("pil.raise")
def raise_impl(ctx, exc, cause):
    if cause is None:
        raise exc
    raise exc from cause


@impl(operator.getitem)
def getitem_impl(ctx, obj, key):
    return obj[key]


@impl(operator.delitem)
def delitem_impl(ctx, obj, key):
    del obj[key]


@impl(operator.setitem)
def setitem_impl(ctx, obj, key, value):
    obj[key] = value


@impl(min)
def min_impl(ctx, *args):
    if has_scalar(list(args)) and len(args) == 2:
        return pypto.min(args[0], args[1])
    return min(args)


@impl(max)
def max_impl(ctx, *args):
    if has_scalar(list(args)) and len(args) == 2:
        return pypto.max(args[0], args[1])
    return max(args)


def _pypto_loop(batch, *args, **kwargs):
    nargs = len(args)
    if nargs == 1:
        start, stop, step = 0, args[0], 1
    elif nargs == 2:
        start, stop, step = args[0], args[1], 1
    elif nargs == 3:
        start, stop, step = args
    else:
        raise TypeError(f"loop() takes 1 to 3 positional arguments but {nargs} were given")

    unroll_list = kwargs.get("unroll_list", None)
    parallel = kwargs.get("parallel", False)
    submit_before_loop = kwargs.get("submit_before_loop", False)
    name = kwargs.get("name", None)
    idx_name = kwargs.get("idx_name", None)
    unroll_list = sorted(set(unroll_list or []) | {1}, reverse=True)
    for u in unroll_list:
        if not isinstance(u, int) or u <= 0:
            raise ValueError(f"unroll factor {u} must be a positive integer")

    return LoopRange(start, stop, step, unroll_list, batch, parallel, submit_before_loop, name, idx_name)


@impl(pypto.loop)
def pypto_loop_impl(ctx: BuildContext, *args, **kwargs):
    return _pypto_loop(False, *args, **kwargs)


@impl(pypto.loop_unroll)
def pypto_loop_unroll_impl(ctx: BuildContext, *args, **kwargs):
    return _pypto_loop(True, *args, **kwargs)


@impl(pypto.is_loop_begin)
def is_loop_begin_impl(ctx: BuildContext, scalar: SymbolicScalar):
    start, _, _ = ctx.loop_stack[-1]
    assert isinstance(start, (SymbolicScalar, int)), "is_loop_begin() must be called in a pypto.loop"
    return scalar == start


@impl(pypto.is_loop_end)
def is_loop_end_impl(ctx: BuildContext, scalar: SymbolicScalar):
    _, end, step = ctx.loop_stack[-1]
    assert isinstance(end, (SymbolicScalar, int)), "is_loop_end() must be called in a pypto.loop"
    assert isinstance(step, (SymbolicScalar, int)), "is_loop_end() must be called in a pypto.loop"
    return scalar + step >= end


@impl(pypto.cond)
def cond_impl(ctx: BuildContext, cond):
    return cond


def _add_jump_stmt(ctx: BuildContext, jump, operands: Optional[list[ir.Expr]] = None):
    if operands is None:
        operands = []
    scope = Scope.current()
    names = ctx.return_var_names

    if jump == Jump.BREAK:
        operands = [ctx.unwrap(scope[name]) for name in names]
        stmt = ctx.create_break_stmt(operands, ctx.span)
    elif jump == Jump.CONTINUE:
        operands = [ctx.unwrap(scope[name]) for name in names]
        stmt = ctx.create_continue_stmt(operands, ctx.span)
    elif jump == Jump.RETURN:
        stmt = ctx.create_return_stmt(operands, ctx.span)
    elif jump == Jump.END_BRANCH:
        stmt = ctx.create_yield_stmt(operands, ctx.span)
    else:
        raise ValueError(f"Unknown jump type: {jump}")

    ctx.emit(stmt)


def _static_for(body: Block, iterator):
    scope = Scope.current()
    loop_var = body.args[0]
    for item in iterator:
        scope.varmap[loop_var.id] = item
        try:
            dispatch_block(body, True)
        except BreakSignal:
            break
        except ContinueSignal:
            continue


def _static_while(body: Block):
    while True:
        try:
            dispatch_block(body, True)
        except BreakSignal:
            break
        except ContinueSignal:
            continue


def _loop_unroll(body: Block, loop: LoopRange, factor, nstart, nstop, ctx: BuildContext):
    scope = Scope.current()
    loop_val = body.args[0]

    # Create loop variable Var
    loop_var_name = f"loop_idx_{loop.idx_name}" if loop.idx_name else f"loop_idx_{loop_val.id}"
    loop_var = ctx.create_scalar_var(loop_var_name)
    scope.varmap[loop_val.id] = loop_var

    iter_args, return_var_names = [], []
    for name in sorted(body.store_names):
        val = scope.locals.get(name)
        if is_opaque_value(val):
            continue
        var = ctx.create_var_like(name, ctx.unwrap(val))
        iter_arg = ctx.create_iter_arg(var, initValue=ctx.unwrap(val))
        scope.store(name, ctx.wrap(var))
        iter_args.append(iter_arg)
        return_var_names.append(name)

    # Compile body into Stmt tree via nested IRBuilder
    body_stmt = ir.SeqStmts(body.span)
    loop_conds = []
    with InsertPoint(body_stmt), ctx.change_span(body.span), ctx.change_return_vars(return_var_names):
        is_positive = SymbolicScalar.check([loop.step > 0]) == SatStatus.SAT
        is_negative = SymbolicScalar.check([loop.step < 0]) == SatStatus.SAT
        if is_positive:
            loop_conds.append(loop_var >= loop.start)
        if is_negative:
            loop_conds.append(loop_var <= loop.start)

        if loop.batch:
            scope.varmap[loop_val.id] = (loop_var, loop.step * factor)
            dispatch_block(body, False)
        else:
            for i in range(factor):
                scope.varmap[loop_val.id] = loop_var + i * loop.step
                dispatch_block(body, False)
        if is_positive:
            loop_conds.append(loop_var + (factor - 1) * loop.step < loop.stop)
        if is_negative:
            loop_conds.append(loop_var + (factor - 1) * loop.step > loop.stop)
        _add_jump_stmt(ctx, body.jump)

    return_vars = []
    for name in return_var_names:
        var = ctx.create_var_like(name, ctx.unwrap(scope[name]))
        return_vars.append(var)
        scope.store(name, ctx.wrap(var))

    loop_attrs = {
        "parallel": loop.parallel,
        "submit_before_loop": loop.submit_before_loop,
        "_loop_conds": loop_conds,  # extra loop condition
        "_config_scope": pypto_impl.CurrentScope(),
        "unroll_times": factor,
    }

    for_stmt = ctx.create_for_stmt(
        loop_var.as_var(),
        ctx.unwrap(nstart),
        ctx.unwrap(nstop),
        ctx.unwrap(factor * loop.step),
        iter_args,
        body_stmt,
        return_vars,
        ctx.span,
        loop_attrs,
    )
    ctx.emit(for_stmt)


def _dyn_for(body: Block, loop: LoopRange, ctx: BuildContext):
    nstart = loop.start
    nstop = loop.stop

    for factor in loop.unroll_list:
        if factor == 1:
            nstop = loop.stop
        else:
            nstep = factor * loop.step
            nstop = nstart + (loop.stop - nstart) // nstep * nstep
        try:
            pypto_impl.BeginScope("loop", {}, ctx.span.filename, ctx.span.begin_line)
            _loop_unroll(body, loop, factor, nstart, nstop, ctx=ctx)
        finally:
            pypto_impl.EndScope()
        nstart = nstop


@impl("pil.loop")
def loop_impl(ctx: BuildContext, body: Block, loop):
    if isinstance(loop, LoopRange):
        ctx.loop_stack.append((loop.start, loop.stop, loop.step))
        _dyn_for(body, loop, ctx)
        ctx.loop_stack.pop()
    elif loop is not None:
        _static_for(body, loop)
    else:
        _static_while(body)


def _snapshot_tensor_lts(scope: Scope, names: set) -> list:
    """Capture storage for tensors this if may MOVE in-place.

    Python rebinding (``t = xv + yv``) is already rolled back by restoring
    ``scope.locals``. In-place MOVE is not: it mutates the Tensor object.
    Only snapshot pre-existing Tensors among ``store_names``, and skip objects
    that still hold the snapshotted logical tensor on restore. Calling
    ``Tensor.Move`` / ``set_logical_tensor`` uniquifies IR names via
    ``AssignStorage`` even when storage did not change.
    """
    snap = []
    seen = set()
    for name in names:
        val = scope.locals.get(name)
        if not isinstance(val, pypto.Tensor) or val.is_empty():
            continue
        tid = id(val)
        if tid in seen:
            continue
        seen.add(tid)
        snap.append((val, val.logical_tensor()))
    return snap


def _restore_tensor_lts(snap: list) -> None:
    for val, lt in snap:
        if not val.is_empty() and val.logical_tensor() is not lt:
            val.set_logical_tensor(lt)


def _if_else_stmt(cond, then_block: Block, else_block: Block, ctx: BuildContext):
    scope = Scope.current()

    saved = dict(scope.locals)
    store_names = then_block.store_names | else_block.store_names
    tensor_snap = _snapshot_tensor_lts(scope, store_names)
    yield_var_names = [
        name
        for name in sorted(store_names)
        if not is_opaque_value(scope.locals.get(name))
    ]

    def _trace_branch(branch: Block, branch_name: str, yield_vars_out: list):
        # Isolate the C++ config scope (vec/cube tile shapes, pass options, ...) so
        # that config mutations inside one branch do not leak into the sibling
        # branch trace or the code traced afterwards. Each branch only sees the
        # config state before the `if` statement plus its own mutations, matching
        # the per-path replay semantics of the legacy frontend.
        try:
            ctx.checkpoint()
            pypto_impl.BeginScope(branch_name, {}, branch.span.filename, branch.span.begin_line)
            branch_body = ir.SeqStmts(branch.span)
            with InsertPoint(branch_body), ctx.change_span(branch.span):
                dispatch_block(branch, False)
                yield_vars_out.extend(ctx.unwrap(scope[name]) for name in yield_var_names)
                _add_jump_stmt(ctx, branch.jump, list(yield_vars_out))
            return branch_body
        finally:
            pypto_impl.EndScope()
            ctx.restore()
            scope.locals = saved
            _restore_tensor_lts(tensor_snap)

    then_yield_vars = []
    then_body = _trace_branch(then_block, "pil.if_else_then", then_yield_vars)

    else_yield_vars = []
    else_body = _trace_branch(else_block, "pil.if_else_else", else_yield_vars)

    yield_vars = []
    for i, name in enumerate(yield_var_names):
        if ir.type_equal(then_yield_vars[i], else_yield_vars[i]):
            var = ctx.create_var_like(name, then_yield_vars[i])
        elif isinstance(then_yield_vars[i].type, ir.NoneType):
            var = ctx.create_var_like(name, else_yield_vars[i])
        elif isinstance(else_yield_vars[i].type, ir.NoneType):
            var = ctx.create_var_like(name, then_yield_vars[i])
        else:
            var = ctx.poison(name, then_block.span)
        yield_vars.append(var)
        scope.store(name, ctx.wrap(var))

    if_stmt = ctx.create_if_stmt(ctx.unwrap(cond), then_body, else_body, yield_vars, ctx.span)
    ctx.emit(if_stmt)


@impl("pil.if_else")
def if_else_impl(ctx: BuildContext, cond, then_block: Block, else_block: Block):
    # Concrete condition: interpret one branch, return early
    if isinstance(cond, pypto.SymbolicScalar):
        cond = cond.simplify()
        if cond.is_concrete():
            block = then_block if cond.concrete() else else_block
            return dispatch_block(block, True)
        else:
            _if_else_stmt(cond, then_block, else_block, ctx)
            return None
    else:
        block = then_block if cond else else_block
        return dispatch_block(block, True)


@impl("pil.fstring")
def format_impl(ctx, joined_str):
    ss = []
    for val in joined_str:
        if isinstance(val, tuple):
            val, conv, spec = val
            # Apply conversion first (mirrors Python f-string semantics)
            if conv == ord('a'):
                val = ascii(val)
            elif conv == ord('s'):
                val = str(val)
            elif conv == ord('r'):
                val = repr(val)
            # conv == -1 means no conversion; val stays as-is
            # Then apply format spec (may be "" for no spec)
            val = format(val, spec)
        ss.append(val)
    return "".join(ss)


@impl("pil.assert")
def assert_impl(ctx, cond, msg):
    if msg:
        assert cond, msg
    else:
        assert cond
