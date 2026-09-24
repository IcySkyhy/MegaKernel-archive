# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import ast
from contextlib import contextmanager
import inspect
import operator
import re
from typing import Any, Callable, NoReturn, Optional, Union

from pypto import ir

from .pir import Block, Call, DoubleStarred, Function, Jump, LoopKind, Operand, Starred, Value, in_, not_in


class Source:
    def __init__(self, pyfunc: Callable):
        pyfunc = getattr(pyfunc, "__wrapped__", pyfunc)
        self.pyfunc = pyfunc
        source_lines, lineno = inspect.getsourcelines(pyfunc)
        self.filename = inspect.getfile(pyfunc)

        source = "if True:\n " + " ".join(source_lines)
        mod = ast.parse(source)
        if not isinstance(mod.body[0], ast.If):
            raise ValueError("Top level must be if statement")

        self.func_def = mod.body[0].body[0]
        if not isinstance(self.func_def, ast.FunctionDef):
            raise ValueError("Top level must be function definition")
        self._fix_line_numbers(lineno)

    def get_span(self, node: ast.AST):
        if hasattr(node, "lineno"):
            return ir.Span(
                self.filename,
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )
        return ir.Span(self.filename, 0, 0)

    def _fix_line_numbers(self, lineno):
        for node in ast.walk(self.func_def):
            if hasattr(node, "lineno"):
                node.lineno += lineno - 2
                node.end_lineno += lineno - 2
                node.col_offset -= 1
                node.end_col_offset -= 1


class _Context:
    def __init__(self, source: Source, entry_point: bool = True):
        self.source = source
        self.entry_point = entry_point
        self.value_id = 0
        self.block_id = 0
        self.current_span = ir.Span.unknown()
        self.loop_kinds = []
        self._current_block: Optional[Block] = None

    @property
    def current_block(self) -> Block:
        if self._current_block is None:
            raise ValueError("Current block is None")
        return self._current_block

    @staticmethod
    def raise_error(node: ast.AST, msg: str = "") -> NoReturn:
        if msg:
            msg = f": {msg}"
        raise SyntaxError(f"Unsupported syntax {type(node)}{msg}")

    def make_temp(self):
        self.value_id += 1
        return Value(self.value_id)

    @contextmanager
    def span(self, span: Union[ast.AST, ir.Span]):
        old = self.current_span
        self.current_span = span if isinstance(span, ir.Span) else self.source.get_span(span)
        try:
            yield
        finally:
            self.current_span = old

    @contextmanager
    def new_block(self, args: tuple[Operand, ...] = ()):
        self.block_id += 1
        new_block = Block(id=self.block_id, args=args, calls=[], result=None, span=self.current_span)
        old, self._current_block = self._current_block, new_block
        try:
            yield new_block
            if self.current_block.jump is None:
                raise ValueError("Block statement must have a jump")
        finally:
            self._current_block = old
            if old is not None:
                old.store_names.update(new_block.store_names)
                old.load_names.update(new_block.load_names)

    def store(self, name: str, value: Union[Value, Any]):
        self.current_block.store_names.add(name)
        return self.call("pil.store", (name, value))

    def mark_store(self, name: str):
        self.current_block.store_names.add(name)

    def load(self, name: str):
        self.current_block.load_names.add(name)
        return self.call("pil.load", (name,))

    def call(self, callee, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = self.make_temp()
        call = Call(ret, callee, args, kwargs, self.current_span)
        self.current_block.calls.append(call)
        return ret

    def call_void(self, callee, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        call = Call(None, callee, args, kwargs, self.current_span)
        self.current_block.calls.append(call)

    def set_jump(self, jump: Jump, result: Optional[Value] = None):
        if self.current_block.jump is not None:
            raise ValueError("Block statement must have only one jump")
        self.current_block.jump = jump
        self.current_block.jump_loc = self.current_span
        self.current_block.result = result


_builtin_ops = {
    # unary ops
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    # binary ops
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
    # cmp ops
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: in_,
    ast.NotIn: not_in,
}


def _pypto_loop_mode(node: ast.AST) -> Optional[str]:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pypto"
        and node.func.attr in ("loop", "loop_unroll")
    ):
        return node.func.attr
    return None


def _parse_params(ctx, func: Union[ast.FunctionDef, ast.Lambda], defaults, kwdefaults):
    if func.args.vararg or func.args.kwarg:
        ctx.raise_error(func, "vararg and kwarg not supported")

    positional = [arg.arg for arg in func.args.posonlyargs + func.args.args]
    kwonly = [arg.arg for arg in func.args.kwonlyargs]

    defaults = defaults or ()
    kwdefaults = kwdefaults or {}
    defvals = [None] * (len(positional) + len(kwonly))
    defvals[len(positional) - len(defaults):len(positional)] = defaults
    defvals[len(positional):] = [kwdefaults.get(name, None) for name in kwonly]

    return tuple(positional + kwonly), tuple(defvals)


class Parser:
    @staticmethod
    def visit_name(name: ast.Name, ctx: _Context):
        if not isinstance(name.ctx, ast.Load):
            ctx.raise_error(name)
        return ctx.load(name.id)

    @staticmethod
    def visit_constant(node: ast.Constant, ctx: _Context):
        return ctx.call("pil.const", (node.value,))

    @staticmethod
    def visit_continue(stmt: ast.Continue, ctx: _Context):
        if ctx.loop_kinds[-1] is LoopKind.DYNAMIC_FOR:
            ctx.raise_error(stmt, "continue is not supported in pypto.loop")
        ctx.set_jump(Jump.CONTINUE)

    @staticmethod
    def visit_break(stmt: ast.Break, ctx: _Context):
        if ctx.loop_kinds[-1] is LoopKind.DYNAMIC_FOR:
            ctx.raise_error(stmt, "break is not supported in pypto.loop")
        ctx.set_jump(Jump.BREAK)

    # expressions
    def visit_bool_op(self, boolop: ast.BoolOp, ctx: _Context):
        if len(boolop.values) < 2:
            ctx.raise_error(boolop, "At least two operands are required for boolop")

        cond0 = self.visit(boolop.values[0], ctx)
        rest = boolop.values[1:]
        if isinstance(boolop.op, ast.And):
            with ctx.new_block() as then_block:
                if len(rest) > 1:
                    sub = ast.BoolOp(op=boolop.op, values=rest)
                    cond1 = self.visit_bool_op(sub, ctx)
                else:
                    cond1 = self.visit(rest[0], ctx)
                ctx.set_jump(Jump.END_BRANCH, cond1)
            with ctx.new_block() as else_block:
                ctx.set_jump(Jump.END_BRANCH, cond0)

            return ctx.call("pil.if_else", (cond0, then_block, else_block))
        elif isinstance(boolop.op, ast.Or):
            with ctx.new_block() as then_block:
                ctx.set_jump(Jump.END_BRANCH, cond0)
            with ctx.new_block() as else_block:
                if len(rest) > 1:
                    sub = ast.BoolOp(op=boolop.op, values=rest)
                    cond1 = self.visit_bool_op(sub, ctx)
                else:
                    cond1 = self.visit(rest[0], ctx)
                ctx.set_jump(Jump.END_BRANCH, cond1)

            return ctx.call("pil.if_else", (cond0, then_block, else_block))
        else:
            ctx.raise_error(boolop)

    def visit_call(self, stmt: ast.Call, ctx: _Context):
        callee = self.visit(stmt.func, ctx)
        args = tuple(self.visit(arg, ctx) for arg in stmt.args)
        kwargs = tuple(
            (None, DoubleStarred(self.visit(kw.value, ctx))) if kw.arg is None else (kw.arg, self.visit(kw.value, ctx))
            for kw in stmt.keywords
        )
        return ctx.call(callee, args, kwargs)

    def visit_unary_op(self, unary: ast.UnaryOp, ctx: _Context):
        ops = _builtin_ops.get(type(unary.op))
        if ops is None:
            ctx.raise_error(unary)
        op0 = self.visit(unary.operand, ctx)
        return ctx.call(ops, (op0,))

    def visit_bin_op(self, bop: ast.BinOp, ctx: _Context):
        ops = _builtin_ops.get(type(bop.op))
        if ops is None:
            ctx.raise_error(bop)
        op0 = self.visit(bop.left, ctx)
        op1 = self.visit(bop.right, ctx)
        return ctx.call(ops, (op0, op1))

    def visit_compare(self, cmp: ast.Compare, ctx: _Context):
        ops = _builtin_ops.get(type(cmp.ops[0]))
        if ops is None:
            ctx.raise_error(cmp)

        op0 = self.visit(cmp.left, ctx)
        op1 = self.visit(cmp.comparators[0], ctx)
        cond0 = ctx.call(ops, (op0, op1))
        if len(cmp.ops) == 1:
            return cond0

        with ctx.new_block() as then_block:
            sub = ast.Compare(
                left=cmp.comparators[0],
                ops=cmp.ops[1:],
                comparators=cmp.comparators[1:],
            )
            cond1 = self.visit_compare(sub, ctx)
            ctx.set_jump(Jump.END_BRANCH, cond1)

        with ctx.new_block() as else_block:
            ctx.set_jump(Jump.END_BRANCH, cond0)

        return ctx.call("pil.if_else", (cond0, then_block, else_block))

    def visit_attribute(self, node: ast.Attribute, ctx: _Context):
        value = self.visit(node.value, ctx)
        return ctx.call(getattr, (value, node.attr))

    def visit_joined_str(self, node: ast.JoinedStr, ctx: _Context):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                val = self.visit(v.value, ctx)
                spec = ""
                if v.format_spec:
                    spec = self.visit(v.format_spec, ctx)
                parts.append((val, v.conversion, spec))
            else:
                ctx.raise_error(v)
        return ctx.call("pil.fstring", (parts,))

    def visit_tuple(self, node: ast.Tuple, ctx: _Context):
        values = tuple(self.visit(v, ctx) for v in node.elts)
        return ctx.call(tuple, (values,))

    def visit_list(self, node: ast.List, ctx: _Context):
        values = list(self.visit(v, ctx) for v in node.elts)
        return ctx.call(list, (values,))

    def visit_set(self, node: ast.Set, ctx: _Context):
        values = list(self.visit(v, ctx) for v in node.elts)
        return ctx.call(set, (values,))

    def visit_dict(self, node: ast.Dict, ctx: _Context):
        pairs = []
        for k, v in zip(node.keys, node.values):
            value = self.visit(v, ctx)
            if k is None:
                pairs.append((None, DoubleStarred(value)))
            else:
                pairs.append((self.visit(k, ctx), value))
        return ctx.call(dict, (pairs,))

    def visit_subscript(self, node: ast.Subscript, ctx: _Context):
        value = self.visit(node.value, ctx)
        index = self.visit(node.slice, ctx)
        return ctx.call(operator.getitem, (value, index))

    def visit_slice(self, node: ast.Slice, ctx: _Context):
        start = self.visit(node.lower, ctx) if node.lower else None
        stop = self.visit(node.upper, ctx) if node.upper else None
        step = self.visit(node.step, ctx) if node.step else None
        return ctx.call(slice, (start, stop, step))

    def visit_if_exp(self, node: ast.IfExp, ctx: _Context):
        cond = self.visit(node.test, ctx)
        with ctx.new_block() as then_block:
            value = self.visit(node.body, ctx)
            ctx.set_jump(Jump.END_BRANCH, value)
        with ctx.new_block() as else_block:
            value = self.visit(node.orelse, ctx)
            ctx.set_jump(Jump.END_BRANCH, value)
        return ctx.call("pil.if_else", (cond, then_block, else_block))

    def visit_starred(self, node: ast.Starred, ctx: _Context):
        if not isinstance(node.ctx, ast.Load):
            ctx.raise_error(node)
        return Starred(self.visit(node.value, ctx))

    def visit_while(self, stmt: ast.While, ctx: _Context):
        if stmt.orelse:
            ctx.raise_error(stmt, "while-else not supported")

        with ctx.new_block() as body:
            cond = self.visit(stmt.test, ctx)
            with ctx.new_block() as then_block:
                ctx.set_jump(Jump.END_BRANCH)
            with ctx.new_block() as else_block:
                ctx.set_jump(Jump.BREAK)
            ctx.call_void("pil.if_else", (cond, then_block, else_block))

            ctx.loop_kinds.append(LoopKind.WHILE)
            self._stmts(stmt.body, ctx)
            if body.jump is None:
                ctx.set_jump(Jump.CONTINUE)
            ctx.loop_kinds.pop()

        ctx.call_void("pil.loop", (body, None))

    def visit_for(self, node: ast.For, ctx: _Context):
        if node.orelse:
            ctx.raise_error(node, "for-else not supported")

        loop_mode = _pypto_loop_mode(node.iter)
        kind = LoopKind.DYNAMIC_FOR if loop_mode else LoopKind.FOR
        if loop_mode == "loop_unroll":
            if not (isinstance(node.target, (ast.Tuple, ast.List)) and len(node.target.elts) == 2):
                ctx.raise_error(
                    node,
                    "pypto.loop_unroll must unpack exactly two targets (index, count), "
                    "e.g. 'for i, k in pypto.loop_unroll(...)'",
                )

        iter = self.visit(node.iter, ctx)
        ctx.loop_kinds.append(kind)
        loop_var = ctx.make_temp()
        with ctx.new_block(args=(loop_var,)) as body:
            self._do_assign(node.target, loop_var, ctx)
            self._stmts(node.body, ctx)
            if body.jump is None:
                ctx.set_jump(Jump.CONTINUE)
        ctx.loop_kinds.pop()

        ctx.call_void("pil.loop", (body, iter))

    def visit_if(self, stmt: ast.If, ctx: _Context):
        cond = self.visit(stmt.test, ctx)
        with ctx.new_block() as then_block:
            self._stmts(stmt.body, ctx)
            if then_block.jump is None:
                ctx.set_jump(Jump.END_BRANCH, None)
        with ctx.new_block() as else_block:
            self._stmts(stmt.orelse, ctx)
            if else_block.jump is None:
                ctx.set_jump(Jump.END_BRANCH, None)
        return ctx.call("pil.if_else", (cond, then_block, else_block))

    def visit_return(self, stmt: ast.Return, ctx: _Context):
        for kind in ctx.loop_kinds:
            if kind is LoopKind.DYNAMIC_FOR:
                ctx.raise_error(stmt, "return is not supported in pypto.loop")

        value = self.visit(stmt.value, ctx) if stmt.value else None
        if ctx.entry_point:
            ctx.raise_error(stmt, "return statement in entry point")
        else:
            ctx.store("$retval", value)
            ctx.set_jump(Jump.RETURN, value)

    def visit_assign(self, stmt: ast.Assign, ctx: _Context):
        value = self.visit(stmt.value, ctx)
        for target in reversed(stmt.targets):
            self._do_assign(target, value, ctx)

    def visit_aug_assign(self, aug: ast.AugAssign, ctx: _Context):
        ops = _builtin_ops.get(type(aug.op))
        if ops is None:
            ctx.raise_error(aug)

        if isinstance(aug.target, ast.Name):
            lhs = ctx.load(aug.target.id)
            rhs = self.visit(aug.value, ctx)
            res = ctx.call(ops, (lhs, rhs))
            ctx.store(aug.target.id, res)
        elif isinstance(aug.target, ast.Subscript):
            obj = self.visit(aug.target.value, ctx)
            key = self.visit(aug.target.slice, ctx)
            rhs = self.visit(aug.value, ctx)
            old_value = ctx.call(operator.getitem, (obj, key))
            new_value = ctx.call(ops, (old_value, rhs))
            ctx.call_void(operator.setitem, (obj, key, new_value))
        elif isinstance(aug.target, ast.Attribute):
            obj = self.visit(aug.target.value, ctx)
            rhs = self.visit(aug.value, ctx)
            old_value = ctx.call(getattr, (obj, aug.target.attr))
            new_value = ctx.call(ops, (old_value, rhs))
            ctx.call_void(setattr, (obj, aug.target.attr, new_value))
        else:
            ctx.raise_error(aug.target)

    def visit_ann_assign(self, stmt: ast.AnnAssign, ctx: _Context):
        if stmt.value is not None:
            value = self.visit(stmt.value, ctx)
            self._do_assign(stmt.target, value, ctx)

    def visit_expr(self, stmt: ast.Expr, ctx: _Context):
        return self.visit(stmt.value, ctx)

    def visit_pass(self, stmt: ast.Pass, ctx: _Context):
        pass

    def visit_delete(self, stmt: ast.Delete, ctx: _Context):
        for target in stmt.targets:
            self._do_delete(target, ctx)

    def visit_function_def(self, stmt: ast.FunctionDef, ctx: _Context):
        defaults = []
        for d in stmt.args.defaults:
            defaults.append(self.visit(d, ctx))

        kwdefaults = {}
        for arg, d in zip(stmt.args.kwonlyargs, stmt.args.kw_defaults):
            if d is not None:
                kwdefaults[arg.arg] = self.visit(d, ctx)
            else:
                kwdefaults[arg.arg] = None

        new_ctx = _Context(ctx.source, entry_point=False)
        parser = Parser()
        with new_ctx.span(stmt), new_ctx.new_block() as blk:
            parser.parse(stmt.body, new_ctx)
            if blk.jump is None:
                new_ctx.set_jump(Jump.RETURN)

        params, param_defaults = _parse_params(ctx, stmt, defaults, kwdefaults)
        func = Function(
            name=stmt.name,
            span=blk.span,
            signature=inspect.Signature(),
            body=blk,
            load_vars=tuple(sorted(blk.load_names)),
            store_vars=tuple(sorted(blk.store_names)),
            global_vars=(),
            global_values=(),
            params=params,
            param_defaults=param_defaults,
        )
        ctx.store(stmt.name, func)

    def visit_lambda(self, node: ast.Lambda, ctx: _Context):
        defaults = []
        for d in node.args.defaults:
            defaults.append(self.visit(d, ctx))

        kwdefaults = {}
        for arg, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if d is not None:
                kwdefaults[arg.arg] = self.visit(d, ctx)
            else:
                kwdefaults[arg.arg] = None

        new_ctx = _Context(ctx.source, entry_point=False)
        with new_ctx.span(node), new_ctx.new_block() as blk:
            value = self.visit(node.body, new_ctx)
            new_ctx.store("$retval", value)
            new_ctx.set_jump(Jump.RETURN, value)

        params, param_defaults = _parse_params(ctx, node, defaults, kwdefaults)
        func = Function(
            name="<lambda>",
            span=blk.span,
            signature=inspect.Signature(),
            body=blk,
            load_vars=tuple(sorted(blk.load_names)),
            store_vars=tuple(sorted(blk.store_names)),
            global_vars=(),
            global_values=(),
            params=params,
            param_defaults=param_defaults,
        )
        return func

    def comprehension(self, node, ctx, acc_init, innermost, name):
        acc = "$c_acc"
        body: list[ast.stmt] = [innermost]
        # wrap innermost -> outermost; every generator is uniform
        for gen in reversed(node.generators):
            for cond in reversed(gen.ifs):
                body = [ast.If(test=cond, body=body, orelse=[])]
            body = [ast.For(target=gen.target, iter=gen.iter, body=body, orelse=[])]

        stmts: list[ast.stmt] = [ast.Assign([ast.Name(acc, ast.Store())], acc_init)]
        stmts += body
        stmts += [ast.Return(ast.Name(acc, ast.Load()))]
        ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[]))

        new_ctx = _Context(ctx.source, entry_point=False)
        with new_ctx.span(node), new_ctx.new_block() as blk:
            self._stmts(stmts, new_ctx)
            if blk.jump is None:
                new_ctx.set_jump(Jump.RETURN)

        func = Function(
            name=name,
            span=blk.span,
            signature=inspect.Signature(),
            body=blk,
            load_vars=tuple(sorted(blk.load_names)),
            store_vars=tuple(sorted(blk.store_names)),
            global_vars=(),
            global_values=(),
        )
        return ctx.call(func, ())

    def visit_list_comp(self, node: ast.ListComp, ctx: _Context):
        acc = "$c_acc"
        innermost = ast.Expr(
            ast.Call(func=ast.Attribute(ast.Name(acc, ast.Load()), "append", ast.Load()), args=[node.elt], keywords=[])
        )
        return self.comprehension(node, ctx, ast.List([], ast.Load()), innermost, "<listcomp>")

    def visit_set_comp(self, node: ast.SetComp, ctx: _Context):
        acc = "$c_acc"
        innermost = ast.Expr(
            ast.Call(func=ast.Attribute(ast.Name(acc, ast.Load()), "add", ast.Load()), args=[node.elt], keywords=[])
        )
        return self.comprehension(
            node, ctx, ast.Call(ast.Name("set", ast.Load()), [ast.List([], ast.Load())], []), innermost, "<setcomp>"
        )

    def visit_dict_comp(self, node: ast.DictComp, ctx: _Context):
        acc = "$c_acc"
        innermost = ast.Assign(
            targets=[ast.Subscript(value=ast.Name(acc, ast.Load()), slice=node.key, ctx=ast.Store())], value=node.value
        )
        return self.comprehension(node, ctx, ast.Dict(keys=[], values=[]), innermost, "<dictcomp>")

    def visit_assert(self, stmt: ast.Assert, ctx: _Context):
        cond = self.visit(stmt.test, ctx)
        msg = self.visit(stmt.msg, ctx) if stmt.msg else None
        ctx.call_void("pil.assert", (cond, msg))

    def visit_raise(self, stmt: ast.Raise, ctx: _Context):
        if stmt.exc is None:
            ctx.raise_error(stmt, "bare raise is not supported")

        exc = self.visit(stmt.exc, ctx)
        cause = self.visit(stmt.cause, ctx) if stmt.cause else None
        ctx.call_void("pil.raise", (exc, cause))

    def visit(self, node: ast.AST, ctx: _Context):
        # CamelCase -> snake_case
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", node.__class__.__name__).lower()
        visitor = getattr(self, "visit_" + name)
        if visitor is None:
            ctx.raise_error(node)
        return visitor(node, ctx)

    def parse(self, stmts: list[ast.stmt], ctx: _Context):
        self._stmts(stmts, ctx)

    def _stmts(self, stmts: list[ast.stmt], ctx: _Context):
        for stmt in stmts:
            with ctx.span(stmt):
                self.visit(stmt, ctx)
                if ctx.current_block.jump is not None:
                    break

    def _do_assign(self, target, value, ctx: _Context):
        with ctx.span(target):
            if isinstance(target, ast.Name):
                ctx.store(target.id, value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for i, elm in enumerate(target.elts):
                    val_i = ctx.call(operator.getitem, (value, i))
                    self._do_assign(elm, val_i, ctx)
            elif isinstance(target, ast.Subscript):
                obj = self.visit(target.value, ctx)
                key = self.visit(target.slice, ctx)
                # `a[:] = x` mark `a` as stored value
                if isinstance(target.value, ast.Name) and self._is_empty_slice(target.slice):
                    ctx.mark_store(target.value.id)
                ctx.call_void(operator.setitem, (obj, key, value))
            elif isinstance(target, ast.Attribute):
                obj = self.visit(target.value, ctx)
                ctx.call_void(setattr, (obj, target.attr, value))
            else:
                ctx.raise_error(target)

    def _do_delete(self, target, ctx: _Context):
        with ctx.span(target):
            if isinstance(target, ast.Name):
                self._do_assign(target, None, ctx)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._do_delete(elt, ctx)
            elif isinstance(target, ast.Subscript):
                obj = self.visit(target.value, ctx)
                key = self.visit(target.slice, ctx)
                ctx.call_void(operator.delitem, (obj, key))
            elif isinstance(target, ast.Attribute):
                obj = self.visit(target.value, ctx)
                ctx.call_void(delattr, (obj, target.attr))
            else:
                ctx.raise_error(target)

    def _is_empty_slice(self, node: ast.AST):
        if isinstance(node, ast.Slice):
            return node.lower is None and node.upper is None and node.step is None
        if isinstance(node, ast.Tuple):
            return all(self._is_empty_slice(e) for e in node.elts)
        return False


def ast2pil(pyfunc, entry_point: bool = True):
    source = Source(pyfunc)
    if not isinstance(source.func_def, ast.FunctionDef):
        raise ValueError("ast2pil must be called with a function definition")

    ctx = _Context(source, entry_point=entry_point)
    parser = Parser()
    with ctx.span(source.func_def), ctx.new_block() as blk:
        parser.parse(source.func_def.body, ctx)
        if blk.jump is None:
            ctx.set_jump(Jump.RETURN)

    builtins = pyfunc.__globals__["__builtins__"]
    if not isinstance(builtins, dict):
        builtins = vars(builtins)
    envs = dict(builtins)
    envs.update(pyfunc.__globals__)

    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    code = pyfunc.__code__
    if pyfunc.__closure__ is not None:
        for var, cell in zip(code.co_freevars, pyfunc.__closure__):
            envs[var] = cell.cell_contents

    global_vars = tuple(sorted(envs.keys()))
    global_values = [envs[k] for k in global_vars]

    params, param_defaults = _parse_params(ctx, source.func_def, pyfunc.__defaults__, pyfunc.__kwdefaults__)

    return Function(
        name=pyfunc.__name__,
        span=blk.span,
        signature=inspect.signature(pyfunc),
        body=blk,
        load_vars=tuple(sorted(blk.load_names)),
        store_vars=tuple(sorted(blk.store_names)),
        global_vars=tuple(global_vars),
        global_values=tuple(global_values),
        params=params,
        param_defaults=param_defaults,
    )
