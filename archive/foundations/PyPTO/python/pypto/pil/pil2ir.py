# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import inspect

import pypto

from ..ir import SeqStmts
from . import ops  # noqa: F401  side-effect import: registers PIL op handlers
from .dispatcher import collect, dispatch_block
from .op_registry import dispatch
from .parser import ast2pil
from .pir import BuildContext, CollectContext, Function, InsertPoint, ReturnSignal, Scope


def _init_scope(func):
    root = Scope(list(func.global_vars))
    for name, val in zip(func.global_vars, func.global_values):
        root[name] = val
    return Scope(list(func.load_vars), parent=root)


def _init_arguments(args):
    all_args = {}
    tensor_args = []
    for key, val in args.items():
        if isinstance(val, pypto.Tensor):
            val = val.copy(name=key)
            tensor_args.append(val)
        all_args[key] = val
    return all_args, tensor_args


def pil2ir(func: Function, args: dict, tensor_args: list[pypto.Tensor]):
    scope = _init_scope(func)
    params = [x.logical_tensor() for x in tensor_args]
    body = SeqStmts(func.span)

    with BuildContext(func.span) as ctx, InsertPoint(body), scope.make_current():
        # Store function arguments
        for key, val in args.items():
            dispatch('pil.store', ctx, key, val)
        try:
            dispatch_block(func.body, True)
        except ReturnSignal:
            pass
        outs = [scope[t.name] for t in tensor_args]
        stmt = ctx.create_return_stmt([ctx.unwrap(t) for t in outs], func.span)
        ctx.emit(stmt)

    return ctx.create_function(func.name, params, [], body, func.span)


def collect_store_names(func: Function, all_args: dict):
    scope = _init_scope(func)
    body = SeqStmts(func.span)

    with CollectContext(func.span) as ctx, InsertPoint(body), scope.make_current():
        with pypto.options(), ops.apply_patches():
            for key, val in all_args.items():
                dispatch('pil.store', ctx, key, val)
            try:
                collect(func.body)
            except ReturnSignal:
                pass


def compile(pyfunc, *args, **kwargs):
    # `has_move=True` if Tensor.move or relative fill_/triu_/tril_ are used
    # we need a pre-pass to collect store names, as move is not treated as store
    # in python, otherwise we could skip it
    has_move = kwargs.pop("has_move", True)
    create_new_logical_tensor = kwargs.pop("create_new_logical_tensor", False)
    pypto.pypto_impl.ir.set_assemble_new_logical_tensor(create_new_logical_tensor)
    sig = inspect.signature(pyfunc)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # c++ addOperation still depends on function
    func = ast2pil(pyfunc)

    # collect store names first
    if has_move:
        pypto.pypto_impl.Reset()
        all_args, tensor_args = _init_arguments(bound.arguments)
        orig_print_graph = pypto.pypto_impl.GetPassDefaultConfig(pypto.pypto_impl.KEY_PRINT_GRAPH, False)
        try:
            pypto.pypto_impl.SetLogRotationEnabled(False)
            pypto.pypto_impl.SetPassDefaultConfig(pypto.pypto_impl.KEY_PRINT_GRAPH, False)
            with pypto.function("__entry__", *tensor_args):
                collect_store_names(func, all_args)
        finally:
            pypto.pypto_impl.SetLogRotationEnabled(True)
            pypto.pypto_impl.SetPassDefaultConfig(pypto.pypto_impl.KEY_PRINT_GRAPH, orig_print_graph)

    pypto.pypto_impl.Reset()
    # arguments maybe changed during `collect_store_names`, reinit it
    all_args, tensor_args = _init_arguments(bound.arguments)
    with pypto.function("__entry__", *tensor_args):
        func_def = pil2ir(func, all_args, tensor_args)
        # function input args still need to be valid, it'll be used later by tensor slot
        setattr(func_def, "__args__", tensor_args)
        return func_def
