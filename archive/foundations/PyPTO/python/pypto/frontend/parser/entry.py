#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Entry point for PTO Script Parser.

This module provides the main entry points for parsing PTO scripts,
including the parse function and JIT decorator.
"""

from enum import IntEnum
import inspect
import itertools
import os
from typing import Any, Callable, Optional, Union

import torch

import pypto
from pypto import pypto_impl
from pypto._utils import get_dtensor_type, get_npu_tensor_format, get_torch_npu
from pypto.cost_model import _cost_model_run_once_data_from_host
from pypto.error import FeError, _catch_and_wrap_error
from pypto.frontend.parser.diagnostics import Source
from pypto.frontend.parser.parser import NestedFunctionMarker, Parser
from pypto.logging import log_debug
from pypto.pil.compile_pipeline import compile_new_ir
from pypto.runtime import _pto_verify_datas, setup_verify_data


def _default_globals() -> dict[str, Any]:
    """Get the default global variables for parsing.

    Returns
    -------
    dict[str, Any]
        Dictionary containing default global variables (pto module).
    """
    return {
        "pypto": pypto,
    }


class RunMode(IntEnum):
    NPU = 0
    SIM = 1


def parse(program: Source, extra_vars: Optional[dict[str, Any]] = None) -> Any:
    """Parse a PTO script program.

    This function parses a PTO script source and returns the parsed result,
    typically a pypto.Function object.

    Parameters
    ----------
    program : Source
        The source code to parse.
    extra_vars : Optional[dict[str, Any]], optional
        Additional variables to make available during parsing.
        These are merged with default globals (pto module).

    Returns
    -------
    Any
        The parsed result, typically a pypto.Function object.

    Examples
    --------
    >>> source = Source("def foo(): ...")
    >>> func = parse(source)
    >>> isinstance(func, pypto.Function)
    True
    """
    if extra_vars is None:
        merged_vars = _default_globals()
    else:
        merged_vars = {**_default_globals(), **extra_vars}
    parser = Parser(program, merged_vars)
    parser.parse()
    return parser.execute()


def _pto_to_tensor_data(
    tensors: list[pypto.Tensor],
) -> list[pypto_impl.DeviceTensorData]:
    """Convert PTO tensors to device tensor data for runtime execution.

    This helper function creates DeviceTensorData objects that encapsulate
    the tensor metadata (dtype, pointer, shape) required by the backend runtime.

    Parameters
    ----------
    tensors : list[pypto.Tensor]
        List of PTO tensors to convert.

    Returns
    -------
    list[pypto_impl.DeviceTensorData]
        List of device tensor data objects ready for runtime execution.

    Raises
    ------
    RuntimeError
        If any tensor's ori_shape is not specified.
    """
    datas = []
    for t in tensors:
        if t.ori_shape is None:
            raise FeError(RuntimeError("The ori_shape of the tensor is not specified."))
        data = pypto_impl.DeviceTensorData(
            t.dtype,
            t.data_ptr,
            list(t.ori_shape),
        )
        datas.append(data)
    return datas


def _current_stream():
    """Retrieve current NPU stream (0 if torch.npu is not present)"""
    npu = getattr(torch, 'npu', None)
    if npu:
        return npu.current_stream().npu_stream
    else:
        return 0


class JitCallableWrapper:
    """Callable wrapper for pypto.Function that integrates frontend parsing with runtime execution.

    This class wraps a pypto.Function and makes it callable with torch tensors,
    integrating the frontend JIT compilation with the runtime execution mechanism.
    Parsing is deferred until the first __call__ invocation (lazy mode), allowing
    dynamic shape binding and cost model evaluation before compilation.

    The wrapper maintains the original function's metadata (__name__, __doc__) and
    provides transparent execution by handling tensor conversion, workspace allocation,
    and device management automatically. It also supports compilation caching to avoid
    redundant recompilation, with cache key generated from compilation-related options.

    Attributes
    ----------
    _pto_function : Optional[pypto.Function]
        The parsed PTO function (None until first call in lazy mode).
    _original_func : Callable
        The original Python function being wrapped.
    _handler : Optional[int]
        The backend runtime handler (None until first call in lazy mode).
    _is_compiled : bool
        Flag indicating whether the function has been compiled.
    _parser : Optional[Parser]
        Parser instance stored for lazy parsing.
    _codegen_options : Optional[dict[str, Any]]
        Options for code generation.
    _host_options : Optional[dict[str, Any]]
        Options for host configuration.
    _runtime_options : Optional[dict[str, Any]]
        Options for runtime execution (including run_mode: NPU or SIM).
    _pass_options : Optional[dict[str, Any]]
        Options for compiler passes.
    _verify_options : Optional[dict[str, Any]]
        Options for verification.
    _debug_options : Optional[dict[str, Any]]
        Options for debugging (including runtime_debug_mode).
    _captured_locals : Optional[dict[str, Any]]
        Captured local variables from the original function's scope (copied to avoid reference issues).
    _kernel_module_cache : dict[tuple, Any]
        Class-level global cache for KernelModule instances, keyed by compilation cache key.
    kmodule : Any
        KernelModule instance associated with this wrapper (from cache if enabled).
    """

    # Global KernelModule cache
    _kernel_module_cache: dict[tuple, Any] = {}
    _global_func_idx_generator = itertools.count(0)

    _dtype_dict = {
        "torch.int8": pypto.DataType.DT_INT8,
        "torch.int16": pypto.DataType.DT_INT16,
        "torch.short": pypto.DataType.DT_INT16,
        "torch.int32": pypto.DataType.DT_INT32,
        "torch.int": pypto.DataType.DT_INT32,
        "torch.int64": pypto.DataType.DT_INT64,
        "torch.long": pypto.DataType.DT_INT64,
        "torch.float16": pypto.DataType.DT_FP16,
        "torch.half": pypto.DataType.DT_FP16,
        "torch.float32": pypto.DataType.DT_FP32,
        "torch.float": pypto.DataType.DT_FP32,
        "torch.bfloat16": pypto.DataType.DT_BF16,
        "torch.float8_e4m3fn": pypto.DataType.DT_FP8E4M3,
        "torch.float8_e5m2": pypto.DataType.DT_FP8E5M2,
        "torch.float8_e8m0fnu": pypto.DataType.DT_FP8E8M0,
        "torch.uint8": pypto.DataType.DT_UINT8,
        "torch.uint16": pypto.DataType.DT_UINT16,
        "torch.uint32": pypto.DataType.DT_UINT32,
        "torch.uint64": pypto.DataType.DT_UINT64,
        "torch.bool": pypto.DataType.DT_BOOL,
    }

    _special_dtype_dict = {
        "torch.uint8": [
            pypto.DataType.DT_HF8,
            pypto.DataType.DT_FP4_E2M1X2,
            pypto.DataType.DT_FP4_E1M2X2,
            pypto.DataType.DT_FP4_E2M1,
            pypto.DataType.DT_FP4_E1M2,
        ]
    }

    _format_dict = {
        "ND": pypto.TileOpFormat.TILEOP_ND,
        "NZ": pypto.TileOpFormat.TILEOP_NZ,
    }

    def __init__(
        self,
        pto_function: Optional[pypto.Function],
        original_func: Callable,
        handler: Optional[int],
        codegen_options: Optional[dict[str, Any]] = None,
        host_options: Optional[dict[str, Any]] = None,
        pass_options: Optional[dict[str, Any]] = None,
        runtime_options: Optional[dict[str, Any]] = None,
        verify_options: Optional[dict[str, Any]] = None,
        debug_options: Optional[dict[str, Any]] = None,
        captured_locals: Optional[dict[str, Any]] = None,
        new_ir: bool = True,
    ):
        """Initialize the JIT callable wrapper with compilation and runtime configurations.

        Parameters
        ----------
        pto_function : Optional[pypto.Function]
            The parsed PTO function (None initially in lazy mode).
        original_func : Callable
            The original Python function to be wrapped and compiled.
        handler : Optional[int]
            The backend runtime handler (None initially in lazy mode).
        codegen_options : Optional[dict[str, Any]], optional
            Options for code generation configuration.
        host_options : Optional[dict[str, Any]], optional
            Options for host environment configuration.
        pass_options : Optional[dict[str, Any]], optional
            Options for compiler pass configuration.
        runtime_options : Optional[dict[str, Any]], optional
            Options for runtime execution (e.g., run_mode: NPU or SIM). Defaults to empty dict if None.
        verify_options : Optional[dict[str, Any]], optional
            Options for verification during compilation.
        debug_options : Optional[dict[str, Any]], optional
            Options for debugging (e.g., runtime_debug_mode).
        captured_locals : Optional[dict[str, Any]], optional
            Local variables captured from the original function's scope (copied to a new dict
            to prevent external modification). Defaults to None.
        new_ir : bool, optional
            Whether to compile through the new IR pipeline. Defaults to False.
        """
        self._pto_function = pto_function
        self._original_func = original_func
        self._handler = handler
        self._is_compiled = pto_function is not None
        self._parser = None  # Store parser for lazy parsing
        self._captured_locals = None if captured_locals is None else dict(captured_locals)

        # Handling options
        self._codegen_options = None if codegen_options is None else dict(codegen_options)
        self._host_options = None if host_options is None else dict(host_options)
        self._runtime_options = runtime_options or {}
        self._pass_options = None if pass_options is None else dict(pass_options)
        self._verify_options = None if verify_options is None else dict(verify_options)
        self._debug_options = None if debug_options is None else dict(debug_options)
        self._use_new_ir = new_ir

        self._set_run_mode()
        self.kwargs = None
        self._cached_signature = self._get_signature()
        self._cached_non_tensor_defaults: dict[str, Any] = self._extract_non_tensor_defaults()
        self._cache_hashes: tuple = self._compute_cache_hashes()

        # Copy metadata from the original function
        if hasattr(original_func, "__name__"):
            self.__name__ = original_func.__name__
        if hasattr(original_func, "__doc__"):
            self.__doc__ = original_func.__doc__

        # kmodule is created lazily in __call__ with cache key including non_tensor_values
        self.kmodule = None

    @_catch_and_wrap_error("call JIT function")
    def __call__(self, *args, **kwargs):
        """Execute the function with torch tensors and optional non-tensor parameters.

        All tensors (including output) are passed as arguments; use out parameter
        and out.move() inside the kernel. This method returns None; caller holds
        output tensors.

        Parameters
        ----------
        *args : Union[torch.Tensor, Any]
            First N arguments must be torch.Tensor (matching tensor params).
            Remaining arguments are non-tensor values (matching non-tensor params in order).
        **kwargs : Any
            Non-tensor parameters as key=value. Overrides positional non-tensor args.

        Returns
        -------
        None
            User holds output tensor(s) passed as arguments; no return value.
        """
        in_tensors, non_tensor_values, input_tensor_defs = self._parse_call_args(args, kwargs)
        self._validate_exact_torch_tensors(in_tensors)
        self._get_or_create_kmodule(non_tensor_values)
        self._execute_kernel(in_tensors, input_tensor_defs)

        return None

    @property
    def function(self) -> Optional[pypto.Function]:
        """Get the underlying pypto.Function.

        Returns
        -------
        Optional[pypto.Function]
            The compiled PTO function, or None if not yet compiled (lazy mode).
        """
        return self._pto_function

    @property
    def handler(self) -> Optional[int]:
        """Get the runtime handler.

        Returns
        -------
        Optional[int]
            The backend runtime handler, or None if not yet compiled (lazy mode).
        """
        return self._handler

    @staticmethod
    def alloc(size):
        """Allocate NPU int8 memory and return its data pointer"""
        return torch.empty(size, dtype=torch.int8, device='npu').data_ptr()

    @staticmethod
    def _validate_exact_torch_tensors(tensors: list) -> None:
        dtensor_type = get_dtensor_type()

        for idx, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise FeError(
                    RuntimeError(f"Input {idx + 1} (index {idx}) must be a torch.Tensor, got {tensor.__class__}")
                )
            if dtensor_type is not None and isinstance(tensor, dtensor_type):
                raise FeError(
                    RuntimeError(f"Input {idx + 1} (index {idx}) must not be a DTensor, got {tensor.__class__}")
                )
            if not tensor.is_contiguous():
                raise FeError(RuntimeError("not all tensors are contiguous"))

    @staticmethod
    def _get_func_nonlocals(func: Callable) -> dict[str, Any]:
        """Extract nonlocal (closure) variables from a function.

        This is a modified version of `inspect.getclosurevars` that specifically
        extracts only the nonlocal variables (captured from enclosing scopes)
        without global or builtin variables. These variables must be made available
        during parsing to properly evaluate the function body.

        Parameters
        ----------
        func : Callable
            The function to extract nonlocal variables from.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their captured values.

        Raises
        ------
        TypeError
            If func is not a Python function.
        """
        if inspect.ismethod(func):
            func = func.__func__

        if not inspect.isfunction(func):
            raise FeError(TypeError(f"{func!r} is not a Python function"))

        code = func.__code__
        # Nonlocal references are named in co_freevars and resolved
        # by looking them up in __closure__ by positional index
        nonlocal_vars = {}
        if func.__closure__ is not None:
            for var, cell in zip(code.co_freevars, func.__closure__):
                try:
                    nonlocal_vars[var] = cell.cell_contents
                except ValueError as err:
                    # cell_contents may raise ValueError if the cell is empty.
                    if "empty" not in str(err):
                        raise
        return nonlocal_vars

    @staticmethod
    def _convert_tensors_with_metadata(torch_tensors: list, tensor_defs: list) -> list:
        """Convert torch tensors to pypto tensors with name and dynamic_axis metadata."""
        pto_tensors = []
        for torch_tensor, tensor_def in zip(torch_tensors, tensor_defs):
            dynamic_axis = [
                i
                for i, dim in enumerate(tensor_def.shape)
                if isinstance(dim, pypto.SymbolicScalar) or dim in (pypto.StatusType.DYN, pypto.StatusType.DYNAMIC)
            ]
            pto_tensors.append(
                pypto.from_torch(
                    torch_tensor,
                    name=tensor_def.name,
                    dynamic_axis=dynamic_axis if dynamic_axis else None,
                    # Use dtype from type annotation when provided; otherwise fallback to torch tensor dtype.
                    dtype=tensor_def.explicit_dtype,
                    tensor_format=tensor_def.explicit_format,
                )
            )
        return pto_tensors

    @staticmethod
    def _make_hashable(obj) -> Any:
        """Convert dict/list/tuple to a hashable representation."""
        if obj is None:
            return None
        t = type(obj)
        if t is dict:
            return frozenset((k, JitCallableWrapper._make_hashable(v)) for k, v in obj.items())
        if t is list or t is tuple:
            return tuple(JitCallableWrapper._make_hashable(item) for item in obj)
        return str(obj)

    @_catch_and_wrap_error("compile function")
    def compile_new(
        self,
        tensors,
        tensor_defs=None,
    ) -> None:
        from pypto.operation import _gathermask_checks, _view_input_ids, _view_input_names, _view_original_shapes
        _view_original_shapes.clear()
        _view_input_ids.clear()
        _view_input_names.clear()
        _gathermask_checks.clear()

        if tensor_defs is not None:
            self._check_input_defs_match_tensors(tensors, tensor_defs)
            pto_tensor = self._convert_tensors_with_metadata(tensors, tensor_defs)
        else:
            pto_tensor = tensors

        self._set_config_option()

        self._pto_function = compile_new_ir(
            self._original_func,
            *pto_tensor,
            create_new_logical_tensor=self._create_new_logical_tensor,
            **(self.kwargs or {}),
        )

        # Check GatherMask viewshape constraint with actual tensor shapes
        if tensor_defs is not None:
            self._check_gathermask_view_constraint(tensors, tensor_defs)

        # Reset golden data after compilation
        _pto_verify_datas.reset()

    @_catch_and_wrap_error("compile function")
    def compile(
        self,
        tensors,
        tensor_defs=None,
    ) -> None:
        """Lazily compile function using PTO tensors for dynamic shape binding & verification.

        Compiles the wrapped function on first call (lazy mode).

        Parameters
        ----------
        tensors : list
            Either list[pypto.Tensor] (PTO tensors) or list[torch.Tensor] (torch tensors).
        tensor_defs : list[pypto.Tensor], optional
            When provided, tensors are torch tensors and will be converted to PTO via from_torch
            using name/dynamic_axis/dtype from each tensor_def. When None, tensors are PTO tensors.
        """
        if self._use_new_ir:
            self.compile_new(tensors, tensor_defs)
            return

        if tensor_defs is not None:
            self._check_input_defs_match_tensors(tensors, tensor_defs)
            args = self._convert_tensors_with_metadata(tensors, tensor_defs)
        else:
            args = tensors

        # Re-create parser for compilation
        self._parser = self._create_parser()
        self._parser.parse()
        self._parser.input_pto_tensor = args

        # Set options AFTER OperatorBegin() to match @pypto.frontend.jit behavior

        self._set_config_option()

        # Initialize backend for compilation
        setup_verify_data(args)

        # Bind dynamic dimensions from concrete inputs
        self._parser.bind_dynamic_dims_to_input_tensors()

        # Execute the deferred parsing (happens on first __call__)
        self._pto_function = self._parser.execute()

        # Reset golden data after compilation
        _pto_verify_datas.reset()

    def _parse_call_args(self, args: tuple, kwargs: dict) -> tuple[list, dict[str, Any], list]:
        """Parse *args and **kwargs into in_tensors and non_tensor_values.

        All tensor parameters (including out) are in input_tensor_defs; no separate
        output tensor definitions.

        Returns
        -------
        in_tensors : list[torch.Tensor]
        non_tensor_values : dict[str, Any]
        input_tensor_defs : list
        """
        input_tensor_defs, non_tensor_param_names = self._cached_signature
        n_tensors = len(input_tensor_defs)
        if len(args) < n_tensors:
            raise FeError(RuntimeError(f"Expected at least {n_tensors} tensor argument(s), got {len(args)}."))
        in_tensors = list(args[:n_tensors])
        non_tensor_from_args = list(args[n_tensors:])

        non_tensor_values = self._merge_non_tensor_params(non_tensor_param_names, non_tensor_from_args, kwargs)

        if len(non_tensor_from_args) > len(non_tensor_param_names):
            raise FeError(
                RuntimeError(
                    f"Too many positional arguments: expected {n_tensors} tensor(s) "
                    f"and up to {len(non_tensor_param_names)} non-tensor(s), "
                    f"got {len(args)} total."
                )
            )
        extra_kwargs = set(kwargs.keys()) - set(non_tensor_param_names)
        if extra_kwargs:
            raise FeError(
                RuntimeError(
                    f"Unknown keyword argument(s): {sorted(extra_kwargs)}. "
                    f"Valid non-tensor parameters: {non_tensor_param_names}."
                )
            )
        return in_tensors, non_tensor_values, input_tensor_defs

    def _merge_non_tensor_params(
        self,
        non_tensor_param_names: list[str],
        non_tensor_from_args: list,
        kwargs: dict,
    ) -> dict[str, Any]:
        """Merge positional non-tensor args with kwargs, using pre-extracted defaults when needed."""
        non_tensor_defaults = self._cached_non_tensor_defaults
        result: dict[str, Any] = {}
        for i, param_name in enumerate(non_tensor_param_names):
            if param_name in kwargs:
                val = kwargs[param_name]
            elif i < len(non_tensor_from_args):
                val = non_tensor_from_args[i]
            elif param_name in non_tensor_defaults:
                val = non_tensor_defaults[param_name]
            else:
                raise FeError(RuntimeError(f"Missing required non-tensor argument '{param_name}'."))
            if isinstance(val, torch.Tensor):
                raise FeError(
                    RuntimeError(
                        f"Non-tensor parameter '{param_name}' must not be a torch.Tensor. "
                        "Use positional arguments for tensors."
                    )
                )
            result[param_name] = val
        return result

    def _ensure_debug_options(self) -> None:
        """Ensure _debug_options is initialized with runtime_debug_mode from global config."""
        if self._debug_options is None:
            self._debug_options = {}
        if "runtime_debug_mode" not in self._debug_options:
            self._debug_options["runtime_debug_mode"] = pypto.get_debug_options().get("runtime_debug_mode", 0)

    def _ensure_host_options(self) -> None:
        """Ensure _host_options is initialized from global config."""
        if self._host_options is None:
            self._host_options = {}
        if "compile_stage" not in self._host_options:
            self._host_options["compile_stage"] = pypto.get_host_options().get(
                "compile_stage", pypto.CompStage.ALL_COMPLETE
            )
        if "compile_monitor_enable" not in self._host_options:
            self._host_options["compile_monitor_enable"] = pypto.get_host_options().get(
                "compile_monitor_enable", 1
            )
        if "compile_timeout" not in self._host_options:
            self._host_options["compile_timeout"] = pypto.get_host_options().get("compile_timeout", 600)
        if "compile_timeout_stage" not in self._host_options:
            self._host_options["compile_timeout_stage"] = pypto.get_host_options().get("compile_timeout_stage", 0)
        if "compile_monitor_print_interval" not in self._host_options:
            self._host_options["compile_monitor_print_interval"] = pypto.get_host_options().get(
                "compile_monitor_print_interval", 60
            )

    def _get_or_create_kmodule(self, non_tensor_values: dict[str, Any]) -> None:
        """Set self.kwargs and resolve kmodule from cache or create new."""
        self.kwargs = non_tensor_values
        self._ensure_debug_options()
        self._ensure_host_options()
        key = self._get_compilation_cache_key(non_tensor_values)
        cache_fields = "source/options/captured locals/non-tensor args"
        if key is not None and key in JitCallableWrapper._kernel_module_cache:
            log_debug(
                f"JIT cache hit for {self._original_func.__name__}: "
                "recreate KernelModule=False, recompile=False, "
                f"cache_fields={cache_fields}, tensor_attrs_checked_by=KernelModule"
            )
            self.kmodule = JitCallableWrapper._kernel_module_cache[key]
        else:
            log_debug(
                f"JIT cache miss for {self._original_func.__name__}: "
                f"cache_key_available={key is not None}, "
                "recreate KernelModule=True, recompile=True, "
                f"cache_fields={cache_fields}, tensor_attrs_checked_by=KernelModule"
            )
            self.kmodule = pypto_impl.KernelModule(self)
            if key is not None:
                JitCallableWrapper._kernel_module_cache[key] = self.kmodule

    def _check_gathermask_view_constraint(
        self,
        torch_tensors: list,
        tensor_defs: list,
    ) -> None:
        """Check GatherMask viewshape constraint with actual tensor shapes."""
        from pypto.operation import _gathermask_checks
        if not _gathermask_checks:
            return

        name_to_shape = {}
        for tensor_def, torch_tensor in zip(tensor_defs, torch_tensors):
            try:
                base = tensor_def._base if hasattr(tensor_def, '_base') else tensor_def
                name = base.GetName() if hasattr(base, 'GetName') else ''
                if name:
                    name_to_shape[name] = list(torch_tensor.shape)
            except (AttributeError, RuntimeError):
                pass

        for check in _gathermask_checks:
            input_name = check.get('input_tensor_name', '')
            viewshape_last = check['viewshape_last']
            if input_name and input_name in name_to_shape and viewshape_last is not None:
                actual_shape = name_to_shape[input_name]
                actual_last = actual_shape[-1]
                if isinstance(actual_last, int) and isinstance(viewshape_last, int) and viewshape_last < actual_last:
                    raise FeError(
                        RuntimeError(
                            f"GatherMask requires the last axis of self.shape not to be split by view. "
                            f"self.shape last axis: {actual_last}, viewshape last axis: {viewshape_last}"
                        )
                    )

    def _execute_kernel(
        self,
        torch_tensors: list,
        tensor_defs: list,
    ) -> None:
        """Run kernel on NPU or CPU (SIM)."""
        cannsim_is_configed: bool = bool(os.environ.get("CAMODEL_LOG_PATH"))
        if cannsim_is_configed:
            self._runtime_options["run_mode"] = RunMode.SIM
        if self._runtime_options.get("run_mode", None) == RunMode.NPU:
            pypto_impl.LaunchKernelTorch(self, _current_stream(), torch_tensors, tensor_defs)
        else:
            cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
            if pypto.get_global_config("simulation.accuracy_level") == 2 and cann_is_configed:
                with pypto.options("jit_scope"):
                    self._set_config_option()
                    get_torch_npu()
                    pypto_impl.LaunchKernelTorch(self, _current_stream(), torch_tensors, tensor_defs)
            else:
                pto_tensors = self._convert_tensors_with_metadata(torch_tensors, tensor_defs)
                with pypto.options("jit_scope"):
                    self._set_config_option()
                    pypto_impl.DeviceInit()
                    self.compile(pto_tensors)
                    self._run_with_cpu(pto_tensors, [])

    def _check_input_defs_match_tensors(self, in_tensors: list, input_tensor_defs: list[pypto.Tensor]) -> None:
        """Check if the input tensor definitions match the input tensors."""

        def get_format(tensor):
            return get_npu_tensor_format(tensor)

        # Check the number of input tensors and input tensor definitions
        if len(in_tensors) != len(input_tensor_defs):
            raise FeError(
                RuntimeError(
                    f"There are {len(in_tensors)} input tensor(s), \
                but {len(input_tensor_defs)} input tensor definition(s)."
                )
            )

        def ordinal(n):
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
            if 11 <= n % 100 <= 13:
                suffix = 'th'
            return f"{n}{suffix}"

        idx = 0
        for in_tensor, input_tensor_def in zip(in_tensors, input_tensor_defs):
            idx += 1

            # Skip checking if the input tensor definition is None or（shape len is 0 && shape object is not list）
            if len(input_tensor_def.shape) != 0 or input_tensor_def.status_shape is not None:
                filtered_shape = len([dim for dim in input_tensor_def.shape if dim is not Ellipsis])
                # def shape len must <= tensor shape len
                is_diff_shape = (
                    len(in_tensor.shape) != len(input_tensor_def.shape)
                    if input_tensor_def.status_shape is None
                    else len(in_tensor.shape) < filtered_shape
                )

                # Check the shape of input tensors and input tensor definitions
                if is_diff_shape:
                    raise FeError(
                        ValueError(
                            f"The number of dimensions of {ordinal(idx)} input tensor {in_tensor.shape} \
                        does not match the number of dimensions of input tensor definition {input_tensor_def.shape}."
                        )
                    )

                in_tensor_shape = list(in_tensor.shape)

                # 检查是否是FP4格式
                is_fp4 = input_tensor_def.dtype in (pypto.DT_FP4_E2M1, pypto.DT_FP4_E1M2)
                if is_fp4:
                    in_tensor_shape[-1] *= 2

                for i, dim in enumerate(input_tensor_def.shape):
                    if isinstance(dim, int) and in_tensor_shape[i] != dim:
                        raise FeError(
                            ValueError(
                                f"The shape of {ordinal(idx)} input tensor {in_tensor.shape} \
                            does not match the shape of input tensor definition {input_tensor_def.shape}."
                            )
                        )

            # Check the dtype of input tensors and input tensor definitions
            if input_tensor_def.explicit_dtype is not None:
                in_tensor_dtype = str(in_tensor.dtype)
                normal_mapped_dtype = self._dtype_dict.get(in_tensor_dtype)
                special_mapped_dtype = self._special_dtype_dict.get(in_tensor_dtype, [])
                if normal_mapped_dtype != input_tensor_def.dtype and input_tensor_def.dtype not in special_mapped_dtype:
                    raise FeError(
                        ValueError(
                            f"The dtype of {ordinal(idx)} input tensor {in_tensor.dtype} \
                        does not match the dtype of input tensor definition {input_tensor_def.dtype}."
                        )
                    )

            # uint8类型的tensor无法通过torch_npu.npu_format_cast转化为NZ，框架无法校验format
            not_check_format = ["torch.uint8"]
            if in_tensor.device.type == "npu":
                if input_tensor_def.explicit_format is not None and str(in_tensor.dtype) not in not_check_format:
                    if self._format_dict[get_format(in_tensor)] != input_tensor_def.format:
                        raise FeError(
                            ValueError(
                                f"The format of {ordinal(idx)} input tensor {get_format(in_tensor)} \
                            does not match the format of input tensor definition {input_tensor_def.format}."
                            )
                        )

    def _extract_non_tensor_defaults(self) -> dict[str, Any]:
        """Extract default values for non-tensor parameters from the original function signature."""
        _, non_tensor_param_names = self._cached_signature
        non_tensor_defaults: dict[str, Any] = {}
        sig = inspect.signature(self._original_func)
        for param_name in non_tensor_param_names:
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.default is not inspect.Parameter.empty:
                    non_tensor_defaults[param_name] = param.default
        return non_tensor_defaults

    def _compute_cache_hashes(self) -> Any:
        """Compute (source_code, options_hash, captured_locals_hash) for cache key generation."""

        def is_defined_globally(func):
            if not inspect.isfunction(func):
                return False
            qualname = func.__qualname__
            return '<locals>' not in qualname and '.' not in qualname

        if is_defined_globally(self._original_func):
            name = self._original_func.__name__
            _loop_idx_generator = next(self._global_func_idx_generator)
            return (name, _loop_idx_generator)

        code_obj = self._original_func.__code__
        source_code = (
            code_obj.co_code,
            code_obj.co_consts,
            code_obj.co_names,
            code_obj.co_varnames,
        )

        options_hash = (
            self._make_hashable(self._codegen_options),
            self._make_hashable(self._host_options),
            self._make_hashable(self._pass_options),
            self._make_hashable(self._runtime_options),
            self._make_hashable(self._verify_options),
            self._make_hashable(self._debug_options),
        )

        if self._captured_locals is not None:
            filtered_locals = {k: v for k, v in self._captured_locals.items() if not isinstance(v, torch.Tensor)}
            captured_locals_hash = self._make_hashable(filtered_locals)
        else:
            captured_locals_hash = None

        return (source_code, options_hash, captured_locals_hash)

    def _get_signature(
        self,
    ) -> tuple[list[pypto.Tensor], list[str]]:
        """Extract function signature: tensor param annotations, non-tensor param names,
        and non-tensor param defaults.

        Does not read or validate return annotation; use out parameter and out.move() instead.

        Parameters
        ----------
        func : Callable
            The function to analyze.

        Returns
        -------
        input_tensor_list : list[pypto.Tensor]
            List of input parameter annotations (tensor definitions, including out).
        non_tensor_param_names : list[str]
            Ordered list of non-tensor parameter names (must come after tensor params).
        """
        func = self._original_func
        code = func.__code__
        annotations = func.__annotations__ or {}
        argcount = code.co_argcount
        param_names = code.co_varnames[:argcount]

        input_tensor_list: list[pypto.Tensor] = []
        non_tensor_param_names: list[str] = []
        seen_non_tensor = False

        for param_name in param_names:
            if param_name == "return":
                continue
            ann = annotations.get(param_name)
            if ann is not None and hasattr(ann, 'to_tensor'):
                if seen_non_tensor:
                    raise FeError(
                        ValueError(
                            "Non-tensor parameters must come after all tensor parameters. "
                            f"Found tensor parameter '{param_name}' after non-tensor "
                            "parameter(s)."
                        )
                    )
                tensor = ann.to_tensor(param_name)
                input_tensor_list.append(tensor)
            else:
                seen_non_tensor = True
                non_tensor_param_names.append(param_name)

        return input_tensor_list, non_tensor_param_names

    def _get_compilation_cache_key(
        self,
        non_tensor_values: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple]:
        """Generate a cache key for compiled functions.

        This enables automatic caching of compilation results for factory function patterns.
        Functions with identical source code, input shapes, options, and non-tensor values
        can reuse compilation results, avoiding redundant compilation overhead.

        The cache key is generated based on:
        1. Source code
        2. Compilation options
        3. Closure variables (nonlocals captured by the function)
        4. Non-tensor parameter values (e.g., tiling)

        Parameters
        ----------
        non_tensor_values : Optional[dict[str, Any]], optional
            Non-tensor parameter values from the call. Defaults to empty dict.

        Returns
        -------
        Optional[tuple]
            A hashable cache key, or None if caching is not applicable.
        """
        if self._cache_hashes is None:
            return None

        non_tensor_hash = self._make_hashable(non_tensor_values) if non_tensor_values else None
        if len(self._cache_hashes) == 2:
            # Globally-defined function: cache key is (func_name, non_tensor_hash)
            fun_name, global_func_id = self._cache_hashes
            combined_key = f"{fun_name}::{global_func_id}"
            return (combined_key, non_tensor_hash)

        source_code, options_hash, captured_locals_hash = self._cache_hashes
        return (source_code, options_hash, captured_locals_hash, non_tensor_hash)

    def _set_run_mode(self) -> None:
        """Configure the runtime execution mode (NPU or SIM).

        Determines whether to run on NPU hardware or in simulation mode based on:
        1. Explicit run_mode in runtime_options
        2. Presence of ASCEND_HOME_PATH environment variable (indicates CANN installation)

        If CANN is configured, defaults to NPU mode; otherwise defaults to SIM mode.

        Raises
        ------
        RuntimeError
            If an invalid run_mode is specified (must be RunMode.NPU or RunMode.SIM).
        """
        if self._runtime_options is None:
            self._runtime_options = {}

        run_mode = self._runtime_options.get("run_mode", None)
        if run_mode is not None:
            if run_mode not in [pypto.RunMode.NPU, pypto.RunMode.SIM, 0, 1]:
                raise FeError(RuntimeError("Invalid run mode, run mode must be RunMode.NPU or RunMode.SIM."))
            else:
                if isinstance(run_mode, pypto.RunMode):
                    self._runtime_options.update({"run_mode": run_mode.value})
                return

        cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
        if cann_is_configed:
            self._runtime_options.update({"run_mode": pypto.RunMode.NPU.value})
            if torch.npu.is_available() is False:
                raise FeError(RuntimeError("NPU is not available."))
        else:
            self._runtime_options.update({"run_mode": pypto.RunMode.SIM.value})

    def _create_parser(self) -> Parser:
        """Create and prepare a parser for the wrapped function.

        Extracts the source code and all captured variables (globals and nonlocals)
        from the original function, then creates a Parser instance ready for parsing.

        Returns
        -------
        Parser
            A configured parser instance with source code and captured variables.
        """
        source = Source(self._original_func)
        closure_vars = inspect.getclosurevars(self._original_func)
        captured_vars = {}
        captured_vars.update(closure_vars.builtins)
        captured_vars.update(self._original_func.__globals__)
        captured_vars.update(closure_vars.globals)
        captured_vars.update(closure_vars.nonlocals)
        captured_vars.update(self._get_func_nonlocals(self._original_func))
        if self._captured_locals:
            captured_vars.update(self._captured_locals)
        if self.kwargs:
            captured_vars.update(self.kwargs)
        parser = Parser(source, captured_vars)
        return parser

    def _set_config_option(self) -> None:
        """Apply all configuration options to the PTO backend.

        This method applies the various option dictionaries provided at initialization
        to configure the backend compilation and runtime behavior. Options include:
        - run_mode (NPU or SIM)
        - codegen options (code generation settings)
        - host options (host environment settings)
        - pass options (compiler pass configurations)
        - runtime options (execution settings)
        - verify options (verification settings)
        - debug options (debugging settings)
        """
        if self._codegen_options:
            pypto.set_codegen_options(**self._codegen_options)
        if self._host_options:
            pypto.set_host_options(**self._host_options)
        if self._pass_options:
            pypto.set_pass_options(**self._pass_options)
        if self._runtime_options:
            options_dict = {k: v for k, v in self._runtime_options.items() if v is not None}
            pypto.set_options(runtime_options=options_dict)
        if self._verify_options:
            pypto.set_verify_options(**self._verify_options)
        if self._debug_options:
            pypto.set_debug_options(**self._debug_options)

    def _run(
        self,
        in_tensor_data: list[pypto_impl.DeviceTensorData],
        out_tensor_data: list[pypto_impl.DeviceTensorData],
        device: torch.device,
        ctrl_cache: int = 0,
    ) -> None:
        """Execute the compiled kernel on device with workspace allocation.

        This is the core execution method that:
        1. Queries required workspace size from the backend
        2. Allocates workspace memory on the target device
        3. Invokes the backend runtime with input/output tensors and workspace
        4. Checks for runtime errors

        Parameters
        ----------
        in_tensor_data : list[pypto_impl.DeviceTensorData]
            Input tensor metadata for the backend.
        out_tensor_data : list[pypto_impl.DeviceTensorData]
            Output tensor metadata for the backend.
        device : torch.device
            The device to execute on (must be NPU for this method).
        ctrl_cache : int
            Device control flow cache.
        Raises
        ------
        RuntimeError
            If runtime execution fails with an error message from the backend.
        """
        if self._handler is None:
            raise FeError(RuntimeError("handler is not initialized"))
        workspace_size = pypto_impl.GetWorkSpaceSize(self._handler, in_tensor_data, out_tensor_data)
        workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        runtime_error_msg = pypto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            in_tensor_data + out_tensor_data,
            [],  # Mark all output tensors as inplace inputs
            torch.npu.current_stream().npu_stream,
            workspace_tensor.data_ptr(),
            ctrl_cache,
        )
        if runtime_error_msg != "":
            raise FeError(RuntimeError(runtime_error_msg))

    def _run_with_npu(
        self,
        in_tensors: list[pypto.Tensor],
        out_tensors: list[pypto.Tensor],
        device: torch.device,
    ) -> None:
        """Execute on NPU hardware with automatic device switching.

        Converts PTO tensors to device tensor data and executes on the specified NPU.
        If the target device differs from the current device, temporarily switches
        to the target device and restores the original device after execution.

        Parameters
        ----------
        in_tensors : list[pypto.Tensor]
            Input PTO tensors.
        out_tensors : list[pypto.Tensor]
            Output PTO tensors.
        device : torch.device
            Target NPU device for execution.

        Raises
        ------
        RuntimeError
            If device type is not NPU or if execution fails.
        """
        if device.type == "npu":
            get_torch_npu()

            in_tensor_data = _pto_to_tensor_data(in_tensors)
            out_tensor_data = _pto_to_tensor_data(out_tensors)
            ori_device = torch.npu.current_device()
            if device.index != ori_device:
                torch.npu.set_device(device.index)
                self._run(in_tensor_data, out_tensor_data, device)
                torch.npu.set_device(ori_device)
            else:
                self._run(in_tensor_data, out_tensor_data, device)
        else:
            raise FeError(RuntimeError(f"Unsupported device type: {device.type}"))

    def _run_with_cpu(self, in_tensors: list[pypto.Tensor], out_tensors: list[pypto.Tensor]) -> None:
        """Execute in simulation mode using the cost model interface.

        This method runs the kernel on CPU using the cost model, which simulates
        the kernel behavior without requiring NPU hardware. Useful for development,
        testing, and performance modeling.

        Parameters
        ----------
        in_tensors : list[pypto.Tensor]
            Input PTO tensors.
        out_tensors : list[pypto.Tensor]
            Output PTO tensors.
        """
        _cost_model_run_once_data_from_host(in_tensors, out_tensors)


def function(
    func: Optional[Callable] = None,
) -> Union[Callable, NestedFunctionMarker]:
    """Decorator to mark a function for inline expansion in PTO kernels.

    Functions decorated with `@pypto.frontend.function` are not compiled as standalone
    kernels. Instead, when called from a JIT-compiled kernel, they are inlined directly
    into the caller's IR. This enables code reuse while maintaining optimal performance
    by avoiding function call overhead.

    Parameters
    ----------
    func : Optional[Callable], optional
        The function to mark for inlining. If None, returns a decorator function.
        This allows both @function and @function() syntax.

    Returns
    -------
    Union[Callable, NestedFunctionMarker]
        A NestedFunctionMarker wrapping the original function, which the parser
        recognizes as eligible for inline expansion.

    Examples
    --------
    >>> @pypto.frontend.function
    ... def helper(x: pypto.Tensor((8,), pypto.DT_FP32)):
    ...     return pypto.add(x, x)
    >>>
    >>> @pypto.frontend.jit
    ... def kernel(a: pypto.Tensor((8,), pypto.DT_FP32)):
    ...     return helper(a)  # helper is inlined here

    Notes
    -----
    - Nested functions must have compatible type signatures with their call sites
    - Parameter names need not match between definition and call
    - Return values from nested functions can be directly used
    - Multiple levels of nesting are supported
    """
    if func is None:

        def decorator(f: Callable) -> NestedFunctionMarker:
            marker = NestedFunctionMarker()
            marker._original_func = f
            marker._func_name = f.__name__
            return marker

        return decorator

    marker = NestedFunctionMarker()
    marker._original_func = func
    marker._func_name = func.__name__
    return marker


def jit(
    func: Optional[Callable] = None,
    *,
    host_options: Optional[dict[str, Any]] = None,
    codegen_options: Optional[dict[str, Any]] = None,
    pass_options: Optional[dict[str, Any]] = None,
    runtime_options: Optional[dict[str, Any]] = None,
    verify_options: Optional[dict[str, Any]] = None,
    debug_options: Optional[dict[str, Any]] = None,
    new_ir: bool = True,
    create_new_logical_tensor: bool = False,
) -> Union[Callable, Callable[[Callable], JitCallableWrapper]]:
    """JIT decorator for compiling Python functions to PTO IR.

    This decorator compiles a Python function into PTO's intermediate representation
    at decoration time. The decorated function will be replaced with the compiled
    PTO function.

    Parameters
    ----------
    func : Optional[Callable], optional
        The function to decorate. If None, returns a decorator function.
        This allows both @jit and @jit() syntax.

    host_options : Optional[dict[str, Any]], optional
        Options to configure the host.
    codegen_options : Optional[dict[str, Any]], optional
        Options to configure the codegen.
    pass_options : Optional[dict[str, Any]], optional
        Options to configure the pass.
    runtime_options : Optional[dict[str, Any]], optional
        Options to configure the runtime.
    verify_options : Optional[dict[str, Any]], optional
        Options to configure the verify.
    debug_options : Optional[dict[str, Any]], optional
        Options to configure the debug.
    new_ir : bool, optional
        Whether to compile through the new IR pipeline. Defaults to False.

    Returns
    -------
    Union[Callable, Callable[[Callable], Callable]]
        Either the decorated function (if func is provided) or a decorator function.

    Raises
    ------
    TypeError
        If the decorator is applied to a non-function object.

    Examples
    --------
    >>> @jit()
    ... def my_kernel(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x + 1
    >>> isinstance(my_kernel, pypto.Function)
    True

    >>> @jit
    ... def my_kernel2(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x * 2
    >>> isinstance(my_kernel2, pypto.Function)
    True

    Notes
    -----
    The decorator extracts closure variables (nonlocals and globals) from the
    original function and makes them available during parsing. The resulting
    PTO function preserves the original function's name and docstring.
    """

    def decorator_wrapper(f: Callable) -> JitCallableWrapper:
        if not inspect.isfunction(f):
            raise FeError(TypeError("jit decorator can only be used on functions"))

        # Create wrapper without compiling - defer to first call
        # This matches the behavior of @pypto.frontend.jit and avoids backend initialization
        # during module load time
        captured_locals = None
        frame = inspect.currentframe()
        if frame and frame.f_back:
            captured_locals = dict(frame.f_back.f_locals)
        # Break reference cycle as soon as possible
        del frame

        wrapper = JitCallableWrapper(
            None,
            f,
            None,
            codegen_options=codegen_options,
            host_options=host_options,
            pass_options=pass_options,
            runtime_options=runtime_options,
            verify_options=verify_options,
            debug_options=debug_options,
            captured_locals=captured_locals,
            new_ir=new_ir,
        )
        wrapper._create_new_logical_tensor = create_new_logical_tensor
        return wrapper

    if func is None:
        # Called with parentheses: @jit()
        return decorator_wrapper

    # Called without parentheses: @jit
    return decorator_wrapper(func)
