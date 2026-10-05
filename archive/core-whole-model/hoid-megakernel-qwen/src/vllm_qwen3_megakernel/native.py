"""Cubin-only auxiliary controls on borrowed torch storage.

These are not decoders: they expose validator/finalizer/delivery guard
operations around a worker. ``BatchedControls`` is the paged ABI of every
bucket: its validator derives each row's liveness from ``seq_lens`` on the
device and writes the worker's ``rows_ctl``.

Per-step checks are O(1): state, thread, context and stream authorization.
The O(n) identity scan of borrowed tensors runs at bind, capture and close
(``verify_bindings``); between those points the retained storages keep every
captured address alive, and the owner invalidates the whole generation when
vLLM replaces an allocation.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import threading

from .config import BUCKETS, TABLE_ENTRIES

_LIVE = {}


class DriverError(RuntimeError):
    pass


class _Driver:
    def __init__(self):
        self.lib = C.CDLL('libcuda.so.1')
        signatures = {
            'cuCtxGetCurrent': [C.POINTER(C.c_void_p)],
            'cuCtxGetDevice': [C.POINTER(C.c_int)],
            'cuDevicePrimaryCtxRetain': [C.POINTER(C.c_void_p), C.c_int],
            'cuDevicePrimaryCtxRelease_v2': [C.c_int],
            'cuStreamGetCtx': [C.c_void_p, C.POINTER(C.c_void_p)],
            'cuModuleLoadData': [C.POINTER(C.c_void_p), C.c_void_p],
            'cuModuleGetFunction': [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
            'cuModuleUnload': [C.c_void_p],
            'cuFuncGetAttribute': [C.POINTER(C.c_int), C.c_int, C.c_void_p],
            'cuFuncSetAttribute': [C.c_void_p, C.c_int, C.c_int],
            'cuOccupancyMaxActiveBlocksPerMultiprocessor': [
                C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t],
            'cuLaunchKernel': [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                               C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                               C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = C.c_int

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            raise DriverError(f'{name} failed: CUDA error {result}')

    def context(self):
        value = C.c_void_p()
        self.call('cuCtxGetCurrent', C.byref(value))
        if not value.value:
            raise DriverError('caller must establish a CUDA context')
        return value.value

    def require_primary(self, device):
        ordinal = C.c_int()
        self.call('cuCtxGetDevice', C.byref(ordinal))
        if ordinal.value != device.index:
            raise DriverError('caller context device mismatch')
        primary = C.c_void_p()
        self.call('cuDevicePrimaryCtxRetain', C.byref(primary), ordinal)
        try:
            if self.context() != primary.value:
                raise DriverError('caller must use the torch device primary context')
        finally:
            self.call('cuDevicePrimaryCtxRelease_v2', ordinal)

    def require_stream_context(self, stream, context):
        actual = C.c_void_p()
        self.call('cuStreamGetCtx', C.c_void_p(stream.cuda_stream), C.byref(actual))
        if actual.value != context:
            raise DriverError('stream belongs to another context')


class _Launch:
    def __init__(self, driver, module, name, values, *, grid=1, threads=256, shared=0,
                 blocks_per_sm=1):
        self.driver = driver
        self.grid, self.threads, self.shared = grid, threads, shared
        self.function = C.c_void_p()
        driver.call('cuModuleGetFunction', C.byref(self.function), module, name.encode())
        maximum = C.c_int()
        driver.call('cuFuncGetAttribute', C.byref(maximum), 0, self.function)
        if shared:
            driver.call('cuFuncSetAttribute', self.function, 8, shared)
        active = C.c_int()
        driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',
                    C.byref(active), self.function, threads, shared)
        if maximum.value < threads or active.value < blocks_per_sm:
            raise DriverError('loaded image cannot admit the requested CTA')
        self.values = values
        self.args = (C.c_void_p * len(values))(*(C.addressof(value) for value in values))

    def submit(self, stream):
        self.driver.call('cuLaunchKernel', self.function, self.grid, 1, 1,
                         self.threads, 1, 1, self.shared,
                         C.c_void_p(stream.cuda_stream), self.args, None)


def identity(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tensor.stride()


def check_disjoint(tensors, message):
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors)
    if any(end > next_begin for (_, end), (next_begin, _) in zip(spans, spans[1:])):
        raise ValueError(message)


class _Controls:
    """Thread-affine ordered control workspace; explicit graph-safe retirement only."""

    rows = 1

    def _validator_values(self, ptr, blocks):
        raise NotImplementedError

    def _setup(self, image, *, stream, blocks, output, scheduler, tensors, specs):
        import torch

        if not isinstance(image, bytes):
            raise TypeError('an admitted control image is required')
        self.image_sha256 = hashlib.sha256(image).hexdigest()
        device = stream.device
        properties = torch.cuda.get_device_properties(device)
        if ('H200' not in properties.name or properties.multi_processor_count != 132
                or (properties.major, properties.minor) != (9, 0)):
            raise ValueError('requires 132-SM H200')
        if type(blocks) is not int or not 1 <= blocks <= 67108864:
            raise ValueError('invalid physical page capacity')
        for tensor, (dtype, shapes) in zip(tensors, specs, strict=True):
            if (tensor.device != device or tensor.dtype != dtype or not tensor.is_contiguous()
                    or tuple(tensor.shape) not in shapes or tensor.requires_grad
                    or tensor.is_conj() or tensor.is_neg()):
                raise ValueError('invalid control tensor device/dtype/extent/layout')
        check_disjoint(tensors, 'control tensors may not alias')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('prepare is forbidden during capture')
        self.driver = _Driver()
        self.context = self.driver.context()
        self.driver.require_primary(device)
        self.driver.require_stream_context(stream, self.context)
        self.thread = threading.get_ident()
        self.stream = stream
        self.capture_stream = None
        self.closed = False
        self.poisoned = False
        self.module = C.c_void_p()
        self.owners = list(tensors)
        self.storages = [tensor.untyped_storage() for tensor in tensors]
        self._graphs = {}
        self._deliveries = {}
        self.identities = [identity(t) for t in tensors]
        self.output_bytes = output.numel() * output.element_size()
        try:
            self.driver.call('cuModuleLoadData', C.byref(self.module), C.c_char_p(image))
            with torch.cuda.stream(stream):
                self.status = torch.zeros(2, dtype=torch.int64, device=device)
                self.dims = torch.zeros(26, dtype=torch.int32, device=device)
                self.consumed = torch.zeros(1, dtype=torch.int64, device=device)
            ptr = lambda tensor: C.c_uint64(tensor.data_ptr())
            self.validator = _Launch(self.driver, self.module, 'mk_validate_paged',
                                     self._validator_values(ptr, blocks))
            self.finalizer = _Launch(self.driver, self.module, 'mk_finalize_paged', [
                ptr(self.status), ptr(self.dims), ptr(scheduler), ptr(output),
                C.c_uint64(self.output_bytes)])
            self.guard = _Launch(self.driver, self.module, 'mk_guard_sampled_tokens', [
                ptr(self.status), ptr(scheduler), ptr(self.consumed), C.c_uint64(0), C.c_uint64(0)])
            self.delivery = None
        except BaseException as error:
            try:
                # Setup zero-fills may be queued even though no handle was published.
                stream.synchronize()
                if self.module.value:
                    self.driver.call('cuModuleUnload', self.module)
            except BaseException as cleanup_error:
                self.poisoned = True
                _LIVE[id(self)] = self
                error.add_note(f'prepare rollback could not prove cleanup: {cleanup_error}')
            else:
                self.closed = True
                self.owners.clear()
                self.storages.clear()
                self.module = C.c_void_p()
                for name in ('status', 'dims', 'consumed'):
                    setattr(self, name, None)
            raise
        _LIVE[id(self)] = self

    def _check(self, stream):
        """O(1) per-step admission: live state, thread, context and stream."""
        import torch
        if self.closed or self.poisoned:
            raise RuntimeError('retired or poisoned control workspace')
        if threading.get_ident() != self.thread or self.driver.context() != self.context:
            raise RuntimeError('caller thread/context changed')
        capture = torch.cuda.is_current_stream_capturing()
        allowed = self.capture_stream if capture else self.stream
        if allowed is None or stream != allowed or torch.cuda.current_stream(stream.device) != stream:
            raise RuntimeError('unauthorized execution/capture stream')

    def verify_bindings(self):
        """O(n) identity scan of every borrowed tensor, at bind/capture/close points."""
        if any(identity(t) != expected for t, expected in zip(self.owners, self.identities)):
            self.poisoned = True
            raise RuntimeError('borrowed storage changed')

    def authorize_capture_stream(self, stream):
        self._check(self.stream)
        self.verify_bindings()
        if self.capture_stream is not None:
            raise RuntimeError('capture authorization cannot be replaced')
        if stream.device != self.stream.device:
            raise ValueError('capture device mismatch')
        self.driver.require_stream_context(stream, self.context)
        self.capture_stream = stream

    def validate(self, stream):
        self._check(stream)
        self.validator.submit(stream)

    def finalize(self, stream):
        self._check(stream)
        self.finalizer.submit(stream)

    def bind_delivery(self, tokens):
        import torch
        self._check(self.stream)
        if torch.cuda.is_current_stream_capturing() or self.delivery is not None:
            raise RuntimeError('delivery can only be bound once outside capture')
        if (tokens.device != self.stream.device or tokens.dtype != torch.int32
                or not tokens.is_contiguous() or not 1 <= tokens.numel() <= self.rows):
            raise ValueError('delivery requires contiguous I32 tokens, at most one per row')
        begin, end = tokens.data_ptr(), tokens.data_ptr() + 4 * tokens.numel()
        for tensor in [*self.owners, self.status, self.dims, self.consumed]:
            if begin < tensor.data_ptr() + tensor.numel() * tensor.element_size() and tensor.data_ptr() < end:
                raise ValueError('delivery must be distinct storage')
        self.delivery = tokens
        self.owners.append(tokens)
        self.storages.append(tokens.untyped_storage())
        self.identities.append(identity(tokens))
        self.guard.values[3].value = tokens.data_ptr()
        self.guard.values[4].value = tokens.numel()

    def guard_delivery(self, stream):
        self._check(stream)
        if self.delivery is None:
            raise RuntimeError('delivery is not bound')
        self.guard.submit(stream)

    def capture_graph(self, record):
        from .graphs import OwnedGraph
        if self.capture_stream is None:
            raise RuntimeError('authorize a capture stream first')
        return OwnedGraph(self, record, self.capture_stream)

    def close(self, *, all_graphs_destroyed):
        self._check(self.stream)
        if self._graphs or self._deliveries or all_graphs_destroyed is not True:
            raise RuntimeError('destroy referencing graphs before retirement')
        try:
            self.stream.synchronize()
            self.driver.call('cuModuleUnload', self.module)
        except BaseException:
            self.poisoned = True
            raise
        self.module = C.c_void_p()
        self.owners.clear()
        self.storages.clear()
        self.identities.clear()
        self.status = None
        self.dims = None
        self.consumed = None
        self.delivery = None
        self.validator = None
        self.finalizer = None
        self.guard = None
        self.capture_stream = None
        self.closed = True
        del _LIVE[id(self)]


class BatchedControls(_Controls):
    """Paged controls of one bucket; padded rows carry ``seq_len == 0`` and are never read."""

    def __init__(self, image, rows, *, stream, blocks, positions, seq_lens, slots, table,
                 rows_ctl, output, scheduler):
        import torch
        if rows not in BUCKETS:
            raise ValueError(f'paged controls serve buckets {BUCKETS}')
        self.rows = rows
        self._values = (positions, seq_lens, slots, table, rows_ctl, output, scheduler)
        specs = [(torch.int64, {(rows,)}), (torch.int32, {(rows,)}), (torch.int64, {(rows,)}),
                 (torch.int32, {(rows, TABLE_ENTRIES)}),
                 (torch.int32, {(5 * rows + TABLE_ENTRIES * rows,)}),
                 (torch.bfloat16, {tuple(output.shape)} if output.numel() == rows * 2560 else set()),
                 (torch.int32, {(8,)})]
        self._setup(image, stream=stream, blocks=blocks, output=output, scheduler=scheduler,
                    tensors=list(self._values), specs=specs)

    def _validator_values(self, ptr, blocks):
        positions, seq_lens, slots, table, rows_ctl, output, scheduler = self._values
        return [ptr(self.status), ptr(self.dims), ptr(positions), ptr(seq_lens), ptr(slots),
                ptr(table), C.c_int(blocks), ptr(rows_ctl), ptr(output),
                C.c_uint64(self.output_bytes), ptr(scheduler)]
