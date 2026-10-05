"""Independent fixed-sequence execution of admitted 36-layer programs.

Not a vLLM serving backend: graph replay, delivery and retirement stay owned by the caller.
"""
import ctypes as C

from .config import TABLE_ENTRIES
from .native import BatchedControls, _Launch, _LIVE, check_disjoint, identity

_DTYPES = {1: 'float32', 3: 'bfloat16', 4: 'int32'}
_LBNHC_STRIDES = (65536, 256, 2048, 1)


class BindingToken:
    """Monotonic generation of the borrowed allocations a decoder was bound to."""

    def __init__(self):
        self.value = 0

    def advance(self):
        self.value += 1


def _check_binding(requirement, tensor, device, strides, blocks):
    import torch
    if (tensor.device != device or tensor.dtype != getattr(torch, _DTYPES[requirement.dtype])
            or tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg()
            or tensor.requires_grad or not 1 <= tensor.ndim <= 4):
        raise ValueError(f'{requirement.name}: dtype/device/layout mismatch')
    if requirement.flags & 2:
        count = tensor.shape[0]
        if (tuple(tensor.shape) != (count, 8, 32, 256) or tensor.stride() != strides
                or not 1 <= count <= 67108864 or (blocks is not None and count != blocks)):
            raise ValueError('inconsistent physical KV capacity/layout')
        minimum = count * requirement.extent
    else:
        if not tensor.is_contiguous():
            raise ValueError('non-cache inputs must be contiguous')
        count = blocks
        minimum = requirement.extent
    if tensor.numel() * tensor.element_size() < minimum or tensor.data_ptr() % requirement.alignment:
        raise ValueError(f'{requirement.name}: extent/alignment mismatch')
    return count


class _Workspace:
    """Common per-step, capture and retirement behaviour of one bucket's decoder."""

    token = None
    token_value = None

    def _bind_token(self, token):
        self.token = token
        self.token_value = None if token is None else token.value

    def _check(self, stream):
        """O(1): live state, binding generation, then the controls' stream admission."""
        if self.closed or self.poisoned:
            raise RuntimeError('decoder is retired or poisoned')
        if self.token is not None and self.token.value != self.token_value:
            self.poisoned = True
            raise RuntimeError('decoder borrowed allocations were replaced')
        self.control._check(stream)

    def verify_bindings(self):
        self._check(self.stream)
        self.control.verify_bindings()
        if any(identity(t) != expected for t, expected in zip(self.owners, self.identities)):
            self.poisoned = True
            raise RuntimeError('decoder borrowed storage changed')

    def authorize_capture_stream(self, stream):
        self.verify_bindings()
        self.control.authorize_capture_stream(stream)

    def enqueue(self, stream):
        self._check(stream)
        from .ops import enqueue
        enqueue(self.op_inputs, self.op_buffers, id(self))
        return self.output

    def _submit_native(self, stream):
        self._check(stream)
        try:
            for launch in self.launches:
                launch.submit(stream)
        except BaseException:
            self.poisoned = True
            raise

    def bind_delivery(self, tokens):
        self._check(self.stream)
        begin, end = tokens.data_ptr(), tokens.data_ptr() + tokens.numel() * tokens.element_size()
        if any(begin < tensor.data_ptr() + tensor.numel() * tensor.element_size()
               and tensor.data_ptr() < end for tensor in self.owners):
            raise ValueError('delivery aliases decoder storage')
        self.control.bind_delivery(tokens)

    def guard_delivery(self, stream):
        self._check(stream)
        self.control.guard_delivery(stream)

    def capture_graph(self):
        from .graphs import OwnedGraph
        if self.control.capture_stream is None:
            raise RuntimeError('authorize a capture stream first')
        self.verify_bindings()
        return OwnedGraph(self, self.enqueue, self.control.capture_stream)

    def _finish_setup(self, launches, op_inputs, op_buffers):
        self.launches = tuple(launches)
        self.op_inputs = op_inputs
        self.op_buffers = op_buffers

    def _rollback(self, error, stream):
        try:
            stream.synchronize()
            if self.module.value:
                self.driver.call('cuModuleUnload', self.module)
            if self.control is not None:
                self.control.close(all_graphs_destroyed=True)
        except BaseException as cleanup_error:
            self.poisoned = True
            _LIVE[id(self)] = self
            error.add_note(f'decoder prepare rollback incomplete: {cleanup_error}')
        else:
            self.closed = True
            self.device_owned.clear()
            self.owners.clear()
            self.storages.clear()

    def close(self, *, all_graphs_destroyed):
        self._check(self.stream)
        if self._graphs or self._deliveries or all_graphs_destroyed is not True:
            raise RuntimeError('referencing graphs must be destroyed')
        try:
            self.stream.synchronize()
            self.driver.call('cuModuleUnload', self.module)
            self.module = C.c_void_p()
            self.control.close(all_graphs_destroyed=True)
        except BaseException:
            self.poisoned = True
            raise
        self.launches = ()
        self.reset = None
        self.worker = None
        self.output = None
        self.control = None
        self.op_inputs.clear()
        self.op_buffers.clear()
        self.device_owned.clear()
        self.owners.clear()
        self.storages.clear()
        self.identities.clear()
        self.closed = True
        del _LIVE[id(self)]

    def _init_state(self, stream, owners):
        self.owners = owners
        self.storages = [tensor.untyped_storage() for tensor in owners]
        self.identities = [identity(t) for t in owners]
        self.stream = stream
        self.closed = False
        self._graphs = {}
        self._deliveries = {}
        self.poisoned = False
        self.module = C.c_void_p()
        self.control = None
        self.device_owned = []


class BatchedDecoder(_Workspace):
    """One bucket's v2 program: rows_ctl is allocated here and written by the validator."""

    def __init__(self, program, *, stream, bindings, output, positions, seq_lens, slots, table,
                 token=None):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare cannot run during capture')
        batch = program.batch
        self.batch = batch
        names = {binding.name for binding in program.bindings}
        if set(bindings) | {program.rows_ctl} != names or program.rows_ctl in bindings:
            raise ValueError('exact execution binding set required (rows_ctl is owned here)')
        if (output.device != stream.device or output.dtype != torch.bfloat16
                or not output.is_contiguous() or output.numel() * 2 != program.output_bytes):
            raise ValueError('output must hold one BF16 residual per row')
        if (tuple(table.shape) != (batch, TABLE_ENTRIES) or table.stride() != (TABLE_ENTRIES, 1)
                or table.dtype != torch.int32):
            raise ValueError(f'block table rows must be I32 with row stride {TABLE_ENTRIES}')
        with torch.cuda.stream(stream):
            rows_ctl = torch.zeros(program.rows_ctl_elements, dtype=torch.int32, device=stream.device)
        bound = dict(bindings)
        bound[program.rows_ctl] = rows_ctl
        owners = []
        blocks = None
        for requirement in program.bindings:
            tensor = bound[requirement.name]
            blocks = _check_binding(requirement, tensor, stream.device, _LBNHC_STRIDES, blocks)
            owners.append(tensor)
        if blocks is None:
            raise ValueError('program binds no paged cache')
        owners.extend([output, positions, seq_lens, slots, table])
        check_disjoint(owners, 'decoder bindings may not overlap')
        self._init_state(stream, owners)
        self._bind_token(token)
        self.program = program
        self.output = output
        self.rows_ctl = rows_ctl
        self.device_owned.append(rows_ctl)
        try:
            with torch.cuda.stream(stream):
                scheduler = torch.zeros(8, dtype=torch.int32, device=stream.device)
                self.control = BatchedControls(
                    program.controls, batch, stream=stream, blocks=blocks, positions=positions,
                    seq_lens=seq_lens, slots=slots, table=table, rows_ctl=rows_ctl,
                    output=output, scheduler=scheduler)
                self.driver = self.control.driver
                self.driver.call('cuModuleLoadData', C.byref(self.module), C.c_char_p(program.worker))
                tasks = torch.frombuffer(bytearray(program.tasks), dtype=torch.uint8).to(stream.device)
                ready_init = torch.frombuffer(bytearray(program.ready), dtype=torch.int32).to(stream.device)
                cursor = torch.zeros(1, dtype=torch.int32, device=stream.device)
                ready = torch.zeros(program.ready_count, dtype=torch.int32, device=stream.device)
                starts = torch.zeros(program.grid, dtype=torch.int64, device=stream.device)
                self.device_owned.extend([scheduler, tasks, ready_init, cursor, ready, starts])
                intermediates = []
                for size in program.scratch[:-1]:
                    # Persistent state is not entirely overwritten by reset/enqueue.
                    tensor = torch.zeros(size, dtype=torch.uint8, device=stream.device)
                    intermediates.append(tensor)
                    self.device_owned.append(tensor)
                pointers = [t.data_ptr() for t in owners[:len(program.bindings)]]
                pointers.extend(t.data_ptr() for t in intermediates)
                pointers.append(output.data_ptr())
                if len(pointers) != program.buffer_count:
                    raise ValueError('pointer table size mismatch')
                pointer_table = torch.tensor(pointers, dtype=torch.uint64, device=stream.device)
                self.device_owned.append(pointer_table)
            ptr = lambda tensor: C.c_uint64(tensor.data_ptr())
            arguments = {'tasks': ptr(tasks), 'cursor': ptr(cursor), 'ready': ptr(ready),
                         'starts': ptr(starts), 'pointers': ptr(pointer_table),
                         'dims': ptr(self.control.dims), 'scheduler': ptr(scheduler)}
            kernels = program.kernels
            self.reset = _Launch(self.driver, self.module, kernels['reset'], [
                ptr(cursor), ptr(ready), ptr(ready_init), C.c_int(program.ready_count),
                C.c_int(program.ready_count)], grid=max(1, -(-program.ready_count // 256)))
            self.worker = _Launch(self.driver, self.module, kernels['worker'], [
                arguments[name] if name != 'null' else C.c_uint64(0)
                for name in ('tasks', 'cursor', 'ready', 'null', 'null', 'null', 'starts',
                             'pointers', 'dims', 'scheduler')],
                grid=program.grid, threads=program.threads, shared=program.dynamic_shared,
                blocks_per_sm=program.blocks_per_sm)
            launches = (self.control.validator, self.reset, self.worker, self.control.finalizer)
            op_inputs = [bound[b.name] for b in program.bindings
                         if not b.flags & 1 and b.name != program.rows_ctl]
            op_inputs.extend([positions, seq_lens, slots, table, tasks, ready_init, pointer_table])
            op_buffers = [bound[b.name] for b in program.bindings if b.flags & 1]
            op_buffers.extend([output, rows_ctl, scheduler, cursor, ready, starts, *intermediates,
                               self.control.status, self.control.dims])
            self._finish_setup(launches, op_inputs, op_buffers)
        except BaseException as error:
            self._rollback(error, stream)
            raise
        _LIVE[id(self)] = self


class DecoderSet:
    """One decoder per bucket, retired together; each member is rooted here."""

    def __init__(self, decoders, stream):
        if not decoders or any(d.stream is not stream for d in decoders.values()):
            raise ValueError('bucket decoders must share one execution stream')
        self.decoders = dict(sorted(decoders.items()))
        self.stream = stream
        self.closed = False
        self._poisoned = False
        self._graphs = {}
        for decoder in self.decoders.values():
            # Members cannot be retired on their own while the set owns them.
            decoder._graphs[id(self)] = self

    @property
    def buckets(self):
        return tuple(self.decoders)

    @property
    def poisoned(self):
        return self._poisoned or any(d.poisoned for d in self.decoders.values())

    @poisoned.setter
    def poisoned(self, value):
        self._poisoned = self._poisoned or bool(value)
        if value:
            for decoder in self.decoders.values():
                decoder.poisoned = True

    @property
    def _deliveries(self):
        merged = {}
        for decoder in self.decoders.values():
            merged.update(decoder._deliveries)
        return merged

    @property
    def capture_stream(self):
        streams = {d.control.capture_stream for d in self.decoders.values()}
        return streams.pop() if len(streams) == 1 else None

    def bucket(self, rows):
        try:
            return self.decoders[rows]
        except KeyError:
            raise RuntimeError(f'no admitted decoder for bucket {rows}') from None

    def _check(self, stream):
        if self.closed or self._poisoned:
            raise RuntimeError('decoder set is retired or poisoned')
        for decoder in self.decoders.values():
            decoder._check(stream)

    def verify_bindings(self):
        for decoder in self.decoders.values():
            decoder.verify_bindings()

    def authorize_capture_stream(self, stream):
        self._check(self.stream)
        for decoder in self.decoders.values():
            if decoder.control.capture_stream is None:
                decoder.authorize_capture_stream(stream)
            elif decoder.control.capture_stream != stream:
                raise RuntimeError('capture stream changed')

    def close(self, *, all_graphs_destroyed):
        self._check(self.stream)
        if self._graphs or self._deliveries or all_graphs_destroyed is not True:
            raise RuntimeError('referencing graphs must be destroyed')
        try:
            self.verify_bindings()
            for decoder in self.decoders.values():
                decoder._graphs.pop(id(self), None)
                decoder.close(all_graphs_destroyed=True)
        except BaseException:
            self.poisoned = True
            raise
        self.decoders.clear()
        self.closed = True
