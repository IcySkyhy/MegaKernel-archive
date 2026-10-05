"""Versioned binary execution metadata admission; no runtime recipe planning.

``Q3MKEX02`` carries its own geometry, so only the shared paged-KV ABI is hard-coded here.
"""
from dataclasses import dataclass
import hashlib
import json
import struct

from .artifacts import AdmissionError, canonical, check_cubin
from .config import MAX_MODEL_LEN as CAPACITY, PAGE, TABLE_ENTRIES

LBNHC = 2
HIDDEN = 2560
PAGE_EXTENT = 131072
LAYERS = 36
MAX_FILE = 16 * 1024 * 1024


@dataclass(frozen=True)
class Binding:
    name: str
    dtype: int
    flags: int
    extent: int
    alignment: int


V2_MAGIC = b'Q3MKEX02'
V2_DTYPES = {'bf16': (3, 2), 'f32': (1, 4), 'i32': (4, 4)}
V2_KERNELS = {'reset': 'megakernel_reset', 'worker': 'worker_kernel',
              'validate': 'mk_validate_paged', 'finalize': 'mk_finalize_paged',
              'guard': 'mk_guard_sampled_tokens'}
V2_WORKER_ARGS = ('tasks', 'cursor', 'ready', 'null', 'null', 'null', 'starts', 'pointers',
                  'dims', 'scheduler')
V2_HEADER_KEYS = {
    'version', 'batch', 'layout', 'page', 'capacity', 'table_entries', 'grid', 'threads',
    'dynamic_shared', 'blocks_per_sm', 'task_count', 'task_stride', 'task_source_offset',
    'task_source_slots', 'ready_count', 'buffer_count', 'externals', 'scratch',
    'output_bytes', 'cache_bindings', 'rows_ctl', 'kernels', 'worker_args',
}
V2_PAYLOAD = ('worker.cubin', 'controls.cubin', 'execution.bin')
EMBEDDED_NAME = 'embed'
# One summed BF16 residual per row; the plugin applies the stock final RMSNorm.
OUTPUT_FIELDS = 1
# Task word 10 is the "dyn_cells" flag: such tasks need a device count launch
# this fixed four-launch sequence does not contain.
TASK_DYN_CELLS_OFFSET = 40
MAX_HEADER = 1024 * 1024
MAX_DYNAMIC_SHARED = 232448


@dataclass(frozen=True)
class ProgramV2:
    batch: int
    grid: int
    threads: int
    dynamic_shared: int
    blocks_per_sm: int
    task_count: int
    task_stride: int
    ready_count: int
    buffer_count: int
    bindings: tuple[Binding, ...]
    scratch: tuple[int, ...]
    output_bytes: int
    rows_ctl: str
    cache_bindings: tuple[str, ...]
    kernels: dict
    tasks: bytes
    ready: bytes
    worker: bytes
    controls: bytes
    header_sha256: str
    embedded: str = EMBEDDED_NAME
    output_fields: int = OUTPUT_FIELDS

    @property
    def layout(self):
        return LBNHC

    @property
    def rows_ctl_elements(self):
        return 5 * self.batch + self.batch * TABLE_ENTRIES


def _int(header, name, minimum, maximum):
    value = header[name]
    if type(value) is not int or not minimum <= value <= maximum:
        raise AdmissionError(f'execution header {name} out of range')
    return value


def parse_execution_v2(data: bytes, batch: int, worker: bytes, controls: bytes) -> ProgramV2:
    """Admit a v2 execution image for exactly the bucket ``batch``."""
    try:
        if len(data) < 12 or data[:8] != V2_MAGIC:
            raise AdmissionError('unsupported execution magic')
        (header_len,) = struct.unpack_from('<I', data, 8)
        if not 0 < header_len <= MAX_HEADER or 12 + header_len > len(data):
            raise AdmissionError('truncated execution header')
        raw = data[12:12 + header_len]
        header = json.loads(raw)
        if not isinstance(header, dict) or set(header) != V2_HEADER_KEYS:
            raise AdmissionError('unknown execution header fields')
        if canonical(header) != raw:
            raise AdmissionError('noncanonical execution header')
        fixed = {'version': 2, 'batch': batch, 'layout': 'LBNHC', 'page': PAGE,
                 'capacity': CAPACITY, 'table_entries': TABLE_ENTRIES, 'blocks_per_sm': 1}
        for key, value in fixed.items():
            if type(header[key]) is not type(value) or header[key] != value:
                raise AdmissionError(f'unsupported execution {key}')
        grid = _int(header, 'grid', 1, 1024)
        threads = _int(header, 'threads', 32, 1024)
        if threads % 32:
            raise AdmissionError('execution threads must be whole warps')
        dynamic_shared = _int(header, 'dynamic_shared', 0, MAX_DYNAMIC_SHARED)
        task_count = _int(header, 'task_count', 1, 1 << 20)
        task_stride = _int(header, 'task_stride', 8, 1 << 16)
        source_offset = _int(header, 'task_source_offset', 0, task_stride)
        source_slots = _int(header, 'task_source_slots', 1, 64)
        ready_count = _int(header, 'ready_count', 1, 1 << 24)
        buffer_count = _int(header, 'buffer_count', 1, 1 << 16)
        # The output buffer index directly follows the source slots in each task.
        if (task_stride % 4 or source_offset % 4 or source_offset + 4 * (source_slots + 1) > task_stride
                or TASK_DYN_CELLS_OFFSET + 4 > task_stride):
            raise AdmissionError('invalid task layout')
        if (header['kernels'] != V2_KERNELS or not isinstance(header['worker_args'], list)
                or tuple(header['worker_args']) != V2_WORKER_ARGS):
            raise AdmissionError('unsupported kernel ABI')

        externals = header['externals']
        if not isinstance(externals, list) or not externals:
            raise AdmissionError('invalid externals')
        bindings, names = [], set()
        for entry in externals:
            if not isinstance(entry, dict) or set(entry) != {'name', 'dtype', 'flags', 'extent', 'alignment'}:
                raise AdmissionError('invalid external descriptor')
            name = entry['name']
            if (not isinstance(name, str) or not 0 < len(name) <= 256 or name in names
                    or not name.isascii() or any(not (c.isalnum() or c in '._') for c in name)):
                raise AdmissionError('duplicate/invalid binding name')
            if entry['dtype'] not in V2_DTYPES:
                raise AdmissionError(f'{name}: unsupported dtype')
            dtype, item = V2_DTYPES[entry['dtype']]
            flags, extent, alignment = entry['flags'], entry['extent'], entry['alignment']
            if (type(flags) is not int or flags & ~3 or type(extent) is not int
                    or not 0 < extent <= 2**31 or type(alignment) is not int or alignment not in (4, 16)):
                raise AdmissionError(f'{name}: invalid binding descriptor')
            if flags & 2 and (dtype != 3 or flags != 3 or extent != PAGE_EXTENT):
                raise AdmissionError(f'{name}: invalid page extent')
            if extent % item:
                raise AdmissionError(f'{name}: partial storage element')
            names.add(name)
            bindings.append(Binding(name, dtype, flags, extent, alignment))

        expected_cache = [f'kv.{i}' for i in range(LAYERS)]
        if header['cache_bindings'] != expected_cache or [b.name for b in bindings if b.flags & 2] != expected_cache:
            raise AdmissionError('invalid paged cache bindings')
        rows = [b for b in bindings if b.name == 'rows_ctl']
        extent = (5 * batch + batch * TABLE_ENTRIES) * 4
        if header['rows_ctl'] != 'rows_ctl' or len(rows) != 1 or (rows[0].dtype, rows[0].flags, rows[0].extent) != (4, 0, extent):
            raise AdmissionError('invalid rows_ctl binding')
        embedded = [b for b in bindings if b.name == EMBEDDED_NAME]
        if len(embedded) != 1 or (embedded[0].dtype, embedded[0].flags, embedded[0].extent) != (3, 0, batch * HIDDEN * 2):
            raise AdmissionError('invalid embedded (embed) binding')

        scratch = header['scratch']
        output_bytes = header['output_bytes']
        if (not isinstance(scratch, list) or not scratch
                or any(type(size) is not int or not 0 < size <= 2**28 for size in scratch)
                or type(output_bytes) is not int or scratch[-1] != output_bytes
                or output_bytes != OUTPUT_FIELDS * batch * HIDDEN * 2):
            raise AdmissionError('invalid scratch/output extent')
        if buffer_count != len(bindings) + len(scratch):
            raise AdmissionError('pointer table size mismatch')

        body = 12 + header_len
        if body + task_count * task_stride + ready_count * 4 != len(data):
            raise AdmissionError('execution size/trailing data mismatch')
        tasks = data[body:body + task_count * task_stride]
        ready = data[body + task_count * task_stride:]
        for task in range(task_count):
            base = task * task_stride
            if struct.unpack_from('<i', tasks, base + TASK_DYN_CELLS_OFFSET)[0] != 0:
                raise AdmissionError('program requires a dynamic barrier count launch')
            indices = struct.unpack_from(f'<{source_slots + 1}i', tasks, base + source_offset)
            if any(index < -1 or index >= buffer_count for index in indices) or indices[-1] < 0:
                raise AdmissionError('task buffer index outside pointer table')
        if any(value < 0 for value in struct.unpack(f'<{ready_count}i', ready)):
            raise AdmissionError('negative initial barrier count')
        check_cubin(worker, 'worker.cubin')
        check_cubin(controls, 'controls.cubin')
        return ProgramV2(batch, grid, threads, dynamic_shared, 1, task_count, task_stride,
                         ready_count, buffer_count, tuple(bindings), tuple(scratch), output_bytes,
                         'rows_ctl', tuple(expected_cache), dict(V2_KERNELS), tasks, ready,
                         worker, controls, hashlib.sha256(raw).hexdigest())
    except (ValueError, TypeError, KeyError, struct.error) as error:
        if isinstance(error, AdmissionError):
            raise
        raise AdmissionError('execution v2 admission failed') from error
