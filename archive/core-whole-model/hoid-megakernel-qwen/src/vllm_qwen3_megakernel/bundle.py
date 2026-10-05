"""Load the per-bucket megakernel cubins from a bundle directory.

Layout: ``index.json`` (the buckets it holds) and one ``b<N>/`` directory per bucket with
``worker.cubin``, ``controls.cubin`` and ``execution.bin``. Nothing is signed; the bytes on
disk are what gets loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .artifacts import AdmissionError
from .config import BUCKETS as SUPPORTED_BUCKETS, MAX_MODEL_LEN, PAGE, TABLE_ENTRIES
from .execution import MAX_FILE, parse_execution_v2

ABI = 'qwen3.decoder.v2'
KV = {'layout': 'LBNHC', 'page': PAGE, 'capacity': MAX_MODEL_LEN, 'table_entries': TABLE_ENTRIES}


@dataclass(frozen=True)
class Bundle:
    programs: dict  # bucket -> execution.ProgramV2
    index_sha256: str

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.programs))


def _read(path: Path) -> bytes:
    data = path.read_bytes()
    if not 0 < len(data) <= MAX_FILE:
        raise AdmissionError(f'{path}: invalid size {len(data)}')
    return data


def load_bundle(directory) -> Bundle:
    root = Path(directory)
    raw = _read(root / 'index.json')
    index = json.loads(raw)
    if index.get('kv') != KV:
        raise AdmissionError(f'bundle KV envelope {index.get("kv")} differs from the plugin {KV}')
    programs = {}
    for key, entry in index['buckets'].items():
        bucket = int(key)
        if bucket not in SUPPORTED_BUCKETS or entry.get('abi') != ABI:
            raise AdmissionError(f'unsupported bucket {key!r}: {entry}')
        folder = root / entry['dir']
        programs[bucket] = parse_execution_v2(_read(folder / 'execution.bin'), bucket,
                                              _read(folder / 'worker.cubin'),
                                              _read(folder / 'controls.cubin'))
    if not programs:
        raise AdmissionError('bundle lists no buckets')
    return Bundle(programs, hashlib.sha256(raw).hexdigest())
