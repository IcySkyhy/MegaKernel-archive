"""Shared admission primitives: bounded no-follow reads, canonical JSON, the cubin format check."""
from __future__ import annotations

import json
import os
import stat
import struct


class AdmissionError(ValueError):
    pass


def _read(directory: int, name: str, maximum: int) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise AdmissionError(f'invalid file type or size: {name}')
        data = source.read(maximum + 1)
        if len(data) != info.st_size:
            raise AdmissionError(f'file changed during admission: {name}')
        return data


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def elf_sections(image: bytes, name: str = 'image') -> dict[str, tuple[int, bytes]]:
    try:
        shoff, = struct.unpack_from('<Q', image, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from('<HHH', image, 0x3a)
        if shentsize != 64 or not 0 < shnum <= 4096 or shstrndx >= shnum or shoff + shnum * 64 > len(image):
            raise AdmissionError(f'{name}: invalid ELF section table')
        headers = [struct.unpack_from('<IIQQQQIIQQ', image, shoff + 64 * i) for i in range(shnum)]
        def body(header):
            kind, offset, size = header[1], header[4], header[5]
            if kind == 8:  # NOBITS occupies no file bytes
                return b''
            if offset + size > len(image):
                raise AdmissionError(f'{name}: ELF section outside image')
            return image[offset:offset + size]
        names = body(headers[shstrndx])
        sections = {}
        for header in headers:
            end = names.find(b'\0', header[0])
            if end < 0:
                raise AdmissionError(f'{name}: invalid ELF section name')
            sections[names[header[0]:end].decode('ascii')] = (header[1], body(header))
        return sections
    except (struct.error, UnicodeError) as error:
        raise AdmissionError(f'{name}: malformed ELF') from error


def check_cubin(image: bytes, name: str = 'image') -> None:
    """A real sm_90a device ELF: never PTX text, a fatbin wrapper or a host object."""
    if (len(image) < 64 or not image.startswith(b'\x7fELF') or image[4] != 2 or image[5] != 1
            or image[7] != 0x41 or image[8] != 8):
        raise AdmissionError(f'{name}: not a 64-bit CUDA ELF (ABI 8) cubin')
    e_type, e_machine = struct.unpack_from('<HH', image, 16)
    (e_flags,) = struct.unpack_from('<I', image, 48)
    # ABI 8 packs the SM in bits 8..15; the arch-specific "a" target is only
    # recorded in the toolkit note, which must name sm_90a.
    if e_type != 2 or e_machine != 190 or (e_flags >> 8) & 0xff != 90:
        raise AdmissionError(f'{name}: cubin is not an sm_90 executable')
    sections = elf_sections(image, name)
    if '.nv_fatbin' in sections or b'sm_90a' not in sections.get('.note.nv.tkinfo', (0, b''))[1]:
        raise AdmissionError(f'{name}: cubin is not an sm_90a executable')
