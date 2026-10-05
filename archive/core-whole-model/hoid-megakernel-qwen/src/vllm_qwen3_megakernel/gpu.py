"""The megakernel is built for the 132-SM H200 only. Checked before any download, environment
build or CUDA context, so a wrong GPU fails in a second instead of an hour in."""
import ctypes as C
import sys

SMS = 132
CAPABILITY = (9, 0)
# CUdevice_attribute values from cuda.h.
_MULTIPROCESSOR_COUNT = 16
_CAPABILITY_MAJOR = 75
_CAPABILITY_MINOR = 76
_NO_DEVICE = 100  # CUDA_ERROR_NO_DEVICE


class GpuError(RuntimeError):
    pass


def first_device():
    """(name, (major, minor), SM count) of the first visible CUDA device."""
    try:
        lib = C.CDLL('libcuda.so.1')
    except OSError as error:
        raise GpuError(f'no NVIDIA driver found ({error})') from None

    def call(name, *args):
        result = getattr(lib, name)(*args)
        if result == _NO_DEVICE:
            raise GpuError('no CUDA device is visible')
        if result:
            raise GpuError(f'{name} failed: CUDA error {result}')

    call('cuInit', C.c_uint(0))
    count = C.c_int()
    call('cuDeviceGetCount', C.byref(count))
    if count.value < 1:
        raise GpuError('no CUDA device is visible')
    device = C.c_int()
    call('cuDeviceGet', C.byref(device), C.c_int(0))
    name = C.create_string_buffer(256)
    call('cuDeviceGetName', name, C.c_int(len(name)), device)
    values = []
    for attribute in (_CAPABILITY_MAJOR, _CAPABILITY_MINOR, _MULTIPROCESSOR_COUNT):
        value = C.c_int()
        call('cuDeviceGetAttribute', C.byref(value), C.c_int(attribute), device)
        values.append(value.value)
    major, minor, sms = values
    return name.value.decode(), (major, minor), sms


def require_h200(describe=first_device):
    requirement = f'This build runs only on an NVIDIA H200 (sm_90, {SMS} SMs)'
    try:
        name, capability, sms = describe()
    except GpuError as error:
        sys.exit(f'{requirement}: {error}.')
    if 'H200' not in name or capability != CAPABILITY or sms != SMS:
        sys.exit(f'{requirement}; the first visible GPU is {name} '
                 f'(sm_{capability[0]}{capability[1]}, {sms} SMs). Refusing to run.')
