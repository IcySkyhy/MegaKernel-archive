import ctypes

from mlx_lm import os_memory


def test_rusage_v4_layout_matches_the_darwin_abi():
    assert ctypes.sizeof(os_memory._RUsageInfoV4) == 296


def test_physical_footprint_returns_none_without_libproc(monkeypatch):
    monkeypatch.setattr(os_memory, "_LIBPROC", None)
    assert os_memory.physical_footprint_bytes() is None


def test_physical_footprint_reads_the_kernel_reply(monkeypatch):
    class FakeLibproc:
        @staticmethod
        def proc_pid_rusage(_pid, _flavor, pointer):
            info = ctypes.cast(
                pointer, ctypes.POINTER(os_memory._RUsageInfoV4)
            ).contents
            info.ri_phys_footprint = 123456
            return 0

    monkeypatch.setattr(os_memory, "_LIBPROC", FakeLibproc())
    assert os_memory.physical_footprint_bytes(42) == 123456


def test_physical_footprint_returns_none_on_probe_failure(monkeypatch):
    class FakeLibproc:
        @staticmethod
        def proc_pid_rusage(_pid, _flavor, _pointer):
            return 1

    monkeypatch.setattr(os_memory, "_LIBPROC", FakeLibproc())
    assert os_memory.physical_footprint_bytes(42) is None
