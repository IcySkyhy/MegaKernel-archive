# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Megakernel backend for the Blade route.
#
# Reuses the whole triton-ascend stage pipeline (ttir -> ttadapter -> npubin) by
# subclassing ``AscendBackend``; the only injection is the megakernel bitcode in
# ``parse_options``, which has to happen before bishengir runs. The compiled
# object itself needs no interception -- the npubin stage output *is* the ``.o``
# bishengir produced, and triton already persists it in the kernel cache.
#
# Lives under a distinct target backend string so that ``make_backend``'s
# single-active-backend assertion still sees exactly one candidate for plain
# "npu" compiles. Registration happens in this package's __init__; selection is
# by passing ``target=`` explicitly to ``triton.compile``.
from __future__ import annotations

from pathlib import Path
from typing import Any

from triton.backends.ascend.compiler import AscendBackend
from triton.backends.compiler import GPUTarget

# NOTE: importing AscendBackend at module scope means this module exposes two
# concrete BaseBackend subclasses, which triton's _find_concrete_subclasses
# rejects. That only runs on the entry-point discovery path; registration here
# is done at runtime from __init__, which bypasses it. Switching to an entry
# point would require hiding this import inside a function.

MEGA_BACKEND = "npu-mega"


def get_tritondist(arch: str, mix_mode: str) -> str:
    """Path to the megakernel device bitcode.

    Installed alongside ``libdevice.10.bc`` in the ascend backend's ``lib/``, so
    the directory is derived from ``get_libdevice()`` rather than recomputed --
    that path has to be correct already or ordinary compiles would fail.
    """
    from triton.backends.ascend.compiler import get_libdevice
    lib_dir = Path(get_libdevice()).parent
    arch_suffix = "c310" if "Ascend950" in arch else "c220"
    core = "mix.aiv" if mix_mode and "mix" in mix_mode else "aiv"
    return str(lib_dir / f"Megakernel.{core}.{arch_suffix}.bc")


def mega_target() -> GPUTarget:
    """The current Ascend target, renamed to select :class:`MegaKernelBackend`.

    Derived from the live driver target rather than rebuilt, so arch and
    warp_size stay whatever the Ascend driver reports (it uses warp_size=0);
    only the backend string differs.
    """
    from triton.runtime.driver import driver
    npu = driver.active.get_current_target()
    return GPUTarget(MEGA_BACKEND, npu.arch, npu.warp_size)


class MegaKernelBackend(AscendBackend):
    """AscendBackend that links the megakernel bitcode and publishes the ``.o``.
    """

    @staticmethod
    def supports_target(target: GPUTarget):
        # Strictly MEGA_BACKEND: accepting "npu" too would make this a second
        # candidate in make_backend() and break every ordinary compile.
        return target.backend == MEGA_BACKEND

    def __init__(self, target: GPUTarget) -> None:
        self.mega_target = target
        super().__init__(target)
        object.__setattr__(self, "target", GPUTarget("npu", target.arch, target.warp_size))
        self.binary_ext = "npubin"
        self.binary_extensions = {"npubin", "mlirbc"}

    def hash(self):
        return str(self.mega_target)

    def parse_options(self, opts) -> Any:
        options = super().parse_options(opts)
        bc = get_tritondist(options.arch, getattr(options, "mix_mode", ""))
        object.__setattr__(options, "bisheng_options",
                           options.bisheng_options + " -cce-link-aicore-ll-module " + bc)
        return options


class MegaKernelDriver:
    """Inactive stub.

    Backend discovery pairs every compiler with a driver. ``is_active()``
    returns False so that ``_create_driver``'s single-active-driver assertion
    keeps resolving to the real Ascend driver -- megakernel objects are launched
    by Blade, not by triton's launcher.
    """

    @staticmethod
    def is_active():
        return False
