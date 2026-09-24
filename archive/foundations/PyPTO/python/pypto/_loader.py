# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from ._build_online import ensure_pypto_impl


def _load_shared_libs():
    import ctypes
    from importlib import metadata
    import logging
    import os
    from pathlib import Path
    from typing import Any, List, Optional

    _log = logging.getLogger(__name__)

    cur_dir: Path = Path(__file__).parent.resolve()
    try:
        pkg_dir: Path = Path(str(metadata.distribution("pypto").locate_file("pypto"))).resolve()
    except Exception:
        pkg_dir = cur_dir

    lib_search_dirs: List[Path] = [Path(cur_dir, "lib")]
    if pkg_dir != cur_dir:
        lib_search_dirs.append(Path(pkg_dir, "lib"))

    use_cann: bool = bool(os.environ.get("ASCEND_HOME_PATH"))

    def _find_lib(_name: str) -> Optional[Path]:
        for _dir in lib_search_dirs:
            _file = _dir / _name
            if _file.exists():
                return _file
        return None

    def _load_shared_lib(_desc: List[Any]):
        _name: str = _desc[0]
        _load: bool = _desc[1]
        if not _load:
            return
        _file = _find_lib(_name)
        if _file is None:
            _log.warning("Shared library %s not found in %s", _name, [str(d) for d in lib_search_dirs])
            return
        ctypes.CDLL(str(_file), mode=ctypes.RTLD_GLOBAL)

    _load_shared_lib(
        _desc=[
            "libc_sec.so",
            not use_cann,
        ]
    )

    # name, load
    desc_lst: List[List[Any]] = [
        [
            "libtile_fwk_utils.so",
            True,
        ],
        [
            "libtile_fwk_adapter.so",
            True,
        ],
        [
            "libtile_fwk_cann_host_runtime.so",
            True,
        ],
        [
            "libtile_fwk_platform.so",
            True,
        ],
        [
            "libtile_fwk_interface.so",
            True,
        ],
        [
            "libtile_fwk_codegen.so",
            True,
        ],
        [
            "libtile_fwk_compiler.so",
            True,
        ],
        [
            "libtile_fwk_runtime.so",
            True,
        ],
        [
            "libtile_fwk_simulation.so",
            True,
        ],
        [
            "libtile_fwk_simulation_pv.so",
            use_cann,
        ],
    ]
    for desc in desc_lst:
        _load_shared_lib(_desc=desc)


_load_shared_libs()
ensure_pypto_impl()
