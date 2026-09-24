#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
在线编译基础设施模块.

本模块不 import pypto_impl 或任何依赖 pypto_impl 的子模块, 以确保在 pypto_impl.so 尚不存在时也可正常加载.

包含两种在线编译管理器:
- BuildOnlinePyptoImplManager: pypto_impl.so 在线编译
- BuildOnlineCalculatorManager: tile_fwk_calculator.so 在线编译
"""

import ctypes
import dataclasses
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

_log = logging.getLogger(__name__)


class _BuildOnlineManager:
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    @dataclasses.dataclass
    class _CMakeContext:
        cmake: Optional[Path] = None

        @dataclasses.dataclass
        class CompileContext:
            src_dir: Path  # 源码根目录, 对应 CMAKE_SOURCE_DIR
            tmp_dir: Path  # 临时目录
            cfg_cmd_ext: Optional[str] = None  # CMake Configure 阶段的扩展命令行
            build_type: str = "Release"  # 编译类型, 默认是 Release
            build_job_num: int = 32  # 32 为默认编译并行度
            install_prefix: Optional[Path] = None
            capture_output: bool = True  # 拦截调用 CMake 过程的输出

            @property
            def install_dir(self) -> Path:
                return self.install_prefix or Path(self.tmp_dir, "install")

            def run_cmd(self, cmd: str):
                ret = subprocess.run(
                    shlex.split(cmd),
                    text=True,
                    encoding='utf-8',
                    capture_output=self.capture_output,
                    check=not self.capture_output,
                )
                if ret.returncode != 0 and self.capture_output:
                    _log.error("cmd: %s, ret: %s", cmd, ret.returncode)
                    _log.error("stdout:\n%s", ret.stdout)
                    _log.error("stderr:\n%s", ret.stderr)
                ret.check_returncode()

        def __init__(self):
            self.cmake = self._which_cmake()
            if self.cmake is None:
                raise RuntimeError(
                    "Can not find cmake, please check your environment.\n"
                    "Hint: install system-level cmake (e.g. `apt install cmake` or `yum install cmake`), "
                    "the cmake pip package is NOT a valid substitute."
                )

        @classmethod
        def _which_cmake(cls) -> Optional[Path]:
            """查找系统级 CMake 可执行文件路径, 委托至 _which_cmake 公共模块."""
            from ._which_cmake import which_cmake

            return which_cmake()

        def compile(self, ctx: CompileContext) -> Path:
            """执行编译流程(包含 CMake 的 Configure, Build 及 Install 阶段)
             本函数会在临时目录 ctx.tmp_dir 路径下以下临时目录:
             1. 创建 build 子目录作为 CMAKE_BINARY_DIR
             2. 创建 install 子目录作为 CMAKE_INSTALL_PREFIX;

            :param ctx: 编译上下文
            :return: 安装路径, 对应 CMAKE_INSTALL_PREFIX
            :rtype: Path
            """
            # 路径准备
            build_dir = Path(ctx.tmp_dir, "build")
            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True)
            # CMake Configure
            cfg_cmd_ext = ctx.cfg_cmd_ext if ctx.cfg_cmd_ext else ""
            cmd = f"{self.cmake} -S {ctx.src_dir} -B {build_dir} -DCMAKE_BUILD_TYPE={ctx.build_type}"
            cmd += f" -DCMAKE_INSTALL_PREFIX={ctx.install_dir} {cfg_cmd_ext}"
            ctx.run_cmd(cmd=cmd)
            # CMake Build
            cmd = f"{self.cmake} --build {build_dir}" + (f" -j {ctx.build_job_num}" if ctx.build_job_num else "")
            ctx.run_cmd(cmd=cmd)
            # CMake Install
            cmd = f"{self.cmake} --install {build_dir} --prefix {ctx.install_dir}"
            ctx.run_cmd(cmd=cmd)
            return ctx.install_dir

    def __new__(cls):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        # 路径识别
        from importlib import metadata

        cur_dir: Path = Path(__file__).parent
        pkg_dir: Path = Path(str(metadata.distribution("pypto").locate_file("pypto"))).resolve()
        pkg_dir = pkg_dir if pkg_dir == cur_dir else cur_dir  # 适配 edit 模式
        # 变量赋值
        self.pkg_dir: Path = pkg_dir
        self.pkg_lib_dir: Path = Path(pkg_dir, "lib")
        self.pypto_version: str = metadata.version("pypto")
        _meta = metadata.metadata("pypto")
        self.run_version: str = _meta["X-Run-Version"] or ""
        self.build_timestamp: str = _meta["X-Build-Timestamp"] or ""
        if pkg_dir != cur_dir:
            # 非 edit 模式下, 必须存在
            if not self.run_version:
                raise RuntimeError("Can't get X-Run-Version in pypto METADATA")
            if not self.build_timestamp:
                raise RuntimeError("Can't get X-Build-Timestamp in pypto METADATA")
        self._initialized: bool = True  # 重复初始化守卫


class BuildOnlineCalculatorManager(_BuildOnlineManager):
    # 独立于 BuildOnlinePyptoImplManager 的编译锁, 避免二者互相阻塞
    _compile_lock: threading.Lock = threading.Lock()

    @dataclasses.dataclass
    class _TorchContext:
        torch_version: str = ""
        torch_root_dir: str = ""
        torch_c_use_cxx11_abi: int = 1

        def __init__(self):
            old_val = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")
            try:
                os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                import torch

                self.torch_version = str(torch.__version__)
                self.torch_root_dir = str(Path(torch.__file__).parent)
                self.torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
            except ImportError as e:
                raise RuntimeError(f"Can not import torch, please check your python environment. Error: {e}") from e
            finally:
                if old_val is None:
                    os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
                else:
                    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = old_val

    def __init__(self):
        if getattr(self, '_initialized', False):
            return  # 复用父类防重复初始化标记
        super().__init__()
        self._target_loaded: bool = False  # 目标二进制已加载进进程空间

    def build_and_load_calculator(self):
        if self._target_loaded:
            return
        # 双重检查锁定: 外层无锁快速检查, 内层加锁后二次检查, 避免多线程重复编译
        with self._compile_lock:
            if self._target_loaded:
                return
            torch_ctx = self._TorchContext()
            cmake_ctx = self._CMakeContext()
            with tempfile.TemporaryDirectory() as _tmp_dir:
                # 编译
                ext = f"-DPY3_MOD_TORCH_VERSION={torch_ctx.torch_version}"
                ext += f" -DPY3_MOD_TORCH_ROOT_PATH={torch_ctx.torch_root_dir}"
                ext += f" -DPY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI={torch_ctx.torch_c_use_cxx11_abi}"
                compile_ctx = self._CMakeContext.CompileContext(
                    src_dir=Path(self.pkg_lib_dir, "framework/src/calculator"), tmp_dir=Path(_tmp_dir)
                )
                compile_ctx.cfg_cmd_ext = ext
                install_prefix = cmake_ctx.compile(ctx=compile_ctx)
                # 加载
                calc_shared = Path(install_prefix, "lib/libtile_fwk_calculator.so")
                if not calc_shared.exists():
                    raise RuntimeError(f"{calc_shared} not exists.")
                ctypes.CDLL(str(calc_shared), mode=ctypes.RTLD_GLOBAL)
            self._target_loaded = True


class BuildOnlinePyptoImplManager(_BuildOnlineManager):
    _FLOCK_TIMEOUT: int = 300  # 多进程等待编译的超时阈值(秒)
    # 独立于 BuildOnlineCalculatorManager 的编译锁, 避免二者互相阻塞
    _compile_lock: threading.Lock = threading.Lock()

    class _PythonContext:
        _PYBIND11_MIN_VERSION = (2, 13, 6)

        def __init__(self):
            self.minor: int = 0
            self.pybind11_cmake_dir: Optional[Path] = None
            self._init_minor_version()
            self._init_development_component()
            self._init_pip_mod_pybind11()

        @staticmethod
        def _init_development_component():
            python_h = Path(sysconfig.get_path("include")) / "Python.h"
            if not python_h.exists():
                raise RuntimeError(
                    f"Python development headers not found (expected {python_h}).\n"
                    "Hint: install python3-dev (e.g. `apt install python3-dev` or `yum install python3-devel`)."
                )

        def _init_minor_version(self):
            minor = int(sys.version_info.minor)
            if minor < 9:
                raise RuntimeError(
                    f"Python version 3.{minor} is not supported for online compilation, require >= 3.9.\n"
                    "Hint: use a Python 3.9+ interpreter."
                )
            self.minor = minor

        def _init_pip_mod_pybind11(self):
            try:
                import pybind11
            except ImportError as e:
                raise RuntimeError(
                    "pybind11 pip package not found.\nHint: install it with `pip install pybind11>=2.13.6`."
                ) from e
            _ver_match = re.match(r'(\d+)\.(\d+)\.(\d+)', pybind11.__version__)
            if not _ver_match:
                raise RuntimeError(
                    f"Can't parse pybind11 version: {pybind11.__version__}.\n"
                    "Hint: install it with `pip install pybind11>=2.13.6`."
                )
            current_ver = tuple(int(x) for x in _ver_match.groups())
            if current_ver < self._PYBIND11_MIN_VERSION:
                raise RuntimeError(
                    f"pybind11 version {pybind11.__version__} is too old, require >= 2.13.6.\n"
                    "Hint: upgrade it with `pip install pybind11>=2.13.6`."
                )
            pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
            if not pybind11_dir or not pybind11_dir.exists():
                raise RuntimeError("pybind11 cmake dir empty.\nHint: install it with `pip install pybind11>=2.13.6`.")
            self.pybind11_cmake_dir = pybind11_dir

    def __init__(self):
        if getattr(self, '_initialized', False):
            return  # 复用父类防重复初始化标记
        super().__init__()
        self._target_compiled: bool = self._find_pypto_impl_so()[0]
        self._cache_dir_value: Optional[Path] = None
        ver_info = sys.version_info
        self._lock_name: str = f".pypto_impl_build.cp{ver_info.major}{ver_info.minor}.lock"

    @staticmethod
    def _get_ascend_path() -> str:
        """获取 CANN 安装路径, 用于缓存 key 区分不同 CANN 环境"""
        for env_key in ("CANN_PATH", "ASCEND_HOME_PATH"):
            val = os.environ.get(env_key)
            if val:
                return val
        opp_path = os.environ.get("ASCEND_OPP_PATH")
        if opp_path:
            return os.path.dirname(opp_path)
        return ""

    def ensure_pypto_impl(self):
        # 双重检查锁定: 外层无锁快速检查, 内层加锁后二次检查, 避免多线程重复编译
        if self._target_compiled:
            return
        with self._compile_lock:
            if self._target_compiled:
                return
            self._ensure_pypto_impl_locked()

    def _ensure_pypto_impl_locked(self):
        # 搜索已有产物 (pkg_dir 优先, cache_dir 次之)
        found, so_path = self._find_pypto_impl_so(cache_dir=self._cache_dir())
        if found:
            self._load_from_cache(so_path=so_path)
            self._target_compiled = True
            return

        src_dir = Path(self.pkg_lib_dir, "pypto_impl/src")
        if not src_dir.exists():
            # 当需要在线编译 pypto_impl 但是 whl 包内没有相关源码时, 不做在线编译.
            # 靠外部 import pypto_impl 的报错来暴露原始问题. 此处不截取该异常, 避免对原始问题的掩盖.
            _log.warning("Can't get pypto_impl, but it's src empty.")
            return

        # 跨用户协同: 检查是否有人正在往 pkg_dir 编译, 若有则等待共享结果
        if self._wait_for_pkg_dir_compilation():
            self._target_compiled = True
            return

        target_dir = self._cache_dir()
        lock_path = target_dir / self._lock_name

        lock_fd = None
        try:
            lock_fd = open(lock_path, 'w')
            if self._try_acquire_lock(lock_fd):
                # 获锁成功: 无其他同版本 Python 进程编译中
                so_path = self._do_compile(src_dir=src_dir, target_dir=target_dir)
            else:
                # 获锁失败: 其他进程正在编译, 轮询等待
                so_path = self._wait_and_compile(lock_fd, src_dir, target_dir, lock_path)

            if so_path is None:
                raise RuntimeError(
                    "Online compilation succeeded but pypto_impl.so not found in install output.\n"
                    f"Target directory: {target_dir}\n"
                    f"Searched suffixes: {importlib.machinery.EXTENSION_SUFFIXES}"
                )
            self._load_from_cache(so_path=so_path)
            self._target_compiled = True
        finally:
            # 在仍持有锁时删除 lock 文件, 防止 unlock 后, unlink 前的竞态窗口:
            # 若先 unlock 再 unlink, 其他进程可能 open 同一 inode 并获取锁,
            # 随后 unlink 使该 inode 不可达, 导致后续进程创建新 inode 并获取新锁,
            # 两个进程同时持有各自的"排他锁"并发编译.
            # 持锁时 unlink 后, 其他进程 open 只能创建新 inode, 无法获取旧 inode 的锁.
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_fd.close()

    def _poll_until(
        self, predicate: Callable[[], bool], timeout: Optional[float] = None, interval: float = 1.0
    ) -> bool:
        """轮询 predicate() 直至返回 True 或超时.

        :param predicate: 条件谓词, 返回 True 时停止轮询.
        :param timeout: 超时秒数, None 时使用 self._FLOCK_TIMEOUT.
        :param interval: 轮询间隔秒数.
        :return: True 表示谓词成功, False 表示超时.
        """
        if timeout is None:
            timeout = self._FLOCK_TIMEOUT
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def _wait_for_pkg_dir_compilation(self) -> bool:
        """检查是否有人正在往 pkg_dir 编译, 若有则等待共享结果.

        通过 os.path.exists() 只读检查 pkg_dir 下的 lock 文件是否存在.
        存在说明有人正在往共享位置编译, 当前进程等待 lock 消失后检查 .so 是否就绪.
        pkg_dir 不可读时跳过, 降级为独立编译.

        :return: True 表示等待后在 pkg_dir 中找到了 .so, 无需自己编译.
        """
        lock_path = self.pkg_dir / self._lock_name
        if not os.access(self.pkg_dir, os.R_OK):
            return False
        if not lock_path.exists():
            return False

        _log.info("Detected compilation in progress at %s, waiting...", lock_path)

        if not self._poll_until(lambda: not lock_path.exists()):
            _log.warning("Timeout waiting for pkg_dir compilation")
            return False

        found, _ = self._find_pypto_impl_so()
        return found

    def _try_acquire_lock(self, lock_fd) -> bool:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _wait_and_compile(self, lock_fd, src_dir: Path, target_dir: Path, lock_path: Path) -> Optional[Path]:
        _log.info("Waiting for compilation by another process (lock: %s)...", lock_path)

        if self._poll_until(lambda: self._try_acquire_lock(lock_fd)):
            _log.info("Acquired compilation lock after waiting")
            found, so_path = self._find_pypto_impl_so(cache_dir=target_dir)
            if found:
                return so_path
            return self._do_compile(src_dir, target_dir)

        _log.warning("Timeout waiting for compilation lock, will compile independently")
        return self._do_compile(src_dir=src_dir, target_dir=target_dir)

    def _do_compile(self, src_dir: Path, target_dir: Path) -> Optional[Path]:
        """执行编译并返回产物路径, 返回 None 表示编译成功但未找到产物"""
        pyenv_ctx = self._PythonContext()
        cmake_ctx = self._CMakeContext()

        with tempfile.TemporaryDirectory(prefix=f".pypto_impl_build.{os.getpid()}.") as tmp_dir:
            ext = f"-DPython3_EXECUTABLE={sys.executable}"
            ext += f" -DPython3_EXECUTABLE_VERSION=3.{pyenv_ctx.minor}"
            ext += f" -DPython3_MOD_PYBIND11_CMAKE_DIR={pyenv_ctx.pybind11_cmake_dir}"
            ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
            if ascend_home:
                ext += f" -DASCEND_CANN_PACKAGE_PATH={ascend_home}"
            compile_ctx = self._CMakeContext.CompileContext(src_dir=src_dir, tmp_dir=Path(tmp_dir))
            compile_ctx.cfg_cmd_ext = ext
            compile_ctx.install_prefix = target_dir
            cmake_ctx.compile(ctx=compile_ctx)

        _log.info("Compiled and installed pypto_impl.so to %s", target_dir)

        found, so_path = self._find_pypto_impl_so(cache_dir=target_dir)
        return so_path if found else None

    def _cache_dir(self) -> Path:
        """决定编译产物的目标路径, 进程内单例缓存"""
        if self._cache_dir_value is not None:
            return self._cache_dir_value

        self._cache_dir_value = self._compute_cache_dir()
        return self._cache_dir_value

    def _compute_cache_dir(self) -> Path:
        """通过归属用户判断 + 实际写入验证双重检测确定编译目标"""
        try:
            stat_info = self.pkg_dir.stat()
            current_uid = os.getuid()
            if stat_info.st_uid == current_uid:
                # 归属用户一致, 尝试实际写入验证
                pid = os.getpid()
                test_file = self.pkg_dir / f".pypto_writable_test.{pid}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    return self.pkg_dir
                except OSError:
                    pass  # 写入失败, 跳到缓存目录分支
        except OSError:
            pass

        # 缓存目录分支
        # 使用 XDG_CACHE_HOME 环境变量做缓存目录, edit 模式下 self.run_version 及 self.build_timestamp 为空
        # 补充说明:
        #   1. 绝大多数情况下: 不会配置 XDG_CACHE_HOME
        #      Linux 发行版 (Ubuntu, openEuler, Debian 等) 默认不预设该环境变量, 需要程序自动 fallback 到 ~/.cache
        #      若没有迁移缓存的需求, 完全没必要改, 所以几乎没人配置.
        #   2. 配置后很少改动
        #      2.1 大量工具会在启动时一次性读取该变量, 缓存目录一旦生成海量文件 (pip, ccache, 浏览器, 编译缓存),
        #          迁移缓存目录需要手动拷贝 / 同步所有缓存文件夹;
        #      2.2 直接改环境变量后旧缓存不会自动迁移, 软件会重新生成一份全新缓存, 浪费磁盘, 拖慢首次运行.
        #   3. 配置逻辑一般是长期规划:
        #      比如把缓存放到高速 SSD, 独立分区, 大容量存储, 属于固定磁盘布局方案, 布局不变就不用改变量.
        cache_dir = None
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            if Path(xdg_cache).is_absolute():
                cache_dir = Path(xdg_cache)
            else:
                _log.warning("XDG_CACHE_HOME=%s is not absolute, fallback to ~/.cache", xdg_cache)
        cache_dir = cache_dir if cache_dir else Path.home() / ".cache"
        cache_dir = cache_dir / "cann" / "pypto"
        if self.run_version:
            cache_dir = cache_dir / self.run_version
        cache_dir = cache_dir / self.pypto_version
        if self.build_timestamp:
            cache_dir = cache_dir / self.build_timestamp
        ascend_path = self._get_ascend_path()
        if ascend_path:
            ascend_hash = hashlib.md5(ascend_path.encode()).hexdigest()[:8]
            cache_dir = cache_dir / f"ascend_{ascend_hash}"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _find_pypto_impl_so(self, cache_dir: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
        """搜索 pypto_impl.so, 返回 (是否找到, 路径)

        搜索顺序: pkg_dir (标准位置) 优先; cache_dir 非空且 != pkg_dir 时再搜缓存目录.
        cache_dir 为 None 时仅搜 pkg_dir (用于 __init__ 快速判断, 不触发 _cache_dir() 的 I/O).
        """
        # 1. 先搜 pkg_dir (标准位置优先级最高)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            so_path = self.pkg_dir / f"pypto_impl{suffix}"
            if so_path.exists():
                _log.info("Found pypto_impl.so: %s", so_path)
                return True, so_path

        # 2. 若 cache_dir 提供且 != pkg_dir, 再搜 cache_dir
        if cache_dir is not None and cache_dir != self.pkg_dir:
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                so_path = cache_dir / f"pypto_impl{suffix}"
                if so_path.exists():
                    _log.info("Found pypto_impl.so in cache: %s", so_path)
                    return True, so_path

        return False, None

    def _load_from_cache(self, so_path: Path):
        """通过 importlib 动态加载缓存中的 .so 并注册到 sys.modules

        .so 在 pkg_dir 下时可被 Python 原生 import 发现, 无需手动注册;
        仅 .so 在缓存目录等非 pkg_dir 路径时才需 importlib 加载.
        """
        # .so 在 pkg_dir 下, Python 原生 import 可达, 无需注册
        if so_path.parent == self.pkg_dir:
            return
        # 幂等保护: 已加载则跳过
        if "pypto.pypto_impl" in sys.modules:
            return

        spec = importlib.util.spec_from_file_location("pypto.pypto_impl", so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {so_path}")

        module = importlib.util.module_from_spec(spec)

        # 注册必须在 exec_module 之前, 防止 pybind11 模块初始化时的递归 import
        sys.modules["pypto.pypto_impl"] = module

        # 同时设为父包属性, 确保 `from pypto import pypto_impl` 可达
        import pypto

        setattr(pypto, "pypto_impl", module)

        spec.loader.exec_module(module)
        _log.info("Loaded pypto_impl.so from cache: %s", so_path)


def ensure_pypto_impl():
    BuildOnlinePyptoImplManager().ensure_pypto_impl()
