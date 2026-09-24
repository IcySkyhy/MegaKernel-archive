#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Ascend NPU 硬件深度检测模块。

瀑布式多级检测策略，从零依赖到高级运行时逐级探测：

  Level 0: lspci PCI device ID 检测（最高优先级，匹配 19e5:d802/d803）
  Level 0a: PCI sysfs 扫描 + /dev/davinci* 枚举（无需任何软件）
  Level 1: npu-smi 命令行（需驱动，含 LD_LIBRARY_PATH 自动修复 + PCI Device ID 解析）
  Level 2: ctypes 加载 libascendcl.so → aclrtGetSocName()（需 CANN）
  Level 3: torch_npu Python 包（需 PyTorch + torch_npu）
  Level 4: import acl Python 绑定（需 Python ACL）

所有可用方法均会执行并交叉验证，以获得最高置信度结果。

SoC 版本映射数据来源：
  https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NpuVariables.h
  https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NpuVariables.cpp
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, field
import glob
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 常量 & 映射表
# ---------------------------------------------------------------------------

ASCEND_ROOT = os.environ.get("ASCEND_INSTALL_PATH", "/usr/local/Ascend")

HUAWEI_PCI_VENDOR_ID = "0x19e5"
NPU_PCI_CLASS_PREFIX = "0x1200"  # Processing accelerators (class 12, subclass 00)

# 已知非 NPU 的海思 PCI 加速器设备（ZIP/SEC 引擎等，共享 class 0x120000）
KNOWN_NON_NPU_DEVICE_IDS: set[str] = {"0xa250", "0xa251", "0xa255", "0xa256"}

# SoC 版本号 → 芯片名映射（来源：Ascend/pytorch NpuVariables.h 的 c10_npu::SocVersion 枚举）
SOC_VERSION_MAP: dict[int, str] = {
    # Ascend 910 (A1)
    100: "Ascend910PremiumA",
    101: "Ascend910ProA",
    102: "Ascend910A",
    103: "Ascend910ProB",
    104: "Ascend910B",
    # Ascend 310P (A1+ 推理)
    200: "Ascend310P1",
    201: "Ascend310P2",
    202: "Ascend310P3",
    203: "Ascend310P4",
    204: "Ascend310P5",
    206: "Ascend310P7",
    # Ascend 910B (A2 训练)
    220: "Ascend910B1",
    221: "Ascend910B2",
    222: "Ascend910B2C",
    223: "Ascend910B3",
    224: "Ascend910B4",
    225: "Ascend910B4-1",
    # Ascend 310B (A2 推理)
    240: "Ascend310B1",
    241: "Ascend310B2",
    242: "Ascend310B3",
    243: "Ascend310B4",
    # Ascend 910_93xx (A3)
    250: "Ascend910_9391",
    251: "Ascend910_9392",
    252: "Ascend910_9381",
    253: "Ascend910_9382",
    254: "Ascend910_9372",
    255: "Ascend910_9362",
    # Ascend 950 (A3+)
    260: "Ascend950",
}

CHIP_NAME_TO_SOC_VERSION: dict[str, int] = {v: k for k, v in SOC_VERSION_MAP.items()}
CHIP_NAME_TO_SOC_VERSION["Ascend950"] = 260  # alias

# Chip family classification: (regex, family, generation, primary_use)
CHIP_FAMILIES: list[tuple[str, str, str, str]] = [
    (r"^Ascend910Premium|^Ascend910Pro[AB]|^Ascend910[AB]$", "Ascend910", "A1", "Training"),
    (r"^Ascend310P", "Ascend310P", "A1+", "Inference"),
    (r"^Ascend910B[1-4]", "Ascend910B", "A2", "Training"),
    (r"^Ascend310B", "Ascend310B", "A2", "Inference"),
    (r"^Ascend910_93", "Ascend910_93", "A3", "Training"),
    (r"^Ascend950", "Ascend950", "A3+", "Training"),
]

# Known Huawei PCI device IDs -> (generation, chip family prefix)
KNOWN_PCI_DEVICE_IDS: dict[str, tuple[str, str]] = {
    "0xd802": ("A2", "Ascend910B"),
    "0xd803": ("A3", "Ascend910_93"),
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class NPUDevice:
    """单个 NPU 设备信息。"""

    index: int
    chip_name: Optional[str] = None
    soc_version: Optional[int] = None
    pci_bus_id: Optional[str] = None
    pci_device_id: Optional[str] = None
    total_memory_bytes: Optional[int] = None


@dataclass
class NPUDetectionResult:
    """聚合检测结果。"""

    npu_present: bool = False
    device_count: int = 0
    chip_name: Optional[str] = None
    chip_family: Optional[str] = None
    generation: Optional[str] = None
    primary_use: Optional[str] = None
    soc_version_int: Optional[int] = None
    driver_version: Optional[str] = None
    cann_version: Optional[str] = None
    devices: list[NPUDevice] = field(default_factory=list)
    detection_methods: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _device_count_confidence: int = 0  # 内部字段：0=无, 1=pci, 2=davinci, 3=npu-smi, 4=acl, 5=torch_npu

    @property
    def device_type(self) -> Optional[str]:
        """将 generation 映射为 PyPTO 的 device_type（a1/a2/a3/a3+）。"""
        if self.generation:
            return self.generation.lower().replace("+", "+")
        return None

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("_device_count_confidence", None)
        return d

    def update_device_count(self, count: int, confidence: int, use_max: bool = False) -> None:
        if confidence > self._device_count_confidence:
            self.device_count = max(self.device_count, count) if use_max else count
            self._device_count_confidence = confidence
            return
        if use_max:
            self.device_count = max(self.device_count, count)

    def summary(self) -> str:
        if not self.npu_present:
            return "No Ascend NPU detected."
        lines = [
            f"Ascend NPU: {self.chip_name or 'Unknown'}",
            f"  Family:     {self.chip_family or 'Unknown'}",
            f"  Generation: {self.generation or 'Unknown'}",
            f"  Use:        {self.primary_use or 'Unknown'}",
            f"  SoC Int:    {self.soc_version_int}",
            f"  Devices:    {self.device_count}",
            f"  Driver:     {self.driver_version or 'N/A'}",
            f"  CANN:       {self.cann_version or 'N/A'}",
            f"  Methods:    {', '.join(self.detection_methods)}",
        ]
        if self.errors:
            lines.append(f"  Warnings:   {'; '.join(self.errors)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 检测方法（瀑布序）
# ---------------------------------------------------------------------------


def _classify_chip(chip_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """芯片名 → (family, generation, primary_use)。"""
    for pattern, family, gen, use in CHIP_FAMILIES:
        if re.match(pattern, chip_name):
            return family, gen, use
    return None, None, None


def _detect_lspci(result: NPUDetectionResult) -> None:
    """Level 0 (highest priority): 通过 lspci -n -D 检测华为 NPU PCI 设备。

    直接匹配已知的 PCI device ID (0xd803=A3, 0xd802=A2)，
    优先级最高，不需要任何软件依赖。
    """
    try:
        out = subprocess.run(
            ["/usr/bin/lspci", "-n", "-D"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    # 匹配格式: 0000:81:00.0 1200: 19e5:d803 (rev 20)
    npu_lines: list[tuple[str, str]] = []  # (bus_id, device_id_hex)
    for line in out.stdout.splitlines():
        m = re.search(r"(\S+)\s+\S+:\s+19e5:(d80[23])\b", line, re.IGNORECASE)
        if m:
            npu_lines.append((m.group(1), m.group(2).lower()))

    if not npu_lines:
        return

    result.npu_present = True
    result.detection_methods.insert(0, "lspci")

    # 以第一个设备的 device ID 决定芯片代际
    first_dev_id = f"0x{npu_lines[0][1]}"
    if first_dev_id in KNOWN_PCI_DEVICE_IDS:
        gen_label, family = KNOWN_PCI_DEVICE_IDS[first_dev_id]
        if result.generation is None:
            result.generation = gen_label
        if result.chip_family is None:
            result.chip_family = family
        # 从 family 推断 primary_use
        for _, fam, _, use in CHIP_FAMILIES:
            if fam == family:
                if result.primary_use is None:
                    result.primary_use = use
                break

    # 更新设备计数（低置信度，但比无检测好）
    result.update_device_count(len(npu_lines), confidence=1)

    # 填充设备列表
    for bus_id, dev_id_hex in npu_lines:
        result.devices.append(
            NPUDevice(
                index=len(result.devices),
                pci_bus_id=bus_id,
                pci_device_id=f"0x{dev_id_hex}",
            )
        )


def _detect_pci_sysfs(result: NPUDetectionResult) -> None:
    """Level 0a: 扫描 /sys/bus/pci/devices/ 中的华为 NPU PCI 设备。"""
    pci_base = Path("/sys/bus/pci/devices")
    if not pci_base.exists():
        return

    npu_pci_devices: list[dict[str, str]] = []
    for dev_dir in pci_base.iterdir():
        try:
            vendor = (dev_dir / "vendor").read_text().strip()
            if vendor != HUAWEI_PCI_VENDOR_ID:
                continue
            device_id = (dev_dir / "device").read_text().strip()
            pci_class = (dev_dir / "class").read_text().strip()
            if not pci_class.startswith(NPU_PCI_CLASS_PREFIX):
                continue
            if device_id in KNOWN_NON_NPU_DEVICE_IDS:
                continue
            npu_pci_devices.append(
                {
                    "bus_id": dev_dir.name,
                    "device_id": device_id,
                    "class": pci_class,
                }
            )
        except (OSError, IOError):
            continue

    if npu_pci_devices:
        result.npu_present = True
        result.detection_methods.append("pci_sysfs")
        result.update_device_count(len(npu_pci_devices), confidence=1)
        first_dev_id = npu_pci_devices[0]["device_id"]
        if first_dev_id in KNOWN_PCI_DEVICE_IDS:
            gen_label, family = KNOWN_PCI_DEVICE_IDS[first_dev_id]
            if result.generation is None:
                result.generation = gen_label
            if result.chip_family is None:
                result.chip_family = family
            for _, fam, _, use in CHIP_FAMILIES:
                if fam == family:
                    if result.primary_use is None:
                        result.primary_use = use
                    break
        # 跳过已由 _detect_lspci() 添加的设备（按 pci_bus_id 去重）
        existing_bus_ids = {d.pci_bus_id for d in result.devices if d.pci_bus_id}
        for pci_dev in npu_pci_devices:
            if pci_dev["bus_id"] not in existing_bus_ids:
                result.devices.append(
                    NPUDevice(
                        index=len(result.devices),
                        pci_bus_id=pci_dev["bus_id"],
                        pci_device_id=pci_dev["device_id"],
                    )
                )


def _detect_dev_davinci(result: NPUDetectionResult) -> None:
    """Level 0b: 检查 /dev/davinci* 设备文件。"""
    davinci_devs = sorted(glob.glob("/dev/davinci[0-9]*"))
    has_manager = os.path.exists("/dev/davinci_manager")
    has_hisi_hdc = os.path.exists("/dev/hisi_hdc")

    if davinci_devs or has_manager or has_hisi_hdc:
        result.npu_present = True
        result.detection_methods.append("dev_davinci")
        if davinci_devs:
            result.update_device_count(len(davinci_devs), confidence=2)


def _detect_driver_version(result: NPUDetectionResult) -> None:
    """读取 Ascend 驱动版本。"""
    for vp in [
        os.path.join(ASCEND_ROOT, "driver", "version.info"),
        os.path.join(ASCEND_ROOT, "latest", "driver", "version.info"),
    ]:
        try:
            content = Path(vp).read_text()
            m = re.search(r"Version\s*=\s*(.+)", content)
            if m:
                result.driver_version = m.group(1).strip()
                return
        except (OSError, IOError):
            continue


def _detect_cann_version(result: NPUDetectionResult) -> None:
    cann_paths = sorted(glob.glob(os.path.join(ASCEND_ROOT, "cann-*")), reverse=True)
    if not cann_paths:
        cann_paths = sorted(glob.glob(os.path.join(ASCEND_ROOT, "ascend-toolkit", "*")), reverse=True)

    for cp in cann_paths:
        m = re.search(r"cann-(\d+\.\d+\.\d+)", cp)
        if m:
            result.cann_version = m.group(1)
            return
        cfg = os.path.join(cp, "version.cfg")
        if os.path.exists(cfg):
            try:
                content = Path(cfg).read_text()
                vm = re.search(r"version\s*=\s*(.+)", content, re.IGNORECASE)
                if vm:
                    result.cann_version = vm.group(1).strip()
                    return
            except (OSError, IOError):
                continue


def _detect_npu_smi(result: NPUDetectionResult) -> None:
    """Level 1: 通过 npu-smi 命令检测。

    若 npu-smi 因缺少共享库失败，会自动尝试补充 LD_LIBRARY_PATH 后重试。
    如果 chip_name 未通过其他方式获取，还会通过 npu-smi info -t board -i 0
    解析 PCI Device ID 来确定芯片代际。
    """
    env = os.environ.copy()

    def _run_npu_smi(args: list[str], run_env: dict[str, str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
                env=run_env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    out = _run_npu_smi(["npu-smi", "info"], env)

    # 如果 npu-smi 失败且 stderr 包含 .so 错误，补充驱动库路径后重试
    if out is not None and out.returncode != 0:
        stderr_text = (out.stderr or "").lower()
        if "error while loading shared librar" in stderr_text or ".so" in stderr_text:
            driver_lib_dirs = [
                os.path.join(ASCEND_ROOT, "driver", "lib64", "common") + "/",
                os.path.join(ASCEND_ROOT, "driver", "lib64", "driver") + "/",
            ]
            existing = [d for d in driver_lib_dirs if os.path.isdir(d)]
            if existing:
                ld_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(existing) + (":" + ld_path if ld_path else "")
                out = _run_npu_smi(["npu-smi", "info"], env)

    if out is None or out.returncode != 0:
        return

    result.detection_methods.append("npu-smi")
    result.npu_present = True

    for line in out.stdout.splitlines():
        m = re.match(r"\s*Chip\s+Name\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip().replace(" ", "")
            if result.chip_name is None:
                result.chip_name = raw_name
            break

    npu_ids: set[int] = set()
    for line in out.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+\d+\s+", line)
        if m:
            npu_ids.add(int(m.group(1)))
    if npu_ids:
        result.update_device_count(len(npu_ids), confidence=3)

    # 如果前面的方法尚未确定代际，通过 npu-smi info -t board -i 0 解析 PCI Device ID
    if result.generation is None:
        board_out = _run_npu_smi(["npu-smi", "info", "-t", "board", "-i", "0"], env)
        if board_out is not None and board_out.returncode == 0:
            for line in board_out.stdout.splitlines():
                m = re.match(r"\s*PCI\s+Device\s+ID\s*:\s*(0x[0-9A-Fa-f]+)", line)
                if m:
                    pci_dev_id = m.group(1).lower()
                    if pci_dev_id in KNOWN_PCI_DEVICE_IDS:
                        gen_label, family = KNOWN_PCI_DEVICE_IDS[pci_dev_id]
                        result.generation = gen_label
                        if result.chip_family is None:
                            result.chip_family = family
                        for _, fam, _g, use in CHIP_FAMILIES:
                            if fam == family:
                                if result.primary_use is None:
                                    result.primary_use = use
                                break
                    break


def _detect_acl_ctypes(result: NPUDetectionResult) -> None:
    """Level 2: 通过 ctypes 加载 libascendcl.so 检测。"""
    acl_lib_paths = [
        os.path.join(ASCEND_ROOT, "cann-*", "lib64", "libascendcl.so"),
        os.path.join(ASCEND_ROOT, "ascend-toolkit", "latest", "lib64", "libascendcl.so"),
        os.path.join(ASCEND_ROOT, "latest", "lib64", "libascendcl.so"),
    ]

    lib_path = None
    for pattern in acl_lib_paths:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            lib_path = matches[0]
            break

    if lib_path is None:
        return

    try:
        acl = ctypes.CDLL(lib_path)
        acl_detected = False

        acl.aclInit.restype = ctypes.c_int
        acl.aclInit.argtypes = [ctypes.c_char_p]
        ret = acl.aclInit(None)
        if ret != 0 and ret != 100000:  # 100000 = ACL_ERROR_REPEAT_INITIALIZE
            result.errors.append(f"aclInit failed with code {ret}")
            return

        acl.aclrtGetSocName.restype = ctypes.c_char_p
        acl.aclrtGetSocName.argtypes = []
        soc_name_raw = acl.aclrtGetSocName()
        if soc_name_raw:
            chip_name = soc_name_raw.decode("utf-8")
            if result.chip_name is None:
                result.chip_name = chip_name
            acl_detected = True
            if chip_name in CHIP_NAME_TO_SOC_VERSION and result.soc_version_int is None:
                result.soc_version_int = CHIP_NAME_TO_SOC_VERSION[chip_name]

        try:
            count = ctypes.c_uint32(0)
            acl.aclrtGetDeviceCount.restype = ctypes.c_int
            acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            ret = acl.aclrtGetDeviceCount(ctypes.byref(count))
            if ret == 0 and count.value > 0:
                result.update_device_count(int(count.value), confidence=4, use_max=True)
                acl_detected = True
        except Exception as e:
            result.errors.append(f"aclrtGetDeviceCount failed: {e}")

        if acl_detected:
            result.npu_present = True
            if "acl_ctypes" not in result.detection_methods:
                result.detection_methods.append("acl_ctypes")

        # 不调用 aclFinalize()，避免影响后续 torch_npu 检测
    except OSError as e:
        result.errors.append(f"Failed to load {lib_path}: {e}")


def _detect_torch_npu(result: NPUDetectionResult) -> None:
    """Level 3: 通过 torch_npu Python 包检测。"""
    probe_code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    import torch_npu\n"
        "except Exception:\n"
        "    print(json.dumps({'ok': False}))\n"
        "    raise SystemExit(0)\n"
        "out = {'ok': True, 'device_count': 0, 'chip_name': None, 'soc_version_int': None}\n"
        "try:\n"
        "    out['device_count'] = int(torch.npu.device_count())\n"
        "except Exception:\n"
        "    out['device_count'] = 0\n"
        "if out['device_count'] > 0:\n"
        "    try:\n"
        "        out['chip_name'] = torch.npu.get_device_name(0)\n"
        "    except Exception:\n"
        "        pass\n"
        "try:\n"
        "    soc_ver = torch_npu._C._npu_get_soc_version()\n"
        "    if isinstance(soc_ver, int) and soc_ver >= 0:\n"
        "        out['soc_version_int'] = soc_ver\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )

    try:
        p = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as e:
        result.errors.append(f"torch_npu probe process failed: {e}")
        return

    if p.returncode != 0 or not p.stdout.strip():
        return

    try:
        probe = json.loads(p.stdout.strip())
    except Exception as e:
        result.errors.append(f"torch_npu probe output parse failed: {e}")
        return

    if not isinstance(probe, dict) or not probe.get("ok"):
        return

    dev_count_raw = probe.get("device_count")
    dev_count = dev_count_raw if isinstance(dev_count_raw, int) else 0
    if dev_count <= 0:
        return

    result.npu_present = True
    result.update_device_count(dev_count, confidence=5)
    result.detection_methods.append("torch_npu")

    chip_name_raw = probe.get("chip_name")
    if isinstance(chip_name_raw, str) and chip_name_raw:
        result.chip_name = result.chip_name or chip_name_raw

    soc_ver_raw = probe.get("soc_version_int")
    soc_ver = soc_ver_raw if isinstance(soc_ver_raw, int) else None
    if soc_ver is not None:
        result.soc_version_int = result.soc_version_int or soc_ver
        if result.chip_name is None and soc_ver in SOC_VERSION_MAP:
            result.chip_name = SOC_VERSION_MAP[soc_ver]

    for i in range(dev_count):
        dev = next((d for d in result.devices if d.index == i), None)
        if dev is None:
            dev = NPUDevice(index=i)
            result.devices.append(dev)
        if dev.chip_name is None and isinstance(chip_name_raw, str) and chip_name_raw:
            dev.chip_name = chip_name_raw
        if dev.soc_version is None and soc_ver is not None:
            dev.soc_version = soc_ver


def _detect_python_acl(result: NPUDetectionResult) -> None:
    """Level 4: 通过 Python acl 包检测。"""
    try:
        import acl  # type: ignore  # pyright: ignore[reportMissingImports]
    except ImportError:
        return

    try:
        python_acl_detected = False
        device_count, ret = acl.rt.get_device_count()
        if ret == 0 and device_count > 0:
            result.npu_present = True
            result.update_device_count(device_count, confidence=4, use_max=True)
            python_acl_detected = True

        soc_name = acl.get_soc_name()
        if soc_name:
            result.chip_name = result.chip_name or soc_name
            python_acl_detected = True

        if python_acl_detected:
            result.npu_present = True
            if "python_acl" not in result.detection_methods:
                result.detection_methods.append("python_acl")

    except Exception as e:
        result.errors.append(f"python acl error: {e}")


# ---------------------------------------------------------------------------
# 主检测编排
# ---------------------------------------------------------------------------


def detect_npu() -> NPUDetectionResult:
    """执行全部检测方法（瀑布序）并返回聚合结果。

    所有方法均会尝试，结果交叉验证以获得最高置信度。
    """
    result = NPUDetectionResult()

    _detect_lspci(result)  # Level 0 最高优先：lspci PCI device ID
    _detect_pci_sysfs(result)  # Level 0a: PCI sysfs 扫描
    _detect_dev_davinci(result)
    _detect_driver_version(result)
    _detect_cann_version(result)
    _detect_npu_smi(result)
    _detect_acl_ctypes(result)
    _detect_torch_npu(result)
    _detect_python_acl(result)

    # 芯片家族分类（仅当 PCI 检测未设置时才从 chip_name 推断）
    if result.chip_name:
        family, gen, use = _classify_chip(result.chip_name)
        if result.chip_family is None:
            result.chip_family = family
        if result.generation is None:
            result.generation = gen
        if result.primary_use is None:
            result.primary_use = use

    # 交叉验证：SoC 版本号与芯片名是否一致
    if result.soc_version_int is not None and result.chip_name:
        expected_name = SOC_VERSION_MAP.get(result.soc_version_int)
        if expected_name and expected_name != result.chip_name:
            result.errors.append(
                f"SoC version {result.soc_version_int} maps to {expected_name} "
                f"but detected chip name is {result.chip_name}"
            )

    # 从全局结果回填每个设备的缺失字段
    for dev in result.devices:
        if dev.chip_name is None and result.chip_name:
            dev.chip_name = result.chip_name
        if (dev.soc_version is None or dev.soc_version < 0) and result.soc_version_int is not None:
            dev.soc_version = result.soc_version_int

    # 按 index 去重
    seen: set[int] = set()
    unique: list[NPUDevice] = []
    for dev in result.devices:
        if dev.index not in seen:
            seen.add(dev.index)
            unique.append(dev)
    result.devices = unique

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Ascend NPU hardware detection')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--summary', action='store_true', help='Human-readable summary')
    args = parser.parse_args()
    result = detect_npu()
    if args.json:
        logging.info(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        logging.info(result.summary())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
