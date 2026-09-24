#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest 配置控制"""

import os
from typing import List, Optional

import pytest


def duration_estimate(seconds: float):
    """
    Decorator: annotate a test case with estimated duration (seconds).

    This decorator marks test cases with their expected execution time,
    allowing pytest to reorder tests for optimal parallel execution.

    Args:
        seconds: Estimated execution time in seconds

    Example:
        @duration_estimate(120)
        def test_something():
            ...
    """

    def decorator(func):
        # Store the time cost as a public attribute on the function
        func.duration_estimate = seconds
        return func

    return decorator


def _set_process_desc(desc: str):
    try:
        import setproctitle

        setproctitle.setproctitle(desc)
    except ModuleNotFoundError:
        pass


def pytest_addoption(parser: pytest.Parser):
    """向 pytest 注册自定义参数

    :param parser: pytest.Parser 类型
    """
    parser.addoption("--device", nargs="+", type=int, help="Device ID, default 0")
    parser.addoption("--test_case_info", action="store", default="", help="Test case info.")
    parser.addoption(
        "--cards-per-case",
        type=int,
        default=1,
        help="Number of cards required for each test case. Default is 1 (single-card cases).",
    )


def pytest_configure(config):
    """pytest 启动时在 root scope 上设置 enable_slice=true。

    覆盖 tile_fwk_config.json 的 false 默认值。root scope 不受 config::Reset() / reset_options()
    影响，所以 UT SetUp 中的 Clear() 不会清除该设置。
    单个测试用例仍可通过 @pypto.options(pass_options={"enable_slice": False}) 覆盖此设置。
    """
    try:
        import pypto.pypto_impl

        pypto.pypto_impl.SetGlobalConfig({"pass.enable_slice": True})
    except Exception:
        pass


def _is_case_match_cards(item, target_cards) -> bool:
    """
    判断测试用例是否匹配目标卡数
    """
    cards_marker = item.get_closest_marker("world_size")
    if cards_marker is None:
        return True
    required_cards = cards_marker.args
    if not required_cards:
        return True

    if isinstance(required_cards[0], int):
        return target_cards == required_cards[0]

    return True


def pytest_configure_node(node):
    """pytest-xdist 回调函数, 在 pytest 主进程 fork 出 worker 进程之前被调用.

    :param node: worker 节点
    """
    # 获取 DeviceId 列表, 当外部传入 --device 时, 是 STest 场景, 否则是 UTest 场景
    device_id_lst: Optional[List[int]] = node.config.getoption("--device")
    cards_per_case: int = node.config.getoption("--cards-per-case", 1)

    if device_id_lst:
        if cards_per_case > 1:
            # 多卡模式
            if len(device_id_lst) % cards_per_case != 0:
                raise ValueError(f"Cannot divide {len(device_id_lst)} devices into groups of {cards_per_case}")

            # 计算worker应该分配的设备组
            num_groups = len(device_id_lst) // cards_per_case
            worker_idx = int(str(node.gateway.id).lstrip("gw"))

            if worker_idx >= num_groups:
                # 没有足够的设备组，这个worker不分配设备
                node.gateway.id = "NoDevices"
                node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID", None)')
                node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID_LIST", None)')
                return

            # 分配设备组
            start_idx = worker_idx * cards_per_case
            end_idx = start_idx + cards_per_case
            device_group = device_id_lst[start_idx:end_idx]
            device_group_str = ",".join(map(str, device_group))

            node.gateway.id = f"Devices[{device_group_str}]"
            node.gateway.remote_exec(f'import os; os.environ["TILE_FWK_DEVICE_ID_LIST"] = "{device_group_str}"')
        else:
            # 单卡模式，保持原有逻辑
            worker_idx = int(str(node.gateway.id).lstrip("gw"))
            if worker_idx >= len(device_id_lst):
                raise ValueError(f"WorkerIdx[{worker_idx}] out of DeviceIdLst{device_id_lst} range.")
            device_id: int = device_id_lst[worker_idx]

            # 修改 worker 名称, 设置 worker 中的 DeviceId
            node.gateway.id = f"Device[{device_id}]"  # 体现在回显中
            node.gateway.remote_exec(f'import os; os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"')
    else:
        node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID", None)')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # 优先使用设备组列表
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Devices[{','.join(device_list)}]")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Device[{device_id}]")
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """在 fork 子进程中手动保存 coverage 数据。

    pytest-forked 的子进程通过 ``os._exit()`` 退出，跳过 coverage 的
    ``atexit`` 数据写入，导致 ``--forked`` 模式下覆盖率全部丢失。
    此钩子复用 pytest-cov 已创建的 Coverage 实例 (已配置 source 和
    data_suffix)，在测试执行后手动 stop + save，数据写入带后缀的
    文件，最终由 pytest-cov 的 finish() 中的 combine() 自动合并。
    """
    cov_file = os.environ.get("COVERAGE_FILE")
    is_forked = item.config.getoption("forked", default=False)
    if not (cov_file and is_forked):
        yield
        return

    cov_plugin = item.config.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or cov_plugin.cov_controller is None:
        yield
        return

    cov = cov_plugin.cov_controller.cov
    if cov is None:
        yield
        return

    try:
        yield
    finally:
        cov.stop()
        cov.save()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """case 进程启动后被调用"""
    # 获取当前运行模式下的卡数
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    if device_list_str is not None:
        # 多卡
        device_list = device_list_str.split(",")
    else:
        # 单卡
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)

    # 设置进程描述
    case_name: str = str(item.name)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Case(Devices[{','.join(device_list)}]::{case_name})")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Case(Device[{device_id}]::{case_name})")
    return None  # 继续执行默认的测试流程


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """用例执行后清理 C++ Program 单例的全局状态。

    PyPTO 的 C++ 侧 ``Program`` 是进程级单例, 用例通过 ``pypto.function()``
    在其中累积 ``Function`` / ``Operation`` 图节点。``EndFunction`` 仅结束当前
    函数录制, 不会清理前序用例遗留的全局图对象, 导致同进程内连续执行多个用例时,
    后续用例在 ``GetGraphInfo`` / ``RemoveCallOpViewAssemble`` 等阶段访问到悬垂的
    ``shared_ptr<Operation>``, 触发 segfault。

    ``--forked`` 模式下每个用例运行在独立子进程中, 进程退出自动回收全局状态,
    因此不会复现。但串行 / xdist worker 内连续执行时, 需要在 teardown 阶段调用
    ``pypto_impl.Reset()`` (对应 ``Program::Reset()``) 清理全局图对象。

    仅对 UTest (python/tests/ut) 路径生效, 避免影响 STest 的 NPU 设备状态。
    """
    item_path = str(item.fspath).replace("\\", "/")
    if "/tests/ut/" not in item_path:
        return
    try:
        import pypto

        pypto.pypto_impl.Reset()
    except Exception:
        pass


def _get_test_time_cost(item):
    """
    获取测试用例的耗时信息

    Args:
        item: pytest 测试项

    Returns:
        int or None: 耗时秒数, 如果未标记则返回None
    """
    # 检查函数是否有duration_estimate属性
    if hasattr(item.function, 'duration_estimate'):
        return item.function.duration_estimate

    # 检查类是否有duration_estimate属性
    if hasattr(item, 'cls') and item.cls and hasattr(item.cls, 'duration_estimate'):
        return item.cls.duration_estimate

    time_marker = item.get_closest_marker("duration_estimate")
    if time_marker and time_marker.args:
        return time_marker.args[0]

    return None


def _get_soc_version():
    """
    从torch_npu获取soc version
    """
    try:
        import torch_npu

        soc_version = torch_npu.npu.get_soc_version()
        return soc_version
    except Exception as e:
        pytest.exit(f"Error: Failed to get soc version, error info: {str(e)}", returncode=1)
        return None


def _is_case_match_soc(item, target_soc):
    """
    判断测试用例是否匹配目标soc版本
    """
    soc_marker = item.get_closest_marker("soc")
    if soc_marker is None:
        supported_socs = ["910"]
    else:
        # 解析标记中的支持版本 (兼容单个/多个版本写法)
        supported_socs = soc_marker.args
        if isinstance(supported_socs[0], str):
            supported_socs = [soc.strip() for soc in supported_socs]
        elif isinstance(supported_socs[0], list):
            supported_socs = [soc.strip() for soc in supported_socs[0]]
    # 核心匹配逻辑：260的标签为 "950", 其余的标签为 "910"
    if target_soc == 260:
        target_tag = "950"
    else:
        target_tag = "910"
    return target_tag in supported_socs


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    在所有conftest.py作用域处理完成后进行全局重排序
    """
    if not items:
        return
    first_item = items[0]
    item_path = str(first_item.fspath)
    has_ut = "ut" in item_path.lower()

    def _is_verify_case(item):
        # Host pass_verify 用例：python/tests/ut/interpreter（原 tests/verify）
        return "ut/interpreter" in str(item.fspath).lower().replace("\\", "/")

    if has_ut:
        filtered_items = items
    else:
        verify_items = [item for item in items if _is_verify_case(item)]
        other_items = [item for item in items if not _is_verify_case(item)]
        # ut/interpreter 看护 Host pass_verify / SIM，不依赖 NPU soc 探测
        if other_items:
            target_soc = _get_soc_version()
            filtered_items = [item for item in other_items if _is_case_match_soc(item, target_soc)]
            filtered_items.extend(verify_items)
        else:
            filtered_items = verify_items

    # 根据卡数要求过滤用例
    cards_per_case = config.getoption("--cards-per-case", 1)

    # 在收集阶段就过滤掉不匹配的用例
    card_filtered_items = [item for item in filtered_items if _is_case_match_cards(item, cards_per_case)]

    # 分离有耗时标识和无耗时标识的测试用例
    timed_tests = []
    untimed_tests = []

    for item in card_filtered_items:
        time_cost = _get_test_time_cost(item)
        if time_cost is not None:
            timed_tests.append((item, time_cost))
        else:
            untimed_tests.append(item)

    timed_tests.sort(key=lambda x: x[1], reverse=True)
    reordered_items = [item for item, _ in timed_tests] + untimed_tests

    items[:] = reordered_items
