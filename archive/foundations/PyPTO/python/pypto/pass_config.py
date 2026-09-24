#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

__all__ = [
    "PassConfigs",
    "PassConfigKey",
    "get_pass_default_config",
    "set_pass_default_config",
    "get_pass_config",
    "set_pass_config",
    "get_pass_configs",
]
from enum import Enum

from pypto.error import PassError

from . import pypto_impl


class PassConfigs:
    """
    PassConfigs data structure returned from C++ (read-only)

    Attributes:
        printGraph: Whether to print function IR.
        dumpGraph: Whether to dump graph to files.
        dumpPassTimeCost: Whether to dump time consumption of pass.
        preCheck: Whether to perform validation checks before pass.
        postCheck: Whether to perform verification checks after pass.
        disablePass: Whether to disable pass.
        healthCheck: Whether to perform health check and generate report.

    Note:
        Instances of this class are entirely created and initialized by the C++ side.
        Python side is only for data access. All attributes are read-only and cannot
        be modified from Python.
    """

    printGraph: bool  # noqa: N815
    dumpGraph: bool  # noqa: N815
    dumpPassTimeCost: bool  # noqa: N815
    preCheck: bool  # noqa: N815
    postCheck: bool  # noqa: N815
    disablePass: bool  # noqa: N815
    healthCheck: bool  # noqa: N815


class PassConfigKey(Enum):
    KEY_DUMP_GRAPH = pypto_impl.KEY_DUMP_GRAPH


def get_pass_default_config(key: PassConfigKey, default_value: bool) -> bool:
    """
    Get default pass configuration value by key.

    Parameters
    ---------
    key: PassConfigKey
        The configuration key to retrieve. Must be one of the valid enum value from `PassConfigKey`.

    default_value: bool
        The default value to return if the key is not found.

    Returns
    ---------
    bool
        The configuration value for the specified key, or `default_value` if the key is not found.

    Raises
    ---------
    ValueError
        If the key is not a valid PassConfigKey enum value.
    """
    if not isinstance(default_value, bool):
        raise PassError(TypeError(f"Expected boolean type, but received {type(default_value).__name__}"))
    if not isinstance(key, PassConfigKey):
        raise PassError(ValueError(f"key must be a member of PassConfigKey, got {key}. "))
    return pypto_impl.GetPassDefaultConfig(key.value, default_value)


def set_pass_default_config(key: PassConfigKey, value: bool):
    """
    Set default pass configuration value by key.

    Parameters
    ---------
    key: PassConfigKey
        The configuration key to update. Must be one of the valid enum value from `PassConfigKey`.

    value: bool
        The new value to associate with the configuration key.

    Raises
    ---------
    ValueError
        If the key is not a valid PassConfigKey enum value.
    """
    if not isinstance(value, bool):
        raise PassError(TypeError(f"Expected boolean type, but received {type(value).__name__}"))
    if not isinstance(key, PassConfigKey):
        raise PassError(ValueError(f"key must be a member of PassConfigKey, got {key}. "))
    pypto_impl.SetPassDefaultConfig(key.value, value)


def get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool:
    """
    Get specific pass configuration value by strategy.identifier.key path.

    Parameters
    ---------
    strategy: str
        The configuration strategy category.

    identifier: str
        The pass name within the strategy.

    key: PassConfigKey
        The configuration key to retrieve. Must be one of the valid enum values from `PassConfigKey`.

    default_value: bool
        The default value to return if the configuration item defined by strategy.identifier.key is not found.

    Returns
    ---------
    bool
        The configuration value for the specified key, or `default_value` if the key is not found.

    Raises
    ---------
    ValueError
        If the key is not a valid PassConfigKey enum value.
    """
    if not isinstance(default_value, bool):
        raise PassError(TypeError(f"Expected boolean type, but received {type(default_value).__name__}"))
    if not isinstance(key, PassConfigKey):
        raise PassError(ValueError(f"key must be a member of PassConfigKey, got {key}. "))
    return pypto_impl.GetPassConfig(strategy, identifier, key.value, default_value)


def set_pass_config(strategy: str, identifier: str, key: PassConfigKey, value: bool):
    """
    Set specific pass configuration value by strategy.identifier.key path.

    Parameters
    ---------
    strategy: str
        The configuration strategy category.

    identifier: str
        The pass name within the strategy.

    key: PassConfigKey
        The configuration key to update. Must be one of the valid enum values from `PassConfigKey`.

    value: bool
        The new value to associate with the configuration key.

    Raises
    ---------
    ValueError
        If the key is not a valid PassConfigKey enum value.
    """
    if not isinstance(value, bool):
        raise PassError(TypeError(f"Expected boolean type, but received {type(value).__name__}"))
    if not isinstance(key, PassConfigKey):
        raise PassError(ValueError(f"key must be a member of PassConfigKey, got {key}. "))
    pypto_impl.SetPassConfig(strategy, identifier, key.value, value)


def get_pass_configs(strategy: str, identifier: str) -> PassConfigs:
    """
    Get the complete configuration object for a specific pass.

    Parameters
    ---------
    strategy: str
        The configuration strategy category.

    identifier: str
        The pass name within the strategy.

    Returns
    ---------
    PassConfigs
        A complete configuration object containing all parameters for the specified pass under the given strategy.
    """
    return pypto_impl.GetPassConfigs(strategy, identifier)
