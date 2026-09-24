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
import logging
from typing import Dict, List, Type

from .rule_base import Rule

logger = logging.getLogger(__name__)

_REGISTERED: Dict[str, Type[Rule]] = {}


def register_rule(cls: Type[Rule]) -> Type[Rule]:
    if not issubclass(cls, Rule):
        raise TypeError(f"{cls.__name__} is not a subclass of Rule")
    rule_id = cls.RULE_ID
    if not rule_id:
        raise ValueError(f"{cls.__name__}.RULE_ID is not set")
    if rule_id in _REGISTERED:
        logger.warning("rule id is already registered, overriding: %s", rule_id)
    _REGISTERED[rule_id] = cls
    return cls


def get_registered_rules() -> List[Type[Rule]]:
    return [_REGISTERED[k] for k in sorted(_REGISTERED.keys())]


def get_rule(rule_id: str) -> Type[Rule]:
    if rule_id not in _REGISTERED:
        raise KeyError(f"unknown rule id: {rule_id}")
    return _REGISTERED[rule_id]
