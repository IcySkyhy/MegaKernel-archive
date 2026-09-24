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
from typing import List

from ..models import Category, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule

_REUSE_KINDS = {"reuse", "Reuse", "REUSE"}


@register_rule
class StitchLegalityRule(Rule):
    RULE_ID = "rule_stitch_legality"
    DESCRIPTION = "Runtime read/write linkage must reference declared producer/consumer tensors"
    CATEGORY = Category.ILLEGAL_LINKAGE

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for edge in ctx.stitch_edges:
            prod_op = ctx.get_static_op(edge.producer_func_key, edge.producer_op_idx)
            cons_op = ctx.get_static_op(edge.consumer_func_key, edge.consumer_op_idx)

            if prod_op is None or cons_op is None:
                violations.append(
                    Violation(
                        rule_id=self.RULE_ID,
                        slot_idx=edge.slot_idx,
                        message=(
                            f"read/write linkage references an undeclared kernel op "
                            f"(producer funcKey={edge.producer_func_key}/"
                            f"opIdx={edge.producer_op_idx}, consumer funcKey="
                            f"{edge.consumer_func_key}/opIdx={edge.consumer_op_idx})"
                        ),
                    )
                )
                continue

            if edge.stitch_kind in _REUSE_KINDS:
                continue

            prod_has = edge.slot_idx in prod_op.outcast_slots
            cons_has = edge.slot_idx in cons_op.incast_slots
            if prod_has and cons_has:
                continue

            issue = []
            if not prod_has:
                issue.append(
                    f"producer kernel funcKey={edge.producer_func_key}/"
                    f"opIdx={edge.producer_op_idx} does not declare this tensor "
                    f"as an output"
                )
            if not cons_has:
                issue.append(
                    f"consumer kernel funcKey={edge.consumer_func_key}/"
                    f"opIdx={edge.consumer_op_idx} does not declare this tensor "
                    f"as an input"
                )
            violations.append(
                Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=edge.slot_idx,
                    message="; ".join(issue),
                )
            )
        return violations
