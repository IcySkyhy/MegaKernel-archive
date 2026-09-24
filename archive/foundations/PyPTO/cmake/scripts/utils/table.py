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
"""辅助表格."""

import copy
from typing import Any, List, Optional


class Table:
    """表格处理, 减少对其他外部库依赖"""

    @staticmethod
    def table(
        datas: List[List[Any]], headers: Optional[List[Any]] = None, col_width_max: int = 128, auto_sort: bool = True
    ) -> str:
        """获取格式化的表格字符串

        :param datas: 二维表格数据，每个子列表代表一行
        :param headers: 可选表头列表
        :param col_width_max: 单列最大列宽
        :param auto_sort: 自动升序排序
        :return: 格式化后的网格表格字符串
        """
        # 归一化为列数相等的表格, 并计算各列所需的最大字符串长度
        _heads = None if headers is None else copy.deepcopy(headers)
        _datas = copy.deepcopy(datas)
        if _heads is not None:
            _datas.append(_heads)
        col_num = max(len(row) for row in _datas)
        col_widths = [0] * col_num
        for i, row in enumerate(_datas):
            if len(row) < col_num:
                _datas[i] = row + [""] * (col_num - len(row))
                row = _datas[i]
            for j, col in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(col)))
        if _heads is not None:
            _datas = _datas[:-1]
            if auto_sort:
                _datas.sort(reverse=False)

        # 构造表头(如果有)
        separator = Table._make_separator(widths=col_widths, width_max=col_width_max, div='-')
        lines = []
        if headers is not None:
            lines.append(separator)
            lines.append(Table._make_details(widths=col_widths, datas=_heads, width_max=col_width_max))
            lines.append(Table._make_separator(widths=col_widths, width_max=col_width_max, div='='))
        else:
            lines.append(separator)

        # 添加数据行
        for row in _datas:
            lines.append(Table._make_details(widths=col_widths, datas=row, width_max=col_width_max, mode="left"))
            lines.append(separator)

        return "\n" + "\n".join(lines)

    @staticmethod
    def _make_separator(widths: List[int], width_max: int = 128, cross: str = '+', div: str = '-') -> str:
        """构建分隔线

        :param widths: 各列的列宽
        :param width_max: 单列最大列宽
        :param cross: 交叉点字符
        :param div: division 行中某元素点字符
        :return 分割线字符串
        """
        line = cross
        for width in widths:
            width = min(width, width_max)
            line += div * (width + 2) + cross
        return line

    @staticmethod
    def _make_details(widths: List[int], datas: List[Any], width_max: int = 128, mode: str = "center") -> str:
        """构建数据行详细信息

        :param widths: 各列的列宽
        :param width_max: 单列最大列宽
        :param datas: 单行各元素
        :param mode: 单元素对齐方式
        :return: 数据行字符串
        """
        line = "|"
        mode = mode.lower()
        for idx, ele in enumerate(datas):
            ele = str(ele)
            if mode in ["right", "r"]:
                ele = ele.rjust(widths[idx])
            elif mode in ["left", "l"]:
                ele = ele.ljust(widths[idx])
            else:
                ele = ele.center(widths[idx])
            ele = ele[:width_max]
            line += f" {ele} |"
        return line
