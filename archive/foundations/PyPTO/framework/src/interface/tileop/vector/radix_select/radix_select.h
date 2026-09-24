/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file radix_select.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_b1.h"
#include "radix_select_b2.h"
#include "radix_select_b4.h"
#ifdef PTO_RS_GET_STRIDE
#undef PTO_RS_GET_STRIDE
#endif
#ifdef PTO_RS_GET_SHAPE
#undef PTO_RS_GET_SHAPE
#endif
#ifdef PTO_RS_PREPARE
#undef PTO_RS_PREPARE
#endif
#ifdef PTO_RS_SORT_ADDR_DEFINE
#undef PTO_RS_SORT_ADDR_DEFINE
#endif
#ifdef PTO_RS_SORT_TILE_DEFINE
#undef PTO_RS_SORT_TILE_DEFINE
#endif
#ifdef PTO_RS_COMMON_TILE_DEFINE
#undef PTO_RS_COMMON_TILE_DEFINE
#endif

namespace RadixSelectUtil {

template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalc(VAL value, IDX index, TMP tmp, SRC src)
{
    constexpr auto srcTypeSize = sizeof(typename SRC::Type);
    if constexpr (srcTypeSize == 1) {
        RadixSelectCalcB1<k, isLargest>(value, index, tmp, src);
    } else if constexpr (srcTypeSize == 2) {
        RadixSelectCalcB2<k, isLargest>(value, index, tmp, src);
    } else if constexpr (srcTypeSize == 4) {
        RadixSelectCalcB4<k, isLargest>(value, index, tmp, src);
    }
}

} // namespace RadixSelectUtil

#define OP_TILE_OP_RADIX_SELECT TRadixSelect
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void TRadixSelect(VAL value, IDX index, TMP tmp, SRC src)
{
    RadixSelectUtil::RadixSelectCalc<k, isLargest>(value, index, tmp, src);
}

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT__H
