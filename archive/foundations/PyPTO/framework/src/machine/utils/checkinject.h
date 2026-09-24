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
 * \file checkinject.h
 * \brief
 */
#ifndef MACHINE_UTILS_CHECKINJECT_H
#define MACHINE_UTILS_CHECKINJECT_H

#include <cstddef>

namespace npu::tile_fwk {
inline int Checkinject(const char cmdStr[], size_t strLen)
{
    if (cmdStr == nullptr || strLen == 0) {
        return -1;
    }
    const char cmdIllegalChar[] = {';', '|', '<', '>', '`'};
    for (size_t i = 0; i < strLen; i++) {
        for (const auto& c : cmdIllegalChar) {
            if (cmdStr[i] == c) {
                return -1;
            }
        }
    }
    return 0;
}
} // namespace npu::tile_fwk

#endif
