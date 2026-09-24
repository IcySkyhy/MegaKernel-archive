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
 * \file hash_order_utils.h
 * \brief Shared utility for parsing func{magic}_{order} hash order format.
 */

#ifndef HASH_ORDER_UTILS_H_
#define HASH_ORDER_UTILS_H_

#include <stdexcept>
#include <string>

namespace npu::tile_fwk {

inline const char* FUNC_HASH_ORDER_DEFAULT_KEY = "DEFAULT";

inline bool ParseFuncHashOrder(const std::string& key, int& funcMagic, int& localOrder)
{
    constexpr size_t kPrefixShift = 4;
    if (key.substr(0, kPrefixShift) != "func") {
        return false;
    }
    size_t pos = key.find('_');
    if (pos == std::string::npos) {
        return false;
    }
    try {
        funcMagic = std::stoi(key.substr(kPrefixShift, pos - kPrefixShift));
        localOrder = std::stoi(key.substr(pos + 1));
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

} // namespace npu::tile_fwk

#endif // HASH_ORDER_UTILS_H_
