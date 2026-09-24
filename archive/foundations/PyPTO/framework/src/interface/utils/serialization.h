/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file serialization.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk {
const std::string T_VERSION = "2.0";
// NOTE: To avoid confuse, only data type are TYPE, other types should use the word kind/sort/categorty/...
enum class Kind {
    T_KIND_RAW_TENSOR = 0,
    T_KIND_TENSOR = 1,
    T_KIND_OPERATION = 2,
    T_KIND_FUNCTION = 3,
};
const std::string T_FIELD_KIND = "kind";
const std::string T_FIELD_RAWTENSOR = "rawtensor";
} // namespace npu::tile_fwk
