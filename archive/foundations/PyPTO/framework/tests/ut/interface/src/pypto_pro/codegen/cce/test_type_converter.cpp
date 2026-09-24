/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "codegen/cce/type_converter.h"

namespace pypto {
namespace codegen {

TEST(TypeConverterTest, ConvertsAllCastRoundModesThroughEnumReflection)
{
    TypeConverter converter;
    const std::vector<std::string> expected = {
        "RoundMode::CAST_NONE", "RoundMode::CAST_RINT",  "RoundMode::CAST_ROUND", "RoundMode::CAST_FLOOR",
        "RoundMode::CAST_CEIL", "RoundMode::CAST_TRUNC", "RoundMode::CAST_ODD",
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(converter.ConvertCastRoundMode(static_cast<int>(i)), expected[i]);
    }
    EXPECT_EQ(converter.ConvertCastRoundMode(99), "UNKNOWN");
}

} // namespace codegen
} // namespace pypto
