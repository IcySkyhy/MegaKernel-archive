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
 * \file test_for_utils.cpp
 * \brief Unit test for codegen utils.
 */

#include "gtest/gtest.h"

#include "codegen/utils/codegen_utils.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
class TestForUtils : public ::testing::Test {};

TEST_F(TestForUtils, TestGetTypeForB16B32InputInt64) { EXPECT_THROW(GetTypeForB16B32(DataType::DT_INT64), Error); }

TEST_F(TestForUtils, TestGetAddrTypeByOperandTypeBufUnknown)
{
    EXPECT_THROW(GetAddrTypeByOperandType(OperandType::BUF_UNKNOWN), Error);
}

TEST_F(TestForUtils, TestCalcLinearOffsetShapeEmpty) { EXPECT_EQ(CalcLinearOffset({}, {}), 0); }

} // namespace npu::tile_fwk
