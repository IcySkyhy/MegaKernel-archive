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
 * \file test_interp_calc_utils.h
 * \brief Shared helpers for interpreter calc unit tests.
 */

#pragma once

#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <vector>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {

class TorchAdaptorTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

template <typename T>
inline LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const T& val)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
inline LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const std::vector<T>& vals)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

#define ASSERT_ALLCLOSE(self, other) \
    ASSERT(calc::AllClose(self, other)) << "lhs:\n" << self->ToString() << "\nrhs:\n" << other->ToString() << "\n"

inline LogicalTensorDataPtr makePartialGolden(int n, int p, float v1, float v2)
{
    std::vector<float> ret(n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ret[i * n + j] = j < p ? v1 : v2;
        }
    }
    return makeTensorData(DT_FP32, {n, n}, ret);
}

inline int64_t alignup(int64_t x, int64_t align) { return (x + (align - 1)) & ~(align - 1); }

} // namespace npu::tile_fwk
