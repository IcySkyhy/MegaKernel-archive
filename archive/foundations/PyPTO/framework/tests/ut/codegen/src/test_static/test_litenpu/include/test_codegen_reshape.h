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
 * \file test_codegen_reshape.h
 * \brief
 */

#pragma once

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "codegen/codegen.h"
#include "codegen/npu/litenpu/codegen_litenpu.h"
#include "test_codegen_common.h"

class TestCodeGenReshape {
public:
    void test_reshape_fp16_001();
    void test_reshape_fp16_002();
    void test_reshape_fp16_003();
    void test_reshape_fp16_004();
    void test_reshape_fp16_005();
    void test_reshape_fp16_006();
    void test_reshape_fp16_007();
    void test_reshape_fp16_008();
    void test_reshape_fp16_009();
    void test_reshape_fp16_010();
    void test_reshape_fp32_001();
    void test_reshape_fp32_002();
    void test_reshape_fp32_003();
    void test_reshape_fp32_004();
    void test_reshape_fp32_005();
    void test_reshape_fp32_006();
    void test_reshape_fp32_007();
    void test_reshape_fp32_008();
    void test_reshape_fp32_009();
    void test_reshape_fp32_010();
    void test_reshape_inplace_fp32_001();
    void test_reshape_inplace_fp32_002();
    void test_reshape_inplace_fp32_003();
    void test_reshape_inplace_fp32_004();
    void test_reshape_inplace_fp32_005();
    void test_reshape_int8_001();
    void test_reshape_int8_002();
    void test_reshape_int8_003();
    void test_reshape_int8_004();
    void test_reshape_int8_005();
    void test_reshape_int16_001();
    void test_reshape_int16_002();
    void test_reshape_int16_003();
    void test_reshape_int16_004();
    void test_reshape_int16_005();
    void test_reshape_int32_001();
    void test_reshape_int32_002();
    void test_reshape_int32_003();
    void test_reshape_int32_004();
    void test_reshape_int32_005();
    static TestCodeGenReshape& Instance();

private:
    TestCodeGenReshape();
    ~TestCodeGenReshape();
};
