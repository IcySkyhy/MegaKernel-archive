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
 * \file test_codegen_full.h
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

class TestCodeGenFull {
public:
    void test_full_001();
    void test_full_002();
    void test_full_003();
    void test_full_004();
    void test_full_005();
    void test_full_006();
    void test_full_007();
    void test_full_008();
    void test_full_009();
    void test_full_010();
    void test_full_011();
    void test_full_012();
    void test_full_013();
    void test_full_014();
    void test_full_015();
    void test_full_016();
    void test_full_017();
    void test_full_018();
    void test_full_019();
    static TestCodeGenFull& Instance();

private:
    TestCodeGenFull();
    ~TestCodeGenFull();
};
