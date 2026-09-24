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
 * \file test_codegen_sub.h
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

class TestCodeGenSub {
public:
    void test_sub_001();
    void test_sub_002();
    void test_sub_003();
    void test_sub_004();
    void test_sub_005();
    void test_sub_006();
    void test_sub_007();
    void test_sub_008();
    void test_sub_009();
    void test_sub_010();
    void test_sub_011();
    void test_sub_012();
    void test_sub_013();
    void test_sub_014();
    void test_sub_015();
    void test_sub_016();
    void test_sub_017();
    void test_sub_018();
    void test_sub_019();
    void test_sub_020();
    void test_sub_021();
    void test_sub_022();
    void test_sub_023();
    void test_sub_024();
    void test_sub_025();
    void test_sub_026();
    void test_sub_027();
    void test_sub_028();
    void test_sub_029();
    void test_sub_030();
    void test_sub_031();
    void test_sub_032();
    void test_sub_033();
    void test_sub_034();
    void test_sub_035();
    void test_sub_036();
    void test_sub_037();
    void test_sub_038();
    void test_sub_039();
    void test_sub_040();
    void test_sub_041();
    void test_sub_042();
    void test_sub_043();
    void test_sub_044();
    void test_sub_045();
    void test_sub_046();
    void test_sub_047();
    void test_sub_048();
    static TestCodeGenSub& Instance();

private:
    TestCodeGenSub();
    ~TestCodeGenSub();
};
