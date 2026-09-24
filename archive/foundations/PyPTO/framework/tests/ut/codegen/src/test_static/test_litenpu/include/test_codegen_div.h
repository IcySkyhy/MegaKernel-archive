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
 * \file test_codegen_div.h
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

class TestCodeGenDiv {
public:
    void test_div_001();
    void test_div_002();
    void test_div_003();
    void test_div_004();
    void test_div_005();
    void test_div_006();
    void test_div_007();
    void test_div_008();
    void test_div_009();
    void test_div_010();
    void test_div_011();
    void test_div_012();
    void test_div_013();
    void test_div_014();
    void test_div_015();
    void test_div_016();
    void test_div_017();
    void test_div_018();
    void test_div_019();
    void test_div_020();
    void test_div_021();
    void test_div_022();
    void test_div_023();
    void test_div_024();
    void test_div_025();
    void test_div_026();
    void test_div_027();
    void test_div_028();
    void test_div_029();
    void test_div_030();
    void test_div_031();
    void test_div_032();
    void test_div_033();
    void test_div_034();
    void test_div_035();
    void test_div_036();
    void test_div_037();
    void test_div_038();
    void test_div_039();
    void test_div_040();
    void test_div_041();
    void test_div_042();
    void test_div_043();
    void test_div_044();
    void test_div_045();
    void test_div_046();
    void test_div_047();
    void test_div_048();
    static TestCodeGenDiv& Instance();

private:
    TestCodeGenDiv();
    ~TestCodeGenDiv();
};
