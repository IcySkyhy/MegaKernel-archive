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
 * \file test_codegen_add.h
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

class TestCodeGenAdd {
public:
    void test_add_001();
    void test_add_002();
    void test_add_003();
    void test_add_004();
    void test_add_005();
    void test_add_006();
    void test_add_007();
    void test_add_008();
    void test_add_009();
    void test_add_010();
    void test_add_011();
    void test_add_012();
    void test_add_013();
    void test_add_014();
    void test_add_015();
    void test_add_016();
    void test_add_017();
    void test_add_018();
    void test_add_019();
    void test_add_020();
    void test_add_021();
    void test_add_022();
    void test_add_023();
    void test_add_024();
    void test_add_025();
    void test_add_026();
    void test_add_027();
    void test_add_028();
    void test_add_029();
    void test_add_030();
    void test_add_031();
    void test_add_032();
    void test_add_033();
    void test_add_034();
    void test_add_035();
    void test_add_036();
    void test_add_037();
    void test_add_038();
    void test_add_039();
    void test_add_040();
    void test_add_041();
    void test_add_042();
    void test_add_043();
    void test_add_044();
    void test_add_045();
    void test_add_046();
    void test_add_047();
    void test_add_048();
    static TestCodeGenAdd& Instance();

private:
    TestCodeGenAdd();
    ~TestCodeGenAdd();
};
