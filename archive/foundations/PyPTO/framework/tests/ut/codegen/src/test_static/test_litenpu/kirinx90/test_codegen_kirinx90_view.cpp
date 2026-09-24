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
 * \file test_codegen_kirinx90_view.cpp
 * \brief
 */

#include "../include/test_codegen_view.h"

using namespace npu::tile_fwk;

class TestCodeGenKirinX90View : public CodegenTestLiteNPU {
public:
    TestCodeGenKirinX90View() : CodegenTestLiteNPU("KirinX90") {};
};

TEST_F(TestCodeGenKirinX90View, test_view_fp16_001) { TestCodeGenView::Instance().test_view_fp16_001(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_002) { TestCodeGenView::Instance().test_view_fp16_002(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_003) { TestCodeGenView::Instance().test_view_fp16_003(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_004) { TestCodeGenView::Instance().test_view_fp16_004(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_005) { TestCodeGenView::Instance().test_view_fp16_005(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_006) { TestCodeGenView::Instance().test_view_fp16_006(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_007) { TestCodeGenView::Instance().test_view_fp16_007(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_008) { TestCodeGenView::Instance().test_view_fp16_008(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_009) { TestCodeGenView::Instance().test_view_fp16_009(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp16_010) { TestCodeGenView::Instance().test_view_fp16_010(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_001) { TestCodeGenView::Instance().test_view_fp32_001(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_002) { TestCodeGenView::Instance().test_view_fp32_002(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_003) { TestCodeGenView::Instance().test_view_fp32_003(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_004) { TestCodeGenView::Instance().test_view_fp32_004(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_005) { TestCodeGenView::Instance().test_view_fp32_005(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_006) { TestCodeGenView::Instance().test_view_fp32_006(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_007) { TestCodeGenView::Instance().test_view_fp32_007(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_008) { TestCodeGenView::Instance().test_view_fp32_008(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_009) { TestCodeGenView::Instance().test_view_fp32_009(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_fp32_010) { TestCodeGenView::Instance().test_view_fp32_010(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int8_001) { TestCodeGenView::Instance().test_view_int8_001(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int8_002) { TestCodeGenView::Instance().test_view_int8_002(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int8_003) { TestCodeGenView::Instance().test_view_int8_003(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int16_001) { TestCodeGenView::Instance().test_view_int16_001(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int16_002) { TestCodeGenView::Instance().test_view_int16_002(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int16_003) { TestCodeGenView::Instance().test_view_int16_003(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int32_001) { TestCodeGenView::Instance().test_view_int32_001(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int32_002) { TestCodeGenView::Instance().test_view_int32_002(); }

TEST_F(TestCodeGenKirinX90View, DISABLED_test_view_int32_003) { TestCodeGenView::Instance().test_view_int32_003(); }
