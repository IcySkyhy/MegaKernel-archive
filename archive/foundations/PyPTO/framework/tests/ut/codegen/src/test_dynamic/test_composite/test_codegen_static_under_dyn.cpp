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
 * \file test_codegen_static_under_dyn.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "test_codegen_common.h"
#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenStaticUnderDyn : public CodegenTestBase {
public:
    TestCodegenStaticUnderDyn() : CodegenTestBase() {}
};

void TestStaticLoop(const Tensor& t0, const Tensor& t1, const Tensor& t2, Tensor& out, int s)
{
    constexpr int LOOP_ITERATIONS = 8;
    FUNCTION("main", {t0, t1, t2}, {out})
    {
        Tensor s0Out;
        config::SetBuildStatic(true);
        FUNCTION("S0") { s0Out = Sub(t1, t0); }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_ITERATIONS))
        {
            Tensor t0s = View(s0Out, {s, s}, {i * s, 0});
            Tensor t3 = Add(t0s, t2);
            Assemble(t3, {i * s, 0}, out);
        }
    }
}

TEST_F(TestCodegenStaticUnderDyn, TestStaticFuncUnderDyn)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [32, 32]
    Tensor t2(DT_FP32, {s, s}, "t2");     // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestStaticLoop(t0, t1, t2, out, s);

    for (auto& ele : Program::GetInstance().GetFunctionMap()) {
        bool isRootExist = ele.second.get()->rootFunc_ != nullptr;
        if (isRootExist) {
            npu::tile_fwk::CodeGenCtx ctx;
            npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
            codeGen.GenCode(*ele.second.get());
        }
    }
}

} // namespace npu::tile_fwk
