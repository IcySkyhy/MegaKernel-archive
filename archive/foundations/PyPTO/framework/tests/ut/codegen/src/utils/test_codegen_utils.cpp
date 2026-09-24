/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_codegen_utils.h"

#include <iostream>
#include <fstream>
#include "gtest/gtest.h"

#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo& info)
{
    if (info.memType == MemoryType::MEM_DEVICE_DDR) {
        std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(
            info.dType, info.shape, TileOpFormat::TILEOP_ND, info.tensorName, info.magic);
        std::vector<int64_t> offset = std::vector<int64_t>(info.shape.size(), 0);
        IRBuilder builder;
        auto ddrTensor = builder.CreateTensorVar(info.function, ddrRawTensor, offset, info.shape, info.dynValidShape);
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
        ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
        return ddrTensor;
    }

    IRBuilder builder;
    auto localTensor = builder.CreateTensorVar(info.function, info.dType, info.shape, info.dynValidShape,
                                               TileOpFormat::TILEOP_ND, info.tensorName);
    localTensor->SetMemoryTypeOriginal(info.memType);
    localTensor->SetMemoryTypeToBe(info.memType);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memoryrange.memId = 0;
    localTensor->memoryrange.start = 0;
    localTensor->memoryrange.end = 0;
    if (info.magic != -1) {
        localTensor->SetMagic(info.magic);
    }
    if (!info.dynValidShape.empty()) {
        localTensor->UpdateDynValidShape(info.dynValidShape);
    }
    return localTensor;
}

std::string GetResultFromCpp(const Function& function)
{
    std::string binPath;
    for (const auto& subFunc : function.rootFunc_->programs_) {
        auto leafFuncAttr = subFunc.second->GetLeafFuncAttribute();
        ASSERT(FwkErr::INVALID_FUNCTION, leafFuncAttr != nullptr) << "can not find leaf func attribute";
        binPath = leafFuncAttr->binPath;
        if (binPath.empty()) {
            continue;
        }
        std::string cppFile = binPath.substr(0, binPath.rfind('.')) + ".cpp";
        std::ifstream ifs(cppFile);
        std::string res((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
        ifs.close();
        return res;
    }

    CODEGEN_LOGW("can not find binPath in testcase: %s", function.GetRawName().c_str());
    return "";
}

void CheckStringExist(const std::string& target, const std::string& content)
{
    bool res = content.find(target) != std::string::npos;
    EXPECT_TRUE(res) << "target: \n" << target << "\n\n ---- not found in content ---- \n\n" << content << std::endl;
}

std::string GenCodeByFunction(Function& function)
{
    CodeGenCtx ctx;
    CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(function);
    return GetResultFromCpp(function);
}

Function* GenMockFuncDyn(const std::string& funcName, const std::vector<int64_t>& shape)
{
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(funcName, {inputA, inputB}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);
    return function;
}

Function* GenMockFuncStatic(const std::string& funcName, const std::vector<int64_t>& shape)
{
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(funcName, {inputA, inputB, output}) { output = Add(inputA, inputB); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    return function;
}

std::shared_ptr<LogicalTensor> CreateConvTensor(Function& function, const DataType& dtype,
                                                const std::vector<int64_t>& shape, const MemoryType& memType,
                                                const bool& isCopyIn)
{
    IRBuilder builder;
    std::shared_ptr<LogicalTensor> tensorPtr = nullptr;
    if (isCopyIn) {
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            tensorPtr = builder.CreateTensorVar(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                                                TileOpFormat::TILEOP_ND, "GmTensor");
        } else {
            tensorPtr = builder.CreateTensorVar(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                                                TileOpFormat::TILEOP_NZ, "L1Tensor");
            tensorPtr->SetAttr(OpAttributeKey::needAlloc, true);
            tensorPtr->memoryrange.memId = 0;
            tensorPtr->memoryrange.start = 0;
            tensorPtr->memoryrange.end = 0;
        }
    } else {
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            tensorPtr = builder.CreateTensorVar(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                                                TileOpFormat::TILEOP_ND, "GmTensor");
        } else {
            tensorPtr = builder.CreateTensorVar(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                                                TileOpFormat::TILEOP_NZ, "L0CTensor");
            tensorPtr->SetAttr(OpAttributeKey::needAlloc, true);
            tensorPtr->memoryrange.memId = 0;
            tensorPtr->memoryrange.start = 0;
            tensorPtr->memoryrange.end = 0;
        }
    }
    tensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(shape));
    tensorPtr->SetMemoryTypeOriginal(memType);
    tensorPtr->SetMemoryTypeToBe(memType);
    return tensorPtr;
}

std::string GenOpCodeFromOp(Function& function, const Operation& op, const GenOpCodeOptions& options)
{
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpNPUCtx opCtx(symbolManager, function, *function.rootFunc_->programs_[0], op, options.isMainBlk);
    CodeGenOpCloudNPU cop(opCtx);
    return cop.GenOpCode();
}

CodeGenOpCloudNPU GenOpCloudNPUFromOp(Function& function, const Operation& op,
                                      std::shared_ptr<SymbolManager>& outSymbolManager, const GenOpCodeOptions& options)
{
    outSymbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, outSymbolManager);
    CodeGenOpNPUCtx opCtx(outSymbolManager, function, *function.rootFunc_->programs_[0], op, options.isMainBlk);
    return CodeGenOpCloudNPU(opCtx);
}

} // namespace npu::tile_fwk
