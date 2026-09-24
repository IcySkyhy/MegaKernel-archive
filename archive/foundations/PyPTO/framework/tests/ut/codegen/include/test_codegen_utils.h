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
 * \file test_codegen_utils.h
 * \brief Unit test for codegen.
 */

#ifndef TEST_CODEGEN_UTILS_H
#define TEST_CODEGEN_UTILS_H

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
struct LogicalTensorInfo {
    LogicalTensorInfo(Function& func, DataType dataType, MemoryType memoryType, const std::vector<int64_t>& tShape)
        : function(func), dType(dataType), memType(memoryType), shape(tShape) {};
    LogicalTensorInfo(Function& func, DataType dataType, MemoryType memoryType, const std::vector<int64_t>& tShape,
                      std::string tName)
        : function(func), dType(dataType), memType(memoryType), shape(tShape), tensorName(std::move(tName)) {};
    LogicalTensorInfo(Function& func, DataType dataType, MemoryType memoryType, const std::vector<int64_t>& tShape,
                      const std::vector<SymbolicScalar>& dynShape)
        : function(func), dType(dataType), memType(memoryType), shape(tShape), dynValidShape(dynShape) {};
    LogicalTensorInfo(Function& func, DataType dataType, MemoryType memoryType, const std::vector<int64_t>& tShape,
                      int magicVal, const std::vector<SymbolicScalar>& dynShape)
        : function(func),
          dType(dataType),
          memType(memoryType),
          shape(tShape),
          magic(magicVal),
          dynValidShape(dynShape) {};
    LogicalTensorInfo(Function& func, DataType dataType, MemoryType memoryType, const std::vector<int64_t>& tShape,
                      std::string tName, int magicVal, const std::vector<SymbolicScalar>& dynShape)
        : function(func),
          dType(dataType),
          memType(memoryType),
          shape(tShape),
          tensorName(std::move(tName)),
          magic(magicVal),
          dynValidShape(dynShape) {};

    Function& function;
    DataType dType;
    MemoryType memType;
    const std::vector<int64_t>& shape;
    const std::string tensorName;
    int magic = -1;
    std::vector<SymbolicScalar> dynValidShape;
};

std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo& info);

std::string GetResultFromCpp(const Function& function);

std::string GenCodeByFunction(Function& function);

void CheckStringExist(const std::string& expect, const std::string& result);

Function* GenMockFuncDyn(const std::string& funcName, const std::vector<int64_t>& shape = {64, 64});
Function* GenMockFuncStatic(const std::string& funcName, const std::vector<int64_t>& shape = {64, 64});

struct MockFuncDynUnaryConf {
    std::vector<int64_t> shape = {64, 64};
    std::vector<int64_t> tileShape = {};
    DataType dtype = DT_FP32;
};

template <typename OpFunc>
Function* GenMockFuncDynUnary(const std::string& funcName, const MockFuncDynUnaryConf& config, OpFunc opFunc)
{
    auto tileShape = config.tileShape.empty() ? config.shape : config.tileShape;
    TileShape::Current().SetVecTile(tileShape);
    Tensor input(config.dtype, config.shape, "input");
    Tensor output(config.dtype, config.shape, "output");

    FUNCTION(funcName, {input}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            opFunc(input, output);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    return function;
}

struct MockFuncDynBinaryConf {
    std::vector<int64_t> shapeA = {64, 64};
    std::vector<int64_t> shapeB = {64, 64};
    std::vector<int64_t> outputShape = {64, 64};
    std::vector<int64_t> tileShape = {};
    DataType dtype = DT_FP32;
};

template <typename OpFunc>
Function* GenMockFuncDynBinary(const std::string& funcName, const MockFuncDynBinaryConf& config, OpFunc opFunc)
{
    auto tileShape = config.tileShape.empty() ? config.outputShape : config.tileShape;
    TileShape::Current().SetVecTile(tileShape);
    Tensor inputA(config.dtype, config.shapeA, "inputA");
    Tensor inputB(config.dtype, config.shapeB, "inputB");
    Tensor output(config.dtype, config.outputShape, "output");

    FUNCTION(funcName, {inputA, inputB}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            opFunc(inputA, inputB, output);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    return function;
}

std::shared_ptr<LogicalTensor> CreateConvTensor(Function& function, const DataType& dtype,
                                                const std::vector<int64_t>& shape, const MemoryType& memType,
                                                const bool& isCopyIn = true);

struct GenOpCodeOptions {
    bool isMainBlk = false;
};

std::string GenOpCodeFromOp(Function& function, const Operation& op, const GenOpCodeOptions& options = {});

CodeGenOpCloudNPU GenOpCloudNPUFromOp(Function& function, const Operation& op,
                                      std::shared_ptr<SymbolManager>& outSymbolManager,
                                      const GenOpCodeOptions& options = {});

} // namespace npu::tile_fwk

#endif // TEST_CODEGEN_UTILS_H
