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
 * \file computational_graph_builder.cpp
 * \brief
 */

#include "computational_graph_builder.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {

bool ComputationalGraphBuilder::AddTensor(DataType dataType, const std::vector<int64_t>& tileShape,
                                          const std::string& name)
{
    if (tensors_.count(name) > 0) {
        return false;
    }
    auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, tileShape, CreateTestConstIntVector(tileShape),
                                                             TileOpFormat::TILEOP_ND, name);
    if (tensor == nullptr) {
        return false;
    }
    function->GetTensorMap().Insert(tensor, false);
    tensors_[name] = tensor;
    return true;
}

bool ComputationalGraphBuilder::AddTensor(DataType dataType, const std::vector<int64_t>& tileShape, MemoryType memType,
                                          const std::string& name, int subGraphID)
{
    (void)subGraphID;
    if (!AddTensor(dataType, tileShape, name)) {
        return false;
    }
    auto tensor = GetTensor(name);
    tensor->SetMemoryTypeBoth(memType, true);
    tensor->memoryrange.memId = tensor->GetMagic();
    tensors_[name] = tensor;
    return true;
}

bool ComputationalGraphBuilder::AddTensors(DataType dataType, const std::vector<int64_t>& tileShape,
                                           const std::vector<std::string>& names)
{
    for (auto& name : names) {
        if (!AddTensor(dataType, tileShape, name)) {
            return false;
        }
    }
    return true;
}

bool ComputationalGraphBuilder::AddTensors(DataType dataType, const std::vector<int64_t>& tileShape,
                                           const std::vector<MemoryType>& memTypes,
                                           const std::vector<std::string>& names, int subGraphID)
{
    if (memTypes.size() != names.size()) {
        return false;
    }
    for (size_t i = 0; i < memTypes.size(); i++) {
        if (!AddTensor(dataType, tileShape, memTypes[i], names[i], subGraphID)) {
            return false;
        }
    }
    return true;
}

bool ComputationalGraphBuilder::AddOp(Opcode opcode, const std::vector<std::string>& ioperands,
                                      const std::vector<std::string>& ooperands, const std::string& name,
                                      bool updateFunctionMap)
{
    if (operations_.count(name) > 0) {
        return false;
    }
    std::vector<std::shared_ptr<LogicalTensor>> itensors;
    for (auto iop : ioperands) {
        if (tensors_.count(iop) == 0) {
            return false;
        }
        itensors.push_back(tensors_[iop]);
    }
    std::vector<std::shared_ptr<LogicalTensor>> otensors;
    for (auto oop : ooperands) {
        if (tensors_.count(oop) == 0) {
            return false;
        }
        otensors.push_back(tensors_[oop]);
    }
    (void)updateFunctionMap;
    IRBuilder builder;
    Operation& op = builder.CreateTensorOpStmt(*function, opcode, itensors, otensors);
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        auto shapeImme = OpImmediate::Specified(itensors[0]->GetShape());
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                            otensors[0]->GetMemoryTypeOriginal(), shapeImme, shapeImme,
                                                            std::vector<OpImmediate>()));
    } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        auto shapeImme = OpImmediate::Specified(itensors[0]->GetShape());
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(itensors[0]->GetMemoryTypeOriginal(),
                                                            OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    }
    operations_[name] = &op;
    operationMagics_[name] = op.GetOpMagic();
    return true;
}

bool ComputationalGraphBuilder::AddOps(const std::vector<Opcode>& opcodes,
                                       const std::vector<std::vector<std::string>>& ioperandss,
                                       const std::vector<std::vector<std::string>>& ooperandss,
                                       const std::vector<std::string>& names, bool updateFunctionMap)
{
    if (opcodes.size() != ioperandss.size() || opcodes.size() != ooperandss.size() || opcodes.size() != names.size()) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(opcodes.size()); i++) {
        if (!AddOp(opcodes[i], ioperandss[i], ooperandss[i], names[i], updateFunctionMap)) {
            return false;
        }
    }
    return true;
}

bool ComputationalGraphBuilder::SetInCast(std::vector<std::string> ioperands)
{
    std::vector<std::shared_ptr<LogicalTensor>> itensors;
    for (auto iop : ioperands) {
        if (tensors_.count(iop) == 0) {
            return false;
        }
        itensors.push_back(tensors_[iop]);
    }
    function->inCasts_ = itensors;
    return true;
}

bool ComputationalGraphBuilder::SetOutCast(std::vector<std::string> ooperands)
{
    std::vector<std::shared_ptr<LogicalTensor>> otensors;
    for (auto oop : ooperands) {
        if (tensors_.count(oop) == 0) {
            return false;
        }
        otensors.push_back(tensors_[oop]);
    }
    function->outCasts_ = otensors;
    return true;
}

Function* ComputationalGraphBuilder::GetFunction() { return function; }

Operation* ComputationalGraphBuilder::GetOp(const std::string& name)
{
    if (operations_.count(name) == 0) {
        return nullptr;
    }
    return function->GetOpByOpMagic(operationMagics_[name]);
}

std::shared_ptr<LogicalTensor> ComputationalGraphBuilder::GetTensor(const std::string& name)
{
    if (tensors_.count(name) == 0) {
        return nullptr;
    }
    return tensors_[name];
}

} // namespace tile_fwk
} // namespace npu
