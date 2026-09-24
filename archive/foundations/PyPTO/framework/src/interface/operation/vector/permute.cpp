/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.cpp
 * \brief Permute operation implementation
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tensor_transformation.h"
#include "permute.h"
#include <algorithm>
#include <sstream>

namespace npu::tile_fwk {

std::vector<int64_t> PermuteTileVector(const std::vector<int64_t>& values, const std::vector<int>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

[[maybe_unused]] std::vector<SymbolicScalar> PermuteTileVector(const std::vector<SymbolicScalar>& values,
                                                               const std::vector<int>& perm)
{
    std::vector<SymbolicScalar> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

void PermuteOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                  const std::vector<LogicalTensorPtr>& oOperand)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "Permute input operand count should be 1";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "Permute output operand count should be 1";
}

std::vector<int64_t> PermuteResultShape(const std::vector<int64_t>& inputShape, const std::vector<int>& perm)
{
    std::vector<int64_t> resultShape;
    resultShape.reserve(perm.size());
    for (int p : perm) {
        resultShape.push_back(inputShape[p]);
    }
    return resultShape;
}

bool IsIdentityPermutation(const std::vector<int>& perm)
{
    if (perm.size() <= 1) {
        return true;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
            return false;
        }
    }
    return true;
}

void NormalizePermutation(std::vector<int>& perm, int shapeSize)
{
    for (int& p : perm) {
        if (p < 0) {
            p += shapeSize;
        }
    }
}

void ValidatePermutation(const std::vector<int>& perm, int shapeSize)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();

    std::vector<bool> used(shapeSize, false);
    for (int p : perm) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, p >= 0 && p < shapeSize)
            << "Permute dim is invalid: " << p << ". Should be in range [0, " << shapeSize << ")";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, !used[p]) << "Permute dims contain duplicate values at index " << p;
        used[p] = true;
    }
}

static LogicalTensorPtr MakePermutedLogicalTensor(Function& function, const LogicalTensorPtr& self,
                                                  const std::vector<int>& perm)
{
    std::vector<int64_t> resultShape = PermuteResultShape(self->shape, perm);
    std::vector<SymbolicScalar> resultValidShape;
    if (!self->GetDynValidShape().empty()) {
        for (int p : perm) {
            resultValidShape.push_back(self->GetDynValidShape()[p]);
        }
    } else {
        resultValidShape = SymbolicScalar::FromConcrete(resultShape);
    }
    return std::make_shared<LogicalTensor>(function, self->tensor->datatype, resultShape, resultValidShape);
}

void TiledPermuteOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& result, const std::vector<int>& perm);

Tensor TensorPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm)
{
    auto result = MakePermutedLogicalTensor(function, self, perm);
    auto& op = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});
    op.SetAttribute(OpAttributeKey::perm, perm);
    return result;
}

void TiledPermuteOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& result, const std::vector<int>& perm)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto& op = function.AddOperation(Opcode::OP_PERMUTE, {srcTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::perm, perm);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck(iOperand, oOperand);

    std::vector<int> perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteOperation(function, tileShape, 0, input, oOperand[0], perm);
}

std::vector<int64_t> PermuteElementTileVector(const std::vector<int64_t>& values, const std::vector<int>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

Tensor TensorElementPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm)
{
    auto result = MakePermutedLogicalTensor(function, self, perm);
    auto& op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {self}, {result});
    op.SetAttribute(OpAttributeKey::perm, perm);
    return result;
}

void TiledPermuteElementOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                                  const LogicalTensorPtr& result, const std::vector<int>& perm)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteElementTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteElementTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto& op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {srcTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::perm, perm);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteElementOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteElementOperationTileFunc(Function& function, const TileShape& tileShape,
                                     const std::vector<LogicalTensorPtr>& iOperand,
                                     const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck(iOperand, oOperand);

    std::vector<int> perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteElementOperation(function, tileShape, 0, input, oOperand[0], perm);
}

Tensor Permute(Function& function, const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    CheckTensorShapeSize(self.GetStorage(), "PERMUTE");
    std::unordered_set<DataType> supportedTypes = {DT_FP8E4M3, DT_FP8E5M2, DT_HF8,    DT_FP8E8M0, DT_FP16,   DT_BF16,
                                                   DT_FP32,    DT_INT8,    DT_UINT8,  DT_INT16,   DT_UINT16, DT_INT32,
                                                   DT_UINT32,  DT_INT64,   DT_UINT64, DT_BOOL};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PERMUTE");

    DataType dtype = self.GetDataType();
    if (dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2 || dtype == DT_HF8 || dtype == DT_FP8E8M0) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
            << "PERMUTE: FP8 types (DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP8E8M0) are only supported on DAV_3510 "
               "architecture.";
    }

    if (dtype == DT_INT64 || dtype == DT_UINT64) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, self.Format() != TileOpFormat::TILEOP_NZ)
            << "PERMUTE: INT64/UINT64 do not support NZ format.";
    }
    CheckTensorDimRange(self.GetStorage(), 1, 5, "PERMUTE");

    const int shapeSize = static_cast<int>(self.GetShape().size());

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();

    if (shapeSize == 1) {
        return self;
    }

    NormalizePermutation(perm, shapeSize);
    ValidatePermutation(perm, shapeSize);

    if (IsIdentityPermutation(perm)) {
        return self;
    }

    bool lastAxisInvolved = (perm[shapeSize - 1] != shapeSize - 1);
    if (lastAxisInvolved) {
        RETURN_CALL(ElementPermuteOperation, function, self.GetStorage(), perm);
    }

    RETURN_CALL(PermuteOperation, function, self.GetStorage(), perm);
}

Tensor Permute(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Permute");

    auto& function = *Program::GetInstance().GetCurrentFunction();
    return Permute(function, self, perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, PermuteElementOperationTileFunc);

} // namespace npu::tile_fwk
