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
 * \file pad_local_buffer.h
 * \brief
 */

#ifndef PAD_LOCAL_BUFFER_H
#define PAD_LOCAL_BUFFER_H
#include <unordered_map>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "axis_combine_marker.h"

namespace npu::tile_fwk {
/*
Vector op默认对尾轴做32B对齐
*/
class PadLocalBuffer : public Pass {
public:
    explicit PadLocalBuffer(std::string name = "PadLocalBuffer") : Pass(name) {}
    ~PadLocalBuffer() override = default;

private:
    Status RunOnFunction(Function& function) override;
    void PadMatmulL1ConvertScene(Operation& op, LogicalTensorPtr& in, size_t lowIndex);
    void PadForMatMulMX(LogicalTensorPtr& in, const int64_t& axisNum);
    void PadMatmul(Operation& op, LogicalTensorPtr& in);
    bool TryPadMatmulMXScene(Operation& op, LogicalTensorPtr& in);
    void TryPadMatmulIsMXScene(Operation& op, LogicalTensorPtr& in);
    int64_t GetMatmulPaddingValue(Operation& op, LogicalTensorPtr& in) const;
    void PadMatmulHighLow(LogicalTensorPtr& in, size_t highIndex, size_t lowIndex, int64_t padValue);
    bool IsGemvL1ATensor(const LogicalTensorPtr& in) const;
    bool IsGemvL0ATensor(const LogicalTensorPtr& in) const;
    void PadMatmulGemvKAlign(LogicalTensorPtr& in, size_t highIndex, size_t lowIndex);
    void PadVector(Operation& op, LogicalTensorPtr& in, std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw);
    void PadVector256(Operation& op, LogicalTensorPtr& in, bool needRowPad);
    void ProcessBroadcast(Operation& op, int64_t blockPadding);
    void PrepareBroadcast(Function& function);
    void PadVectorForAxisCombine(Operation& op, LogicalTensorPtr& in,
                                 std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw);
    bool IsUb2L1CopyOp(const Operation& op);
    bool HandleUb2L1CopyOp(Operation& op, LogicalTensorPtr& in);
    bool ShouldSkipVectorPad(Operation& op, LogicalTensorPtr& in);
    bool IsElementwiseLikeOp(OpCalcType calcType, Operation* producerOp) const;
    void DoBrcbOpPadding(Operation& op, LogicalTensorPtr& in, size_t lastIdx, int64_t paddingValue,
                         std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw);
    void DoElementwiseLikePadding(const Operation& op, LogicalTensorPtr& in, size_t lastIdx, int64_t paddingValue);
    bool IsMatmul(const LogicalTensorPtr& tensor) const;
    bool IsVector(const LogicalTensorPtr& tensor) const;
    void PadSingleTensor(Operation& op, LogicalTensorPtr& tensor,
                         std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw, bool needRowPad = false);
    void DoPadding(Function& function);
    void ProcessReduceForAxisCombine(Operation& op, LogicalTensorPtr& in, int64_t paddingValue);
    int64_t AlignedRawTensorIfNeed(LogicalTensorPtr& in, int64_t pos, const int64_t base);
    bool IsInputDataType(const Operation& op, const LogicalTensorPtr& in,
                         const std::unordered_set<DataType>& targetTypes) const;
    // 设置原始 rawshape（替代 RawTensor::oriRawshape = rawshape）
    Shape& SetOriRawshape(LogicalTensorPtr& in);
    // 获取已保存的原始 rawshape
    Shape& GetOriRawshape(LogicalTensorPtr& in);
    std::unordered_map<int64_t, int64_t> broadcastLastAxis_;
    bool combineAxis_ = false;
    AxisCombineMarker axisCombineMarker_;
    std::unordered_map<int, Shape> oriRawshapeMap_; // 存储原始 rawshape，替代 RawTensor::oriRawshape
};
} // namespace npu::tile_fwk
#endif // PAD_LOCAL_BUFFER_H
