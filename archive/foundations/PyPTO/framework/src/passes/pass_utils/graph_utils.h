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
 * \file graph_utils.h
 * \brief
 */

#pragma once
#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H
#include <vector>
#include <queue>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "pass_common_defs.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
struct CompareTensorByMagic {
    bool operator()(const LogicalTensorPtr& a, const LogicalTensorPtr& b) const
    {
        if (a == b) {
            return false;
        }
        if (!a) {
            return b != nullptr;
        }
        if (!b) {
            return false;
        }

        return a->GetMagic() > b->GetMagic();
    }
};

using TensorSet = std::set<LogicalTensorPtr, CompareTensorByMagic>;

class GraphUtils {
public:
    /**
     * @brief Add an operation and set the DynValidShape of the output.
     *
     * @param function the target function for the operation to be added.
     * @param opCode type of the operation to be added (Besides Assemble, View, Convert, CopyIn, CopyOut, Reshape)
     * @param iOperands LogicalTensors, indicating the input of the op to be added
     * @param oOperands LogicalTensors, indicating the output of the op to be added
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     * @return the operation to be added
     */
    static Operation& AddDynOperation(Function& function, const Opcode opCode, LogicalTensors iOperands,
                                      const LogicalTensors& oOperands,
                                      const std::vector<std::vector<SymbolicScalar>>& outDynShape = {});
    /**
     * @brief Add an assemble operation.
     *        Update the AssembleOpAttribute of the assemble operation. The fromDynValidShape value is set by the
     * DynValidShape of input. Inherit the operation attribute and scopeId when given an origin assemble op. Set the
     * DynValidShape of the output.
     *
     * @param function the target function for the assemble operation.
     * @param assemble AssembleOp, indicating the basic information of the added assemble operation.
     *                 The information includes the memoryType of assemble OpAttribute, assemble offset, input and
     * output of assemble. The information also indicates the origin assemble op (if exist) that the added operation
     *                 should inherit attribute and scope id from.
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     *                    The AssembleOpAttribute does not require dynamic attributes for output, so the SetDynShape is
     * executed at last.
     * @return the operation to be added
     */
    static Operation& AddAssembleOperation(Function& function, const AssembleOp& assemble,
                                           const std::vector<std::vector<SymbolicScalar>>& outDynShape = {});
    /**
     * @brief Add a reshape operation.
     *        Set the DynValidShape of the output.
     *        Inherit the operation attribute and scope id when given a legal origin reshape operation pointer.
     *        Update the op_attr_validShape of the reshape operation by the DynValidShape of output.
     *
     * @param function the target function for the reshape operation.
     * @param iOperand LogicalTensorPtr, indicating the input of the op
     * @param oOperand LogicalTensorPtr, indicating the output of the op
     * @param originOp Pointer of operation, indicating an origin operation the added reshape operation should inherit
     * attribute and scopeId from. Skip inherit attribute and scopeId if the pointer is nullptr.
     * @param outDynShape the DynValidShape of the output and the value of op_attr_validShape. The default value is {}.
     *                    If outDynShape is empty, uses CallInferShapeFunc to calculate the DynValidShape.
     * @return the operation to be added
     */
    static Operation& AddReshapeOperation(Function& function, const LogicalTensorPtr iOperand,
                                          const LogicalTensorPtr& oOperand, const ReshapeOp& reshapeOp,
                                          const std::vector<SymbolicScalar>& outDynShape = {});
    /**
     * @brief Set the DynValidShape of dstTensor by the DynValidShape of srcTensor.
     *
     * @param function the target function, consisting the target op.
     * @param op the target view op.
     */
    static void CopyDynStatus(const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& srcTensor);
    /**
     * @brief Update FromDynOffset of a view op when the input or output is incast or outcast.
     *
     * @param function the target function, consisting the target op.
     * @param op the target view op.
     */
    static void UpdateViewAttr(Function& function, Operation& op);
    /**
     * @brief Set the DynValidShape of the output for the specified op.
     *
     * @param newOp the target operation having oOperands without DynValidShape.
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, set the DynValidShape of each output by CallInferShapeFunc.
     */
    static void SetDynShape(Operation* newOp, const std::vector<std::vector<SymbolicScalar>>& outDynShape = {});
    /**
     * @brief Set the AssembleOpAttribute for a assemble op.
     *
     * @param op the target assemble operation.
     * @param copy AssembleOp, consisting of input, output, fromtype, toOffset.
     */
    static void SetAssembleAttr(Operation& op, const AssembleOp& assemble);
    /**
     * @brief Determine it is a CV separate or CV mix platform.
     */
    static bool IsCVMixPlatform();
    /**
     * @brief Get all tensors in the function that match the given rawMagic.
     *        This method traverses inCasts, outCasts, and all operation inputs/outputs.
     *        It represents the logical-tensor bucket keyed by rawmagic.
     *
     * @param function the target function to search in.
     * @param rawMagic the raw magic ID to match.
     * @return a TensorSet containing LogicalTensorPtrs matching the rawMagic.
     */
    static TensorSet GetTensorsByRawMagic(Function& function, int64_t rawMagic);
    /**
     * @brief Get the shared RawTensor represented by the given rawMagic bucket.
     *
     * @param function the target function to search in.
     * @param rawMagic the raw magic ID of the bucket.
     * @return the shared RawTensor for the bucket, or nullptr if the bucket is empty.
     */
    static std::shared_ptr<RawTensor> GetRawTensorByRawMagic(Function& function, int64_t rawMagic);
    /**
     * @brief Get all tensors in the function that match the given actualRawMagic.
     *        This method traverses inCasts, outCasts, and all operation outputs.
     *
     * @param function the target function to search in.
     * @param actualRawMagic the actual raw magic ID to match.
     * @return a TensorSet containing LogicalTensorPtrs matching the actualRawMagic.
     */
    static TensorSet GetTensorsByActualRawMagic(Function& function, int64_t actualRawMagic);
    /**
     * @brief Get overlapped tensors in the function that match the given tensor's rawMagic.
     *        If current function has no rawMagic bucket, recursively look up parent function.
     *
     * @param function the target function to search in.
     * @param tensor the tensor to match overlap against.
     * @return a list of overlapped LogicalTensorPtrs.
     */
    static std::vector<LogicalTensorPtr> FindOverlappedTensors(Function& function, const LogicalTensorPtr& tensor);
    /**
     * @brief Get the unique tensor in the function that matches the given magic.
     *        This method searches inCasts, outCasts, and all operation outputs.
     *
     * @param function the target function to search in.
     * @param magic the magic ID to match.
     * @return the LogicalTensorPtr matching the magic, or nullptr if not found.
     */
    static LogicalTensorPtr GetTensorByMagic(Function& function, int magic);
    /**
     * @brief Get all tensors in the function.
     *        This method replaces the old global tensorMap_ traversal.
     *
     * @param function the target function to search in.
     * @return a TensorSet containing all LogicalTensorPtrs in the graph.
     */
    static TensorSet GetAllTensors(Function& function);

    // 判断 op 是否为跨核 move(CONVERT/COPY 输入输出跨 AIC/AIV), IntraSubgraphAdapter 会将其
    // 改写为 OP_COPY_OUT 复用原 op 承担 DDR 搬运. ReduceCopyMerge 预判 DDR 化时复用此逻辑.
    static bool IsCrossCoreMoveOp(Operation* op);
};
} // namespace tile_fwk
} // namespace npu
#endif
