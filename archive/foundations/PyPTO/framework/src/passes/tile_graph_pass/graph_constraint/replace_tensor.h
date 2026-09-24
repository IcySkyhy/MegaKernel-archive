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
 * \file replace_tensor.h
 * \brief
 */

#ifndef REPLACE_TENSOR_H
#define REPLACE_TENSOR_H
#include <vector>
#include <climits>
#include <queue>

#include "tilefwk/data_type.h"
#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {

struct OperandCount {
    constexpr static size_t VIEW_INPUT = 1;
    constexpr static size_t VIEW_OUTPUT = 1;
    constexpr static size_t ASSEMBLE_INPUT = 1;
    constexpr static size_t ASSEMBLE_OUTPUT = 1;
    constexpr static size_t RESHAPE_INPUT = 1;
    constexpr static size_t RESHAPE_OUTPUT = 1;
    constexpr static size_t INDEX_OUTCAST_INPUTS = 3;
    constexpr static size_t INDEX_OUTCAST_OUTPUT = 1;
    constexpr static size_t A_MULACC_B_MIN_INPUTS = 3;
    constexpr static size_t A_MULACC_B_MAX_INPUTS = 4;
    constexpr static size_t A_MULACC_B_MAX_INPUTS_A5 = 5;
    constexpr static size_t A_MULACC_B_OUTPUT = 1;
};

/*
key: Opcode类型
vaule: vector of pair, 每个pair记录了第几个输入和第几个输出存在inplace关系
*/
const std::unordered_map<Opcode, std::vector<std::pair<size_t, size_t>>> inplaceOpMap = {
    {Opcode::OP_A_MULACC_B, {std::pair<size_t, size_t>{2, 0}}},
    {Opcode::OP_INDEX_OUTCAST, {std::pair<size_t, size_t>{2, 0}}},
};

const std::unordered_set<Opcode> inplaceOpSet = {Opcode::OP_VIEW,       Opcode::OP_ASSEMBLE,      Opcode::OP_RESHAPE,
                                                 Opcode::OP_A_MULACC_B, Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW_TYPE};

class UnionFind {
public:
    explicit UnionFind(std::unordered_map<LogicalTensorPtr, int>& tensorToOrderIndex)
    {
        for (auto it = tensorToOrderIndex.begin(); it != tensorToOrderIndex.end(); it++) {
            parentMap[it->first] = it->first;
            rankMap[it->first] = 1;
        }
    }

    void Unite(const LogicalTensorPtr x, const LogicalTensorPtr y)
    {
        LogicalTensorPtr xRoot = Find(x);
        LogicalTensorPtr yRoot = Find(y);
        if (xRoot == yRoot) {
            return;
        }
        if (rankMap[xRoot] < rankMap[yRoot]) {
            parentMap[xRoot] = yRoot;
        } else if (rankMap[xRoot] > rankMap[yRoot]) {
            parentMap[yRoot] = xRoot;
        } else {
            parentMap[yRoot] = xRoot;
            rankMap[xRoot]++;
        }
    }

    std::vector<LogicalTensors> GetGroups() const
    {
        std::unordered_map<LogicalTensorPtr, LogicalTensors> rootToGroup;
        for (const auto& pair : parentMap) {
            LogicalTensorPtr obj = pair.first;
            LogicalTensorPtr root = Find(obj);
            rootToGroup[root].push_back(obj);
        }
        std::vector<LogicalTensors> groups;
        for (const auto& pair : rootToGroup) {
            groups.push_back(pair.second);
        }
        return groups;
    }

private:
    LogicalTensorPtr Find(const LogicalTensorPtr& x) const
    {
        if (parentMap[x] != x) {
            parentMap[x] = Find(parentMap[x]);
        }
        return parentMap[x];
    }
    mutable std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> parentMap;
    std::unordered_map<LogicalTensorPtr, int> rankMap;
};

class ReplaceTensor : public Pass {
public:
    ReplaceTensor() : Pass("ReplaceTensor") {}
    ~ReplaceTensor() override = default;

private:
    /*
    补齐: Status PreCheck(Function &function) override;
    补齐: Status PostCheck(Function &function) override;
    */
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    bool HasSameConsecutive(Operation& op);
    bool CheckAddrConflict(const Operation& op);
    bool CheckIndexProducer(const Operation& op);
    bool CheckAssembleConflict(const Operation& op);
    bool CheckIndexOutcastConflict(const Operation& op, Function& function);
    bool CheckReshapeConflict(const Operation& op, Function& function);
    bool CheckAMulAccBConflict(const Operation& op);
    Status InplaceCheck(Function& function);
    bool CheckInplace(const Operation& op);

    std::unordered_map<LogicalTensorPtr, int> BuildTensorOrderIndexMap(Function& function);
    Status FindBaseTensor(Function& function, const std::unordered_map<LogicalTensorPtr, int>& tensorToOderIndex,
                          LogicalTensors& group, LogicalTensorPtr& baseTensor);
    Status ProcessHubOp(Function& function);
    void ProcessHubAssembleOp(Function& function, Operation& hubOp, Operation& assembleOp,
                              std::shared_ptr<LogicalTensor> hubInput, std::shared_ptr<LogicalTensor> hubOutput);
    LogicalTensorPtr FindReplaceSource(Function& function, Operation& op,
                                       std::unordered_map<Operation*, LogicalTensorPtr>& visited);
    Status RefactorViewConnectForReplace(Function& function);
    void UniteTensor(Function& function, UnionFind& uf);

    Status AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const;
    Status AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const;
    Status AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const;

    Status ForwardProcess(Function& function);
    Status BackwardProcess(Function& function);

    Status ForwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardAssemble(Operation* op, LogicalTensorPtr& rootTensor);
    Status ForwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardViewType(Operation* op, LogicalTensorPtr& rootTensor);
    Status ForwardCopyOut(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function);

    Status BackwardAssemble(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardReshape(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardView(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardViewType(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function);

    Status ForUpdateView(Operation* op);
    Status BackUpdateAssemble(Operation* op);
    // L0C2UB 直写折叠：将刷新到 assemble 输入 tensor 上的目的偏移落入其唯一 producer（L0C2UB copy）
    // 的 toOffset，并将 copy 模式改为 INSERT
    Status FoldL0C2UBCopyOffset(Operation* op);
    std::vector<OpImmediate> SumOffsetForCopyIn(const std::vector<OpImmediate> offset1,
                                                const std::vector<OpImmediate> offset2);
    Status UpdateCopyInAttr(Operation* copyInOp);

    Status MarkTensorAsPartialMem(Function& function);

    Status InsertCopyUBOp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input);
    Status InsertCopyDDROp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input);
    Status FindNeedToCopyAssemble(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                  std::unordered_set<int>& visitedAssOps, Operation& op, Function& function);
    Status FindNeedToCopyReshape(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                 std::unordered_set<int>& visitedReshapeOps, Operation& op, Function& function);
    Status InsertNeedCopy(Function& function);
    bool isBoundTensor(LogicalTensorPtr& curTensor);

    std::unordered_map<DataType, int> viewTypeTable = {{DT_INT8, 1},    {DT_BF16, 2},    {DT_FP16, 2},   {DT_FP32, 4},
                                                       {DT_FP8E4M3, 1}, {DT_FP8E5M2, 1}, {DT_FP8E8M0, 1}};
    std::queue<LogicalTensorPtr> backRoots;
    std::queue<LogicalTensorPtr> forRoots;
    std::unordered_set<int> processedOp;
    std::unordered_set<int> backwardOps;
    std::unordered_set<int> forwardOps;
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // REPLACE_TENSOR_H
