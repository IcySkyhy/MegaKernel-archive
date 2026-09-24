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
 * \file spill_engine.h
 * \brief Spill execution engine — pure execution layer extracted from OoOScheduler.
 *        Public interface = SpillBuffer (execute a single spill) + factory helpers.
 *        Orchestration methods (GenBufferSpill, SelectSpillBuffers, HasEnoughBuffer,
 *        RearrangeBuffer, ApplySpillContext, PrintSpillFailedInfo, GetSpillGroup,
 *        GetDualSpillGroup, GetGroupNextUseTime) remain in OoOScheduler as decision layer.
 */

#ifndef PASS_SPILL_ENGINE_H
#define PASS_SPILL_ENGINE_H

#include <optional>
#include <unordered_set>
#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "passes/statistics/schedule_observer.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "SpillEngine"

namespace npu::tile_fwk {

class SpillEngine {
public:
    SpillEngine(ScheduleState& state, Function& function) : state_(state), function_(function) {}
    ~SpillEngine() {}

    Status SpillBuffer(int memId, Operation* spillAllocOp, SpillContext& ctx);

    void EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind);
    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType);

    Operation* GetSpillOp(int memId);
    int GetBufNextUseTime(int curMemId);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* op);

private:
    ScheduleState& state_;
    Function& function_;
    IRBuilder irBuilder_;
    int64_t workspaceOffset_{0};
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;
    void EraseSchedulerSideMaps(Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*>& filterLtags);
    bool CheckMachineAndL1(Operation* spillOp, Operation* allocOp);

    Status HandleSpillMode(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp,
                           SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillBufferFromDDR(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp,
                              SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp,
                              SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillL1BufferFor3510(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                       Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillReshapeFromDDRFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp,
                                      LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext& ctx,
                                      SingleSpillCreatedOps& created);
    Status SpillReshapeL1BufferFor3510(int memId, Operation* actualSpillOp, Operation* spillOp,
                                       LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext& ctx,
                                       SingleSpillCreatedOps& created);
    Status SpillL0CBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp,
                          SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
                                    Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBufferFor3510(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
                                           Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created);

    Status CopyoutParticalBuffer(LogicalTensorPtr spillTensor, LogicalTensorPtr gmTensor, SpillContext& ctx);
    Status CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleOOperand,
                                Operation* copyoutOp, Operation* spillAllocOp);
    Status FillSpillAssembleBuffer(int spillMemid, LogicalTensorPtr spillTensor, LogicalTensorPtr assembleTensor,
                                   Operation* copyoutOp, LogicalTensorPtr gmTensor, Operation* spillAllocOp,
                                   Operation*& wholeCopyinOut);

    LogicalTensorPtr CreateLocalTensor(LogicalTensorPtr spillTensor);
    LogicalTensorPtr CreateGMTensor(LogicalTensorPtr spillTensor, LogicalTensorPtr actualSpillTensor, int spillMemId,
                                    DataType gmDtype = DT_BOTTOM);
    LogicalTensorPtr CreateParticalTensor(LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand,
                                          LogicalTensorPtr spillTensor, std::vector<int64_t> toOffset);
    Operation* CreateAllocOp(LogicalTensorPtr oOperand);
    Operation* CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset,
                              bool isND2NZ = false);
    Operation* CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                               std::vector<OpImmediate> offset);
    Operation* CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<int64_t> toOffset,
                                std::vector<SymbolicScalar> toDynOffset, std::vector<SymbolicScalar> fromDynValidShape);

    const std::vector<int64_t>& GetLargerShape(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2);

    bool IsUnusedTensor(Operation* spillOp);
    Status UpdateSuccessorDependencies(Operation* succOp, Operation* spillOp, Operation* reloadCopyin, int spillMemId,
                                       int reloadMemId);
    void UpdatePredecessorAllocDependencies(Operation* succOp, Operation* reloadAlloc, int spillMemId);
    Status UpdateSmallShapeDependAndBuf(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int spillMemId,
                                        Operation* spillOp);

    void CollectUBSceneOpsAndTensors(Operation* producerOp, std::vector<Operation*>& opsToDelete,
                                     std::vector<LogicalTensorPtr>& tensorsToDelete);
    void CollectProducerChainForDeletion(LogicalTensorPtr spillTensor, std::vector<Operation*>& opsToDelete,
                                         std::vector<LogicalTensorPtr>& tensorsToDelete);
    void ReleaseDeletedOpBufRefs(Operation* op, const std::vector<LogicalTensorPtr>& tensorsToDelete);
    size_t CleanupCollectedOperations(const std::vector<Operation*>& opsToDelete,
                                      const std::vector<LogicalTensorPtr>& tensorsToDelete);
    void CleanupCollectedTensors(const std::vector<LogicalTensorPtr>& tensorsToDelete);
    void EraseOrphanedTensors(const std::vector<LogicalTensorPtr>& tensorsToDelete,
                              const std::vector<Operation*>& opsToDelete);
    Status RemoveSmallShapeSpillResources(int spillMemId, LogicalTensorPtr spillTensor, SpillContext& ctx);

    LogicalTensorPtr GetSpillTensor(Operation* spillOp, int spillMemId);
    void CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*>& consumers);
    Status GetActualSpillForNd2nz(Operation*& spillOp, LogicalTensorPtr& spillTensor);
    Status GetActualSpill(Operation* op, Operation*& actualOp, LogicalTensorPtr& actualTensor);
    Status UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId);

    Status UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr reloadTensor, int spillMemId);
    Status UpdateSkipOpInput(Operation* chainTail, Operation* spillOp, Operation* targetOp,
                             LogicalTensorPtr reloadTensor, size_t index);
    LogicalTensorPtr CloneSkipChain(Operation* targetOp, const std::vector<Operation*>& chain,
                                    LogicalTensorPtr reloadTensor);
    void DetachOrphanedSkipChain(const std::vector<Operation*>& chain, Operation* targetOp);
    void ReplaceSkipOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId);
    void ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId);
    Status UpdateRemainMemid(int oldMemId, int newMemId);
    void UpdateOpInternalSubgraphID(Operation& op, Operation* srcOp);

    Status UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId,
                                     Operation* spillAllocOp, bool isRetired = true);
    void UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp);
    Status InsertOps(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, Operation* spillAllocOp,
                     int memId);
    Status UpdateScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
                                Operation* spillAllocOp, LogicalTensorPtr localTensor, Operation* spillOp);
    Status UpdateNeedDeleteScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
                                          Operation* spillAllocOp, LogicalTensorPtr spillTensor, Operation* spillOp,
                                          SpillContext& ctx);
    bool IsMultiProducerTensor(LogicalTensorPtr tensor);
    Status GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t>& toOffset,
                                     std::vector<SymbolicScalar>& toDynOffset,
                                     std::vector<SymbolicScalar>& fromDynValidShape) const;

    void UpdateScheduleStatusForDualDst(const std::vector<std::pair<Operation*, std::vector<int>>>& opMemidMap,
                                        int memId, Operation* spillAllocOp, Operation* spillOp);
};

} // namespace npu::tile_fwk
#endif // PASS_SPILL_ENGINE_H
