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
 * \file tensor_slot.h
 * \brief
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "tilefwk/tensor.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/runtime_slot.h"

namespace npu::tile_fwk {

enum class SlotProperty : uint32_t {
    NONE,
    OUTPUT,
    ASSEMBLE_DST,
    SHMEM_TENSOR,
};

struct TensorSlot {
public:
    TensorSlot() {}
    TensorSlot(int64_t id, const void* slot) : id_(id), slot_(slot) {}

    const void* GetSlot() const { return slot_; }
    int64_t GetId() const { return id_; }

    std::string GetSymbolName() const;

    std::shared_ptr<LogicalTensor> GetSlotValue() const;
    void SetSlotValue(const std::shared_ptr<LogicalTensor>& value) const;

    std::string Dump() const;
    std::string DumpHead(const std::string& name) const;

    bool operator==(const TensorSlot& oth) const { return id_ == oth.id_; }

    static TensorSlot CreateTensor(const Tensor& tensor) { return TensorSlot(tensor.Id(), &tensor); }

private:
    int64_t id_{-1};
    const void* slot_{nullptr};
};
} // namespace npu::tile_fwk

template <>
struct std::hash<npu::tile_fwk::TensorSlot> {
    std::size_t operator()(const npu::tile_fwk::TensorSlot& t) const { return std::hash<int64_t>()(t.GetId()); }
};

namespace npu::tile_fwk {

class Function;

struct TensorSlotAccess {
public:
    TensorSlotAccess() {}

    const std::shared_ptr<LogicalTensor>& GetFirstReadTensor() const { return firstReadTensor_; }
    std::shared_ptr<LogicalTensor> GetFirstReadTensor() { return firstReadTensor_; }

    const std::shared_ptr<LogicalTensor>& GetLastWriteTensor() const { return lastWriteTensor_; }
    std::shared_ptr<LogicalTensor> GetLastWriteTensor() { return lastWriteTensor_; }

    void Read(const std::shared_ptr<LogicalTensor>& tensor)
    {
        if (!written_) {
            firstReadTensor_ = tensor;
        }
    }
    void Write(const std::shared_ptr<LogicalTensor>& tensor)
    {
        written_ = true;
        lastWriteTensor_ = tensor;
    }

    std::string Dump() const
    {
        std::ostringstream oss;
        oss << "<" << (written_ ? 'W' : ' ') << ","
            << (firstReadTensor_ ? firstReadTensor_->Dump() : std::string("noread")) << ","
            << (lastWriteTensor_ ? lastWriteTensor_->Dump() : std::string("nowrite")) << ">";
        return oss.str();
    }

private:
    bool written_{false};
    std::shared_ptr<LogicalTensor> firstReadTensor_; // read before write
    std::shared_ptr<LogicalTensor> lastWriteTensor_;
};

struct IncastOutcastSlot {
    /* One tensor might be passed via multiple slots. An example code:
     *
     *      Tensor a("a"), b("b");
     *      Tensor v0, v1;
     *      FUNCTION("A") {
     *          Tensor t = a + b;
     *          v0 = t; // t outcast via v0
     *          v1 = t; // t outcast via v1
     *      }
     *
     *      Tensor x0;
     *      FUNCTION("B") {
     *          // t incast via both v0 and v1
     *          x0 = v0 + v1;
     *      }
     */
    std::vector<std::vector<int>> incastSlot;
    std::vector<std::vector<int>> outcastSlot;
    std::vector<int> partialUpdateOutcastList;
};

struct TensorSlotScope {
    Function* tensorFunc = nullptr;
    std::unordered_map<TensorSlot, TensorSlotAccess> accessRecord;
    std::unordered_set<int> aliveSlots;

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>> incastToInArgumentDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>> outcastToOutArgumentDict;

    std::vector<std::unordered_set<TensorSlot>> incastReadSlotSet;
    std::vector<std::unordered_set<TensorSlot>> outcastWriteSlotSet;

    std::vector<std::unordered_set<TensorSlot>> oriIncastReadSlotSet;
    std::vector<std::unordered_set<TensorSlot>> oriOutcastWriteSlotSet;

    std::unordered_map<LogicalTensorPtr, int> partialUpdateOutcastDict;

    std::vector<int> constructAssembleSlotList;

    IncastOutcastSlot ioslot;
    IncastOutcastSlot originalIocastsSlot;

    explicit TensorSlotScope(Function* tfunc) : tensorFunc(tfunc) {}
    TensorSlotScope(TensorSlotScope&& scope) = default;
    TensorSlotScope& operator=(TensorSlotScope&& scope) = default;

    std::unordered_set<TensorSlot> LookupIncastReadFrom(const std::shared_ptr<LogicalTensor>& tensor) const;
    std::unordered_set<TensorSlot> LookupOutcastWriteTo(const std::shared_ptr<LogicalTensor>& tensor) const;
    std::unordered_set<TensorSlot> LoopupArgSlot(std::shared_ptr<RawTensor> tensor);
    void InitAliveSlot();
    bool IsAliveSlot(int slotId) const { return aliveSlots.count(slotId) > 0; }

    void BuildSlotSet();
    void BuildIncastOutcastSlot(const std::unordered_map<TensorSlot, int>& slotIndexDict);
    std::string Dump() const;
};

struct IncastOutcastLink {
    explicit IncastOutcastLink(int slotNum = 0) : totalSlot(slotNum) {}

    int totalSlot;
    std::unordered_map<Function*, IncastOutcastSlot> ioslotDict;

    std::vector<int> inputSlotIndexList;
    std::vector<int> outputSlotIndexList;
    std::vector<int> assembleSlotIndexList;
    std::vector<int> shmemTensorSlotIndexList;
    std::vector<int> inplaceSlotIndexList;
    std::vector<int> partialUpdateSlotIdexList;

    std::vector<RuntimeSlotKindSet> runtimeSlotKindSetList;

    void UpdateRuntimeSlotKindSetList()
    {
        runtimeSlotKindSetList.resize(totalSlot);
        for (int inputSlotIndex : inputSlotIndexList) {
            runtimeSlotKindSetList[inputSlotIndex].Add(RuntimeSlotKind::INPUT);
        }
        for (int outputSlotIndex : outputSlotIndexList) {
            runtimeSlotKindSetList[outputSlotIndex].Add(RuntimeSlotKind::OUTPUT);
        }
        for (int assembleSlotIndex : assembleSlotIndexList) {
            runtimeSlotKindSetList[assembleSlotIndex].Add(RuntimeSlotKind::ASSEMBLE_OUTCAST);
        }
        for (int shmemSlotIndex : shmemTensorSlotIndexList) {
            runtimeSlotKindSetList[shmemSlotIndex].Add(RuntimeSlotKind::ADDRESS_EXPRESSION);
        }
    }
};

struct SlotInfo {
    std::shared_ptr<LogicalTensor> tensor;
    int refCount;
};

struct TensorSlotCheckpoint {
    std::unordered_map<TensorSlot, SlotInfo> slotDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::set<Operation*, LogicalTensor::CompareOp>> producerDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::set<Operation*, LogicalTensor::CompareOp>> consumerDict;
};

struct TensorSlotUsage {
    Function* construct{nullptr};
    Function* destruct{nullptr};
    Function* readFirst{nullptr};
    Function* readLast{nullptr};
    Function* writeFirst{nullptr};
    Function* writeLast{nullptr};
};

struct TensorSlotManager {
    std::vector<std::shared_ptr<TensorSlotScope>> scopeList;

    std::shared_ptr<TensorSlotScope> currScope;

    /* Mapping from slot to its index */
    std::unordered_map<TensorSlot, int> slotIndexDict;
    std::unordered_map<TensorSlot, TensorSlotUsage> slotUsageDict;
    std::unordered_map<LogicalTensorPtr, std::shared_ptr<Tensor>> slotTensorDict;

    std::unordered_set<TensorSlot> liveSlotSet;
    std::unordered_set<TensorSlot> assembleSlotSet;
    std::unordered_set<TensorSlot> shmemTensorSlotSet;

    std::unordered_map<std::string, TensorSlot> symbolNameDict;
    std::unordered_map<TensorSlot, std::string> slotNameDict;
    std::unordered_map<TensorSlot, std::string> slotFuncNameDict;
    std::vector<TensorSlot> inputSlotList;
    std::unordered_map<TensorSlot, int> inputSlotDict;
    std::vector<std::string> inputNameList;

    std::vector<TensorSlot> outputSlotList;
    std::unordered_map<TensorSlot, int> outputSlotDict;
    std::vector<std::string> outputNameList;
    std::unordered_map<TensorSlot, TensorSlot> inplaceDict;
    std::unordered_map<std::string, int> nameDict;
    std::unordered_map<TensorSlot, TensorSlot> reshapeInplaceDict;

    std::set<int> partialUpdateSlotIndexSet;

    std::vector<TensorSlotCheckpoint> checkpointStack;
    std::unordered_set<TensorSlot> recycleSlotSet;

    void SetRecording(bool isRecording);
    void BeginScope(Function* tensorFunc);
    std::shared_ptr<TensorSlotScope> EndScope();
    void ConnectSlot(std::shared_ptr<TensorSlotScope> scope);

    void TensorSlotRead(const TensorSlot& slot, const std::shared_ptr<LogicalTensor>& tensor);
    void TensorSlotWrite(const TensorSlot& slot, const std::shared_ptr<LogicalTensor>& tensor);
    void TensorSlotConstruct(const TensorSlot& slot);
    void TensorSlotDestruct(const TensorSlot& slot);

    void TensorRead(const Tensor& tensor);
    void TensorWrite(const Tensor& tensor, SlotProperty property = SlotProperty::NONE);
    void TensorConstruct(const Tensor& tensor);
    void TensorDestruct(const Tensor& tensor);

    void TensorSymbol(const Tensor& tensor, const std::string& symbolName);

    TensorSlotUsage& GetTensorSlotUsage(const TensorSlot& slot);

    std::vector<int> LookupSlotIndex(const std::vector<std::reference_wrapper<Tensor>>& tensorList);
    std::vector<int> LookupSlotIndexConst(const std::vector<std::reference_wrapper<const Tensor>>& tensorList);

    void MarkInput(const Tensor& tensor);
    void MarkOutput(const Tensor& tensor);
    void MarkInplace(const Tensor& out, const Tensor& in);

    const std::vector<std::string>& GetInputNameList() const { return inputNameList; }
    const std::vector<std::string>& GetOutputNameList() const { return outputNameList; }

    int GetInputIndex(const Tensor& tensor);
    int GetOutputIndex(const Tensor& tensor);
    int GetSlotIndex(const Tensor& tensor);
    std::shared_ptr<Tensor> GetSlotTensor(std::shared_ptr<LogicalTensor> lt, bool createIfMissing = true);
    void ClearSlotTensors() { slotTensorDict.clear(); }

    void Checkpoint();
    void Restore();
    void UnwindAllCheckpoints();
    bool HasActiveCheckpoints() const { return !checkpointStack.empty(); }

    void UpdateReshapeInplaceSlots(IncastOutcastLink& link);
    void SetSameSlot(const Tensor& operand, const Tensor& dst);
    void SetSameSlot(const LogicalTensorPtr& src, const LogicalTensorPtr& dst);
    IncastOutcastLink BuildIncastOutcastLink(const std::string& rawname = "");

    static bool HasSameSlot(const std::vector<int>& slots1, const std::vector<int>& slots2);

    std::string Dump() const;

private:
    void LogOperation(const TensorSlot& slot, const std::string& op);
    void InsertLiveSlot(const TensorSlot& slot);
    void TensorSlotRecycle(const TensorSlot& slot);
    bool isRecording_{false};
};
} // namespace npu::tile_fwk
