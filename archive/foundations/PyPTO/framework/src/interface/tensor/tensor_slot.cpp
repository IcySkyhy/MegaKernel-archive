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
 * \file tensor_slot.cpp
 * \brief
 */

#include "tensor_slot.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/utils/error.h"
#include "interface/program/program.h"
#include "interface/utils/string_utils.h"
#include "irbuilder.h"
namespace npu::tile_fwk {

// 根据nameDict里记录的name出现次数，为多次同命名的名称添加递增后缀
static void AddNameSuffix(std::string& name, std::unordered_map<std::string, int>& nameDict)
{
    auto it = nameDict.find(name);
    if (it != nameDict.end()) {
        ++(it->second);
        name += "_" + std::to_string(it->second);
        AddNameSuffix(name, nameDict);
    } else {
        nameDict[name] = 0;
    }
}

std::string TensorSlot::GetSymbolName() const
{
    std::string name;
    const Tensor* t = reinterpret_cast<const Tensor*>(GetSlot());
    if (t->GetStorage(false) != nullptr) {
        name = t->GetStorage(false)->tensor->symbol;
    }
    return name;
}

std::shared_ptr<LogicalTensor> TensorSlot::GetSlotValue() const
{
    std::shared_ptr<LogicalTensor> value;
    const Tensor* tensor = reinterpret_cast<const Tensor*>(GetSlot());
    value = tensor->GetStorage(false);
    return value;
}

void TensorSlot::SetSlotValue(const std::shared_ptr<LogicalTensor>& value) const
{
    Tensor* tensor = reinterpret_cast<Tensor*>(const_cast<void*>(GetSlot()));
    tensor->GetStorage(false) = value;
}

std::string TensorSlot::DumpHead(const std::string& name) const
{
    constexpr int width = 15;
    std::ostringstream oss;
    std::string symbol = name;
    if (symbol != "") {
        symbol = "(" + symbol + ")";
    }
    oss << "Id:" << id_ << " slot:" << GetSlot() << std::setw(width) << std::left << symbol;
    return oss.str();
}

std::string TensorSlot::Dump() const
{
    std::ostringstream oss;
    oss << DumpHead(GetSymbolName());
    std::shared_ptr<LogicalTensor> value = GetSlotValue();
    if (value != nullptr) {
        oss << " value:" << value.get() << "(" << value->Dump(true, true) << ")";
    }
    return oss.str();
}

std::unordered_set<TensorSlot> TensorSlotScope::LookupIncastReadFrom(const std::shared_ptr<LogicalTensor>& tensor) const
{
    std::unordered_set<TensorSlot> tensorSlot;
    for (auto& [slot, access] : accessRecord) {
        /* Match by raw tensor */
        if (access.GetFirstReadTensor() && access.GetFirstReadTensor()->tensor == tensor->tensor) {
            tensorSlot.insert(slot);
        }
    }
    return tensorSlot;
}

void TensorSlotScope::InitAliveSlot()
{
    aliveSlots.clear();
    for (auto tensor : Program::GetInstance().GetAliveTensors()) {
        aliveSlots.insert(tensor->Id());
    }
}

std::unordered_set<TensorSlot> TensorSlotScope::LookupOutcastWriteTo(const std::shared_ptr<LogicalTensor>& tensor) const
{
    std::unordered_set<TensorSlot> tensorSlot;
    for (auto& [slot, access] : accessRecord) {
        /* Match by raw tensor */
        if (access.GetLastWriteTensor() && access.GetLastWriteTensor()->tensor == tensor->tensor) {
            if (!Program::GetInstance().GetTensorSlotManager()->liveSlotSet.count(slot)) {
                continue;
            }
            tensorSlot.insert(slot);
        }
    }
    return tensorSlot;
}

std::unordered_set<TensorSlot> TensorSlotScope::LoopupArgSlot(std::shared_ptr<RawTensor> tensor)
{
    auto realArgs = tensor;

    for (auto& [incast, arg] : incastToInArgumentDict) {
        if (incast->tensor == tensor) {
            realArgs = arg->tensor;
            break;
        }
    }
    for (auto& [outcast, arg] : outcastToOutArgumentDict) {
        if (outcast->tensor == tensor) {
            realArgs = arg->tensor;
            break;
        }
    }

    std::unordered_set<TensorSlot> tensorSlot;
    for (auto& [slot, access] : accessRecord) {
        if (access.GetFirstReadTensor() && access.GetFirstReadTensor()->tensor == realArgs) {
            tensorSlot.insert(slot);
        }
        if (access.GetLastWriteTensor() && access.GetLastWriteTensor()->tensor == realArgs) {
            tensorSlot.insert(slot);
        }
    }
    return tensorSlot;
}

void TensorSlotScope::BuildSlotSet()
{
    if (accessRecord.size() == 0) {
        return;
    }
    for (size_t idx = 0; idx < tensorFunc->GetIncast().size(); idx++) {
        auto& i = tensorFunc->GetIncast()[idx];
        FE_ASSERT(FeError::NOT_EXIST, incastToInArgumentDict.count(i))
            << "LogicalTensor[" << i->GetMagic() << "] not found in incastToInArgumentDict.";
        auto iarg = incastToInArgumentDict[i];
        auto slot = LookupIncastReadFrom(iarg);
        incastReadSlotSet.push_back(slot);
    }
    for (size_t idx = 0; idx < tensorFunc->GetOutcast().size(); idx++) {
        auto& o = tensorFunc->GetOutcast()[idx];
        FE_ASSERT(FeError::NOT_EXIST, outcastToOutArgumentDict.count(o))
            << "LogicalTensor[" << o->GetMagic() << "] not found in outcastToOutArgumentDict.";
        auto oarg = outcastToOutArgumentDict[o];
        auto slot = LookupOutcastWriteTo(oarg);
        outcastWriteSlotSet.push_back(slot);
    }
}

void TensorSlotScope::BuildIncastOutcastSlot(const std::unordered_map<TensorSlot, int>& slotIndexDict)
{
    ioslot.incastSlot.resize(tensorFunc->GetIncast().size());
    for (size_t idx = 0; idx < tensorFunc->GetIncast().size(); idx++) {
        for (auto& h : incastReadSlotSet[idx]) {
            FE_ASSERT(FeError::NOT_EXIST, slotIndexDict.count(h) != 0)
                << "TensorSlot[" << h.GetSymbolName() << "] not found in slotIndexDict.";
            ioslot.incastSlot[idx].push_back(slotIndexDict.find(h)->second);
        }
        std::sort(ioslot.incastSlot[idx].begin(), ioslot.incastSlot[idx].end());
    }

    ioslot.outcastSlot.resize(tensorFunc->GetOutcast().size());
    for (size_t idx = 0; idx < tensorFunc->GetOutcast().size(); idx++) {
        for (auto& h : outcastWriteSlotSet[idx]) {
            FE_ASSERT(FeError::NOT_EXIST, slotIndexDict.count(h) != 0)
                << "TensorSlot[" << h.GetSymbolName() << "] not found in slotIndexDict.";
            ioslot.outcastSlot[idx].push_back(slotIndexDict.find(h)->second);
        }
        std::sort(ioslot.outcastSlot[idx].begin(), ioslot.outcastSlot[idx].end());

        auto outcast = tensorFunc->GetOutcast()[idx];
        auto itor = partialUpdateOutcastDict.find(outcast);
        if (itor != partialUpdateOutcastDict.end()) {
            ioslot.partialUpdateOutcastList.push_back(idx);
        }
    }
}

std::string TensorSlotScope::Dump() const
{
    std::string INDENT = "  ";
    std::ostringstream oss;
    oss << "scope {\n" << INDENT << "#name:" << tensorFunc->GetMagicName() << "\n";
    for (auto& [slot, access] : accessRecord) {
        oss << INDENT << "slot:" << slot.GetSlot() << " id "
            << Program::GetInstance().GetTensorSlotManager()->slotIndexDict[slot] << " access:" << access.Dump()
            << "\n";
    }
    for (auto& [incast, inarg] : incastToInArgumentDict) {
        oss << INDENT << "incast:" << incast->Dump() << " inarg:" << inarg->Dump() << "\n";
    }
    for (auto& [outcast, outarg] : outcastToOutArgumentDict) {
        oss << INDENT << "outcast:" << outcast->Dump() << " outarg:" << outarg->Dump() << "\n";
    }
    oss << "}\n";
    return oss.str();
}

void TensorSlotManager::TensorSlotRecycle(const TensorSlot& slot)
{
    slotIndexDict.erase(slot);
    slotUsageDict.erase(slot);
    auto name = slotNameDict.find(slot);
    if (name != slotNameDict.end()) {
        symbolNameDict.erase(name->second);
        slotNameDict.erase(slot);
    }
    slotFuncNameDict.erase(slot);
}

void TensorSlotManager::SetRecording(bool isRecording)
{
    isRecording_ = isRecording;
    if (!isRecording_) {
        for (auto& slot : recycleSlotSet) {
            TensorSlotRecycle(slot);
        }
        recycleSlotSet.clear();
    }
}

void TensorSlotManager::BeginScope(Function* tensorFunc)
{
    std::shared_ptr<TensorSlotScope> scope = std::make_shared<TensorSlotScope>(tensorFunc);
    scopeList.push_back(scope);
    currScope = scope;
    scope->InitAliveSlot();
    tensorFunc->SetSlotScope(scope);
}

std::shared_ptr<TensorSlotScope> TensorSlotManager::EndScope()
{
    std::shared_ptr<TensorSlotScope> lastScope = currScope;

    currScope = nullptr;
    return lastScope;
}

void TensorSlotManager::ConnectSlot(std::shared_ptr<TensorSlotScope> scope)
{
    scope->BuildSlotSet();
    scope->BuildIncastOutcastSlot(slotIndexDict);
    scope->tensorFunc->SetSlotScope(scope);
}

void TensorSlotManager::InsertLiveSlot(const TensorSlot& slot)
{
    if (slotIndexDict.count(slot) == 0) {
        slotIndexDict[slot] = slot.GetId();
        slotUsageDict[slot] = TensorSlotUsage();
    }
    liveSlotSet.insert(slot);
}

TensorSlotUsage& TensorSlotManager::GetTensorSlotUsage(const TensorSlot& slot) { return slotUsageDict[slot]; }

static Function* GetCurrentNonHiddenFunction()
{
    Function* currNonHiddenFunction = Program::GetInstance().GetCurrentFunction();
    while (currNonHiddenFunction && currNonHiddenFunction->IsHiddenFunction()) {
        FE_ASSERT(currNonHiddenFunction->HasParent()) << "currNonHiddenFunction doesn't have parent func.";
        currNonHiddenFunction = &currNonHiddenFunction->Parent();
    }
    FE_ASSERT(currNonHiddenFunction != nullptr);
    return currNonHiddenFunction;
}

void TensorSlotManager::TensorSlotRead(const TensorSlot& slot, const std::shared_ptr<LogicalTensor>& tensor)
{
    InsertLiveSlot(slot);
    if (currScope) {
        currScope->accessRecord[slot].Read(tensor);
    }

    TensorSlotUsage& slotUsage = GetTensorSlotUsage(slot);
    if (slotUsage.readFirst == nullptr) {
        slotUsage.readFirst = GetCurrentNonHiddenFunction();
    }
    slotUsage.readLast = GetCurrentNonHiddenFunction();
}

void TensorSlotManager::TensorSlotWrite(const TensorSlot& slot, const std::shared_ptr<LogicalTensor>& tensor)
{
    InsertLiveSlot(slot);
    if (currScope) {
        currScope->accessRecord[slot].Write(tensor);
    }

    TensorSlotUsage& slotUsage = GetTensorSlotUsage(slot);
    if (slotUsage.writeFirst == nullptr) {
        slotUsage.writeFirst = GetCurrentNonHiddenFunction();
    }
    slotUsage.writeLast = GetCurrentNonHiddenFunction();
}

void TensorSlotManager::TensorSlotConstruct(const TensorSlot& slot)
{
    InsertLiveSlot(slot);

    TensorSlotUsage& slotUsage = GetTensorSlotUsage(slot);
    slotUsage.construct = GetCurrentNonHiddenFunction();
}

void TensorSlotManager::TensorSlotDestruct(const TensorSlot& slot)
{
    if (slotIndexDict.count(slot) == 0) {
        return;
    }

    TensorSlotUsage& slotUsage = GetTensorSlotUsage(slot);
    slotUsage.destruct = GetCurrentNonHiddenFunction();

    if (liveSlotSet.count(slot)) {
        liveSlotSet.erase(slot);
    }
    if (isRecording_) {
        // slot info maybe used during end function, could not freed directly
        recycleSlotSet.insert(slot);
    } else {
        TensorSlotRecycle(slot);
    }
}

static std::string Width(const std::string& suffix, int width)
{
    std::ostringstream oss;
    oss << std::setw(width) << std::left << suffix;
    return oss.str();
}

void TensorSlotManager::LogOperation(const TensorSlot& slot, const std::string& op)
{
    std::string ops = Width(op, 10);
    FE_LOGD("[slotManager] %zu op:%s %s", slotIndexDict.size(), ops.c_str(), slot.Dump().c_str());
}

void TensorSlotManager::TensorRead(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    std::shared_ptr<LogicalTensor> storage = tensor.GetStorage(false);
    TensorSlotRead(slot, storage);

    LogOperation(slot, "read");
}

void TensorSlotManager::TensorWrite(const Tensor& tensor, SlotProperty property)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    std::shared_ptr<LogicalTensor> storage = tensor.GetStorage(false);
    TensorSlotWrite(slot, storage);
    if (property == SlotProperty::ASSEMBLE_DST) {
        assembleSlotSet.insert(slot);
    } else if (property == SlotProperty::SHMEM_TENSOR) {
        shmemTensorSlotSet.insert(slot);
    }
    FE_ASSERT(tensor.GetStorage(false) != nullptr) << "Assigning uninitialized Tensor variable is forbidden";
    LogOperation(slot, "write");
}

void TensorSlotManager::TensorConstruct(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);

    TensorSlotConstruct(slot);

    LogOperation(slot, "construct");
}

void TensorSlotManager::TensorDestruct(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);

    TensorSlotDestruct(slot);

    LogOperation(slot, "destruct");
}

void TensorSlotManager::TensorSymbol(const Tensor& tensor, const std::string& symbolName)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    symbolNameDict[symbolName] = slot;
    slotNameDict[slot] = symbolName;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) != CFG_RUNTIME_DEBUG_VERIFY) {
        return;
    }
    Function* currFunc = Program::GetInstance().GetCurrentFunction();
    if (currFunc != nullptr) {
        StringUtils::AppendUniqueToken(slotFuncNameDict[slot], currFunc->GetRawName());
    }
}

std::vector<int> TensorSlotManager::LookupSlotIndex(const std::vector<std::reference_wrapper<Tensor>>& tensorList)
{
    std::vector<int> indexList;
    for (auto& tensor : tensorList) {
        TensorSlot slot = TensorSlot::CreateTensor(tensor);
        if (slotIndexDict.count(slot)) {
            indexList.push_back(slotIndexDict[slot]);
        } else {
            indexList.push_back(-1);
        }
    }
    return indexList;
}

std::vector<int> TensorSlotManager::LookupSlotIndexConst(
    const std::vector<std::reference_wrapper<const Tensor>>& tensorList)
{
    std::vector<int> indexList;
    for (auto& tensor : tensorList) {
        TensorSlot slot = TensorSlot::CreateTensor(tensor);
        if (slotIndexDict.count(slot)) {
            indexList.push_back(slotIndexDict[slot]);
        } else {
            indexList.push_back(-1);
        }
    }
    return indexList;
}

std::shared_ptr<Tensor> TensorSlotManager::GetSlotTensor(LogicalTensorPtr lt, bool createIfMissing)
{
    auto it = slotTensorDict.find(lt);
    if (it == slotTensorDict.end()) {
        if (!createIfMissing) {
            return nullptr;
        }
        it = slotTensorDict.emplace(lt, std::make_shared<Tensor>(lt)).first;
    }
    return it->second;
}

void TensorSlotManager::SetSameSlot(const LogicalTensorPtr& src, const LogicalTensorPtr& dst)
{
    auto srcTensor = GetSlotTensor(src);
    auto dstTensor = GetSlotTensor(dst, false);
    if (dstTensor == nullptr) {
        slotTensorDict[dst] = srcTensor;
        return;
    }
    if (srcTensor->Id() == dstTensor->Id()) {
        return;
    }
    // Migrate all tensors pointing to dst's old slot to src's slot, so no prior
    // links are lost regardless of link order or direction.
    for (auto& [lt, tensor] : slotTensorDict) {
        (void)lt;
        if (tensor->Id() == dstTensor->Id()) {
            tensor = srcTensor;
        }
    }
}

void TensorSlotManager::MarkInput(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    FE_ASSERT(inputSlotDict.count(slot) == 0)
        << "TensorSlot[" << slot.GetSymbolName() << "] already exists in inputSlotDict.";
    inputSlotDict[slot] = inputSlotList.size();
    inputSlotList.push_back(slot);
    auto logicalTensor = tensor.GetStorage();

    std::string inputName = logicalTensor ? logicalTensor->tensor->symbol : "untitled";
    AddNameSuffix(inputName, nameDict);
    inputNameList.push_back(inputName);
    FE_LOGD("MarkInput push input name[%s].", inputName.c_str());

    LogOperation(slot, "input");
}

void TensorSlotManager::MarkOutput(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    FE_ASSERT(outputSlotDict.count(slot) == 0)
        << "TensorSlot[" << slot.GetSymbolName() << "] already exists in outputSlotDict.";
    outputSlotDict[slot] = outputSlotList.size();
    outputSlotList.push_back(slot);
    auto logicalTensor = tensor.GetStorage(false);

    std::string outputName = logicalTensor ? logicalTensor->tensor->symbol : "untitled";
    AddNameSuffix(outputName, nameDict);
    outputNameList.push_back(outputName);
    FE_LOGD("MarkOutput push output name[%s].", outputName.c_str());

    LogOperation(slot, "output");
}

void TensorSlotManager::MarkInplace(const Tensor& out, const Tensor& in)
{
    MarkOutput(out);
    TensorSlot outSlot = TensorSlot::CreateTensor(out);
    TensorSlot inSlot = TensorSlot::CreateTensor(in);
    FE_ASSERT(inputSlotDict.count(inSlot) != 0)
        << "TensorSlot[" << inSlot.GetSymbolName() << "] not found in inputSlotDict.";
    inplaceDict[outSlot] = inSlot;
    FE_LOGD("Slot already inplace [%s, %s].", inSlot.GetSymbolName().c_str(), outSlot.GetSymbolName().c_str());
}

int TensorSlotManager::GetInputIndex(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    for (size_t i = 0; i < inputSlotList.size(); i++) {
        if (slot == inputSlotList[i]) {
            return i;
        }
    }
    return -1;
}

int TensorSlotManager::GetOutputIndex(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    for (size_t i = 0; i < outputSlotList.size(); i++) {
        if (slot == outputSlotList[i]) {
            return i;
        }
    }
    return -1;
}

int TensorSlotManager::GetSlotIndex(const Tensor& tensor)
{
    TensorSlot slot = TensorSlot::CreateTensor(tensor);
    return slotIndexDict[slot];
}

void TensorSlotManager::Checkpoint()
{
    TensorSlotCheckpoint checkpoint;

    std::unordered_set<std::shared_ptr<LogicalTensor>> tensorSet;
    for (auto& slot : liveSlotSet) {
        auto storage = slot.GetSlotValue();
        int refCount = 0;
        if (storage && storage->tensor) {
            refCount = storage->tensor->GetRefCount();
        }
        checkpoint.slotDict[slot] = {storage, refCount};
        tensorSet.insert(storage);

        LogOperation(slot, "checkpoint");
    }
    for (auto& tensor : tensorSet) {
        if (tensor == nullptr) {
            continue;
        }
        checkpoint.producerDict[tensor] = tensor->GetProducers();
        checkpoint.consumerDict[tensor] = tensor->GetConsumers();
    }
    checkpointStack.push_back(std::move(checkpoint));
}

void TensorSlotManager::UnwindAllCheckpoints()
{
    while (!checkpointStack.empty()) {
        Restore();
    }
}

void TensorSlotManager::Restore()
{
    FE_ASSERT(checkpointStack.size() != 0) << "checkpointStack.size(): " << checkpointStack.size();
    TensorSlotCheckpoint& checkpoint = checkpointStack.back();
    for (auto& [slot, value] : checkpoint.slotDict) {
        if (!liveSlotSet.count(slot)) {
            continue;
        }
        auto storage = value.tensor;
        slot.SetSlotValue(storage);
        if (storage && storage->tensor) {
            storage->tensor->SetRefCount(value.refCount);
        }
        LogOperation(slot, "restore");
    }

    std::vector<std::shared_ptr<LogicalTensor>> tensorList;
    for (auto& ele : checkpoint.producerDict) {
        tensorList.push_back(ele.first);
    }

    for (auto tensor : tensorList) {
        tensor->GetProducers() = checkpoint.producerDict[tensor];
        tensor->GetConsumers() = checkpoint.consumerDict[tensor];
    }
    checkpointStack.pop_back();
}

std::string TensorSlotManager::Dump() const
{
    std::vector<TensorSlot> slotList(slotIndexDict.size());
    for (auto& [slot, index] : slotIndexDict) {
        slotList[index] = slot;
    }
    constexpr int width2 = 2;
    constexpr int width6 = 6;
    constexpr int width7 = 7;

    std::ostringstream oss;
    for (size_t i = 0; i < slotList.size(); i++) {
        bool live = liveSlotSet.count(slotList[i]);
        bool assemble = assembleSlotSet.count(slotList[i]);
        bool shmemTensor = shmemTensorSlotSet.count(slotList[i]);
        bool input = inputSlotDict.count(slotList[i]);
        bool output = outputSlotDict.count(slotList[i]);
        bool named = slotNameDict.count(slotList[i]);
        bool parial = partialUpdateSlotIndexSet.count(i);
        if (live || input || output || named) {
            oss << "slot[" << std::setw(width2) << i << "]: ";
            oss << std::setw(width2) << (live ? 'L' : ' ');
            oss << std::setw(width2) << (assemble ? 'A' : ' ');
            oss << std::setw(width2) << (shmemTensor ? 'S' : ' ');
            oss << std::setw(width2) << (parial ? 'P' : ' ');
            oss << std::setw(width6)
                << (input ? "in:" + std::to_string(inputSlotDict.find(slotList[i])->second) : std::string(" "));
            oss << std::setw(width7)
                << (output ? "out:" + std::to_string(outputSlotDict.find(slotList[i])->second) : std::string(" "));
            if (live) {
                oss << " " << slotList[i].Dump() << "\n";
            } else {
                oss << " " << slotList[i].DumpHead(slotNameDict.find(slotList[i])->second) << "\n";
            }
        }
    }
    oss << "slotSize:" << slotList.size() << "\n";
    return oss.str();
}

void TensorSlotManager::UpdateReshapeInplaceSlots(IncastOutcastLink& link)
{
    for (auto& [slotIn, slotOut] : reshapeInplaceDict) {
        FE_ASSERT(slotIndexDict.find(slotIn) != slotIndexDict.end())
            << "slotIn[" << slotIn.GetSymbolName() << "]is not in slotIndexDict";
        FE_ASSERT(slotIndexDict.find(slotOut) != slotIndexDict.end())
            << "slotOut[" << slotOut.GetSymbolName() << "]is not in slotIndexDict";

        for (auto& iter : link.ioslotDict) {
            auto& ioslot = iter.second;
            // update incast for all funcs
            for (std::vector<int>& slotsIdxIn : ioslot.incastSlot) {
                for (auto& slotIdxIn : slotsIdxIn) {
                    if (slotIdxIn == slotIndexDict[slotIn]) {
                        FE_LOGD("replace slot %d to %d.", slotIdxIn, slotIndexDict[slotOut]);
                        slotIdxIn = slotIndexDict[slotOut];
                    }
                }
            }

            for (std::vector<int>& slotsIdxOut : ioslot.outcastSlot) {
                for (auto& slotIdxOut : slotsIdxOut) {
                    if (slotIdxOut == slotIndexDict[slotIn]) {
                        FE_LOGD("replace slot %d to %d.", slotIdxOut, slotIndexDict[slotOut]);
                        slotIdxOut = slotIndexDict[slotOut];
                    }
                }
            }
        }
    }
}

IncastOutcastLink TensorSlotManager::BuildIncastOutcastLink([[maybe_unused]] const std::string& rawname)
{
    IncastOutcastLink link(slotIndexDict.size());

    for (auto& scope : scopeList) {
        Function* tensorFunc = scope->tensorFunc;
        if (!tensorFunc->IsGraphType(GraphType::TILE_GRAPH)) {
            continue;
        }
        link.ioslotDict[tensorFunc] = scope->ioslot;
        for (auto& outcast : scope->ioslot.partialUpdateOutcastList) {
            for (auto& slot : scope->ioslot.outcastSlot[outcast]) {
                partialUpdateSlotIndexSet.insert(slot);
            }
        }
    }

    for (auto& input : inputSlotList) {
        FE_ASSERT(slotIndexDict.count(input) != 0)
            << "TensorSlot[" << input.GetSymbolName() << "] not found in slotIndexDict.";
        link.inputSlotIndexList.push_back(slotIndexDict[input]);
    }
    for (auto& output : outputSlotList) {
        FE_ASSERT(slotIndexDict.count(output) != 0)
            << "TensorSlot[" << output.GetSymbolName() << "] not found in slotIndexDict.";
        link.outputSlotIndexList.push_back(slotIndexDict[output]);
        auto iter = inplaceDict.find(output);
        if (iter != inplaceDict.end()) {
            link.inplaceSlotIndexList.push_back(slotIndexDict[iter->second]);
        } else {
            link.inplaceSlotIndexList.push_back(-1);
        }
    }

    std::unordered_set<std::shared_ptr<TensorSlotScope>> constructAssembleSlotScopeSet;
    for (auto& [slot, index] : slotIndexDict) {
        TensorSlotUsage& usage = GetTensorSlotUsage(slot);
        if (assembleSlotSet.count(slot)) {
            link.assembleSlotIndexList.push_back(index);

            if (usage.construct) {
                std::shared_ptr<TensorSlotScope> scope = usage.construct->GetSlotScope();
                if (scope) {
                    /* Some Tensor might be defined out of FUNCTION. */
                    scope->constructAssembleSlotList.push_back(index);
                    constructAssembleSlotScopeSet.insert(scope);
                }
            }
        }
        if (shmemTensorSlotSet.count(slot)) {
            link.shmemTensorSlotIndexList.push_back(index);
        }
    }
    for (auto scope : constructAssembleSlotScopeSet) {
        std::sort(scope->constructAssembleSlotList.begin(), scope->constructAssembleSlotList.end());
    }

    for (auto& slotIndex : partialUpdateSlotIndexSet) {
        link.partialUpdateSlotIdexList.push_back(slotIndex);
    }

    for (auto& [func, ioslot] : link.ioslotDict) {
        for (size_t idx = 0; idx < func->GetIncast().size(); idx++) {
            if (ioslot.incastSlot[idx].empty()) {
                FE_LOGW("!!! incast[%zu] slot not found, %s", idx, func->GetIncast()[idx]->Dump().c_str());
            }
        }
    }
    UpdateReshapeInplaceSlots(link);
    return link;
}

void TensorSlotManager::SetSameSlot(const Tensor& operand, const Tensor& dst)
{
    TensorSlot slotIn = TensorSlot::CreateTensor(operand);
    TensorSlot slotOut = TensorSlot::CreateTensor(dst);
    FE_ASSERT(outputSlotDict.count(slotOut) != 0)
        << "TensorSlot[" << slotOut.GetSymbolName() << "] not found in outputSlotDict.";
    reshapeInplaceDict[slotIn] = slotOut;
}

bool TensorSlotManager::HasSameSlot(const std::vector<int>& slots1, const std::vector<int>& slots2)
{
    std::unordered_set<int> slotSet(slots2.begin(), slots2.end());
    for (int slot1 : slots1) {
        if (slotSet.count(slot1)) {
            return true;
        }
    }
    return false;
}

} // namespace npu::tile_fwk
