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
 * \file tensor.cpp
 * \brief
 */

#include "tilefwk/tensor.h"
#include "logical_tensor.h"
#include "raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/error.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "interface/utils/id_gen.h"
#include "irbuilder.h"

using namespace npu::tile_fwk;

Tensor::Tensor() : storage_(nullptr), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId())
{
    Program::GetInstance().InsertAliveTensor(this);
}

Tensor::~Tensor()
{
    Program::GetInstance().GetTensorSlotManager()->TensorDestruct(*this);

    Program::GetInstance().EraseAliveTensor(this);
    if (storage_ == nullptr) {
        return;
    }
    FE_ASSERT(FeError::INVALID_PTR, storage_->tensor != nullptr);
    storage_->tensor->AddRefCount(-1);
}

Tensor::Tensor(std::shared_ptr<LogicalTensor> s)
    : storage_(std::move(s)), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId())
{
    FE_ASSERT(FeError::INVALID_PTR, storage_ != nullptr);
    FE_ASSERT(FeError::INVALID_PTR, storage_->tensor != nullptr);
    Program::GetInstance().InsertAliveTensor(this);
    storage_->tensor->AddRefCount(1);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
}

static std::vector<SymbolicScalar> ToDynShape(const std::string& tname, const Shape& shape)
{
    auto dynShape = SymbolicScalar::FromConcrete(shape);
    for (size_t dim = 0; dim < shape.size(); dim++) {
        CHECK(ExternalError::INVALID_VAL, shape[dim] >= -1) << "Invalid shape " << shape[dim];
        if (shape[dim] == -1) {
            auto name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputShapeDim);
            auto handler = SymbolicScalar(AddRuntimePrefix(name));
            auto input = SymbolicScalar(AddArgPrefix(tname));
            dynShape[dim] = handler(input, dim);
        }
    }
    return dynShape;
}

void CheckShapeValid(DataType& dataType, const Shape& shape, TileOpFormat& format)
{
    if (shape.empty() || shape.back() == -1) {
        return;
    }
    static const std::vector<DataType> FP4_TYPES = {DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E1M2X2,
                                                    DataType::DT_FP4_E2M1, DataType::DT_FP4_E1M2};
    bool isB4 = std::find(FP4_TYPES.begin(), FP4_TYPES.end(), dataType) != FP4_TYPES.end();
    if (format == TileOpFormat::TILEOP_NZ) {
        size_t alignSize = isB4 ? ALIGN_SIZE_64 : ALIGN_SIZE_32;
        const bool dataBytesAlign = ((shape.back() * BytesOf(dataType)) % alignSize == 0);
        CHECK(ExternalError::INVALID_SHAPE, dataBytesAlign)
            << "Current inner axis: " << (size_t)shape.back()
            << ", when input is NZ format, inner axis shape must be 32-byte aligned(4bit dtype must be aligned to 64)";
    }
    if (format == TileOpFormat::TILEOP_ND && isB4) {
        const bool inputIsEven = ((shape.back() & 1) == 0);
        CHECK(ExternalError::INVALID_SHAPE, inputIsEven)
            << "Current inner axis: " << (size_t)shape.back()
            << ", when input is ND format and 4bit dtype, inner axis must be even number";
    }
}

Tensor::Tensor(DataType dataType, const Shape& shape, std::string name, TileOpFormat format)
    : index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId())
{
    CheckShapeValid(dataType, shape, format);
    auto dynShape = ToDynShape(name, shape);
    storage_ = std::make_shared<LogicalTensor>(*Program::GetInstance().GetCurrentFunction(), dataType, shape, dynShape,
                                               format, name);
    storage_->tensor->AddRefCount(1);
    storage_->tensor->UpdateDynRawShape(dynShape);
    Program::GetInstance().GetTensorSlotManager()->TensorConstruct(*this);

    Program::GetInstance().InsertAliveTensor(this);
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    Program::GetInstance().GetTensorSlotManager()->TensorSymbol(*this, name);
}

Tensor::Tensor(DataType dataType, std::vector<SymbolicScalar> shape, std::string name, TileOpFormat format)
    : Tensor(dataType, SymbolicScalar::Concrete(shape, -1), name, format)
{
    auto rawTensor = storage_->GetRawTensor();
    for (size_t axis = 0; axis < shape.size(); axis++) {
        if (shape[axis].ConcreteValid() && shape[axis].Concrete() == -1) {
            shape[axis] = rawTensor->GetDynRawShape(axis);
        }
    }
    storage_->UpdateDynValidShape(shape);
    rawTensor->UpdateDynRawShape(shape);
}

const LogicalTensor& Tensor::operator*() const
{
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage_;
}

LogicalTensor& Tensor::operator*()
{
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage_;
}

const std::shared_ptr<LogicalTensor>& Tensor::GetStorage(bool readSlot) const
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage_;
}

std::shared_ptr<LogicalTensor>& Tensor::GetStorage(bool readSlot)
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage_;
}

void Tensor::AssignStorage(const Tensor& rhs, std::shared_ptr<LogicalTensor> newStorage, bool addRef)
{
    if (storage_ != nullptr && storage_->tensor != nullptr) {
        storage_->tensor->AddRefCount(-1);
        newStorage->tensor->symbol = storage_->tensor->symbol;
        newStorage->name_ = IRContext::Get().GetVarName(IRContext::Get().GetOriginName(storage_));
    }
    storage_ = std::move(newStorage);
    if (storage_ != nullptr) {
        if (addRef) {
            storage_->tensor->AddRefCount(1);
        }
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

Tensor& Tensor::operator=(const Tensor& rhs)
{
    if (this == &rhs) {
        return *this;
    }
    AssignStorage(rhs, rhs.GetStorage(), true);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& rhs) noexcept
{
    if (this == &rhs) {
        return *this;
    }
    AssignStorage(rhs, std::move(rhs.GetStorage()), false);
    return *this;
}

Tensor::Tensor(const Tensor& rhs) : storage_(rhs.GetStorage()), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId())
{
    if (storage_ != nullptr) {
        storage_->tensor->AddRefCount(1);
    }

    Program::GetInstance().InsertAliveTensor(this);
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

Tensor::Tensor(Tensor&& rhs)
    : storage_(std::move(rhs.GetStorage())), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId())
{
    Program::GetInstance().InsertAliveTensor(this);
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

DataType Tensor::GetDataType() const { return storage_->Datatype(); }

const Shape& Tensor::GetShape() const { return storage_->shape; }

uint64_t Tensor::Dim() const
{
    if (storage_ != nullptr) {
        return storage_->shape.size();
    } else {
        return 0;
    }
}

bool Tensor::IsEmpty() const { return storage_ == nullptr; }

int32_t Tensor::GetShape(int axis) const
{
    const size_t dimCount = storage_->shape.size();
    FE_ASSERT(FeError::INVALID_VAL, dimCount > 0) << "Tensor has no dimensions! disCount: " << dimCount;
    if (axis < 0) {
        axis += static_cast<int>(dimCount);
    }
    FE_ASSERT(FeError::OUT_OF_RANGE, axis >= 0 && static_cast<size_t>(axis) < dimCount)
        << "Axis index " << axis << " is out of range [0, " << (dimCount - 1) << "].";
    return storage_->shape[axis];
}

std::vector<SymbolicScalar>& Tensor::GetValidShape() const { return storage_->GetDynValidShape(); }

TileOpFormat Tensor::Format() const { return storage_->Format(); }

void Tensor::SetCachePolicy(CachePolicy policy, bool value)
{
    if (storage_ != nullptr) {
        storage_->SetCachePolicy(policy, value);
    }
}

bool Tensor::GetCachePolicy(CachePolicy policy) const
{
    if (storage_ != nullptr) {
        return storage_->GetCachePolicy(policy);
    }
    return false;
}

void Tensor::SetName(const std::string& name) const
{
    if (storage_) {
        storage_->tensor->SetSymbol(name);
    }
}

std::string Tensor::GetName() const { return storage_ ? storage_->tensor->GetSymbol() : ""; }

SymbolicScalar npu::tile_fwk::GetInputShape(const Tensor& t, int n)
{
    auto rawTensor = t.GetStorage(false)->GetRawTensor();
    return rawTensor->GetDynRawShape(n);
}

const std::vector<SymbolicScalar>& npu::tile_fwk::GetInputShape(const Tensor& t)
{
    auto rawTensor = t.GetStorage(false)->GetRawTensor();
    return rawTensor->GetDynRawShape();
}

namespace npu::tile_fwk {
SymbolicScalar GetInputData(const Tensor& t, const std::vector<SymbolicScalar>& offset)
{
    FE_ASSERT(FeError::INVALID_VAL, t.Dim() == offset.size())
        << "t.Dim(): " << t.Dim() << "!= offset.size(): " << offset.size();
    FE_ASSERT(FeError::INVALID_VAL, t.Dim() > 0 && t.Dim() <= 0x4) << "t.Dim(): " << t.Dim() << ", limit: [1, 4]";
    FE_ASSERT(FeError::INVALID_VAL, IsInteger(t.GetDataType()))
        << "Not Int type, got: " << DataType2String(t.GetDataType());

    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string handlerName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputData);
    handlerName = AddRuntimePrefix(handlerName);
    SymbolicScalar getInputData(handlerName);

    int inputIndex = slotManager->GetInputIndex(t);
    std::string inputName = slotManager->GetInputNameList()[inputIndex];
    inputName = AddArgPrefix(inputName);
    SymbolicScalar input(inputName);
    SymbolicScalar type(AddRuntimePrefix(DataType2CCEStr(t.GetDataType())));

    std::vector<SymbolicScalar> argList = {input, type};
    for (const auto& off : offset) {
        argList.push_back(off);
    }
    return getInputData(argList);
}

static SymbolicScalar DoGetTensorDataInt32(SymbolHandlerId handlerId, const Tensor& t,
                                           const std::vector<SymbolicScalar>& offset)
{
    FE_ASSERT(FeError::INVALID_VAL, t.GetShape().size() == offset.size())
        << "Mismatch dimension: " << t.GetShape().size() << " vs " << offset.size() << "\n";
    Program::GetInstance().GetTensorSlotManager()->TensorRead(t);

    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    FE_ASSERT(FeError::INVALID_OPERATION, currDynFunc != nullptr) << "Not under dynamic function!\n";

    auto assemble = std::make_shared<Tensor>(TensorExtract(t, offset));
    auto builder = IRBuilder();
    int getTensorDataIndex = builder.AddTensorDataDesc(assemble);

    auto emuopAssemble = *assemble->GetStorage()->GetProducers().begin();
    auto emuopMark = *emuopAssemble->GetIOperands()[0]->GetProducers().begin();
    auto emuopView = *emuopMark->GetIOperands()[0]->GetProducers().begin();
    GetTensorDataSetIndex(emuopView, getTensorDataIndex);
    GetTensorDataSetIndex(emuopMark, getTensorDataIndex);
    GetTensorDataSetIndex(emuopAssemble, getTensorDataIndex);

    std::string getName = SymbolHandler::GetNameByHandlerId(handlerId);
    std::string getRuntimeName = AddRuntimePrefix(getName);
    SymbolicScalar getRuntimeHandler(getRuntimeName);
    std::vector<SymbolicScalar> argList = {getTensorDataIndex, -1, -1, -1};
    argList.insert(argList.end(), offset.begin(), offset.end());
    auto scalar = getRuntimeHandler(argList);

    emuopAssemble->result_token_ = {builder.CreateTokenVar(emuopAssemble->GetSpan())};
    builder.AddDependToken(scalar, emuopAssemble->result_token_.front());
    return scalar;
}

static std::vector<std::reference_wrapper<const Tensor>>::iterator FindTensor(
    const Tensor& key, std::vector<std::reference_wrapper<const Tensor>>& vec)
{
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (&key == &(it->get())) {
            return it;
        }
    }
    return vec.end();
}

constexpr int MAX_GET_TENSOR_DATA_DIM = 4;
SymbolicScalar GetTensorData(const Tensor& t, const std::vector<SymbolicScalar>& offset)
{
    auto funcPtr = Program::GetInstance().GetCurrentDynamicFunction();
    if (funcPtr) {
        auto inputTensorList = funcPtr->GetDyndevAttribute()->startArgsInputTensorList;
        if (FindTensor(t, inputTensorList) != inputTensorList.end()) {
            FE_LOGD("Tensor[%s] already exists in inputTensorList", t.GetName().c_str());
            return GetInputData(t, offset);
        }
    }
    CHECK(ExternalError::INVALID_TYPE, t.GetDataType() == DT_INT32) << "Tensor dtype must be DT_INT32.";
    CHECK(ExternalError::OUT_OF_RANGE, offset.size() <= MAX_GET_TENSOR_DATA_DIM)
        << "Offset.size() must be less than " << MAX_GET_TENSOR_DATA_DIM;
    SymbolHandlerId handlerId = static_cast<SymbolHandlerId>(static_cast<int>(SymbolHandlerId::GetTensorDataInt32Dim1) +
                                                             offset.size() - 1);
    return DoGetTensorDataInt32(handlerId, t, offset);
}

void TensorMove(Tensor& src, Tensor& dst)
{
    auto func = Program::GetInstance().GetCurrentDynamicFunction();
    if (func) {
        auto inputs = func->GetDyndevAttribute()->startArgsInputTensorList;
        if (FindTensor(dst, inputs) != inputs.end()) {
            auto offsets = std::vector<SymbolicScalar>(src.Dim(), 0);
            Assemble(src, offsets, dst);
            return;
        }
    }
    dst = std::move(src);
}

static void DoSetTensorDataInt32(const SymbolicScalar& v, const std::vector<SymbolicScalar>& off, Tensor& t)
{
    CHECK(ExternalError::INVALID_VAL, t.GetShape().size() == off.size())
        << "Mismatch dimen:" << t.GetShape().size() << " vs " << off.size() << "\n";
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(t);

    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    FE_ASSERT(FeError::INVALID_OPERATION, currDynFunc != nullptr) << "Not under dynamic function!\n";

    Shape vShape = Shape(t.GetShape().size(), 1);
    auto tmp = Full(v, t.GetDataType(), vShape);
    TensorInsert(tmp, off, t);
}

void SetTensorData(const SymbolicScalar& v, const std::vector<SymbolicScalar>& off, Tensor& dst)
{
    CHECK(ExternalError::INVALID_TYPE, dst.GetDataType() == DT_INT32) << "Tensor dtype must be DT_INT32.";
    FE_LOGD("Set tensor[%s] data.", dst.GetName().c_str());
    return DoSetTensorDataInt32(v, off, dst);
}

} // namespace npu::tile_fwk

SymbolicScalar npu::tile_fwk::IsLoopBegin(const SymbolicScalar& symbol, const SymbolicScalar& begin)
{
    std::string isLoopBeginName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin);
    isLoopBeginName = AddRuntimePrefix(isLoopBeginName);
    SymbolicScalar isLoopBegin(isLoopBeginName);
    auto result = isLoopBegin(symbol, begin);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}

SymbolicScalar npu::tile_fwk::IsLoopEnd(const SymbolicScalar& symbol, const SymbolicScalar& end)
{
    std::string isLoopEndName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd);
    isLoopEndName = AddRuntimePrefix(isLoopEndName);
    SymbolicScalar isLoopEnd(isLoopEndName);
    auto result = isLoopEnd(symbol, end);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}
