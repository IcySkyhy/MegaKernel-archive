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
 * \file shmem_operation_impl.cpp
 * \brief
 */

#include <type_traits>
#include "distributed_common.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::Distributed {

void ValidateGroup(const char* group)
{
    ASSERT(DistributedErrorCode::INVALID_GROUP_NAME, group != nullptr) << "\"group\" cannot be nullptr";
    auto groupLen = std::string(group).size();
    ASSERT(DistributedErrorCode::INVALID_GROUP_NAME, (groupLen >= 1) && (groupLen < MAX_GROUP_NAME_LENGTH))
        << "The length of \"group\" only supports [1, " << MAX_GROUP_NAME_LENGTH << "), but got " << groupLen;
}

void ValidateTiling(const Opcode& opCode, const Tensor& target, const std::string& desc, bool isData = false)
{
    const auto vecTile = TileShape::Current().GetVecTile();
    ASSERT(DistributedErrorCode::INVALID_TILE_SHAPE, vecTile.valid())
        << ToString(opCode)
        << ": vecTile should not empty, and all value should > 0, but got:" << ToString(vecTile.tile);
    if (isData) {
        ASSERT(DistributedErrorCode::INVALID_TILE_DIM, target.Dim() == vecTile.size())
            << ToString(opCode) << " dim of vectile shape must be equal to " << std::to_string(target.Dim())
            << ", which is same as " << desc << ", but got " << vecTile.size();
    }
}

void ValidateDataType(const Tensor& tensor, const std::string& desc, const std::unordered_set<DataType>& allowedTypes,
                      const std::string& context = "")
{
    auto dataType = tensor.GetDataType();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DTYPE, allowedTypes.empty() || allowedTypes.count(dataType))
        << "Invalid data type: " << desc << " data type must be " << ToString(allowedTypes)
        << ", but got:" << ToString(dataType) << (context.empty() ? "" : ". " + context);
}

void ValidateDim(const Shape& shape, const std::string& desc, const std::set<size_t>& allowedDims)
{
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, allowedDims.empty() || allowedDims.count(shape.size()))
        << "Invalid dimensional: " << desc << " dimensional must be " << ToString(allowedDims)
        << ", but got dimensional=" << shape.size();
}

void ValidateFormat(const Tensor& tensor, const std::string& desc,
                    const std::unordered_set<TileOpFormat>& allowedFormats = {TileOpFormat::TILEOP_ND})
{
    ASSERT(DistributedErrorCode::INVALID_TENSOR_FORMAT, allowedFormats.empty() || allowedFormats.count(tensor.Format()))
        << "Invalid format: " << desc << " only support ND format, but got NZ format";
}

void ValidateShape(const Tensor& tensor, const std::string& desc, const Shape& expectShape)
{
    const auto& shape = tensor.GetShape();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_SHAPE,
           std::all_of(shape.begin(), shape.end(), [](int64_t val) { return val > 0; }))
        << "Invaild shape value: " << desc << ", all value must be greater than 0, but got " << ToString(shape);
    ASSERT(DistributedErrorCode::INVALID_TENSOR_SHAPE, expectShape.empty() || expectShape == shape)
        << "Invalid shape: " << desc << " expect:" << ToString(expectShape) << ", but got: " << ToString(shape);
}

void ValidateTensor(const Tensor& tensor, const std::string& desc, const std::set<size_t>& allowedDims = {},
                    const std::unordered_set<DataType>& allowedTypes = {},
                    const std::unordered_set<TileOpFormat>& allowedFormats = {}, const Shape& expectShape = {},
                    const std::string& dataTypeContext = "")
{
    ValidateDim(tensor.GetShape(), desc, allowedDims);
    ValidateDataType(tensor, desc, allowedTypes, dataTypeContext);
    ValidateFormat(tensor, desc, allowedFormats);
    ValidateShape(tensor, desc, expectShape);
}

void ValidateOpType(OpType cmp, const std::unordered_set<OpType>& allowedOpTypes)
{
    ASSERT(DistributedErrorCode::INVALID_OP_TYPE, allowedOpTypes.empty() || allowedOpTypes.count(cmp))
        << "Invaild OP type, only support:" << ToString(allowedOpTypes) << ", but got:" << ToString(cmp);
}

void ValidateShmemTensor(const ShmemTensor& t, bool hasData = false, bool hasSignal = false)
{
    static std::unordered_map<std::string, int64_t> groupWorldSizeMap;
    ValidateGroup(t.group.c_str());
    auto groupWorldSize = groupWorldSizeMap.find(t.group);
    if (groupWorldSize == groupWorldSizeMap.end()) {
        ASSERT(DistributedErrorCode::INVALID_WORLD_SIZE, t.worldSize > 0)
            << "Invalid world size for group " << t.group << ": world size must be greather than 0"
            << ", but got " << t.worldSize;
        groupWorldSizeMap.emplace(t.group, t.worldSize);
    } else {
        ASSERT(DistributedErrorCode::INVALID_WORLD_SIZE, t.worldSize == groupWorldSize->second)
            << "WorldSize mismatch for group " << t.group << ": expected " << groupWorldSize->second << ", but got "
            << t.worldSize;
    }
    if (hasData) {
        ASSERT(DistributedErrorCode::INVALID_SHMEM_TENSOR, t.data.GetStorage() != nullptr)
            << "shmem tensor's data should not be empty";
    }
    if (hasSignal) {
        ASSERT(DistributedErrorCode::INVALID_SHMEM_TENSOR, t.signal.GetStorage() != nullptr)
            << "shmem tensor's signal should not be empty";
    }
}

static uint64_t GetSignalBufferSize(const ShmemTensor& t, uint64_t maxTileNum)
{
    ASSERT(DistributedErrorCode::INVALID_SHMEM_TENSOR, t.signal.GetStorage() != nullptr)
        << "shmem tensor's signal should not be empty";
    return AlignUp(BytesOf(t.signal.GetDataType()) * t.worldSize * SHMEM_SIGNAL_STRIDE * maxTileNum, SHMEM_SIZE_ALIGN);
}

static void CreateShmemSignalImpl(ShmemTensor& t, const Shape& shape)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(t.group));
    Shape signalShape{t.worldSize};
    signalShape.insert(signalShape.end(), shape.begin(), shape.end());
    auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
    t.signal = signalInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
    auto& signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
    int64_t maxTileNum = 1;
    signalOp.SetAttribute(OpAttributeKey::bindTensor,
                          BindTensor(hcclGroupIndex, 1, GetSignalBufferSize(t, maxTileNum), maxTileNum));
    signalOp.SetAttribute(OpAttributeKey::maxTileNum, maxTileNum);
    t.signalOp = &signalOp;
}

ShmemTensor CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape)
{
    ShmemTensor t;
    static uint64_t s_index = 0;
    LOOP("CreateShmemTensor" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
    {
        (void)index;
        CreateShmemTensor(group, worldSize, dataType, shape, t);
    }
    return t;
}

void CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor& t)
{
    ValidateGroup(group);
    ValidateDim(shape, "shmem Tensor", {2, 3, 4});
    t.group = std::string(group);
    t.worldSize = worldSize;
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape dataShape = shape;
    auto dataInner = std::make_shared<LogicalTensor>(function, dataType, dataShape);
    t.data = dataInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.data, SlotProperty::SHMEM_TENSOR);
    auto& dataOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {dataInner});
    dataOp.SetAttribute(OpAttributeKey::bindTensor,
                        BindTensor(hcclGroupIndex, 0,
                                   AlignUp(BytesOf(dataType) * std::accumulate(dataShape.begin(), dataShape.end(), 1,
                                                                               std::multiplies<int64_t>()),
                                           SHMEM_SIZE_ALIGN)));

    CreateShmemSignalImpl(t, shape);

    ValidateShmemTensor(t, true, true);
}

ShmemTensor CreateShmemSignal(const char* group, int64_t worldSize)
{
    ShmemTensor t;
    static uint64_t s_index = 0;
    LOOP("CreateShmemSignal" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
    {
        (void)index;
        CreateShmemSignal(group, worldSize, t);
    }
    return t;
}

void CreateShmemSignal(const char* group, int64_t worldSize, ShmemTensor& t)
{
    ValidateGroup(group);
    t.group = std::string(group);
    t.worldSize = worldSize;
    CreateShmemSignalImpl(t, {1, SHMEM_SIGNAL_STRIDE});
    ValidateShmemTensor(t, false, true);
}

template <typename OffsetType, bool HasValidShape = false>
ShmemTensor ShmemViewImpl(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                          const std::vector<OffsetType>& offsets, const std::vector<SymbolicScalar>& validShapes = {})
{
    ASSERT(DistributedErrorCode::INVALID_SHMEM_VIEW_PARAM, operand.data.GetStorage() != nullptr)
        << "shmem tensor which has no valid data not support view";

    auto data = [&]() {
        if constexpr (HasValidShape) {
            return View(operand.data, shapes, validShapes, offsets);
        } else {
            return View(operand.data, shapes, offsets);
        }
    }();
    ASSERT(DistributedErrorCode::INVALID_SHMEM_VIEW_PARAM, operand.data.Dim() == shapes.size())
        << "input shape dim should be equal to shmem data dim, input shape dim:" << shapes.size()
        << ", shmem data dim:" << operand.data.Dim();
    ASSERT(DistributedErrorCode::INVALID_SHMEM_VIEW_PARAM, operand.data.Dim() == offsets.size())
        << "input offsets dim should be equal to shmem data dim, input offsets dim:" << offsets.size()
        << ", shmem data dim:" << operand.data.Dim();

    Shape signalShape = operand.signal.GetShape();
    std::copy(shapes.begin(), shapes.end(), signalShape.end() - shapes.size());
    std::vector<OffsetType> signalOffset(operand.signal.GetShape().size(), 0);
    std::copy(offsets.begin(), offsets.end(), signalOffset.end() - offsets.size());
    auto signal = View(operand.signal, signalShape, signalOffset);
    return ShmemTensor{operand.group, operand.worldSize, data, signal, operand.signalOp};
}

ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<int64_t>& offsets)
{
    return ShmemViewImpl<int64_t>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<SymbolicScalar>& offsets)
{
    return ShmemViewImpl<SymbolicScalar>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<SymbolicScalar>& newValidShapes, const std::vector<SymbolicScalar>& newOffsets)
{
    return ShmemViewImpl<SymbolicScalar, true>(operand, shapes, newOffsets, newValidShapes);
}

ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::initializer_list<SymbolicScalar>& newOffsets)
{
    return ShmemView(operand, shapes, std::vector<SymbolicScalar>(newOffsets));
}

static Tensor ShmemPutImpl(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp,
                           const Tensor& pred, bool isStore)
{
    ValidateShmemTensor(dst, true);
    std::unordered_set<DataType> allowedTypes = {DT_INT32, DT_FP32, DT_FP16, DT_BF16};
    ValidateTensor(src, "local tensor", {2, 3, 4}, allowedTypes, {TileOpFormat::TILEOP_ND});
    std::unordered_set<DataType> allowedShmemTypes = {src.GetDataType()};
    if ((putOp == AtomicType::ADD) && ((src.GetDataType() == DT_BF16) || (src.GetDataType() == DT_FP16))) {
        allowedShmemTypes.emplace(DT_FP32);
    }
    ValidateTensor(dst.data, "data of shmem tensor", {}, allowedShmemTypes, {TileOpFormat::TILEOP_ND}, src.GetShape(),
                   "Shmem tensor dtype must match input tensor dtype");
    ValidateTensor(pred, "pred tensor", {2, 3, 4});
    ValidateTiling(isStore ? Opcode::OP_SHMEM_STORE : Opcode::OP_SHMEM_PUT, src, "src");
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, src.GetShape());
    auto& op = isStore ? function.AddOperation(Opcode::OP_SHMEM_STORE,
                                               {src.GetStorage(), dst.data.GetStorage(), pred.GetStorage()}, {out}) :
                         function.AddOperation(Opcode::OP_SHMEM_PUT,
                                               {pred.GetStorage(), src.GetStorage(), dst.data.GetStorage()}, {out});
    if (src.GetValidShape().size() == 0) {
        src.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(src.GetShape()));
    }
    MemoryType fromType = isStore ? MemoryType::MEM_UB : MemoryType::MEM_DEVICE_DDR;
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        fromType, OpImmediate::Specified({0, 0}), OpImmediate::Specified({src.GetShape()}),
        OpImmediate::Specified({src.GetShape()}), OpImmediate::Specified(src.GetValidShape())));
    op.SetAttr(OpAttributeKey::isDistCopyOut, true);

    ShmemPutAttr distOpAttr;
    distOpAttr.atomicType = putOp;
    distOpAttr.ownerRank = dstRank;
    distOpAttr.group = dst.group;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemPut(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp,
                const Tensor& pred)
{
    return ShmemPutImpl(src, dst, dstRank, putOp, pred, false);
}

Tensor ShmemStore(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp,
                  const Tensor& pred)
{
    return ShmemPutImpl(src, dst, dstRank, putOp, pred, true);
}

Tensor ShmemGet(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred, DataType targetDataType)
{
    ValidateShmemTensor(src, true);
    ValidateTensor(src.data, "data of shmem tensor", {2, 3, 4}, {DT_INT32, DT_FP32, DT_FP16, DT_BF16},
                   {TileOpFormat::TILEOP_ND});
    ValidateTensor(pred, "pred tensor", {2, 3, 4});
    ValidateTiling(Opcode::OP_SHMEM_GET, src.data, "src", true);
    if (targetDataType == DT_BOTTOM) {
        targetDataType = src.data.GetDataType();
    }
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, targetDataType, src.data.GetShape(), src.data.Format());
    auto& op = function.AddOperation(Opcode::OP_SHMEM_GET, {pred.GetStorage(), src.data.GetStorage()}, {out});
    if (src.data.GetValidShape().size() == 0) {
        src.data.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(src.data.GetShape()));
    }
    out->UpdateDynValidShape(src.data.GetValidShape());
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 0}), OpImmediate::Specified(src.data.GetShape()),
        OpImmediate::Specified(src.data.GetShape()), OpImmediate::Specified(src.data.GetValidShape())));

    ShmemGetAttr distOpAttr;
    distOpAttr.ownerRank = srcRank;
    distOpAttr.group = src.group;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    op.SetAttr(OpAttributeKey::isDistCopyOut, true);
    return out;
}

Tensor ShmemLoad(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred, DataType nonShmemDataType)
{
    ValidateShmemTensor(src, true);
    ValidateTensor(src.data, "data of shmem tensor", {2, 3, 4}, {DT_INT32, DT_FP32, DT_FP16, DT_BF16},
                   {TileOpFormat::TILEOP_ND});
    ValidateTensor(pred, "pred tensor", {2, 3, 4});
    ValidateTiling(Opcode::OP_SHMEM_LOAD, src.data, "src", true);
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = src.data.GetDataType();
    }
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, nonShmemDataType, src.data.GetShape());
    auto& op = function.AddOperation(Opcode::OP_SHMEM_LOAD, {pred.GetStorage(), src.data.GetStorage()}, {out});
    if (src.data.GetValidShape().size() == 0) {
        src.data.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(src.data.GetShape()));
    }
    out->UpdateDynValidShape(src.data.GetValidShape());
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, OpImmediate::Specified(src.data.GetShape()),
        OpImmediate::Specified(out->shape), OpImmediate::Specified(src.data.GetValidShape())));

    ShmemGetAttr distOpAttr;
    distOpAttr.ownerRank = srcRank;
    distOpAttr.group = src.group;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    op.SetAttr(OpAttributeKey::isDistCopyOut, false);
    return out;
}

static void UpdataSignalMaxTile(const ShmemTensor& src, ShmemSignalAttr& distOpAttr)
{
    const auto& vecTile = TileShape::Current().GetVecTile();
    auto [totalTileNum, viewTileNum, viewshapes, viewTileStrides, viewIndexStrides] = GetTotalTileNum(vecTile, src);
    (void)viewTileStrides;
    (void)viewIndexStrides;
    distOpAttr.viewshapes = viewshapes;
    distOpAttr.viewTileNum = viewTileNum;
    distOpAttr.totalTileNum = totalTileNum;

    int64_t cur = ((Operation*)src.signalOp)->GetIntAttribute(OpAttributeKey::maxTileNum);
    if (totalTileNum > cur) {
        auto hcclGroupIndex = static_cast<uint64_t>(CommGroupRecorder::GetInstance().Input(src.group));
        ((Operation*)src.signalOp)
            ->SetAttribute(OpAttributeKey::bindTensor,
                           BindTensor(hcclGroupIndex, 1, GetSignalBufferSize(src, totalTileNum), totalTileNum));
        ((Operation*)src.signalOp)->SetAttribute(OpAttributeKey::maxTileNum, totalTileNum);
    }
}

static Tensor ShmemSignalImpl(const ShmemTensor& src, const SymbolicScalar& srcRank, const SymbolicScalar& targetRank,
                              int32_t signal, AtomicType sigOp, const Tensor& pred, bool notifyAll = false)
{
    ValidateShmemTensor(src, false, true);
    ValidateTensor(pred, "pred tensor", {2, 3, 4});
    ValidateTensor(src.signal, "signal of shmem tensor", {3, 4, 5});
    ValidateTiling(Opcode::OP_SHMEM_SIGNAL, src.signal, "src");
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = src.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = srcRank;
    auto signalTensor = View(src.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {pred.GetStorage(), signalTensor.GetStorage()}, {out});
    op.SetAttr(OpAttributeKey::dontTouch, true);
    ShmemSignalAttr distOpAttr;
    distOpAttr.group = src.group;
    distOpAttr.signalValue = signal;
    distOpAttr.atomicType = sigOp;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.notifyAll = notifyAll;
    distOpAttr.worldSize = src.worldSize;
    distOpAttr.ownerRank = targetRank;

    UpdataSignalMaxTile(src, distOpAttr);

    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);

    return out;
}

Tensor ShmemSignal(const ShmemTensor& src, const SymbolicScalar& srcRank, const SymbolicScalar& targetRank,
                   int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    return ShmemSignalImpl(src, srcRank, targetRank, signal, sigOp, pred);
}

Tensor ShmemSignalAll(const ShmemTensor& src, const SymbolicScalar& srcRank, int32_t signal, AtomicType sigOp,
                      const Tensor& pred)
{
    return ShmemSignalImpl(src, srcRank, 0, signal, sigOp, pred, true);
}

Tensor ShmemWaitUntil(const ShmemTensor& src, const SymbolicScalar& srcRank, OpType cmp, int32_t cmpValue,
                      bool clearSignal, const Tensor& pred)
{
    ValidateOpType(cmp, {OpType::EQ});
    ValidateShmemTensor(src, false, true);
    ValidateTensor(pred, "pred tensor", {2, 3, 4});
    ValidateTensor(src.signal, "signal of shmem tensor", {3, 4, 5});
    ValidateTiling(Opcode::OP_SHMEM_WAIT_UNTIL, src.signal, "src");
    (void)cmp;
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = src.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = srcRank;
    auto signalTensor = View(src.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto& op = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {pred.GetStorage(), signalTensor.GetStorage()},
                                     {out});
    ShmemWaitUntilAttr distOpAttr;
    distOpAttr.group = src.group;
    distOpAttr.expectedSum = cmpValue;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.resetSignal = clearSignal;
    distOpAttr.ownerRank = GetHcclRankId(src.group);

    const auto& vecTile = TileShape::Current().GetVecTile();
    auto [totalTileNum, viewTileNum, viewshapes, viewTileStrides, viewIndexStrides] = GetTotalTileNum(vecTile, src);
    distOpAttr.viewshapes = viewshapes;
    distOpAttr.viewTileStrides = viewTileStrides;
    distOpAttr.viewIndexStrides = viewIndexStrides;
    distOpAttr.viewTileNum = viewTileNum;
    distOpAttr.totalTileNum = totalTileNum;

    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

static Tensor ShmemClearImpl(const ShmemTensor& src, Tensor& pred, bool clearData)
{
    if (clearData) {
        ValidateShmemTensor(src, true);
        ValidateTensor(src.data, "data of shmem tensor", {2, 3, 4});
    } else {
        ValidateShmemTensor(src, false, true);
        ValidateTensor(src.signal, "signal of shmem tensor", {3, 4, 5});
    }
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(
        Opcode::OP_SHMEM_SET, {pred.GetStorage(), clearData ? src.data.GetStorage() : src.signal.GetStorage()}, {out});
    ShmemSetAttr distOpAttr;
    distOpAttr.group = src.group;
    distOpAttr.isSetData = clearData;
    distOpAttr.ownerRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemClearData(const ShmemTensor& src, Tensor& pred) { return ShmemClearImpl(src, pred, true); }

Tensor ShmemClearSignal(const ShmemTensor& src, Tensor& pred) { return ShmemClearImpl(src, pred, false); }

Tensor ShmemBarrier(const ShmemTensor& src, const Tensor& pred)
{
    ShmemSignalAll(src, 0, 1, AtomicType::ADD, pred);
    return ShmemWaitUntil(src, 0, OpType::EQ, src.worldSize, true, pred);
}

void AllGather(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "pred tensor", {2});
    ValidateTensor(in, "input tensor", {2});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar validRow = in.GetValidShape()[0];
    SymbolicScalar validCol = in.GetValidShape()[1];
    ValidateTensor(shmemTensor.data, "data of shmem tensor", {}, {in.GetDataType()}, {in.Format()},
                   {worldSize * row, col});
    ValidateTensor(out, "output tensor", {}, {in.GetDataType()}, {in.Format()}, {row * worldSize, col});
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {row, col}, std::vector<SymbolicScalar>{thisRank * row, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::SET, predToken);
        auto shmemSignalOut = ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::SET, shmemPutOut);
        auto shmemDataLocal = ShmemView(shmemTensor, {row, col}, std::vector<SymbolicScalar>{validRow, validCol},
                                        std::vector<SymbolicScalar>{dynRankId * row, 0});
        auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, 1, true, shmemSignalOut);
        auto shmemGetOut = ShmemGet(shmemDataLocal, thisRank, waitUntilOut);
        Assemble(shmemGetOut, {dynRankId * validRow, 0}, out);
    }
}

void ReduceScatter(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, DistReduceType reduceType,
                   Tensor& out)
{
    (void)reduceType;
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "pred tensor", {2});
    ValidateTensor(in, "input tensor", {2});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowOut = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "data of shmem tensor", {}, {}, {in.Format()}, {rowOut, col});
    ValidateTensor(out, "output tensor", {}, {in.GetDataType()}, {in.Format()}, {rowOut, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {rowOut, col}, std::vector<SymbolicScalar>{0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto shmemPutOut = ShmemPut(inTile, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {rowOut, col}, std::vector<SymbolicScalar>{0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, predToken);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void OneShotAllReduce(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "pred tensor", {2});
    ValidateTensor(in, "input tensor", {2});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "data of shmem tensor", {}, {}, {in.Format()}, {row, col});
    ValidateTensor(out, "output tensor", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {row, col}, std::vector<SymbolicScalar>{0, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {row, col}, in.GetValidShape(), std::vector<SymbolicScalar>{0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "pred tensor", {2});
    ValidateTensor(in, "input tensor", {2});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "data of shmem tensor", {}, {}, {in.Format()}, {rowPerRank, col});
    ValidateTensor(out, "output tensor", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {rowPerRank, col}, std::vector<SymbolicScalar>{0, 0});
        auto inTile = View(in, {rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId * rowPerRank, 0});
        auto shmemPutOut = ShmemPut(inTile, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignalAll(shmemDataTile, dynRankId, 1, AtomicType::ADD, shmemPutOut);
        auto waitUntilOut = ShmemWaitUntil(shmemDataTile, dynRankId, OpType::EQ, worldSize, true, predToken);
        auto tmp = ShmemGet(shmemDataTile, dynRankId, waitUntilOut, in.GetDataType());
        Assemble(tmp, {rowPerRank * dynRankId, 0}, out);
    }
}
} // namespace npu::tile_fwk::Distributed
