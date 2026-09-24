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
 * \file dev_encode_tensor.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"

namespace npu::tile_fwk {
class Storage;
}

namespace npu::tile_fwk::dynamic {
struct EncodeRawTensorAttr {
    std::shared_ptr<Storage> storage;
    uint64_t storageOffset = 0;
};

struct DevAscendStride {
    int64_t dimSize;
    uint64_t dimStride[DEV_SHAPE_DIM_MAX];

    const uint64_t& operator[](int index) const { return dimStride[index]; }
    uint64_t& operator[](int index) { return dimStride[index]; }

    int GetShape(int dim) const
    {
        if (dim == dimSize - 1) {
            return dimStride[dim];
        } else {
            return dimStride[dim + 1] != 0 ? dimStride[dim] / dimStride[dim + 1] : 0;
        }
    }

    void SetShape(const int* shape, int dim)
    {
        /* For shape [d0, d1, d2], the stride is [d0 * d1 * d2, d1 * d2, d2] */
        dimSize = dim;
        for (int i = DEV_SHAPE_DIM_MAX - 1; i >= 0; i--) {
            if (i > dimSize - 1) {
                dimStride[i] = 0;
            } else if (i == dimSize - 1) {
                dimStride[i] = shape[i];
            } else {
                dimStride[i] = shape[i] * dimStride[i + 1];
            }
        }
    }
    void SetShape(const std::vector<int>& shape) { SetShape(shape.data(), (int)shape.size()); }
    void SetShape(const DevShape& shape) { SetShape(shape.dim, shape.dimSize); }
};

static inline std::string DumpStride(const DevAscendStride& stride)
{
    std::ostringstream oss;
    oss << "<";
    for (int k = 0; k < stride.dimSize; k++) {
        oss << Delim(k != 0, ",") << stride.dimStride[k];
    }
    oss << ">";
    return oss.str();
}

struct DevCellMatchTableDesc {
    DevShape cellShape;
    DevAscendStride stride;

    // Index corresponds to CellMatchOpType enum: NORMAL_WRITE=0, ATOMIC_WRITE=1, READ=2
    static constexpr uint32_t CELL_MATCH_OP_TYPE_MAX_NUM = 3;
    uint32_t cacheOpMaxCount[CELL_MATCH_OP_TYPE_MAX_NUM]{1, 0, 0};
    uint32_t opMemLayOutIndex[CELL_MATCH_OP_TYPE_MAX_NUM]{};
    uint32_t cellUint64Size{2};

    int GetDimensionSize() const { return cellShape.dimSize; }

    const int& GetCellShape(int index) const { return cellShape.dim[index]; }

    const uint64_t& GetStride(int index) const { return stride.dimStride[index]; }
    int GetStrideShape(int index) const { return stride.GetShape(index); }

    void SetCellShape(const std::vector<int>& shape)
    {
        cellShape.dimSize = shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            cellShape.dim[i] = shape[i];
        }
    }
    void SetStrideShape(const std::vector<int>& shape) { stride.SetShape(shape); }

    uint32_t GetCacheOpMaxCount(uint32_t opType) const { return cacheOpMaxCount[opType]; }

    void SetCacheOpMaxCount(const std::vector<uint32_t>& counts)
    {
        for (uint32_t i = 0; i < CELL_MATCH_OP_TYPE_MAX_NUM; i++) {
            cacheOpMaxCount[i] = counts[i];
        }
        UpdateCellMemLayOut();
    }

    void UpdateCellMemLayOut()
    {
        uint32_t offset = 1;
        for (uint32_t i = 0; i < CELL_MATCH_OP_TYPE_MAX_NUM; i++) {
            opMemLayOutIndex[i] = offset;
            offset += cacheOpMaxCount[i];
        }
        cellUint64Size = offset;
    }

    bool MaybeHaveAtomic() const
    {
        return GetCacheOpMaxCount(1) > 0; // ATOMIC_WRITE
    }
};

static inline std::string DumpMemLayOut(const DevCellMatchTableDesc& tableDesc)
{
    std::ostringstream oss;
    oss << " memLayout = [" << tableDesc.cellUint64Size << "]{";
    for (uint32_t k = 0; k < DevCellMatchTableDesc::CELL_MATCH_OP_TYPE_MAX_NUM; k++) {
        oss << Delim(k != 0, ",") << tableDesc.cacheOpMaxCount[k];
    }
    oss << "}";
    return oss.str();
}

static inline std::string DumpCellMatchTableDesc(const DevCellMatchTableDesc& desc)
{
    return DumpShape(desc.cellShape) + " x " + DumpStride(desc.stride) + DumpMemLayOut(desc);
}

struct DevAscendProgram;

struct DevDynamicCellMatchStridePatch {
    uint64_t descOffset{0};
    DevAscendStride stride{};
};

inline void ApplyDynamicCellMatchDescPatchesFromLaunchArgs(DevAscendProgram* devProg, int64_t* inputs)
{
    if (devProg == nullptr || inputs == nullptr) {
        return;
    }
    const uint64_t inputCount = static_cast<uint64_t>(inputs[0]);
    const uint64_t outputCount = static_cast<uint64_t>(inputs[1]);
    auto* patchCountPtr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(inputs + DEV_TENSOR_DATA_OFFSET) + inputCount + outputCount);
    const uint64_t patchCount = *patchCountPtr;
    if (patchCount == 0) {
        return;
    }
    auto* patches = reinterpret_cast<DevDynamicCellMatchStridePatch*>(patchCountPtr + 1);
    auto* devProgBytes = reinterpret_cast<uint8_t*>(devProg);
    for (uint64_t i = 0; i < patchCount; ++i) {
        auto* dstDesc = reinterpret_cast<DevCellMatchTableDesc*>(devProgBytes + patches[i].descOffset);
        dstDesc->stride = patches[i].stride;
    }
}

struct DevSymShape {
    int dimSize{0};
    SymInt dim[DEV_SHAPE_DIM_MAX];

    void SetShape(const std::vector<SymInt>& shape)
    {
        for (std::size_t i = 0; i < shape.size(); i++) {
            dim[i] = shape[i];
        }
        dimSize = static_cast<int>(shape.size());
    }

    uint64_t At(size_t idx, const uint64_t* exprTbl) const
    {
        if (dim[idx].IsExpression())
            return exprTbl[dim[idx].Value()];
        else
            return dim[idx].Value();
    }

    void ToStride(uint64_t* stides, const uint64_t* exprTbl) const
    {
        stides[dimSize - 1] = 1;
        for (int i = dimSize - 1; i > 0; i--) {
            stides[i - 1] = stides[i] * At(i, exprTbl);
        }
    }
};

struct DevAscendRawTensor {
    // Offset in DevAscendFunction (root outcasts & non i/o raw tensors, separately recorded)
    uint64_t addrOffset{UINT64_MAX};
    uint64_t memoryRequirement; // Only available for incast/outcast
                                // For workspace tensors, the memoryRequirement property is deprecated
    uint64_t maxStaticMemReq;   // 0 if cannot find a non-symbolic raw shape
    DataType dataType;
    DevSymShape shape;
    DevIOProperty ioProperty{DevIOProperty::NONE};
    int32_t ioIndex;
    int32_t linkedIncastId; // outcast shared same addr with incast
    int rawMagic;

    int GetDim() const { return shape.dimSize; }

    uint64_t GetMemoryRequirement(const uint64_t* exprTbl) const
    {
        if (memoryRequirement != 0)
            return memoryRequirement;
        int64_t numel = 1;
        for (int i = 0; i < GetDim(); i++) {
            numel *= shape.At(i, exprTbl);
        }
        return static_cast<uint64_t>(DataSizeOf(numel, dataType));
    }

    std::string DumpType() const
    {
        std::ostringstream oss;
        oss << "<";
        for (int i = 0; i < shape.dimSize; i++) {
            if (shape.dim[i].IsExpression()) {
                oss << "? x ";
            } else {
                oss << shape.dim[i].Value() << " x ";
            }
        }
        oss << DataType2String(dataType);
        oss << ">";
        return oss.str();
    }

    std::string DumpAttr() const
    {
        std::ostringstream oss;
        if (ioProperty == DevIOProperty::ROOT_INCAST) {
            oss << schema::incast(ioIndex).Dump() << " ";
        } else if (ioProperty == DevIOProperty::ROOT_OUTCAST) {
            oss << schema::outcast(ioIndex).Dump() << " ";
        }
        oss << schema::mem(memoryRequirement).Dump() << " ";
        oss << schema::off(addrOffset).Dump();
        return oss.str();
    }

    static std::string DumpAttrDesc(const DevRawTensorDesc* desc)
    {
        std::ostringstream oss;
        if (desc->location == RAW_TENSOR_LOCATION_INCAST) {
            oss << schema::incast(desc->offsetOrIndex).Dump();
        } else if (desc->location == RAW_TENSOR_LOCATION_OUTCAST) {
            oss << schema::outcast(desc->offsetOrIndex).Dump();
        } else {
            oss << schema::off(desc->offsetOrIndex).Dump();
        }
        return oss.str();
    }
};

struct DevAscendTensor {
    uint64_t rawIndex;
};
} // namespace npu::tile_fwk::dynamic
