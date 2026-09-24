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
 * \file dev_tensor_creator.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_program.h"

namespace npu::tile_fwk::dynamic {
struct InputsHeader {
    uint32_t dim;
    uint32_t cnt;
    int64_t dimVal[0];

    uint32_t size() { return sizeof(InputsHeader) + dim * sizeof(uint64_t); }
    InputsHeader* next() { return reinterpret_cast<InputsHeader*>(reinterpret_cast<uint64_t>(this) + size()); }
};

struct DevAscendTensorDataCreator {
    template <typename T>
    static DevTensorData Create(uintdevptr_t tensorAddress, const std::vector<T>& tensorShape)
    {
        DevTensorData tensorData;
        Init(&tensorData, tensorAddress, tensorShape.data(), tensorShape.size());
        return tensorData;
    }

    template <typename T>
    static void Init(DevTensorData* tensorData, uintdevptr_t tensorAddress, const T* dims, int n)
    {
        if (n > DEV_SHAPE_DIM_MAX) {
            DEV_ERROR(TensorMetaErr::TENSOR_DIM_COUNT_EXCEEDED,
                      "#task..tensor.init: Dimension count (%d) exceeds maximum allowed (%d)", n, DEV_SHAPE_DIM_MAX);
        }
        DEV_ASSERT(TensorMetaErr::TENSOR_DIM_COUNT_EXCEEDED, n <= DEV_SHAPE_DIM_MAX);

        tensorData->address = tensorAddress;
        tensorData->shape.dimSize = n;
        for (int i = 0; i < n; i++) {
            tensorData->shape.dim[i] = dims[i];
        }
    }

    static int Decode(int64_t* inputs, DevAscendProgram* devProg, int idxOffset, DevTensorData* tensorData)
    {
        int64_t addrOffset = *inputs;
        int64_t* ptrBase = reinterpret_cast<int64_t*>(reinterpret_cast<uint64_t>(inputs) + addrOffset);
        int n = 0;
        InputsHeader* h = reinterpret_cast<InputsHeader*>(inputs + 1);
        int64_t* ptr = ptrBase;
        while (reinterpret_cast<int64_t*>(h) < ptrBase) {
            int64_t addr = *ptr;
            if (devProg->disableL2List[idxOffset + n] == 1) {
                DEV_INFO("Tensor index=%d disable l2.", idxOffset + n);
                addr += static_cast<int64_t>(devProg->l2CacheOffset);
            }
            Init(&tensorData[n], addr, h->dimVal, h->dim);
            n++;
            ptr++;
            h = h->next();
        }
        return n;
    }

    /*
     *                  |    8 bytes  |
     *  start -->       |  ptr_offset |
     *  input0 -->      |  dim | cnt  |
     *                  | dim * int64 |
     *  input1 -->      |  dim | cnt  |
     *                  | dim * int64 |
     *                  |     ...     |
     *   ptrstart -->   |    ptr1     |
     *                  |    ptr2     |
     *                  |     ...     |
     */
    static std::vector<int64_t> Encode(const std::vector<DevTensorData>& tensors)
    {
        size_t size = tensors.size() * 0x2 + 1;
        for (auto& t : tensors) {
            size += t.shape.dimSize;
        }

        std::vector<int64_t> data(size);
        int64_t* ptr = data.data() + (data.size() - tensors.size());
        data[0] = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(data.data()); // ptroffset
        auto h = reinterpret_cast<InputsHeader*>(&data[1]);
        for (auto& t : tensors) {
            h->dim = t.shape.dimSize;
            h->cnt = 1;
            for (int i = 0; i < static_cast<int>(h->dim); i++) {
                h->dimVal[i] = t.shape.dim[i];
            }
            *ptr++ = t.address;
            h = h->next();
        }
        if (ptr != data.data() + data.size()) {
            DEV_ERROR(TensorMetaErr::TENSOR_ENCODE_PTR_MISMATCH,
                      "#task..tensor.encode: Pointer mismatch: ptr (0x%p) != data.data() + data.size() (0x%p)",
                      (void*)ptr, (void*)(data.data() + data.size()));
        }
        DEV_ASSERT(TensorMetaErr::TENSOR_ENCODE_PTR_MISMATCH, ptr == data.data() + data.size());

        return data;
    }
};
} // namespace npu::tile_fwk::dynamic
