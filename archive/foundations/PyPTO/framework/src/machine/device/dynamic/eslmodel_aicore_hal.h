/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <utility>
#include <dlfcn.h>

const int REG_SPR_DATA_MAIN_BASE_ADDR = 0x000000D0;
const int REG_SPR_COND_ADDR = 0x00005108;
const int SUBBLOCKDIM_NUM = 2;
const int BLOCKDIM = 32;
using CaReadReg64Func = uint32_t (*)(uint32_t coreId, uint32_t subcoreId, uint64_t addr, uint64_t* data);
using CaWriteReg64Func = uint32_t (*)(uint32_t coreId, uint32_t subcoreId, uint64_t addr, uint64_t* data);
using BusDirectReadFunc = uint32_t (*)(void* ptr, uint64_t size, uint64_t address);
using BusDirectWriteFunc = uint32_t (*)(uint64_t address, uint64_t size, void* ptr);

namespace npu::tile_fwk::dynamic {

class EslAicoreHal {
public:
    void Init()
    {
        void* eslDriverHandle = dlopen("libnpu_drv.so", RTLD_LAZY | RTLD_NOLOAD);
        caReadReg64_ = reinterpret_cast<CaReadReg64Func>(dlsym(eslDriverHandle, "ca_read_reg64"));
        caWriteReg64_ = reinterpret_cast<CaWriteReg64Func>(dlsym(eslDriverHandle, "ca_write_reg64"));
        busDirectRead_ = reinterpret_cast<BusDirectReadFunc>(dlsym(eslDriverHandle, "ca_read_ddr"));
        busDirectWrite_ = reinterpret_cast<BusDirectWriteFunc>(dlsym(eslDriverHandle, "ca_write_ddr"));
    }

    inline std::pair<int, int> GetSubCoreId(int coreIdx)
    {
        if (coreIdx < BLOCKDIM) {
            return {coreIdx, 0};
        }
        int primaryCoreIdx = (coreIdx - BLOCKDIM) / SUBBLOCKDIM_NUM;
        int subCoreIdx = (coreIdx - BLOCKDIM - (primaryCoreIdx * 2)) % 2 + 1;
        return {primaryCoreIdx, subCoreIdx};
    }

    inline void WriteEslReg(int coreIdx, uint64_t* val)
    {
        auto coreInfo = GetSubCoreId(coreIdx);
        caWriteReg64_(coreInfo.first, coreInfo.second, REG_SPR_DATA_MAIN_BASE_ADDR, val);
    }

    inline uint64_t ReadEslReg(int coreIdx)
    {
        auto coreInfo = GetSubCoreId(coreIdx);
        uint64_t regVal;
        caReadReg64_(coreInfo.first, coreInfo.second, REG_SPR_COND_ADDR, &regVal);
        return regVal;
    }

    inline void WriteEslMem(uint64_t address, uint64_t size, void* value) { busDirectWrite_(address, size, value); }

    inline void SendDynFuncData(DynDeviceTask* dyntask, uint64_t stitchedSize, uint64_t dynFuncDataSize)
    {
        auto dyndata = &dyntask->dynFuncDataList->At(0);
        for (size_t funcIdx = 0; funcIdx < stitchedSize; ++funcIdx) {
            WriteEslMem(static_cast<int64_t>(PtrToValue(dyndata->rawTensorAddr)),
                        dyndata->rawTensorAddrSize * sizeof(uint64_t), dyndata->rawTensorAddr);
            WriteEslMem(static_cast<int64_t>(PtrToValue(dyndata->exprTbl)), dyndata->exprNum * sizeof(uint64_t),
                        dyndata->exprTbl);
            dyndata++;
        }
        WriteEslMem(static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList())), dynFuncDataSize,
                    dyntask->GetDynFuncDataList());
    }

private:
    CaReadReg64Func caReadReg64_;
    CaWriteReg64Func caWriteReg64_;
    BusDirectReadFunc busDirectRead_;
    BusDirectWriteFunc busDirectWrite_;
};

#ifndef __DEVICE__
inline void HandleEslModelTransmission(DevAscendProgram* devProg, DynDeviceTask* dyntask, size_t dynFuncDataSize)
{
    if (devProg->devArgs.enableEslModel && dyntask) {
        EslAicoreHal eslAicoreHal;
        eslAicoreHal.Init();
        eslAicoreHal.SendDynFuncData(dyntask, dyntask->stitchedList.size(), dynFuncDataSize);
    }
}
#endif
} // namespace npu::tile_fwk::dynamic
