/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_host_topo.h
 * \brief
 */

#ifndef DUMP_HOST_TOPO_H
#define DUMP_HOST_TOPO_H

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "interface/tensor/tensor_slot.h"
#include "machine/utils/dynamic/dev_encode_function.h"
namespace npu::tile_fwk {
namespace topo_dump {

void DumpSlotMapping(const TensorSlotManager& slotManager, const std::unordered_map<int, int>& slotIdxMapping,
                     const IncastOutcastLink& inoutLink);

class StaticTopoCsvWriter {
public:
    StaticTopoCsvWriter();
    ~StaticTopoCsvWriter();

    StaticTopoCsvWriter(const StaticTopoCsvWriter&) = delete;
    StaticTopoCsvWriter& operator=(const StaticTopoCsvWriter&) = delete;

    bool Enabled() const;
    void WriteFunction(int devRootKey, dynamic::DevAscendFunction& funcBin);

private:
    std::ofstream ofs_;
    std::string path_;
};

class SlotCellTableCsvWriter {
public:
    explicit SlotCellTableCsvWriter(bool fillContent);
    ~SlotCellTableCsvWriter();

    SlotCellTableCsvWriter(const SlotCellTableCsvWriter&) = delete;
    SlotCellTableCsvWriter& operator=(const SlotCellTableCsvWriter&) = delete;

    bool Enabled() const;
    void WritePartial(int slotIdx, const dynamic::DevCellMatchTableDesc& desc, size_t outcastCount);
    void WriteFullCover(int slotIdx, uint64_t rootHash, int funcKey, const dynamic::DevCellMatchTableDesc& desc);

private:
    std::ofstream ofs_;
    std::string path_;
};

} // namespace topo_dump
} // namespace npu::tile_fwk

#endif // DUMP_HOST_TOPO_H
