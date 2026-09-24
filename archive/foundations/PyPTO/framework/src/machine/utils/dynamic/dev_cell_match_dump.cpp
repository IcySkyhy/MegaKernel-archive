/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/utils/dynamic/dev_cell_match_dump.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "tilefwk/aikernel_data.h"
#include <sstream>

namespace npu::tile_fwk::dynamic {

std::string DumpCellMatchPartialUpdateTable(const uint64_t* cellMatchTableData, uint32_t dataSize,
                                            const DevCellMatchTableDesc& desc)
{
    std::ostringstream oss;
    if (desc.cellUint64Size == 0 || dataSize == 0 || cellMatchTableData == nullptr) {
        return oss.str();
    }
    static const char* opNames[] = {"NW", "AW", "RD"};
    auto opTypeName = [&](uint32_t t) -> const char* {
        return (t == CELL_MATCH_OP_TYPE_NONE) ? "NONE" : (t <= CELL_MATCH_OP_TYPE_READ) ? opNames[t] : "?";
    };
    for (uint32_t ci = 0; ci < desc.GetStride(0); ci++) {
        uint64_t cellBase = ci * desc.cellUint64Size;
        uint64_t meta = cellMatchTableData[cellBase];
        uint32_t curType = CellMatchGetCurrentOpType(meta);
        uint32_t curCnt = CellMatchGetCurrentOpCount(meta);
        uint32_t prevType = CellMatchGetPrevMutexOpType(meta);
        uint32_t prevCnt = CellMatchGetPrevMutexOpCount(meta);
        uint64_t tagId = CellMatchGetTagId(meta);
        oss << "cell[" << ci << "]: meta=0x" << std::hex << meta << std::dec;
        oss << " curType=" << opTypeName(curType) << " curCnt=" << curCnt << " prevType=" << opTypeName(prevType);
        if (prevCnt != CELL_MATCH_INVALID_OP_COUNT) {
            oss << " prevCnt=" << prevCnt;
        } else {
            oss << " prevCnt=INVALID";
        }
        oss << " tagId=0x" << std::hex << tagId << std::dec;
        auto dumpOps = [&](uint32_t opType, uint32_t count, const char* label) {
            if (count == 0 || opType > CELL_MATCH_OP_TYPE_READ)
                return;
            oss << " " << label << "=[";
            uint32_t printed = 0;
            for (uint32_t i = 0; i < count; i++) {
                uint64_t opId = CellMatchGetOpId(const_cast<uint64_t*>(cellMatchTableData), cellBase, opType, i, desc);
                if (opId != AICORE_TASK_INIT) {
                    oss << (printed > 0 ? "," : "") << FuncID(static_cast<uint32_t>(opId)) << "!"
                        << TaskID(static_cast<uint32_t>(opId));
                    printed++;
                }
            }
            oss << "]";
        };
        if (prevType <= CELL_MATCH_OP_TYPE_READ && prevCnt > 0 && prevCnt != CELL_MATCH_INVALID_OP_COUNT) {
            dumpOps(prevType, prevCnt, "prev");
        }
        if (curType <= CELL_MATCH_OP_TYPE_READ && curCnt > 0) {
            dumpOps(curType, curCnt, "cur");
        }
        oss << "\n";
    }
    return oss.str();
}

} // namespace npu::tile_fwk::dynamic
