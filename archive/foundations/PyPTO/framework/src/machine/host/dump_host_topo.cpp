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
 * \file dump_host_topo.cpp
 * \brief
 */
#include "dump_host_topo.h"

#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/utils/string_utils.h"
#include "machine/utils/dep_verify_dump_path.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_operation.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::topo_dump {
namespace {

inline bool DumpEnabled()
{
    static const bool cached = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_RUNTIME_DEBUG_VERIFY);
    return cached;
}

/// CSV-quote a string per RFC4180 (wrap with ", double inner ").
std::string CsvQuote(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::ofstream OpenCsv(const std::string& fileName, std::initializer_list<const char*> header, std::string& outPath,
                      bool enabled)
{
    std::ofstream ofs;
    if (!enabled) {
        return ofs;
    }
    outPath = GetDepVerifyDumpDir() + "/" + fileName;
    ofs.open(outPath);
    if (!ofs.is_open()) {
        return ofs;
    }
    bool first = true;
    for (const char* col : header) {
        if (!first) {
            ofs << ',';
        }
        ofs << col;
        first = false;
    }
    ofs << '\n';
    return ofs;
}

void WriteCellMatchDesc(std::ostream& os, const dynamic::DevCellMatchTableDesc& desc)
{
    int dim = desc.GetDimensionSize();
    os << '[';
    for (int d = 0; d < dim; ++d) {
        if (d != 0) {
            os << ';';
        }
        os << desc.GetCellShape(d);
    }
    os << "]," << desc.GetStride(0);
}

void CollectTensorAndFuncNamesPerFrontendSlot(const TensorSlotManager& slotManager,
                                              std::unordered_map<int, std::string>& feSlotToTensorName,
                                              std::unordered_map<int, std::string>& feSlotToFuncName)
{
    for (const auto& kv : slotManager.slotIndexDict) {
        int feIdx = kv.second;
        auto nameIt = slotManager.slotNameDict.find(kv.first);
        if (nameIt != slotManager.slotNameDict.end() && !nameIt->second.empty()) {
            StringUtils::AppendUniqueToken(feSlotToTensorName[feIdx], nameIt->second);
        }
        auto funcIt = slotManager.slotFuncNameDict.find(kv.first);
        if (funcIt == slotManager.slotFuncNameDict.end() || funcIt->second.empty()) {
            continue;
        }
        const std::string& joined = funcIt->second;
        size_t start = 0;
        while (start <= joined.size()) {
            size_t sep = joined.find(';', start);
            size_t end = (sep == std::string::npos) ? joined.size() : sep;
            if (end > start) {
                StringUtils::AppendUniqueToken(feSlotToFuncName[feIdx], joined.substr(start, end - start));
            }
            if (sep == std::string::npos) {
                break;
            }
            start = sep + 1;
        }
    }
}

} // namespace

void DumpSlotMapping(const TensorSlotManager& slotManager, const std::unordered_map<int, int>& slotIdxMapping,
                     const IncastOutcastLink& inoutLink)
{
    if (!DumpEnabled()) {
        return;
    }

    std::set<int> inputRuntimeSlots(inoutLink.inputSlotIndexList.begin(), inoutLink.inputSlotIndexList.end());
    std::set<int> outputRuntimeSlots(inoutLink.outputSlotIndexList.begin(), inoutLink.outputSlotIndexList.end());

    std::unordered_map<int, std::string> feSlotToTensorName;
    std::unordered_map<int, std::string> feSlotToFuncName;
    CollectTensorAndFuncNamesPerFrontendSlot(slotManager, feSlotToTensorName, feSlotToFuncName);

    std::ostringstream blob;
    blob << "frontendSlotIdx,runtimeSlotIdx,slotRole,tensorName,funcRawName\n";
    for (const auto& slot : slotIdxMapping) {
        const char* role = "INTERNAL";
        bool isInput = inputRuntimeSlots.count(slot.second) != 0;
        bool isOutput = outputRuntimeSlots.count(slot.second) != 0;
        if (isInput && isOutput) {
            role = "INOUT";
        } else if (isInput) {
            role = "INPUT";
        } else if (isOutput) {
            role = "OUTPUT";
        }
        blob << slot.first << ',' << slot.second << ',' << role << ',' << CsvQuote(feSlotToTensorName[slot.first])
             << ',' << CsvQuote(feSlotToFuncName[slot.first]) << '\n';
    }

    const std::string path = GetDepVerifyDumpDir() + "/slot_mapping.csv";
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return;
    }
    ofs << blob.str();
    ofs.close();
    MACHINE_LOGD("SlotMapping dumped to %s, total %zu entries", path.c_str(), slotIdxMapping.size());
}

StaticTopoCsvWriter::StaticTopoCsvWriter()
{
    ofs_ = OpenCsv("static_topo.csv",
                   {"funcKey", "rootHash", "rawName", "opIdx", "incastSlots", "outcastSlots", "staticSuccessors"},
                   path_, DumpEnabled());
}

StaticTopoCsvWriter::~StaticTopoCsvWriter()
{
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

bool StaticTopoCsvWriter::Enabled() const { return ofs_.is_open(); }

void StaticTopoCsvWriter::WriteFunction(int devRootKey, dynamic::DevAscendFunction& funcBin)
{
    if (!ofs_.is_open()) {
        return;
    }
    auto& os = ofs_;
    for (size_t opIdx = 0; opIdx < funcBin.GetOperationSize(); opIdx++) {
        os << devRootKey << ',' << funcBin.rootHash << ',' << CsvQuote(funcBin.GetRawName()) << ',' << opIdx << ',';
        os << '[';
        for (size_t i = 0; i < funcBin.GetIncastSize(); i++) {
            auto& incast = funcBin.GetIncast(i);
            if (i > 0) {
                os << ';';
            }
            for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
                if (j > 0) {
                    os << '/';
                }
                os << funcBin.At(incast.fromSlotList, j);
            }
        }
        os << "],[";

        for (size_t i = 0; i < funcBin.GetOutcastSize(); i++) {
            auto& outcast = funcBin.GetOutcast(i);
            if (i > 0) {
                os << ';';
            }
            for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
                if (j > 0) {
                    os << '/';
                }
                os << funcBin.At(outcast.toSlotList, j);
            }
        }
        os << ']';

        // staticSuccessors: trailing variable-length opIdx list, one per column.
        auto& succList = funcBin.GetOperationDepGraphSuccList(opIdx);
        for (size_t j = 0; j < succList.size(); j++) {
            os << ',' << funcBin.At(succList, j);
        }
        os << '\n';
    }
}

SlotCellTableCsvWriter::SlotCellTableCsvWriter(bool fillContent)
{
    ofs_ = OpenCsv("slot_cell_table.csv",
                   {"slotIdx", "stitchPolicy", "rootHash", "funcKey", "cellShape", "cellCount", "outcastCount"}, path_,
                   fillContent && DumpEnabled());
}

SlotCellTableCsvWriter::~SlotCellTableCsvWriter()
{
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

bool SlotCellTableCsvWriter::Enabled() const { return ofs_.is_open(); }

void SlotCellTableCsvWriter::WritePartial(int slotIdx, const dynamic::DevCellMatchTableDesc& desc, size_t outcastCount)
{
    if (!ofs_.is_open()) {
        return;
    }
    // rootHash=0, funcKey=-1 are sentinels meaning "aggregated across roots".
    ofs_ << slotIdx << ",partial,0,-1,";
    WriteCellMatchDesc(ofs_, desc);
    ofs_ << ',' << outcastCount << '\n';
}

void SlotCellTableCsvWriter::WriteFullCover(int slotIdx, uint64_t rootHash, int funcKey,
                                            const dynamic::DevCellMatchTableDesc& desc)
{
    if (!ofs_.is_open()) {
        return;
    }
    ofs_ << slotIdx << ",fullcover," << rootHash << ',' << funcKey << ',';
    WriteCellMatchDesc(ofs_, desc);
    ofs_ << ",1\n";
}

} // namespace npu::tile_fwk::topo_dump
