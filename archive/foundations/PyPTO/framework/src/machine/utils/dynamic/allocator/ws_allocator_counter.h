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
 * \file ws_allocator_counter.h
 * \brief
 */

#pragma once

#include "ws_allocator_basics.h"

#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/sheet_formatter.h"

#include <map>
#include <vector>
#include <utility>
#include <string>

#include <cstdint>

namespace npu::tile_fwk::dynamic {

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

class DelayedDumper {
public:
    void AddRootFuncDump(std::string name, std::map<std::pair<WsMemCategory, size_t>, uint32_t> memReqs)
    {
        if (stitchWindowDumpInfo_.size() < currWindowIdx_ + 1) {
            stitchWindowDumpInfo_.resize(currWindowIdx_ + 1);
        }
        auto& dumpInfo = stitchWindowDumpInfo_[currWindowIdx_].rootFuncs[std::move(name)];
        dumpInfo.cnt++;
        for (auto&& [key, value] : memReqs) {
            dumpInfo.memReqCounter[key] += value;
        }
    }

    void AddAicpuMemDump(const std::map<std::pair<WsMemCategory, size_t>, uint32_t>& memReqs)
    {
        if (stitchWindowDumpInfo_.size() < currWindowIdx_ + 1) {
            stitchWindowDumpInfo_.resize(currWindowIdx_ + 1);
        }
        auto& windowInfo = stitchWindowDumpInfo_[currWindowIdx_];
        for (auto&& [key, value] : memReqs) {
            windowInfo.aicpuMemInfo[key] += value;
        }
    }

    void LogTensorMalloc(std::string name, WsAllocation allocation)
    {
        if (stitchWindowDumpInfo_.size() < currWindowIdx_ + 1) {
            stitchWindowDumpInfo_.resize(currWindowIdx_ + 1);
        }
        auto& dumpInfo = stitchWindowDumpInfo_[currWindowIdx_].rootFuncs[std::move(name)];
        dumpInfo.memReqCounter[std::make_pair(allocation.category_, allocation.rawMemReq_)]++;
    }

    void Rewind() { currWindowIdx_ = 0; }

    void MarkAsNewStitchWindow() { currWindowIdx_++; }

    void DumpStitchWindowMemoryUsage() const
    {
        DEV_MEM_DUMP("Stitch Window Memory Usage:\n");

        auto rangeToString = [](size_t l, size_t r) {
            if (l == r) {
                return std::to_string(l);
            }
            return std::to_string(l) + " ~ " + std::to_string(r);
        };

        SheetFormatter sheet({"Window Idx", "Root Func Name", "Root Func Cnt", "Mem Category", "Mem Req", "Alloc Num"});
        for (size_t i = 0; i < stitchWindowDumpInfo_.size(); i++) {
            auto& windowInfo = stitchWindowDumpInfo_[i];
            if (windowInfo.rootFuncs.empty()) {
                continue;
            }

            sheet.AddRowSeparator();

            size_t lastEqual = i;
            while (lastEqual + 1 < stitchWindowDumpInfo_.size() &&
                   stitchWindowDumpInfo_[lastEqual + 1].rootFuncs == windowInfo.rootFuncs) {
                lastEqual++;
            }

            bool isFirstLine = true;
            for (auto&& [rootFuncName, dumpInfo] : windowInfo.rootFuncs) {
                bool isFirstLine2 = true;
                if (!isFirstLine) {
                    sheet.AddRowSeparator(1);
                }
                for (auto&& [memInfo, cnt] : dumpInfo.memReqCounter) {
                    sheet.AddRow(isFirstLine ? rangeToString(i, lastEqual) : std::string{},
                                 isFirstLine2 ? rootFuncName : std::string{},
                                 isFirstLine2 ? sheet::Integer(dumpInfo.cnt) : std::string{},
                                 GetCategoryName(memInfo.first), memInfo.second, cnt);
                    isFirstLine = false;
                    isFirstLine2 = false;
                }
            }

            if (!windowInfo.aicpuMemInfo.empty()) {
                if (!isFirstLine) {
                    sheet.AddRowSeparator(1);
                }
                bool isFirstLine2 = true;
                for (auto&& [memInfo, cnt] : windowInfo.aicpuMemInfo) {
                    sheet.AddRow(isFirstLine ? rangeToString(i, lastEqual) : std::string{},
                                 isFirstLine2 ? "N/A (Metadata)" : "", isFirstLine2 ? "N/A" : "",
                                 GetCategoryName(memInfo.first), memInfo.second, cnt);
                    isFirstLine = false;
                    isFirstLine2 = false;
                }
            }

            i = lastEqual;
        }
        auto lines = sheet.DumpLines();
        for (auto&& line : lines) {
            DEV_MEM_DUMP("%s\n", line.c_str());
            (void)line;
        }
    }

private:
    using MemReqCounter = std::map<std::pair<WsMemCategory, size_t>, uint32_t>;

    struct RootFuncDumpInfo {
        size_t cnt{0};
        MemReqCounter memReqCounter;

        bool operator==(const RootFuncDumpInfo& oth) const
        {
            return cnt == oth.cnt && memReqCounter == oth.memReqCounter;
        }
    };

    struct WindowInfo {
        std::map<std::string, RootFuncDumpInfo> rootFuncs;
        MemReqCounter aicpuMemInfo;
    };
    std::vector<WindowInfo> stitchWindowDumpInfo_;
    size_t currWindowIdx_{0};
};

class WsAllocatorCounter {
public:
    void LogMalloc(WsAllocation allocation)
    {
        memReqCounter_[std::make_pair(allocation.category_, allocation.rawMemReq_)]++;
        statistics_.totalMemReq += allocation.rawMemReq_;
    }

    void LogDealloc(WsAllocation allocation) { statistics_.totalMemReq -= allocation.rawMemReq_; }

    void Merge(const WsAllocatorCounter& oth)
    {
        for (auto&& [key, value] : oth.memReqCounter_) {
            memReqCounter_[key] += value;
        }
        statistics_.totalMemReq += oth.statistics_.totalMemReq;
    }

    size_t TotalMemReq() const { return statistics_.totalMemReq; }

    void Reset() { memReqCounter_.clear(); }

    void DelayedDumpAsRootFuncAndReset(DelayedDumper& dumper, const char* rootFuncName)
    {
        dumper.AddRootFuncDump(rootFuncName, std::move(memReqCounter_));
        memReqCounter_.clear();
    }

    void DelayedDumpAsAicpuCounterAndReset(DelayedDumper& dumper)
    {
        dumper.AddAicpuMemDump(memReqCounter_);
        memReqCounter_.clear();
    }

private:
    std::map<std::pair<WsMemCategory, size_t>, uint32_t> memReqCounter_;

    struct Statistics {
        size_t totalMemReq{0};
    } statistics_;
};

#else // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL: ^^^ true / false vvv

class DelayedDumper {
public:
    void AddRootFuncDump(std::string name, std::map<std::pair<WsMemCategory, size_t>, uint32_t> memReqs)
    {
        (void)name;
        (void)memReqs;
    }

    void AddAicpuMemDump(const std::map<std::pair<WsMemCategory, size_t>, uint32_t>& memReqs) { (void)memReqs; }

    void LogTensorMalloc(std::string name, WsAllocation allocation)
    {
        (void)name;
        (void)allocation;
    }

    void Rewind() {}

    void MarkAsNewStitchWindow() {}

    void DumpStitchWindowMemoryUsage() const {}
};

class WsAllocatorCounter {
public:
    void LogMalloc(WsAllocation allocation) { (void)allocation; }

    void LogDealloc(WsAllocation allocation) { (void)allocation; }

    void Merge(const WsAllocatorCounter& oth) { (void)oth; }

    size_t TotalMemReq() const { return 0; }

    void Reset() {}

    void DelayedDumpAsRootFuncAndReset(DelayedDumper& dp, const char* rootFuncName)
    {
        (void)dp;
        (void)rootFuncName;
    }

    void DelayedDumpAsAicpuCounterAndReset(DelayedDumper& dumper) { (void)dumper; }
};

#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

} // namespace npu::tile_fwk::dynamic
