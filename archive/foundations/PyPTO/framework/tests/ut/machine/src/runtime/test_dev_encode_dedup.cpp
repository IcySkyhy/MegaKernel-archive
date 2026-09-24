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
 * \file test_dev_encode_outcast_lastwins_dedup.cpp
 * \brief UT for encode outcast consumer READ last-wins dedup.
 */

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#define private public
#define protected public

#include "machine/utils/dynamic/dev_encode_program.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

constexpr int kOutcastWrwrTile = 32;

void SetupEncodeOutcastDedupTest()
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(kOutcastWrwrTile, kOutcastWrwrTile);
    TileShape::Current().SetCubeTile({kOutcastWrwrTile, kOutcastWrwrTile}, {kOutcastWrwrTile, kOutcastWrwrTile},
                                     {kOutcastWrwrTile, kOutcastWrwrTile});
}

DevAscendProgram* EncodeAndGetDevProg()
{
    Function* func = Program::GetInstance().GetLastFunction();
    EXPECT_NE(func, nullptr);
    if (func == nullptr) {
        return nullptr;
    }
    auto dynAttr = func->GetDyndevAttribute();
    EXPECT_NE(dynAttr, nullptr);
    if (dynAttr == nullptr) {
        return nullptr;
    }
    EXPECT_FALSE(dynAttr->devProgBinary.empty());
    if (dynAttr->devProgBinary.empty()) {
        return nullptr;
    }
    auto* devProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    if (devProg == nullptr) {
        return nullptr;
    }
    // After encode, RelocProgram(this, 0) stores relative offsets; restore before At()/list access.
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg));
    return devProg;
}

struct OutcastUseEntry {
    CellMatchOpType opType{CellMatchOpType::READ};
    int operationIdx{-1};
};

bool CollectOutcastUseEntries(DevAscendProgram* devProg, std::vector<OutcastUseEntry>& entries)
{
    entries.clear();
    if (devProg == nullptr) {
        return false;
    }
    for (int fi = 0; fi < static_cast<int>(devProg->GetFunctionSize()); ++fi) {
        DevAscendFunction* func = devProg->GetFunction(fi);
        if (func == nullptr) {
            continue;
        }
        for (size_t oi = 0; oi < func->GetOutcastSize(); ++oi) {
            auto& outcast = func->GetOutcast(static_cast<int>(oi));
            if (outcast.producerConsumerList.size() < 3) {
                continue;
            }
            std::vector<OutcastUseEntry> cur;
            cur.reserve(outcast.producerConsumerList.size());
            size_t writeCount = 0;
            size_t readCount = 0;
            for (size_t k = 0; k < outcast.producerConsumerList.size(); ++k) {
                const auto& use = func->At(outcast.producerConsumerList, k);
                cur.push_back(OutcastUseEntry{use.opType, use.operationIdx});
                if (use.opType == CellMatchOpType::READ) {
                    ++readCount;
                } else {
                    ++writeCount;
                }
            }
            // Need interleaved-capable list: >=2 writes and >=1 read.
            if (writeCount >= 2 && readCount >= 1) {
                entries.swap(cur);
                return true;
            }
        }
    }
    return false;
}

} // namespace

// Last-wins only drops earlier registrations of the *same* READ operationIdx; does not force all R after all W.
TEST(DevEncodeOutcastLastWinsDedupTest, OutcastUseList_WRWR_LastWinsNoDupReadOp)
{
    SetupEncodeOutcastDedupTest();

    Tensor a(DT_FP32, {kOutcastWrwrTile, kOutcastWrwrTile}, "a");
    Tensor b(DT_FP32, {kOutcastWrwrTile, kOutcastWrwrTile}, "b");
    Tensor out(DT_FP32, {2 * kOutcastWrwrTile, kOutcastWrwrTile}, "out");
    Tensor scaled(DT_FP32, {2 * kOutcastWrwrTile, kOutcastWrwrTile}, "scaled");
    FUNCTION("outcast_wrwr_lastwins", {a, b}, {out, scaled})
    {
        LOOP("loop_wrwr", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            // tile0: W then R (atomic_add -> read -> mul -> assemble)
            auto w0 = Add(a, b);
            AtomicRMW(w0, {0, 0}, out, AtomicRMWMode::ADD);
            auto r0 = View(out, {kOutcastWrwrTile, kOutcastWrwrTile}, {0, 0});
            Assemble(Mul(r0, Element(DT_FP32, 2.0f)), {0, 0}, scaled);

            // tile1: another W then R -> discovery can be W->R->W->R
            auto w1 = Add(a, b);
            AtomicRMW(w1, {kOutcastWrwrTile, 0}, out, AtomicRMWMode::ADD);
            auto r1 = View(out, {kOutcastWrwrTile, kOutcastWrwrTile}, {kOutcastWrwrTile, 0});
            Assemble(Mul(r1, Element(DT_FP32, 2.0f)), {kOutcastWrwrTile, 0}, scaled);
        }
    }

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);

    std::vector<OutcastUseEntry> entries;
    ASSERT_TRUE(CollectOutcastUseEntries(devProg, entries))
        << "expected an outcast producerConsumerList with >=2 writes and >=1 read";

    // Last-wins: each READ operationIdx appears at most once (earlier dups removed).
    std::unordered_map<int, size_t> readOpCount;
    size_t writeCount = 0;
    for (const auto& e : entries) {
        if (e.opType == CellMatchOpType::READ) {
            ++readOpCount[e.operationIdx];
        } else {
            ++writeCount;
        }
    }
    EXPECT_GE(writeCount, 2u);
    EXPECT_FALSE(readOpCount.empty());
    for (const auto& kv : readOpCount) {
        EXPECT_EQ(kv.second, 1u) << "duplicate READ operationIdx=" << kv.first << " should have been last-wins deduped";
    }
}
