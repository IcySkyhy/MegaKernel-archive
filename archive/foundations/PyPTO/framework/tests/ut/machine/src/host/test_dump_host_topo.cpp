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
 * \file test_dump_host_topo.cpp
 * \brief UT for machine/host/dump_host_topo.cpp
 */

#include <gtest/gtest.h>
#include "machine/host/dump_host_topo.h"
#include "interface/configs/config_manager.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::topo_dump;

namespace {
class DumpHostTopoEnv : public ::testing::Environment {
public:
    void SetUp() override { config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, CFG_RUNTIME_DEBUG_VERIFY); }
};
} // namespace

static const auto kEnvRegistry = []() {
    testing::AddGlobalTestEnvironment(new DumpHostTopoEnv());
    return 0;
}();

TEST(DumpHostTopoTest, WritersEnabledAndWriteOps)
{
    StaticTopoCsvWriter staticWriter;
    EXPECT_TRUE(staticWriter.Enabled());

    SlotCellTableCsvWriter slotWriterTrue(true);
    EXPECT_TRUE(slotWriterTrue.Enabled());

    SlotCellTableCsvWriter slotWriterFalse(false);
    EXPECT_FALSE(slotWriterFalse.Enabled());

    dynamic::DevCellMatchTableDesc desc;
    desc.SetCellShape({2, 4});
    desc.SetStrideShape({4, 1});
    desc.SetCacheOpMaxCount({1, 0, 0});

    slotWriterTrue.WritePartial(0, desc, 3);
    slotWriterTrue.WriteFullCover(1, 12345, 0, desc);
}

TEST(DumpHostTopoTest, DumpSlotMapping_WhenEnabled)
{
    TensorSlotManager slotManager;
    std::unordered_map<int, int> slotIdxMapping;
    IncastOutcastLink inoutLink;
    DumpSlotMapping(slotManager, slotIdxMapping, inoutLink);
    SUCCEED();
}

TEST(DumpHostTopoTest, FillContentFalse_DisablesWriter)
{
    SlotCellTableCsvWriter slotWriterFalse(false);
    EXPECT_FALSE(slotWriterFalse.Enabled());

    dynamic::DevCellMatchTableDesc desc;
    slotWriterFalse.WritePartial(0, desc, 0);
    slotWriterFalse.WriteFullCover(0, 0, 0, desc);
    SUCCEED();
}
