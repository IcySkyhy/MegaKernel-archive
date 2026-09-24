/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_runtime_data.h
 * \brief
 */
#include "gtest/gtest-spi.h"

#include "interface/utils/string_utils.h"
#include "interface/utils/common.h"

#include "test_machine_common.h"

#include "machine/utils/dynamic/dev_start_args.h"

struct RuntimeDataTest : UnitTestBase {};

TEST_F(RuntimeDataTest, AllocateDeallocate)
{
    const uint64_t rawSize = 0xf;
    const uint64_t size = 0x10;
    const uint64_t count = 0x4;
    EXPECT_EQ(sizeof(RuntimeDataRingBufferHead) + size * count,
              RuntimeDataRingBufferHead::GetRingBufferSize(rawSize, count));

    std::vector<uint8_t> buf(rawSize + sizeof(RuntimeDataRingBufferHead));
    auto& head = *reinterpret_cast<RuntimeDataRingBufferHead*>(buf.data());
    head.Initialize(rawSize, count);
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData() + size);
    EXPECT_EQ(0x1, head.GetIndexPending());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData() + size * 0x2);
    EXPECT_EQ(0x2, head.GetIndexPending());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData() + size * 0x3);
    EXPECT_EQ(0x3, head.GetIndexPending());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData());
    EXPECT_EQ(0x4, head.GetIndexPending());
    EXPECT_TRUE(head.Full());
    head.Deallocate(head.GetRuntimeData() + size);
    EXPECT_EQ(0x1, head.GetIndexFinished());
    head.Deallocate(head.GetRuntimeData() + size * 0x2);
    EXPECT_EQ(0x2, head.GetIndexFinished());
    EXPECT_FALSE(head.Full());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData() + size);
    EXPECT_EQ(0x5, head.GetIndexPending());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData() + size * 0x2);
    EXPECT_EQ(0x6, head.GetIndexPending());
}

TEST_F(RuntimeDataTest, FullAndAllocate)
{
    const uint64_t size = 0x10;
    const uint64_t count = 0x2;

    std::vector<uint8_t> buf(size + sizeof(RuntimeDataRingBufferHead));
    auto& head = *reinterpret_cast<RuntimeDataRingBufferHead*>(buf.data());
    head.Initialize(size, count);
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData(0x1));
    EXPECT_EQ(0x1, head.GetIndexPending());
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData(0x0));
    EXPECT_EQ(0x2, head.GetIndexPending());

    std::thread th([&head]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        head.Deallocate(head.GetRuntimeData(0x1));
    });
    EXPECT_EQ(head.Allocate(), head.GetRuntimeData(0x1));
    th.join();
}
