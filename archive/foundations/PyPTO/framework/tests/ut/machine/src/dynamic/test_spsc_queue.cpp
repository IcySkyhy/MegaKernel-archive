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
 * \file test_spsc_queue.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <thread>
#include "machine/utils/dynamic/spsc_queue.h"

TEST(spscqueue, normal)
{
    int n = 100000;
    SPSCQueue<int64_t, 64> queue;

    std::thread t([&]() {
        for (int i = 0; i < n; i++) {
            queue.Enqueue(i);
        }
    });
    t.detach();
    for (int i = 0; i < n; i++) {
        int val = queue.Dequeue();
        EXPECT_EQ(val, i);
    }
}
