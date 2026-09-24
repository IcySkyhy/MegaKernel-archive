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
 * \file test_interp_calculator.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <math.h>

#include "interface/interpreter/thread_pool.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"

#include <chrono>

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::util;

namespace {
TEST(ThreadPoolTest, Dispatch)
{
    const int nproc = 2;
    struct Handler {
        static void Entry(void* ctx)
        {
            int threadIndex = (intptr_t)ctx;
            VERIFY_LOGI("Before: %d", threadIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            VERIFY_LOGI("After: %d", threadIndex);
        }
    };
    {
        ThreadPool pool(nproc);
        for (int i = 0; i < nproc * 2; i++) {
            pool.SubmitTask((void*)(intptr_t)i, Handler::Entry);
        }
        pool.NotifyAll();
        pool.WaitForAll();
        pool.Stop();
    }
}
} // namespace
