/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "../comm_mpi.h"
#include "../tput_async_rdma/tput_async_rdma_kernel.h"

template <typename T, size_t count>
void ExpectGetPlan(
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode, int nRanks = 2,
    int nDevices = 2)
{
    RdmaTestResult result = RunGetAsyncRdmaRootGetPlan<T, count>(
        nRanks, nDevices, 0, 0, elemOffset, elemCount, operationCount, completionMode);
    if (result == RdmaTestResult::SKIPPED) {
        GTEST_SKIP() << "RDMA runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RdmaTestResult::PASSED);
}

template <typename T, size_t count>
void ExpectGetResult(int nRanks = 2, int nDevices = 2)
{
    ExpectGetPlan<T, count>(0, static_cast<int>(count), 1, RdmaCompletionMode::STATUS_WAIT_EACH, nRanks, nDevices);
}

// GET is covered by its own target, parallel to the PUT target.
TEST(TGetAsyncRdma, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 256>();
}

TEST(TGetAsyncRdma, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<int32_t, 4096>();
}

TEST(TGetAsyncRdma, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<uint8_t, 512>();
}

TEST(TGetAsyncRdma, Vec_Uint8_64B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<uint8_t, 64>();
}

TEST(TGetAsyncRdma, Vec_Float_256B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 64>();
}

TEST(TGetAsyncRdma, Vec_Uint8_Offset_63B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<uint8_t, 512>(17, 63, 1, RdmaCompletionMode::STATUS_WAIT_EACH);
}

TEST(TGetAsyncRdma, Vec_Int32_MultiWqe_WaitEach)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<int32_t, 4096>(0, 256, 16, RdmaCompletionMode::STATUS_WAIT_EACH);
}

TEST(TGetAsyncRdma, Vec_Int32_MultiWqe_WaitLast)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<int32_t, 4096>(0, 256, 16, RdmaCompletionMode::STATUS_WAIT_LAST);
}

TEST(TGetAsyncRdma, Vec_Float_PublicEventWaitTest)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<float, 256>(0, 256, 1, RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
}

TEST(TGetAsyncRdma, Vec_Float_MR_6MB)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 524288>();
}

TEST(TGetAsyncRdma, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ExpectGetResult<float, 256>(4, 4);
}

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
