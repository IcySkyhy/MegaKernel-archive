/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int kRows, int kCols>
void launchTMOV_nd2zn_hif8(uint8_t* out, uint8_t* src, void* stream);
template <int kRows, int kCols>
void launchTMOV_nd2zn_half(uint16_t* out, uint16_t* src, void* stream);
template <int kRows, int kCols>
void launchTMOV_nd2zn_b32(uint32_t* out, uint32_t* src, void* stream);

class TMovNd2ZnTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename ElemT, int kRows, int kCols>
void run_tmov_nd2zn(void (*launcher)(ElemT*, ElemT*, void*))
{
    size_t elemSize = sizeof(ElemT);
    size_t inputSize = kRows * kCols * elemSize;
    size_t outputSize = inputSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    ElemT *dstHost, *dstDevice, *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputSize);
    aclrtMallocHost((void**)(&srcHost), inputSize);
    aclrtMalloc((void**)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", inputSize, srcHost, inputSize);
    aclrtMemset(dstDevice, outputSize, 0, outputSize);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launcher(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<ElemT> golden(outputSize / elemSize);
    std::vector<ElemT> devFinal(outputSize / elemSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outputSize, devFinal.data(), outputSize);

    bool ret = ResultCmp<ElemT>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovNd2ZnTest, case_hif8_32x32) { run_tmov_nd2zn<uint8_t, 32, 32>(launchTMOV_nd2zn_hif8<32, 32>); }
TEST_F(TMovNd2ZnTest, case_hif8_32x64) { run_tmov_nd2zn<uint8_t, 32, 64>(launchTMOV_nd2zn_hif8<32, 64>); }
TEST_F(TMovNd2ZnTest, case_hif8_64x64) { run_tmov_nd2zn<uint8_t, 64, 64>(launchTMOV_nd2zn_hif8<64, 64>); }
TEST_F(TMovNd2ZnTest, case_hif8_128x128) { run_tmov_nd2zn<uint8_t, 128, 128>(launchTMOV_nd2zn_hif8<128, 128>); }

TEST_F(TMovNd2ZnTest, case_half_32x32) { run_tmov_nd2zn<uint16_t, 32, 32>(launchTMOV_nd2zn_half<32, 32>); }
TEST_F(TMovNd2ZnTest, case_half_32x64) { run_tmov_nd2zn<uint16_t, 32, 64>(launchTMOV_nd2zn_half<32, 64>); }
TEST_F(TMovNd2ZnTest, case_half_64x64) { run_tmov_nd2zn<uint16_t, 64, 64>(launchTMOV_nd2zn_half<64, 64>); }
TEST_F(TMovNd2ZnTest, case_half_128x128) { run_tmov_nd2zn<uint16_t, 128, 128>(launchTMOV_nd2zn_half<128, 128>); }

TEST_F(TMovNd2ZnTest, case_b32_32x32) { run_tmov_nd2zn<uint32_t, 32, 32>(launchTMOV_nd2zn_b32<32, 32>); }
TEST_F(TMovNd2ZnTest, case_b32_32x64) { run_tmov_nd2zn<uint32_t, 32, 64>(launchTMOV_nd2zn_b32<32, 64>); }
TEST_F(TMovNd2ZnTest, case_b32_64x64) { run_tmov_nd2zn<uint32_t, 64, 64>(launchTMOV_nd2zn_b32<64, 64>); }
TEST_F(TMovNd2ZnTest, case_b32_128x128) { run_tmov_nd2zn<uint32_t, 128, 128>(launchTMOV_nd2zn_b32<128, 128>); }
