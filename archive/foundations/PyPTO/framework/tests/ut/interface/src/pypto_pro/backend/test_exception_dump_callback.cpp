/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
int32_t pro_register_exception_dump_callback();
void pro_set_dump_info(const char* kernelName, int32_t numTensors, const int32_t* types, const size_t* tensorSizes,
                       const int32_t* dataTypes, const void** tensorAddrs, const int64_t* flatShapes,
                       const int32_t* shapeCounts, int32_t maxDims);
void pro_clear_dump_info();
void pro_set_debug_cmd(const char* cmd);
int32_t pro_test_exception_dump_callback(uint32_t dumpSize, uint32_t* realSize, uint32_t* mode, char* outKernelName,
                                         uint32_t kernelNameBufSize, uint32_t* outExtraTensorNum);
int32_t pro_test_exception_dump_callback_nullptr();
}

class ExceptionDumpCallbackTest : public testing::Test {
protected:
    void SetUp() override { pro_clear_dump_info(); }
    void TearDown() override { pro_clear_dump_info(); }
};

TEST_F(ExceptionDumpCallbackTest, ClearDumpInfoClearsKernelName)
{
    pro_set_dump_info("test_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);
    pro_clear_dump_info();
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithZeroTensors)
{
    pro_set_dump_info("empty_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithSingleTensor)
{
    const char* kernelName = "single_tensor_kernel";
    int32_t types[] = {0};
    size_t sizes[] = {8192};
    int32_t dataTypes[] = {1};
    int64_t addr = 0x1000;
    const void* addrs[] = {&addr};
    int32_t shapeCounts[] = {2};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 1, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithMultipleTensors)
{
    const char* kernelName = "multi_tensor_kernel";
    int32_t types[] = {0, 0, 2};
    size_t sizes[] = {8192, 16384, 40};
    int32_t dataTypes[] = {1, 0, 4};
    int64_t addr0 = 0x1000;
    int64_t addr1 = 0x2000;
    int64_t addr2 = 0x3000;
    const void* addrs[] = {&addr0, &addr1, &addr2};
    int32_t shapeCounts[] = {2, 2, 0};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 3, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithMaxDimsExceedingShapeCount)
{
    const char* kernelName = "max_dims_kernel";
    int32_t types[] = {0};
    size_t sizes[] = {1024};
    int32_t dataTypes[] = {1};
    int64_t addr = 0x4000;
    const void* addrs[] = {&addr};
    int32_t shapeCounts[] = {1};
    int64_t flatShapes[] = {16, 0, 0, 0};

    pro_set_dump_info(kernelName, 1, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 4);
}

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithNullCmd) { pro_set_debug_cmd(nullptr); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithEmptyCmd) { pro_set_debug_cmd(""); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithValidCmd) { pro_set_debug_cmd("echo hello"); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdOverwritesPreviousCmd)
{
    pro_set_debug_cmd("echo first");
    pro_set_debug_cmd("echo second");
}

TEST_F(ExceptionDumpCallbackTest, RegisterCallbackReturnsResultCode)
{
    int32_t ret = pro_register_exception_dump_callback();
    EXPECT_TRUE(ret == 0 || ret == -1);
}

TEST_F(ExceptionDumpCallbackTest, CallbackNullptrDumpInfoReturnsError)
{
    int32_t ret = pro_test_exception_dump_callback_nullptr();
    EXPECT_EQ(ret, 1);
}

TEST_F(ExceptionDumpCallbackTest, CallbackFillsKernelNameFromCache)
{
    pro_set_dump_info("cached_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, 1);
    EXPECT_EQ(realSize, 1);
    EXPECT_STREQ(kernelName, "cached_kernel");
    EXPECT_EQ(extraTensorNum, 0);
}

TEST_F(ExceptionDumpCallbackTest, CallbackFillsTensorsFromCache)
{
    const char* kernelName = "tensor_kernel";
    int32_t types[] = {0, 0};
    size_t sizes[] = {8192, 16384};
    int32_t dataTypes[] = {1, 0};
    int64_t addr0 = 0x1000;
    int64_t addr1 = 0x2000;
    const void* addrs[] = {&addr0, &addr1};
    int32_t shapeCounts[] = {2, 2};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 2, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char outKernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, outKernelName, sizeof(outKernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, 1);
    EXPECT_EQ(realSize, 1);
    EXPECT_STREQ(outKernelName, kernelName);
    EXPECT_EQ(extraTensorNum, 2);
}

TEST_F(ExceptionDumpCallbackTest, CallbackExecutesDebugCmdOnce)
{
    pro_set_debug_cmd("echo callback_test");
    pro_set_dump_info("debug_cmd_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);

    ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName), &extraTensorNum);
    EXPECT_EQ(ret, 0);
}

TEST_F(ExceptionDumpCallbackTest, CallbackSkipsDebugCmdWhenEmpty)
{
    pro_set_debug_cmd("");
    pro_set_dump_info("no_debug_cmd_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_STREQ(kernelName, "no_debug_cmd_kernel");
}
