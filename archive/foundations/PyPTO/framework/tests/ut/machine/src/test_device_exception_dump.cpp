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
 * \file test_device_exception_dump.cpp
 * \brief UT for device_exception_dump.cpp - 入口函数 ExceptionDumpCallBack
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <memory>
#include "securec.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "adapter/api/runtime_define.h"
#include "adapter/api/adump_define.h"
#include "adapter/api/runtime_api.h"
#include "adapter/api/adump_api.h"
#include "machine/runtime/runner/device_exception_dump.h"
#include "machine/utils/machine_ws_intf.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error_code.h"
#define private public
#define protected public
#include "interface/function/function.h"
#include "interface/program/program.h"
#undef private
#undef protected

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

constexpr int32_t MAX_AICPU_ARG_NUM = 7;
constexpr int32_t KERNEL_NAME_IDX = 0;
constexpr int32_t INPUT_OUTPUT_IDX = 4;
constexpr int32_t TENSOR_DATA_IDX = 6;

class DeviceExceptionDumpTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        auto dynAttr = std::make_shared<DyndevFunctionAttribute>();
        dynAttr->disableL2List.clear();
        auto func = std::make_shared<Function>(Program::GetInstance(), "TestFunc", "TENSOR_TestFunc", nullptr);
        func->dyndevAttr_ = dynAttr;
        Program::GetInstance().SetLastFunction(func.get());
        testFunc_ = func;
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().SetLastFunction(nullptr);
        testFunc_.reset();
    }
    std::shared_ptr<Function> testFunc_;
};

TEST_F(DeviceExceptionDumpTest, TestExceptionDumpInfoIsNullptr)
{
    RtExceptionInfo exceptionInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, nullptr, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX));
}

TEST_F(DeviceExceptionDumpTest, TestAdumpRegExeptionDump)
{
    auto ret = AdumpRegExceptionDump();
    EXPECT_EQ(ret, 0);
}

TEST_F(DeviceExceptionDumpTest, TestBothPtrIsNullptr)
{
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    auto ret = ExceptionDumpCallBack(nullptr, nullptr, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX));
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithNullArgAddr)
{
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = nullptr;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX));
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithMismatchedArgsize)
{
    char kernelName[] = "PyPTO_TestKernel";
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = kernelName;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * 4;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithNonPyptoKernel)
{
    char exceptionKernelName[] = "PyPTO_Kernel";
    int64_t inputOutputInfo[2] = {1, 0};
    uint8_t binBuf[64] = {};
    std::vector<uint8_t> tensorBuf(32, 0xAB);
    DevTensorData tensorData[1] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 4;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = const_cast<char*>("OtherKernelName");
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = exceptionKernelName;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin = reinterpret_cast<RtBinHandle>(binBuf);
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.binSize = sizeof(binBuf);
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    // kernelName from exceptionKernelInfo, not from kernelArg[0]
    EXPECT_STREQ(dumpInfo.kernelName, exceptionKernelName);
    EXPECT_STREQ(dumpInfo.kernelDisplayName, exceptionKernelName);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithValidTensors)
{
    char kernelName[] = "PyPTO_ValidTensorKernel";
    int64_t inputOutputInfo[2] = {2, 0};
    std::vector<uint8_t> tensorBuf0(32, 0xAB);
    std::vector<uint8_t> tensorBuf1(16, 0xCD);
    DevTensorData tensorData[2] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf0.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 4;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    tensorData[1].address = reinterpret_cast<uint64_t>(tensorBuf1.data());
    tensorData[1].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[1].shape.dim[0] = 4;
    tensorData[1].shape.dim[1] = 4;
    tensorData[1].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = kernelName;
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_EQ(dumpInfo.extraTensorNum, 2);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithZeroAddressTensor)
{
    char kernelName[] = "PyPTO_ZeroAddrKernel";
    int64_t inputOutputInfo[2] = {1, 0};
    DevTensorData tensorData[1] = {};
    tensorData[0].address = 0;
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 16;
    tensorData[0].shape.dimSize = 1;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = kernelName;
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_EQ(dumpInfo.extraTensorNum, 1);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithMultipleTensors)
{
    char kernelName[] = "PyPTO_MultiTensorKernel";
    int64_t inputOutputInfo[2] = {3, 0};
    std::vector<uint8_t> tensorBuf0(64, 0xAB);
    std::vector<uint8_t> tensorBuf1(32, 0xCD);
    std::vector<uint8_t> tensorBuf2(16, 0xEF);
    DevTensorData tensorData[3] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf0.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 8;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    tensorData[1].address = reinterpret_cast<uint64_t>(tensorBuf1.data());
    tensorData[1].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[1].shape.dim[0] = 4;
    tensorData[1].shape.dim[1] = 8;
    tensorData[1].shape.dimSize = 2;
    tensorData[2].address = reinterpret_cast<uint64_t>(tensorBuf2.data());
    tensorData[2].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[2].shape.dim[0] = 4;
    tensorData[2].shape.dim[1] = 4;
    tensorData[2].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = kernelName;
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_EQ(dumpInfo.extraTensorNum, 3);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithOutputTensors)
{
    char kernelName[] = "PyPTO_OutputTensorKernel";
    int64_t inputOutputInfo[2] = {2, 1};
    std::vector<uint8_t> tensorBuf0(32, 0xAB);
    std::vector<uint8_t> tensorBuf1(16, 0xCD);
    DevTensorData tensorData[2] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf0.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 4;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    tensorData[1].address = reinterpret_cast<uint64_t>(tensorBuf1.data());
    tensorData[1].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[1].shape.dim[0] = 4;
    tensorData[1].shape.dim[1] = 4;
    tensorData[1].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = kernelName;
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_EQ(dumpInfo.extraTensorNum, 1);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithExceptionKernelInfoBin)
{
    char exceptionKernelName[] = "PyPTO_BinTestKernel";
    int64_t inputOutputInfo[2] = {1, 0};
    uint8_t binBuf[128] = {};
    std::vector<uint8_t> tensorBuf(32, 0xAB);
    DevTensorData tensorData[1] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 4;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = exceptionKernelName;
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = exceptionKernelName;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin = reinterpret_cast<RtBinHandle>(binBuf);
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.binSize = sizeof(binBuf);
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_EQ(dumpInfo.bin, exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin);
    EXPECT_STREQ(dumpInfo.kernelName, exceptionKernelName);
    EXPECT_STREQ(dumpInfo.kernelDisplayName, exceptionKernelName);
}

TEST_F(DeviceExceptionDumpTest, TestAicoreExceptionWithNullExceptionKernelName)
{
    int64_t inputOutputInfo[2] = {1, 0};
    uint8_t binBuf[64] = {};
    std::vector<uint8_t> tensorBuf(32, 0xAB);
    DevTensorData tensorData[1] = {};
    tensorData[0].address = reinterpret_cast<uint64_t>(tensorBuf.data());
    tensorData[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensorData[0].shape.dim[0] = 4;
    tensorData[0].shape.dim[1] = 8;
    tensorData[0].shape.dimSize = 2;
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    kernelArg[KERNEL_NAME_IDX] = const_cast<char*>("PyPTO_NullKernelName");
    kernelArg[INPUT_OUTPUT_IDX] = inputOutputInfo;
    kernelArg[TENSOR_DATA_IDX] = tensorData;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argAddr = kernelArg.data();
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.argsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = nullptr;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin = reinterpret_cast<RtBinHandle>(binBuf);
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.binSize = sizeof(binBuf);
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    // bin should be filled from exceptionKernelInfo
    EXPECT_EQ(dumpInfo.bin, exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin);
    // kernelName should remain empty when exceptionKernelInfo.kernelName is nullptr
    EXPECT_STREQ(dumpInfo.kernelName, "");
    EXPECT_STREQ(dumpInfo.kernelDisplayName, "");
}

TEST_F(DeviceExceptionDumpTest, TestNonAicoreExceptionTypeFFTSPlus)
{
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::FFTS_PLUS;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
}

// ============================ AICPU exception dump tests ============================

static void BuildAicpuArgBuffer(std::vector<uint8_t>& buffer, int64_t inputSize,
                                const std::vector<DevTensorData>& tensors)
{
    size_t totalSize = sizeof(AiCpuArgs) + DEV_TENSOR_DATA_OFFSET * sizeof(int64_t) +
                       tensors.size() * sizeof(DevTensorData);
    buffer.assign(totalSize, 0);
    auto* args = reinterpret_cast<AiCpuArgs*>(buffer.data());
    args->kArgs = DeviceKernelArgs{};
    int64_t* tensorInfo = reinterpret_cast<int64_t*>(buffer.data() + sizeof(AiCpuArgs));
    tensorInfo[0] = inputSize;
    tensorInfo[1] = 0;
    auto* tensorData = reinterpret_cast<DevTensorData*>(tensorInfo + DEV_TENSOR_DATA_OFFSET);
    for (size_t i = 0; i < tensors.size(); i++) {
        tensorData[i] = tensors[i];
    }
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithNullArgAddr)
{
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = nullptr;
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = sizeof(AiCpuArgs);
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX));
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithZeroArgSize)
{
    int64_t dummy = 0;
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = &dummy;
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = 0;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithNonPyptoFunctionName)
{
    std::vector<uint8_t> buffer;
    BuildAicpuArgBuffer(buffer, 0, {});
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = buffer.data();
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = static_cast<uint32_t>(buffer.size());
    exceptionInfo.expandInfo.u.aicpuInfo.functionName = "OtherAicpuKernel";
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithNullFunctionName)
{
    std::vector<uint8_t> buffer;
    BuildAicpuArgBuffer(buffer, 0, {});
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = buffer.data();
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = static_cast<uint32_t>(buffer.size());
    exceptionInfo.expandInfo.u.aicpuInfo.functionName = nullptr;
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_STREQ(dumpInfo.kernelName, "PyPTO_Aicpu_TestFunc");
    EXPECT_STREQ(dumpInfo.kernelDisplayName, "PyPTO_Aicpu_TestFunc");
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithValidTensors)
{
    std::vector<uint8_t> tensorBuf0(32, 0xAB);
    std::vector<uint8_t> tensorBuf1(16, 0xCD);
    std::vector<DevTensorData> tensors(2);
    tensors[0].address = reinterpret_cast<uint64_t>(tensorBuf0.data());
    tensors[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensors[0].shape.dim[0] = 4;
    tensors[0].shape.dim[1] = 8;
    tensors[0].shape.dimSize = 2;
    tensors[1].address = reinterpret_cast<uint64_t>(tensorBuf1.data());
    tensors[1].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensors[1].shape.dim[0] = 4;
    tensors[1].shape.dim[1] = 4;
    tensors[1].shape.dimSize = 2;

    std::vector<uint8_t> buffer;
    BuildAicpuArgBuffer(buffer, 2, tensors);
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = buffer.data();
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = static_cast<uint32_t>(buffer.size());
    exceptionInfo.expandInfo.u.aicpuInfo.functionName = "DynTileFwkKernelServer";
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE);
    EXPECT_EQ(realSize, 1);
    EXPECT_STREQ(dumpInfo.kernelName, "PyPTO_Aicpu_TestFunc");
    EXPECT_STREQ(dumpInfo.kernelDisplayName, "PyPTO_Aicpu_TestFunc");
    EXPECT_EQ(dumpInfo.extraTensorNum, 2);
}

TEST_F(DeviceExceptionDumpTest, TestAicpuExceptionWithZeroAddressTensor)
{
    std::vector<DevTensorData> tensors(1);
    tensors[0].address = 0;
    tensors[0].dataType = static_cast<int32_t>(DataType::DT_FP32);
    tensors[0].shape.dim[0] = 16;
    tensors[0].shape.dimSize = 1;

    std::vector<uint8_t> buffer;
    BuildAicpuArgBuffer(buffer, 1, tensors);
    RtExceptionInfo exceptionInfo = {};
    AdxExceptionDumpInfo dumpInfo = {};
    uint32_t dumpSize = 1;
    uint32_t realSize = 0;
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.argAddr = buffer.data();
    exceptionInfo.expandInfo.u.aicpuInfo.argsize = static_cast<uint32_t>(buffer.size());
    exceptionInfo.expandInfo.u.aicpuInfo.functionName = "DynTileFwkKernelServer";
    auto ret = ExceptionDumpCallBack(&exceptionInfo, &dumpInfo, dumpSize, &realSize, &mode);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(dumpInfo.extraTensorNum, 1);
}
