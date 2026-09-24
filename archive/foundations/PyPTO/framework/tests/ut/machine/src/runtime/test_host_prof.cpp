/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_host_prof.cpp
 * \brief UT for machine/runtime/runner/host_prof.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#undef private
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "adapter/api/acl_api.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(HostProfTest, AllProfOps_NoCrash)
{
    EXPECT_EQ(HostProf::GetProfSwitch(), 0ULL);
    EXPECT_EQ(HostProf::GetProfType(), 0u);

    HostProf::RegHostProf();

    HostProf prof;
    prof.SetProfFunction(nullptr);
    prof.HostProfReportApi(100, 200);
    prof.HostProfReportNodeInfo(200, 24, 1);
    prof.HostProfReportContextInfo(200);
    prof.HostProfReportCacheTaskInfo(nullptr, 1, 1);
}

TEST(HostProfTest, HostProfInit_NullData) { EXPECT_EQ(HostProf::HostProfInit(0, nullptr, 0), -1); }

TEST(HostProfTest, HostProfInit_ZeroLen)
{
    int dummy = 0;
    EXPECT_EQ(HostProf::HostProfInit(0, &dummy, 0), -1);
}

TEST(HostProfTest, HostProfInit_InvalidType)
{
    MspfCommandHandle handle{};
    EXPECT_EQ(HostProf::HostProfInit(999, &handle, sizeof(handle)), -1);
}

TEST(HostProfTest, HostProfInit_TooSmallLen)
{
    MspfCommandHandle handle{};
    EXPECT_EQ(HostProf::HostProfInit(static_cast<uint32_t>(RtProfCtrlType::SWITCH), &handle, 1), -1);
}

TEST(HostProfTest, HostProfInit_ValidInput)
{
    MspfCommandHandle handle{};
    handle.profSwitch = 0xABCD;
    handle.type = 42;
    EXPECT_EQ(HostProf::HostProfInit(static_cast<uint32_t>(RtProfCtrlType::SWITCH), &handle, sizeof(handle)), 0);
    EXPECT_EQ(HostProf::GetProfSwitch(), 0xABCDULL);
    EXPECT_EQ(HostProf::GetProfType(), 42u);
    HostProf::profSwitch_ = 0;
    HostProf::profType_ = 0;
}

TEST(HostProfTest, BuildTensor_NullTensorInfo)
{
    HostProf prof;
    MspfTensorData tensorData{};
    prof.BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, static_cast<RawTensorDataPtr>(nullptr), tensorData);
    EXPECT_EQ(tensorData.tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(tensorData.format, 2u);
    EXPECT_EQ(tensorData.dataType, 0u);
    EXPECT_EQ(tensorData.shape[0], 0u);
}

TEST(HostProfTest, BuildCacheTensorInfo_NullTaskInfo)
{
    HostProf prof;
    prof.BuildCacheTensorInfo(nullptr);
}

TEST(HostProfTest, HostProfReportCacheTaskInfo_NullStream)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.HostProfReportCacheTaskInfo(nullptr, 1, 1);
}

TEST(HostProfTest, IsCacheOpInfoEnable_NullStream)
{
    HostProf prof;
    EXPECT_FALSE(prof.IsCacheOpInfoEnable(nullptr));
}

TEST(HostProfTest, BuildTensor_WithDeviceTensorData)
{
    HostProf prof;
    MspfTensorData tensorData{};

    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3, 4});

    prof.BuildTensor(MSPF_GE_TENSOR_TYPE_OUTPUT, tensor, tensorData);

    EXPECT_EQ(tensorData.tensorType, MSPF_GE_TENSOR_TYPE_OUTPUT);
    EXPECT_EQ(tensorData.format, 2u);
    EXPECT_EQ(tensorData.dataType, static_cast<uint32_t>(DataType2CannType(DataType::DT_FP32)));
    EXPECT_EQ(tensorData.shape[0], 2u);
    EXPECT_EQ(tensorData.shape[1], 3u);
    EXPECT_EQ(tensorData.shape[2], 4u);
}

TEST(HostProfTest, BuildTensor_WithDeviceTensorData_NZFormat)
{
    HostProf prof;
    MspfTensorData tensorData{};

    DeviceTensorData tensor(DataType::DT_FP16, nullptr, std::vector<int64_t>{16, 16}, TileOpFormat::TILEOP_NZ);

    prof.BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, tensor, tensorData);

    EXPECT_EQ(tensorData.format, 29u);
    EXPECT_EQ(tensorData.dataType, static_cast<uint32_t>(DataType2CannType(DataType::DT_FP16)));
}

TEST(HostProfTest, BuildCacheTensorInfo_WithInputOutputData)
{
    ProgramData::GetInstance().Reset();
    HostProf prof;
    prof.opName_ = "test_op";

    DeviceTensorData inputTensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    prof.iDeviceTensorData_.push_back(inputTensor);

    DeviceTensorData outputTensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    prof.oDeviceTensorData_.push_back(outputTensor);

    size_t bufferSize = sizeof(CacheTaskInfo) + sizeof(MspfTensorData) * 2;
    std::vector<uint8_t> buffer(bufferSize, 0);
    CacheTaskInfo* taskInfo = reinterpret_cast<CacheTaskInfo*>(buffer.data());

    prof.BuildCacheTensorInfo(taskInfo);

    EXPECT_EQ(taskInfo->tensorData[0].tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(taskInfo->tensorData[1].tensorType, MSPF_GE_TENSOR_TYPE_OUTPUT);
    ProgramData::GetInstance().Reset();
}

TEST(HostProfTest, HostProfReportCacheTaskInfo_WithRealStream)
{
    AclRtStream stream = nullptr;
    AclError ret = AclRtCreateStream(&stream);
    if (ret != ACLRT_SUCCESS) {
        GTEST_SKIP() << "Failed to create stream";
    }

    HostProf prof;
    prof.opName_ = "test_op";

    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    prof.iDeviceTensorData_.push_back(tensor);

    prof.HostProfReportCacheTaskInfo(stream, 1, MSPF_GE_TASK_TYPE_AI_CORE);

    AclRtDestroyStream(stream);
}

TEST(HostProfTest, GetIOTensor_WithValidFunction)
{
    HostProf prof;

    auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_magic", "test_func", nullptr);
    auto dyndevAttr = std::make_shared<DyndevFunctionAttribute>();
    dyndevAttr->startArgsDirectionList = {ParamDirection::IN, ParamDirection::OUT};
    func->SetDyndevAttribute(dyndevAttr);

    prof.profFunction_ = func.get();

    std::vector<DeviceTensorData> tensors;
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});

    prof.GetIOTensor(tensors);

    EXPECT_EQ(prof.iDeviceTensorData_.size(), 1u);
    EXPECT_EQ(prof.oDeviceTensorData_.size(), 1u);
}

TEST(HostProfTest, GetIOTensor_MismatchedSizes)
{
    HostProf prof;

    auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_magic2", "test_func2", nullptr);
    auto dyndevAttr = std::make_shared<DyndevFunctionAttribute>();
    dyndevAttr->startArgsDirectionList = {ParamDirection::IN};
    func->SetDyndevAttribute(dyndevAttr);

    prof.profFunction_ = func.get();

    std::vector<DeviceTensorData> tensors(2);
    prof.GetIOTensor(tensors);

    EXPECT_TRUE(prof.iDeviceTensorData_.empty());
    EXPECT_TRUE(prof.oDeviceTensorData_.empty());
}

TEST(HostProfTest, SetProfFunction_WithValidFunction)
{
    HostProf prof;

    auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_magic3", "test_func3", nullptr);
    auto dyndevAttr = std::make_shared<DyndevFunctionAttribute>();
    dyndevAttr->startArgsDirectionList = {ParamDirection::IN, ParamDirection::OUT};
    func->SetDyndevAttribute(dyndevAttr);

    std::vector<DeviceTensorData> tensors;
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});

    prof.SetProfFunction(func.get(), tensors);

    EXPECT_EQ(prof.profFunction_, func.get());
    EXPECT_FALSE(prof.opName_.empty());
    EXPECT_EQ(prof.inputsSize_, 1u);
}

TEST(HostProfTest, SetProfFunction_NullFunction)
{
    HostProf prof;
    prof.SetProfFunction(nullptr);
    EXPECT_EQ(prof.profFunction_, nullptr);
}

TEST(HostProfTest, HostProfReportTensorInfo_WithData)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.profFunction_ = reinterpret_cast<Function*>(0x1);

    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    prof.iDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 1;

    prof.HostProfReportTensorInfo(1000);
}

TEST(HostProfTest, PackTensorInfo_InputTensor)
{
    HostProf prof;
    prof.opName_ = "test_op";

    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3, 4});
    prof.iDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 1;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 0);

    EXPECT_EQ(tensorInfo.tensorData[0].tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[0], 2u);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[1], 3u);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[2], 4u);
}

TEST(HostProfTest, PackTensorInfo_OutputTensor)
{
    HostProf prof;
    prof.opName_ = "test_op";

    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{5, 6}, TileOpFormat::TILEOP_NZ);
    prof.oDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 0;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 0);

    EXPECT_EQ(tensorInfo.tensorData[0].tensorType, MSPF_GE_TENSOR_TYPE_OUTPUT);
    EXPECT_EQ(tensorInfo.tensorData[0].format, 29u);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[0], 5u);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[1], 6u);
}

TEST(HostProfTest, PackTensorInfo_ShapeExceedsMaxLen)
{
    HostProf prof;
    prof.opName_ = "test_op";
    std::vector<int64_t> longShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    DeviceTensorData tensor(DataType::DT_FP32, nullptr, longShape);
    prof.iDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 1;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 0);

    EXPECT_EQ(tensorInfo.tensorData[0].shape[0], 1u);
    EXPECT_EQ(tensorInfo.tensorData[0].shape[7], 8u);
}

TEST(HostProfTest, PackTensorInfo_ShapePaddingZeros)
{
    HostProf prof;
    prof.opName_ = "test_op";
    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{3});
    prof.iDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 1;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 0);

    EXPECT_EQ(tensorInfo.tensorData[0].shape[0], 3u);
    for (int j = 1; j < MSPF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
        EXPECT_EQ(tensorInfo.tensorData[0].shape[j], 0u);
    }
}

TEST(HostProfTest, PackTensorInfo_OutputWithNDFormat)
{
    HostProf prof;
    prof.opName_ = "test_op";
    DeviceTensorData tensor(DataType::DT_INT8, nullptr, std::vector<int64_t>{4, 8});
    prof.oDeviceTensorData_.push_back(tensor);
    prof.inputsSize_ = 0;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 0);

    EXPECT_EQ(tensorInfo.tensorData[0].tensorType, MSPF_GE_TENSOR_TYPE_OUTPUT);
    EXPECT_EQ(tensorInfo.tensorData[0].format, 2u);
    EXPECT_EQ(tensorInfo.tensorData[0].dataType, static_cast<uint32_t>(DataType2CannType(DataType::DT_INT8)));
}

TEST(HostProfTest, PackTensorInfo_GroupedIO_MultipleMods)
{
    HostProf prof;
    prof.opName_ = "test_op";
    for (int i = 0; i < 3; i++) {
        prof.iDeviceTensorData_.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{i + 1, 2});
    }
    prof.inputsSize_ = 3;

    MspfTensorInfo tensorInfo{};
    prof.PackTensorInfo(&tensorInfo, 0, 2);

    EXPECT_EQ(tensorInfo.tensorData[2].tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(tensorInfo.tensorData[2].shape[0], 3u);
}

TEST(HostProfTest, HostProfReportTensorInfo_NullProfFunction)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.profFunction_ = nullptr;
    prof.HostProfReportTensorInfo(1000);
    SUCCEED();
}

TEST(HostProfTest, HostProfReportTensorInfo_MultipleGroups)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.profFunction_ = reinterpret_cast<Function*>(0x1);

    for (int i = 0; i < 7; i++) {
        prof.iDeviceTensorData_.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    }
    prof.inputsSize_ = 7;

    prof.HostProfReportTensorInfo(2000);
    SUCCEED();
}

TEST(HostProfTest, BuildTensor_DeviceTensorData_EmptyShape)
{
    HostProf prof;
    MspfTensorData tensorData{};
    DeviceTensorData tensor(DataType::DT_FP32, nullptr, std::vector<int64_t>{});
    prof.BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, tensor, tensorData);
    EXPECT_EQ(tensorData.tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(tensorData.format, 2u);
}

TEST(HostProfTest, BuildCacheTensorInfo_OnlyInputs)
{
    ProgramData::GetInstance().Reset();
    HostProf prof;
    prof.opName_ = "test_op";

    DeviceTensorData inputTensor(DataType::DT_FP16, nullptr, std::vector<int64_t>{4, 4});
    prof.iDeviceTensorData_.push_back(inputTensor);

    size_t bufferSize = sizeof(CacheTaskInfo) + sizeof(MspfTensorData) * 1;
    std::vector<uint8_t> buffer(bufferSize, 0);
    CacheTaskInfo* taskInfo = reinterpret_cast<CacheTaskInfo*>(buffer.data());

    prof.BuildCacheTensorInfo(taskInfo);

    EXPECT_EQ(taskInfo->tensorData[0].tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(taskInfo->tensorData[0].shape[0], 4u);
    ProgramData::GetInstance().Reset();
}

TEST(HostProfTest, HostProfReportCacheTaskInfo_AICPUTaskType)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.HostProfReportCacheTaskInfo(nullptr, 4, MSPF_GE_TASK_TYPE_AI_CPU);
    SUCCEED();
}

TEST(HostProfTest, SetProfFunction_NullFunction_NoChange)
{
    HostProf prof;
    prof.profFunction_ = nullptr;
    prof.SetProfFunction(nullptr);
    EXPECT_EQ(prof.profFunction_, nullptr);
}

TEST(HostProfTest, GetIOTensor_INOUTDirection)
{
    HostProf prof;
    auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_inout", "test_func_inout", nullptr);
    auto dyndevAttr = std::make_shared<DyndevFunctionAttribute>();
    dyndevAttr->startArgsDirectionList = {ParamDirection::IN, ParamDirection::INOUT, ParamDirection::OUT};
    func->SetDyndevAttribute(dyndevAttr);
    prof.profFunction_ = func.get();

    std::vector<DeviceTensorData> tensors;
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2});
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{3});
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{4});

    prof.GetIOTensor(tensors);

    EXPECT_EQ(prof.iDeviceTensorData_.size(), 2u);
    EXPECT_EQ(prof.oDeviceTensorData_.size(), 1u);
}
