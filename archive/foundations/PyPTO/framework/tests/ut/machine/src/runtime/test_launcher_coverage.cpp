/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_launcher_coverage.cpp
 * \brief High-coverage tests for launcher zero-dependency functions
 */

#include <gtest/gtest.h>
#define private public
#define protected public
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#undef private
#undef protected
#include "interface/program/program.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

// ============================================================================
// DeviceTensorData - 张量数据结构
// ============================================================================

TEST(LauncherCoverageTest, DeviceTensorData_AllMethods)
{
    // 构造
    std::vector<int64_t> shape = {2, 3, 4};
    DeviceTensorData data(DT_FP32, reinterpret_cast<void*>(0x1000), shape);

    // GetDataType
    EXPECT_EQ(data.GetDataType(), DT_FP32);

    // GetAddr
    EXPECT_EQ(data.GetAddr(), reinterpret_cast<void*>(0x1000));

    // GetShape
    auto& shapeRef = data.GetShape();
    EXPECT_EQ(shapeRef.size(), 3u);
    EXPECT_EQ(shapeRef[0], 2);
    EXPECT_EQ(shapeRef[1], 3);
    EXPECT_EQ(shapeRef[2], 4);

    // GetDataSize
    size_t expectedSize = 2 * 3 * 4 * sizeof(float);
    EXPECT_EQ(data.GetDataSize(), expectedSize);
}

TEST(LauncherCoverageTest, DeviceTensorData_EmptyShape)
{
    DeviceTensorData data(DT_INT32, nullptr, {});
    EXPECT_EQ(data.GetShape().size(), 0u);
    // 空shape的numel为1（累乘的初始值），所以DataSize为sizeof(DT_INT32)=4
    EXPECT_EQ(data.GetDataSize(), 4);
}

TEST(LauncherCoverageTest, DeviceTensorData_Format)
{
    // 测试默认格式
    DeviceTensorData data1(DT_FP32, nullptr, {2, 3});
    EXPECT_EQ(data1.Format(), TileOpFormat::TILEOP_ND);

    // 测试指定格式
    DeviceTensorData data2(DT_FP32, nullptr, {2, 3}, TileOpFormat::TILEOP_NZ);
    EXPECT_EQ(data2.Format(), TileOpFormat::TILEOP_NZ);
}

TEST(LauncherCoverageTest, DeviceTensorData_UintptrConstructor)
{
    uintptr_t addr = 0x2000;
    DeviceTensorData data(DT_FP16, addr, {4, 5});
    EXPECT_EQ(data.GetAddr(), reinterpret_cast<void*>(addr));
    EXPECT_EQ(data.GetDataType(), DT_FP16);
    EXPECT_EQ(data.GetShape().size(), 2u);
}

// ============================================================================
// DeviceLauncherConfig - 配置结构体
// ============================================================================

TEST(LauncherCoverageTest, DeviceLauncherConfig_DefaultConstructor)
{
    DeviceLauncherConfig config;
    EXPECT_EQ(config.blockdim, 0);
    EXPECT_EQ(config.aicpuNum, 5);
    EXPECT_TRUE(config.onBoard);
    EXPECT_EQ(config.dynWorkspaceSize, 0);
    EXPECT_EQ(config.repeatNum, 1);
    EXPECT_TRUE(config.runModel);
    EXPECT_TRUE(config.hcclContext.empty());
    EXPECT_FALSE(config.controlFlowCache);
    EXPECT_FALSE(config.cpuSeparate);
    EXPECT_EQ(config.workspaceAddr, 0u);
    EXPECT_FALSE(config.workspaceAllocByTorch);
    EXPECT_TRUE(config.isCacheOriginShape);
}

TEST(LauncherCoverageTest, DeviceLauncherConfig_ParameterizedConstructor)
{
    DeviceLauncherConfig config(false, 10, 8);
    EXPECT_EQ(config.blockdim, 10);
    EXPECT_EQ(config.aicpuNum, 8);
    EXPECT_FALSE(config.onBoard);
}

TEST(LauncherCoverageTest, DeviceLauncherConfig_WorkspaceConstructor)
{
    DeviceLauncherConfig config(1024);
    EXPECT_EQ(config.dynWorkspaceSize, 1024);
}

TEST(LauncherCoverageTest, DeviceLauncherConfig_WorkspaceAndRepeatConstructor)
{
    DeviceLauncherConfig config(2048, 5);
    EXPECT_EQ(config.dynWorkspaceSize, 2048);
    EXPECT_EQ(config.repeatNum, 5);
}

TEST(LauncherCoverageTest, DeviceLauncherConfig_HcclContextConstructor)
{
    std::vector<uint64_t> addrs = {0x1000, 0x2000, 0x3000};
    DeviceLauncherConfig config(addrs);
    EXPECT_EQ(config.hcclContext.size(), 3u);
    EXPECT_EQ(config.hcclContext[0], 0x1000u);
    EXPECT_EQ(config.hcclContext[1], 0x2000u);
    EXPECT_EQ(config.hcclContext[2], 0x3000u);
}

TEST(LauncherCoverageTest, DeviceLauncherConfig_CreateConfigWithWorkspaceAddr)
{
    auto config = DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(0x5000);
    EXPECT_EQ(config.workspaceAddr, 0x5000u);
}

// ============================================================================
// OperatorTensorPara - 算子张量参数
// ============================================================================

TEST(LauncherCoverageTest, OperatorTensorPara_EqualityOperator)
{
    OperatorTensorPara para1;
    OperatorTensorPara para2;

    // 空参数应该相等
    EXPECT_TRUE(para1 == para2);

    // 添加相同的输入
    DevTensorData tensor1;
    tensor1.shape.dimSize = 2;
    tensor1.shape.dim[0] = 2;
    tensor1.shape.dim[1] = 3;

    para1.inputTensorParaList.push_back(tensor1);
    para2.inputTensorParaList.push_back(tensor1);
    EXPECT_TRUE(para1 == para2);

    // 添加不同的输出
    DevTensorData tensor2;
    tensor2.shape.dimSize = 1;
    tensor2.shape.dim[0] = 4;
    para1.outputTensorParaList.push_back(tensor2);
    EXPECT_FALSE(para1 == para2);

    para2.outputTensorParaList.push_back(tensor2);
    EXPECT_TRUE(para1 == para2);
}

TEST(LauncherCoverageTest, OperatorTensorPara_InequalityInputSize)
{
    OperatorTensorPara para1;
    OperatorTensorPara para2;

    DevTensorData tensor;
    tensor.shape.dimSize = 1;
    tensor.shape.dim[0] = 2;

    para1.inputTensorParaList.push_back(tensor);
    EXPECT_FALSE(para1 == para2);
}

TEST(LauncherCoverageTest, OperatorTensorPara_InequalityInputShape)
{
    OperatorTensorPara para1;
    OperatorTensorPara para2;

    DevTensorData tensor1;
    tensor1.shape.dimSize = 2;
    tensor1.shape.dim[0] = 2;
    tensor1.shape.dim[1] = 3;

    DevTensorData tensor2;
    tensor2.shape.dimSize = 2;
    tensor2.shape.dim[0] = 2;
    tensor2.shape.dim[1] = 4; // 不同的shape

    para1.inputTensorParaList.push_back(tensor1);
    para2.inputTensorParaList.push_back(tensor2);
    EXPECT_FALSE(para1 == para2);
}

// ============================================================================
// OperatorTensorParaHash - 哈希函数
// ============================================================================

TEST(LauncherCoverageTest, OperatorTensorParaHash_EmptyPara)
{
    OperatorTensorParaHash hasher;
    OperatorTensorPara para;
    size_t hash = hasher(para);
    // 空参数的hash应该是一个固定值
    EXPECT_NE(hash, 0u);
}

TEST(LauncherCoverageTest, OperatorTensorParaHash_SamePara)
{
    OperatorTensorParaHash hasher;

    OperatorTensorPara para1;
    OperatorTensorPara para2;

    DevTensorData tensor;
    tensor.shape.dimSize = 2;
    tensor.shape.dim[0] = 2;
    tensor.shape.dim[1] = 3;

    para1.inputTensorParaList.push_back(tensor);
    para2.inputTensorParaList.push_back(tensor);

    EXPECT_EQ(hasher(para1), hasher(para2));
}

TEST(LauncherCoverageTest, OperatorTensorParaHash_DifferentPara)
{
    OperatorTensorParaHash hasher;

    OperatorTensorPara para1;
    OperatorTensorPara para2;

    DevTensorData tensor1;
    tensor1.shape.dimSize = 2;
    tensor1.shape.dim[0] = 2;
    tensor1.shape.dim[1] = 3;

    DevTensorData tensor2;
    tensor2.shape.dimSize = 2;
    tensor2.shape.dim[0] = 4;
    tensor2.shape.dim[1] = 5;

    para1.inputTensorParaList.push_back(tensor1);
    para2.inputTensorParaList.push_back(tensor2);

    EXPECT_NE(hasher(para1), hasher(para2));
}

// ============================================================================
// CachedOperator - 缓存操作符
// ============================================================================

TEST(LauncherCoverageTest, CachedOperator_GetWorkspaceDevAddrHolder_Null)
{
    auto holder = CachedOperator::GetWorkspaceDevAddrHolder(nullptr);
    EXPECT_EQ(holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetWorkspaceDevAddrHolder_Valid)
{
    CachedOperator op;
    auto holder = CachedOperator::GetWorkspaceDevAddrHolder(&op);
    EXPECT_NE(holder, nullptr);
    EXPECT_EQ(*holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetCfgDataDevAddrHolder_Null)
{
    auto holder = CachedOperator::GetCfgDataDevAddrHolder(nullptr);
    EXPECT_EQ(holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetCfgDataDevAddrHolder_Valid)
{
    CachedOperator op;
    auto holder = CachedOperator::GetCfgDataDevAddrHolder(&op);
    EXPECT_NE(holder, nullptr);
    EXPECT_EQ(*holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetMetaDataDevAddrHolder_Null)
{
    auto holder = CachedOperator::GetMetaDataDevAddrHolder(nullptr);
    EXPECT_EQ(holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetMetaDataDevAddrHolder_Valid)
{
    CachedOperator op;
    auto holder = CachedOperator::GetMetaDataDevAddrHolder(&op);
    EXPECT_NE(holder, nullptr);
    EXPECT_EQ(*holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetBinHandleHolder_Null)
{
    auto holder = CachedOperator::GetBinHandleHolder(nullptr);
    EXPECT_EQ(holder, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_GetBinHandleHolder_Valid)
{
    CachedOperator op;
    auto holder = CachedOperator::GetBinHandleHolder(&op);
    EXPECT_NE(holder, nullptr);
    // GetBinHandleHolder 返回 void*，不能解引用
}

TEST(LauncherCoverageTest, CachedOperator_FindCtrlFlowCache_NotFound)
{
    CachedOperator op;
    std::vector<DeviceTensorData> inputList;
    std::vector<DeviceTensorData> outputList;

    auto cache = op.FindCtrlFlowCache(inputList, outputList);
    EXPECT_EQ(cache, nullptr);
}

TEST(LauncherCoverageTest, CachedOperator_InsertAndFindCtrlFlowCache)
{
    CachedOperator op;
    std::vector<DeviceTensorData> inputList;
    std::vector<DeviceTensorData> outputList;

    uint8_t cacheData = 42;
    op.InsertCtrlFlowCache(inputList, outputList, &cacheData);

    auto cache = op.FindCtrlFlowCache(inputList, outputList);
    EXPECT_EQ(cache, &cacheData);
}

TEST(LauncherCoverageTest, CachedOperator_FindCtrlFlowCache_DifferentInput)
{
    CachedOperator op;
    std::vector<DeviceTensorData> inputList1;
    std::vector<DeviceTensorData> outputList1;

    uint8_t cacheData = 42;
    op.InsertCtrlFlowCache(inputList1, outputList1, &cacheData);

    std::vector<DeviceTensorData> inputList2;
    std::vector<DeviceTensorData> outputList2;
    DeviceTensorData tensor(DT_FP32, nullptr, {2, 3});
    inputList2.push_back(tensor);

    auto cache = op.FindCtrlFlowCache(inputList2, outputList2);
    EXPECT_EQ(cache, nullptr);
}
