/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "machine/runtime/launcher/device_launcher_types.h"

using namespace npu::tile_fwk::dynamic;
using npu::tile_fwk::DevTensorData;
using npu::tile_fwk::DT_FP32;
using npu::tile_fwk::DT_INT32;
using npu::tile_fwk::TileOpFormat;

TEST(DeviceTensorDataTest, Constructors_AndGetDataSize)
{
    DeviceTensorData d;
    EXPECT_TRUE(d.GetShape().empty());

    int buf = 42;
    std::vector<int64_t> shape = {2, 3};
    DeviceTensorData d2(DT_FP32, &buf, shape);
    EXPECT_EQ(d2.GetAddr(), &buf);
    EXPECT_EQ(d2.GetShape().size(), 2u);
    EXPECT_EQ(d2.GetShape()[0], 2);
    EXPECT_EQ(d2.GetShape()[1], 3);
    EXPECT_EQ(d2.GetDataType(), DT_FP32);
    EXPECT_EQ(d2.Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(d2.GetDataSize(), 2 * 3 * 4);

    uintptr_t addr = 0x1000;
    DeviceTensorData d3(DT_INT32, addr, {4}, TileOpFormat::TILEOP_ND);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(d3.GetAddr()), addr);
    EXPECT_EQ(d3.GetDataType(), DT_INT32);

    DeviceTensorData dEmpty(DT_FP32, nullptr, std::vector<int64_t>{});
    EXPECT_EQ(dEmpty.GetDataSize(), 4);
}

TEST(DeviceLauncherConfigTest, AllConstructors_AndFactory)
{
    DeviceLauncherConfig cfg;
    EXPECT_TRUE(cfg.onBoard);
    EXPECT_EQ(cfg.blockdim, 0);
    EXPECT_EQ(cfg.aicpuNum, 5);
    EXPECT_EQ(cfg.dynWorkspaceSize, 0);
    EXPECT_EQ(cfg.repeatNum, 1);
    EXPECT_TRUE(cfg.runModel);
    EXPECT_FALSE(cfg.controlFlowCache);
    EXPECT_FALSE(cfg.cpuSeparate);
    EXPECT_EQ(cfg.workspaceAddr, 0u);
    EXPECT_FALSE(cfg.workspaceAllocByTorch);
    EXPECT_TRUE(cfg.isCacheOriginShape);

    DeviceLauncherConfig cfg2(false, 8, 3);
    EXPECT_FALSE(cfg2.onBoard);
    EXPECT_EQ(cfg2.blockdim, 8);
    EXPECT_EQ(cfg2.aicpuNum, 3);

    DeviceLauncherConfig cfg3(1024);
    EXPECT_EQ(cfg3.dynWorkspaceSize, 1024);

    DeviceLauncherConfig cfg4(2048, 5);
    EXPECT_EQ(cfg4.dynWorkspaceSize, 2048);
    EXPECT_EQ(cfg4.repeatNum, 5);

    std::vector<uint64_t> addrs = {0x100, 0x200, 0x300};
    DeviceLauncherConfig cfg5(addrs);
    EXPECT_EQ(cfg5.hcclContext.size(), 3u);
    EXPECT_EQ(cfg5.hcclContext[0], 0x100u);

    auto cfg6 = DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(0xDEAD);
    EXPECT_EQ(cfg6.workspaceAddr, 0xDEADu);
}

TEST(OperatorTensorParaTest, Equality_AndHash)
{
    OperatorTensorPara a, b;
    EXPECT_TRUE(a == b);

    DevTensorData td{};
    td.shape.dimSize = 1;
    td.shape.dim[0] = 2;
    a.inputTensorParaList.push_back(td);
    EXPECT_FALSE(a == b);

    OperatorTensorPara c, d;
    td.shape.dim[0] = 4;
    c.outputTensorParaList.push_back(td);
    EXPECT_FALSE(c == d);

    OperatorTensorParaHash hasher;
    OperatorTensorPara empty;
    EXPECT_NE(hasher(empty), 0u);

    OperatorTensorPara h1, h2;
    DevTensorData td2{};
    td2.shape.dimSize = 2;
    td2.shape.dim[0] = 4;
    td2.shape.dim[1] = 8;
    h1.inputTensorParaList.push_back(td2);
    h2.inputTensorParaList.push_back(td2);
    EXPECT_EQ(hasher(h1), hasher(h2));

    OperatorTensorPara h3, h4;
    DevTensorData td3{}, td4{};
    td3.shape.dimSize = 1;
    td3.shape.dim[0] = 4;
    td4.shape.dimSize = 1;
    td4.shape.dim[0] = 8;
    h3.inputTensorParaList.push_back(td3);
    h4.inputTensorParaList.push_back(td4);
    EXPECT_NE(hasher(h3), hasher(h4));
}

TEST(CachedOperatorTest, StaticGetters_AndCtrlFlowCache)
{
    EXPECT_EQ(CachedOperator::GetWorkspaceDevAddrHolder(nullptr), nullptr);
    EXPECT_EQ(CachedOperator::GetCfgDataDevAddrHolder(nullptr), nullptr);
    EXPECT_EQ(CachedOperator::GetMetaDataDevAddrHolder(nullptr), nullptr);
    EXPECT_EQ(CachedOperator::GetBinHandleHolder(nullptr), nullptr);

    CachedOperator op;
    EXPECT_NE(CachedOperator::GetWorkspaceDevAddrHolder(&op), nullptr);
    EXPECT_EQ(*CachedOperator::GetWorkspaceDevAddrHolder(&op), nullptr);
    EXPECT_NE(CachedOperator::GetCfgDataDevAddrHolder(&op), nullptr);
    EXPECT_EQ(*CachedOperator::GetCfgDataDevAddrHolder(&op), nullptr);
    EXPECT_NE(CachedOperator::GetMetaDataDevAddrHolder(&op), nullptr);
    EXPECT_EQ(*CachedOperator::GetMetaDataDevAddrHolder(&op), nullptr);
    EXPECT_NE(CachedOperator::GetBinHandleHolder(&op), nullptr);

    std::vector<DeviceTensorData> inputs = {DeviceTensorData(DT_FP32, (void*)nullptr, {2, 3})};
    std::vector<DeviceTensorData> outputs = {DeviceTensorData(DT_FP32, (void*)nullptr, {2, 3})};
    EXPECT_EQ(op.FindCtrlFlowCache(inputs, outputs), nullptr);

    uint8_t* cache = reinterpret_cast<uint8_t*>(0x1234);
    op.InsertCtrlFlowCache(inputs, outputs, cache);
    EXPECT_EQ(op.FindCtrlFlowCache(inputs, outputs), cache);

    std::vector<DeviceTensorData> inputs2 = {DeviceTensorData(DT_FP32, (void*)nullptr, {4, 5})};
    std::vector<DeviceTensorData> outputs2 = {DeviceTensorData(DT_FP32, (void*)nullptr, {4, 5})};
    EXPECT_EQ(op.FindCtrlFlowCache(inputs2, outputs2), nullptr);
}
