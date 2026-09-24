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
 * \file test_pv_model.cpp
 * \brief
 */

#include <fstream>
#include "gtest/gtest.h"
#include "tilefwk/platform.h"
#include "cost_model/simulation_pv/PvModelImpl.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "cost_model/simulation/common/CommonTools.h"
#include "cost_model/simulation/common/BaseQueue.h"

namespace CostModel {

TEST(PvModelTest, TestAddGlobalAttr)
{
    std::string path("./local_function.cpp");
    std::fstream file(path, std::ios::out);

    if (!file.is_open()) {
        std::cerr << "open config file error: " << path << std::endl;
        return;
    }

    std::string code = R"!!!(
[aicore] void TENSOR_Matmul_T_root_3_0(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat) {
}
[aicore] void TENSOR_Matmul_T_root_3_1(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat) {
}
)!!!";

    file << code << std::endl;
    file.close();

    CostModel::PvModelCodegen::AddGlobalAttr(path);

    std::ifstream ifile(path);
    if (!file) {
        std::cerr << "open config file error: " << path << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << ifile.rdbuf();
    std::string actual(buffer.str());
    ifile.close();

    std::string expect = R"!!!(
extern "C" __global__ [aicore] void TENSOR_Matmul_T_root_3_0(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat) {
}
extern "C" __global__ [aicore] void TENSOR_Matmul_T_root_3_1(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat) {
}

)!!!";

    ASSERT_EQ(actual, expect);
}

TEST(PvModelTest, TestDynFactory)
{
    auto pv = CostModel::PvModelFactory::CreateDyn();
    EXPECT_NE(pv, nullptr);
    npu::tile_fwk::Platform::Instance().GetSoc().SetNPUArch(npu::tile_fwk::NPUArch::DAV_3510);
    pv = CostModel::PvModelFactory::CreateDyn();
    EXPECT_NE(pv, nullptr);
}

TEST(PvModelTest, TestDynCodegen)
{
    std::string org = R"!!!(
#include "TileOpImpl.h"
[aicore] void TENSOR_PATH0_4_0(CoreFuncParam *param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat *taskStat) {
}
)!!!";
    std::string srcFile("TENSOR_PATH0_4_0.cpp");
    std::ofstream ofs(srcFile);
    ofs << org;
    ofs.close();

    std::string dstFile("TENSOR_PATH0_4_0_pvmodel.cpp");
    npu::tile_fwk::CopyFile(srcFile, dstFile);
    PvModelCodegen::AddKernelEntry(dstFile);
    std::ifstream file(dstFile);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    std::string expect = R"!!!(#include "TileOpImpl.h"

extern "C" [aicore] void TENSOR_PATH0_4_0(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat);


extern "C" __global__ [aicore] void PvModelKernelEntry(__gm__ npu::tile_fwk::DynFuncHeader *funcHeader, __gm__ uint64_t *opAttrs) {
    __gm__ npu::tile_fwk::DynFuncData *funcData = (__gm__ npu::tile_fwk::DynFuncData*)(funcHeader + 1);
    CoreFuncParam param = {funcData, funcHeader, opAttrs, funcData->exprTbl};
    TENSOR_PATH0_4_0(&param, funcHeader->stackWorkSpaceAddr, (__gm__ int64_t *)funcHeader->startArgs->commContexts, (__gm__ TaskStat*)NULL);
}


[aicore] void TENSOR_PATH0_4_0(CoreFuncParam *param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat *taskStat) {
}
)!!!";
    EXPECT_EQ(expect, content);
}

TEST(PvModelTest, TestOutputSilencerSilenceAndRestore)
{
    CostModel::OutputSilencer silencer;
    silencer.silence();
    fflush(stdout);
    silencer.restore();
    fflush(stdout);
}

TEST(PvModelTest, TestSimQueueReset)
{
    SimQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Step();
    EXPECT_GT(queue.Size(), 0);
    queue.Reset();
    EXPECT_EQ(queue.Size(), 0);
    EXPECT_EQ(queue.WriteQueueSize(), 0);
}

} // namespace CostModel
