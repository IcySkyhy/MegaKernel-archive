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
 * \file test_distributed.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager_ng.h"
#include "distributed_test_framework.h"
#include "test_distributed.h"
#include "adapter/api/hcomm_api.h"

namespace npu::tile_fwk::Distributed {
class DistributedTest : public testing::TestWithParam<OpMetaData> {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() { GegisterOps(); }

    void SetUp() override
    {
        const nlohmann::json& testData = GetParam().testData_;
        std::string opName = testData["operation"].get<std::string>();
        if (opName == "MultiCommGroupsOp") {
            Distributed::TestFrameworkInit2Groups(testParam, hcomTestParam, physicalDeviceId);
        } else {
            Distributed::TestFrameworkInit(testParam, hcomTestParam, physicalDeviceId);
        }
        std::string outputDir = "output";
        bool res = CreateDir(outputDir);
        CHECK(res) << "Failed to create directory: " << outputDir;
        std::string folderPath = outputDir + "/output_" + getTimeStamp() + "_" + std::to_string(physicalDeviceId);
        setenv("TILE_FWK_OUTPUT_DIR", folderPath.c_str(), 0);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        DistributedTestDestroy();
        Distributed::TestFrameworkDestroy(timeout);
    }

    // 暴露超时设置接口
    void SetDestroyTimeout(int32_t destroyTimeout) { timeout = destroyTimeout; }

    // 通用测试入口
    void RunDistributedTestGeneric(const nlohmann::json& testData, const std::string& fileName)
    {
        if (!testData.contains("input_tensors") || testData["input_tensors"].empty()) {
            FAIL() << "No input tensors in testData: " << testData.dump();
        }
        std::string opName = testData["operation"].get<std::string>();
        std::string dtype = testData["input_tensors"][0]["dtype"].get<std::string>();
        std::string caseName = testData["case_name"].get<std::string>();
        std::string goldenDir = GetGoldenDirPath(testData, fileName);
        DisOpRegister::GetRegister().Run(opName, testParam, dtype, goldenDir);
        DISTRIBUTED_LOGI("test case finished successfully: op=%s, case=%s, json file=%s.", opName.c_str(),
                         caseName.c_str(), fileName.c_str());
    }

protected:
    void DistributedTestDestroy()
    {
        // 销毁集合通信域
        CHECK(HcommCommDestroy(hcomTestParam.hcclComm) == 0) << "HcommCommDestroy failed";
        // 重置设备
        CHECK(AclRtResetDevice(physicalDeviceId) == 0) << "aclResetDevice failed";
        // 设备去初始化
        CHECK(AclFinalize() == 0) << "AclFinalize failed";
    }

    Distributed::OpTestParam testParam;
    Distributed::HcomTestParam hcomTestParam;
    int32_t timeout = 10;
    int physicalDeviceId = 0;
};

// 注册所有算子
void GegisterOps()
{
    auto& reg = DisOpRegister::GetRegister();
    // 模板算子
    reg.RegisterOp("AllGather", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllGather<T>(testParam, goldenDir);
    });
    reg.RegisterOp("ReduceScatter", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestReduceScatter<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduce", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceAddAllReduce", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduceAddAllReduce<T>(testParam, goldenDir);
    });
    reg.disRegisterMap["AllGatherAttnPostReduceScatter"] = [](OpTestParam& testParam, const std::string&,
                                                              std::string& goldenDir) {
        Distributed::TestAllGatherAttentionPostReducescatter(testParam, goldenDir);
    };
    reg.RegisterOp("MultiCommGroupsOp", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::Test2GroupsAllGather<T>(testParam, goldenDir);
    });
    // 后续按照上面格式增加算子
}

INSTANTIATE_TEST_SUITE_P(TestDistributedOps, DistributedTest, ::testing::ValuesIn(GetOpMetaData<OpMetaData>()));
TEST_P(DistributedTest, TestOps)
{
    config::SetPassOption(ENABLE_SLICE, false);
    RunDistributedTestGeneric(GetParam().testData_, GetParam().fileName_);
}
} // namespace npu::tile_fwk::Distributed
