/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aicpu_interface.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <dlfcn.h>

#define private public
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#undef private

using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void* args);

TEST(PyptoAicpuInterfaceTest, NullInitArgReturnsError) { EXPECT_EQ(DynPyptoKernelServerNull(nullptr), 1U); }

TEST(PyptoAicpuInterfaceTest, ExecuteBeforeLoadReturnsError)
{
    EXPECT_EQ(DynPyptoKernelServer(nullptr), 1U);
    EXPECT_EQ(DynPyptoKernelServerInit(nullptr), 1U);
}

namespace {
int MockKernelFunc(void*) { return 0; }
} // namespace

class BackendServerHandleManagerTest : public testing::Test {
protected:
    std::string root_;
    std::string soDir_;
    std::string soPath_;

    void SetUp() override
    {
        const auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = "/tmp/aicpu_kernels_" + std::to_string(getpid()) + "_" + testInfo->test_suite_name() + "_" +
                testInfo->name();
        soDir_ = root_ + "/0/aicpu_kernels_device";
        soPath_ = soDir_ + "/libpypto_server0.so";
        std::filesystem::create_directories(soDir_);
    }
    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }
};

TEST_F(BackendServerHandleManagerTest, SaveSoFile_ShortLenAndWriteFail)
{
    BackendServerHandleManager mgr;
    EXPECT_EQ(mgr.SaveSoFile(nullptr, 0), true);
    char data[] = "test_so_data";
    mgr.pyptoServerSoName_ = soPath_;
    EXPECT_EQ(mgr.SaveSoFile(data, sizeof(data) - 1), true);
    EXPECT_EQ(mgr.firstCreatSo_, true);

    BackendServerHandleManager mgrFail;
    mgrFail.firstCreatSo_ = false;
    mgrFail.pyptoServerSoName_ = soPath_;
    std::error_code ec;
    std::filesystem::remove(soPath_, ec);
    std::filesystem::create_symlink("/dev/full", soPath_, ec);
    std::vector<char> bigData(8192, 'A');
    EXPECT_EQ(mgrFail.SaveSoFile(bigData.data(), bigData.size()), false);
}

TEST_F(BackendServerHandleManagerTest, SetTileFwkKernelMap_AlreadyLoaded)
{
    BackendServerHandleManager mgrLoaded;
    mgrLoaded.firstLoadSo_ = true;
    mgrLoaded.SetTileFwkKernelMap();
    EXPECT_EQ(mgrLoaded.soHandle_, nullptr);
}

TEST_F(BackendServerHandleManagerTest, LoadTileFwkKernelFunc_DlsymSuccessAndDuplicate)
{
    BackendServerHandleManager mgr;
    mgr.soHandle_ = dlopen(nullptr, RTLD_LAZY);
    ASSERT_NE(mgr.soHandle_, nullptr);
    mgr.LoadTileFwkKernelFunc(dynServerKernelFun);
    EXPECT_NE(mgr.kernelKey2FuncHandle_.count(dyExecFuncKey), 0U);
    mgr.LoadTileFwkKernelFunc(dynServerKernelFun);
    EXPECT_EQ(mgr.kernelKey2FuncHandle_.size(), 1U);
}

TEST_F(BackendServerHandleManagerTest, Destructor_WithAndWithoutHandle)
{
    {
        BackendServerHandleManager mgr;
    }
    void* handle = dlopen(nullptr, RTLD_LAZY);
    ASSERT_NE(handle, nullptr);
    {
        BackendServerHandleManager mgr;
        mgr.soHandle_ = handle;
    }
    SUCCEED();
}

TEST_F(BackendServerHandleManagerTest, GetTileFwkKernelFunc_HitAndMiss)
{
    BackendServerHandleManager mgr;
    mgr.kernelKey2FuncHandle_[dyExecFuncKey] = MockKernelFunc;
    EXPECT_NE(mgr.GetTileFwkKernelFunc(dyExecFuncKey), nullptr);
    EXPECT_EQ(mgr.GetTileFwkKernelFunc(dyInitFuncKey), nullptr);
}
