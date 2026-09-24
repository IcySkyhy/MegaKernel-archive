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
 * \file test_device_launcher_driver_gate.cpp
 * \brief UT for Ascend driver version gate (DeviceLauncher::CheckAscendDriverVersionOnboard).
 */

#include <gtest/gtest.h>

#include "securec.h"

#include "adapter/api/acl_api.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_driver_gate.h"
#include "tilefwk/error.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk::dynamic::AscendDriverVersionGate;

namespace {

constexpr size_t kDriverVersionBufSize = 128;

AclError WriteVersionToBuf(const char* pkgName, char* versionStr, const char* version)
{
    (void)pkgName;
    if (versionStr == nullptr) {
        return 1;
    }
    if (version == nullptr) {
        versionStr[0] = '\0';
        return ACLRT_SUCCESS;
    }
    if (strncpy_s(versionStr, kDriverVersionBufSize, version, kDriverVersionBufSize - 1) != EOK) {
        versionStr[0] = '\0';
        return 1;
    }
    return ACLRT_SUCCESS;
}

AclError FailGetVersion(const char* pkgName, char* versionStr)
{
    (void)pkgName;
    (void)versionStr;
    return 1;
}

AclError HookVersion25_5_1(const char* pkgName, char* versionStr)
{
    return WriteVersionToBuf(pkgName, versionStr, "25.5.1");
}

AclError HookVersion25_4_0(const char* pkgName, char* versionStr)
{
    return WriteVersionToBuf(pkgName, versionStr, "25.4.0");
}

AclError HookVersionNonOfficial(const char* pkgName, char* versionStr)
{
    return WriteVersionToBuf(pkgName, versionStr, "7.0.t9.0.b811");
}

AclError HookVersionEmpty(const char* pkgName, char* versionStr) { return WriteVersionToBuf(pkgName, versionStr, ""); }

int gSysGetVersionCallCount = 0;

AclError CountingVersionHook(const char* pkgName, char* versionStr)
{
    ++gSysGetVersionCallCount;
    return HookVersion25_5_1(pkgName, versionStr);
}

int gDav3510HookCallCount = 0;

AclError Dav3510ProbeHook(const char* pkgName, char* versionStr)
{
    ++gDav3510HookCallCount;
    return WriteVersionToBuf(pkgName, versionStr, "25.5.1");
}

class DriverVersionGateTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        savedArch_ = Platform::Instance().GetSoc().GetNPUArch();
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
        ClearSysGetVersionStrHookForTest();
        ResetOnceFlagForTest();
        gSysGetVersionCallCount = 0;
        gDav3510HookCallCount = 0;
    }

    void TearDown() override
    {
        ClearSysGetVersionStrHookForTest();
        Platform::Instance().GetSoc().SetNPUArch(savedArch_);
        ResetOnceFlagForTest();
    }

    NPUArch savedArch_{NPUArch::DAV_UNKNOWN};
};

} // namespace

TEST_F(DriverVersionGateTest, TryParseOfficialDriverThreeLevel_AcceptsUpToFourSegments)
{
    int major = 0;
    int minor = 0;
    EXPECT_TRUE(TryParseOfficialDriverThreeLevel("25.5.1", major, minor));
    EXPECT_EQ(major, 25);
    EXPECT_EQ(minor, 5);

    EXPECT_TRUE(TryParseOfficialDriverThreeLevel("26.6.rc1.1", major, minor));
    EXPECT_EQ(major, 26);
    EXPECT_EQ(minor, 6);
}

TEST_F(DriverVersionGateTest, TryParseOfficialDriverThreeLevel_RejectsTooManySegments)
{
    int major = 0;
    int minor = 0;
    EXPECT_FALSE(TryParseOfficialDriverThreeLevel("7.0.t9.0.b811", major, minor));
}

TEST_F(DriverVersionGateTest, TryParseOfficialDriverThreeLevel_RejectsNonNumericMajorMinor)
{
    int major = 0;
    int minor = 0;
    EXPECT_FALSE(TryParseOfficialDriverThreeLevel("25.rc.1", major, minor));
}

TEST_F(DriverVersionGateTest, MeetsMinAscendDriverMajorMinor_Boundary)
{
    EXPECT_TRUE(MeetsMinAscendDriverMajorMinor(25, 5));
    EXPECT_TRUE(MeetsMinAscendDriverMajorMinor(25, 6));
    EXPECT_TRUE(MeetsMinAscendDriverMajorMinor(26, 0));
    EXPECT_FALSE(MeetsMinAscendDriverMajorMinor(25, 4));
    EXPECT_FALSE(MeetsMinAscendDriverMajorMinor(24, 99));
}

TEST_F(DriverVersionGateTest, RunGateBody_SkipsOnDav3510WithoutQueryingDriver)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    SetSysGetVersionStrHookForTest(Dav3510ProbeHook);
    EXPECT_NO_THROW(RunDriverVersionCheck());
    EXPECT_EQ(gDav3510HookCallCount, 0);
}

TEST_F(DriverVersionGateTest, RunGateBody_SkipsWhenAclFails)
{
    SetSysGetVersionStrHookForTest(FailGetVersion);
    EXPECT_NO_THROW(RunDriverVersionCheck());
}

TEST_F(DriverVersionGateTest, RunGateBody_SkipsWhenVersionEmpty)
{
    SetSysGetVersionStrHookForTest(HookVersionEmpty);
    EXPECT_NO_THROW(RunDriverVersionCheck());
}

TEST_F(DriverVersionGateTest, RunGateBody_SkipsWhenNonOfficialFormat)
{
    SetSysGetVersionStrHookForTest(HookVersionNonOfficial);
    EXPECT_NO_THROW(RunDriverVersionCheck());
}

TEST_F(DriverVersionGateTest, RunGateBody_PassesWhenVersionMeetsMinimum)
{
    SetSysGetVersionStrHookForTest(HookVersion25_5_1);
    EXPECT_NO_THROW(RunDriverVersionCheck());
}

TEST_F(DriverVersionGateTest, RunGateBody_ThrowsWhenVersionBelowMinimum)
{
    SetSysGetVersionStrHookForTest(HookVersion25_4_0);
    // Host UT: ASSERT maps to throw Error (see tilefwk/error.h), not abort — EXPECT_DEATH is invalid here.
    EXPECT_THROW(RunDriverVersionCheck(), Error);
}

TEST_F(DriverVersionGateTest, EnsureAscendDriverVersionForOnboardOnce_UsesCallOnce)
{
    SetSysGetVersionStrHookForTest(CountingVersionHook);
    DeviceLauncher::CheckAscendDriverVersionOnboard();
    DeviceLauncher::CheckAscendDriverVersionOnboard();
    EXPECT_EQ(gSysGetVersionCallCount, 1);
}
