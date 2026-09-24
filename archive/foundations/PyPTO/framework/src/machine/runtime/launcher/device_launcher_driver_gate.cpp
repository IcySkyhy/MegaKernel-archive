/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/launcher/device_launcher_driver_gate.h"

#include <cctype>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "interface/utils/string_utils.h"
#include "machine/device/dynamic/device_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {
namespace {

constexpr size_t kDriverVersionBufSize = 128;
constexpr uint32_t kCheckVersionLength = 4;

std::once_flag gDriverVersionGateOnce;

AscendDriverVersionGate::SysGetVersionStrFn gSysGetVersionStrImpl = AclSysGetVersionStr;

bool SegmentIsAllDigits(const std::string& seg)
{
    if (seg.empty()) {
        return false;
    }
    for (unsigned char c : seg) {
        if (std::isdigit(c) == 0) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace AscendDriverVersionGate {

bool TryParseOfficialDriverThreeLevel(const std::string& ver, int& major, int& minor)
{
    const std::vector<std::string> parts = StringUtils::Split(ver, ".");
    if (parts.size() > kCheckVersionLength) {
        return false;
    }
    if (parts.size() < 2) {
        return false;
    }
    if (!SegmentIsAllDigits(parts[0]) || !SegmentIsAllDigits(parts[1])) {
        return false;
    }
    major = std::stoi(parts[0]);
    minor = std::stoi(parts[1]);
    return true;
}

bool MeetsMinAscendDriverMajorMinor(int maj, int minVer)
{
    return maj > kMinAscendDriverMajor || (maj == kMinAscendDriverMajor && minVer >= kMinAscendDriverMinor);
}

void RunDriverVersionCheck()
{
    const auto arch = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    if (arch == ArchInfo::DAV_3510) {
        return;
    }

    std::string versionBuf(kDriverVersionBufSize, '\0');
    AclError ret = gSysGetVersionStrImpl("driver", versionBuf.data());
    if (ret != ACLRT_SUCCESS) {
        MACHINE_LOGW("AclSysGetVersionStr(\"driver\") failed, ret=%d; skip Ascend driver version gate.", ret);
        return;
    }
    std::string ver(versionBuf.c_str());
    StringUtils::Trim(ver);
    if (ver.empty()) {
        MACHINE_LOGW("Ascend driver version string is empty after trim; skip Ascend driver version gate.");
        return;
    }
    int major = 0;
    int minor = 0;
    if (!TryParseOfficialDriverThreeLevel(ver, major, minor)) {
        MACHINE_LOGW(
            "Ascend driver version \"%s\" is not official three-level release format (x.y.z); skip Ascend driver "
            "version gate.",
            ver.c_str());
        return;
    }
    if (MeetsMinAscendDriverMajorMinor(major, minor)) {
        return;
    }
    MACHINE_LOGE(DevCommonErr::PARAM_INVALID,
                 "Ascend driver version \"%s\" is below required %d.%d for onboard; kernel launch aborted.",
                 ver.c_str(), kMinAscendDriverMajor, kMinAscendDriverMinor);
    ASSERT(DevCommonErr::PARAM_INVALID, false)
        << "Ascend driver package version " << ver << " is not supported for onboard (need >= " << kMinAscendDriverMajor
        << "." << kMinAscendDriverMinor << ").";
}

void SetSysGetVersionStrHookForTest(SysGetVersionStrFn hook)
{
    gSysGetVersionStrImpl = hook != nullptr ? hook : AclSysGetVersionStr;
}

void ClearSysGetVersionStrHookForTest() { gSysGetVersionStrImpl = AclSysGetVersionStr; }

void ResetOnceFlagForTest() { new (&gDriverVersionGateOnce) std::once_flag(); }

void EnsureDriverVersionForOnboardOnce() { std::call_once(gDriverVersionGateOnce, RunDriverVersionCheck); }

} // namespace AscendDriverVersionGate
} // namespace npu::tile_fwk::dynamic
