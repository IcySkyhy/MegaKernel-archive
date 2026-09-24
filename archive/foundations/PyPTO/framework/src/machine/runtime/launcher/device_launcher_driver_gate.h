/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_launcher_driver_gate.h
 * \brief Ascend driver version gate helpers (onboard launch path).
 */

#pragma once

#include <string>

#include "adapter/api/acl_api.h"

namespace npu::tile_fwk::dynamic {

constexpr int kMinAscendDriverMajor = 25;
constexpr int kMinAscendDriverMinor = 5;

namespace AscendDriverVersionGate {

/** Official release: at most four dot-separated segments; first two numeric major/minor. */
bool TryParseOfficialDriverThreeLevel(const std::string& ver, int& major, int& minor);

bool MeetsMinAscendDriverMajorMinor(int maj, int minVer);

/** Run driver version gate once (no std::call_once); used by DeviceLauncher and UT. */
void RunDriverVersionCheck();

using SysGetVersionStrFn = AclError (*)(const char* pkgName, char* versionStr);

void SetSysGetVersionStrHookForTest(SysGetVersionStrFn hook);
void ClearSysGetVersionStrHookForTest();

void ResetOnceFlagForTest();

void EnsureDriverVersionForOnboardOnce();

} // namespace AscendDriverVersionGate
} // namespace npu::tile_fwk::dynamic
