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
 * \file runtime_capture_context.cpp
 * \brief
 */

#include "adapter/api/runtime_capture_context.h"

#include "adapter/api/acl_api.h"

namespace npu::tile_fwk {
namespace {
bool g_captureMode = false;
bool g_testThreadModeOverride = false;
AclMdlRICaptureMode g_testThreadMode = AclMdlRICaptureMode::RELAXED;
} // namespace

void RuntimeCaptureContext::SetCaptureMode(bool captureMode) { g_captureMode = captureMode; }

bool RuntimeCaptureContext::IsCaptureMode() { return g_captureMode; }

bool RuntimeCaptureContext::QueryThreadCaptureMode(AclMdlRICaptureMode& mode)
{
    if (g_testThreadModeOverride) {
        mode = g_testThreadMode;
        return true;
    }

    // ACL has no get-only API for thread capture mode; only AclMdlRICaptureThreadExchangeMode is available.
    // It atomically swaps *mode with the thread's current value: input = new mode, output = old mode (written back into
    // *mode).
    //
    // First exchange: read the old mode into previousMode. The returned value does not depend on the input,
    // but the thread is temporarily set to the input until the restore below. Use RELAXED (not an arbitrary
    // placeholder): rtMemcpy paths require RELAXED in capture, and if restore fails the thread stays in a safe mode
    // (see ChangeCaptureModeRelax).
    AclMdlRICaptureMode previousMode = AclMdlRICaptureMode::RELAXED;
    const AclError exchangeRet = AclMdlRICaptureThreadExchangeMode(&previousMode);
    if (exchangeRet != ACLRT_SUCCESS) {
        return false;
    }
    const AclMdlRICaptureMode currentMode = previousMode;

    // Second exchange: swap back to currentMode so the probe leaves no lasting side effect
    // (same two-call pattern as AclModeGuard in device_launcher.h).
    AclMdlRICaptureMode restoreMode = currentMode;
    const AclError restoreRet = AclMdlRICaptureThreadExchangeMode(&restoreMode);
    if (restoreRet != ACLRT_SUCCESS) {
        return false;
    }
    mode = currentMode;
    return true;
}

void RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode mode, bool enable)
{
    g_testThreadMode = mode;
    g_testThreadModeOverride = enable;
}
} // namespace npu::tile_fwk
