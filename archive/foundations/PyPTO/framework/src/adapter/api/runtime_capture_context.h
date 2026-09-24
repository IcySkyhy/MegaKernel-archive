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
 * \file runtime_capture_context.h
 * \brief Process-wide capture flag and per-thread capture mode query for RuntimeMemcpy checks.
 */

#pragma once

#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
class RuntimeCaptureContext {
public:
    static void SetCaptureMode(bool captureMode);
    static bool IsCaptureMode();

    /// Probe thread capture mode via ACL exchange (no get-only API); restores mode before return.
    static bool QueryThreadCaptureMode(AclMdlRICaptureMode& mode);

    /// Unit test hook: override QueryThreadCaptureMode result.
    static void SetTestThreadCaptureMode(AclMdlRICaptureMode mode, bool enable);
};
} // namespace npu::tile_fwk
