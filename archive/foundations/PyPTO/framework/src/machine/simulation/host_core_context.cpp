/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/simulation/host_core_context.h"

namespace npu::tile_fwk::dynamic {

namespace {
thread_local HostCoreContext g_ctx{};
}

const HostCoreContext& HostCoreCtx::Current() { return g_ctx; }

void HostCoreCtx::SetCurrent(const HostCoreContext& ctx) { g_ctx = ctx; }

} // namespace npu::tile_fwk::dynamic
