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
 * \file msprof_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/msprof_define.h"

namespace npu::tile_fwk {
uint64_t StubProfSysCycleTime(void);
uint64_t StubProfGetHashId(const char* hashInfo, size_t length);
int32_t StubProfReportApi(uint32_t nonPersistantFlag, const struct MspfApi* api);
int32_t StubProfReportCompactInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length);
int32_t StubProfReportAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length);
int32_t StubProfRegisterCallback(uint32_t moduleId, MspfCommandHandleFunc handle);
} // namespace npu::tile_fwk
