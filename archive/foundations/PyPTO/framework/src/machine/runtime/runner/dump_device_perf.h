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
 * \file dump_device_perf.h
 * \brief
 */

#ifndef DUMP_DEVICE_PERF_H
#define DUMP_DEVICE_PERF_H

#include <vector>
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
void DumpAicoreTaskExectInfo(const DeviceArgs& args, const std::vector<void*>& perfData);
void DumpAicpuPerfInfo(DeviceArgs& args, const std::vector<void*>& perfData, uint32_t freq, bool isLast);
void DumpDevTaskPerfData(DeviceArgs& args, const std::vector<void*>& perfData, bool isLast);
} // namespace npu::tile_fwk::dynamic
#endif
