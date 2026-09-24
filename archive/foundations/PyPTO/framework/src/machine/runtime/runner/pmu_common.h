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
 * \file pmu_common.h
 * \brief
 */

#ifndef SRC_MACHINE_RUNTIME_PMU_COMMON_H
#define SRC_MACHINE_RUNTIME_PMU_COMMON_H

#include <vector>
#include <cstdint>
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {
class PmuCommon {
public:
    static void InitPmuEventType(const ArchInfo& archInfo, std::vector<int64_t>& pmuEvtType);
};
} // namespace npu::tile_fwk
#endif // SRC_MACHINE_RUNTIME_PMU_COMMON_H
