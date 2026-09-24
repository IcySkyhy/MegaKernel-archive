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
 * \file scheduler.h
 * \brief
 */

#ifndef OSP_SCHEDULER_H
#define OSP_SCHEDULER_H

#include <string>

#include "passes/algorithms/osp/bsp/model/bsp_instance.h"
#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"

namespace npu::tile_fwk {
namespace osp {
enum class ReturnStatus { OSP_SUCCESS, OSP_BEST_FOUND, OSP_ERROR };

/**
 * @class Scheduler
 * @brief Interface for BSP schedulers.
 *
 * The Scheduler class defines the common interface for all scheduling algorithms computing BSP schedules.
 * It specifies the contract for computing standard BSP schedules (BspSchedule) and communication-aware schedules
 * (BspScheduleCS).
 */
template <typename GraphT>
class Scheduler {
public:
    Scheduler() = default;
    virtual ~Scheduler() = default;

    /**
     * @brief Computes a BSP schedule for the given BSP instance.
     *
     * This pure virtual function must be implemented by derived classes to provide
     * the specific scheduling logic. It modifies the passed BspSchedule object.
     *
     * @param schedule The BspSchedule object to be computed. It contains the BspInstance.
     * @return ReturnStatus::OSP_SUCCESS if a schedule was successfully computed,
     *         ReturnStatus::OSP_ERROR if an error occurred, or other status codes as appropriate.
     */
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT>& schedule) = 0;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SCHEDULER_HPP
