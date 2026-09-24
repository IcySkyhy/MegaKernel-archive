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
 * \file improvement_scheduler.h
 * \brief
 */

#ifndef OSP_IMPROVEMENT_SCHEDULER_H
#define OSP_IMPROVEMENT_SCHEDULER_H

#include "scheduler.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class ImprovementScheduler
 * @brief Abstract base class for improvement scheduling scheduler.
 *
 * The ImprovementScheduler class provides a common interface for improvement scheduling scheduler.
 * Subclasses of this class can implement specific improvement scheduler by overriding the virtual methods.
 */
template <typename GraphT>
class ImprovementScheduler {
public:
    /**
     * @brief Constructor for ImprovementScheduler.
     * @param timelimit The time limit in seconds for the improvement algorithm. Default is 3600 seconds (1 hour).
     */
    ImprovementScheduler(unsigned timelimit = 3600) : timeLimitSeconds_(timelimit) {}
    virtual ~ImprovementScheduler() = default;

    /**
     * @brief Improve the given BspSchedule.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT>& schedule) = 0;

    /**
     * @brief Improve the given BspSchedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT>& schedule) = 0;

protected:
    unsigned timeLimitSeconds_; /**< The time limit in seconds for the improvement algorithm. */
};

template <typename GraphT>
class ComboScheduler : public Scheduler<GraphT> {
public:
    ComboScheduler(Scheduler<GraphT>& base, ImprovementScheduler<GraphT>& improvement)
        : Scheduler<GraphT>(), baseScheduler_(base), improvementScheduler_(improvement)
    {}

    virtual ~ComboScheduler() = default;

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT>& schedule) override
    {
        ReturnStatus status = baseScheduler_.ComputeSchedule(schedule);
        if (status != ReturnStatus::OSP_SUCCESS and status != ReturnStatus::OSP_BEST_FOUND) {
            return status;
        }

        return improvementScheduler_.ImproveSchedule(schedule);
    }

private:
    Scheduler<GraphT>& baseScheduler_;
    ImprovementScheduler<GraphT>& improvementScheduler_;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_IMPROVEMENT_SCHEDULER_HPP
