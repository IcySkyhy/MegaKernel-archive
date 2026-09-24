/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file slog_stub.cpp
 * \brief
 */

#ifdef __DEVICE__
#include "dlog_pub.h"

#include <stdarg.h>
#include <stdio.h>
static int log_level = DLOG_ERROR;

void dav_log([[maybe_unused]] int module_id, [[maybe_unused]] const char* fmt, ...) {}

void DlogRecord([[maybe_unused]] int moduleId, int level, [[maybe_unused]] const char* fmt, ...)
{
    if (log_level > level) {
        return;
    }
}

void DlogErrorInner([[maybe_unused]] int module_id, [[maybe_unused]] const char* fmt, ...)
{
    if (log_level > DLOG_ERROR) {
        return;
    }
}

void DlogWarnInner([[maybe_unused]] int module_id, [[maybe_unused]] const char* fmt, ...)
{
    if (log_level > DLOG_WARN) {
        return;
    }
}

void DlogInfoInner([[maybe_unused]] int module_id, [[maybe_unused]] const char* fmt, ...)
{
    if (log_level > DLOG_INFO) {
        return;
    }
}

void DlogDebugInner([[maybe_unused]] int module_id, [[maybe_unused]] const char* fmt, ...)
{
    if (log_level > DLOG_DEBUG) {
        return;
    }
}

void DlogEventInner([[maybe_unused]] int module_id, const char* fmt, ...) { dav_log(module_id, fmt); }

void DlogInner(int module_id, [[maybe_unused]] int level, const char* fmt, ...) { dav_log(module_id, fmt); }

int dlog_setlevel([[maybe_unused]] int module_id, int level, [[maybe_unused]] int enable_event)
{
    log_level = level;
    return log_level;
}

int dlog_getlevel([[maybe_unused]] int module_id, [[maybe_unused]] int* enable_event) { return log_level; }

int CheckLogLevel([[maybe_unused]] int moduleId, int log_level_check) { return log_level_check >= log_level; }

/**
 * @ingroup plog
 * @brief DlogReportInitialize: init log in service process before all device setting.
 * @return: 0: SUCCEED, others: FAILED
 */
int DlogReportInitialize() { return 0; }

/**
 * @ingroup plog
 * @brief DlogReportFinalize: release log resource in service process after all device reset.
 * @return: 0: SUCCEED, others: FAILED
 */
int DlogReportFinalize() { return 0; }

#endif
