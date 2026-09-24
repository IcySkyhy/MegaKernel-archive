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
 * \file schema.h
 * \brief
 */

#pragma once
#ifndef SCHEMA_TRACE_H
#define SCHEMA_TRACE_H

#include "schema_base.h"
#include "schema_parser.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::schema {

template <typename Ty0>
static inline std::string DumpAttr(const Ty0& arg0)
{
    return arg0.Dump();
}

template <typename Ty0, typename... Tys>
static inline std::string DumpAttr(const Ty0& arg0, const Tys&... args)
{
    std::string tail = DumpAttr(args...);
    std::string head = arg0.Dump();
    return head + " " + tail;
}

#include "schema_def_common.h"
#include "schema_def_attr.h"
#include "schema_def_trace.h"

static inline range Range(uint64_t begin, uint64_t end) { return range(begin, end, end - begin); }

#define DEV_TRACE_PREFIX "#trace:"

#define DEV_TRACE_DEBUG(arg, args...)                                             \
    do {                                                                          \
        using namespace npu::tile_fwk::schema;                                    \
        DEV_VERBOSE_DEBUG(DEV_TRACE_PREFIX " %s", DumpAttr(arg, ##args).c_str()); \
    } while (0)
#define DEV_TRACE_INFO(arg, args...)                                     \
    do {                                                                 \
        using namespace npu::tile_fwk::schema;                           \
        DEV_INFO(DEV_TRACE_PREFIX " %s", DumpAttr(arg, ##args).c_str()); \
    } while (0)
#define DEV_TRACE_WARN(arg, args...)                                     \
    do {                                                                 \
        using namespace npu::tile_fwk::schema;                           \
        DEV_WARN(DEV_TRACE_PREFIX " %s", DumpAttr(arg, ##args).c_str()); \
    } while (0)
#define DEV_TRACE_ERROR(arg, args...)                                                                       \
    do {                                                                                                    \
        using namespace npu::tile_fwk::schema;                                                              \
        DEV_ERROR(DevCommonErr::PARAM_CHECK_FAILED, DEV_TRACE_PREFIX " %s", DumpAttr(arg, ##args).c_str()); \
    } while (0)
#define DEV_TRACE_DEBUG_SPLIT(arg, args...)                                     \
    do {                                                                        \
        using namespace npu::tile_fwk::schema;                                  \
        DEV_DEBUG_SPLIT(DEV_TRACE_PREFIX " %s", DumpAttr(arg, ##args).c_str()); \
    } while (0)
} // namespace npu::tile_fwk::schema

#endif // SCHEMA_TRACE_H
