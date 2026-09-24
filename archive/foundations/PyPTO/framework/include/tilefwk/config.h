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
 * \file config.h
 * \brief
 */

#pragma once
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

namespace npu::tile_fwk {

enum class MachineScheduleConfig {
    /**
     * \brief Default schedule mode: L2CACHE_AFFINITY_SCH(disable) MULTI_CORE_FAIR_SCH(disable)
     */
    DEFAULT_SCH = 0x0,

    /**
     * \brief Dispatch the most recently ready task to maximize cache reuse
     */
    L2CACHE_AFFINITY_SCH = 0x1,

    /**
     * \brief Fair scheduling refers to maintaining as balanced a distribution of tasks across cores as possible,
     *        Enabling this configuration will introduce some additional public scheduling overhead.
     */
    MULTI_CORE_FAIR_SCH = 0x2
};

namespace config {

template <typename T>
void SetOptionsNg(const std::string& key, const T& value);

/**
 * \brief Set pass options
 *
 * \param key config option key
 *  - pg_upper_bound:
 *      upper bound of schedule cycles for each subgraph
 *      default: 512
 *  - pg_lower_bound:
 *      lower bound of schedule cycles for each subgraph
 *      default: 10000
 * \param value config option value
 */
template <typename T>
void SetPassOption(const std::string& key, const T& value)
{
    SetOptionsNg("pass." + key, value);
}

/**
 * \brief Set codegen options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetCodeGenOption(const std::string& key, const T& value)
{
    SetOptionsNg("codegen." + key, value);
}

/**
 * \brief Set runtime options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetRuntimeOption(const std::string& key, const T& value)
{
    SetOptionsNg("runtime." + key, value);
}

/**
 * \brief Set host options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetHostOption(const std::string& key, const T& value)
{
    SetOptionsNg("host." + key, value);
}

/**
 * \brief Set host options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetVerifyOption(const std::string& key, const T& value)
{
    SetOptionsNg("verify." + key, value);
}

/**
 * \brief Set Operation options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetOperationOption(const std::string& key, const T& value)
{
    SetOptionsNg("operation." + key, value);
}

/**
 * \brief Set Debug options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetDebugOption(const std::string& key, const T& value)
{
    SetOptionsNg("debug." + key, value);
}

/**
 * \brief Set tensor print options
 *
 * \param edgeItems print max items in tensor head and tail
 * \param precision print precision
 * \param threshold threshold to use ...
 * \param linewidth max line width
 */
void SetPrintOptions(int edgeItems, int precision, int threshold, int linewidth);

/**
 * \brief Set the Semantic Label object
 *
 * \param label semantic label
 * \note label will be attached to subsequent operations
 */
void SetSemanticLabel(const std::string& label, const char* filename = __builtin_FILE(), int lineno = __builtin_LINE());

/**
 * \brief Set the Build static function or not
 *
 * \param isStatic true: build static function, false: build dynamic function
 */
void SetBuildStatic(bool isStatic);

/**
 * \brief Dump all config options
 *
 * \return std::string config options string
 */
std::string Dump();

/**
 * \brief Reset config options to default values
 */
void Reset();
}; // namespace config

} // namespace npu::tile_fwk
