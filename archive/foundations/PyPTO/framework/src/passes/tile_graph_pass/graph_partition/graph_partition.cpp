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
 * \file graph_partitioner.cpp
 * \brief
 */

#include "graph_partition.h"
#include <algorithm>
#include "interface/function/function.h"
#include "passes/pass_check/iso_partitioner_checker.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

namespace {

Status RunIsoPartition(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start GraphPartition. Mode: IsoPartitioner.");
    IsoPartitioner partitioner;
    if (partitioner.SetParameter(function.paramConfigs_.sgParallelNum, function.paramConfigs_.sgPgLowerBound, true) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Config, "Set parameters of GraphPartition failed.");
        return FAILED;
    }
    if (partitioner.PartitionGraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "GraphPartition failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End GraphPartition.");
    return SUCCESS;
}

Status RunOspPartition(Function& function, const std::string& partitionMode)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start GraphPartition. Mode: %s", partitionMode.c_str());
    OspMode mode = (partitionMode == "OspBsp") ? OspMode::MERKLEBSP : OspMode::SARKAR;

    OspPartitioner partitioner(mode);
    if (partitioner.SetParameter(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Config, "Set parameters of GraphPartition failed.");
        return FAILED;
    }
    if (partitioner.PartitionGraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "GraphPartitionOSP failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End GraphPartitionOSP.");
    return SUCCESS;
}

} // namespace

Status GraphPartition::RunOnFunction(Function& function)
{
    const std::string partitionMode = function.paramConfigs_.sgPartitionAlgorithm;

    if (partitionMode == "Iso") {
        return RunIsoPartition(function);
    } else if (partitionMode == "OspSarkar" || partitionMode == "OspBsp") {
        return RunOspPartition(function, partitionMode);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid partition mode.");
        return FAILED;
    }
}

Status GraphPartition::PreCheck(Function& function)
{
    GraphPartitionChecker checker;
    return checker.DoPreCheck(function);
}

Status GraphPartition::PostCheck(Function& function)
{
    GraphPartitionChecker checker;
    return checker.DoPostCheck(function);
}

} // namespace npu::tile_fwk
