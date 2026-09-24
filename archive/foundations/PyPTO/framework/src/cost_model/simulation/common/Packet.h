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
 * \file Packet.h
 * \brief
 */

#pragma once
#include <vector>
#include "cost_model/simulation/common/ISA.h"

namespace CostModel {

struct DataIDType {
    int id;

    bool operator==(const DataIDType& oth) const { return id == oth.id; }
    bool operator<(const DataIDType& oth) const { return id < oth.id; }

    std::string Dump() const;
};

/**
 * For transmitting between machines
 */
struct TaskMeta {
    std::shared_ptr<Task> taskPtr = nullptr;
    int functionMagic = -1;
    uint64_t functionHash = -1;
    int taskId = -1;
    // FunctionHash functionHash;
    std::string functionName = "";
    MachineType coreMachineType = MachineType::UNKNOWN;
};

struct TileOpMeta : TaskMeta {
    int magic = -1;
    TileOpPtr tileOp = nullptr;
    TilePtr tile = nullptr;
};

struct DataMeta {
    int tensorId;
};

// submission queue msg type
struct TaskPack {
    TaskMeta task;
    TileOpMeta tileopTask;
    std::vector<DataMeta> incasts;
    std::vector<DataMeta> outcasts;

    uint64_t taskId;
    std::vector<DataIDType> incastIdsOnParent;
    std::vector<DataIDType> outcastIdsOnParent;
    TaskCycleInfo cycleInfo;
    std::vector<bool> externalDependencyReleased; // True: All dependencies have been pushed to this submachine
};

struct CachePacket {
    uint64_t pid = 0;
    uint64_t tid = 0;
    CachePacketType type = CachePacketType::REQUEST;
    CacheRequestType requestType = CacheRequestType::DATA_READ_REQ;
    uint64_t addr = 0; // functionHash; data address;
    uint64_t size = 0;
    CycleInfo cycleInfo;
};

struct PipeCompletedMsg {
    int magic = 0;
};

struct CompletedPacket {
    uint64_t taskId = 0;
    MachineType currentType = MachineType::TOTAL_MACHINE_TYPE;
    PipeCompletedMsg pipeMsg;
    TaskCycleInfo cycleInfo;
};

} // namespace CostModel
