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
 * \file mix_info.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
namespace mix_info {
int DumpMixInfo(Function* topFunc);

// Profiling-only event ID for force-sync ops in mix_event_info.json.
// Hardware still uses EVENT_ID7 in generated CCE; 256 is outside the normal
// cross-core event id range (0-15) so mix swimlane metadata can distinguish them.
constexpr int MIX_INFO_FORCE_SYNC_EVENT_ID = 256;

int GetMixInfoEventId(const Operation& op);
struct SyncInfo {
    bool isSet;
    int eventID;
};

struct CoreTask {
    uint64_t hashValue;
    std::vector<SyncInfo> syncMsg;
};

struct WrapInfo {
    int wrapID;
    std::vector<CoreTask> coreTask;
};

struct MixInfo {
    uint64_t mixId;
    std::vector<WrapInfo> wrapInfos;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SyncInfo, isSet, eventID)

// 绑定CoreTask
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CoreTask, hashValue, syncMsg)

// 绑定WrapInfo
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WrapInfo, wrapID, coreTask)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MixInfo, mixId, wrapInfos)
} // namespace mix_info
} // namespace tile_fwk
} // namespace npu
