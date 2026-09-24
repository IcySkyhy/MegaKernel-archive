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
 * \file CommonData.h
 * \brief
 */

// Chrome trace format reference:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0

#pragma once

namespace CostModel {

using TimeStamp = uint64_t;
using Pid = uint64_t;
using Tid = uint64_t;

struct PTid {
    Pid pid = -1;
    Tid tid = -1;

    bool operator==(const PTid& oth) const { return pid == oth.pid && tid == oth.tid; }
    bool operator!=(const PTid& oth) const { return !(*this == oth); }

    bool operator<(const PTid& oth) const { return pid != oth.pid ? pid < oth.pid : tid < oth.tid; }
};

struct EventId {
    PTid ptid;    // by default { -1, -1 }
    int eid = -1; // event id

    bool operator==(const EventId& oth) const { return ptid == oth.ptid && eid == oth.eid; }
    bool operator!=(const EventId& oth) const { return !(*this == oth); }

    bool operator<(const EventId& oth) const { return ptid != oth.ptid ? ptid < oth.ptid : eid < oth.eid; }

    bool Valid() const { return *this != EventId{}; }
};

struct LogData {
    bool isLogTileOp = false;
    std::string name = "";
    Pid pid = 0;
    Tid tid = 0;
    TimeStamp sTime = 0;
    TimeStamp eTime = 0;
    std::string hint = "";
};
} // namespace CostModel
