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
 * \file CommonTools.h
 * \brief
 */

#pragma once

#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include "cost_model/simulation/common/CommonType.h"

#ifdef _WIN32
#include <io.h>
#define DUP _dup
#define DUP2 _dup2
#define FILENO _fileno
#define NULL_DEVICE "NUL"
#else
#include <unistd.h>
#define DUP dup
#define DUP2 dup2
#define FILENO fileno
#define NULL_DEVICE "/dev/null"
#endif

namespace CostModel {

const int PROCESS_ID_OFFSET = 10000;

inline uint64_t GetProcessID(CostModel::MachineType type, size_t sequence)
{
    return (static_cast<uint64_t>(type) * PROCESS_ID_OFFSET) + sequence;
}

inline int GetMachineType(CostModel::Pid pid) { return (pid / PROCESS_ID_OFFSET); }

inline int GetMachineSeq(CostModel::Pid pid) { return (pid % PROCESS_ID_OFFSET); }

class OutputSilencer {
private:
    int saved_stdout;
    bool is_silenced;

public:
    OutputSilencer() : is_silenced(false) { saved_stdout = DUP(FILENO(stdout)); }

    void silence()
    {
        if (is_silenced)
            return;

        int dev_null = open(NULL_DEVICE, O_WRONLY);
        if (dev_null != -1) {
            DUP2(dev_null, FILENO(stdout));
            close(dev_null);
            is_silenced = true;
        }
    }

    void restore()
    {
        if (!is_silenced)
            return;

        fflush(stdout); // 恢复前先清空缓冲区，防止内容错乱
        DUP2(saved_stdout, FILENO(stdout));
        is_silenced = false;
    }

    ~OutputSilencer()
    {
        restore();
        if (saved_stdout != -1) {
            close(saved_stdout);
        }
    }
};
} // namespace CostModel
