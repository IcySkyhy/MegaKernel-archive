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
 * \file function.h
 * \brief
 */

#pragma once

#include <string>

#define PROGRAM(name, ...)                                                                             \
    if (auto recordProg = npu::tile_fwk::DefineProg(name, ##__VA_ARGS__); !recordProg.IsRecording()) { \
    } else

namespace npu::tile_fwk {
class DefineProg {
public:
    bool IsRecording() const { return isRecording_; }
    explicit DefineProg(const std::string& name);
    ~DefineProg();

private:
    bool isRecording_;
};
} // namespace npu::tile_fwk
