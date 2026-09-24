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
 * \file stream_context.h
 * \brief
 */

#pragma once

#include "adapter/api/runtime_api.h"

namespace npu::tile_fwk {
class StreamContext {
public:
    StreamContext()
    {
        RuntimeStreamCreate(&aicoreStream_, RT_STREAM_PRIORITY_DEFAULT);
        RuntimeStreamCreate(&scheStream_, RT_STREAM_PRIORITY_DEFAULT);
        RuntimeStreamCreate(&ctrlStream_, RT_STREAM_PRIORITY_DEFAULT);
    }

    ~StreamContext()
    {
        RuntimeStreamDestroy(aicoreStream_);
        RuntimeStreamDestroy(scheStream_);
        RuntimeStreamDestroy(ctrlStream_);
    }

    RtStream& GetAiCoreStream() { return aicoreStream_; }

    RtStream& GetScheStream() { return scheStream_; }

    RtStream& GetCtrlStream() { return ctrlStream_; }

    RtStream& GetCurrentStream() { return currentStream_; }

    void SetCurrentStream(const RtStream& stream) { currentStream_ = stream; }

private:
    RtStream aicoreStream_{nullptr};
    RtStream ctrlStream_{nullptr};
    RtStream scheStream_{nullptr};
    RtStream currentStream_{nullptr};
};
StreamContext& GetStreamContext();
RtStream GetContextAiCoreStream();
RtStream GetContextScheStream();
RtStream GetContextCtrlStream();
} // namespace npu::tile_fwk
