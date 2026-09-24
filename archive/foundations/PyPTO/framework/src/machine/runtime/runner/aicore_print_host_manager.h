/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_print_host_manager.h
 * \brief Host-side aicore print buffer management: allocates per-core HBM ring
 *        buffers, injects addresses into dfxBuffer[5] via H2D, and after task
 *        sync D2H-copies + decodes the TLV records to host log output.
 *        Functionally isolated — all print-related host logic lives here.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "interface/machine/device/tilefwk/aicpu_common.h"

namespace npu::tile_fwk {

class AicorePrintHostManager {
public:
    AicorePrintHostManager();
    ~AicorePrintHostManager();

    AicorePrintHostManager(const AicorePrintHostManager&) = delete;
    AicorePrintHostManager& operator=(const AicorePrintHostManager&) = delete;

    int Init(const DeviceArgs& args);
    int SetPrintBufferAddrs() const;
    int DumpAicoreLog();
    void Release();

private:
    uint32_t numCores_{0};
    uint8_t* sharedBuffer_{nullptr};
    std::vector<void*> devBuffers_;
    std::vector<uint8_t> hostBuf_;
};

} // namespace npu::tile_fwk
