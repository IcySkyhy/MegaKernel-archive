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
 * \file aicore_print_host_manager.cpp
 * \brief Host-side aicore print buffer management implementation.
 */

#include "machine/runtime/runner/aicore_print_host_manager.h"
#include "tilefwk/aicore_print_logger.h"
#include "adapter/api/runtime_api.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "machine/runtime/memory_utils/memory_pool.h"

namespace npu::tile_fwk {
AicorePrintHostManager::AicorePrintHostManager() = default;

AicorePrintHostManager::~AicorePrintHostManager() { Release(); }

int AicorePrintHostManager::Init(const DeviceArgs& args)
{
    Release();
    numCores_ = static_cast<uint32_t>(args.GetBlockNum());
    sharedBuffer_ = reinterpret_cast<uint8_t*>(args.sharedBuffer);
    MACHINE_LOGD("AicorePrintHostManager Init: numCores=%u, sharedBuffer=%p", numCores_, sharedBuffer_);
    if (sharedBuffer_ == nullptr || numCores_ == 0) {
        MACHINE_LOGW("AicorePrintHostManager init skipped: sharedBuffer=0 or numCores=0");
        return static_cast<int>(DevCommonErr::PARAM_INVALID);
    }
    for (uint32_t i = 0; i < numCores_; i++) {
        void* devPtr = nullptr;
        if (RuntimeMalloc(&devPtr, PRINT_BUFFER_SIZE, RT_MEMORY_HBM, 0) != RT_SUCCESS) {
            MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to alloc aicore print buffer, size=%lu bytes, coreId=%u",
                         PRINT_BUFFER_SIZE, i);
            Release();
            return static_cast<int>(DevCommonErr::ALLOC_FAILED);
        }
        MACHINE_LOGD("AicorePrintHostManager alloc buffer=%p size=%lu bytes, coreId=%u", devPtr, PRINT_BUFFER_SIZE, i);
        if (RuntimeMemset(devPtr, PRINT_BUFFER_SIZE, 0, PRINT_BUFFER_SIZE) != RT_SUCCESS) {
            MACHINE_LOGE(DevCommonErr::MEMRESET_FAILED,
                         "Failed to memset aicore print buffer, size=%lu bytes, coreId=%u", PRINT_BUFFER_SIZE, i);
            RuntimeFree(devPtr);
            Release();
            return static_cast<int>(DevCommonErr::MEMRESET_FAILED);
        }
        devBuffers_.push_back(devPtr);
    }
    if (SetPrintBufferAddrs() != 0) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "SetPrintBufferAddrs failed, releasing");
        Release();
        return static_cast<int>(DevCommonErr::MEMCPY_FAILED);
    }
    hostBuf_.resize(PRINT_BUFFER_SIZE);
    MACHINE_LOGI("AicorePrintHostManager init success: %u cores, buffer size %lu bytes", numCores_, PRINT_BUFFER_SIZE);
    return 0;
}

int AicorePrintHostManager::SetPrintBufferAddrs() const
{
    int retCode = 0;
    MACHINE_LOGD("SetPrintBufferAddrs: %u cores, sharedBuffer=%p", numCores_, sharedBuffer_);
    for (uint32_t i = 0; i < numCores_; i++) {
        uint64_t devAddr = reinterpret_cast<uint64_t>(devBuffers_[i]);
        uint8_t* dfxBufAddr = sharedBuffer_ + offsetof(KernelArgs, dfxBuffer) +
                              sizeof(uint64_t) * SHAK_BUF_PRINT_BUFFER_INDEX + i * SHARED_BUFFER_SIZE;
        auto ret = NormalizedRtMemcpy(dfxBufAddr, sizeof(uint64_t), &devAddr, sizeof(uint64_t),
                                      RtMemcpyKind::HOST_TO_DEVICE);
        if (ret != RT_SUCCESS) {
            MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "H2D write dfxBuffer failed, core %u, ret %d", i, ret);
            retCode = static_cast<int>(DevCommonErr::MEMCPY_FAILED);
        } else {
            MACHINE_LOGD("SetPrintBufferAddrs core %u: dfxBufAddr=%p devAddr=0x%lx", i, static_cast<void*>(dfxBufAddr),
                         devAddr);
        }
    }
    return retCode;
}

int AicorePrintHostManager::DumpAicoreLog()
{
    if (devBuffers_.empty()) {
        MACHINE_LOGW("DumpAicoreLog: no buffers, skip");
        return 0;
    }
    MACHINE_LOGD("DumpAicoreLog: start, %u cores", numCores_);
    int retCode = 0;
    for (uint32_t i = 0; i < numCores_; i++) {
        auto ret = NormalizedRtMemcpy(hostBuf_.data(), PRINT_BUFFER_SIZE, devBuffers_[i], PRINT_BUFFER_SIZE,
                                      RtMemcpyKind::DEVICE_TO_HOST);
        if (ret != RT_SUCCESS) {
            MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED,
                         "D2H copy aicore print buffer failed, size=%lu bytes, coreId=%u, ret %d", PRINT_BUFFER_SIZE, i,
                         ret);
            retCode = static_cast<int>(DevCommonErr::MEMCPY_FAILED);
            continue;
        }
        AicoreLogger logger;
        logger.BindHostBuffer(hostBuf_.data(), hostBuf_.size());
        constexpr int lineSize = 512;
        char line[lineSize];
        uint8_t level = static_cast<uint8_t>(AicoreLogLevel::NONE);
        uint64_t timestamp = 0;
        while (logger.Read(line, lineSize, &level, &timestamp) > 0) {
            switch (static_cast<AicoreLogLevel>(level)) {
                case AicoreLogLevel::DEBUG:
                    AICORE_HOST_LOGD("[AICORE][Core-%u](time:%lu) %s", i, timestamp, line);
                    break;
                case AicoreLogLevel::WARN:
                    AICORE_HOST_LOGW("[AICORE][Core-%u](time:%lu) %s", i, timestamp, line);
                    break;
                case AicoreLogLevel::ERROR:
                    AICORE_HOST_LOGE("[AICORE][Core-%u](time:%lu) %s", i, timestamp, line);
                    break;
                case AicoreLogLevel::INFO:
                default:
                    AICORE_HOST_LOGI("[AICORE][Core-%u](time:%lu) %s", i, timestamp, line);
                    break;
            }
        }
    }
    MACHINE_LOGD("DumpAicoreLog: done, retCode=%d", retCode);
    return retCode;
}

void AicorePrintHostManager::Release()
{
    for (auto* ptr : devBuffers_) {
        if (ptr != nullptr) {
            RuntimeFree(ptr);
        }
    }
    devBuffers_.clear();
    numCores_ = 0;
    sharedBuffer_ = nullptr;
}

} // namespace npu::tile_fwk
