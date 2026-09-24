/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP
#define PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP

#include "pto/pto-inst.hpp"
#include "pto/comm/async/rdma/rdma_types.hpp"

namespace pto {
namespace comm {
namespace rdma {

constexpr uint32_t kHandleRankShift = 32;
constexpr uint32_t kErrorHandleRank = 0xffffffffU;

AICORE inline uint64_t EncodeHandle(uint32_t destRankId, uint32_t queueHead)
{
    return (static_cast<uint64_t>(destRankId) << kHandleRankShift) | static_cast<uint64_t>(queueHead);
}

AICORE inline uint64_t EncodeErrorHandle(uint32_t status)
{
    return EncodeHandle(kErrorHandleRank, status == 0 ? kRdmaInvalidWorkspaceError : status);
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t& destRankId, uint32_t& queueHead)
{
    destRankId = static_cast<uint32_t>(handle >> kHandleRankShift);
    queueHead = static_cast<uint32_t>(handle & 0xffffffffULL);
}

AICORE inline bool IsErrorHandle(uint64_t handle)
{
    return static_cast<uint32_t>(handle >> kHandleRankShift) == kErrorHandleRank;
}

AICORE inline bool IsWorkspaceHeaderValid(__gm__ const RdmaInfo* info)
{
    return info != nullptr && info->magic == kRdmaWorkspaceMagic && info->version == kRdmaWorkspaceVersion &&
           info->backend != RdmaBackend::NONE && info->rankCount > 0 && info->qpNum > 0;
}

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP
