/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-only hcomm ABI definitions for the RoCE (HNS_1825) control plane.
//
// Public hcomm types come from <hcomm/hcomm_res.h>. ChannelEntity / SqContext /
// CqContext / RegedBufferEntity are not in the public include path; they are
// defined here ABI-compatible with hcomm_res_entity_defs.h. PTO supports the
// split-doorbell and vendor-specified RoCE context layouts used by
// HNS1825; legacy HCOMM V1 contexts are outside this backend's scope.

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_HCOMM_DEFS_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_HCOMM_DEFS_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_hcomm_defs.hpp is a host-only header and cannot be included in device code."
#endif

#include <hcomm/hcomm_res.h>

#include <cstddef>
#include <cstdint>

#include "securec.h"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {
namespace host {

enum ProtectionType {
    PROTECTION_TYPE_RESERVED = -1,
    PROTECTION_TYPE_ROCE = 0,
    PROTECTION_TYPE_UB = 1,
};

enum RegedBufferType {
    REGED_BUFFER_TYPE_RESERVED = -1,
    REGED_BUFFER_TYPE_IPC = 0,
    REGED_BUFFER_TYPE_RMA = 1,
};

enum RegedNotifyType {
    REGED_NOTIFY_TYPE_RESERVED = -1,
    REGED_NOTIFY_TYPE_IPC_RT = 0,
    REGED_NOTIFY_TYPE_IPC_MEM = 1,
    REGED_NOTIFY_TYPE_RMA_RT = 2,
    REGED_NOTIFY_TYPE_RMA_MEM = 3,
};

enum SqContextType {
    SQ_CONTEXT_TYPE_RESERVED = -1,
    SQ_CONTEXT_TYPE_UB_JFS = 0,
    SQ_CONTEXT_TYPE_ROCE = 1,
};

enum CqContextType {
    CQ_CONTEXT_TYPE_RESERVED = -1,
    CQ_CONTEXT_TYPE_UB_JFC = 0,
    CQ_CONTEXT_TYPE_ROCE = 1,
};

struct ProtectionInfo {
    ProtectionType type;
    union {
        struct {
            uint32_t lkey;
            uint32_t rkey;
        } roce;
        struct {
            uint32_t tokenId;
            uint32_t tokenValue;
        } ub;
        uint8_t raws[24];
    } memInfo;
};

struct RegedBufferEntity {
    RegedBufferType type;
    union {
        struct {
            uint64_t addr;
            uint64_t size;
        } ipc;
        struct {
            uint64_t addr;
            uint64_t size;
            ProtectionInfo protectionInfo;
        } rma;
        uint8_t raws[56];
    } bufferInfo;
};

struct RegedNotifyEntity {
    RegedNotifyType type;
    union {
        struct {
            uint64_t addr;
            uint32_t size;
            int32_t notifyId;
        } ipcRt;
        struct {
            uint64_t addr;
            uint32_t size;
        } ipcMem;
        struct {
            uint64_t addr;
            uint32_t size;
            int32_t notifyId;
            ProtectionInfo protectionInfo;
        } rmaRt;
        struct {
            uint64_t addr;
            uint32_t size;
            ProtectionInfo protectionInfo;
        } rmaMem;
        uint8_t raws[56];
    } notifyInfo;
};

// HCOMM has shipped both a split-doorbell layout and a vendor-specified QoS
// layout without adding an ABI tag to these private contexts. Keep explicit
// byte-layout views and decode from contextInfo.raws below.
#define PTO_HNS1825_ROCE_SQ_CONTEXT_COMMON_FIELDS \
    uint64_t sqVa;                                \
    uint64_t headAddr;                            \
    uint64_t tailAddr;                            \
    uint64_t dbHwVa;                              \
    uint64_t dbSwVa;                              \
    uint32_t qpn;                                 \
    uint32_t wqeSize;                             \
    uint32_t depth;                               \
    uint8_t sl

struct RoceSqContextSplitDb {
    PTO_HNS1825_ROCE_SQ_CONTEXT_COMMON_FIELDS;
    uint8_t mtuShift;
};

struct RoceSqContextVendorSpecified {
    PTO_HNS1825_ROCE_SQ_CONTEXT_COMMON_FIELDS;
    uint64_t DbVendorSpecified;
};

#undef PTO_HNS1825_ROCE_SQ_CONTEXT_COMMON_FIELDS

#define PTO_HNS1825_ROCE_CQ_CONTEXT_COMMON_FIELDS \
    uint64_t cqVa;                                \
    uint64_t headAddr;                            \
    uint64_t tailAddr;                            \
    uint64_t dbHwVa;                              \
    uint64_t dbSwVa;                              \
    uint32_t cqn;                                 \
    uint32_t cqeSize;                             \
    uint32_t cqDepth

struct RoceCqContextSplitDb {
    PTO_HNS1825_ROCE_CQ_CONTEXT_COMMON_FIELDS;
};

struct RoceCqContextVendorSpecified {
    PTO_HNS1825_ROCE_CQ_CONTEXT_COMMON_FIELDS;
    uint64_t DbVendorSpecified;
};

#undef PTO_HNS1825_ROCE_CQ_CONTEXT_COMMON_FIELDS

struct SqContext {
    SqContextType type;
    union {
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfsID;
            uint32_t wqeSize;
            uint32_t sqDepth;
            uint32_t tpID;
            uint8_t remoteEID[16];
        } ubJfs;
        RoceSqContextVendorSpecified roceSq;
        uint8_t raws[120];
    } contextInfo;
};

struct CqContext {
    CqContextType type;
    union {
        struct {
            uint64_t scqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfcID;
            uint32_t cqeSize;
            uint32_t cqDepth;
        } ubJfc;
        RoceCqContextVendorSpecified roceCq;
        uint8_t raws[120];
    } contextInfo;
};

static_assert(offsetof(RoceSqContextSplitDb, mtuShift) == 53, "unexpected split-db SQ context layout");
static_assert(
    offsetof(RoceSqContextVendorSpecified, DbVendorSpecified) == 56, "unexpected vendor-specified SQ context layout");
static_assert(
    offsetof(RoceCqContextVendorSpecified, DbVendorSpecified) == 56, "unexpected vendor-specified CQ context layout");

template <typename RoceContext, typename Context>
inline RoceContext ExtractRoceContext(const Context& context)
{
    RoceContext roce{};
    static_assert(sizeof(roce) <= sizeof(context.contextInfo.raws), "ROCE context view is too large");
    (void)memcpy_s(&roce, sizeof(roce), context.contextInfo.raws, sizeof(roce));
    return roce;
}

inline RoceSqContextSplitDb ExtractRoceSqContextSplitDb(const SqContext& context)
{
    return ExtractRoceContext<RoceSqContextSplitDb>(context);
}

inline RoceSqContextVendorSpecified ExtractRoceSqContextVendorSpecified(const SqContext& context)
{
    return ExtractRoceContext<RoceSqContextVendorSpecified>(context);
}

inline RoceCqContextSplitDb ExtractRoceCqContextSplitDb(const CqContext& context)
{
    return ExtractRoceContext<RoceCqContextSplitDb>(context);
}

struct ChannelEntity {
    CommAbiHeader abiHeader;
    CommEngine engine;
    CommProtocol protocol;
    uint32_t localNotifyNum;
    uint32_t remoteNotifyNum;
    uint32_t localBufferNum;
    uint32_t remoteBufferNum;
    uint32_t sqNum;
    uint32_t cqNum;
    RegedNotifyEntity* localNotifyAddr;
    RegedNotifyEntity* remoteNotifyAddr;
    RegedBufferEntity* localBufferAddr;
    RegedBufferEntity* remoteBufferAddr;
    SqContext* sqContextAddr;
    CqContext* cqContextAddr;
    uint8_t reserve[160];
};

static_assert(sizeof(ProtectionInfo) == 28, "unexpected hcomm ProtectionInfo ABI");
static_assert(sizeof(RegedBufferEntity) == 64, "unexpected hcomm RegedBufferEntity ABI");
static_assert(sizeof(RegedNotifyEntity) == 64, "unexpected hcomm RegedNotifyEntity ABI");
static_assert(sizeof(SqContext) == 128, "unexpected hcomm SqContext ABI");
static_assert(sizeof(CqContext) == 128, "unexpected hcomm CqContext ABI");
static_assert(sizeof(ChannelEntity) == 256, "unexpected hcomm ChannelEntity ABI");
static_assert(offsetof(SqContext, contextInfo) == 8, "unexpected hcomm SqContext union offset");
static_assert(offsetof(CqContext, contextInfo) == 8, "unexpected hcomm CqContext union offset");
static_assert(offsetof(ChannelEntity, localNotifyAddr) == 48, "unexpected hcomm ChannelEntity pointer offset");

} // namespace host
} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_HCOMM_DEFS_HPP
