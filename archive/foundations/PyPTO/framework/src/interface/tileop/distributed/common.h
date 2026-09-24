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
 * \file common.h
 * \brief
 */

#ifndef DISTRIBUTED_COMMON_H
#define DISTRIBUTED_COMMON_H

#include "../tileop_common.h"
#include "comm_context.h"

#define PIPE_SYNC_EVENT(from, to, eventId)  \
    do {                                    \
        set_flag((from), (to), (eventId));  \
        wait_flag((from), (to), (eventId)); \
    } while (0)

namespace TileOp::Distributed {
enum class AtomicType { SET, ADD };

struct CopyParams {
    uint16_t nBurst;
    uint16_t lenBurst;
    uint16_t srcStride;
    uint16_t dstStride;
};

constexpr uint32_t
    ATOMIC_ADD_BLOCK_BYTE_SIZE = 32; // AtomicAdd 每次操作 32B 的数据，对同一 32B 的数据进行 AtomicAdd 需要排队
constexpr uint32_t FLAG_BYTE_SIZE = ATOMIC_ADD_BLOCK_BYTE_SIZE *
                                    4; // 为了消除 AtomicAdd 并发，以 32B 为最小单位，视情况调节每个 flag 占用的字节数
constexpr uint32_t MOE_COMBINE_SIGNAL_OFFSET = 512 / sizeof(int32_t); // 每 512B 放一个 signal，避免同地址访问性能下降
constexpr uint32_t MOE_COMBINE_INFO_NUM = 3; // combine info 每行有 3 个元素：rankId, tokenId, kOffset
constexpr uint16_t COPY_BLOCK_BYTE_SIZE = 32;
constexpr uint16_t VECTOR_INSTRUCTION_BYTE_SIZE = 256;

#define GM_ADDR __gm__ uint8_t*
#define UB_ADDR __ubuf__ uint8_t*

struct DataCopyParams {
    uint8_t sid;
    uint16_t nBurst;
    uint16_t lenBurst;
    uint16_t srcStride;
    uint16_t dstStride;
};

struct GatherMaskParams {
    uint16_t repeat;
    uint8_t src0BlockStride;
    uint8_t patternMode;
    uint16_t src0RepeatStride;
    uint8_t src1RepeatStride;
};

struct SumParams {
    uint8_t repeat;
    uint16_t dstRepeatStride;
    uint16_t srcBlockStride;
    uint16_t srcRepeatStride;
};

constexpr uint32_t MASK_SELECT_SEND_FLAG = 0x1010101;      // 每 8 个数取第一个
constexpr uint32_t MASK_SELECT_SEND_COUNT = 0x2020202;     // 每 8 个数取第二个
constexpr uint32_t MASK_SELECT_RECV_TOKEN_CNT = 0x1010101; // 每 8 个数取第一个

template <typename T>
constexpr TILEOP T AlignUp(const T value, const T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

TILEOP void DevWinLog(__gm__ int64_t* hcclContext, __ubuf__ uint8_t* tmpBuf, size_t len, size_t offset = 0)
{
    pipe_barrier(PIPE_ALL);
    __gm__ CommContext* winContext = (__gm__ CommContext*)(hcclContext[0]);
    GM_ADDR winBaseAddr = (GM_ADDR)(winContext->winAddr[winContext->debugIndex + winContext->rankId]);
    GM_ADDR dstWinGMAddr = winBaseAddr + offset;
    int32_t lenBurst = AlignUp<int32_t>(len, 32) / 32;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(dstWinGMAddr, tmpBuf, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

TILEOP void DevWinLog(__gm__ int64_t* hcclContext, __gm__ uint8_t* srcGm, __ubuf__ uint8_t* tmpBuf, size_t len,
                      size_t offset = 0)
{
    pipe_barrier(PIPE_ALL);
    __gm__ CommContext* winContext = (__gm__ CommContext*)(hcclContext[0]);
    GM_ADDR winBaseAddr = (GM_ADDR)(winContext->winAddr[winContext->debugIndex + winContext->rankId]);
    GM_ADDR dstWinGMAddr = winBaseAddr + offset;
    int32_t lenBurst = AlignUp<int32_t>(len, 32) / 32;
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    copy_gm_to_ubuf(tmpBuf, srcGm, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(dstWinGMAddr, tmpBuf, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
TILEOP void SetAttomicType()
{
    if constexpr (std::is_same_v<T, float>) {
        set_atomic_f32();
    } else if constexpr (std::is_same_v<T, half>) {
        set_atomic_f16();
    } else if constexpr (std::is_same_v<T, int16_t>) {
        set_atomic_s16();
    } else if constexpr (std::is_same_v<T, int32_t>) {
        set_atomic_s32();
    } else if constexpr (std::is_same_v<T, int8_t>) {
        set_atomic_s8();
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        set_atomic_bf16();
    }
}

struct DispatchInfo {
    int tileIndex;
    int groupIndex;
    int rowPerRank;
    int colPerRank;
    int rankShape;
    int rankOffset;
    int rowShape;
    int rowOffset;
    int colShape;
    int colOffset;
    int totalTileNum;
    int shareRankCnt;
    int expertNumPerRank;
    int rankNum;
    int expertIndex;
};

template <typename T, uint32_t memType = 0>
TILEOP __gm__ T* MapVirtualAddr(__gm__ int64_t* hcclContext, __gm__ T* vAddr, uint32_t dstRankId)
{
    uint64_t addrVal = (uint64_t)vAddr;
    uint64_t groupIndex = TileOp::Distributed::DecodeShmemAddrGroupIndex(addrVal);
    uint64_t offset = TileOp::Distributed::DecodeShmemAddrOffset(addrVal);
    __gm__ TileOp::CommContext* commCtxParam = (__gm__ TileOp::CommContext*)hcclContext[groupIndex];
    if constexpr (memType == 0) {
        return (__gm__ T*)(commCtxParam->winAddr[dstRankId] + offset);
    } else {
        return (__gm__ T*)(commCtxParam->winAddr[commCtxParam->statusIndex + dstRankId] + offset);
    }
}

/* UB 清 0 */
TILEOP void ClearFlagBuf(__ubuf__ int32_t* flagBuf)
{
    /*
    每次处理 8 个 block，8 * 32 = 256B，所以使用 vector_dup 时建议 flag 内存对齐 256B
    BlockStride 是每次迭代内 block 的距离（stride，前一个头和后一个头，0 会按照 1 来处理），单位是 block
    RepeatStride 是每次迭代间 block 的距离，如果内存是连续的，值一般是 8
    */

    uint8_t repeat = 1;
    int32_t src = 0;
    uint16_t dstBlockStride = 0;
    uint16_t srcBlockStride = 0;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 0;
    vector_dup(flagBuf, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

TILEOP void GatherMask(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1,
                       GatherMaskParams& gatherMaskParams)
{
    set_mask_norm();
    set_vector_mask(-1, -1);
    vreducev2(dst, src0, src1, gatherMaskParams.repeat, gatherMaskParams.src0BlockStride, gatherMaskParams.patternMode,
              gatherMaskParams.src0RepeatStride, gatherMaskParams.src1RepeatStride);
    set_mask_norm();
    set_vector_mask(-1, -1); // 重置 mask
}

TILEOP void Sum(__ubuf__ float* result, __ubuf__ float* src, SumParams& sumParams, uint32_t cnt)
{
    set_mask_count();        // 设置 counter mode
    set_vector_mask(0, cnt); // 只计算 cnt 个
    vcadd(result, src, sumParams.repeat, sumParams.dstRepeatStride, sumParams.srcBlockStride, sumParams.srcRepeatStride,
          0);
    set_mask_norm();         // 重置 mode
    set_vector_mask(-1, -1); // 重置 mask
}

template <typename T>
TILEOP void GatherMaskAndSum(__gm__ T* out, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, __ubuf__ uint32_t* dst,
                             uint32_t mask, uint32_t cnt, __gm__ int64_t* hcclContext)
{
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t*>(src1));
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t*>(dst));

    src1[0] = mask; // 设置前 32 个数
    src1[1] = mask; // 设置后 32 个数，共计 64 个数，256B
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    GatherMaskParams gatherMaskParams;
    uint32_t gatherMaskRepeat = (cnt * 32 + 255) / 256; // 重复次数，向上对齐 256B（gather 每次处理 256B）
    gatherMaskParams.repeat = gatherMaskRepeat; // 重复次数，最多只会处理 8 个数，每 8 个取一个，正好 64 个，256B
    // 单次迭代内 blk stride，表示 mask 后 32 个数相对前 32 个数的 stride（u32 类型），1 表示连续，0
    // 表示两次处理同一块，一般取 1
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.patternMode = 0; // 自定义模式需为 0
    // 可能可以调为 1，winFlag 搬运进 UB 的时候可以按照 32B 对齐，不搬运全部 512B 大小
    gatherMaskParams.src0RepeatStride = 16; // 迭代间 stride，16 * 32 = 512B，符合 dispatch 的 flag 排序要求
    gatherMaskParams.src1RepeatStride = 0;  // 迭代间 stride，0 表示每次 repeat 都取同样的 src1 mask
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    GatherMask(dst, src0, src1, gatherMaskParams);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    __ubuf__ float* sumSrc = reinterpret_cast<__ubuf__ float*>(dst);
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t*>(src1)); // 可以不加，其余位置的数据并不重要，可以不清空
    __ubuf__ float* sumDst = reinterpret_cast<__ubuf__ float*>(src1);

    SumParams sumParams;
    sumParams.repeat = 1;                // 只计算一次
    sumParams.dstRepeatStride = 8;       // 不重要
    sumParams.srcBlockStride = 1;        // 表示 src 连续取值
    sumParams.srcRepeatStride = 8;       // 不重要
    Sum(sumDst, sumSrc, sumParams, cnt); // sum 的输出这里是一个 float 值
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}

TILEOP void CalcOccurrences(__ubuf__ int32_t* expertTable, uint32_t dstExpertId, uint32_t cnt, __ubuf__ int32_t* result)
{
    (*result) = 0;
    if (cnt == 0) {
        return;
    }
    __ubuf__ int32_t* tmp = expertTable;
    for (int32_t i = 0; i < cnt; i++) {
        if ((*tmp++) == dstExpertId) {
            pipe_barrier(PIPE_ALL);
            (*result)++;
            pipe_barrier(PIPE_ALL);
        }
    }
}

TILEOP int32_t CalcOccurrencesVector(__ubuf__ int32_t* expertTable, uint32_t dstExpertId, uint32_t cnt,
                                     __ubuf__ int32_t* tmpBuf)
{
    if (cnt == 0) {
        return 0;
    }
    int32_t bufferLen = AlignUp<int32_t>(cnt * sizeof(int32_t), 32);
    uint32_t repeatCnt = bufferLen / 32;
    if (bufferLen % 32 != 0) {
        repeatCnt++;
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    __ubuf__ int32_t* subBuf = tmpBuf + bufferLen;
    // dst, srcImm, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vector_dup(tmpBuf, dstExpertId, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
    // src1RepeatStride
    vsub(subBuf, expertTable, tmpBuf, repeatCnt, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vabs((__ubuf__ float*)tmpBuf, (__ubuf__ float*)subBuf, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vmins(subBuf, tmpBuf, 1, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // 求和
    set_mask_count();        // 设置 counter mode
    set_vector_mask(0, cnt); // 只计算 cnt 个
    // dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride
    vcadd((__ubuf__ float*)tmpBuf, (__ubuf__ float*)subBuf, 1, 8, 1, 8, 0);
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    return cnt - tmpBuf[0];
}

template <typename T>
TILEOP void WaitFlagV2(__gm__ T* out, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, __ubuf__ uint32_t* dst,
                       uint32_t cnt, __gm__ int64_t* hcclContext)
{
    GatherMaskAndSum(out, src0, src1, dst, MASK_SELECT_SEND_FLAG, cnt, hcclContext);
}

TILEOP void ClearFlagV2(__ubuf__ int32_t* flag, uint32_t offset, uint32_t repeat, __gm__ int64_t* hcclContext,
                        DispatchInfo& dispatchInfo, __gm__ int32_t* shmemFlagBaseAddr)
{
    __gm__ CommContext* winContext = (__gm__ CommContext*)(hcclContext[dispatchInfo.groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR winFlagBaseAddr = (GM_ADDR)MapVirtualAddr<int32_t>(hcclContext, shmemFlagBaseAddr,
                                                               localUsrRankId); // flag 在 win 区的基地址
    GM_ADDR winFlagReadStartAddr = winFlagBaseAddr + offset;

    ClearFlagBuf(flag);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    flag[0] = -1;
    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = 1;     // 搬运次数
    dataCopyParams.lenBurst = 1;   // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 0;  // 前一个尾巴和下一个的开头，gap
    dataCopyParams.dstStride = 15; // 前一个尾巴和下一个开头，gap
    set_atomic_s32();
    for (int i = 0; i < repeat; i++) {
        copy_ubuf_to_gm(winFlagReadStartAddr, flag, dataCopyParams.sid, dataCopyParams.nBurst, dataCopyParams.lenBurst,
                        dataCopyParams.srcStride, dataCopyParams.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        winFlagReadStartAddr += 512;
    }
    set_atomic_none();
}

template <typename T>
TILEOP void ReadFlagV2(__ubuf__ uint32_t* flag, uint32_t offset, uint32_t repeat, __gm__ int64_t* hcclContext,
                       __gm__ T* shmemFlagBaseAddr, DispatchInfo& dispatchInfo)
{
    __gm__ CommContext* winContext = (__gm__ CommContext*)(hcclContext[dispatchInfo.groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    __gm__ T* winFlagBaseAddr = MapVirtualAddr<T>(hcclContext, shmemFlagBaseAddr,
                                                  localUsrRankId); // flag 在 win 区的基地址
    GM_ADDR winFlagReadStartAddr = (GM_ADDR)winFlagBaseAddr + static_cast<uint32_t>(offset);

    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = repeat; // 搬运次数
    dataCopyParams.lenBurst = 1;    // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 15;  // 前一个尾巴和下一个的开头，gap，15 * 32 = 480B
    dataCopyParams.dstStride = 0;   // 前一个尾巴和下一个开头，gap，搬到 UB 上需要连续
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    copy_gm_to_ubuf(flag, winFlagReadStartAddr, dataCopyParams.sid, dataCopyParams.nBurst, dataCopyParams.lenBurst,
                    dataCopyParams.srcStride, dataCopyParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

TILEOP void CumSum(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* gatherMaskDst, uint32_t mask,
                   uint32_t cnt)
{
    __ubuf__ uint32_t* gatherMask = reinterpret_cast<__ubuf__ uint32_t*>(dst);
    // 主逻辑
    gatherMask[0] = mask; // 每 8 个数挑一个，一次处理 64 个数，256B，一共需要处理 32 * 48 / 256 = 6 次
    gatherMask[1] = mask;
    pipe_barrier(PIPE_ALL);
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeat = (cnt * 32 + 255) / 256; // 重复次数，只搬运所需要的部分
    // 单次迭代内 blk stride，表示 mask 后 32 个数相对前 32 个数的 stride（u32 类型），单位应该是 128B，1 表示连续，0
    // 表示两次处理同一块，一般取 1
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.patternMode = 0; // 自定义模式需为 0
    gatherMaskParams.src0RepeatStride = 8; // 迭代间 stride，搬运进 UB 后调整为 32B 间隔，64 个数 * 4 = 256B / 32 = 8
    gatherMaskParams.src1RepeatStride = 0; // 迭代间 stride，0 表示每次 repeat 都取同样的 src1 mask
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    GatherMask(gatherMaskDst, src, gatherMask, gatherMaskParams); // 把所有的 src 都收集起来了，共计 48 个
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    __ubuf__ float* sumSrc = reinterpret_cast<__ubuf__ float*>(gatherMaskDst);
    // 在 TileOpIndex = 0 的场景下，cnt 是 0，需要手动清空上面赋值的 mask，否则后续结果错误
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t*>(dst));
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    __ubuf__ float* sumDst = reinterpret_cast<__ubuf__ float*>(dst); // GatherMask 的 mask 复用为 Sum 的输出

    SumParams sumParams;
    sumParams.repeat = 1;                // 只计算一次
    sumParams.dstRepeatStride = 8;       // 不重要
    sumParams.srcBlockStride = 1;        // 表示 src 连续取值
    sumParams.srcRepeatStride = 8;       // 不重要
    Sum(sumDst, sumSrc, sumParams, cnt); // sum 的输出这里是一个 float 值
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}

TILEOP void CopyGmToGm(__gm__ void* dst, __gm__ void* src, __ubuf__ void* tmpUbuf, DataCopyParams gmToUbParams,
                       DataCopyParams ubToGmParams)
{
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    copy_gm_to_ubuf(tmpUbuf, src, gmToUbParams.sid, gmToUbParams.nBurst, gmToUbParams.lenBurst, gmToUbParams.srcStride,
                    gmToUbParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_gm(dst, tmpUbuf, ubToGmParams.sid, ubToGmParams.nBurst, ubToGmParams.lenBurst, ubToGmParams.srcStride,
                    ubToGmParams.dstStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}
} // namespace TileOp::Distributed
#endif
