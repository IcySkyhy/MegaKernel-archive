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
 * \file msprof_define.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace npu::tile_fwk {
#define MSPF_REPORT_NODE_BASIC_INFO_TYPE 0U      /* type info: node_basic_info */
#define MSPF_REPORT_NODE_TENSOR_INFO_TYPE 1U     /* type info: tensor_info */
#define MSPF_REPORT_NODE_CONTEXT_ID_INFO_TYPE 4U /* type info: context_id_info */
#define MSPF_REPORT_NODE_LAUNCH_TYPE 5U          /* type info: launch */
#define MSPF_REPORT_NODE_LEVEL 10000U
#define MSPF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPF_GE_TENSOR_DATA_NUM 5
#define MSPF_TASK_TIME_L1_MASK 0x00000002ULL
#define MSPF_TASK_TIME_L2_MASK 0x00002000ULL
#define MSPF_CTX_ID_MAX_NUM 55
#define MSPF_REPORT_DATA_MAGIC_NUM 0x5A5AU
#define MSPF_ADDTIONAL_INFO_DATA_LENGTH (232)
#define MSPF_COMPACT_INFO_DATA_LENGTH 40

enum MspfGeTensorType {
    MSPF_GE_TENSOR_TYPE_INPUT = 0,
    MSPF_GE_TENSOR_TYPE_OUTPUT,
};

enum MspfGeTaskType {
    MSPF_GE_TASK_TYPE_AI_CORE = 0,
    MSPF_GE_TASK_TYPE_AI_CPU,
    MSPF_GE_TASK_TYPE_AIV,
    MSPF_GE_TASK_TYPE_WRITE_BACK,
    MSPF_GE_TASK_TYPE_MIX_AIC,
    MSPF_GE_TASK_TYPE_MIX_AIV,
    MSPF_GE_TASK_TYPE_FFTS_PLUS,
    MSPF_GE_TASK_TYPE_DSA,
    MSPF_GE_TASK_TYPE_DVPP,
    MSPF_GE_TASK_TYPE_HCCL,
    MSPF_GE_TASK_TYPE_FUSION,
    MSPF_GE_TASK_TYPE_INVALID
};

#pragma pack(1)

struct MspfTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPF_GE_TENSOR_DATA_SHAPE_LEN];
};
struct MspfTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MspfTensorData tensorData[MSPF_GE_TENSOR_DATA_NUM];
};

struct MspfContextIdInfo {
    uint64_t opName;
    uint32_t ctxIdNum;
    uint32_t ctxIds[MSPF_CTX_ID_MAX_NUM];
};

struct MspfNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};

struct MspfHCCLOPInfo { // for MspfReportCompactInfo buffer data
    uint8_t relay : 1;  // Communication
    uint8_t retry : 1;  // Retransmission flag
    uint8_t dataType;   // Consistent with Type HcclDataType preservation
    uint64_t algType;   // The algorithm used by the communication operator.
    uint64_t count;     // Number of data sent
    uint64_t groupName; // group hash id
};

#pragma pack()

struct MspfApi { // for MspfReportApi
#ifdef __cplusplus
    uint16_t magicNumber = MSPF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

struct MspfAdditionalInfo { // for MspfReportAdditionalInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t data[MSPF_ADDTIONAL_INFO_DATA_LENGTH];
};

struct MspfRuntimeTrack { // for MspfReportCompactInfo buffer data
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;   // task message hash id
    uint64_t kernelName; // kernelname hash id
};

struct MspfCaptureStreamInfo { // for MspfReportCompactInfo buffer data
    uint16_t captureStatus;    // Whether the mark is destroyed: 0 indicates normal, 1 indicates destroyed.
    uint16_t modelStreamId;    // capture stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t originalStreamId; // ori stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t modelId;          // capture model id, independent of GE
    uint16_t deviceId;
};

struct MspfDpuTrack {  // for MspfReportCompactInfo buffer data
    uint16_t deviceId; // high 4 bits, devType: dpu: 1, low 12 bits device id
    uint16_t streamId;
    uint32_t taskId;
    uint32_t taskType; // task type enum
    uint32_t res;
    uint64_t startTime; // start time
};

struct MspfStreamExpandSpecInfo {
    uint8_t expandStatus;
    uint8_t reserve1;
    uint16_t reserve2;
};

struct MspfCompactInfo { // for MspfReportCompactInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPF_COMPACT_INFO_DATA_LENGTH];
        struct MspfRuntimeTrack runtimeTrack;
        struct MspfCaptureStreamInfo captureStreamInfo;
        struct MspfNodeBasicInfo nodeBasicInfo;
        struct MspfHCCLOPInfo hcclopInfo;
        struct MspfDpuTrack dpuTack;
        struct MspfStreamExpandSpecInfo streamExpandInfo;
    } data;
};

typedef int32_t (*MspfCommandHandleFunc)(uint32_t type, void* data, uint32_t len);

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
#define MSPF_MAX_DEV_NUM 64

struct MspfCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit; // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

/**
 * @brief profiling command info
 */
struct MspfCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MspfCommandHandleParams params;
};

enum MspfCommandHandleType {
    MSPF_COMMANDHANDLE_TYPE_INIT = 0,
    MSPF_COMMANDHANDLE_TYPE_START,
    MSPF_COMMANDHANDLE_TYPE_STOP,
    MSPF_COMMANDHANDLE_TYPE_FINALIZE,
    MSPF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE,
    MSPF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE,
    MSPF_COMMANDHANDLE_TYPE_MAX
};
} // namespace npu::tile_fwk
