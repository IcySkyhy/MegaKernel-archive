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
 * \file hccl_context.h
 * \brief
 */

#ifndef __LOGICALTENSOR_HCCL_CONTEXT__
#define __LOGICALTENSOR_HCCL_CONTEXT__

#include <type_traits>
#include <cstdint>

namespace npu::tile_fwk {
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;
constexpr uint32_t AICPU_MAX_RANK_NUM_V1 = 32;
constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;

struct HcclSignalInfo {
    uint64_t resId; // 在代表event时为eventid，notify时为notifyid
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};
struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;      // 记录物理cqId
    uint32_t logicCqids; // 记录逻辑cqId
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM]; // 集合通信AICPU展开资源
    ListCommon nextTagRes;                                 // HccltagLocalResV2
};

struct AlgoTopoInfo {
    uint32_t userRank;     // 通信域 RankID
    uint32_t userRankSize; // 通信域的Rank数量
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation; // 每个Module中的Device数量
    uint32_t superPodNum;             // 集群中总的超节点数
    uint32_t devicePhyId;
    uint32_t topoType; // TopoType
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;                     // niclist数组指针
    uint64_t complanRankLength;           // complanRank占用的字节数
    uint64_t complanRank;                 // 指针
    uint64_t bridgeRankNum;               // bridgeRank占用的个数
    uint64_t bridgeRank;                  // 指针
    uint64_t serverAndsuperPodRankLength; // serverAndsuperPodRank占用的字节数
    uint64_t serverAndsuperPodRank;       // 指针
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct HcclOpConfig {
    uint8_t deterministic; // 确定性计算开关
    uint8_t retryEnable;   // 是否重执行
    uint8_t highPerfEnable;
    uint8_t padding[5];      // 大小需要64By对齐，未来添加参数时减小padding
    uint8_t linkTimeOut[8];  // 发送超时时长
    uint64_t notifyWaitTime; // 超时时长，同HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable = false; // 使能rdma开关
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512; // 多QP每个QP分担数据量最小阈值
};

struct HcclMC2WorkSpace {
    uint64_t workSpace;
    uint64_t workSpaceSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HDCommunicateParams {
    uint64_t hostAddr{0};
    uint64_t deviceAddr{0};
    uint64_t readCacheAddr{0};
    uint32_t devMemSize{0};
    uint32_t buffLen{0};
    uint32_t flag{0};
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParam {
    // 本地资源
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId; // usrrankid
    uint32_t rankSize;       // 通信域内total rank个数
    uint64_t winSize; // 每个win大小，静态图时，可能是0，如果通信域内也有动态图，则可能为非0
    uint64_t localWindowsIn;  // 全F为无效值
    uint64_t localWindowsOut; // 全F为无效值
    char hcomId[128];
    // aicore识别remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;  // 为HcclRankRelationRes起始位置
    uint32_t rWinOffset; // 为HcclRemoteRes的大小
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // 外部配置参数
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;                      // 有效的remoteResNum
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM]; // 数组指针，指向HcclRankRelationResV2，下标为remoteUserRankId

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem; // for all2all
    uint64_t tinyMemSize;
    // 零拷贝场景使用
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];     // 保存集合通信时每个对端的输入输出内存地址
    uint64_t zeroCopyDevicePhyId[16]; // 保存每个rank对应的物理卡Id
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId; // usrrankid
    uint32_t rankSize;       // 通信域内total rank个数
    uint64_t winSize; // 每个win大小，静态图时，可能是0，如果通信域内也有动态图，则可能为非0
    uint64_t localWindowsIn;  // 全F为无效值
    uint64_t localWindowsOut; // 全F为无效值
    char hcomId[128];
    // aicore识别remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

// TP8卡
struct HcclCombinOpSignalParam {
    HcclSignalInfo noIpcNotifys[AICPU_MAX_RANK_NUM_V1 * 2];
    HcclSignalInfo ipcNotifys[AICPU_MAX_RANK_NUM_V1 * 4];
    HcclSignalInfo noIpcEvents[AICPU_MAX_RANK_NUM_V1];
    HcclSignalInfo aicpuNotify;
    HcclSignalInfo aicpuOpNotify[2]; // 集合通信AICPU展开资源
};

struct HcclCombinOpParamA5 {
    uint64_t workSpace;                         // client和server之间通信的地址
    uint64_t workSpaceSize;                     // client和server之间通信的空间大小
    uint32_t rankId;                            // 当前卡rankId
    uint32_t rankNum;                           // 总卡数
    uint64_t winSize;                           // ccu不使用
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];  // ccu不使用, MTE 数据区
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用，MTE 状态区

    // for ccu
    uint64_t xnAddr;  // Xn寄存器起始地址
    uint64_t ckeAddr; // CKE寄存器起始地址
    uint64_t msAddr;  // MS地址，预留
    uint64_t msSize;  // 可写的MS个数，预留
};

struct HcclCombinOpParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t rankId = 0; // 当前卡rankId
    uint32_t rankNum = 0;
    uint64_t winSize = 0; // 每个win大小
    uint64_t windowsIn[AICPU_MAX_RANK_NUM_V1];
    uint64_t windowsOut[AICPU_MAX_RANK_NUM_V1];
    char hcomId[128] = "\0";
    HcclStreamInfo streamInfo[AICPU_MAX_RANK_NUM_V1];
    HcclCombinOpSignalParam signalInfo;
    HcclOpConfig config; // 配置参数
    uint64_t overFlowAddr = 0;
    uint8_t onlyRead = 0; // 只使用读模式，不使用写模式

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;

    uint8_t padding[16]; // 大小需要64By对齐，未来添加参数时减小padding
    uint64_t winExpSize = 0;
    uint64_t windowsExp[AICPU_MAX_RANK_NUM_V1];

    uint8_t multiServerFlag = 0;
    uint64_t ibverbsData = 0;     // TransportDeviceNormalIbverbsData数组的首地址
    uint64_t ibverbsDataSize = 0; // TransportDeviceNormalIbverbsData数组的字节长度
};

#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;
struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};
} // namespace npu::tile_fwk

#endif
