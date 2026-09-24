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
 * \file CommonType.h
 * \brief
 */

#pragma once

#include <iostream>
#include <cstdint>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <regex>
#include <unordered_set>
#include "CycleInfo.h"
#include "CommonData.h"
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "tilefwk/error.h"

#define PRIOR_SCHEDULING // comment it to disable PriorScheduling pass

#ifdef PRIOR_SCHEDULING
using setType = std::conditional<true, std::unordered_set<int>, std::set<int>>::type;
#else
using setType = std::conditional<false, std::unordered_set<int>, std::set<int>>::type;
#endif

namespace CostModel {
enum class SimMode { NORMAL = 0, EMULATOR, LEAF_FUNCTION, PV_MODEL };

inline bool IsNeedInput(CostModel::SimMode simMode)
{
    if (simMode == CostModel::SimMode::NORMAL || simMode == CostModel::SimMode::LEAF_FUNCTION ||
        simMode == CostModel::SimMode::EMULATOR) {
        return true;
    }
    return false;
}

enum class NodeType { LOCAL, INCAST, OUTCAST };

using namespace npu::tile_fwk;

inline CostModel::OperandType BufferNameToType(std::string& name)
{
    static const std::unordered_map<std::string, CostModel::OperandType> bufferMap = {
        {"MEM_UB", OperandType::BUF_UB},       {"MEM_L1", OperandType::BUF_L1},
        {"MEM_L0A", OperandType::BUF_L0A},     {"MEM_L0B", OperandType::BUF_L0B},
        {"MEM_L0C", OperandType::BUF_L0C},     {"MEM_FIX", OperandType::BUF_FIX},
        {"MEM_BT", OperandType::BUF_BT},       {"MEM_DEVICE_DDR", OperandType::BUF_DDR},
        {"MEM_L0AMX", OperandType::BUF_L0AMX}, {"MEM_L0BMX", OperandType::BUF_L0BMX},
    };

    auto it = bufferMap.find(name);
    return (it != bufferMap.end()) ? it->second : OperandType::BUF_UNKNOWN;
}

inline NodeType ToNodeType(std::string& type)
{
    if (type == "INCAST") {
        return NodeType::INCAST;
    } else if (type == "OUTCAST") {
        return NodeType::OUTCAST;
    } else {
        return NodeType::LOCAL;
    }
}

inline DataType ToDataType(std::string& name)
{
    static std::unordered_map<std::string, DataType> type_map = {
        {"DT_INT4", DataType::DT_INT4},         {"DT_INT8", DataType::DT_INT8},
        {"DT_INT16", DataType::DT_INT16},       {"DT_INT32", DataType::DT_INT32},
        {"DT_INT64", DataType::DT_INT64},       {"DT_FP8", DataType::DT_FP8},
        {"DT_FP16", DataType::DT_FP16},         {"DT_FP32", DataType::DT_FP32},
        {"DT_BF16", DataType::DT_BF16},         {"DT_HF4", DataType::DT_HF4},
        {"DT_HF8", DataType::DT_HF8},           {"DT_UINT8", DataType::DT_UINT8},
        {"DT_UINT16", DataType::DT_UINT16},     {"DT_UINT32", DataType::DT_UINT32},
        {"DT_UINT64", DataType::DT_UINT64},     {"DT_BOOL", DataType::DT_BOOL},
        {"DT_DOUBLE", DataType::DT_DOUBLE},     {"DT_FP8E5M2", DataType::DT_FP8E5M2},
        {"DT_FP8E4M3", DataType::DT_FP8E4M3},   {"DT_FP8E8M0", DataType::DT_FP8E8M0},
        {"DT_FP4_E2M1", DataType::DT_FP4_E2M1}, {"DT_FP4_E2M1X2", DataType::DT_FP4_E2M1X2},
        {"DT_FP4_E1M2", DataType::DT_FP4_E1M2}, {"DT_FP4_E1M2X2", DataType::DT_FP4_E1M2X2}};
    auto it = type_map.find(name);
    if (it == type_map.end()) {
        std::cout << "Unrecognized DataType" << name << std::endl;
        return DataType::DT_FP16;
    }
    return it->second;
}

// CostModel
enum class CorePipeType {
    PIPE_UNKNOW = -1,
    PIPE_TILE_ALLOC = 0,
    PIPE_VECTOR_BMU,
    PIPE_CUBE_BMU_L1,
    PIPE_CUBE_BMU_L0A,
    PIPE_CUBE_BMU_L0B,
    PIPE_CUBE_BMU_L0C,
    PIPE_MTE_IN, // FOR TILE_COPY_IN
    PIPE_MTE1,   // FOR L1 TO L0A/B
    PIPE_VECTOR_ALU,
    PIPE_CUBE,
    PIPE_MTE_OUT, // FOR TILE_COPY_OUT
    PIPE_S,       // FOR VIEW,ASSEMBLE,RESHAPE
    PIPE_CALL,
    PIPE_FIX,
    TOTAL_CORE_PIPE_TYPE
};

inline bool IsTileAlloc(CorePipeType type)
{
    if (type == CorePipeType::PIPE_TILE_ALLOC || type == CorePipeType::PIPE_VECTOR_BMU ||
        type == CorePipeType::PIPE_CUBE_BMU_L1 || type == CorePipeType::PIPE_CUBE_BMU_L0A ||
        type == CorePipeType::PIPE_CUBE_BMU_L0B || type == CorePipeType::PIPE_CUBE_BMU_L0C) {
        return true;
    }
    return false;
}

inline bool IsTileBufferAlloc(CorePipeType type)
{
    if (type == CorePipeType::PIPE_VECTOR_BMU || type == CorePipeType::PIPE_CUBE_BMU_L1 ||
        type == CorePipeType::PIPE_CUBE_BMU_L0A || type == CorePipeType::PIPE_CUBE_BMU_L0B ||
        type == CorePipeType::PIPE_CUBE_BMU_L0C) {
        return true;
    }
    return false;
}

inline bool IsReadCache(CorePipeType type)
{
    if (type == CorePipeType::PIPE_MTE_IN) {
        return true;
    }
    return false;
}

inline bool IsWriteCache(CorePipeType type)
{
    if (type == CorePipeType::PIPE_MTE_OUT) {
        return true;
    }
    return false;
}

inline bool IsMTEPipe(CorePipeType type)
{
    if (type == CorePipeType::PIPE_MTE_IN || type == CorePipeType::PIPE_MTE1 || type == CorePipeType::PIPE_MTE_OUT) {
        return true;
    }
    return false;
}

inline std::string CorePipeName(CorePipeType type)
{
    switch (type) {
        case CorePipeType::PIPE_TILE_ALLOC:
            return "TILE_ALLOC";
        case CorePipeType::PIPE_VECTOR_BMU:
            return "VECTOR_BMU";
        case CorePipeType::PIPE_CUBE_BMU_L1:
            return "CUBE_BMU_L1";
        case CorePipeType::PIPE_CUBE_BMU_L0A:
            return "CUBE_BMU_L0A";
        case CorePipeType::PIPE_CUBE_BMU_L0B:
            return "CUBE_BMU_L0B";
        case CorePipeType::PIPE_CUBE_BMU_L0C:
            return "CUBE_BMU_L0C";
        case CorePipeType::PIPE_MTE_IN:
            return "MTE_IN";
        case CorePipeType::PIPE_MTE1:
            return "MTE1";
        case CorePipeType::PIPE_MTE_OUT:
            return "MTE_OUT";
        case CorePipeType::PIPE_VECTOR_ALU:
            return "VECTOR_ALU";
        case CorePipeType::PIPE_CUBE:
            return "CUBE";
        case CorePipeType::PIPE_CALL:
            return "SIM_CALL";
        case CorePipeType::PIPE_S:
            return "PIPE_S";
        case CorePipeType::PIPE_FIX:
            return "PIPE_FIX";
        default:
            return "ILLEGAL";
    }
}

enum class MachineType { UNKNOWN, DEVICE, CPU, AIC, AIV, MIXAICORE, PIPE, CACHE, HUB, TOTAL_MACHINE_TYPE };

inline std::string MachineName(MachineType type)
{
    switch (type) {
        case MachineType::DEVICE:
            return "DEVICE";
        case MachineType::CPU:
            return "AICPU";
        case MachineType::AIV:
            return "AIV";
        case MachineType::AIC:
            return "AIC";
        case MachineType::MIXAICORE:
            return "MIXAICORE";
        case MachineType::PIPE:
            return "PIPE";
        case MachineType::CACHE:
            return "CACHE";
        case MachineType::HUB:
            return "HUB";
        default:
            return "ILLEGAL";
    }
}

inline bool IsCoreMachine(MachineType type)
{
    if (type == MachineType::AIC || type == MachineType::AIV || type == MachineType::MIXAICORE ||
        type == MachineType::HUB) {
        return true;
    }
    return false;
}

inline bool IsCoreMachine(int type) { return IsCoreMachine(static_cast<MachineType>(type)); }

// convert string to MachineType
inline MachineType ToMachineType(const std::string& machineTypeStr)
{
    if (machineTypeStr == "AIV") {
        return MachineType::AIV;
    } else if (machineTypeStr == "AIC") {
        return MachineType::AIC;
    } else if (machineTypeStr == "MIXAICORE") {
        return MachineType::MIXAICORE;
    } else if (machineTypeStr == "HUB") {
        return MachineType::HUB;
    }
    return MachineType::UNKNOWN;
}

enum class CalendarMode { DEVICE, GLOBAL_COUNTER, OPTIONAL_COUNTERS };

const std::map<MachineType, std::set<CorePipeType>> MACHINE_PIPE_SET = {
    {MachineType::AIV,
     {CorePipeType::PIPE_VECTOR_BMU, CorePipeType::PIPE_MTE_IN, CorePipeType::PIPE_VECTOR_ALU,
      CorePipeType::PIPE_MTE_OUT}},
    {MachineType::AIC,
     {CorePipeType::PIPE_CUBE_BMU_L1, CorePipeType::PIPE_CUBE_BMU_L0A, CorePipeType::PIPE_CUBE_BMU_L0B,
      CorePipeType::PIPE_CUBE_BMU_L0C, CorePipeType::PIPE_MTE_IN, CorePipeType::PIPE_MTE1, CorePipeType::PIPE_CUBE,
      CorePipeType::PIPE_MTE_OUT}},
    {MachineType::MIXAICORE,
     {CorePipeType::PIPE_CUBE_BMU_L1, CorePipeType::PIPE_CUBE_BMU_L0A, CorePipeType::PIPE_CUBE_BMU_L0B,
      CorePipeType::PIPE_CUBE_BMU_L0C, CorePipeType::PIPE_MTE_IN, CorePipeType::PIPE_MTE1, CorePipeType::PIPE_CUBE,
      CorePipeType::PIPE_VECTOR_ALU, CorePipeType::PIPE_MTE_OUT}},
};

enum class CacheType { FUNCTION_CACHE, L2CACHE, TOTAL_CACHE_TYPE };

inline std::string CacheName(CacheType type)
{
    switch (type) {
        case CacheType::FUNCTION_CACHE:
            return "FunctionCache";
        case CacheType::L2CACHE:
            return "L2CACHE";
        default:
            return "ILLEGAL";
    }
}

enum class CachePacketType {
    REQUEST = 0,
    RESPONSE,
};

enum class CacheRequestType {
    FUNCTION_REQ = 0,
    DATA_READ_REQ,
    DATA_WRITE_REQ,
};

inline std::string CacheRequestName(CacheRequestType type)
{
    switch (type) {
        case CacheRequestType::FUNCTION_REQ:
            return "Function_Read";
        case CacheRequestType::DATA_READ_REQ:
            return "Data_Read";
        case CacheRequestType::DATA_WRITE_REQ:
            return "Data_Write";
        default:
            return "ILLEGAL";
    }
}

const std::vector<std::string> LETTERS = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                                          "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};

inline std::string DecimalTo26(int num)
{
    num = std::abs(num);
    int baseDivisor = 26;
    if (num == 0) {
        return "a";
    }
    std::string result;
    while (num > 0) {
        int remainder = num % baseDivisor;
        result = CostModel::LETTERS[remainder] + result;
        num = num / baseDivisor;
    }
    return result;
}

const uint64_t TASK_ID_OFFSET20 = 20;
const uint64_t TASK_ID_OFFSET11 = 11;

class ReplayTaskEntry {
public:
    uint64_t seqNo;
    uint64_t taskId;
    uint64_t sCycles;
    uint64_t eCycles;
    ReplayTaskEntry(uint64_t seq, uint64_t id, uint64_t sTime, uint64_t eTime)
        : seqNo(seq), taskId(id), sCycles(sTime), eCycles(eTime)
    {}
};

class Task {
public:
    uint64_t seqNo = 0;
    uint64_t taskId = -1;
    uint64_t functionHash;
    std::string functionName = "";
    bool status = false;
    int remainingPredecessors = 0;
    std::vector<uint64_t> predecessors;
    std::vector<uint64_t> successors;
    std::vector<int> incasts;
    std::vector<int> outcasts;
    std::string semanticLabel;
    MachineType machineType = MachineType::UNKNOWN;

    // fixed latency
    bool fixedLatency = false;
    uint64_t fixedLatencyVal = 0;

    bool scaleExecuteTime = false;
    double proportion = 1.0;
    bool printRelativeCycle = false;

    uint64_t leafIndex = 0;
    uint64_t opmagic = 0;
    int psgId = -1;
    uint64_t rootIndex = 0;
    uint64_t uniqueKey = 0;
    uint64_t taskStartCycle = 0;
    uint64_t taskEndCycle = 0;

    std::string GetColorLabel(uint64_t mode)
    {
        std::string colorLabel;
        if (mode == 1) {
            colorLabel = semanticLabel;
        } else if (mode > 1) {
            colorLabel = semanticLabel + " " + DecimalTo26(psgId);
        }
        if (colorLabel.empty()) {
            colorLabel = DecimalTo26(psgId);
        }
        return colorLabel;
    }

    std::string GetFormalName()
    {
        std::ostringstream os;
        uint64_t funcIdStitch = ((taskId >> TASK_ID_OFFSET20) & ((1 << TASK_ID_OFFSET11) - 1));
        uint64_t opIndex = (taskId & ((1 << TASK_ID_OFFSET20) - 1));
        os << seqNo << "-" << funcIdStitch << "-" << opIndex;
        return os.str();
    }

    std::string GetTaskName()
    {
        return GetFormalName() + "-" + std::to_string(rootIndex) + "-" + std::to_string(psgId);
    }

    std::string GetTaskFullName()
    {
        std::ostringstream os;
        os << "[" << GetTaskName() << "] Executing SeqNo:" << seqNo << " TaskId:" << taskId << " pSgId:" << psgId;
        os << " Function:" << functionName << ", hash:" << functionHash;
        return os.str();
    }
};

using TaskMap = std::map<uint64_t, std::shared_ptr<Task>>;

struct TopoInfoEntry {
    uint64_t eSgId;
    int readyState;
    uint64_t calleeHash = 0;

    // fixed latency
    bool fixedLatency = false;
    uint64_t fixedLatencyVal = 0;
    MachineType mType = MachineType::UNKNOWN;
    setType outGraph;
};

// For AICPU workload balance dispatch task.(SMT)
struct AICoreWorkLoadStatus {
    std::vector<std::vector<size_t>> smtGroupIndexs;
    size_t maxLevel = 0;
    size_t groups = 0;
    AICoreWorkLoadStatus()
    {
        smtGroupIndexs.clear();
        maxLevel = 0;
    }

    void AddMachineGroup()
    {
        smtGroupIndexs.emplace_back();
        groups++;
    }

    void AddMachineIndex(size_t index) { smtGroupIndexs[groups - 1].emplace_back(index); }
};
} // namespace CostModel
