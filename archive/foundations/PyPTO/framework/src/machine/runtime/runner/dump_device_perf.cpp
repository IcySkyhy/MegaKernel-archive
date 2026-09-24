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
 * \file dump_device_perf.cpp
 * \brief
 */

#include "machine/runtime/runner/dump_device_perf.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "utils/file_utils.h"
#include "interface/configs/config_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/distributed/common.h"
#include "machine/utils/checkinject.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace npu::tile_fwk::dynamic {
namespace {
constexpr int DUMP_LEVEL_FOUR = 4;
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
uint32_t g_last_round_num = 0;
} // namespace

void DumpDevTaskPerfData(DeviceArgs& args, const std::vector<void*>& perfData, bool isLast)
{
    if (GetEnvVar("DUMP_DEVICE_PERF") == "true" && !perfData.empty()) {
        uint64_t freq = (args.archInfo == ArchInfo::DAV_2201) ? FREQ_DAV_2201 : FREQ_DAV_3510;
        DumpAicpuPerfInfo(args, perfData, freq, isLast);
    }
}

json BuildSyncEventsJson(const TaskStat& taskStat, const uint8_t* perfDataPtr)
{
    json syncEventsArr = json::array();

    const uint8_t* perfDataBase = perfDataPtr;
    const uint64_t* setEventBase = reinterpret_cast<const uint64_t*>(perfDataBase + taskStat.setEventAddr);
    const uint64_t* waitEventBase = reinterpret_cast<const uint64_t*>(perfDataBase + taskStat.waitEventAddr);

    for (int k = 0; k < taskStat.setEventNum; ++k) {
        uint64_t timestamp = setEventBase[k];
        if (timestamp != 0) {
            json setEvent;
            setEvent["idx"] = k;
            setEvent["type"] = "CV_SYNC_SET";
            setEvent["time"] = timestamp;
            syncEventsArr.push_back(setEvent);
        }
    }
    for (int k = 0; k < taskStat.waitEventNum; ++k) {
        uint64_t timestamp = waitEventBase[k];
        if (timestamp != 0) {
            json waitEvent;
            waitEvent["idx"] = k;
            waitEvent["type"] = "CV_SYNC_WAIT";
            waitEvent["time"] = timestamp;
            syncEventsArr.push_back(waitEvent);
        }
    }

    std::sort(syncEventsArr.begin(), syncEventsArr.end(),
              [](const json& a, const json& b) { return a.value("time", 0) < b.value("time", 0); });
    return syncEventsArr;
}

void ConstructTaskInfo(const uint32_t& index, json& rootTaskStats, const std::vector<void*>& perfData,
                       const std::string& coreType)
{
    void* devPtr = perfData[index];
    size_t dataSize = PERF_DATA_TOTAL_SIZE;
    std::vector<uint8_t> hostBuffer(dataSize);
    auto ret = NormalizedRtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (ret != 0) {
        MACHINE_LOGW("task perf D2H copy failed ret: %d, index: %u", ret, index);
    }
    Metrics* aicpuMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
    if (aicpuMetric->taskCount > MAX_DFX_TASK_NUM_PER_CORE) {
        aicpuMetric->taskCount = MAX_DFX_TASK_NUM_PER_CORE;
    }
    TaskStat* taskStats = aicpuMetric->tasks;
    size_t numTasks = aicpuMetric->taskCount;
    json coreObj;
    coreObj["blockIdx"] = index;
    coreObj["coreType"] = coreType;
    if (aicpuMetric->coreType != -1) {
        coreObj["coreType"] = aicpuMetric->coreType == static_cast<int16_t>(CoreType::AIC) ? "AIC" : "AIV";
    }
    json tasksArr = json::array();
    for (size_t j = 0; j < numTasks; ++j) {
        if (taskStats[j].execEnd != 0) {
            json taskObj;
            taskObj["seqNo"] = taskStats[j].seqNo;
            taskObj["leafIndex"] = taskStats[j].subGraphId;
            taskObj["taskId"] = taskStats[j].taskId;
            taskObj["execStart"] = taskStats[j].execStart;
            taskObj["execEnd"] = taskStats[j].execEnd;
            json syncEventsArr = BuildSyncEventsJson(taskStats[j], hostBuffer.data());
            if (!syncEventsArr.empty()) {
                taskObj["syncEvents"] = syncEventsArr;
            }
            tasksArr.push_back(taskObj);
        }
    }
    coreObj["tasks"] = tasksArr;
    if (!tasksArr.empty()) {
        rootTaskStats.push_back(coreObj);
    }
    aicpuMetric->taskCount = 0;
    ret = NormalizedRtMemcpy(perfData[index], sizeof(Metrics), aicpuMetric, sizeof(Metrics),
                             RtMemcpyKind::HOST_TO_DEVICE);
    if (ret != 0) {
        MACHINE_LOGW("task perf H2D copy failed ret: %d, index: %u", ret, index);
    }
}

void DumpAicoreTaskExectInfo(const DeviceArgs& args, const std::vector<void*>& perfData)
{
    json rootTaskStatus = json::array();
    auto blockNum = args.GetBlockNum();
    MACHINE_LOGI("GetBlockNum : %lu", blockNum);
    for (uint32_t i = 0; i < blockNum; i++) {
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        ConstructTaskInfo(i, rootTaskStatus, perfData, coreType);
    }
    uint32_t aicoreBlockNum = args.nrAic + args.nrAiv;
    for (uint32_t i = aicoreBlockNum; i < aicoreBlockNum + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        ConstructTaskInfo(i, rootTaskStatus, perfData, "AI-CPU");
    }
    std::string jsonFilePath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    if (!SaveFile(jsonFilePath, rootTaskStatus.dump(DUMP_LEVEL_FOUR))) {
        MACHINE_LOGW("Contrust custom op json failed");
        return;
    }
    MACHINE_LOGD("tilefwk_L1_prof_data have saved in: %s", jsonFilePath.c_str());
    std::string topo_txt_path = npu::tile_fwk::config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = npu::tile_fwk::config::LogTopFolder() + "/program.json";
    std::string mix_event_path = npu::tile_fwk::config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::string draw_swim_lane_py_path = GetPyptoLibPath() + "/scripts/draw_swim_lane.py";
    npu::tile_fwk::config::SetRunDataOption(KEY_SWIM_GRAPH_PATH,
                                            npu::tile_fwk::config::GetAbsoluteTopFolder() + "/merged_swimlane.json");
    uint64_t freq = (args.archInfo == ArchInfo::DAV_2201) ? FREQ_DAV_2201 : FREQ_DAV_3510;

    if (IsPathExist(program_json_path) && IsPathExist(topo_txt_path)) {
        MACHINE_LOGI("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 " + draw_swim_lane_py_path + " \"" + jsonFilePath + "\" \"" + topo_txt_path +
                              "\" \"" + program_json_path +
                              "\" --label_type=1 --time_convert_denominator=" + std::to_string(freq) +
                              " --mix_event_info=\"" + mix_event_path + "\"";
        int ret = Checkinject(command.c_str(), command.size());
        if (ret != 0) {
            MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Draw swimlane cmd illegal char.");
            return;
        }
        if (system(command.c_str()) != 0) {
            MACHINE_LOGW("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        MACHINE_LOGW("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
}

inline void DevTaskPerfFormat(uint32_t tid, uint32_t type, json& devTaskJson, const MetricPerf* aicpuPer,
                              const uint32_t& turnIdx)
{
    json per_dev_task;
    uint32_t idx = DEVTASK_PERF_ARRY_INDEX(type);
    const DevTaskPerf& perfSlot = aicpuPer->devTaskPerfs[tid][idx];
    for (uint32_t i = 0; i < perfSlot.cnt; i++) {
        std::string name = PerfTraceName[type];
        name = name + "_" + std::to_string(turnIdx);
        if (type != PERF_TRACE_DEV_TASK_SEND_FIRST_LEAF_TASK) {
            name = name + "(" + std::to_string(i) + ")";
        }
        per_dev_task["name"] = name;
        per_dev_task["end"] = perfSlot.timeStamp[i];
        devTaskJson.push_back(per_dev_task);
    }
}

inline void SparateCore(int total, int idx, int part, const int& offset, std::vector<int>& coreArray)
{
    if (part == 0) {
        return;
    }
    int perCpu = total / part;
    int remain = total % part;
    int start = idx * perCpu + ((idx < remain) ? idx : remain);
    int end = start + perCpu + ((idx < remain) ? 1 : 0);
    for (int i = start; i < end; i++) {
        coreArray[i + offset] = idx + 1;
    }
}

inline void ConstructAicorePerfInfo(json& tasksArr, Metrics* aicoreMetric, const uint32_t& turnNum)
{
    uint64_t curCycle = 0;
    for (uint32_t turnIdx = g_last_round_num; turnIdx < turnNum; turnIdx++) {
        AicoreMetric* aicoreMetricPref = &(aicoreMetric->aicoreDevTaskInfo[turnIdx]);
        for (uint64_t cnt = 0; cnt < aicoreMetricPref->cnt; cnt++) {
            AicoreDevTaskPerf perf = aicoreMetricPref->aicoreEveryDevTypeTimeStamp[cnt];

            curCycle = perf.aicoreDevTimeStamp;
            if (curCycle == 0) {
                continue;
            }

            uint32_t type = static_cast<uint32_t>(perf.type);
            if (type >= PERF_TRACE_CORE_MAX) {
                continue;
            }

            json aicoreTaskType;
            std::string name = AicorePerfTraceName[type];
            name = name + "_" + std::to_string(turnIdx);
            if (perf.devTaskIdx != INVALID_DEV_TASK_ID) {
                name = name + "(" + std::to_string(perf.devTaskIdx) + ")";
            }
            aicoreTaskType["name"] = name;
            aicoreTaskType["end"] = curCycle;
            tasksArr.push_back(aicoreTaskType);
        }
    }
}

inline void DumpAicoreDevTask(DeviceArgs& args, json& aicpuPrefArray, const std::vector<void*>& perfData,
                              const uint32_t& freq, const uint32_t& turnNum)
{
    std::vector<int> coreArray;
    coreArray.resize(args.GetBlockNum());
    for (uint32_t i = 0; i < args.scheCpuNum; i++) {
        SparateCore(args.nrValidAic, i, args.scheCpuNum, 0, coreArray);
        SparateCore(args.nrValidAic * AIV_NUM_PER_AI_CORE, i, args.scheCpuNum, args.nrValidAic, coreArray);
    }
    for (uint32_t i = 0; i < args.GetBlockNum(); i++) {
        void* devPtr = perfData[i];
        size_t dataSize = PERF_DATA_TOTAL_SIZE;
        std::vector<uint8_t> hostBuffer(dataSize);
        auto ret = NormalizedRtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RtMemcpyKind::DEVICE_TO_HOST);
        if (ret != 0) {
            MACHINE_LOGW("aicore perf D2H copy failed ret: %d, block: %u", ret, i);
        }
        Metrics* aicoreMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
        std::string coreType = aicoreMetric->coreType == static_cast<int16_t>(CoreType::AIC) ? "AIC" : "AIV";
        json aicoreTask;
        aicoreTask["blockIdx"] = i + 1;
        aicoreTask["coreType"] = "SCHED" + std::to_string(aicoreMetric->scheCpuIdx) + "-" + coreType;
        aicoreTask["freq"] = freq;
        json tasksArr = json::array();
        ConstructAicorePerfInfo(tasksArr, aicoreMetric, turnNum);
        aicoreTask["tasks"] = tasksArr;
        aicpuPrefArray.push_back(aicoreTask);
    }
}

inline std::unique_ptr<MetricPerf> GetAicpuPrefAddr(const DeviceArgs& args, const uint32_t& turnIdx)
{
    auto aicpuMetric = std::make_unique<MetricPerf>();
    auto aicpuPer = (ValueToPtr(args.aicpuPerfAddr + turnIdx * sizeof(MetricPerf)));
    if (aicpuPer == nullptr) {
        MACHINE_LOGW("Aicpu per ptr is null");
        return aicpuMetric;
    }

    auto ret = NormalizedRtMemcpy(PtrToPtr<MetricPerf, void>(aicpuMetric.get()), sizeof(MetricPerf), aicpuPer,
                                  sizeof(MetricPerf), RtMemcpyKind::DEVICE_TO_HOST);
    if (ret != 0) {
        MACHINE_LOGW("aicpu meter copy failed ret: %d", ret);
    }
    return aicpuMetric;
}

inline void DumpAicpuDevTask(const DeviceArgs& args, json& aicpuPrefArray, const uint32_t& freq,
                             const uint32_t& turnNum)
{
    for (uint32_t i = 0; i < args.nrAicpu - 1; i++) {
        json aicpu;
        std::string coreType = "AICPU";
        if (i == 0) {
            coreType = "AICPU-CTRL";
        } else if (i <= args.scheCpuNum) {
            coreType = "AICPU-SCHED";
        }
        aicpu["blockIdx"] = i;
        aicpu["coreType"] = coreType;
        aicpu["freq"] = freq;
        json aicpuDevTasks = json::array();
        for (uint32_t turnIdx = g_last_round_num; turnIdx < turnNum; turnIdx++) {
            auto aicpuMetric = GetAicpuPrefAddr(args, turnIdx);
            for (uint32_t type = 0; type < PERF_TRACE_MAX; type++) {
                if (IsDevTaskType(type)) {
                    DevTaskPerfFormat(i, type, aicpuDevTasks, aicpuMetric.get(), turnIdx);
                    continue;
                }
                if (aicpuMetric->perfAicpuTrace[i][type] == 0) {
                    continue;
                }
                json schCtrAicpu;
                std::string name = PerfTraceName[type];
                schCtrAicpu["name"] = name + "_" + std::to_string(turnIdx);
                schCtrAicpu["end"] = aicpuMetric->perfAicpuTrace[i][type];
                aicpuDevTasks.push_back(schCtrAicpu);
            }
        }
        aicpu["tasks"] = aicpuDevTasks;
        aicpuPrefArray.push_back(aicpu);
    }
}

void DumpAicpuPerfInfo(DeviceArgs& args, const std::vector<void*>& perfData, uint32_t freq, bool isLast)
{
    void* devPtr = perfData[0];
    size_t dataSize = PERF_DATA_TOTAL_SIZE;
    std::vector<uint8_t> hostBuffer(dataSize);
    auto memcpyRet = NormalizedRtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (memcpyRet != 0) {
        MACHINE_LOGW("aicpu perf header D2H copy failed ret: %d", memcpyRet);
    }
    Metrics* aicoreMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
    auto sumRoundNum = (aicoreMetric->turnNum > MAX_ROUND_NUM) ? MAX_ROUND_NUM : aicoreMetric->turnNum;
    MACHINE_LOGD("CoreId 0 devAddr: %p, sumRoundNum: %ld", devPtr, sumRoundNum);
    if (sumRoundNum == g_last_round_num) {
        return;
    }
    if ((sumRoundNum < 50 || sumRoundNum % 50 != 0) && !isLast) {
        return;
    }
    json aicpuPrefArray = json::array();
    DumpAicpuDevTask(args, aicpuPrefArray, freq, sumRoundNum);
    DumpAicoreDevTask(args, aicpuPrefArray, perfData, freq, sumRoundNum);

    std::string aicpuPerfilePath = config::LogTopFolder() + "/machine_trace_perf_data_" +
                                   std::to_string(g_last_round_num) + ".json";
    if (!SaveFile(aicpuPerfilePath, aicpuPrefArray.dump(DUMP_LEVEL_FOUR))) {
        MACHINE_LOGW("Contrust custom op json failed");
        return;
    }

    // toolkit drawm aicpu preffto
    std::string scriptPath = GetPyptoLibPath() + "/scripts/machine_perf_trace.py";
    std::string cmd = "python3 " + scriptPath + " gen_perfetto " + aicpuPerfilePath + " " +
                      npu::tile_fwk::config::LogTopFolder() + "/machine_runtime_operator_trace_" +
                      std::to_string(g_last_round_num) + ".json " + npu::tile_fwk::config::LogTopFolder() +
                      "/merged_swimlane.json";
    int ret = Checkinject(cmd.c_str(), cmd.size());
    if (ret != 0) {
        MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Draw swimlane cmd illegal char.");
        return;
    }
    if (system(cmd.c_str()) != 0) {
        MACHINE_LOGW("Failed to execute machine_perf_trace.py, cannot get aicpu perfetto.json.");
    }
    g_last_round_num = sumRoundNum;
    npu::tile_fwk::config::SetRunDataOption(KEY_AICPU_PERF_GRAPH_PATH, npu::tile_fwk::config::GetAbsoluteTopFolder() +
                                                                           "/machine_runtime_operator_trace_" +
                                                                           std::to_string(g_last_round_num) + ".json");
}
} // namespace npu::tile_fwk::dynamic
