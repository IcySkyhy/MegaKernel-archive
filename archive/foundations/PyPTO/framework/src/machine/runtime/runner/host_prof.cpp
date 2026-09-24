/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/runner/host_prof.h"

#include <sys/syscall.h>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "tilefwk/error.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "passes/pass_utils/pass_utils.h"

#define CCECPU 25

namespace npu::tile_fwk {
namespace {
const std::string kOpType = "PyPTO";
constexpr uint32_t kFormatNd = 2;
constexpr uint32_t kFormatNz = 29;
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;
} // namespace

HostProf& HostProf::GetInstance()
{
    thread_local HostProf instance;
    return instance;
}

HostProf::~HostProf() {}

uint32_t HostProf::GetHostProfType() const { return (GetProfSwitch() & MSPF_TASK_TIME_L2_MASK) != 0; }

void HostProf::ReportHostProfInfo(RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType,
                                  bool isCore) const
{
    if (GetProfType() == MSPF_COMMANDHANDLE_TYPE_START) {
        uint64_t endTime = MspfSysCycleTime();
        if (isCore) {
            uint32_t mixBlockDim = MIX_BLOCK_DIM;
            blockDim = (mixBlockDim << HIGHT_BIT) | blockDim;
            HostProfReportContextInfo(endTime);
        }
        if ((GetProfSwitch() & MSPF_TASK_TIME_L1_MASK) != 0) {
            HostProfReportNodeInfo(endTime, blockDim, taskType);
        }
        endTime = MspfSysCycleTime();
        (void)HostProfReportApi(startTime, endTime);
    }
    if (taskType == MSPF_GE_TASK_TYPE_MIX_AIC) {
        HostProfReportCacheTaskInfo(stream, blockDim, taskType);
    }
}

uint64_t HostProf::GetProfSwitch() { return profSwitch_; }

uint32_t HostProf::GetProfType() { return profType_; }

uint64_t HostProf::profSwitch_ = 0;
uint32_t HostProf::profType_ = 0;

int32_t HostProf::HostProfInit(uint32_t type, void* data, uint32_t len)
{
    if (data == nullptr || len == 0) {
        MACHINE_LOGW("Para is invalid");
        return -1;
    }
    if (type != static_cast<uint32_t>(RtProfCtrlType::SWITCH)) {
        MACHINE_LOGW("Prof type [%u] is invalid", type);
        return -1;
    }
    if (len < sizeof(MspfCommandHandle)) {
        MACHINE_LOGW("Prof CommandHandle len [%u] is invalid", len);
        return -1;
    }
    MspfCommandHandle* hostProfHandleConfig = reinterpret_cast<MspfCommandHandle*>(data);
    profSwitch_ = hostProfHandleConfig->profSwitch;
    profType_ = hostProfHandleConfig->type;
    MACHINE_LOGD("Host prof profSwitch is %lu profType is %u", profSwitch_, profType_);
    return 0;
}

void HostProf::RegHostProf() { MspfRegisterCallback(CCECPU, HostProfInit); }

bool HostProf::HostProfReportApi(const uint64_t& startTime, const uint64_t& endTime) const
{
    struct MspfApi apiInfo;
    apiInfo.level = MSPF_REPORT_NODE_LEVEL;
    apiInfo.type = MSPF_REPORT_NODE_LAUNCH_TYPE;
    apiInfo.beginTime = startTime;
    apiInfo.endTime = endTime;
    apiInfo.itemId = MspfGetHashId(opName_.c_str(), opName_.length());
    apiInfo.threadId = syscall(SYS_gettid);
    auto ret = MspfReportApi(true, &apiInfo);
    if (ret != 0) {
        MACHINE_LOGW("Report Api not success");
        return false;
    }
    return true;
}

void HostProf::HostProfReportNodeInfo(const uint64_t& endTime, const uint32_t blockDim, const uint16_t taskType) const
{
    HostProfReportBasicInfo(endTime, blockDim, taskType);
    HostProfReportTensorInfo(endTime);
}

void HostProf::HostProfReportBasicInfo(const uint64_t& endTime, const uint32_t blockDim, const uint16_t taskType) const
{
    struct MspfCompactInfo nodeBasicInfo;
    nodeBasicInfo.level = MSPF_REPORT_NODE_LEVEL;
    nodeBasicInfo.type = MSPF_REPORT_NODE_BASIC_INFO_TYPE;
    nodeBasicInfo.timeStamp = endTime;
    nodeBasicInfo.threadId = syscall(SYS_gettid);
    nodeBasicInfo.data.nodeBasicInfo.opName = MspfGetHashId(opName_.c_str(), opName_.length());
    nodeBasicInfo.data.nodeBasicInfo.opType = MspfGetHashId(kOpType.c_str(), kOpType.length());
    nodeBasicInfo.data.nodeBasicInfo.taskType = taskType;
    nodeBasicInfo.data.nodeBasicInfo.blockDim = blockDim;
    nodeBasicInfo.data.nodeBasicInfo.opFlag = true;
    auto ret = MspfReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
                                     static_cast<uint32_t>(sizeof(MspfCompactInfo)));
    if (ret != 0) {
        MACHINE_LOGW("Compact node[%s] basic info failed", opName_.c_str());
    }
}

void HostProf::HostProfReportContextInfo(const uint64_t& endTime) const
{
    struct MspfAdditionalInfo contextInfo;
    contextInfo.level = MSPF_REPORT_NODE_LEVEL;
    contextInfo.type = MSPF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    contextInfo.threadId = syscall(SYS_gettid);
    contextInfo.timeStamp = endTime;
    struct MspfContextIdInfo ctxId;
    ctxId.opName = MspfGetHashId(opName_.c_str(), opName_.length());
    ctxId.ctxIdNum = 1;
    ctxId.ctxIds[0] = 0;
    MemcpyS(contextInfo.data, MSPF_ADDTIONAL_INFO_DATA_LENGTH, &ctxId, sizeof(MspfContextIdInfo));
    auto ret = MspfReportAdditionalInfo(false, reinterpret_cast<void*>(&contextInfo), sizeof(MspfAdditionalInfo));
    if (ret != 0) {
        MACHINE_LOGW("Op[%s] Msprof report context info not success", opName_.c_str());
    }
}

void HostProf::HostProfReportTensorInfo(const uint64_t& endTime) const
{
    if (profFunction_ == nullptr) {
        MACHINE_LOGW("Op [%s] is null", opName_.c_str());
        return;
    }
    uint32_t iONums = inputsSize_ + oDeviceTensorData_.size();
    MACHINE_LOGD("Op [%s] with inputs[%u], outputs[%zu]", opName_.c_str(), inputsSize_, oDeviceTensorData_.size());
    uint32_t groupNums = iONums / MSPF_GE_TENSOR_DATA_NUM;
    uint32_t modulus = iONums % MSPF_GE_TENSOR_DATA_NUM;
    for (uint32_t i = 0; i < groupNums; i++) {
        ReportTensoInfo(i, MSPF_GE_TENSOR_DATA_NUM, endTime);
    }

    if (modulus > 0) {
        ReportTensoInfo(groupNums, modulus, endTime);
    }
}

void HostProf::ReportTensoInfo(const uint32_t& groupId, const uint32_t mods, const uint64_t& endTime) const
{
    struct MspfAdditionalInfo tensorInfo;
    tensorInfo.level = MSPF_REPORT_NODE_LEVEL;
    tensorInfo.type = MSPF_REPORT_NODE_TENSOR_INFO_TYPE;
    tensorInfo.threadId = syscall(SYS_gettid);
    tensorInfo.timeStamp = endTime;
    auto profTensorData = reinterpret_cast<MspfTensorInfo*>(tensorInfo.data);
    profTensorData->opName = MspfGetHashId(opName_.c_str(), opName_.length());
    profTensorData->tensorNum = mods;
    for (uint32_t j = 0; j < mods; j++) {
        PackTensorInfo(profTensorData, groupId, j);
    }
    auto ret = MspfReportAdditionalInfo(false, reinterpret_cast<void*>(&tensorInfo), sizeof(MspfAdditionalInfo));
    if (ret != 0) {
        MACHINE_LOGW("Op[%s] Msprof report tensor info not success", opName_.c_str());
    }
}

void HostProf::PackTensorInfo(MspfTensorInfo* profTensorData, const uint32_t groupId, const uint32_t modId) const
{
    uint32_t iOIdx = groupId * MSPF_GE_TENSOR_DATA_NUM + modId;
    const npu::tile_fwk::dynamic::DeviceTensorData* iOTensor;
    std::stringstream iOtensorInfo;
    if (inputsSize_ > iOIdx) {
        iOTensor = &iDeviceTensorData_[iOIdx];
        profTensorData->tensorData[modId].tensorType = MSPF_GE_TENSOR_TYPE_INPUT;
        profTensorData->tensorData[modId].format = iOTensor->Format() == TileOpFormat::TILEOP_NZ ? kFormatNz :
                                                                                                   kFormatNd;
        profTensorData->tensorData[modId].dataType = static_cast<uint32_t>(DataType2CannType(iOTensor->GetDataType()));
        iOtensorInfo << "Input " << iOIdx << " shape: ";
    } else {
        auto outputIdx = iOIdx - inputsSize_;
        iOTensor = &oDeviceTensorData_[outputIdx];
        profTensorData->tensorData[modId].tensorType = MSPF_GE_TENSOR_TYPE_OUTPUT;
        profTensorData->tensorData[modId].format = iOTensor->Format() == TileOpFormat::TILEOP_NZ ? kFormatNz :
                                                                                                   kFormatNd;
        profTensorData->tensorData[modId].dataType = static_cast<uint32_t>(DataType2CannType(iOTensor->GetDataType()));
        iOtensorInfo << "output " << outputIdx << " shape: ";
    }
    size_t shapeLen = iOTensor->GetShape().size();
    if (shapeLen > MSPF_GE_TENSOR_DATA_SHAPE_LEN) {
        MACHINE_LOGW("Op [%s] tensor[%u] size[%zu] len over [%d]", opName_.c_str(), iOIdx, shapeLen,
                     MSPF_GE_TENSOR_DATA_SHAPE_LEN);
        shapeLen = MSPF_GE_TENSOR_DATA_SHAPE_LEN;
    }

    for (size_t j = 0; j < shapeLen; j++) {
        profTensorData->tensorData[modId].shape[j] = iOTensor->GetShape()[j];
        iOtensorInfo << iOTensor->GetShape()[j] << " ";
    }
    for (size_t j = shapeLen; j < MSPF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
        profTensorData->tensorData[modId].shape[j] = 0;
    }
    iOtensorInfo << "\n";
    MACHINE_LOGD("tensorInfo %s", iOtensorInfo.str().c_str());
}

void HostProf::BuildTensor(const uint32_t tensorType, const RawTensorDataPtr& tensorInfo, MspfTensorData& tensorData)
{
    tensorData.tensorType = tensorType;
    if (tensorInfo == nullptr) {
        tensorData.format = kFormatNd;
        tensorData.dataType = 0U;
        tensorData.shape[0U] = 0U;
        return;
    }
    tensorData.format = kFormatNd;
    tensorData.dataType = DataType2CannType(tensorInfo->GetDataType());
    for (size_t i = 0; i < tensorInfo->GetShape().size(); i++) {
        tensorData.shape[i] = static_cast<uint32_t>(tensorInfo->GetShape().at(i));
    }
}

void HostProf::BuildTensor(const uint32_t tensorType, const dynamic::DeviceTensorData& tensorInfo,
                           MspfTensorData& tensorData)
{
    tensorData.tensorType = tensorType;
    tensorData.format = tensorInfo.Format() == TileOpFormat::TILEOP_NZ ? kFormatNz : kFormatNd;
    tensorData.dataType = DataType2CannType(tensorInfo.GetDataType());
    size_t dimIdx = 0;
    for (const int64_t dim : tensorInfo.GetShape()) {
        tensorData.shape[dimIdx++] = static_cast<uint32_t>(dim);
    }
}

void HostProf::BuildCacheTensorInfo(CacheTaskInfo* taskInfo) const
{
    if (taskInfo == nullptr) {
        return;
    }
    size_t inputSize = 0;
    const std::vector<RawTensorDataPtr>& inputTensorList = ProgramData::GetInstance().GetInputDataList();
    if (!inputTensorList.empty()) {
        for (size_t i = 0; i < inputTensorList.size(); ++i) {
            BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, inputTensorList.at(i), taskInfo->tensorData[i]);
        }
        inputSize = inputTensorList.size();
        MACHINE_LOGD("Assemble input tensor from program data, input tensor size is [%zu].", inputTensorList.size());
    } else {
        for (size_t i = 0; i < iDeviceTensorData_.size(); ++i) {
            BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, iDeviceTensorData_.at(i), taskInfo->tensorData[i]);
        }
        inputSize = iDeviceTensorData_.size();
        MACHINE_LOGD("Assemble input tensor from device data, input tensor size is [%zu].", iDeviceTensorData_.size());
    }

    const std::vector<RawTensorDataPtr>& outputTensorList = ProgramData::GetInstance().GetOutputDataList();
    if (!outputTensorList.empty()) {
        for (size_t i = 0; i < outputTensorList.size(); ++i) {
            BuildTensor(MSPF_GE_TENSOR_TYPE_OUTPUT, outputTensorList.at(i), taskInfo->tensorData[i + inputSize]);
        }
        MACHINE_LOGD("Assemble output tensor from program data, output tensor size is [%zu].", outputTensorList.size());
    } else {
        for (size_t i = 0; i < oDeviceTensorData_.size(); ++i) {
            BuildTensor(MSPF_GE_TENSOR_TYPE_OUTPUT, oDeviceTensorData_.at(i), taskInfo->tensorData[i + inputSize]);
        }
        MACHINE_LOGD("Assemble output tensor from device data, output tensor size is [%zu].",
                     oDeviceTensorData_.size());
    }
}

bool HostProf::IsCacheOpInfoEnable(const AclRtStream stream)
{
    if (stream == nullptr) {
        return false;
    }
    AclRtStreamAttrValue value = {};
    value.cacheOpInfoSwitch = 0;
    AclError ret = AclRtGetStreamAttribute(stream, AclRtStreamAttr::CACHE_OP_INFO, &value);
    if (ret != ACLRT_SUCCESS) {
        MACHINE_LOGW("Get stream attribute failed, ret is [%d]", ret);
        return false;
    }
    return static_cast<bool>(value.cacheOpInfoSwitch);
}

void HostProf::HostProfReportCacheTaskInfo(const AclRtStream stream, const uint32_t numBlocks,
                                           const uint32_t taskType) const
{
    if (!IsCacheOpInfoEnable(stream)) {
        MACHINE_LOGD("Op cache for AclGraph is disabled.");
        return;
    }
    MACHINE_LOGD("Begin to report op cache [%s], block num[%u], task type [%u].", opName_.c_str(), numBlocks, taskType);
    uint32_t tensorSize = 0;
    if (taskType != MSPF_GE_TASK_TYPE_AI_CPU) {
        tensorSize = ProgramData::GetInstance().GetInputDataList().size() +
                     ProgramData::GetInstance().GetOutputDataList().size();
        if (tensorSize == 0) {
            tensorSize = iDeviceTensorData_.size() + oDeviceTensorData_.size();
        }
    }
    MACHINE_LOGD("Tensor size of op[%s] is [%u].", opName_.c_str(), tensorSize);
    size_t bufferSize = sizeof(CacheTaskInfo) + sizeof(MspfTensorData) * tensorSize;
    void* buffer = malloc(bufferSize);
    if (buffer == nullptr) {
        MACHINE_LOGW("Fail to malloc memory, size is [%zu] bytes", bufferSize);
        return;
    }
    (void)memset_s(buffer, bufferSize, 0, bufferSize);
    CacheTaskInfo* taskInfo = reinterpret_cast<CacheTaskInfo*>(buffer);
    taskInfo->taskType = taskType;
    taskInfo->numBlocks = numBlocks;
    taskInfo->nodeId = MspfGetHashId(opName_.c_str(), opName_.length());
    taskInfo->opType = MspfGetHashId(kOpType.c_str(), kOpType.length());
    taskInfo->attrId = 0;
    taskInfo->opFlag = 0;
    taskInfo->tensorNum = tensorSize;

    if (taskType != MSPF_GE_TASK_TYPE_AI_CPU) {
        BuildCacheTensorInfo(taskInfo);
    }

    if (AclRtCacheLastTaskOpInfo(buffer, bufferSize) != ACLRT_SUCCESS) {
        MACHINE_LOGW("Report op info cache failed for op[%s, %s].", opName_.c_str(), kOpType.c_str());
    } else {
        MACHINE_LOGI("Report op[%s, %s] info cache, task type[%u], numBlocks[%u], attrId[%lu] size[%zu]",
                     opName_.c_str(), kOpType.c_str(), taskType, numBlocks, taskInfo->attrId, bufferSize);
    }
    free(buffer);
}

void HostProf::GetIOTensor(const std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors)
{
    auto directions = profFunction_->GetDyndevAttribute()->startArgsDirectionList;
    iDeviceTensorData_.clear();
    oDeviceTensorData_.clear();
    if (tensors.size() != directions.size()) {
        MACHINE_LOGW("Direction size != tensorData size not support to msprof");
        return;
    }
    for (size_t idx = 0; idx < directions.size(); idx++) {
        if (directions[idx] == ParamDirection ::IN || directions[idx] == ParamDirection ::INOUT) {
            iDeviceTensorData_.emplace_back(tensors[idx]);
        } else if (directions[idx] == ParamDirection ::OUT) {
            oDeviceTensorData_.emplace_back(tensors[idx]);
        }
    }
}

void HostProf::SetProfFunction(Function* function, const std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors)
{
    if (function == nullptr) {
        MACHINE_LOGW("Function is invalid, please check function");
        return;
    }
    // current using functionHashId as opName;
    opName_ = PROFILING_PREFIX + function->GetOriginalRawName();
    profFunction_ = function;
    GetIOTensor(tensors);
    inputsSize_ = iDeviceTensorData_.size();
}
} // namespace npu::tile_fwk
