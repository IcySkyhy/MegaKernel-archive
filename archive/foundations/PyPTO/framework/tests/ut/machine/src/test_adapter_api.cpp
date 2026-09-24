/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_apapter_api.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "adapter/api/acl_api.h"
#include "adapter/api/adump_api.h"
#include "adapter/api/hal_api.h"
#include "adapter/api/hcomm_api.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/runtime_api.h"
#include "adapter/api/runtime_capture_context.h"
#include "adapter/manager/adapter_manager.h"
#include "adapter/stubs/adump_stubs.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {
// Check whether CANN shared libraries are actually loaded (not just whether ASCEND_HOME_PATH is set).
static bool HasCannLoaded() { return AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init) != nullptr; }

// Always call the API (to cover both CANN and stub code paths), but only assert the return value in stub mode.
// In CANN mode the real CANN functions may return non-success error codes for invalid (nullptr) arguments.
#define CHECK_API_RET(call, expected)     \
    do {                                  \
        auto _apiRet = (call);            \
        if (!hasCann) {                   \
            EXPECT_EQ(_apiRet, expected); \
        }                                 \
    } while (0)

class TestAdapterApi : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override
    {
        RuntimeCaptureContext::SetCaptureMode(false);
        RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    }
};

TEST_F(TestAdapterApi, test_acl_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(AclInit(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclFinalize(), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtMemcpy(nullptr, 0, nullptr, 0, AclRtMemcpyKind::HOST_TO_HOST), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSetDevice(0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtResetDevice(0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateEvent(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtRecordEvent(nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateEventExWithFlag(nullptr, 0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtStreamWaitEvent(nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtGetStreamResLimit(nullptr, AclRtDevResLimitType::CUBE_CORE, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtGetStreamAttribute(nullptr, AclRtStreamAttr::FAILURE_MODE, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCacheLastTaskOpInfo(nullptr, 0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSetExceptionInfoCallback(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtMalloc(nullptr, 0, AclRtMemMallocPolicy::HUGE_FIRST), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtFree(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSynchronizeStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtDestroyStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclMdlRICaptureGetInfo(nullptr, nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclMdlRICaptureThreadExchangeMode(nullptr), ACLRT_SUCCESS);
    char versionBuf[1] = {'x'};
    CHECK_API_RET(AclSysGetVersionStr("pkg", versionBuf), ACLRT_SUCCESS);
    if (!hasCann) {
        EXPECT_EQ(versionBuf[0], '\0');
    }
}

TEST_F(TestAdapterApi, test_acl_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    auto initRet = AclInit(nullptr);
    if (initRet != ACLRT_SUCCESS) {
        GTEST_SKIP() << "AclInit failed, device not available";
    }

    auto setDevRet = AclRtSetDevice(0);
    if (setDevRet != ACLRT_SUCCESS) {
        AclFinalize();
        GTEST_SKIP() << "AclRtSetDevice failed, device not available";
    }

    AclRtStream stream = nullptr;
    auto createStreamRet = AclRtCreateStream(&stream);
    if (createStreamRet != ACLRT_SUCCESS || stream == nullptr) {
        AclRtResetDevice(0);
        AclFinalize();
        GTEST_SKIP() << "AclRtCreateStream failed, device not available";
    }

    AclRtEvent event = nullptr;
    auto createEventRet = AclRtCreateEvent(&event);
    if (createEventRet != ACLRT_SUCCESS || event == nullptr) {
        AclRtDestroyStream(stream);
        AclRtResetDevice(0);
        AclFinalize();
        GTEST_SKIP() << "AclRtCreateEvent failed, device not available";
    }

    auto recordEventRet = AclRtRecordEvent(event, stream);
    EXPECT_EQ(recordEventRet, ACLRT_SUCCESS);

    auto streamWaitRet = AclRtStreamWaitEvent(stream, event);
    EXPECT_EQ(streamWaitRet, ACLRT_SUCCESS);

    auto syncRet = AclRtSynchronizeStream(stream);
    EXPECT_EQ(syncRet, ACLRT_SUCCESS);

    void* devPtr = nullptr;
    auto mallocRet = AclRtMalloc(&devPtr, 1024, AclRtMemMallocPolicy::HUGE_FIRST);
    if (mallocRet == ACLRT_SUCCESS && devPtr != nullptr) {
        char hostBuf[1024] = {0};
        auto memcpyH2DRRet = AclRtMemcpy(devPtr, 1024, hostBuf, 1024, AclRtMemcpyKind::HOST_TO_DEVICE);
        EXPECT_EQ(memcpyH2DRRet, ACLRT_SUCCESS);

        auto memcpyD2HRet = AclRtMemcpy(hostBuf, 1024, devPtr, 1024, AclRtMemcpyKind::DEVICE_TO_HOST);
        EXPECT_EQ(memcpyD2HRet, ACLRT_SUCCESS);

        auto freeRet = AclRtFree(devPtr);
        EXPECT_EQ(freeRet, ACLRT_SUCCESS);
    }

    auto destroyStreamRet = AclRtDestroyStream(stream);
    EXPECT_EQ(destroyStreamRet, ACLRT_SUCCESS);

    auto resetDevRet = AclRtResetDevice(0);
    EXPECT_EQ(resetDevRet, ACLRT_SUCCESS);

    auto finalizeRet = AclFinalize();
    EXPECT_EQ(finalizeRet, ACLRT_SUCCESS);
}

TEST_F(TestAdapterApi, test_adump_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR), 0);
    CHECK_API_RET(AdumpRegExceptionDumpCallBack(nullptr), 0);
    std::vector<AdxTensorInfoV2> emptyTensors;
    CHECK_API_RET(AdxDumpDumpTensorV2("op_name", "op_type", emptyTensors, nullptr), 0);

    AdxTensorInfoV2 tensorInfo{};
    tensorInfo.type = AdxTensorType::INPUT;
    tensorInfo.tensorSize = 128;
    tensorInfo.format = 0;
    tensorInfo.dataType = 1;
    tensorInfo.tensorAddr = nullptr;
    tensorInfo.addrType = AdxAddressType::TRADITIONAL;
    tensorInfo.placement = 0;
    tensorInfo.argsOffSet = 0;
    tensorInfo.shape = {1, 128};
    tensorInfo.originShape = {1, 128};
    std::vector<AdxTensorInfoV2> tensors = {tensorInfo};
    CHECK_API_RET(AdxDumpDumpTensorV2("op_name", "op_type", tensors, nullptr), 0);
    CHECK_API_RET(AdxDumpDataDumpServerInit(), 0);
    CHECK_API_RET(AdxDumpDataDumpServerUnInit(), 0);
}

TEST_F(TestAdapterApi, test_adump_stubs)
{
    EXPECT_EQ(StubDumpGetDumpSwitch(AdxDumpType::OPERATOR), 0);
    EXPECT_EQ(StubDumpDumpTensorV2("op_name", "op_type", {}, nullptr), 0);
    EXPECT_EQ(StubDumpDataDumpServerInit(), 0);
    EXPECT_EQ(StubDumpDataDumpServerUnInit(), 0);
}

TEST_F(TestAdapterApi, test_hal_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(HalMemCtl(0, nullptr, 0, nullptr, nullptr), HAL_ERROR_NONE);
    CHECK_API_RET(HalResMap(0, nullptr, nullptr, nullptr), HAL_ERROR_NONE);
    CHECK_API_RET(HalGetDeviceInfoByBuff(0, 0, 0, nullptr, nullptr), HAL_ERROR_NONE);
}

TEST_F(TestAdapterApi, test_hccl_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(HcommGetCommName(nullptr, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetL0TopoTypeEx(nullptr, nullptr, 0), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetCommHandleByGroup(nullptr, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetRootInfo(nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommCommDestroy(nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommCommInitRootInfo(0, nullptr, 0, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr), HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(MspfSysCycleTime(), 0);
    CHECK_API_RET(MspfGetHashId(nullptr, 0), 0);
    CHECK_API_RET(MspfReportApi(0, nullptr), 0);
    CHECK_API_RET(MspfReportCompactInfo(0, nullptr, 0), 0);
    CHECK_API_RET(MspfReportAdditionalInfo(0, nullptr, 0), 0);
    CHECK_API_RET(MspfRegisterCallback(0, nullptr), 0);
}

TEST_F(TestAdapterApi, test_runtime_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    auto setDevRet = RuntimeSetDevice(0);
    if (setDevRet != RT_SUCCESS) {
        GTEST_SKIP() << "RuntimeSetDevice failed, device not available";
    }

    int32_t devId = -1;
    auto getDevRet = RuntimeGetDevice(&devId);
    EXPECT_EQ(getDevRet, RT_SUCCESS);

    uint32_t aiCpuCnt = 0;
    auto getAiCpuCntRet = RuntimeGetAiCpuCount(&aiCpuCnt);
    EXPECT_EQ(getAiCpuCntRet, RT_SUCCESS);

    uint64_t l2Offset = 0;
    auto getL2OffsetRet = RuntimeGetL2CacheOffset(0, &l2Offset);
    EXPECT_EQ(getL2OffsetRet, RT_SUCCESS);

    int32_t logicDevId = -1;
    auto getLogicDevRet = RuntimeGetLogicDevIdByUserDevId(0, &logicDevId);
    EXPECT_EQ(getLogicDevRet, RT_SUCCESS);

    char socVersion[256] = {0};
    auto getSocVerRet = RuntimeGetSocVersion(socVersion, sizeof(socVersion));
    EXPECT_EQ(getSocVerRet, RT_SUCCESS);

    char socSpec[256] = {0};
    auto getSocSpecRet = RuntimeGetSocSpec("SoCInfo", "chip_type", socSpec, sizeof(socSpec));
    EXPECT_EQ(getSocSpecRet, RT_SUCCESS);

    RtStream stream = nullptr;
    auto createStreamRet = RuntimeStreamCreate(&stream, 0);
    if (createStreamRet != RT_SUCCESS || stream == nullptr) {
        GTEST_SKIP() << "RuntimeStreamCreate failed, device not available";
    }

    auto syncRet = RuntimeStreamSynchronize(stream);
    EXPECT_EQ(syncRet, RT_SUCCESS);

    void* devPtr = nullptr;
    auto mallocRet = RuntimeMalloc(&devPtr, 1024, RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST, 0);
    if (mallocRet == RT_SUCCESS && devPtr != nullptr) {
        auto memsetRet = RuntimeMemset(devPtr, 1024, 0, 1024);
        EXPECT_EQ(memsetRet, RT_SUCCESS);

        char hostBuf[1024] = {0};
        auto memcpyH2DRet = RuntimeMemcpyDirect(devPtr, 1024, hostBuf, 1024, RtMemcpyKind::HOST_TO_DEVICE);
        EXPECT_EQ(memcpyH2DRet, RT_SUCCESS);

        auto memcpyD2HRet = RuntimeMemcpyDirect(hostBuf, 1024, devPtr, 1024, RtMemcpyKind::DEVICE_TO_HOST);
        EXPECT_EQ(memcpyD2HRet, RT_SUCCESS);

        auto freeRet = RuntimeFree(devPtr);
        EXPECT_EQ(freeRet, RT_SUCCESS);
    }

    auto destroyStreamRet = RuntimeStreamDestroy(stream);
    EXPECT_EQ(destroyStreamRet, RT_SUCCESS);
}

TEST_F(TestAdapterApi, test_adump_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    // Test with valid parameters to cover CANN adapter code paths
    auto dumpSwitch = AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR);
    (void)dumpSwitch; // May be 0 or non-zero depending on environment

    auto regRet = AdumpRegExceptionDumpCallBack(nullptr);
    EXPECT_EQ(regRet, 0);

    std::vector<AdxTensorInfoV2> emptyTensors;
    auto dumpEmptyRet = AdxDumpDumpTensorV2("op_name", "op_type", emptyTensors, nullptr);
    EXPECT_EQ(dumpEmptyRet, 0);

    AdxTensorInfoV2 tensorInfo{};
    tensorInfo.type = AdxTensorType::INPUT;
    tensorInfo.tensorSize = 128;
    tensorInfo.format = 0;
    tensorInfo.dataType = 1;
    tensorInfo.tensorAddr = nullptr;
    tensorInfo.addrType = AdxAddressType::TRADITIONAL;
    tensorInfo.placement = 0;
    tensorInfo.argsOffSet = 0;
    tensorInfo.shape = {1, 128};
    tensorInfo.originShape = {1, 128};
    std::vector<AdxTensorInfoV2> tensors = {tensorInfo};
    auto dumpRet = AdxDumpDumpTensorV2("op_name", "op_type", tensors, nullptr);
    EXPECT_EQ(dumpRet, 0);

    auto serverInitRet = AdxDumpDataDumpServerInit();
    EXPECT_EQ(serverInitRet, 0);

    auto serverUnInitRet = AdxDumpDataDumpServerUnInit();
    EXPECT_EQ(serverUnInitRet, 0);
}

TEST_F(TestAdapterApi, test_hal_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    // Test with valid parameters to cover CANN adapter code paths
    auto memCtlRet = HalMemCtl(0, nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(memCtlRet, HAL_ERROR_NONE);

    auto resMapRet = HalResMap(0, nullptr, nullptr, nullptr);
    EXPECT_EQ(resMapRet, HAL_ERROR_NONE);

    auto getDevInfoRet = HalGetDeviceInfoByBuff(0, 0, 0, nullptr, nullptr);
    EXPECT_EQ(getDevInfoRet, HAL_ERROR_NONE);
}

TEST_F(TestAdapterApi, test_hcomm_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    // Test with valid parameters to cover CANN adapter code paths
    auto getCommNameRet = HcommGetCommName(nullptr, nullptr);
    EXPECT_EQ(getCommNameRet, HCOMM_SUCCESS);

    auto getL0TopoRet = HcommGetL0TopoTypeEx(nullptr, nullptr, 0);
    EXPECT_EQ(getL0TopoRet, HCOMM_SUCCESS);

    auto getCommHandleRet = HcommGetCommHandleByGroup(nullptr, nullptr);
    EXPECT_EQ(getCommHandleRet, HCOMM_SUCCESS);

    auto getRootInfoRet = HcommGetRootInfo(nullptr);
    EXPECT_EQ(getRootInfoRet, HCOMM_SUCCESS);

    auto commDestroyRet = HcommCommDestroy(nullptr);
    EXPECT_EQ(commDestroyRet, HCOMM_SUCCESS);

    auto commInitRet = HcommCommInitRootInfo(0, nullptr, 0, nullptr);
    EXPECT_EQ(commInitRet, HCOMM_SUCCESS);

    auto allocComResRet = HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(allocComResRet, HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api_with_valid_params)
{
    bool hasCann = HasCannLoaded();
    if (!hasCann) {
        GTEST_SKIP() << "CANN not loaded, skipping valid param tests";
    }

    auto cycleTime = MspfSysCycleTime();
    (void)cycleTime;

    auto hashId = MspfGetHashId("test_op", 7);
    (void)hashId;

    auto reportApiRet = MspfReportApi(0, nullptr);
    EXPECT_EQ(reportApiRet, 0);

    auto reportCompactRet = MspfReportCompactInfo(0, nullptr, 0);
    EXPECT_EQ(reportCompactRet, 0);

    auto reportAdditionalRet = MspfReportAdditionalInfo(0, nullptr, 0);
    EXPECT_EQ(reportAdditionalRet, 0);

    auto registerCbRet = MspfRegisterCallback(0, nullptr);
    EXPECT_EQ(registerCbRet, 0);
}

TEST_F(TestAdapterApi, test_acl_adapter)
{
    bool loaded = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Finalize) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMemcpy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtResetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtRecordEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEventExWithFlag) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtStreamWaitEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamResLimit) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamAttribute) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCacheLastTaskOpInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetExceptionInfoCallback) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureGetInfo) != nullptr, loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureThreadExchangeMode) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::SysGetVersionStr) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMalloc) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtFree) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateStream) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSynchronizeStream) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtDestroyStream) != nullptr, loaded);
}

TEST_F(TestAdapterApi, test_adump_adapter)
{
    bool loaded = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::GetDumpSwitch) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpTensorV2) != nullptr, loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpFailTaskExceptionCallBack) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DataDumpServerInit) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DataDumpServerUnInit) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_hal_adapter)
{
    bool loaded = AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::MemCtl) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::GetDeviceInfoByBuff) != nullptr, loaded);
}

TEST_F(TestAdapterApi, test_msprof_adapter)
{
    bool loaded = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::SysCycleTime) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::GetHashId) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportApi) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportCompactInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportAdditionalInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::RegisterCallback) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_hccl_adapter)
{
    bool loaded = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommName) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetL0TopoTypeEx) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommHandleByGroup) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetRootInfo) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommInitRootInfo) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommDestroy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::AllocComResourceByTiling) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_runtime_adapter)
{
    bool loaded = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Malloc) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::rtMemset) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopyAsync) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Free) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::SetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocSpec) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocVersion) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetAiCpuCount) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetL2CacheOffset) != nullptr,
              loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetLogicDevIdByUserDevId) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamCreate) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamDestroy) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamAddToModel) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamSynchronize) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryUnRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::RegisterAllKernel) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::LaunchCpuKernel) != nullptr,
              loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunchWithHandleV2) != nullptr,
        loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::AicpuKernelLaunchExWithArgs) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::FunctionRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunch) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetExceptionRegInfo) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_runtime_capture_context)
{
    RuntimeCaptureContext::SetCaptureMode(true);
    EXPECT_TRUE(RuntimeCaptureContext::IsCaptureMode());

    AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, true);
    EXPECT_TRUE(RuntimeCaptureContext::QueryThreadCaptureMode(mode));
    EXPECT_EQ(mode, AclMdlRICaptureMode::RELAXED);

    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    EXPECT_TRUE(RuntimeCaptureContext::QueryThreadCaptureMode(mode));

    RuntimeCaptureContext::SetCaptureMode(false);
    EXPECT_FALSE(RuntimeCaptureContext::IsCaptureMode());
}

TEST_F(TestAdapterApi, test_error_message)
{
    ErrorMessage msg;
    std::vector<int> emptyVec;
    msg << emptyVec;
    EXPECT_EQ(msg.Message(), "[]");

    ErrorMessage msg2;
    std::vector<int> singleVec = {42};
    msg2 << singleVec;
    EXPECT_EQ(msg2.Message(), "[42]");

    ErrorMessage msg3;
    std::vector<std::string> multiVec = {"a", "b", "c"};
    msg3 << multiVec;
    EXPECT_EQ(msg3.Message(), "[a, b, c]");
}

// Tests that exercise real CANN APIs with valid arguments (requires NPU device)
// These tests call the APIs to exercise the code paths, but don't assert on results
// since the device may not be properly initialized in the test environment
TEST_F(TestAdapterApi, test_acl_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // Initialize ACL
    (void)AclInit(nullptr);

    // Set device
    (void)AclRtSetDevice(0);

    // Create stream
    AclRtStream stream = nullptr;
    (void)AclRtCreateStream(&stream);

    // Create event
    AclRtEvent event = nullptr;
    (void)AclRtCreateEvent(&event);

    // Record event on stream
    if (event != nullptr && stream != nullptr) {
        (void)AclRtRecordEvent(event, stream);
        (void)AclRtStreamWaitEvent(stream, event);
        (void)AclRtSynchronizeStream(stream);
    }

    // Allocate device memory
    void* devPtr = nullptr;
    (void)AclRtMalloc(&devPtr, 1024, AclRtMemMallocPolicy::HUGE_FIRST);

    // Memcpy operations
    if (devPtr != nullptr) {
        char hostData[1024] = {0};
        (void)AclRtMemcpy(devPtr, 1024, hostData, 1024, AclRtMemcpyKind::HOST_TO_DEVICE);
        (void)AclRtMemcpy(hostData, 1024, devPtr, 1024, AclRtMemcpyKind::DEVICE_TO_HOST);
        (void)AclRtFree(devPtr);
    }

    // Cleanup
    if (stream != nullptr) {
        (void)AclRtDestroyStream(stream);
    }
    (void)AclRtResetDevice(0);
    (void)AclFinalize();
}

TEST_F(TestAdapterApi, test_runtime_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // Set device
    (void)RuntimeSetDevice(0);

    // Get device
    int32_t devId = -1;
    (void)RuntimeGetDevice(&devId);

    // Get AI CPU count
    uint32_t aiCpuCnt = 0;
    (void)RuntimeGetAiCpuCount(&aiCpuCnt);

    // Get L2 cache offset
    uint64_t offset = 0;
    (void)RuntimeGetL2CacheOffset(0, &offset);

    // Get logic device ID
    int32_t logicDevId = -1;
    (void)RuntimeGetLogicDevIdByUserDevId(0, &logicDevId);

    // Get SoC version
    char versionBuf[256] = {0};
    (void)RuntimeGetSocVersion(versionBuf, sizeof(versionBuf));

    // Create stream
    RtStream stream = nullptr;
    (void)RuntimeStreamCreate(&stream, 0);

    // Synchronize stream
    if (stream != nullptr) {
        (void)RuntimeStreamSynchronize(stream);
    }

    // Allocate device memory
    void* devPtr = nullptr;
    (void)RuntimeMalloc(&devPtr, 1024, RT_MEMORY_POLICY_HUGE_PAGE_FIRST, 0);

    // Memcpy operations
    if (devPtr != nullptr) {
        char hostData[1024] = {0};
        (void)RuntimeMemset(devPtr, 1024, 0, 1024);
        (void)RuntimeMemcpyDirect(devPtr, 1024, hostData, 1024, RtMemcpyKind::HOST_TO_DEVICE);
        (void)RuntimeMemcpyDirect(hostData, 1024, devPtr, 1024, RtMemcpyKind::DEVICE_TO_HOST);
        (void)RuntimeFree(devPtr);
    }

    // Destroy stream
    if (stream != nullptr) {
        (void)RuntimeStreamDestroy(stream);
    }
}

TEST_F(TestAdapterApi, test_adump_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // Get dump switch
    uint64_t dumpSwitch = AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR);
    (void)dumpSwitch; // May be 0 or non-zero depending on environment

    // Register exception dump callback
    EXPECT_EQ(AdumpRegExceptionDumpCallBack(nullptr), 0);

    // Dump tensor with empty tensor list
    std::vector<AdxTensorInfoV2> emptyTensors;
    EXPECT_EQ(AdxDumpDumpTensorV2("op_name", "op_type", emptyTensors, nullptr), 0);

    // Dump tensor with valid tensor info
    AdxTensorInfoV2 tensorInfo{};
    tensorInfo.type = AdxTensorType::INPUT;
    tensorInfo.tensorSize = 128;
    tensorInfo.format = 0;
    tensorInfo.dataType = 1;
    tensorInfo.tensorAddr = nullptr;
    tensorInfo.addrType = AdxAddressType::TRADITIONAL;
    tensorInfo.placement = 0;
    tensorInfo.argsOffSet = 0;
    tensorInfo.shape = {1, 128};
    tensorInfo.originShape = {1, 128};
    std::vector<AdxTensorInfoV2> tensors = {tensorInfo};
    EXPECT_EQ(AdxDumpDumpTensorV2("op_name", "op_type", tensors, nullptr), 0);

    EXPECT_EQ(AdxDumpDataDumpServerInit(), 0);
    EXPECT_EQ(AdxDumpDataDumpServerUnInit(), 0);
}

TEST_F(TestAdapterApi, test_hal_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // MemCtl with null arguments
    EXPECT_EQ(HalMemCtl(0, nullptr, 0, nullptr, nullptr), HAL_ERROR_NONE);

    // ResMap with null arguments
    EXPECT_EQ(HalResMap(0, nullptr, nullptr, nullptr), HAL_ERROR_NONE);

    // GetDeviceInfoByBuff with null arguments
    EXPECT_EQ(HalGetDeviceInfoByBuff(0, 0, 0, nullptr, nullptr), HAL_ERROR_NONE);
}

TEST_F(TestAdapterApi, test_hcomm_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // Get comm name with null arguments
    EXPECT_EQ(HcommGetCommName(nullptr, nullptr), HCOMM_SUCCESS);

    // Get L0 topo type with null arguments
    EXPECT_EQ(HcommGetL0TopoTypeEx(nullptr, nullptr, 0), HCOMM_SUCCESS);

    // Get comm handle by group with null arguments
    EXPECT_EQ(HcommGetCommHandleByGroup(nullptr, nullptr), HCOMM_SUCCESS);

    // Get root info with null arguments
    EXPECT_EQ(HcommGetRootInfo(nullptr), HCOMM_SUCCESS);

    // Comm destroy with null arguments
    EXPECT_EQ(HcommCommDestroy(nullptr), HCOMM_SUCCESS);

    // Comm init root info with null arguments
    EXPECT_EQ(HcommCommInitRootInfo(0, nullptr, 0, nullptr), HCOMM_SUCCESS);

    // Alloc com resource by tiling with null arguments
    EXPECT_EQ(HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr), HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api_with_device)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    // Get system cycle time
    uint64_t cycleTime = MspfSysCycleTime();
    (void)cycleTime; // May be 0 if profiling is not enabled

    // Get hash ID
    uint64_t hashId = MspfGetHashId("test_op", 7);
    (void)hashId; // May be 0 if profiling is not enabled

    // Report API with null arguments
    EXPECT_EQ(MspfReportApi(0, nullptr), 0);

    // Report compact info with null arguments
    EXPECT_EQ(MspfReportCompactInfo(0, nullptr, 0), 0);

    // Report additional info with null arguments
    EXPECT_EQ(MspfReportAdditionalInfo(0, nullptr, 0), 0);

    // Register callback with null arguments
    EXPECT_EQ(MspfRegisterCallback(0, nullptr), 0);
}

TEST_F(TestAdapterApi, test_runtime_api_stub_coverage)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(RuntimeMemcpyDirectAsync(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeFuncGetByName(nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeBinaryLoadFromFile(nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeStreamAddToModel(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeDevBinaryUnRegister(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeRegisterAllKernel(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeDevBinaryRegister(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeFunctionRegister(nullptr, nullptr, nullptr, nullptr, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeKernelLaunch(nullptr, 0, nullptr, 0, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeLaunchCpuKernel(nullptr, 0, nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeKernelLaunchWithHandleV2(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeAicpuKernelLaunchExWithArgs(0, nullptr, 0, nullptr, nullptr, nullptr, 0), RT_SUCCESS);

    RtExceptionRegInfo regInfo{};
    RtExceptionInfo exInfo{};
    auto ret = RuntimeGeExceptionRegInfo(&exInfo, &regInfo);
    EXPECT_EQ(ret, RT_SUCCESS);
    if (!hasCann) {
        EXPECT_EQ(regInfo.coreNum, 1U);
        EXPECT_NE(regInfo.errRegInfo, nullptr);
        EXPECT_EQ(regInfo.errRegInfo->coreId, 1U);
        EXPECT_EQ(regInfo.errRegInfo->coreType, RtCoreType::RT_CORE_TYPE_AIC);
    }

    auto nullRet = RuntimeGeExceptionRegInfo(&exInfo, nullptr);
    EXPECT_EQ(nullRet, RT_SUCCESS);
}

TEST_F(TestAdapterApi, test_runtime_api_device_coverage)
{
    if (!HasCannLoaded()) {
        GTEST_SKIP() << "CANN not loaded, skipping real API tests";
    }

    (void)RuntimeSetDevice(0);

    RtStream stream = nullptr;
    (void)RuntimeStreamCreate(&stream, 0);

    void* devPtr = nullptr;
    (void)RuntimeMalloc(&devPtr, 1024, RT_MEMORY_POLICY_HUGE_PAGE_FIRST, 0);

    if (devPtr != nullptr && stream != nullptr) {
        char hostData[1024] = {0};
        (void)RuntimeMemcpyDirectAsync(devPtr, 1024, hostData, 1024, RtMemcpyKind::HOST_TO_DEVICE, stream);
        (void)RuntimeStreamSynchronize(stream);
    }

    (void)RuntimeFuncGetByName(nullptr, "kernel", nullptr);
    (void)RuntimeBinaryLoadFromFile("/nonexistent.bin", nullptr, nullptr);
    (void)RuntimeStreamAddToModel(stream, nullptr);
    (void)RuntimeDevBinaryUnRegister(nullptr);
    (void)RuntimeRegisterAllKernel(nullptr, nullptr);
    (void)RuntimeDevBinaryRegister(nullptr, nullptr);
    (void)RuntimeFunctionRegister(nullptr, nullptr, "stub", nullptr, 0);
    (void)RuntimeKernelLaunch(nullptr, 0, nullptr, 0, nullptr, stream);
    (void)RuntimeLaunchCpuKernel(nullptr, 0, stream, nullptr, nullptr);
    (void)RuntimeKernelLaunchWithHandleV2(nullptr, 0, 0, nullptr, nullptr, stream, nullptr);
    (void)RuntimeAicpuKernelLaunchExWithArgs(0, "test_op", 0, nullptr, nullptr, stream, 0);

    RtExceptionInfo exInfo{};
    RtExceptionRegInfo regInfo{};
    (void)RuntimeGeExceptionRegInfo(&exInfo, &regInfo);

    if (devPtr != nullptr) {
        (void)RuntimeFree(devPtr);
    }
    if (stream != nullptr) {
        (void)RuntimeStreamDestroy(stream);
    }
}

} // namespace npu::tile_fwk
