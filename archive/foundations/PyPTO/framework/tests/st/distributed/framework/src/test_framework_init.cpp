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
 * \file test_framework_init.cpp
 * \brief
 */

#include <string>
#include <future>
#include <vector>
#include <sstream>
#include <mutex>
#include <dlfcn.h>
#include <algorithm>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/hcomm_api.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "distributed_test_framework.h"
namespace npu::tile_fwk {
namespace Distributed {
namespace {

// Thread-safe environment variable accessor
class ThreadSafeEnv {
private:
    static std::once_flag initFlag_;
    static std::string mpiHomePath_;
    static std::string deviceIdList_;

    static void initialize()
    {
        const char* envPath = std::getenv("MPI_HOME");
        if (envPath) {
            mpiHomePath_ = envPath;
        }

        const char* envDeviceList = std::getenv("TILE_FWK_DEVICE_ID_LIST");
        if (envDeviceList) {
            deviceIdList_ = envDeviceList;
        }
    }

public:
    static const std::string& getMPIHomePath()
    {
        std::call_once(initFlag_, initialize);
        return mpiHomePath_;
    }

    static const std::string& getDeviceIdList()
    {
        std::call_once(initFlag_, initialize);
        return deviceIdList_;
    }
};

std::once_flag ThreadSafeEnv::initFlag_;
std::string ThreadSafeEnv::mpiHomePath_;
std::string ThreadSafeEnv::deviceIdList_;

// 定义MPI类型
using MPI_Comm = int;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
using MPI_Datatype = int;
#define MPI_CHAR ((MPI_Datatype)0x4c000101)
using MPI_Group = int;
struct MPI_Status {
    int data[10];
};

// 定义MPI函数类型
using MpiInitFunc = int (*)(int*, char***);
using MpiCommSizeFunc = int (*)(MPI_Comm, int*);
using MpiCommRankFunc = int (*)(MPI_Comm, int*);
using MpiBcastFunc = int (*)(void*, int, MPI_Datatype, int, MPI_Comm);
using MpiBarrierFunc = int (*)(MPI_Comm);
using MpiAbortFunc = int (*)(MPI_Comm, int);
using MpiFinalizeFunc = int (*)();

// Try several common MPI library paths/names so the test can find MPICH/MPILib without
// requiring system-level changes (e.g., no sudo inside container).
static void* TryOpen(const std::string& path, int flags = RTLD_NOW)
{
    void* h = dlopen(path.c_str(), flags);
    if (h)
        return h;
    return nullptr;
}

std::vector<std::string> BuildMpiCandidatePaths()
{
    std::vector<std::string> candidates;

    // First, try paths from MPI_HOME environment variable (highest priority)
    const std::string& mpiHome = ThreadSafeEnv::getMPIHomePath();
    if (!mpiHome.empty()) {
        DISTRIBUTED_LOGI("Searching MPI libraries in MPI_HOME: %s", mpiHome.c_str());

        std::vector<std::string> mpiHomePaths = {mpiHome + "/lib/libmpi.so",
                                                 mpiHome + "/lib/libmpich.so",
                                                 mpiHome + "/lib/libmpich.so.12",
                                                 mpiHome + "/lib64/libmpi.so",
                                                 mpiHome + "/lib64/libmpich.so",
                                                 mpiHome + "/lib64/libmpich.so.12",
                                                 mpiHome + "/lib/aarch64-linux-gnu/libmpi.so",
                                                 mpiHome + "/lib/x86_64-linux-gnu/libmpi.so"};
        candidates.insert(candidates.end(), mpiHomePaths.begin(), mpiHomePaths.end());
    }

    // Then add hardcoded paths (backwards compatibility and fallback)
    if (candidates.empty() || mpiHome.empty()) {
        const std::vector<std::string> systemPaths = {
            // Original default path - try first for backwards compatibility
            "/usr/local/mpich/lib/libmpi.so",

            // Automatic discovery - common installation paths as fallback
            "/lib/aarch64-linux-gnu/libmpich.so",
            "/lib/x86_64-linux-gnu/libmpich.so",
            "/usr/lib/libmpi.so",
            "/usr/lib/libmpich.so",
            "/lib/libmpi.so",
            "/lib/libmpich.so",
            "/usr/lib/x86_64-linux-gnu/libmpi.so",
            "/usr/lib/aarch64-linux-gnu/libmpi.so",
        };

        candidates.insert(candidates.end(), systemPaths.begin(), systemPaths.end());
    }
    return candidates;
}

void* GetLibHandle()
{
    static std::vector<std::string> candidates = BuildMpiCandidatePaths();
    static auto handle = []() -> void* {
        for (const auto& path : candidates) {
            // If absolute path, try RTLD_NOW|RTLD_NOLOAD first to see if already loaded via that path
            if (!path.empty() && path.front() == '/') {
                void* h = TryOpen(path, RTLD_NOW | RTLD_NOLOAD);
                if (h) {
                    DISTRIBUTED_LOGI("Found already-loaded MPI library: %s", path.c_str());
                    return h;
                }
                h = TryOpen(path, RTLD_NOW);
                if (h) {
                    DISTRIBUTED_LOGI("Loaded MPI library from path: %s", path.c_str());
                    return h;
                }
            } else {
                // symbolic name: let the dynamic loader resolve it using standard search paths
                void* h = TryOpen(path, RTLD_NOW);
                if (h) {
                    DISTRIBUTED_LOGI("Loaded MPI library by name: %s", path.c_str());
                    return h;
                }
            }
        }

        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR,
                         "Failed to load MPI library from common candidate paths/names");
        return static_cast<void*>(nullptr);
    }();
    return handle;
}

// 消除reinterpret_cast
template <typename FuncType>
struct FunctionConverter {
    static auto Convert(void* ptr) -> FuncType
    {
        union {
            void* from;
            FuncType to;
        } converter;

        converter.from = ptr;
        return converter.to;
    }
};

template <typename FuncType>
auto GetFunction(const std::string& funcName) -> FuncType
{
    auto handle = GetLibHandle();
    if (!handle) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "Failed to load MPI library");
        return nullptr;
    }

    auto func = dlsym(handle, funcName.c_str());
    if (!func) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "Failed to find function %s: %s", funcName.c_str(),
                         dlerror());
        return nullptr;
    }
    return FunctionConverter<FuncType>::Convert(func);
}

void InitMpiAndDeviceId(OpTestParam& testParam, int& physicalDeviceId)
{
    auto mpiInit = GetFunction<MpiInitFunc>("MPI_Init");
    CHECK(mpiInit != nullptr) << "MpiInitFunc ptr not found";
    auto mpiCommSize = GetFunction<MpiCommSizeFunc>("MPI_Comm_size");
    CHECK(mpiCommSize != nullptr) << "MpiCommSizeFunc ptr not found";
    auto mpiCommRank = GetFunction<MpiCommRankFunc>("MPI_Comm_rank");
    CHECK(mpiCommRank != nullptr) << "MpiCommRankFunc ptr not found";

    mpiInit(NULL, NULL);

    mpiCommSize(MPI_COMM_WORLD, &testParam.rankSize);
    mpiCommRank(MPI_COMM_WORLD, &testParam.rankId);

    const std::string& dev_list_str = ThreadSafeEnv::getDeviceIdList();
    if (!dev_list_str.empty()) {
        std::vector<int> device_list;
        std::stringstream ss(dev_list_str);
        std::string id;
        while (std::getline(ss, id, ',')) {
            device_list.push_back(std::stoi(id));
        }
        CHECK(testParam.rankId < static_cast<int>(device_list.size())) << "RankID out of range";
        physicalDeviceId = device_list[testParam.rankId];
    } else {
        physicalDeviceId = testParam.rankId;
    }
}

void InitAclNpu(int physicalDeviceId, int rankId)
{
    CHECK(AclInit(NULL) == 0) << "AclInit failed";
    if (rankId == 0) {
        CHECK(RuntimeSetDevice(physicalDeviceId) == 0) << "Set device failed";
    }
    CHECK(AclRtSetDevice(physicalDeviceId) == 0) << "Set device failed";
}
} // namespace

void TestFrameworkInit(OpTestParam& testParam, HcomTestParam& hcomTestParam, int& physicalDeviceId)
{
    auto mpiBcast = GetFunction<MpiBcastFunc>("MPI_Bcast");
    CHECK(mpiBcast != nullptr) << "MpiBcastFunc ptr not found";
    auto mpiBarrier = GetFunction<MpiBarrierFunc>("MPI_Barrier");
    CHECK(mpiBarrier != nullptr) << "MpiBarrierFunc ptr not found";

    InitMpiAndDeviceId(testParam, physicalDeviceId);
    InitAclNpu(physicalDeviceId, testParam.rankId);

    hcomTestParam.rootRank = 0;
    if (testParam.rankId == hcomTestParam.rootRank) {
        CHECK(HcommGetRootInfo(&hcomTestParam.rootInfo) == 0) << "HcommGetRootInfo failed";
    }
    mpiBcast(&hcomTestParam.rootInfo, HCOMM_ROOT_INFO_BYTES, MPI_CHAR, hcomTestParam.rootRank, MPI_COMM_WORLD);
    mpiBarrier(MPI_COMM_WORLD);
    CHECK(HcommCommInitRootInfo(testParam.rankSize, &hcomTestParam.rootInfo, testParam.rankId,
                                &hcomTestParam.hcclComm) == 0)
        << "HcommCommInitRootInfo failed";

    CHECK(HcommGetCommName(hcomTestParam.hcclComm, testParam.group) == 0) << "HcommGetCommName failed";
    setenv("TILE_FWK_DEVICE_ID", std::to_string(physicalDeviceId).c_str(), 1);

    DISTRIBUTED_LOGI("testParam.rankSize %d\n", testParam.rankSize);
    DISTRIBUTED_LOGI("testParam.rankId %d\n", testParam.rankId);
    DISTRIBUTED_LOGI("testParam.group %s\n", testParam.group);
    DISTRIBUTED_LOGI("rootInfo.internal %s\n", hcomTestParam.rootInfo.internal);

    return;
}

void TestFrameworkInit2Groups(OpTestParam& testParam, HcomTestParam& hcomTestParam, int& physicalDeviceId)
{
    auto mpiBarrier = GetFunction<MpiBarrierFunc>("MPI_Barrier");
    CHECK(mpiBarrier != nullptr) << "MpiBarrierFunc ptr not found";
    auto mpiSend = GetFunction<int (*)(void*, int, MPI_Datatype, int, int, MPI_Comm)>("MPI_Send");
    CHECK(mpiSend != nullptr) << "MPI_Send ptr not found";
    auto mpiRecv = GetFunction<int (*)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*)>("MPI_Recv");
    CHECK(mpiRecv != nullptr) << "MPI_Recv ptr not found";

    InitMpiAndDeviceId(testParam, physicalDeviceId);
    InitAclNpu(physicalDeviceId, testParam.rankId);

    HcommRootInfo subRootInfo;
    HcommHandle subComm;

    int subCommId = (testParam.rankId % 2 == 0) ? 0 : 1;
    int subRootRank = subCommId;

    if (testParam.rankId == subRootRank) {
        CHECK(HcommGetRootInfo(&subRootInfo) == 0) << "SubGroup HcommGetRootInfo failed";
        for (int r = subRootRank + 2; r < testParam.rankSize; r += 2) {
            mpiSend(&subRootInfo, HCOMM_ROOT_INFO_BYTES, MPI_CHAR, r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Status status;
        mpiRecv(&subRootInfo, HCOMM_ROOT_INFO_BYTES, MPI_CHAR, subRootRank, 0, MPI_COMM_WORLD, &status);
    }
    mpiBarrier(MPI_COMM_WORLD);

    std::vector<int> subRanks;
    for (int r = subCommId; r < testParam.rankSize; r += 2) {
        subRanks.push_back(r);
    }
    int subRankSize = static_cast<int>(subRanks.size());
    auto it = std::find(subRanks.begin(), subRanks.end(), testParam.rankId);
    CHECK(it != subRanks.end()) << "Current rank not found in subRanks logic error";
    int subRankId = std::distance(subRanks.begin(), it);

    CHECK(HcommCommInitRootInfo(subRankSize, &subRootInfo, subRankId, &subComm) == 0)
        << "SubGroup HcommCommInitRootInfo failed";
    CHECK(HcommGetCommName(subComm, testParam.group) == 0) << "SubGroup HcommGetCommName failed";
    testParam.worldRankId = testParam.rankId;
    testParam.rankId = subRankId;
    testParam.rankSize = subRankSize;
    hcomTestParam.hcclComm = subComm;
    DISTRIBUTED_LOGI("Init SubGroup: %s, SubRank: %d, SubSize: %d", testParam.group, testParam.rankId,
                     testParam.rankSize);
    return;
}

void TestFrameworkDestroy(int32_t timeout)
{
    auto mpiAbort = GetFunction<MpiAbortFunc>("MPI_Abort");
    CHECK(mpiAbort != nullptr) << "MpiAbortFunc ptr not found";
    std::future<void> finalizeTask = std::async([] {
        auto mpiFinalize = GetFunction<MpiFinalizeFunc>("MPI_Finalize");
        CHECK(mpiFinalize != nullptr) << "MpiFinalizeFunc ptr not found";
        mpiFinalize();
    });
    if (finalizeTask.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "MPI_Finalize timeout, forcing exit");
        mpiAbort(MPI_COMM_WORLD, 1);
    }
}

std::string getTimeStamp()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;

    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    constexpr int NUM_SIX = 6;
    timestamp << "_" << std::setw(NUM_SIX) << std::setfill('0') << us;
    return timestamp.str();
}
} // namespace Distributed
} // namespace npu::tile_fwk
