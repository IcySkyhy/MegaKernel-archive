/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_FWK_AICPU_INTERFACE_H
#define TILE_FWK_AICPU_INTERFACE_H

#include <cstdint>
#include <fstream>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include "machine/utils/device_log.h"

using TileFwkKernelServelEnty = int (*)(void*);
namespace npu::tile_fwk {
const std::string dynServerKernelFun = "DynTileFwkBackendKernelServer";
const std::string dynServerKernelInitFun = "DynTileFwkBackendKernelServerInit";
const uint64_t dyInitFuncKey = 2;
const uint64_t dyExecFuncKey = 3;
const uint64_t minSoLen = 1;

class BackendServerHandleManager {
public:
    bool SaveSoFile(char* data, const uint64_t& len, uint8_t deviceId = 0)
    {
        std::lock_guard<std::mutex> lock(funcLock_);
        if (len < minSoLen || firstCreatSo_) {
            DEV_WARN("Aicpu so len less than 1, don't to copy");
            return true;
        }
        if (pyptoServerSoName_.empty()) {
            pyptoServerSoName_ = "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libpypto_server" +
                                 std::to_string(deviceId) + ".so";
        }
        std::ofstream file(pyptoServerSoName_, std::ios::out | std::ios::binary);
        DEV_DEBUG("Begin to create server.so");
        if (!file) {
            DEV_ERROR(DevCommonErr::FILE_ERROR, "#sche.task.pre.init.resource: Coundn't create file [%s]",
                      pyptoServerSoName_.c_str());
            return false;
        }

        // write bin to file
        file.write(data, len);

        if (!file) {
            DEV_ERROR(DevCommonErr::FILE_ERROR, "#sche.task.pre.init.resource: Write to file [%s] not success",
                      pyptoServerSoName_.c_str());
            return false;
        }
        DEV_DEBUG("Create device[%u] server so [%s] success", deviceId, pyptoServerSoName_.c_str());
        file.close();
        firstCreatSo_ = true;
        return true;
    }

    BackendServerHandleManager() = default;

    void SetTileFwkKernelMap()
    {
        std::lock_guard<std::mutex> lock(funcLock_);
        if (firstLoadSo_) {
            return;
        }
        (void)LoadTileFwkKernelFunc(dynServerKernelInitFun);
        (void)LoadTileFwkKernelFunc(dynServerKernelFun);
        firstLoadSo_ = true;
    }

    inline int32_t ExecuteFunc(void* args, const uint64_t funcKey)
    {
        auto func = GetTileFwkKernelFunc(funcKey);
        if (func == nullptr) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#sche.func.exec: kernel func[%lu] is invalid, cannot get from so %s",
                      funcKey, pyptoServerSoName_.c_str());
            return -1;
        }
        return func(args);
    }

    ~BackendServerHandleManager()
    {
        if (soHandle_) {
            DEV_INFO("Close handle");
            (void)dlclose(soHandle_);
        }
    }

private:
    void LoadTileFwkKernelFunc(const std::string& kernelName)
    {
        if (soHandle_ == nullptr) {
            soHandle_ = dlopen(pyptoServerSoName_.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
        }
        if (!soHandle_) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#sche.task.pre.kernel.load: Cannot open so %s",
                      pyptoServerSoName_.c_str());
            return;
        }
        uint64_t funcKey = 0;
        if (kernelName == dynServerKernelInitFun) {
            funcKey = dyInitFuncKey;
        } else if (kernelName == dynServerKernelFun) {
            funcKey = dyExecFuncKey;
        }
        DEV_DEBUG("Current to open kernel func: name=%s, funcKey=%lu.", kernelName.c_str(), funcKey);
        auto iter = kernelKey2FuncHandle_.find(funcKey);
        if (iter != kernelKey2FuncHandle_.end()) {
            return;
        }

        TileFwkKernelServelEnty tileFwkServrFuncEnty = reinterpret_cast<TileFwkKernelServelEnty>(
            dlsym(soHandle_, kernelName.c_str()));
        if (tileFwkServrFuncEnty == nullptr) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#sche.task.pre.kernel.load: Current KernelName [%s] is null",
                      kernelName.c_str());
            (void)dlclose(soHandle_);
            return;
        }
        DEV_INFO("kernelName=%s has been loaded", kernelName.c_str());
        kernelKey2FuncHandle_[funcKey] = tileFwkServrFuncEnty;
        return;
    }

    TileFwkKernelServelEnty GetTileFwkKernelFunc(const uint64_t funcKey)
    {
        auto iter = kernelKey2FuncHandle_.find(funcKey);
        if (iter != kernelKey2FuncHandle_.end()) {
            return iter->second;
        }
        DEV_ERROR(DevCommonErr::NULLPTR, "#sche.kernel.check: Function[%lu] is null.", funcKey);
        return nullptr;
    }

    std::unordered_map<uint64_t, TileFwkKernelServelEnty> kernelKey2FuncHandle_;
    std::mutex funcLock_;
    void* soHandle_ = nullptr;
    bool firstCreatSo_ = false;
    bool firstLoadSo_ = false;
    std::string pyptoServerSoName_;
};

} // namespace npu::tile_fwk
extern "C" {
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void* args);
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void* args);
}
#endif // TILE_FWK_AICPU_INTERFACE_H
