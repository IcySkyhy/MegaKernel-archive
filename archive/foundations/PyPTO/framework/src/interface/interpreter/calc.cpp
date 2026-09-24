/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <dlfcn.h>

#include "tilefwk/error.h"
#include "interface/interpreter/interpreter_log.h"
#include "utils/file_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::calc {

typedef struct CalcOps* (*GetCalcOpsFunc)();

struct CalcOps* GetCalcOps()
{
    static std::once_flag once_;
    static struct CalcOps* calcOps = nullptr;

    std::call_once(once_, []() {
    /* whl 包场景在 whl 包 init 时即完成 so 加载, 故可直接从当前进程加载符号 */
#ifndef ENABLE_TESTS
        auto handle = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);
        if (handle == nullptr) {
            INTERPRETER_LOGE_FULL(VerifyEnableScene::VERIFY_LOAD_CALC_OPS_FAILED, "Can't get program handle");
            return;
        }
#else
        std::string path = GetPyptoLibPath() + "/libtile_fwk_calculator.so";
        if (!IsPathExist(path)) {
            return;
        }
        auto handle = dlopen(path.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            INTERPRETER_LOGI("torch not found please check the library path or import torch first");
            return;
        }
#endif
        auto func = dlsym(handle, "GetCalcOps");
        calcOps = reinterpret_cast<GetCalcOpsFunc>(func)();
    });

    return calcOps;
}
} // namespace npu::tile_fwk::calc
