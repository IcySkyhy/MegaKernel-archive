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
 * \file compile_control_bin.cpp
 * \brief
 */

#include <iosfwd>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/program/program.h"
#include "utils/file_utils.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "machine/utils/checkinject.h"
#include "machine/utils/machine_utils.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
using Json = nlohmann::json;

namespace {
const std::string CustomOpCtrolRunFuncName = "PyptoKernelCtrlServer";
const std::string CustomOpCtrolInitFuncName = "PyptoKernelCtrlServerInit";
const std::string CustomKerneLib = "CUSTKFCKernel";

std::string GetMachineCompilerPath()
{
    constexpr const char* kAscendHomeEnv = "ASCEND_HOME_PATH";
    const std::string homePath = npu::tile_fwk::GetEnvVar(kAscendHomeEnv);
    if (homePath.empty()) {
        MACHINE_LOGW(
            "%s is unset, empty after trimming whitespace, or exceeds max env length; device control-flow compiler "
            "path is disabled.",
            kAscendHomeEnv);
        return "";
    }
    const std::string compiler = homePath + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++";
    if (!npu::tile_fwk::IsPathExist(compiler)) {
        MACHINE_LOGE(npu::tile_fwk::DevCommonErr::FILE_ERROR,
                     "%s is set but aarch64 toolchain g++ was not found at \"%s\" (canonical path must exist).",
                     kAscendHomeEnv, compiler.c_str());
        return "";
    }
    return compiler;
}
const std::string DeviceMahineCompiler = GetMachineCompilerPath();
} // namespace
namespace npu::tile_fwk {

constexpr int DUMP_LEVEL_FOUR = 4;

void GenCustomOpInfo(const std::string& funcName, const std::string& controlAicpuPath,
                     const std::string& constrolSoName)
{
    Json customOp;
    AicpuOpConfig costomInit;
    costomInit.functionName = CustomOpCtrolInitFuncName;
    costomInit.kernelSo = constrolSoName + ".so";
    costomInit.opKernelLib = CustomKerneLib;
    costomInit.opType = funcName + "Init";

    AicpuOpConfig costomRun = costomInit;
    costomRun.opType = funcName + "Run";
    costomRun.functionName = CustomOpCtrolRunFuncName;

    GenAicpuOpInfoJson(customOp, {costomInit, costomRun});
    std::string fileName = controlAicpuPath + "/" + constrolSoName + ".json";
    if (!SaveFile(fileName, customOp.dump(DUMP_LEVEL_FOUR))) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Save custom op json failed");
        return;
    }
    OpInfoManager::GetInstance().GetCustomOpJsonPath() = fileName;
}

bool GenTilingFunc(const std::string& funcName, const std::string& controlAicpuPath)
{
    std::ostringstream oss;
    oss << "#include <vector>\n";
    oss << "#include <string>\n";
    oss << "#include \"controlFlow_dev" << funcName << ".h\"\n";
    oss << "#include <map>\n";
    oss << "#include \""
        << "tilefwk/aicpu_runtime.h"
        << "\"\n";
    oss << "namespace npu::tile_fwk { \n";
    oss << "using controlFlowFuncPtr = uint64_t (*)(void*, int64_t*, RuntimeCallEntryType*, DevStartArgsBase*);\n";
    oss << "namespace " << funcName << "{\n";
    oss << "controlFlowFuncPtr controlFlowptr = ControlFlowEntry;\n";
    oss << "} // end namespace " << funcName << "\n";
    oss << "extern \"C\" __attribute__((visibility(\"default\"))) void* GetCtrlFlowFunc() {\n";
    oss << "return reinterpret_cast<void*>(" << funcName << "::controlFlowptr);\n";
    oss << "}\n";
    oss << "}\n";
    std::string fileName = controlAicpuPath + "/control_flow_kernel.cpp";
    return SaveFile(fileName, oss.str());
}

bool TieFwkAicpuPreCompile(std::string& preCompileO, std::string& controlAicpuPath)
{
    std::stringstream preCompileStream;
    std::string ext = "cpp";
    auto files = GetFiles(controlAicpuPath, ext);
    std::string includePath = GetPyptoLibPath() + "/../include/tilefwk";
    for (auto file : files) {
        std::string objFile = controlAicpuPath + file.substr(0, file.find(".")) + ".o";
        std::string compileCmd = DeviceMahineCompiler + " -Wall -O2 -fPIC -c -std=gnu++17 -fno-common " +
                                 controlAicpuPath + file + " -I" + includePath + " -I" + includePath + "/include/" +
                                 " -I" + GetPyptoLibPath() + "/include/" + " -o " + objFile;
        if (npu::tile_fwk::IsAicoreResolveEnabled()) {
            compileCmd += " -DENABLE_AICORE_RESOLVE=1";
        } else {
            compileCmd += " -DENABLE_AICORE_RESOLVE=0";
        }
        MACHINE_LOGD("PreCompileCmd is %s, file is %s.\n", compileCmd.c_str(), file.c_str());
        int ret = Checkinject(compileCmd.c_str(), compileCmd.size());
        if (ret != 0) {
            MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Precompile %s cmd illegal char.\n", file.c_str());
            return false;
        }
        ret = std::system(compileCmd.c_str());
        if (ret != 0) {
            MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Precompile %s fail\n", file.c_str());
            return false;
        }
        preCompileStream << objFile << " ";
    }
    preCompileO = preCompileStream.str();
    return true;
}

bool SharedAicpuCompile(const std::string& funcName, const std::string& aicpuDirPath, const std::string& preCompileO)
{
    std::string cmdGccCompile = "LD_PRELOAD= " + DeviceMahineCompiler +
                                " -std=gnu++17 -fno-common -shared -fPIC -O2 -Wl,--no-warn-rwx-segments -o " +
                                aicpuDirPath + "/lib" + funcName + "_control.so " + preCompileO +
                                " -Wl,--whole-archive " + GetPyptoLibPath() + "/libpypto_ctrl_server.a" +
                                " -Wl,--no-whole-archive";
    int ret = Checkinject(cmdGccCompile.c_str(), cmdGccCompile.size());
    if (ret != 0) {
        MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "RunDeviceMachine compile cmd illegal char.\n");
        return false;
    }
    ret = std::system(cmdGccCompile.c_str());
    if (ret != 0) {
        MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "RUNDeviceMachine compile fail\n");
        return false;
    }
    std::string srcSoPath = aicpuDirPath + "/lib" + funcName + "_control.so";
    std::string constrolSoName = "lib" + funcName + "_control";
    GenCustomOpInfo(funcName, aicpuDirPath, constrolSoName);
    bool succ = false;
    OpInfoManager::GetInstance().SetControlBuffer(ReadFile(srcSoPath, &succ));
    return succ;
}

bool TileFwkAiCpuCompile(const std::string& funcName, const std::string& aicpuDirPath)
{
    OpInfoManager::GetInstance().GetOpFuncName() = funcName;
    std::string controlAicpuPath = aicpuDirPath + "/" + funcName + "/aicpu/";
    if (!GenTilingFunc(funcName, controlAicpuPath)) {
        MACHINE_LOGE(HostBackEndErr::GEN_DYNAMIC_OP_FAILED, "Gen op[%s]  not success\n", funcName.c_str());
        return false;
    }
    // preCompile
    std::string preCompileO = "";
    if (!TieFwkAicpuPreCompile(preCompileO, controlAicpuPath)) {
        MACHINE_LOGE(HostBackEndErr::PRECOMPILE_FAILED, "Op %s preCompile fail\n", funcName.c_str());
        return false;
    }
    return SharedAicpuCompile(funcName, aicpuDirPath, preCompileO);
}
} // namespace npu::tile_fwk
