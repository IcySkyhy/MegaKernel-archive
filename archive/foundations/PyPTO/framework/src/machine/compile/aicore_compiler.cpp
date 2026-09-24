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
 * \file aicore_compiler.cpp
 * \brief
 */

#include "machine/compile/aicore_compiler.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "codegen/utils/parallel_execute.h"
#include "utils/file_utils.h"
#include "interface/utils/common.h"
#include "interface/utils/op_info_manager.h"
#include "machine/compile/gen_aicore_code.h"
#include "machine/host/main_block.h"
#include "machine/utils/checkinject.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace {
constexpr const size_t CMD_SIZE_1K = 1024;
constexpr const size_t CMD_SIZE_2K = 2048;
constexpr const char* BISHENG_PROGRAM_CMD = "bisheng";
constexpr const char* BISHENG_LD_CMD = "ld.lld";
} // namespace

static int CompileCoreMachine(const std::string& objFile, bool isCube, uint64_t tilingKey, const std::string& headFile,
                              const std::string& aicoreSrcFile, const std::string& funcRawName)
{
    MACHINE_LOGI("Compile src file is [%s], kernel type[%d].", aicoreSrcFile.c_str(), isCube);
    const std::string cc_opt = isCube ? Platform::Instance().GetSoc().GetCCECVersion("AIC") :
                                        Platform::Instance().GetSoc().GetCCECVersion("AIV");
    const std::string coreType = isCube ? "-D__AIC__" : "-D__AIV__";
    const auto& opType = OpInfoManager::GetInstance().GetOpType();
    std::string hasSubFunc = headFile.empty() ? "" : "-D__HAS_SUB_FUNC__";
    const std::string davArch = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) ? "-D__DAV_V220" :
                                                                                                    "-D__DAV_V310";
    const std::string enableMainBlock = ((config::GetPassGlobalConfig(KEY_ENABLE_VF, false) ||
                                          config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1)) ?
                                            "-D__ENABLE_MAIN_BLOCK" :
                                            "";
    const std::string enableAicoreResolve = IsAicoreResolveEnabled() ? "-DENABLE_AICORE_RESOLVE=1" :
                                                                       "-DENABLE_AICORE_RESOLVE=0";
    std::string ccecCmd;
    ccecCmd.resize(CMD_SIZE_2K);
    std::string includePath = GetPyptoLibPath() + "/../include/tile_fwk";
    int ret = snprintf_s(ccecCmd.data(), CMD_SIZE_2K, CMD_SIZE_2K - 1,
                         "%s -c -O3 -g -x cce -Wall -Werror -std=c++17 "
                         "--cce-aicore-only "
                         "--cce-aicore-arch=%s "
                         "-mllvm -cce-aicore-stack-size=0x8000 "
                         "-mllvm -cce-aicore-function-stack-size=0x8000 "
                         "-mllvm -cce-aicore-record-overflow=false "
                         "-mllvm -cce-aicore-addr-transform "
                         "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
                         "-D__MIX__ "
                         "-D__TILINGKEY__=%s "
                         "-D__OPTYPE__=%s "
                         "-D__OPNAME__=%s "
                         "-D__HEAD_FILE__=%s "
                         "%s "
                         "%s "
                         "%s "
                         "-I%s/tileop/arch32 "
                         "-I%s/ "
                         "-I%s/include/tileop/arch32 "
                         "-I%s/include/ "
                         "-o %s %s %s %s",
                         BISHENG_PROGRAM_CMD, cc_opt.c_str(), std::to_string(tilingKey).c_str(), opType.c_str(),
                         funcRawName.c_str(), headFile.c_str(), hasSubFunc.c_str(), coreType.c_str(),
                         enableAicoreResolve.c_str(), includePath.c_str(), includePath.c_str(),
                         GetPyptoLibPath().c_str(), GetPyptoLibPath().c_str(), objFile.c_str(), aicoreSrcFile.c_str(),
                         davArch.c_str(), enableMainBlock.c_str());
    if (ret < 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_AICORE_FAILED, "Compile aicore construct cmd failed.");
        return ret;
    }
    ret = Checkinject(ccecCmd.c_str(), ccecCmd.size());
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_AICORE_FAILED, "Compile aicore cmd illegal char.");
        return ret;
    }
    ret = std::system(ccecCmd.c_str());
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED, "Compile ccec failed.");
    }
    return ret;
}

std::string GenSubFuncCall(std::map<uint64_t, Function*>& leafDict, CoreType coreType,
                           dynamic::EncodeDevAscendFunctionParam& param, const std::string& ccePath, uint64_t tilingKey,
                           std::stringstream& src_obj, bool enableSubFunc)
{
    if (!enableSubFunc) {
        return "";
    }
    std::stringstream code;
    std::map<int, std::string> idxNameMap;
    // Declare extern leaf func.
    for (const auto& iter : leafDict) {
        const auto leaf = iter.second;
        int leafIndex = param.calleeHashIndexDict[leaf->ComputeHash().GetHash()];
        auto leafFuncAttr = leaf->GetLeafFuncAttribute();
        ASSERT(ProgEncodeErr::LEAF_CALLEE_ATTR_NULL, leafFuncAttr != nullptr) << "LeafFuncAttr is null for " << leaf;
        if (coreType != leafFuncAttr->coreType) {
            continue;
        }
        int baseIdx = leafFuncAttr->binPathMainBlock.empty() ? leafIndex : leafIndex * MAIN_BLOCK_SIZE - 1;
        src_obj << leafFuncAttr->binPath << " ";
        code << leafFuncAttr->kernelDeclare << std::endl;
        idxNameMap[baseIdx] = leafFuncAttr->kernelName;
        if (!leafFuncAttr->binPathMainBlock.empty()) {
            src_obj << leafFuncAttr->binPathMainBlock << " ";
            code << leafFuncAttr->kernelDeclareMainBlock << std::endl;
            idxNameMap[baseIdx + 1] = leafFuncAttr->kernelNameMainBlock;
        }
        MACHINE_LOGD("Func[%d], kernel_name[%s].", leafIndex, leafFuncAttr->kernelName.c_str());
    }
    if (idxNameMap.empty()) {
        return "";
    }
    // Define call sub func inline function.
    code << "__attribute__((always_inline)) inline __aicore__ void CallSubFuncTask(uint64_t funcIdx, ";
    code << "CoreFuncParam *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat) {\n";
    code << "    switch (funcIdx) {\n";
    for (const auto& iter : idxNameMap) {
        MACHINE_LOGD("Call sub func id[%d], kernel_name[%s].", iter.first, iter.second.c_str());
        code << "        case " << std::to_string(iter.first) << ": {\n";
        code << "            " << iter.second << "(param, gmStackAddr, hcclContext, taskStat);\n";
        code << "            break;\n";
        code << "        }\n";
    }
    code << "        default:\n";
    code << "            return;\n";
    code << "    };\n    return;\n}\n";
    MACHINE_LOGD("Sub func call code [\n%s\n].", code.str().c_str());

    std::string head_file = (coreType == CoreType::AIC) ? "sub_func_aic_call_" : "sub_func_aiv_call_";
    head_file = ccePath + head_file + std::to_string(tilingKey) + ".h";
    FILE* fsrc = fopen(head_file.c_str(), "w");
    if (fsrc == nullptr) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Fail to open call.h.");
        return "";
    }
    (void)fprintf(fsrc, "%s", code.str().c_str());
    (void)fclose(fsrc);
    return head_file;
}

static int RunLdViaSecureTempScript(const std::string& ccePath, const std::string& key, const std::string& ccecCmd)
{
    std::string linkScriptTemplate = ccePath + "link_" + key + std::to_string(getpid()) + "_XXXXXX";
    std::vector<char> linkScriptBuffer(linkScriptTemplate.begin(), linkScriptTemplate.end());
    linkScriptBuffer.push_back('\0');
    int fd = mkstemp(linkScriptBuffer.data());
    if (fd < 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Create secure link script failed.");
        return -1;
    }
    const std::string linkScript(linkScriptBuffer.data());
    if (fchmod(fd, S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Set secure link script permission failed, file[%s].",
                     linkScript.c_str());
        (void)close(fd);
        (void)unlink(linkScript.c_str());
        return -1;
    }
    const std::string scriptContent = std::string("#!/bin/bash\n") + ccecCmd;
    ssize_t writtenSize = write(fd, scriptContent.data(), scriptContent.size());
    if (writtenSize < 0 || static_cast<size_t>(writtenSize) != scriptContent.size()) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Write secure link script failed, file[%s].", linkScript.c_str());
        (void)close(fd);
        (void)unlink(linkScript.c_str());
        return -1;
    }
    if (close(fd) != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Close secure link script failed, file[%s].", linkScript.c_str());
        (void)unlink(linkScript.c_str());
        return -1;
    }
    const std::string ldCmd = "bash " + linkScript;
    int ret = Checkinject(ldCmd.c_str(), ldCmd.size());
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Link kernel cmd illegal char.");
        return ret;
    }
    ret = std::system(ldCmd.c_str());
    if (unlink(linkScript.c_str()) != 0) {
        MACHINE_LOGW("Cleanup secure link script failed, file[%s].", linkScript.c_str());
    }
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Link kernel failed.");
    }
    return ret;
}

static int LinkObject(const std::string& src_objs, std::string& objPath, const std::string& ccePath, bool relocate,
                      const std::string& key)
{
    size_t cmdSize = src_objs.size() + objPath.size() + CMD_SIZE_1K;
    std::string ccecCmd;
    ccecCmd.resize(cmdSize);
    int ret = snprintf_s(ccecCmd.data(), cmdSize, cmdSize - 1,
                         "%s -m aicorelinux -Ttext=0 -static %s -o "
                         "%s "
                         "%s",
                         BISHENG_LD_CMD, relocate ? "-r" : "", objPath.c_str(), src_objs.c_str());
    if (ret < 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "LinkCoreMachine construct cmd failed.");
        return ret;
    }
    MACHINE_LOGD("Link ccec command: [%s].", ccecCmd.c_str());
    return RunLdViaSecureTempScript(ccePath, key, ccecCmd);
}

int CompileAICoreKernel(std::map<uint64_t, Function*>& leafDict, dynamic::EncodeDevAscendFunctionParam& param,
                        const std::string& ccePath, const std::string& funcHash, const std::string& funcRawName,
                        std::string& kernelPath, bool enableSubFunc)
{
    if (ccePath.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "No cce path.");
        return -1;
    }
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    std::string aic_obj = ccePath + "dy_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
    std::string aiv_obj = ccePath + "dy_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
    std::string aicoreSrcFile = ccePath + "aicore.cpp";
    if (!GenAicoreSrcFile(aicoreSrcFile, funcHash)) {
        MACHINE_LOGE(HostBackEndErr::GEN_AICORE_FILE_FAILED, "Fail to generate aicore src file.");
        return -1;
    }
    std::deque<std::function<void(void)>> tasks;
    auto task = [&ccePath, &funcHash, &funcRawName, &leafDict, &param, &aic_obj, &aicoreSrcFile, &tilingKey,
                 enableSubFunc]() {
        // gen switch case func
        std::stringstream src_aic_obj;
        auto headFile = GenSubFuncCall(leafDict, CoreType::AIC, param, ccePath, tilingKey, src_aic_obj, enableSubFunc);
        std::string mid_aic_obj = ccePath + "mid_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
        auto ret = CompileCoreMachine(mid_aic_obj, true, tilingKey, headFile, aicoreSrcFile, funcRawName);
        ASSERT(HostBackEndErr::COMPILE_AICORE_FAILED, ret == 0)
            << "CompileCoreMachine failed with return code  " << ret;
        src_aic_obj << mid_aic_obj;
        ret = LinkObject(src_aic_obj.str(), aic_obj, ccePath, true, "aic");
        ASSERT(HostBackEndErr::LINK_FAILED, ret == 0) << "LinkObject failed with return code  " << ret;
        return;
    };
    tasks.push_back(task);

    auto task1 = [&ccePath, &funcHash, &funcRawName, &leafDict, &param, &aiv_obj, &aicoreSrcFile, &tilingKey,
                  enableSubFunc]() {
        std::stringstream src_aiv_obj;
        auto headFile = GenSubFuncCall(leafDict, CoreType::AIV, param, ccePath, tilingKey, src_aiv_obj, enableSubFunc);
        std::string mid_aiv_obj = ccePath + "mid_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
        auto ret = CompileCoreMachine(mid_aiv_obj, false, tilingKey, headFile, aicoreSrcFile, funcRawName);
        ASSERT(HostBackEndErr::COMPILE_AICORE_FAILED, ret == 0)
            << "CompileCoreMachine failed with return code  " << ret;
        src_aiv_obj << mid_aiv_obj;
        ret = LinkObject(src_aiv_obj.str(), aiv_obj, ccePath, true, "aiv");
        ASSERT(HostBackEndErr::LINK_FAILED, ret == 0) << "LinkObject failed with return code  " << ret;
        return;
    };
    tasks.push_back(task1);
    ParallelExecuteAndWait(tasks.size(), tasks);

    std::stringstream src_obj;
    src_obj << " " << aic_obj << " " << aiv_obj;
    kernelPath = ccePath + "dy_kernel_" + funcHash + "_" + std::to_string(tilingKey) + ".o";
    MACHINE_LOGD("Compile dynamic kernel to %s.", kernelPath.c_str());
    auto ret = LinkObject(src_obj.str(), kernelPath, ccePath, false, "mix");
    return ret;
}

} // namespace npu::tile_fwk
