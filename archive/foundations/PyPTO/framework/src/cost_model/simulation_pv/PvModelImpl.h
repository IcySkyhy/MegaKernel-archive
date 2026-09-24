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
 * \file PvModelImpl.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <list>
#include <regex>
#include <dlfcn.h>
#include <fstream>
#include "utils/file_utils.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation/common/CommonTools.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "tilefwk/core_func_data.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

constexpr int INVALID_ARG_INDEX = 0xFFFFFFFF;
using namespace npu::tile_fwk;

namespace CostModel {
constexpr uint8_t MAIN_BLOCK_SIZE = 2;
const uint32_t PV_REG_PC = 0;
const uint32_t PV_REG_PARA_BASE = 4;
const uint32_t PV_REG_BLOCK_DIM = 9;
const uint32_t PV_REG_TASK_CFG = 163;
const uint32_t PV_STEP_PIPE_ID = 0;
const uint32_t PV_SYS_VA_BASE = 67;
const uint32_t PV_SYS_PHY_BASE = 68;
constexpr uint32_t PV_MEM_GM = 0;
constexpr uint32_t PV_REG_SPR = 1;
constexpr uint32_t PV_MODE_NORMAL = 0;
uint32_t HBM_PARA_BASE = 0xffff8000;

class PvModelCodegen {
public:
    static void AddGlobalAttr(std::string srcPath)
    {
        const std::string searchStr = "[aicore]";
        const std::string replaceStr = "extern \"C\" __global__ [aicore]";

        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        content = ReplaceAll(content, searchStr, replaceStr);

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        outFile << content;
        outFile.close();
    }

    static void AddKernelEntry(std::string srcPath)
    {
        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        std::string line;
        std::string include_lines;
        std::string other_lines;
        SeparateHeadersAndContent(include_lines, content, other_lines);

        outFile << include_lines;
        auto name = ExtractFunctionName(content);

        std::string decName = R"!!!(
extern "C" [aicore] void {KernelName}(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat);

)!!!";
        std::string entry = R"!!!(
extern "C" __global__ [aicore] void PvModelKernelEntry(__gm__ npu::tile_fwk::DynFuncHeader *funcHeader, __gm__ uint64_t *opAttrs) {
    __gm__ npu::tile_fwk::DynFuncData *funcData = (__gm__ npu::tile_fwk::DynFuncData*)(funcHeader + 1);
    CoreFuncParam param = {funcData, funcHeader, opAttrs, funcData->exprTbl};
    {KernelName}(&param, funcHeader->stackWorkSpaceAddr, (__gm__ int64_t *)funcHeader->startArgs->commContexts, (__gm__ TaskStat*)NULL);
}

)!!!";
        decName = ReplaceAll(decName, "{KernelName}", name);
        entry = ReplaceAll(entry, "{KernelName}", name);
        outFile << decName;
        outFile << entry;
        outFile << other_lines;
        outFile.close();
    }

private:
    static void SeparateHeadersAndContent(std::string& headers, const std::string& content, std::string& otherContent)
    {
        std::istringstream stream(content);
        std::string line;

        while (std::getline(stream, line)) {
            if (line.find("#include") == 0) {
                headers += line + "\n";
            } else {
                otherContent += line + "\n";
            }
        }
    }

    static std::string ReplaceAll(std::string str, const std::string& from, const std::string& to)
    {
        size_t startPos = 0;
        while ((startPos = str.find(from, startPos)) != std::string::npos) {
            str.replace(startPos, from.length(), to);
            startPos += to.length();
        }
        return str;
    }

    static std::string ExtractFunctionName(const std::string& code)
    {
        std::string functionName;
        std::regex functionPattern(R"(\b\w+\s+(\w+)\s*\([^)]*\))");
        std::smatch match;

        std::string::const_iterator searchStart(code.cbegin());
        std::regex_search(searchStart, code.cend(), match, functionPattern);
        if (match.size() > 1) {
            functionName = match[1].str();
        }

        return functionName;
    }
};

// Dynamic
class DynPvModelImpl : public DynPvModel {
private:
    npu::tile_fwk::Function* func_;
    std::string dir_;
    struct DataMap {
        uint8_t* data;
        uint64_t devPtr;
        uint64_t size;
    };
    std::vector<DataMap> data_;
    DataMap workspace_;
    std::vector<std::vector<uint8_t>> storage_;

    struct PvModelCceBin {
        uint32_t psgId;
        uint64_t funcHash;
        npu::tile_fwk::CoreType coreType;
        std::string srcPath;
        std::string binPath;
        PvModelCceBin(uint32_t p, uint64_t h, npu::tile_fwk::CoreType t, std::string s = "", std::string b = "")
            : psgId(p), funcHash(h), coreType(t), srcPath(s), binPath(b)
        {}
    };
    std::vector<PvModelCceBin> cceBin;
    uint64_t subcoreId_ = 0;
    uint64_t coreId_ = 0;
    uint64_t binAddr_ = 0xffffc000;

public:
    using PvInitFunc = void (*)(int pv_mode, int hj_switch, int pv_wrap, const char* out_dir, uint32_t core_id);
    using PvLaunchSubCoreFunc = void (*)(uint64_t pc, const char* bin_file, uint32_t sub_core_id, uint32_t core_id);
    using PvStepFunc = uint32_t (*)(uint32_t pipe_id, uint32_t sub_core_id, uint32_t core_id, uint32_t warp_id);
    using PvMemWriteFunc = void (*)(uint32_t mem_type, uint64_t addr, uint64_t size, uint8_t* buf, uint32_t sub_core_id,
                                    uint32_t core_id);
    using PvMemReadFunc = void (*)(uint32_t mem_type, uint64_t addr, uint64_t size, uint8_t* buf, uint32_t sub_core_id,
                                   uint32_t core_id);
    using PvRegWriteFunc = void (*)(uint32_t reg_type, uint32_t reg_id, uint8_t* buf, uint32_t sub_core_id,
                                    uint32_t core_id);

    explicit DynPvModelImpl()
    {
        dir_ = npu::tile_fwk::config::LogTopFolder() + "/PvModelOutput";
        if (npu::tile_fwk::IsPathExist(dir_)) {
            npu::tile_fwk::DeleteDir(dir_, true);
        }
        npu::tile_fwk::CreateDir(dir_);
    }

    void InitPv()
    {
        auto archType = npu::tile_fwk::Platform::Instance().GetSoc().GetNPUArch();
        const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
        if (ascendHome == nullptr) {
            throw std::runtime_error("ASCEND_HOME_PATH environment variable not set");
        }
        std::string archTypeStr = NPUArchToString(archType);
        std::transform(archTypeStr.begin(), archTypeStr.end(), archTypeStr.begin(), ::tolower);
        std::string soPath = std::string(ascendHome) + "/toolkit/tools/simulator/" + archTypeStr +
                             (archType == NPUArch::DAV_3510 ? "/camodel/libpem_davinci.so" : "/lib/libpem_davinci.so");
        if (archType == NPUArch::DAV_3510) {
            std::string camodelDir = std::string(ascendHome) + "/toolkit/tools/simulator/" + archTypeStr + "/camodel";
            std::string lib64Dir = std::string(ascendHome) + "/lib64";
            std::string loaderC = dir_ + "/camodel_loader.c";
            std::string loaderSo = dir_ + "/camodel_loader.so";
            std::ofstream ofs(loaderC);
            ofs << "int pypto_camodel_loader_marker = 0;\n";
            ofs.close();
            constexpr int loaderCmdLen = 4096;
            char loaderCmd[loaderCmdLen];
            int ret = snprintf_s(loaderCmd, sizeof(loaderCmd), sizeof(loaderCmd) - 1,
                                 "cc -shared -fPIC -o %s %s -Wl,--disable-new-dtags -Wl,-rpath,%s:%s "
                                 "-Wl,--no-as-needed -L%s -lnpu_drv_camodel -Wl,--as-needed",
                                 loaderSo.c_str(), loaderC.c_str(), camodelDir.c_str(), lib64Dir.c_str(),
                                 camodelDir.c_str());
            if (ret < 0 || ret >= static_cast<int>(sizeof(loaderCmd))) {
                throw std::runtime_error("failed to build camodel loader command");
            }
            auto loaderArgs = SplitString(loaderCmd);
            ret = SafeExecCommand(loaderArgs);
            if (ret != 0) {
                SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::CMD_ERROR, "cmd error: %s", loaderCmd);
                throw std::runtime_error("failed to compile camodel loader");
            }
            void* loader = dlopen(loaderSo.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (!loader) {
                SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::NO_SO_EXISTS, "can not load library: %s",
                                loaderSo.c_str());
                throw std::runtime_error("can not load camodel loader: " + loaderSo);
            }
        }
        void* handle = dlopen((soPath.c_str()), RTLD_LAZY);
        if (!handle) {
            SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::NO_SO_EXISTS, "can not load library: %s",
                            soPath.c_str());
            throw std::runtime_error("can not load library: " + soPath);
        }
        // Load function symbols
        this->pv_init_ = (PvInitFunc)load_symbol(handle, "pv_init");
        this->pv_launch_sub_core_ = (PvLaunchSubCoreFunc)load_symbol(handle, "pv_launch_sub_core");
        this->pv_step_ = (PvStepFunc)load_symbol(handle, "pv_step");
        this->pv_mem_write_ = (PvMemWriteFunc)load_symbol(handle, "pv_mem_write");
        this->pv_mem_read_ = (PvMemReadFunc)load_symbol(handle, "pv_mem_read");
        this->pv_reg_write_ = (PvRegWriteFunc)load_symbol(handle, "pv_reg_write");

        CostModel::OutputSilencer silencer;
        silencer.silence();
        uint8_t* value_0_ptr = reinterpret_cast<uint8_t*>(new uint64_t(0));
        uint8_t* value_1_ptr = reinterpret_cast<uint8_t*>(new uint64_t(1));
        uint8_t* value_34603008_ptr = reinterpret_cast<uint8_t*>(new uint64_t(34603008));
        pv_init_(PV_MODE_NORMAL, 0, 1, (dir_ + std::string("/pvlog/")).c_str(), coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_PARA_BASE, (uint8_t*)&HBM_PARA_BASE, 0, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_PARA_BASE, (uint8_t*)&HBM_PARA_BASE, 1, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_BLOCK_DIM, value_1_ptr, 0, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_BLOCK_DIM, value_1_ptr, 1, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_TASK_CFG, value_1_ptr, 0, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_TASK_CFG, value_1_ptr, 1, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_SYS_VA_BASE, value_0_ptr, 0, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_SYS_VA_BASE, value_0_ptr, 1, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_SYS_PHY_BASE, value_34603008_ptr, 0, coreId_);
        pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_SYS_PHY_BASE, value_34603008_ptr, 1, coreId_);
        silencer.restore();
        SIMULATION_LOGI("pvlog path: %s", (dir_ + std::string("/pvlog/")).c_str());
    }

    void* load_symbol(void* handle, std::string symbol)
    {
        void* func = dlsym(handle, symbol.c_str());
        if (!func) {
            dlclose(handle);
            throw std::runtime_error("Cannot load symbol: " + symbol);
        }
        return func;
    }

    void Codegen(npu::tile_fwk::Function* func)
    {
        auto attr = func->GetDyndevAttribute();
        std::map<std::uint64_t, npu::tile_fwk::Function*> leafDict;
        for (size_t i = 0; i < attr->funcGroup.devRootList.size(); i++) {
            npu::tile_fwk::Function* devRoot = attr->funcGroup.devRootList[i];
            for (auto& [hash, leaf] : devRoot->programs_) {
                (void)hash;
                if (!leafDict.count(leaf->GetFunctionHash().GetHash())) {
                    leafDict[leaf->GetFunctionHash().GetHash()] = leaf;
                }
            }
        }

        cceBin.emplace_back(PvModelCceBin(0, 0, npu::tile_fwk::CoreType::HUB));
        cceBin.emplace_back(PvModelCceBin(0, HUB_MIX_DUMMY_HASH, npu::tile_fwk::CoreType::HUB_MIX));
        int leafIndex = 2;
        for (auto& [hash, leaf] : leafDict) {
            (void)hash;
            auto leafFuncAttr = leaf->GetLeafFuncAttribute();
            bool hasMainBlock = (leafFuncAttr != nullptr) && !leafFuncAttr->binPathMainBlock.empty();
            int baseIdx = hasMainBlock ? (leafIndex * MAIN_BLOCK_SIZE - 1) : leafIndex;
            while (static_cast<int>(cceBin.size()) < baseIdx) {
                cceBin.emplace_back(PvModelCceBin(0, 0, npu::tile_fwk::CoreType::INVALID));
            }
            if (leaf->IsDummyFunction()) {
                cceBin.emplace_back(PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(),
                                                  npu::tile_fwk::CoreType::HUB));
                if (hasMainBlock) {
                    cceBin.emplace_back(PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(),
                                                      npu::tile_fwk::CoreType::HUB));
                }
            } else {
                ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, leafFuncAttr != nullptr)
                    << "LeafFuncAttr is null for " << leaf;
                CompileCode(func, leaf, leafFuncAttr->binPath);
                if (hasMainBlock) {
                    CompileCode(func, leaf, leafFuncAttr->binPathMainBlock);
                }
            }
            leafIndex++;
        }
    }

    void CompileCode(npu::tile_fwk::Function* func, npu::tile_fwk::Function* leaf, std::string binPath)
    {
        int Len2 = 2;
        int Len3 = 3;
        auto leafFuncAttr = leaf->GetLeafFuncAttribute();
        auto orgSrcPath = binPath.substr(0, binPath.length() - 1) + "cpp";
        auto srcPath = binPath.substr(0, binPath.length() - Len2) + "_pvmodel.cpp";
        npu::tile_fwk::CopyFile(orgSrcPath, srcPath);
        PvModelCodegen::AddKernelEntry(srcPath);

        auto objPath = srcPath.substr(0, srcPath.length() - Len3) + "o";
        npu::tile_fwk::CodeGenCtx ctx;
        npu::tile_fwk::CodeGenCloudNPU cga(ctx);
        auto coreType = leafFuncAttr == nullptr ? npu::tile_fwk::CoreType::INVALID : leafFuncAttr->coreType;
        bool isCube = coreType == npu::tile_fwk::CoreType::AIC;
        npu::tile_fwk::CompileInfo compileInfo(*func, ctx, {leaf->GetProgramId(), leaf}, isCube);
        compileInfo.SetCCEAbsPath(srcPath);
        compileInfo.SetBinAbsPath(objPath);
        cga.CompileCode(cga.PrepareCmd(compileInfo, ""));

        binPath = srcPath.substr(0, srcPath.length() - Len3) + "bin";
        constexpr int cmdLen = 2048;
        char cmd[cmdLen];
        CHECK(static_cast<unsigned>(CostModel::ExternalErrorScene::INVALID_PATH), npu::tile_fwk::IsPathExist(objPath))
            << "obj file does not exist. objPath: " << objPath;
        int ret = snprintf_s(cmd, sizeof(cmd), sizeof(cmd) - 1, "llvm-objcopy -O binary -j .text %s %s",
                             objPath.c_str(), binPath.c_str());
        if (ret < 0 || ret >= static_cast<int>(sizeof(cmd))) {
            SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::CMD_ERROR, "snprintf_s: %s", cmd);
        }
        auto args = SplitString(cmd);
        ret = SafeExecCommand(args);
        if (ret != 0) {
            SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::CMD_ERROR, "cmd error: %s", cmd);
        }

        cceBin.emplace_back(
            PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(), coreType, srcPath, binPath));
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size)
    {
        std::vector<uint8_t> s(data, data + size);
        uint8_t* devPtr = s.data();
        storage_.emplace_back(std::move(s));
        return devPtr;
    }

    uint8_t* CopyTensorToDev(uint8_t* data, uint64_t size)
    {
        std::vector<uint8_t> s(data, data + size);
        uint8_t* devPtr = s.data();
        pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(devPtr), size, devPtr, 0, 0);
        storage_.emplace_back(std::move(s));
        DataMap m = {data, reinterpret_cast<uint64_t>(devPtr), size};
        data_.emplace_back(m);
        return devPtr;
    }

    uint8_t* AllocWorkspace(uint64_t size)
    {
        std::vector<uint8_t> s(size, 0);
        uint8_t* devPtr = s.data();
        storage_.emplace_back(std::move(s));
        return devPtr;
    }

    void CopyTensorFromDev()
    {
        for (auto& d : data_) {
            pv_mem_read_(PV_MEM_GM, d.devPtr, d.size, d.data, 0, 0);
        }
    }

    void Run(DynFuncData* funcdata, int coreId, int funcId, int taskId);

private:
    void RunModel(PvModelCceBin* cce, DynFuncData* funcdata, uint64_t* opAttrs);

    PvInitFunc pv_init_;
    PvLaunchSubCoreFunc pv_launch_sub_core_;
    PvStepFunc pv_step_;
    PvMemWriteFunc pv_mem_write_;
    PvMemReadFunc pv_mem_read_;
    PvRegWriteFunc pv_reg_write_;
    enum class step_status_t { END = 0, NORMAL = 1, TIME_OUT = 2, CONTINUE = 3, UNDEF };
};
} // namespace CostModel
