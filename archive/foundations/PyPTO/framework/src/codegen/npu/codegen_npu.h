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
 * \file codegen_npu.h
 * \brief
 */

#ifndef CODEGEN_NPU_H
#define CODEGEN_NPU_H

#include <unordered_set>
#include <utility>
#include <mutex>
#include <vector>
#include <chrono>
#include <stack>
#include <thread>

#include "tilefwk/platform.h"
#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
const std::string ENV_ASCEND_HOME_PATH = "ASCEND_HOME_PATH";
const std::string ENV_PTO_TILE_LIB_CODE_PATH = "PTO_TILE_LIB_CODE_PATH";
constexpr const int64_t CODE_RESERVED_SIZE = 1024 * 1024;

struct CompileTaskInfo {
    std::string outputPath;
    std::string inputPath;
    std::string compileCmd;
};

/* The 64-bit subTilingKey embedded in the kernel name must be reproducible under
 * multi-threaded codegen; otherwise fixed-CCE debug cannot match on-disk symbol
 * names with the names allocated in this compile. The id is therefore derived
 * from the task identity, not from completion order.
 * Layout (pure concat, LSB -> MSB):
 *   [0:0]     isMainBlock   (1)
 *   [1:19]    subProgramId  (19)
 *   [20:31]   devRootIndex  (12)
 *   [32:63]   folded topFuncHash (32) = low32(hash ^ (hash >> 32))
 * All fields are fixed before tasks are dispatched and can be read safely in parallel.
 */
constexpr uint32_t SUB_ID_MAIN_BLOCK_BITS = 1U;
constexpr uint32_t SUB_ID_PROGRAM_BITS = 19U;
constexpr uint32_t SUB_ID_DEV_ROOT_BITS = 12U;
constexpr uint32_t SUB_ID_HASH_BITS = 32U;
constexpr uint32_t SUB_ID_PROGRAM_SHIFT = SUB_ID_MAIN_BLOCK_BITS;
constexpr uint32_t SUB_ID_DEV_ROOT_SHIFT = SUB_ID_PROGRAM_SHIFT + SUB_ID_PROGRAM_BITS;
constexpr uint32_t SUB_ID_HASH_SHIFT = SUB_ID_DEV_ROOT_SHIFT + SUB_ID_DEV_ROOT_BITS;
constexpr uint64_t SUB_ID_PROGRAM_MAX = (1UL << SUB_ID_PROGRAM_BITS) - 1UL;
constexpr uint64_t SUB_ID_DEV_ROOT_MAX = (1UL << SUB_ID_DEV_ROOT_BITS) - 1UL;
constexpr uint64_t SUB_ID_HASH_MASK = (1UL << SUB_ID_HASH_BITS) - 1UL;

class CompileInfo {
public:
    CompileInfo(Function& topFunc, const CodeGenCtx& ctx, const std::pair<uint64_t, Function*>& subFuncPair,
                bool isCube)
        : userSpecCCEDir_(ctx.cceDir),
          isCube_(isCube),
          attr_(subFuncPair.second->GetLeafFuncAttribute()),
          isMainBlock_(ctx.isMainBlock),
          subProgramId_(subFuncPair.first),
          devRootIndex_(ctx.devRootIndex),
          topFuncHash_(topFunc.GetFunctionHash().GetHash())
    {
        Init(topFunc, subFuncPair.first);
    };
    virtual ~CompileInfo() = default;

    std::string GetCCEAbsPath() const { return cceAbsPath_; }
    void SetCCEAbsPath(const std::string& cceAbsPath) { cceAbsPath_ = cceAbsPath; }

    std::string GetBinAbsPath() const { return binAbsPath_; }
    void SetBinAbsPath(const std::string& binAbsPath) { binAbsPath_ = binAbsPath; }
    void SetKernelName(const std::string& kernelName) { kernelName_ = kernelName; }
    std::string GetKernelName() const { return kernelName_; }
    void SetFuncDeclare(const std::string& funcDeclare) { funcDeclare_ = funcDeclare; }
    std::string GetFuncDeclare() const { return funcDeclare_; }
    bool IsCube() const { return isCube_; }
    uint64_t GetSubProgramId() const { return subProgramId_; }
    const std::string& GetCCEFileName() const { return cceFileName_; }
    uint64_t GetDeterministicSubId() const;

protected:
    void Init(Function& topFunc, uint64_t subProgramId)
    {
        std::string coreType = isCube_ ? "aic" : "aiv";
        std::ostringstream ss;
        std::ostringstream tailStr;

        if ((attr_ != nullptr) && (attr_->mixId != -1)) {
            tailStr << "mix" << attr_->mixId << "_" << coreType;
            if (!isCube_) {
                int aivId = static_cast<int>(attr_->aivCore);
                if (aivId == -1) {
                    tailStr << "x";
                } else {
                    tailStr << aivId;
                }
            }
        } else {
            tailStr << coreType;
        }

        if (isMainBlock_) {
            tailStr << "_main";
        }

        ss << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subProgramId << "_" << tailStr.str();
        cceFileName_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << GetSuffix();
        cceAbsPath_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".o";
        binAbsPath_ = ss.str();
    }
    std::string GetSuffix() const
    {
        std::string suffix = ".cpp";
        return suffix;
    }

    std::string userSpecCCEDir_;
    bool isCube_{false};
    std::string cceFileName_;
    std::string cceAbsPath_;
    std::string binAbsPath_;
    std::string kernelName_;
    std::string funcDeclare_;
    std::shared_ptr<LeafFuncAttribute> attr_{nullptr};
    bool isMainBlock_{false};
    uint64_t subProgramId_{0};
    int devRootIndex_{0};
    uint64_t topFuncHash_{0};
};

class CodeGenNPU : public CodeGenCCE {
public:
    explicit CodeGenNPU(const CodeGenCtx& cgCtx) : CodeGenCCE(cgCtx)
    {
        platform_ = Platform::Instance().GetSoc().GetNPUArch();
    };
    ~CodeGenNPU() override = default;

    void GenCode(Function& topFunc) override;
    std::string PrepareCmd(const CompileInfo& compileInfo, const std::string& compileOptions) const;
    // only used to compile code directly when running under simulation mode.
    void CompileCode(const std::string& compileCmd) const;
    void GenAllocForLocalBuffer(const Operation& op, const std::shared_ptr<SymbolManager>& sm);
    virtual std::string GetCoreArch(const CompileInfo& compileInfo) const;
    static void AppendVFOptions(std::ostringstream& oss, NPUArch platform, bool isCube);

protected:
    void GenExtraAlloc(const std::shared_ptr<SymbolManager>& sm, const std::shared_ptr<LogicalTensor>& tensor);
    void GenFuncBodyBefore(const std::pair<uint64_t, Function*>& subFuncPair, Function& topFunc,
                           CompileInfo& compileInfo, std::ostringstream& oss) const;
    void GenInclude(const Function& topFunc, std::ostringstream& oss) const;
    void GenCommentBeforeFuncHeader(Function& subFunc, std::ostringstream& oss) const;
    std::string GenFuncHeader(Function& topFunc, CompileInfo& compileInfo) const;
    virtual void GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss) = 0;
    void GenFuncEnd(std::ostringstream& oss) const;
    static std::string GenKernelName(Function& topFunc, const CompileInfo& compileInfo);

    void GenCodeToBinaryTask(std::ostringstream& code, const CompileInfo& compileInfo,
                             const std::string& compileOptions) const;
    bool IsNeedDumpCode(const std::string& inputFile) const;
    void DumpCode(const std::string& name, std::ostringstream& code) const;
    int DoCompileCmd(const std::string& compileCmd) const;

    virtual void BuildArchOptions(std::ostringstream& oss, const CompileInfo& compileInfo) const;
    void Prepare(const Function& topFunc);

    virtual void BuildIncludes(std::ostringstream& oss) const;
    virtual void BuildExtraOptions(std::ostringstream& oss, const CompileInfo& compileInfo,
                                   const std::string& compileOptions) const;

    void GenAlloc(const std::shared_ptr<SymbolManager>& sm, BufferType bufferType,
                  const std::shared_ptr<LogicalTensor>& tensor);

    void GenDynParamForExpr(std::ostringstream& oss, const Function& func) const;

    bool HandleForAICpuSubFunc(Function& subFunc);

    void UpdateSubFunc(std::pair<uint64_t, Function*> subFuncPair, const CompileInfo& compileInfo) const;

    std::string GetIncludePathForCompileCCE() const;
    std::string GetPtoTileLibPathByEnv() const;

    void CollectCompileTask(const CompileTaskInfo& task) const;
    void GenerateMakefile(const std::string& makefilePath) const;
    void ExecuteParallelCompile(const Function& topFunc);
    std::string GetOutputDir() const;
    void GenDDRChecker(std::ostringstream& oss) const;

    mutable std::mutex compileTasksMutex_;
    mutable std::vector<CompileTaskInfo> compileTasks_;
    NPUArch platform_;
    std::string rootFuncName_;
    int rootFuncOpSize_{0};
};

class FloatSpecValMgr {
public:
    void UpdateByOp(const Operation& op);
    void PrintFloatSpecVal(std::ostringstream& oss);

private:
    std::set<FloatSpecVal> floatSpecVals_;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_NPU_H
