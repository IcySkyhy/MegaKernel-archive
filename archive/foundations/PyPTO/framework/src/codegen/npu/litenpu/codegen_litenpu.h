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
 * \file codegen_litenpu.h
 * \brief
 */

#ifndef CODEGEN_LITENPU_H
#define CODEGEN_LITENPU_H

#include "codegen/npu/codegen_npu.h"

namespace npu::tile_fwk {
const std::string CODEGEN_LITENPU_WORKSPACE = "workspace";

class CompileInfoLiteNPU : public CompileInfo {
public:
    CompileInfoLiteNPU(Function& topFunc, const CodeGenCtx& ctx, const std::pair<uint64_t, Function*>& subFuncPair,
                       bool isCube)
        : CompileInfo(topFunc, ctx, subFuncPair, isCube)
    {
        std::ostringstream ss;
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".json";
        jsonAbsPath_ = ss.str();
    };
    ~CompileInfoLiteNPU() override = default;
    std::string GetJsonAbsPath() const { return jsonAbsPath_; }
    void SetJsonAbsPath(const std::string& jsonAbsPath) { jsonAbsPath_ = jsonAbsPath; }

private:
    std::string jsonAbsPath_;
};

class CodeGenLiteNPU : public CodeGenNPU {
public:
    explicit CodeGenLiteNPU(const CodeGenCtx& cgCtx) : CodeGenNPU(cgCtx) {};
    ~CodeGenLiteNPU() override = default;

    void GenCode(Function& topFunc) override;

private:
    void GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss) override;

    void BuildArchOptions(std::ostringstream& oss, const CompileInfo& compileInfo) const override;

    std::string GetCoreArch(const CompileInfo& compileInfo) const override;

    void AppendLiteNPUVFOptions(std::ostringstream& oss) const;
    void BuildExtraOptions(std::ostringstream& oss, const CompileInfo& compileInfo,
                           const std::string& compileOptions) const override;

    void BuildIncludes(std::ostringstream& oss) const override;

    std::string GenFuncGlobalCodeAfterReplace(const Function& func, std::pair<uint64_t, Function*> subFuncPair,
                                              const std::string& subProgramCode);

    std::unordered_map<int, std::string> GenParamsSymbolMap(const SubfuncParam& subFuncParam,
                                                            std::vector<std::string>& params,
                                                            std::map<std::string, std::string>& dTypeMap);

    std::vector<std::string> GetInOutParams(std::pair<uint64_t, Function*> subFuncPair);

    void GenConfigJson(const std::string& jsonName, const std::string& cppName, const std::string& binName,
                       const std::string& kernelName, const int& workspaceSize,
                       const std::vector<std::string>& argNames, const int& blockDim) const;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
