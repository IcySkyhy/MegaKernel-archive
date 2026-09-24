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
 * \file symbolic_scalar_evaluate.h
 * \brief
 */
/*for flow Verify Tool */

#pragma once

#include "interface/operation/attribute.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "interface/interpreter/communication.h"
#include "tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {

constexpr int SIZE_TWO = 2;
constexpr int SIZE_THREE = 3;
constexpr int SIZE_FOUR = 4;
constexpr int SIZE_SIX = 6;

struct FunctionIODataPair;
struct FunctionFrame;
class EvaluateSymbol {
public:
    EvaluateSymbol() {}

    void EvaluateDynParam(const std::map<std::string, DynParamInfo>& dynParamTable,
                          const std::vector<SymbolicScalar>& linearArgList)
    {
        for (auto& paramInfo : dynParamTable) {
            std::string symbolName = paramInfo.first;
            if (paramInfo.second.dim.IsValid()) {
                ScalarImmediateType val = EvaluateSymbolicScalar(paramInfo.second.dim, linearArgList);
                UpdateSymbolDict(symbolName, val);
                continue;
            }
            int n = paramInfo.second.tensorIndex;
            (void)n;
            int base = paramInfo.second.tensorBaseAddrCoaIndex;
            int dim = paramInfo.second.dimSize;
            int idx = paramInfo.second.dimIndex;
            int argIndex = ((base) + 1) + 3 * (dim) + idx;

            ScalarImmediateType val = EvaluateSymbolicScalar(linearArgList[argIndex]);
            UpdateSymbolDict(symbolName, val);
        }
    }

    int EvaluateRankId(int hcclIdx)
    {
        const std::vector<std::string> groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
        const std::string& groupName = groupNames[hcclIdx];
        auto ctx = SimulationCommManager::Instance().GetCommContext(groupName);
        return ctx->GetRank();
    }

    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar>& dynValidShape,
                                            const std::vector<SymbolicScalar>& linearArgList = {})
    {
        std::vector<int64_t> result;
        for (auto& shape : dynValidShape) {
            result.push_back(EvaluateSymbolicScalar(shape, linearArgList));
        }
        return result;
    }

    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t>& offset,
                                        const std::vector<SymbolicScalar>& dynOffset,
                                        const std::vector<SymbolicScalar>& linearArgList = {})
    {
        std::vector<int64_t> resultOffset;
        if (dynOffset.size() != 0) {
            for (auto& off : dynOffset) {
                resultOffset.push_back(EvaluateSymbolicScalar(off, linearArgList));
            }
        } else {
            for (auto& off : offset) {
                resultOffset.push_back(off);
            }
        }
        return resultOffset;
    }

    bool RuntimeIsLoopBegin(ScalarImmediateType idx, ScalarImmediateType begin) { return idx == begin; }
    bool RuntimeIsLoopEnd(ScalarImmediateType idx, ScalarImmediateType end) { return idx >= end; }

    ScalarImmediateType EvaluateSymbolicCall(const std::string& name, const std::vector<ScalarImmediateType>& dataList,
                                             const std::vector<SymbolicScalar>& linearArgList);
    ScalarImmediateType EvaluateSymbolicScalar(const RawSymbolicScalarPtr& ss,
                                               const std::vector<SymbolicScalar>& linearArgList = {});
    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar& ss) { return EvaluateSymbolicScalar(ss.Raw()); }
    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar& ss,
                                               const std::vector<SymbolicScalar>& linearArgList)
    {
        return EvaluateSymbolicScalar(ss.Raw(), linearArgList);
    }

    std::unordered_map<std::string, ScalarImmediateType> GetSymbolDict() const { return symbolDict_; }
    void UpdateSymbolDict(const std::string key, const ScalarImmediateType value) { symbolDict_[key] = value; }
    void SetSymbolDict(const std::unordered_map<std::string, ScalarImmediateType>& symbolDict)
    {
        symbolDict_ = symbolDict;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> GetInputDataViewList() const { return inputDataViewList_; }
    void UpdateInputDataViewList(size_t index, const std::shared_ptr<LogicalTensorData>& inputDataView)
    {
        inputDataViewList_[index] = inputDataView;
    }
    void InitInputDataViewList(const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList)
    {
        inputDataViewList_ = inputDataViewList;
    }

    std::shared_ptr<FunctionIODataPair> GetInoutDataPair() const { return inoutDataPair_; }
    void UpdateIODataPair(std::shared_ptr<FunctionIODataPair>& inoutDataPair) { inoutDataPair_ = inoutDataPair; }

private:
    std::unordered_map<std::string, ScalarImmediateType> symbolDict_;
    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList_;
    std::shared_ptr<FunctionIODataPair> inoutDataPair_;
};

} // namespace npu::tile_fwk
