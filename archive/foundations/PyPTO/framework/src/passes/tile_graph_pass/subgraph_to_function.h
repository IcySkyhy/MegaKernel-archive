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
 * \file subgraph_to_function.h
 * \brief
 */

#ifndef PASS_SUGGRAPH_TO_FUNCTION_H_
#define PASS_SUGGRAPH_TO_FUNCTION_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "passes/pass_utils/subfunc_utils.h"
#include "passes/statistics/execute_graph_statistic.h"
#include "passes/tile_graph_pass/static_subgraph_processor.h"

namespace npu::tile_fwk {
struct RecordInfo {
    size_t i;
    size_t j;
    size_t k;
    LogicalTensorPtr operand;
    Shape shape;
    Offset offset;
};

class SubgraphToFunction : public Pass {
public:
    SubgraphToFunction() : Pass("SubgraphToFunction") {}
    ~SubgraphToFunction() override = default;
    friend class MixSubgraphSplit;
    friend class MixCallOperationBuilder; // 提供接口规避编译错误，后续整改
    friend class MixDependencyAnalyzer;

    void SetupStaticProcessor() { staticProcessor_.SetNList(nLIST); }

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;

    void Init();
    Status GetTensorDataDependencyInsert(Function& function);
    Status GetTensorDataDependencyClear(Function& function);

    void DoHealthCheckAfter(Function& function, const std::string& folderPath) override;

    Status ProcessSubgraph(Function& function, size_t i, size_t& programIdx, std::vector<Function*>& outputFuncList);
    Status ProcessCacheResult(const std::tuple<Function*, Operation*, bool>& result, size_t i, size_t& programIdx,
                              std::vector<Function*>& outputFuncList, Operation& callOp);
    void SetSemanticLabel(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation& callOp);
    void CopyHashOrderInfoToCallOp(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation& callOp,
                                   const std::string& hashOrderKey, const std::string& countKey) const;
    void SetHashOrderInfoOnCallOp(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation& callOp) const;
    void InitializeRootFunction(Function& function, Function& rootFunc);
    Status IslandToFunction(Function& function);
    void ConstructnList(Function& function);

    // 与 subFuncInvokeInfos相关的函数
    void RecordIncastOutcast(Function& function);
    void RecordEsgIncast(Function& function, size_t i, size_t j, size_t k);
    void RecordIncastInfo(Function& function, RecordInfo recordInfo, SubfuncInvokeInfoTy& iter);
    void RecordConnectionWithProducers(RecordInfo recordInfo, SubfuncInvokeInfoTy& iter);
    void RecordEsgOutcast(Function& function, size_t i, size_t j, size_t k);
    void RecordEsgIncastOutcast(Function& function);
    void InsertParameter(size_t i, Function& leafFunc);
    void ConstructParamMap(Function& function);
    void RecordOutcastInfo(Function& function, RecordInfo recordInfo, SubfuncInvokeInfoTy& iter);

    // 符号化相关函数
    void ProcessInputOperands(Function& rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc,
                              int& iParamLoc) const;
    void ProcessOutputOperands(Function& rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc,
                               int& oParamLoc) const;
    void ProcessCopyInOperand(Operation& tileOp, std::vector<int64_t>& offset, std::vector<int64_t>& shape) const;
    void ProcessCopyOutOperand(Operation& tileOp, std::vector<int64_t>& offset, std::vector<int64_t>& shape) const;
    void ProcessSymbolOfReshape(Function& function, Operation& op) const;
    void SymbolizeEachFunction(Function& rootFunc, std::vector<Function*>& mergedFuncList1, size_t i) const;
    void SymbolizeFunction(Function& rootFunc, std::vector<Function*>& mergedFuncList1) const;
    std::string FindSymbolName(std::shared_ptr<LogicalTensor> op, int magic) const;

    void GenerateAndExportCombinedReport(Function& func, const std::multimap<int, int>& psgToESgMapParam,
                                         const std::vector<std::vector<OperationPtr>>& subgraphGroups,
                                         const std::string& filename = "ExecuteGraph_Health_Report.json");

    // 在子图生成前将View转成CopyIn用于coa记录，子图生成后转回View
    Status TransViewToCopyInBeforeGenSubgraph(Function& function);
    Status RecoverCopyInToViewAfterGenSubgraph(Function& function);

    // 静态流程处理器
    StaticSubgraphProcessor staticProcessor_;

    std::vector<std::vector<OperationPtr>> nLIST;
    std::vector<Function*> mergedFuncList;
    std::multimap<int, int> psgToESgMap;
    std::vector<SubfuncInvokeInfoTy> subFuncInvokeInfos;
    std::unordered_set<int> mixSubgraphIds_;
    std::unordered_map<const Operation*, std::shared_ptr<OpAttribute>> viewToCopyInMapping_;
    static constexpr int kShapePlaceholderForParameterized = -2;
};
} // namespace npu::tile_fwk
#endif // PASS_SUGGRAPH_TO_FUNCTION_H_
