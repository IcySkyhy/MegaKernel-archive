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
 * \file infer_tensor_format.h
 * \brief Derive TileOpFormat for each tensor in the compute graph,
 *        and insert TransData operations when format mismatches occur.
 */

#ifndef PASS_INFER_TENSOR_FORMAT_H_
#define PASS_INFER_TENSOR_FORMAT_H_

#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class InferTensorFormat : public Pass {
public:
    InferTensorFormat() : Pass("InferTensorFormat") {}
    ~InferTensorFormat() override = default;

private:
    Status RunOnFunction(Function& function) override;

    // ---- 透传 op: 输入 format 直接复制到输出 ----
    static const std::unordered_set<Opcode> kPassThroughOps;

    // ---- Format 查询 ----
    static TileOpFormat GetRequiredInputFormat(Opcode opcode, const std::string& arch, size_t inputPos);
    static TileOpFormat GetOutputFormat(Opcode opcode, const std::string& arch, size_t outputPos);
    static bool IsSupportedTransData(TileOpFormat srcFormat, TileOpFormat targetFormat);
    static bool IsNdNzCompatibleFormat(TileOpFormat srcFormat, TileOpFormat targetFormat);
    static bool IsValidTileOpFormat(int64_t format);
    static int GetOpGroupValue(Operation* op);
    static int GetTransDataGroupValue(const std::shared_ptr<LogicalTensor>& srcTensor,
                                      const std::shared_ptr<LogicalTensor>& fakeDstTensor, Operation* relatedOp);
    static void ApplyTransDataVecTile(const std::shared_ptr<LogicalTensor>& srcTensor, TileOpFormat targetFormat,
                                      Operation* relatedOp);

    // ---- 图操作 ----
    static bool IsFunctionOutcast(const Function& function, const std::shared_ptr<LogicalTensor>& tensor);
    static int FindInputPosition(const Operation& op, const std::shared_ptr<LogicalTensor>& tensor);
    static Operation* FindRelatedConvOp(const std::shared_ptr<LogicalTensor>& srcTensor,
                                        const std::shared_ptr<LogicalTensor>& fakeDstTensor, TileOpFormat targetFormat);
    static std::shared_ptr<LogicalTensor> InsertTransDataOp(Function& function,
                                                            const std::shared_ptr<LogicalTensor>& srcTensor,
                                                            const std::shared_ptr<LogicalTensor>& fakeDstTensor,
                                                            Operation* relatedOp, TileOpFormat targetFormat);
    std::shared_ptr<LogicalTensor> InsertFakeTransOp(Function& function,
                                                     const std::shared_ptr<LogicalTensor>& srcTensor,
                                                     TileOpFormat targetFormat, Operation* relatedOp = nullptr);
    Status EnsureTensorFormat(Function& function, std::shared_ptr<LogicalTensor>& tensor, Operation* relatedOp,
                              TileOpFormat targetFormat);
    static Status GetFakeTransFormat(const Operation& op, const std::string& attrName, TileOpFormat& format);

    // ---- 输出 format 推导 ----
    static void DetermineOutputFormat(const Function& function, const Operation& op, const std::string& arch,
                                      std::unordered_map<int, bool>& visitedTensors,
                                      std::queue<std::shared_ptr<LogicalTensor>>& worklist);
    static void EnqueueTensorIfNeeded(const std::shared_ptr<LogicalTensor>& tensor,
                                      std::unordered_map<int, bool>& visitedTensors,
                                      std::queue<std::shared_ptr<LogicalTensor>>& worklist);
    static void EnqueueFunctionInputs(const Function& function, std::unordered_map<int, bool>& visitedTensors,
                                      std::queue<std::shared_ptr<LogicalTensor>>& worklist);
    static TileOpFormat ResolveRequiredInputFormat(const Function& function, const Operation& consumer,
                                                   const std::shared_ptr<LogicalTensor>& tensor,
                                                   const std::string& arch, int inputPos,
                                                   std::unordered_set<int>& assembledOutputs);
    Status EnsureConsumerInputFormat(Function& function, Operation& consumer,
                                     const std::shared_ptr<LogicalTensor>& tensor, TileOpFormat required);
    static void MarkConsumerInputProcessed(const Function& function, const Operation& consumer, const std::string& arch,
                                           std::unordered_map<int, int>& processedInputs,
                                           std::unordered_map<int, bool>& visitedTensors,
                                           std::queue<std::shared_ptr<LogicalTensor>>& worklist);
    Status ProcessConsumerFormat(Function& function, Operation* consumer, const std::shared_ptr<LogicalTensor>& tensor,
                                 const std::string& arch, std::unordered_map<int, int>& processedInputs,
                                 std::unordered_map<int, bool>& visitedTensors,
                                 std::unordered_set<int>& assembledOutputs,
                                 std::queue<std::shared_ptr<LogicalTensor>>& worklist);
    Status ProcessTensorConsumers(Function& function, const std::shared_ptr<LogicalTensor>& tensor,
                                  const std::string& arch, std::unordered_map<int, int>& processedInputs,
                                  std::unordered_map<int, bool>& visitedTensors,
                                  std::unordered_set<int>& assembledOutputs,
                                  std::queue<std::shared_ptr<LogicalTensor>>& worklist);

    // ---- 主算法 ----
    Status DeriveFormats(Function& function);

    // ---- Phase 2: 值编号+并查集消除冗余 FakeTrans ----
    Status EliminateRedundantFakeTrans(Function& function);
    static std::shared_ptr<LogicalTensor> FindRoot(const std::shared_ptr<LogicalTensor>& tensor,
                                                   std::unordered_map<int, std::shared_ptr<LogicalTensor>>& rootCache);
    static TileOpFormat GetEffectiveFormat(const std::shared_ptr<LogicalTensor>& tensor);
    static bool IsFakeTransOutput(const std::shared_ptr<LogicalTensor>& tensor);
    static bool IsValidFakeTransOp(const Operation& op);
    using Signature = std::pair<int, int>; // (root magic, format int)
    static void CollectSignatures(Function& function,
                                  std::unordered_map<int, std::shared_ptr<LogicalTensor>>& rootCache,
                                  std::map<Signature, std::vector<std::shared_ptr<LogicalTensor>>>& equivalenceClasses);
    static bool ReplaceRedundantTensors(
        std::map<Signature, std::vector<std::shared_ptr<LogicalTensor>>>& equivalenceClasses);
    static bool DeleteDeadFakeTrans(Function& function);

    // ---- Phase 3: 物化剩余 FakeTrans 为真实 TransData ----
    Status MaterializeFakeTrans(Function& function);

    IRBuilder irBuilder_;
};

} // namespace npu::tile_fwk
#endif // PASS_INFER_TENSOR_FORMAT_H_
