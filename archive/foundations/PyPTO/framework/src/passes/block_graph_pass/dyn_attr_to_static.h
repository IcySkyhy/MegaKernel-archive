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
 * \file dyn_attr_to_static.h
 * \brief
 */

#ifndef PASS_DYNATTR_TO_STATIC_H_
#define PASS_DYNATTR_TO_STATIC_H_

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/symbolic_scalar.h"

#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "DynAttrToStatic"

namespace npu {
namespace tile_fwk {

enum class CoaType { PARAM_OFFSET, PARAM_VALID_SHAPE, PARAM, PARAM_RAW_SHAPE, INVALID };

static const std::string COA_PREFIX = "RUNTIME_COA_GET_PARAM";
static const std::string MAYBE_CONST_POSTFIX = "MAYBE_CONST";

static const SymbolicScalar MAYBE_CONST_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM_MAYBE_CONST");
static const SymbolicScalar GET_PARAM_ADDR_MAYBE_CONST = AddRuntimeCoaPrefix("GET_PARAM_ADDR_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetRawShape = AddRuntimeCoaPrefix("GET_PARAM_RAW_SHAPE_MAYBE_CONST");

Status SToIWrapper(const std::string str, int& result);

constexpr int OFFSET_INDEX_ORDER = 0;
constexpr int SHAPE_INDEX_ORDER = 1;
constexpr int RAWSHAPE_INDEX_ORDER = 2;
constexpr int VALID_SHAPE_INDEX_ORDER = 3;
constexpr int INPUT_PARAM_POS_ONE = 1;
constexpr int INPUT_PARAM_POS_TWO = 2;
constexpr int INPUT_PARAM_POS_THREE = 3;

/**
 * @brief 校验vector形参调用时，是否存在索引组满足「每次调用内组内索引值相同」
 */
class VectorParamConsistencyChecker {
public:
    /**
     * @brief 注册一次函数调用的vector实参
     * @param args 本次调用的vector实参
     * @return 注册是否成功（长度非法时失败）
     */
    bool RegisterCall(const std::vector<SymbolicScalar>& args)
    {
        if (args.empty())
            return false; // 空vector无意义

        // 首次调用：记录vector长度，后续调用需保持长度一致
        if (callCount_ == 0) {
            vecLen_ = args.size();
        } else if (args.size() != vecLen_) {
            isValid_ = false; // 长度不一致，直接标记无效
            return false;
        }

        // 步骤1：生成本次调用的「值-索引列表」（仅用operator==，线性遍历）
        std::unordered_map<std::string, std::set<size_t>> currValIdxs;
        for (size_t idx = 0; idx < args.size(); ++idx) {
            currValIdxs[args[idx].Dump()].insert(idx);
        }

        // 步骤2：更新候选索引组（首次调用初始化，后续调用筛选）
        if (callCount_ == 0) {
            // 首次调用：所有非空索引列表都作为候选组（去重+排序）
            for (auto& pair : currValIdxs) {
                const auto& values = pair.second;
                if (!values.empty()) {
                    // 索引组排序（保证相同索引组合的一致性，避免重复）
                    std::vector<size_t> vec{values.begin(), values.end()};
                    candidateGroups_.push_back(std::move(vec));
                }
            }
            // 对候选组去重（避免首次调用就有重复的索引组）
            DeduplicateGroups(candidateGroups_);
        } else {
            // 非首次调用：筛选候选组（仅保留在本次调用中值相同的索引组）
            std::vector<std::vector<size_t>> newCandidates;
            for (const auto& candidate : candidateGroups_) {
                // 检查候选组是否在本次调用的某值索引列表中（子集判断）
                if (IsCandidateValidInCurrCall(candidate, currValIdxs)) {
                    newCandidates.push_back(candidate);
                }
            }
            // 去重后替换候选组
            DeduplicateGroups(newCandidates);
            candidateGroups_.swap(newCandidates);
        }

        callCount_++;
        return true;
    }

    /**
     * @brief 获取首个满足条件的索引组（便捷接口，兼容旧逻辑）
     * @return 第一个满足条件的索引组（空则无）
     */
    std::vector<size_t> GetConsistentIndexGroup() const
    {
        if (!isValid_ || candidateGroups_.empty()) {
            return {};
        }
        return candidateGroups_.front();
    }

    /**
     * @brief 获取所有满足条件的索引组（核心接口，返回全部有效组）
     * @return 所有符合条件的索引组（空则无）
     */
    std::vector<std::vector<size_t>> GetAllConsistentIndexGroups() const
    {
        if (!isValid_ || candidateGroups_.empty()) {
            return {};
        }
        // 返回完整的候选组列表（已去重、有序）
        return candidateGroups_;
    }

    std::string PrintIndexGroups(const std::vector<std::vector<size_t>>& groups) const
    {
        std::stringstream ss;
        ss << std::endl << "ALL Consistent Index Group:  {" << std::endl;
        if (groups.empty()) {
            ss << "}";
        }
        for (size_t i = 0; i < groups.size(); ++i) {
            ss << "Consistent Index Group: " << (i + 1) << "{";
            for (size_t idx : groups[i]) {
                ss << idx << ", ";
            }
            ss << "}" << std::endl;
        }
        ss << "}";
        return ss.str();
    }

    /**
     * @brief 重置校验器（清空所有调用记录）
     */
    void Reset()
    {
        callCount_ = 0;
        vecLen_ = 0;
        isValid_ = true;
        candidateGroups_.clear();
    }

private:
    /**
     * @brief 检查候选索引组是否在本次调用中有效（组内索引值相同）
     * @param candidate 候选索引组
     * @param currValIdxs 本次调用的「值-索引列表」
     * @return 是否有效
     */
    bool IsCandidateValidInCurrCall(const std::vector<size_t>& candidate,
                                    const std::unordered_map<std::string, std::set<size_t>>& currValIdxs) const
    {
        // 步骤1：查找第一个索引在当前args所在的索引组
        if (candidate.empty())
            return false;
        size_t firstIdx = candidate[0];
        for (const auto& pair : currValIdxs) {
            const auto& values = pair.second;
            if (values.count(firstIdx)) {
                // 步骤2：校验候选组中的索引在新的索引组中是否全部存在
                for (size_t idx : candidate) {
                    if (!values.count(idx)) {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    }

    /**
     * @brief 对索引组列表去重（避免重复的索引组合）
     * @param groups 待去重的索引组列表
     */
    void DeduplicateGroups(std::vector<std::vector<size_t>>& groups)
    {
        if (groups.empty())
            return;
        // 步骤1：先对每个索引组内部排序（保证 {3,1} 和 {1,3} 视为同一组）
        for (auto& group : groups) {
            std::sort(group.begin(), group.end());
        }
        // 步骤2：对索引组列表排序，便于去重
        std::sort(groups.begin(), groups.end(), [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
            if (a.size() != b.size())
                return a.size() < b.size();
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i] != b[i])
                    return a[i] < b[i];
            }
            return false;
        });
        // 步骤3：去重
        auto last = std::unique(groups.begin(), groups.end());
        groups.erase(last, groups.end());
    }

    size_t callCount_ = 0; // 调用次数
    size_t vecLen_ = 0;    // vector固定长度
    bool isValid_ = true;  // 是否有效（长度一致）
    // 候选索引组：用普通vector存储，无需排序/哈希（已去重）
    std::vector<std::vector<size_t>> candidateGroups_;
};

class DynAttrToStatic : public Pass {
public:
    DynAttrToStatic() : Pass("DynAttrToStatic") {}
    ~DynAttrToStatic() override = default;

private:
    friend class LoopaxesProc;

    std::unordered_map<Function*, std::vector<Operation*>> leaf2Caller;

    Status RunOnFunction(Function& function) override;
    std::vector<std::reference_wrapper<SymbolicScalar>> GetOpDynamicAttributeList(Operation& op);
    Status GetCallee(const Operation& callop, Function*& callFunc);
    void RefSpecifiedValue(std::vector<SymbolicScalar>& oriList,
                           std::vector<std::reference_wrapper<SymbolicScalar>>& newList) const;
    void FilterSpecifiedValue(std::vector<OpImmediate>& oriList,
                              std::vector<std::reference_wrapper<SymbolicScalar>>& newList) const;
    Status BuildLeafToCaller(Function* func);
    Status BuildNewCoa(std::reference_wrapper<SymbolicScalar>& dynScalar,
                       std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim);
    Status TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList);
    Status GetTileFunction(Function* function, std::unordered_set<Function*>& tileFunctionSet);
    Status DumpFunctionJson(Function& function, const std::string& logFolder, bool beforeFunction = true) override;
    Status PrintFunction(Function& function, const std::string& logFolder, bool beforeFunction = true) override;
    void BuildParamAddr(Operation& op);
    std::set<LogicalTensorPtr> visitedTensors_;
    std::set<int> rootInOutCast_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_DYNATTR_TO_STATIC_H_
