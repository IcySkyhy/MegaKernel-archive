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
 * \file function.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <unordered_set>
#include <optional>
#include "tilefwk/tensor.h"
#include "tilefwk/symbolic_scalar.h"
#include "ir/span.h"

using namespace pypto;

namespace npu::tile_fwk {
// Forward declaration for types used in this file
class Function;
} // namespace npu::tile_fwk

// Helper macros to count arguments
#define RECORD_FUNC_VAR_NAME_COUNTER_HELPER(var, cnt) var##cnt
#define RECORD_FUNC_VAR_NAME_COUNTER(var, cnt) RECORD_FUNC_VAR_NAME_COUNTER_HELPER(var, cnt)
#define RECORD_FUNC_VAR_NAME(var) RECORD_FUNC_VAR_NAME_COUNTER(var, __COUNTER__)

/**
 * @brief Start a tile_fwk function. All computational logic must be enclosed by this macro
 * @param name: Name of the function;
 * @param explicitOpArgs: The inputs and outputs of the function. Be effective in static shape scen.
 * @param startArgsInputTensorList: The inputs of the function. Be effective in dynamic shape scen.
 * @param startArgsOutputTensorList: The outputs of the function. Be effective in dynamic shape scen.
 * @param inplaceArgs: A inpute and a output have same addr. Be effective in dynamic shape scen. optional, default is
 * empty;
 */
#define FUNCTION(name, ...) \
    for ([[maybe_unused]] auto& RECORD_FUNC_VAR_NAME(recordFunc) : npu::tile_fwk::RecordFunc(name, ##__VA_ARGS__))

/**
 * @brief Start a tile_fwk dynamic loop.
 * @param name: Name of the loops;
 * @param type: Type of function;
 * @param index: The index of loops.
 * @param range: The range of loops, including start\end and step length.
 * @param unrollList: The list for unroll; Optional, default is empty.
 * @param submitBeforeLoop: Submit task before next loop. Optional, default value is false.
 */
#define LOOP(name, funcType, index, ...) \
    for (auto& index : npu::tile_fwk::RecordLoopFunc(name, funcType, #index, ##__VA_ARGS__))

/**
 * @brief Describe an 'if' branch in dynamic scen.
 * @param cond: The conditions;
 */
#define IF(cond) if (npu::tile_fwk::RecordIfBranch(cond, __FILE__, __LINE__))

/**
 * @brief Describe an 'else' branch in dynamic scen.
 *
 */
#define ELSE else

/**
 * @brief Expend some loops to compile.
 * @param unrollTimes: The times of loops to compile;
 */
#define UNROLL(X) if (npu::tile_fwk::RecordLoopFunc::MatchUnrollTimes(X))

/**
 * @brief Expend 1 loop to compile.
 *
 */
#define UNROLL_DEFAULT if (npu::tile_fwk::RecordLoopFunc::MatchUnrollTimes(1))

namespace npu::tile_fwk {
struct DynloopFunctionAttribute;

enum class FunctionType {
    EAGER,
    STATIC,
    DYNAMIC,
    DYNAMIC_LOOP,
    DYNAMIC_LOOP_PATH,
    INVALID,
    MAX,
};

const std::string FUNCTION_PREFIX = "TENSOR_";
const std::string PROFILING_PREFIX = "PYPTO_";

class LoopRange {
public:
    LoopRange(const SymbolicScalar& rangeBegin, const SymbolicScalar& rangeEnd, const SymbolicScalar& rangeStep)
        : begin_(rangeBegin), end_(rangeEnd), step_(rangeStep)
    {}

    explicit LoopRange(const SymbolicScalar& rangeBegin, const SymbolicScalar& rangeEnd)
        : LoopRange(rangeBegin, rangeEnd, 1)
    {}

    explicit LoopRange(const SymbolicScalar& rangeEnd) : LoopRange(0, rangeEnd, 1) {}

    SymbolicScalar& Begin() { return begin_; }
    const SymbolicScalar& Begin() const { return begin_; }

    SymbolicScalar& End() { return end_; }
    const SymbolicScalar& End() const { return end_; }

    SymbolicScalar& Step() { return step_; }
    const SymbolicScalar& Step() const { return step_; }

    std::string Dump()
    {
        std::stringstream ss;
        ss << "LoopRange(" << begin_.Dump() << ", " << end_.Dump() << ", " << step_.Dump() << ")";
        return ss.str();
    }

private:
    SymbolicScalar begin_;
    SymbolicScalar end_;
    SymbolicScalar step_;
};

class RecordLoopFunc {
public:
    struct IteratorEnd {
        RecordLoopFunc& func;
        SymbolicScalar scalar;
    };

    class Iterator {
    public:
        Iterator(RecordLoopFunc& rlf, const SymbolicScalar& scalar)
            : rlf_(rlf), scalar_(scalar), originalScalar_(scalar)
        {}

        Iterator operator++();
        bool operator!=(const IteratorEnd& rhs);
        bool operator==(const IteratorEnd& rhs) { return !this->operator!=(rhs); }
        const SymbolicScalar& operator*() const { return scalar_; }
        SymbolicScalar& operator*() { return scalar_; }

    private:
        RecordLoopFunc& rlf_;
        SymbolicScalar scalar_;
        SymbolicScalar originalScalar_;
        int cur_{0};
    };

    explicit RecordLoopFunc(const std::string& name, FunctionType funcType, const std::string& iterName,
                            const LoopRange& range, const std::set<int>& unrollList = {}, bool submitBeforeLoop = false,
                            bool parallel = false);
    ~RecordLoopFunc();

    void BeginLoopFunction();
    void EndLoopFunction();

    std::shared_ptr<DynloopFunctionAttribute> GetLoopAttr();

    Iterator begin();
    IteratorEnd end();
    void IterationBegin();
    void IterationNext();
    bool IterationEnd();
    bool Condition(const SymbolicScalar& cond, const std::string& file, int line);

    const SymbolicScalar& LoopBegin() const;
    const SymbolicScalar& LoopStep() const;
    const SymbolicScalar& LoopEnd() const;

    bool VisitedUnroll(int unrollTimes) const { return visited_.count(unrollTimes) > 0; }
    void VisitUnroll(int unrollTimes);
    bool IsCustomUnrollTimes(int unrollTimes) const
    {
        return customUnrollTimes_.count(unrollTimes) > 0;
    } // check is user defined
    bool StillHaveUnrollTimes() const { return !unrollTimes_.empty(); }
    size_t UnrollTimesSize() const { return unrollTimes_.size(); }
    int CurUnrollTimes() const;
    void NextUnrollTimes();
    bool Getparallel() const { return parallel_; }

    bool CustomUnrollTimesMatched() const { return customUnrollTimes_.count(CurUnrollTimes()) > 0; }
    static bool MatchUnrollTimes(int unrollTimes);

private:
    std::string GetLoopSuffix(int count) { return "_PATH" + std::to_string(count); }
    std::string name_;
    std::string iterName_;
    std::string curPathFuncName_;
    std::shared_ptr<LoopRange> loopRange_;
    bool submitBeforeLoop_;
    bool parallel_;
    FunctionType funcType_{FunctionType::STATIC};
    Function* currentLoopFunc_{nullptr};
    bool dryRun_{false};
    bool hasManualUnroll_{false};
    int endCount_{0};

    std::vector<std::shared_ptr<LoopRange>> rangeOfEaceUnroll_;
    std::set<int, std::greater<>> unrollTimes_;
    std::unordered_set<int> visited_;
    std::unordered_set<int> customUnrollTimes_;
    ir::Span span_;

    void GenDefaultUnrollTimes(const std::set<int>& unrollList);
};

class RecordFunc {
public:
    struct IteratorEnd {
        RecordFunc& func_;
        std::optional<RecordLoopFunc::IteratorEnd> wrappedEnd;

        IteratorEnd(RecordFunc& func, RecordLoopFunc::IteratorEnd end) : func_(func), wrappedEnd(end) {}

        IteratorEnd(RecordFunc& func) : func_(func), wrappedEnd(std::nullopt) {}
    };

    class Iterator {
    public:
        Iterator(RecordFunc& func, RecordLoopFunc::Iterator iter) : func_(func), wrappedIter_(iter), cur_(0) {}

        Iterator(RecordFunc& func) : func_(func), wrappedIter_(std::nullopt), cur_(0) {}

        Iterator operator++();
        bool operator!=(const IteratorEnd& rhs);
        bool operator==(const IteratorEnd& rhs) { return !this->operator!=(rhs); }
        const SymbolicScalar& operator*() const
        {
            if (wrappedIter_.has_value()) {
                return wrappedIter_->operator*();
            }
            return cur_;
        }
        SymbolicScalar& operator*()
        {
            if (wrappedIter_.has_value()) {
                return wrappedIter_->operator*();
            }
            return cur_;
        }

    private:
        RecordFunc& func_;
        std::optional<RecordLoopFunc::Iterator> wrappedIter_;
        SymbolicScalar cur_;
    };

    explicit RecordFunc(const std::string& name);
    RecordFunc(const std::string& name, const std::vector<std::reference_wrapper<const Tensor>>& explicitOpArgs);
    RecordFunc(const std::string& name,
               const std::vector<std::reference_wrapper<const Tensor>>& startArgsInputTensorList,
               const std::vector<std::reference_wrapper<const Tensor>>& startArgsOutputTensorList,
               const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>&
                   inplaceArgs = {});

    void EndFunction();
    void AbortRecording();

    ~RecordFunc()
    {
        if (!isEnd_)
            EndFunction();
    }

    Iterator begin();
    IteratorEnd end();

private:
    void RecordDynFuncInner(
        const std::vector<std::reference_wrapper<const Tensor>>& startArgsInputTensorList,
        const std::vector<std::reference_wrapper<const Tensor>>& startArgsOutputTensorList,
        const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>&
            inplaceArgs);
    Function* dynFunc_{nullptr};
    std::string funcName;
    std::unique_ptr<RecordLoopFunc> recordLoopFunc_;
    bool isEnd_{false};
};

class RecordIfBranch {
public:
    explicit RecordIfBranch(const SymbolicScalar& cond, const std::string& file, int line)
        : cond_(cond), file_(file), line_(line)
    {}

    operator bool() const;

private:
    SymbolicScalar cond_{0};
    std::string file_;
    int line_{-1};
};
} // namespace npu::tile_fwk
