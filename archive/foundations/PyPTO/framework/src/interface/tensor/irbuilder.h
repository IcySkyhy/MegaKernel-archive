/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "ir/builder.h"
#include "ir/type.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"

using namespace pypto;

namespace npu::tile_fwk {

class IRContext {
public:
    IRContext(const IRContext&) = delete;
    IRContext& operator=(const IRContext&) = delete;
    IRContext(IRContext&&) = delete;
    IRContext& operator=(IRContext&&) = delete;

    ir::VarPtr MakeVar(std::string name, ir::TypePtr type, ir::Span span)
    {
        auto var_name = GetVarName(name);
        type_map_[var_name] = type;
        return std::make_shared<ir::Var>(var_name, type, span);
    }

    ir::VarPtr MakeTempVar(ir::TypePtr type, ir::Span span)
    {
        auto var_name = GetVarName();
        type_map_[var_name] = type;
        return std::make_shared<ir::Var>(var_name, type, span);
    }

    ir::IterArgPtr MakeIterArg(std::string name, ir::TypePtr type, ir::ExprPtr initVal, ir::Span span)
    {
        auto var_name = GetVarName(name);
        type_map_[var_name] = type;
        auto var = std::make_shared<ir::Var>(var_name, type, span);
        return std::make_shared<ir::IterArg>(var, initVal);
    }

    ir::VarPtr MakeToken() { return MakeTempVar(ir::GetTokenType(), ir::Span::Unknown()); }

    ir::TypePtr GetType(ir::VarPtr var) { return type_map_[var->name_]; }

    void SetType(ir::VarPtr var, ir::TypePtr type) { type_map_[var->name_] = type; }

    std::string GetOriginName(ir::VarPtr var)
    {
        auto it = all_vars_.find(var->name_);
        return it != all_vars_.end() ? it->second : var->name_;
    }

    std::string GetVarName(const std::string& name = "")
    {
        std::string origin = name;
        if (origin.empty()) {
            origin = "$" + std::to_string(temp_counter_++);
        } else {
            // Propagate: if `name` is an existing unique var name, carry its origin forward
            // so clones/substitutes share the source var's base name.
            auto it = all_vars_.find(origin);
            if (it != all_vars_.end()) {
                origin = it->second;
            }
        }

        auto var_name = origin;
        if (all_vars_.find(var_name) != all_vars_.end()) {
            auto& counter = var_counter_[origin];
            do {
                var_name = origin + "_" + std::to_string(counter++);
            } while (all_vars_.count(var_name));
        }

        all_vars_.emplace(var_name, origin);
        return var_name;
    }

    void Reset()
    {
        temp_counter_ = 0;
        type_map_.clear();
        var_counter_.clear();
        all_vars_.clear();
        token_map_.clear();
        semantic_to_normal_.clear();
        ClearTensorDataDescList();
    }

    void SetAssembleNewLogicalTensor(bool enabled) { assembleNewLogicalTensor_ = enabled; }
    bool AssembleNewLogicalTensor() const { return assembleNewLogicalTensor_; }

    void ClearTensorDataDescList() { getTensorDataDescList.clear(); }

    void AddDependToken(const ir::ExprPtr& val, const ir::VarPtr& token) { token_map_[val].push_back(token); }
    std::vector<ir::VarPtr>& GetDependToken(const ir::ExprPtr& val);

    std::vector<ir::VarPtr> GetDependToken(Operation& op);

    /**
     * Create a READ/WRITE semantic token and its paired NORMAL token.
     * The NORMAL token is registered internally and is not returned.
     */
    ir::VarPtr MakeSemanticToken(std::string name, ir::TokenKind kind, ir::Span span);

    /** Look up the NORMAL token paired with a READ/WRITE semantic token. */
    ir::VarPtr GetNormalToken(const ir::VarPtr& semantic) const;

    int AddTensorDataDesc(const std::shared_ptr<Tensor>& assembleTensor)
    {
        int index = static_cast<int>(getTensorDataDescList.size());
        getTensorDataDescList.push_back(assembleTensor);
        return index;
    }
    std::shared_ptr<Tensor> GetTensorDataDesc(int index) const
    {
        FE_ASSERT(FeError::INVALID_VAL, index >= 0 && static_cast<size_t>(index) < getTensorDataDescList.size())
            << "Invalid GetTensorDataDesc index: " << index;
        auto assembleTensor = getTensorDataDescList[index];
        FE_ASSERT(FeError::INVALID_PTR, assembleTensor != nullptr) << "GetTensorDataDesc is nullptr, index: " << index;
        return assembleTensor;
    }

    static IRContext& Get();

private:
    IRContext() = default;
    int64_t temp_counter_{0};                     // counter for temporary variables
    std::map<std::string, ir::TypePtr> type_map_; // type for each variable
    std::map<std::string, int64_t> var_counter_;  // counter for named variable
    std::map<std::string, std::string> all_vars_; // unique var name -> var name
    std::unordered_map<ir::ExprPtr, std::vector<ir::VarPtr>> token_map_;
    std::unordered_map<ir::VarPtr, ir::VarPtr> semantic_to_normal_;
    std::vector<std::shared_ptr<Tensor>> getTensorDataDescList;
    bool assembleNewLogicalTensor_{false};
};

class IRBuilder : public ir::IRBuilder {
public:
    IRBuilder();

    // Disable copying and moving since we have unique_ptr members
    IRBuilder(const IRBuilder&) = delete;
    IRBuilder& operator=(const IRBuilder&) = delete;
    IRBuilder(IRBuilder&&) = delete;
    IRBuilder& operator=(IRBuilder&&) = delete;

    /* create raw tensor with static shape */
    std::shared_ptr<RawTensor> CreateRawTensor(DataType t, Shape shape, TileOpFormat format = TileOpFormat::TILEOP_ND,
                                               std::string name = "");

    /* create raw tensor with dynamic shape */
    std::shared_ptr<RawTensor> CreateRawTensor(DataType t, std::vector<SymbolicScalar> shape,
                                               TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create logical tensor with static shape */
    LogicalTensorPtr CreateTensorVar(DataType t, Shape shape, TileOpFormat format = TileOpFormat::TILEOP_ND,
                                     std::string name = "");

    /* create logical tensor with dynamic shape */
    LogicalTensorPtr CreateTensorVar(DataType t, Shape shape, std::vector<SymbolicScalar> validShape,
                                     TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create logical tensor from raw tensor */
    LogicalTensorPtr CreateTensorVar(std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape,
                                     std::vector<SymbolicScalar> validShape = {});

    /* create logical tensor with static shape, associate with function */
    LogicalTensorPtr CreateTensorVar(Function& f, DataType t, Shape shape,
                                     TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create logical tensor with dynamic shape, associate with function */
    LogicalTensorPtr CreateTensorVar(Function& f, DataType t, Shape shape, std::vector<SymbolicScalar> validShape,
                                     TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create logical tensor from raw tensor, associate with function */
    LogicalTensorPtr CreateTensorVar(Function& f, std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape,
                                     std::vector<SymbolicScalar> validShape = {});

    /* create tensor operation statement */
    Operation& CreateTensorOpStmt(Function& f, const Opcode opCode, const LogicalTensors& iOperands,
                                  const LogicalTensors& oOperands, ir::Span span = ir::Span::Unknown());

    ir::TensorOpStmtPtr CreateTensorOpStmt(std::vector<ir::VarPtr> result, ir::VarPtr result_token, std::string opcode,
                                           std::vector<ir::ExprPtr> args, std::vector<ir::VarPtr> tokens,
                                           std::vector<std::pair<std::string, std::any>> attrs, ir::Span span);

    /* create symbolic scalar */
    SymbolicScalar CreateConstInt(int64_t value);

    SymbolicScalar CreateScalarVar(std::string sym);

    ir::VarPtr CreateVarLike(std::string name, ir::ExprPtr value);

    /* ==== scf statement ==== */

    ir::AssignStmtPtr CreateAssignStmt(ir::VarPtr var, ir::ExprPtr value, ir::Span span);

    ir::SeqStmtsPtr CreateSeqStmts(std::vector<ir::StmtPtr> stmts, ir::Span span);

    ir::IfStmtPtr CreateIfStmt(ir::ExprPtr cond, ir::StmtPtr thenBody, std::optional<ir::StmtPtr> elseBody,
                               std::vector<ir::VarPtr> returnVars, ir::Span span);

    ir::YieldStmtPtr CreateYieldStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ReturnStmtPtr CreateReturnStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ForStmtPtr CreateForStmt(ir::VarPtr loopVar, ir::ExprPtr start, ir::ExprPtr stop, ir::ExprPtr step,
                                 std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body,
                                 std::vector<ir::VarPtr> returnVars, ir::Span span,
                                 std::vector<std::pair<std::string, std::any>> attrs = {});

    ir::IterArgPtr CreateIterArg(std::string name, ir::TypePtr type, ir::ExprPtr initValue, ir::Span span);

    ir::IterArgPtr CreateIterArg(ir::VarPtr var, ir::ExprPtr initValue);

    ir::WhileStmtPtr CreateWhileStmt(ir::ExprPtr cond, std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body,
                                     std::vector<ir::VarPtr> returnVars, ir::Span span);

    ir::BreakStmtPtr CreateBreakStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ContinueStmtPtr CreateContinueStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::FunctionPtr CreateFunction(std::string name, std::vector<ir::VarPtr> params,
                                   std::vector<ir::TypePtr> returnTypes, ir::StmtPtr body, ir::Span span);

    ir::ProgramPtr CreateProgram(std::vector<ir::FunctionPtr> functions, std::string name, ir::Span span);

    ir::VarPtr CreateTokenVar(ir::Span span);

    void AddDependToken(SymbolicScalar scalar, ir::VarPtr token) { irContext_.AddDependToken(scalar.AsExpr(), token); }
    std::vector<ir::VarPtr>& GetDependToken(SymbolicScalar scalar)
    {
        return irContext_.GetDependToken(scalar.AsExpr());
    }

    void AddDependToken(ir::ExprPtr expr, ir::VarPtr token) { irContext_.AddDependToken(expr, token); }
    std::vector<ir::VarPtr>& GetDependToken(ir::ExprPtr expr) { return irContext_.GetDependToken(expr); }
    std::vector<ir::VarPtr> GetDependToken(Operation& op) { return irContext_.GetDependToken(op); }

    int AddTensorDataDesc(const std::shared_ptr<Tensor>& assembleTensor)
    {
        return irContext_.AddTensorDataDesc(assembleTensor);
    }
    void ClearTensorDataDescList() { irContext_.ClearTensorDataDescList(); }
    std::shared_ptr<Tensor> GetTensorDataDesc(int index) const { return irContext_.GetTensorDataDesc(index); }

    void Checkpoint();

    void Restore();

    void EmitTensorStmts();

    ir::ExprPtr None();

private:
    IRContext& irContext_;
};

} // namespace npu::tile_fwk
