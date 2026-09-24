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
#include "ir/kind_traits.h"
#include "irbuilder.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
struct AccessRegion {
    RawTensor* raw;
    std::vector<int64_t> shape;
    std::vector<SymbolicScalar> offset;
};

struct Access {
    Operation* op;
    AccessRegion region;
    bool isWrite;

    std::string Dump()
    {
        std::stringstream ss;
        ss << (isWrite ? "W" : "R") << "@" << region.raw->GetRawMagic() << region.raw->GetSymbol() << " "
           << " shape=" << region.shape << " offset=" << region.offset << std::endl;
        return ss.str();
    }
};

using AccessMap = std::unordered_map<RawTensor*, std::vector<Access>>;

class TokenPass {
public:
    TokenPass() : builder_(IRBuilder()) {}

    ir::FunctionPtr operator()(const ir::FunctionPtr& func)
    {
        VisitStmt(func->body_);
        return func;
    }

private:
    std::vector<SymbolicScalar> GetOffset(const std::vector<SymbolicScalar>& dyn, const std::vector<int64_t>& concrete)
    {
        return dyn.empty() ? SymbolicScalar::FromConcrete(concrete) : dyn;
    }

    bool RegionOverlap(const AccessRegion& a, const AccessRegion& b)
    {
        if (a.raw != b.raw || a.raw == nullptr) {
            return false;
        }
        auto check = [](SymbolicScalar cond) {
            cond = cond.Simplify();
            return cond.ConcreteValid() && cond.Concrete() == true;
        };
        ASSERT(a.shape.size() == b.shape.size());
        for (size_t i = 0; i < a.shape.size(); i++) {
            if (check(a.offset[i] + a.shape[i] <= b.offset[i]) || check(b.offset[i] + b.shape[i] <= a.offset[i])) {
                return false;
            }
        }
        return true;
    }

    bool AddReadRegion(Operation& op, const LogicalTensorPtr& lt, AccessRegion& r)
    {
        /* view is partial read */
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            ASSERT(attr != nullptr);
            r.offset = GetOffset(attr->GetFromDynOffset(), attr->GetFromOffset());
            r.shape = op.GetOOperands()[0]->GetShape();
            r.raw = lt->GetRawTensor().get();
        } else {
            r.offset = std::vector<SymbolicScalar>(lt->GetShape().size(), 0);
            r.shape = lt->GetShape();
            r.raw = lt->GetRawTensor().get();
        }
        return true;
    }

    bool GetWriteRegion(Operation& op, AccessRegion& r)
    {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE && op.GetOpcode() != Opcode::OP_ASSEMBLE_SSA &&
            op.GetOpcode() != Opcode::OP_ATOMIC_RMW) {
            return false;
        }
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        ASSERT(attr != nullptr);
        r.offset = GetOffset(attr->GetToDynOffset(), attr->GetToOffset());
        r.shape = op.GetIOperands()[0]->GetShape();
        r.raw = op.GetOOperands()[0]->GetRawTensor().get();
        return true;
    }

    std::vector<Access> CollectAccessRecord(Operation* op)
    {
        AccessRegion r;
        std::vector<Access> acc;
        for (const auto& in : op->GetIOperands()) {
            if (AddReadRegion(*op, in, r)) {
                acc.push_back({op, std::move(r), /*isWrite=*/false});
            }
        }
        if (GetWriteRegion(*op, r)) {
            acc.push_back({op, std::move(r), /*isWrite=*/true});
        }
        return acc;
    }

    void GetRWDepends(AccessMap& history, const std::vector<Access>& access, std::set<ir::VarPtr>& tokenSet)
    {
        for (const auto& a : access) {
            if (!a.isWrite) {
                continue;
            }
            auto it = history.find(a.region.raw);
            if (it == history.end()) {
                continue;
            }
            for (const auto& prev : it->second) {
                if (!RegionOverlap(prev.region, a.region)) {
                    continue;
                }
                // prev is a read (WAR) or a write (WAW); both serialize
                // behind a producer->consumer token edge.
                if (prev.op->result_token_.empty()) {
                    prev.op->result_token_ = {builder_.CreateTokenVar(prev.op->GetSpan())};
                }
                tokenSet.insert(prev.op->result_token_.front());
            }
        }
    }

    void GetScalarDepends(Operation* op, std::set<ir::VarPtr>& tokenSet)
    {
        for (auto& scalar : op->GetDynamicAttributeList()) {
            auto token = builder_.GetDependToken(scalar.get().AsExpr());
            tokenSet.insert(token.begin(), token.end());
        }
    }

    void AddTokenDepend(AccessMap& history, Operation* op)
    {
        std::set<ir::VarPtr> tokenSet;
        auto access = CollectAccessRecord(op);
        GetRWDepends(history, access, tokenSet);
        for (const auto& a : access) {
            history[a.region.raw].push_back(a);
        }

        GetScalarDepends(op, tokenSet);
        std::vector<ir::VarPtr> tokenList(tokenSet.begin(), tokenSet.end());
        std::sort(tokenList.begin(), tokenList.end(), [](ir::VarPtr a, ir::VarPtr b) { return a->name_ < b->name_; });
        op->tokens_ = tokenList;
    }

    void VisitStmt(ir::SeqStmtsPtr stmts)
    {
        AccessMap history;
        for (const auto& stmt : stmts->stmts_) {
            if (auto ifStmt = ir::As<ir::IfStmt>(stmt)) {
                VisitStmt(ifStmt->thenBody_);
                if (ifStmt->elseBody_.has_value()) {
                    VisitStmt(ifStmt->elseBody_.value());
                }
            } else if (auto forStmt = ir::As<ir::ForStmt>(stmt)) {
                VisitStmt(forStmt->body_);
            } else if (auto whileStmt = ir::As<ir::WhileStmt>(stmt)) {
                VisitStmt(whileStmt->body_);
            } else if (auto section = ir::As<ir::SectionStmt>(stmt)) {
                VisitStmt(section->body_);
            } else if (auto tensorOp = ir::AsMut<ir::TensorOpStmt>(stmt)) {
                AddTokenDepend(history, std::static_pointer_cast<Operation>(tensorOp).get());
            }
        }
    }

private:
    IRBuilder builder_;
};
} // namespace npu::tile_fwk
