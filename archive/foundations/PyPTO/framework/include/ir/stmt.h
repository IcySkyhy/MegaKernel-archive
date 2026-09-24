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
#include <cstdint>
#include <algorithm>
#include <memory>
#include <any>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

// Forward declarations for friend classes
class IRVisitor;
class IRMutator;

enum class SectionKind : uint8_t { Vector = 0, Cube = 1, VF = 2 };

inline std::string SectionKindToString(SectionKind kind)
{
    switch (kind) {
        case SectionKind::Vector:
            return "Vector";
        case SectionKind::Cube:
            return "Cube";
        case SectionKind::VF:
            return "VF";
        default:
            break;
    }
    CHECK(false) << "Unknown SectionKind";
    return "Unknown";
}

inline SectionKind StringToSectionKind(const std::string& str)
{
    if (str == "Vector")
        return SectionKind::Vector;
    if (str == "Cube")
        return SectionKind::Cube;
    if (str == "VF")
        return SectionKind::VF;
    CHECK(false) << "Unknown SectionKind: " << str;
    return SectionKind::Vector;
}

/**
 * \brief Base class for all statements in the IR
 *
 * Statements represent operations that perform side effects or control flow.
 * All statements are immutable.
 */
class Stmt : public IRNode {
public:
    /**
     * \brief Create a statement
     *
     * \param span Source location
     */
    explicit Stmt(Span s) : IRNode(std::move(s)) {}
    ~Stmt() override = default;

    /**
     * \brief Get the type name of this statement
     *
     * \return Human-readable type name (e.g., "Stmt", "Assign", "Return")
     */
    [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

    static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using StmtPtr = std::shared_ptr<const Stmt>;

/**
 * \brief Assignment statement
 *
 * Represents an assignment operation: var = value
 * where var is a variable and value is an expression.
 */
class AssignStmt : public Stmt {
public:
    VarPtr var_;    // Variable
    ExprPtr value_; // Expression

    /**
     * \brief Create an assignment statement
     *
     * \param var Variable
     * \param value Expression
     * \param span Source location
     */
    AssignStmt(VarPtr var, ExprPtr value, Span span)
        : Stmt(std::move(span)), var_(std::move(var)), value_(std::move(value))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::AssignStmt; }
    [[nodiscard]] std::string TypeName() const override { return "AssignStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (var and value as DEF and USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::DefField(&AssignStmt::var_, "var"),
                                              reflection::UsualField(&AssignStmt::value_, "value")));
    }
};

using AssignStmtPtr = std::shared_ptr<const AssignStmt>;

/**
 * \brief Sequence of statements
 *
 * Represents a sequence of statements: stmt1; stmt2; ... stmtN
 * where stmts is a list of statements.
 */
class SeqStmts : public Stmt {
public:
    /**
     * \brief Create a sequence of statements
     *
     * \param stmts List of statements
     * \param span Source location
     */
    SeqStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

    /**
     * \brief Create a sequence of statements with no statements
     *
     * \param span Source location
     */
    SeqStmts(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SeqStmts; }
    [[nodiscard]] std::string TypeName() const override { return "SeqStmts"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (stmts as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&SeqStmts::stmts_, "stmts")));
    }

    /**
     * @brief Create a normalized statement from a list of statements
     *
     * Flattens nested SeqStmts and unwraps single-child sequences:
     * - Flatten({a, SeqStmts({b, c}), d}, span) → SeqStmts({a, b, c, d})
     * - Flatten({a}, span) → a
     * - Flatten({}, span) → SeqStmts({})
     */
    static StmtPtr Flatten(std::vector<StmtPtr> stmts, Span span)
    {
        std::vector<StmtPtr> flat;
        for (auto& s : stmts) {
            if (auto seq = AsMut(s)) {
                // Recursively flatten nested SeqStmts
                for (const auto& inner : seq->stmts_) {
                    if (auto inner_seq = AsMut(inner)) {
                        flat.insert(flat.end(), inner_seq->stmts_.begin(), inner_seq->stmts_.end());
                    } else {
                        flat.push_back(inner);
                    }
                }
            } else {
                flat.push_back(std::move(s));
            }
        }
        if (flat.size() == 1) {
            return flat[0];
        }
        return std::make_shared<SeqStmts>(std::move(flat), std::move(span));
    }

    static std::shared_ptr<SeqStmts> AsMut(StmtPtr stmt)
    {
        return std::dynamic_pointer_cast<SeqStmts>(std::const_pointer_cast<Stmt>(stmt));
    }

    /**
     * \brief Wrap a statement in a SeqStmts if it's not already one
     *
     * \param stmt Statement to wrap
     * \param span Source location
     * \return Wrapped statement
     */
    static std::shared_ptr<SeqStmts> Wrap(StmtPtr stmt, Span span)
    {
        if (auto seq = AsMut(stmt)) {
            return seq;
        }
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{stmt}, std::move(span));
    }

    static std::optional<std::shared_ptr<SeqStmts>> Wrap(std::optional<StmtPtr> stmt, Span span)
    {
        if (!stmt) {
            return std::nullopt;
        }
        return Wrap(stmt.value(), span);
    }

public:
    std::vector<StmtPtr> stmts_;
};

using SeqStmtsPtr = std::shared_ptr<SeqStmts>;

/**
 * \brief Conditional statement
 *
 * Represents an if-else statement: if condition then then_body else else_body
 * where condition is an expression and then_body/else_body is statement.
 */
class IfStmt : public Stmt {
public:
    /**
     * \brief Create a conditional statement with then and else branches
     *
     * \param condition Condition expression
     * \param thenBody Then branch statement
     * \param elseBody Else branch statement (can be optional)
     * \param returnVars Return variables (can be empty)
     * \param span Source location
     */
    IfStmt(ExprPtr condition, StmtPtr thenBody, std::optional<StmtPtr> elseBody, std::vector<VarPtr> returnVars,
           Span span)
        : Stmt(std::move(span)),
          condition_(std::move(condition)),
          thenBody_(SeqStmts::Wrap(thenBody, span)),
          elseBody_(SeqStmts::Wrap(elseBody, span)),
          returnVars_(std::move(returnVars))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::IfStmt; }
    [[nodiscard]] std::string TypeName() const override { return "IfStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (condition, then_body, else_body as USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&IfStmt::condition_, "condition"),
                                              reflection::UsualField(&IfStmt::thenBody_, "then_body"),
                                              reflection::UsualField(&IfStmt::elseBody_, "else_body"),
                                              reflection::DefField(&IfStmt::returnVars_, "return_vars")));
    }

public:
    ExprPtr condition_;                   // Condition expression
    SeqStmtsPtr thenBody_;                // Then branch statement
    std::optional<SeqStmtsPtr> elseBody_; // Else branch statement (optional)
    std::vector<VarPtr> returnVars_;      // Return variables (can be empty)
};

using IfStmtPtr = std::shared_ptr<const IfStmt>;

/**
 * \brief Yield statement
 *
 * Represents a yield operation: yield value
 * where value is a list of variables to yield.
 */
class YieldStmt : public Stmt {
public:
    /**
     * \brief Create a yield statement
     *
     * \param value List of variables to yield (can be empty)
     * \param span Source location
     */
    YieldStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

    /**
     * \brief Create a yield statement without values
     *
     * \param span Source location
     */
    explicit YieldStmt(Span span) : Stmt(std::move(span)), value_() {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::YieldStmt; }
    [[nodiscard]] std::string TypeName() const override { return "YieldStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&YieldStmt::value_, "value")));
    }

public:
    std::vector<ExprPtr> value_; // List of expressions to yield
};

using YieldStmtPtr = std::shared_ptr<const YieldStmt>;

/**
 * \brief Return statement
 *
 * Represents a return operation: return value
 * where value is a list of expressions to return.
 */
class ReturnStmt : public Stmt {
public:
    /**
     * \brief Create a return statement
     *
     * \param value List of expressions to return (can be empty)
     * \param span Source location
     */
    ReturnStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

    /**
     * \brief Create a return statement without values
     *
     * \param span Source location
     */
    explicit ReturnStmt(Span span) : Stmt(std::move(span)), value_() {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ReturnStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ReturnStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ReturnStmt::value_, "value")));
    }

public:
    std::vector<ExprPtr> value_; // List of expressions to return
};

using ReturnStmtPtr = std::shared_ptr<const ReturnStmt>;

/**
 * \brief For loop statement
 *
 * Represents a for loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** for loop_var in range(start, stop, step): body
 *
 * **Loop with iteration arguments:**
 * for loop_var, (iter_arg1, iter_arg2) in pl.range(start, stop, step, init_values=[...]):
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 * - IterArgs cannot be directly accessed outside the loop; use return_vars instead
 */
class ForStmt : public Stmt {
public:
    /**
     * \brief Create a for loop statement
     *
     * \param loopVar Loop variable
     * \param start Start value expression
     * \param stop Stop value expression
     * \param step Step value expression
     * \param iterArgs Iteration arguments (loop-carried values, scoped to loop body)
     * \param body Loop body statement (must yield values matching iterArgs if non-empty)
     * \param returnVars Return variables (capture final values, accessible after loop)
     * \param span Source location
     */
    ForStmt(VarPtr loopVar, ExprPtr start, ExprPtr stop, ExprPtr step, std::vector<IterArgPtr> iterArgs, StmtPtr body,
            std::vector<VarPtr> returnVars, Span span, std::vector<std::pair<std::string, std::any>> attrs = {})
        : Stmt(std::move(span)),
          loopVar_(std::move(loopVar)),
          start_(std::move(start)),
          stop_(std::move(stop)),
          step_(std::move(step)),
          iterArgs_(std::move(iterArgs)),
          body_(SeqStmts::Wrap(body, span)),
          returnVars_(std::move(returnVars)),
          attrs_(std::move(attrs))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ForStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ForStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (loop_var as DEF field, others as USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::DefField(&ForStmt::loopVar_, "loop_var"), reflection::UsualField(&ForStmt::start_, "start"),
                reflection::UsualField(&ForStmt::stop_, "stop"), reflection::UsualField(&ForStmt::step_, "step"),
                reflection::DefField(&ForStmt::iterArgs_, "iter_args"), reflection::UsualField(&ForStmt::body_, "body"),
                reflection::DefField(&ForStmt::returnVars_, "return_vars"),
                reflection::UsualField(&ForStmt::attrs_, "attrs")));
    }

    /// Get a typed attribute value (returns default_value if key not found)
    template <typename T>
    [[nodiscard]] T GetAttr(const std::string& key, const T& default_value = T{}) const
    {
        for (const auto& [k, v] : attrs_) {
            if (k == key)
                return AnyCast<T>(v, "for_stmt attr key: " + key);
        }
        return default_value;
    }

    /// Check if an attribute exists
    [[nodiscard]] bool HasAttr(const std::string& key) const
    {
        return std::any_of(attrs_.begin(), attrs_.end(), [&key](const auto& pair) { return pair.first == key; });
    }

public:
    VarPtr loopVar_;                   // Loop variable (e.g., i in "for i in range(...)")
    ExprPtr start_;                    // Start value expression
    ExprPtr stop_;                     // Stop value expression
    ExprPtr step_;                     // Step value expression
    std::vector<IterArgPtr> iterArgs_; // Loop-carried values (scoped to loop body)
    SeqStmtsPtr body_;                 // Loop body statement (must yield if iter_args non-empty)
    std::vector<VarPtr> returnVars_;   // Variables capturing final iteration values (accessible after loop)
    std::vector<std::pair<std::string, std::any>> attrs_; // Loop attributes
};

using ForStmtPtr = std::shared_ptr<const ForStmt>;

/**
 * \brief While loop statement
 *
 * Represents a while loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** while condition: body
 *
 * **Loop with iteration arguments:**
 * while condition, (iter_arg1, iter_arg2) with init_values=[...]:
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - condition: Boolean expression evaluated each iteration using current iter_args
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 */
class WhileStmt : public Stmt {
public:
    /**
     * \brief Create a while loop statement
     *
     * \param condition Boolean condition expression
     * \param iterArgs Iteration arguments (loop-carried values, scoped to loop body)
     * \param body Loop body statement (must yield values matching iterArgs if non-empty)
     * \param returnVars Return variables (capture final values, accessible after loop)
     * \param span Source location
     */
    WhileStmt(ExprPtr condition, std::vector<IterArgPtr> iterArgs, StmtPtr body, std::vector<VarPtr> returnVars,
              Span span)
        : Stmt(std::move(span)),
          condition_(std::move(condition)),
          iterArgs_(std::move(iterArgs)),
          body_(SeqStmts::Wrap(body, span)),
          returnVars_(std::move(returnVars))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::WhileStmt; }
    [[nodiscard]] std::string TypeName() const override { return "WhileStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (iter_args as DEF, condition as USUAL, body as USUAL, return_vars as
     * DEF). Iter args must be visited before condition/body so structural comparison can bind loop-carried vars first.
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::DefField(&WhileStmt::iterArgs_, "iter_args"),
                                              reflection::UsualField(&WhileStmt::condition_, "condition"),
                                              reflection::UsualField(&WhileStmt::body_, "body"),
                                              reflection::DefField(&WhileStmt::returnVars_, "return_vars")));
    }

public:
    ExprPtr condition_;                // Condition expression (evaluated each iteration)
    std::vector<IterArgPtr> iterArgs_; // Loop-carried values (scoped to loop body)
    SeqStmtsPtr body_;                 // Loop body statement (must yield if iter_args non-empty)
    std::vector<VarPtr> returnVars_;   // Variables capturing final iteration values (accessible after loop)
};

using WhileStmtPtr = std::shared_ptr<const WhileStmt>;

/**
 * \brief Section statement for hardware-specific execution regions
 */
class SectionStmt : public Stmt {
public:
    SectionStmt(SectionKind sectionKind, StmtPtr body, Span span)
        : Stmt(std::move(span)), sectionKind_(sectionKind), body_(SeqStmts::Wrap(body, span))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SectionStmt; }
    [[nodiscard]] std::string TypeName() const override { return "SectionStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&SectionStmt::sectionKind_, "section_kind"),
                                              reflection::UsualField(&SectionStmt::body_, "body")));
    }

public:
    SectionKind sectionKind_;
    SeqStmtsPtr body_;
};

using SectionStmtPtr = std::shared_ptr<const SectionStmt>;

/**
 * \brief Evaluation statement
 *
 * Represents an expression executed as a statement: expr
 * where expr is an expression (typically a Call).
 * This is used for expressions that have side effects but no return value
 * (or return value is ignored).
 */
class EvalStmt : public Stmt {
public:
    /**
     * \brief Create an evaluation statement
     *
     * \param expr Expression to execute
     * \param span Source location
     */
    EvalStmt(ExprPtr expr, Span span) : Stmt(std::move(span)), expr_(std::move(expr)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::EvalStmt; }
    [[nodiscard]] std::string TypeName() const override { return "EvalStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (expr as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&EvalStmt::expr_, "expr")));
    }

public:
    ExprPtr expr_; // Expression
};

using EvalStmtPtr = std::shared_ptr<const EvalStmt>;

/**
 * \brief Break statement
 *
 * Represents a break statement used to exit a loop.
 */
class BreakStmt : public Stmt {
public:
    std::vector<ExprPtr> value_;

    BreakStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}
    explicit BreakStmt(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::BreakStmt; }
    [[nodiscard]] std::string TypeName() const override { return "BreakStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&BreakStmt::value_, "value")));
    }
};

using BreakStmtPtr = std::shared_ptr<const BreakStmt>;

/**
 * \brief Continue statement
 *
 * Represents a continue statement used to skip to the next loop iteration.
 */
class ContinueStmt : public Stmt {
public:
    std::vector<ExprPtr> value_;

    ContinueStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}
    explicit ContinueStmt(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ContinueStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ContinueStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Stmt::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ContinueStmt::value_, "value")));
    }
};

using ContinueStmtPtr = std::shared_ptr<const ContinueStmt>;

/**
 * \brief Scalar operation statement
 */
class ScalarOpStmt : public Stmt {
public:
    VarPtr result_;
    VarPtr result_token_;
    std::string opcode_;
    std::vector<ExprPtr> args_;

    ScalarOpStmt(VarPtr result, VarPtr result_token, std::string opcode, std::vector<ExprPtr> args, Span span)
        : Stmt(std::move(span)),
          result_(std::move(result)),
          result_token_(std::move(result_token)),
          opcode_(std::move(opcode)),
          args_(std::move(args))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScalarOpStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ScalarOpStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (var and value as DEF and USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        auto newFields = std::make_tuple(reflection::DefField(&ScalarOpStmt::result_, "result"),
                                         reflection::UsualField(&ScalarOpStmt::result_token_, "result_token"),
                                         reflection::UsualField(&ScalarOpStmt::opcode_, "opcode"),
                                         reflection::UsualField(&ScalarOpStmt::args_, "args"));
        return std::tuple_cat(Stmt::GetFieldDescriptors(), newFields);
    }
};

using ScalarOpStmtPtr = std::shared_ptr<const ScalarOpStmt>;

/**
 * \brief Tensor operation statement
 */
class TensorOpStmt : public Stmt {
public:
    std::vector<VarPtr> result_;
    std::vector<VarPtr> result_token_;
    std::string opcode_;
    std::vector<ExprPtr> args_;
    std::vector<VarPtr> tokens_;
    std::vector<std::pair<std::string, std::any>> attrs_;

    TensorOpStmt(std::vector<VarPtr> result, std::vector<VarPtr> result_token, std::string opcode,
                 std::vector<ExprPtr> args, std::vector<VarPtr> tokens,
                 std::vector<std::pair<std::string, std::any>> attrs, Span span)
        : Stmt(std::move(span)),
          result_(std::move(result)),
          result_token_(std::move(result_token)),
          opcode_(std::move(opcode)),
          args_(std::move(args)),
          tokens_(std::move(tokens)),
          attrs_(std::move(attrs))
    {}

    TensorOpStmt(std::vector<VarPtr> result, VarPtr result_token, std::string opcode, std::vector<ExprPtr> args,
                 std::vector<VarPtr> tokens, std::vector<std::pair<std::string, std::any>> attrs, Span span)
        : TensorOpStmt(std::move(result),
                       result_token ? std::vector<VarPtr>{std::move(result_token)} : std::vector<VarPtr>{},
                       std::move(opcode), std::move(args), std::move(tokens), std::move(attrs), std::move(span))
    {}

    explicit TensorOpStmt(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TensorOpStmt; }
    [[nodiscard]] std::string TypeName() const override { return "TensorOpStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (var and value as DEF and USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        auto newFields = std::make_tuple(reflection::DefField(&TensorOpStmt::result_, "result"),
                                         reflection::UsualField(&TensorOpStmt::result_token_, "result_token"),
                                         reflection::UsualField(&TensorOpStmt::opcode_, "opcode"),
                                         reflection::UsualField(&TensorOpStmt::args_, "args"),
                                         reflection::UsualField(&TensorOpStmt::tokens_, "tokens"),
                                         reflection::UsualField(&TensorOpStmt::attrs_, "attrs"));
        return std::tuple_cat(Stmt::GetFieldDescriptors(), newFields);
    }
};

using TensorOpStmtPtr = std::shared_ptr<const TensorOpStmt>;
} // namespace ir
} // namespace pypto
