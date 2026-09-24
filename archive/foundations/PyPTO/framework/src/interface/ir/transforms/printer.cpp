/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstddef>
#include <cstring>
#include <any>
#include <iomanip>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/type.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/stmt.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/printer.h"

#include "tilefwk/symbolic_scalar.h"
#include "interface/tensor/ir.h"

using npu::tile_fwk::SymbolicScalar;

namespace pypto {
namespace ir {

Precedence GetPrecedence(const ExprPtr& expr)
{
    static const std::unordered_map<std::type_index, Precedence> kPrecedenceMap = {
        // Logical operators
        {std::type_index(typeid(Or)), Precedence::kOr},
        {std::type_index(typeid(Xor)), Precedence::kXor},
        {std::type_index(typeid(And)), Precedence::kAnd},
        {std::type_index(typeid(Not)), Precedence::kNot},

        // Comparison operators
        {std::type_index(typeid(Eq)), Precedence::kComparison},
        {std::type_index(typeid(Ne)), Precedence::kComparison},
        {std::type_index(typeid(Lt)), Precedence::kComparison},
        {std::type_index(typeid(Le)), Precedence::kComparison},
        {std::type_index(typeid(Gt)), Precedence::kComparison},
        {std::type_index(typeid(Ge)), Precedence::kComparison},

        // Bitwise operators
        {std::type_index(typeid(BitOr)), Precedence::kBitOr},
        {std::type_index(typeid(BitXor)), Precedence::kBitXor},
        {std::type_index(typeid(BitAnd)), Precedence::kBitAnd},
        {std::type_index(typeid(BitShiftLeft)), Precedence::kBitShift},
        {std::type_index(typeid(BitShiftRight)), Precedence::kBitShift},

        // Arithmetic operators
        {std::type_index(typeid(Add)), Precedence::kAddSub},
        {std::type_index(typeid(Sub)), Precedence::kAddSub},
        {std::type_index(typeid(Mul)), Precedence::kMulDivMod},
        {std::type_index(typeid(FloorDiv)), Precedence::kMulDivMod},
        {std::type_index(typeid(FloatDiv)), Precedence::kMulDivMod},
        {std::type_index(typeid(FloorMod)), Precedence::kMulDivMod},
        {std::type_index(typeid(Pow)), Precedence::kPow},

        // Unary operators
        {std::type_index(typeid(Neg)), Precedence::kUnary},
        {std::type_index(typeid(BitNot)), Precedence::kUnary},

        // Function-like operators and atoms
        {std::type_index(typeid(Abs)), Precedence::kCall},
        {std::type_index(typeid(Cast)), Precedence::kCall},
        {std::type_index(typeid(Min)), Precedence::kCall},
        {std::type_index(typeid(Max)), Precedence::kCall},
        {std::type_index(typeid(Call)), Precedence::kCall},
        {std::type_index(typeid(Var)), Precedence::kAtom},
        {std::type_index(typeid(ConstInt)), Precedence::kAtom},
        {std::type_index(typeid(ConstFloat)), Precedence::kAtom},
        {std::type_index(typeid(ConstBool)), Precedence::kAtom},
        {std::type_index(typeid(ScalarExpr)), Precedence::kAtom},
        {std::type_index(typeid(GetItemExpr)), Precedence::kAtom},
    };

    INTERNAL_CHECK(expr) << "Expression is null";
    const Expr& expr_ref = *expr;
    const auto it = kPrecedenceMap.find(std::type_index(typeid(expr_ref)));
    if (it != kPrecedenceMap.end()) {
        return it->second;
    }

    // Default for any other expression types.
    return Precedence::kAtom;
}

bool IsRightAssociative(const ExprPtr& expr)
{
    // Only ** (power) is right-associative in Python
    return IsA<Pow>(expr);
}

bool NeedsParensForPrint(const ExprPtr& parent, const ExprPtr& child, bool is_left)
{
    Precedence parent_prec = GetPrecedence(parent);
    Precedence child_prec = GetPrecedence(child);
    if (child_prec < parent_prec) {
        return true;
    }

    if (child_prec == parent_prec) {
        if (IsRightAssociative(parent)) {
            return is_left;
        } else {
            return !is_left;
        }
    }
    return false;
}

void PrintIRNodeWithVisitor(IRVisitor& visitor, std::ostream& stream, const IRNodePtr& node)
{
    if (auto program = As<Program>(node)) {
        visitor.VisitProgram(program);
    } else if (auto func = As<Function>(node)) {
        visitor.VisitFunction(func);
    } else if (auto stmt = As<Stmt>(node)) {
        visitor.VisitStmt(stmt);
    } else if (auto expr = As<Expr>(node)) {
        visitor.VisitExpr(expr);
    } else {
        stream << "<unsupported IRNode type>";
    }
}

void PrintChildExprWithParens(IRVisitor& visitor, std::ostream& stream, const ExprPtr& parent, const ExprPtr& child,
                              bool is_left)
{
    bool needs_parens = NeedsParensForPrint(parent, child, is_left);
    if (needs_parens) {
        stream << "(";
    }

    visitor.VisitExpr(child);

    if (needs_parens) {
        stream << ")";
    }
}

void PrintReturnStmtValues(IRVisitor& visitor, std::ostream& stream, const std::vector<ExprPtr>& values)
{
    stream << "return";
    if (!values.empty()) {
        stream << " ";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0)
                stream << ", ";
            visitor.VisitExpr(values[i]);
        }
    }
}

void PrintFunctionReturnAnnotation(std::ostream& stream, const std::vector<TypePtr>& return_types,
                                   const std::function<std::string(const TypePtr&)>& print_type)
{
    if (!return_types.empty()) {
        stream << " -> ";
        if (return_types.size() == 1) {
            stream << print_type(return_types[0]);
        } else {
            stream << "tuple[";
            for (size_t i = 0; i < return_types.size(); ++i) {
                if (i > 0)
                    stream << ", ";
                stream << print_type(return_types[i]);
            }
            stream << "]";
        }
    }
}

namespace {

std::string FormatFloatLiteral(double value)
{
    // Check if the value is an integer (no fractional part)
    if (std::fabs(value) - std::floor(value) < 1e-10) {
        // For integer values, format as X.0
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << value;
        return oss.str();
    } else {
        // For non-integer values, use default formatting with enough precision
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

const char* TileLayoutToPythonName(TileLayout layout)
{
    switch (layout) {
        case TileLayout::none_box:
            return "none_box";
        case TileLayout::row_major:
            return "row_major";
        case TileLayout::col_major:
            return "col_major";
        default:
            INTERNAL_CHECK(false) << "Unknown TileLayout in PrintTileView";
            return "";
    }
}

const char* TilePadToPythonName(TilePad pad)
{
    switch (pad) {
        case TilePad::null:
            return "null";
        case TilePad::zero:
            return "zero";
        case TilePad::max:
            return "max";
        case TilePad::min:
            return "min";
        default:
            INTERNAL_CHECK(false) << "Unknown TilePad in PrintTileView";
            return "";
    }
}

const char* CompactModeToPythonName(CompactMode compact)
{
    switch (compact) {
        case CompactMode::null:
            return "null";
        case CompactMode::normal:
            return "normal";
        case CompactMode::row_plus_one:
            return "row_plus_one";
        default:
            INTERNAL_CHECK(false) << "Unknown CompactMode in PrintTileView";
            return "";
    }
}

const char* TensorLayoutToPythonName(TensorLayout layout)
{
    const char* full = EnumToString(layout);
    const char* sep = std::strstr(full, "::");
    return sep ? sep + 2 : full;
}

static void PrintKwargValue(std::ostream& stream, const std::string& prefix, const std::string& key,
                            const std::any& value)
{
    stream << std::quoted(key) << ": ";

    if (value.type() == typeid(int)) {
        auto intVal = AnyCast<int>(value, key);
        if (key == "set_pipe" || key == "wait_pipe") {
            stream << prefix << ".PipeType." << PipeTypeToString(static_cast<PipeType>(intVal));
        } else {
            stream << intVal;
        }
    } else if (value.type() == typeid(uint64_t)) {
        stream << (AnyCast<uint64_t>(value));
    } else if (value.type() == typeid(double)) {
        stream << FormatFloatLiteral(AnyCast<double>(value, key));
    } else if (value.type() == typeid(float)) {
        stream << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, key)));
    } else if (value.type() == typeid(bool)) {
        stream << (AnyCast<bool>(value, key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
        stream << std::quoted(AnyCast<std::string>(value, key));
    } else if (value.type() == typeid(DataType)) {
        stream << prefix << "." << DTypeToString(AnyCast<DataType>(value, key));
    } else if (value.type() == typeid(MemorySpace)) {
        stream << prefix << ".MemorySpace." << MemorySpaceToString(AnyCast<MemorySpace>(value, key));
    } else if (value.type() == typeid(SymbolicScalar)) {
        stream << AnyCast<SymbolicScalar>(value, key).Dump();
    } else if (value.type() == typeid(std::vector<int>)) {
        const auto& values = AnyCast<std::vector<int>>(value, key);
        stream << "[";
        for (size_t j = 0; j < values.size(); ++j) {
            if (j != 0)
                stream << ", ";
            stream << values[j];
        }
        stream << "]";
    } else if (value.type() == typeid(std::vector<int64_t>)) {
        const auto& values = AnyCast<std::vector<int64_t>>(value, key);
        stream << "[";
        for (size_t j = 0; j < values.size(); ++j) {
            if (j != 0)
                stream << ", ";
            stream << values[j];
        }
        stream << "]";
    } else if (value.type() == typeid(std::vector<SymbolicScalar>)) {
        const auto& values = AnyCast<std::vector<SymbolicScalar>>(value, key);
        stream << "[";
        for (size_t j = 0; j < values.size(); ++j) {
            if (j != 0)
                stream << ", ";
            stream << values[j].Dump();
        }
        stream << "]";
    } else {
        stream << "Unsupported";
    }
}

void PrintIterArgNames(std::ostringstream& stream, const std::vector<IterArgPtr>& iter_args)
{
    stream << "(";
    for (size_t i = 0; i < iter_args.size(); ++i) {
        if (i > 0)
            stream << ", ";
        stream << iter_args[i]->iterVar_->name_;
    }
    if (iter_args.size() == 1) {
        stream << ",";
    }
    stream << ")";
}

template <typename VisitExprFn>
void PrintIterArgInitValues(std::ostringstream& stream, const std::vector<IterArgPtr>& iter_args,
                            const VisitExprFn& visit_expr)
{
    stream << "init_values=(";
    for (size_t i = 0; i < iter_args.size(); ++i) {
        if (i > 0)
            stream << ", ";
        visit_expr(iter_args[i]->initValue_);
    }
    if (iter_args.size() == 1) {
        stream << ",";
    }
    stream << ")";
}

template <typename VisitExprFn>
void PrintForRangeHeader(std::ostringstream& stream, const std::string& prefix, const ForStmtPtr& op,
                         const VisitExprFn& visit_expr)
{
    stream << "for " << op->loopVar_->name_;
    if (!op->iterArgs_.empty()) {
        stream << ", ";
        PrintIterArgNames(stream, op->iterArgs_);
    }

    stream << " in " << prefix << ".range(";
    visit_expr(op->start_);
    stream << ", ";
    visit_expr(op->stop_);
    stream << ", ";
    visit_expr(op->step_);

    if (!op->iterArgs_.empty()) {
        stream << ", ";
        PrintIterArgInitValues(stream, op->iterArgs_, visit_expr);
    }
    if (!op->attrs_.empty()) {
        stream << ", attrs={";
        for (size_t i = 0; i < op->attrs_.size(); ++i) {
            const auto& [key, value] = op->attrs_[i];
            if (key.size() > 1 && key[0] == '_') // internal attrs, skip it
                continue;
            if (i != 0)
                stream << ", ";
            PrintKwargValue(stream, prefix, key, value);
        }
        stream << "}";
    }
    stream << "):\n";
}

template <typename VisitExprFn>
void PrintWhileIterArgsHeader(std::ostringstream& stream, const std::string& prefix, const WhileStmtPtr& op,
                              const VisitExprFn& visit_expr)
{
    stream << "for ";
    PrintIterArgNames(stream, op->iterArgs_);
    stream << " in " << prefix << ".while_(";
    PrintIterArgInitValues(stream, op->iterArgs_, visit_expr);
    stream << "):\n";
}
} // namespace

/**
 * \brief Python-style IR printer
 *
 * Prints IR nodes in Python syntax with type annotations and SSA-style control flow.
 * This is the recommended printer for new code that outputs valid Python syntax.
 *
 * Key features:
 * - Type annotations (e.g., x: pl.INT64, a: pl.Tensor[[4, 8], pl.FP32])
 * - SSA-style if/for with pl.yield_() and pl.range()
 * - Op attributes as keyword arguments
 * - Program headers with # pypto.program: name
 */
class IRPrinter : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    explicit IRPrinter(std::string prefix = "ir", bool concise = false) : prefix_(std::move(prefix)), concise_(concise)
    {}
    ~IRPrinter() override = default;

    /**
     * \brief Print an IR node to a string in Python IR syntax
     *
     * \param node IR node to print (can be Expr, Stmt, Function, or Program)
     * \return Python-style string representation
     */
    std::string Print(const IRNodePtr& node);
    std::string Print(const TypePtr& type);

protected:
    PYPTO_IR_PRINTER_COMMON_VISITOR_OVERRIDES();
    void VisitExpr_(const ScalarExprPtr& op) override;
    void VisitStmt_(const TensorOpStmtPtr& op) override;
    void VisitStmt_(const ScalarOpStmtPtr& op) override;
    void VisitStmt_(const SectionStmtPtr& op) override;

private:
    std::ostringstream stream_;
    int indent_ = 0;
    std::string prefix_; // Prefix for type names (e.g., "pl" or "ir")
    bool concise_;       // When true, omit intermediate type annotations

    std::string GetIndent() const;

    // SeqStmts is a transparent container - recursed into without extra indent.
    void PrintStmtBlock(const StmtPtr& stmt);

    // Statement body visitor with SSA-style handling
    void VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars = {});
    void VisitFunctionBody(const StmtPtr& body);

    // If `stmt` is a value-carrying terminator (yield/break/continue) and `return_vars` is
    // non-empty, print it as `return_vars = <terminator>`
    bool PrintTerminator(const StmtPtr& stmt, const std::vector<VarPtr>& return_vars);

    // Binary/unary operator helpers (reuse precedence logic)
    void PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol);
    void PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name);
    void PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left);
    void PrintCallKwargs(const CallPtr& op, bool need_comma);

    // Shape printing helper
    void PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape);
    void PrintExprList(std::ostringstream& oss, const std::vector<ExprPtr>& exprs);

    // Print an expression for use in type annotations (shapes, views).
    std::string PrintExprForType(const ExprPtr& expr);

    // MemRef and TileView printing helpers
    std::string PrintMemRef(const MemRef& memref);
    std::string PrintTileView(const TileView& tile_view);
    std::string PrintHardwareInfo(const HardwareInfo& hw_info);
    std::string PrintTensorView(const TensorView& tensor_view);
};

// DataTypeToPythonString removed — now uses DTypeToString from dtype.h

// IRPrinter implementation
std::string IRPrinter::Print(const IRNodePtr& node)
{
    stream_.str("");
    stream_.clear();
    indent_ = 0;
    PrintIRNodeWithVisitor(*this, stream_, node);
    return stream_.str();
}

std::string IRPrinter::Print(const TypePtr& type)
{
    if (auto scalar_type = As<ScalarType>(type)) {
        // Print as pl.Scalar[pl.INT64] for proper round-trip support
        return prefix_ + ".Scalar[" + prefix_ + "." + DTypeToString(scalar_type->dtype_) + "]";
    }

    if (auto tensor_type = As<TensorType>(type)) {
        std::ostringstream oss;
        // Subscript-style: pl.Tensor[[shape], dtype]
        oss << prefix_ << ".Tensor[[";
        PrintShapeDims(oss, tensor_type->shape_);
        oss << "], " << prefix_ << "." << DTypeToString(tensor_type->dtype_);

        if (tensor_type->tensor_view_.has_value()) {
            oss << ", tensor_view=" << PrintTensorView(tensor_type->tensor_view_.value());
        }

        // Add optional memref as positional arg
        if (tensor_type->memref_.has_value()) {
            oss << ", " << PrintMemRef(*tensor_type->memref_.value());
        }

        oss << "]";
        return oss.str();
    }

    if (auto tile_type = As<TileType>(type)) {
        std::ostringstream oss;
        oss << prefix_ << ".TileType([";
        PrintShapeDims(oss, tile_type->shape_);
        oss << "], " << prefix_ << "." << DTypeToString(tile_type->dtype_);

        if (tile_type->memref_.has_value()) {
            oss << ", " << PrintMemRef(*tile_type->memref_.value());
        }

        if (tile_type->tileView_.has_value()) {
            oss << ", tile_view=" << PrintTileView(tile_type->tileView_.value());
        }

        if (tile_type->hardwareInfo_.has_value()) {
            oss << ", hardware_info=" << PrintHardwareInfo(tile_type->hardwareInfo_.value());
        }

        oss << ")";
        return oss.str();
    }

    if (auto tuple_type = As<TupleType>(type)) {
        std::ostringstream oss;
        if (tuple_type->types_.empty()) {
            oss << prefix_ << ".Tuple[()]";
        } else {
            oss << prefix_ << ".Tuple[";
            for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
                if (i > 0)
                    oss << ", ";
                oss << Print(tuple_type->types_[i]);
            }
            oss << "]";
        }
        return oss.str();
    }

    if (auto memref_type = As<MemRefType>(type)) {
        return prefix_ + ".MemRefType";
    }

    if (auto ptr_type = As<PtrType>(type)) {
        return prefix_ + ".Ptr";
    }

    if (auto token_type = As<TokenType>(type)) {
        switch (token_type->kind_) {
            case TokenKind::READ:
                return prefix_ + ".ReadToken";
            case TokenKind::WRITE:
                return prefix_ + ".WriteToken";
            case TokenKind::NORMAL:
            default:
                return prefix_ + ".Token";
        }
    }

    if (auto none_type = As<NoneType>(type)) {
        return prefix_ + ".None";
    }

    if (auto logical_tensor_type = As<LogicalTensorType>(type)) {
        return prefix_ + ".Tensor";
    }

    return prefix_ + ".Unknown";
}

std::string IRPrinter::GetIndent() const { return std::string(static_cast<size_t>(indent_ * 4), ' '); }

// Expression visitors - reuse precedence logic from base printer
void IRPrinter::VisitExpr_(const VarPtr& op)
{
    if (auto type = As<LogicalTensorType>(op->GetType())) {
        stream_ << DumpTensorVar(op);
    } else {
        stream_ << op->name_;
    }
}

void IRPrinter::VisitExpr_(const MemRefPtr& op) { stream_ << PrintMemRef(*op); }

void IRPrinter::VisitExpr_(const ConstIntPtr& op) { stream_ << op->value_; }

void IRPrinter::VisitExpr_(const ConstFloatPtr& op) { stream_ << FormatFloatLiteral(op->value_); }

void IRPrinter::VisitExpr_(const ConstBoolPtr& op) { stream_ << (op->value_ ? "True" : "False"); }

void IRPrinter::VisitExpr_(const CallPtr& op)
{
    stream_ << prefix_ << ".call @" << op->name_ << "(";
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->args_[i]);
    }
    PrintCallKwargs(op, !op->args_.empty());
    stream_ << ")";
}

void IRPrinter::VisitExpr_(const MakeTuplePtr& op)
{
    stream_ << "[";
    for (size_t i = 0; i < op->elements_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->elements_[i]);
    }
    stream_ << "]";
}

void IRPrinter::VisitExpr_(const GetItemExprPtr& op)
{
    VisitExpr(op->value_);
    stream_ << "[";
    VisitExpr(op->slice_);
    stream_ << "]";
}

void IRPrinter::VisitExpr_(const ScalarExprPtr& op)
{
    auto scalar_type = As<ScalarType>(op->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "ScalarExpr has non-scalar type";
    stream_ << DumpScalarExpr(op);
}

// Binary and unary operators - reuse from base printer logic
void IRPrinter::PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left)
{
    PrintChildExprWithParens(*this, stream_, parent, child, is_left);
}

void IRPrinter::PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol)
{
    PrintChild(op, op->left_, true);
    stream_ << " " << op_symbol << " ";
    PrintChild(op, op->right_, false);
}

void IRPrinter::PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name)
{
    stream_ << prefix_ << "." << func_name << "(";
    VisitExpr(op->left_);
    stream_ << ", ";
    VisitExpr(op->right_);
    stream_ << ")";
}

void IRPrinter::PrintCallKwargs(const CallPtr& op, bool need_comma)
{
    for (const auto& [key, value] : op->kwargs_) {
        if (need_comma) {
            stream_ << ", ";
        }
        need_comma = true;
        stream_ << key << "=";
        PrintKwargValue(stream_, prefix_, key, value);
    }
}

// Arithmetic binary operators
void IRPrinter::VisitExpr_(const AddPtr& op) { PrintBinaryOp(op, "+"); }
void IRPrinter::VisitExpr_(const SubPtr& op) { PrintBinaryOp(op, "-"); }
void IRPrinter::VisitExpr_(const MulPtr& op) { PrintBinaryOp(op, "*"); }
void IRPrinter::VisitExpr_(const FloorDivPtr& op) { PrintBinaryOp(op, "//"); }
void IRPrinter::VisitExpr_(const FloorModPtr& op) { PrintBinaryOp(op, "%"); }
void IRPrinter::VisitExpr_(const FloatDivPtr& op) { PrintBinaryOp(op, "/"); }
void IRPrinter::VisitExpr_(const PowPtr& op) { PrintBinaryOp(op, "**"); }

// Function-style binary operators
void IRPrinter::VisitExpr_(const MinPtr& op) { PrintFunctionBinaryOp(op, "min"); }
void IRPrinter::VisitExpr_(const MaxPtr& op) { PrintFunctionBinaryOp(op, "max"); }

// Comparison operators
void IRPrinter::VisitExpr_(const EqPtr& op) { PrintBinaryOp(op, "=="); }
void IRPrinter::VisitExpr_(const NePtr& op) { PrintBinaryOp(op, "!="); }
void IRPrinter::VisitExpr_(const LtPtr& op) { PrintBinaryOp(op, "<"); }
void IRPrinter::VisitExpr_(const LePtr& op) { PrintBinaryOp(op, "<="); }
void IRPrinter::VisitExpr_(const GtPtr& op) { PrintBinaryOp(op, ">"); }
void IRPrinter::VisitExpr_(const GePtr& op) { PrintBinaryOp(op, ">="); }

// Logical operators
void IRPrinter::VisitExpr_(const AndPtr& op) { PrintBinaryOp(op, "and"); }
void IRPrinter::VisitExpr_(const OrPtr& op) { PrintBinaryOp(op, "or"); }
void IRPrinter::VisitExpr_(const XorPtr& op) { PrintBinaryOp(op, "xor"); }

// Bitwise operators
void IRPrinter::VisitExpr_(const BitAndPtr& op) { PrintBinaryOp(op, "&"); }
void IRPrinter::VisitExpr_(const BitOrPtr& op) { PrintBinaryOp(op, "|"); }
void IRPrinter::VisitExpr_(const BitXorPtr& op) { PrintBinaryOp(op, "^"); }
void IRPrinter::VisitExpr_(const BitShiftLeftPtr& op) { PrintBinaryOp(op, "<<"); }
void IRPrinter::VisitExpr_(const BitShiftRightPtr& op) { PrintBinaryOp(op, ">>"); }

// Unary operators
void IRPrinter::VisitExpr_(const NegPtr& op)
{
    stream_ << "-";
    Precedence operand_prec = GetPrecedence(op->operand_);
    if (operand_prec < Precedence::kUnary) {
        stream_ << "(";
        VisitExpr(op->operand_);
        stream_ << ")";
    } else {
        VisitExpr(op->operand_);
    }
}

void IRPrinter::VisitExpr_(const AbsPtr& op)
{
    stream_ << prefix_ << ".abs(";
    VisitExpr(op->operand_);
    stream_ << ")";
}

void IRPrinter::VisitExpr_(const CastPtr& op)
{
    auto scalar_type = As<ScalarType>(op->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "Cast has non-scalar type";
    stream_ << prefix_ << ".cast(";
    VisitExpr(op->operand_);
    stream_ << ", " << prefix_ << "." << DTypeToString(scalar_type->dtype_) << ")";
}

void IRPrinter::VisitExpr_(const NotPtr& op)
{
    stream_ << "not ";
    Precedence operand_prec = GetPrecedence(op->operand_);
    if (operand_prec < Precedence::kNot) {
        stream_ << "(";
        VisitExpr(op->operand_);
        stream_ << ")";
    } else {
        VisitExpr(op->operand_);
    }
}

void IRPrinter::VisitExpr_(const BitNotPtr& op)
{
    stream_ << "~";
    Precedence operand_prec = GetPrecedence(op->operand_);
    if (operand_prec < Precedence::kUnary) {
        stream_ << "(";
        VisitExpr(op->operand_);
        stream_ << ")";
    } else {
        VisitExpr(op->operand_);
    }
}

// Statement visitors with proper Python syntax
void IRPrinter::VisitStmt_(const AssignStmtPtr& op)
{
    // Print with type annotation: var: type = value
    // In concise mode, omit the type annotation: var = value
    VisitExpr(op->var_);
    if (!concise_ && !As<TileType>(op->var_->GetType())) {
        stream_ << ": " << Print(op->var_->GetType());
    }
    stream_ << " = ";
    VisitExpr(op->value_);
}

void IRPrinter::VisitStmt_(const IfStmtPtr& op)
{
    // SSA-style if with pl.yield_()
    stream_ << "if ";
    VisitExpr(op->condition_);
    stream_ << ":\n";

    indent_++;
    VisitStmtBody(op->thenBody_, op->returnVars_);
    indent_--;

    if (op->elseBody_.has_value()) {
        stream_ << "\n" << GetIndent() << "else:\n";
        indent_++;
        VisitStmtBody(*op->elseBody_, op->returnVars_);
        indent_--;
    }
}

void IRPrinter::VisitStmt_(const YieldStmtPtr& op)
{
    // Note: In function context, this will be changed to "return" by VisitFunction
    stream_ << prefix_ << ".yield_(";
    for (size_t i = 0; i < op->value_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->value_[i]);
    }
    stream_ << ")";
}

void IRPrinter::VisitStmt_(const ReturnStmtPtr& op) { PrintReturnStmtValues(*this, stream_, op->value_); }

void IRPrinter::VisitStmt_(const ForStmtPtr& op)
{
    PrintForRangeHeader(stream_, prefix_, op, [this](const ExprPtr& expr) { VisitExpr(expr); });

    indent_++;
    VisitStmtBody(op->body_, op->returnVars_);
    indent_--;
}

void IRPrinter::VisitStmt_(const WhileStmtPtr& op)
{
    // Check if this is SSA-style (with iter_args) or natural style
    if (op->iterArgs_.empty()) {
        // Natural while loop without iter_args
        stream_ << "while ";
        VisitExpr(op->condition_);
        stream_ << ":\n";

        indent_++;
        VisitStmtBody(op->body_, op->returnVars_);
        indent_--;
    } else {
        // SSA-style while with iter_args - print as explicit DSL syntax
        PrintWhileIterArgsHeader(stream_, prefix_, op, [this](const ExprPtr& expr) { VisitExpr(expr); });

        indent_++;

        // Print condition as pl.cond() call as first body statement
        stream_ << GetIndent() << prefix_ << ".cond(";
        VisitExpr(op->condition_);
        stream_ << ")\n";

        VisitStmtBody(op->body_, op->returnVars_);
        indent_--;
    }
}

void IRPrinter::VisitStmt_(const SeqStmtsPtr& op)
{
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        PrintStmtBlock(op->stmts_[i]);
        if (i < op->stmts_.size() - 1) {
            stream_ << "\n";
        }
    }
}

void IRPrinter::PrintStmtBlock(const StmtPtr& stmt)
{
    if (auto seq = As<SeqStmts>(stmt)) {
        for (size_t i = 0; i < seq->stmts_.size(); ++i) {
            PrintStmtBlock(seq->stmts_[i]);
            if (i < seq->stmts_.size() - 1)
                stream_ << "\n";
        }
    } else {
        stream_ << GetIndent();
        VisitStmt(stmt);
    }
}

void IRPrinter::VisitStmt_(const EvalStmtPtr& op)
{
    // Print expression statement: expr
    stream_ << prefix_ << ".eval(";
    VisitExpr(op->expr_);
    stream_ << ")";
}

void IRPrinter::VisitStmt_(const BreakStmtPtr& op)
{
    stream_ << "break";
    for (size_t i = 0; i < op->value_.size(); ++i) {
        stream_ << (i == 0 ? " " : ", ");
        VisitExpr(op->value_[i]);
    }
}

void IRPrinter::VisitStmt_(const ContinueStmtPtr& op)
{
    stream_ << "continue";
    for (size_t i = 0; i < op->value_.size(); ++i) {
        stream_ << (i == 0 ? " " : ", ");
        VisitExpr(op->value_[i]);
    }
}

void IRPrinter::VisitStmt_(const TensorOpStmtPtr& op)
{
    // res [, res_token] = opcode(args, tokens=[], attrs=[])
    if (op->result_.size() == 1) {
        VisitExpr(op->result_[0]);
    } else {
        stream_ << "[";
        for (size_t i = 0; i < op->result_.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            VisitExpr(op->result_[i]);
        }
        stream_ << "]";
    }
    for (const auto& token : op->result_token_) {
        stream_ << ", ";
        VisitExpr(token);
    }
    stream_ << " = " << op->opcode_ << "(";
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->args_[i]);
    }
    if (!op->tokens_.empty()) {
        stream_ << ", tokens=[";
        for (size_t i = 0; i < op->tokens_.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            VisitExpr(op->tokens_[i]);
        }
        stream_ << "]";
    }

    if (!op->attrs_.empty()) {
        stream_ << ", attrs=[";
        for (size_t i = 0; i < op->attrs_.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            PrintKwargValue(stream_, prefix_, op->attrs_[i].first, op->attrs_[i].second);
        }
        stream_ << "]";
    } else {
        auto attrs = CollectTensorOpAttrs(op);
        if (!attrs.empty()) {
            stream_ << ", attrs=[";
            bool need_comma = false;
            for (auto [name, val] : attrs) {
                if (need_comma)
                    stream_ << ", ";
                need_comma = true;
                PrintKwargValue(stream_, prefix_, name, val);
            }
            stream_ << "]";
        }
    }
    stream_ << ")";
}

void IRPrinter::VisitStmt_(const ScalarOpStmtPtr& op)
{
    // res [, res_token] = opcode(args)
    VisitExpr(op->result_);
    if (op->result_token_) {
        stream_ << ", ";
        VisitExpr(op->result_token_);
    }
    stream_ << " = " << op->opcode_ << "(";
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->args_[i]);
    }
    stream_ << ")";
}

void IRPrinter::VisitStmt_(const SectionStmtPtr& op)
{
    static const std::unordered_map<SectionKind, std::string> section_kind_to_name = {
        {SectionKind::Vector, "section_vector"},
        {SectionKind::Cube, "section_cube"},
        {SectionKind::VF, "section_vf"},
    };

    auto it = section_kind_to_name.find(op->sectionKind_);
    INTERNAL_CHECK(it != section_kind_to_name.end())
        << "Internal error: Unknown SectionKind in printer: " << SectionKindToString(op->sectionKind_);

    stream_ << "with " << prefix_ << "." << it->second << "():\n";
    indent_++;
    PrintStmtBlock(op->body_);
    indent_--;
}

void IRPrinter::VisitStmt_(const StmtPtr& op) { stream_ << op->TypeName(); }

bool IRPrinter::PrintTerminator(const StmtPtr& stmt, const std::vector<VarPtr>& return_vars)
{
    auto has_values = [](const StmtPtr& s) -> bool {
        if (auto yield = As<YieldStmt>(s)) {
            return !yield->value_.empty();
        }
        if (auto brk = As<BreakStmt>(s)) {
            return !brk->value_.empty();
        }
        if (auto cont = As<ContinueStmt>(s)) {
            return !cont->value_.empty();
        }
        return false;
    };
    if (return_vars.empty() || !has_values(stmt)) {
        return false;
    }
    stream_ << GetIndent();
    for (size_t i = 0; i < return_vars.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        stream_ << return_vars[i]->name_;
    }
    stream_ << " = ";
    VisitStmt(stmt);
    return true;
}

void IRPrinter::VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars)
{
    if (auto seq_stmts = As<SeqStmts>(body)) {
        // Process each statement in sequence
        if (seq_stmts->stmts_.empty()) {
            stream_ << GetIndent() << "pass";
            return;
        }
        for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
            auto& stmt = seq_stmts->stmts_[i];
            if (i < seq_stmts->stmts_.size() - 1) {
                PrintStmtBlock(stmt);
                stream_ << "\n";
            } else if (!PrintTerminator(stmt, return_vars)) {
                PrintStmtBlock(stmt);
            }
        }
    } else {
        PrintStmtBlock(body);
    }
}

void IRPrinter::VisitFunctionBody(const StmtPtr& body)
{
    if (auto seq_stmts = As<SeqStmts>(body)) {
        if (seq_stmts->stmts_.empty()) {
            stream_ << GetIndent() << "pass";
        } else {
            for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
                // Convert yield to return in function context
                if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
                    stream_ << GetIndent() << "return";
                    if (!yield_stmt->value_.empty()) {
                        stream_ << " ";
                        for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
                            if (j > 0)
                                stream_ << ", ";
                            VisitExpr(yield_stmt->value_[j]);
                        }
                    }
                } else {
                    PrintStmtBlock(seq_stmts->stmts_[i]);
                }
                if (i < seq_stmts->stmts_.size() - 1) {
                    stream_ << "\n";
                }
            }
        }
    } else if (auto yield_stmt = As<YieldStmt>(body)) {
        stream_ << GetIndent() << "return";
        if (!yield_stmt->value_.empty()) {
            stream_ << " ";
            for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
                if (i > 0)
                    stream_ << ", ";
                VisitExpr(yield_stmt->value_[i]);
            }
        }
    } else {
        PrintStmtBlock(body);
    }
}

void IRPrinter::VisitFunction(const FunctionPtr& func)
{
    // Print decorator
    if (func->funcType_ == FunctionType::SIMT_VF || func->funcType_ == FunctionType::SIMT_CALLEE) {
        stream_ << GetIndent() << "@" << prefix_ << ".simt.function(type=" << prefix_ << ".FunctionType."
                << FunctionTypeToString(func->funcType_) << ")";
    } else {
        stream_ << GetIndent() << "@" << prefix_ << ".function";
        if (func->funcType_ != FunctionType::OPAQUE) {
            stream_ << "(type=" << prefix_ << ".FunctionType." << FunctionTypeToString(func->funcType_) << ")";
        }
    }
    stream_ << "\n";

    // Print function signature
    stream_ << GetIndent() << "def " << func->name_ << "(";

    // Print parameters with type annotations and direction wrappers
    for (size_t i = 0; i < func->params_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        const auto& var = func->params_[i];
        VisitExpr(var);
        stream_ << ": " << Print(var->GetType());
    }

    stream_ << ")";

    PrintFunctionReturnAnnotation(stream_, func->returnTypes_, [this](const TypePtr& type) { return Print(type); });

    stream_ << ":\n";

    // Print body - convert yield to return in function context
    indent_++;
    if (func->body_) {
        VisitFunctionBody(func->body_);
    }
    indent_--;
}

void IRPrinter::VisitProgram(const ProgramPtr& program)
{
    stream_ << "# ir.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";
    bool first = true;
    for (const auto& entry : program->functions_) {
        if (!first) {
            stream_ << "\n"; // Blank line between functions
        }
        first = false;
        VisitFunction(entry.second);
    }
}

std::string IRPrinter::PrintExprForType(const ExprPtr& expr)
{
    if (auto const_int = As<ConstInt>(expr)) {
        return std::to_string(const_int->value_);
    }
    if (auto var = As<Var>(expr)) {
        return var->name_;
    }
    IRPrinter temp_printer(prefix_);
    return temp_printer.Print(expr);
}

void IRPrinter::PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape)
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << PrintExprForType(shape[i]);
    }
}

void IRPrinter::PrintExprList(std::ostringstream& oss, const std::vector<ExprPtr>& exprs)
{
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (i > 0)
            oss << ", ";
        IRPrinter temp_printer(prefix_);
        oss << temp_printer.Print(exprs[i]);
    }
}

std::string IRPrinter::PrintMemRef(const MemRef& memref)
{
    std::ostringstream oss;
    oss << prefix_ << ".MemRef(";
    oss << prefix_ << ".MemorySpace." << MemorySpaceToString(memref.memorySpace_) << ", ";

    IRPrinter temp_printer(prefix_);
    oss << temp_printer.Print(memref.addr_);
    // Print size
    oss << ", " << memref.size_ << ")";
    return oss.str();
}

std::string IRPrinter::PrintTileView(const TileView& tile_view)
{
    std::ostringstream oss;
    oss << prefix_ << ".TileView(valid_shape=[";
    PrintExprList(oss, tile_view.validShape);
    oss << "], stride=[";
    PrintExprList(oss, tile_view.stride);
    oss << "], start_offset=";
    IRPrinter temp_printer(prefix_);
    oss << temp_printer.Print(tile_view.startOffset);
    oss << ")";
    return oss.str();
}

std::string IRPrinter::PrintHardwareInfo(const HardwareInfo& hw_info)
{
    std::ostringstream oss;
    oss << prefix_ << ".HardwareInfo(";
    oss << "blayout=" << prefix_ << ".TileLayout." << TileLayoutToPythonName(hw_info.blayout);
    oss << ", slayout=" << prefix_ << ".TileLayout." << TileLayoutToPythonName(hw_info.slayout);
    oss << ", fractal=" << hw_info.fractal;
    oss << ", pad=" << prefix_ << ".TilePad." << TilePadToPythonName(hw_info.pad);
    oss << ", compact=" << prefix_ << ".CompactMode." << CompactModeToPythonName(hw_info.compact);
    oss << ")";
    return oss.str();
}

std::string IRPrinter::PrintTensorView(const TensorView& tensor_view)
{
    std::ostringstream oss;
    oss << prefix_ << ".TensorView(";
    if (!tensor_view.validShape.empty()) {
        oss << "valid_shape=[";
        PrintExprList(oss, tensor_view.validShape);
        oss << "], ";
    }
    oss << "stride=[";
    PrintExprList(oss, tensor_view.stride);
    oss << "], layout=" << prefix_ << ".TensorLayout." << TensorLayoutToPythonName(tensor_view.layout);
    oss << ")";
    return oss.str();
}

std::string PythonPrint(const IRNodePtr& node, const std::string& prefix, bool concise)
{
    IRPrinter printer(prefix, concise);
    return printer.Print(node);
}

std::string PythonPrint(const TypePtr& type, const std::string& prefix)
{
    IRPrinter printer(prefix);
    return printer.Print(type);
}
} // namespace ir
} // namespace pypto
