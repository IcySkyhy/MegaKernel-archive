/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file io_text.h
 * \brief IR text I/O — dumper and loader following the declared IR Syntax grammar.
 *
 * IRTextDumper produces the canonical text representation declared in the
 * Doxygen "IR Syntax" blocks of each IR node header.  IRTextLoader parses that
 * text back into IR nodes.  Together they provide full round-trip serialisation.
 */
#ifndef PYPTO_IR_TRANSFORMS_IO_TEXT_H_
#define PYPTO_IR_TRANSFORMS_IO_TEXT_H_

#include <sstream>
#include <string>
#include <string_view>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"

#include "interface/tensor/symbolic_scalar.h"
#include "tilefwk/data_type.h"

/* ===========================================================================
 * IR Text Grammar (EBNF) — see io_text.md in framework/src/interface/ir/transforms/
 * =========================================================================== */

/* ===========================================================================
 * IR text grammar keywords and punctuation
 * =========================================================================== */
/* ---- Word keywords ---- */
#define IR_KW_PROGRAM "program"
#define IR_KW_FUNCTION "function"
#define IR_KW_INCAST "incast"
#define IR_KW_OUTCAST "outcast"
#define IR_KW_IF "if"
#define IR_KW_THEN "then"
#define IR_KW_ELSE "else"
#define IR_KW_FOR "for"
#define IR_KW_INRANGE "inrange"
#define IR_KW_ITER "iter"
#define IR_KW_WHILE "while"
#define IR_KW_YIELD "yield"
#define IR_KW_RETURN "return"
#define IR_KW_SECTION "section"
#define IR_KW_EVAL "eval"
#define IR_KW_BREAK "break"
#define IR_KW_CONTINUE "continue"
#define IR_KW_TUPLE "tuple"
#define IR_KW_GETITEM "getitem"
#define IR_KW_SCALAR_EXPR "scalar_expr"
#define IR_KW_UNKNOWN "unknown"
#define IR_KW_TENSOR "tensor"
#define IR_KW_TILE "tile"
#define IR_KW_PTR "ptr"
#define IR_KW_TOKEN "token"
#define IR_KW_NONE "none"
#define IR_KW_NULL "null"
#define IR_KW_TENSOR_VIEW "tensor_view"
#define IR_KW_TILE_VIEW "tile_view"
#define IR_KW_MEMREF "memref"
#define IR_KW_MEMREF_TYPE "memref_type"
#define IR_KWV0_LOGICAL_TENSOR "v0_logical_tensor"
#define IR_KW_HW_INFO "hw_info"
#define IR_KW_TRUE "true"
#define IR_KW_FALSE "false"
#define IR_KW_ENTRY "entry"
#define IR_KW_TYPE "type"
#define IR_KW_DIM "x"

// ---- Scalar binary operator names ----
#define IR_KW_SCALAR_BOP_ADD "add"
#define IR_KW_SCALAR_BOP_SUB "sub"
#define IR_KW_SCALAR_BOP_MUL "mul"
#define IR_KW_SCALAR_BOP_DIV "div"
#define IR_KW_SCALAR_BOP_MOD "mod"
#define IR_KW_SCALAR_BOP_FDIV "fdiv"
#define IR_KW_SCALAR_BOP_MIN "min"
#define IR_KW_SCALAR_BOP_MAX "max"
#define IR_KW_SCALAR_BOP_POW "pow"
#define IR_KW_SCALAR_BOP_EQ "eq"
#define IR_KW_SCALAR_BOP_NE "ne"
#define IR_KW_SCALAR_BOP_LT "lt"
#define IR_KW_SCALAR_BOP_LE "le"
#define IR_KW_SCALAR_BOP_GT "gt"
#define IR_KW_SCALAR_BOP_GE "ge"
#define IR_KW_SCALAR_BOP_LAND "land"
#define IR_KW_SCALAR_BOP_LOR "lor"
#define IR_KW_SCALAR_BOP_LXOR "lxor"
#define IR_KW_SCALAR_BOP_AND "and"
#define IR_KW_SCALAR_BOP_OR "or"
#define IR_KW_SCALAR_BOP_XOR "xor"
#define IR_KW_SCALAR_BOP_SHL "shl"
#define IR_KW_SCALAR_BOP_SHR "shr"

// ---- Scalar unary operator names ----
#define IR_KW_SCALAR_UOP_ABS "abs"
#define IR_KW_SCALAR_UOP_NEG "neg"
#define IR_KW_SCALAR_UOP_NOT "not"
#define IR_KW_SCALAR_UOP_INV "inv"
#define IR_KW_SCALAR_UOP_CAST "cast"

// ---- Symbolic scalar (scalarv0) opcode keywords ----
#define IR_KWV0_SCALAR_UOP_POS "v0pos"
#define IR_KWV0_SCALAR_UOP_NEG "v0neg"
#define IR_KWV0_SCALAR_UOP_NOT "v0not"

#define IR_KWV0_SCALAR_BOP_ADD "v0add"
#define IR_KWV0_SCALAR_BOP_SUB "v0sub"
#define IR_KWV0_SCALAR_BOP_MUL "v0mul"
#define IR_KWV0_SCALAR_BOP_DIV "v0div"
#define IR_KWV0_SCALAR_BOP_MOD "v0mod"
#define IR_KWV0_SCALAR_BOP_EQ "v0eq"
#define IR_KWV0_SCALAR_BOP_NE "v0ne"
#define IR_KWV0_SCALAR_BOP_LT "v0lt"
#define IR_KWV0_SCALAR_BOP_LE "v0le"
#define IR_KWV0_SCALAR_BOP_GT "v0gt"
#define IR_KWV0_SCALAR_BOP_GE "v0ge"

#define IR_KWV0_SCALAR_MOP_CALL "v0call"
#define IR_KWV0_SCALAR_MOP_MIN "v0min"
#define IR_KWV0_SCALAR_MOP_MAX "v0max"
#define IR_KWV0_SCALAR_MOP_AND "v0and"
#define IR_KWV0_SCALAR_MOP_OR "v0or"

// ---- Punctuation ----
#define IR_PUN_VARNAME "%"
#define IR_PUN_MEMREF "@"
#define IR_PUN_LBRACE "{"
#define IR_PUN_RBRACE "}"
#define IR_PUN_LPAREN "("
#define IR_PUN_RPAREN ")"
#define IR_PUN_LT "<"
#define IR_PUN_GT ">"
#define IR_PUN_COMMA ","
#define IR_PUN_EQ "="
#define IR_PUN_SEMI ";"
#define IR_PUN_ATTRNAME "#"
#define IR_PUN_LBRACKET "["
#define IR_PUN_RBRACKET "]"
#define IR_PUN_OPMAGIC "!"

namespace pypto {
namespace ir {

enum class IRTextLexerTokenKind {
    // Word keywords
    KwProgram,
    KwFunction,
    KwIncast,
    KwOutcast,
    KwIf,
    KwThen,
    KwElse,
    KwFor,
    KwInrange,
    KwIter,
    KwWhile,
    KwYield,
    KwReturn,
    KwSection,
    KwEval,
    KwBreak,
    KwContinue,
    KwTuple,
    KwGetItem,
    KwScalarExpr,
    KwUnknown,
    KwTensor,
    KwTile,
    KwPtr,
    KwToken,
    KwNone,
    KwTensorView,
    KwTileView,
    KwMemref,
    KwMemrefType,
    KwV0LogicalTensor,
    KwHwInfo,
    KwTrue,
    KwFalse,
    KwEntry,
    KwType,
    KwDim,
    KwNull,
    // Scalar binary op keywords
    KwBOpAdd,
    KwBOpSub,
    KwBOpMul,
    KwBOpDiv,
    KwBOpMod,
    KwBOpFdiv,
    KwBOpMin,
    KwBOpMax,
    KwBOpPow,
    KwBOpEq,
    KwBOpNe,
    KwBOpLt,
    KwBOpLe,
    KwBOpGt,
    KwBOpGe,
    KwBOpLand,
    KwBOpLor,
    KwBOpLxor,
    KwBOpAnd,
    KwBOpOr,
    KwBOpXor,
    KwBOpShl,
    KwBOpShr,
    // Scalar unary op keywords
    KwUOpAbs,
    KwUOpNeg,
    KwUOpNot,
    KwUOpInv,
    KwUOpCast,
    // Symbolic scalar (scalarv0) op keywords
    KwV0ScalarUOpPos,
    KwV0ScalarUOpNeg,
    KwV0ScalarUOpNot,
    KwV0ScalarBOpAdd,
    KwV0ScalarBOpSub,
    KwV0ScalarBOpMul,
    KwV0ScalarBOpDiv,
    KwV0ScalarBOpMod,
    KwV0ScalarBOpEq,
    KwV0ScalarBOpNe,
    KwV0ScalarBOpLt,
    KwV0ScalarBOpLe,
    KwV0ScalarBOpGt,
    KwV0ScalarBOpGe,
    KwV0ScalarMOpCall,
    KwV0ScalarMOpMin,
    KwV0ScalarMOpMax,
    KwV0ScalarMOpAnd,
    KwV0ScalarMOpOr,
    KwV0ScalarBegin = KwV0ScalarUOpPos,
    KwV0ScalarEnd = KwV0ScalarMOpOr,
    // Punctuation
    PunVarName,
    PunAttrName,
    PunLBrace,
    PunRBrace,
    PunLParen,
    PunRParen,
    PunLt,
    PunGt,
    PunComma,
    PunEq,
    PunSemi,
    PunLBracket,
    PunRBracket,

    // Data Type
    KwTypeBool,
    KwTypeInt8,
    KwTypeInt16,
    KwTypeInt32,
    KwTypeInt64,
    KwTypeUint8,
    KwTypeUint16,
    KwTypeUint32,
    KwTypeUint64,
    KwTypeFp16,
    KwTypeFp32,
    KwTypeFp64,
    KwTypeBf16,
    KwTypeFp8e4m3fn,
    KwTypeFp8e5m2,
    KwTypeHf4,
    KwTypeHf8,
    KwTypeUnknown,
    KwTypeBegin = KwTypeBool,
    KwTypeEnd = KwTypeUnknown,

    // Valuable
    TokIdent,
    TokV0OpMagic,
    TokInt,
    TokFloat,
    TokVarName,
    TokAttrName,
    Invalid,
};

static inline const npu::tile_fwk::BiMap<IRTextLexerTokenKind>& IRTextGetLexerTokenDict()
{
    static npu::tile_fwk::BiMap<IRTextLexerTokenKind> dict{{
        // Keywords
        {IRTextLexerTokenKind::KwProgram, IR_KW_PROGRAM},
        {IRTextLexerTokenKind::KwFunction, IR_KW_FUNCTION},
        {IRTextLexerTokenKind::KwIncast, IR_KW_INCAST},
        {IRTextLexerTokenKind::KwOutcast, IR_KW_OUTCAST},
        {IRTextLexerTokenKind::KwIf, IR_KW_IF},
        {IRTextLexerTokenKind::KwThen, IR_KW_THEN},
        {IRTextLexerTokenKind::KwElse, IR_KW_ELSE},
        {IRTextLexerTokenKind::KwFor, IR_KW_FOR},
        {IRTextLexerTokenKind::KwInrange, IR_KW_INRANGE},
        {IRTextLexerTokenKind::KwIter, IR_KW_ITER},
        {IRTextLexerTokenKind::KwWhile, IR_KW_WHILE},
        {IRTextLexerTokenKind::KwYield, IR_KW_YIELD},
        {IRTextLexerTokenKind::KwReturn, IR_KW_RETURN},
        {IRTextLexerTokenKind::KwSection, IR_KW_SECTION},
        {IRTextLexerTokenKind::KwEval, IR_KW_EVAL},
        {IRTextLexerTokenKind::KwBreak, IR_KW_BREAK},
        {IRTextLexerTokenKind::KwContinue, IR_KW_CONTINUE},
        {IRTextLexerTokenKind::KwTuple, IR_KW_TUPLE},
        {IRTextLexerTokenKind::KwGetItem, IR_KW_GETITEM},
        {IRTextLexerTokenKind::KwScalarExpr, IR_KW_SCALAR_EXPR},
        {IRTextLexerTokenKind::KwUnknown, IR_KW_UNKNOWN},
        {IRTextLexerTokenKind::KwTensor, IR_KW_TENSOR},
        {IRTextLexerTokenKind::KwTile, IR_KW_TILE},
        {IRTextLexerTokenKind::KwPtr, IR_KW_PTR},
        {IRTextLexerTokenKind::KwToken, IR_KW_TOKEN},
        {IRTextLexerTokenKind::KwNone, IR_KW_NONE},
        {IRTextLexerTokenKind::KwTensorView, IR_KW_TENSOR_VIEW},
        {IRTextLexerTokenKind::KwTileView, IR_KW_TILE_VIEW},
        {IRTextLexerTokenKind::KwMemref, IR_KW_MEMREF},
        {IRTextLexerTokenKind::KwMemrefType, IR_KW_MEMREF_TYPE},
        {IRTextLexerTokenKind::KwV0LogicalTensor, IR_KWV0_LOGICAL_TENSOR},
        {IRTextLexerTokenKind::KwHwInfo, IR_KW_HW_INFO},
        {IRTextLexerTokenKind::KwTrue, IR_KW_TRUE},
        {IRTextLexerTokenKind::KwFalse, IR_KW_FALSE},
        {IRTextLexerTokenKind::KwEntry, IR_KW_ENTRY},
        {IRTextLexerTokenKind::KwType, IR_KW_TYPE},
        {IRTextLexerTokenKind::KwDim, IR_KW_DIM},
        {IRTextLexerTokenKind::KwNull, IR_KW_NULL},
        // Scalar binary ops
        {IRTextLexerTokenKind::KwBOpAdd, IR_KW_SCALAR_BOP_ADD},
        {IRTextLexerTokenKind::KwBOpSub, IR_KW_SCALAR_BOP_SUB},
        {IRTextLexerTokenKind::KwBOpMul, IR_KW_SCALAR_BOP_MUL},
        {IRTextLexerTokenKind::KwBOpDiv, IR_KW_SCALAR_BOP_DIV},
        {IRTextLexerTokenKind::KwBOpMod, IR_KW_SCALAR_BOP_MOD},
        {IRTextLexerTokenKind::KwBOpFdiv, IR_KW_SCALAR_BOP_FDIV},
        {IRTextLexerTokenKind::KwBOpMin, IR_KW_SCALAR_BOP_MIN},
        {IRTextLexerTokenKind::KwBOpMax, IR_KW_SCALAR_BOP_MAX},
        {IRTextLexerTokenKind::KwBOpPow, IR_KW_SCALAR_BOP_POW},
        {IRTextLexerTokenKind::KwBOpEq, IR_KW_SCALAR_BOP_EQ},
        {IRTextLexerTokenKind::KwBOpNe, IR_KW_SCALAR_BOP_NE},
        {IRTextLexerTokenKind::KwBOpLt, IR_KW_SCALAR_BOP_LT},
        {IRTextLexerTokenKind::KwBOpLe, IR_KW_SCALAR_BOP_LE},
        {IRTextLexerTokenKind::KwBOpGt, IR_KW_SCALAR_BOP_GT},
        {IRTextLexerTokenKind::KwBOpGe, IR_KW_SCALAR_BOP_GE},
        {IRTextLexerTokenKind::KwBOpLand, IR_KW_SCALAR_BOP_LAND},
        {IRTextLexerTokenKind::KwBOpLor, IR_KW_SCALAR_BOP_LOR},
        {IRTextLexerTokenKind::KwBOpLxor, IR_KW_SCALAR_BOP_LXOR},
        {IRTextLexerTokenKind::KwBOpAnd, IR_KW_SCALAR_BOP_AND},
        {IRTextLexerTokenKind::KwBOpOr, IR_KW_SCALAR_BOP_OR},
        {IRTextLexerTokenKind::KwBOpXor, IR_KW_SCALAR_BOP_XOR},
        {IRTextLexerTokenKind::KwBOpShl, IR_KW_SCALAR_BOP_SHL},
        {IRTextLexerTokenKind::KwBOpShr, IR_KW_SCALAR_BOP_SHR},
        // Scalar unary ops
        {IRTextLexerTokenKind::KwUOpAbs, IR_KW_SCALAR_UOP_ABS},
        {IRTextLexerTokenKind::KwUOpNeg, IR_KW_SCALAR_UOP_NEG},
        {IRTextLexerTokenKind::KwUOpNot, IR_KW_SCALAR_UOP_NOT},
        {IRTextLexerTokenKind::KwUOpInv, IR_KW_SCALAR_UOP_INV},
        {IRTextLexerTokenKind::KwUOpCast, IR_KW_SCALAR_UOP_CAST},
        // Symbolic scalar (scalarv0) op keywords
        {IRTextLexerTokenKind::KwV0ScalarUOpPos, IR_KWV0_SCALAR_UOP_POS},
        {IRTextLexerTokenKind::KwV0ScalarUOpNeg, IR_KWV0_SCALAR_UOP_NEG},
        {IRTextLexerTokenKind::KwV0ScalarUOpNot, IR_KWV0_SCALAR_UOP_NOT},
        {IRTextLexerTokenKind::KwV0ScalarBOpAdd, IR_KWV0_SCALAR_BOP_ADD},
        {IRTextLexerTokenKind::KwV0ScalarBOpSub, IR_KWV0_SCALAR_BOP_SUB},
        {IRTextLexerTokenKind::KwV0ScalarBOpMul, IR_KWV0_SCALAR_BOP_MUL},
        {IRTextLexerTokenKind::KwV0ScalarBOpDiv, IR_KWV0_SCALAR_BOP_DIV},
        {IRTextLexerTokenKind::KwV0ScalarBOpMod, IR_KWV0_SCALAR_BOP_MOD},
        {IRTextLexerTokenKind::KwV0ScalarBOpEq, IR_KWV0_SCALAR_BOP_EQ},
        {IRTextLexerTokenKind::KwV0ScalarBOpNe, IR_KWV0_SCALAR_BOP_NE},
        {IRTextLexerTokenKind::KwV0ScalarBOpLt, IR_KWV0_SCALAR_BOP_LT},
        {IRTextLexerTokenKind::KwV0ScalarBOpLe, IR_KWV0_SCALAR_BOP_LE},
        {IRTextLexerTokenKind::KwV0ScalarBOpGt, IR_KWV0_SCALAR_BOP_GT},
        {IRTextLexerTokenKind::KwV0ScalarBOpGe, IR_KWV0_SCALAR_BOP_GE},
        {IRTextLexerTokenKind::KwV0ScalarMOpCall, IR_KWV0_SCALAR_MOP_CALL},
        {IRTextLexerTokenKind::KwV0ScalarMOpMin, IR_KWV0_SCALAR_MOP_MIN},
        {IRTextLexerTokenKind::KwV0ScalarMOpMax, IR_KWV0_SCALAR_MOP_MAX},
        {IRTextLexerTokenKind::KwV0ScalarMOpAnd, IR_KWV0_SCALAR_MOP_AND},
        {IRTextLexerTokenKind::KwV0ScalarMOpOr, IR_KWV0_SCALAR_MOP_OR},
        // Punctuation (VarName/AttrName handled as whole tokens, not here)
        {IRTextLexerTokenKind::PunLBrace, IR_PUN_LBRACE},
        {IRTextLexerTokenKind::PunRBrace, IR_PUN_RBRACE},
        {IRTextLexerTokenKind::PunLParen, IR_PUN_LPAREN},
        {IRTextLexerTokenKind::PunRParen, IR_PUN_RPAREN},
        {IRTextLexerTokenKind::PunLt, IR_PUN_LT},
        {IRTextLexerTokenKind::PunGt, IR_PUN_GT},
        {IRTextLexerTokenKind::PunComma, IR_PUN_COMMA},
        {IRTextLexerTokenKind::PunEq, IR_PUN_EQ},
        {IRTextLexerTokenKind::PunSemi, IR_PUN_SEMI},
        {IRTextLexerTokenKind::PunLBracket, IR_PUN_LBRACKET},
        {IRTextLexerTokenKind::PunRBracket, IR_PUN_RBRACKET},

        {IRTextLexerTokenKind::KwTypeBool, IR_KW_TYPE_BOOL},
        {IRTextLexerTokenKind::KwTypeInt8, IR_KW_TYPE_INT8},
        {IRTextLexerTokenKind::KwTypeInt16, IR_KW_TYPE_INT16},
        {IRTextLexerTokenKind::KwTypeInt32, IR_KW_TYPE_INT32},
        {IRTextLexerTokenKind::KwTypeInt64, IR_KW_TYPE_INT64},
        {IRTextLexerTokenKind::KwTypeUint8, IR_KW_TYPE_UINT8},
        {IRTextLexerTokenKind::KwTypeUint16, IR_KW_TYPE_UINT16},
        {IRTextLexerTokenKind::KwTypeUint32, IR_KW_TYPE_UINT32},
        {IRTextLexerTokenKind::KwTypeUint64, IR_KW_TYPE_UINT64},
        {IRTextLexerTokenKind::KwTypeFp16, IR_KW_TYPE_FP16},
        {IRTextLexerTokenKind::KwTypeFp32, IR_KW_TYPE_FP32},
        {IRTextLexerTokenKind::KwTypeFp64, IR_KW_TYPE_FP64},
        {IRTextLexerTokenKind::KwTypeBf16, IR_KW_TYPE_BF16},
        {IRTextLexerTokenKind::KwTypeFp8e4m3fn, IR_KW_TYPE_FP8E4M3FN},
        {IRTextLexerTokenKind::KwTypeFp8e5m2, IR_KW_TYPE_FP8E5M2},
        {IRTextLexerTokenKind::KwTypeHf4, IR_KW_TYPE_HF4},
        {IRTextLexerTokenKind::KwTypeHf8, IR_KW_TYPE_HF8},
        {IRTextLexerTokenKind::KwTypeUnknown, IR_KW_TYPE_UNKNOWN},

        // Valuable tokens (angle-bracketed to distinguish from keywords/punctuation)
        {IRTextLexerTokenKind::TokIdent, "<identifier>"},
        {IRTextLexerTokenKind::TokV0OpMagic, "<op_magic>"},
        {IRTextLexerTokenKind::TokInt, "<integer>"},
        {IRTextLexerTokenKind::TokFloat, "<float>"},
        {IRTextLexerTokenKind::TokVarName, "<var_name>"},
        {IRTextLexerTokenKind::TokAttrName, "<attr_name>"},
        {IRTextLexerTokenKind::Invalid, "<invalid>"},
    }};
    return dict;
}

// ===========================================================================
// Dumper
// ===========================================================================
class IRTextDumper : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;
    using IRVisitor::VisitType_;

public:
    IRTextDumper() = default;
    ~IRTextDumper() override = default;

    static std::string Dump(const IRNodePtr& node);
    static std::string DumpType(const TypePtr& type);

protected:
    void VisitExpr_(const VarPtr& op) override;
    void VisitExpr_(const MemRefPtr& op) override;
    void VisitExpr_(const ConstIntPtr& op) override;
    void VisitExpr_(const ConstFloatPtr& op) override;
    void VisitExpr_(const ConstBoolPtr& op) override;
    void VisitExpr_(const CallPtr& op) override;
    void VisitExpr_(const MakeTuplePtr& op) override;
    void VisitExpr_(const GetItemExprPtr& op) override;
    void VisitExpr_(const ScalarExprPtr& op) override;

    void VisitBinaryExpr_(const BinaryExprPtr& op) override;
    void VisitUnaryExpr_(const UnaryExprPtr& op) override;

    void VisitType_(const UnknownTypePtr& op) override;
    void VisitType_(const ScalarTypePtr& op) override;
    void VisitType_(const TensorTypePtr& op) override;
    void VisitType_(const TileTypePtr& op) override;
    void VisitType_(const TupleTypePtr& op) override;
    void VisitType_(const MemRefTypePtr& op) override;
    void VisitType_(const PtrTypePtr& op) override;
    void VisitType_(const TokenTypePtr& op) override;
    void VisitType_(const NoneTypePtr& op) override;
    void VisitType_(const LogicalTensorTypePtr& op) override;

    void VisitStmt_(const AssignStmtPtr& op) override;
    void VisitStmt_(const SeqStmtsPtr& op) override;
    void VisitStmt_(const IfStmtPtr& op) override;
    void VisitStmt_(const YieldStmtPtr& op) override;
    void VisitStmt_(const ReturnStmtPtr& op) override;
    void VisitStmt_(const ForStmtPtr& op) override;
    void VisitStmt_(const WhileStmtPtr& op) override;
    void VisitStmt_(const SectionStmtPtr& op) override;
    void VisitStmt_(const EvalStmtPtr& op) override;
    void VisitStmt_(const BreakStmtPtr& op) override;
    void VisitStmt_(const ContinueStmtPtr& op) override;
    void VisitStmt_(const ScalarOpStmtPtr& op) override;
    void VisitStmt_(const TensorOpStmtPtr& op) override;
    void VisitStmt_(const StmtPtr& op) override;

    void VisitFunction(const FunctionPtr& func) override;
    void VisitProgram(const ProgramPtr& program) override;

private:
    std::ostringstream stream_;
    int indent_ = 0;

    std::string GetIndent() const { return std::string(static_cast<size_t>(indent_ * 4), ' '); }

    void PrintVarDef(const VarPtr& var);
    void PrintVarRef(const VarPtr& var);
    void PrintVarDefList(const std::vector<VarPtr>& varList, const std::vector<VarPtr>& tokenList = {});
    void PrintIterArgs(const std::vector<IterArgPtr>& iterArgs);

    void PrintIntListAttr(const std::string& key, const std::vector<int64_t>& vals);
    void PrintSymbolicScalarListAttr(const std::string& key, const std::vector<npu::tile_fwk::SymbolicScalar>& vals);

    void PrintType(const TypePtr& type);
    void PrintShape(const std::vector<ExprPtr>& shape);
    void PrintExprList(const std::vector<ExprPtr>& exprs);
    void PrintTokenList(const std::vector<VarPtr>& tokenList);
    void PrintAttrValue(const std::string& key, const std::any& value);
    void PrintAttr(const std::string& key, const std::any& value);
    void PrintDataArray(const std::vector<int64_t>& vals);
    template <typename T>
    void PrintAttr(const std::string& key, const T& value)
    {
        stream_ << " " << IR_PUN_ATTRNAME << key << IR_PUN_LPAREN << value << IR_PUN_RPAREN;
    }

    void PrintBody(const StmtPtr& body);

    static const char* BinaryOpName(const BinaryExprPtr& op);
    static const char* UnaryOpName(const UnaryExprPtr& op);
};

// ===========================================================================
// Loader
// ===========================================================================
class IRTextLoader {
public:
    IRTextLoader() = default;

    ProgramPtr LoadProgram(const std::string& text, std::string& error);
    FunctionPtr LoadFunction(const std::string& text, std::string& error);
    StmtPtr LoadStmt(const std::string& text, std::string& error);
    ExprPtr LoadExpr(const std::string& text, std::string& error);
    TypePtr LoadType(const std::string& text, std::string& error);
};

// ===========================================================================
// Public free functions
// ===========================================================================
std::string TextDump(const IRNodePtr& node);
std::string TextDumpType(const TypePtr& type);

ProgramPtr TextLoadProgram(const std::string& text, std::string& error);
FunctionPtr TextLoadFunction(const std::string& text, std::string& error);
StmtPtr TextLoadStmt(const std::string& text, std::string& error);
ExprPtr TextLoadExpr(const std::string& text, std::string& error);
TypePtr TextLoadType(const std::string& text, std::string& error);

// ===========================================================================
// Operator BiMap dictionaries
// ===========================================================================
static inline const npu::tile_fwk::BiMap<ObjectKind>& GetBinaryOpDict()
{
    static npu::tile_fwk::BiMap<ObjectKind> dict{{
        {ObjectKind::Add, IR_KW_SCALAR_BOP_ADD},
        {ObjectKind::Sub, IR_KW_SCALAR_BOP_SUB},
        {ObjectKind::Mul, IR_KW_SCALAR_BOP_MUL},
        {ObjectKind::FloorDiv, IR_KW_SCALAR_BOP_DIV},
        {ObjectKind::FloorMod, IR_KW_SCALAR_BOP_MOD},
        {ObjectKind::FloatDiv, IR_KW_SCALAR_BOP_FDIV},
        {ObjectKind::Min, IR_KW_SCALAR_BOP_MIN},
        {ObjectKind::Max, IR_KW_SCALAR_BOP_MAX},
        {ObjectKind::Pow, IR_KW_SCALAR_BOP_POW},
        {ObjectKind::Eq, IR_KW_SCALAR_BOP_EQ},
        {ObjectKind::Ne, IR_KW_SCALAR_BOP_NE},
        {ObjectKind::Lt, IR_KW_SCALAR_BOP_LT},
        {ObjectKind::Le, IR_KW_SCALAR_BOP_LE},
        {ObjectKind::Gt, IR_KW_SCALAR_BOP_GT},
        {ObjectKind::Ge, IR_KW_SCALAR_BOP_GE},
        {ObjectKind::And, IR_KW_SCALAR_BOP_LAND},
        {ObjectKind::Or, IR_KW_SCALAR_BOP_LOR},
        {ObjectKind::Xor, IR_KW_SCALAR_BOP_LXOR},
        {ObjectKind::BitAnd, IR_KW_SCALAR_BOP_AND},
        {ObjectKind::BitOr, IR_KW_SCALAR_BOP_OR},
        {ObjectKind::BitXor, IR_KW_SCALAR_BOP_XOR},
        {ObjectKind::BitShiftLeft, IR_KW_SCALAR_BOP_SHL},
        {ObjectKind::BitShiftRight, IR_KW_SCALAR_BOP_SHR},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<ObjectKind>& GetUnaryOpDict()
{
    static npu::tile_fwk::BiMap<ObjectKind> dict{{
        {ObjectKind::Abs, IR_KW_SCALAR_UOP_ABS},
        {ObjectKind::Neg, IR_KW_SCALAR_UOP_NEG},
        {ObjectKind::Not, IR_KW_SCALAR_UOP_NOT},
        {ObjectKind::BitNot, IR_KW_SCALAR_UOP_INV},
        {ObjectKind::Cast, IR_KW_SCALAR_UOP_CAST},
    }};
    return dict;
}

// Bidirectional map between tile_fwk::DataType (LogicalTensor dtype) and the IR
// type keyword text (IR_KW_TYPE_*). Types without a distinct IR keyword text are
// omitted: DT_INT4, DT_FP8, DT_HF4, DT_FP4_E2M1, DT_FP4_E1M2.
static inline const npu::tile_fwk::BiMap<npu::tile_fwk::DataType>& GetTileFwkDataTypeDict()
{
    using npu::tile_fwk::DataType;
    static npu::tile_fwk::BiMap<DataType> dict{{
        {DataType::DT_BOOL, IR_KW_TYPE_BOOL},
        {DataType::DT_INT8, IR_KW_TYPE_INT8},
        {DataType::DT_INT16, IR_KW_TYPE_INT16},
        {DataType::DT_INT32, IR_KW_TYPE_INT32},
        {DataType::DT_INT64, IR_KW_TYPE_INT64},
        {DataType::DT_UINT8, IR_KW_TYPE_UINT8},
        {DataType::DT_UINT16, IR_KW_TYPE_UINT16},
        {DataType::DT_UINT32, IR_KW_TYPE_UINT32},
        {DataType::DT_UINT64, IR_KW_TYPE_UINT64},
        {DataType::DT_FP16, IR_KW_TYPE_FP16},
        {DataType::DT_FP32, IR_KW_TYPE_FP32},
        {DataType::DT_DOUBLE, IR_KW_TYPE_FP64},
        {DataType::DT_BF16, IR_KW_TYPE_BF16},
        {DataType::DT_FP8E4M3, IR_KW_TYPE_FP8E4M3FN},
        {DataType::DT_FP8E5M2, IR_KW_TYPE_FP8E5M2},
        {DataType::DT_FP8E8M0, IR_KW_TYPE_FP8E8M0},
        {DataType::DT_FP4_E2M1X2, IR_KW_TYPE_FP4E2M1},
        {DataType::DT_FP4_E1M2X2, IR_KW_TYPE_FP4E1M2},
        {DataType::DT_HF8, IR_KW_TYPE_HF8},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<npu::tile_fwk::SymbolicOpcode>& GetSymbolicOpcodeDict()
{
    using npu::tile_fwk::SymbolicOpcode;
    static npu::tile_fwk::BiMap<SymbolicOpcode> dict{{
        // Unary ops
        {SymbolicOpcode::T_UOP_POS, IR_KWV0_SCALAR_UOP_POS},
        {SymbolicOpcode::T_UOP_NEG, IR_KWV0_SCALAR_UOP_NEG},
        {SymbolicOpcode::T_UOP_NOT, IR_KWV0_SCALAR_UOP_NOT},
        // Binary ops
        {SymbolicOpcode::T_BOP_ADD, IR_KWV0_SCALAR_BOP_ADD},
        {SymbolicOpcode::T_BOP_SUB, IR_KWV0_SCALAR_BOP_SUB},
        {SymbolicOpcode::T_BOP_MUL, IR_KWV0_SCALAR_BOP_MUL},
        {SymbolicOpcode::T_BOP_DIV, IR_KWV0_SCALAR_BOP_DIV},
        {SymbolicOpcode::T_BOP_MOD, IR_KWV0_SCALAR_BOP_MOD},
        {SymbolicOpcode::T_BOP_EQ, IR_KWV0_SCALAR_BOP_EQ},
        {SymbolicOpcode::T_BOP_NE, IR_KWV0_SCALAR_BOP_NE},
        {SymbolicOpcode::T_BOP_LT, IR_KWV0_SCALAR_BOP_LT},
        {SymbolicOpcode::T_BOP_LE, IR_KWV0_SCALAR_BOP_LE},
        {SymbolicOpcode::T_BOP_GT, IR_KWV0_SCALAR_BOP_GT},
        {SymbolicOpcode::T_BOP_GE, IR_KWV0_SCALAR_BOP_GE},
        // Multiple ops
        {SymbolicOpcode::T_MOP_CALL, IR_KWV0_SCALAR_MOP_CALL},
        {SymbolicOpcode::T_MOP_MIN, IR_KWV0_SCALAR_MOP_MIN},
        {SymbolicOpcode::T_MOP_MAX, IR_KWV0_SCALAR_MOP_MAX},
        {SymbolicOpcode::T_MOP_AND, IR_KWV0_SCALAR_MOP_AND},
        {SymbolicOpcode::T_MOP_OR, IR_KWV0_SCALAR_MOP_OR},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TensorLayout>& GetTensorLayoutDict()
{
    static npu::tile_fwk::BiMap<TensorLayout> dict{{
        {TensorLayout::ND, "ND"},
        {TensorLayout::DN, "DN"},
        {TensorLayout::NZ, "NZ"},
        {TensorLayout::ZN, "ZN"},
        {TensorLayout::NN, "NN"},
        {TensorLayout::ZZ, "ZZ"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TileLayout>& GetTileLayoutDict()
{
    static npu::tile_fwk::BiMap<TileLayout> dict{{
        {TileLayout::none_box, "none_box"},
        {TileLayout::row_major, "row_major"},
        {TileLayout::col_major, "col_major"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TilePad>& GetTilePadDict()
{
    static npu::tile_fwk::BiMap<TilePad> dict{{
        {TilePad::null, "null"},
        {TilePad::zero, "zero"},
        {TilePad::max, "max"},
        {TilePad::min, "min"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<CompactMode>& GetCompactModeDict()
{
    static npu::tile_fwk::BiMap<CompactMode> dict{{
        {CompactMode::null, "null"},
        {CompactMode::normal, "normal"},
        {CompactMode::row_plus_one, "row_plus_one"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<MemorySpace>& GetMemorySpaceDict()
{
    static npu::tile_fwk::BiMap<MemorySpace> dict{{
        {MemorySpace::DDR, "ddr"},
        {MemorySpace::Vec, "vec"},
        {MemorySpace::Mat, "mat"},
        {MemorySpace::Left, "left"},
        {MemorySpace::Right, "right"},
        {MemorySpace::Scaling, "scaling"},
        {MemorySpace::Acc, "acc"},
        {MemorySpace::Bias, "bias"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<SectionKind>& GetSectionKindDict()
{
    static npu::tile_fwk::BiMap<SectionKind> dict{{
        {SectionKind::Vector, "vector"},
        {SectionKind::Cube, "cube"},
        {SectionKind::VF, "vf"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<FunctionType>& GetFunctionTypeDict()
{
    static npu::tile_fwk::BiMap<FunctionType> dict{{
        {FunctionType::OPAQUE, "opaque"},
        {FunctionType::ORCHESTRATION, "orchestration"},
        {FunctionType::IN_CORE, "in_core"},
        {FunctionType::HELPER, "helper"},
    }};
    return dict;
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_IO_TEXT_H_
