/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file loader_text.cpp
 * \brief IR text loader (parser) implementation.
 *
 * Recursive-descent parser that reads the text grammar produced by
 * IRTextDumper and reconstructs IR nodes.
 */

#include "ir/transforms/io_text.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/span.h"
#include "ir/stmt.h"

#include "interface/tensor/irbuilder.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/function/function.h"

namespace pypto {
namespace ir {

// ===========================================================================
// Tokenizer (regex-based)
// ===========================================================================
namespace {

struct IRTextLexerToken {
    IRTextLexerTokenKind kind = IRTextLexerTokenKind::Invalid;
    std::string text; // ident text, punct char, or string content (unquoted)
    int64_t intVal = 0;
    double floatVal = 0.0;
    size_t begin = 0; // byte offset of token start in source
    size_t end = 0;   // byte offset of token end in source
};

static bool IsKw(IRTextLexerTokenKind k)
{
    return k >= IRTextLexerTokenKind::KwProgram && k <= IRTextLexerTokenKind::KwUOpCast;
}

static bool IsDTypeKw(IRTextLexerTokenKind k)
{
    return k >= IRTextLexerTokenKind::KwTypeBegin && k <= IRTextLexerTokenKind::KwTypeEnd;
}

static bool IsIdentOrKw(IRTextLexerTokenKind k)
{
    return k == IRTextLexerTokenKind::TokIdent || IsKw(k) || IsDTypeKw(k);
}

/// Regex-based lexer.  Patterns are tried in priority order at the current
/// position using match_continuous, so the longest correct token always wins.
class IRTextLexer {
public:
    explicit IRTextLexer(const std::string& src) : src_(src)
    {
        using R = std::pair<IRTextLexerTokenKind, std::regex>;
        rules_ = {
            R{IRTextLexerTokenKind::TokVarName,
              std::regex(R"(%([A-Za-z_][A-Za-z0-9_.]*|[0-9]+)(@([A-Za-z_][A-Za-z0-9_.]*|[0-9]+))?)")},
            R{IRTextLexerTokenKind::TokAttrName, std::regex(R"(#[A-Za-z_][A-Za-z0-9_.]*)")},
            R{IRTextLexerTokenKind::TokV0OpMagic, std::regex(R"(!\d+)")},
            R{IRTextLexerTokenKind::TokFloat, std::regex(R"(-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][-+]?\d+)")},
            R{IRTextLexerTokenKind::TokInt, std::regex(R"(-?\d+)")},
            R{IRTextLexerTokenKind::TokIdent, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)")},
        };
        wsRe_ = std::regex(R"(\s+)");
    }

    std::optional<std::vector<IRTextLexerToken>> Tokenize()
    {
        std::vector<IRTextLexerToken> toks;
        auto it = src_.cbegin();
        const auto end = src_.cend();
        while (it != end) {
            // 1. Skip whitespace.
            std::smatch ws;
            if (std::regex_search(it, end, ws, wsRe_, std::regex_constants::match_continuous)) {
                it = ws[0].second;
                continue;
            }
            // 2. Try each token pattern in priority order.
            size_t startPos = static_cast<size_t>(it - src_.cbegin());
            bool matched = false;
            for (const auto& [kind, re] : rules_) {
                std::smatch m;
                if (std::regex_search(it, end, m, re, std::regex_constants::match_continuous)) {
                    size_t endPos = static_cast<size_t>(m[0].second - src_.cbegin());
                    toks.push_back(MakeToken(kind, m.str(), startPos, endPos));
                    it = m[0].second;
                    matched = true;
                    break;
                }
            }
            // 3. Try punctuation (single-char tokens).
            if (!matched) {
                IRTextLexerTokenKind punKind = ClassifyPunct(*it);
                if (punKind != IRTextLexerTokenKind::Invalid) {
                    std::string lexeme(1, *it);
                    size_t endPos = startPos + 1;
                    toks.push_back(IRTextLexerToken{punKind, lexeme, 0, 0.0, startPos, endPos});
                    ++it;
                    matched = true;
                }
            }
            if (!matched) {
                errorPos_ = startPos;
                return std::nullopt;
            }
        }
        toks.push_back(IRTextLexerToken{IRTextLexerTokenKind::Invalid, "", 0, 0.0, static_cast<size_t>(src_.size()),
                                        static_cast<size_t>(src_.size())});
        return toks;
    }

    size_t GetErrorPos() const { return errorPos_; }

private:
    const std::string& src_;
    std::vector<std::pair<IRTextLexerTokenKind, std::regex>> rules_;
    std::regex wsRe_;
    size_t errorPos_ = 0;

    static IRTextLexerTokenKind ClassifyKeyword(const std::string& s)
    {
        const auto& dict = IRTextGetLexerTokenDict();
        return dict.Count(s) ? dict.Find(s, IRTextLexerTokenKind::TokIdent) : IRTextLexerTokenKind::TokIdent;
    }

    static IRTextLexerTokenKind ClassifyPunct(char c)
    {
        std::string s(1, c);
        const auto& dict = IRTextGetLexerTokenDict();
        return dict.Count(s) ? dict.Find(s, IRTextLexerTokenKind::Invalid) : IRTextLexerTokenKind::Invalid;
    }

    static IRTextLexerToken MakeToken(IRTextLexerTokenKind kind, const std::string& lexeme, size_t begin, size_t end)
    {
        // Reclassify TokIdent as specific keyword if it matches.
        if (kind == IRTextLexerTokenKind::TokIdent) {
            kind = ClassifyKeyword(lexeme);
        }
        switch (kind) {
            case IRTextLexerTokenKind::TokV0OpMagic: {
                int64_t val = std::stoll(lexeme.substr(1));
                return IRTextLexerToken{kind, lexeme, val, 0.0, begin, end};
            }
            case IRTextLexerTokenKind::TokInt: {
                int64_t val = std::stoll(lexeme);
                return IRTextLexerToken{kind, lexeme, val, 0.0, begin, end};
            }
            case IRTextLexerTokenKind::TokFloat:
                return IRTextLexerToken{kind, lexeme, 0, std::stod(lexeme), begin, end};
            case IRTextLexerTokenKind::TokVarName:
            case IRTextLexerTokenKind::TokAttrName:
                return IRTextLexerToken{kind, lexeme.substr(1), 0, 0.0, begin, end};
            default:
                return IRTextLexerToken{kind, lexeme, 0, 0.0, begin, end};
        }
    }
};

} // namespace

// ===========================================================================
// Parser
// ===========================================================================
#define IR_TEXT_PARSER_RESULT "IRTextParserResult"
#define IR_TEXT_PARSER_SUCCESS "IRTextParserSuccess"

struct IRTextParserResult : public IRNode {
    IRTextParserResult() : IRNode(Span::Unknown()) {}
    ObjectKind GetKind() const override { return ObjectKind::Invalid; }
    std::string TypeName() const override { return IR_TEXT_PARSER_RESULT; }
};

struct IRTextParserSuccess : public IRTextParserResult {
    IRTextParserSuccess() : IRTextParserResult() {}
    std::string TypeName() const override { return IR_TEXT_PARSER_SUCCESS; }
};
using IRTextParserSuccessPtr = std::shared_ptr<const IRTextParserSuccess>;

template <typename T>
struct IRTextParserData : public IRTextParserResult {
    IRTextParserData() : IRTextParserResult() {}
    IRTextParserData(const T& d) : data(d) {}
    T data;
    std::string TypeName() const override { return typeid(T).name(); }
};
template <typename T>
using IRTextParserDataPtr = std::shared_ptr<IRTextParserData<T>>;

struct VarDefList {
    std::vector<VarPtr> valueList;
    std::vector<VarPtr> tokenList;

    const std::vector<VarPtr>& GetValueList() const { return valueList; }
    const std::vector<VarPtr>& GetTokenList() const { return tokenList; }
    std::vector<VarPtr> GetVarDefList() const
    {
        std::vector<VarPtr> resultList;
        resultList.insert(resultList.end(), valueList.begin(), valueList.end());
        resultList.insert(resultList.end(), tokenList.begin(), tokenList.end());
        return resultList;
    }
};

using IRTextParserStringPtr = IRTextParserDataPtr<std::string>;
using IRTextParserAttrValuePtr = IRTextParserDataPtr<std::any>;
using IRTextParserExprListPtr = IRTextParserDataPtr<std::vector<ExprPtr>>;
using IRTextParserIterArgListPtr = IRTextParserDataPtr<std::vector<IterArgPtr>>;
using IRTextParserAttrListPtr = IRTextParserDataPtr<std::vector<std::pair<std::string, std::any>>>;
using IRTextParserOptionalTensorViewPtr = IRTextParserDataPtr<std::optional<TensorView>>;
using IRTextParserOptionalTileViewPtr = IRTextParserDataPtr<std::optional<TileView>>;
using IRTextParserOptionalHardwareInfoPtr = IRTextParserDataPtr<std::optional<HardwareInfo>>;
using IRTextParserVarDefListPtr = IRTextParserDataPtr<VarDefList>;

#define MUST_MATCH(call)    \
    do {                    \
        if (!(call))        \
            return nullptr; \
    } while (0)

#define MUST_VALID(name, call) \
    auto name = (call);        \
    do {                       \
        if (!(name))           \
            return nullptr;    \
    } while (0)

class IRTextParser {
public:
    explicit IRTextParser(const std::string& text) : cur_(0)
    {
        success_ = std::make_shared<IRTextParserSuccess>();
        IRTextLexer lexer(text);
        auto result = lexer.Tokenize();
        if (result) {
            tokens_ = std::move(*result);
        } else {
            tokens_.push_back(IRTextLexerToken{IRTextLexerTokenKind::Invalid, "", 0, 0.0, 0, 0});
            errorMsg_ = "lex error at byte offset " + std::to_string(lexer.GetErrorPos());
            errorPos_ = lexer.GetErrorPos();
        }
    }

    friend class IRTextLoader;

    // ---- Helpers ----
    const IRTextLexerToken& Peek() const { return tokens_[cur_]; }
    const IRTextLexerToken& PeekAt(size_t offset) const { return tokens_[std::min(cur_ + offset, tokens_.size() - 1)]; }
    bool CheckToken(IRTextLexerTokenKind kind, bool consumeIfPass = false)
    {
        bool result = Peek().kind == kind;
        if (result) {
            if (consumeIfPass) {
                ++cur_;
            }
        }
        return result;
    }
    bool CheckVarName() const { return Peek().kind == IRTextLexerTokenKind::TokVarName; }
    bool CheckAttrName() const { return Peek().kind == IRTextLexerTokenKind::TokAttrName; }
    std::nullptr_t Error(const std::string& msg)
    {
        errorMsg_ = msg;
        errorPos_ = Peek().begin;
        return nullptr;
    }
    IRNodePtr ParseToken(IRTextLexerTokenKind kind)
    {
        if (!CheckToken(kind, true)) {
            return Error("expected token '" + IRTextGetLexerTokenDict().Find(kind, std::string()) + "', got '" +
                         Peek().text + "'");
        }
        return success_;
    }
    IRTextParserStringPtr ParseIdent()
    {
        if (!IsIdentOrKw(Peek().kind)) {
            return Error("expected identifier, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserStringPtr ParseVarName()
    {
        if (!CheckVarName()) {
            return Error("expected variable name, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserStringPtr ParseAttrName()
    {
        if (!CheckAttrName()) {
            return Error("expected attribute name, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserDataPtr<int64_t> ParseOpMagic()
    {
        if (Peek().kind != IRTextLexerTokenKind::TokV0OpMagic) {
            return Error("expected opmagic, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<int64_t>>();
        result->data = Peek().intVal;
        ++cur_;
        return result;
    }
    IRTextParserDataPtr<DataType> ParseDType()
    {
        if (!IsIdentOrKw(Peek().kind)) {
            return Error("expected dtype, got '" + Peek().text + "'");
        }
        const auto& dict = DataType::GetDataTypeCTypeStringDict();
        const std::string& s = Peek().text;
        if (!dict.Count(s)) {
            return Error("unknown C-type string: " + s);
        }
        auto result = std::make_shared<IRTextParserData<DataType>>();
        result->data = DataType(dict.Find(s, static_cast<uint8_t>(0)));
        ++cur_;
        return result;
    }
    IRTextParserExprListPtr ParseExprList(IRTextLexerTokenKind split, IRTextLexerTokenKind stop)
    {
        auto result = std::make_shared<IRTextParserData<std::vector<ExprPtr>>>();
        if (!CheckToken(stop)) {
            do {
                MUST_VALID(expr, ParseExpr());
                result->data.push_back(expr);
            } while (CheckToken(split, true));
        }
        return result;
    }
    IRTextParserIterArgListPtr ParseIterArgs()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<IterArgPtr>>>();
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwIter));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBrace));
        while (!CheckToken(IRTextLexerTokenKind::PunRBrace)) {
            MUST_VALID(type, ParseType());
            MUST_VALID(name, ParseVarName());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunEq));
            MUST_VALID(initVal, ParseExpr());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
            auto ia = std::make_shared<IterArg>(std::make_shared<Var>(name->data, type, Span::Unknown()), initVal);
            symtab_[name->data] = ia->iterVar_;
            result->data.push_back(ia);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBrace));
        return result;
    }
    IRTextParserAttrListPtr ParseAttrList()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<std::pair<std::string, std::any>>>>();
        while (CheckAttrName()) {
            MUST_VALID(key, ParseAttrName());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
            MUST_VALID(val, ParseAttrValueRaw());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
            result->data.emplace_back(key->data, val->data);
        }
        return result;
    }

    // ---- Var definition:  type %name->data ----
    VarPtr ParseVarDef()
    {
        MUST_VALID(type, ParseType());
        MUST_VALID(name, ParseVarName());
        std::string varName = name->data;
        if (As<LogicalTensorType>(type)) {
            // LogicalTensor var_def carries dtype/shape/offset/dynvalidshape/dynvalidoffset attrs
            return ParseLogicalTensorVar(varName);
        }
        // Non-tensor vars keep no attrs; consume the attr_list (if any) for format tolerance
        MUST_VALID(attrs, ParseAttrList());
        auto v = std::make_shared<Var>(name->data, type, Span::Unknown());
        symtab_[name->data] = v;
        return v;
    }

    void FillAttr(npu::tile_fwk::DataType& dst, const std::any& src)
    {
        const auto& dtypeStr = AnyCast<std::string>(src);
        dst = GetTileFwkDataTypeDict().Find(dtypeStr, npu::tile_fwk::DT_INT64);
    }

    void FillAttr(std::vector<int64_t>& dst, const std::any& src)
    {
        const auto& list = std::any_cast<const std::vector<std::any>&>(src);
        for (const auto& elem : list) {
            dst.push_back(AnyCast<int64_t>(elem));
        }
    }

    void FillAttr(std::vector<npu::tile_fwk::SymbolicScalar>& dst, const std::any& src)
    {
        const auto& list = std::any_cast<const std::vector<std::any>&>(src);
        for (const auto& elem : list) {
            if (elem.type() == typeid(int64_t)) {
                dst.emplace_back(static_cast<int64_t>(AnyCast<int64_t>(elem)));
            } else if (elem.type() == typeid(std::string)) {
                dst.emplace_back(AnyCast<std::string>(elem));
            } else if (elem.type() == typeid(ExprPtr)) {
                auto expr = AnyCast<ExprPtr>(elem);
                dst.emplace_back(std::dynamic_pointer_cast<npu::tile_fwk::RawSymbolicScalar>(
                    std::const_pointer_cast<ir::Expr>(expr)));
            } else {
                Error("unsupported symbolic scalar attr element type");
                return;
            }
        }
    }

    // v0_logical_tensor %name #shape(..) #offset(..) #dynvalidshape(..) #dynvalidoffset(..)
    // Attributes are parsed generically via ParseAttrList, then dispatched by name
    // to initialize the LogicalTensor fields.
    VarPtr ParseLogicalTensorVar(const std::string& varName)
    {
        auto atPos = varName.find('@');
        int memoryId = std::stoi(varName.substr(atPos + 1));
        std::string name = varName.substr(0, atPos);

        MUST_VALID(attrs, ParseAttrList());

        npu::tile_fwk::DataType dtype = npu::tile_fwk::DT_INT64;
        std::vector<int64_t> shape;
        std::vector<int64_t> offset;
        std::vector<npu::tile_fwk::SymbolicScalar> dynValidShape;
        std::vector<npu::tile_fwk::SymbolicScalar> dynOffset;
        for (const auto& [key, value] : attrs->data) {
            if (key == "dtype") {
                FillAttr(dtype, value);
            } else if (key == "shape") {
                FillAttr(shape, value);
            } else if (key == "offset") {
                FillAttr(offset, value);
            } else if (key == "dynvalidshape") {
                FillAttr(dynValidShape, value);
            } else if (key == "dynvalidoffset") {
                FillAttr(dynOffset, value);
            } else {
                return Error("unknown logical tensor attr: #" + key);
            }
        }

        npu::tile_fwk::IRBuilder builder;
        npu::tile_fwk::LogicalTensorPtr tensor = builder.CreateTensorVar(dtype, shape,
                                                                         npu::tile_fwk::TileOpFormat::TILEOP_ND, name);
        if (memoryId >= 0) {
            tensor->tensor->memoryId = memoryId;
        }
        if (!offset.empty()) {
            tensor->offset = offset;
        }
        tensor->dynValidShape_ = dynValidShape;
        tensor->dynOffset_ = dynOffset;
        symtab_[name] = tensor;
        return tensor;
    }

    // ---- Shape parsing:  dim0 x dim1 x ... ----
    IRTextParserExprListPtr ParseShape()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<ExprPtr>>>();
        MUST_VALID(dim, ParseExpr());
        result->data.push_back(dim);
        while (CheckToken(IRTextLexerTokenKind::KwDim)) {
            ++cur_; // consume 'x'
            MUST_VALID(dim2, ParseExpr());
            result->data.push_back(dim2);
        }
        return result;
    }

    // ---- Attr value parsing (per-branch, dispatched by token) ----
    using ParseAttrValueFn = IRTextParserAttrValuePtr (IRTextParser::*)();

    IRTextParserAttrValuePtr ParseAttrValueInt()
    {
        const IRTextLexerToken& tok = Peek();
        ++cur_;
        std::any data = std::any(static_cast<int64_t>(tok.intVal));
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueFloat()
    {
        const IRTextLexerToken& tok = Peek();
        ++cur_;
        std::any data = std::any(tok.floatVal);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueTrue()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTrue));
        std::any data = true;
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueFalse()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwFalse));
        std::any data = false;
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueIdent()
    {
        const IRTextLexerToken& tok = Peek();
        std::string s = tok.text;
        ++cur_;
        std::any data = std::any(s);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueList()
    {
        // '[' [ attr_value { ',' attr_value } ] ']' -> std::vector<std::any>
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBracket));
        std::vector<std::any> vals;
        if (!CheckToken(IRTextLexerTokenKind::PunRBracket)) {
            do {
                MUST_VALID(val, ParseAttrValueRaw());
                vals.push_back(val->data);
            } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBracket));
        std::any data = std::any(vals);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }

    IRTextParserAttrValuePtr ParseAttrValueExpr()
    {
        // expr fallback: keep the parsed expr as std::any(ExprPtr)
        MUST_VALID(expr, ParseExpr());
        std::any data = std::any(expr);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }

    IRTextParserAttrValuePtr ParseAttrValueRaw()
    {
        static const std::unordered_map<IRTextLexerTokenKind, ParseAttrValueFn> dispatch = {
            {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseAttrValueInt},
            {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseAttrValueFloat},
            {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseAttrValueIdent},
            // dtype keywords (int32_t / float / bfloat16_t ...) are treated as ident-shaped attr values
            {IRTextLexerTokenKind::KwTypeBool, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeInt8, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeInt16, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeInt32, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeInt64, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeUint8, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeUint16, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeUint32, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeUint64, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeFp16, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeFp32, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeFp64, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeBf16, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeFp8e4m3fn, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeFp8e5m2, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeHf4, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeHf8, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTypeUnknown, &IRTextParser::ParseAttrValueIdent},
            {IRTextLexerTokenKind::KwTrue, &IRTextParser::ParseAttrValueTrue},
            {IRTextLexerTokenKind::KwFalse, &IRTextParser::ParseAttrValueFalse},
            {IRTextLexerTokenKind::PunLBracket, &IRTextParser::ParseAttrValueList},
            // expr-start tokens -> std::any(ExprPtr)
            {IRTextLexerTokenKind::TokVarName, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::PunLParen, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwTuple, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwGetItem, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwMemref, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarUOpPos, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarUOpNeg, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarUOpNot, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpAdd, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpSub, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpMul, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpDiv, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpMod, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpEq, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpNe, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpLt, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpLe, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpGt, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpGe, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpCall, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpMin, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpMax, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpAnd, &IRTextParser::ParseAttrValueExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpOr, &IRTextParser::ParseAttrValueExpr},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            auto result = (this->*(it->second))();
            return result;
        }
        return Error("expected attr value");
    }

    ExprPtr MakeUnary(const std::string& opName, const ExprPtr& operand, const TypePtr& type)
    {
        DataType dt = DataType::INT64;
        if (auto st = std::dynamic_pointer_cast<const ScalarType>(type)) {
            dt = st->dtype_;
        }
        auto result = MakeUnaryOp(opName, operand, dt);
        if (!result) {
            return Error("Unknown unary op: " + opName);
        }
        return result;
    }

    ExprPtr MakeBinary(const std::string& opName, const ExprPtr& left, const ExprPtr& right)
    {
        auto result = MakeBinaryOp(opName, left, right);
        if (!result) {
            return Error("Unknown binary op: " + opName);
        }
        return result;
    }

    // ---- Operator helpers ----
    static bool IsUnaryOpName(const std::string& s)
    {
        return s == IR_KW_SCALAR_UOP_ABS || s == IR_KW_SCALAR_UOP_NEG || s == IR_KW_SCALAR_UOP_NOT ||
               s == IR_KW_SCALAR_UOP_INV || s == IR_KW_SCALAR_UOP_CAST;
    }

    // ---- Type parsing ----
    // ---- Type parsing (per-branch, dispatched by token) ----
    TypePtr ParseTypeUnknown()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwUnknown));
        return GetUnknownType();
    }
    TypePtr ParseTypeMemRefType()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwMemrefType));
        return GetMemRefType();
    }
    TypePtr ParseTypeToken()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwToken));
        return GetTokenType();
    }
    TypePtr ParseTypeNone()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwNone));
        return GetNoneType();
    }
    TypePtr ParseTypeLogicalTensor()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwV0LogicalTensor));
        return GetLogicalTensorType();
    }
    TypePtr ParseTypePtr()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwPtr));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));
        MUST_VALID(dt, ParseDType());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<PtrType>(dt->data);
    }
    TypePtr ParseTypeTuple()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTuple));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));
        std::vector<TypePtr> types;
        if (!CheckToken(IRTextLexerTokenKind::PunGt)) {
            do {
                MUST_VALID(t, ParseType());
                types.push_back(t);
            } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<TupleType>(types);
    }
    TypePtr ParseTypeTensor()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTensor));
        return ParseTensorType();
    }
    TypePtr ParseTypeTile()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTile));
        return ParseTileType();
    }
    TypePtr ParseTypeScalar()
    {
        MUST_VALID(dt, ParseDType());
        return std::make_shared<ScalarType>(dt->data);
    }

    IRTextParserOptionalTensorViewPtr ParserTensorTypeTensorView()
    {
        // tensor_view<...> or tensor_view<>
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTensorView));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));

        std::optional<TensorView> tensorView;
        if (!CheckToken(IRTextLexerTokenKind::PunGt)) {
            MUST_VALID(validShape, ParseShape());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(stride, ParseShape());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(layoutStr, ParseIdent());
            TensorLayout layout = GetTensorLayoutDict().Find(layoutStr->data, TensorLayout::ND);
            if (CheckToken(IRTextLexerTokenKind::PunComma, true)) {
                MUST_VALID(ptrExpr, ParseExpr());
                tensorView = TensorView(validShape->data, stride->data, layout, ptrExpr);
            } else {
                tensorView = TensorView(validShape->data, stride->data, layout);
            }
        }

        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalTensorViewPtr::element_type>(tensorView);
    }

    // ---- Type-specific parsers ----
    TypePtr ParseTensorType()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));
        MUST_VALID(shape, ParseShape());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(dtype, ParseDType());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(tensorView, ParserTensorTypeTensorView());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<TensorType>(shape->data, dtype->data, std::nullopt, tensorView->data);
    }

    IRTextParserOptionalTileViewPtr ParseTileTypeTileView()
    {
        // tile_view<...> or tile_view<>
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTileView));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));

        std::optional<TileView> tileView;
        if (!CheckToken(IRTextLexerTokenKind::PunGt)) {
            MUST_VALID(validShape, ParseShape());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(stride, ParseShape());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            ExprPtr startOffset = nullptr;
            if (!CheckToken(IRTextLexerTokenKind::KwNull)) {
                MUST_VALID(offset, ParseExpr());
                startOffset = offset;
            }
            tileView = TileView(validShape->data, stride->data, startOffset);
        }

        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalTileViewPtr::element_type>(tileView);
    }

    IRTextParserOptionalHardwareInfoPtr ParseTileTypeHardwareInfo()
    {
        // hw_info<...> or hw_info<>
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwHwInfo));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));

        std::optional<HardwareInfo> hwInfo;
        if (!CheckToken(IRTextLexerTokenKind::PunGt)) {
            MUST_VALID(blayoutStr, ParseIdent());
            TileLayout blayout = GetTileLayoutDict().Find(blayoutStr->data, TileLayout::row_major);
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(slayoutStr, ParseIdent());
            TileLayout slayout = GetTileLayoutDict().Find(slayoutStr->data, TileLayout::none_box);
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            uint64_t fractal = HardwareInfo::kDefaultFractal;
            if (Peek().kind == IRTextLexerTokenKind::TokInt) {
                fractal = static_cast<uint64_t>(tokens_[cur_].intVal);
                ++cur_;
            }
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(padStr, ParseIdent());
            TilePad pad = GetTilePadDict().Find(padStr->data, TilePad::null);
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
            MUST_VALID(compactStr, ParseIdent());
            CompactMode compact = GetCompactModeDict().Find(compactStr->data, CompactMode::null);
            hwInfo = HardwareInfo(blayout, slayout, fractal, pad, compact);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalHardwareInfoPtr::element_type>(hwInfo);
    }

    TypePtr ParseTileType()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLt));
        MUST_VALID(shape, ParseShape());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(dtype, ParseDType());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(tileView, ParseTileTypeTileView());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(hwInfo, ParseTileTypeHardwareInfo());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunGt));
        return std::make_shared<TileType>(shape->data, dtype->data, std::nullopt, tileView->data, hwInfo->data);
    }

    TypePtr ParseType()
    {
        using ParseTypeFn = TypePtr (IRTextParser::*)();
        if (IsDTypeKw(Peek().kind)) {
            return ParseTypeScalar();
        }
        static const std::unordered_map<IRTextLexerTokenKind, ParseTypeFn> dispatch = {
            {IRTextLexerTokenKind::KwUnknown, &IRTextParser::ParseTypeUnknown},
            {IRTextLexerTokenKind::KwMemrefType, &IRTextParser::ParseTypeMemRefType},
            {IRTextLexerTokenKind::KwToken, &IRTextParser::ParseTypeToken},
            {IRTextLexerTokenKind::KwNone, &IRTextParser::ParseTypeNone},
            {IRTextLexerTokenKind::KwV0LogicalTensor, &IRTextParser::ParseTypeLogicalTensor},
            {IRTextLexerTokenKind::KwPtr, &IRTextParser::ParseTypePtr},
            {IRTextLexerTokenKind::KwTuple, &IRTextParser::ParseTypeTuple},
            {IRTextLexerTokenKind::KwTensor, &IRTextParser::ParseTypeTensor},
            {IRTextLexerTokenKind::KwTile, &IRTextParser::ParseTypeTile},
            {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseTypeScalar},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))();
        }
        return Error("expected type");
    }

    // ---- Expression parsing (per-branch, dispatched by token) ----
    ExprPtr ParseExprVarName()
    {
        std::string name = Peek().text;
        if (Peek().kind != IRTextLexerTokenKind::TokVarName) {
            return Error("expected variable name, got '" + name + "'");
        }
        ++cur_;
        // strip '@memoryId' suffix for symbol table lookup
        auto atPos = name.find('@');
        if (atPos != std::string::npos) {
            name = name.substr(0, atPos);
        }
        auto it = symtab_.find(name);
        if (it != symtab_.end()) {
            return it->second;
        } else {
            return Error("Unknown var name: " + name);
        }
    }
    ExprPtr ParseExprInt()
    {
        auto val = Peek().intVal;
        ++cur_;
        return std::make_shared<ConstInt>(val, DataType::INT64, Span::Unknown());
    }
    ExprPtr ParseExprFloat()
    {
        auto val = Peek().floatVal;
        ++cur_;
        return std::make_shared<ConstFloat>(val, DataType::FP64, Span::Unknown());
    }
    ExprPtr ParseExprTrue()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTrue));
        return std::make_shared<ConstBool>(true, Span::Unknown());
    }
    ExprPtr ParseExprFalse()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwFalse));
        return std::make_shared<ConstBool>(false, Span::Unknown());
    }
    ExprPtr ParseExprTuple()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTuple));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(elts, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        return std::make_shared<MakeTuple>(elts->data, Span::Unknown());
    }
    ExprPtr ParseExprGetItem()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwGetItem));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(value, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(slice, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        return std::make_shared<GetItemExpr>(value, slice, Span::Unknown());
    }
    ExprPtr ParseExprMemref()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwMemref));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(spaceStr, ParseIdent());
        MemorySpace space = StringToMemorySpace(spaceStr->data);
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(addr, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        uint64_t size = 0;
        if (Peek().kind == IRTextLexerTokenKind::TokInt) {
            size = static_cast<uint64_t>(tokens_[cur_++].intVal);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        return std::make_shared<MemRef>(space, addr, size, Span::Unknown());
    }
    ExprPtr ParseExprScalarExpr()
    {
        using npu::tile_fwk::RawSymbolicExpression;
        using npu::tile_fwk::RawSymbolicImmediate;
        using npu::tile_fwk::RawSymbolicScalar;
        using npu::tile_fwk::SymbolicOpcode;

        // Entry token is a scalarv0_op keyword: v0add, v0call, ...
        if (Peek().kind < IRTextLexerTokenKind::KwV0ScalarBegin || Peek().kind > IRTextLexerTokenKind::KwV0ScalarEnd) {
            return Error("expected scalarv0_op, got '" + Peek().text + "'");
        }
        auto opcode = GetSymbolicOpcodeDict().Find(Peek().text, SymbolicOpcode::T_MOP_CALL);
        ++cur_; // consume scalarv0_op keyword token
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        std::vector<std::shared_ptr<RawSymbolicScalar>> operands;
        if (!CheckToken(IRTextLexerTokenKind::PunRParen)) {
            do {
                if (Peek().kind == IRTextLexerTokenKind::TokInt) {
                    operands.push_back(std::make_shared<RawSymbolicImmediate>(tokens_[cur_++].intVal));
                } else if (Peek().kind == IRTextLexerTokenKind::TokIdent) {
                    MUST_VALID(ident, ParseIdent());
                    operands.push_back(std::make_shared<npu::tile_fwk::RawSymbolicSymbol>(ident->data));
                } else if (Peek().kind >= IRTextLexerTokenKind::KwV0ScalarBegin &&
                           Peek().kind <= IRTextLexerTokenKind::KwV0ScalarEnd) {
                    MUST_VALID(sub, ParseExprScalarExpr());
                    operands.push_back(std::const_pointer_cast<RawSymbolicScalar>(
                        std::dynamic_pointer_cast<const RawSymbolicScalar>(sub)));
                } else {
                    return Error("unexpected token in scalar_expr arg: '" + Peek().text + "'");
                }
            } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        auto expr = std::make_shared<RawSymbolicExpression>(opcode, operands);
        return std::dynamic_pointer_cast<const Expr>(expr);
    }
    ExprPtr ParseExprCall()
    {
        std::string name = Peek().text;
        MUST_VALID(ident, ParseIdent());
        if (!CheckToken(IRTextLexerTokenKind::PunLParen)) {
            return Error("unexpected identifier: " + name);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        std::vector<ExprPtr> args;
        std::vector<std::pair<std::string, std::any>> kwargs;
        if (!CheckToken(IRTextLexerTokenKind::PunRParen)) {
            do {
                if (Peek().kind == IRTextLexerTokenKind::TokIdent && PeekAt(1).kind == IRTextLexerTokenKind::PunEq) {
                    MUST_VALID(key, ParseIdent());
                    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunEq));
                    MUST_VALID(val, ParseAttrValueRaw());
                    kwargs.emplace_back(key->data, val->data);
                } else {
                    MUST_VALID(arg, ParseExpr());
                    args.push_back(arg);
                }
            } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        return std::make_shared<Call>(name, args, kwargs, Span::Unknown());
    }
    ExprPtr ParseExprParen()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        const IRTextLexerToken& first = Peek();

        // Determine if this is unary: first token is an TokIdent that is a known unary op
        if (IsIdentOrKw(first.kind) && IsUnaryOpName(first.text)) {
            std::string opName = first.text;
            ++cur_;
            DataType castDtype = DataType::INT64;
            if (opName == IR_KW_SCALAR_UOP_CAST) {
                MUST_VALID(castDt, ParseDType());
                castDtype = castDt->data;
            }
            MUST_VALID(operand, ParseExpr());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
            auto scalarType = std::make_shared<ScalarType>(castDtype);
            return MakeUnary(opName, operand, scalarType);
        } else {
            // Binary: expr op expr
            MUST_VALID(left, ParseExpr());
            MUST_VALID(op, ParseIdent());
            MUST_VALID(right, ParseExpr());
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
            return MakeBinary(op->data, left, right);
        }
    }

    ExprPtr ParseExpr()
    {
        using ParseExprFn = ExprPtr (IRTextParser::*)();
        static const std::unordered_map<IRTextLexerTokenKind, ParseExprFn> dispatch = {
            {IRTextLexerTokenKind::TokVarName, &IRTextParser::ParseExprVarName},
            {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseExprInt},
            {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseExprFloat},
            {IRTextLexerTokenKind::PunLParen, &IRTextParser::ParseExprParen},
            {IRTextLexerTokenKind::KwTrue, &IRTextParser::ParseExprTrue},
            {IRTextLexerTokenKind::KwFalse, &IRTextParser::ParseExprFalse},
            {IRTextLexerTokenKind::KwTuple, &IRTextParser::ParseExprTuple},
            {IRTextLexerTokenKind::KwGetItem, &IRTextParser::ParseExprGetItem},
            {IRTextLexerTokenKind::KwMemref, &IRTextParser::ParseExprMemref},
            {IRTextLexerTokenKind::KwV0ScalarUOpPos, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarUOpNeg, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarUOpNot, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpAdd, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpSub, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpMul, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpDiv, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpMod, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpEq, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpNe, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpLt, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpLe, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpGt, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarBOpGe, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpCall, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpMin, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpMax, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpAnd, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::KwV0ScalarMOpOr, &IRTextParser::ParseExprScalarExpr},
            {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseExprCall},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))();
        }
        return Error("unexpected token in expression: '" + Peek().text + "'");
    }

    // ---- Statement parsing ----

    IRTextParserVarDefListPtr ParseVarDefList(IRTextLexerTokenKind split, IRTextLexerTokenKind stop)
    {
        VarDefList varDefList;
        if (!CheckToken(stop)) {
            do {
                MUST_VALID(varDef, ParseVarDef());
                if (std::dynamic_pointer_cast<const TokenType>(varDef->GetType())) {
                    varDefList.tokenList.push_back(varDef);
                } else {
                    varDefList.valueList.push_back(varDef);
                }
            } while (CheckToken(split, true));
        }
        return std::make_shared<IRTextParserVarDefListPtr::element_type>(varDefList);
    }

    StmtPtr ParseStmtSeq(const VarDefList& = {})
    {
        // Reads statements until '}' or end
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBrace));
        std::vector<StmtPtr> stmts;
        while (!CheckToken(IRTextLexerTokenKind::PunRBrace) && Peek().kind != IRTextLexerTokenKind::Invalid) {
            MUST_VALID(stmt, ParseStmt());
            stmts.push_back(stmt);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBrace));
        return std::make_shared<SeqStmts>(stmts, Span::Unknown());
    }

    // ---- Individual statement parsers ----
    StmtPtr ParseStmtYield(const VarDefList&)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwYield));
        MUST_VALID(values, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<YieldStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtReturn(const VarDefList&)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwReturn));
        MUST_VALID(values, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<ReturnStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtIf(const VarDefList& defVarList)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwIf));
        MUST_VALID(cond, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwThen));
        MUST_VALID(thenBody, ParseStmtSeq());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwElse));
        MUST_VALID(elseBody, ParseStmtSeq());
        return std::make_shared<IfStmt>(cond, thenBody, elseBody, defVarList.GetVarDefList(), Span::Unknown());
    }

    StmtPtr ParseStmtFor(const VarDefList& defVarList)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwFor));
        // for %loopVar inrange start, stop, step [iter (...)] {body}
        MUST_VALID(loopVarName, ParseVarName());
        auto loopVar = std::make_shared<Var>(loopVarName->data, std::make_shared<ScalarType>(DataType::INT64),
                                             Span::Unknown());
        symtab_[loopVarName->data] = loopVar;
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwInrange));
        MUST_VALID(start, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(stop, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
        MUST_VALID(step, ParseExpr());
        MUST_VALID(iterArgs, ParseIterArgs());
        MUST_VALID(attrs, ParseAttrList());
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs->data, body, defVarList.GetVarDefList(),
                                         Span::Unknown(), attrs->data);
    }

    StmtPtr ParseStmtWhile(const VarDefList& defVarList)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwWhile));
        MUST_VALID(cond, ParseExpr());
        MUST_VALID(iterArgs, ParseIterArgs());
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<WhileStmt>(cond, iterArgs->data, body, defVarList.GetVarDefList(), Span::Unknown());
    }

    StmtPtr ParseStmtSection(const VarDefList&)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwSection));
        MUST_VALID(sectionName, ParseIdent());
        SectionKind kind = StringToSectionKind(sectionName->data);
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<SectionStmt>(kind, body, Span::Unknown());
    }

    StmtPtr ParseStmtEval(const VarDefList&)
    {
        MUST_VALID(expr, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<EvalStmt>(expr, Span::Unknown());
    }

    StmtPtr ParseStmtBreak(const VarDefList&)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwBreak));
        MUST_VALID(values, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<BreakStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtContinue(const VarDefList&)
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwContinue));
        MUST_VALID(values, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<ContinueStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtTensorOrScalarOp(const VarDefList& defVarList)
    {
        MUST_VALID(opcode, ParseIdent());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(args, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));

        std::vector<VarPtr> tokenList;
        bool hasToken = false;
        if (CheckToken(IRTextLexerTokenKind::KwToken, true)) {
            hasToken = true;
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
            MUST_VALID(tokens, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
            MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
            for (const auto& token : tokens->data) {
                tokenList.push_back(std::dynamic_pointer_cast<const Var>(token));
            }
        }
        MUST_VALID(attrs, ParseAttrList());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));

        VarPtr resultToken = defVarList.GetTokenList().empty() ? nullptr : defVarList.GetTokenList()[0];
        if (!hasToken) {
            return std::make_shared<ScalarOpStmt>(defVarList.GetValueList().front(), resultToken, opcode->data,
                                                  args->data, Span::Unknown());
        } else {
            return std::make_shared<TensorOpStmt>(defVarList.GetValueList(), resultToken, opcode->data, args->data,
                                                  tokenList, attrs->data, Span::Unknown());
        }
    }

    // Assignment or op statement:
    //   type %result [= expr] ;
    //   type %result, type %token = opcode(args) [tokens(...)] [#attr(val)...] ;
    //   type %result, type %token = opcode(args) ;
    StmtPtr ParseStmtV0TensorOp(const VarDefList& defVarList)
    {
        MUST_VALID(opmagic, ParseOpMagic());
        MUST_VALID(opcode, ParseIdent());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(args, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));

        std::vector<VarPtr> tokenList;
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwToken));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(tokens, ParseExprList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        for (const auto& token : tokens->data) {
            tokenList.push_back(std::dynamic_pointer_cast<const Var>(token));
        }

        MUST_VALID(attrs, ParseAttrList());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));

        npu::tile_fwk::LogicalTensors iOperands;
        for (const auto& arg : args->data) {
            iOperands.push_back(
                std::dynamic_pointer_cast<npu::tile_fwk::LogicalTensor>(std::const_pointer_cast<ir::Expr>(arg)));
        }
        npu::tile_fwk::LogicalTensors oOperands;
        for (const auto& res : defVarList.valueList) {
            oOperands.push_back(
                std::dynamic_pointer_cast<npu::tile_fwk::LogicalTensor>(std::const_pointer_cast<ir::Var>(res)));
        }

        auto op = std::make_shared<npu::tile_fwk::Operation>(*tempFunc_, npu::tile_fwk::FindOpcode(opcode->data),
                                                             iOperands, oOperands, static_cast<int>(opmagic->data));
        op->result_token_ = defVarList.tokenList;
        op->tokens_ = tokenList;
        return op;
    }

    // Assignment: type %result = expr;
    StmtPtr ParseStmtAssign(const VarDefList& defVarList)
    {
        MUST_VALID(value, ParseExpr());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunSemi));
        return std::make_shared<AssignStmt>(defVarList.valueList.front(), value, Span::Unknown());
    }

    using ParseStmtFn = StmtPtr (IRTextParser::*)(const VarDefList&);

    StmtPtr ParseStmtVarDef()
    {
        MUST_VALID(varDefList, ParseVarDefList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunEq));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunEq));

        static const std::unordered_map<IRTextLexerTokenKind, ParseStmtFn> dispatch = {
            {IRTextLexerTokenKind::KwIf, &IRTextParser::ParseStmtIf},
            {IRTextLexerTokenKind::KwFor, &IRTextParser::ParseStmtFor},
            {IRTextLexerTokenKind::KwWhile, &IRTextParser::ParseStmtWhile},
            {IRTextLexerTokenKind::TokV0OpMagic, &IRTextParser::ParseStmtV0TensorOp},
            {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseStmtTensorOrScalarOp},
            {IRTextLexerTokenKind::PunLParen, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::TokVarName, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwTrue, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwFalse, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwTuple, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwGetItem, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwMemref, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarUOpPos, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarUOpNeg, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarUOpNot, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpAdd, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpSub, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpMul, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpDiv, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpMod, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpEq, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpNe, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpLt, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpLe, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpGt, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarBOpGe, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarMOpCall, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarMOpMin, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarMOpMax, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarMOpAnd, &IRTextParser::ParseStmtAssign},
            {IRTextLexerTokenKind::KwV0ScalarMOpOr, &IRTextParser::ParseStmtAssign},
        };
        const IRTextLexerToken& tok = Peek();
        auto it = dispatch.find(tok.kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))(varDefList->data);
        } else {
            return Error("Unknown var def: " + tok.text);
        }
    }

    // ---- Statement dispatch ----
    StmtPtr ParseStmt()
    {
        static const std::unordered_map<IRTextLexerTokenKind, ParseStmtFn> dispatch = {
            {IRTextLexerTokenKind::PunLBrace, &IRTextParser::ParseStmtSeq},
            {IRTextLexerTokenKind::KwYield, &IRTextParser::ParseStmtYield},
            {IRTextLexerTokenKind::KwReturn, &IRTextParser::ParseStmtReturn},
            {IRTextLexerTokenKind::KwSection, &IRTextParser::ParseStmtSection},
            {IRTextLexerTokenKind::KwBreak, &IRTextParser::ParseStmtBreak},
            {IRTextLexerTokenKind::KwContinue, &IRTextParser::ParseStmtContinue},
            {IRTextLexerTokenKind::KwIf, &IRTextParser::ParseStmtIf},
            {IRTextLexerTokenKind::KwFor, &IRTextParser::ParseStmtFor},
            {IRTextLexerTokenKind::KwWhile, &IRTextParser::ParseStmtWhile},

            {IRTextLexerTokenKind::TokV0OpMagic, &IRTextParser::ParseStmtV0TensorOp},
            {IRTextLexerTokenKind::PunLParen, &IRTextParser::ParseStmtEval},
            {IRTextLexerTokenKind::KwTrue, &IRTextParser::ParseStmtEval},
            {IRTextLexerTokenKind::KwFalse, &IRTextParser::ParseStmtEval},
            {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseStmtEval},
            {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseStmtEval},
            {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseStmtEval},
        };
        const IRTextLexerToken& tok = Peek();
        auto it = dispatch.find(tok.kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))({});
        }

        static const std::unordered_set<IRTextLexerTokenKind> typeKeywords = {
            IRTextLexerTokenKind::KwTensor,      IRTextLexerTokenKind::KwTile,
            IRTextLexerTokenKind::KwTuple,       IRTextLexerTokenKind::KwPtr,
            IRTextLexerTokenKind::KwToken,       IRTextLexerTokenKind::KwNone,
            IRTextLexerTokenKind::KwMemrefType,  IRTextLexerTokenKind::KwV0LogicalTensor,
            IRTextLexerTokenKind::KwTypeBool,    IRTextLexerTokenKind::KwTypeInt8,
            IRTextLexerTokenKind::KwTypeInt16,   IRTextLexerTokenKind::KwTypeInt32,
            IRTextLexerTokenKind::KwTypeInt64,   IRTextLexerTokenKind::KwTypeUint8,
            IRTextLexerTokenKind::KwTypeUint16,  IRTextLexerTokenKind::KwTypeUint32,
            IRTextLexerTokenKind::KwTypeUint64,  IRTextLexerTokenKind::KwTypeFp16,
            IRTextLexerTokenKind::KwTypeFp32,    IRTextLexerTokenKind::KwTypeFp64,
            IRTextLexerTokenKind::KwTypeBf16,    IRTextLexerTokenKind::KwTypeFp8e4m3fn,
            IRTextLexerTokenKind::KwTypeFp8e5m2, IRTextLexerTokenKind::KwTypeHf4,
            IRTextLexerTokenKind::KwTypeHf8,     IRTextLexerTokenKind::KwTypeUnknown,
        };
        if (typeKeywords.count(tok.kind)) {
            return ParseStmtVarDef();
        } else {
            return Error("unknown statement");
        }
    }

    // ---- Function parsing ----
    FunctionPtr ParseFunction()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwFunction));
        MUST_VALID(name, ParseIdent());

        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwIncast));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(params, ParseVarDefList(IRTextLexerTokenKind::PunComma, IRTextLexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));

        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwOutcast));
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        std::vector<TypePtr> retTypes;
        if (!CheckToken(IRTextLexerTokenKind::PunRParen)) {
            do {
                MUST_VALID(t, ParseType());
                retTypes.push_back(t);
            } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        MUST_VALID(attrs, ParseAttrList());

        FunctionType funcType = FunctionType::OPAQUE;
        bool isEntry = false;
        for (const auto& [key, value] : attrs->data) {
            if (key == IR_KW_TYPE) {
                const auto& typeStr = AnyCast<std::string>(value);
                try {
                    funcType = StringToFunctionType(typeStr);
                } catch (const std::invalid_argument&) {
                    return Error("unknown function type: " + typeStr);
                }
            } else if (key == IR_KW_ENTRY) {
                isEntry = AnyCast<bool>(value);
            }
        }

        tempProgram_ = std::make_unique<npu::tile_fwk::Program>();
        tempFunc_ = std::make_shared<npu::tile_fwk::Function>(*tempProgram_, name->data, name->data, nullptr);

        MUST_VALID(body, ParseStmtSeq());

        tempFunc_.reset();
        tempProgram_.reset();

        return std::make_shared<Function>(name->data, params->data.GetVarDefList(), retTypes, body, Span::Unknown(),
                                          funcType, isEntry);
    }

    // ---- Program parsing ----
    ProgramPtr ParseProgram()
    {
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwProgram));
        MUST_MATCH(ParseAttrList());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBrace));

        std::vector<FunctionPtr> funcs;
        while (!CheckToken(IRTextLexerTokenKind::PunRBrace)) {
            MUST_VALID(func, ParseFunction());
            funcs.push_back(func);
        }
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBrace));

        auto prog = std::make_shared<Program>(funcs, IR_KW_PROGRAM, Span::Unknown());
        return prog;
    }

private:
    // ---- Member variables ----
    std::vector<IRTextLexerToken> tokens_;
    size_t cur_;
    std::unordered_map<std::string, VarPtr> symtab_;
    IRTextParserSuccessPtr success_;
    std::string errorMsg_;
    size_t errorPos_ = 0;
    std::unique_ptr<npu::tile_fwk::Program> tempProgram_;
    std::shared_ptr<npu::tile_fwk::Function> tempFunc_;
};

// ===========================================================================
// Public API
// ===========================================================================
ProgramPtr IRTextLoader::LoadProgram(const std::string& text, std::string& error)
{
    IRTextParser parser(text);
    auto prog = parser.ParseProgram();
    error = parser.errorMsg_;
    return prog;
}

FunctionPtr IRTextLoader::LoadFunction(const std::string& text, std::string& error)
{
    IRTextParser parser(text);
    auto func = parser.ParseFunction();
    error = parser.errorMsg_;
    return func;
}

StmtPtr IRTextLoader::LoadStmt(const std::string& text, std::string& error)
{
    IRTextParser parser(text);
    StmtPtr stmt;
    if (parser.Peek().kind == IRTextLexerTokenKind::PunLBrace) {
        stmt = parser.ParseStmtSeq();
    } else {
        stmt = parser.ParseStmt();
    }
    error = parser.errorMsg_;
    return stmt;
}

ExprPtr IRTextLoader::LoadExpr(const std::string& text, std::string& error)
{
    IRTextParser parser(text);
    auto expr = parser.ParseExpr();
    error = parser.errorMsg_;
    return expr;
}

TypePtr IRTextLoader::LoadType(const std::string& text, std::string& error)
{
    IRTextParser parser(text);
    auto type = parser.ParseType();
    error = parser.errorMsg_;
    return type;
}

ProgramPtr TextLoadProgram(const std::string& text, std::string& error)
{
    return IRTextLoader().LoadProgram(text, error);
}
FunctionPtr TextLoadFunction(const std::string& text, std::string& error)
{
    return IRTextLoader().LoadFunction(text, error);
}
StmtPtr TextLoadStmt(const std::string& text, std::string& error) { return IRTextLoader().LoadStmt(text, error); }
ExprPtr TextLoadExpr(const std::string& text, std::string& error) { return IRTextLoader().LoadExpr(text, error); }
TypePtr TextLoadType(const std::string& text, std::string& error) { return IRTextLoader().LoadType(text, error); }

} // namespace ir
} // namespace pypto
