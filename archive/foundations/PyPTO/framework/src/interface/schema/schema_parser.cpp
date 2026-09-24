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
 * \file schema_parser.cpp
 * \brief
 */

#include "interface/schema/schema.h"
#include <vector>
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::schema {
std::string SchemaNode::Dump() const
{
    std::ostringstream oss;
    if (name == "") {
        oss << "[";
        for (size_t i = 0; i < size(); i++) {
            oss << ((i == 0) ? "" : ",");
            oss << at(i)->Dump();
        }
        oss << "]";
    } else {
        oss << GetName();
        if (size() != 0) {
            oss << "{";
            for (size_t i = 0; i < size(); i++) {
                oss << ((i == 0) ? "" : ",");
                oss << at(i)->Dump();
            }
            oss << "}";
        }
    }
    return oss.str();
}

struct Parser {
    const std::string& text;
    int base;

    Parser(const std::string& text_, int base_) : text(text_), base(base_) {}

    struct Token {
        static constexpr int id = 0;

        int kind;
        std::string text;
        Token(int kind_, const std::string& text_ = "") : kind(kind_), text(text_) {}

        int Kind() { return kind; }
        std::string Text() { return text; }
    };
    std::vector<Token> tokenList;

    void Tokenization()
    {
        std::string curr;
        for (size_t idx = base; idx < text.size(); idx++) {
            switch (text[idx]) {
                case '#':
                case ',':
                case '[':
                case ']':
                case '{':
                case '}': {
                    if (curr.size() != 0) {
                        tokenList.emplace_back(Token::id, curr);
                        curr.clear();
                    }
                    tokenList.emplace_back(text[idx]);
                } break;
                case ' ': {
                    if (curr.size() != 0) {
                        tokenList.emplace_back(Token::id, curr);
                        curr.clear();
                    }
                } break;
                default:
                    curr.push_back(text[idx]);
                    break;
            }
        }
        if (curr.size() != 0) {
            tokenList.emplace_back(Token::id, curr);
            curr.clear();
        }
    }

    int pos = 0;
    Token& Current() { return tokenList[pos]; }
    bool Accessible() { return pos < (int)tokenList.size(); }
    void MoveNext() { pos++; }

    std::shared_ptr<SchemaNode> ParseNode()
    {
        std::shared_ptr<SchemaNode> curr;
        if (Current().Kind() == '[') {
            curr = std::make_shared<SchemaNode>("");
            MoveNext();
            for (;;) {
                auto child = ParseNode();
                curr->push_back(child);
                if (Current().Kind() == ',') {
                    MoveNext();
                } else if (Current().Kind() == ']') {
                    MoveNext();
                    break;
                } else {
                    // invalid format
                    FE_ASSERT(FeError::INVALID_VAL, false);
                }
            }
        } else {
            FE_ASSERT(FeError::INVALID_VAL, Current().Kind() == Token::id);
            curr = std::make_shared<SchemaNode>(Current().Text());
            MoveNext();
            if (Current().Kind() == '{') {
                MoveNext();
                for (;;) {
                    auto child = ParseNode();
                    curr->push_back(child);
                    if (Current().Kind() == ',') {
                        MoveNext();
                    } else if (Current().Kind() == '}') {
                        MoveNext();
                        break;
                    } else {
                        // invalid format
                        FE_ASSERT(FeError::INVALID_VAL, false);
                    }
                }
            } else if (Current().Kind() == ',' || Current().Kind() == ']' || Current().Kind() == '}') {
                // only id
            } else {
                // invalid format
                FE_ASSERT(FeError::INVALID_VAL, false);
            }
        }
        return curr;
    }

    std::vector<std::shared_ptr<SchemaNode>> Parse()
    {
        Tokenization();
        std::vector<std::shared_ptr<SchemaNode>> nodeList;
        pos = 0;
        while (Accessible()) {
            while (Accessible() && Current().Kind() != '#') {
                MoveNext();
            }
            if (!Accessible()) {
                break;
            }
            FE_ASSERT(FeError::INVALID_VAL, Current().Kind() == '#');
            MoveNext();
            nodeList.push_back(ParseNode());
        }
        return nodeList;
    }
};

std::vector<std::shared_ptr<SchemaNode>> SchemaNode::ParseSchema(const std::string& schema)
{
    auto pos = schema.find(DEV_TRACE_PREFIX);
    if (pos == std::string::npos) {
        return {};
    }

    pos += std::string(DEV_TRACE_PREFIX).size();
    std::vector<std::shared_ptr<SchemaNode>> nodeList = Parser(schema, pos).Parse();
    return nodeList;
}

std::vector<std::shared_ptr<SchemaNode>> SchemaNode::ParseSchema(const std::vector<std::string>& schemaList)
{
    std::vector<std::shared_ptr<SchemaNode>> nodeList;
    for (auto& schema : schemaList) {
        auto childList = ParseSchema(schema);
        nodeList.insert(nodeList.end(), childList.begin(), childList.end());
    }
    return nodeList;
}

static void BuildSchemaDict(std::map<std::string, std::vector<std::shared_ptr<SchemaNode>>>& dict,
                            const std::shared_ptr<SchemaNode>& node)
{
    dict[node->GetName()].push_back(node);
    for (auto& child : *node) {
        BuildSchemaDict(dict, child);
    }
}

std::map<std::string, std::vector<std::shared_ptr<SchemaNode>>> SchemaNode::BuildDict(
    const std::vector<std::shared_ptr<SchemaNode>>& nodeList)
{
    std::map<std::string, std::vector<std::shared_ptr<SchemaNode>>> dict;
    for (auto& node : nodeList) {
        BuildSchemaDict(dict, node);
    }
    return dict;
}

} // namespace npu::tile_fwk::schema
