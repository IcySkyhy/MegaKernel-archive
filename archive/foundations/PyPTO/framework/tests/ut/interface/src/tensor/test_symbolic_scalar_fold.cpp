/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "gtest/gtest.h"
#include "interface/tensor/symbolic_scalar.h"

namespace npu::tile_fwk {
namespace test {

using ImmediateDiff = SymbolicExpressionTable::ImmediateDiff;

static RawSymbolicScalarPtr Imm(int64_t v) { return RawSymbolicImmediate::Create(v); }
static RawSymbolicScalarPtr Sym(const std::string& n) { return RawSymbolicSymbol::Create(n); }
static RawSymbolicScalarPtr Add(const RawSymbolicScalarPtr& a, const RawSymbolicScalarPtr& b)
{
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_ADD, std::vector<RawSymbolicScalarPtr>{a, b});
}
static RawSymbolicScalarPtr Mul(const RawSymbolicScalarPtr& a, const RawSymbolicScalarPtr& b)
{
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_MUL, std::vector<RawSymbolicScalarPtr>{a, b});
}

// 同指针 / 内容相等不同指针 / 单点浅差异 / 多处深差异 / 同值 imm 出现在不同位置无误报
TEST(TestFindAllImmediateDifferences, EqualAndDiffs)
{
    // 同指针 → true, empty
    auto e = Add(Sym("a"), Imm(4));
    std::vector<ImmediateDiff> diffs;
    EXPECT_TRUE(SymbolicExpressionTable::FindAllImmediateDifferences(e, e, diffs));
    EXPECT_TRUE(diffs.empty());

    // 内容相等不同指针 → true, empty；模板里 12 出现两次但只动右边那个 → 单点差异
    auto e1 = Add(Mul(Sym("x"), Imm(12)), Imm(12));
    auto e2 = Add(Mul(Sym("x"), Imm(12)), Imm(12));
    auto e3 = Add(Mul(Sym("x"), Imm(12)), Imm(99));
    diffs.clear();
    EXPECT_TRUE(SymbolicExpressionTable::FindAllImmediateDifferences(e1, e2, diffs));
    EXPECT_TRUE(diffs.empty());
    EXPECT_TRUE(SymbolicExpressionTable::FindAllImmediateDifferences(e1, e3, diffs));
    ASSERT_EQ(diffs.size(), 1u);
    EXPECT_EQ(diffs[0].path, std::vector<int>({1}));
    EXPECT_EQ(diffs[0].immLhs, 12);
    EXPECT_EQ(diffs[0].immRhs, 99);

    // 多处深差异：((x+4) * (y+100)) vs ((x+12) * (y+200))
    auto m1 = Mul(Add(Sym("x"), Imm(4)), Add(Sym("y"), Imm(100)));
    auto m2 = Mul(Add(Sym("x"), Imm(12)), Add(Sym("y"), Imm(200)));
    diffs.clear();
    EXPECT_TRUE(SymbolicExpressionTable::FindAllImmediateDifferences(m1, m2, diffs));
    ASSERT_EQ(diffs.size(), 2u);
    EXPECT_EQ(diffs[0].path, std::vector<int>({0, 1}));
    EXPECT_EQ(diffs[1].path, std::vector<int>({1, 1}));
}

// Symbol 名 / Opcode / Kind / Operand 数 任一不同 → 拒绝
TEST(TestFindAllImmediateDifferences, StructuralRejection)
{
    std::vector<ImmediateDiff> diffs;
    EXPECT_FALSE(
        SymbolicExpressionTable::FindAllImmediateDifferences(Add(Sym("a"), Imm(4)), Add(Sym("b"), Imm(4)), diffs));
    EXPECT_FALSE(
        SymbolicExpressionTable::FindAllImmediateDifferences(Add(Sym("a"), Imm(4)), Mul(Sym("a"), Imm(4)), diffs));
    EXPECT_FALSE(SymbolicExpressionTable::FindAllImmediateDifferences(Add(Sym("a"), Imm(4)), Imm(4), diffs));
    auto call1 = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL,
                                                         std::vector<RawSymbolicScalarPtr>{Sym("F"), Sym("a")});
    auto call2 = std::make_shared<RawSymbolicExpression>(
        SymbolicOpcode::T_MOP_CALL, std::vector<RawSymbolicScalarPtr>{Sym("F"), Sym("a"), Sym("b")});
    EXPECT_FALSE(SymbolicExpressionTable::FindAllImmediateDifferences(call1, call2, diffs));
}

// 浅位置 / 深位置 / 多处替换 / affine 子树替换
TEST(TestBuildExpressionWithPlaceholders, SingleMultiAndAffine)
{
    auto k = Sym("sym_k");

    // 浅位置单替换：(a + 4) → 立即数被覆盖
    auto t1 = Add(Sym("a"), Imm(4));
    auto out1 = SymbolicExpressionTable::BuildExpressionWithPlaceholders(t1, {{{1}, k}});
    EXPECT_NE(out1.find("sym_k"), std::string::npos);
    EXPECT_EQ(out1.find("4"), std::string::npos);

    // 多处深替换：((x+4) * (y+100))，仅替换叶子立即数，旁路 x / y 保持
    auto t2 = Mul(Add(Sym("x"), Imm(4)), Add(Sym("y"), Imm(100)));
    auto out2 = SymbolicExpressionTable::BuildExpressionWithPlaceholders(t2, {{{0, 1}, k}, {{1, 1}, k}});
    EXPECT_EQ(out2.find("4"), std::string::npos);
    EXPECT_EQ(out2.find("100"), std::string::npos);
    EXPECT_NE(out2.find("x"), std::string::npos);
    EXPECT_NE(out2.find("y"), std::string::npos);

    // affine 子树占位：(a + (10 + 2*k)) —— 10 / 2 / k 全部出现
    auto affine = Add(Imm(10), Mul(Imm(2), k));
    auto out3 = SymbolicExpressionTable::BuildExpressionWithPlaceholders(t1, {{{1}, affine}});
    EXPECT_NE(out3.find("sym_k"), std::string::npos);
    EXPECT_NE(out3.find("10"), std::string::npos);
    EXPECT_NE(out3.find("2"), std::string::npos);
}

// 结构排序的关键不变式
// - Kind 主导（Immediate 排 Symbol 之前，与值无关）
// - 同 kind 同 Opcode 时，等差家族 a+4, a+8, a+12 在 CompareRaw 下严格递增（折叠依赖此不变式）
TEST(TestCompareRaw, OrderingInvariants)
{
    EXPECT_LT(SymbolicExpressionTable::CompareRaw(Imm(99999), Sym("a")), 0);
    auto a4 = Add(Sym("a"), Imm(4));
    auto a8 = Add(Sym("a"), Imm(8));
    auto a12 = Add(Sym("a"), Imm(12));
    EXPECT_LT(SymbolicExpressionTable::CompareRaw(a4, a8), 0);
    EXPECT_LT(SymbolicExpressionTable::CompareRaw(a8, a12), 0);
}

static RawSymbolicScalarPtr CallGetInputShapeDim(const std::string& argName, int64_t dim)
{
    return std::make_shared<RawSymbolicExpression>(
        SymbolicOpcode::T_MOP_CALL,
        std::vector<RawSymbolicScalarPtr>{Sym("RUNTIME_GetInputShapeDim"), Sym(argName), Imm(dim)});
}

static RawSymbolicScalarPtr RuntimeAnd(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs)
{
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL,
                                                   std::vector<RawSymbolicScalarPtr>{Sym("RUNTIME_And"), lhs, rhs});
}

// CSE map 命中只发生在 GetInput CALL；外层 Add/And 不得把自己整树展开当 key，叶子仍替换。
TEST(TestBuildExpressionCse, LookupOnlyGetInputCallNodes)
{
    auto shape0 = CallGetInputShapeDim("ARG_x", 0);
    const std::string key = SymbolicExpressionTable::BuildExpression(shape0);
    std::unordered_map<std::string, std::string> cse{{key, "CSE_sd[0]"}};

    EXPECT_EQ(SymbolicExpressionTable::BuildExpression(shape0, &cse), "CSE_sd[0]");

    auto wrapped = Add(shape0, Imm(1));
    const std::string wrappedStr = SymbolicExpressionTable::BuildExpression(wrapped, &cse);
    EXPECT_NE(wrappedStr.find("CSE_sd[0]"), std::string::npos);
    EXPECT_EQ(wrappedStr.find("RUNTIME_GetInputShapeDim"), std::string::npos);

    auto chain = shape0;
    for (int i = 0; i < 32; ++i) {
        chain = RuntimeAnd(chain, shape0);
    }
    const std::string chainStr = SymbolicExpressionTable::BuildExpression(chain, &cse);
    EXPECT_NE(chainStr.find("CSE_sd[0]"), std::string::npos);
    EXPECT_EQ(chainStr.find("RUNTIME_GetInputShapeDim"), std::string::npos);
    EXPECT_NE(chainStr.find("RUNTIME_And"), std::string::npos);
}

} // namespace test
} // namespace npu::tile_fwk
