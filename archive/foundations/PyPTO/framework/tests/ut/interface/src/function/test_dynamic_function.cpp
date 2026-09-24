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
 * \file test_dynamic_function.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <regex>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;
using namespace std;
using Json = nlohmann::json;

class DynamicFunctionTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

namespace {
// Constants to replace magic numbers
constexpr int LOOP_COUNT = 8;
constexpr int SECOND_LOOP_COUNT = 2;
constexpr int CONDITION_THRESHOLD = 6;
} // namespace

static bool HasDuplicateElem(const std::vector<std::vector<int>>& vec)
{
    std::set<std::vector<int>> seen;
    for (const auto& subvec : vec) {
        if (seen.find(subvec) != seen.end()) {
            return true;
        }
        seen.insert(subvec);
    }
    return false;
}

static bool HasSameIoSlots(std::shared_ptr<Function> func)
{
    auto& inSlots = func->GetSlotScope()->ioslot.incastSlot;
    auto& outSlots = func->GetSlotScope()->ioslot.incastSlot;
    return HasDuplicateElem(inSlots) || HasDuplicateElem(outSlots);
}

TEST_F(DynamicFunctionTest, TestSymbolic)
{
    {
        SymbolicScalar a("a");
        SymbolicScalar b("b");
        SymbolicScalar c = a + b;
        EXPECT_FALSE(c.ConcreteValid());
        EXPECT_EQ(c.Raw()->Kind(), SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION);

        SymbolicClosure closure;
        closure.Insert("a", 100);
        closure.Insert("b", 50);
        EXPECT_EQ(150, closure.Evaluate(c));
    }

    {
        SymbolicScalar a("a", 100);
        SymbolicScalar b("b", 50);
        {
            auto c0 = -a;
            EXPECT_EQ(c0.Concrete(), -100);

            c0 = +a;
            EXPECT_EQ(c0.Concrete(), 100);

            c0 = !a;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a + b;
            EXPECT_EQ(c0.Concrete(), 150);

            c0 = a - b;
            EXPECT_EQ(c0.Concrete(), 50);

            c0 = a * b;
            EXPECT_EQ(c0.Concrete(), 5000);

            c0 = a / b;
            EXPECT_EQ(c0.Concrete(), 2);

            c0 = a % b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a == b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a != b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = a < b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a <= b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a > b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = a >= b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = std::min(a, b);
            EXPECT_EQ(c0.Concrete(), 50);

            c0 = std::max(a, b);
            EXPECT_EQ(c0.Concrete(), 100);

            std::cout << c0 << "\n";
        }
        {
            SymbolicScalar c0 = a + 50;
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 150);
        }
        {
            SymbolicScalar c0 = 50 + a;
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 150);
        }
        {
            SymbolicScalar c0 = std::max(a, b);
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 100);
        }
        {
            SymbolicScalar c0 = std::min(a, b);
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 50);
        }
        {
            a.AsIntermediateVariable();
            EXPECT_TRUE(a.IsIntermediateVariable());
        }
    }
}

TEST_F(DynamicFunctionTest, TestDynOffset)
{
    SymbolicScalar b("b");
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {b, 0};

    FUNCTION("main", {}, {})
    {
        Tensor t(DT_FP32, {4, 4}, "t0");
        auto v = View(t, {1, 1}, {b, 2}, {b, 0});
        v.GetStorage()->UpdateOffset(TensorOffset(offset, dynoffset));
    }

    auto func = Program::GetInstance().GetFunctionByRawName("TENSOR_main");
    Tensor t(DT_FP32, {4, 4}, "t0");
    t.GetStorage()->UpdateOffset(TensorOffset(offset, dynoffset));
    auto tt = LogicalTensor::LoadJson(*func, {}, t.GetStorage()->DumpJson(*func));
    EXPECT_EQ(tt->GetShape(), t.GetStorage()->GetShape());
}

TEST_F(DynamicFunctionTest, TestLoopRange)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{16, 64};
    std::vector<int64_t> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int STATIC_LOOP_COUNT = 4;
    constexpr int DYNAMIC_LOOP_START = 0;
    constexpr int DYNAMIC_LOOP_END = 4;
    constexpr int DYNAMIC_LOOP_STEP = 1;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    int count = 0;

    FUNCTION("main", {a, b}, {c})
    {
        LOOP("D1", FunctionType::STATIC, k, LoopRange(STATIC_LOOP_COUNT))
        {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }

        LOOP("D2", FunctionType::DYNAMIC_LOOP, k, LoopRange(DYNAMIC_LOOP_START, DYNAMIC_LOOP_END))
        {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }

        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(DYNAMIC_LOOP_START, DYNAMIC_LOOP_END, DYNAMIC_LOOP_STEP))
        {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }
    }
    EXPECT_EQ(count, 3);
}

TEST_F(DynamicFunctionTest, TestOnlyExpression)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{16, 64};
    std::vector<int64_t> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    FUNCTION("main", {a, b}, {c})
    {
        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END))
        {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
        }
    }

#if ENABLE_HIDDENLOOP
    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_D3_Unroll1_PATH0_hiddenfunc0_root_11");
#else
    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_D3_Unroll1_PATH0_root_5");
#endif
    EXPECT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetCallopAttrList().size(), 1);
    auto attr = rootFunc->GetCallopAttrList().front();
    FE_LOGI("%s", attr->DumpAttr().c_str());
    auto attrStr = attr->DumpAttr(2);
    std::regex idRegex(R"(attr\[2\]\[\d+,(.*)\]\])");
    std::smatch idMatch;
    ASSERT_TRUE(std::regex_match(attrStr, idMatch, idRegex)) << "attr format mismatch: " << attrStr;
    EXPECT_EQ(idMatch[1].str(), "  0,(k*16), 16, 16, 16, 64, 16,RUNTIME_Min(RUNTIME_Max((64-(k*16)), 0), 16)");
}

TEST_F(DynamicFunctionTest, TestOnlySymbol)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(1, 64);

    std::vector<int64_t> shape{4, 64};
    std::vector<int64_t> childShape{1, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;

    FUNCTION("main", {a, b}, {c})
    {
        LOOP("DynSymbol", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END))
        {
            auto a0 = View(a, childShape, {k, 0});
            auto b0 = View(b, childShape, {k, 0});
            auto c0 = Add(a0, b0);
            Assemble(c0, {k, 0}, c);
        }
    }
#if ENABLE_HIDDENLOOP
    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_DynSymbol_Unroll1_PATH0_hiddenfunc0_root_11");
#else
    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_DynSymbol_Unroll1_PATH0_root_5");
#endif
    EXPECT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetCallopAttrList().size(), 1);
    auto attr = rootFunc->GetCallopAttrList().front();
    FE_LOGI("%s", attr->DumpAttr().c_str());
    auto attrStr = attr->DumpAttr(2);
    std::regex idRegex(R"(attr\[2\]\[\d+,(.*)\]\])");
    std::smatch idMatch;
    ASSERT_TRUE(std::regex_match(attrStr, idMatch, idRegex)) << "attr format mismatch: " << attrStr;
    EXPECT_EQ(idMatch[1].str(), "  k,  0,  1, 64,  4, 64,RUNTIME_Min(RUNTIME_Max((4-k), 0), 1), 64");
}

void TestHybridLoopIf(const Tensor& t0, const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                      Tensor& out)
{
    Tensor r0;
    SymbolicScalar loopCount = 0;
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    FUNCTION("main", {t0, t1, t2, t3, t4}, {out})
    {
        config::SetBuildStatic(true);
        FUNCTION("spre") { r0 = Add(t0, t1); }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            TileShape::Current().SetVecTile(32, 32);
            r0 = Mul(r0, t1); // +t0, +t1
            IF(i < CONDITION_THRESHOLD)
            {
                r0 = Sub(r0, t3); // +t2 * 6
                r0 = Sub(r0, t3); // +t2 * 6
            }
            ELSE
            {
                r0 = Sub(r0, t4); // +t3 * 8
                r0 = Add(r0, t4); // +t3 * 8
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(SECOND_LOOP_COUNT))
        {
            TileShape::Current().SetVecTile(32, 32);
            loopCount = loopCount + i;
            r0 = Add(r0, t4);
        }
        FUNCTION("spost")
        {
            TileShape::Current().SetVecTile(32, 32);
            r0 = Add(r0, t0);
            r0 = Add(r0, t1);
            r0 = Add(r0, t2);
            r0 = Add(r0, t3);
            out = Add(r0, t4);
        }
    }

    for (auto& ele : Program::GetInstance().GetFunctionMap()) {
        auto func = ele.second;
        std::shared_ptr<DynloopFunctionAttribute>& attr = func->GetDynloopAttribute();
        if (attr != nullptr) {
            auto node = attr->BuildPathNode();
            node->Dump();
            attr->DumpBranch();
        }
    }
}

void TestHybridLoopIf2(const Tensor& t0, const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                       Tensor& out)
{
    FUNCTION("main", {t0, t1, t2, t3, t4}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            auto r0 = Add(t0, t1);
            r0 = Mul(r0, t1); // +t0, +t1
            IF(i < CONDITION_THRESHOLD)
            {
                r0 = Sub(r0, t2); // +t2 * 6
            }
            ELSE
            {
                r0 = Sub(r0, t3); // +t3 * 8
            }
            out = Add(r0, t4);
        }
    }
}

void TestStaticLoopStatic(const Tensor& t0, const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                          Tensor& out, int s)
{
    SymbolicScalar GetInt32Value1("GetInt32Value1");
    SymbolicScalar GetInt32Value2("GetInt32Value2");
    SymbolicScalar blockTableAddr("blockTableAddr");
    SymbolicScalar batchAddr("batchAddr");
    Tensor r0;
    SymbolicScalar loopCount = 0;

    FUNCTION("main", {t0, t1, t2, t3, t4}, {out})
    {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            loopCount = loopCount + i;
            r0 = Add(t0, t1);
            r0 = Sub(r0, t2);
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t3, 1) / s))
        {
            loopCount = loopCount + i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(t3.Dim() / s))
        {
            loopCount = loopCount + i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("s1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            loopCount = loopCount + i;
            out = Sub(r0, t4);
        }
    }
    auto& functions = Program::GetInstance().GetFunctionMap();
    for (auto& [name, function] : functions) {
        if (name == "PROGRAM_ENTRY" || !function->IsGraphType(GraphType::TENSOR_GRAPH)) {
            continue;
        }
        size_t incastSize = function->inCasts_.size();
        size_t outcastSize = function->outCasts_.size();
        auto& scope = function->GetSlotScope();
        EXPECT_EQ(scope->ioslot.incastSlot.size(), incastSize);
        EXPECT_EQ(scope->ioslot.outcastSlot.size(), outcastSize);
        EXPECT_EQ(HasSameIoSlots(function), false);
    }
}

TEST_F(DynamicFunctionTest, TestStaticLoopStatic)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    // Representation
    int s = 32;
    int n = 1;
    int m = 1;
    int t3CountValue = 7;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s * t3CountValue}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor blockTable(DT_INT32, {16, 16}, "blockTable");
    Tensor batch(DT_INT32, {1, 1}, "batch");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    TestStaticLoopStatic(t0, t1, t2, t3, t4, out, s);
}

TEST_F(DynamicFunctionTest, TestHybridLoopIf)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    TestHybridLoopIf(t0, t1, t2, t3, t4, out);
    Program::GetInstance().DumpStack();
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->Dump();
    slotManager->BuildIncastOutcastLink("test");
    slotManager->GetOutputIndex(out);
    slotManager->GetOutputIndex(t4);
    slotManager->LookupSlotIndex({t0, t1, t2});
    TensorSlot slot;
    slotManager->TensorSlotRead(slot, t1.GetStorage());

    auto programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);

    auto programJson2 = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJson.dump(), programJson2.dump());
    Program::GetInstance().LoadJson(programJson2);

    auto programJson3 = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJson3.dump(), programJson2.dump());

    auto& functions = Program::GetInstance().GetFunctionMap();
    for (auto& [name, function] : functions) {
        if (name == "PROGRAM_ENTRY" || !function->IsGraphType(GraphType::TENSOR_GRAPH)) {
            continue;
        }
        size_t incastSize = function->inCasts_.size();
        size_t outcastSize = function->outCasts_.size();
        auto& scope = function->GetSlotScope();
        bool ret = HasSameIoSlots(function);
        EXPECT_EQ(scope->ioslot.incastSlot.size(), incastSize);
        EXPECT_EQ(scope->ioslot.outcastSlot.size(), outcastSize);
        EXPECT_EQ(ret, false);
    }
}

Tensor TestLoopWithRank(const Tensor& t0, Tensor& r0, Tensor& out, int s, int maxRank)
{
    FUNCTION("main", {t0, r0}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s), PowersOf2(maxRank))
        {
            Tensor t0v = View(t0, {s, s}, {s * i, 0});
            r0 = Add(t0v, r0);
        }
        config::SetBuildStatic(true);
        FUNCTION("S1") { out = Add(r0, Element(DataType::DT_FP32, 3.0)); }
    }
    return out;
}

Tensor TestLoopIfWithRank(const Tensor& t0, Tensor& r0, Tensor& out, int s, int maxRank)
{
    FUNCTION("main", {t0, r0}, {out})
    {
        auto len = GetInputShape(t0, 0) / s;

        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len), PowersOf2(maxRank))
        {
            IF(IsLoopBegin(i, 0))
            {
                IF(IsLoopEnd(i, len)) { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
                ELSE { r0 = Add(r0, Element(DataType::DT_FP32, 2.0)); }
            }
            ELSE
            {
                IF(IsLoopEnd(i, len)) { r0 = Add(r0, Element(DataType::DT_FP32, 0.0)); }
                ELSE
                {
                    Tensor t0v = View(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
        }
        config::SetBuildStatic(true);
        FUNCTION("S1") { out = Add(r0, Element(DataType::DT_FP32, 3.0)); }
    }
    return out;
}

Tensor TestLoopWithManualRank(const Tensor& t0, Tensor& r0, Tensor& out, int s, int maxRank)
{
    auto func = [](const Tensor& lt0, Tensor& lr0, Tensor& lout, int ls, const npu::tile_fwk::SymbolicScalar& i,
                   int r) {
        TileShape::Current().SetVecTile({ls * r, ls * r});
        Tensor t0v = View(lt0, {ls * r, ls}, {ls * i, 0});
        Tensor r0v = View(lr0, {ls * r, ls}, {ls * i, 0});
        Tensor tmp = Add(t0v, r0v);
        Assemble(tmp, {ls * i, 0}, lout);
    };

    FUNCTION("main", {t0, r0}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s), PowersOf2(maxRank))
        {
            UNROLL_DEFAULT { func(t0, r0, out, s, i, 1); }
            UNROLL(2) { func(t0, r0, out, s, i, 2); }
            UNROLL(3) { func(t0, r0, out, s, i, 3); }
        }
        config::SetBuildStatic(true);
        FUNCTION("S1") { out = Add(out, Element(DataType::DT_FP32, 1.0)); }
    }
    return out;
}

TEST_F(DynamicFunctionTest, TestLoopWithRank)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopWithRank(t0, r0, out, s, maxUnrollTimes);

#if ENABLE_HIDDENLOOP
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_TENSOR_main_loop_Unroll1_PATH0_4");
#else
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
#endif
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 6);
    for (auto& callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, maxUnrollTimes);
            EXPECT_EQ(loopAttr->pathList.size(), 1);
            maxUnrollTimes /= 2;
        }
    }
    for (auto& op : mainFunc->Operations()) {
        EXPECT_EQ(op.GetOpcode(), Opcode::OP_CALL);
    }
}

TEST_F(DynamicFunctionTest, TestLoopIfWithRank)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopIfWithRank(t0, r0, out, s, maxUnrollTimes);

#if ENABLE_HIDDENLOOP
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_TENSOR_main_loop_Unroll1_PATH0_4");
#else
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
#endif
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 6);
    for (auto& callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, maxUnrollTimes);
            EXPECT_EQ(loopAttr->pathList.size(), 4);
            maxUnrollTimes /= 2;
        }
    }

    std::unordered_set<string> outcastSymbol;
    for (auto outcast : mainFunc->outCasts_) {
        std::string outcastName = outcast->tensor->GetSymbol();
        EXPECT_EQ(outcastSymbol.count(outcastName), 0);
        outcastSymbol.insert(outcastName);
    }
}

TEST_F(DynamicFunctionTest, TestLoopWithManualRank)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {n * s, s}, "r0");
    Tensor out(DT_FP32, {n * s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopWithManualRank(t0, r0, out, s, maxUnrollTimes);

#if ENABLE_HIDDENLOOP
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_TENSOR_main_loop_Unroll1_PATH0_4");
#else
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
#endif
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 7);

    std::vector<int64_t> ranks = {16, 8, 4, 3, 2, 1};
    int idx = 0;
    for (auto& callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, ranks[idx++]);
            FE_LOGE(FeError::UNKNOWN, "unrollTimes: %d range: %s", loopAttr->unrollTimes,
                    loopAttr->loopRange.Dump().c_str());
            EXPECT_EQ(loopAttr->pathList.size(), 1);
        }
    }
}

TEST_F(DynamicFunctionTest, TestSymbolicScalarDumpLoad)
{
    const Json expJson = {2, 6, 2, 16, 3, 1, "RUNTIME_GetInputDataInt32Dim1", 1, "INPUT_actSeqs", 1, "bIdx", 0, 256};
    SymbolicScalar exp = LoadSymbolicScalar(expJson);
    EXPECT_EQ(expJson, ToJson(exp));
}

#if ENABLE_HIDDENLOOP
TEST_F(DynamicFunctionTest, HiddenLoop)
{
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    FUNCTION("Main", {t1, t2, t3}, {out})
    {
        // LOOP("L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
        //     (void)_;
        Tensor r0(DT_FP32, {n * s, m * s}, "r0");
        LOOP("L01", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(LOOP_COUNT))
        {
            (void)idx1;
            t2 = Add(t2, t3);
            out = Add(out, t1); // out=0+20*8=160
        }
        // LOOP("L02",FunctionType::DYNAMIC_LOOP,idx2,LoopRange(1)){
        //     (void)idx2;
        r0 = Add(t1, t2);
        out = Add(r0, t2);
        //     }
        // }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_Main_2"); // outest function
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1); // one hidden loop

    auto outerLoopFunc = mainFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopFunc->GetMagicName(), "TENSOR_TENSOR_Main_loop_Unroll1_3");
    EXPECT_EQ(outerLoopFunc->GetCalleeFunctionList().size(), 1);

    auto innerLoopFunc1 = outerLoopFunc->GetCalleeFunctionList()[0];
    std::vector<std::string> LoopPathFuncNames1 = {"TENSOR_L01_Unroll1_6",
                                                   "TENSOR_TENSOR_Main_loop_Unroll1_PATH0_hiddenfunc1_9"};
    int idx = 0;
    for (auto& LoopPathFuc1 : innerLoopFunc1->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", LoopPathFuc1->GetMagicName().c_str());
        EXPECT_EQ(LoopPathFuc1->GetMagicName(), LoopPathFuncNames1[idx++]);
    }

    auto innerLoopFunc2 = Program::GetInstance().GetFunctionByMagicName(
        "TENSOR_L01_Unroll1_PATH0_7"); // one of the innermost loops
    EXPECT_NE(innerLoopFunc2, nullptr);
    EXPECT_EQ(innerLoopFunc2->GetCalleeFunctionList().size(), 1); // Excessive hidden loop
}

void HiddenLoopWithIf(Tensor& t0, Tensor& t1, Tensor& out)
{
    FUNCTION("Main", {t0, t1}, {out})
    {
        // LOOP("L0",FunctionType::DYNAMIC_LOOP,_,LoopRange(1)){
        //     (void)_;
        //     LOOP("L01",FunctionType::DYNAMIC_LOOP,idx1,LoopRange(1)){
        //         (void)idx1;
        out = Add(t0, t1);
        //     }
        IF(SymbolicScalar(0) < SymbolicScalar("x"))
        {
            LOOP("L02", FunctionType::DYNAMIC_LOOP, _, LoopRange(SECOND_LOOP_COUNT))
            {
                (void)_;
                t0 = Add(t0, t1);
                out = Add(t0, out);
            }
        }
        ELSE
        {
            LOOP("L03", FunctionType::DYNAMIC_LOOP, _, LoopRange(SECOND_LOOP_COUNT))
            {
                (void)_;
                t0 = Add(t0, t1);
                out = Sub(t0, out);
            }
        }
        // }
    }
}

TEST_F(DynamicFunctionTest, HiddenLoopWithIf)
{
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    HiddenLoopWithIf(t0, t1, out);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_Main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1); // one hidden loop

    auto outerLoopFunc = mainFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopFunc->GetMagicName(), "TENSOR_TENSOR_Main_loop_Unroll1_3");
    EXPECT_EQ(outerLoopFunc->GetCalleeFunctionList().size(), 2); // one hidden loop has two path

    int idx1 = 0;
    std::vector<std::string> LoopPathFuncNames = {"TENSOR_TENSOR_Main_loop_Unroll1_PATH0_4",
                                                  "TENSOR_TENSOR_Main_loop_Unroll1_PATH1_10"};
    for (auto& LoopPathFuc : outerLoopFunc->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", LoopPathFuc->GetMagicName().c_str());
        EXPECT_EQ(LoopPathFuc->GetMagicName(), LoopPathFuncNames[idx1++]);
        EXPECT_EQ(LoopPathFuc->GetCalleeFunctionList().size(), 2);
    }

    auto innerLoopFunc1 = Program::GetInstance().GetFunctionByMagicName("TENSOR_TENSOR_Main_loop_Unroll1_PATH1_10");
    EXPECT_NE(innerLoopFunc1, nullptr);
    EXPECT_EQ(innerLoopFunc1->GetCalleeFunctionList().size(), 2);

    int idx2 = 0;
    std::vector<std::string> innerLoopPathFuncNames1 = {"TENSOR_TENSOR_Main_loop_Unroll1_PATH1_hiddenfunc0_11",
                                                        "TENSOR_L02_Unroll1_12"};
    for (auto& innerLoopPathFuc : innerLoopFunc1->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", innerLoopPathFuc->GetMagicName().c_str());
        EXPECT_EQ(innerLoopPathFuc->GetMagicName(), innerLoopPathFuncNames1[idx2++]);
    }

    auto innerLoopFunc2 = Program::GetInstance().GetFunctionByMagicName("TENSOR_TENSOR_Main_loop_Unroll1_PATH0_4");
    EXPECT_NE(innerLoopFunc2, nullptr);
    EXPECT_EQ(innerLoopFunc2->GetCalleeFunctionList().size(), 2);

    int idx3 = 0;
    std::vector<std::string> innerLoopPathFuncNames2 = {"TENSOR_TENSOR_Main_loop_Unroll1_PATH0_hiddenfunc0_5",
                                                        "TENSOR_L03_Unroll1_6"};
    for (auto& innerLoopPathFuc : innerLoopFunc2->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", innerLoopPathFuc->GetMagicName().c_str());
        EXPECT_EQ(innerLoopPathFuc->GetMagicName(), innerLoopPathFuncNames2[idx3++]);
    }

    auto innerLoopFunc3 = Program::GetInstance().GetFunctionByMagicName("TENSOR_L02_Unroll1_PATH0_13");
    EXPECT_NE(innerLoopFunc3, nullptr);
    EXPECT_EQ(innerLoopFunc3->GetCalleeFunctionList().size(), 1); // Excessive hidden loop
}

TEST_F(DynamicFunctionTest, HiddenLoopNestedWithIf)
{
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    FUNCTION("Main", {t0, t1, t2, t3, t4}, {out})
    {
        // LOOP("L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
        //     (void)_;
        // LOOP("L01",FunctionType::DYNAMIC_LOOP,idx1,LoopRange(1)){
        //     (void)idx1;
        IF(SymbolicScalar(0) < SymbolicScalar("x")) { t0 = Add(t1, t1); }
        ELSE { t0 = Add(t2, t2); }
        //}
        LOOP("L02", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(LOOP_COUNT))
        {
            (void)idx2;
            t0 = Add(t0, t4);
        }
        // LOOP("L03",FunctionType::DYNAMIC_LOOP,idx3,LoopRange(1)){
        //     (void)idx3;
        IF(SymbolicScalar(0) < SymbolicScalar("x")) { t3 = Mul(t0, t2); }
        ELSE { t3 = Sub(t0, t2); }
        if (SymbolicScalar(0) < 1) {
            out = Sub(t3, t0);
        }
        //}
        //  }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_Main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1); // one hidden loop

    auto outerLoopFunc = mainFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopFunc->GetMagicName(), "TENSOR_TENSOR_Main_loop_Unroll1_3");
    // const and duplicate cond will be optimized
    EXPECT_EQ(outerLoopFunc->GetCalleeFunctionList().size(), 2);

    int idx = 0;
    std::vector<std::string> LoopPathFuncNames = {
        "TENSOR_TENSOR_Main_loop_Unroll1_PATH0_4", "TENSOR_TENSOR_Main_loop_Unroll1_PATH1_10",
        "TENSOR_TENSOR_Main_loop_Unroll1_PATH2_16", "TENSOR_TENSOR_Main_loop_Unroll1_PATH3_22"};
    for (auto& LoopPathFuc : outerLoopFunc->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", LoopPathFuc->GetMagicName().c_str());
        EXPECT_EQ(LoopPathFuc->GetMagicName(), LoopPathFuncNames[idx++]);
        EXPECT_EQ(LoopPathFuc->GetCalleeFunctionList().size(), 3);
    }

    auto innerLoopFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_L02_Unroll1_PATH0_7");
    EXPECT_NE(innerLoopFunc, nullptr);
    EXPECT_EQ(innerLoopFunc->GetCalleeFunctionList().size(), 1); // Excessive hidden loop
}

TEST_F(DynamicFunctionTest, HiddenLoopNestedWithIfComplex)
{
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    FUNCTION("Main", {t0, t1, t2, t3, t4}, {out})
    {
        // LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        //    (void)i;
        IF(SymbolicScalar(0) < SymbolicScalar("x")) { t0 = Add(t1, t1); }
        ELSE { t0 = Add(t2, t2); }
        LOOP("L02", FunctionType::DYNAMIC_LOOP, k, LoopRange(LOOP_COUNT))
        {
            (void)k;
            t3 = Mul(t0, t2);
        }
        // LOOP("L03", FunctionType::DYNAMIC_LOOP, l, LoopRange(1)) {
        //      (void)l;
        out = Sub(t3, t0);
        // }
        LOOP("L04", FunctionType::DYNAMIC_LOOP, h, LoopRange(LOOP_COUNT))
        {
            (void)h;
            t0 = Mul(t0, t2);
        }
        // LOOP("L05", FunctionType::DYNAMIC_LOOP, q, LoopRange(1)) {
        //     (void)q;
        out = Add(out, t0);
        //}
        //  }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_Main_2"); // outest function
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1); // one hidden loop

    auto outerLoopFunc = mainFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopFunc->GetMagicName(), "TENSOR_TENSOR_Main_loop_Unroll1_3");
    EXPECT_EQ(outerLoopFunc->GetCalleeFunctionList().size(), 2); // one hidden loop has two paths

    int idx = 0;
    std::vector<std::string> LoopPathFuncNames = {"TENSOR_TENSOR_Main_loop_Unroll1_PATH0_4",
                                                  "TENSOR_TENSOR_Main_loop_Unroll1_PATH1_14"};
    for (auto& LoopPathFuc : outerLoopFunc->GetCalleeFunctionList()) {
        FE_LOGI("LoopPathFuc: %s", LoopPathFuc->GetMagicName().c_str());
        EXPECT_EQ(LoopPathFuc->GetMagicName(), LoopPathFuncNames[idx++]);
        EXPECT_EQ(LoopPathFuc->GetCalleeFunctionList().size(), 5);
    }

    auto innerLoopFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_L02_Unroll1_PATH0_7");
    EXPECT_NE(innerLoopFunc, nullptr);
    EXPECT_EQ(innerLoopFunc->GetCalleeFunctionList().size(), 1); // Excessive hidden loop
}
#endif

TEST_F(DynamicFunctionTest, TestGetInputDataInt32Dim3)
{
    TileShape::Current().SetVecTile(32, 32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t5(DT_INT32, {2, n * s, m * s}, "t5");
    Tensor out(DT_INT32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t5, 64.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    SymbolicScalar loopCount = 0;

    FUNCTION("main", {t5}, {out})
    {
        LOOP("s1", FunctionType::DYNAMIC_LOOP, i,
             LoopRange(GetTensorData(t5, {npu::tile_fwk::SymbolicScalar("0"), npu::tile_fwk::SymbolicScalar("1"),
                                          npu::tile_fwk::SymbolicScalar("2")}) /
                       s))
        {
            loopCount = loopCount + i;
            out = Add(t5, Element(DataType::DT_FP32, static_cast<double>(1.0)));
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> attr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
}

TEST_F(DynamicFunctionTest, TestGetInputDataInt32Dim4)
{
    TileShape::Current().SetVecTile(16, 16, 16, 16);

    int s = 16;
    int k = 1;
    int n = 1;
    int m = 1;
    Tensor t5(DT_INT32, {2, k * s, n * s, m * s}, "t5");
    Tensor out(DT_FP32, {k * s, n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t5, 64.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    SymbolicScalar loopCount = 0;

    FUNCTION("main", {t5}, {out})
    {
        LOOP("s1", FunctionType::DYNAMIC_LOOP, i,
             LoopRange(GetTensorData(t5, {npu::tile_fwk::SymbolicScalar("0"), npu::tile_fwk::SymbolicScalar("1"),
                                          npu::tile_fwk::SymbolicScalar("2"), npu::tile_fwk::SymbolicScalar("3")}) /
                       s))
        {
            loopCount = loopCount + i;
            out = Add(t5, Element(DataType::DT_FP32, static_cast<double>(1.0)));
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> attr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
}
