/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mix_internal_components_analyzer.cpp
 * \brief Unit test for MixInternalComponentsAnalyzer
 */
#include <gtest/gtest.h>
#include "passes/block_graph_pass/mix_subgraph_split/mix_internal_components_analyzer.h"
#include "computational_graph_builder.h"
#include "passes/pass_utils/pass_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/tensor/irbuilder.h"

#define IS_SYNC_OPERATION(op)                                                                               \
    ((op) && ((op)->GetOpcode() == Opcode::OP_SYNC_SRC || (op)->GetOpcode() == Opcode::OP_SYNC_DST ||       \
              (op)->GetOpcode() == Opcode::OP_CV_SYNC_SRC || (op)->GetOpcode() == Opcode::OP_CV_SYNC_DST || \
              (op)->GetOpcode() == Opcode::OP_PHASE1 || (op)->GetOpcode() == Opcode::OP_PHASE2 ||           \
              (op)->GetOpcode() == Opcode::OP_BAR_V || (op)->GetOpcode() == Opcode::OP_BAR_M ||             \
              (op)->GetOpcode() == Opcode::OP_BAR_ALL))

namespace npu {
namespace tile_fwk {
// 全局常量定义
constexpr int MS_NUM0 = 0;
constexpr int MS_NUM1 = 1;
constexpr int MS_NUM2 = 2;
constexpr int MS_NUM3 = 3;
constexpr int MS_NEG1 = -1;
constexpr int MS_TENSOR_DIM = 16;
constexpr int64_t MS_SUB_BLOCK_IDX0 = 0;
constexpr int64_t MS_SUB_BLOCK_IDX1 = 1;

// 测试工具类：封装所有场景构建+结果校验辅助函数
namespace test_utils {
// 场景构建：创建Cube算子（isCube=true，指定InternalSubgraphID）
Operation& CreateCubeOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                        std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId);

// 场景构建：创建Vector算子（无isCube，指定AIVCore和InternalSubgraphID）
Operation& CreateVectorOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                          std::shared_ptr<LogicalTensor>& outTensor, AIVCore aivCore, int internalSubgraphId);

// 场景构建：创建同步算子（初始InternalSubgraphID=-1，指定同步Opcode）
Operation& CreateSyncOp(Function& mixFunc, Opcode syncOpcode, std::shared_ptr<LogicalTensor>& inTensor,
                        std::shared_ptr<LogicalTensor>& outTensor);

// 场景构建：创建COPY_IN算子（PHASE1/PHASE2合并专用）
Operation& CreateCopyInOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                          std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId);

// 场景构建：创建L0C_COPY_UB算子（CubeScope专用，subBlockIdx设置）
Operation& CreateL0CCopyUbOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                             std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId);

// 场景构建：创建基础张量（默认FP32，16*16）
std::shared_ptr<LogicalTensor> CreateBasicTensor();

// 结果校验：Scope基础信息（数量、ID、类型）
void VerifyScopeBasicInfo(const std::vector<InternalComponentInfo>& components, int expectedCount,
                          const std::vector<int>& expectedInternalIds, const std::vector<ComponentType>& expectedTypes);

// 结果校验：Scope内算子属性（数量、isCube、AIVCore）
void VerifyScopeOperands(const InternalComponentInfo& component, int expectedOpCount, bool isCube,
                         AIVCore expectedAivCore);

bool IsSyncOperation(const Operation* op);

// 结果校验：L0C_COPY_UB算子的subBlockIdx属性
void VerifyL0CCopyUbSubBlockIdx(Operation& copyUbOp, int64_t expectedSubBlockIdx);

// 结果校验：算子的InternalSubgraphID是否符合预期
void VerifyOpInternalId(const Operation& op, int expectedId);
} // namespace test_utils

// 测试基类（统一SetUp/TearDown，复用分析器和Mix子图）
class MixInternalComponentsAnalyzerTest : public ::testing::Test {
public:
    static void SetUpTestCase()
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    static void TearDownTestCase()
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void SetUp() override
    {
        // 初始化核心测试对象
        analyzer_ = std::make_shared<MixInternalComponentsAnalyzer>();
        mixFuncPtr_ = std::make_shared<Function>(Program::GetInstance(), "test_mix_func", "test_mix_func", nullptr);
        // 设置Mix子图基础属性
        mixFuncPtr_->SetGraphType(GraphType::BLOCK_GRAPH);
        mixFuncPtr_->SetFunctionType(FunctionType::STATIC);
    }

    void TearDown() override
    {
        // 资源释放
        analyzer_.reset();
        mixFuncPtr_.reset();
    }

protected:
    std::shared_ptr<MixInternalComponentsAnalyzer> analyzer_; // 核心分析器
    std::shared_ptr<Function> mixFuncPtr_;                    // 测试用Mix子图
};

// -------------------------- 辅助函数实现 --------------------------
namespace test_utils {
Operation& CreateCubeOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                        std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId)
{
    auto& op = IRBuilder().CreateTensorOpStmt(mixFunc, Opcode::OP_A_MUL_B, {inTensor}, {outTensor});
    op.UpdateInternalSubgraphID(internalSubgraphId);
    op.SetAttr(OpAttributeKey::isCube, true);
    op.SetAIVCore(AIVCore::UNSPECIFIED);
    return op;
}

Operation& CreateVectorOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                          std::shared_ptr<LogicalTensor>& outTensor, AIVCore aivCore, int internalSubgraphId)
{
    auto& op = IRBuilder().CreateTensorOpStmt(mixFunc, Opcode::OP_ADD, {inTensor}, {outTensor});
    op.UpdateInternalSubgraphID(internalSubgraphId);
    op.SetAIVCore(aivCore);
    return op;
}

Operation& CreateSyncOp(Function& mixFunc, Opcode syncOpcode, std::shared_ptr<LogicalTensor>& inTensor,
                        std::shared_ptr<LogicalTensor>& outTensor)
{
    auto& op = IRBuilder().CreateTensorOpStmt(mixFunc, syncOpcode, {inTensor}, {outTensor});
    op.UpdateInternalSubgraphID(MS_NEG1); // 初始无ID，触发未分配逻辑
    return op;
}

Operation& CreateCopyInOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                          std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId)
{
    auto& op = IRBuilder().CreateTensorOpStmt(mixFunc, Opcode::OP_COPY_IN, {inTensor}, {outTensor});
    op.UpdateInternalSubgraphID(internalSubgraphId);
    op.SetAIVCore(AIVCore::AIV0);
    return op;
}

Operation& CreateL0CCopyUbOp(Function& mixFunc, std::shared_ptr<LogicalTensor>& inTensor,
                             std::shared_ptr<LogicalTensor>& outTensor, int internalSubgraphId)
{
    auto& op = IRBuilder().CreateTensorOpStmt(mixFunc, Opcode::OP_L0C_COPY_UB, {inTensor}, {outTensor});
    op.UpdateInternalSubgraphID(internalSubgraphId);
    op.SetAttr(OpAttributeKey::isCube, true);
    op.SetAIVCore(AIVCore::UNSPECIFIED);
    return op;
}

std::shared_ptr<LogicalTensor> CreateBasicTensor()
{
    return npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, Shape({MS_TENSOR_DIM, MS_TENSOR_DIM}),
                                                      CreateTestConstIntVector(Shape({MS_TENSOR_DIM, MS_TENSOR_DIM})));
}

void VerifyScopeBasicInfo(const std::vector<InternalComponentInfo>& components, int expectedCount,
                          const std::vector<int>& expectedInternalIds, const std::vector<ComponentType>& expectedTypes)
{
    ASSERT_EQ(components.size(), expectedCount) << "Scope count mismatch";
    for (int i = 0; i < expectedCount; ++i) {
        const auto& comp = components[i];
        EXPECT_EQ(comp.internalSubgraphID, expectedInternalIds[i]) << "Scope ID mismatch at index " << i;
        EXPECT_EQ(comp.componentType, expectedTypes[i]) << "Scope type mismatch at index " << i;
        EXPECT_EQ(comp.suffix, "_internal_" + std::to_string(expectedInternalIds[i]))
            << "Scope suffix mismatch at index " << i;
    }
}

void VerifyScopeOperands(const InternalComponentInfo& component, int expectedOpCount, bool isCube,
                         AIVCore expectedAivCore)
{
    ASSERT_EQ(component.operations.size(), expectedOpCount) << "Scope inner op count mismatch";
    for (const auto* op : component.operations) {
        ASSERT_NE(op, nullptr) << "Null op found in scope " << component.internalSubgraphID;
        if (isCube) {
            EXPECT_TRUE(op->HasAttribute(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube))
                << "Cube scope op missing isCube=true";
        } else {
            EXPECT_TRUE(!op->HasAttribute(OpAttributeKey::isCube) || !op->GetBoolAttribute(OpAttributeKey::isCube))
                << "Vector scope op has unexpected isCube attribute";
        }
        if (!IsSyncOperation(op)) {
            EXPECT_EQ(op->GetAIVCore(), expectedAivCore) << "Scope op AIVCore mismatch";
        }
    }
}

bool IsSyncOperation(const Operation* op) { return IS_SYNC_OPERATION(op); }

void VerifyL0CCopyUbSubBlockIdx(Operation& copyUbOp, int64_t expectedSubBlockIdx)
{
    EXPECT_TRUE(copyUbOp.HasAttribute(OpAttributeKey::subBlockIdx)) << "L0C_COPY_UB missing subBlockIdx attr";
    int64_t actualSubBlockIdx = copyUbOp.GetIntAttribute(OpAttributeKey::subBlockIdx);
    EXPECT_EQ(actualSubBlockIdx, expectedSubBlockIdx) << "L0C_COPY_UB subBlockIdx mismatch";
}

void VerifyOpInternalId(const Operation& op, int expectedId)
{
    EXPECT_EQ(op.GetInternalSubgraphID(), expectedId) << "Op " << op.GetOpcodeStr() << " internalID mismatch";
}
} // namespace test_utils

// -------------------------- 基础正常场景用例（覆盖核心流程） --------------------------
// 用例1：单Cube Scope基础划分（无同步算子，纯Cube算子）
TEST_F(MixInternalComponentsAnalyzerTest, TestSingleCubeScopeBasicSplit)
{
    // 1. 构建场景
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    auto t4 = test_utils::CreateBasicTensor();
    test_utils::CreateCubeOp(*mixFuncPtr_, t1, t2, MS_NUM0);
    test_utils::CreateCubeOp(*mixFuncPtr_, t2, t3, MS_NUM0);
    test_utils::CreateCubeOp(*mixFuncPtr_, t3, t4, MS_NUM0);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "Single Cube scope analyze failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::C_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM3, true, AIVCore::UNSPECIFIED);
}

// 用例2：单Vector Scope基础划分（含NOP隐式跳过，纯Vector算子）
TEST_F(MixInternalComponentsAnalyzerTest, TestSingleVectorScopeBasicSplit)
{
    // 1. 构建场景
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    test_utils::CreateVectorOp(*mixFuncPtr_, t1, t2, AIVCore::AIV0, MS_NUM0);
    test_utils::CreateVectorOp(*mixFuncPtr_, t2, t3, AIVCore::AIV0, MS_NUM0);
    // 插入NOP算子（验证隐式跳过逻辑）
    IRBuilder().CreateTensorOpStmt(*mixFuncPtr_, Opcode::OP_NOP, {}, {});

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "Single Vector scope analyze failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::V_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM2, false, AIVCore::AIV0);
}

// 用例3：多Scope基础划分（1Cube+1Vector，无交叉，非连续ID）
TEST_F(MixInternalComponentsAnalyzerTest, TestMultiScopeBasicSplit_Cube_Vector)
{
    // 1. 构建场景（ID=0:Cube，ID=2:Vector，非连续ID验证）
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    auto t4 = test_utils::CreateBasicTensor();
    test_utils::CreateCubeOp(*mixFuncPtr_, t1, t2, MS_NUM0);
    test_utils::CreateVectorOp(*mixFuncPtr_, t3, t4, AIVCore::AIV1, MS_NUM2);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "Multi scope analyze failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM2, {MS_NUM0, MS_NUM2},
                                     {ComponentType::C_SCOPE, ComponentType::V_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM1, true, AIVCore::UNSPECIFIED);
    test_utils::VerifyScopeOperands(components[1], MS_NUM1, false, AIVCore::AIV1);
}

// -------------------------- 同步算子合并场景用例  --------------------------
// 用例4：同步算子OP_SYNC_SRC合并（向前找非同步算子）
TEST_F(MixInternalComponentsAnalyzerTest, TestSyncOpMerge_SyncSrc_Backward)
{
    // 1. 构建场景：Vector(ID=0) + OP_SYNC_SRC(初始-1)
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    test_utils::CreateVectorOp(*mixFuncPtr_, t1, t2, AIVCore::AIV0, MS_NUM0);
    auto& syncSrcOp = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_SYNC_SRC, t2, t3);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "OP_SYNC_SRC merge failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::V_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM2, false, AIVCore::AIV0);
    test_utils::VerifyOpInternalId(syncSrcOp, MS_NUM0);
}

// 用例5：同步算子OP_BAR_ALL合并（向后找非同步算子）
TEST_F(MixInternalComponentsAnalyzerTest, TestSyncOpMerge_BarAll_Forward)
{
    // 1. 构建场景：OP_BAR_ALL(初始-1) + Vector(ID=2)
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    auto& barAllOp = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_BAR_ALL, t1, t2);
    test_utils::CreateVectorOp(*mixFuncPtr_, t2, t3, AIVCore::AIV1, MS_NUM2);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "OP_BAR_ALL merge failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM2, {MS_NUM2, MS_NUM3},
                                     {ComponentType::V_SCOPE, ComponentType::C_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM1, false, AIVCore::AIV1);
    test_utils::VerifyOpInternalId(barAllOp, MS_NUM3);
}

// 用例6：同步算子OP_PHASE2合并（向前找COPY_IN算子）
TEST_F(MixInternalComponentsAnalyzerTest, TestSyncOpMerge_Phase2_CopyIn)
{
    // 1. 构建场景：COPY_IN(ID=1) + OP_PHASE2(初始-1)
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    test_utils::CreateCopyInOp(*mixFuncPtr_, t1, t2, MS_NUM1);
    auto& phase2Op = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_PHASE2, t2, t3);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "OP_PHASE2 merge failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM1}, {ComponentType::V_SCOPE});
    test_utils::VerifyScopeOperands(components[0], MS_NUM2, false, AIVCore::AIV0);
    test_utils::VerifyOpInternalId(phase2Op, MS_NUM1);
}

// -------------------------- CubeScope L0C_COPY_UB处理用例  --------------------------
// 用例7：CubeScope含L0C_COPY_UB，下游连接AIV1 Vector（subBlockIdx=1）
TEST_F(MixInternalComponentsAnalyzerTest, TestCubeScope_WithL0CCopyUb_AIV1)
{
    // 1. 构建场景：L0C_COPY_UB(ID=0,Cube) → Vector(ID=1,AIV1)
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    auto& copyUbOp = test_utils::CreateL0CCopyUbOp(*mixFuncPtr_, t1, t2, MS_NUM0);
    test_utils::CreateVectorOp(*mixFuncPtr_, t2, t3, AIVCore::AIV1, MS_NUM1);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, SUCCESS) << "CubeScope L0C_COPY_UB process failed";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM2, {MS_NUM0, MS_NUM1},
                                     {ComponentType::C_SCOPE, ComponentType::V_SCOPE});
    test_utils::VerifyL0CCopyUbSubBlockIdx(copyUbOp, MS_SUB_BLOCK_IDX1);
}

// -------------------------- 异常校验场景用例（覆盖所有失败分支） --------------------------
// 用例8：Scope内isCube属性不一致（校验失败，返回FAILED）
TEST_F(MixInternalComponentsAnalyzerTest, TestException_InconsistentIsCube)
{
    // 1. 构建场景：2个算子同ID=0，一个isCube=true，一个isCube=false
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    test_utils::CreateCubeOp(*mixFuncPtr_, t1, t2, MS_NUM0);
    auto& vecOp = test_utils::CreateVectorOp(*mixFuncPtr_, t2, t3, AIVCore::AIV0, MS_NUM0);
    vecOp.SetAttr(OpAttributeKey::isCube, false); // 制造属性不一致

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, FAILED) << "Should return FAILED when isCube inconsistent";
}

// 用例9：V_SCOPE内AIVCore属性不一致（校验失败，返回FAILED）
TEST_F(MixInternalComponentsAnalyzerTest, TestException_InconsistentAIVCore)
{
    // 1. 构建场景：2个Vector算子同ID=1，一个AIV0，一个AIV1
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto t3 = test_utils::CreateBasicTensor();
    test_utils::CreateVectorOp(*mixFuncPtr_, t1, t2, AIVCore::AIV0, MS_NUM1);
    test_utils::CreateVectorOp(*mixFuncPtr_, t2, t3, AIVCore::AIV1, MS_NUM1); // 制造属性不一致

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, FAILED) << "Should return FAILED when AIVCore inconsistent";
}

// 用例10：Scope仅含同步算子（AIVCore=UNSPECIFIED，推断为C_SCOPE）
TEST_F(MixInternalComponentsAnalyzerTest, TestOnlySyncOpInComponent_CubeScope)
{
    // 1. 构建场景：仅OP_BAR_ALL算子，手动设置ID=0，AIVCore默认UNSPECIFIED
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto& barAllOp = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_BAR_ALL, t1, t2);
    barAllOp.UpdateInternalSubgraphID(MS_NUM0);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验：全同步算子+AIVCore=UNSPECIFIED → C_SCOPE
    ASSERT_EQ(status, SUCCESS) << "All-sync component with UNSPECIFIED AIVCore should succeed as C_SCOPE";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::C_SCOPE});
    ASSERT_EQ(components[0].operations.size(), MS_NUM1);
    EXPECT_EQ(components[0].aivCore, AIVCore::UNSPECIFIED);
}

// 用例10b：Scope仅含同步算子（AIVCore=AIV1，推断为V_SCOPE）
TEST_F(MixInternalComponentsAnalyzerTest, TestOnlySyncOpInComponent_VectorScope)
{
    // 1. 构建场景：仅OP_BAR_ALL算子，手动设置ID=0和AIVCore=AIV1
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto& barAllOp = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_BAR_ALL, t1, t2);
    barAllOp.UpdateInternalSubgraphID(MS_NUM0);
    barAllOp.SetAIVCore(AIVCore::AIV1);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验：全同步算子+AIVCore=AIV1 → V_SCOPE
    ASSERT_EQ(status, SUCCESS) << "All-sync component with AIV1 should succeed as V_SCOPE";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::V_SCOPE});
    EXPECT_EQ(components[0].aivCore, AIVCore::AIV1);
}

// 用例11：同步算子无合并目标（通过AIVCore创建新component，返回SUCCESS）
TEST_F(MixInternalComponentsAnalyzerTest, TestSyncOpMergeFail_CreateNewComponent)
{
    // 1. 构建场景：仅OP_SYNC_SRC算子，无任何非同步算子可合并，AIVCore默认UNSPECIFIED
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();
    auto& syncSrcOp = test_utils::CreateSyncOp(*mixFuncPtr_, Opcode::OP_SYNC_SRC, t1, t2);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验：无匹配component时按AIVCore创建新component（UNSPECIFIED→C_SCOPE）
    ASSERT_EQ(status, SUCCESS) << "Sync op with no merge target should create new component by AIVCore";
    test_utils::VerifyScopeBasicInfo(components, MS_NUM1, {MS_NUM0}, {ComponentType::C_SCOPE});
    test_utils::VerifyOpInternalId(syncSrcOp, MS_NUM0);
}

// 用例12：同步算子无合并目标（前校验失败, 未标记subgraphID, 返回FAILED）
TEST_F(MixInternalComponentsAnalyzerTest, TestNonSyncFailed)
{
    // 1. 构建场景
    auto t1 = test_utils::CreateBasicTensor();
    auto t2 = test_utils::CreateBasicTensor();

    test_utils::CreateCubeOp(*mixFuncPtr_, t1, t2, MS_NEG1);

    // 2. 执行分析
    std::vector<InternalComponentInfo> components;
    Status status = analyzer_->AnalyzeInternalComponents(*mixFuncPtr_, components);

    // 3. 结果校验
    ASSERT_EQ(status, FAILED) << "Should return FAILED";
}
} // namespace tile_fwk
} // namespace npu
