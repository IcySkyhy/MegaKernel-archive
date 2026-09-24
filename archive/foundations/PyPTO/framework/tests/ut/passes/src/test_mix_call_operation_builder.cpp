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
 * \file test_mix_call_operation_builder.cpp
 * \brief Unit test for MixCallOperationBuilder
 * */
#include <gtest/gtest.h>
#include "interface/tensor/irbuilder.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_call_operation_builder.h"
#include "computational_graph_builder.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {

constexpr int MS_NUM1 = 1;
constexpr int MS_NUM2 = 2;
constexpr int MS_NUM4 = 4;
constexpr int MS_NUM10 = 10;
constexpr int MS_NUM16 = 16;
constexpr uint64_t TEST_PROGRAM_ID = 100;
constexpr int32_t OP_MAGIC_BASE = 10000;
constexpr int TENSOR_DIMENSIONS = 2;

constexpr int OFFSET_INPUT1 = 100;
constexpr int OFFSET_INPUT2 = 101;
constexpr int OFFSET_ADD_INPUT1 = 200;
constexpr int OFFSET_ADD_INPUT2 = 201;
constexpr int OFFSET_ADD_OUTPUT = 300;
constexpr int OFFSET_OUTPUT = 400;

constexpr int COMPONENT_ID_0 = 0;
constexpr int COMPONENT_ID_1 = 1;

constexpr int TENSOR_INDEX_0 = 0;
constexpr int TENSOR_INDEX_1 = 1;
constexpr int TENSOR_INDEX_2 = 2;

class MixCallOperationBuilderTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig("KEY_ENABLE_COST_MODEL", false);

        rootFunc = std::make_shared<Function>(Program::GetInstance(), "test_root", "test_root", nullptr);
        rootFunc->rootFunc_ = rootFunc.get();

        builder = std::make_unique<MixCallOperationBuilder>();
    }

    void TearDown() override { builder.reset(); }

protected:
    // 测试场景结构体
    struct TestScenario {
        std::vector<int64_t> shape;
        std::shared_ptr<LogicalTensor> inputTensor1;
        std::shared_ptr<LogicalTensor> inputTensor2;
        std::shared_ptr<LogicalTensor> outputTensor;
        std::shared_ptr<Function> originalMixFunc;
        Operation* originalCallOp = nullptr;
        std::shared_ptr<CallOpAttribute> originalCallAttr;
    };

    // 传播依赖张量结构体
    struct PropagatedTensors {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
    };

    std::shared_ptr<Function> createSimpleFunction(const std::string& name)
    {
        auto func = std::make_shared<Function>(Program::GetInstance(), name, name, rootFunc.get());
        func->SetGraphType(GraphType::BLOCK_GRAPH);
        func->SetFunctionType(FunctionType::STATIC);
        return func;
    }

    std::shared_ptr<Function> createLeafFuncWithPropagatedTensors(const std::string& name, int componentId,
                                                                  const TestScenario& scenario,
                                                                  const PropagatedTensors& propagatedTensors)
    {
        auto leafFunc = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        leafFunc->inCasts_.push_back(scenario.inputTensor1);
        leafFunc->inCasts_.push_back(scenario.inputTensor2);
        leafFunc->outCasts_.push_back(scenario.outputTensor);
        addPropagatedIncastOutcast(leafFunc, propagatedTensors.input, propagatedTensors.output);

        createOperationsWithOffsets(leafFunc, componentId, scenario.inputTensor1, scenario.inputTensor2,
                                    scenario.outputTensor, shape);
        addPropagatedTensorOperations(leafFunc, componentId, propagatedTensors.input, propagatedTensors.output);

        return leafFunc;
    }

    Operation* createSimpleCallOp(Function& func)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        auto& callOp = IRBuilder().CreateTensorOpStmt(func, Opcode::OP_CALL, {input1, input2}, {output1});

        auto callAttr = std::make_shared<CallOpAttribute>();
        callOp.SetOpAttribute(callAttr);

        return &callOp;
    }

    std::vector<InternalComponentInfo> createMixedComponents()
    {
        std::vector<InternalComponentInfo> components;
        components.emplace_back(COMPONENT_ID_0, "comp_c_scope", AIVCore::AIV0, ComponentType::C_SCOPE);
        components.emplace_back(COMPONENT_ID_1, "comp_v_scope", AIVCore::AIV1, ComponentType::V_SCOPE);
        return components;
    }

    SubfuncInvokeInfoTy createInvokeInfoWithTensorParams(uint64_t programId,
                                                         const std::shared_ptr<LogicalTensor>& input1,
                                                         const std::shared_ptr<LogicalTensor>& input2,
                                                         const std::shared_ptr<LogicalTensor>& output)
    {
        SubfuncInvokeInfoTy invokeInfo;
        invokeInfo.UpdateProgramSubgraphId(programId);

        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        const std::vector<int64_t> offset = {0, 0};
        const std::vector<int64_t> rawShape = {MS_NUM16, MS_NUM16};

        invokeInfo.RecordConnection(-1, programId, TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
                                    input1, OP_MAGIC_BASE + MS_NUM1);
        invokeInfo.RecordConnection(-1, programId, TENSOR_INDEX_1, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
                                    input2, OP_MAGIC_BASE + MS_NUM2);

        SubfuncInvokeInfoTy::SuccessorIncastInfoTy emptySuccessorInfo;
        invokeInfo.RecordOutcast(programId, TENSOR_INDEX_0, TENSOR_INDEX_1, TENSOR_INDEX_0, emptySuccessorInfo, offset,
                                 shape, rawShape, DT_FP32, output, OP_MAGIC_BASE + MS_NUM4);

        invokeInfo.RecordTensorArg(TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32, false, input1,
                                   OP_MAGIC_BASE + MS_NUM1);
        invokeInfo.RecordTensorArg(TENSOR_INDEX_1, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32, false, input2,
                                   OP_MAGIC_BASE + MS_NUM2);
        invokeInfo.RecordTensorArg(TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32, true, output,
                                   OP_MAGIC_BASE + MS_NUM4);

        invokeInfo.DoFinishRecord();

        return invokeInfo;
    }

    SubfuncInvokeInfoTy createInvokeInfoWithIncastOutcast(uint64_t programId,
                                                          const std::shared_ptr<LogicalTensor>& input1,
                                                          const std::shared_ptr<LogicalTensor>& input2,
                                                          const std::shared_ptr<LogicalTensor>& output,
                                                          Function* leafFunc = nullptr)
    {
        SubfuncInvokeInfoTy invokeInfo;
        invokeInfo.UpdateProgramSubgraphId(programId);

        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        const std::vector<int64_t> offset = {0, 0};
        const std::vector<int64_t> rawShape = {MS_NUM16, MS_NUM16};

        int input1OpMagic = OP_MAGIC_BASE + MS_NUM1;
        int input2OpMagic = OP_MAGIC_BASE + MS_NUM2;
        int outputOpMagic = OP_MAGIC_BASE + MS_NUM4;

        if (leafFunc) {
            auto operations = leafFunc->Operations(false);
            for (auto& op : operations) {
                auto iOperands = op.GetIOperands();
                for (size_t i = 0; i < iOperands.size(); i++) {
                    if (iOperands[i] == input1) {
                        input1OpMagic = op.GetOpMagic();
                    }
                    if (iOperands[i] == input2) {
                        input2OpMagic = op.GetOpMagic();
                    }
                }
                auto oOperands = op.GetOOperands();
                for (size_t i = 0; i < oOperands.size(); i++) {
                    if (oOperands[i] == output) {
                        outputOpMagic = op.GetOpMagic();
                    }
                }
            }
        }

        invokeInfo.RecordConnection(-1, programId, 0, input1->GetRawMagic(), offset, shape, rawShape, DT_FP32, input1,
                                    input1OpMagic);
        invokeInfo.RecordConnection(-1, programId, 0, input2->GetRawMagic(), offset, shape, rawShape, DT_FP32, input2,
                                    input2OpMagic);

        SubfuncInvokeInfoTy::SuccessorIncastInfoTy emptySuccessorInfo;
        invokeInfo.RecordOutcast(programId, 0, 1, output->GetRawMagic(), emptySuccessorInfo, offset, shape, rawShape,
                                 DT_FP32, output, outputOpMagic);

        invokeInfo.RecordTensorArg(0, input1->GetRawMagic(), offset, shape, rawShape, DT_FP32, false, input1,
                                   input1OpMagic);
        invokeInfo.RecordTensorArg(0, input2->GetRawMagic(), offset, shape, rawShape, DT_FP32, false, input2,
                                   input2OpMagic);
        invokeInfo.RecordTensorArg(0, output->GetRawMagic(), offset, shape, rawShape, DT_FP32, true, output,
                                   outputOpMagic);

        invokeInfo.DoFinishRecord();
        invokeInfo.ConstructActualInvokeParam(programId);

        return invokeInfo;
    }

    SubgraphToFunction createSubgraphToFunctionForComponents(size_t componentCount,
                                                             uint64_t baseProgramId = TEST_PROGRAM_ID)
    {
        SubgraphToFunction subgraphToFunction;
        for (size_t i = 0; i < componentCount; ++i) {
            auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
            invokeInfo->UpdateProgramSubgraphId(baseProgramId + i);
            subgraphToFunction.subFuncInvokeInfos.push_back(*invokeInfo);
        }
        return subgraphToFunction;
    }

    TestScenario createBasicTestScenario()
    {
        TestScenario scenario;

        scenario.shape = {MS_NUM16, MS_NUM16};
        scenario.inputTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, scenario.shape,
                                                                           CreateTestConstIntVector(scenario.shape));
        scenario.inputTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, scenario.shape,
                                                                           CreateTestConstIntVector(scenario.shape));
        scenario.outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, scenario.shape,
                                                                           CreateTestConstIntVector(scenario.shape));

        return scenario;
    }

    void buildOriginalMixFuncAndCallOp(const std::string& funcName, TestScenario& scenario,
                                       uint64_t programId = TEST_PROGRAM_ID)
    {
        scenario.originalMixFunc = createFunctionWithRealOffsetOps(funcName, COMPONENT_ID_0, scenario.inputTensor1,
                                                                   scenario.inputTensor2, scenario.outputTensor);

        scenario.originalCallOp = createCallOpWithArgList(programId, scenario.inputTensor1, scenario.inputTensor2,
                                                          scenario.outputTensor);
        scenario.originalCallAttr = std::dynamic_pointer_cast<CallOpAttribute>(
            scenario.originalCallOp->GetOpAttribute());
    }

    void verifyWrapIdSet(CallOpAttribute* callAttr)
    {
        EXPECT_NE(callAttr->wrapId, static_cast<uint64_t>(-1)) << "wrapId should be set (not -1)";
    }

    void verifyCallOpIOCount(Operation* callOp, size_t expectedInputs, size_t expectedOutputs)
    {
        auto iOperands = callOp->GetIOperands();
        auto oOperands = callOp->GetOOperands();
        EXPECT_EQ(iOperands.size(), expectedInputs) << "Input count should match";
        EXPECT_EQ(oOperands.size(), expectedOutputs) << "Output count should match";
    }

    void verifyCallOpOffsets(Operation* callOp, const std::vector<int>& expectedInputOffsets,
                             const std::vector<int>& expectedOutputOffsets)
    {
        ASSERT_NE(callOp, nullptr) << "CallOp should not be null";

        if (!expectedInputOffsets.empty()) {
            for (size_t i = 0; i < expectedInputOffsets.size(); ++i) {
                int actualOffset = callOp->GetIOpAttrOffset(i);
                EXPECT_EQ(actualOffset, expectedInputOffsets[i])
                    << "Input offset at index " << i << " should match expected value";
            }
        }

        if (!expectedOutputOffsets.empty()) {
            for (size_t i = 0; i < expectedOutputOffsets.size(); ++i) {
                int actualOffset = callOp->GetOOpAttrOffset(i);
                EXPECT_EQ(actualOffset, expectedOutputOffsets[i])
                    << "Output offset at index " << i << " should match expected value";
            }
        }
    }

    PropagatedTensors createPropagatedTensors()
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        PropagatedTensors tensors;
        tensors.input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        tensors.output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        return tensors;
    }

    void addPropagatedIncastOutcast(const std::shared_ptr<Function>& func,
                                    const std::shared_ptr<LogicalTensor>& propagatedInput,
                                    const std::shared_ptr<LogicalTensor>& propagatedOutput)
    {
        func->inCasts_.push_back(propagatedInput);
        func->outCasts_.push_back(propagatedOutput);
    }

    void createLeafFunctionsAndPointers(int count, const std::string& baseName,
                                        std::vector<std::shared_ptr<Function>>& leafFuncs,
                                        std::vector<Function*>& newFunctions)
    {
        leafFuncs.clear();
        newFunctions.clear();

        for (int i = 0; i < count; ++i) {
            auto leafFunc = createSimpleFunction(baseName + std::to_string(i));
            leafFuncs.push_back(leafFunc);
            newFunctions.push_back(leafFunc.get());
        }
    }

    std::vector<uint64_t> createProgramIds(size_t count, uint64_t baseProgramId = TEST_PROGRAM_ID)
    {
        std::vector<uint64_t> programIds;
        for (size_t i = 0; i < count; ++i) {
            programIds.push_back(baseProgramId + i);
        }
        return programIds;
    }

    void createComponentsAndSubgraphInfo(const std::vector<ComponentType>& componentTypes, const TestScenario& scenario,
                                         std::vector<InternalComponentInfo>& components,
                                         std::vector<std::shared_ptr<Function>>& leafFuncs,
                                         std::vector<Function*>& newFunctions, SubgraphToFunction& subgraphToFunction,
                                         std::vector<uint64_t>& newProgramIDs)
    {
        components.clear();
        leafFuncs.clear();
        newFunctions.clear();
        subgraphToFunction.subFuncInvokeInfos.clear();
        newProgramIDs.clear();

        for (size_t i = 0; i < componentTypes.size(); ++i) {
            int componentId = static_cast<int>(i);
            components.emplace_back(componentId, "comp_" + std::to_string(i), AIVCore::UNSPECIFIED, componentTypes[i]);

            auto leafFunc = createFunctionWithRealOffsetOps("leaf_" + std::to_string(i), componentId,
                                                            scenario.inputTensor1, scenario.inputTensor2,
                                                            scenario.outputTensor);
            leafFuncs.push_back(leafFunc);
            newFunctions.push_back(leafFunc.get());

            auto invokeInfo = createInvokeInfoWithTensorParams(TEST_PROGRAM_ID + i, scenario.inputTensor1,
                                                               scenario.inputTensor2, scenario.outputTensor);
            subgraphToFunction.subFuncInvokeInfos.push_back(invokeInfo);

            newProgramIDs.push_back(TEST_PROGRAM_ID + i);
        }
    }

    std::shared_ptr<Function> createFunctionWithOps(const std::string& name)
    {
        auto func = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        func->inCasts_.push_back(input1);
        func->inCasts_.push_back(input2);
        func->outCasts_.push_back(output);

        auto& copyIn1 = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_IN, {input1}, {internal1});
        copyIn1.opmagic = OP_MAGIC_BASE + 1;
        copyIn1.SetIOpAtt(0, 100);

        auto& copyIn2 = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_IN, {input2}, {internal2});
        copyIn2.opmagic = OP_MAGIC_BASE + 2;
        copyIn2.SetIOpAtt(0, 101);

        auto& addOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ADD, {internal1, internal2}, {output});
        addOp.opmagic = OP_MAGIC_BASE + 3;
        addOp.SetIOpAtt(0, 200);
        addOp.SetIOpAtt(1, 201);
        addOp.SetOOpAtt(0, 300);

        return func;
    }

    std::shared_ptr<Function> createFunctionWithInvokeInfo(const std::string& name)
    {
        auto func = createFunctionWithOps(name);

        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(TEST_PROGRAM_ID);

        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        const std::vector<int64_t> offset = {0, 0};
        const std::vector<int64_t> rawShape = {MS_NUM16, MS_NUM16};

        auto operations = func->Operations(false).DuplicatedOpList();
        auto& addOp = operations[2];

        invokeInfo->RecordConnection(-1, TEST_PROGRAM_ID, 0, 0, offset, shape, rawShape, DT_FP32, func->inCasts_[0],
                                     addOp->GetOpMagic());
        invokeInfo->RecordConnection(-1, TEST_PROGRAM_ID, 1, 0, offset, shape, rawShape, DT_FP32, func->inCasts_[1],
                                     addOp->GetOpMagic());

        SubfuncInvokeInfoTy::SuccessorIncastInfoTy emptySuccessorInfo;
        invokeInfo->RecordOutcast(TEST_PROGRAM_ID, 0, 1, 0, emptySuccessorInfo, offset, shape, rawShape, DT_FP32,
                                  func->outCasts_[0], addOp->GetOpMagic());

        invokeInfo->DoFinishRecord();

        return func;
    }

    Operation* createCallOpWithoutAttribute(Function& func)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        auto& callOp = IRBuilder().CreateTensorOpStmt(func, Opcode::OP_CALL, {input1, input2}, {output1});

        return &callOp;
    }

protected:
    std::shared_ptr<Function> rootFunc;
    std::unique_ptr<MixCallOperationBuilder> builder;

private:
    void createCopyInOperation(const std::shared_ptr<Function>& func, int internalSubgraphId,
                               const std::shared_ptr<LogicalTensor>& input,
                               const std::shared_ptr<LogicalTensor>& output, int offset)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;

        static int copyInOpMagicCounter = 50000;
        int opMagic = copyInOpMagicCounter++;

        auto& copyIn = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_IN, {input}, {output});
        copyIn.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
        copyIn.SetIOpAtt(TENSOR_INDEX_0, offset);
        copyIn.UpdateInternalSubgraphID(internalSubgraphId);
        copyIn.SetAttr(OpAttributeKey::isCube, true);
        copyIn.opmagic = opMagic;
    }

    void createAddOperation(const std::shared_ptr<Function>& func, int internalSubgraphId,
                            const std::shared_ptr<LogicalTensor>& input1, const std::shared_ptr<LogicalTensor>& input2,
                            const std::shared_ptr<LogicalTensor>& output)
    {
        auto& addOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ADD, {input1, input2}, {output});
        addOp.SetIOpAtt(TENSOR_INDEX_0, OFFSET_ADD_INPUT1);
        addOp.SetIOpAtt(TENSOR_INDEX_1, OFFSET_ADD_INPUT2);
        addOp.SetOOpAtt(TENSOR_INDEX_0, OFFSET_ADD_OUTPUT);
        addOp.UpdateInternalSubgraphID(internalSubgraphId);
        addOp.SetAttr(OpAttributeKey::isCube, true);
    }

    void createCopyOutOperation(const std::shared_ptr<Function>& func, int internalSubgraphId,
                                const std::shared_ptr<LogicalTensor>& input,
                                const std::shared_ptr<LogicalTensor>& output, int offset)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;

        static int copyOutOpMagicCounter = 60000;
        int opMagic = copyOutOpMagicCounter++;

        auto& copyOut = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_OUT, {input}, {output});
        copyOut.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
        copyOut.SetOOpAtt(TENSOR_INDEX_0, offset);
        copyOut.UpdateInternalSubgraphID(internalSubgraphId);
        copyOut.SetAttr(OpAttributeKey::isCube, true);
        copyOut.opmagic = opMagic;
    }

    std::shared_ptr<Function> createOperationsWithOffsets(const std::shared_ptr<Function>& func, int internalSubgraphId,
                                                          const std::shared_ptr<LogicalTensor>& input1,
                                                          const std::shared_ptr<LogicalTensor>& input2,
                                                          const std::shared_ptr<LogicalTensor>& output,
                                                          const std::vector<int64_t>& shape)
    {
        auto internal1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        createCopyInOperation(func, internalSubgraphId, input1, internal1, OFFSET_INPUT1);
        createCopyInOperation(func, internalSubgraphId, input2, internal2, OFFSET_INPUT2);
        createAddOperation(func, internalSubgraphId, internal1, internal2, internal3);
        createCopyOutOperation(func, internalSubgraphId, internal3, output, OFFSET_OUTPUT);

        return func;
    }

    std::shared_ptr<Function> createFunctionWithRealOffsetOps(const std::string& name, int internalSubgraphId,
                                                              const std::shared_ptr<LogicalTensor>& input1,
                                                              const std::shared_ptr<LogicalTensor>& input2,
                                                              const std::shared_ptr<LogicalTensor>& output)
    {
        auto func = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        func->inCasts_.push_back(input1);
        func->inCasts_.push_back(input2);
        func->outCasts_.push_back(output);

        auto internal1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto internal3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        static int opMagicCounter = OP_MAGIC_BASE;

        auto& copyIn1 = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_IN, {input1}, {internal1});
        copyIn1.opmagic = opMagicCounter++;

        auto& copyIn2 = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_IN, {input2}, {internal2});
        copyIn2.opmagic = opMagicCounter++;

        auto& addOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ADD, {internal1, internal2}, {internal3});
        addOp.opmagic = opMagicCounter++;

        auto& copyOut = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_COPY_OUT, {internal3}, {output});
        copyOut.opmagic = opMagicCounter++;

        copyIn1.SetIOpAtt(TENSOR_INDEX_0, OFFSET_INPUT1);
        copyIn1.UpdateInternalSubgraphID(internalSubgraphId);

        copyIn2.SetIOpAtt(TENSOR_INDEX_0, OFFSET_INPUT2);
        copyIn2.UpdateInternalSubgraphID(internalSubgraphId);

        addOp.SetIOpAtt(TENSOR_INDEX_0, OFFSET_ADD_INPUT1);
        addOp.SetIOpAtt(TENSOR_INDEX_1, OFFSET_ADD_INPUT2);
        addOp.SetOOpAtt(TENSOR_INDEX_0, OFFSET_ADD_OUTPUT);
        addOp.UpdateInternalSubgraphID(internalSubgraphId);

        copyOut.SetOOpAtt(TENSOR_INDEX_0, OFFSET_OUTPUT);
        copyOut.UpdateInternalSubgraphID(internalSubgraphId);

        return func;
    }

    void addPropagatedTensorOperations(const std::shared_ptr<Function>& func, int componentId,
                                       const std::shared_ptr<LogicalTensor>& propagatedInput,
                                       const std::shared_ptr<LogicalTensor>& propagatedOutput)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        auto internalPropagatedInput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                                  CreateTestConstIntVector(shape));
        createCopyInOperation(func, componentId, propagatedInput, internalPropagatedInput, OFFSET_INPUT1 + MS_NUM10);

        auto internalPropagatedOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                                   CreateTestConstIntVector(shape));
        createCopyOutOperation(func, componentId, internalPropagatedOutput, propagatedOutput, OFFSET_OUTPUT + MS_NUM10);
    }

    std::shared_ptr<Function> createOriginalMixFuncWithPropagatedTensors(const std::string& name,
                                                                         const TestScenario& scenario,
                                                                         const PropagatedTensors& propagatedTensors)
    {
        auto originalMixFunc = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        originalMixFunc->inCasts_.push_back(scenario.inputTensor1);
        originalMixFunc->inCasts_.push_back(scenario.inputTensor2);
        originalMixFunc->outCasts_.push_back(scenario.outputTensor);
        addPropagatedIncastOutcast(originalMixFunc, propagatedTensors.input, propagatedTensors.output);

        createOperationsWithOffsets(originalMixFunc, COMPONENT_ID_0, scenario.inputTensor1, scenario.inputTensor2,
                                    scenario.outputTensor, shape);
        addPropagatedTensorOperations(originalMixFunc, COMPONENT_ID_0, propagatedTensors.input,
                                      propagatedTensors.output);

        return originalMixFunc;
    }

    std::vector<SymbolicScalar> createTensorArgsFor2D(int tensorIndex)
    {
        std::vector<SymbolicScalar> tensorArgs;
        tensorArgs.push_back(IRBuilder().CreateConstInt(tensorIndex));

        for (int i = 0; i < MS_NUM4; ++i) {
            for (int d = 0; d < TENSOR_DIMENSIONS; ++d) {
                tensorArgs.push_back(IRBuilder().CreateConstInt(i == 0 ? 0 : MS_NUM16));
            }
        }
        return tensorArgs;
    }

    Operation* createCallOpWithArgList(uint64_t programId, const std::shared_ptr<LogicalTensor>& input1,
                                       const std::shared_ptr<LogicalTensor>& input2,
                                       const std::shared_ptr<LogicalTensor>& output)
    {
        auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFunc, Opcode::OP_CALL, {input1, input2}, {output});

        std::vector<std::vector<SymbolicScalar>> argList;
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_0));
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_1));
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_2));

        auto callAttr = std::make_shared<CallOpAttribute>(
            FunctionHash(programId), argList, "", std::map<int, SymbolicScalar>(), std::vector<SymbolicScalar>());

        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programId);
        callAttr->invokeInfo_ = invokeInfo;

        callOp.SetOpAttribute(callAttr);
        callOp.UpdateSubgraphID(programId);

        return &callOp;
    }

    // 用于重构 TestPropagatedIncastOutcast 的辅助函数
    std::shared_ptr<Function> createOriginalMixFuncWithMultiplePropagatedTensors(
        const std::string& name, const TestScenario& scenario, const PropagatedTensors& propagatedTensors,
        const std::shared_ptr<LogicalTensor>& propagatedInput2, const std::shared_ptr<LogicalTensor>& propagatedOutput2)
    {
        auto originalMixFunc = createOriginalMixFuncWithPropagatedTensors(name, scenario, propagatedTensors);

        addPropagatedTensorOperations(originalMixFunc, COMPONENT_ID_0, propagatedInput2, propagatedOutput2);
        originalMixFunc->inCasts_.push_back(propagatedInput2);
        originalMixFunc->outCasts_.push_back(propagatedOutput2);

        return originalMixFunc;
    }

    Operation* createOriginalCallOpWithMultiplePropagatedTensors(
        const TestScenario& scenario, const PropagatedTensors& propagatedTensors,
        const std::shared_ptr<LogicalTensor>& propagatedInput2, const std::shared_ptr<LogicalTensor>& propagatedOutput2)
    {
        auto& originalCallOp = IRBuilder().CreateTensorOpStmt(
            *rootFunc, Opcode::OP_CALL,
            {scenario.inputTensor1, scenario.inputTensor2, propagatedTensors.input, propagatedInput2},
            {scenario.outputTensor, propagatedTensors.output, propagatedOutput2});

        auto originalCallAttr = std::make_shared<CallOpAttribute>();
        originalCallOp.SetOpAttribute(originalCallAttr);

        return &originalCallOp;
    }

    std::vector<std::shared_ptr<Function>> createLeafFunctionsForPropagatedTest(
        const TestScenario& scenario, const PropagatedTensors& propagatedTensors,
        const std::shared_ptr<LogicalTensor>& propagatedInput2, const std::shared_ptr<LogicalTensor>& propagatedOutput2,
        const std::vector<InternalComponentInfo>& components)
    {
        std::vector<std::shared_ptr<Function>> leafFuncs;

        for (size_t i = 0; i < components.size(); ++i) {
            std::string leafName = "leaf" + std::to_string(i);
            auto leafFunc = createLeafFuncWithPropagatedTensors(leafName, static_cast<int>(i), scenario,
                                                                propagatedTensors);

            leafFunc->inCasts_.push_back(propagatedInput2);
            leafFunc->outCasts_.push_back(propagatedOutput2);
            addPropagatedTensorOperations(leafFunc, static_cast<int>(i), propagatedInput2, propagatedOutput2);

            leafFuncs.push_back(leafFunc);
        }

        return leafFuncs;
    }

    SubgraphToFunction createSubgraphInfoForPropagatedTest(const TestScenario& scenario,
                                                           const std::vector<Function*>& newFunctions)
    {
        SubgraphToFunction subgraphToFunction;

        for (size_t i = 0; i < newFunctions.size(); ++i) {
            auto leafFunc = newFunctions[i];
            auto invokeInfo = createInvokeInfoWithIncastOutcast(TEST_PROGRAM_ID + i, scenario.inputTensor1,
                                                                scenario.inputTensor2, scenario.outputTensor, leafFunc);
            subgraphToFunction.subFuncInvokeInfos.push_back(invokeInfo);
        }

        return subgraphToFunction;
    }

    void verifyPropagatedTestResults(Operation* originalCallOp, size_t expectedComponentCount)
    {
        auto newCallOps = rootFunc->GetCallopList();
        size_t newCallOpCount = 0;

        for (auto* callOpPtr : newCallOps) {
            if (callOpPtr != originalCallOp) {
                newCallOpCount++;
                auto callAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
                if (callAttr) {
                    EXPECT_NE(callAttr->wrapId, static_cast<uint64_t>(-1)) << "wrapId should be set (not -1)";
                }
            }
        }

        EXPECT_EQ(newCallOpCount, expectedComponentCount) << "Should create one CallOp per component";
    }
};

// ==================== 测试用例 ====================

// 测试同一个originalCallOp对应的不同组件有相同wrapId
TEST_F(MixCallOperationBuilderTest, TestSameWrapIdForSameOriginalCallOp)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    auto originalCallOp = createSimpleCallOp(*rootFunc);
    auto components = createMixedComponents();
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());

    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctions);

    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());

    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
                                           newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed";

    auto newCallOps = rootFunc->GetCallopList();
    Operation* newCallOp1 = nullptr;
    Operation* newCallOp2 = nullptr;

    for (auto* op : newCallOps) {
        if (op != originalCallOp) {
            if (newCallOp1 == nullptr) {
                newCallOp1 = op;
            } else {
                newCallOp2 = op;
                break;
            }
        }
    }

    ASSERT_NE(newCallOp1, nullptr) << "Should have created first new CallOp";
    ASSERT_NE(newCallOp2, nullptr) << "Should have created second new CallOp";

    auto callAttr1 = dynamic_cast<CallOpAttribute*>(newCallOp1->GetOpAttribute().get());
    auto callAttr2 = dynamic_cast<CallOpAttribute*>(newCallOp2->GetOpAttribute().get());

    ASSERT_NE(callAttr1, nullptr) << "CallOp1 should have CallOpAttribute";
    ASSERT_NE(callAttr2, nullptr) << "CallOp2 should have CallOpAttribute";

    EXPECT_EQ(callAttr1->wrapId, callAttr2->wrapId)
        << "Same originalCallOp should have same wrapId for different components";
}

// 测试不同的originalCallOp有不同的wrapId
TEST_F(MixCallOperationBuilderTest, TestDifferentWrapIdForDifferentOriginalCallOps)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    auto originalCallOp1 = createSimpleCallOp(*rootFunc);
    auto originalCallOp2 = createSimpleCallOp(*rootFunc);
    auto components = createMixedComponents();
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());

    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctions);

    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());

    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp1, originalCallOp2}, originalMixFunc.get(),
                                           components, newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed";

    auto newCallOps = rootFunc->GetCallopList();
    std::set<uint64_t> wrapIds;

    for (auto* op : newCallOps) {
        if (op != originalCallOp1 && op != originalCallOp2) {
            auto callAttr = dynamic_cast<CallOpAttribute*>(op->GetOpAttribute().get());
            if (callAttr) {
                wrapIds.insert(callAttr->wrapId);
            }
        }
    }

    EXPECT_GE(wrapIds.size(), 2U) << "Different originalCallOps should have different wrapIds";
}

// 测试Global Tensor的处理
TEST_F(MixCallOperationBuilderTest, TestGlobalTensorHandling)
{
    auto scenario = createBasicTestScenario();
    buildOriginalMixFuncAndCallOp("original_mix", scenario);

    std::vector<InternalComponentInfo> components = {
        {COMPONENT_ID_0, "comp_cube", AIVCore::UNSPECIFIED, ComponentType::C_SCOPE},
        {COMPONENT_ID_1, "comp_vector", AIVCore::AIV0, ComponentType::V_SCOPE}};

    auto leafFuncCube = createFunctionWithRealOffsetOps("leaf_cube", COMPONENT_ID_0, scenario.inputTensor1,
                                                        scenario.inputTensor2, scenario.outputTensor);
    auto leafFuncVector = createFunctionWithRealOffsetOps("leaf_vector", COMPONENT_ID_1, scenario.inputTensor1,
                                                          scenario.inputTensor2, scenario.outputTensor);
    std::vector<Function*> newFunctions = {leafFuncCube.get(), leafFuncVector.get()};

    SubgraphToFunction subgraphToFunction;
    for (int i = 0; i < MS_NUM2; ++i) {
        auto leafInvokeInfo = createInvokeInfoWithTensorParams(TEST_PROGRAM_ID + i, scenario.inputTensor1,
                                                               scenario.inputTensor2, scenario.outputTensor);
        subgraphToFunction.subFuncInvokeInfos.push_back(leafInvokeInfo);
    }

    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());

    Status status = builder->CreateCallOps(*rootFunc, {scenario.originalCallOp}, scenario.originalMixFunc.get(),
                                           components, newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed with global tensors";

    auto newCallOps = rootFunc->GetCallopList();
    EXPECT_GE(newCallOps.size(), 2U) << "Should have created at least 2 new CallOps";

    for (auto* callOp : newCallOps) {
        if (callOp != scenario.originalCallOp) {
            verifyCallOpIOCount(callOp, 2U, 1U);

            auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
            ASSERT_NE(callAttr, nullptr) << "CallOp should have CallOpAttribute";

            const auto& argList = callAttr->GetArgList();
            EXPECT_FALSE(argList.empty()) << "argList should not be empty";
            verifyWrapIdSet(callAttr);
        }
    }
}

// 测试带内部依赖的情况
TEST_F(MixCallOperationBuilderTest, TestInternalDependencies)
{
    auto originalMixFunc = createSimpleFunction("TestInternalDependencies_mix");
    auto originalCallOp = createSimpleCallOp(*rootFunc);
    auto components = createMixedComponents();
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());

    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctionVec;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctionVec);

    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());

    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
                                           newProgramIDs, subgraphToFunction, newFunctionVec);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed with internal dependencies";
}

TEST_F(MixCallOperationBuilderTest, TestOffsets)
{
    auto scenario = createBasicTestScenario();
    buildOriginalMixFuncAndCallOp("test_component_types", scenario);

    const std::vector<ComponentType> componentTypes = {ComponentType::C_SCOPE, ComponentType::V_SCOPE};

    std::vector<InternalComponentInfo> components;
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    SubgraphToFunction subgraphToFunction;
    std::vector<uint64_t> newProgramIDs;

    createComponentsAndSubgraphInfo(componentTypes, scenario, components, leafFuncs, newFunctions, subgraphToFunction,
                                    newProgramIDs);

    Status status = builder->CreateCallOps(*rootFunc, {scenario.originalCallOp}, scenario.originalMixFunc.get(),
                                           components, newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed for different component types";

    auto newCallOps = rootFunc->GetCallopList();
    size_t newCallOpCount = 0;

    for (auto* callOp : newCallOps) {
        if (callOp != scenario.originalCallOp) {
            newCallOpCount++;
            verifyCallOpOffsets(callOp, {OFFSET_INPUT1, OFFSET_INPUT2}, {OFFSET_OUTPUT});
        }
    }

    EXPECT_EQ(newCallOpCount, componentTypes.size()) << "Should create one CallOp per component";
}

// 测试传播依赖的Incast/Outcast处理
TEST_F(MixCallOperationBuilderTest, TestPropagatedIncastOutcast)
{
    // 准备测试数据
    auto scenario = createBasicTestScenario();
    auto propagatedTensors = createPropagatedTensors();

    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
    auto propagatedInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto propagatedOutput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                        CreateTestConstIntVector(shape));

    // 创建原始混合函数
    auto originalMixFunc = createOriginalMixFuncWithMultiplePropagatedTensors(
        "original_mix", scenario, propagatedTensors, propagatedInput2, propagatedOutput2);

    // 创建原始调用操作
    auto originalCallOp = createOriginalCallOpWithMultiplePropagatedTensors(scenario, propagatedTensors,
                                                                            propagatedInput2, propagatedOutput2);

    // 创建组件信息
    std::vector<InternalComponentInfo> components = {
        {COMPONENT_ID_0, "comp_c_scope", AIVCore::UNSPECIFIED, ComponentType::C_SCOPE},
        {COMPONENT_ID_1, "comp_v_scope", AIVCore::AIV0, ComponentType::V_SCOPE}};

    // 创建叶子函数
    auto leafFuncs = createLeafFunctionsForPropagatedTest(scenario, propagatedTensors, propagatedInput2,
                                                          propagatedOutput2, components);

    // 准备函数指针列表
    std::vector<Function*> newFunctions;
    for (auto& func : leafFuncs) {
        newFunctions.push_back(func.get());
    }

    // 创建子图映射信息
    auto subgraphToFunction = createSubgraphInfoForPropagatedTest(scenario, newFunctions);

    // 执行测试
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());

    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
                                           newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed with mixed incast/outcast";

    // 验证结果
    verifyPropagatedTestResults(originalCallOp, components.size());
}

// =====================日志覆盖======================
TEST_F(MixCallOperationBuilderTest, TestCreateCallOpWithNullCallAttribute)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    auto originalCallOp = createCallOpWithoutAttribute(*rootFunc);

    std::vector<InternalComponentInfo> components = {{0, "comp_0", AIVCore::UNSPECIFIED, ComponentType::C_SCOPE}};

    auto leafFunc = createFunctionWithOps("leaf_0");
    std::vector<Function*> newFunctions = {leafFunc.get()};

    std::vector<uint64_t> newProgramIDs = {TEST_PROGRAM_ID};
    auto subgraphToFunction = createSubgraphToFunctionForComponents(1);

    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
                                           newProgramIDs, subgraphToFunction, newFunctions);

    EXPECT_EQ(status, FAILED) << "CreateCallOps should fail with null CallOpAttribute";
}

TEST_F(MixCallOperationBuilderTest, TestFindIOpAttrOffsetFromActualIncastsWithInvalidTensor)
{
    auto leafFunc = createFunctionWithInvokeInfo("leaf_func");
    auto originalMixFunc = createFunctionWithOps("original_mix");

    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
    auto invalidTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    std::vector<std::shared_ptr<LogicalTensor>> actualIncasts = {invalidTensor};

    std::vector<OperandAttribute> iOffsets;
    std::vector<OperandAttribute> oOffsets;
    std::set<LogicalTensorPtr> processedIncasts;
    std::set<LogicalTensorPtr> processedOutcasts;
    ExtractInfo extractInfo{iOffsets, oOffsets, processedIncasts, processedOutcasts};

    bool result = builder->FindIOpAttrOffsetFromActualIncasts(actualIncasts, extractInfo, originalMixFunc.get());

    EXPECT_FALSE(result) << "FindIOpAttrOffsetFromActualIncasts should return false for invalid tensor";
}

TEST_F(MixCallOperationBuilderTest, TestFindOOpAttrOffsetFromActualOutcastsWithEmptyShape)
{
    auto leafFunc = createFunctionWithInvokeInfo("leaf_func");
    auto originalMixFunc = createFunctionWithOps("original_mix");

    const std::vector<int64_t> emptyShape = {};
    auto tensorWithEmptyShape = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, emptyShape,
                                                                           CreateTestConstIntVector(emptyShape));

    std::vector<std::shared_ptr<LogicalTensor>> actualOutcasts = {tensorWithEmptyShape};

    std::vector<OperandAttribute> iOffsets;
    std::vector<OperandAttribute> oOffsets;
    std::set<LogicalTensorPtr> processedIncasts;
    std::set<LogicalTensorPtr> processedOutcasts;
    ExtractInfo extractInfo{iOffsets, oOffsets, processedIncasts, processedOutcasts};

    bool result = builder->FindOOpAttrOffsetFromActualOutcasts(actualOutcasts, extractInfo, originalMixFunc.get());

    EXPECT_FALSE(result) << "FindOOpAttrOffsetFromActualOutcasts should return false for tensor with empty shape";
}

TEST_F(MixCallOperationBuilderTest, TestFindOOpAttrOffsetFromActualOutcastsWithInvalidTensor)
{
    auto leafFunc = createFunctionWithInvokeInfo("leaf_func");
    auto originalMixFunc = createFunctionWithOps("original_mix");

    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
    auto invalidTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    std::vector<std::shared_ptr<LogicalTensor>> actualOutcasts = {invalidTensor};

    std::vector<OperandAttribute> iOffsets;
    std::vector<OperandAttribute> oOffsets;
    std::set<LogicalTensorPtr> processedIncasts;
    std::set<LogicalTensorPtr> processedOutcasts;
    ExtractInfo extractInfo{iOffsets, oOffsets, processedIncasts, processedOutcasts};

    bool result = builder->FindOOpAttrOffsetFromActualOutcasts(actualOutcasts, extractInfo, originalMixFunc.get());

    EXPECT_FALSE(result) << "FindOOpAttrOffsetFromActualOutcasts should return false for invalid tensor";
}

} // namespace tile_fwk
} // namespace npu
