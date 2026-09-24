/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#define private public
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#undef private

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
struct MiniDup {
    static constexpr size_t kBuf = 4096;
    std::unique_ptr<uint8_t[]> funcBuf;
    std::unique_ptr<uint8_t[]> dupBuf;
    DevAscendFunction* func{nullptr};
    DevAscendFunctionDuppedData* dupData{nullptr};
    DevAscendFunctionDupped dup;
    size_t cursor{0};

    void Align(size_t a) { cursor = (cursor + a - 1) & ~(a - 1); }

    template <typename T>
    T* Alloc(size_t n = 1)
    {
        Align(alignof(T));
        auto* p = reinterpret_cast<T*>(funcBuf.get() + cursor);
        cursor += sizeof(T) * n;
        EXPECT_LT(cursor, kBuf);
        return p;
    }

    void Build(bool withOutcast)
    {
        funcBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(funcBuf.get(), 0, kBuf);
        cursor = sizeof(DevAscendFunction);
        func = reinterpret_cast<DevAscendFunction*>(funcBuf.get());
        func->funcKey = 1;

        func->operationList_.AssignOffsetSize(cursor, 1);
        auto* op = Alloc<DevAscendOperation>();
        new (op) DevAscendOperation();
        op->stitchIndex = 0;
        op->depGraphPredCount = 0;
        op->depGraphSuccList.AssignOffsetSize(0, 0);

        if (withOutcast) {
            Align(alignof(DevAscendFunctionOutcast));
            func->outcastList.AssignOffsetSize(cursor, 1);
            auto* oc = Alloc<DevAscendFunctionOutcast>();
            new (oc) DevAscendFunctionOutcast();
            oc->stitchPolicyFullCoverProducerHubOpIdx = -1;
            oc->producerConsumerList.AssignOffsetSize(0, 0);
            oc->stitchPolicyFullCoverProducerList.AssignOffsetSize(0, 0);
            oc->toSlotList.AssignOffsetSize(0, 0);
            oc->cellMatchRuntimeFullUpdateTable.AssignOffsetSize(0, 0);
        }

        dupBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(dupBuf.get(), 0, kBuf);
        dupData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.get());
        uint8_t* p = dupBuf.get() + sizeof(DevAscendFunctionDuppedData);
        dupData->source_ = func;
        dupData->operationList_.size = 1;
        dupData->operationList_.predCountBase = static_cast<uint32_t>(p - dupBuf.get());
        *reinterpret_cast<predcount_t*>(p) = 0;
        p += sizeof(predcount_t);
        dupData->operationList_.stitchBase = static_cast<uint32_t>(p - dupBuf.get());
        dupData->operationList_.stitchCount = 1;
        new (p) DevAscendFunctionDuppedStitchList();
        p += sizeof(DevAscendFunctionDuppedStitchList);
        dupData->expressionList_.base = static_cast<uint32_t>(p - dupBuf.get());
        dupData->expressionList_.size = 1;
        *reinterpret_cast<uint64_t*>(p) = 0;

        WsAllocation a;
        a.ptr = reinterpret_cast<uint64_t>(dupData);
        dup = DevAscendFunctionDupped(a);
    }
};
} // namespace

TEST(ParallelForContextTest, Begin_FirstLoop)
{
    ParallelForContext ctx;
    ctx.info.forId = 0;
    ctx.info.iterId = 0;
    ctx.info.wsId = 0;
    ctx.info.parallelism = 2;
    ctx.Begin();
    EXPECT_EQ(ctx.info.forId, 1u);
    EXPECT_EQ(ctx.info.iterId, 1u);
    EXPECT_EQ(ctx.info.wsId, 1u);
    EXPECT_TRUE(ctx.isInParallelForScope);
}

TEST(ParallelForContextTest, Begin_SubsequentLoop)
{
    ParallelForContext ctx;
    ctx.info.forId = 1;
    ctx.info.iterId = 5;
    ctx.info.wsId = 0;
    ctx.info.parallelism = 2;
    ctx.Begin();
    EXPECT_EQ(ctx.info.forId, 1u);
    EXPECT_EQ(ctx.info.iterId, 6u);
    EXPECT_EQ(ctx.info.wsId, 1u);
    EXPECT_TRUE(ctx.isInParallelForScope);
}

TEST(ParallelForContextTest, Begin_WrapAround)
{
    ParallelForContext ctx;
    ctx.info.forId = 1;
    ctx.info.iterId = 5;
    ctx.info.wsId = 1;
    ctx.info.parallelism = 2;
    ctx.Begin();
    EXPECT_EQ(ctx.info.wsId, 0u);
}

TEST(ParallelForContextTest, End)
{
    ParallelForContext ctx;
    ctx.isInParallelForScope = true;
    ctx.End();
    EXPECT_FALSE(ctx.isInParallelForScope);
}

TEST(ParallelForContextTest, SwitchDefaultWorkspace)
{
    ParallelForContext ctx;
    ctx.info.wsId = 5;
    ctx.SwitchDefaultWorkspace();
    EXPECT_EQ(ctx.info.wsId, 0u);
}

TEST(ParallelForContextTest, ChangeForId)
{
    ParallelForContext ctx;
    ctx.info.forId = 3;
    ctx.info.iterId = 10;
    ctx.ChangeForId();
    EXPECT_EQ(ctx.info.forId, 4u);
    EXPECT_EQ(ctx.info.iterId, 0u);
}

TEST(ParallelForContextTest, InitParallel)
{
    ParallelForContext ctx;
    ctx.InitParallel(8);
    EXPECT_EQ(ctx.info.parallelism, 8u);
}

TEST(DevAscendFunctionDuppedDataTest, GetSource)
{
    MiniDup dup;
    dup.Build(false);
    EXPECT_EQ(dup.dupData->GetSource(), dup.func);
}

TEST(DevAscendFunctionDuppedDataTest, GetRuntimeWorkspace)
{
    MiniDup dup;
    dup.Build(false);
    dup.dupData->runtimeWorkspace_ = 0x1234;
    EXPECT_EQ(dup.dupData->GetRuntimeWorkspace(), 0x1234u);
}

TEST(DevAscendFunctionDuppedDataTest, GetRuntimeReuseInfo)
{
    MiniDup dup;
    dup.Build(false);
    dup.dupData->runtimeWsReuseInfo_.poolResetTimes = 5;
    auto reuseInfo = dup.dupData->GetRuntimeReuseInfo();
    EXPECT_EQ(reuseInfo.poolResetTimes, 5u);
}

TEST(DevAscendFunctionDuppedDataTest, GetOperationSize)
{
    MiniDup dup;
    dup.Build(false);
    EXPECT_EQ(dup.dupData->GetOperationSize(), 1u);
}

TEST(DevAscendFunctionDuppedDataTest, GetExpressionSize)
{
    MiniDup dup;
    dup.Build(false);
    EXPECT_EQ(dup.dupData->GetExpressionSize(), 1u);
}

TEST(DevAscendFunctionDuppedDataTest, GetOperationCurrPredCount)
{
    MiniDup dup;
    dup.Build(false);
    auto& predCount = dup.dupData->GetOperationCurrPredCount(0);
    EXPECT_EQ(predCount, 0u);
}

TEST(DevAscendFunctionDuppedDataTest, GetOperationStitch)
{
    MiniDup dup;
    dup.Build(false);
    auto& stitch = dup.dupData->GetOperationStitch(0);
    EXPECT_EQ(stitch.Head(), nullptr);
}

TEST(DevAscendFunctionDuppedTest, GetSource)
{
    MiniDup dup;
    dup.Build(false);
    EXPECT_EQ(dup.dup.GetSource(), dup.func);
}

TEST(DevAscendFunctionDuppedTest, GetExpressionAddr)
{
    MiniDup dup;
    dup.Build(false);
    uint64_t* exprAddr = dup.dup.GetExpressionAddr();
    EXPECT_NE(exprAddr, nullptr);
    EXPECT_EQ(*exprAddr, 0u);
}

TEST(DevAscendFunctionDuppedTest, GetExpression)
{
    MiniDup dup;
    dup.Build(false);
    uint64_t expr = dup.dup.GetExpression(0);
    EXPECT_EQ(expr, 0u);
}

TEST(DevAscendFunctionDuppedTest, SchemaGetExpressionTable)
{
    MiniDup dup;
    dup.Build(false);
    auto exprTable = dup.dup.SchemaGetExpressionTable();
    // Just verify it doesn't crash
    (void)exprTable;
}

TEST(DeviceExecuteContextExtraTest, GetRuid_Basic)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    ctx.taskId = 42;
    ctx.stitchContext.stitchedList_.size_ = 5;

    auto ruid = ctx.GetRuid(100, false);
    (void)ruid;
}

TEST(DeviceExecuteContextExtraTest, GetRuid_AfterAppend)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    ctx.taskId = 10;
    ctx.stitchContext.stitchedList_.size_ = 3;

    auto ruid = ctx.GetRuid(200, true);
    (void)ruid;
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_Finish)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_TRUE(ctx.NeedSubmmitDevTask(RUNTIME_FUNCKEY_FINISH));
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_LoopBarrier)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_TRUE(ctx.NeedSubmmitDevTask(RUNTIME_FUNCKEY_LOOP_BARRIER));
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_ParallelForEnd)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_TRUE(ctx.NeedSubmmitDevTask(RUNTIME_FUNCKEY_PARALLEL_FOR_END));
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_ParallelForBegin)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_TRUE(ctx.NeedSubmmitDevTask(RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN));
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_Other)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_FALSE(ctx.NeedSubmmitDevTask(12345));
}

TEST(DeviceExecuteContextExtraTest, NeedSubmmitDevTask_Zero)
{
    DevStartArgs args{};
    DeviceExecuteContext ctx(&args);
    EXPECT_FALSE(ctx.NeedSubmmitDevTask(0));
}

TEST(DeviceStitchContextTest, DumpStitchInfo_Empty)
{
    DeviceStitchContext ctx;
    ctx.DumpStitchInfo(nullptr, 0);
}

TEST(DeviceStitchContextTest, DumpStitchInfo_WithMiniDup)
{
    MiniDup dup;
    dup.Build(false);

    DeviceStitchContext ctx;
    ctx.DumpStitchInfo(&dup.dup, 1);
}

TEST(DevAscendFunctionDuppedDataTest, Dump_Indent)
{
    MiniDup dup;
    dup.Build(false);

    std::string result = dup.dup.Dump(2);
    EXPECT_FALSE(result.empty());
}

TEST(DevAscendFunctionDuppedDataTest, Dump_ZeroIndent)
{
    MiniDup dup;
    dup.Build(false);

    std::string result = dup.dup.Dump(0);
    EXPECT_FALSE(result.empty());
}
