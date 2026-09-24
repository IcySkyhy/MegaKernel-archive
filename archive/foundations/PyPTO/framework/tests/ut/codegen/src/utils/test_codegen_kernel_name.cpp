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
 * \file test_codegen_kernel_name.cpp
 * \brief Unit test for deterministic kernel numbering under parallel codegen.
 */

#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "codegen/npu/codegen_npu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
namespace {
constexpr int THREAD_NUM = 128;
constexpr int DEV_ROOT_NUM = 8;
constexpr uint64_t PROGRAM_NUM = 16;

struct KernelIdentity {
    int devRootIndex;
    uint64_t subProgramId;
    bool isMainBlock;
};

std::vector<KernelIdentity> BuildIdentities()
{
    std::vector<KernelIdentity> identities;
    for (int devRoot = 0; devRoot < DEV_ROOT_NUM; ++devRoot) {
        for (uint64_t program = 0; program < PROGRAM_NUM; ++program) {
            identities.push_back({devRoot, program, false});
            identities.push_back({devRoot, program, true});
        }
    }
    return identities;
}

uint64_t MakeKernelKey(Function& topFunc, const KernelIdentity& identity)
{
    CodeGenCtx ctx("", "", identity.isMainBlock, false, identity.devRootIndex);
    std::pair<uint64_t, Function*> subFuncPair{identity.subProgramId, topFunc.rootFunc_->programs_[0]};
    CompileInfo compileInfo(topFunc, ctx, subFuncPair, false);
    return compileInfo.GetDeterministicSubId();
}
} // namespace

class TestCodegenKernelName : public CodegenTestBase {
public:
    TestCodegenKernelName()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH,
                           .setTileTensor = true,
                           .tileTensorValue = true,
                           .setIdGen = true,
                           .resetTileTensorOnTearDown = true})
    {}
};

TEST_F(TestCodegenKernelName, SubTilingKeyDiffersPerFuncHashWithSameStructuredSlot)
{
    // Two GenMockFuncDyn in one Program without Reset() crashes in RemoveCallOpViewAssemble
    // (GetGraphInfo dereferences empty producers). Build them sequentially with Reset.
    KernelIdentity identity{1, 0, false};

    Function* functionA = GenMockFuncDyn("kernel_hash_a");
    ASSERT_NE(functionA, nullptr);
    const uint64_t hashA = functionA->GetFunctionHash().GetHash();
    const uint64_t keyA = MakeKernelKey(*functionA, identity);

    Program::GetInstance().Reset();
    config::Reset();
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);

    Function* functionB = GenMockFuncDyn("kernel_hash_b");
    ASSERT_NE(functionB, nullptr);
    const uint64_t hashB = functionB->GetFunctionHash().GetHash();
    const uint64_t keyB = MakeKernelKey(*functionB, identity);

    ASSERT_NE(hashA, hashB);
    EXPECT_NE(keyA, keyB);
}

TEST_F(TestCodegenKernelName, SubTilingKeyPacksDecodableFields)
{
    Function* function = GenMockFuncDyn("kernel_id_decode");
    ASSERT_NE(function, nullptr);

    KernelIdentity identity{3, 7, true};
    uint64_t key = MakeKernelKey(*function, identity);

    EXPECT_EQ(key & 1UL, 1UL);
    EXPECT_EQ((key >> SUB_ID_PROGRAM_SHIFT) & SUB_ID_PROGRAM_MAX, identity.subProgramId);
    EXPECT_EQ((key >> SUB_ID_DEV_ROOT_SHIFT) & SUB_ID_DEV_ROOT_MAX, static_cast<uint64_t>(identity.devRootIndex));

    uint64_t hash = function->GetFunctionHash().GetHash();
    uint64_t folded = (hash ^ (hash >> SUB_ID_HASH_BITS)) & SUB_ID_HASH_MASK;
    EXPECT_EQ((key >> SUB_ID_HASH_SHIFT) & SUB_ID_HASH_MASK, folded);
}

TEST_F(TestCodegenKernelName, SubTilingKeyIsUniquePerIdentity)
{
    Function* function = GenMockFuncDyn("kernel_id_unique");
    ASSERT_NE(function, nullptr);
    ASSERT_NE(function->rootFunc_->programs_[0], nullptr);

    auto identities = BuildIdentities();
    std::set<uint64_t> keys;
    for (const auto& identity : identities) {
        keys.insert(MakeKernelKey(*function, identity));
    }
    EXPECT_EQ(keys.size(), identities.size());
}

TEST_F(TestCodegenKernelName, SubTilingKeyIsStableAcrossThreadCounts)
{
    Function* function = GenMockFuncDyn("kernel_id_stable");
    ASSERT_NE(function, nullptr);
    ASSERT_NE(function->rootFunc_->programs_[0], nullptr);

    auto identities = BuildIdentities();
    std::vector<uint64_t> serialKeys;
    for (const auto& identity : identities) {
        serialKeys.push_back(MakeKernelKey(*function, identity));
    }

    // Ids are derived from task identity only, so parallel allocation must match serial keys exactly.
    std::vector<uint64_t> parallelKeys(identities.size(), 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < THREAD_NUM; ++t) {
        threads.emplace_back([&identities, &parallelKeys, function, t]() {
            for (size_t i = t; i < identities.size(); i += THREAD_NUM) {
                parallelKeys[i] = MakeKernelKey(*function, identities[i]);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(parallelKeys, serialKeys);
}

} // namespace npu::tile_fwk
