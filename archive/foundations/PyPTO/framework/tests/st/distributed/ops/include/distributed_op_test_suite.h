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
 * \file distributed_op_test_suite.h
 * \brief
 */

#ifndef DISTRIBUTED_OP_TEST_SUITE_H
#define DISTRIBUTED_OP_TEST_SUITE_H
#include <string>

namespace npu::tile_fwk {
namespace Distributed {

struct OpTestParam {
    char group[128]{0};
    int rankSize;
    int rankId;
    int worldRankId;
};

void TestAllGatherAttentionPostReducescatter(OpTestParam& testParam, std::string& goldenDir);
template <typename T>
void TestAllGather(OpTestParam& testParam, std::string& goldenDir);
template <typename T>
void TestReduceScatter(OpTestParam& testParam, std::string& goldenDir);
template <typename T>
void TestAllReduce(OpTestParam& testParam, std::string& goldenDir);
template <typename T>
void TestAllReduceAddAllReduce(OpTestParam& testParam, std::string& goldenDir);
template <typename T>
void Test2GroupsAllGather(OpTestParam& testParam, std::string& goldenDir);
} // namespace Distributed
} // namespace npu::tile_fwk

#endif // DISTRIBUTED_OP_TEST_SUITE_H
