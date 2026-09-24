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
 * \file test_subfunction.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include <sstream>
#include "passes/pass_utils/subfunc_utils.h"

using namespace npu::tile_fwk;

class SubFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "SubFunctionTest SetUpTestCase" << std::endl; }

    static void TearDownTestCase() { std::cout << "SubFunctionTest TearDownTestCase" << std::endl; }

    void SetUp() override { std::cout << "SubFunctionTest SetUp" << std::endl; }

    void TearDown() override { std::cout << "SubFunctionTest TearDown" << std::endl; }
};

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_PrintInvokeInfo)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    subfuncInvokeInfo.PrintInvokeInfo("extra_info");
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_PrettyPrintInvokeInfo1)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    subfuncInvokeInfo.PrettyPrintInvokeInfo(123);
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_DumpInvokeInfo1)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    std::vector<int64_t> invokeParamVec(10, 10);
    subfuncInvokeInfo.DumpInvokeInfo(0, invokeParamVec.data());
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_LookupInvokeArgs1)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    std::tuple<int, int, int> tp1(123, 0, 0);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0), tp1);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0x10000000), tp1);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0x20000000), tp1);
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_Print1)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.Print("extra_info");

    subfuncInvokeInfo.SetGraphType(CoreType::AIC);
    EXPECT_EQ(subfuncInvokeInfo.GetGraphType(), CoreType::AIC);
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_ToJson)
{
    SubfuncInvokeInfoTy subfuncInvokeInfo;
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 1}, {16, 32}, {16, 32}, DataType::DT_FP32, true, nullptr, 20);
    subfuncInvokeInfo.RecordConnection(2, 3, 4, 123, {2, 3}, {64, 65}, {64, 65}, DataType::DT_FP16, nullptr, 30);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    inCasts.emplace_back(100, 7, nullptr, 88);
    inCasts.emplace_back(101, 8, nullptr, 99);
    subfuncInvokeInfo.RecordOutcast(4, 0, 2, 789, inCasts, {6, 7}, {128, 129}, {128, 129}, DataType::DT_INT32, nullptr,
                                    40);

    Json result = subfuncInvokeInfo.ToJson();

    ASSERT_TRUE(result.contains("incasts"));
    ASSERT_TRUE(result.contains("outcasts"));
    ASSERT_TRUE(result.contains("tensors"));

    ASSERT_EQ(result["incasts"].size(), 1);
    EXPECT_EQ(result["incasts"][0]["operandIdx"].get<int>(), 4);
    EXPECT_EQ(result["incasts"][0]["ddrId"].get<int>(), 123);
    EXPECT_EQ(result["incasts"][0]["shape"].get<std::vector<int64_t>>(), std::vector<int64_t>({64, 65}));
    EXPECT_EQ(result["incasts"][0]["offset"].get<std::vector<int64_t>>(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(result["incasts"][0]["dtype"].get<int>(), static_cast<int>(DataType::DT_FP16));

    ASSERT_EQ(result["outcasts"].size(), 1);
    EXPECT_EQ(result["outcasts"][0]["operandIdx"].get<int>(), 0);
    EXPECT_EQ(result["outcasts"][0]["ddrId"].get<int>(), 789);
    EXPECT_EQ(result["outcasts"][0]["shape"].get<std::vector<int64_t>>(), std::vector<int64_t>({128, 129}));
    EXPECT_EQ(result["outcasts"][0]["offset"].get<std::vector<int64_t>>(), std::vector<int64_t>({6, 7}));
    EXPECT_EQ(result["outcasts"][0]["dtype"].get<int>(), static_cast<int>(DataType::DT_INT32));
    EXPECT_EQ(result["outcasts"][0]["succEsgIds"].get<std::vector<int>>(), std::vector<int>({100, 101}));

    ASSERT_EQ(result["tensors"].size(), 1);
    EXPECT_EQ(result["tensors"][0]["operandIdx"].get<int>(), 1);
    EXPECT_EQ(result["tensors"][0]["ddrId"].get<int>(), 456);
    EXPECT_EQ(result["tensors"][0]["shape"].get<std::vector<int64_t>>(), std::vector<int64_t>({16, 32}));
    EXPECT_EQ(result["tensors"][0]["offset"].get<std::vector<int64_t>>(), std::vector<int64_t>({0, 1}));
    EXPECT_EQ(result["tensors"][0]["dtype"].get<int>(), static_cast<int>(DataType::DT_FP32));
    EXPECT_EQ(result["tensors"][0]["isOutput"].get<bool>(), true);
}

TEST_F(SubFunctionTest, SubfuncParam_PrettyPrint)
{
    SubfuncParam subfuncParam;
    subfuncParam.AppendTensorParam(1, 11, {16, 32}, {0, 1}, "tensor_sym", 0, "tensor_symbol", DataType::DT_FP32);
    subfuncParam.AppendIncastParam(2, 22, {64, 65}, {2, 3}, "incast_sym", 1, "incast_symbol", DataType::DT_FP16);
    subfuncParam.AppendOutcastParam(3, 33, 2, {128, 129}, {4, 5}, "outcast_sym", 2, "outcast_symbol",
                                    DataType::DT_INT32);

    std::ostringstream osm;
    subfuncParam.PrettyPrint(123, osm);
    std::string output = osm.str();

    EXPECT_NE(output.find("PARAM_LIST[123]"), std::string::npos);
    EXPECT_NE(output.find("tensor_sym"), std::string::npos);
    EXPECT_NE(output.find("INCAST"), std::string::npos);
    EXPECT_NE(output.find("incast_sym"), std::string::npos);
    EXPECT_NE(output.find("OUTCAST"), std::string::npos);
    EXPECT_NE(output.find("outcast_sym"), std::string::npos);
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_TopoSort)
{
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    subfuncTopoInfo.TopoSort();

    EXPECT_EQ(subfuncTopoInfo.IsEsgReady(1), true);
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_Print)
{
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    subfuncTopoInfo.SetMaxM(10);
    subfuncTopoInfo.Print();
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_DumpEachEntryInfo)
{
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    std::vector<int64_t> entryParam(10, 10);
    std::vector<int32_t> readyState(10, 10);
    Status status = subfuncTopoInfo.DumpEachEntryInfo(1, CoreType::AIC, 0, entryParam.data(),
                                                      entryParam.size() * sizeof(int64_t), readyState.data(),
                                                      readyState.size());

    EXPECT_EQ(status, SUCCESS);
    const int64_t graphId = static_cast<int64_t>((static_cast<uint64_t>(CoreType::AIC) << 32) | 1U);
    EXPECT_EQ(entryParam[0], graphId);
    EXPECT_EQ(entryParam[1], 0);
    EXPECT_EQ(entryParam[2], 1);
    EXPECT_EQ(readyState[2], 0);
    EXPECT_EQ(readyState[3], static_cast<int32_t>(CoreType::AIC));
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_DumpEachEntryInfoEntryParamOverflow)
{
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});

    std::vector<int64_t> entryParam(2, 10);
    std::vector<int32_t> readyState(4, 10);
    Status status = subfuncTopoInfo.DumpEachEntryInfo(1, CoreType::AIC, 0, entryParam.data(),
                                                      entryParam.size() * sizeof(int64_t), readyState.data(),
                                                      readyState.size());

    EXPECT_EQ(status, FAILED);
    EXPECT_EQ(entryParam[0], 10);
    EXPECT_EQ(entryParam[1], 10);
    EXPECT_EQ(readyState[2], 10);
    EXPECT_EQ(readyState[3], 10);
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_DumpEachEntryInfoReadyStateOverflow)
{
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});

    std::vector<int64_t> entryParam(10, 10);
    std::vector<int32_t> readyState(3, 10);
    Status status = subfuncTopoInfo.DumpEachEntryInfo(1, CoreType::AIC, 0, entryParam.data(),
                                                      entryParam.size() * sizeof(int64_t), readyState.data(),
                                                      readyState.size());

    EXPECT_EQ(status, FAILED);
    EXPECT_EQ(entryParam[0], 10);
    EXPECT_EQ(entryParam[1], 10);
    EXPECT_EQ(entryParam[2], 10);
    EXPECT_EQ(readyState[2], 10);
}
