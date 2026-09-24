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
 * \file test_raw_tensor_data.cpp
 * \brief Unit tests for LogicalTensorData Dump/Save/Load/ToString to improve code coverage.
 */

#include <gtest/gtest.h>

#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

namespace {
constexpr const char* TMP_DIR = "/tmp/opencode";

template <typename T>
LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const T& val)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const std::vector<T>& vals)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

std::string tmpPath(const std::string& name) { return std::string(TMP_DIR) + "/" + name; }
} // namespace

class RawTensorDataTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        mkdir(TMP_DIR, 0755);
    }
};

// ===== DumpRange (nullptr branch) / DumpCoord / DumpData =====
// Shape {2,2,4}: rowSize=4, colSize=4. Data designed to cover:
//   - DumpRange: multi-element group (L62), single-element group (L60), comma (L56)
//   - DumpCoord: k!=0 comma branch (L92) via 3D shape
//   - DumpData: single-row group (L124), multi-row group (L126)
TEST_F(RawTensorDataTest, Dump_NullElementDumpList)
{
    // row0=[1,1,2,3], row1=[1,1,2,3] (same as row0)
    // row2=[4,4,5,6], row3=[4,4,5,6] (same as row2, diff from row1)
    std::vector<int32_t> data = {1, 1, 2, 3, 1, 1, 2, 3, 4, 4, 5, 6, 4, 4, 5, 6};
    auto tensor = makeTensorData(DT_INT32, {2, 2, 4}, data);

    std::string result = tensor->Dump(nullptr);
    EXPECT_FALSE(result.empty());
    // Multi-row range uses "..." (DumpData L126), single-row uses DumpCoord only (L124)
    EXPECT_NE(result.find("..."), std::string::npos);
}

// ===== DumpRange (non-null branch) =====
// Covers L44-47 (elementDumpList diff) and L65-68 (elementDumpList output)
TEST_F(RawTensorDataTest, Dump_WithElementDumpList)
{
    std::vector<int32_t> data = {1, 1, 2, 3, 1, 1, 2, 3, 4, 4, 5, 6, 4, 4, 5, 6};
    auto tensor = makeTensorData(DT_INT32, {2, 2, 4}, data);

    std::vector<ElementDump> dumpList;
    for (int i = 0; i < 16; i++) {
        ElementDump ed;
        ed.DumpElement(static_cast<int64_t>(data[i]));
        dumpList.push_back(ed);
    }

    std::string result = tensor->Dump(&dumpList);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("..."), std::string::npos);
}

// ===== Save / SaveFile / Load round-trip for all dtypes =====
// Covers the full dtype switch in Save (L155-198) and Load (L292-312)
TEST_F(RawTensorDataTest, SaveLoadRoundTrip_AllDtypes)
{
#define SAVE_LOAD_DTYPE(dtype, cppType, value)                                    \
    do {                                                                          \
        auto tensor = makeTensorData(dtype, {2, 2}, static_cast<cppType>(value)); \
        std::string path = tmpPath("rt_save_" #dtype ".bin");                     \
        tensor->Save(path);                                                       \
        tensor->SaveFile(path.c_str());                                           \
        auto loaded = LogicalTensorData::Load(path);                              \
        ASSERT_NE(loaded, nullptr);                                               \
        EXPECT_EQ(loaded->GetDataType(), dtype);                                  \
        EXPECT_EQ(loaded->GetShape(), tensor->GetShape());                        \
        std::remove(path.c_str());                                                \
    } while (0)

    SAVE_LOAD_DTYPE(DT_INT8, int8_t, 5);
    SAVE_LOAD_DTYPE(DT_INT16, int16_t, 5);
    SAVE_LOAD_DTYPE(DT_INT32, int32_t, 5);
    SAVE_LOAD_DTYPE(DT_INT64, int64_t, 5);
    SAVE_LOAD_DTYPE(DT_FP16, npu::tile_fwk::float16, 2.5f);
    SAVE_LOAD_DTYPE(DT_FP32, float, 2.5f);
    SAVE_LOAD_DTYPE(DT_BF16, npu::tile_fwk::bfloat16, 2.5f);
    SAVE_LOAD_DTYPE(DT_UINT8, uint8_t, 5);
    SAVE_LOAD_DTYPE(DT_UINT16, uint16_t, 5);
    SAVE_LOAD_DTYPE(DT_UINT32, uint32_t, 5);
    SAVE_LOAD_DTYPE(DT_UINT64, uint64_t, 5);
    SAVE_LOAD_DTYPE(DT_DOUBLE, double, 2.5);
    SAVE_LOAD_DTYPE(DT_BOOL, bool, true);

#undef SAVE_LOAD_DTYPE
}

// ===== Save default case (ASSERT false) =====
TEST_F(RawTensorDataTest, Save_InvalidDtypeThrows)
{
    auto data = std::make_shared<RawTensorData>(DT_BOTTOM, std::vector<int64_t>{2, 2});
    auto tensor = std::make_shared<LogicalTensorData>(data);
    EXPECT_THROW(tensor->Save(tmpPath("rt_save_bottom.bin")), Error);
    std::remove(tmpPath("rt_save_bottom.bin").c_str());
}

// ===== Load failure: nonexistent file (fopen returns nullptr) =====
TEST_F(RawTensorDataTest, Load_NonexistentFile)
{
    auto result = LogicalTensorData::Load(tmpPath("rt_nonexistent_file.bin"));
    EXPECT_EQ(result, nullptr);
}

// ===== Load failure: truncated head (fread != 1) =====
TEST_F(RawTensorDataTest, Load_TruncatedHead)
{
    std::string path = tmpPath("rt_truncated_head.bin");
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << "x"; // 1 byte, far less than sizeof(LogicalTensorDataHead) = 64
    }
    auto result = LogicalTensorData::Load(path);
    EXPECT_EQ(result, nullptr);
    std::remove(path.c_str());
}

// ===== Load failure: valid head but truncated data (second fread != size) =====
TEST_F(RawTensorDataTest, Load_TruncatedData)
{
    std::string path = tmpPath("rt_truncated_data.bin");
    {
        auto tensor = makeTensorData(DT_INT32, {2, 2}, std::vector<int32_t>{1, 2, 3, 4});
        tensor->Save(path);
        // Truncate to head size only (64 bytes), removing the 16 bytes of int32 data
        int ret = truncate(path.c_str(), 64);
        ASSERT_EQ(ret, 0);
    }
    auto result = LogicalTensorData::Load(path);
    EXPECT_EQ(result, nullptr);
    std::remove(path.c_str());
}

// ===== ToString unsigned branch (L242) =====
TEST_F(RawTensorDataTest, ToString_UnsignedBranch)
{
    auto tensor = makeTensorData(DT_UINT32, {2, 2}, std::vector<uint32_t>{10, 20, 30, 40});
    PrintOptions opts{2, 6, 1000, 80};
    std::string result = tensor->ToString(&opts);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("10"), std::string::npos);
    EXPECT_NE(result.find("40"), std::string::npos);
}

} // namespace npu::tile_fwk
