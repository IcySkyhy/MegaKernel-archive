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
 * \file test_gather_in_l1.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <random>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "interface/program/program.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <stdexcept>

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

// ----------------- 配置结构体（含类型） -----------------
// IndexT  : topk_indices / page_table 的整数类型
// DataT   : buffer / golden_result 的数据类型
template <typename IndexT, typename DataT>
struct PageAttentionTestConfig {
    using IndexType = IndexT;
    using DataType = DataT;

    int topk_count;         // topk 的 k 值：选出的 token 个数
    int num_logical_blocks; // 逻辑块个数（page_table 长度）
    int num_buffer_tokens;  // buffer 第一维长度：物理 token 容量
    int hidden_dim;         // buffer 第二维长度：隐藏维度大小
    int block_size;         // 每个块里有多少个 token
};

// ----------------- 基础打印工具 -----------------
template <typename T>
void print_1d(const std::vector<T>& v, const std::string& name, int max_print = 32)
{
    std::cout << name << " (size=" << v.size() << "): ";
    int n = std::min<int>(v.size(), max_print);
    for (int i = 0; i < n; ++i) {
        std::cout << v[i];
        if (i + 1 != n)
            std::cout << ", ";
    }
    if ((int)v.size() > max_print)
        std::cout << " ...";
    std::cout << "\n";
}

template <typename T>
void print_2d(const std::vector<T>& v, int rows, int cols, const std::string& name, int max_rows = 8)
{
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    int r_limit = std::min(rows, max_rows);
    for (int r = 0; r < r_limit; ++r) {
        std::cout << "  [";
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(10) << v[r * cols + c];
            if (c + 1 != cols)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
    if (rows > r_limit) {
        std::cout << "  ... (" << (rows - r_limit) << " more rows)\n";
    }
}

// ----------------- 参数合法性检查 -----------------
template <typename Config>
bool validate_config(const Config& cfg, std::string& err)
{
    if (cfg.topk_count <= 0 || cfg.num_logical_blocks <= 0 || cfg.num_buffer_tokens <= 0 || cfg.hidden_dim <= 0 ||
        cfg.block_size <= 0) {
        err = "topk_count, num_logical_blocks, num_buffer_tokens, hidden_dim, block_size 都必须为正整数";
        return false;
    }

    // 每个逻辑块有 block_size 个 token，
    // 总逻辑 token 数 = num_logical_blocks * block_size
    int total_logical_tokens = cfg.num_logical_blocks * cfg.block_size;

    // 强制 topk 的 k 不超过逻辑 token 总数
    if (cfg.topk_count > total_logical_tokens) {
        err = "topk_count 必须 <= num_logical_blocks * block_size（topk 的 k 不能超过逻辑 token 总数）";
        return false;
    }

    // 物理块总数 = num_buffer_tokens / block_size
    if (cfg.num_buffer_tokens < cfg.block_size) {
        err = "num_buffer_tokens 必须至少 >= block_size,才能容纳一个物理块";
        return false;
    }
    int num_physical_blocks = cfg.num_buffer_tokens / cfg.block_size;
    if (num_physical_blocks <= 0) {
        err = "num_buffer_tokens / block_size 必须 > 0";
        return false;
    }

    return true;
}

// ----------------- 构造 buffer[num_buffer_tokens, hidden_dim] -----------------
template <typename Config>
std::vector<typename Config::DataType> make_buffer(const Config& cfg)
{
    using DataType = typename Config::DataType;
    std::vector<DataType> buffer(cfg.num_buffer_tokens * cfg.hidden_dim);

    for (int token_index = 0; token_index < cfg.num_buffer_tokens; ++token_index) {
        for (int h = 0; h < cfg.hidden_dim; ++h) {
            // 简单可区分的 pattern：1000 * token_index + hidden_idx
            buffer[token_index * cfg.hidden_dim + h] = static_cast<DataType>(10.0f * token_index + h);
        }
    }
    return buffer;
}

// ----------------- 构造 page_table[1, num_logical_blocks] -----------------
// 使用 Config::IndexType，代表逻辑块 -> 物理块映射的类型
// 随机选取一个合法的物理块 ID（0..num_physical_blocks-1）作为映射，与实际网络中不符
// 实际网络中，除了前缀或者swap，一般逻辑和物理都是一一对应的，单轮对话中也不会出现多个逻辑映射同一个物理块。
// 但是模拟过程中，这个并不影响功能
template <typename Config>
std::vector<typename Config::IndexType> make_page_table(const Config& cfg, uint32_t seed = 42)
{
    using IndexType = typename Config::IndexType;

    int num_physical_blocks = static_cast<int>(std::ceil(cfg.num_buffer_tokens / cfg.block_size));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, num_physical_blocks - 1);

    std::vector<IndexType> page_table(cfg.num_logical_blocks);
    for (int logical_block_id = 0; logical_block_id < cfg.num_logical_blocks; ++logical_block_id) {
        page_table[logical_block_id] = static_cast<IndexType>(dist(rng));
    }
    return page_table;
}

// ----------------- 构造 topk_indices[1, topk_count] -----------------
// 使用 Config::IndexType，代表逻辑 token id 类型
// 与网络不符的，实际网络中topk的应该不会重复
template <typename Config>
std::vector<typename Config::IndexType> make_topk_indices(const Config& cfg, uint32_t seed = 123)
{
    using IndexType = typename Config::IndexType;

    int total_logical_tokens = cfg.num_logical_blocks * cfg.block_size;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, total_logical_tokens - 1);

    std::vector<IndexType> indices(cfg.topk_count);
    for (int i = 0; i < cfg.topk_count; ++i) {
        indices[i] = static_cast<IndexType>(dist(rng));
    }
    return indices;
}

// ----------------- 逻辑 index -> 物理 index 的核心函数 -----------------
template <typename Config>
typename Config::IndexType compute_physical_index(typename Config::IndexType logical_index,
                                                  const std::vector<typename Config::IndexType>& page_table,
                                                  const Config& cfg)
{
    using IndexType = typename Config::IndexType;

    IndexType logical_block_id = logical_index / static_cast<IndexType>(cfg.block_size);
    IndexType physical_block_id = page_table[logical_block_id];
    IndexType block_offset = logical_index % static_cast<IndexType>(cfg.block_size);
    IndexType physical_index = physical_block_id * static_cast<IndexType>(cfg.block_size) + block_offset;
    return physical_index;
}

// ----------------- 根据 pageattention 逻辑进行 gather -----------------
// topk_indices   : [1, topk_count] flatten -> size = topk_count
// page_table     : [1, num_logical_blocks] -> size = num_logical_blocks
// buffer         : [num_buffer_tokens, hidden_dim] -> size = num_buffer_tokens * hidden_dim
// 输出 result    : [topk_count, hidden_dim] -> size = topk_count * hidden_dim
template <typename Config>
void gather_golden(const std::vector<typename Config::IndexType>& topk_indices,
                   const std::vector<typename Config::IndexType>& page_table,
                   const std::vector<typename Config::DataType>& buffer, const Config& cfg,
                   std::vector<typename Config::DataType>& result)
{
    using IndexType = typename Config::IndexType;
    // using DataType = typename Config::DataType;

    if (static_cast<int>(topk_indices.size()) != cfg.topk_count) {
        throw std::runtime_error("topk_indices.size() != topk_count");
    }
    if (static_cast<int>(page_table.size()) != cfg.num_logical_blocks) {
        throw std::runtime_error("page_table.size() != num_logical_blocks");
    }
    if (static_cast<int>(buffer.size()) != cfg.num_buffer_tokens * cfg.hidden_dim) {
        throw std::runtime_error("buffer.size() != num_buffer_tokens * hidden_dim");
    }

    result.resize(cfg.topk_count * cfg.hidden_dim);

    int total_logical_tokens = cfg.num_logical_blocks * cfg.block_size;

    for (int j = 0; j < cfg.topk_count; ++j) {
        IndexType logical_index = topk_indices[j];

        // 逻辑 index 范围 [0, num_logical_blocks * block_size)
        if (logical_index < 0 || logical_index >= static_cast<IndexType>(total_logical_tokens)) {
            throw std::runtime_error("logical_index 越界: topk_indices[" + std::to_string(j) + "]");
        }
        IndexType physical_index = compute_physical_index<Config>(logical_index, page_table, cfg);
        if (physical_index < 0 || physical_index >= static_cast<IndexType>(cfg.num_buffer_tokens)) {
            throw std::runtime_error("physical_index 越界: " + std::to_string(physical_index));
        }

        // 拷贝 hidden 维：buffer[physical_index, :] -> result[j, :]
        for (int h = 0; h < cfg.hidden_dim; ++h) {
            result[j * cfg.hidden_dim + h] = buffer[static_cast<int>(physical_index) * cfg.hidden_dim + h];
        }
    }
}

class GatherInUBTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
    void SetUp() override
    {
        TestSuite_STest_Ops_Aihac::SetUp();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }
    void TearDown() override
    {
        config::SetHostOption(COMPILE_STAGE, 0);
        TestSuite_STest_Ops_Aihac::TearDown();
    }
};

template <typename Config>
void BasicGatherTest(Config& cfg)
{
    Shape srcShapes{cfg.num_buffer_tokens, cfg.hidden_dim}; // 网络中，kvcache对应的内存
    Shape offsetsShapes{1, cfg.topk_count};                 // topk的结果
    Shape pageTableShapes{1, cfg.num_logical_blocks};       // page attention 对应的页表
    Shape dstShapes{cfg.topk_count, cfg.hidden_dim};        // 结果，将topk个数据拿出来

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor pageTable(DT_INT32, pageTableShapes, "pageTable");
    Tensor dst(DT_FP16, dstShapes, "dst");

    std::string err;
    if (!validate_config<Config>(cfg, err)) {
        std::cerr << "配置非法: " << err << "\n";
        return;
    }
    auto srcData = make_buffer<Config>(cfg);
    auto offsetsData = make_topk_indices<Config>(cfg, /*seed=*/123);
    auto pageTableData = make_page_table<Config>(cfg, /*seed=*/42);
    // 4. 用 pageattention 逻辑做 gather，生成 golden 结果
    std::vector<typename Config::DataType> golden;
    gather_golden<Config>(offsetsData, pageTableData, srcData, cfg, golden);
    std::cout << "simu finished" << std::endl;

    FUNCTION("test", {src, offsets, pageTable}, {dst})
    {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1))
        {
            (void)sIdx;
            TileShape::Current().SetVecTile({32, 64});

            std::vector<SymbolicScalar> srcValidShape = {src.GetShape()[0], src.GetShape()[1]};
            Tensor dynSrc = View(src, src.GetShape(), srcValidShape, {0, 0});

            std::vector<SymbolicScalar> offsetsValidShape = {offsets.GetShape()[0], offsets.GetShape()[1]};
            Tensor dynOffsets = View(offsets, offsets.GetShape(), offsetsValidShape, {0, 0});

            dst = experimental::GatherInUB(dynSrc, dynOffsets, pageTable, cfg.block_size, -2);
        }
    }
    std::cout << "compile finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<float16>(src, srcData),
                                             RawTensorData::CreateTensor<int32_t>(offsets, offsetsData),
                                             RawTensorData::CreateTensor<int32_t>(pageTable, pageTableData)});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float16>(dst, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    int maxErrorPrintNum = 50;
    int curErrorPrintNum = 0;
    float eps = 1e-6f;
    for (size_t i = 0; i < golden.size(); i++) {
        auto actual = ((float16*)out->data())[i];
        auto expect = golden[i];
        if (fabs(actual - expect) > eps && curErrorPrintNum < maxErrorPrintNum) {
            std::cout << i << ": output: " << actual << "; expect: " << expect << std::endl;
            curErrorPrintNum++;
        }
    }
    EXPECT_TRUE(resultCmp(golden, (float16*)out->data(), eps));
}

TEST_F(GatherInUBTest, gather_in_a_)
{
    using Config = PageAttentionTestConfig<int32_t, float16>;
    Config cfg;
    cfg.topk_count = 8;         // topk结果
    cfg.num_logical_blocks = 3; // 逻辑块个数
    cfg.num_buffer_tokens = 32; // buffer token 维度（物理 token 容量）
    cfg.hidden_dim = 4;         // 隐藏维度大小
    cfg.block_size = 4;         // 每个块的 token 数
    BasicGatherTest(cfg);
}
TEST_F(GatherInUBTest, gather_in_a)
{
    using Config = PageAttentionTestConfig<int32_t, float16>;
    Config cfg;
    cfg.topk_count = 512;         // topk 结果
    cfg.num_logical_blocks = 8;   // 逻辑块个数，
    cfg.num_buffer_tokens = 2048; // buffer token 维度（物理 token 容量）
    cfg.hidden_dim = 256;         // 隐藏维度大小
    cfg.block_size = 128;         // 每个块的 token 数
    BasicGatherTest(cfg);
}
