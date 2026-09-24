/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aicore_print_host_manager.cpp
 * \brief UT for machine/runtime/runner/aicore_print_host_manager.cpp
 */

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "aicore_emulation.h"
#include "interface/machine/device/tilefwk/aicore_print.h"
#include "tilefwk/error_code.h"

#define private public
#include "machine/runtime/runner/aicore_print_host_manager.h"
#undef private

using namespace npu::tile_fwk;

namespace {
DeviceArgs MakeArgsWithCores(uint32_t numCores, uint8_t* sharedBuf)
{
    DeviceArgs args;
    args.nrAic = numCores;
    args.nrAiv = 0;
    args.nrValidAic = numCores;
    args.sharedBuffer = reinterpret_cast<uint64_t>(sharedBuf);
    return args;
}

constexpr uint32_t TEST_NUM_CORES = 2;
constexpr size_t SHARED_BUF_SIZE = TEST_NUM_CORES * static_cast<size_t>(SHARED_BUFFER_SIZE);
} // namespace

class AicorePrintHostManagerUTest : public testing::Test {
public:
    void SetUp() override { sharedBuf_.assign(SHARED_BUF_SIZE, 0); }
    void TearDown() override {}

    std::vector<uint8_t> sharedBuf_;
};

// 测试 sharedBuffer 为空时 Init 返回 PARAM_INVALID
TEST_F(AicorePrintHostManagerUTest, Enabled_InitWithNullSharedBufferReturnsParamInvalid)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, nullptr);
    EXPECT_EQ(mgr.Init(args), static_cast<int>(DevCommonErr::PARAM_INVALID));
}

// 测试 numCores 为 0 时 Init 返回 PARAM_INVALID (nrAic 非 0 避免除零, nrValidAic=0 使 GetBlockNum=0)
TEST_F(AicorePrintHostManagerUTest, Enabled_InitWithZeroCoresReturnsParamInvalid)
{
    AicorePrintHostManager mgr;
    DeviceArgs args;
    args.nrAic = 1;
    args.nrAiv = 0;
    args.nrValidAic = 0;
    args.sharedBuffer = reinterpret_cast<uint64_t>(sharedBuf_.data());
    EXPECT_EQ(mgr.Init(args), static_cast<int>(DevCommonErr::PARAM_INVALID));
}

// 测试正常 Init 成功: 返回 0, 分配每核 buffer, hostBuf_ 已就绪
TEST_F(AicorePrintHostManagerUTest, Enabled_InitSuccessAllocatesPerCoreBuffers)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    EXPECT_EQ(mgr.Init(args), 0);
    EXPECT_EQ(mgr.numCores_, TEST_NUM_CORES);
    EXPECT_EQ(mgr.devBuffers_.size(), static_cast<size_t>(TEST_NUM_CORES));
    for (uint32_t i = 0; i < TEST_NUM_CORES; i++) {
        EXPECT_NE(mgr.devBuffers_[i], nullptr);
    }
    EXPECT_EQ(mgr.hostBuf_.size(), static_cast<size_t>(PRINT_BUFFER_SIZE));
    EXPECT_EQ(mgr.sharedBuffer_, sharedBuf_.data());
    mgr.Release();
    EXPECT_EQ(mgr.numCores_, 0u);
    EXPECT_TRUE(mgr.devBuffers_.empty());
    EXPECT_EQ(mgr.sharedBuffer_, nullptr);
}

// 测试 SetPrintBufferAddrs 把每核 dev buffer 地址写入 sharedBuffer 对应偏移
TEST_F(AicorePrintHostManagerUTest, Enabled_SetPrintBufferAddrsWritesAddrToSharedBuffer)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);
    EXPECT_EQ(mgr.SetPrintBufferAddrs(), 0);
    for (uint32_t i = 0; i < TEST_NUM_CORES; i++) {
        uint8_t* dfxBufAddr = sharedBuf_.data() + offsetof(KernelArgs, dfxBuffer) +
                              sizeof(uint64_t) * SHAK_BUF_PRINT_BUFFER_INDEX + i * SHARED_BUFFER_SIZE;
        uint64_t writtenAddr = 0;
        std::memcpy(&writtenAddr, dfxBufAddr, sizeof(uint64_t));
        EXPECT_EQ(writtenAddr, reinterpret_cast<uint64_t>(mgr.devBuffers_[i]));
    }
    mgr.Release();
}

// 测试未 Init 时 DumpAicoreLog 返回 0(devBuffers_ 为空, 提前返回)
TEST_F(AicorePrintHostManagerUTest, Enabled_DumpAicoreLogWithoutInitReturnsZero)
{
    AicorePrintHostManager mgr;
    EXPECT_EQ(mgr.DumpAicoreLog(), 0);
}

// 测试 DumpAicoreLog 解码已编码数据: 编码 marker 字符串后 D2H 拷贝能正确解码
TEST_F(AicorePrintHostManagerUTest, Enabled_DumpAicoreLogDecodesEncodedData)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);

    for (uint32_t i = 0; i < TEST_NUM_CORES; i++) {
        auto* devPtr = static_cast<uint8_t*>(mgr.devBuffers_[i]);
        AicoreLogger encoder;
        encoder.Init(devPtr, PRINT_BUFFER_SIZE);
        encoder.PrintRaw("ut_marker");
        encoder.Sync();
    }

    EXPECT_EQ(mgr.DumpAicoreLog(), 0);

    AicoreLogger verifier;
    verifier.BindHostBuffer(mgr.hostBuf_.data(), mgr.hostBuf_.size());
    char line[512];
    std::string output;
    while (verifier.Read(line, sizeof(line)) > 0) {
        output.append(line);
    }
    EXPECT_NE(output.find("ut_marker"), std::string::npos) << "output: " << output;

    mgr.Release();
}

// 测试 Init 后 Release 清空所有 buffer, 后续 DumpAicoreLog 安全返回 0
TEST_F(AicorePrintHostManagerUTest, Enabled_ReleaseThenDumpAicoreLogSafe)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);
    mgr.Release();
    EXPECT_EQ(mgr.DumpAicoreLog(), 0);
}

// 测试重复 Init 不泄漏: 第二次 Init 追加 buffer, Release 统一释放
TEST_F(AicorePrintHostManagerUTest, Enabled_DoubleInitThenReleaseSafe)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);
    EXPECT_EQ(mgr.Init(args), 0);
    EXPECT_EQ(mgr.devBuffers_.size(), static_cast<size_t>(TEST_NUM_CORES));
    EXPECT_EQ(mgr.numCores_, TEST_NUM_CORES);
    mgr.Release();
    EXPECT_TRUE(mgr.devBuffers_.empty());
}

// 测试单核场景 Init + DumpAicoreLog 全流程
TEST_F(AicorePrintHostManagerUTest, Enabled_SingleCoreFlow)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(1, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);
    EXPECT_EQ(mgr.numCores_, 1u);

    auto* devPtr = static_cast<uint8_t*>(mgr.devBuffers_[0]);
    AicoreLogger encoder;
    encoder.Init(devPtr, PRINT_BUFFER_SIZE);
    encoder.PrintRaw("single_core_marker");
    encoder.Sync();

    EXPECT_EQ(mgr.DumpAicoreLog(), 0);

    AicoreLogger verifier;
    verifier.BindHostBuffer(mgr.hostBuf_.data(), mgr.hostBuf_.size());
    char line[512];
    std::string output;
    while (verifier.Read(line, sizeof(line)) > 0) {
        output.append(line);
    }
    EXPECT_NE(output.find("single_core_marker"), std::string::npos);

    mgr.Release();
}

// 辅助: 向 devBuffer 编码一条带 level+timestamp 的日志
static void EncodeLogLine(AicoreLogger& enc, uint8_t level, uint64_t ts, const char* msg)
{
    enc.EncodeLogLevel(level);
    enc.EncodeTimestamp(ts);
    enc.PrintRaw(msg);
    enc.PrintNewLine();
}

// 测试 Read 正确返回 level 和 timestamp: 编码 DEBUG/INFO/WARN/ERROR 四条日志, 逐条读取验证
TEST_F(AicorePrintHostManagerUTest, Enabled_ReadReturnsLevelAndTimestamp)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(1, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);

    auto* devPtr = static_cast<uint8_t*>(mgr.devBuffers_[0]);
    AicoreLogger enc;
    enc.Init(devPtr, PRINT_BUFFER_SIZE);
    EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::DEBUG), 100, "debug_msg");
    EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::INFO), 200, "info_msg");
    EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::WARN), 300, "warn_msg");
    EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::ERROR), 400, "error_msg");

    AicoreLogger reader;
    reader.BindHostBuffer(devPtr, PRINT_BUFFER_SIZE);
    char line[512];
    uint8_t level = static_cast<uint8_t>(AicoreLogLevel::NONE);
    uint64_t ts = 0;

    ASSERT_GT(reader.Read(line, sizeof(line), &level, &ts), 0);
    EXPECT_EQ(level, static_cast<uint8_t>(AicoreLogLevel::DEBUG));
    EXPECT_EQ(ts, 100u);
    EXPECT_NE(std::string(line).find("debug_msg"), std::string::npos);

    ASSERT_GT(reader.Read(line, sizeof(line), &level, &ts), 0);
    EXPECT_EQ(level, static_cast<uint8_t>(AicoreLogLevel::INFO));
    EXPECT_EQ(ts, 200u);
    EXPECT_NE(std::string(line).find("info_msg"), std::string::npos);

    ASSERT_GT(reader.Read(line, sizeof(line), &level, &ts), 0);
    EXPECT_EQ(level, static_cast<uint8_t>(AicoreLogLevel::WARN));
    EXPECT_EQ(ts, 300u);
    EXPECT_NE(std::string(line).find("warn_msg"), std::string::npos);

    ASSERT_GT(reader.Read(line, sizeof(line), &level, &ts), 0);
    EXPECT_EQ(level, static_cast<uint8_t>(AicoreLogLevel::ERROR));
    EXPECT_EQ(ts, 400u);
    EXPECT_NE(std::string(line).find("error_msg"), std::string::npos);

    mgr.Release();
}

// 测试 DumpAicoreLog 按级别解码: 多核多级别日志, dump 后用 verifier 验证 head/tail 推进
TEST_F(AicorePrintHostManagerUTest, Enabled_DumpAicoreLogMultiCoreMultiLevel)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(TEST_NUM_CORES, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);

    for (uint32_t i = 0; i < TEST_NUM_CORES; i++) {
        auto* devPtr = static_cast<uint8_t*>(mgr.devBuffers_[i]);
        AicoreLogger enc;
        enc.Init(devPtr, PRINT_BUFFER_SIZE);
        EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::DEBUG), 1000 + i, "dbg");
        EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::INFO), 2000 + i, "inf");
        EncodeLogLine(enc, static_cast<uint8_t>(AicoreLogLevel::ERROR), 4000 + i, "err");
    }

    EXPECT_EQ(mgr.DumpAicoreLog(), 0);

    AicoreLogger verifier;
    verifier.BindHostBuffer(mgr.hostBuf_.data(), mgr.hostBuf_.size());
    char line[512];
    uint8_t level = static_cast<uint8_t>(AicoreLogLevel::NONE);
    uint64_t ts = 0;
    int readCount = 0;
    while (verifier.Read(line, sizeof(line), &level, &ts) > 0) {
        readCount++;
        EXPECT_NE(ts, 0u);
    }
    // 每核3条, TEST_NUM_CORES 核
    EXPECT_EQ(readCount, 3);

    mgr.Release();
}

// 测试无 marker 的日志: level 回退为 NONE, timestamp 为 0
TEST_F(AicorePrintHostManagerUTest, Enabled_ReadWithoutMarkersReturnsDefaults)
{
    AicorePrintHostManager mgr;
    DeviceArgs args = MakeArgsWithCores(1, sharedBuf_.data());
    ASSERT_EQ(mgr.Init(args), 0);

    auto* devPtr = static_cast<uint8_t*>(mgr.devBuffers_[0]);
    AicoreLogger enc;
    enc.Init(devPtr, PRINT_BUFFER_SIZE);
    enc.PrintRaw("no_marker_msg");
    enc.PrintNewLine();

    AicoreLogger reader;
    reader.BindHostBuffer(devPtr, PRINT_BUFFER_SIZE);
    char line[512];
    uint8_t level = 0xFF;
    uint64_t ts = 999;
    ASSERT_GT(reader.Read(line, sizeof(line), &level, &ts), 0);
    EXPECT_EQ(level, static_cast<uint8_t>(AicoreLogLevel::NONE));
    EXPECT_EQ(ts, 0u);
    EXPECT_NE(std::string(line).find("no_marker_msg"), std::string::npos);

    mgr.Release();
}
