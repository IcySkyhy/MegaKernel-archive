/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#define private public
#include "utils/host_log/dlog_handler.h"
#include "utils/host_log/log_manager.h"
#undef private

namespace npu::tile_fwk {

TEST(DLogHandlerExtraTest, CheckLogLevel_NullFunc)
{
    DLogHandler handler;
    handler.checkLevelFunc_ = nullptr;
    EXPECT_EQ(handler.CheckLogLevel(0, 0), 0);
}

TEST(DLogHandlerExtraTest, GetLogLevel_NullFunc)
{
    DLogHandler handler;
    handler.getLevelFunc_ = nullptr;
    int32_t enableEvent = 0;
    EXPECT_EQ(handler.GetLogLevel(0, &enableEvent), -1);
}

TEST(DLogHandlerExtraTest, SetLogLevel_NullFunc)
{
    DLogHandler handler;
    handler.setLevelFunc_ = nullptr;
    EXPECT_EQ(handler.SetLogLevel(0, 0, 0), -1);
}

TEST(DLogHandlerExtraTest, CloseHandle_NullHandle)
{
    DLogHandler handler;
    handler.handle_ = nullptr;
    handler.CloseHandle();
    EXPECT_EQ(handler.handle_, nullptr);
}

TEST(DLogHandlerExtraTest, IsAvailable_NullHandle)
{
    DLogHandler handler;
    handler.handle_ = nullptr;
    handler.checkLevelFunc_ = nullptr;
    handler.logRecordFunc_ = nullptr;
    handler.getLevelFunc_ = nullptr;
    handler.setLevelFunc_ = nullptr;
    EXPECT_FALSE(handler.IsAvailable());
}

TEST(LogManagerExtraTest, SetLogLevel_InvalidLevel)
{
    LogManager mgr;
    mgr.SetLogLevel(LogLevel::NONE);
    EXPECT_NE(mgr.level_, LogLevel::NONE);
}

TEST(LogManagerExtraTest, SetLogLevel_ValidLevel)
{
    LogManager mgr;
    mgr.SetLogLevel(LogLevel::DEBUG);
    EXPECT_EQ(mgr.level_, LogLevel::DEBUG);
    mgr.SetLogLevel(LogLevel::WARN);
    EXPECT_EQ(mgr.level_, LogLevel::WARN);
}

TEST(LogManagerExtraTest, CheckLevel_EventEnabled)
{
    LogManager mgr;
    mgr.enableEvent_ = true;
    EXPECT_TRUE(mgr.CheckLevel(LogLevel::EVENT));
}

TEST(LogManagerExtraTest, CheckLevel_EventDisabled)
{
    LogManager mgr;
    mgr.enableEvent_ = false;
    EXPECT_FALSE(mgr.CheckLevel(LogLevel::EVENT));
}

TEST(LogManagerExtraTest, CheckLevel_InvalidLevel)
{
    LogManager mgr;
    EXPECT_FALSE(mgr.CheckLevel(LogLevel::NONE));
}

TEST(LogManagerExtraTest, ConstructMsgHeader_Valid)
{
    LogManager mgr;
    LogMsg logMsg{};
    mgr.ConstructMsgHeader(LogLevel::INFO, logMsg);
    EXPECT_GT(logMsg.length, 0u);
    std::string msg(logMsg.msg);
    EXPECT_NE(msg.find("[INFO ]"), std::string::npos);
    EXPECT_NE(msg.find("PYPTO"), std::string::npos);
}

TEST(LogManagerExtraTest, ConstructMsgTail_AddsNewline)
{
    LogManager mgr;
    LogMsg logMsg{};
    logMsg.msg[0] = 'H';
    logMsg.msg[1] = 'i';
    logMsg.length = 2;
    mgr.ConstructMsgTail(logMsg);
    EXPECT_EQ(logMsg.msg[2], '\n');
    EXPECT_EQ(logMsg.length, 3u);
}

TEST(LogManagerExtraTest, ConstructMsgTail_AlreadyHasNewline)
{
    LogManager mgr;
    LogMsg logMsg{};
    logMsg.msg[0] = 'H';
    logMsg.msg[1] = '\n';
    logMsg.length = 2;
    mgr.ConstructMsgTail(logMsg);
    EXPECT_EQ(logMsg.length, 2u);
}

TEST(LogManagerExtraTest, ConstructMsgTail_MaxLength)
{
    LogManager mgr;
    LogMsg logMsg{};
    logMsg.length = MAX_MSG_LENGTH;
    logMsg.msg[MAX_MSG_LENGTH - 1] = 'x';
    mgr.ConstructMsgTail(logMsg);
    EXPECT_EQ(logMsg.msg[MAX_MSG_LENGTH - 1], '\n');
}

TEST(LogManagerExtraTest, WriteToStdOut_Valid)
{
    LogManager mgr;
    LogMsg logMsg{};
    const char* testMsg = "test stdout write\n";
    size_t len = strlen(testMsg);
    memcpy(logMsg.msg, testMsg, len);
    logMsg.length = len;
    mgr.WriteToStdOut(logMsg);
}

TEST(LogManagerExtraTest, Destructor_ClosesStreams)
{
    auto* mgr = new LogManager();
    mgr->enableStdOut_ = true;
    mgr->level_ = LogLevel::DEBUG;
    delete mgr;
}

TEST(LogManagerExtraTest, WriteToFile_StreamNotOpen)
{
    LogManager mgr;
    mgr.enableStdOut_ = false;
    LogMsg logMsg{};
    const char* testMsg = "test file write\n";
    size_t len = strlen(testMsg);
    memcpy(logMsg.msg, testMsg, len);
    logMsg.length = len;
    mgr.WriteToFile(logMsg);
}

TEST(LogManagerExtraTest, WriteMessage_ToStdOut)
{
    LogManager mgr;
    mgr.enableStdOut_ = true;
    LogMsg logMsg{};
    const char* testMsg = "test message\n";
    size_t len = strlen(testMsg);
    memcpy(logMsg.msg, testMsg, len);
    logMsg.length = len;
    mgr.WriteMessage(logMsg);
}

TEST(LogManagerExtraTest, WriteMessage_ToFile)
{
    LogManager mgr;
    mgr.enableStdOut_ = false;
    LogMsg logMsg{};
    const char* testMsg = "test message\n";
    size_t len = strlen(testMsg);
    memcpy(logMsg.msg, testMsg, len);
    logMsg.length = len;
    mgr.WriteMessage(logMsg);
}

} // namespace npu::tile_fwk
