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
 * \file test_host_log.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <dirent.h>
#include <sys/syscall.h>
#include <regex>
#include <iostream>
#define private public
#include "utils/host_log/log_manager.h"
#include "utils/host_log/log_module_manager.h"
#undef private
#include "tilefwk/pypto_fwk_log.h"
#include "utils/host_log/dlog_handler.h"

namespace npu::tile_fwk {
class TestHostLog : public testing::Test {
public:
    void SetUp() override
    {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_GLOBAL_EVENT_ENABLE");
        unsetenv("ASCEND_HOST_LOG_FILE_NUM");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
        unsetenv("ASCEND_WORK_PATH");
    }
    void TearDown() override
    {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_GLOBAL_EVENT_ENABLE");
        unsetenv("ASCEND_HOST_LOG_FILE_NUM");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
        unsetenv("ASCEND_WORK_PATH");
    }
    uint64_t GetTestThreadId()
    {
        thread_local uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
        return tid;
    }
    size_t GetLogFileSizeOfSpecifiedDir(const std::string& dirPath, const std::string& filePrefix)
    {
        DIR* dir = opendir(dirPath.c_str());
        if (dir == nullptr) {
            return 0;
        }
        size_t fileSize = 0;
        struct dirent* dirp = nullptr;
        while ((dirp = readdir(dir)) != nullptr) {
            if (dirp->d_name[0] == '.') {
                continue;
            }
            std::string fileName = dirp->d_name;
            if (fileName.find(filePrefix) != 0) {
                continue;
            }
            if (fileName.find(".log") == std::string::npos) {
                continue;
            }
            fileSize++;
        }
        closedir(dir);
        return fileSize;
    }
    void RecoreLog(LogManager& log_manager, const LogLevel logLevel, const char* fmt, ...)
    {
        va_list list;
        va_start(list, fmt);
        log_manager.Record(logLevel, fmt, list);
        va_end(list);
    }
    void CheckLogContent(const std::string& expectedStr, const char* fmt, ...)
    {
        va_list list;
        va_start(list, fmt);
        LogMsg logMsg{};
        LogManager::Instance().ConstructMessage(LogLevel::INFO, fmt, list, logMsg);
        va_end(list);
        std::string retStr(logMsg.msg);
        bool ret = retStr.find(expectedStr) == retStr.size() - expectedStr.size() - 1;
        if (!ret) {
            std::cout << retStr << "|" << expectedStr << std::endl;
        }
        EXPECT_TRUE(ret);

        std::string expectStr = expectedStr;
        size_t pos = expectStr.find("+");
        while (pos != std::string::npos) {
            expectStr.replace(pos, 1, "\\+");
            pos = expectStr.find("+", pos + 2);
        }

        std::string regexStr = "\\[INFO \\] PYPTO\\(\\d+\\):\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}.\\d{3} " +
                               expectStr + "\n";
        std::regex logMsgRegex(regexStr);
        ret = std::regex_match(retStr, logMsgRegex);
        if (!ret) {
            std::cout << retStr << "|" << regexStr << std::endl;
        }
        EXPECT_TRUE(ret);
    }
};

TEST_F(TestHostLog, test_tilefwk_log_case0)
{
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE, "I'm a space-bound %s and your heart's the moon", "rocketship");
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE, "And I aiming it right at you, right at you %f", 3.14f);
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE, "%d miles on a clear night in %s", 250000, "June");
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE, "And I'm so lost without you, without you %x", (uint32_t)626);
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "I'm a space-bound %s and your heart's the moon", "rocketship");
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "And I aiming it right at you, right at you %f", 3.14f);
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "%d miles on a clear night in %s", 250000, "June");
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "And I'm so lost without you, without you");

    std::ostringstream oss;
    for (size_t i = 0; i < 200; i++) {
        oss << "0123456789";
    }
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "Hello %s", oss.str().c_str());
}

namespace {
void FunctionWithNoReturn()
{
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE,
                   "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE,
                         "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(
        DLOG_INFO, MACHINE, "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    std::ostringstream oss;
    for (size_t i = 0; i < 200; i++) {
        oss << "0123456789";
    }
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "Hello %s", oss.str().c_str());
}
int FunctionWithReturn()
{
    PYPTO_HOST_LOG(DLOG_ERROR, MACHINE,
                   "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE,
                         "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(
        DLOG_INFO, MACHINE, "In the year of %d assembled here the volunteers in the days when lands were few", 39);
    std::ostringstream oss;
    for (size_t i = 0; i < 200; i++) {
        oss << "0123456789";
    }
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, MACHINE, "Hello %s", oss.str().c_str());
    return 0;
}
} // namespace
TEST_F(TestHostLog, test_tilefwk_log_case1)
{
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    FunctionWithNoReturn();
    FunctionWithReturn();
}

TEST_F(TestHostLog, test_log_manager_case0)
{
    LogManager::Instance().SetLogLevel(LogLevel::ERROR);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::EVENT), false);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
    RecoreLog(log_manager, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
    RecoreLog(log_manager, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
    RecoreLog(log_manager, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
    RecoreLog(log_manager, LogLevel::INFO, "And I'm so lost without you, without you %x", (uint32_t)626);
}

TEST_F(TestHostLog, test_log_manager_case1)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case2)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_MODULE_LOG_LEVEL", "PYPTO=2", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case3)
{
    setenv("ASCEND_GLOBAL_EVENT_ENABLE", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), true);
}

TEST_F(TestHostLog, test_log_manager_case4)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "abc", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case5)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_PROCESS_LOG_PATH", "./temp_pypto_log", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.enableStdOut_, false);
    EXPECT_EQ(log_manager.maxLogFileNum_, 10);
    for (size_t i = 0; i < 200000; ++i) {
        if (i % 2 == 0) {
            log_manager.EnableHostLog();
        } else {
            log_manager.EnableDeviceLog();
        }
        RecoreLog(log_manager, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
        RecoreLog(log_manager, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
        RecoreLog(log_manager, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
        RecoreLog(log_manager, LogLevel::INFO, "And I'm so lost without you, without you %x", (uint32_t)626);
    }
    std::string hostLogFilePrefix = "pypto-log-" + std::to_string(GetTestThreadId());
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager.hostLogDir_, hostLogFilePrefix), 2);
    std::string devLogFilePrefix = "pypto-simulation-" + std::to_string(GetTestThreadId());
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager.deviceLogDir_, devLogFilePrefix), 2);
}

TEST_F(TestHostLog, test_log_manager_case6)
{
    setenv("ASCEND_PROCESS_LOG_PATH", "./temp_process_log", 1);
    setenv("ASCEND_HOST_LOG_FILE_NUM", "15", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.maxLogFileNum_, 15);
    EXPECT_NE(log_manager.hostLogDir_.find("/temp_process_log/debug/plog"), std::string::npos);
    EXPECT_NE(log_manager.deviceLogDir_.find("/temp_process_log/debug/device-0"), std::string::npos);
}

TEST_F(TestHostLog, test_log_manager_case7)
{
    setenv("ASCEND_WORK_PATH", "./temp_work_log", 1);
    setenv("ASCEND_HOST_LOG_FILE_NUM", "0", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.maxLogFileNum_, 10);
    EXPECT_NE(log_manager.hostLogDir_.find("/temp_work_log/log/debug/plog"), std::string::npos);
    EXPECT_NE(log_manager.deviceLogDir_.find("/temp_work_log/log/debug/device-0"), std::string::npos);
}

TEST_F(TestHostLog, test_log_manager_case8)
{
    setenv("ASCEND_PROCESS_LOG_PATH", "./temp_process_log", 1);
    setenv("ASCEND_WORK_PATH", "./temp_work_log", 1);
    setenv("ASCEND_HOST_LOG_FILE_NUM", "-1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.maxLogFileNum_, 10);
    EXPECT_NE(log_manager.hostLogDir_.find("/temp_process_log/debug/plog"), std::string::npos);
    EXPECT_NE(log_manager.deviceLogDir_.find("/temp_process_log/debug/device-0"), std::string::npos);
    EXPECT_EQ(log_manager.hostLogDir_.find("/temp_work_log/log/debug/plog"), std::string::npos);
    EXPECT_EQ(log_manager.deviceLogDir_.find("/temp_work_log/log/debug/device-0"), std::string::npos);
}

TEST_F(TestHostLog, test_log_manager_case9)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_PROCESS_LOG_PATH", "./temp_pypto_host_log", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.enableStdOut_, false);
    EXPECT_EQ(log_manager.maxLogFileNum_, 10);
    for (size_t i = 0; i < 200000; ++i) {
        RecoreLog(log_manager, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
        RecoreLog(log_manager, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
        RecoreLog(log_manager, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
        RecoreLog(log_manager, LogLevel::INFO, "And I'm so lost without you, without you %x", (uint32_t)626);
    }
    std::string hostLogFilePrefix = "pypto-log-" + std::to_string(GetTestThreadId());
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager.hostLogDir_, hostLogFilePrefix), 4);

    setenv("ASCEND_HOST_LOG_FILE_NUM", "2", 1);
    LogManager log_manager2;
    EXPECT_EQ(log_manager2.enableStdOut_, false);
    EXPECT_EQ(log_manager2.maxLogFileNum_, 2);
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager2.hostLogDir_, "pypto-log-"), 2);
    for (size_t i = 0; i < 200000; ++i) {
        RecoreLog(log_manager2, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
        RecoreLog(log_manager2, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
        RecoreLog(log_manager2, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
        RecoreLog(log_manager2, LogLevel::INFO, "And I'm so lost without you, without you %x", (uint32_t)626);
    }
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager2.hostLogDir_, "pypto-log-"), 2);
}

TEST_F(TestHostLog, test_log_construct_case0)
{
    int32_t int32_val = -234;
    int32_t int32_val2 = 567;
    uint32_t uint32_val = 432;
    CheckLogContent("-234,-234,+567,432", "%d,%+d,%+d,%u", int32_val, int32_val, int32_val2, uint32_val);
    CheckLogContent("37777777426,0660,ffffff16,1B0,0xffffff16,0X1B0", "%o,%#o,%x,%X,%#x,%#X", int32_val, uint32_val,
                    int32_val, uint32_val, int32_val, uint32_val);

    std::ostringstream oss1;
    oss1 << std::hex << &int32_val << "," << &uint32_val;
    CheckLogContent(oss1.str(), "%p,%p", &int32_val, &uint32_val);

    int64_t int64_val = -789;
    uint64_t uint64_val = 987;
    CheckLogContent("-789,987,fffffffffffffceb,3DB,0xfffffffffffffceb,0X3DB", "%ld,%lu,%lx,%lX,%#lx,%#lX", int64_val,
                    uint64_val, int64_val, uint64_val, int64_val, uint64_val);

    float float_val = 123.456f;
    CheckLogContent("123.456001,123.46, 123.45600", "%f,%.2f,%10.5f", float_val, float_val, float_val);
    CheckLogContent("1.234560e+02,1.235e+02,1.234560E+02", "%e,%.3e,%E", float_val, float_val, float_val);

    double double_val = -456.987321;
    CheckLogContent("-456.987321,-456.99,-456.98732", "%f,%.2f,%10.5f", double_val, double_val, double_val);
    CheckLogContent("-4.569873e+02,-4.570e+02,-4.569873E+02", "%e,%.3e,%E", double_val, double_val, double_val);

    CheckLogContent("Hello", "%.5s", "Hello world");
    CheckLogContent("     Hello", "%10s", "Hello");
    CheckLogContent("Hello     ", "%-10s", "Hello");
    CheckLogContent("Hello world", "%c%s%c", 'H', "ello worl", 100);
}

TEST_F(TestHostLog, test_log_construct_case1)
{
    CheckLogContent("Hello world!", "Hello world!", 123, "morgan");
    CheckLogContent("4294967173,4294966840", "%u,%lu", -123, -456);
    CheckLogContent("3.140000,4294967173", "%f,%lu", -123, 3.14f);

    std::ostringstream oss;
    for (size_t i = 0; i < 100; i++) {
        oss << "0123456789";
    }
    LogManager log_manager;
    RecoreLog(log_manager, LogLevel::INFO, "%s", oss.str().c_str());
}

TEST_F(TestHostLog, test_dlog_handler_case0)
{
    DLogHandler log_handler;
    EXPECT_NE(log_handler.checkLevelFunc_, nullptr);
    EXPECT_NE(log_handler.logRecordFunc_, nullptr);
    EXPECT_NE(log_handler.getLevelFunc_, nullptr);
    EXPECT_NE(log_handler.setLevelFunc_, nullptr);

    EXPECT_EQ(log_handler.IsAvailable(), true);
    EXPECT_EQ(log_handler.CheckLogLevel(PYPTO, DLOG_ERROR), 1);
    EXPECT_EQ(log_handler.CheckLogLevel(PYPTO, DLOG_WARN), 0);
    int32_t enableEvent = 0;
    EXPECT_EQ(log_handler.GetLogLevel(PYPTO, &enableEvent), DLOG_ERROR);
    EXPECT_EQ(enableEvent, 1);
    EXPECT_EQ(log_handler.SetLogLevel(PYPTO, DLOG_WARN, enableEvent), 0);
    EXPECT_EQ(log_handler.CheckLogLevel(PYPTO, DLOG_WARN), 1);

    EXPECT_EQ(DLogHandler::Instance().IsAvailable(), true);
    EXPECT_NE(DLogHandler::Instance().checkLevelFunc_, nullptr);
    EXPECT_NE(DLogHandler::Instance().logRecordFunc_, nullptr);
    EXPECT_NE(DLogHandler::Instance().getLevelFunc_, nullptr);
    EXPECT_NE(DLogHandler::Instance().setLevelFunc_, nullptr);
}

TEST_F(TestHostLog, test_log_module_manager_case0)
{
    LogModuleManager log_module_manager;
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::FUNCTION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::PASS), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::CODEGEN), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::MACHINE), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::DISTRIBUTED), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::SIMULATION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::VERIFY), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::COMPILER_MONITOR), -1);
}

TEST_F(TestHostLog, test_log_module_manager_case1)
{
    setenv("ASCEND_MODULE_LOG_LEVEL", "MACHINE=2", 1);
    LogModuleManager log_module_manager;
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::FUNCTION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::PASS), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::CODEGEN), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::MACHINE), 2);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::DISTRIBUTED), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::SIMULATION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::VERIFY), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::COMPILER_MONITOR), -1);
}

TEST_F(TestHostLog, test_log_module_manager_case2)
{
    setenv("ASCEND_MODULE_LOG_LEVEL", " MACHINE = 0 : PASS =1", 1);
    LogModuleManager log_module_manager;
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::FUNCTION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::PASS), 1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::CODEGEN), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::MACHINE), 0);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::DISTRIBUTED), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::SIMULATION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::VERIFY), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::COMPILER_MONITOR), -1);
}

TEST_F(TestHostLog, test_log_module_manager_case3)
{
    setenv("ASCEND_MODULE_LOG_LEVEL", " MACHINE =  : CODEGEN =3", 1);
    LogModuleManager log_module_manager;
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::FUNCTION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::PASS), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::CODEGEN), 3);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::MACHINE), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::DISTRIBUTED), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::SIMULATION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::VERIFY), -1);
}

TEST_F(TestHostLog, test_log_module_manager_case4)
{
    setenv("ASCEND_MODULE_LOG_LEVEL", " MACHINE: CODEGEN =3", 1);
    LogModuleManager log_module_manager;
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::FUNCTION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::PASS), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::CODEGEN), 3);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::MACHINE), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::DISTRIBUTED), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::SIMULATION), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::VERIFY), -1);
    EXPECT_EQ(log_module_manager.GetModuleLogLevel(LogModule::COMPILER_MONITOR), -1);
}
} // namespace npu::tile_fwk
