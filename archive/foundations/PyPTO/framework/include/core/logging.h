/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logging.h
 * \brief Logging framework with support for console, file, and in-memory logging
 *
 * This header provides a flexible logging system with:
 * - Multiple log levels (DEBUG, INFO, WARN, ERROR, FATAL, EVENT)
 * - Colored terminal output using ANSI escape codes
 * - File-based logging with append/overwrite modes
 * - In-memory line-based logging for programmatic access
 * - Thread-safe logging operations
 * - Stream-style and printf-style logging interfaces
 */

#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/error.h"
#include "securec.h"

namespace pypto {
/**
 * \brief Enumeration of available log levels
 *
 * Log levels in ascending order of severity:
 * - DEBUG: Detailed information for debugging
 * - INFO: General informational messages
 * - WARN: Warning messages for potentially harmful situations
 * - ERROR: Error messages for failures
 * - FATAL: Critical errors that may cause termination
 * - EVENT: Special events and milestones
 * - NONE: Disable all logging
 */
enum class LogLevel : uint8_t {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4,
    EVENT = 5,
    NONE = 6,
};

/**
 * \brief Standard error logger that writes to stderr
 *
 * Supports colored output using TTY commands.
 */
class StdLogger {
public:
    /**
     * \brief Log a value of any type
     * \tparam T Type of value to log
     * \param t Value to log
     * \return Reference to this logger for chaining
     */
    template <typename T>
    StdLogger& Log(T&& t)
    {
        std::cerr << (std::forward<T>(t));
        return *this;
    }

    StdLogger() = default;
    StdLogger(const StdLogger&) = delete;
    StdLogger& operator=(const StdLogger&) = delete;
};

/**
 * \brief File-based logger that writes to a file stream
 *
 * Supports both append and overwrite modes. TTY commands are ignored
 * since file output typically doesn't support colored text.
 */
class FileLogger {
public:
    std::ofstream ofs;

    /**
     * \brief Construct a file logger
     * \param filepath Path to the log file
     * \param append If true, append to existing file; otherwise overwrite
     */
    FileLogger(const std::string& filepath, bool append)
    {
        if (append) {
            ofs.open(filepath, std::ios_base::app);
        } else {
            ofs.open(filepath);
        }
    }

    /**
     * \brief Log a value to the file
     * \tparam T Type of value to log
     * \param t Value to log
     * \return Reference to this logger for chaining
     */
    template <typename T>
    FileLogger& Log(T&& t)
    {
        ofs << (std::forward<T>(t));
        return *this;
    }
    FileLogger(const FileLogger&) = delete;
    FileLogger& operator=(const FileLogger&) = delete;
};

/**
 * \brief In-memory logger that stores log lines as strings
 *
 * Useful for programmatic access to log messages or testing.
 * Inherits from std::vector<std::string> for direct access to stored lines.
 */
class LineLogger : public std::vector<std::string> {
public:
    /**
     * \brief Log a string value
     * \param t String to store
     * \return Reference to this logger for chaining
     */
    LineLogger& Log(std::string&& t)
    {
        this->emplace_back(t);
        return *this;
    }
};

/**
 * \brief Central manager for all loggers
 *
 * This singleton class manages:
 * - Global log level threshold
 * - Standard output logger enable/disable
 * - Multiple file loggers
 * - Multiple line (in-memory) loggers
 *
 * All logging operations are thread-safe.
 */
class LoggerManager {
public:
    std::mutex logMtx;
#ifdef NDEBUG
    LogLevel level{LogLevel::ERROR};
#else
    LogLevel level{LogLevel::DEBUG};
#endif
    bool stdEnabled{true};
    StdLogger stdLogger;
    std::unordered_map<std::string, std::unique_ptr<FileLogger>> fileLoggerDict;
    std::unordered_map<std::string, std::shared_ptr<LineLogger>> lineLoggerDict;

    LoggerManager() = default;

    /**
     * \brief Log a message to all active loggers
     * \tparam T Type of the log message
     * \param l Log level
     * \param t Plain message (for file/line loggers)
     * \param tRich Rich message with formatting (for std logger)
     */
    template <typename T>
    void Log(LogLevel l, T&& t, T&& tRich)
    {
        std::scoped_lock lock(logMtx);
        if (l >= level) {
            if (stdEnabled) {
                stdLogger.Log(std::forward<T>(tRich));
            }
        }
        for (auto& [filepath, logger] : fileLoggerDict) {
            (void)filepath;
            logger->Log(std::forward<T>(t));
        }
        for (auto& [name, logger] : lineLoggerDict) {
            (void)name;
            logger->Log(std::forward<T>(t));
        }
    }

    /**
     * \brief Set the global log level threshold
     * \param l New log level
     */
    static void SetLevel(LogLevel l) { GetManager().level = l; }

    /**
     * \brief Get the current global log level threshold
     * \return Current log level
     */
    static LogLevel GetLevel() { return GetManager().level; }

    /**
     * \brief Enable or disable standard output logging
     * \param enabled True to enable, false to disable
     */
    static void StdLoggerEnable(bool enabled) { GetManager().stdEnabled = enabled; }

    /**
     * \brief Register a file logger
     * \param filepath Path to the log file
     * \param append If true, append to existing file; otherwise overwrite
     */
    static void FileLoggerRegister(const std::string& filepath, bool append)
    {
        GetManager().fileLoggerDict.try_emplace(filepath, std::make_unique<FileLogger>(filepath, append));
    }

    /**
     * \brief Unregister and close a file logger
     * \param filepath Path to the log file to unregister
     */
    static void FileLoggerUnregister(const std::string& filepath) { GetManager().fileLoggerDict.erase(filepath); }

    /**
     * \brief Replace one file logger with another
     * \param oldFilepath Path to the old log file
     * \param newFilepath Path to the new log file
     * \param append If true, append to new file; otherwise overwrite
     */
    static void FileLoggerReplace(const std::string& oldFilepath, const std::string& newFilepath, bool append)
    {
        FileLoggerUnregister(oldFilepath);
        FileLoggerRegister(newFilepath, append);
    }

    /**
     * \brief Register an in-memory line logger
     * \param name Name identifier for the logger
     * \return Shared pointer to the line logger for access to stored lines
     */
    static std::shared_ptr<LineLogger> LineLoggerRegister(const std::string& name)
    {
        auto& dict = GetManager().lineLoggerDict;
        auto it = dict.find(name);
        if (it != dict.end()) {
            return it->second;
        }
        auto logger = std::make_shared<LineLogger>();
        dict[name] = logger;
        return logger;
    }

    /**
     * \brief Unregister an in-memory line logger
     * \param name Name identifier of the logger to unregister
     */
    static void LineLoggerUnregister(const std::string& name) { GetManager().lineLoggerDict.erase(name); }

    friend class Logger;

    /**
     * \brief Get the singleton LoggerManager instance
     * \return Reference to the singleton LoggerManager
     */
    static LoggerManager& GetManager()
    {
        static LoggerManager manager;
        return manager;
    }
};

constexpr uint32_t MAX_LOG_BUF_SIZE = 1024;

/**
 * \brief Main logger class for creating log messages
 *
 * This class handles:
 * - Automatic timestamp generation
 * - Log level prefixing
 * - Message buffering
 * - Stream-style output via operator<<
 * - Variadic output via operator()
 *
 * Logger objects should be created per log message (they flush on destruction).
 */
class Logger {
private:
    std::stringstream ss;
    std::stringstream ssRich;
#ifdef NDEBUG
    LogLevel level{LogLevel::ERROR};
#else
    LogLevel level{LogLevel::DEBUG};
#endif
    bool enableLog = false;

public:
    /**
     * \brief Construct a logger for a single log message
     * \param levelIn Log level for this message
     * \param func Function name (currently unused but available for future use)
     * \param line Line number (currently unused but available for future use)
     */
    Logger(LogLevel levelIn, [[maybe_unused]] int line) : level(levelIn)
    {
        enableLog = LoggerManager::GetManager().level <= level;
        if (enableLog) {
            static const char* MSG = "DIWEFVN";
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
            (void)localtime_r(&time, &tm);

            // Format timestamp
            char timeBuf[128];
            std::strftime(timeBuf, sizeof(timeBuf), "%F %T.", &tm);
            Log(timeBuf);

            // Add milliseconds and log level
            char buf[MAX_LOG_BUF_SIZE];
            auto epoch = now.time_since_epoch();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count() % 1000;
            sprintf_s(buf, sizeof(buf), "%03d %c | ", static_cast<int>(ms), MSG[static_cast<int>(level)]);

            Log(buf);
        }
    }

    /**
     * \brief Destructor flushes the log message to all active loggers
     */
    ~Logger()
    {
        if (enableLog) {
            Log("\n");
            LoggerManager::GetManager().Log(level, ss.str(), ssRich.str());
        }
    }

    /**
     * \brief Log a value
     * \tparam T Type of value to log
     * \param val Value to log
     * \return Reference to this logger for chaining
     */
    template <typename T>
    Logger& Log(T&& val)
    {
        ss << (std::forward<T>(val));
        ssRich << (std::forward<T>(val));
        return *this;
    }

    /**
     * \brief Stream operator for convenient logging
     * \tparam T Type of value to log
     * \param val Value to log
     * \return Reference to this logger for chaining
     */
    template <typename T>
    Logger& operator<<(T&& val)
    {
        if (enableLog) {
            return Log(std::forward<T>(val));
        } else {
            return *this;
        }
    }

    /**
     * \brief Variadic operator for logging multiple values at once
     * \tparam Tys Types of values to log
     * \param vals Values to log
     * \return Reference to this logger for chaining
     */
    template <typename... Tys>
    Logger& operator()(Tys&&... vals)
    {
        if (enableLog) {
            if constexpr (sizeof...(Tys) > 0) {
                (Log(std::forward<Tys>(vals)), ...);
            }
        }
        return *this;
    }
};

// Convenience macros for logging at different levels
#define IR_LOGI() pypto::Logger(pypto::LogLevel::INFO, __LINE__)
#define IR_LOGW() pypto::Logger(pypto::LogLevel::WARN, __LINE__)
#define IR_LOGE() pypto::Logger(pypto::LogLevel::ERROR, __LINE__)
#define IR_LOGF() pypto::Logger(pypto::LogLevel::FATAL, __LINE__)
#define IR_LOGV() pypto::Logger(pypto::LogLevel::EVENT, __LINE__)

/**
 * \brief Helper function for formatted logging to avoid redefining safe functions in macros
 * \param fmt Format string
 * \param ... Variable arguments
 * \return Formatted string
 */
inline std::string FormatLogMessage(const char* fmt, ...)
{
    constexpr int defaultBufSize = 1024;
    std::string buf(defaultBufSize, '\0');

    va_list args;
    va_start(args, fmt);
    int msgLength = vsnprintf_s(buf.data(), buf.size(), buf.size() - 1, fmt, args);
    va_end(args);

    if (msgLength < 0) {
        return "[FormatLogMessage error]";
    }

    if (msgLength > defaultBufSize) {
        buf.resize(msgLength + 1, '\0');
        va_start(args, fmt);
        int ret = vsnprintf_s(buf.data(), buf.size(), buf.size() - 1, fmt, args);
        va_end(args);
        if (ret < 0) {
            return "[FormatLogMessage error]";
        }
    }

    return buf;
}

// Printf-style logging macros
#define LOG_F(lvl, fmt, args...)                                                \
    do {                                                                        \
        if (pypto::LoggerManager::GetManager().level <= pypto::LogLevel::lvl) { \
            LOG_##lvl(pypto::FormatLogMessage(fmt, ##args).c_str());            \
        }                                                                       \
    } while (false)

#define LOG_DEBUG_F(fmt, args...) LOG_F(DEBUG, fmt, ##args)
#define LOG_INFO_F(fmt, args...) LOG_F(INFO, fmt, ##args)
#define LOG_WARN_F(fmt, args...) LOG_F(WARN, fmt, ##args)
#define LOG_ERROR_F(fmt, args...) LOG_F(ERROR, fmt, ##args)
#define LOG_EVENT_F(fmt, args...) LOG_F(EVENT, fmt, ##args)

/**
 * \brief Helper class for IRCHECK, INTERNAL_CHECK, UNREACHABLE, and INTERNAL_UNREACHABLE macros
 *
 * This class collects error messages via operator<< and throws
 * an exception on destruction if the check condition failed.
 *
 * \tparam ExceptionType The type of exception to throw (ValueError or InternalError)
 */
template <typename ExceptionType>
class FatalLogger;

// Specialization for conditional checks (IRCHECK, INTERNAL_CHECK)
template <typename ExceptionType>
class FatalLogger {
private:
    std::stringstream ss;
    const char* file;
    int line;
    const char* exprStr;

public:
    FatalLogger(const char* exprStr_, const char* file_, int line_) : file(file_), line(line_), exprStr(exprStr_) {}

    [[noreturn]] ~FatalLogger() noexcept(false)
    {
        ss << "\n"
           << "Check failed: " << exprStr << " at " << file << ":" << line;
        throw ExceptionType(ss.str());
    }

    template <typename T>
    FatalLogger& operator<<(T&& val)
    {
        ss << (std::forward<T>(val));
        return *this;
    }

    std::stringstream& GetStream() { return ss; }

    FatalLogger(const FatalLogger&) = delete;
    FatalLogger& operator=(const FatalLogger&) = delete;
    FatalLogger(FatalLogger&&) = delete;
    FatalLogger& operator=(FatalLogger&&) = delete;
};

/**
 * \brief Check a condition and throw ValueError if it fails
 *
 * Usage: IRCHECK(condition) << "error message";
 */
#define IRCHECK(expr) \
    if (!(expr))      \
    pypto::FatalLogger<pypto::ir::ValueError>(#expr, __FILE__, __LINE__)

/**
 * \brief Check an internal invariant and throw InternalError if it fails
 *
 * Usage: INTERNAL_CHECK(condition) << "error message";
 */
#define INTERNAL_CHECK(expr) \
    if (!(expr))             \
    pypto::FatalLogger<pypto::ir::InternalError>(#expr, __FILE__, __LINE__)

/**
 * \brief Mark a code path as unreachable and throw ValueError if reached
 *
 * Usage: UNREACHABLE << "optional message";
 */
#define UNREACHABLE pypto::FatalLogger<pypto::ir::ValueError>("unreachable", __FILE__, __LINE__)

/**
 * \brief Mark a code path as internally unreachable and throw InternalError if reached
 *
 * Usage: INTERNAL_UNREACHABLE << "optional message";
 */
#define INTERNAL_UNREACHABLE pypto::FatalLogger<pypto::ir::InternalError>("unreachable", __FILE__, __LINE__)

/**
 * @brief Check an internal invariant with IR source location and throw InternalError if it fails
 *
 * Usage: INTERNAL_CHECK_SPAN(condition, node->span_) << "error message";
 */
#define INTERNAL_CHECK_SPAN(expr, span) \
    if (!!(expr))                       \
        ;                               \
    else                                \
        pypto::FatalLogger<pypto::ir::InternalError>(#expr, span.Filename().c_str(), span.BeginLine())
} // namespace pypto
