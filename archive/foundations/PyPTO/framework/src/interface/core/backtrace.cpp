/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/error.h"
#include "interface/utils/common.h"

#include <cxxabi.h>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace pypto {
namespace ir {

/// Patterns to filter out from backtraces (internal/infrastructure frames)
const std::vector<std::string> kFileNameFilter = {
    "nanobind",     // Python binding layer
    "__libc_",      // C library internals
    "include/c++/", // C++ standard library
    "object.h",     // Python object.h
    "error.h"       // exception throwing infrastructure
};

std::string StackFrame::ToString() const
{
    std::ostringstream oss;

    if (!function.empty()) {
        oss << function;
    } else {
        oss << "0x" << std::hex << pc;
    }

    if (!filename.empty()) {
        oss << " at " << filename;
        if (lineno > 0) {
            oss << ":" << std::dec << lineno;
        }
    }

    return oss.str();
}

// Helper function to read a specific line from a source file
static std::string ReadSourceLine(const std::string& filename, int lineno)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    int currentLine = 0;
    while (std::getline(file, line)) {
        currentLine++;
        if (currentLine == lineno) {
            // Trim leading whitespace for display
            size_t start = line.find_first_not_of(" \t");
            if (start != std::string::npos) {
                return line.substr(start);
            }
            return line;
        }
    }
    return "";
}

// Structure to hold file location information
struct FileLocation {
    std::string filename;
    int lineno;
};

// Cache for symbol resolution to avoid repeated addr2line calls
static std::mutex locMapMutex;
static std::unordered_map<void*, FileLocation> locMap;

static bool IsUnknownLocation(const std::string& location)
{
    return location == "??" || location.find("??:") == 0 || location.find("?? ??") == 0;
}

static int ParseLineNumber(const std::string& lineStr)
{
    try {
        return std::stoi(lineStr);
    } catch (...) {
        return 0;
    }
}

static FileLocation ParseLocationWithAt(const std::string& location)
{
    FileLocation loc{"", 0};
    size_t colonPos = location.rfind(':');
    if (colonPos == std::string::npos) {
        return loc;
    }

    size_t prevColonPos = location.rfind(':', colonPos - 1);
    if (prevColonPos != std::string::npos) {
        loc.filename = location.substr(0, prevColonPos);
        loc.lineno = ParseLineNumber(location.substr(prevColonPos + 1, colonPos - prevColonPos - 1));
    } else {
        loc.filename = location.substr(0, colonPos);
        loc.lineno = ParseLineNumber(location.substr(colonPos + 1));
    }

    if (IsUnknownLocation(loc.filename) || loc.filename.empty()) {
        return FileLocation{"", 0};
    }

    return loc;
}

static FileLocation ParseAlternateFormat(const std::string& output)
{
    FileLocation loc{"", 0};
    size_t colonPos = output.rfind(':');
    if (colonPos == std::string::npos) {
        return loc;
    }

    loc.filename = output.substr(0, colonPos);
    if (IsUnknownLocation(loc.filename)) {
        return FileLocation{"", 0};
    }

    loc.lineno = ParseLineNumber(output.substr(colonPos + 1));
    return loc;
}

static FileLocation ParseAddr2lineOutput(const std::string& output)
{
    if (output.empty()) {
        return FileLocation{"", 0};
    }

    std::string trimmed = output;
    if (trimmed.back() == '\n') {
        trimmed.pop_back();
    }

    size_t atPos = trimmed.find(" at ");
    if (atPos != std::string::npos) {
        std::string location = trimmed.substr(atPos + 4);
        if (IsUnknownLocation(location)) {
            return FileLocation{"", 0};
        }
        return ParseLocationWithAt(location);
    }

    if (trimmed.find(":") != std::string::npos) {
        if (IsUnknownLocation(trimmed)) {
            return FileLocation{"", 0};
        }
        return ParseAlternateFormat(trimmed);
    }

    return FileLocation{"", 0};
}

static std::string ExecuteAddr2line(void* addr)
{
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_fname == nullptr) {
        return "";
    }

    std::stringstream addrStr;
    addrStr << addr;
    std::vector<std::string> args = {"addr2line", "-e", info.dli_fname, "-f", "-C", "-p", addrStr.str()};
    return npu::tile_fwk::SafeExecCommandWithOutput(args);
}

static FileLocation GetFileLineFromAddr2line(void* addr)
{
    {
        std::lock_guard<std::mutex> lock(locMapMutex);
        auto it = locMap.find(addr);
        if (it != locMap.end()) {
            return it->second;
        }
    }

    std::string output = ExecuteAddr2line(addr);
    FileLocation loc = ParseAddr2lineOutput(output);

    {
        std::lock_guard<std::mutex> lock(locMapMutex);
        locMap[addr] = loc;
    }

    return loc;
}

/// Clean up file paths from debug info that may contain temp build directory prefixes.
/// When building via pip in a temp directory, paths may look like:
///   /private/var/folders/.../build/./python/nanobind/modules/logging.cpp
/// This function extracts just the relative path portion.
static std::string CleanupFilePath(const std::string& path)
{
    if (path.empty()) {
        return path;
    }

    // Look for "/./", which indicates where the relative path begins
    // (this is created by -fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.)
    // Replace the prefix up to "/./" with "./"
    size_t markerPos = path.find("/./");
    if (markerPos != std::string::npos) {
        return "./" + path.substr(markerPos + 0x3);
    }

    // If path already starts with "./", keep it as-is
    if (path.size() >= 0x2 && path[0] == '.' && path[1] == '/') {
        return path;
    }

    return path;
}

Backtrace& Backtrace::GetInstance()
{
    static Backtrace instance;
    return instance;
}

Backtrace::Backtrace()
{
    // No initialization needed for execinfo-based implementation
}

std::vector<StackFrame> Backtrace::CaptureStackTrace(int skip)
{
    std::vector<StackFrame> frames;

    // Capture raw stack frames using execinfo
    constexpr int kMaxFrames = 128;
    void* callstack[kMaxFrames];
    int nrFrames = ::backtrace(callstack, kMaxFrames);

    // Skip requested frames plus this function itself
    int startFrame = skip + 1;
    if (startFrame >= nrFrames) {
        return frames;
    }

    // Get symbol information
    char** symbols = backtrace_symbols(callstack, nrFrames);
    if (symbols == nullptr) {
        return frames;
    }

    for (int i = startFrame; i < nrFrames; i++) {
        void* addr = callstack[i];

        // Parse the symbol string to get function name, library name, and offset
        // Format: "path/libname(function+offset) [address]"
        std::string symbol_str(symbols[i]);
        std::string funcNameStr;
        std::string libName;
        std::string funcOffsetStr;

        // Try to demangle the function name and extract library/offset
        char* funcName = strchr(symbols[i], '(');
        char* funcOffset = strchr(symbols[i], '+');
        char* closeParen = strchr(symbols[i], ')');

        if (funcName != nullptr && funcOffset != nullptr && closeParen != nullptr) {
            // Extract library name (everything before '(')
            *funcName = '\0';
            char* libnameStart = strrchr(symbols[i], '/');
            libName = (libnameStart != nullptr) ? (libnameStart + 1) : symbols[i];

            // Extract function name (between '(' and '+')
            funcName++;
            *funcOffset = '\0';

            // Extract offset (between '+' and ')')
            funcOffset++;
            *closeParen = '\0';
            funcOffsetStr = std::string("+") + funcOffset;

            // Demangle function name
            int status = 0;
            std::unique_ptr<char, std::function<void(char*)>> demangled(
                abi::__cxa_demangle(funcName, nullptr, nullptr, &status), free);
            if (status == 0 && demangled) {
                funcNameStr = demangled.get();
            } else {
                funcNameStr = funcName;
            }
        }

        // Try to get file and line information using addr2line
        FileLocation loc = GetFileLineFromAddr2line(addr);
        if (!loc.filename.empty()) {
            loc.filename = CleanupFilePath(loc.filename);
        }

        // Create frame with all information
        StackFrame frame(funcNameStr, loc.filename, loc.lineno, reinterpret_cast<uintptr_t>(addr));
        frame.libname = libName;
        frame.offset = funcOffsetStr;
        frames.push_back(frame);
    }

    free(symbols);
    return frames;
}

std::string Backtrace::FormatStackTrace(const std::vector<StackFrame>& frames)
{
    if (frames.empty()) {
        return "";
    }

    std::ostringstream oss;

    // Reverse the frames to show most recent last (like Python)
    std::vector<StackFrame> reversedFrames(frames.rbegin(), frames.rend());

    auto isFileNameFiltered = [](const std::string& filename) {
        return std::any_of(kFileNameFilter.begin(), kFileNameFilter.end(), [&filename](const std::string& filter) {
            return filename.find(filter) != std::string::npos;
        });
    };

    // Filter and deduplicate frames by PC address to handle Clang's debug info issues.
    // When Clang generates DWARF info for inlined functions/templates, it may
    // report multiple "virtual" frames for the same PC with incorrect source
    // locations. We keep only the first frame for each unique PC.
    std::vector<StackFrame> deduplicatedFrames;
    for (const auto& frame : reversedFrames) {
        // Filter out infrastructure frames before deduplication.
        // This prevents filtered frames from being used in duplicate PC checks.
        if (!frame.filename.empty() && isFileNameFiltered(frame.filename)) {
            continue;
        } else if (frame.pc != 0 && !deduplicatedFrames.empty() && deduplicatedFrames.back().pc == frame.pc) {
            // Same PC as the previous frame - this is likely a spurious inline frame.
            // Skip it to keep only the first frame for each unique PC.
            continue;
        } else {
            deduplicatedFrames.push_back(frame);
        }
    }

    for (const auto& frame : deduplicatedFrames) {
        // Format: File "filename", line X in function_name
        if (!frame.filename.empty() && frame.lineno > 0) {
            oss << " File \"" << frame.filename << "\", line " << frame.lineno << "\n";

            // Try to read and display the source line
            std::string sourceLine = ReadSourceLine(frame.filename, frame.lineno);
            if (!sourceLine.empty()) {
                oss << "   " << sourceLine << "\n";
            }
        } else if (!frame.function.empty() && frame.pc != 0) {
            // Fallback to traditional format if we don't have file/line info (Release mode)
            // Format: libname(function+offset) [0xaddress]
            if (!frame.libname.empty() && !frame.offset.empty()) {
                oss << " " << frame.libname << "(" << frame.function << frame.offset << ") [0x" << std::hex << frame.pc
                    << std::dec << "]\n";
            } else if (!frame.libname.empty()) {
                oss << " " << frame.libname << "(" << frame.function << ") [0x" << std::hex << frame.pc << std::dec
                    << "]\n";
            } else {
                oss << " " << frame.function << " [0x" << std::hex << frame.pc << std::dec << "]\n";
            }
        }
    }

    return oss.str();
}

} // namespace ir
} // namespace pypto
