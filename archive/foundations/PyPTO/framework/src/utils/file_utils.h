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
 * \file file_utils.h
 * \brief Path / file / directory helpers (host side).
 */
#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <dlfcn.h>

namespace npu::tile_fwk {
// --- Path utilities ---
// Declared inline because leaf translation units that do not link the
// file_utils object (e.g. tile_fwk_cann_host_runtime) use them.

inline std::string RealPath(const std::string& path)
{
    char fullpath[PATH_MAX];
    if (realpath(path.c_str(), fullpath) == nullptr) {
        return {};
    }
    return std::string(fullpath);
}

inline bool IsPathExist(const std::string& path)
{
    struct stat buf;
    return (::stat(path.c_str(), &buf) == 0);
}

inline uint64_t GetFileSize(const std::string& path, bool* isSuccess = nullptr)
{
    struct stat buf;
    bool succ = ::stat(path.c_str(), &buf) == 0;
    if (isSuccess) {
        *isSuccess = succ;
    }
    return succ ? buf.st_size : 0;
}

inline void DeleteFile(const std::string& path) { remove(path.c_str()); }

std::string GetPyptoLibPath();
std::string GetCwd();

bool CreateDir(const std::string& path, bool recursive = false);
void DeleteDir(const std::string& path, bool recursive = false);
void Rename(const std::string& src, const std::string& dst);
bool SaveFile(const std::string& path, const uint8_t* data, size_t size);
void SaveFileSafe(const std::string& path, const uint8_t* data, size_t size);
std::vector<uint8_t> ReadFile(const std::string& path, bool* isSuccess = nullptr);
bool CopyFile(const std::string& src, const std::string& dst);

void RemoveOldDirectories(const std::string& path, const std::string& prefix, size_t kept);
std::vector<std::string> GetFiles(const std::string& path, const std::string& ext = "");

inline bool SaveFile(const std::string& path, const std::string& content)
{
    return SaveFile(path, (const uint8_t*)content.c_str(), content.size());
}
inline bool SaveFile(const std::string& path, const std::vector<uint8_t>& content)
{
    return SaveFile(path, (const uint8_t*)content.data(), content.size());
}
inline void SaveFileSafe(const std::string& path, const std::string& content)
{
    SaveFileSafe(path, (const uint8_t*)content.c_str(), content.size());
}
inline void SaveFileSafe(const std::string& path, const std::vector<uint8_t>& content)
{
    SaveFileSafe(path, (const uint8_t*)content.data(), content.size());
}
} // namespace npu::tile_fwk
