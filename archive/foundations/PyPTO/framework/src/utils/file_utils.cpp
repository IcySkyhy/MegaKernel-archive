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
 * \file file_utils.cpp
 * \brief
 */

#include "file_utils.h"

#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <climits>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <dlfcn.h>
#include <ftw.h>
#include <iostream>

#include "securec.h"

#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
namespace {
// 0755 for created directories, 0640 for lock files.
constexpr mode_t DIR_MODE = 0755;
constexpr mode_t FILE_MODE = 0640;

int RemoveFile(const char* path, const struct stat* sb, int flag, struct FTW* ftwbuf)
{
    (void)sb;
    (void)ftwbuf;
    if (flag == FTW_F) {
        return remove(path);
    } else if (flag == FTW_DP) {
        return rmdir(path);
    }
    return 0;
}
} // namespace

// do not use PYPTO_LOGX in it's used by log_manager
bool CreateDir(const std::string& path, bool recursive)
{
    char fullpath[PATH_MAX];
    if (strcpy_s(fullpath, PATH_MAX, path.c_str()) != 0) {
        std::cerr << "Create dir[" << path << "] failed, path is too long" << std::endl;
        return false;
    }
    if (recursive) {
        // Walk one '/' at a time, creating every non-empty prefix. Empty prefixes
        // come from leading '/' (e.g. "/a") or consecutive slashes (e.g. "a///b")
        // and are skipped, so "///" patterns are treated as a single separator.
        for (char* p = fullpath + 1; *p != '\0'; ++p) {
            if (*p != '/') {
                continue;
            }
            *p = '\0';
            if (fullpath[0] != '\0' && mkdir(fullpath, DIR_MODE) != 0 && errno != EEXIST) {
                std::cerr << "Create dir[" << fullpath << "] failed, reason is " << errno << std::endl;
                return false;
            }
            *p = '/';
        }
    }
    if (mkdir(fullpath, DIR_MODE) != 0 && errno != EEXIST) {
        std::cerr << "Create dir[" << fullpath << "] failed, reason is " << errno << std::endl;
        return false;
    }
    return true;
}

void DeleteDir(const std::string& path, bool recursive)
{
    if (recursive) {
        if (nftw(path.c_str(), RemoveFile, 0x10, FTW_DEPTH | FTW_PHYS)) {
            PYPTO_LOGW("Delete dir[%s] failed, reason is %d", path.c_str(), errno);
        }
    } else {
        if (rmdir(path.c_str()) != 0) {
            PYPTO_LOGW("Delete dir[%s] failed, reason is %d", path.c_str(), errno);
        }
    }
}

bool SaveFile(const std::string& path, const uint8_t* data, size_t size)
{
    std::ofstream of(path, std::ios::binary);
    if (!of.is_open()) {
        return false;
    }
    of.write((const char*)data, size);
    return of.good();
}

void SaveFileSafe(const std::string& path, const uint8_t* data, size_t size)
{
    std::string tmpfile = std::string(path) + ".tmp";
    if (SaveFile(tmpfile, data, size)) {
        Rename(tmpfile, path);
    }
}

void Rename(const std::string& src, const std::string& dst)
{
    if (rename(src.c_str(), dst.c_str()) != 0) {
        PYPTO_LOGW("Rename file %s to %s failed, reason is %d", src.c_str(), dst.c_str(), errno);
    }
}

std::vector<uint8_t> ReadFile(const std::string& path, bool* isSuccess)
{
    if (isSuccess != nullptr) {
        *isSuccess = false;
    }
    std::vector<uint8_t> binary;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        PYPTO_LOGW("Open file %s failed.", path.c_str());
        return binary;
    }

    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size < 0) {
        PYPTO_LOGW("Read file %s failed.", path.c_str());
        return binary;
    }
    binary.resize(static_cast<size_t>(size));
    ifs.read(reinterpret_cast<char*>(binary.data()), size);
    if (!ifs.good()) {
        PYPTO_LOGW("Read file %s failed.", path.c_str());
        return binary;
    }

    if (isSuccess != nullptr) {
        *isSuccess = true;
    }
    return binary;
}

void Rename(const char* oldPath, const char* newPath)
{
    if (rename(oldPath, newPath) != 0) {
        PYPTO_LOGW("Rename file %s to %s failed, reason is %d", oldPath, newPath, errno);
    }
}

bool CopyFile(const std::string& src, const std::string& dst)
{
    std::ifstream ifs(src, std::ios::binary);
    std::ofstream ofs(dst, std::ios::binary | std::ios::trunc);
    if (!ifs.is_open() || !ofs.is_open()) {
        return false;
    }

    constexpr size_t kBufferSize = 16 * 1024;
    char buffer[kBufferSize];
    while (ofs.good()) {
        ifs.read(buffer, kBufferSize);
        std::streamsize n = ifs.gcount();
        if (n <= 0) {
            break;
        }
        ofs.write(buffer, n);
    }
    return ofs.good() && !ifs.bad();
}

std::string GetCwd()
{
    char path[PATH_MAX];
    auto cwd = getcwd(path, PATH_MAX);
    if (cwd == nullptr) {
        PYPTO_LOGW("Failed to call getcwd()");
        return ".";
    }
    return cwd;
}

void RemoveOldDirectories(const std::string& path, const std::string& prefix, size_t kept)
{
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        PYPTO_LOGW("Failed to opendir: %s", path.c_str());
        return;
    }

    struct dirent* entry;
    struct stat statBuf;
    std::vector<std::pair<std::string, int64_t>> dirList;
    while ((entry = readdir(dir)) != nullptr) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
            continue;
        }
        if (strncmp(entry->d_name, prefix.c_str(), prefix.size()) != 0) {
            continue;
        }
        std::string dirname = path + "/" + entry->d_name;
        if (stat(dirname.c_str(), &statBuf) == 0 && S_ISDIR(statBuf.st_mode)) {
            dirList.emplace_back(dirname, statBuf.st_mtime);
        }
    }

    closedir(dir);

    std::sort(dirList.begin(), dirList.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    for (size_t i = kept; i < dirList.size(); ++i) {
        DeleteDir(dirList[i].first, true);
    }
}

// do not use PYPTO_LOGX in it's used by log_manager
std::vector<std::string> GetFiles(const std::string& path, const std::string& ext)
{
    std::vector<std::string> files;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return files;
    }

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_type == DT_DIR) {
            continue;
        }
        if (!ext.empty()) {
            auto pos = strrchr(ent->d_name, '.');
            if (pos == nullptr || strcmp(pos + 1, ext.c_str())) {
                continue;
            }
        }
        files.push_back(ent->d_name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

std::string GetPyptoLibPath()
{
    static std::string path;
    static std::once_flag once;
    std::call_once(once, [&] {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(GetPyptoLibPath), &info) || info.dli_fname == nullptr) {
            return; // leave empty
        }
        if (const char* pos = strrchr(info.dli_fname, '/'); pos != nullptr) {
            path.assign(info.dli_fname, pos - info.dli_fname);
        } else {
            path = info.dli_fname;
        }
    });
    return path;
}

} // namespace npu::tile_fwk
