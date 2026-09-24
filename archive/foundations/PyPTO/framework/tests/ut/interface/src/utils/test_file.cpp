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
 * \file test_file.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>

#include "utils/file_utils.h"

using namespace npu::tile_fwk;

#define PATH_MAX 4096

class FileTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST(FileTest, NullptrTest)
{
    uint8_t a = 255;
    std::vector<uint8_t> data{a};
    SaveFile("", data);

    bool ret = SaveFile("", &a, 1);
    EXPECT_EQ(ret, false);

    EXPECT_FALSE(CopyFile("", ""));
}

TEST(FileTest, LoadFileTest)
{
    uint8_t a = 255;
    std::vector<uint8_t> data{a};

    data = ReadFile("");
    EXPECT_EQ(data.size(), 0);
}

TEST(FileTest, ReadFileReturnsFullContents)
{
    // 回归：ReadFile 必须以 std::ios::ate 打开，否则 tellg() 返回 0 导致读回空缓冲。
    const std::string path = "/tmp/test_read_file_" + std::to_string(::getpid());
    const std::vector<uint8_t> payload{0xde, 0xad, 0xbe, 0xef};
    ASSERT_TRUE(SaveFile(path, payload));

    const auto data = ReadFile(path);
    EXPECT_EQ(data.size(), payload.size());
    EXPECT_EQ(data, payload);

    DeleteFile(path);
}

TEST(FileTest, RemoveOldDirectoriesKeepsNewestAndOnlyMatchingPrefix)
{
    // 回归：筛选条件曾被写反，导致既不按前缀过滤、又误把 "."/".." 放入候选。
    const std::string base = "/tmp/test_remove_old_dirs_" + std::to_string(::getpid());
    const std::string prefix = "run_";
    DeleteDir(base, true);
    ASSERT_TRUE(CreateDir(base, true));

    // 前缀匹配的目录（应受 kept 数量约束，最旧的被删）。
    for (int i = 0; i < 3; ++i) {
        const std::string d = base + "/" + prefix + std::to_string(i);
        ASSERT_TRUE(CreateDir(d, true));
        // 写入一个文件，验证删除走的是递归路径。
        ASSERT_TRUE(SaveFile(d + "/f.bin", std::vector<uint8_t>{0x01}));
    }
    // 不匹配前缀的目录（必须被忽略，不被删除）。
    const std::string other = base + "/unrelated";
    ASSERT_TRUE(CreateDir(other, true));

    // 保留最新的 1 个：按 mtime 排序后删除其余 prefix 目录。
    RemoveOldDirectories(base, prefix, 1);

    EXPECT_TRUE(IsPathExist(other)); // 非匹配前缀必须保留
    int surviving = 0;
    for (int i = 0; i < 3; ++i) {
        if (IsPathExist(base + "/" + prefix + std::to_string(i))) {
            ++surviving;
        }
    }
    EXPECT_EQ(surviving, 1); // 只保留 1 个 prefix 目录

    DeleteDir(base, true);
}

TEST(FileTest, RealPathTest)
{
    EXPECT_TRUE(RealPath("").empty());
    EXPECT_TRUE(RealPath("/nonexitent/file").empty());
    EXPECT_FALSE(RealPath("/tmp").empty());
    EXPECT_TRUE(RealPath(std::string(PATH_MAX, 'a')).empty());

    EXPECT_FALSE(CreateDir(std::string(PATH_MAX, 'a')));
}

TEST(FileTest, CreateDirFailedWhenParentNotExist)
{
    // 父目录不存在 -> mkdir 返回 -1，errno=ENOENT -> CreateDir 返回 false
    std::string path = "/tmp/no_such_parent_dir" + std::to_string(::getpid()) + "/child";
    EXPECT_FALSE(CreateDir(path));
}

TEST(FileTest, CreateDirHandlesConsecutiveSlashes)
{
    // "///" 等连续斜杠应被当作单个分隔符处理。
    const std::string base = "/tmp/create_dir_slashes_" + std::to_string(::getpid());
    const std::string path = base + "///a///b//c";
    DeleteDir(base);
    ASSERT_TRUE(CreateDir(path, true));
    EXPECT_TRUE(IsPathExist(path));
    // 重复创建应幂等成功。
    EXPECT_TRUE(CreateDir(path, true));
    // 结尾斜杠也应支持。
    EXPECT_TRUE(CreateDir(path + "/", true));
    DeleteDir(base, true);

    EXPECT_TRUE(CreateDir("/tmp", true)); // 已存在
}

TEST(FileTest, CreateMultiLevelDirFailedWhenParentIsFile)
{
    const std::string parent = "/tmp/parent_is_file_" + std::to_string(::getpid());
    const std::string target = parent + "/child";
    DeleteDir(target);
    DeleteFile(parent);
    SaveFile(parent, std::vector<uint8_t>{0x01});
    EXPECT_FALSE(CreateDir(target, true));

    // 清理
    DeleteDir(target, true);
    DeleteFile(parent);
}
