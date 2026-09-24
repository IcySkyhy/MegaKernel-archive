/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file io_text_dump_test.cpp
 * \brief Data-driven round-trip tests: load -> dump -> compare for IR text format.
 *
 * Each ``.pypto`` file in ``io_text_dump_test_data/`` is loaded via TextLoadFunction,
 * re-dumped via TextDump, and compared against the original file content.
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ir/transforms/io_text.h"
#include "interface/tensor/irbuilder.h"

namespace pypto {
namespace ir {

namespace fs = std::filesystem;

class IoTextDumpTest : public testing::Test {};

static const fs::path kDataDir = fs::path(IO_TEXT_TEST_DATA_DIR);

/// Read a file into a string, stripping trailing whitespace for comparison.
static std::string ReadFile(const fs::path& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
        content.pop_back();
    }
    return content;
}

/// Collect all .pypto files in the data directory.
static std::vector<fs::path> CollectDataFiles()
{
    std::vector<fs::path> files;
    if (fs::exists(kDataDir)) {
        for (const auto& entry : fs::directory_iterator(kDataDir)) {
            if (entry.path().extension() == ".pypto") {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

TEST_F(IoTextDumpTest, LoadDumpMatch)
{
    std::vector<fs::path> dataFilePathList = CollectDataFiles();
    ASSERT_FALSE(dataFilePathList.empty()) << "No .pypto data files found in: " << kDataDir;
    for (auto dataFilePath : dataFilePathList) {
        npu::tile_fwk::IRContext::Get().Reset();
        std::string original = ReadFile(dataFilePath);
        ASSERT_FALSE(original.empty()) << "Empty or missing file: " << dataFilePath;

        // Load -> Dump -> Compare
        std::string error;
        auto prog = TextLoadProgram(original, error);
        ASSERT_NE(prog, nullptr) << "Load failed for: " << dataFilePath.filename() << ": " << error;
        EXPECT_TRUE(error.empty()) << "Unexpected parse error in: " << dataFilePath.filename() << ": " << error;

        std::string dumped = TextDump(prog);
        EXPECT_EQ(original, dumped) << "Round-trip mismatch in: " << dataFilePath.filename();
    }
}

} // namespace ir
} // namespace pypto
