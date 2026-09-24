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
 * \file test_dump_json.cpp
 * \brief Unit test for DumpJson.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/serialization.h"
#include "utils/file_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace npu::tile_fwk;

class JsonOutputValidationTest : public testing::Test {
protected:
    static std::string jsonFilePath;

    static void SetUpTestCase()
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassConfig("PVC2_OOO", "CodegenPreproc", "print_graph", true);
        config::SetPassConfig("PVC2_OOO", "CodegenPreproc", "dump_graph", true);
        constexpr int32_t tilex = 8;
        constexpr int32_t tiley = 8;
        TileShape::Current().SetVecTile(tilex, tiley);

        std::vector<int64_t> shape = {8, 16};
        Tensor input(DT_FP32, shape, "input");
        Tensor output(DT_FP32, shape, "output");
        config::SetSemanticLabel("AddFunction");
        FUNCTION("AddFunction") { output = Add(input, input); }
        auto folder = config::LogTopFolder() + "/test_json_dumo";
        CreateDir(folder);
        jsonFilePath = folder + "/test_json.json";
        auto runFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");
        if (runFunc == nullptr) {
            GTEST_SKIP() << "Get func empty.";
        }
        runFunc->DumpJsonFile(jsonFilePath);
        if (RealPath(jsonFilePath).empty()) {
            GTEST_SKIP() << "Json file does not exist." << jsonFilePath;
        }
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    json loadJsonFile()
    {
        std::ifstream f(jsonFilePath);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open file: " + jsonFilePath);
        }
        return json::parse(f);
    }

    void TearDown() override {}
};

std::string JsonOutputValidationTest::jsonFilePath;

TEST_F(JsonOutputValidationTest, VerifyBasicStructure)
{
    ASSERT_FALSE(jsonFilePath.empty()) << "No valid output directory found";

    json data;
    ASSERT_NO_THROW(data = loadJsonFile()) << "Failed to parse json file";

    EXPECT_TRUE(data.contains("version")) << "Missing version field";
    EXPECT_TRUE(data.contains("functions")) << "Missing functions field";
    EXPECT_TRUE(data["functions"].is_array()) << "Functions shoule be an array";
}

TEST_F(JsonOutputValidationTest, VerifyFunctionStructure)
{
    ASSERT_FALSE(jsonFilePath.empty()) << "No valid output directory found";

    json data = loadJsonFile();
    json functions = data["functions"];

    for (const auto& func : functions) {
        // Check required function fields
        EXPECT_TRUE(func.contains(T_FIELD_KIND)) << "Function missing kind field";
        EXPECT_EQ(func[T_FIELD_KIND], static_cast<int>(Kind::T_KIND_FUNCTION)) << "Incorrect function kind";
        EXPECT_TRUE(func.contains("rawname")) << "Function missing rawname field";
        EXPECT_TRUE(func.contains("funcmagic")) << "Function missing funcmagic field";
        EXPECT_TRUE(func.contains("functype")) << "Function missing functype field";

        // Check casts
        EXPECT_TRUE(func.contains("incasts")) << "Function missing incasts field";
        EXPECT_TRUE(func.contains("outcasts")) << "Function missing outcasts field";

        // Check operations
        EXPECT_TRUE(func.contains("operations")) << "Function missing operations field";
        EXPECT_TRUE(func["operations"].is_array()) << "Operations should be an array";

        // Check hash
        EXPECT_TRUE(func.contains("hash")) << "Function missing hash field";
    }
}

TEST_F(JsonOutputValidationTest, VerifyTensorStructure)
{
    ASSERT_FALSE(jsonFilePath.empty()) << "No valid output directory found";

    json data = loadJsonFile();
    json functions = data["functions"];

    for (const auto& func : functions) {
        // Check if tensor tables exist when useTable is true
        if (func.contains("tensors") && func["tensors"].is_array()) {
            for (const auto& tensor : func["tensors"]) {
                EXPECT_TRUE(tensor.contains(T_FIELD_KIND)) << "Tensor missing kind field";
                EXPECT_EQ(tensor[T_FIELD_KIND], static_cast<int>(Kind::T_KIND_TENSOR)) << "Incorrect tensor kind";
                EXPECT_TRUE(tensor.contains("offset")) << "Tensor missing offset field";
                EXPECT_TRUE(tensor.contains("shape")) << "Tensor missing shape field";
                EXPECT_TRUE(tensor.contains("nodetype")) << "Tensor missing nodetype field";
                EXPECT_TRUE(tensor.contains(T_FIELD_RAWTENSOR)) << "Tensor missing rawtensor reference field";
                EXPECT_TRUE(tensor.contains("magic")) << "Tensor missing magic field";
            }
        }

        // Check raw tensors if present
        if (func.contains("rawtensors") && func["rawtensors"].is_array()) {
            for (const auto& rawTensor : func["rawtensors"]) {
                EXPECT_TRUE(rawTensor.contains(T_FIELD_KIND)) << "RawTensor missing kind field";
                EXPECT_EQ(rawTensor[T_FIELD_KIND], static_cast<int>(Kind::T_KIND_RAW_TENSOR))
                    << "Incorrect raw tensor kind";
                EXPECT_TRUE(rawTensor.contains("datatype")) << "RawTensor missing datatype field";
                EXPECT_TRUE(rawTensor.contains("rawshape")) << "RawTensor missing rawshape field";
                EXPECT_TRUE(rawTensor.contains("rawmagic")) << "RawTensor missing rawmagic field";

                if (rawTensor.contains("symbol")) {
                    EXPECT_TRUE(rawTensor["symbol"].is_string()) << "symbol should be a string";
                }
            }
        }
    }
}

TEST_F(JsonOutputValidationTest, VerifyOperationStructure)
{
    ASSERT_FALSE(jsonFilePath.empty()) << "No valid output directory found";

    json data = loadJsonFile();
    json functions = data["functions"];

    for (const auto& func : functions) {
        for (const auto& op : func["operations"]) {
            EXPECT_TRUE(op.contains(T_FIELD_KIND)) << "Operation missing kind field";
            EXPECT_EQ(op[T_FIELD_KIND], static_cast<int>(Kind::T_KIND_OPERATION)) << "Incorrect operation kind";
            EXPECT_TRUE(op.contains("ioperands")) << "Operation missing ioperands field";
            EXPECT_TRUE(op["ioperands"].is_array()) << "ioperands should be an array";

            EXPECT_TRUE(op.contains("ooperands")) << "Operation missing ooperands field";
            EXPECT_TRUE(op["ooperands"].is_array()) << "ooperands should be an array";

            EXPECT_TRUE(op.contains("opcode")) << "Operation missing opcode field";
            EXPECT_TRUE(op["opcode"].is_string()) << "opcode should be a string";

            EXPECT_TRUE(op.contains("opmagic")) << "Operation missing opmagic field";

            // Check semantic labels
            EXPECT_TRUE(op.contains("semantic_label")) << "Operation missing semantic_label field";
            EXPECT_TRUE(op["semantic_label"].is_object()) << "Semantic labels should be an array";

            EXPECT_TRUE(op.contains("subgraphid")) << "Operation missing subgraphid field";

            // Special handling for call operations
            if (op["opcode"] == "OP_CALL") {
                EXPECT_TRUE(op.contains("calleehash")) << "Call operation missing calleehash field";

                if (op.contains("program_funcmagic")) {
                    EXPECT_TRUE(op["program_funcmagic"].is_number()) << "program_funcmagic should be a number";
                }
            }

            // Check tile information if present
            if (op.contains("tile")) {
                EXPECT_TRUE(op["tile"].contains("vec")) << "Tile missing vec field";
                EXPECT_TRUE(op["tile"].contains("cube")) << "Tile missing cube field";
                EXPECT_TRUE(op["tile"].contains("comm")) << "Tile missing comm field";
            }
            // Check attributes if present
            if (op.contains("attr")) {
                EXPECT_TRUE(op["attr"].is_array()) << "attr should be an array";
            }
        }
    }
}
