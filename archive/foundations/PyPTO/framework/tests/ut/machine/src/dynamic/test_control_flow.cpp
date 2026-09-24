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
 * \file test_control_flow.cpp
 * \brief
 */

#include "interface/utils/string_utils.h"
#include "tilefwk/platform.h"

#include "test_machine_common.h"

struct ControlFlowTest : UnitTestBase {};

std::string GetDeclName(const std::string& name)
{
    std::vector<std::string> descList = StringUtils::Split(name, "_");
    return descList[1];
}

TEST_F(ControlFlowTest, RunDeviceContext)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1))
        {
            (void)_;
            output = Add(inputA, inputB);
        }
        LOOP("s1", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x40 - 0x1))
        {
            (void)_;
            output = Add(output, inputB);
        }
    }

    struct Inspector {
        int count{0};
        std::vector<std::string> nameList;
        std::vector<DevAscendFunction*> rootList;
        static void Entry(void* inspector_, DeviceExecuteContext* execCtx, DynDeviceTask* task)
        {
            Inspector* inspector = reinterpret_cast<Inspector*>(inspector_);
            (void)execCtx;
            (void)task;
            inspector->count++;
            DynFuncDataCache* cacheList = task->GetDynFuncDataCacheList();
            for (size_t k = 0; k < task->dynFuncDataCacheListSize; k++) {
                inspector->rootList.push_back(cacheList->At(k).devFunc);
                std::string currName = GetDeclName(cacheList->At(k).devFunc->GetRawName());
                inspector->nameList.push_back(currName);
            }
        }
    };
    Inspector inspector;
    PyptoKernelCtrlServerRegisterTaskInspector(Inspector::Entry, &inspector);

    DeviceLauncherConfig config;
    config.blockdim = 24; // 24: max blockdim
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config));
    EXPECT_EQ(0x10, inspector.count);
    EXPECT_EQ(0x40, inspector.rootList.size());
    EXPECT_EQ("s0", inspector.nameList[0]);
    for (size_t k = 1; k < 0x40; k++) {
        EXPECT_EQ("s1", inspector.nameList[k]);
    }
    PyptoKernelCtrlServerRegisterTaskInspector(nullptr, nullptr);
}

TEST_F(ControlFlowTest, TestDD)
{
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    std::vector<float> golden(n * s * s, 128.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {t0, t1, blockTable}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s))
        {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2 * s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2 * s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, qi, ki, false, true);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }

    DeviceLauncherConfig config;
    config.blockdim = 25;
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config));
}

TEST_F(ControlFlowTest, TensorRecycleDestruct)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 100);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, k, LoopRange(0x2))
        {
            Tensor mid(DT_INT32, {n, n}, "O");
            LOOP("s0", FunctionType::DYNAMIC_LOOP, i, LoopRange(0x4))
            {
                LOOP("s1", FunctionType::DYNAMIC_LOOP, j, LoopRange(0x4))
                {
                    Tensor t0 = View(inputA, {tiling, tiling}, {i * tiling, j * tiling});
                    Tensor t1 = View(inputB, {tiling, tiling}, {i * tiling, j * tiling});
                    Tensor ts = Add(t0, t1);
                    Assemble(ts, {i * tiling, j * tiling}, mid);
                }
            }
            LOOP("sum", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1))
            {
                (void)_;
                IF(k == 0) { output = Add(mid, Element(DT_INT32, 0)); }
                ELSE { output = Add(output, mid); }
            }
        }
    }

    struct CapturedTaskData {
        bool isAddr0Valid;
        bool isAddr1Valid;
        uint64_t addr0Value;
        uint64_t addr1Value;
    };

    struct Inspector {
        std::vector<CapturedTaskData> dataList;

        static void Entry(void* inspector_, DeviceExecuteContext* execCtx, DynDeviceTask* task)
        {
            (void)execCtx;
            Inspector* inspector = reinterpret_cast<Inspector*>(inspector_);

            DynFuncDataCache* cacheList = task->GetDynFuncDataCacheList();
            DevAscendFunctionDuppedData* dup0 = cacheList->At(0).duppedData;
            DevAscendFunctionDuppedData* dup1 = cacheList->At(0x4 * 0x4 + 0x1).duppedData;

            CapturedTaskData data;
            data.isAddr0Valid = dup0->GetOutcastAddress(0).IsAddress();
            data.isAddr1Valid = dup1->GetOutcastAddress(0).IsAddress();
            data.addr0Value = dup0->GetOutcastAddress(0).GetAddressValue();
            data.addr1Value = dup1->GetOutcastAddress(0).GetAddressValue();

            inspector->dataList.push_back(data);
        }
    };

    Inspector inspector;
    PyptoKernelCtrlServerRegisterTaskInspector(Inspector::Entry, &inspector);

    DeviceLauncherConfig config;
    config.blockdim = 25;
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config));

    EXPECT_EQ(1, inspector.dataList.size());

    const auto& data = inspector.dataList[0];
    EXPECT_TRUE(data.isAddr0Valid);
    EXPECT_TRUE(data.isAddr1Valid);
    EXPECT_NE(data.addr0Value, data.addr1Value);
    PyptoKernelCtrlServerRegisterTaskInspector(nullptr, nullptr);
}

static DeviceTensorData toTensorData(const std::shared_ptr<LogicalTensor>& t)
{
    return DeviceTensorData(t->Datatype(), nullptr, t->GetShape());
}

TEST_F(ControlFlowTest, CtrlFlowPartialCache)
{
    // every task 4 root func
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);

    int tiling = 32;
    int n = tiling * 4;
    TileShape::Current().SetVecTile(tiling, tiling);

    Tensor input1(DT_INT32, {n, n}, "A");
    Tensor input2(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input1, 1),
        RawTensorData::CreateConstantTensor<int32_t>(input2, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    // 17 root func in total
    FUNCTION("main", {input1, input2}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(input1, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(input1, 1) / tiling))
            {
                auto a = View(input1, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(input2, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("Use", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }

    std::vector<DeviceTensorData> inputList = {toTensorData(input1.GetStorage()), toTensorData(input2.GetStorage())};
    std::vector<DeviceTensorData> outputList = {toTensorData(output.GetStorage())};
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    EmulationMemoryUtils memUtils;
    DevControlFlowCache* ctrolCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, inputList,
                                                          outputList, &ctrolCache, config));
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());

    EXPECT_EQ(0x5, ctrolCache->deviceTaskCount);
    EXPECT_EQ(0x0, ctrolCache->deviceTaskSkippedCount);

    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrolCache->RelocMetaCache(0, (intptr_t)ctrolCache);
    ctrolCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrolCache);

    for (int i = 0; i < 0x4; i++) {
        auto dynTaskBase = ctrolCache->deviceTaskCacheList[i].dynTaskBase;
        EXPECT_EQ(0x4, dynTaskBase->GetDynFuncDataList()->Size());
    }
    auto dynTaskBase = ctrolCache->deviceTaskCacheList[0x4].dynTaskBase;
    EXPECT_EQ(0x1, dynTaskBase->GetDynFuncDataList()->Size());

    ctrolCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrolCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);
    ctrolCache->RelocMetaCache((intptr_t)ctrolCache, 0);
    EXPECT_EQ(false, ctrolCache->isRelocMetaDev);
    EXPECT_EQ(true, ctrolCache->isActivated);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrolCache, config));
}

TEST_F(ControlFlowTest, TestMainBlock)
{
    config::SetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE, 1);

    int tile_size = 32;
    TileShape::Current().SetVecTile(tile_size, tile_size);
    int tensor_dim = tile_size * 4;

    Tensor tensor_a(DT_INT32, {tensor_dim, tensor_dim}, "A");
    Tensor tensor_b(DT_INT32, {tensor_dim, tensor_dim}, "B");
    Tensor output_tensor(DT_INT32, {tensor_dim, tensor_dim}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(tensor_a, 1),
        RawTensorData::CreateConstantTensor<int32_t>(tensor_b, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output_tensor, 0),
    });

    FUNCTION("main", {tensor_a, tensor_b}, {output_tensor})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, loop_idx, LoopRange(0x2))
        {
            Tensor middle_tensor(DT_INT32, {tensor_dim, tensor_dim}, "O");
            LOOP("s0", FunctionType::DYNAMIC_LOOP, row_idx, LoopRange(0x4))
            {
                LOOP("s1", FunctionType::DYNAMIC_LOOP, col_idx, LoopRange(0x4))
                {
                    Tensor tile_a = View(tensor_a, {tile_size, tile_size}, {row_idx * tile_size, col_idx * tile_size});
                    Tensor tile_b = View(tensor_b, {tile_size, tile_size}, {row_idx * tile_size, col_idx * tile_size});
                    Tensor tile_sum = Add(tile_a, tile_b);
                    Assemble(tile_sum, {row_idx * tile_size, col_idx * tile_size}, middle_tensor);
                }
            }

            LOOP("sum", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
            {
                (void)_;
                IF(loop_idx == 0) { output_tensor = Add(middle_tensor, Element(DT_INT32, 0)); }
                else
                {
                    output_tensor = Add(output_tensor, middle_tensor);
                }
            }
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 25; // 25: block dim
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config));
}

TEST_F(ControlFlowTest, TestParallelLoop)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);

    int parallel_tile_size = 32;
    TileShape::Current().SetVecTile(parallel_tile_size, parallel_tile_size);
    int parallel_tensor_dim = parallel_tile_size * 4;

    Tensor input_tensor_x(DT_INT32, {parallel_tensor_dim, parallel_tensor_dim}, "X");
    Tensor input_tensor_y(DT_INT32, {parallel_tensor_dim, parallel_tensor_dim}, "Y");
    Tensor result_tensor(DT_INT32, {parallel_tensor_dim, parallel_tensor_dim}, "R");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input_tensor_x, 1),
        RawTensorData::CreateConstantTensor<int32_t>(input_tensor_y, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(result_tensor, 0),
    });

    FUNCTION("main", {input_tensor_x, input_tensor_y}, {result_tensor})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, outer_iter, LoopRange(0x2), {}, false, true)
        {
            Tensor accum_tensor(DT_INT32, {parallel_tensor_dim, parallel_tensor_dim}, "A");
            LOOP("s0", FunctionType::DYNAMIC_LOOP, row_iter, LoopRange(0x4))
            {
                LOOP("s1", FunctionType::DYNAMIC_LOOP, col_iter, LoopRange(0x4))
                {
                    Tensor view_x = View(input_tensor_x, {parallel_tile_size, parallel_tile_size},
                                         {row_iter * parallel_tile_size, col_iter * parallel_tile_size});
                    Tensor view_y = View(input_tensor_y, {parallel_tile_size, parallel_tile_size},
                                         {row_iter * parallel_tile_size, col_iter * parallel_tile_size});
                    Tensor add_result = Add(view_x, view_y);
                    Assemble(add_result, {row_iter * parallel_tile_size, col_iter * parallel_tile_size}, accum_tensor);
                }
            }

            LOOP("sum", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
            {
                (void)_;
                IF(outer_iter == 0) { result_tensor = Add(accum_tensor, Element(DT_INT32, 0)); }
                else
                {
                    result_tensor = Add(result_tensor, accum_tensor);
                }
            }
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 25; // 25: block dim
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config));
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}
