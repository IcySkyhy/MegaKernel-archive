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
 * \file test_tile_op_add.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "utils/file_utils.h"

namespace npu::tile_fwk {

class TestTileOpAdd : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

int CompileCCEForSingleOpTest(const std::string& srcFile, const std::string& objFile, bool isCube)
{
    std::string curPath = GetCwd();
    std::string codeSrcPath = curPath.append("/../../../../");

    std::string coreType = isCube ? "dav-c220-cube" : "dav-c220-vec";
    const std::string envPath = std::string(std::getenv("ASCEND_AICPU_PATH"));
    std::string runtimePath = envPath + "/machine/include";
    std::string lib64Path = envPath + "/lib64";

    char ccecCmd[2048];
    std::string compileOptions = "";

    int ret = snprintf_s(ccecCmd, sizeof(ccecCmd), sizeof(ccecCmd) - 1,
                         "ccec %s -lstdc++ -O2 -g -x cce -std=c++17 --shared -fPIC "
                         "--cce-aicore-arch=%s "
                         "--cce-enable-print "
                         "-mllvm -cce-aicore-stack-size=0x8000 "
                         "-mllvm -cce-aicore-function-stack-size=0x8000 "
                         "-mllvm -cce-aicore-record-overflow=false "
                         "-mllvm -cce-aicore-addr-transform "
                         "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
                         "-L%s "
                         "-lruntime "
                         "-I%s "
                         "-I%s/framework/src/interface/tileop/arch32 "
                         "-I%s/framework/src/machine/kernel/ "
                         "-I%s/framework/src/ "
                         "-I%s/framework/src/interface "
                         "-o %s "
                         "%s",
                         compileOptions.c_str(), coreType.c_str(), lib64Path.c_str(), runtimePath.c_str(),
                         codeSrcPath.c_str(), codeSrcPath.c_str(), codeSrcPath.c_str(), codeSrcPath.c_str(),
                         objFile.c_str(), srcFile.c_str());
    ret = std::system(ccecCmd);
    return ret;
}
void CompileTestCCE(const std::string& cceFileName)
{
    CodeGenCtx ctx;
    CodeGenCloudNPU codegen(ctx); // used to PrepareDefaultOutputPath
    std::string cwd = GetCwd();
    std::string testDir = "test_add";
    std::string cceFilePath = cwd + "/../../../../" + TEST_TILE_OP_PATH + testDir + "/" + cceFileName + ".cce";

    std::string outputDirPath = cwd + "/kernel_meta";
    std::string outputFilePath = outputDirPath + "/" + cceFileName + ".o";

    int ret = CompileCCEForSingleOpTest(cceFilePath, outputFilePath, false);
    ASSERT(ret == 0) << "CompileCCEForSingleOpTest failed!!";
}

TEST_F(TestTileOpAdd, TestAddDim2)
{
    AclInit(nullptr);
    RuntimeSetDevice(GetDeviceIdByEnvVar());

    RtStream stream;
    RuntimeStreamCreate(&stream, 0);

    std::vector<int> shape = {64, 64};
    int capacity = shape[0] * shape[1];
    int size = shape[0] * shape[1] * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(size);

    std::string cceFileName = "test_tileop_add_dim2";
    CompileTestCCE(cceFileName);

    void* src0 = readToDev(GetGoldenDir() + "/add_x.bin", capacity);
    void* src1 = readToDev(GetGoldenDir() + "/add_y.bin", capacity);

    typedef void (*KernelFnPtrTy)(const std::vector<uint64_t>&, RtStream&);
    KernelFnPtrTy KernelFn = nullptr;
    void* soHandle = nullptr;
    const std::string kernelSoPath = "./kernel_meta/" + cceFileName + ".o";
    soHandle = dlopen(kernelSoPath.c_str(), RTLD_LAZY);
    *(void**)&KernelFn = dlsym(soHandle, "TestTileOpKernelLaunch");
    printf("KernelFn is %p\n", KernelFn);

    const std::vector<uint64_t> addrList = {(uint64_t)out_ptr, (uint64_t)src0, (uint64_t)src1};
    KernelFn(addrList, stream);

    // Invoke a kernel
    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    CopyFromTensor((uint8_t*)res.data(), (uint8_t*)out_ptr, size);
    readInput(GetGoldenDir() + "/add_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

} // namespace npu::tile_fwk
