# M1 MXFP4 GEMM 单核移植：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m1_mxfp4_gemm/build -S m1_mxfp4_gemm -DCMAKE_BUILD_TYPE=Release
#   cmake --build m1_mxfp4_gemm/build -j4
# 运行：
#   ./m1_mxfp4_gemm/build/m1_mxfp4_gemm

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m1_mxfp4_gemm LANGUAGES ASC CXX)

add_executable(m1_mxfp4_gemm
    m1_mxfp4_gemm.asc
)

target_compile_definitions(m1_mxfp4_gemm PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m1_mxfp4_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
