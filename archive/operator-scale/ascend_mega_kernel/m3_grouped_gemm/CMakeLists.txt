# M3 多核 grouped MXFP4 GEMM：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m3_grouped_gemm/build -S m3_grouped_gemm -DCMAKE_BUILD_TYPE=Release
#   cmake --build m3_grouped_gemm/build -j4
# 运行：
#   ./m3_grouped_gemm/build/m3_grouped_gemm

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m3_grouped_gemm LANGUAGES ASC CXX)

add_executable(m3_grouped_gemm
    m3_grouped_gemm.asc
)

target_compile_definitions(m3_grouped_gemm PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m3_grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
