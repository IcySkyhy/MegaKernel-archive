# M11 bf16 GEMM 单 AIC 核：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m11_bf16_gemm/build -S m11_bf16_gemm -DCMAKE_BUILD_TYPE=Release
#   cmake --build m11_bf16_gemm/build -j4
# 运行：
#   ./m11_bf16_gemm/build/m11_bf16_gemm

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m11_bf16_gemm LANGUAGES ASC CXX)

add_executable(m11_bf16_gemm
    m11_bf16_gemm.asc
)

target_compile_definitions(m11_bf16_gemm PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m11_bf16_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
