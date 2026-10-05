# M2 MXFP4 激活量化：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m2_mxfp4_quant/build -S m2_mxfp4_quant -DCMAKE_BUILD_TYPE=Release
#   cmake --build m2_mxfp4_quant/build -j4
# 运行：
#   ./m2_mxfp4_quant/build/m2_mxfp4_quant

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m2_mxfp4_quant LANGUAGES ASC CXX)

add_executable(m2_mxfp4_quant
    m2_mxfp4_quant.asc
)

target_compile_definitions(m2_mxfp4_quant PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m2_mxfp4_quant PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
