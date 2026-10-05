# M0  bring-up：mix(1,2) 全核启动 + BufferID 核内同步 + CrossCore mode0/mode2 最小示例
# 构建：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行：
#   ./build/m0_bringup

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(ascend_mega_kernel LANGUAGES ASC CXX)

add_executable(m0_bringup
    m0/m0_bringup.asc
)

target_compile_definitions(m0_bringup PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m0_bringup PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
