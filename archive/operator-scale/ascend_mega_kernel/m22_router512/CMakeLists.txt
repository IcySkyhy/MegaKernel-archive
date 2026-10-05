# M22 真实规模 MoE router（512 专家 / top-10）：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m22_router512/build -S m22_router512 -DCMAKE_BUILD_TYPE=Release
#   cmake --build m22_router512/build -j4
# 运行（读 m22_router512/data/router_weight.bin）：
#   M22_OUT=./m22_out ./m22_router512/build/m22_router512
# 归并树 A/B（同一 .asc，编译期切换 2 路 / 4 路）：
#   ./m22_router512/build/m22_router512_m4
# 交叉验证（对 tools/golden/moe_block_ref.py）：
#   /usr/local/python3.12.13/bin/python3.12 m22_router512/check_ref.py ./m22_out/real_m33 --w m22_router512/data/router_weight.bin

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m22_router512 LANGUAGES ASC CXX)

# 默认目标：M22_MERGE_MODE=2（m7 现行 2 路归并树）
add_executable(m22_router512
    m22_router512.asc
)
target_compile_definitions(m22_router512 PRIVATE
    NPU_ARCH_DAV_3510
    M22_MERGE_MODE=2
)
target_compile_options(m22_router512 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_link_libraries(m22_router512 m)

# A/B 目标：M22_MERGE_MODE=4（4 路归并树，2 级）
add_executable(m22_router512_m4
    m22_router512.asc
)
target_compile_definitions(m22_router512_m4 PRIVATE
    NPU_ARCH_DAV_3510
    M22_MERGE_MODE=4
)
target_compile_options(m22_router512_m4 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_link_libraries(m22_router512_m4 m)
