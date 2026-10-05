# M7 Router softmax topk：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m7_router_topk/build -S m7_router_topk -DCMAKE_BUILD_TYPE=Release
#   cmake --build m7_router_topk/build -j4
# 运行（仓库根目录，读 m7_router_topk/data/router_weight.bin）：
#   ./m7_router_topk/build/m7_router_topk
# 交叉验证（对 moe_block_ref.py）：
#   M7_OUT=./m7_out ./m7_router_topk/build/m7_router_topk
#   /usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m1
#   /usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m33

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m7_router_topk LANGUAGES ASC CXX)

add_executable(m7_router_topk
    m7_router_topk.asc
)

target_compile_definitions(m7_router_topk PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m7_router_topk PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_link_libraries(m7_router_topk m)
