# M22 GDN layer kernel 骨架：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录，或本目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m14_gdn_layer/build -S m14_gdn_layer -DCMAKE_BUILD_TYPE=Release
#   cmake --build m14_gdn_layer/build -j4
# 运行（在空目录里跑，dump 会落在 cwd）：
#   ./m14_gdn_layer/build/m14_gdn_layer [chain|slice|chain2 ...]
#   M14_DUMP=1 ./m14_gdn_layer/build/m14_gdn_layer      # 落盘供 check_ref.py
#   M14_STAGE_LIMIT=<n> ...                             # 段序截断（bring-up 定位）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m14_gdn_layer LANGUAGES ASC CXX)

add_executable(m14_gdn_layer
    m14_gdn_layer.asc
)

target_compile_definitions(m14_gdn_layer PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m14_gdn_layer PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host C 参考需与 kernel 内向量指令同一 IEEE 运算序列（禁止 FMA 融合），
    # 否则 res1/res2/conv_state 无法做位级判据
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

# host 侧 C 参考用 sqrt/exp/log
target_link_libraries(m14_gdn_layer PRIVATE m)
