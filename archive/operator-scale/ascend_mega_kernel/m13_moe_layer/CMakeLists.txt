# M13 MoE block layer kernel 骨架：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m13_moe_layer/build -S m13_moe_layer -DCMAKE_BUILD_TYPE=Release
#   cmake --build m13_moe_layer/build -j2
# 运行（消费 tools/golden/data/{m1,m33}，路径相对仓库根）：
#   ./m13_moe_layer/build/m13_moe_layer [tools/golden/data] [case ...]

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m13_moe_layer LANGUAGES ASC CXX)

add_executable(m13_moe_layer
    m13_moe_layer.asc
)

target_compile_definitions(m13_moe_layer PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m13_moe_layer PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host C 参考需与 kernel 内独立 Mul/Add 向量指令同一 IEEE 运算序列
    # （禁止 FMA 融合），否则 unpermute/combine 无法逐位对齐
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

# host 侧 C 参考用 sqrtf/expf
target_link_libraries(m13_moe_layer PRIVATE m)
