# M17 MoE block 真实权重端到端（M29）：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m17_moe_real/build -S m17_moe_real -DCMAKE_BUILD_TYPE=Release
#   cmake --build m17_moe_real/build -j2        # 并行度别拉满（cgroup 内存上限 32GB）
#
# 运行（先确认 m17_weight_manifest.txt 已生成）：
#   ./m17_moe_real/build/m17_moe_real m17_moe_real/m17_weight_manifest.txt --m 1 --token 1000

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m17_moe_real LANGUAGES ASC CXX)

add_executable(m17_moe_real
    m17_moe_layer.asc
)

target_compile_definitions(m17_moe_real PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m17_moe_real PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 侧参考复算需与 kernel 内独立 Mul/Add 向量指令同一 IEEE 运算序列（禁 FMA 融合），
    # 否则 unpermute/combine 无法逐位对齐
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

# host 用 sqrtf/expf/ldexp
target_link_libraries(m17_moe_real PRIVATE m)

# host 头文件依赖（改了要重编）
set_property(SOURCE m17_moe_layer.asc APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/m17_resources.h")
