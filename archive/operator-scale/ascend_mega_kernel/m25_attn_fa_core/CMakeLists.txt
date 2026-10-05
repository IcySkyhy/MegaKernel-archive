# M117 / Wave-B3 —— 稠密 causal prefill attention core 的**独立验证路**（不碰 m15_layer_loop/CMakeLists.txt）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m25_attn_fa_core/build -S m25_attn_fa_core -DCMAKE_BUILD_TYPE=Release
#   cmake --build m25_attn_fa_core/build -j4
# 运行（**必须在设备槽里**）：
#   flock -w 900 /tmp/npu0.lock ./m25_attn_fa_core/build/m25_attn_fa_core core
# 对拍：
#   python3 m25_attn_fa_core/check_ref.py m25_attn_fa_core/out

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m25_attn_fa_core LANGUAGES ASC CXX)

# 段体头 `m15_attn_fa_core.h` 交付到 `m15_layer_loop/`（融合形态），独立验证路通过 -I 引用它，
# **不复制**（复制会让「被测对象」与「融合用的那份」漂移）。
add_executable(m25_attn_fa_core
    m25_attn_fa_core.asc
)

target_include_directories(m25_attn_fa_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop
)

target_compile_definitions(m25_attn_fa_core PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m25_attn_fa_core PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 参考需与 kernel 同一 IEEE 运算序列（禁止 FMA 融合）
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m25_attn_fa_core PRIVATE m)
