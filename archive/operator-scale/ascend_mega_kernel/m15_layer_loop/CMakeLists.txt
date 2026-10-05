# M25 48 层循环骨架：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
#   cmake --build m15_layer_loop/build -j4
# 运行（权重 manifest 由 slice_layer_manifest.py 生成并入库）：
#   ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all
#   M15_SYNTH=1 ./m15_layer_loop/build/m15_layer_loop "" all      # 不等 checkpoint 的冒烟路径

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m15_layer_loop LANGUAGES ASC CXX)

add_executable(m15_layer_loop
    m15_layer_loop.asc
)

target_compile_definitions(m15_layer_loop PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m15_layer_loop PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 参考需与 kernel 内向量指令同一 IEEE 运算序列（禁止 FMA 融合）：
    # Ver B 的 conv_state 位级判据与 ssm_state 容差判据都依赖这一条
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

# host 侧 C 参考用 sqrt/exp/log
target_link_libraries(m15_layer_loop PRIVATE m)

# ---- M85：PLE 独立 kernel（自己的入口符号 + 自己的 main()；见 m15_ple.asc 顶部 PROLOGUE）----
# 构建： cmake --build m15_layer_loop/build -j4 --target m15_ple
# 数据： python3.12 m15_layer_loop/ple/gen_ple_data.py
# 运行： M15_PLE_OUT=m15_layer_loop/ple/out ./m15_layer_loop/build/m15_ple
# 对拍： python3.12 m15_layer_loop/m15_ple_check.py m15_layer_loop/ple/out
add_executable(m15_ple
    m15_ple.asc
)

target_compile_definitions(m15_ple PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m15_ple PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # 与 host 参考同一 IEEE 运算序列（禁止 FMA 融合）——判据依赖 bf16 舍入点全在源码里
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m15_ple PRIVATE m)

# ---- M101：attention 核心段的独立设备验证路（自己的入口符号 + 自己的 main()；见 m15_attn_core.asc 顶部）----
# 该 target 只验证「段」，不接进层 kernel（挂载点归 M102）。
# 构建： cmake --build m15_layer_loop/build -j4 --target m15_attn_core
# 生成： python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py
# 校验： python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py --check
# 运行： M15AC_OUT=m15_layer_loop/evidence/attn_core/out \
#          M15AC_REFDIR=m10_attn_decode/data ./m15_layer_loop/build/m15_attn_core core
# 判据： 在 dump 所在目录用 donor 的判据原样跑（不复制判据）：
#        cd m15_layer_loop/evidence/attn_core/out && PYTHONDONTWRITEBYTECODE=1 \
#          /usr/local/python3.12.13/bin/python3 ../../../../m10_attn_decode/check_ref.py
add_executable(m15_attn_core
    m15_attn_core.asc
)

target_compile_definitions(m15_attn_core PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m15_attn_core PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 侧判据（o_proj 的逐位参考）与 kernel 内向量指令必须走同一 IEEE 运算序列
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m15_attn_core PRIVATE m)
