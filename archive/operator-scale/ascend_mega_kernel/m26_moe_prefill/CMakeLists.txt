# M118（B4）MoE prefill 段：独立验证工程（不依赖仓库顶层 CMakeLists.txt，也不碰
# m15_layer_loop/CMakeLists.txt —— 它归在飞的 M101）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m26_moe_prefill/build -S m26_moe_prefill -DCMAKE_BUILD_TYPE=Release
#   cmake --build m26_moe_prefill/build -j4
# 运行（设备槽自助排队；跑前 npu-smi 复查）：
#   flock -w 900 /tmp/npu0.lock ./m26_moe_prefill/build/m26_moe_prefill --m 64
# 对拍（numpy float64 参考 + 非空洞性 + 负向对照）：
#   /usr/local/python3.12.13/bin/python3 m26_moe_prefill/check_ref.py

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m26_moe_prefill LANGUAGES ASC CXX)

add_executable(m26_moe_prefill
    m26_moe_prefill.asc
)

# 段体头交付到 m15_layer_loop/（那才是要被融合进层 kernel 的东西）；本工程只**读**它。
target_include_directories(m26_moe_prefill PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop"
)

target_compile_definitions(m26_moe_prefill PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m26_moe_prefill PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 参考与 kernel 内向量指令同一 IEEE 运算序列（禁止 FMA 融合）：
    # T1/T3 判据都依赖这一条
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m26_moe_prefill PRIVATE m)

set_property(SOURCE m26_moe_prefill.asc APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop/m15_moe_prefill.h")
set_property(SOURCE m26_moe_prefill.asc APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop/m15_moe_prefill_res.h")
set_property(SOURCE m26_moe_prefill.asc APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop/m15_moe_prefill_host.h")

# m26_consts：只打印资源头的编译期读数（不碰设备；见该文件头注）
add_executable(m26_consts m26_consts.asc)
target_include_directories(m26_consts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m26_consts PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(m26_consts PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_link_libraries(m26_consts PRIVATE m)
set_property(SOURCE m26_consts.asc APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop/m15_moe_prefill_res.h")
