# M116（Wave B2）：attention 前端的 prefill 段体 + cache 填的 **独立验证路**
#
# 为什么独立工程（而不是注册进 m15_layer_loop/CMakeLists.txt）：
#   `docs/15` 的 `M103-2.0` P1（塔裁 2026-09-27）：融合 TU 是串行资源，**Wave B 一律不碰**
#   `m15_layer_loop/CMakeLists.txt`（归在飞的 M101）。=> 每条 Wave B 自带 CMakeLists + 带 main() 的 .asc。
#
# 构建（在本目录或仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m24_attn_prefill/build -S m24_attn_prefill -DCMAKE_BUILD_TYPE=Release
#   cmake --build m24_attn_prefill/build -j8
# 运行（设备槽纪律：flock -w 900 /tmp/npu0.lock ...；进锁后再 npu-smi 复查）：
#   flock -w 900 /tmp/npu0.lock ./m24_attn_prefill/build/m24_attn_prefill
#   M24_ONLY=big4097 flock -w 900 /tmp/npu0.lock ./m24_attn_prefill/build/m24_attn_prefill

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m24_attn_prefill LANGUAGES ASC CXX)

add_executable(m24_attn_prefill
    m24_attn_prefill.asc
)

target_compile_definitions(m24_attn_prefill PRIVATE
    NPU_ARCH_DAV_3510
)

# 段体头在 ../m15_layer_loop/（段体交付到那里；本工程只把它 include 进来）
target_include_directories(m24_attn_prefill PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop
)

target_compile_options(m24_attn_prefill PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 参考需与 kernel 内同一 IEEE 运算序列（禁止 FMA 融合）：判据的 ε 推导依赖这一条
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m24_attn_prefill PRIVATE m pthread)
