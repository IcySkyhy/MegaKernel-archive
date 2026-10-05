# m23_gdn_prefill —— M115 / Wave B1 的**独立验证路**（自带 main()，不碰 m15_layer_loop/CMakeLists.txt）
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行（设备槽内，dump 写 cwd）：
#   cd build && flock -w 900 /tmp/npu0.lock ./m23_gdn_prefill
# 判据（numpy float64 逐句参考）：
#   <python> m23_gdn_prefill/check_ref.py

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m23_gdn_prefill LANGUAGES ASC CXX)

add_executable(m23_gdn_prefill m23_gdn_prefill.asc)

target_include_directories(m23_gdn_prefill PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")

target_compile_definitions(m23_gdn_prefill PRIVATE NPU_ARCH_DAV_3510)

target_compile_options(m23_gdn_prefill PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 最小核间握手探针（塔裁 ③：先验 mode 形态再收 J）
add_executable(m23_cc_probe m23_cc_probe.asc)
target_compile_definitions(m23_cc_probe PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(m23_cc_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 诊断收窄档（塔裁 ④）：只跑 J1，用于把挂死定位到"J1/脚手架"还是"J2/J3"
add_executable(m23_gdn_prefill_j1 m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j1 PRIVATE NPU_ARCH_DAV_3510 M15GP_J1_ONLY=1)
target_compile_options(m23_gdn_prefill_j1 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 收窄第二档：只跑 J1+J2
add_executable(m23_gdn_prefill_j12 m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12 PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1)
target_compile_options(m23_gdn_prefill_j12 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 收窄第三档：J1+J2，且 J2 的 Nd2Nz 从第 1 个 chunk 起跳过（诊断 L1 令牌跨次复用）
add_executable(m23_gdn_prefill_j12nz m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12nz PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12nz PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1)
target_compile_options(m23_gdn_prefill_j12nz PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12nz_q m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12nz_q PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12nz_q PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=1)
target_compile_options(m23_gdn_prefill_j12nz_q PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12nz_st m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12nz_st PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12nz_st PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=2)
target_compile_options(m23_gdn_prefill_j12nz_st PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12nz_w m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12nz_w PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12nz_w PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=3)
target_compile_options(m23_gdn_prefill_j12nz_w PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12_onlyq m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12_onlyq PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12_onlyq PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=4)
target_compile_options(m23_gdn_prefill_j12_onlyq PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12_onlyst m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12_onlyst PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12_onlyst PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=5)
target_compile_options(m23_gdn_prefill_j12_onlyst PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

add_executable(m23_gdn_prefill_j12_onlyw m23_gdn_prefill.asc)
target_include_directories(m23_gdn_prefill_j12_onlyw PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop")
target_compile_definitions(m23_gdn_prefill_j12_onlyw PRIVATE NPU_ARCH_DAV_3510 M15GP_J1J2_ONLY=1 M15GP_J2_NZ_SKIP_FROM=1 M15GP_J2_NZ_WHICH=6)
target_compile_options(m23_gdn_prefill_j12_onlyw PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 最小 AIC 探针：L0A/L0B/L0C 的 16/17 tile 边界（1 个 AIC）
add_executable(probe_l0 probe_l0/probe_l0.asc)
target_compile_definitions(probe_l0 PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_l0 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
