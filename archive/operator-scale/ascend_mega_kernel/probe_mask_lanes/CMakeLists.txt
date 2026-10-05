# probe_mask_lanes：M94 —— CreateMask<T, ALL>() 的 lane 数与落盘字节数实测（独立 CMake 工程）
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
# 运行：见 README.md（每个变体一个独立进程；run_probes.sh 全量重跑 + 归档）
#
# 目标：
#   probe_mask_lanes   主探针（list / run <op>）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_mask_lanes LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_mask_lanes probe_mask_lanes.asc)
target_compile_definitions(probe_mask_lanes PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_mask_lanes PRIVATE ${ASC_ARCH_FLAGS})
