# probe_ub2l1：M121 —— 3510 UB→L1 硬件通道真机探针（独立 CMake 工程，不依赖仓库顶层 CMakeLists.txt）
#
# 同一份源码编两个可执行：
#   probe_ub2l1        —— 默认（无 -DENABLE_CV_COMM_VIA_SSBUF）⇒ 对照组 / C API 档
#   probe_ub2l1_ssbuf  —— 带 -DENABLE_CV_COMM_VIA_SSBUF=true（官方样例的配置）⇒ 基础 API 硬通道档
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行见 README.md；设备槽用 flock -w 900 /tmp/npu0.lock <命令>

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_ub2l1 LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_ub2l1 probe_ub2l1.asc)
target_compile_definitions(probe_ub2l1 PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_ub2l1 PRIVATE ${ASC_ARCH_FLAGS})

add_executable(probe_ub2l1_ssbuf probe_ub2l1.asc)
target_compile_definitions(probe_ub2l1_ssbuf PRIVATE NPU_ARCH_DAV_3510 ENABLE_CV_COMM_VIA_SSBUF=true)
target_compile_options(probe_ub2l1_ssbuf PRIVATE ${ASC_ARCH_FLAGS})
