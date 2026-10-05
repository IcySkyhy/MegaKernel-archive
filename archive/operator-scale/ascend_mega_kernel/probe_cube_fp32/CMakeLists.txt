# probe_cube_fp32：M106 —— cube（AIC）吃 fp32 操作数的能力 / 精度形态 / 吞吐 标定探针
# 独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行：见 README.md（每个 mode 一个独立进程；run_probes.sh 全量重跑 + 归档）
#
# 目标：
#   probe_cube_fp32   主探针（prec / hf32 / perf / perf_hf32 / dump）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_cube_fp32 LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_cube_fp32 probe_cube_fp32.asc)
target_compile_definitions(probe_cube_fp32 PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_cube_fp32 PRIVATE ${ASC_ARCH_FLAGS})
target_link_libraries(probe_cube_fp32 m)
