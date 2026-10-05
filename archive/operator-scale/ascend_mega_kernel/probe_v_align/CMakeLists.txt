# probe_v_align：M57 逐 API 的 UB 对齐要求探针（独立 CMake 工程，不依赖仓库顶层 CMakeLists.txt）
# 构建：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
# 运行（每个变体一次进程，逐变体抓 aclError / 逐字节对拍）：
#   bash run_probes.sh            # 全量矩阵 + 证据归档

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_v_align LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_v_align probe_v_align.asc)
target_compile_definitions(probe_v_align PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_v_align PRIVATE ${ASC_ARCH_FLAGS})

add_executable(probe_v_ildl probe_v_ildl.asc)
target_compile_definitions(probe_v_ildl PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_v_ildl PRIVATE ${ASC_ARCH_FLAGS})
