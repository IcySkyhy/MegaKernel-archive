# probe_ub_bufid_tail —— M126 独立探针工程（不依赖仓库顶层 CMakeLists.txt）
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行（设备槽纪律见 run_probe.sh）：
#   ./build/probe_ub_bufid_tail tail base_N3 1 out --trace
#   ./build/probe_ub_bufid_tail sp   mte2_drain_8 1 out

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_ub_bufid_tail LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_ub_bufid_tail probe_ub_bufid_tail.asc)

target_compile_definitions(probe_ub_bufid_tail PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_ub_bufid_tail PRIVATE ${ASC_ARCH_FLAGS})
target_link_libraries(probe_ub_bufid_tail PRIVATE m)
