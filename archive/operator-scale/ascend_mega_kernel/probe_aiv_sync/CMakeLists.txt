# probe_aiv_sync —— AIV↔AIV 核间屏障语义与可靠性独立探针工程（不依赖仓库顶层 CMakeLists.txt）
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行 + 归档证据：
#   bash run_probe.sh                 # 全量变体 × 9 次独立进程 + 落盘 + sha256
#
# 目标：
#   probe_aiv_sync       主探针（屏障配方 × 写几何 × 核数 × AIC/AIV 对照）
#   probe_visibility     跨核 GM 可见性最小实验（flag vs 纯轮询 vs DCCI）
#   probe_m53_reverse    M53 probe_aiv_barrier.asc 的逐字副本（反向验证用，不得改动逻辑）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_aiv_sync LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_aiv_sync probe_aiv_sync.asc)
add_executable(probe_visibility probe_visibility.asc)
add_executable(probe_m53_reverse probe_m53_reverse.asc)
add_executable(probe_aic_aiv probe_aic_aiv.asc)

foreach(tgt probe_aiv_sync probe_visibility probe_m53_reverse probe_aic_aiv)
    target_compile_definitions(${tgt} PRIVATE NPU_ARCH_DAV_3510)
    target_compile_options(${tgt} PRIVATE ${ASC_ARCH_FLAGS})
    target_link_libraries(${tgt} m)
endforeach()
