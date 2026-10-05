# probe_crosscore_tail —— M134 独立探针工程（不依赖仓库顶层 CMakeLists.txt）
#
# 目标：把 M117/B3 `m=64` 挂死收窄为「跨核一侧（AIC Fixpipe L0C→对侧 AIV UB +
#       CrossCore mode 2 flagId 记账）」的最小 mix 复现路。
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行：
#   ./build/probe_crosscore_tail list                        # 变体清单
#   ./build/probe_crosscore_tail base 16 1 out               # <变体> <N> <runNo> [outdir]
#   ./build/probe_crosscore_tail base 16 1 out --trace       # 带逐轮打印
# 证据（脚本自带 flock -w 300 + 进锁先 npu-smi + 锁内 timeout；每档各自进锁）：
#   N=16 REPS=3 bash run_probe.sh                            # 默认 VARIANTS 全跑
#   N=16 REPS=3 VARIANTS="base c_m2canon" bash run_probe.sh  # 指定变体
#   TRACE=1 TAG=_trace N=4 REPS=2 VARIANTS="base" bash run_probe.sh
#   bash tools/summarize.sh                                  # 从逐次日志重算 matrix_N<N>.txt

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_crosscore_tail LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_crosscore_tail probe_crosscore_tail.asc)

target_compile_definitions(probe_crosscore_tail PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_crosscore_tail PRIVATE ${ASC_ARCH_FLAGS})
target_link_libraries(probe_crosscore_tail PRIVATE m)
