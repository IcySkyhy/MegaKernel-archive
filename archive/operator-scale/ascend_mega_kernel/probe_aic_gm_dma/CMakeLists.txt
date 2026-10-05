# probe_aic_gm_dma —— M141 独立探针工程（不依赖仓库顶层 CMakeLists.txt）
#
# 目标：把「AIC（cube 核）能否把数据从 UB 搬到 GM；若不能卡在哪条腿」缩成一个
#       **只让 AICore 0 的 AIC 干活**的最小 mix 核，逐档只换「UB→GM 这一条腿」的走法；
#       同时给出对照腿（AIC 经 FIXP 的 L0C→GM）与 AIV 的 UB→GM 正对照。
#
# 复用机制（带 文件:行；仓库内 M134/M121 既有探针）：
#   · 独立 `__mix__(1,2)` kernel + 自带 CMakeLists + run_probe.sh 的工程形态，
#     沿用 probe_crosscore_tail/CMakeLists.txt:1-31 与
#           probe_crosscore_tail/run_probe.sh:1-96（同一套 flock 纪律）。
#   · `host 侧 ACL 装载 + <<<blockDim>>> 启动 + 逐元素对拍` 沿用
#           probe_crosscore_tail/probe_crosscore_tail.asc:600-740。
#   · FIXP L0C→GM 对照腿的算子小助手（GmToL1Nz/LoadL0Bf16/MmadBf16/FixpToGm）沿用
#           probe_crosscore_tail/probe_crosscore_tail.asc:187-273。
#   · L1→UB 的 C API 走法 `asc_copy_l12ub` 沿用 probe_ub2l1/probe_ub2l1.asc:178-179。
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 运行：
#   ./build/probe_aic_gm_dma list
#   ./build/probe_aic_gm_dma <variant> <runNo> [outdir] [--trace]
# 证据（脚本自带 flock + 进锁先 npu-smi + 每条 timeout）：
#   bash run_probe.sh                 # 默认变体集（每档各自进一次锁）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_aic_gm_dma LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(probe_aic_gm_dma probe_aic_gm_dma.asc)

target_compile_definitions(probe_aic_gm_dma PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_aic_gm_dma PRIVATE ${ASC_ARCH_FLAGS})
target_link_libraries(probe_aic_gm_dma PRIVATE m)
