# M9 GDN decode 预处理核（conv1d+SiLU/l2norm/gating 融合）：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m9_gdn_prolog/build -S m9_gdn_prolog -DCMAKE_BUILD_TYPE=Release
#   cmake --build m9_gdn_prolog/build -j4
# 运行（在 build 目录内运行，dump 文件写 cwd）：
#   cd m9_gdn_prolog/build && ./m9_gdn_prolog
# 数值校验（numpy fp32 参考，逐输出相对容差 1e-5）：
#   /usr/local/python3.12.13/bin/python3 ../check_ref.py

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m9_gdn_prolog LANGUAGES ASC CXX)

add_executable(m9_gdn_prolog
    m9_gdn_prolog.asc
)

target_compile_definitions(m9_gdn_prolog PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m9_gdn_prolog PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
